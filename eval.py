from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 300000000
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

from dust3r.utils.metrics import (
    normalize_to_minus1_plus1,
    normalize_to_0_1,
    correspondence_rmse,
    rotation_angle_error,
    normalized_translation_error,
    PoseMetrics,
)
from dust3r.utils.pose_estimation import (
    COORD_TRANSFORM,
    estimate_relative_pose,
    compute_epipolar_geometry,
    select_best_pose_by_ground_truth,
    is_camera_upright,
    mirror_pose_across_correspondences,
    get_plan_intrinsics,
)


# =============================================================================
# Configuration
# =============================================================================

RESOLUTION = (512, 512)
PLAN_FOCAL_LENGTH = 1e7  # Large focal length for orthographic approximation


# =============================================================================
# Data Loading
# =============================================================================

def load_ground_truth_data(
    image_pairs_csv: str,
    geometric_dir: str,
    visual_dir: str,
) -> list[dict]:
    """
    Load pose ground truth from CSV metadata and .npy files.
    
    Args:
        pose_csv: Path to the CSV file with image pair metadata.
        geometric_dir: Directory containing camera poses and intrinsics.
        visual_dir: Directory containing floor plan images and photos.
    
    Returns:
        List of dictionaries, each containing:
        - plan_corrs: Nx2 array of plan correspondences (numpy array)
        - photo_corrs: Nx2 array of photo correspondences (numpy array)
        - R_gt: 3x3 rotation matrix (numpy array)
        - t_gt: 3x1 translation vector (numpy array)
        - K_photo: 3x3 camera intrinsics (numpy array)
        - plan_size: (width, height) tuple
        - uid: Sample UID
        - scene_name: Scene name
    """
    import csv
    from os.path import join
    
    pose_data = []
    
    with open(image_pairs_csv, 'r') as f:
        reader = csv.reader(f)
        
        for row in reader:
            if len(row) < 4:
                continue
            
            uid, scene_name, plan_path, photo_path = row[0], row[1], row[2], row[3]
            
            try:
                uid = int(uid)
            except ValueError:
                # Skip header row if present
                continue
            
            corrs_path = join(
                geometric_dir,
                "correspondences",
                f"{uid // 1000}",
                f"{uid:06d}.npy",
            )

            # Load pose from .npy file
            pose_path = join(
                geometric_dir,
                "camera_poses",
                f"{uid // 1000}",
                f"{uid:06d}.npy",
            )
            
            # Load plan image to get size
            plan_full_path = join(visual_dir, scene_name, plan_path)
            with Image.open(plan_full_path) as img:
                plan_size = img.size  # (width, height)

            plan_corrs, photo_corrs = np.load(corrs_path, allow_pickle=True)
            mask = valid_xy_mask(img_size=plan_size, corrs=plan_corrs, size=RESOLUTION[0])
            photo_corrs = photo_corrs[mask]
            R_plan2cam, t_plan, K_photo = np.load(pose_path, allow_pickle=True)
            
            pose_data.append({
                'plan_corrs': np.array(plan_corrs.tolist(), dtype=np.float64),
                'photo_corrs': np.array(photo_corrs.tolist(), dtype=np.float64),
                'R_gt': np.array(R_plan2cam.tolist(), dtype=np.float64),
                't_gt': np.array(t_plan, dtype=np.float64).reshape(3),
                'K_photo': np.array(K_photo.tolist(), dtype=np.float64),
                'plan_size': plan_size,
                'uid': uid,
                'scene_name': scene_name,
            })
    
    return pose_data


# =============================================================================
# Data Processing
# =============================================================================

def valid_xy_mask(img_size: tuple[int, int], corrs: np.ndarray, size: int) -> np.ndarray:
    """
    Create a mask for valid (x, y) coordinates after resizing and padding.
    
    Args:
        img_size: Original image size (width, height).
        corrs: Nx2 array of correspondences.
        size: Target padded size (int).
    
    Returns:
        mask: Boolean array indicating valid correspondences after resize and pad.
    """
    W, H = img_size
    ratio = min(size / W, size / H)
    W_target = int(W * ratio)
    H_target = int(H * ratio)
    
    # Resize and create padded image
    w_offset = (size - W_target) // 2
    h_offset = (size - H_target) // 2
    
    # Apply same transformations to correspondences
    offset = np.array([0, h_offset]) if W_target > H_target else np.array([w_offset, 0])
    corrs_updated = corrs * ratio + offset
    
    # Create mask for valid coordinates (within bounds)
    mask = np.all((corrs_updated >= 0) & (corrs_updated <= size - 1), axis=1)
    
    return mask


def process_correspondences(view1: dict, view2: dict, pred1: dict, pred2: dict) -> tuple:
    """
    Extract and normalize correspondence predictions and ground truth.
    
    Args:
        view1: First view data (plan image).
        view2: Second view data (photo image).
        pred1: Model predictions for view 1.
        pred2: Model predictions for view 2.
    
    Returns:
        Tuple of (pred_coords, gt_coords, photo_corrs) where:
        - pred_coords: Predicted correspondences in normalized [-1, 1] space
        - gt_coords: Ground truth correspondences in normalized [-1, 1] space
        - photo_corrs: Photo correspondence indices for pose estimation
    """
    plan_corrs = view1["corrs"][0]
    photo_corrs = view2["corrs"][0]
    
    # Normalize ground truth coordinates to [-1, 1]
    gt = normalize_to_minus1_plus1(plan_corrs, RESOLUTION[0])
    
    # Extract predictions from 3D points (using X and Z coordinates)
    pred_xyz = pred2["pts3d_in_other_view"]
    pred_x = pred_xyz[..., 0:1]
    pred_z = pred_xyz[..., 2:3]
    pred = torch.cat([pred_x, pred_z], dim=-1)
    pred = pred[0, photo_corrs[:, 1], photo_corrs[:, 0]]
    
    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
    
    return pred, gt


def predictions_to_plan_coords(
    pred_normalized: torch.Tensor,
    plan_size: tuple[int, int],
) -> np.ndarray:
    """
    Convert normalized predictions to plan image pixel coordinates.
    
    Reverses the normalization applied during preprocessing:
    1. Undo [-1, 1] normalization
    2. Undo padding offset
    3. Undo resize scaling
    
    Args:
        pred_normalized: Predictions in [-1, 1] normalized space.
        plan_size: Original plan image size (width, height).
    
    Returns:
        Nx2 array of coordinates in original image space.
    """
    corrs_norm = pred_normalized.cpu().numpy()
    W, H = plan_size
    padded_size = RESOLUTION[0]
    
    # Undo [-1, 1] normalization
    corrs = (corrs_norm + 1) * (padded_size - 1) / 2
    
    # Compute resize and pad parameters
    ratio = min(padded_size / W, padded_size / H)
    W_resized = int(W * ratio)
    H_resized = int(H * ratio)
    w_offset = (padded_size - W_resized) // 2
    h_offset = (padded_size - H_resized) // 2
    
    # Undo padding and resize
    offset = np.array([w_offset, h_offset])
    corrs_orig = (corrs - offset) / ratio
    
    return corrs_orig.astype(float)


# =============================================================================
# Pose Estimation
# =============================================================================

def estimate_pose_from_correspondences(
    photo_corrs: np.ndarray,
    pred_plan_corrs: np.ndarray,
    K_photo: np.ndarray,
    plan_size: tuple[int, int],
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    ransac_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Estimate camera pose from predicted correspondences.
    
    Args:
        photo_corrs: Nx2 correspondences in photo image.
        pred_plan_corrs: Nx2 predicted correspondences in plan image.
        K_photo: 3x3 photo camera intrinsics.
        plan_size: Plan image dimensions.
        R_gt: Ground truth rotation matrix.
        t_gt: Ground truth translation vector.
        ransac_threshold: RANSAC inlier threshold (auto-computed if None).
    
    Returns:
        Tuple of (best_R, second_best_R, t, K_plan, rot_error, trans_error) where:
        - best_R: Best rotation matrix
        - second_best_R: Second best rotation matrix
        - t: Translation vector
        - K_plan: Plan camera intrinsics
        - rot_error: Rotation error in degrees
        - trans_error: Normalized translation error
    """
    K_plan = get_plan_intrinsics(plan_size, PLAN_FOCAL_LENGTH)
    
    if ransac_threshold is None:
        ransac_threshold = 1.5 / PLAN_FOCAL_LENGTH
    
    # Estimate essential matrix and decompose
    E, R1, R2, t = estimate_relative_pose(
        img1_pts=photo_corrs,
        img2_pts=pred_plan_corrs,
        K1=K_photo,
        K2=K_plan,
        ransac_threshold=ransac_threshold,
        confidence=0.999,
    )
    
    # Select best pose using ground truth
    best_R, second_best_R, t, rot_error, trans_error = select_best_pose_by_ground_truth(
        R_gt=R_gt,
        t_gt=t_gt,
        R1_pred=R1,
        R2_pred=R2,
        t_pred=t,
        K1=K_photo,
        K2=K_plan,
        img_size=plan_size,
    )
    
    return best_R, second_best_R, t, K_plan, rot_error, trans_error


def compute_final_pose(
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    K_photo: np.ndarray,
    K_plan: np.ndarray,
    pred_corrs: np.ndarray,
    second_best_R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute final camera pose with upright correction.
    
    Args:
        R_pred: Predicted rotation matrix (from essential matrix decomposition).
        t_pred: Predicted translation vector.
        K_photo: Photo camera intrinsics.
        K_plan: Plan camera intrinsics.
        pred_corrs: Predicted correspondences (for mirror correction).
        second_best_R: Second rotation candidate (for mirror correction).
    
    Returns:
        Tuple of (R_world, cc_pred) where:
        - R_world: Final rotation matrix in world coordinates
        - cc_pred: Camera center in world coordinates
    """
    # Compute camera center from epipolar geometry
    _, e_plan, F, E = compute_epipolar_geometry(R_pred, t_pred, K1=K_photo, K2=K_plan)
    
    # Transform rotation to world coordinates
    R_world = COORD_TRANSFORM @ R_pred
    second_best_R_world = COORD_TRANSFORM @ second_best_R
    cc_pred = np.insert(e_plan, 1, 0)  # Camera center in XZ plane
    
    # Check if camera is upright
    is_upright = is_camera_upright(R_world)
    
    # If camera is not upright, mirror pose across correspondences
    if not is_upright:
        R_world, cc_pred = mirror_pose_across_correspondences(
            second_best_R_world, cc_pred, pred_corrs
        )
    
    return R_world, cc_pred


def compute_pose_errors(
    R_world: np.ndarray,
    cc_pred: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    plan_size: tuple[int, int],
) -> tuple[float, float]:
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    
    Args:
        R_world: Predicted rotation matrix in world coordinates.
        cc_pred: Predicted camera center.
        R_gt: Ground truth rotation matrix.
        t_gt: Ground truth translation vector.
        plan_size: Plan image dimensions.
    
    Returns:
        Tuple of (rotation_error_deg, translation_error_normalized).
    """
    # Ground truth camera center
    cc_gt = -R_gt.T @ t_gt
    
    # Compute errors
    rot_error = rotation_angle_error(R_gt, R_world.T)
    trans_error = normalized_translation_error(cc_gt, cc_pred, plan_size)
    
    return rot_error, trans_error


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate(
    model,
    dataloader,
    device: torch.device,
    gt_data: list[dict] | None = None,
    eval_pose: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run full evaluation on the dataset.
    
    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        device: Device to run evaluation on.
        gt_data: List of ground truth data dictionaries (one per sample).
        verbose: Whether to show progress bar.
    
    Returns:
        Dictionary containing all evaluation metrics.
    """
    model.eval()
    
    all_preds = []
    all_gts = []
    pose_metrics = PoseMetrics()
    
    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating") if verbose else enumerate(dataloader)
    
    for idx, batch in iterator:
        view1, view2 = batch
        
        # Remove unused keys
        view1.pop("instance", None)
        view2.pop("instance", None)
        
        # Move tensors to device
        for view in (view1, view2):
            for key, val in view.items():
                if isinstance(val, torch.Tensor):
                    view[key] = val.to(device, non_blocking=True)
        
        with torch.no_grad():
            pred1, pred2 = model(view1, view2)
            
            # Process correspondence predictions
            pred, gt = process_correspondences(view1, view2, pred1, pred2)
            
            all_preds.append(pred.cpu().numpy())
            all_gts.append(gt.cpu().numpy())
            
            # Pose estimation 
            if eval_pose:
                try:
                    # Get ground truth for this sample
                    gt_sample = gt_data[idx]
                    
                    # Skip if pose data is missing for this sample
                    if gt_sample is None:
                        continue
                    
                    R_gt = gt_sample['R_gt']
                    t_gt = gt_sample['t_gt']
                    K_photo = gt_sample['K_photo']
                    plan_size = gt_sample['plan_size']
                    plan_corrs = gt_sample['plan_corrs']
                    photo_corrs = gt_sample['photo_corrs']
                    
                    # Convert predictions to plan coordinates
                    pred_plan_corrs = predictions_to_plan_coords(pred, plan_size)

                    # Estimate pose
                    best_R, second_best_R, t_pred, K_plan, _, _ = estimate_pose_from_correspondences(
                        photo_corrs=photo_corrs,
                        pred_plan_corrs=pred_plan_corrs,
                        K_photo=K_photo,
                        plan_size=plan_size,
                        R_gt=R_gt,
                        t_gt=t_gt,
                    )
                    
                    # Compute final pose with upright correction
                    R_world, cc_pred = compute_final_pose(
                        R_pred=best_R,
                        t_pred=t_pred,
                        K_photo=K_photo,
                        K_plan=K_plan,
                        pred_corrs=pred_plan_corrs,
                        second_best_R=second_best_R,
                    )
                    
                    # Compute errors
                    rot_err, trans_err = compute_pose_errors(
                        R_world=R_world,
                        cc_pred=cc_pred,
                        R_gt=R_gt,
                        t_gt=t_gt,
                        plan_size=plan_size,
                    )
                    
                    pose_metrics.add(rot_err, trans_err)
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Pose estimation failed for sample {idx}: {e}")
    
    # Compute correspondence RMSE in [0, 1] normalized space
    all_preds = torch.tensor(np.concatenate(all_preds))
    all_gts = torch.tensor(np.concatenate(all_gts))
    
    rmse = correspondence_rmse(
        normalize_to_0_1(all_preds),
        normalize_to_0_1(all_gts)
    )
    
    # Compile results
    results = {
        'correspondence_rmse': rmse.item(),
        'num_samples': len(dataloader),
    }
    
    if eval_pose and len(pose_metrics) > 0:
        results.update(pose_metrics.summary())
    
    return results


def print_results(results: dict) -> None:
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Correspondence Metrics':^60}")
    print("-" * 60)
    print(f"  RMSE (normalized [0,1]):  {results['correspondence_rmse']:.4f}")
    print(f"  Number of samples:        {results['num_samples']}")
    
    print(f"\n{'Pose Estimation Metrics':^60}")
    print("-" * 60)
    print(f"  Rotation Acc @5°:         {results['rotation_acc_5deg']*100:.1f}")
    print(f"  Rotation Acc @10°:        {results['rotation_acc_10deg']*100:.1f}")
    print(f"  Rotation Acc @20°:        {results['rotation_acc_20deg']*100:.1f}")
    print(f"  Rotation Acc @30°:        {results['rotation_acc_30deg']*100:.1f}")
    print()
    print(f"  Translation Acc @0.05:    {results['translation_acc_0.05']*100:.1f}")
    print(f"  Translation Acc @0.1:     {results['translation_acc_0.1']*100:.1f}")
    print(f"  Translation Acc @0.2:     {results['translation_acc_0.2']*100:.1f}")
    print()
    print(f"  Rotation Acc @30°, Translation Acc @0.2:        {results['rotation_acc_30deg, translation_acc_0.2']*100:.1f}")
    
    print("\n" + "=" * 60)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate C3PO model predictions")
    parser.add_argument("--weights", type=str, default="demo/ckpt.pth",
                        help="Path to model weights")
    parser.add_argument("--image-pairs-dir", type=str, 
                        help="Path to image pairs directory")
    parser.add_argument("--geometric-dir", type=str,
                        help="Path to geometric data directory")
    parser.add_argument("--visual-dir", type=str, default=None,
                        help="Path to visual data directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--eval-camera-poses", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    from dust3r.datasets import get_data_loader, C3
    from dust3r.model import AsymmetricCroCo3DStereo, inf
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = C3(
        image_pairs_dir=args.image_pairs_dir,
        geometric_dir=args.geometric_dir,
        visual_dir=args.visual_dir,
        split=args.split,
        resolution=[RESOLUTION],
        augmentation_factor=1
    )
    
    dataloader = get_data_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        test=True
    )
    
    # Load pose ground truth
    image_pairs_file = Path(args.image_pairs_dir) / args.split / "image_pairs.csv"
    geometric_dir_with_split = Path(args.geometric_dir) / args.split
    print(f"Loading ground truth data ...")
    gt_data = load_ground_truth_data(
        image_pairs_csv=image_pairs_file,
        geometric_dir=geometric_dir_with_split,
        visual_dir=args.visual_dir,
    )
    valid_poses = sum(1 for p in gt_data if p is not None)
    print(f"  Loaded {valid_poses}/{len(gt_data)} entries")
    
    # Load model
    print("Loading model...")
    model = AsymmetricCroCo3DStereo(
        pos_embed='RoPE100',
        patch_embed_cls='ManyAR_PatchEmbed',
        img_size=(512, 512),
        head_type='dpt',
        output_mode='pts3d',
        depth_mode=('exp', -inf, inf),
        conf_mode=('exp', 1, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12
    )
    
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        gt_data=gt_data,
        eval_pose=args.eval_camera_poses,
        verbose=True,
    )
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
