from __future__ import annotations

import numpy as np
import torch


# =============================================================================
# Coordinate Normalization
# =============================================================================

def normalize_to_minus1_plus1(coords: np.ndarray | torch.Tensor, image_dim: int) -> torch.Tensor:
    """
    Normalize coordinates from [0, image_dim - 1] to [-1, 1].
    """
    if isinstance(coords, np.ndarray):
        coords = torch.tensor(coords, dtype=torch.float32)
    return 2 * (coords / (image_dim - 1)) - 1


def normalize_to_0_1(coords: torch.Tensor) -> torch.Tensor:
    """
    Normalize coordinates from [-1, 1] to [0, 1].
    """
    return (coords + 1) / 2.0


# =============================================================================
# Correspondence Metrics
# =============================================================================

def correspondence_rmse(
    pred: torch.Tensor | np.ndarray,
    gt: torch.Tensor | np.ndarray,
    per_image: bool = False,
) -> torch.Tensor:
    """
    Compute Root Mean Square Error between predicted and ground truth correspondences.
    """
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.tensor(gt)
    
    if per_image:
        return torch.sqrt(torch.mean((pred - gt) ** 2, dim=1))
    return torch.sqrt(torch.mean((pred - gt) ** 2))


# =============================================================================
# Pose Error Metrics
# =============================================================================

def rotation_angle_error(R_gt: np.ndarray, R_pred: np.ndarray) -> float:
    """
    Compute rotation error as angular difference between camera forward directions.
    
    Args:
        R_gt: Ground truth 3x3 rotation matrix (cam to plan).
        R_pred: Predicted 3x3 rotation matrix (cam to plan).
    
    Returns:
        Absolute angular difference in degrees.
    """
    # Camera forward is +Z axis
    forward_gt = R_gt.T[:, 2]
    forward_pred = R_pred.T[:, 2]
    
    # Project onto XZ plane (zero out Y component)
    forward_gt_xz = np.array([forward_gt[0], 0.0, forward_gt[2]])
    forward_pred_xz = np.array([forward_pred[0], 0.0, forward_pred[2]])
    
    # # Normalize projected vectors
    forward_gt_xz = forward_gt_xz / np.linalg.norm(forward_gt_xz)
    forward_pred_xz = forward_pred_xz / np.linalg.norm(forward_pred_xz)
    
    # Calculate angles
    angle_gt = np.arctan2(forward_gt_xz[0], forward_gt_xz[2])
    angle_pred = np.arctan2(forward_pred_xz[0], forward_pred_xz[2])
    
    # Compute relative angle and normalize to [-pi, pi]
    relative_angle = angle_pred - angle_gt
    relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
    
    return float(np.abs(np.degrees(relative_angle)))


def normalized_translation_error(
    cc_gt: np.ndarray,
    cc_pred: np.ndarray,
    img_size: tuple[int, int],
) -> float:
    """
    Compute translation error in the XZ plane normalized by the image diagonal.
    
    Args:
        cc_gt: Ground truth camera center (3D).
        cc_pred: Predicted camera center (3D).
        img_size: Image dimensions (width, height) for normalization.
    
    Returns:
        Normalized L2 distance between camera centers in XZ plane.
    """
    # Compute XZ plane distance
    xz_distance = np.linalg.norm(cc_gt[[0, 2]] - cc_pred[[0, 2]])
    
    # Normalize by image diagonal
    normalization = np.linalg.norm(img_size)
    
    return float(xz_distance / normalization)


# =============================================================================
# Aggregated Metrics
# =============================================================================

class PoseMetrics:
    """
    Accumulator for pose estimation metrics.
    
    Collects rotation and translation errors across multiple samples
    and computes summary statistics.
    """
    
    def __init__(self):
        self.rotation_errors: list[float] = []
        self.translation_errors: list[float] = []
    
    def add(self, rotation_error: float, translation_error: float) -> None:
        """Add a single sample's errors."""
        self.rotation_errors.append(rotation_error)
        self.translation_errors.append(translation_error)
    
    def rotation_accuracy_at_threshold(self, threshold_deg: float) -> float:
        """Fraction of samples with rotation error below threshold."""
        if not self.rotation_errors:
            return float('nan')
        return float(np.mean(np.array(self.rotation_errors) < threshold_deg))
    
    def translation_accuracy_at_threshold(self, threshold: float) -> float:
        """Fraction of samples with translation error below threshold."""
        if not self.translation_errors:
            return float('nan')
        return float(np.mean(np.array(self.translation_errors) < threshold))
    
    def rotation_translation_accuracy_at_threshold(self, rot_threshold_deg: float, trans_threshold: float) -> float:
        """Fraction of samples with both rotation and translation errors below thresholds."""
        if not self.rotation_errors or not self.translation_errors:
            return float('nan')
        rot_array = np.array(self.rotation_errors)
        trans_array = np.array(self.translation_errors)
        return float(np.mean((rot_array < rot_threshold_deg) & (trans_array < trans_threshold)))
    
    def summary(self) -> dict[str, float]:
        """
        Get a summary dictionary of all metrics.
        
        Returns:
            Dictionary containing:
            - rotation_acc_5deg: Accuracy at 5° threshold
            - rotation_acc_10deg: Accuracy at 10° threshold
            - rotation_acc_15deg: Accuracy at 15° threshold
            - translation_acc_0.05: Accuracy at 0.05 threshold
            - translation_acc_0.1: Accuracy at 0.1 threshold
            - translation_acc_0.1: Accuracy at 0.1 threshold
            - translation_acc_0.2: Accuracy at 0.2 threshold
        """
        return {
            'rotation_acc_5deg': self.rotation_accuracy_at_threshold(5.0),
            'rotation_acc_10deg': self.rotation_accuracy_at_threshold(10.0),
            'rotation_acc_20deg': self.rotation_accuracy_at_threshold(20.0),
            'rotation_acc_30deg': self.rotation_accuracy_at_threshold(30.0),
            'translation_acc_0.05': self.translation_accuracy_at_threshold(0.05),
            'translation_acc_0.1': self.translation_accuracy_at_threshold(0.1),
            'translation_acc_0.2': self.translation_accuracy_at_threshold(0.2),
            'rotation_acc_30deg, translation_acc_0.2': self.rotation_translation_accuracy_at_threshold(30.0, 0.2),
        }
    
    def __len__(self) -> int:
        return len(self.rotation_errors)
    
    def __repr__(self) -> str:
        return (
            f"PoseMetrics({len(self)} samples)"
        )
