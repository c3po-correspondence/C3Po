from __future__ import annotations

import cv2
import numpy as np
from dust3r.utils.metrics import (normalized_translation_error,
                                  rotation_angle_error)

# =============================================================================
# Constants
# =============================================================================

# Transformation matrix for camera rotation matrix from the XY plane to the floor
# plan frame, which lies on the XZ plane (+90° rotation about the X-axis)
COORD_TRANSFORM = np.array([
    [1, 0, 0],   
    [0, 0, -1],  
    [0, 1, 0],   
])


# =============================================================================
# Epipolar Geometry
# =============================================================================

def essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute the essential matrix from rotation and translation.
    
    Args:
        R: 3x3 rotation matrix.
        t: 3x1 or (3,) translation vector.
    
    Returns:
        3x3 essential matrix E = [t]_x @ R.
    """
    t = t.reshape(3)
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0],
    ])
    return t_skew @ R


def fundamental_matrix(
    R: np.ndarray,
    t: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
) -> np.ndarray:
    """
    Compute fundamental matrix from pose and intrinsics.
    
    Args:
        R: 3x3 rotation matrix.
        t: 3x1 translation vector.
        K1: 3x3 intrinsic matrix for image 1.
        K2: 3x3 intrinsic matrix for image 2.
    
    Returns:
        3x3 fundamental matrix F = K2^(-T) @ E @ K1^(-1).
    """
    E = essential_matrix(R, t)
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def compute_epipolar_geometry(
    R: np.ndarray,
    t: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute epipoles and fundamental matrix from camera pose.
    
    Args:
        R: 3x3 rotation matrix from image 1 to image 2.
        t: 3x1 translation vector.
        K1: 3x3 intrinsic matrix for image 1 (photo).
        K2: 3x3 intrinsic matrix for image 2 (plan).
    
    Returns:
        Tuple of (epipole1, epipole2, F, E) where:
        - epipole1: 2D epipole in image 1
        - epipole2: 2D epipole in image 2
        - F: 3x3 fundamental matrix
        - E: 3x3 essential matrix
    """
    E = essential_matrix(R, t)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    
    # Epipole in image 1 (right null space of F)
    _, _, Vt = np.linalg.svd(F)
    e1 = Vt[-1]
    e1 = e1 / e1[2]
    
    # Epipole in image 2 (left null space of F)
    _, _, Vt = np.linalg.svd(F.T)
    e2 = Vt[-1]
    e2 = e2 / e2[2]
    
    return e1[:2].flatten(), e2[:2].flatten(), F, E


def get_plan_intrinsics(plan_size: tuple[int, int], focal_length: float = 1e7) -> np.ndarray:
    """
    Create camera intrinsics for plan/floorplan image (orthographic approximation).
    
    For floor plans, we use a very large focal length to approximate
    an orthographic projection.
    
    Args:
        plan_size: Plan image dimensions (width, height).
        focal_length: Focal length (default: 1e7 for near-orthographic).
    
    Returns:
        3x3 intrinsic matrix.
    """
    return np.array([
        [focal_length, 0, plan_size[0] // 2],
        [0, focal_length, plan_size[1] // 2],
        [0, 0, 1],
    ], dtype=np.float64)


# =============================================================================
# Pose Estimation
# =============================================================================

def estimate_relative_pose(
    img1_pts: np.ndarray,
    img2_pts: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    ransac_threshold: float = 0.001,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate relative pose between two views using point correspondences.
    
    Args:
        img1_pts: Nx2 array of (x, y) coordinates in image 1.
        img2_pts: Nx2 array of (x, y) coordinates in image 2.
        K1: 3x3 intrinsic matrix for image 1.
        K2: 3x3 intrinsic matrix for image 2.
        ransac_threshold: RANSAC threshold in normalized coordinates.
        confidence: RANSAC confidence level.
    
    Returns:
        Tuple of (E, R1, R2, t) where:
        - E: 3x3 essential matrix
        - R1, R2: Two possible 3x3 rotation matrices
        - t: 3x1 translation vector
    """
    img1_pts = img1_pts.astype(np.float32)
    img2_pts = img2_pts.astype(np.float32)
    
    # Normalize points to camera coordinates
    img1_norm = cv2.undistortPoints(
        img1_pts.reshape(-1, 1, 2), K1, None
    ).reshape(-1, 2)
    img2_norm = cv2.undistortPoints(
        img2_pts.reshape(-1, 1, 2), K2, None
    ).reshape(-1, 2)
    
    # Find essential matrix
    E, mask = cv2.findEssentialMat(
        img1_norm,
        img2_norm,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=confidence,
        threshold=ransac_threshold,
    )
    
    # Decompose into rotation and translation
    R1, R2, t = cv2.decomposeEssentialMat(E)
    
    return E, R1, R2, t


def select_best_pose_by_ground_truth(
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    R1_pred: np.ndarray,
    R2_pred: np.ndarray,
    t_pred: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    img_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Select the best pose from two rotation candidates using ground truth.
    
    This is used for oracle evaluation to measure the best possible
    pose estimation performance.
    
    Args:
        R_gt: Ground truth rotation matrix.
        t_gt: Ground truth translation vector.
        R1_pred, R2_pred: Two candidate rotation matrices.
        t_pred: Predicted translation vector.
        K1, K2: Intrinsic matrices for both cameras.
        img_size: Image dimensions for error normalization.
    
    Returns:
        Tuple of (best_R, second_best_R, t, rotation_error, translation_error).
    """    
    results = []
    
    # Evaluate both rotation matrices
    for R_candidate in [R1_pred, R2_pred]:
        rot_error = rotation_angle_error(R_gt, (COORD_TRANSFORM @ R_candidate).T)
        _, cc_oracle, _, _ = compute_epipolar_geometry(
            R_candidate, t_pred, K1=K1, K2=K2
        )
        cc_oracle = np.insert(cc_oracle, 1, 0)  # Insert Y=0 for XZ plane
        trans_error = normalized_translation_error(
            -R_gt.T @ t_gt, cc_oracle, img_size
        )
        results.append((rot_error, trans_error, R_candidate))
    
    best_rot_error, best_trans_error, best_R = min(
        results, key=lambda x: (x[0], round(x[1], 2))
    )
    _, _, second_best_R = max(results, key=lambda x: (x[0], round(x[1], 2)))
    
    return best_R, second_best_R, t_pred, best_rot_error, best_trans_error


# =============================================================================
# Camera Orientation Utilities
# =============================================================================

def is_camera_upright(R: np.ndarray) -> tuple[bool, float]:
    """
    Check if camera orientation is approximately upright.
    
    Args:
        R: 3x3 rotation matrix.
    
    Returns:
        Boolean indicating if the camera is upright.
    """
    up_world = np.array([0.0, 1.0, 0.0])
    up_camera = R[:, 1]
    dot_product = np.clip(np.dot(up_camera, up_world), -1.0, 1.0)
    return dot_product < 0.03


def mirror_pose_across_correspondences(
    R: np.ndarray,
    C: np.ndarray,
    xz_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reflect camera pose across best-fit line through correspondences in XZ plane.
    
    Args:
        R: 3x3 rotation matrix.
        C: Camera center (3D).
        xz_points: Nx2 array of points to fit the reflection line.
    
    Returns:
        Tuple of (reflected_R, reflected_C).
    """
    pts = np.asarray(xz_points)
    mean = pts.mean(axis=0)
    
    # PCA to get line direction in XZ
    _, _, Vt = np.linalg.svd(pts - mean)
    dx, dz = Vt[0] / np.linalg.norm(Vt[0])
    
    # Reflection matrix (reflects in XZ plane, Y unchanged)
    M = np.array([
        [dx * dx - dz * dz, 0.0, 2 * dx * dz],
        [0.0, 1.0, 0.0],
        [2 * dx * dz, 0.0, dz * dz - dx * dx],
    ])
    
    # Reflect rotation
    R_reflected = M @ R
    
    # Reflect camera center about the line
    C_shift = C - np.array([mean[0], 0.0, mean[1]])
    C_reflected = M @ C_shift + np.array([mean[0], 0.0, mean[1]])
    
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R_reflected) < 0:
        R_reflected[:, 1] *= -1
    
    return R_reflected, C_reflected
