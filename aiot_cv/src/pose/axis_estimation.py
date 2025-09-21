"""
Axis estimation fallback for thin objects when reference matching fails.
Estimates object axis from segmentation mask and depth using 3D line fitting.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Any
import logging


def backproject_points(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Backproject masked depth pixels to 3D points.
    
    Args:
        depth: Depth image in meters
        mask: Binary mask (255=object, 0=background)
        K: Camera intrinsics matrix (3x3)
        
    Returns:
        3D points array (N, 3)
    """
    fy, fx = K[1, 1], K[0, 0]
    cy, cx = K[1, 2], K[0, 2]
    
    # Get mask coordinates
    ys, xs = np.where(mask > 0)
    z = depth[ys, xs].astype(np.float32)
    
    # Filter valid depth
    valid = (z > 0) & np.isfinite(z)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    
    if len(xs) == 0:
        return np.empty((0, 3))
    
    # Backproject to 3D
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts = np.stack([X, Y, z], axis=1)
    
    return pts


def extract_shaft_mask(full_mask: np.ndarray) -> np.ndarray:
    """
    Extract shaft portion from full object mask (removes handle/thick parts).
    
    Args:
        full_mask: Full object mask (uint8, 0 or 255)
        
    Returns:
        Shaft-only mask
    """
    m = full_mask.copy()
    
    # Opening to remove thin connections
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Use distance transform to identify thick parts (handle)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    
    # Remove thick parts (typically the handle)
    if np.any(m > 0):
        threshold = np.percentile(dist[m > 0], 60)  # Keep thinner parts
        shaft = np.uint8((dist <= threshold) & (m > 0)) * 255
        
        # Clean up
        shaft = cv2.morphologyEx(shaft, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    else:
        shaft = m
    
    return shaft


def ransac_line3d(points: np.ndarray, thresh: float = 0.006, iters: int = 300) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Fit 3D line to points using RANSAC.
    
    Args:
        points: 3D points array (N, 3)
        thresh: Distance threshold for inliers (meters)
        iters: Number of RANSAC iterations
        
    Returns:
        (origin, direction, num_inliers) or None if failed
    """
    if len(points) < 50:
        logging.warning(f"[AxisEst] Too few points for RANSAC: {len(points)} < 50")
        return None
    
    best_inliers = None
    best_count = 0
    rng = np.random.default_rng(42)
    
    for _ in range(iters):
        # Sample two points
        idx = rng.integers(0, len(points), size=2)
        p0, p1 = points[idx[0]], points[idx[1]]
        
        # Line direction
        v = p1 - p0
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            continue
        v /= nv
        
        # Point-to-line distance
        diff = points - p0
        distances = np.linalg.norm(np.cross(diff, v), axis=1)
        inliers = distances < thresh
        
        if inliers.sum() > best_count:
            best_inliers = inliers
            best_count = inliers.sum()
    
    if best_inliers is None or best_count < 30:
        logging.warning(f"[AxisEst] RANSAC failed: best_inliers={best_count}")
        return None
    
    # Refine using inliers (PCA)
    inlier_points = points[best_inliers]
    center = inlier_points.mean(axis=0)
    
    # SVD for best-fit line
    centered = inlier_points - center
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]  # Principal component
    direction /= np.linalg.norm(direction)
    
    logging.info(f"[AxisEst] RANSAC success: {best_count}/{len(points)} inliers ({best_count/len(points):.1%})")
    
    return center, direction, best_count


def estimate_axis_from_depth(depth_m: np.ndarray, mask_u8: np.ndarray, K: np.ndarray, 
                           z_range: Tuple[float, float] = (0.18, 0.30)) -> Optional[Tuple[np.ndarray, np.ndarray, int, dict]]:
    """
    Estimate object axis from depth and mask when reference matching fails.
    
    Args:
        depth_m: Depth image in meters
        mask_u8: Object mask (uint8, 0 or 255)
        K: Camera intrinsics matrix (3x3)
        z_range: Valid depth range (min, max) in meters
        
    Returns:
        (origin, axis_direction, num_inliers, stats) or None if failed
    """
    # Quality gate: depth range
    valid_depth = depth_m[mask_u8 > 0]
    valid_depth = valid_depth[np.isfinite(valid_depth) & (valid_depth > 0)]
    
    if len(valid_depth) == 0:
        logging.warning("[AxisEst] No valid depth in mask")
        return None
    
    median_z = np.median(valid_depth)
    if not (z_range[0] <= median_z <= z_range[1]):
        logging.warning(f"[AxisEst] Depth out of range: {median_z:.3f}m not in {z_range}")
        return None
    
    # Quality gate: valid depth ratio
    mask_pixels = (mask_u8 > 0).sum()
    valid_depth_pixels = len(valid_depth)
    valid_ratio = valid_depth_pixels / mask_pixels if mask_pixels > 0 else 0
    
    if valid_ratio < 0.95:
        logging.warning(f"[AxisEst] Low valid depth ratio: {valid_ratio:.1%} < 95%")
        return None
    
    logging.info(f"[AxisEst] Quality checks passed: median_z={median_z:.3f}m, valid_ratio={valid_ratio:.1%}")
    
    # Try shaft extraction first
    shaft_mask = extract_shaft_mask(mask_u8)
    shaft_points = backproject_points(depth_m, shaft_mask, K)
    
    result = None
    if len(shaft_points) >= 100:
        logging.info(f"[AxisEst] Trying shaft-only estimation with {len(shaft_points)} points")
        result = ransac_line3d(shaft_points, thresh=0.006, iters=400)
    
    # Fallback to full mask if shaft extraction failed
    if result is None:
        logging.info("[AxisEst] Shaft extraction failed, trying full mask")
        full_points = backproject_points(depth_m, mask_u8, K)
        if len(full_points) >= 100:
            result = ransac_line3d(full_points, thresh=0.007, iters=500)
    
    if result is None:
        logging.warning("[AxisEst] All axis estimation attempts failed")
        return None
    
    origin, direction, num_inliers = result
    
    # Calculate statistics
    stats = {
        'median_z': median_z,
        'valid_ratio': valid_ratio,
        'total_points': len(shaft_points) if len(shaft_points) >= 100 else len(full_points),
        'inlier_ratio': num_inliers / (len(shaft_points) if len(shaft_points) >= 100 else len(full_points)),
        'used_shaft': len(shaft_points) >= 100
    }
    
    logging.info(f"[AxisEst] Success: origin={origin}, direction={direction}")
    logging.info(f"[AxisEst] Stats: {stats}")
    
    return origin, direction, num_inliers, stats


def axis_from_mask_2d(mask_u8: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Estimate 2D axis from mask when depth is not available.
    
    Args:
        mask_u8: Object mask (uint8, 0 or 255)
        
    Returns:
        ((center_x, center_y), (dir_x, dir_y)) or None if failed
    """
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 50:
        return None
    
    # PCA on 2D coordinates
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    
    # Covariance matrix
    cov = (centered.T @ centered) / len(centered)
    eigenvals, eigenvecs = np.linalg.eig(cov)
    
    # Major axis (largest eigenvalue)
    major_idx = np.argmax(eigenvals)
    major_axis = eigenvecs[:, major_idx]
    major_axis = major_axis / np.linalg.norm(major_axis)
    
    return (center[0], center[1]), (major_axis[0], major_axis[1])
