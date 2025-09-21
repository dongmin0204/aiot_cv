"""
Mask overlay and 3D axis visualization utilities.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging


def draw_mask_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Draw semi-transparent mask overlay on image.
    
    Args:
        img_bgr: Input image (BGR)
        mask: Binary mask (0/255)
        alpha: Overlay transparency
        
    Returns:
        Image with mask overlay
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    # Create green overlay
    color = np.zeros_like(img_bgr)
    color[:, :, 1] = 255  # Green channel
    
    # Apply mask
    mask_binary = (mask > 0).astype(np.uint8)[:, :, None]
    overlay = (img_bgr * (1 - alpha) + color * alpha).astype(np.uint8)
    
    return np.where(mask_binary == 1, overlay, img_bgr)


def project_pts(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """
    Project 3D camera points to 2D image coordinates.
    
    Args:
        K: Camera intrinsics matrix (3x3)
        pts_cam: 3D points in camera frame (N, 3)
        
    Returns:
        2D image coordinates (N, 2)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    Z = np.clip(Z, 1e-6, None)  # Prevent division by zero
    
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    
    return np.stack([u, v], axis=1).astype(int)


def draw_axes_from_pose(img: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, 
                       axis_len: float = 0.05, thickness: int = 2) -> np.ndarray:
    """
    Draw 3D coordinate axes from pose (R, t).
    
    Args:
        img: Input image (BGR)
        K: Camera intrinsics matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3,)
        axis_len: Length of axes in meters
        thickness: Line thickness
        
    Returns:
        Image with axes drawn
    """
    try:
        # Axis endpoints in object frame
        origin = np.zeros(3)
        x_axis = np.array([axis_len, 0, 0])
        y_axis = np.array([0, axis_len, 0])
        z_axis = np.array([0, 0, axis_len])
        
        # Transform to camera frame: p_cam = R @ p_obj + t
        origin_cam = R @ origin + t
        x_cam = R @ x_axis + t
        y_cam = R @ y_axis + t
        z_cam = R @ z_axis + t
        
        # Stack points for projection
        pts_3d = np.stack([origin_cam, x_cam, origin_cam, y_cam, origin_cam, z_cam], axis=0)
        pts_2d = project_pts(K, pts_3d)
        
        # Colors: X=Red, Y=Green, Z=Blue (BGR format)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        
        out = img.copy()
        
        # Draw axes
        for i, color in enumerate(colors):
            start_idx = i * 2
            end_idx = i * 2 + 1
            
            start_pt = tuple(pts_2d[start_idx])
            end_pt = tuple(pts_2d[end_idx])
            
            # Check if points are within image bounds
            h, w = img.shape[:2]
            if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
                0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
                
                cv2.arrowedLine(out, start_pt, end_pt, color, thickness, tipLength=0.3)
                cv2.circle(out, start_pt, 3, color, -1)
        
        return out
        
    except Exception as e:
        logging.warning(f"Failed to draw axes from pose: {e}")
        return img


def pca_axis_from_depth(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, 
                       step: int = 2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract main axis from depth using PCA (for elongated objects like screwdriver).
    
    Args:
        mask: Binary mask (H, W)
        depth: Depth image in meters (H, W)
        K: Camera intrinsics matrix (3x3)
        step: Downsampling step for efficiency
        
    Returns:
        (center, axis_direction, axis_endpoints) or (None, None, None) if failed
    """
    try:
        h, w = depth.shape
        ys, xs = np.where(mask > 0)
        
        if len(xs) == 0:
            return None, None, None
        
        # Downsample for efficiency
        xs, ys = xs[::step], ys[::step]
        z = depth[ys, xs].astype(np.float32)
        
        # Filter valid depth
        valid = (z > 0) & np.isfinite(z) & (z < 5.0)
        xs, ys, z = xs[valid], ys[valid], z[valid]
        
        if len(z) < 50:  # Minimum points required
            return None, None, None
        
        # Backproject to 3D camera coordinates
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        X = (xs - cx) / fx * z
        Y = (ys - cy) / fy * z
        
        pts = np.stack([X, Y, z], axis=1)  # (N, 3) in camera frame
        
        # Compute center and PCA
        center = pts.mean(axis=0)
        centered = pts - center
        
        # SVD for PCA
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        axis_direction = Vt[0]  # First principal component (main axis)
        
        # Estimate object length from 90th percentile of distances
        distances = np.linalg.norm(centered, axis=1)
        length = np.percentile(distances, 90)
        
        # Axis endpoints for visualization
        p0 = center - axis_direction * length
        p1 = center + axis_direction * length
        axis_endpoints = np.stack([p0, p1], axis=0)
        
        return center, axis_direction, axis_endpoints
        
    except Exception as e:
        logging.warning(f"PCA axis extraction failed: {e}")
        return None, None, None


def draw_axis_from_pca(img: np.ndarray, K: np.ndarray, center: np.ndarray, 
                      axis_endpoints: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), 
                      thickness: int = 2) -> np.ndarray:
    """
    Draw axis line from PCA results.
    
    Args:
        img: Input image (BGR)
        K: Camera intrinsics matrix (3x3)
        center: Axis center point (3,)
        axis_endpoints: Axis endpoints (2, 3)
        color: Line color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with axis drawn
    """
    try:
        # Project center and endpoints
        pts_3d = np.vstack([center, axis_endpoints[0], center, axis_endpoints[1]])
        pts_2d = project_pts(K, pts_3d).reshape(2, 2, 2)
        
        out = img.copy()
        
        # Draw axis lines
        for line_pts in pts_2d:
            start_pt = tuple(line_pts[0])
            end_pt = tuple(line_pts[1])
            
            # Check bounds
            h, w = img.shape[:2]
            if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
                0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
                
                cv2.line(out, start_pt, end_pt, color, thickness)
                cv2.circle(out, start_pt, 3, color, -1)
        
        return out
        
    except Exception as e:
        logging.warning(f"Failed to draw PCA axis: {e}")
        return img


def visualize_pose_and_mask(img_bgr: np.ndarray, mask: Optional[np.ndarray], 
                           pose: Optional[np.ndarray], depth: Optional[np.ndarray], 
                           K: np.ndarray, axis_len: float = 0.05) -> np.ndarray:
    """
    Comprehensive visualization combining mask overlay and axis visualization.
    
    Args:
        img_bgr: Input image (BGR)
        mask: Binary mask (optional)
        pose: 4x4 pose matrix (optional)
        depth: Depth image in meters (optional)
        K: Camera intrinsics matrix
        axis_len: Axis length for pose visualization
        
    Returns:
        Visualization image
    """
    vis = img_bgr.copy()
    
    # 1) Mask overlay
    if mask is not None:
        vis = draw_mask_overlay(vis, mask, alpha=0.4)
    
    # 2) Axis visualization
    if pose is not None:
        # Use pose for accurate axes
        R = pose[:3, :3]
        t = pose[:3, 3]
        vis = draw_axes_from_pose(vis, K, R, t, axis_len=axis_len)
        
        # Add pose info text
        cv2.putText(vis, f"Pose: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})", 
                   (10, vis.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    elif mask is not None and depth is not None:
        # Fallback to PCA-based axis estimation
        center, axis_dir, axis_endpoints = pca_axis_from_depth(mask, depth, K)
        if center is not None:
            vis = draw_axis_from_pca(vis, K, center, axis_endpoints, color=(0, 255, 255))
            
            # Add axis info text
            cv2.putText(vis, f"PCA Axis: {axis_dir}", 
                       (10, vis.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return vis
