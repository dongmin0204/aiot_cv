"""
Point cloud processing utilities for FoundationPose pipeline.
Includes RGB-D to point cloud conversion, noise filtering, and geometric analysis.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass

try:
    import open3d as o3d
    from sklearn.neighbors import KDTree
    HAS_O3D = True
    HAS_SKLEARN = True
except ImportError:
    HAS_O3D = False
    HAS_SKLEARN = False
    o3d = None
    KDTree = None


@dataclass
class PointCloudData:
    """Point cloud data container."""
    points: np.ndarray      # (N, 3) 3D points
    colors: Optional[np.ndarray] = None  # (N, 3) RGB colors (0-1)
    normals: Optional[np.ndarray] = None  # (N, 3) surface normals
    
    def __len__(self):
        return len(self.points)
    
    def __post_init__(self):
        if self.colors is not None and len(self.colors) != len(self.points):
            raise ValueError("Colors length must match points length")
        if self.normals is not None and len(self.normals) != len(self.points):
            raise ValueError("Normals length must match points length")


def rgbd_to_pcl(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, 
                depth_scale: float = 0.001, mask: Optional[np.ndarray] = None,
                max_depth: float = 5.0) -> PointCloudData:
    """
    Convert RGB-D image to point cloud.
    
    Args:
        rgb: RGB image (H, W, 3) uint8
        depth: Depth image (H, W) uint16 or float
        K: Camera intrinsics (3, 3)
        depth_scale: Depth scale factor (meters per depth unit)
        mask: Binary mask (H, W) to filter points
        max_depth: Maximum depth in meters
        
    Returns:
        PointCloudData object
    """
    h, w = depth.shape
    
    # Create coordinate grids
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # Generate pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert depth to meters and filter
    depth_m = depth.astype(np.float32) * depth_scale
    valid_depth = (depth_m > 0) & (depth_m < max_depth)
    
    # Apply mask if provided
    if mask is not None:
        valid_depth = valid_depth & (mask > 0)
    
    # Extract valid pixels
    u_valid = u[valid_depth]
    v_valid = v[valid_depth]
    depth_valid = depth_m[valid_depth]
    
    # Backproject to 3D
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid
    
    points = np.stack([x, y, z], axis=1)
    
    # Extract colors for valid points
    colors = None
    if rgb is not None:
        colors = rgb[v_valid, u_valid].astype(np.float32) / 255.0
    
    return PointCloudData(points=points, colors=colors)


def statistical_outlier_removal(points: np.ndarray, k: int = 20, 
                               std_ratio: float = 1.5) -> np.ndarray:
    """
    Remove statistical outliers from point cloud.
    
    Args:
        points: Point cloud (N, 3)
        k: Number of neighbors to consider
        std_ratio: Standard deviation ratio threshold
        
    Returns:
        Filtered points (M, 3) where M <= N
    """
    if not HAS_SKLEARN or len(points) <= k:
        return points
    
    # Build KDTree for efficient neighbor search
    tree = KDTree(points)
    
    # Find k-nearest neighbors for each point
    distances, _ = tree.query(points, k=k)
    
    # Calculate mean distance to neighbors (excluding self)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # Remove outliers based on standard deviation
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    threshold = global_mean + std_ratio * global_std
    valid_mask = mean_distances <= threshold
    
    return points[valid_mask]


def radius_outlier_removal(points: np.ndarray, radius: float = 0.02,
                          min_neighbors: int = 8) -> np.ndarray:
    """
    Remove radius outliers from point cloud.
    
    Args:
        points: Point cloud (N, 3)
        radius: Search radius
        min_neighbors: Minimum number of neighbors required
        
    Returns:
        Filtered points (M, 3) where M <= N
    """
    if not HAS_SKLEARN or len(points) < min_neighbors:
        return points
    
    tree = KDTree(points)
    
    # Count neighbors within radius
    neighbor_counts = tree.query_radius(points, r=radius, count_only=True)
    
    # Keep points with sufficient neighbors
    valid_mask = neighbor_counts >= min_neighbors
    
    return points[valid_mask]


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.003) -> np.ndarray:
    """
    Downsample point cloud using voxel grid.
    
    Args:
        points: Point cloud (N, 3)
        voxel_size: Voxel size for downsampling
        
    Returns:
        Downsampled points (M, 3) where M <= N
    """
    if len(points) == 0:
        return points
    
    # Simple voxel downsampling: assign points to voxels and take centroid
    voxel_coords = np.floor(points / voxel_size).astype(int)
    
    # Find unique voxels
    unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
    
    # Compute centroid for each voxel
    downsampled_points = np.zeros((len(unique_voxels), 3))
    for i, voxel in enumerate(unique_voxels):
        voxel_mask = inverse_indices == i
        downsampled_points[i] = np.mean(points[voxel_mask], axis=0)
    
    return downsampled_points


def compute_obb_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Oriented Bounding Box using PCA.
    
    Args:
        points: Point cloud (N, 3)
        
    Returns:
        center: OBB center (3,)
        axes: OBB axes as rotation matrix (3, 3)
        extents: OBB extents (3,)
    """
    if len(points) == 0:
        return np.zeros(3), np.eye(3), np.zeros(3)
    
    # Center the points
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    # Compute PCA
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (largest first)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, sorted_indices]
    
    # Ensure right-handed coordinate system
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    
    # Project points onto principal axes and compute extents
    projected = centered_points @ axes
    min_proj = np.min(projected, axis=0)
    max_proj = np.max(projected, axis=0)
    extents = max_proj - min_proj
    
    # Adjust center to account for extents
    center += axes @ ((min_proj + max_proj) / 2)
    
    return center, axes, extents


def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Estimate surface normals for point cloud.
    
    Args:
        points: Point cloud (N, 3)
        k: Number of neighbors for normal estimation
        
    Returns:
        Normals (N, 3)
    """
    if not HAS_SKLEARN or len(points) < k:
        return np.zeros_like(points)
    
    tree = KDTree(points)
    normals = np.zeros_like(points)
    
    for i, point in enumerate(points):
        # Find k-nearest neighbors
        _, neighbor_indices = tree.query([point], k=k)
        neighbors = points[neighbor_indices[0]]
        
        # Compute covariance matrix
        centered_neighbors = neighbors - np.mean(neighbors, axis=0)
        cov_matrix = np.cov(centered_neighbors.T)
        
        # Normal is the eigenvector with smallest eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, 0]
        
        # Ensure consistent orientation (pointing towards camera)
        if normal[2] > 0:
            normal *= -1
            
        normals[i] = normal
    
    return normals


def filter_pointcloud(pcd: PointCloudData, sor_k: int = 20, sor_std: float = 1.5,
                     ror_radius: float = 0.02, ror_min_neighbors: int = 8,
                     voxel_size: float = 0.003) -> PointCloudData:
    """
    Apply comprehensive filtering to point cloud.
    
    Args:
        pcd: Input point cloud
        sor_k: SOR neighbor count
        sor_std: SOR standard deviation ratio
        ror_radius: ROR search radius
        ror_min_neighbors: ROR minimum neighbors
        voxel_size: Voxel downsampling size
        
    Returns:
        Filtered point cloud
    """
    if len(pcd) == 0:
        return pcd
    
    # Apply filters in sequence
    points = pcd.points.copy()
    colors = pcd.colors.copy() if pcd.colors is not None else None
    normals = pcd.normals.copy() if pcd.normals is not None else None
    
    # Statistical outlier removal
    if HAS_SKLEARN and len(points) > sor_k:
        valid_mask = np.ones(len(points), dtype=bool)
        tree = KDTree(points)
        distances, _ = tree.query(points, k=sor_k)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + sor_std * global_std
        valid_mask = mean_distances <= threshold
        
        points = points[valid_mask]
        if colors is not None:
            colors = colors[valid_mask]
        if normals is not None:
            normals = normals[valid_mask]
    
    # Radius outlier removal
    if HAS_SKLEARN and len(points) > ror_min_neighbors:
        tree = KDTree(points)
        neighbor_counts = tree.query_radius(points, r=ror_radius, count_only=True)
        valid_mask = neighbor_counts >= ror_min_neighbors
        
        points = points[valid_mask]
        if colors is not None:
            colors = colors[valid_mask]
        if normals is not None:
            normals = normals[valid_mask]
    
    # Voxel downsampling
    if len(points) > 0 and voxel_size > 0:
        voxel_coords = np.floor(points / voxel_size).astype(int)
        unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
        
        downsampled_points = np.zeros((len(unique_voxels), 3))
        downsampled_colors = None
        downsampled_normals = None
        
        if colors is not None:
            downsampled_colors = np.zeros((len(unique_voxels), 3))
        if normals is not None:
            downsampled_normals = np.zeros((len(unique_voxels), 3))
        
        for i, voxel in enumerate(unique_voxels):
            voxel_mask = inverse_indices == i
            downsampled_points[i] = np.mean(points[voxel_mask], axis=0)
            if colors is not None:
                downsampled_colors[i] = np.mean(colors[voxel_mask], axis=0)
            if normals is not None:
                downsampled_normals[i] = np.mean(normals[voxel_mask], axis=0)
        
        points = downsampled_points
        colors = downsampled_colors
        normals = downsampled_normals
    
    return PointCloudData(points=points, colors=colors, normals=normals)


def create_o3d_pointcloud(pcd: PointCloudData) -> Optional[object]:
    """
    Convert PointCloudData to Open3D PointCloud object.
    
    Args:
        pcd: PointCloudData object
        
    Returns:
        Open3D PointCloud or None if Open3D not available
    """
    if not HAS_O3D or len(pcd) == 0:
        return None
    
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    
    if pcd.colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
    
    if pcd.normals is not None:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals)
    
    return o3d_pcd
