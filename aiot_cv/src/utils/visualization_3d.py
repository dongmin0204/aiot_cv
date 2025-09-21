"""
3D visualization utilities for pose estimation and object detection.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class Pose3DVisualizer:
    """3D pose visualization with Open3D integration."""
    
    def __init__(self, enable_o3d: bool = True):
        """
        Initialize 3D visualizer.
        
        Args:
            enable_o3d: Whether to enable Open3D visualization
        """
        self.enable_o3d = enable_o3d and OPEN3D_AVAILABLE
        self.vis = None
        self.coordinate_frame = None
        self.point_cloud = None
        self.bounding_box = None
        
        if self.enable_o3d:
            self._setup_o3d_visualizer()
    
    def _setup_o3d_visualizer(self):
        """Setup Open3D visualizer."""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="3D Pose Estimation", width=800, height=600)
            
            # Add coordinate frame
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.vis.add_geometry(self.coordinate_frame)
            
            # Setup camera view
            view_ctl = self.vis.get_view_control()
            view_ctl.set_front([0, 0, -1])
            view_ctl.set_lookat([0, 0, 0.3])
            view_ctl.set_up([0, -1, 0])
            view_ctl.set_zoom(0.8)
            
            logging.info("Open3D visualizer initialized")
        except Exception as e:
            logging.warning(f"Failed to setup Open3D visualizer: {e}")
            self.enable_o3d = False
    
    def update_pose(self, pose: np.ndarray, point_cloud: Optional[np.ndarray] = None,
                   bbox_size: Optional[Tuple[float, float, float]] = None):
        """
        Update 3D pose visualization.
        
        Args:
            pose: 4x4 pose matrix
            point_cloud: Optional point cloud (N, 3)
            bbox_size: Optional bounding box size (length, width, height)
        """
        if not self.enable_o3d or self.vis is None:
            return
        
        try:
            # Update coordinate frame pose
            if self.coordinate_frame is not None:
                self.vis.remove_geometry(self.coordinate_frame, reset_bounding_box=False)
            
            # Create new coordinate frame at pose
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            self.coordinate_frame.transform(pose)
            self.vis.add_geometry(self.coordinate_frame, reset_bounding_box=False)
            
            # Update point cloud
            if point_cloud is not None and len(point_cloud) > 0:
                if self.point_cloud is not None:
                    self.vis.remove_geometry(self.point_cloud, reset_bounding_box=False)
                
                self.point_cloud = o3d.geometry.PointCloud()
                self.point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
                
                # Color point cloud
                colors = np.tile([0.7, 0.3, 0.3], (len(point_cloud), 1))  # Red
                self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
                
                self.vis.add_geometry(self.point_cloud, reset_bounding_box=False)
            
            # Update bounding box
            if bbox_size is not None:
                if self.bounding_box is not None:
                    self.vis.remove_geometry(self.bounding_box, reset_bounding_box=False)
                
                # Create oriented bounding box
                center = pose[:3, 3]
                R = pose[:3, :3]
                
                self.bounding_box = o3d.geometry.OrientedBoundingBox(
                    center=center,
                    R=R,
                    extent=bbox_size
                )
                self.bounding_box.color = [0.0, 1.0, 0.0]  # Green
                
                self.vis.add_geometry(self.bounding_box, reset_bounding_box=False)
            
            # Update visualization
            self.vis.poll_events()
            self.vis.update_renderer()
            
        except Exception as e:
            logging.warning(f"Failed to update 3D visualization: {e}")
    
    def draw_pose_on_image(self, image: np.ndarray, pose: np.ndarray, K: np.ndarray,
                          axis_length: float = 0.1) -> np.ndarray:
        """
        Draw 3D pose axes on 2D image.
        
        Args:
            image: Input image
            pose: 4x4 pose matrix
            K: Camera intrinsics matrix
            axis_length: Length of axis lines
            
        Returns:
            Image with pose overlay
        """
        try:
            if pose is None or np.any(~np.isfinite(pose)):
                return image
            
            # 3D axis points in object frame
            origin = np.array([0, 0, 0, 1])
            x_axis = np.array([axis_length, 0, 0, 1])
            y_axis = np.array([0, axis_length, 0, 1])
            z_axis = np.array([0, 0, axis_length, 1])
            
            # Transform to camera frame
            points_3d = np.array([origin, x_axis, y_axis, z_axis])
            points_cam = (pose @ points_3d.T).T[:, :3]
            
            # Project to 2D
            points_2d, _ = cv2.projectPoints(
                points_cam.reshape(-1, 1, 3),
                np.zeros(3), np.zeros(3), K, None
            )
            points_2d = points_2d.reshape(-1, 2).astype(int)
            
            # Check if points are within image bounds
            h, w = image.shape[:2]
            valid = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & 
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h))
            
            if not valid[0]:  # Origin must be visible
                return image
            
            origin_2d = tuple(points_2d[0])
            
            # Draw axes with colors: X=Red, Y=Green, Z=Blue
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR format
            
            for i, (color, valid_point) in enumerate(zip(colors, valid[1:]), 1):
                if valid_point:
                    end_2d = tuple(points_2d[i])
                    cv2.arrowedLine(image, origin_2d, end_2d, color, 3, tipLength=0.3)
            
            # Draw pose info text
            t = pose[:3, 3]
            text = f"Pose: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})"
            cv2.putText(image, text, (10, image.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return image
            
        except Exception as e:
            logging.warning(f"Failed to draw pose on image: {e}")
            return image
    
    def draw_axis_from_direction(self, image: np.ndarray, origin_3d: np.ndarray, 
                                direction_3d: np.ndarray, K: np.ndarray,
                                axis_length: float = 0.1) -> np.ndarray:
        """
        Draw axis line from 3D origin and direction.
        
        Args:
            image: Input image
            origin_3d: 3D origin point
            direction_3d: 3D direction vector (normalized)
            K: Camera intrinsics matrix
            axis_length: Length of axis line
            
        Returns:
            Image with axis overlay
        """
        try:
            # Create axis endpoints
            end_3d = origin_3d + direction_3d * axis_length
            points_3d = np.array([origin_3d, end_3d])
            
            # Project to 2D
            points_2d, _ = cv2.projectPoints(
                points_3d.reshape(-1, 1, 3),
                np.zeros(3), np.zeros(3), K, None
            )
            points_2d = points_2d.reshape(-1, 2).astype(int)
            
            # Check bounds
            h, w = image.shape[:2]
            valid = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & 
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h))
            
            if np.all(valid):
                origin_2d = tuple(points_2d[0])
                end_2d = tuple(points_2d[1])
                
                # Draw axis in yellow for single direction
                cv2.arrowedLine(image, origin_2d, end_2d, (0, 255, 255), 3, tipLength=0.3)
                
                # Draw axis info
                text = f"Axis: {direction_3d}"
                cv2.putText(image, text, (10, image.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            return image
            
        except Exception as e:
            logging.warning(f"Failed to draw axis: {e}")
            return image
    
    def close(self):
        """Close visualization."""
        if self.enable_o3d and self.vis is not None:
            self.vis.destroy_window()
            self.vis = None


def create_3d_bbox_from_pose(pose: np.ndarray, size: Tuple[float, float, float]) -> np.ndarray:
    """
    Create 3D bounding box corners from pose and size.
    
    Args:
        pose: 4x4 pose matrix
        size: (length, width, height)
        
    Returns:
        8x3 array of bbox corners
    """
    l, w, h = size
    
    # Define bbox corners in object frame
    corners = np.array([
        [-l/2, -w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2],
        [-l/2,  w/2,  h/2]
    ])
    
    # Transform to world frame
    corners_h = np.hstack([corners, np.ones((8, 1))])
    corners_world = (pose @ corners_h.T).T[:, :3]
    
    return corners_world


def draw_3d_bbox_on_image(image: np.ndarray, bbox_corners: np.ndarray, 
                         K: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw 3D bounding box on image.
    
    Args:
        image: Input image
        bbox_corners: 8x3 array of bbox corners
        K: Camera intrinsics matrix
        color: Line color (BGR)
        
    Returns:
        Image with bbox overlay
    """
    try:
        # Project to 2D
        points_2d, _ = cv2.projectPoints(
            bbox_corners.reshape(-1, 1, 3),
            np.zeros(3), np.zeros(3), K, None
        )
        points_2d = points_2d.reshape(8, 2).astype(int)
        
        # Define bbox edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        # Draw edges
        for start, end in edges:
            cv2.line(image, tuple(points_2d[start]), tuple(points_2d[end]), color, 2)
        
        return image
        
    except Exception as e:
        logging.warning(f"Failed to draw 3D bbox: {e}")
        return image
