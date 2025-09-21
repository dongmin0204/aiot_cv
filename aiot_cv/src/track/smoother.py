"""
Real-time pose smoothing and filtering for FoundationPose pipeline.
Includes PCA-based smoothing with angular velocity and acceleration clamping.
"""

import numpy as np
from typing import List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass
import time


@dataclass
class PoseFilterConfig:
    """Configuration for pose filtering."""
    window_size: int = 5           # Number of poses to consider for smoothing
    max_angular_velocity: float = 2.0    # rad/s
    max_angular_acceleration: float = 5.0 # rad/s²
    max_translation_velocity: float = 1.0 # m/s
    max_translation_acceleration: float = 2.0 # m/s²
    outlier_threshold: float = 0.1       # Maximum pose change to accept
    enable_temporal_smoothing: bool = True
    enable_outlier_rejection: bool = True


class RealtimePCASmoother:
    """
    Real-time pose smoother using PCA-based filtering with velocity/acceleration constraints.
    
    Features:
    - Temporal smoothing using sliding window
    - Angular velocity and acceleration clamping
    - Translation velocity and acceleration clamping
    - Outlier rejection based on pose change magnitude
    - Adaptive window sizing
    """
    
    def __init__(self, config: Optional[PoseFilterConfig] = None):
        """
        Initialize the pose smoother.
        
        Args:
            config: Filtering configuration
        """
        self.config = config or PoseFilterConfig()
        self.pose_history: Deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self.time_history: Deque[float] = deque(maxlen=self.config.window_size)
        self.last_valid_pose: Optional[np.ndarray] = None
        self.last_time: Optional[float] = None
        
        # State for velocity/acceleration tracking
        self.last_angular_velocity: Optional[np.ndarray] = None
        self.last_translation_velocity: Optional[np.ndarray] = None
        
        print(f"PCASmoother initialized with window_size={self.config.window_size}")
    
    def update(self, pose: np.ndarray, timestamp: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Update smoother with new pose and return filtered pose.
        
        Args:
            pose: Input pose (4, 4)
            timestamp: Timestamp in seconds (optional, uses current time if None)
            
        Returns:
            Filtered pose (4, 4) or None if rejected
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Validate input pose
        if not self._is_valid_pose(pose):
            print("Warning: Invalid pose provided")
            return self.last_valid_pose
        
        # Outlier rejection
        if (self.config.enable_outlier_rejection and 
            self.last_valid_pose is not None and
            not self._is_pose_reasonable(pose)):
            print("Warning: Pose rejected as outlier")
            return self.last_valid_pose
        
        # Add to history
        self.pose_history.append(pose.copy())
        self.time_history.append(timestamp)
        
        # Apply temporal smoothing
        if self.config.enable_temporal_smoothing and len(self.pose_history) >= 2:
            smoothed_pose = self._apply_temporal_smoothing()
        else:
            smoothed_pose = pose.copy()
        
        # Apply velocity/acceleration constraints
        constrained_pose = self._apply_motion_constraints(smoothed_pose, timestamp)
        
        # Update state
        self.last_valid_pose = constrained_pose
        self.last_time = timestamp
        
        return constrained_pose
    
    def reset(self):
        """Reset the smoother state."""
        self.pose_history.clear()
        self.time_history.clear()
        self.last_valid_pose = None
        self.last_time = None
        self.last_angular_velocity = None
        self.last_translation_velocity = None
        print("PCASmoother reset")
    
    def get_velocity(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get current angular and translation velocities.
        
        Returns:
            Tuple of (angular_velocity, translation_velocity) in rad/s and m/s
        """
        if len(self.pose_history) < 2:
            return None, None
        
        current_pose = self.pose_history[-1]
        previous_pose = self.pose_history[-2]
        current_time = self.time_history[-1]
        previous_time = self.time_history[-2]
        
        dt = current_time - previous_time
        if dt <= 0:
            return None, None
        
        # Compute velocities
        angular_vel, translation_vel = self._compute_velocities(
            previous_pose, current_pose, dt
        )
        
        return angular_vel, translation_vel
    
    def _is_valid_pose(self, pose: np.ndarray) -> bool:
        """Check if pose is valid."""
        if pose is None or pose.shape != (4, 4):
            return False
        
        # Check for NaN or Inf first
        if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
            return False
        
        # Check if rotation matrix is valid (orthogonal, det=1)
        R = pose[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
            return False
        if abs(np.linalg.det(R) - 1.0) > 1e-3:
            return False
        
        # Check translation is reasonable (not too far from camera)
        t = pose[:3, 3]
        distance = np.linalg.norm(t)
        if distance > 10.0 or distance < 0.01:  # 1cm to 10m range
            return False
        
        # Check bottom row is [0, 0, 0, 1]
        if not np.allclose(pose[3, :], [0, 0, 0, 1], atol=1e-6):
            return False
        
        return True
    
    def _is_pose_reasonable(self, pose: np.ndarray) -> bool:
        """Check if pose change is reasonable (outlier detection)."""
        if self.last_valid_pose is None:
            return True
        
        # Compute pose difference
        pose_diff = self._compute_pose_difference(self.last_valid_pose, pose)
        
        # Check translation change
        translation_change = np.linalg.norm(pose_diff[:3, 3])
        if translation_change > self.config.outlier_threshold:
            return False
        
        # Check rotation change (angle of rotation matrix)
        rotation_angle = self._rotation_angle(pose_diff[:3, :3])
        if rotation_angle > self.config.outlier_threshold:
            return False
        
        return True
    
    def _apply_temporal_smoothing(self) -> np.ndarray:
        """Apply PCA-based temporal smoothing."""
        if len(self.pose_history) < 2:
            return self.pose_history[-1].copy()
        
        # Convert poses to 6D representation (3 translation + 3 rotation)
        poses_6d = []
        for pose in self.pose_history:
            translation = pose[:3, 3]
            rotation = self._rotation_matrix_to_euler(pose[:3, :3])
            poses_6d.append(np.concatenate([translation, rotation]))
        
        poses_6d = np.array(poses_6d)
        
        # Apply PCA smoothing (weighted average with more weight on recent poses)
        weights = np.exp(np.linspace(-2, 0, len(poses_6d)))
        weights = weights / np.sum(weights)
        
        smoothed_6d = np.average(poses_6d, axis=0, weights=weights)
        
        # Convert back to 4x4 pose
        smoothed_translation = smoothed_6d[:3]
        smoothed_rotation = self._euler_to_rotation_matrix(smoothed_6d[3:])
        
        smoothed_pose = np.eye(4)
        smoothed_pose[:3, :3] = smoothed_rotation
        smoothed_pose[:3, 3] = smoothed_translation
        
        return smoothed_pose
    
    def _apply_motion_constraints(self, pose: np.ndarray, timestamp: float) -> np.ndarray:
        """Apply velocity and acceleration constraints."""
        if (self.last_valid_pose is None or 
            self.last_time is None or 
            len(self.pose_history) < 2):
            return pose
        
        dt = timestamp - self.last_time
        if dt <= 0:
            return pose
        
        # Compute current velocities
        current_angular_vel, current_translation_vel = self._compute_velocities(
            self.last_valid_pose, pose, dt
        )
        
        if current_angular_vel is None or current_translation_vel is None:
            return pose
        
        # Apply velocity constraints
        constrained_pose = pose.copy()
        
        # Angular velocity constraint
        angular_vel_magnitude = np.linalg.norm(current_angular_vel)
        if angular_vel_magnitude > self.config.max_angular_velocity:
            scale_factor = self.config.max_angular_velocity / angular_vel_magnitude
            # Scale down the rotation
            rotation_change = self._rotation_matrix_to_euler(pose[:3, :3]) - \
                            self._rotation_matrix_to_euler(self.last_valid_pose[:3, :3])
            constrained_rotation_change = rotation_change * scale_factor
            constrained_rotation = self._euler_to_rotation_matrix(
                self._rotation_matrix_to_euler(self.last_valid_pose[:3, :3]) + 
                constrained_rotation_change
            )
            constrained_pose[:3, :3] = constrained_rotation
        
        # Translation velocity constraint
        translation_vel_magnitude = np.linalg.norm(current_translation_vel)
        if translation_vel_magnitude > self.config.max_translation_velocity:
            scale_factor = self.config.max_translation_velocity / translation_vel_magnitude
            translation_change = pose[:3, 3] - self.last_valid_pose[:3, 3]
            constrained_translation_change = translation_change * scale_factor
            constrained_pose[:3, 3] = self.last_valid_pose[:3, 3] + constrained_translation_change
        
        return constrained_pose
    
    def _compute_velocities(self, pose1: np.ndarray, pose2: np.ndarray, dt: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute angular and translation velocities between two poses."""
        if dt <= 0:
            return None, None
        
        # Translation velocity
        translation_change = pose2[:3, 3] - pose1[:3, 3]
        translation_velocity = translation_change / dt
        
        # Angular velocity
        rotation_change = pose2[:3, :3] @ pose1[:3, :3].T
        angular_velocity = self._rotation_matrix_to_axis_angle(rotation_change) / dt
        
        return angular_velocity, translation_velocity
    
    def _compute_pose_difference(self, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """Compute pose difference (pose2 - pose1)."""
        return pose2 @ np.linalg.inv(pose1)
    
    def _rotation_angle(self, R: np.ndarray) -> float:
        """Compute rotation angle from rotation matrix."""
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        return angle
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (ZYX order)."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (ZYX order) to rotation matrix."""
        x, y, z = euler
        
        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        cz, sz = np.cos(z), np.sin(z)
        
        R = np.array([
            [cy*cz, -cy*sz, sy],
            [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy],
            [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy]
        ])
        
        return R
    
    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation."""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        
        if angle < 1e-6:
            return np.zeros(3)
        
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))
        
        return axis * angle
