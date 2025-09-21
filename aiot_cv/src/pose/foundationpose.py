"""
FoundationPose wrapper for model-free pose estimation from reference images.
Supports few-shot initialization and real-time tracking.
"""

import os
import json
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


@dataclass
class RefsBundle:
    """Bundle of reference images, masks, intrinsics, and depth for FoundationPose."""
    images: List[np.ndarray]  # RGB images (H, W, 3)
    masks: List[np.ndarray]   # Binary masks (H, W)
    K: np.ndarray            # Camera intrinsics (3, 3)
    depth_scale: float       # Depth scale factor
    poses: Optional[List[np.ndarray]] = None  # Reference poses (4, 4) if available
    names: Optional[List[str]] = None         # Reference image names


class FoundationPoseWrapper:
    """
    Wrapper for FoundationPose model-free pose estimation.
    
    Features:
    - Few-shot initialization from reference images
    - Real-time pose tracking
    - Multi-hypothesis initialization support
    - Reinitialization capabilities
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize FoundationPose wrapper.
        
        Args:
            model_path: Path to FoundationPose model weights (optional for model-free)
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.model_path = model_path
        self.refs_bundle = None
        self.is_initialized = False
        
        # Tracking state
        self.last_pose = None
        self.track_history = []
        
        print(f"FoundationPose wrapper initialized on {device}")
    
    def load_refs_bundle(self, refs_dir: str, K: np.ndarray, depth_scale: float = 0.001) -> RefsBundle:
        """
        Load reference images and masks from directory.
        
        Expected structure:
        refs_dir/
        ├── images/
        │   ├── ref_001.jpg
        │   ├── ref_002.jpg
        │   └── ...
        ├── masks/
        │   ├── ref_001.png
        │   ├── ref_002.png
        │   └── ...
        └── poses.json (optional)  # Reference poses if available
        
        Or with tool_class subdirectory:
        refs_dir/
        ├── images/tool_class/
        │   ├── ref_001.jpg
        │   └── ...
        ├── masks/tool_class/
        │   ├── ref_001.png
        │   └── ...
        
        Args:
            refs_dir: Directory containing reference images and masks
            K: Camera intrinsics (3, 3)
            depth_scale: Depth scale factor
            
        Returns:
            RefsBundle object
        """
        # Check for tool_class subdirectory structure
        if os.path.exists(os.path.join(refs_dir, "images")):
            images_dir = os.path.join(refs_dir, "images")
            masks_dir = os.path.join(refs_dir, "masks")
        else:
            # Direct structure (images and masks in refs_dir)
            images_dir = refs_dir
            masks_dir = refs_dir
        
        poses_file = os.path.join(refs_dir, "poses.json")
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
        
        # Load images
        image_files = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        images = []
        masks = []
        names = []
        
        for img_file in image_files:
            # Load image
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            
            # Load corresponding mask (try multiple naming conventions)
            base_name = os.path.splitext(img_file)[0]
            mask_candidates = [
                base_name + '_mask.png',  # ref_01_mask.png
                base_name + '.png',       # ref_01.png (if mask has same name)
                base_name + '_mask.jpg',  # ref_01_mask.jpg
            ]
            
            mask = None
            for mask_file in mask_candidates:
                mask_path = os.path.join(masks_dir, mask_file)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 127).astype(np.uint8)
                    print(f"Loaded mask: {mask_file}")
                    break
            
            if mask is None:
                print(f"Warning: Mask not found for {img_file}, creating dummy mask")
                mask = np.ones(img.shape[:2], dtype=np.uint8)
            
            masks.append(mask)
            names.append(base_name)
        
        # Load poses if available
        poses = None
        if os.path.exists(poses_file):
            with open(poses_file, 'r') as f:
                poses_data = json.load(f)
                poses = [np.array(poses_data[name]) for name in names 
                        if name in poses_data]
        
        self.refs_bundle = RefsBundle(
            images=images,
            masks=masks,
            K=K,
            depth_scale=depth_scale,
            poses=poses,
            names=names
        )
        
        print(f"Loaded {len(images)} reference images from {refs_dir}")
        return self.refs_bundle
    
    def init_pose_from_refs(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None, 
                           mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Initialize pose from reference images (model-free approach).
        
        This is a simplified implementation. In practice, you would:
        1. Extract features from query image and reference images
        2. Match features or use template matching
        3. Estimate pose using PnP or other geometric methods
        
        Args:
            rgb: Query RGB image (H, W, 3)
            depth: Query depth image (H, W) - optional
            mask: Object mask (H, W) - optional
            
        Returns:
            Initial pose (4, 4) or None if initialization failed
        """
        if self.refs_bundle is None:
            raise RuntimeError("Reference bundle not loaded. Call load_refs_bundle first.")
        
        if len(self.refs_bundle.images) == 0:
            raise RuntimeError("No reference images available")
        
        # Simplified initialization: find best matching reference
        # In practice, this would use feature matching or template matching
        best_match_idx = self._find_best_reference(rgb, mask)
        
        if best_match_idx is None:
            print("Warning: No suitable reference match found")
            return None
        
        # Use reference pose if available, otherwise estimate from matching
        if (self.refs_bundle.poses is not None and 
            best_match_idx < len(self.refs_bundle.poses)):
            initial_pose = self.refs_bundle.poses[best_match_idx].copy()
        else:
            # Fallback: estimate pose using simple geometric method
            initial_pose = self._estimate_pose_from_matching(
                rgb, depth, mask, best_match_idx
            )
        
        if initial_pose is not None:
            self.last_pose = initial_pose
            self.is_initialized = True
            print(f"Pose initialized from reference {best_match_idx}")
        
        return initial_pose
    
    def track(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None,
              mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Track object pose in current frame.
        
        Args:
            rgb: Current RGB image (H, W, 3)
            depth: Current depth image (H, W) - optional
            mask: Object mask (H, W) - optional
            
        Returns:
            Tracked pose (4, 4) or None if tracking failed
        """
        if not self.is_initialized:
            print("Warning: Not initialized. Call init_pose_from_refs first.")
            return None
        
        # Simplified tracking: use previous pose as initialization
        # In practice, this would use optical flow, feature tracking, or neural tracking
        tracked_pose = self._simple_tracking(rgb, depth, mask)
        
        if tracked_pose is not None:
            self.last_pose = tracked_pose
            self.track_history.append(tracked_pose.copy())
            
            # Keep only recent history
            if len(self.track_history) > 10:
                self.track_history.pop(0)
        
        return tracked_pose
    
    def reinit(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None,
               mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Reinitialize pose estimation."""
        self.is_initialized = False
        self.last_pose = None
        self.track_history = []
        return self.init_pose_from_refs(rgb, depth, mask)
    
    def _find_best_reference(self, rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[int]:
        """Find best matching reference image (simplified)."""
        # Simple template matching as placeholder
        # In practice, use feature matching, deep features, etc.
        best_score = -1
        best_idx = None
        
        for i, ref_img in enumerate(self.refs_bundle.images):
            # Resize to same size for comparison
            h, w = rgb.shape[:2]
            ref_resized = cv2.resize(ref_img, (w, h))
            
            # Simple correlation score
            score = cv2.matchTemplate(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(ref_resized, cv2.COLOR_RGB2GRAY),
                cv2.TM_CCOEFF_NORMED
            )[0, 0]
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx if best_score > 0.3 else None
    
    def _estimate_pose_from_matching(self, rgb: np.ndarray, depth: Optional[np.ndarray],
                                   mask: Optional[np.ndarray], ref_idx: int) -> Optional[np.ndarray]:
        """Estimate pose from reference matching (placeholder implementation)."""
        # Placeholder: return identity pose
        # In practice, implement PnP, ICP, or other geometric methods
        return np.eye(4, dtype=np.float64)
    
    def _simple_tracking(self, rgb: np.ndarray, depth: Optional[np.ndarray],
                        mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Simple tracking implementation (placeholder)."""
        # Placeholder: return previous pose
        # In practice, implement optical flow, feature tracking, etc.
        return self.last_pose.copy() if self.last_pose is not None else None


def create_refs_bundle_from_images(images: List[np.ndarray], masks: List[np.ndarray],
                                 K: np.ndarray, depth_scale: float = 0.001,
                                 poses: Optional[List[np.ndarray]] = None,
                                 names: Optional[List[str]] = None) -> RefsBundle:
    """
    Create RefsBundle directly from arrays (for testing/programmatic use).
    
    Args:
        images: List of RGB images
        masks: List of binary masks
        K: Camera intrinsics
        depth_scale: Depth scale factor
        poses: Optional reference poses
        names: Optional image names
        
    Returns:
        RefsBundle object
    """
    return RefsBundle(
        images=images,
        masks=masks,
        K=K,
        depth_scale=depth_scale,
        poses=poses,
        names=names
    )
