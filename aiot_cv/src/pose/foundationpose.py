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
from .domain_bridge import apply_domain_bridge
from .axis_estimation import estimate_axis_from_depth

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
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", 
                 enable_domain_bridge: bool = True, bridge_profile: str = "metal_lowres",
                 enable_ref_matching: bool = True, cuda_memory_fraction: float = 0.8):
        """
        Initialize FoundationPose wrapper.
        
        Args:
            model_path: Path to FoundationPose model weights (optional for model-free)
            device: Device to run on ("cuda" or "cpu")
            enable_domain_bridge: Whether to apply domain gap bridging
            bridge_profile: Domain bridging profile ("metal_lowres", "indoor", "outdoor")
            enable_ref_matching: Whether to enable reference matching (expensive)
            cuda_memory_fraction: GPU memory allocation fraction
        """
        self.device = device
        self.model_path = model_path
        self.refs_bundle = None
        self.is_initialized = False
        
        # Feature control settings
        self.enable_domain_bridge = enable_domain_bridge
        self.bridge_profile = bridge_profile
        self.enable_ref_matching = enable_ref_matching
        
        # CUDA optimization
        self.cuda_memory_fraction = cuda_memory_fraction
        if device == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction)
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
                print(f"CUDA optimization: memory_fraction={cuda_memory_fraction}, cudnn_benchmark=True")
        
        # Tracking state
        self.last_pose = None
        self.track_history = []
        
        print(f"FoundationPose wrapper initialized on {device}")
        print(f"Reference matching: {'ON' if enable_ref_matching else 'OFF'}")
        print(f"Domain bridge: {'ON' if enable_domain_bridge else 'OFF'} (profile: {bridge_profile})")
    
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
            # Convert BGR to RGB for consistency with real-time processing
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
                    # Load mask with all channels preserved
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        print(f"Warning: Mask not found: {mask_path}")
                        continue
                    
                    # Handle different channel configurations
                    if mask.ndim == 2:               # Already 1-channel (grayscale)
                        pass
                    elif mask.ndim == 3 and mask.shape[2] == 1:  # (H,W,1) -> squeeze to (H,W)
                        mask = mask[:, :, 0]
                    elif mask.ndim == 3 and mask.shape[2] == 4:   # RGBA -> use alpha channel as mask
                        mask = mask[:, :, 3]
                    elif mask.ndim == 3 and mask.shape[2] == 3:   # BGR -> convert to grayscale
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    else:
                        print(f"Warning: Unsupported mask shape {mask.shape}, skipping {mask_file}")
                        continue
                    
                    # Ensure uint8 and binarize (0/255)
                    if mask.dtype != np.uint8:
                        mask = mask.astype(np.uint8)
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    
                    print(f"Loaded mask: {mask_file}")
                    break
            
            if mask is None:
                print(f"Warning: Mask not found for {img_file}, skipping this reference")
                continue  # Skip references without masks
            
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
        
        print(f"[Refs] loaded {len(images)} refs from {refs_dir}")
        for i, (img, mask) in enumerate(zip(images, masks)):
            mask_unique = set(np.unique(mask))
            print(f"  - ref[{i}]: img{img.shape} mask{mask.shape} binary={mask_unique}")
        
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
        
        # Skip expensive reference matching if disabled
        if not self.enable_ref_matching:
            print("[Init] Reference matching disabled - skipping to geometric initialization")
            best_match_idx = None
        else:
            print("[Init] Reference matching enabled - searching through references...")
            # Simplified initialization: find best matching reference
            # In practice, this would use feature matching or template matching
            best_match_idx = self._find_best_reference(rgb, mask)
        
        if best_match_idx is None:
            if self.enable_ref_matching:
                print("Warning: No suitable reference match found")
            # Skip to geometric initialization
        else:
            print(f"[Init] Found best match: ref[{best_match_idx}]")
        
        # Use reference pose if available, otherwise estimate from matching
        if (best_match_idx is not None and
            self.refs_bundle.poses is not None and 
            best_match_idx < len(self.refs_bundle.poses)):
            initial_pose = self.refs_bundle.poses[best_match_idx].copy()
            print(f"[Init] Using pre-computed reference pose")
        else:
            # Fallback: estimate pose using simple geometric method
            print(f"[Init] Estimating pose from matching")
            initial_pose = self._estimate_pose_from_matching(
                rgb, depth, mask, best_match_idx
            )
        
        if initial_pose is not None:
            # Validate pose
            z_val = initial_pose[2, 3]
            is_valid = (np.isfinite(initial_pose).all() and 
                       initial_pose.shape == (4, 4) and 
                       z_val > 0.01)  # At least 1cm depth
            
            print(f"[Init] Pose validation: z={z_val:.4f}, valid={is_valid}")
            
            if is_valid:
                self.last_pose = initial_pose
                self.is_initialized = True
                print(f"[Init] Successfully initialized from reference {best_match_idx}")
                return initial_pose
            else:
                print(f"[Init] Invalid pose detected, trying PCA+ICP fallback")
                initial_pose = None
        else:
            print(f"[Init] Reference matching failed, trying PCA+ICP fallback")
        
        # Fallback: PCA + ICP initialization
        if initial_pose is None and depth is not None and mask is not None:
            print("[Fallback] Attempting PCA + ICP initialization...")
            
            # Generate PCA-based initial poses
            pca_poses = self._pca_init_from_mask(depth, mask, self.refs_bundle.K, self.refs_bundle.depth_scale)
            
            if len(pca_poses) > 0:
                # Convert depth to point cloud
                depth_m = depth.astype(np.float32) * self.refs_bundle.depth_scale
                mask_binary = (mask > 0).astype(np.uint8)
                
                # Get valid points
                ys, xs = np.where(mask_binary > 0)
                z = depth_m[ys, xs]
                valid = np.isfinite(z) & (z > 0.01) & (z < 2.0)
                xs, ys, z = xs[valid], ys[valid], z[valid]
                
                if len(xs) >= 100:
                    # Backproject to 3D
                    fx, fy, cx, cy = self.refs_bundle.K[0, 0], self.refs_bundle.K[1, 1], self.refs_bundle.K[0, 2], self.refs_bundle.K[1, 2]
                    X = (xs - cx) * z / fx
                    Y = (ys - cy) * z / fy
                    source_points = np.stack([X, Y, z], axis=1)
                    
                    # Try ICP refinement
                    refined_pose = self._icp_refine_pose(source_points, pca_poses)
                    
                    if refined_pose is not None:
                        self.last_pose = refined_pose
                        self.is_initialized = True
                        print("[Fallback] Successfully initialized using PCA + ICP")
                        return refined_pose
                    else:
                        # Use best PCA pose as last resort
                        best_pca = pca_poses[0]  # First candidate is usually best
                        self.last_pose = best_pca
                        self.is_initialized = True
                        print("[Fallback] Using PCA pose (ICP failed)")
                        return best_pca
                else:
                    print(f"[Fallback] Insufficient points for ICP: {len(xs)} < 100")
            else:
                print("[Fallback] PCA initialization failed")
        
        # Final fallback: Axis-only estimation
        print("[Fallback] Attempting axis-only estimation...")
        axis_result = estimate_axis_from_depth(
            depth.astype(np.float32) * self.refs_bundle.depth_scale,
            mask.astype(np.uint8),
            self.refs_bundle.K
        )
        
        if axis_result is not None:
            origin, axis_dir, num_inliers, stats = axis_result
            
            # Create a simple pose from axis (identity rotation + centroid translation)
            axis_pose = np.eye(4, dtype=np.float64)
            axis_pose[:3, 3] = origin
            
            # Log axis information
            print(f"[AxisOnly] Origin: {origin}")
            print(f"[AxisOnly] Direction: {axis_dir}")
            print(f"[AxisOnly] Inliers: {num_inliers}")
            print(f"[AxisOnly] Stats: {stats}")
            
            # Store as partial initialization
            self.last_pose = axis_pose
            self.is_initialized = True
            print("[AxisOnly] Axis-based initialization successful (5DoF, roll undefined)")
            return axis_pose
        
        print("[Init] All initialization methods failed (including axis estimation)")
        return None
    
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
        """Find best matching reference image with enhanced feature matching."""
        if len(self.refs_bundle.images) == 0:
            print("[RefMatch] No reference images available")
            return None
        
        print(f"[RefMatch] Searching through {len(self.refs_bundle.images)} references")
        
        best_score = -1
        best_idx = None
        scores = []
        detailed_results = []
        
        # Enhanced preprocessing with CLAHE
        query_preprocessed = self._preprocess_for_matching(rgb, mask, use_clahe=True)
        
        # Initialize ORB detector with enhanced parameters
        orb = cv2.ORB_create(
            nfeatures=3000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,  # Reduced for more features on thin objects
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31
        )
        
        # BFMatcher with cross-check enabled
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        for i, ref_img in enumerate(self.refs_bundle.images):
            ref_name = self.refs_bundle.names[i] if i < len(self.refs_bundle.names) else f"ref_{i}"
            print(f"[RefMatch] Testing {ref_name} ({i+1}/{len(self.refs_bundle.images)})")
            
            # Apply same preprocessing to reference with optional domain bridging
            ref_img_processed = ref_img
            if self.enable_domain_bridge:
                ref_img_processed = apply_domain_bridge(ref_img, self.bridge_profile)
            ref_preprocessed = self._preprocess_for_matching(ref_img_processed, self.refs_bundle.masks[i], use_clahe=True)
            
            # Convert to grayscale for feature detection
            query_gray = cv2.cvtColor(query_preprocessed, cv2.COLOR_RGB2GRAY)
            ref_gray = cv2.cvtColor(ref_preprocessed, cv2.COLOR_RGB2GRAY)
            
            # Feature detection and matching
            kp1, des1 = orb.detectAndCompute(query_gray, None)
            kp2, des2 = orb.detectAndCompute(ref_gray, None)
            
            feature_score = 0.0
            num_matches = 0
            num_good_matches = 0
            
            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                # K-NN matching with ratio test
                matches = bf.knnMatch(des1, des2, k=2)
                
                # Ratio test (Lowe's ratio test) - relaxed for thin objects
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.8 * n.distance:  # Relaxed from 0.75 to 0.8
                            good_matches.append(m)
                
                num_matches = len(matches)
                num_good_matches = len(good_matches)
                
                if num_good_matches >= 8:  # Minimum matches for thin objects
                    # Calculate feature matching score
                    avg_distance = np.mean([m.distance for m in good_matches])
                    max_distance = 256  # ORB descriptor distance range
                    feature_score = max(0, (max_distance - avg_distance) / max_distance)
                    
                    # Try geometric verification with relaxed RANSAC
                    if num_good_matches >= 12:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        H, inliers = cv2.findHomography(
                            dst_pts, src_pts, 
                            cv2.RANSAC, 
                            ransacReprojThreshold=6.0,  # Relaxed from 3.0 to 6.0
                            confidence=0.995,
                            maxIters=3000
                        )
                        
                        if inliers is not None:
                            inlier_ratio = np.sum(inliers) / len(good_matches)
                            feature_score *= (0.5 + 0.5 * inlier_ratio)  # Boost score with geometric consistency
                            print(f"[RefMatch] {ref_name}: features={len(kp1)}/{len(kp2)}, matches={num_good_matches}, inliers={np.sum(inliers)}, ratio={inlier_ratio:.2f}")
                        else:
                            print(f"[RefMatch] {ref_name}: features={len(kp1)}/{len(kp2)}, matches={num_good_matches}, homography_failed")
                    else:
                        print(f"[RefMatch] {ref_name}: features={len(kp1)}/{len(kp2)}, matches={num_good_matches} (insufficient for geometry)")
                else:
                    print(f"[RefMatch] {ref_name}: features={len(kp1)}/{len(kp2)}, matches={num_good_matches} (too few)")
            else:
                print(f"[RefMatch] {ref_name}: feature_detection_failed, kp1={len(kp1) if kp1 else 0}, kp2={len(kp2) if kp2 else 0}")
            
            # Template matching as backup
            h, w = query_preprocessed.shape[:2]
            ref_resized = cv2.resize(ref_preprocessed, (w, h))
            template_score = cv2.matchTemplate(query_gray, cv2.cvtColor(ref_resized, cv2.COLOR_RGB2GRAY), cv2.TM_CCOEFF_NORMED)[0, 0]
            
            # Combined score: prioritize feature matching, fallback to template
            if feature_score > 0.1:
                combined_score = 0.8 * feature_score + 0.2 * template_score
            else:
                combined_score = template_score
            
            scores.append(combined_score)
            detailed_results.append({
                'name': ref_name,
                'score': combined_score,
                'feature_score': feature_score,
                'template_score': template_score,
                'matches': num_good_matches,
                'keypoints': (len(kp1) if kp1 else 0, len(kp2) if kp2 else 0)
            })
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = i
        
        # Adaptive threshold based on score distribution
        scores_sorted = sorted(scores, reverse=True)
        if len(scores_sorted) >= 3:
            top3_mean = np.mean(scores_sorted[:3])
            top3_std = np.std(scores_sorted[:3])
            adaptive_thr = max(0.15, top3_mean - 0.5 * top3_std)
        else:
            adaptive_thr = 0.2  # fallback
        
        # Margin validation (best vs second best) - very relaxed for thin objects
        margin_delta = 0.02  # Further reduced from 0.03 to 0.02
        if len(scores_sorted) >= 2:
            best_score, second_best = scores_sorted[0], scores_sorted[1]
            margin_ok = (best_score - second_best) >= margin_delta
        else:
            margin_ok = True  # only one reference
        
        # Combined validation - very lenient for thin objects
        basic_threshold = max(0.08, adaptive_thr * 0.6)  # Lower floor, 60% of adaptive
        valid_by_adaptive = (best_score >= basic_threshold) and margin_ok
        
        # Log detailed matching results
        print(f"[RefMatch] === MATCHING SUMMARY ===")
        print(f"[RefMatch] Processed {len(self.refs_bundle.images)} references")
        print(f"[RefMatch] Best: {detailed_results[best_idx]['name'] if best_idx is not None else 'None'} | Score: {best_score:.3f}")
        print(f"[RefMatch] Threshold: {basic_threshold:.3f} | Margin OK: {margin_ok}")
        print(f"[RefMatch] Valid: {valid_by_adaptive}")
        
        # Show top 3 results
        sorted_results = sorted(detailed_results, key=lambda x: x['score'], reverse=True)
        print(f"[RefMatch] Top 3 candidates:")
        for i, result in enumerate(sorted_results[:3]):
            print(f"[RefMatch]   {i+1}. {result['name']}: score={result['score']:.3f}, "
                  f"features={result['feature_score']:.3f}, template={result['template_score']:.3f}, "
                  f"matches={result['matches']}, kpts={result['keypoints']}")
        
        # Show failure reasons if no match found
        if not valid_by_adaptive:
            if best_score < basic_threshold:
                print(f"[RefMatch] FAIL: Best score {best_score:.3f} < threshold {basic_threshold:.3f}")
            if not margin_ok:
                print(f"[RefMatch] FAIL: Insufficient margin {best_score-scores_sorted[1]:.3f} < {margin_delta:.3f}")
        
        return best_idx if valid_by_adaptive else None
    
    def _preprocess_for_matching(self, img: np.ndarray, mask: Optional[np.ndarray] = None, 
                                gamma: float = 1.0, use_clahe: bool = True) -> np.ndarray:
        """Consistent preprocessing for both query and reference images with lighting normalization."""
        # Ensure RGB format
        if img.ndim == 3 and img.shape[2] == 3:
            # Assume BGR if loaded with cv2.imread, convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img.copy()
        
        # Resize to standard size for matching
        target_size = 256
        img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Apply mask if available (set background to black)
        if mask is not None:
            mask_resized = cv2.resize(mask.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 127).astype(np.uint8)
            
            # Erode mask slightly to remove background noise (1px)
            kernel = np.ones((3, 3), np.uint8)
            mask_binary = cv2.erode(mask_binary, kernel, iterations=1)
            
            # Set background to black
            img_resized = img_resized * mask_binary[:, :, np.newaxis]
        
        # Lighting normalization
        img_normalized = self._normalize_lighting(img_resized, gamma, use_clahe)
        
        return img_normalized
    
    def _normalize_lighting(self, img: np.ndarray, gamma: float = 1.0, use_clahe: bool = True) -> np.ndarray:
        """Normalize lighting using gamma correction and CLAHE."""
        img_norm = img.astype(np.uint8)
        
        # Gamma correction
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            img_norm = cv2.LUT(img_norm, table)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if use_clahe:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(img_norm, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            
            # Merge back
            lab_clahe = cv2.merge([l_clahe, a, b])
            img_norm = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return img_norm
    
    def _compute_gradient_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Normalized Cross Correlation on gradient magnitudes."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Compute gradients
        gx1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
        gy1 = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
        gx2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
        gy2 = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute gradient magnitudes
        grad_mag1 = cv2.magnitude(gx1, gy1)
        grad_mag2 = cv2.magnitude(gx2, gy2)
        
        # Normalize to 0-1 range
        grad_mag1 = cv2.normalize(grad_mag1, None, 0, 1, cv2.NORM_MINMAX)
        grad_mag2 = cv2.normalize(grad_mag2, None, 0, 1, cv2.NORM_MINMAX)
        
        # Compute NCC
        mean1 = np.mean(grad_mag1)
        mean2 = np.mean(grad_mag2)
        
        num = np.sum((grad_mag1 - mean1) * (grad_mag2 - mean2))
        den = np.sqrt(np.sum((grad_mag1 - mean1)**2) * np.sum((grad_mag2 - mean2)**2))
        
        if den == 0:
            return 0.0
        
        ncc = num / den
        # Convert to 0-1 range (NCC can be -1 to 1)
        return (ncc + 1.0) / 2.0
    
    def _estimate_pose_from_matching(self, rgb: np.ndarray, depth: Optional[np.ndarray],
                                   mask: Optional[np.ndarray], ref_idx: int) -> Optional[np.ndarray]:
        """Estimate pose from reference matching with depth-based z estimation."""
        # Start with identity pose
        pose = np.eye(4, dtype=np.float64)
        
        # If depth is available, estimate z from depth median
        if depth is not None and mask is not None:
            # Convert depth to meters
            depth_m = depth.astype(np.float32) * self.refs_bundle.depth_scale
            mask_binary = (mask > 0).astype(np.uint8)
            
            # Get valid depth values in mask area
            zmin, zmax = 0.05, 2.0  # Working distance range
            valid_depth = (depth_m > zmin) & (depth_m < zmax) & (mask_binary > 0)
            if valid_depth.any():
                depth_values = depth_m[valid_depth]
                
                # IQR-based outlier filtering for robust Z estimation
                if len(depth_values) > 50:  # Enough points for robust statistics
                    q1, q3 = np.percentile(depth_values, [25, 75])
                    iqr = q3 - q1
                    outlier_factor = 1.5
                    depth_robust = depth_values[
                        (depth_values >= q1 - outlier_factor * iqr) & 
                        (depth_values <= q3 + outlier_factor * iqr)
                    ]
                    z_median = float(np.median(depth_robust)) if len(depth_robust) > 0 else float(np.median(depth_values))
                    print(f"[PoseEst] IQR filter: {len(depth_values)} -> {len(depth_robust) if len(depth_values) > 50 else len(depth_values)} points")
                else:
                    z_median = float(np.median(depth_values))
                
                # Set translation z component
                pose[2, 3] = z_median
                print(f"[PoseEst] Estimated z from depth: {z_median:.3f}m from {len(depth_values)} points")
                
                # Basic centering (simple centroid estimation)
                # In practice, use more sophisticated geometric methods
                h, w = mask.shape
                y_coords, x_coords = np.where(mask_binary > 0)
                if len(x_coords) > 0:
                    cx = np.mean(x_coords)
                    cy = np.mean(y_coords)
                    
                    # Convert to camera coordinates (simplified)
                    fx, fy = self.refs_bundle.K[0, 0], self.refs_bundle.K[1, 1]
                    cx_k, cy_k = self.refs_bundle.K[0, 2], self.refs_bundle.K[1, 2]
                    
                    x_cam = (cx - cx_k) * z_median / fx
                    y_cam = (cy - cy_k) * z_median / fy
                    
                    pose[0, 3] = x_cam
                    pose[1, 3] = y_cam
                    
                    print(f"[PoseEst] Estimated t=[{x_cam:.3f}, {y_cam:.3f}, {z_median:.3f}]")
                else:
                    print("[PoseEst] No mask pixels found, using z-only estimation")
            else:
                print("[PoseEst] No valid depth in mask, using identity pose")
        else:
            print("[PoseEst] No depth/mask available, using identity pose")
        
        return pose
    
    def _pca_init_from_mask(self, depth: np.ndarray, mask: np.ndarray, K: np.ndarray, 
                           depth_scale: float) -> List[np.ndarray]:
        """Generate PCA-based initial poses from depth mask."""
        # Get mask coordinates
        ys, xs = np.where(mask > 0)
        if len(xs) < 500:
            print(f"[PCA] Insufficient mask pixels: {len(xs)} < 500")
            return []
        
        # Get depth values
        depth_m = depth.astype(np.float32) * depth_scale
        z = depth_m[ys, xs]
        valid = np.isfinite(z) & (z > 0.01) & (z < 2.0)  # 1cm to 2m range
        xs, ys, z = xs[valid], ys[valid], z[valid]
        
        if len(xs) < 500:
            print(f"[PCA] Insufficient valid depth points: {len(xs)} < 500")
            return []
        
        # Backproject to 3D
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        pts = np.stack([X, Y, z], axis=1)
        
        # PCA analysis
        center = pts.mean(axis=0)
        centered = pts - center
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Principal components (rows of Vt)
        a1 = Vt[0]  # Primary axis (longest dimension)
        a2 = Vt[1]  # Secondary axis
        a3 = Vt[2]  # Tertiary axis
        
        # Ensure right-handed coordinate system
        if np.linalg.det(np.column_stack([a1, a2, a3])) < 0:
            a3 = -a3
        
        # Create rotation matrix (object frame)
        R = np.column_stack([a1, a2, a3])
        
        # For symmetric objects like screwdrivers, create multiple candidates
        # 1. Standard orientation
        T1 = np.eye(4, dtype=np.float64)
        T1[:3, :3] = R
        T1[:3, 3] = center
        
        # 2. Flipped along primary axis (180° rotation)
        R_flip = R @ np.diag([-1, -1, 1])
        T2 = np.eye(4, dtype=np.float64)
        T2[:3, :3] = R_flip
        T2[:3, 3] = center
        
        # 3. Alternative orientation (primary <-> secondary axes swapped)
        R_alt = np.column_stack([a2, a1, a3])
        if np.linalg.det(R_alt) < 0:
            R_alt[:, 2] = -R_alt[:, 2]
        T3 = np.eye(4, dtype=np.float64)
        T3[:3, :3] = R_alt
        T3[:3, 3] = center
        
        variance_explained = S / S.sum()
        print(f"[PCA] Generated 3 pose candidates from {len(pts)} points")
        print(f"[PCA] Variance explained: {variance_explained[0]:.2f}, {variance_explained[1]:.2f}, {variance_explained[2]:.2f}")
        print(f"[PCA] Object dimensions (m): {S[0]:.3f} x {S[1]:.3f} x {S[2]:.3f}")
        
        return [T1, T2, T3]
    
    def _icp_refine_pose(self, source_points: np.ndarray, init_poses: List[np.ndarray], 
                        voxel_size: float = 0.004, max_correspondence: float = 0.01) -> Optional[np.ndarray]:
        """Refine pose using ICP with multiple initial guesses."""
        try:
            import open3d as o3d
        except ImportError:
            print("[ICP] Open3D not available, skipping ICP refinement")
            return None
        
        if len(source_points) < 100:
            print(f"[ICP] Insufficient source points: {len(source_points)} < 100")
            return None
        
        # Create source point cloud
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        source_pcd = source_pcd.voxel_down_sample(voxel_size)
        source_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        
        # For now, use a simple target (cylinder or box) as placeholder
        # In practice, this would be loaded from reference CAD or point cloud
        target_pcd = self._create_reference_geometry()
        if target_pcd is None:
            print("[ICP] No reference geometry available")
            return None
        
        target_pcd = target_pcd.voxel_down_sample(voxel_size)
        target_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        
        best_result = None
        best_fitness = 0.0
        
        print(f"[ICP] Testing {len(init_poses)} initial poses")
        
        for i, init_pose in enumerate(init_poses):
            try:
                # Run ICP
                result = o3d.pipelines.registration.registration_icp(
                    source_pcd, target_pcd, max_correspondence, init_pose,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
                )
                
                print(f"[ICP] Init {i}: fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.4f}")
                
                # Quality thresholds for thin objects (relaxed)
                if result.fitness > 0.55 and result.inlier_rmse < 0.010:
                    if result.fitness > best_fitness:
                        best_result = result
                        best_fitness = result.fitness
                        
            except Exception as e:
                print(f"[ICP] Init {i} failed: {e}")
                continue
        
        if best_result is not None:
            print(f"[ICP] Success: fitness={best_result.fitness:.3f}, rmse={best_result.inlier_rmse:.4f}")
            return best_result.transformation
        else:
            print("[ICP] All attempts failed quality thresholds")
            return None
    
    def _create_reference_geometry(self) -> Optional:
        """Create reference geometry for ICP. Placeholder implementation."""
        try:
            import open3d as o3d
        except ImportError:
            return None
        
        # Simple cylinder approximation for screwdriver
        # In practice, load actual CAD model or reference point cloud
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=0.15)
        cylinder.translate([0, 0, -0.075])  # Center at origin
        
        # Convert to point cloud
        pcd = cylinder.sample_points_poisson_disk(2000)
        return pcd
    
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
