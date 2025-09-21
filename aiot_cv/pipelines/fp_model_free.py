"""
End-to-end FoundationPose pipeline for model-free pose estimation.
Camera -> YOLO Detection/Segmentation -> ROI -> FoundationPose -> Track/Smooth
"""

import argparse
import time
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Import our modules
from aiot_cv.src.detect.yolo import YoloSeg, Detection
from aiot_cv.src.pose.foundationpose import FoundationPoseWrapper
from aiot_cv.src.track.smoother import RealtimePCASmoother, PoseFilterConfig
from aiot_cv.src.pc.pointcloud import rgbd_to_pcl, filter_pointcloud
from aiot_cv.src.io.load_k import load_K


class FoundationPosePipeline:
    """
    End-to-end FoundationPose pipeline for model-free pose estimation.
    
    Pipeline:
    1. Load RGB-D images (camera or files)
    2. Run YOLO detection/segmentation
    3. Extract ROIs and masks
    4. Initialize FoundationPose from reference images
    5. Track pose with real-time smoothing
    6. Log performance metrics (FPS, ADD-S error)
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = self._load_config(config_path)
        self.detector = None
        self.fp_wrapper = None
        self.smoother = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.pose_history = []
        self.fps_history = []
        
        # State
        self.is_initialized = False
        self.current_pose = None
        
        print(f"FoundationPose pipeline initialized with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set defaults
        defaults = {
            'yolo': {
                'weights': None,
                'conf': 0.5,
                'imgsz': 640
            },
            'foundationpose': {
                'device': 'cuda',
                'model_path': None
            },
            'smoothing': {
                'window_size': 5,
                'max_angular_velocity': 2.0,
                'max_translation_velocity': 1.0,
                'outlier_threshold': 0.1
            },
            'camera': {
                'camera_yaml': 'aiot_cv/configs/camera.yaml'
            },
            'references': {
                'refs_dir': 'aiot_cv/data/references',
                'tool_class': 'screwdriver'
            }
        }
        
        # Merge with defaults
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
            elif isinstance(default_value, dict):
                for subkey, subdefault in default_value.items():
                    if subkey not in config[key]:
                        config[key][subkey] = subdefault
        
        return config
    
    def setup(self):
        """Setup all components of the pipeline."""
        print("Setting up pipeline components...")
        
        # Load camera intrinsics
        self.K, self.depth_scale, _ = load_K(self.config['camera']['camera_yaml'])
        print(f"Camera intrinsics loaded: K shape {self.K.shape}")
        
        # Initialize YOLO detector
        self.detector = YoloSeg(
            weights=self.config['yolo']['weights'],
            conf=self.config['yolo']['conf'],
            imgsz=self.config['yolo']['imgsz']
        )
        print("YOLO detector initialized")
        
        # Initialize FoundationPose wrapper
        self.fp_wrapper = FoundationPoseWrapper(
            model_path=self.config['foundationpose']['model_path'],
            device=self.config['foundationpose']['device']
        )
        
        # Load reference bundle
        refs_dir = self.config['references']['refs_dir']
        if not Path(refs_dir).exists():
            raise FileNotFoundError(f"Reference directory not found: {refs_dir}")
        
        self.fp_wrapper.load_refs_bundle(refs_dir, self.K, self.depth_scale)
        print(f"Reference bundle loaded from {refs_dir}")
        
        # Initialize pose smoother
        smoother_config = PoseFilterConfig(**self.config['smoothing'])
        self.smoother = RealtimePCASmoother(smoother_config)
        print("Pose smoother initialized")
        
        print("Pipeline setup complete!")
    
    def process_frame(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Process a single frame through the pipeline.
        
        Args:
            rgb: RGB image (H, W, 3)
            depth: Depth image (H, W) - optional
            
        Returns:
            Estimated pose (4, 4) or None
        """
        if not self.is_initialized:
            self.setup()
            self.is_initialized = True
        
        # Start timing
        frame_start_time = time.time()
        
        # Step 1: YOLO detection
        detections = self.detector(rgb, depth)
        
        # Class filter - only allow screwdriver for now
        allowed_classes = {"screwdriver"}
        detections = [d for d in detections if d.class_name.lower() in allowed_classes]
        
        if not detections:
            print("No detections found (after class filter)")
            return None
        
        # Get best detection (highest confidence)
        detection = detections[0]
        print(f"Best detection: {detection.class_name} (conf={detection.confidence:.3f})")
        
        # Step 2: Auto-switch references based on detected class
        tool = detection.class_name.lower()
        tool_dir = Path(self.config['references']['refs_dir']) / tool
        
        if tool_dir.exists():
            if getattr(self, "_loaded_tool", None) != tool:
                print(f"[Refs] Switching to: {tool_dir}")
                self.fp_wrapper.load_refs_bundle(str(tool_dir), self.K, self.depth_scale)
                self._loaded_tool = tool
        else:
            print(f"[Refs] Missing directory: {tool_dir}")
            # Try fallback to generic references
            generic_dir = Path(self.config['references']['refs_dir'])
            if generic_dir.exists() and getattr(self, "_loaded_tool", None) != "generic":
                print(f"[Refs] Using generic references: {generic_dir}")
                self.fp_wrapper.load_refs_bundle(str(generic_dir), self.K, self.depth_scale)
                self._loaded_tool = "generic"
        
        # Step 3: Extract ROI and mask
        roi_rgb = detection.roi_rgb
        roi_depth = detection.roi_depth
        mask = detection.mask
        
        if roi_rgb is None:
            print("No ROI RGB available")
            return None
        
        # Convert BGR to RGB for FoundationPose
        roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2RGB)
        
        # Safe mask booleanization (handle different mask formats)
        if mask is not None:
            if mask.dtype == np.uint8:
                mask = mask > 0
            else:
                mask = mask > 0.5
        
        # Resize ROI to standard size for better matching
        target_size = 256
        h, w = roi_rgb.shape[:2]
        
        if roi_rgb.shape[:2] != (target_size, target_size):
            roi_rgb = cv2.resize(roi_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        if roi_depth is not None and roi_depth.shape != (target_size, target_size):
            roi_depth = cv2.resize(roi_depth, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        if mask is not None and mask.shape != (target_size, target_size):
            mask = cv2.resize(mask.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Ensure all arrays have consistent shapes
        if roi_depth is not None and mask is not None:
            if roi_depth.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), roi_depth.shape[:2], interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Step 4: FoundationPose estimation
        if self.current_pose is None:
            # Initialize pose from reference images
            print("Initializing pose from references...")
            
            # Debug: Save ROI and mask for analysis
            dbg_dir = Path("debug_roi")
            dbg_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(dbg_dir / f"roi_rgb_{self.frame_count:06d}.png"), 
                       cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
            if roi_depth is not None:
                np.save(str(dbg_dir / f"roi_depth_{self.frame_count:06d}.npy"), roi_depth)
            if mask is not None:
                cv2.imwrite(str(dbg_dir / f"roi_mask_{self.frame_count:06d}.png"), 
                           (mask.astype(np.uint8) * 255))
            print(f"[Debug] Saved ROI to {dbg_dir}")
            
            pose = self.fp_wrapper.init_pose_from_refs(roi_rgb, roi_depth, mask)
            
            # Fallback: Coarse initialization from point cloud if reference matching fails
            if pose is None and roi_depth is not None and mask is not None:
                print("[Fallback] Attempting coarse initialization from point cloud...")
                
                # Debug: Check valid depth points
                valid = (roi_depth > 0)
                if mask is not None:
                    valid = valid & mask
                nz = int(valid.sum())
                print(f"[Debug] Valid depth points in ROI: {nz}")
                
                if nz < 500:
                    print(f"[Fallback] Insufficient valid points ({nz} < 500)")
                    return self.current_pose
                
                try:
                    # Generate point cloud from depth and mask
                    from aiot_cv.src.pc.pointcloud import rgbd_to_pcl, filter_pointcloud
                    pcl = rgbd_to_pcl(roi_rgb, roi_depth, self.K, self.depth_scale, mask)
                    if pcl is not None and len(pcl) > 500:
                        print(f"[Debug] Generated point cloud with {len(pcl)} points")
                        pcl = filter_pointcloud(pcl)
                        print(f"[Debug] After filtering: {len(pcl)} points")
                        
                        if len(pcl) > 100:
                            # Simple PCA-based initialization
                            centroid = np.mean(pcl, axis=0)
                            centered_pcl = pcl - centroid
                            _, _, Vt = np.linalg.svd(centered_pcl)
                            R0 = Vt.T
                            if np.linalg.det(R0) < 0:
                                R0[:, 2] *= -1
                            
                            pose = np.eye(4, dtype=np.float64)
                            pose[:3, :3] = R0
                            pose[:3, 3] = centroid
                            print(f"[Fallback] Coarse pose initialized from {len(pcl)} points")
                        else:
                            print("[Fallback] Insufficient points after filtering")
                    else:
                        print("[Fallback] Insufficient point cloud data")
                except Exception as e:
                    print(f"[Fallback] Point cloud initialization failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Track pose
            print("Tracking pose...")
            pose = self.fp_wrapper.track(roi_rgb, roi_depth, mask)
        
        # Validate pose
        if pose is None or not np.all(np.isfinite(pose)):
            print("FoundationPose estimation failed or invalid pose")
            return self.current_pose  # Return previous pose if available
        
        # Step 4: Apply smoothing
        smoothed_pose = self.smoother.update(pose, frame_start_time)
        
        # Validate smoothed pose
        if smoothed_pose is not None and np.all(np.isfinite(smoothed_pose)):
            self.current_pose = smoothed_pose
            
            # Log pose
            self.pose_history.append({
                'timestamp': frame_start_time,
                'pose': smoothed_pose.tolist(),
                'detection_confidence': detection.confidence
            })
            
            # Compute and log FPS
            if self.start_time is None:
                self.start_time = frame_start_time
            else:
                fps = self.frame_count / (frame_start_time - self.start_time)
                self.fps_history.append(fps)
                print(f"FPS: {fps:.2f}")
        else:
            print("Smoothed pose is invalid, keeping previous pose")
        
        # Update frame count
        self.frame_count += 1
        
        return self.current_pose
    
    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Process a video file through the pipeline.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {width}x{height} @ {fps} FPS")
        
        # Setup output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                print(f"Processing frame {frame_idx}...")
                
                # Process frame (assume no depth for now)
                pose = self.process_frame(frame)
                
                # Visualize results
                if pose is not None:
                    frame = self._draw_pose_axes(frame, pose)
                
                # Write output frame
                if writer is not None:
                    writer.write(frame)
                
                frame_idx += 1
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
        
        print(f"Processed {frame_idx} frames")
        self._save_results(output_path)
    
    def process_realsense(self, output_path: Optional[str] = None):
        """
        Process RealSense camera stream in real-time.
        
        Args:
            output_path: Path to save output video (optional)
        """
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("Error: pyrealsense2 not installed. Install with: pip install pyrealsense2")
            return
        
        # Configure RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable color and depth streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start streaming
        try:
            profile = pipeline.start(config)
            print("RealSense camera started successfully")
            
            # Get camera intrinsics from RealSense
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            color_intrinsics = color_profile.get_intrinsics()
            
            # Update camera matrix from RealSense
            self.K = np.array([
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            print(f"RealSense intrinsics: fx={color_intrinsics.fx:.1f}, fy={color_intrinsics.fy:.1f}")
            print(f"Depth scale: {self.depth_scale}")
            
        except Exception as e:
            print(f"Failed to start RealSense camera: {e}")
            return
        
        # Setup output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
            print(f"Recording to: {output_path}")
        
        # Create align object to align depth to color
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # Setup depth filters for better quality
        decimation = rs.decimation_filter(2)  # Reduce resolution, reduce noise
        spatial = rs.spatial_filter()         # Spatial filtering
        temporal = rs.temporal_filter()       # Temporal filtering
        hole_fill = rs.hole_filling_filter(1) # Fill holes
        threshold = rs.threshold_filter(min=0.15, max=1.2)  # Depth range filter
        
        frame_count = 0
        
        try:
            print("Starting RealSense processing. Press 'q' to quit, 'r' to reset tracking.")
            
            while True:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                
                # Align depth to color
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Apply depth filters
                depth_frame = decimation.process(depth_frame)
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_frame = hole_fill.process(depth_frame)
                depth_frame = threshold.process(depth_frame)
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Process frame through pipeline
                pose = self.process_frame(color_image, depth_image)
                
                # Draw visualization
                if pose is not None:
                    color_image = self._draw_pose_axes(color_image, pose)
                    # Draw info text
                    cv2.putText(color_image, f"Frame: {frame_count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(color_image, "Pose: OK", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(color_image, "No pose", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Write output frame
                if writer is not None:
                    writer.write(color_image)
                
                # Display image
                cv2.imshow('FoundationPose RealSense', color_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('r'):
                    print("Resetting tracking...")
                    self.reset()
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Cleanup
            pipeline.stop()
            cv2.destroyAllWindows()
            if writer is not None:
                writer.release()
            
            print(f"Processed {frame_count} frames from RealSense")
            self._save_results(output_path)
    
    def _draw_pose_axes(self, image: np.ndarray, pose: np.ndarray, 
                       length: float = 0.1) -> np.ndarray:
        """
        Draw pose axes safely. Guards against z<=0/NaN and wrong pose convention.
        """
        K = self.K
        R = pose[:3, :3].copy()
        t = pose[:3, 3].copy()

        # Dynamic length based on distance from camera (10% of distance)
        dyn_len = float(max(0.05, min(0.2, t[2] * 0.1))) if np.isfinite(t[2]) else length

        # Axis endpoints in object frame
        axes_3d = np.array([
            [0, 0, 0],            # origin
            [dyn_len, 0, 0],       # x
            [0, dyn_len, 0],       # y
            [0, 0, dyn_len],       # z
        ], dtype=np.float64).T  # (3,4)

        def project(R_, t_):
            # camera-frame points
            cam = (R_ @ axes_3d) + t_[:, None]  # (3,4)
            z = cam[2]
            return cam, z

        cam, z = project(R, t)

        # If any point has z<=eps, try inverting pose once (in case pose convention is reversed)
        if np.any(~np.isfinite(z)) or np.any(z <= 1e-6):
            Pinv = np.linalg.inv(pose)
            R = Pinv[:3, :3]
            t = Pinv[:3, 3]
            cam, z = project(R, t)

        # If still invalid, skip drawing
        if np.any(~np.isfinite(z)) or np.any(z <= 1e-6):
            return image

        # Use OpenCV projectPoints for robust projection/type handling
        rvec, _ = cv2.Rodrigues(R)
        obj_pts = np.array([
            [0, 0, 0],
            [dyn_len, 0, 0],
            [0, dyn_len, 0],
            [0, 0, dyn_len],
        ], dtype=np.float64)
        dist = np.zeros(5)  # assume no distortion
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, t, K, dist)  # (4,1,2)
        pts = img_pts.reshape(-1, 2)

        # Validate finite + in-bounds-ish
        if not np.all(np.isfinite(pts)):
            return image

        origin = tuple(np.round(pts[0]).astype(int))
        x_end  = tuple(np.round(pts[1]).astype(int))
        y_end  = tuple(np.round(pts[2]).astype(int))
        z_end  = tuple(np.round(pts[3]).astype(int))

        # Optional: reject absurdly off-screen points to avoid OpenCV issues
        H, W = image.shape[:2]
        def on_screen(p): return -W <= p[0] <= 2*W and -H <= p[1] <= 2*H
        if not (on_screen(origin) and on_screen(x_end) and on_screen(y_end) and on_screen(z_end)):
            return image

        cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 3)  # X (red)
        cv2.arrowedLine(image, origin, y_end, (0, 255, 0), 3)  # Y (green)
        cv2.arrowedLine(image, origin, z_end, (255, 0, 0), 3)  # Z (blue)
        return image
    
    def _save_results(self, output_path: Optional[str] = None):
        """Save processing results."""
        results = {
            'total_frames': self.frame_count,
            'average_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'pose_history': self.pose_history,
            'config': self.config
        }
        
        # Save to JSON
        results_path = output_path.replace('.mp4', '_results.json') if output_path else 'fp_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_path}")
    
    def reset(self):
        """Reset pipeline state."""
        self.smoother.reset()
        self.fp_wrapper.reinit(None, None, None)  # Will be called with actual data
        self.current_pose = None
        self.frame_count = 0
        self.start_time = None
        self.pose_history = []
        self.fps_history = []
        print("Pipeline reset")


def create_config_template(output_path: str):
    """Create a configuration template file."""
    config = {
        'yolo': {
            'weights': None,  # Auto-detect from local files
            'conf': 0.5,
            'imgsz': 640
        },
        'foundationpose': {
            'device': 'cuda',
            'model_path': None  # Model-free mode
        },
        'smoothing': {
            'window_size': 5,
            'max_angular_velocity': 2.0,
            'max_translation_velocity': 1.0,
            'outlier_threshold': 0.1
        },
        'camera': {
            'camera_yaml': 'aiot_cv/configs/camera.yaml'
        },
        'references': {
            'refs_dir': 'aiot_cv/data/references',
            'tool_class': 'screwdriver'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration template created: {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="FoundationPose Model-Free Pipeline")
    parser.add_argument('--config', required=True, help='Pipeline configuration file')
    parser.add_argument('--video', help='Input video file')
    parser.add_argument('--output', help='Output video file')
    parser.add_argument('--realsense', action='store_true', help='Use RealSense camera')
    parser.add_argument('--create-config', help='Create config template at specified path')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_config_template(args.create_config)
        return
    
    # Initialize pipeline
    pipeline = FoundationPosePipeline(args.config)
    
    if args.realsense:
        pipeline.process_realsense(args.output)
    elif args.video:
        pipeline.process_video(args.video, args.output)
    else:
        print("Please specify --video or --realsense")
        return


if __name__ == '__main__':
    main()
