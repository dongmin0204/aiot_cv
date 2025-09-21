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
        
        if not detections:
            print("No detections found")
            return None
        
        # Get best detection (highest confidence)
        detection = detections[0]
        print(f"Best detection: {detection.class_name} (conf={detection.confidence:.3f})")
        
        # Step 2: Extract ROI and mask
        roi_rgb = detection.roi_rgb
        roi_depth = detection.roi_depth
        mask = detection.mask
        
        if roi_rgb is None:
            print("No ROI RGB available")
            return None
        
        # Convert BGR to RGB for FoundationPose
        roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2RGB)
        
        # Step 3: FoundationPose estimation
        if self.current_pose is None:
            # Initialize pose from reference images
            print("Initializing pose from references...")
            pose = self.fp_wrapper.init_pose_from_refs(roi_rgb, roi_depth, mask)
        else:
            # Track pose
            print("Tracking pose...")
            pose = self.fp_wrapper.track(roi_rgb, roi_depth, mask)
        
        if pose is None:
            print("FoundationPose estimation failed")
            return None
        
        # Step 4: Apply smoothing
        smoothed_pose = self.smoother.update(pose, frame_start_time)
        
        if smoothed_pose is not None:
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
        
        # Update frame count
        self.frame_count += 1
        
        return smoothed_pose
    
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
    
    def process_realsense(self):
        """Process RealSense camera stream (placeholder)."""
        print("RealSense processing not implemented yet")
        print("Use process_frame() with RealSense RGB-D data")
    
    def _draw_pose_axes(self, image: np.ndarray, pose: np.ndarray, 
                       length: float = 0.1) -> np.ndarray:
        """Draw pose coordinate axes on image."""
        # Extract rotation and translation
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Define axis endpoints in 3D
        axes_3d = np.array([
            [0, 0, 0],           # origin
            [length, 0, 0],      # x-axis (red)
            [0, length, 0],      # y-axis (green)
            [0, 0, length]       # z-axis (blue)
        ]).T
        
        # Project to 2D
        axes_2d = self.K @ (R @ axes_3d + t[:, np.newaxis])
        axes_2d = axes_2d[:2] / axes_2d[2]
        
        # Draw axes
        origin = tuple(axes_2d[:, 0].astype(int))
        
        # X-axis (red)
        x_end = tuple(axes_2d[:, 1].astype(int))
        cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 3)
        
        # Y-axis (green)
        y_end = tuple(axes_2d[:, 2].astype(int))
        cv2.arrowedLine(image, origin, y_end, (0, 255, 0), 3)
        
        # Z-axis (blue)
        z_end = tuple(axes_2d[:, 3].astype(int))
        cv2.arrowedLine(image, origin, z_end, (255, 0, 0), 3)
        
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
        pipeline.process_realsense()
    elif args.video:
        pipeline.process_video(args.video, args.output)
    else:
        print("Please specify --video or --realsense")
        return


if __name__ == '__main__':
    main()
