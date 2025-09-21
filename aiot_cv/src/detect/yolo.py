import os
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from dataclasses import dataclass

try:
	from ultralytics import YOLO
except Exception:
	YOLO = None


@dataclass
class Detection:
	"""Standardized detection output for FoundationPose pipeline."""
	bbox: np.ndarray        # Bounding box [x1, y1, x2, y2] in image coordinates
	mask: Optional[np.ndarray] = None  # Binary mask (H, W) if segmentation available
	class_id: int = 0       # Class ID
	class_name: str = ""    # Class name (if available)
	confidence: float = 0.0 # Detection confidence score
	roi_rgb: Optional[np.ndarray] = None  # Cropped RGB region
	roi_depth: Optional[np.ndarray] = None  # Cropped depth region (if available)


class YoloSeg:
	"""
	YOLOv8 segmentation wrapper with standardized output for FoundationPose.
	
	Features:
	- Standardized detection format
	- Automatic ROI cropping (RGB + depth)
	- Class name mapping
	- Confidence filtering
	"""
	
	def __init__(self, weights: str=None, conf: float=0.5, imgsz: int=640, 
	             class_names: Optional[List[str]] = None):
		"""
		Initialize YOLO detector.
		
		Args:
			weights: Path to YOLO weights file
			conf: Confidence threshold
			imgsz: Input image size
			class_names: List of class names (optional)
		"""
		if weights is None:
			# Prefer local trained weights if exist
			cand = [
				"train_results/exp1/weights/last.pt",
				"tool_seg/yolov8n.pt",
			]
			for c in cand:
				if os.path.exists(c):
					weights = c; break
			if weights is None:
				raise FileNotFoundError("No YOLO weights found. Place a .pt file or set weights path.")
		
		self.model = YOLO(weights) if YOLO else None
		self.conf = conf
		self.imgsz = imgsz
		self.class_names = class_names or ["object"]  # Default class name
		
		# Try to get class names from model
		if self.model is not None and hasattr(self.model, 'names'):
			self.class_names = list(self.model.names.values())
		
		print(f"YOLO initialized with {len(self.class_names)} classes: {self.class_names}")

	def __call__(self, img_bgr: np.ndarray, depth: Optional[np.ndarray] = None, **kwargs) -> List[Detection]:
		"""
		Run detection and return standardized Detection objects.
		
		Args:
			img_bgr: Input BGR image (H, W, 3)
			depth: Optional depth image (H, W) - will be cropped to match detections
			**kwargs: Additional arguments for YOLO inference (conf, classes, etc.)
			
		Returns:
			List of Detection objects with ROIs
		"""
		assert self.model is not None, "ultralytics not available"
		
		# Prepare inference arguments
		infer_args = {
			'source': img_bgr,
			'imgsz': self.imgsz,
			'conf': kwargs.get('conf', self.conf),
			'verbose': False
		}
		
		# Add classes filter if provided
		if 'classes' in kwargs:
			infer_args['classes'] = kwargs['classes']
		
		# Run YOLO inference
		res = self.model.predict(**infer_args)[0]
		
		detections = []
		boxes = res.boxes
		masks = getattr(res, 'masks', None)
		
		if boxes is None or len(boxes) == 0:
			return detections
		
		# Process each detection
		for i in range(len(boxes)):
			# Extract basic detection info
			xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
			score = float(boxes.conf[i].cpu().numpy())
			cls_id = int(boxes.cls[i].cpu().numpy())
			
			# Get class name
			cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
			
			# Extract mask if available
			mask = None
			if masks is not None and i < len(masks.data):
				mask_data = masks.data[i].cpu().numpy()
				# Resize mask to original image size
				h_orig, w_orig = img_bgr.shape[:2]
				mask_resized = cv2.resize(mask_data, (w_orig, h_orig))
				mask = (mask_resized > 0.5).astype(np.uint8)
			
			# Crop ROI regions
			roi_rgb = self._crop_roi(img_bgr, xyxy)
			roi_depth = self._crop_roi(depth, xyxy) if depth is not None else None
			
			# Create detection object
			detection = Detection(
				bbox=xyxy,
				mask=mask,
				class_id=cls_id,
				class_name=cls_name,
				confidence=score,
				roi_rgb=roi_rgb,
				roi_depth=roi_depth
			)
			
			detections.append(detection)
		
		# Sort by confidence (highest first)
		detections.sort(key=lambda x: x.confidence, reverse=True)
		
		return detections
	
	def _crop_roi(self, img: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
		"""Crop region of interest from image."""
		if img is None:
			return None
		
		x1, y1, x2, y2 = bbox
		h, w = img.shape[:2]
		
		# Ensure coordinates are within image bounds
		x1 = max(0, min(x1, w-1))
		y1 = max(0, min(y1, h-1))
		x2 = max(x1+1, min(x2, w))
		y2 = max(y1+1, min(y2, h))
		
		return img[y1:y2, x1:x2]
	
	def get_best_detection(self, img_bgr: np.ndarray, depth: Optional[np.ndarray] = None,
	                      target_class: Optional[str] = None, **kwargs) -> Optional[Detection]:
		"""
		Get the best (highest confidence) detection.
		
		Args:
			img_bgr: Input BGR image
			depth: Optional depth image
			target_class: Filter by class name (optional)
			**kwargs: Additional arguments for YOLO inference
			
		Returns:
			Best Detection or None
		"""
		detections = self(img_bgr, depth, **kwargs)
		
		if not detections:
			return None
		
		# Filter by class if specified
		if target_class is not None:
			detections = [d for d in detections if d.class_name == target_class]
		
		return detections[0] if detections else None
