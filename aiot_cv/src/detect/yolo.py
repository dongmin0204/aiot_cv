import os
from typing import List, Dict, Any
import numpy as np

try:
	from ultralytics import YOLO
except Exception:
	YOLO = None

class YoloSeg:
	def __init__(self, weights: str=None, conf: float=0.5, imgsz: int=640):
		if weights is None:
			# Prefer local trained weights if exist
			cand = [
				"train_results/exp1/weights/best.pt",
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

	def __call__(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
		assert self.model is not None, "ultralytics not available"
		res = self.model.predict(source=img_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
		out = []
		boxes = res.boxes
		masks = getattr(res, 'masks', None)
		for i in range(len(boxes)):
			xyxy = boxes.xyxy[i].cpu().numpy().tolist()
			score = float(boxes.conf[i].cpu().numpy())
			cls = int(boxes.cls[i].cpu().numpy())
			mask = None
			if masks is not None:
				m = masks.data[i].cpu().numpy()
				mask = (m > 0.5).astype(np.uint8)
			out.append({"xyxy": xyxy, "score": score, "cls": cls, "mask": mask})
		return out
