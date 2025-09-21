import yaml
import numpy as np


def build_K(fx, fy, cx, cy):
	"""Build 3x3 camera matrix from scalar intrinsics."""
	return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def scale_K(K, sx, sy):
	"""Scale K when the image is resized from (W0,H0) to (W1,H1) with sx=W1/W0, sy=H1/H0."""
	K2 = K.copy().astype(np.float64)
	K2[0,0] *= sx
	K2[1,1] *= sy
	K2[0,2] *= sx
	K2[1,2] *= sy
	return K2


def crop_adjust_K(K, x1, y1):
	"""Adjust principal point when the image is cropped with top-left offset (x1,y1)."""
	K2 = K.copy().astype(np.float64)
	K2[0,2] -= x1
	K2[1,2] -= y1
	return K2


def load_K(path):
	"""Load intrinsics from minimal or extended YAML schema.
	- If extended: choose color/depth by align_to_color flag, return K and depth scale.
	- If minimal: expects fx,fy,cx,cy,depth_scale.
	"""
	cfg = yaml.safe_load(open(path, 'r',encoding='utf-8'))
	if 'intrinsics' in cfg:  # extended schema
		intr = cfg['intrinsics']['color'] if cfg.get('align_to_color', True) else cfg['intrinsics']['depth']
		fx, fy, cx, cy = float(intr['fx']), float(intr['fy']), float(intr['cx']), float(intr['cy'])
		K = build_K(fx, fy, cx, cy)
		depth_scale = float(cfg.get('depth_units', cfg.get('depth_scale', 0.001)))
		return K, depth_scale, cfg
	else:  # minimal schema
		fx, fy, cx, cy = cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy']
		K = build_K(fx, fy, cx, cy)
		depth_scale = float(cfg.get('depth_scale', 0.001))
		return K, depth_scale, cfg
