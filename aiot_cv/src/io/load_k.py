import yaml
import numpy as np


def build_K(fx, fy, cx, cy):
	return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def scale_K(K, sx, sy):
	K2 = K.copy().astype(np.float64)
	K2[0,0] *= sx
	K2[1,1] *= sy
	K2[0,2] *= sx
	K2[1,2] *= sy
	return K2


def crop_adjust_K(K, x1, y1):
	K2 = K.copy().astype(np.float64)
	K2[0,2] -= x1
	K2[1,2] -= y1
	return K2


def load_K(path):
	cfg = yaml.safe_load(open(path, 'r'))
	# Support minimal or extended schema
	if 'intrinsics' in cfg:
		intr = cfg['intrinsics']['color'] if cfg.get('align_to_color', True) else cfg['intrinsics']['depth']
		fx, fy, cx, cy = float(intr['fx']), float(intr['fy']), float(intr['cx']), float(intr['cy'])
		K = build_K(fx, fy, cx, cy)
		depth_scale = float(cfg.get('depth_units', cfg.get('depth_scale', 0.001)))
		return K, depth_scale, cfg
	else:
		fx, fy, cx, cy = cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy']
		K = build_K(fx, fy, cx, cy)
		depth_scale = float(cfg.get('depth_scale', 0.001))
		return K, depth_scale, cfg
