import yaml
import numpy as np

def load_K(path):
	with open(path, 'r') as f:
		cfg = yaml.safe_load(f)
	K = np.array([[cfg['fx'], 0, cfg['cx']], [0, cfg['fy'], cfg['cy']], [0,0,1]], dtype=np.float64)
	depth_scale = float(cfg.get('depth_scale', 0.001))
	return K, depth_scale, cfg
