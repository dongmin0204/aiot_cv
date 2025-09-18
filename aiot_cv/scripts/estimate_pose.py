import argparse
import json
import numpy as np
from pathlib import Path

from aiot_cv.pose.frames import make_T, project_points
from aiot_cv.pose.depth_to_cloud import backproject_mask, ror, sor, voxel_down
from aiot_cv.pose.pca_init import pca_pose, stabilize_axes


def load_K_from_yaml(path):
	import yaml
	with open(path, 'r') as f:
		cfg = yaml.safe_load(f)
	K = np.array([[cfg['fx'], 0, cfg['cx']], [0, cfg['fy'], cfg['cy']], [0,0,1]], dtype=np.float64)
	return K, float(cfg.get('depth_scale', 0.001))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--rgb', type=str, required=False)
	ap.add_argument('--depth', type=str, required=True)
	ap.add_argument('--mask', type=str, required=True)
	ap.add_argument('--K', type=str, required=True)
	ap.add_argument('--tool_id', type=str, default='tool')
	ap.add_argument('--sym_type', type=str, default='none')
	ap.add_argument('--use-pca', action='store_true')
	ap.add_argument('--out', type=str, default='pose_out.json')
	args = ap.parse_args()

	# I/O
	import cv2
	depth = cv2.imread(args.depth, -1)
	mask = cv2.imread(args.mask, 0)
	K, depth_scale = load_K_from_yaml(args.K)

	# cloud
	cloud = backproject_mask(depth, mask, K, depth_scale)
	cloud = ror(cloud)
	cloud = sor(cloud)
	cloud = voxel_down(cloud)

	R, t = pca_pose(cloud)
	T = make_T(R, t)

	out = {
		'R': R.tolist(),
		't': t.tolist(),
		'T': T.tolist(),
		'frame': 'camera',
		'tool_id': args.tool_id,
	}
	Path(args.out).write_text(json.dumps(out, indent=2))
	print(f"saved: {args.out}")


if __name__ == '__main__':
	main()
