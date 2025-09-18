import numpy as np
from typing import Tuple

try:
	from sklearn.neighbors import KDTree  # optional; used for filters
except Exception:
	KDTree = None

def backproject_mask(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, depth_scale: float=0.001) -> np.ndarray:
	assert depth.shape == mask.shape
	h, w = depth.shape
	ys, xs = np.where(mask > 0)
	z = depth[ys, xs].astype(np.float32) * depth_scale
	valid = z > 0
	xs, ys, z = xs[valid], ys[valid], z[valid]
	fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
	X = (xs - cx) * z / fx
	Y = (ys - cy) * z / fy
	return np.stack([X, Y, z], axis=1)

def ror(points: np.ndarray, radius: float=0.02, min_neighbors: int=8) -> np.ndarray:
	if KDTree is None or len(points) < min_neighbors:
		return points
	tree = KDTree(points)
	cnts = tree.query_radius(points, r=radius, count_only=True)
	return points[cnts >= min_neighbors]

def sor(points: np.ndarray, k: int=20, std_ratio: float=1.5) -> np.ndarray:
	if KDTree is None or len(points) <= k:
		return points
	tree = KDTree(points)
	dists, _ = tree.query(points, k=k)
	mean = dists[:, 1:].mean(axis=1)
	m, s = mean.mean(), mean.std()
	keep = np.abs(mean - m) <= std_ratio * s
	return points[keep]

def voxel_down(points: np.ndarray, voxel: float=0.003) -> np.ndarray:
	if len(points) == 0:
		return points
	keys = np.floor(points / voxel)
	_, idx = np.unique(keys, axis=0, return_index=True)
	return points[idx]
