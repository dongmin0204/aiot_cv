import numpy as np
from typing import Optional, Tuple


def pca_pose(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	assert points.ndim == 2 and points.shape[1] == 3
	c = np.median(points, axis=0)
	X = points - c
	_, _, Vt = np.linalg.svd(X, full_matrices=False)
	R = Vt.T
	if np.linalg.det(R) < 0:
		R[:, 2] *= -1
	return R.astype(np.float64), c.astype(np.float64)


def stabilize_axes(R_prev: Optional[np.ndarray], R_curr: np.ndarray, dot_thresh: float=0.0) -> np.ndarray:
	R = R_curr.copy()
	if R_prev is not None:
		for j in range(3):
			if R_prev[:, j].dot(R[:, j]) < dot_thresh:
				R[:, j] *= -1
	if np.linalg.det(R) < 0:
		R[:, 2] *= -1
	return R
