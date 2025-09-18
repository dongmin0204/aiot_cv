import numpy as np, cv2

def solve_pnp(kps2d: np.ndarray, kps3d: np.ndarray, K: np.ndarray, dist=None):
	if dist is None:
		dist = np.zeros(5)
	succ, rvec, tvec, inl = cv2.solvePnPRansac(
		kps3d.astype(np.float32), kps2d.astype(np.float32), K.astype(np.float64), dist,
		reprojectionError=3.0, flags=cv2.SOLVEPNP_ITERATIVE
	)
	if not succ:
		return None, None
	R, _ = cv2.Rodrigues(rvec)
	T = np.eye(4, dtype=np.float64)
	T[:3,:3] = R
	T[:3, 3] = tvec.reshape(3)
	return T, inl
