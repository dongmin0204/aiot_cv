import numpy as np
from typing import List, Tuple


def symmetry_candidates(T: np.ndarray, sym_type: str) -> list:
	cands = [T]
	if sym_type == 'none':
		return cands
	R = T[:3, :3].copy()
	t = T[:3, 3].copy()
	if sym_type == 'cyl_90':
		for k in [1,2,3]:
			Rz = rot_z(np.deg2rad(90*k))
			Tc = np.eye(4)
			Tc[:3,:3] = R @ Rz
			Tc[:3,3] = t
			cands.append(Tc)
	elif sym_type == 'mirror_x':
		Mx = np.diag([-1,1,1])
		Tc = np.eye(4)
		Tc[:3,:3] = R @ Mx
		Tc[:3,3] = t
		cands.append(Tc)
	return cands


def rot_z(theta: float) -> np.ndarray:
	c, s = np.cos(theta), np.sin(theta)
	return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def score_by_reprojection(T_list, model_pts_samp, rgb, depth, K, mask) -> tuple:
	# Placeholder; to be implemented with actual scoring
	scores = np.linspace(0, 1, num=len(T_list))
	best = int(np.argmin(scores))
	return T_list[best], float(scores[best])
