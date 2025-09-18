import numpy as np

def rot_z(theta):
	c, s = np.cos(theta), np.sin(theta)
	return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def symmetry_candidates(T: np.ndarray, sym_type: str):
	cands = [T]
	R = T[:3,:3]; t = T[:3,3]
	if sym_type == 'cyl_90':
		for k in [1,2,3]:
			Tc = np.eye(4)
			Tc[:3,:3] = R @ rot_z(np.deg2rad(90*k))
			Tc[:3,3] = t
			cands.append(Tc)
	return cands
