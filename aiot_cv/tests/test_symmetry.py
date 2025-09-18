import numpy as np
from aiot_cv.pose.symmetry import symmetry_candidates

def test_symmetry_candidates_shape():
	T = np.eye(4)
	cands = symmetry_candidates(T, 'cyl_90')
	for Tc in cands:
		assert Tc.shape == (4,4)
