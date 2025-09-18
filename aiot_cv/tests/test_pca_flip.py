import numpy as np
from aiot_cv.pose.pca_init import stabilize_axes

def test_stabilize_axes():
	Rprev = np.eye(3)
	Rcurr = np.eye(3)
	Rcurr[:,0] *= -1
	R = stabilize_axes(Rprev, Rcurr, dot_thresh=0.0)
	assert np.linalg.det(R) > 0
	assert R[:,0].dot(Rprev[:,0]) >= 0
