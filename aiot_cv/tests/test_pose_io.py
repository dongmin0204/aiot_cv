import numpy as np
from aiot_cv.pose.frames import make_T, invert, project_points

def test_make_invert_roundtrip():
	R = np.eye(3)
	t = np.array([1.0, 2.0, 3.0])
	T = make_T(R, t)
	Ti = invert(T)
	I = T @ Ti
	assert np.allclose(I, np.eye(4), atol=1e-8)

def test_project_points_identity():
	P = np.array([[0,0,1],[0,0,2],[0,0,3]], dtype=np.float64)
	K = np.array([[600,0,320],[0,600,240],[0,0,1]], dtype=np.float64)
	T = np.eye(4)
	uv = project_points(P, T, K)
	assert np.allclose(uv[:,0], 320)
	assert np.allclose(uv[:,1], 240)
