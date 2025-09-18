import torch
from aiot_cv.src.keypoints.decoder import soft_argmax_2d

def test_soft_argmax_center():
	h, w = 64, 64
	hm = torch.zeros(1,1,h,w)
	hm[0,0,32,40] = 10.0
	xy = soft_argmax_2d(hm)
	x, y = xy[0,0,0].item(), xy[0,0,1].item()
	assert abs(x-40) < 1.0 and abs(y-32) < 1.0
