import torch
from .decoder import soft_argmax_2d

def infer_keypoints(net, roi_rgb):
	# roi_rgb: (H,W,3) np.uint8 RGB [0..255]
	x = torch.from_numpy(roi_rgb.transpose(2,0,1)).float().unsqueeze(0)/255.0
	with torch.no_grad():
		hm = net(x.cuda())
	kps = soft_argmax_2d(hm).cpu().numpy()[0]
	return hm.cpu().numpy()[0], kps
