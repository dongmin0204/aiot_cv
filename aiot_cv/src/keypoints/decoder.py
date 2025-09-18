import torch

def soft_argmax_2d(hm):
	B,K,H,W = hm.shape
	hm = hm.view(B*K, H, W)
	hm = torch.softmax(hm.reshape(B*K, -1), dim=-1).reshape(B*K, H, W)
	xs = torch.linspace(0, W-1, W, device=hm.device)
	ys = torch.linspace(0, H-1, H, device=hm.device)
	xs = xs.view(1,1,W).expand(B*K, H, W)
	ys = ys.view(1,H,1).expand(B*K, H, W)
	x = (hm * xs).sum(dim=(1,2))
	y = (hm * ys).sum(dim=(1,2))
	return torch.stack([x,y], dim=-1).view(B, K, 2)
