import torch, torch.nn as nn

def conv_bn(inp, oup, k=3, s=1, p=1):
	return nn.Sequential(
		nn.Conv2d(inp, oup, k, s, p, bias=False),
		nn.BatchNorm2d(oup),
		nn.SiLU(True)
	)

class KPHead(nn.Module):
	def __init__(self, in_ch=3, kpts=4, hm_sz=64):
		super().__init__()
		self.enc = nn.Sequential(
			conv_bn(in_ch, 32, 3, 2, 1),
			conv_bn(32, 64, 3, 2, 1),
			conv_bn(64, 128, 3, 1, 1),
			conv_bn(128, 128, 3, 1, 1),
		)
		self.dec = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			conv_bn(128, 64, 3,1,1),
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			conv_bn(64, 32, 3,1,1),
		)
		self.head = nn.Sequential(
			nn.Conv2d(32, kpts, 3, 1, 1),
			nn.Upsample(size=(hm_sz, hm_sz), mode='bilinear', align_corners=False)
		)

	def forward(self, x):
		f = self.enc(x)
		f = self.dec(f)
		hm = self.head(f)
		return hm
