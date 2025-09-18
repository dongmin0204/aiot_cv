import torch, torch.nn.functional as F

def heatmap_mse(pred, target, vis):
	return ((pred - target)**2 * vis).mean()
