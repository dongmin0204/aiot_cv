import json, cv2, numpy as np, torch
from torch.utils.data import Dataset


def make_heatmap(h, w, xy, sigma):
	K = xy.shape[0]
	hm = np.zeros((K,h,w), np.float32)
	yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
	for k,(x,y) in enumerate(xy):
		if x<0 or y<0: continue
		hm[k] = np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))
	return hm


class KPDataset(Dataset):
	def __init__(self, ann_path, tool_id, cfg):
		self.items = json.load(open(ann_path))
		self.tool_id = tool_id
		self.cfg = cfg

	def __len__(self): return len(self.items)

	def __getitem__(self, i):
		rec = self.items[i]
		img = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
		obj = next(o for o in rec["objects"] if o["tool_id"]==self.tool_id)
		x1,y1,x2,y2 = obj["bbox"]
		w, h = x2-x1, y2-y1
		c = np.array([(x1+x2)/2, (y1+y2)/2])
		s = int(1.2*max(w,h)/2)
		x1n,y1n = (c - s).astype(int)
		x2n,y2n = (c + s).astype(int)
		H, W = img.shape[:2]
		x1n,y1n = np.clip([x1n,y1n], 0, [W-1,H-1])
		x2n,y2n = np.clip([x2n,y2n], 0, [W-1,H-1])

		patch = img[y1n:y2n, x1n:x2n]
		patch = cv2.resize(patch, (self.cfg["image_size"], self.cfg["image_size"]))
		patch = patch[:,:,::-1].astype(np.float32)/255.0

		order = json.load(open(f"aiot_cv/configs/tools/{self.tool_id}.json"))["keypoints_order"]
		kps = np.array([obj["keypoints"][k][:2] for k in order], np.float32)
		vis = np.array([obj["keypoints"][k][2] for k in order], np.float32)
		kps_roi = (kps - np.array([x1n,y1n])) * (self.cfg["image_size"]/(x2n-x1n))
		scale = self.cfg["heatmap_size"]/self.cfg["image_size"]
		kps_hm = kps_roi * scale

		hm = make_heatmap(self.cfg["heatmap_size"], self.cfg["heatmap_size"], kps_hm, self.cfg["sigma"])
		vis = vis.reshape(-1,1,1)
		return torch.from_numpy(patch.transpose(2,0,1)), \
		       torch.from_numpy(hm), \
		       torch.from_numpy(vis), \
		       torch.tensor([x1n,y1n,x2n,y2n], dtype=torch.float32)
