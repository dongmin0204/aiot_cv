import argparse, yaml, torch, torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import KPDataset
from .model import KPHead
from .losses import heatmap_mse

def train(ann, tool_id, cfg_path, out="weights/kp_head.pt"):
	cfg = yaml.safe_load(open(cfg_path))
	ds = KPDataset(ann, tool_id, cfg)
	dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
	net = KPHead(in_ch=3, kpts=cfg["num_keypoints"], hm_sz=cfg["heatmap_size"]).cuda()
	optimz = optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

	for ep in range(cfg["epochs"]):
		net.train(); tot=0
		for img, hm_t, vis, _ in dl:
			img, hm_t, vis = img.cuda(), hm_t.cuda(), vis.cuda()
			hm_p = net(img)
			loss = heatmap_mse(hm_p, hm_t, vis)
			optimz.zero_grad(); loss.backward(); optimz.step()
			tot += loss.item()*img.size(0)
		print(f"ep{ep+1}: loss={tot/len(ds):.4f}")
	torch.save(net.state_dict(), out)

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('--ann', required=True)
	ap.add_argument('--tool_id', required=True)
	ap.add_argument('--cfg', default='aiot_cv/configs/train_kp.yaml')
	ap.add_argument('--out', default='weights/kp_head.pt')
	args = ap.parse_args()
	train(args.ann, args.tool_id, args.cfg, args.out)
