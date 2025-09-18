import json, yaml, cv2, torch, numpy as np, argparse, os
from aiot_cv.src.io.load_k import load_K
from aiot_cv.src.detect.yolo import YoloSeg
from aiot_cv.src.detect.mask_ops import crop_square_roi
from aiot_cv.src.keypoints.model import KPHead
from aiot_cv.src.keypoints.decoder import soft_argmax_2d
from aiot_cv.src.pose.pnp import solve_pnp
from aiot_cv.src.pose.symmetry import symmetry_candidates

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('--image', default='aiot_cv/data/images/sample.png')
	ap.add_argument('--camera', default='aiot_cv/configs/camera.yaml')
	ap.add_argument('--tool', default='aiot_cv/configs/tools/screwdriver.json')
	ap.add_argument('--kp-weights', default=None)
	args = ap.parse_args()

	cfg = yaml.safe_load(open('aiot_cv/configs/train_kp.yaml'))
	K, depth_scale, _ = load_K(args.camera)
	tool = json.load(open(args.tool))
	order = tool['keypoints_order']; KPTS = len(order)

	det = YoloSeg(weights=None, conf=0.5)
	net = KPHead(3, KPTS, cfg['heatmap_size']).cuda()
	# auto-detect kp weights if not provided
	cand = [args.kp_weights, 'weights/kp_head.pt', 'aiot_cv/weights/kp_head.pt']
	for c in cand:
		if c and os.path.exists(c):
			net.load_state_dict(torch.load(c))
			print(f"loaded KP weights: {c}")
			break
	net.eval()

	rgb = cv2.imread(args.image)
	objs = det(rgb); obj = max(objs, key=lambda o:o['score'])
	roi, T_roi = crop_square_roi(rgb, obj['xyxy'], out_size=cfg['image_size'])

	x = torch.from_numpy(roi[:,:,::-1].transpose(2,0,1)).float().unsqueeze(0)/255.0
	with torch.no_grad():
		hm = net(x.cuda())
	kps_hm = soft_argmax_2d(hm).cpu().numpy()[0]

	scale = cfg['image_size']/T_roi['size']
	kps_roi = kps_hm * (1/ (cfg['image_size']/cfg['heatmap_size']))
	kps_full = kps_roi/scale + np.array([T_roi['x1'], T_roi['y1']])

	kps3d = np.array([tool['keypoints_3d'][k] for k in order], float)
	T_pnp, inl = solve_pnp(kps_full, kps3d, K)
	Ts = symmetry_candidates(T_pnp, tool['symmetry'])
	T = Ts[0]
	print('T:', np.array(T))
