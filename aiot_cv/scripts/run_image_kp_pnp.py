import json, yaml, cv2, torch, numpy as np, argparse, os
from aiot_cv.src.io.load_k import load_K, crop_adjust_K, scale_K
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
	K_full, depth_scale, Kcfg = load_K(args.camera)
	tool = json.load(open(args.tool))
	order = tool['keypoints_order']; KPTS = len(order)

	det = YoloSeg(weights=None, conf=0.5)
	net = KPHead(3, KPTS, cfg['heatmap_size']).cuda()
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

	# Build ROI-adjusted K_in corresponding to network input SxS
	x1, y1, size = T_roi['x1'], T_roi['y1'], T_roi['size']
	K_crop = crop_adjust_K(K_full, x1, y1)
	sx = cfg['image_size'] / size
	sy = cfg['image_size'] / size
	K_in = scale_K(K_crop, sx, sy)

	x = torch.from_numpy(roi[:,:,::-1].transpose(2,0,1)).float().unsqueeze(0)/255.0
	with torch.no_grad():
		hm = net(x.cuda())
	kps_hm = soft_argmax_2d(hm).cpu().numpy()[0]
	# heatmap coords -> network input coords
	kps_in = kps_hm * (1/ (cfg['image_size']/cfg['heatmap_size']))

	kps3d = np.array([tool['keypoints_3d'][k] for k in order], float)
	T_pnp, inl = solve_pnp(kps_in, kps3d, K_in)
	Ts = symmetry_candidates(T_pnp, tool['symmetry'])
	T = Ts[0]
	print('T:', np.array(T))
