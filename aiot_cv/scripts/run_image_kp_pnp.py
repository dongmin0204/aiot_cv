import json, yaml, cv2, torch, numpy as np
from aiot_cv.src.io.load_k import load_K
from aiot_cv.src.detect.yolo import YoloSeg
from aiot_cv.src.detect.mask_ops import crop_square_roi
from aiot_cv.src.keypoints.model import KPHead
from aiot_cv.src.keypoints.decoder import soft_argmax_2d
from aiot_cv.src.pose.pnp import solve_pnp
from aiot_cv.src.pose.symmetry import symmetry_candidates

if __name__ == '__main__':
	cfg = yaml.safe_load(open('aiot_cv/configs/train_kp.yaml'))
	K, depth_scale, _ = load_K('tool_seg/data.yaml'.replace('data.yaml','camera.yaml')) if False else (np.array([[600,0,320],[0,600,240],[0,0,1]], float), 0.001, {})
	tool = json.load(open('aiot_cv/configs/tools/screwdriver.json'))
	order = tool['keypoints_order']; KPTS = len(order)

	det = YoloSeg(weights=None, conf=0.5)
	net = KPHead(3, KPTS, cfg['heatmap_size']).cuda()
	# TODO: load weights
	# net.load_state_dict(torch.load('weights/kp_head.pt'))
	net.eval()

	rgb = cv2.imread('aiot_cv/data/images/sample.png')
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
