import cv2import cv2
import time
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


# --------------------
# Config (keep it short)
# --------------------
WEIGHTS = "../train_results/exp1/weights/best.pt"
CONF_TH = 0.5
IOU_TH  = 0.5
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30
SAMPLE_STRIDE = 2
Z_MIN, Z_MAX = 0.1, 2.0
FONT = cv2.FONT_HERSHEY_SIMPLEX


# Base←Cam extrinsic (example; replace with calibrated values)
RX_DEG, RY_DEG, RZ_DEG = -180.0, 0.0, -80.0
TX, TY, TZ = 0.20, 0.0, 0.50
ORDER = "XYZ"


# --------------------
# Small math helpers
# --------------------
def Rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)


def Ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], float)


def Rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)


def H_from_axis_angles(tx, ty, tz, rx, ry, rz, order="XYZ"):
    R_map = {"X": Rx(rx), "Y": Ry(ry), "Z": Rz(rz)}
    R = np.eye(3)
    for ax in order:
        R = R @ R_map[ax]
    H = np.eye(4)
    H[:3,:3] = R
    H[:3, 3] = [tx, ty, tz]
    return H


def ensure_right_handed(R):
    R = np.asarray(R, float).copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


# --------------------
# Camera projections
# --------------------
def precompute_xy_maps(intr, H, W):
    js = np.arange(W, dtype=np.float32)
    is_ = np.arange(H, dtype=np.float32)
    gy, gx = np.meshgrid(is_, js, indexing="ij")
    x_map = (gx - intr.ppx) / intr.fx
    y_map = (gy - intr.ppy) / intr.fy
    return x_map, y_map


def project_points_intr(intr, pts3d):
    pts3d = np.asarray(pts3d, dtype=np.float32)
    Z = pts3d[:, 2]
    valid = Z > 1e-6
    uv = np.full((pts3d.shape[0], 2), np.nan, np.float32)
    uv[valid, 0] = intr.fx * (pts3d[valid, 0] / Z[valid]) + intr.ppx
    uv[valid, 1] = intr.fy * (pts3d[valid, 1] / Z[valid]) + intr.ppy
    return uv, valid


# --------------------
# Depth → 3D points (mask-based)
# --------------------
def mask_to_points3d(depth_frame, mask, depth_scale, intr, x_map, y_map,
                     sample_stride=1, z_min=0.0, z_max=10.0, erosion=3):
    depth_u16 = np.asanyarray(depth_frame.get_data())
    H, W = depth_u16.shape[:2]
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    if erosion > 0:
        k = np.ones((erosion*2+1, erosion*2+1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), k, 1)
    m = (mask > 0)
    if sample_stride > 1:
        m = m[::sample_stride, ::sample_stride]
        Zf = depth_u16.astype(np.float32) * depth_scale
        Z = Zf[::sample_stride, ::sample_stride]
        X = (x_map * Zf)[::sample_stride, ::sample_stride]
        Y = (y_map * Zf)[::sample_stride, ::sample_stride]
    else:
        Z = depth_u16.astype(np.float32) * depth_scale
        X = x_map * Z
        Y = y_map * Z
    if not np.any(m):
        return None
    Xv, Yv, Zv = X[m], Y[m], Z[m]
    valid = (Zv > 0) & np.isfinite(Zv) & (Zv >= z_min) & (Zv <= z_max)
    if not np.any(valid):
        return None
    pts = np.stack([Xv[valid], Yv[valid], Zv[valid]], axis=1)
    return pts if pts.shape[0] >= 30 else None


# --------------------
# PCA OBB (fast, minimal)
# --------------------
def pca_obb_3d(points_xyz):
    pts = points_xyz.astype(np.float32)
    mean = pts.mean(axis=0)
    C = np.cov((pts - mean), rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    axes = axes / (np.linalg.norm(axes, axis=0, keepdims=True) + 1e-9)
    proj = (pts - mean) @ axes
    mins, maxs = proj.min(axis=0), proj.max(axis=0)
    c_local = (mins + maxs) * 0.5
    half = (maxs - mins) * 0.5
    center = mean + axes @ c_local
    corners = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                corners.append(center + s1*half[0]*axes[:,0] + s2*half[1]*axes[:,1] + s3*half[2]*axes[:,2])
    corners = np.stack(corners, axis=0)
    lengths = 2.0 * half
    return center, axes, lengths, corners


# --------------------
# NEW: Robust pose stabilization (roll/pitch lock + temporal smoothing)
# --------------------


def so3_log(R):
    """Matrix log for SO(3) -> axis*angle (vector)"""
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * theta


def so3_exp(w):
    """Axis-angle vector -> SO(3) using Rodrigues"""
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K@K)


def slerp_SO3(R0, R1, alpha):
    """Geodesic interpolation on SO(3)."""
    dR = R0.T @ R1
    w = so3_log(dR)
    return R0 @ so3_exp(alpha * w)


def fit_plane_ransac(pts, iters=100, tau=0.01):
    """RANSAC plane fit: returns unit normal (z_up) and offset d so that n^T x + d = 0"""
    N = pts.shape[0]
    if N < 50:
        return None, None
    best_inl, best = 0, (None, None)
    rng = np.random.default_rng()
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        a,b,c = pts[idx]
        n = np.cross(b-a, c-a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            continue
        n = n / n_norm
        d = -np.dot(n, a)
        dist = np.abs(pts @ n + d)
        inl = np.count_nonzero(dist < tau)
        if inl > best_inl:
            best_inl = inl; best = (n, d)
    if best[0] is None:
        return None, None
    # ensure z-up (flip if upside down)
    n = best[0]
    if n[2] < 0: n = -n
    return n / (np.linalg.norm(n)+1e-9), best[1]


class PoseStabilizer:
    """Temporal smoothing + roll/pitch lock via plane normal or IMU gravity."""
    def __init__(self, alpha_R=0.25, alpha_t=0.3, use_plane_lock=True):
        self.R_prev = None
        self.t_prev = None
        self.alpha_R = alpha_R
        self.alpha_t = alpha_t
        self.use_plane_lock = use_plane_lock
        self.z_up_ref = None  # set from first reliable plane or external IMU


    def lock_roll_pitch(self, R_obj_cam, z_up):
        """Project object X/Y axes onto plane orthogonal to z_up to remove roll/pitch."""
        # build a camera/world frame where z = z_up
        z = z_up / (np.linalg.norm(z_up)+1e-9)
        # choose x as projection of current x onto plane, then y = z×x
        x_raw = R_obj_cam[:,0]
        x = x_raw - np.dot(x_raw, z)*z
        if np.linalg.norm(x) < 1e-6:
            return R_obj_cam
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R_locked = np.stack([x, y, z], axis=1)
        return R_locked


    def update(self, center3d, axes3, pts3d):
        # 1) optional plane lock to stabilize roll/pitch
        if self.use_plane_lock:
            n, _ = fit_plane_ransac(pts3d)
            if n is not None:
                self.z_up_ref = n if self.z_up_ref is None else 0.8*self.z_up_ref + 0.2*n
                axes3 = self.lock_roll_pitch(axes3, self.z_up_ref)


        # 2) temporal smoothing (SO(3) SLERP + EMA for translation)
        if self.R_prev is None:
            self.R_prev = axes3.copy()
            self.t_prev = center3d.copy()
            return center3d, axes3


        R_s = slerp_SO3(self.R_prev, axes3, self.alpha_R)
        t_s = (1.0 - self.alpha_t)*self.t_prev + self.alpha_t*center3d


        self.R_prev = R_s
        self.t_prev = t_s
        return t_s, R_s


# --------------------
# Simple drawing
# --------------------
def draw_obb3d_on_image(img, intr, corners3d, color=(200,50,200), th=2):
    uv, valid = project_points_intr(intr, corners3d)
    H, W = img.shape[:2]
    edges = [(i, i ^ (1<<b)) for i in range(8) for b in (0,1,2) if i < (i ^ (1<<b))]
    for i, j in edges:
        if not (valid[i] and valid[j]):
            continue
        x1,y1 = int(round(uv[i,0])), int(round(uv[i,1]))
        x2,y2 = int(round(uv[j,0])), int(round(uv[j,1]))
        if 0<=x1<W and 0<=y1<H and 0<=x2<W and 0<=y2<H:
            cv2.line(img, (x1,y1), (x2,y2), color, th, cv2.LINE_AA)


def draw_axes3d(img, intr, center3d, axes, lengths):
    pts = np.vstack([center3d.reshape(1,3),
                     center3d + axes[:,0]*(0.5*lengths[0]),
                     center3d + axes[:,1]*(0.5*lengths[1]),
                     center3d + axes[:,2]*(0.5*lengths[2])])
    uv, valid = project_points_intr(intr, pts)
    if not np.all(valid):
        return
    c = tuple(map(int, np.round(uv[0]).tolist()))
    cols = [(0,0,255),(0,255,0),(255,0,0)]
    for k in range(3):
        e = tuple(map(int, np.round(uv[k+1]).tolist()))
        cv2.arrowedLine(img, c, e, cols[k], 2, tipLength=0.12)


# --------------------
# Main loop (minimal)
# --------------------


def main():
    model = YOLO(WEIGHTS)
    names = model.names


    pipe = rs.pipeline(); cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    prof = pipe.start(cfg)


    depth_sensor = prof.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)


    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    x_map, y_map = precompute_xy_maps(intr, COLOR_H, COLOR_W)


    # example extrinsic
    H_BC = H_from_axis_angles(TX, TY, TZ,
                              math.radians(RX_DEG), math.radians(RY_DEG), math.radians(RZ_DEG),
                              order=ORDER)
    R_BC = H_BC[:3,:3]


    # NEW: pose stabilizer (roll/pitch lock + temporal smoothing)
    stabilizer = PoseStabilizer(alpha_R=0.25, alpha_t=0.3, use_plane_lock=True)


    t0 = time.time(); n=0; fps=None


    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            d = aligned.get_depth_frame(); c = aligned.get_color_frame()
            if not d or not c:
                continue


            color = np.asanyarray(c.get_data())
            overlay = color.copy()


            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            r = model(rgb, conf=CONF_TH, iou=IOU_TH, verbose=False)[0]


            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
            clses = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), int)
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None


            for i in range(len(boxes)):
                cls_name = names.get(int(clses[i]), str(int(clses[i])))
                conf = float(confs[i])


                # prefer mask → depth backprojection
                if masks is not None and i < masks.shape[0]:
                    mask = (masks[i] > 0.5).astype(np.uint8)
                else:
                    # bbox fallback mask
                    H, W = COLOR_H, COLOR_W
                    x1,y1,x2,y2 = boxes[i].astype(int)
                    mask = np.zeros((H,W), np.uint8)
                    mask[max(0,y1):min(H,y2+1), max(0,x1):min(W,x2+1)] = 1


                pts3d = mask_to_points3d(d, mask, depth_scale, intr, x_map, y_map,
                                         sample_stride=SAMPLE_STRIDE, z_min=Z_MIN, z_max=Z_MAX, erosion=3)
                if pts3d is None:
                    continue


                center3d, axes3, lens3, corners3d = pca_obb_3d(pts3d)
                axes3 = ensure_right_handed(axes3)
                # --- stabilize pose (roll/pitch lock via plane + temporal smoothing)
                center3d, axes3 = stabilizer.update(center3d, axes3, pts3d)


                # draw 3D box and axes
                draw_obb3d_on_image(overlay, intr, corners3d)
                draw_axes3d(overlay, intr, center3d, axes3, lens3)


                # label with class, conf, Z
                uv_c, ok = project_points_intr(intr, center3d.reshape(1,3))
                if ok[0]:
                    cx, cy = int(round(uv_c[0,0])), int(round(uv_c[0,1]))
                    label = f"{cls_name} {conf:.2f}  Z:{center3d[2]:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
                    y_text = max(cy, th+8)
                    cv2.rectangle(overlay, (cx, y_text-th-6), (cx+tw+6, y_text), (200,50,200), -1)
                    cv2.putText(overlay, label, (cx+3, y_text-4), FONT, 0.55, (0,0,0), 2, cv2.LINE_AA)


                    # yaw around base Z (object X axis projected onto base XY)
                    objX = (H_BC[:3,:3] @ axes3)[:,0]
                    yaw_deg = math.degrees(math.atan2(float(objX[1]), float(objX[0])))
                    yaw_label = f"Yaw(Z): {yaw_deg:+.1f}"
                    (tw2, th2), _ = cv2.getTextSize(yaw_label, FONT, 0.7, 2)
                    y2 = y_text + th2 + 12
                    cv2.rectangle(overlay, (cx, y2-th2-6), (cx+tw2+6, y2), (70,180,255), -1)
                    cv2.putText(overlay, yaw_label, (cx+3, y2-4), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)


            # FPS
            n += 1
            if n >= 10:
                now = time.time(); fps = n / (now - t0); t0 = now; n = 0
            if fps is not None:
                cv2.putText(overlay, f"FPS: {fps:.1f}", (12, 28), FONT, 0.8, (50,50,255), 2, cv2.LINE_AA)


            cv2.imshow("RealSense YOLO (Core)", overlay)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        pipe.stop()
        try: cv2.destroyAllWindows()
        except: pass


if __name__ == "__main__":
    main()




import time
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


# --------------------
# Config (keep it short)
# --------------------
WEIGHTS = "../train_results/exp1/weights/best.pt"
CONF_TH = 0.5
IOU_TH  = 0.5
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30
SAMPLE_STRIDE = 2
Z_MIN, Z_MAX = 0.1, 2.0
FONT = cv2.FONT_HERSHEY_SIMPLEX


# Base←Cam extrinsic (example; replace with calibrated values)
RX_DEG, RY_DEG, RZ_DEG = -180.0, 0.0, -80.0
TX, TY, TZ = 0.20, 0.0, 0.50
ORDER = "XYZ"


# --------------------
# Small math helpers
# --------------------
def Rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)


def Ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], float)


def Rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)


def H_from_axis_angles(tx, ty, tz, rx, ry, rz, order="XYZ"):
    R_map = {"X": Rx(rx), "Y": Ry(ry), "Z": Rz(rz)}
    R = np.eye(3)
    for ax in order:
        R = R @ R_map[ax]
    H = np.eye(4)
    H[:3,:3] = R
    H[:3, 3] = [tx, ty, tz]
    return H


def ensure_right_handed(R):
    R = np.asarray(R, float).copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


# --------------------
# Camera projections
# --------------------
def precompute_xy_maps(intr, H, W):
    js = np.arange(W, dtype=np.float32)
    is_ = np.arange(H, dtype=np.float32)
    gy, gx = np.meshgrid(is_, js, indexing="ij")
    x_map = (gx - intr.ppx) / intr.fx
    y_map = (gy - intr.ppy) / intr.fy
    return x_map, y_map


def project_points_intr(intr, pts3d):
    pts3d = np.asarray(pts3d, dtype=np.float32)
    Z = pts3d[:, 2]
    valid = Z > 1e-6
    uv = np.full((pts3d.shape[0], 2), np.nan, np.float32)
    uv[valid, 0] = intr.fx * (pts3d[valid, 0] / Z[valid]) + intr.ppx
    uv[valid, 1] = intr.fy * (pts3d[valid, 1] / Z[valid]) + intr.ppy
    return uv, valid


# --------------------
# Depth → 3D points (mask-based)
# --------------------
def mask_to_points3d(depth_frame, mask, depth_scale, intr, x_map, y_map,
                     sample_stride=1, z_min=0.0, z_max=10.0, erosion=3):
    depth_u16 = np.asanyarray(depth_frame.get_data())
    H, W = depth_u16.shape[:2]
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    if erosion > 0:
        k = np.ones((erosion*2+1, erosion*2+1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), k, 1)
    m = (mask > 0)
    if sample_stride > 1:
        m = m[::sample_stride, ::sample_stride]
        Zf = depth_u16.astype(np.float32) * depth_scale
        Z = Zf[::sample_stride, ::sample_stride]
        X = (x_map * Zf)[::sample_stride, ::sample_stride]
        Y = (y_map * Zf)[::sample_stride, ::sample_stride]
    else:
        Z = depth_u16.astype(np.float32) * depth_scale
        X = x_map * Z
        Y = y_map * Z
    if not np.any(m):
        return None
    Xv, Yv, Zv = X[m], Y[m], Z[m]
    valid = (Zv > 0) & np.isfinite(Zv) & (Zv >= z_min) & (Zv <= z_max)
    if not np.any(valid):
        return None
    pts = np.stack([Xv[valid], Yv[valid], Zv[valid]], axis=1)
    return pts if pts.shape[0] >= 30 else None


# --------------------
# PCA OBB (fast, minimal)
# --------------------
def pca_obb_3d(points_xyz):
    pts = points_xyz.astype(np.float32)
    mean = pts.mean(axis=0)
    C = np.cov((pts - mean), rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    axes = axes / (np.linalg.norm(axes, axis=0, keepdims=True) + 1e-9)
    proj = (pts - mean) @ axes
    mins, maxs = proj.min(axis=0), proj.max(axis=0)
    c_local = (mins + maxs) * 0.5
    half = (maxs - mins) * 0.5
    center = mean + axes @ c_local
    corners = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                corners.append(center + s1*half[0]*axes[:,0] + s2*half[1]*axes[:,1] + s3*half[2]*axes[:,2])
    corners = np.stack(corners, axis=0)
    lengths = 2.0 * half
    return center, axes, lengths, corners


# --------------------
# NEW: Robust pose stabilization (roll/pitch lock + temporal smoothing)
# --------------------


def so3_log(R):
    """Matrix log for SO(3) -> axis*angle (vector)"""
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * theta


def so3_exp(w):
    """Axis-angle vector -> SO(3) using Rodrigues"""
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K@K)


def slerp_SO3(R0, R1, alpha):
    """Geodesic interpolation on SO(3)."""
    dR = R0.T @ R1
    w = so3_log(dR)
    return R0 @ so3_exp(alpha * w)


def fit_plane_ransac(pts, iters=100, tau=0.01):
    """RANSAC plane fit: returns unit normal (z_up) and offset d so that n^T x + d = 0"""
    N = pts.shape[0]
    if N < 50:
        return None, None
    best_inl, best = 0, (None, None)
    rng = np.random.default_rng()
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        a,b,c = pts[idx]
        n = np.cross(b-a, c-a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            continue
        n = n / n_norm
        d = -np.dot(n, a)
        dist = np.abs(pts @ n + d)
        inl = np.count_nonzero(dist < tau)
        if inl > best_inl:
            best_inl = inl; best = (n, d)
    if best[0] is None:
        return None, None
    # ensure z-up (flip if upside down)
    n = best[0]
    if n[2] < 0: n = -n
    return n / (np.linalg.norm(n)+1e-9), best[1]


class PoseStabilizer:
    """Temporal smoothing + roll/pitch lock via plane normal or IMU gravity."""
    def __init__(self, alpha_R=0.25, alpha_t=0.3, use_plane_lock=True):
        self.R_prev = None
        self.t_prev = None
        self.alpha_R = alpha_R
        self.alpha_t = alpha_t
        self.use_plane_lock = use_plane_lock
        self.z_up_ref = None  # set from first reliable plane or external IMU


    def lock_roll_pitch(self, R_obj_cam, z_up):
        """Project object X/Y axes onto plane orthogonal to z_up to remove roll/pitch."""
        # build a camera/world frame where z = z_up
        z = z_up / (np.linalg.norm(z_up)+1e-9)
        # choose x as projection of current x onto plane, then y = z×x
        x_raw = R_obj_cam[:,0]
        x = x_raw - np.dot(x_raw, z)*z
        if np.linalg.norm(x) < 1e-6:
            return R_obj_cam
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R_locked = np.stack([x, y, z], axis=1)
        return R_locked


    def update(self, center3d, axes3, pts3d):
        # 1) optional plane lock to stabilize roll/pitch
        if self.use_plane_lock:
            n, _ = fit_plane_ransac(pts3d)
            if n is not None:
                self.z_up_ref = n if self.z_up_ref is None else 0.8*self.z_up_ref + 0.2*n
                axes3 = self.lock_roll_pitch(axes3, self.z_up_ref)


        # 2) temporal smoothing (SO(3) SLERP + EMA for translation)
        if self.R_prev is None:
            self.R_prev = axes3.copy()
            self.t_prev = center3d.copy()
            return center3d, axes3


        R_s = slerp_SO3(self.R_prev, axes3, self.alpha_R)
        t_s = (1.0 - self.alpha_t)*self.t_prev + self.alpha_t*center3d


        self.R_prev = R_s
        self.t_prev = t_s
        return t_s, R_s


# --------------------
# Simple drawing
# --------------------
def draw_obb3d_on_image(img, intr, corners3d, color=(200,50,200), th=2):
    uv, valid = project_points_intr(intr, corners3d)
    H, W = img.shape[:2]
    edges = [(i, i ^ (1<<b)) for i in range(8) for b in (0,1,2) if i < (i ^ (1<<b))]
    for i, j in edges:
        if not (valid[i] and valid[j]):
            continue
        x1,y1 = int(round(uv[i,0])), int(round(uv[i,1]))
        x2,y2 = int(round(uv[j,0])), int(round(uv[j,1]))
        if 0<=x1<W and 0<=y1<H and 0<=x2<W and 0<=y2<H:
            cv2.line(img, (x1,y1), (x2,y2), color, th, cv2.LINE_AA)


def draw_axes3d(img, intr, center3d, axes, lengths):
    pts = np.vstack([center3d.reshape(1,3),
                     center3d + axes[:,0]*(0.5*lengths[0]),
                     center3d + axes[:,1]*(0.5*lengths[1]),
                     center3d + axes[:,2]*(0.5*lengths[2])])
    uv, valid = project_points_intr(intr, pts)
    if not np.all(valid):
        return
    c = tuple(map(int, np.round(uv[0]).tolist()))
    cols = [(0,0,255),(0,255,0),(255,0,0)]
    for k in range(3):
        e = tuple(map(int, np.round(uv[k+1]).tolist()))
        cv2.arrowedLine(img, c, e, cols[k], 2, tipLength=0.12)


# --------------------
# Main loop (minimal)
# --------------------


def main():
    model = YOLO(WEIGHTS)
    names = model.names


    pipe = rs.pipeline(); cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    prof = pipe.start(cfg)


    depth_sensor = prof.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)


    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    x_map, y_map = precompute_xy_maps(intr, COLOR_H, COLOR_W)


    # example extrinsic
    H_BC = H_from_axis_angles(TX, TY, TZ,
                              math.radians(RX_DEG), math.radians(RY_DEG), math.radians(RZ_DEG),
                              order=ORDER)
    R_BC = H_BC[:3,:3]


    # NEW: pose stabilizer (roll/pitch lock + temporal smoothing)
    stabilizer = PoseStabilizer(alpha_R=0.25, alpha_t=0.3, use_plane_lock=True)


    t0 = time.time(); n=0; fps=None


    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            d = aligned.get_depth_frame(); c = aligned.get_color_frame()
            if not d or not c:
                continue


            color = np.asanyarray(c.get_data())
            overlay = color.copy()


            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            r = model(rgb, conf=CONF_TH, iou=IOU_TH, verbose=False)[0]


            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
            clses = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), int)
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None


            for i in range(len(boxes)):
                cls_name = names.get(int(clses[i]), str(int(clses[i])))
                conf = float(confs[i])


                # prefer mask → depth backprojection
                if masks is not None and i < masks.shape[0]:
                    mask = (masks[i] > 0.5).astype(np.uint8)
                else:
                    # bbox fallback mask
                    H, W = COLOR_H, COLOR_W
                    x1,y1,x2,y2 = boxes[i].astype(int)
                    mask = np.zeros((H,W), np.uint8)
                    mask[max(0,y1):min(H,y2+1), max(0,x1):min(W,x2+1)] = 1


                pts3d = mask_to_points3d(d, mask, depth_scale, intr, x_map, y_map,
                                         sample_stride=SAMPLE_STRIDE, z_min=Z_MIN, z_max=Z_MAX, erosion=3)
                if pts3d is None:
                    continue


                center3d, axes3, lens3, corners3d = pca_obb_3d(pts3d)
                axes3 = ensure_right_handed(axes3)
                # --- stabilize pose (roll/pitch lock via plane + temporal smoothing)
                center3d, axes3 = stabilizer.update(center3d, axes3, pts3d)


                # draw 3D box and axes
                draw_obb3d_on_image(overlay, intr, corners3d)
                draw_axes3d(overlay, intr, center3d, axes3, lens3)


                # label with class, conf, Z
                uv_c, ok = project_points_intr(intr, center3d.reshape(1,3))
                if ok[0]:
                    cx, cy = int(round(uv_c[0,0])), int(round(uv_c[0,1]))
                    label = f"{cls_name} {conf:.2f}  Z:{center3d[2]:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
                    y_text = max(cy, th+8)
                    cv2.rectangle(overlay, (cx, y_text-th-6), (cx+tw+6, y_text), (200,50,200), -1)
                    cv2.putText(overlay, label, (cx+3, y_text-4), FONT, 0.55, (0,0,0), 2, cv2.LINE_AA)


                    # yaw around base Z (object X axis projected onto base XY)
                    objX = (H_BC[:3,:3] @ axes3)[:,0]
                    yaw_deg = math.degrees(math.atan2(float(objX[1]), float(objX[0])))
                    yaw_label = f"Yaw(Z): {yaw_deg:+.1f}"
                    (tw2, th2), _ = cv2.getTextSize(yaw_label, FONT, 0.7, 2)
                    y2 = y_text + th2 + 12
                    cv2.rectangle(overlay, (cx, y2-th2-6), (cx+tw2+6, y2), (70,180,255), -1)
                    cv2.putText(overlay, yaw_label, (cx+3, y2-4), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)


            # FPS
            n += 1
            if n >= 10:
                now = time.time(); fps = n / (now - t0); t0 = now; n = 0
            if fps is not None:
                cv2.putText(overlay, f"FPS: {fps:.1f}", (12, 28), FONT, 0.8, (50,50,255), 2, cv2.LINE_AA)


            cv2.imshow("RealSense YOLO (Core)", overlay)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        pipe.stop()
        try: cv2.destroyAllWindows()
        except: pass


if __name__ == "__main__":
    main()



