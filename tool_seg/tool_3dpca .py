import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import time
import math
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

WEIGHTS = "../train_results/exp1/weights/best.pt" 
DEVICE  = "0"  # CPU 사용으로 변경 # written by DM 20250904
CONF_TH = 0.5                      
IOU_TH  = 0.5                      
LABEL_FILTER = None                
DRAW_MASK_ALPHA = 0.4               
DEPTH_KERNEL = 5                    # bbox 중심 주변 (k x k) median depth (디텍션 모델용)
FONT = cv2.FONT_HERSHEY_SIMPLEX


EXCLUDE_CLASS_IDS = []

COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30

TARGET_LABELS = {"nipper", "vernier_calipers", "wire_cutter", "wire_stripper"}

SHOW_WINDOW = True              
FALLBACK_SAVE_EVERY_N = 15         
FALLBACK_SAVE_DIR = "/tmp/tool_pca_frames"

DRAW_3D_OBB = True          
SAMPLE_STRIDE = 2                   # 3 → 2로 완화 (포인트 밀도 2배 증가) # written by DM 20250904
Z_MIN, Z_MAX = 0.1, 2.0             # 0.15-1.20 → 0.1-2.0으로 범위 확대 # written by DM 20250904


def Rx(theta):
    """x축 회전행렬 (3x3), theta: rad"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=float)

def Ry(theta):
    """y축 회전행렬 (3x3), theta: rad"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

def Rz(theta):
    """z축 회전행렬 (3x3), theta: rad"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=float)

def H_from_R_t(R, t):
    """회전 R(3x3), 평행이동 t(3,) -> 4x4 동차변환"""
    H = np.eye(4, dtype=float)
    H[:3, :3] = np.asarray(R, dtype=float)
    H[:3, 3]  = np.asarray(t, dtype=float).reshape(3,)
    return H

def H_from_axis_angles(tx, ty, tz, rx, ry, rz, order="XYZ"):
    """
    축별 회전각(rx,ry,rz)과 평행이동(tx,ty,tz)로 4x4 생성.
    order: 회전 적용 순서 문자열. "XYZ"라면 최종 R = Rz@Ry@Rx (오른쪽부터 적용)
    """
    R_map = {"X": Rx(rx), "Y": Ry(ry), "Z": Rz(rz)}
    R = np.eye(3, dtype=float)
    for ax in order[::-1]:
        R = R @ R_map[ax] 
    return H_from_R_t(R, [tx, ty, tz])

def transform_points(H, pts):
    """H(4x4)로 점(3,) 또는 (N,3) 변환."""
    P = np.asarray(pts, dtype=float)
    if P.ndim == 1:
        Ph = np.hstack([P, 1.0])
        Qh = H @ Ph
        return Qh[:3]
    elif P.ndim == 2 and P.shape[1] == 3:
        ones = np.ones((P.shape[0], 1), dtype=float)
        Ph = np.hstack([P, ones])
        Qh = (H @ Ph.T).T
        return Qh[:, :3]
    else:
        raise ValueError("pts shape must be (3,) or (N,3)")


# =======================
# 유틸 함수 (공통)
# =======================
def apply_mask_overlay(overlay_bgr, mask, alpha=0.4, color=(0, 255, 255)):
    """세그멘테이션 마스크를 색상 오버레이로 표시"""
    H, W = overlay_bgr.shape[:2]
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    if mask.shape[0] != H or mask.shape[1] != W:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    out = overlay_bgr.copy()
    color_img = np.zeros_like(out, dtype=np.uint8); color_img[:] = color
    m = mask.astype(bool)
    out[m] = ((1.0 - alpha) * out[m] + alpha * color_img[m]).astype(np.uint8)
    return out

def median_depth_meters_from_center(depth_frame, cx, cy, k=5, depth_scale=0.001):
    """깊이 프레임에서 bbox 중심 주변 kxk 패치의 median depth(m)를 계산."""
    depth_image = np.asanyarray(depth_frame.get_data())
    H, W = depth_image.shape[:2]
    x1 = max(0, cx - k // 2); x2 = min(W - 1, cx + k // 2)
    y1 = max(0, cy - k // 2); y2 = min(H - 1, cy + k // 2)
    patch = depth_image[y1:y2 + 1, x1:x2 + 1]
    if patch.size == 0:
        return None
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale

def median_depth_meters_from_mask(depth_frame, mask, depth_scale=0.001):
    """세그 마스크 내부 픽셀의 median depth(m)를 계산."""
    depth_image = np.asanyarray(depth_frame.get_data())
    H_d, W_d = depth_image.shape[:2]
    if mask.shape[0] != H_d or mask.shape[1] != W_d:
        mask = cv2.resize(mask.astype(np.uint8), (W_d, H_d), interpolation=cv2.INTER_NEAREST)
    m = mask.astype(bool)
    if not np.any(m):
        return None
    valid = depth_image[m]; valid = valid[valid > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale

def pca_obb_from_mask(mask, out_size_hw):
    """2D PCA → OBB(중심, 반축 길이, 각도, 꼭짓점)."""
    H, W = out_size_hw
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 0)
    if xs.size < 10:
        return None, None, None, None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    u = vecs[:, order[0]]; v = vecs[:, order[1]]
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    R = np.stack([u, v], axis=1)
    proj = centered @ R
    mins = proj.min(axis=0); maxs = proj.max(axis=0)
    c_local = (mins + maxs) * 0.5
    center = mean + R @ c_local
    a = (maxs[0] - mins[0]) * 0.5
    b = (maxs[1] - mins[1]) * 0.5
    c = center
    c1 = c + (+a)*u + (+b)*v
    c2 = c + (+a)*u + (-b)*v
    c3 = c + (-a)*u + (-b)*v
    c4 = c + (-a)*u + (+b)*v
    corners = np.stack([c1, c2, c3, c4], axis=0)
    angle_deg = math.degrees(math.atan2(u[1], u[0]))
    return corners.astype(np.int32), (2*a, 2*b), angle_deg, center

def draw_obb(overlay, corners, color=(0, 180, 255), thickness=2):
    """corners: (4,2) int32 시계방향"""
    cv2.polylines(overlay, [corners], isClosed=True, color=color, thickness=thickness)


# =======================
# 3D용 보조 함수
# =======================
def precompute_xy_maps(intr, H, W):
    """내참수로부터 X/Z, Y/Z 계수 맵 계산."""
    js = np.arange(W, dtype=np.float32)
    is_ = np.arange(H, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(is_, js, indexing="ij")
    x_map = (grid_x - intr.ppx) / intr.fx
    y_map = (grid_y - intr.ppy) / intr.fy
    return x_map, y_map

def mask_to_points3d(depth_frame, mask, depth_scale, intr, x_map, y_map,
                     sample_stride=1, z_min=0.0, z_max=10.0):
    """마스크 영역의 3D 점(Nx3)을 추출."""
    depth_u16 = np.asanyarray(depth_frame.get_data())
    H, W = depth_u16.shape[:2]
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    m = (mask > 0)
    if sample_stride > 1:
        m = m[::sample_stride, ::sample_stride]
        # [OPT] depth_u16.astype(np.float32) * depth_scale는 한 번만 계산하고 재사용하세요.
        Zfull = (depth_u16.astype(np.float32) * depth_scale)
        Z = Zfull[::sample_stride, ::sample_stride]
        X = (x_map * Zfull)[::sample_stride, ::sample_stride]
        Y = (y_map * Zfull)[::sample_stride, ::sample_stride]
    else:
        Z = depth_u16.astype(np.float32) * depth_scale
        X = x_map * Z
        Y = y_map * Z
    if not np.any(m):
        return None
    Xv = X[m]; Yv = Y[m]; Zv = Z[m]
    valid = (Zv > 0) & np.isfinite(Zv) & (Zv >= z_min) & (Zv <= z_max)
    if not np.any(valid):
        return None
    pts = np.stack([Xv[valid], Yv[valid], Zv[valid]], axis=1)
    if pts.shape[0] < 30:
        return None
    return pts

def pca_obb_3d(points_xyz):
    """3D PCA로 OBB 계산."""
    pts = points_xyz.astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    for k in range(3):
        n = np.linalg.norm(axes[:, k])
        if n > 0:
            axes[:, k] /= n
    proj = centered @ axes
    mins = proj.min(axis=0); maxs = proj.max(axis=0)
    c_local = (mins + maxs) * 0.5
    half = (maxs - mins) * 0.5
    center = mean + axes @ c_local
    corners = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                corner = center + s1*half[0]*axes[:,0] + s2*half[1]*axes[:,1] + s3*half[2]*axes[:,2]
                corners.append(corner)
    corners = np.stack(corners, axis=0)
    lengths = 2.0 * half
    return center, axes, lengths, corners

def project_points_intr(intr, pts3d):
    """내참수(intr)로 3D 점들을 픽셀 좌표로 투영."""
    pts3d = np.asarray(pts3d, dtype=np.float32)
    Z = pts3d[:, 2]
    valid = Z > 1e-6
    uv = np.full((pts3d.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = intr.fx * (pts3d[valid, 0] / Z[valid]) + intr.ppx
    uv[valid, 1] = intr.fy * (pts3d[valid, 1] / Z[valid]) + intr.ppy
    return uv, valid

def draw_obb3d_on_image(overlay, intr, corners3d, color=(200, 50, 200), thickness=2):
    """3D OBB 모서리를 이미지에 그림."""
    uv, valid = project_points_intr(intr, corners3d)
    edges = []
    for i in range(8):
        for bit in (0, 1, 2):
            j = i ^ (1 << bit)
            if i < j:
                edges.append((i, j))
    H, W = overlay.shape[:2]
    for i, j in edges:
        if not (valid[i] and valid[j]):
            continue
        x1, y1 = int(round(uv[i, 0])), int(round(uv[i, 1]))
        x2, y2 = int(round(uv[j, 0])), int(round(uv[j, 1]))
        if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

def draw_axes3d(overlay, intr, center3d, axes, lengths, scale=0.5):
    """3개 주성분 축을 색상으로 표시(X=red, Y=green, Z=blue)."""
    Ls = 0.5 * lengths * float(scale)
    endpoints = [center3d + axes[:, k] * Ls[k] for k in range(3)]
    pts = np.vstack([center3d.reshape(1, 3), np.stack(endpoints, axis=0)])
    uv, valid = project_points_intr(intr, pts)
    if not np.all(valid):
        return
    c = tuple(map(int, np.round(uv[0]).tolist()))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
    H, W = overlay.shape[:2]
    for k in range(3):
        e = tuple(map(int, np.round(uv[k + 1]).tolist()))
        if (0 <= c[0] < W and 0 <= c[1] < H and 0 <= e[0] < W and 0 <= e[1] < H):
            cv2.arrowedLine(overlay, c, e, colors[k], 2, tipLength=0.1)


def main():
    # YOLO 로드
    model = YOLO(WEIGHTS)
    names = model.names  # {class_id: class_name}

    # Screwdriver(드라이버) 클래스 ID 찾기 (대소문자 무시)
    target_lower = {t.lower() for t in TARGET_LABELS}
    ALLOWED_CLASS_IDS = [i for i, n in names.items() if n.lower() in target_lower]
    if not ALLOWED_CLASS_IDS:
        print("[WARN] TARGET_LABELS가 모델 클래스에 없습니다. 전체 클래스로 추론합니다.")

    # RealSense 파이프라인
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    profile = pipeline.start(config)

    # 깊이 스케일, 정렬(깊이를 컬러에 정렬)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # 일반적으로 0.001 (mm→m)
    align = rs.align(rs.stream.color)

    # 색상 카메라 내참수
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()  # fx, fy, ppx, ppy, width, height

    # X/Z, Y/Z 맵 미리 계산 (프레임마다 재계산 불필요)
    x_map, y_map = precompute_xy_maps(intr, COLOR_H, COLOR_W)

    # 선택: depth post-processing
    spat_filter = rs.spatial_filter()
    temp_filter = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter(1)

    # === 추가: 베이스←카메라 H_BC 설정 (각도: deg → rad) ===
    # ▼ 실제 캘리브레이션 값으로 교체하세요.
    RX_DEG, RY_DEG, RZ_DEG = -180.0, 0.0, -90.0   # 베이스 기준 카메라의 회전(도)
    TX, TY, TZ = 0.20, 0.0, 0.50               # 베이스 기준 카메라의 위치(m)
    ORDER = "XYZ"                             # 회전 적용 순서 문자열 (예: "XYZ", "ZYX" 등)

    H_BC = H_from_axis_angles(
        TX, TY, TZ,
        math.radians(RX_DEG), math.radians(RY_DEG), math.radians(RZ_DEG),
        order=ORDER
    )
    # H_BC: Base ← Cam (카메라 좌표를 베이스 좌표로 보냄)

    fps_time = time.time()
    frame_count = 0
    fps = None

    # 헤드리스 저장 준비
    show_window = SHOW_WINDOW
    if not show_window:
        os.makedirs(FALLBACK_SAVE_DIR, exist_ok=True)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 선택적 필터
            depth_frame = spat_filter.process(depth_frame)
            depth_frame = temp_filter.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            color = np.asanyarray(color_frame.get_data())  # BGR uint8 (H x W x 3)
            overlay = color.copy()

            # YOLO 추론: RGB 입력 권장
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            results = model(
                rgb,
                conf=CONF_TH,
                iou=IOU_TH,
                device=DEVICE,
                verbose=False,
                classes=ALLOWED_CLASS_IDS if ALLOWED_CLASS_IDS else None,
            )

            if results and len(results) > 0:
                r = results[0]

                # 박스/클래스/신뢰도
                if getattr(r, "boxes", None) is not None and r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    clses = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                else:
                    boxes = np.zeros((0, 4))
                    clses = np.zeros((0,), dtype=int)
                    confs = np.zeros((0,))

                # 마스크 (세그 모델일 때만)
                if getattr(r, "masks", None) is not None and r.masks is not None:
                    masks_np = r.masks.data.cpu().numpy()
                else:
                    masks_np = None

                N = len(boxes)
                for i in range(N):
                    c = int(clses[i])
                    if c in EXCLUDE_CLASS_IDS:
                        continue

                    x1, y1, x2, y2 = boxes[i]
                    cls_name = names.get(c, str(c))
                    conf = float(confs[i])

                    if LABEL_FILTER and cls_name not in LABEL_FILTER:
                        continue

                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

                    three_d_done = False
                    y_text = None  # 라벨 y 위치 보관(베이스 라벨 위치 잡기용)

                    if DRAW_3D_OBB:
                        if masks_np is not None and i < masks_np.shape[0]:
                            raw_mask = masks_np[i]
                            pts3d = mask_to_points3d(
                                depth_frame, raw_mask, depth_scale, intr,
                                x_map, y_map, sample_stride=SAMPLE_STRIDE,
                                z_min=Z_MIN, z_max=Z_MAX
                            )
                        else:
                            # 세그가 없으면 bbox 내부를 간이 마스크로 사용
                            mask_bbox = np.zeros((COLOR_H, COLOR_W), dtype=np.uint8)
                            x1c = max(0, x1i); y1c = max(0, y1i)
                            x2c = min(COLOR_W - 1, x2i); y2c = min(COLOR_H - 1, y2i)
                            mask_bbox[y1c:y2c + 1, x1c:x2c + 1] = 1
                            pts3d = mask_to_points3d(
                                depth_frame, mask_bbox, depth_scale, intr,
                                x_map, y_map, sample_stride=SAMPLE_STRIDE,
                                z_min=Z_MIN, z_max=Z_MAX
                            )

                        if pts3d is not None:
                            res = pca_obb_3d(pts3d)
                            if res is not None:
                                center3d, axes3, lens3, corners3d = res
                                # 3D OBB와 축 그리기 (카메라 프레임 기준 투영)
                                draw_obb3d_on_image(overlay, intr, corners3d, color=(200, 50, 200), thickness=2)
                                draw_axes3d(overlay, intr, center3d, axes3, lens3, scale=0.8)

                                # 중심 투영 위치에 라벨 (카메라 기준)
                                uv_c, ok = project_points_intr(intr, center3d.reshape(1, 3))
                                if ok[0]:
                                    cxp, cyp = int(round(uv_c[0, 0])), int(round(uv_c[0, 1]))
                                    dist_m = float(center3d[2])
                                    label3 = f"{cls_name} {conf:.2f} | Z={dist_m:.2f}m | 3D[{lens3[0]:.2f},{lens3[1]:.2f},{lens3[2]:.2f}]m"
                                    (tw, th), _ = cv2.getTextSize(label3, FONT, 0.55, 2)
                                    y_text = max(cyp, th + 8)
                                    cv2.rectangle(overlay, (cxp, y_text - th - 6), (cxp + tw + 6, y_text), (200, 50, 200), -1)
                                    cv2.putText(overlay, label3, (cxp + 3, y_text - 4), FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

                                    # === 추가: base 좌표 라벨 ===
                                    base_center3d = transform_points(H_BC, center3d)  # (3,)
                                    base_label = f"BASE: X={base_center3d[0]:.3f} Y={base_center3d[1]:.3f} Z={base_center3d[2]:.3f} m"
                                    (twb, thb), _ = cv2.getTextSize(base_label, FONT, 0.5, 1)
                                    y_text_b = (y_text + thb + 10) if y_text is not None else (cyp + thb + 10)
                                    cv2.rectangle(overlay, (cxp, y_text_b - thb - 6), (cxp + twb + 6, y_text_b), (120, 90, 240), -1)
                                    cv2.putText(overlay, base_label, (cxp + 3, y_text_b - 4), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                                three_d_done = True

                    # ───────────── 2D 마스크/디텍션 표시 (보조용/폴백) ─────────────
                    if masks_np is not None and i < masks_np.shape[0]:
                        mask = masks_np[i]
                        overlay = apply_mask_overlay(overlay, mask, alpha=DRAW_MASK_ALPHA, color=(0, 255, 255))

                        # 3D 실패시 2D OBB와 거리(마스크 median)
                        if not three_d_done:
                            dist_m = median_depth_meters_from_mask(depth_frame, mask, depth_scale)
                            corners, (w_len, h_len), angle_deg, center = pca_obb_from_mask(
                                mask, (overlay.shape[0], overlay.shape[1])
                            )
                            if corners is not None:
                                draw_obb(overlay, corners, color=(0, 180, 255), thickness=2)
                                label = f"{cls_name} {conf:.2f}"
                                if dist_m is not None:
                                    label += f" | {dist_m:.2f}m"
                                label2 = f"θ={angle_deg:.1f}°  {w_len:.1f}x{h_len:.1f}px"
                                (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
                                y_text = max(int(center[1]), th + 8)
                                x_text = int(center[0])
                                cv2.rectangle(overlay, (x_text, y_text - th - 6), (x_text + tw + 6, y_text), (0, 180, 255), -1)
                                cv2.putText(overlay, label, (x_text + 3, y_text - 4), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                                (tw2, th2), _ = cv2.getTextSize(label2, FONT, 0.5, 1)
                                y_text2 = y_text + th2 + 10
                                cv2.rectangle(overlay, (x_text, y_text2 - th2 - 6), (x_text + tw2 + 6, y_text2), (0, 180, 255), -1)
                                cv2.putText(overlay, label2, (x_text + 3, y_text2 - 4), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    else:
                        if not three_d_done:
                            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                            cx, cy = (x1i + x2i) // 2, (y1i + y2i) // 2
                            dist_m = median_depth_meters_from_center(
                                depth_frame, cx, cy, k=DEPTH_KERNEL, depth_scale=depth_scale
                            )
                            cv2.circle(overlay, (cx, cy), 3, (0, 255, 0), -1)
                            label = f"{cls_name} {conf:.2f}"
                            if dist_m is not None:
                                label += f" | {dist_m:.2f}m"
                            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
                            y_text = max(y1i, th + 8)
                            cv2.rectangle(overlay, (x1i, y_text - th - 6), (x1i + tw + 6, y_text), (0, 255, 0), -1)
                            cv2.putText(overlay, label, (x1i + 3, y_text - 4), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # FPS 계산/표시
            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - fps_time)
                fps_time = now
                frame_count = 0

            if fps is not None:
                txt = f"FPS: {fps:.1f}"
                cv2.putText(overlay, txt, (12, 28), FONT, 0.8, (50, 50, 255), 2, cv2.LINE_AA)

            # ── 화면 표시 또는 헤드리스 저장 ──
            if show_window:
                try:
                    cv2.imshow("RealSense YOLO (3D PCA OBB + Depth)", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except Exception as e:
                    print(f"[WARN] imshow 실패, 헤드리스로 전환합니다: {e}")
                    show_window = False
                    os.makedirs(FALLBACK_SAVE_DIR, exist_ok=True)
            else:
                if (fps is None) or (int(time.time() * 10) % FALLBACK_SAVE_EVERY_N == 0):
                    fp = f"{FALLBACK_SAVE_DIR}/frame_{int(time.time()*1000)}.jpg"
                    cv2.imwrite(fp, overlay)

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        try:
            cv2.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":
    print("ROS2 humble is activated!")  # 사용자가 남긴 로그 형식 유지
    main()
