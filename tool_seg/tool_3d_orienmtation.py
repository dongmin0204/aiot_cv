import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import time
import math
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import matplotlib.pyplot as plt  # 포인트 클라우드 시각화용 # written by DM 20250904
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯용 # written by DM 20250904

WEIGHTS = "../train_results/exp1/weights/best.pt"
DEVICE  = "0"  # CPU 사용으로 변경 # written by DM 20250904
CONF_TH = 0.5
IOU_TH  = 0.5
LABEL_FILTER = None
DRAW_MASK_ALPHA = 0.4
DEPTH_KERNEL = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX

EXCLUDE_CLASS_IDS = []

COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30
TARGET_LABELS = {"nipper", "vernier_calipers", "wire_cutter", "wire_stripper"}


SHOW_WINDOW = True
FALLBACK_SAVE_EVERY_N = 15
FALLBACK_SAVE_DIR = "/tmp/tool_pca_frames"

DRAW_3D_OBB = True
SAMPLE_STRIDE = 2  # 3 → 2로 완화 (포인트 밀도 2배 증가) # written by DM 20250904
Z_MIN, Z_MAX = 0.1, 2.0  # 0.15-1.20 → 0.1-2.0으로 범위 확대 # written by DM 20250904

# 포인트 클라우드 시각화 옵션 # written by DM 20250904
SHOW_POINTCLOUD = True  # 포인트 클라우드 창 표시
POINTCLOUD_UPDATE_INTERVAL = 5  # N프레임마다 업데이트
VISUALIZATION_MODE = "none"  # "matplotlib", "text", "none" # written by DM 20250904

# Matplotlib 인터랙티브 상태/객체 (재사용) # written by DM 20250904
_pc_fig = None
_pc_ax = None
_pc_scatter = None

# 포인트 클라우드 전처리 옵션 # written by DM 20250904
ENABLE_OUTLIER_REMOVAL = True  # outlier 제거 활성화
OUTLIER_RADIUS = 0.05  # 0.02 → 0.05로 반경 확대 (m) # written by DM 20250904
MIN_NEIGHBORS = 5  # 8 → 5로 최소 이웃 수 감소 # written by DM 20250904
STATISTICAL_OUTLIER = True  # SOR 사용
SOR_K_NEIGHBORS = 15  # 20 → 15로 k값 감소 # written by DM 20250904
SOR_STD_RATIO = 2.0  # 1.5 → 2.0으로 표준편차 비율 완화 # written by DM 20250904

# 마스크 품질 개선 옵션 # written by DM 20250904
ENABLE_MASK_REFINEMENT = True  # 마스크 정제 활성화
MASK_MIN_AREA = 50  # 100 → 50으로 최소 마스크 면적 완화 (픽셀) # written by DM 20250904
MASK_MIN_CONFIDENCE = 0.2  # 0.3 → 0.2로 최소 마스크 신뢰도 완화 # written by DM 20250904
MORPH_KERNEL_SIZE = 3  # 형태학적 연산 커널 크기
ENABLE_MULTI_FRAME_VOTING = True  # 멀티프레임 투표 활성화
VOTING_FRAMES = 3  # 투표에 사용할 프레임 수
MASK_IOU_THRESHOLD = 0.5  # 마스크 IoU 임계값

# =======================
# 좌표계/회전 유틸
# =======================
def Rx(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=float)

def Ry(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

def Rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=float)

def H_from_R_t(R, t):
    H = np.eye(4, dtype=float)
    H[:3, :3] = np.asarray(R, dtype=float)
    H[:3, 3]  = np.asarray(t, dtype=float).reshape(3,)
    return H

def H_from_axis_angles(tx, ty, tz, rx, ry, rz, order="XYZ"):
    """
    order="XYZ"면 R = Rx @ Ry @ Rz
    """
    R_map = {"X": Rx(rx), "Y": Ry(ry), "Z": Rz(rz)}
    R = np.eye(3, dtype=float)
    for ax in order:
        R = R @ R_map[ax]
    return H_from_R_t(R, [tx, ty, tz])

def transform_points(H, pts):
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

def ensure_right_handed(R):
    """PCA 축행렬(열벡터 [u1 u2 u3])이 오른손이 되도록 det<0이면 u3 반전"""
    R = np.asarray(R, dtype=float).copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R

def angle_between(v1, v2):
    """3D 단위벡터 각도(라디안)"""
    v1 = np.asarray(v1, dtype=float); v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = float(np.dot(v1/n1, v2/n2))
    c = max(-1.0, min(1.0, c))
    return math.acos(c)

# =======================
# 공통 유틸
# =======================
def apply_mask_overlay(overlay_bgr, mask, alpha=0.4, color=(0, 255, 255)):
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
    cv2.polylines(overlay, [corners], isClosed=True, color=color, thickness=thickness)

# =======================
# 3D 보조
# =======================
def precompute_xy_maps(intr, H, W):
    js = np.arange(W, dtype=np.float32)
    is_ = np.arange(H, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(is_, js, indexing="ij")
    x_map = (grid_x - intr.ppx) / intr.fx
    y_map = (grid_y - intr.ppy) / intr.fy
    return x_map, y_map

def mask_to_points3d(depth_frame, mask, depth_scale, intr, x_map, y_map,
                     sample_stride=1, z_min=0.0, z_max=10.0, erosion_pixels=3):
    """마스크 안쪽으로 들어가서 포인트 클라우드 추출 (Erosion 방식) # written by DM 20250904"""
    depth_u16 = np.asanyarray(depth_frame.get_data())
    H, W = depth_u16.shape[:2]
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    
    # 마스크 Erosion으로 경계에서 안쪽으로 들어가기 # written by DM 20250904
    if erosion_pixels > 0:
        kernel = np.ones((erosion_pixels*2+1, erosion_pixels*2+1), np.uint8)
        mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        m = (mask_eroded > 0)
        print(f"[DEBUG] Mask erosion: {np.sum(mask > 0)} → {np.sum(mask_eroded > 0)} pixels")  # written by DM 20250904
    else:
        m = (mask > 0)
    
    if sample_stride > 1:
        m = m[::sample_stride, ::sample_stride]
        # [OPT] 아래 depth_u16.astype(np.float32) * depth_scale 계산은 한 번만 수행하고 X/Y에 재사용하세요.
        Zfull = (depth_u16.astype(np.float32) * depth_scale)
        Z = Zfull[::sample_stride, ::sample_stride]
        X = (x_map * Zfull)[::sample_stride, ::sample_stride]
        Y = (y_map * Zfull)[::sample_stride, ::sample_stride]
    else:
        Z = depth_u16.astype(np.float32) * depth_scale
        X = x_map * Z
        Y = y_map * Z
    if not np.any(m):
        print("[DEBUG] No valid mask pixels after erosion")  # written by DM 20250904
        return None
    Xv = X[m]; Yv = Y[m]; Zv = Z[m]
    valid = (Zv > 0) & np.isfinite(Zv) & (Zv >= z_min) & (Zv <= z_max)
    if not np.any(valid):
        print(f"[DEBUG] No valid depth values (Z range: {z_min}-{z_max})")  # written by DM 20250904
        return None
    pts = np.stack([Xv[valid], Yv[valid], Zv[valid]], axis=1)
    print(f"[DEBUG] Final point cloud: {len(pts)} points")  # written by DM 20250904
    if pts.shape[0] < 30:
        print(f"[DEBUG] Too few points ({len(pts)} < 30), returning None")  # written by DM 20250904
        return None
    return pts

# =======================
# 실시간 PCA 안정화 클래스 written by dongmin. 2025.08.28
# =======================
class RealtimePCASmoother:
    
    def __init__(self, stability_thresh=1.5, power_iter_trigger=0.8):
        self.prev_axes = None
        self.frame_count = 0
        self.stability_thresh = stability_thresh  # 고유값 비율 임계치
        self.power_iter_trigger = power_iter_trigger  # Power iteration 트리거 임계치
        
    def power_iteration(self, cov_matrix, max_iter=3):

        """
        3D Power iteration for OBB 상당한 실시간 최적화.
        cov_matrix: (3,3) 공분산 행렬
        max_iter: 최대 반복 횟수
        반환:
          main_axis(3,), main_eigenval 기본 주축과 고유값
        """
        n = cov_matrix.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(max_iter):
            v = cov_matrix @ v
            v = v / np.linalg.norm(v)
        
        # 고유값 추정
        eigenval = v.T @ cov_matrix @ v
        return v, eigenval
    
    def check_projection_stability(self, vals):
        """
        정사영 분산 체크 - 고유값 비율로 안정성 판단.
        vals: (3,) float32 : 고유값 : 내림차순
        return bool (true/false) (true : 안정적, false: 불안정)
        """

        if len(vals) < 3:
            return False
        ratio1 = vals[0] / vals[1] if vals[1] > 1e-6 else float('inf')
        ratio2 = vals[1] / vals[2] if vals[2] > 1e-6 else float('inf')
        return ratio1 > self.stability_thresh and ratio2 > self.stability_thresh
    
    def fast_axis_alignment(self, axes_new):
        """
        제한없는 임의 축과 기존 축의 내적 기반 계산. 센서 노이즈가 난무해도 안정적.
        axes_new: (3,3) 새 장축 행렬 (정규화된 벡터로 주축 감지됨)
        return (3,3) 정렬된 축 행렬
        """
        if self.prev_axes is None:
            self.prev_axes = axes_new.copy()
            return axes_new
        
        axes_aligned = axes_new.copy()
        # 벡터화된 내적 계산
        dots = np.sum(axes_new * self.prev_axes, axis=0)
        flip_mask = dots < 0
        axes_aligned[:, flip_mask] *= -1
        
        self.prev_axes = axes_aligned
        return axes_aligned


# =======================
# 실시간 최적화된 3D PCA OBB 계산 함수 written by dongmin. 2025.08.28
# =======================
def pca_obb_3d(points_xyz, smoother=None):
    """
    실시간 최적화된 3D PCA OBB 계산
    반환:
      center(3,), axes(3,3) [열벡터 u1,u2,u3 = 장축, 단축, 세번째축], lengths(3,), corners(8,3)
    - 장축 = u1 = X축
    - 단축 = u2 = Y축
    """
    pts = points_xyz.astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    
    # 기본 고유분해
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    
    # 정규화
    for k in range(3):
        n = np.linalg.norm(axes[:, k])
        if n > 0:
            axes[:, k] /= n
    
    # 실시간 안정화 적용
    if smoother is not None:
        smoother.frame_count += 1
        
        # 1) 정사영 분산 체크
        is_stable = smoother.check_projection_stability(vals)
        
        # 2) 불안정할 때만 Power iteration 적용
        if not is_stable and smoother.frame_count % 5 == 0:  # 5프레임마다 체크
            try:
                # 주축만 Power iteration으로 재계산
                main_axis, main_eigenval = smoother.power_iteration(cov, max_iter=3)
                axes[:, 0] = main_axis
            except:
                pass  # 실패시 기본 결과 사용
        
        # 3) 빠른 축 정렬 (항상 적용)
        axes = smoother.fast_axis_alignment(axes)
    
    # OBB 계산
    proj = centered @ axes
    mins = proj.min(axis=0); maxs = proj.max(axis=0)
    c_local = (mins + maxs) * 0.5
    half = (maxs - mins) * 0.5
    center = mean + axes @ c_local
    
    # 8개 꼭짓점 생성
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
    pts3d = np.asarray(pts3d, dtype=np.float32)
    Z = pts3d[:, 2]
    valid = Z > 1e-6
    uv = np.full((pts3d.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = intr.fx * (pts3d[valid, 0] / Z[valid]) + intr.ppx
    uv[valid, 1] = intr.fy * (pts3d[valid, 1] / Z[valid]) + intr.ppy
    return uv, valid

def draw_obb3d_on_image(overlay, intr, corners3d, color=(200, 50, 200), thickness=2):
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

def draw_axes3d(overlay, intr, center3d, axes, lengths, scale=0.8):
    """PCA 3축: X(장축, 빨강), Y(단축, 초록), Z(파랑)"""
    Ls = 0.5 * lengths * float(scale)
    endpoints = [center3d + axes[:, k] * Ls[k] for k in range(3)]
    pts = np.vstack([center3d.reshape(1, 3), np.stack(endpoints, axis=0)])
    uv, valid = project_points_intr(intr, pts)
    if not np.all(valid):
        return
    c = tuple(map(int, np.round(uv[0]).tolist()))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X=red, Y=green, Z=blue
    H, W = overlay.shape[:2]
    for k in range(3):
        e = tuple(map(int, np.round(uv[k + 1]).tolist()))
        if (0 <= c[0] < W and 0 <= c[1] < H and 0 <= e[0] < W and 0 <= e[1] < H):
            cv2.arrowedLine(overlay, c, e, colors[k], 2, tipLength=0.12)

# =======================
# 회전 정보 배너(UI)
# =======================
def draw_rotation_banner(img, order, rx_deg, ry_deg, rz_deg, org=(12, 28), pad=8):
    line1 = f"Base \u2190 Cam Rotation  (order={order})"
    line2 = f"Rx={rx_deg:.1f}\u00B0,  Ry={ry_deg:.1f}\u00B0,  Rz={rz_deg:.1f}\u00B0"
    (w1, h1), _ = cv2.getTextSize(line1, FONT, 0.65, 2)
    (w2, h2), _ = cv2.getTextSize(line2, FONT, 0.65, 2)
    x, y = org
    box_w = max(w1, w2) + pad*2
    box_h = (h1 + h2) + pad*3
    x1, y1 = x - 4, y - h1 - pad
    x2, y2 = x1 + box_w + 8, y1 + box_h
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)
    cv2.putText(img, line1, (x + pad, y), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, line2, (x + pad, y + h1 + pad + 2), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

# 포인트 클라우드 시각화 함수 (Matplotlib 버전) # written by DM 20250904
def visualize_pointcloud(pts3d, center3d=None, axes3=None, lengths=None, title="Point Cloud"):
    """Matplotlib로 포인트 클라우드 시각화 (인터랙티브, 단일 Axes 재사용) # written by DM 20250904"""
    global _pc_fig, _pc_ax, _pc_scatter
    if pts3d is None or len(pts3d) < 10:
        return

    # 인터랙티브 모드 켜기 및 단일 Figure 재사용
    if _pc_fig is None or _pc_ax is None:
        plt.ion()
        _pc_fig = plt.figure("Tool Point Cloud", figsize=(8, 6))
        _pc_ax = _pc_fig.add_subplot(111, projection='3d')
        plt.show(block=False)

    ax = _pc_ax
    ax.clear()

    # 색상 설정 (Z값에 따라)
    z_min, z_max = pts3d[:, 2].min(), pts3d[:, 2].max()
    colors = (pts3d[:, 2] - z_min) / (z_max - z_min + 1e-9)

    # 포인트 클라우드 그리기 (단일 scatter 갱신)
    _pc_scatter = ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],
                             c=colors, cmap='viridis', s=1, alpha=0.7)

    # 중심점 표시
    if center3d is not None:
        ax.scatter(center3d[0], center3d[1], center3d[2], c='red', s=50, marker='o', label='Center')

    # PCA 축 표시
    if axes3 is not None and lengths is not None and center3d is not None:
        colors_axis = ['red', 'green', 'blue']
        labels_axis = ['Major', 'Minor', 'Third']
        for i, (axis, length) in enumerate(zip(axes3.T, lengths)):
            start = center3d
            end = start + axis * length * 0.3
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    color=colors_axis[i], linewidth=2, label=f'{labels_axis[i]} Axis')

    # 축 설정 및 제목
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend(loc='upper right')

    # 비율 맞추기
    max_range = np.array([pts3d[:, 0].max() - pts3d[:, 0].min(),
                         pts3d[:, 1].max() - pts3d[:, 1].min(),
                         pts3d[:, 2].max() - pts3d[:, 2].min()]).max() / 2.0
    mid_x = (pts3d[:, 0].max() + pts3d[:, 0].min()) * 0.5
    mid_y = (pts3d[:, 1].max() + pts3d[:, 1].min()) * 0.5
    mid_z = (pts3d[:, 2].max() + pts3d[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 렌더링 업데이트
    _pc_fig.canvas.draw_idle()
    _pc_fig.canvas.flush_events()
    plt.pause(0.001)

# 텍스트 출력 버전 (백업) # written by DM 20250904
def print_pointcloud_info(pts3d, center3d=None, axes3=None, lengths=None, title="Point Cloud"):
    """포인트 클라우드 정보를 텍스트로 출력"""
    if pts3d is None or len(pts3d) < 10:
        return
    
    print(f"\n=== {title} ===")
    print(f"포인트 개수: {len(pts3d)}")
    print(f"X 범위: {pts3d[:, 0].min():.3f} ~ {pts3d[:, 0].max():.3f} m")
    print(f"Y 범위: {pts3d[:, 1].min():.3f} ~ {pts3d[:, 1].max():.3f} m")
    print(f"Z 범위: {pts3d[:, 2].min():.3f} ~ {pts3d[:, 2].max():.3f} m")
    
    if center3d is not None:
        print(f"중심점: ({center3d[0]:.3f}, {center3d[1]:.3f}, {center3d[2]:.3f}) m")
    
    if axes3 is not None and lengths is not None:
        print("PCA 축:")
        axis_names = ['주축', '부축', '세번째축']
        for i, (axis, length) in enumerate(zip(axes3.T, lengths)):
            print(f"  {axis_names[i]}: 방향({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}), 길이 {length:.3f}m")
    print("=" * 30)

# 포인트 클라우드 outlier 제거 함수들 # written by DM 20250904
def radius_outlier_removal(points, radius=0.02, min_neighbors=8):
    """반경 기반 outlier 제거 (ROR)"""
    if len(points) < min_neighbors:
        return points
    
    # 간단한 거리 기반 필터링
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    
    valid_indices = []
    for i, point in enumerate(points):
        # 반경 내 이웃 개수 계산
        neighbors = tree.query_ball_point(point, radius)
        if len(neighbors) >= min_neighbors:
            valid_indices.append(i)
    
    return points[valid_indices] if valid_indices else points

def statistical_outlier_removal(points, k=20, std_ratio=1.5):
    """통계적 outlier 제거 (SOR)"""
    if len(points) < k:
        return points
    
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    
    distances = []
    for point in points:
        # k개 최근접 이웃까지의 거리
        dists, _ = tree.query(point, k=k+1)  # +1 because point itself is included
        avg_dist = np.mean(dists[1:])  # 자기 자신 제외
        distances.append(avg_dist)
    
    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # 임계값 계산
    threshold = mean_dist + std_ratio * std_dist
    
    # 유효한 점들만 선택
    valid_mask = distances < threshold
    return points[valid_mask]

def remove_outliers(points):
    """포인트 클라우드 outlier 제거 통합 함수"""
    if not ENABLE_OUTLIER_REMOVAL or points is None or len(points) < 10:
        print(f"[DEBUG] Outlier removal skipped: enabled={ENABLE_OUTLIER_REMOVAL}, points={len(points) if points is not None else 0}")  # written by DM 20250904
        return points
    
    original_count = len(points)
    print(f"[DEBUG] Starting outlier removal: {original_count} points")  # written by DM 20250904
    
    # 1단계: 반경 기반 outlier 제거 (ROR)
    if OUTLIER_RADIUS > 0:
        points_before_ror = len(points)
        points = radius_outlier_removal(points, OUTLIER_RADIUS, MIN_NEIGHBORS)
        print(f"[DEBUG] ROR: {points_before_ror} → {len(points)} points (radius={OUTLIER_RADIUS}, min_neighbors={MIN_NEIGHBORS})")  # written by DM 20250904
    
    # 2단계: 통계적 outlier 제거 (SOR)
    if STATISTICAL_OUTLIER and len(points) > SOR_K_NEIGHBORS:
        points_before_sor = len(points)
        points = statistical_outlier_removal(points, SOR_K_NEIGHBORS, SOR_STD_RATIO)
        print(f"[DEBUG] SOR: {points_before_sor} → {len(points)} points (k={SOR_K_NEIGHBORS}, std_ratio={SOR_STD_RATIO})")  # written by DM 20250904
    
    # 최소 점 수 보장
    if len(points) < 30:
        print(f"[WARN] Outlier removal reduced points to {len(points)}, keeping original")
        return points
    
    removed_count = original_count - len(points)
    if removed_count > 0:
        print(f"[INFO] Removed {removed_count} outliers ({original_count} → {len(points)})")
    
    return points

# 마스크 품질 개선 함수들 # written by DM 20250904
def calculate_mask_iou(mask1, mask2):
    """두 마스크 간 IoU 계산"""
    if mask1.shape != mask2.shape:
        return 0.0
    
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def refine_mask(mask, kernel_size=3):
    """마스크 정제: 구멍 채우기, 노이즈 제거"""
    if not ENABLE_MASK_REFINEMENT:
        return mask
    
    # 이진화
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # 형태학적 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 1. Opening: 작은 노이즈 제거
    mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    
    # 2. Closing: 작은 구멍 채우기
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    
    # 3. 최소 면적 필터링
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 가장 큰 컨투어만 유지
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) >= MASK_MIN_AREA:
            mask_final = np.zeros_like(mask_clean)
            cv2.fillPoly(mask_final, [largest_contour], 1)
            return mask_final
    
    return mask_clean

def validate_mask_quality(mask, confidence=None):
    """마스크 품질 검증"""
    if mask is None:
        return False
    
    # 면적 체크
    mask_area = np.sum(mask > 0)
    if mask_area < MASK_MIN_AREA:
        return False
    
    # 신뢰도 체크
    if confidence is not None and confidence < MASK_MIN_CONFIDENCE:
        return False
    
    # 형태 체크 (너무 얇거나 이상한 형태)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # 원형도 체크 (너무 복잡한 형태 제외)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.1:  # 너무 복잡한 형태
            return False
    
    return True

class MaskVotingSystem:
    """멀티프레임 마스크 투표 시스템"""
    def __init__(self, max_frames=3):
        self.max_frames = max_frames
        self.mask_history = []
        self.confidence_history = []
    
    def add_mask(self, mask, confidence):
        """새 마스크 추가"""
        if not ENABLE_MULTI_FRAME_VOTING:
            return mask
        
        # 히스토리 업데이트
        self.mask_history.append(mask.copy() if mask is not None else None)
        self.confidence_history.append(confidence)
        
        # 최대 프레임 수 유지
        if len(self.mask_history) > self.max_frames:
            self.mask_history.pop(0)
            self.confidence_history.pop(0)
        
        # 투표 수행
        return self.vote_masks()
    
    def vote_masks(self):
        """마스크 투표 수행"""
        if len(self.mask_history) < 2:
            return self.mask_history[-1] if self.mask_history else None
        
        valid_masks = []
        valid_confidences = []
        
        for mask, conf in zip(self.mask_history, self.confidence_history):
            if mask is not None and validate_mask_quality(mask, conf):
                valid_masks.append(mask)
                valid_confidences.append(conf)
        
        if len(valid_masks) < 2:
            return valid_masks[0] if valid_masks else None
        
        # 가중 평균 투표 (신뢰도 기반)
        weights = np.array(valid_confidences)
        weights = weights / weights.sum()
        
        # 마스크 크기 통일
        target_shape = valid_masks[0].shape
        aligned_masks = []
        for mask in valid_masks:
            if mask.shape != target_shape:
                mask = cv2.resize(mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            aligned_masks.append(mask)
        
        # 가중 평균
        weighted_mask = np.zeros_like(aligned_masks[0], dtype=np.float32)
        for mask, weight in zip(aligned_masks, weights):
            weighted_mask += mask.astype(np.float32) * weight
        
        # 임계값 적용
        final_mask = (weighted_mask > 0.5).astype(np.uint8)
        
        return final_mask

# =======================
# 메인
# =======================
def main():
    # YOLO
    model = YOLO(WEIGHTS)
    names = model.names

    # 대상 클래스 필터
    target_lower = {t.lower() for t in TARGET_LABELS}
    ALLOWED_CLASS_IDS = [i for i, n in names.items() if n.lower() in target_lower]
    if not ALLOWED_CLASS_IDS:
        print("[WARN] TARGET_LABELS가 모델 클래스에 없습니다. 전체 클래스로 추론합니다.")

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    x_map, y_map = precompute_xy_maps(intr, COLOR_H, COLOR_W)

    # depth post-processing
    spat_filter = rs.spatial_filter()
    temp_filter = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter(1)

    # === 베이스←카메라 H_BC (예시값: 실제 캘리브레이션으로 교체) ===
    RX_DEG, RY_DEG, RZ_DEG = -180.0, 0.0, -80.0
    TX, TY, TZ = 0.20, 0.0, 0.50
    ORDER = "XYZ"

    H_BC = H_from_axis_angles(
        TX, TY, TZ,
        math.radians(RX_DEG), math.radians(RY_DEG), math.radians(RZ_DEG),
        order=ORDER
    )
    R_BC = H_BC[:3, :3]

    # === 실시간 PCA 안정화 초기화 ===
    # --- written by dongmin. 2025.08.28
    pca_smoother = RealtimePCASmoother(stability_thresh=1.5, power_iter_trigger=0.8)

    fps_time = time.time()
    frame_count = 0
    fps = None
    pc_frame_count = 0  # 포인트 클라우드 업데이트 카운터 # written by DM 20250904
    
    # 마스크 투표 시스템 초기화 # written by DM 20250904
    mask_voting_systems = {}  # 클래스별 투표 시스템

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

            # optional filters
            depth_frame = spat_filter.process(depth_frame)
            depth_frame = temp_filter.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            color = np.asanyarray(color_frame.get_data())
            overlay = color.copy()

            # 카메라 자세 배너(참고)
            draw_rotation_banner(overlay, ORDER, RX_DEG, RY_DEG, RZ_DEG, org=(12, 28), pad=8)

            # YOLO
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
                if getattr(r, "boxes", None) is not None and r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    clses = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                else:
                    boxes = np.zeros((0, 4))
                    clses = np.zeros((0,), dtype=int)
                    confs = np.zeros((0,))

                masks_np = r.masks.data.cpu().numpy() if getattr(r, "masks", None) is not None and r.masks is not None else None

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
                    y_text = None

                    if DRAW_3D_OBB:
                        if masks_np is not None and i < masks_np.shape[0]:
                            raw_mask = masks_np[i]
                            pts3d = mask_to_points3d(
                                depth_frame, raw_mask, depth_scale, intr,
                                x_map, y_map, sample_stride=SAMPLE_STRIDE,
                                z_min=Z_MIN, z_max=Z_MAX, erosion_pixels=3  # written by DM 20250904
                            )
                        else:
                            mask_bbox = np.zeros((COLOR_H, COLOR_W), dtype=np.uint8)
                            x1c = max(0, x1i); y1c = max(0, y1i)
                            x2c = min(COLOR_W - 1, x2i); y2c = min(COLOR_H - 1, y2i)
                            mask_bbox[y1c:y2c + 1, x1c:x2c + 1] = 1
                            pts3d = mask_to_points3d(
                                depth_frame, mask_bbox, depth_scale, intr,
                                x_map, y_map, sample_stride=SAMPLE_STRIDE,
                                z_min=Z_MIN, z_max=Z_MAX, erosion_pixels=3  # written by DM 20250904
                            )

                        if pts3d is not None:
                            # 포인트 클라우드 outlier 제거 # written by DM 20250904
                            pts3d_clean = remove_outliers(pts3d)
                            
                            # --- written by dongmin. 2025.08.28
                            res = pca_obb_3d(pts3d_clean, smoother=pca_smoother)
                            if res is not None:
                                center3d, axes3, lens3, corners3d = res
                                # 3D 그리기 (카메라 프레임 투영)
                                draw_obb3d_on_image(overlay, intr, corners3d, color=(200, 50, 200), thickness=2)
                                draw_axes3d(overlay, intr, center3d, axes3, lens3, scale=0.8)
                                
                                # 포인트 클라우드 시각화 (주기적 업데이트) # written by DM 20250904
                                if SHOW_POINTCLOUD and pc_frame_count % POINTCLOUD_UPDATE_INTERVAL == 0:
                                    if VISUALIZATION_MODE == "matplotlib":
                                        visualize_pointcloud(pts3d_clean, center3d, axes3, lens3, f"Tool Point Cloud - {cls_name}")
                                    elif VISUALIZATION_MODE == "text":
                                        print_pointcloud_info(pts3d_clean, center3d, axes3, lens3, f"Tool Point Cloud - {cls_name}")

                                # 카메라 기준 라벨
                                uv_c, ok = project_points_intr(intr, center3d.reshape(1, 3))
                                if ok[0]:
                                    cxp, cyp = int(round(uv_c[0, 0])), int(round(uv_c[0, 1]))
                                    dist_m = float(center3d[2])
                                    label3 = f"{cls_name} {conf:.2f}  Z:{dist_m:.2f}"
                                    (tw, th), _ = cv2.getTextSize(label3, FONT, 0.55, 2)
                                    y_text = max(cyp, th + 8)
                                    cv2.rectangle(overlay, (cxp, y_text - th - 6), (cxp + tw + 6, y_text), (200, 50, 200), -1)
                                    cv2.putText(overlay, label3, (cxp + 3, y_text - 4), FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

                                    # ====== 베이스 Z축 기준 yaw(°)만 표시 ======
                                    # 1) PCA 축을 베이스 프레임으로 변환
                                    R_obj_cam  = ensure_right_handed(axes3)   # [u1 u2 u3], u1=장축(X), u2=단축(Y)
                                    R_obj_base = R_BC @ R_obj_cam

                                    # 2) 장축(객체 X축)을 XY 평면에 투영해 yaw 계산 (Base X축 기준, 반시계 +)
                                    objX = R_obj_base[:, 0]
                                    vx, vy = float(objX[0]), float(objX[1])
                                    yaw_deg = math.degrees(math.atan2(vy, vx))  # -180~+180°

                                    # 3) UI 출력 (단위 명확히 °)
                                    yaw_label = f"Yaw(Z): {yaw_deg:+.1f}"
                                    (twy, thy), _ = cv2.getTextSize(yaw_label, FONT, 0.7, 2)
                                    y_text_yaw = y_text + thy + 12
                                    cv2.rectangle(overlay, (cxp, y_text_yaw - thy - 6), (cxp + twy + 6, y_text_yaw), (70, 180, 255), -1)
                                    cv2.putText(overlay, yaw_label, (cxp + 3, y_text_yaw - 4), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                                three_d_done = True

                    # ── 2D 폴백 ──
                    if masks_np is not None and i < masks_np.shape[0]:
                        mask = masks_np[i]
                        
                        # 마스크 품질 검증 및 정제 # written by DM 20250904
                        if not validate_mask_quality(mask, conf):
                            print(f"[SKIP] Poor quality mask for {cls_name} (conf: {conf:.3f})")
                            continue
                        
                        # 마스크 정제
                        mask_refined = refine_mask(mask, MORPH_KERNEL_SIZE)
                        
                        # 멀티프레임 투표
                        if cls_name not in mask_voting_systems:
                            mask_voting_systems[cls_name] = MaskVotingSystem(VOTING_FRAMES)
                        mask_final = mask_voting_systems[cls_name].add_mask(mask_refined, conf)
                        
                        if mask_final is None:
                            continue
                        
                        overlay = apply_mask_overlay(overlay, mask_final, alpha=DRAW_MASK_ALPHA, color=(0, 255, 255))
                        if not three_d_done:
                            dist_m = median_depth_meters_from_mask(depth_frame, mask_final, depth_scale)
                            corners, (w_len, h_len), angle_deg, center = pca_obb_from_mask(
                                mask_final, (overlay.shape[0], overlay.shape[1])
                            )
                            if corners is not None:
                                draw_obb(overlay, corners, color=(0, 180, 255), thickness=2)
                                label = f"{cls_name} {conf:.2f}"
                                if dist_m is not None:
                                    label += f"{dist_m:.2f}"
                                (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
                                y_text = max(int(center[1]), th + 8)
                                x_text = int(center[0])
                                cv2.rectangle(overlay, (x_text, y_text - th - 6), (x_text + tw + 6, y_text), (0, 180, 255), -1)
                                cv2.putText(overlay, label, (x_text + 3, y_text - 4), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
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
                                label += f" {dist_m:.2f}"
                            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
                            y_text = max(y1i, th + 8)
                            cv2.rectangle(overlay, (x1i, y_text - th - 6), (x1i + tw + 6, y_text), (0, 255, 0), -1)
                            cv2.putText(overlay, label, (x1i + 3, y_text - 4), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # FPS
            frame_count += 1
            pc_frame_count += 1  # 포인트 클라우드 카운터 증가 # written by DM 20250904
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - fps_time)
                fps_time = now
                frame_count = 0
            if fps is not None:
                txt = f"FPS: {fps:.1f}"
                cv2.putText(overlay, txt, (12, 28 + 48), FONT, 0.8, (50, 50, 255), 2, cv2.LINE_AA)

            # 화면/저장
            if show_window:
                try:
                    cv2.imshow("RealSense YOLO (Yaw-only UI)", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except Exception as e:
                    print(f"[WARN] imshow 실패, 헤드리스로 전환: {e}")
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
    print("ROS2 humble is activated!")
    main()
