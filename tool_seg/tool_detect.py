# -*- coding: utf-8 -*-
"""
ROS2 Vision Node (통합판)
- tool_detect 의 ROS2 통신 구조 유지: target_tool_selection 구독 → tool_detections 퍼블리시
[토픽]
- 구독:  target_tool_selection (std_msgs/String)
- 발행:  tool_detections (std_msgs/Float32MultiArray)  # [x, y, z, roll_deg]

[설명]
- x, y, z는 카메라 좌표계(m) (Realsense intrinsics 사용)
- roll_deg는 화면기준 b=(0,-1) 대비 x축(손잡이 -> 날 방향) 회전각(CW+), 프레임 간 지수평활 적용
좌표계: RealSense **color optical frame** 기준. x=+우측, y=+아래, z=+전방(카메라가 보는 방향). 모든 길이 단위는 미터(m).
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import re
import cv2
import time
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray

WEIGHTS = "../train_results/exp1/weights/best.pt"  # seg/det 둘 다 허용
DEVICE = "0"  # CPU 사용으로 변경 # written by DM 20250904
CONF_TH = 0.28
IOU_TH  = 0.45
IMG_SIZE = 640  # CPU 성능 고려해 해상도 축소 # written by DM 20250904
USE_RGB_INPUT = False

# 표시/후처리
DRAW_MASK_ALPHA = 0.4
COLOR_W, COLOR_H, COLOR_FPS = 640, 480, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 480, 30
TARGET_LABELS = {"nipper", "vernier_calipers", "wire_cutter", "wire_stripper"}
SHOW_WINDOW = True

# --- 밴드(HEAD/TAIL) 전처리 프리뷰 오버레이 ---
SHOW_BAND_PREVIEW = False  # ★ True면 후보군 시각화; 성능 위해 기본 비활성화 # written by DM 20250904
# [OPT] 디버그 시각화로 프레임 비용 증가(밴드 마스크 생성/오버레이/텍스트/라인). 성능 우선 시 False 권장.

# 깊이 폴백(중앙 패치) 커널
DEPTH_KERNEL = 7  # 중앙 패치 median용 커널 확장(깊이 튐 완화) # written by DM 20250904

# 각도 평활화
SMOOTHING_ALPHA = 0.1  # 롤 각도 프레임간 스무딩 강화(흔들림 감소) # written by DM 20250904

# 손잡이/날 시각화
DRAW_HANDLE_TIP = True
DRAW_ENDPOINTS  = True

# 오인(손잡이/날) 방지 가중치
WIDTH_W   = 0.55
RADIUS_W  = 0.35
POINTY_W  = 0.10
WIDTH_MARGIN_RATIO = 0.15

FONT = cv2.FONT_HERSHEY_SIMPLEX

def norm_label(s: str) -> str:
    """라벨 비교를 위한 정규화(공백/하이픈/언더스코어 제거, 소문자)"""
    return re.sub(r"[\s\-_]+", "", s.lower())
# [OPT] 동일 문자열에 대해 반복 호출 시 프레임 범위 캐시(dict)로 미세 최적화 가능.

def apply_mask_overlay(overlay_bgr, mask, alpha=0.4, color=(0, 255, 255)):
    """마스크 색상화 오버레이"""
    H, W = overlay_bgr.shape[:2]
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    if mask.shape[0] != H or mask.shape[1] != W:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    out = overlay_bgr.copy()
    color_img = np.zeros_like(out, dtype=np.uint8)
    color_img[:] = color
    m = mask.astype(bool)
    out[m] = ((1.0 - alpha) * out[m] + alpha * color_img[m]).astype(np.uint8)
    return out

def median_depth_meters_from_center(depth_frame, cx, cy, k=5, depth_scale=0.001):
    """깊이 프레임에서 (cx,cy)의 유효 픽셀 중앙값(m) 반환."""
    depth_image = np.asanyarray(depth_frame.get_data())
    H, W = depth_image.shape[:2]
    x1, x2 = max(0, cx - k // 2), min(W - 1, cx + k // 2)
    y1, y2 = max(0, cy - k // 2), min(H - 1, cy + k // 2)
    patch = depth_image[y1:y2 + 1, x1:x2 + 1]
    if patch.size == 0:
        return None
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale

def median_depth_meters_from_mask(depth_frame, mask, depth_scale=0.001):
    """깊이 프레임에서 마스크 내 유효 픽셀 중앙값(m) 반환."""
    depth_image = np.asanyarray(depth_frame.get_data())
    H_d, W_d = depth_image.shape[:2]
    if mask.shape[0] != H_d or mask.shape[1] != W_d:
        mask = cv2.resize(mask.astype(np.uint8), (W_d, H_d), interpolation=cv2.INTER_NEAREST)
    m = mask.astype(bool)
    if not np.any(m):
        return None
    valid = depth_image[m]
    valid = valid[valid > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale

def _band_metrics_with_dt(mask_bin, band_mask, centered, pts, u_major, v_minor):
    """
    한 끝단 밴드의 지표 계산:
      - width_v : v축 폭
      - area    : 픽셀 수
      - radius_med : distance transform 반경 중앙값(두께)
      - pointiness : (u 길이)/(v 폭) (클수록 뾰족)
    """
    if not np.any(band_mask):
        return dict(valid=False)

    pts_b = pts[band_mask]
    ctr_b = centered[band_mask]

    v_b   = ctr_b @ v_minor
    width_v = float(v_b.max() - v_b.min())
    area    = float(pts_b.shape[0])

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_smooth = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k, iterations=1)
    dt = cv2.distanceTransform(mask_smooth, cv2.DIST_L2, 3).astype(np.float32)

    yy = pts_b[:,1].astype(np.int32).clip(0, mask_bin.shape[0]-1)
    xx = pts_b[:,0].astype(np.int32).clip(0, mask_bin.shape[1]-1)
    radii = dt[yy, xx]
    radius_med = float(np.median(radii)) if radii.size else 0.0

    u_b = ctr_b @ u_major
    u_len = float(u_b.max() - u_b.min()) if u_b.size else 0.0
    pointiness = (u_len + 1e-6) / (width_v + 1e-6)

    return dict(valid=True, width_v=width_v, area=area, radius_med=radius_med, pointiness=pointiness)


def choose_handle_tip(mask_bin, mask_head, mask_tail, centered, pts, u_major, v_minor,
                      width_w=WIDTH_W, radius_w=RADIUS_W, pointy_w=POINTY_W,
                      width_margin_ratio=WIDTH_MARGIN_RATIO,
                      force_width_only=False):
    """
    손잡이/날 결정:
      - force_width_only=True 이면 폭만으로 결정
      - 아니면 (1) 폭 즉결 → (2) (폭+반경+뾰족도) 가중합
    반환: band_handle_mask, band_tip_mask, decision_conf(0~1)
    """
    mh = _band_metrics_with_dt(mask_bin, mask_head, centered, pts, u_major, v_minor)
    mt = _band_metrics_with_dt(mask_bin, mask_tail, centered, pts, u_major, v_minor)

    if not mh["valid"] or not mt["valid"]:
        if mh.get("width_v",0) >= mt.get("width_v",0):
            return mask_head, mask_tail, 0.5
        else:
            return mask_tail, mask_head, 0.5

    wh, wt = mh["width_v"], mt["width_v"]
    big = max(wh, wt) + 1e-6
    width_gap_ratio = abs(wh - wt) / big

    # ★ 폭만으로 결정(버니어 캘리퍼스 전용)
    if force_width_only:
        if wh >= wt:
            return mask_tail, mask_head, min(1.0, 0.6 + 0.4*width_gap_ratio)
        else:
            return mask_head, mask_tail, min(1.0, 0.6 + 0.4*width_gap_ratio)

    # 1) 폭 즉결
    if wh > wt * (1.0 + width_margin_ratio):
        return mask_head, mask_tail, min(1.0, 0.6 + 0.4*width_gap_ratio)
    if wt > wh * (1.0 + width_margin_ratio):
        return mask_tail, mask_head, min(1.0, 0.6 + 0.4*width_gap_ratio)

    # 2) 가중합
    width_score  = (wh - wt) / big  # 양수면 head가 넓음 → head=손잡이
    rh, rt       = mh["radius_med"], mt["radius_med"]
    radius_score = (rh - rt) / (max(rh, rt) + 1e-6)  # 양수면 head 두꺼움 → head=손잡이
    ph, pt       = mh["pointiness"], mt["pointiness"]
    pointy_score = (pt - ph)  # 양수면 tail이 더 뾰족 → tail=날

    combined = width_w*width_score + radius_w*radius_score + pointy_w*pointy_score

    if combined >= 0:
        conf = min(1.0, 0.5 + 0.5*abs(combined))
        return mask_head, mask_tail, conf
    else:
        conf = min(1.0, 0.5 + 0.5*abs(combined))
        return mask_tail, mask_head, conf

# ---------- 컨벡스 헐 + minAreaRect 기반 OBB ----------
def obb_handle_tip_from_mask(mask, out_size_hw, end_band_ratio=0.18, min_pts=30, force_width_only=False):
    """
    세그 마스크로 OBB 계산 (cv2.minAreaRect + Convex Hull 기반)
    - force_width_only=True: 손잡이/날을 폭만으로 결정(버니어 캘리퍼스용)
    """
    H, W = out_size_hw
    if mask.dtype != np.uint8:
        mask_bin = (mask > 0.5).astype(np.uint8)
    else:
        mask_bin = mask
    if mask_bin.shape[:2] != (H, W):
        mask_bin = cv2.resize(mask_bin, (W, H), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None, None, (None, None, None, None), 0.0, None

    # 가장 큰 컨투어 사용
    main_contour = max(contours, key=cv2.contourArea)
    if main_contour.shape[0] < min_pts:
        return None, None, None, None, None, None, (None, None, None, None), 0.0, None

    # 1. 컨벡스 헐 기반 최소 면적 사각형 찾기
    hull = cv2.convexHull(main_contour)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    
    # 2. 사각형의 장축을 주축으로 설정
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        u_major = edge1
    else:
        u_major = edge2
    u_major = u_major / (np.linalg.norm(u_major) + 1e-9)
    v_minor = np.array([-u_major[1], u_major[0]], dtype=np.float32)

    # 마스크의 모든 점과 중심(mean)
    ys, xs = np.where(mask_bin > 0)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0)
    centered = pts - mean

    # 3. 주축 기준으로 밴드 생성
    proj_u = centered @ u_major
    u_min, u_max = proj_u.min(), proj_u.max()
    L = u_max - u_min
    if L < 1e-6:
        return None, None, None, None, None, None, (None, None, None, None), 0.0, None

    # 종횡비에 따른 밴드 두께 조정
    proj_v = centered @ v_minor
    ar = L / (proj_v.max() - proj_v.min() + 1e-6)
    band_ratio = end_band_ratio if ar >= 1.25 else max(end_band_ratio, 0.25)
    band = band_ratio * L

    mask_head = proj_u <= (u_min + band)
    mask_tail = proj_u >= (u_max - band)

    # 4. 손잡이/날 결정
    band_handle_mask, band_tip_mask, decide_conf = choose_handle_tip(
        mask_bin, mask_head, mask_tail, centered, pts, u_major, v_minor,
        force_width_only=force_width_only
    )
    
    # 밴드 중심 및 끝점 계산
    def _endpoints_and_center(band_mask):
        if not np.any(band_mask): return None, None, None
        pts_b = pts[band_mask]
        ctr_b = centered[band_mask]
        v_b = ctr_b @ v_minor
        p_min = pts_b[int(np.argmin(v_b))]
        p_max = pts_b[int(np.argmax(v_b))]
        center_mid = (p_min + p_max) * 0.5
        return p_min, p_max, center_mid

    h_end_a, h_end_b, handle_ctr = _endpoints_and_center(band_handle_mask)
    t_end_a, t_end_b, tip_ctr = _endpoints_and_center(band_tip_mask)

    # 디버그 정보
    def _create_band_img(band_mask):
        img = np.zeros((H, W), dtype=np.uint8)
        if np.any(band_mask):
            pts_b = np.round(pts[band_mask]).astype(np.int32)
            pts_b[:,0] = np.clip(pts_b[:,0], 0, W-1)
            pts_b[:,1] = np.clip(pts_b[:,1], 0, H-1)
            img[pts_b[:,1], pts_b[:,0]] = 1
        return img
    
    debug_head_ctr = pts[mask_head].mean(axis=0) if np.any(mask_head) else None
    debug_tail_ctr = pts[mask_tail].mean(axis=0) if np.any(mask_tail) else None
    
    debug = dict(
        band_head_img=_create_band_img(mask_head),
        band_tail_img=_create_band_img(mask_tail),
        u_major=u_major.copy(), v_minor=v_minor.copy(), mean=mean.copy(),
        t_min=u_min, t_max=u_max,
        head_ctr=debug_head_ctr, tail_ctr=debug_tail_ctr,
        chosen_axis="major_from_hull", decide_conf=decide_conf,
        sep=np.linalg.norm(tip_ctr - handle_ctr) if handle_ctr is not None and tip_ctr is not None else 0.0,
        ar=ar
    )

    if handle_ctr is None or tip_ctr is None:
        return None, None, None, None, None, None, (None, None, None, None), decide_conf, debug

    # 5. 최종 방향 및 OBB 계산
    xdir = (tip_ctr - handle_ctr).astype(np.float32)
    n = np.linalg.norm(xdir)
    if n < 1e-6:
        xdir = u_major.copy()
    else:
        xdir /= n
    ydir = np.array([-xdir[1], xdir[0]], dtype=np.float32)

    R = np.stack([xdir, ydir], axis=1)
    proj_xy = (pts - mean) @ R
    mins = proj_xy.min(axis=0); maxs = proj_xy.max(axis=0)
    c_local = (mins + maxs) * 0.5
    center = mean + (R @ c_local)
    a = (maxs[0] - mins[0]) * 0.5
    b = (maxs[1] - mins[1]) * 0.5
    corners = np.stack([
        center + a*xdir + b*ydir,
        center + a*xdir - b*ydir,
        center - a*xdir - b*ydir,
        center - a*xdir + b*ydir
    ], axis=0).astype(np.int32)

    # 각도 계산
    b_up = np.array([0.0, -1.0], dtype=np.float32)
    dot = b_up @ xdir
    det = b_up[0]*xdir[1] - b_up[1]*xdir[0]
    angle_rad_ccw = math.atan2(det, dot)
    angle_deg = -math.degrees(angle_rad_ccw)
    if angle_deg > 180: angle_deg -= 360
    elif angle_deg < -180: angle_deg += 360

    len_x = float(maxs[0] - mins[0])
    len_y = float(maxs[1] - mins[1])

    return corners, (len_x, len_y), angle_deg, center, handle_ctr, tip_ctr, \
           (h_end_a, h_end_b, t_end_a, t_end_b), decide_conf, debug


def draw_obb(overlay, corners, color=(0, 180, 255), thickness=2):
    cv2.polylines(overlay, [corners], isClosed=True, color=color, thickness=thickness)


def draw_handle_tip_viz(overlay, handle_ctr, tip_ctr, end_points_tuple,
                        draw_endpoints=True,
                        color_center_handle=(0,200,255),
                        color_center_tip=(0,50,255),
                        color_arrow=(0,255,0),
                        color_end_handle=(255,200,0),
                        color_end_tip=(255,0,100)):
    """중앙점 2개 + 끝점 4개(손잡이 2, 날 2) 시각화"""
    if handle_ctr is None or tip_ctr is None:
        return overlay
    hc = tuple(np.int32(handle_ctr)); tc = tuple(np.int32(tip_ctr))
    cv2.circle(overlay, hc, 5, color_center_handle, -1)
    cv2.circle(overlay, tc, 5, color_center_tip, -1)
    cv2.arrowedLine(overlay, hc, tc, color_arrow, 2, tipLength=0.25)
    if draw_endpoints and end_points_tuple is not None:
        h_end_a, h_end_b, t_end_a, t_end_b = end_points_tuple
        for p in [h_end_a, h_end_b]:
            if p is not None:
                cv2.circle(overlay, tuple(np.int32(p)), 6, color_end_handle, -1)
        for p in [t_end_a, t_end_b]:
            if p is not None:
                cv2.circle(overlay, tuple(np.int32(p)), 6, color_end_tip, -1)
    return overlay


# ROS2 노드
class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.detection_pub = self.create_publisher(Float32MultiArray, '/info/array/target_obj_array', 10)
        self.target_tool_sub = self.create_subscription(String, '/info/string/obj_name', self.target_tool_callback, 10)

        self.target_tool = None        
        self.detection_active = False  

        self.frame_id = 'camera_color_optical_frame'  # 좌표계 프레임 이름
        self.timer    = self.create_timer(0.1, self.vision_callback)  
        self.setup_vision_system()
        self.get_logger().info('Vision Node 시작됨. 제어단에서 도구 요청을 대기 중...')

        self.smoothed_angle = None

    def setup_vision_system(self):
        # YOLO
        self.model = YOLO(WEIGHTS)
        self.names = self.model.names
        self.get_logger().info(f"모델 클래스({len(self.names)}): {self.names}")

        # 라벨 정규화 매칭(소문자 + 공백/하이픈/언더스코어 제거)
        target_norm = {norm_label(t) for t in TARGET_LABELS}
        names_norm = {i: norm_label(n) for i, n in self.names.items()}
        self.allowed_ids = [i for i, n in names_norm.items() if n in target_norm]
        self.get_logger().info(f"매칭된 클래스 ID: {self.allowed_ids}")

        # RealSense 파이프라인
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
        config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
        profile = self.pipeline.start(config)

        # 깊이 관련
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.spat_filter = rs.spatial_filter()
        self.temp_filter = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter(1)

        # Intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.get_logger().info("RealSense 카메라 초기화 완료")

    # 카메라 좌표계로 좌표 변환 
    def pixel_to_3d_point(self, pixel_x, pixel_y, depth_m):
        if depth_m is None or depth_m <= 0:
            return None
        x = (pixel_x - self.intrinsics.ppx) * depth_m / self.intrinsics.fx
        y = (pixel_y - self.intrinsics.ppy) * depth_m / self.intrinsics.fy
        z = depth_m
        return [float(x), float(y), float(z)]

    # 콜백 루프 
    def vision_callback(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)

            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not color_frame:
                return

            if depth_frame:
                depth_frame = self.spat_filter.process(depth_frame)
                depth_frame = self.temp_filter.process(depth_frame)
                depth_frame = self.hole_filling.process(depth_frame)
            # [OPT] 후단에 SOR/ROR 등 점군 outlier 제거가 추가되면 필터 강도 축소/일부 생략으로 중복 비용 절감 가능.
            else:
                depth_frame = None

            color = np.asanyarray(color_frame.get_data())
            overlay = color.copy()
            # [OPT] 시각화가 불필요하면 overlay 복사 생략 가능. YOLO 입력만 필요 시 inp만 생성.
            inp = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) if USE_RGB_INPUT else color

            results = self.model(
                inp,
                conf=CONF_TH,
                iou=IOU_TH,
                device=DEVICE,
                classes=self.allowed_ids if self.allowed_ids else None,
                imgsz=IMG_SIZE,
                verbose=False,
            ) if self.detection_active else []

            found_payload = None

            if results and len(results) > 0:
                r = results[0]

                # boxes
                if getattr(r, "boxes", None) is not None and r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    clses = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                else:
                    boxes, clses, confs = np.zeros((0,4)), np.zeros((0,), dtype=int), np.zeros((0,))

                # masks
                if getattr(r, "masks", None) is not None and r.masks is not None:
                    masks_np = r.masks.data.cpu().numpy()
                else:
                    masks_np = None

                # 각 후보 순회 → target_tool 일치 항목 우선
                N = len(boxes)
                for i in range(N):
                    c = int(clses[i])
                    x1, y1, x2, y2 = boxes[i]
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

                    if isinstance(self.names, dict):
                        cls_name = self.names.get(c, str(c))
                    else:
                        try: cls_name = self.names[c]
                        except Exception: cls_name = str(c)

                    if self.target_tool and norm_label(cls_name) != norm_label(self.target_tool):
                        continue
                    conf = float(confs[i])
                    depth_m = None
                    roll_deg = 0.0
                    decide_conf = 0.0
                    handle_ctr = None
                    tip_ctr = None

                    if masks_np is not None and i < masks_np.shape[0]:
                        mask = masks_np[i]
                        # 원 마스크
                        overlay = apply_mask_overlay(overlay, mask, alpha=DRAW_MASK_ALPHA, color=(0,255,255))

                        # 마스크 erode 후 median(튀는 깊이 억제) # written by DM 20250904
                        if depth_frame is not None:
                            dimg = np.asanyarray(depth_frame.get_data())
                            H_d, W_d = dimg.shape[:2]
                            mask_d = mask
                            if mask_d.shape[:2] != (H_d, W_d):
                                mask_d = cv2.resize(mask_d.astype(np.uint8), (W_d, H_d), interpolation=cv2.INTER_NEAREST)
                            if mask_d.dtype != np.uint8:
                                mask_d = (mask_d > 0.5).astype(np.uint8)
                            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                            mask_erode = cv2.erode(mask_d, k, iterations=1)
                            depth_m = median_depth_meters_from_mask(depth_frame, mask_erode, self.depth_scale)
                        else:
                            depth_m = None

                        # ★ 버니어 캘리퍼스면 폭만으로 결정
                        is_vc = norm_label(cls_name) == norm_label("vernier_calipers")

                        (corners, (w_len, h_len), angle_deg, center_px,
                         handle_ctr, tip_ctr, end_points_tuple, decide_conf, debug) = obb_handle_tip_from_mask(
                            mask, (overlay.shape[0], overlay.shape[1]),
                            force_width_only=is_vc
                        )

                        # ★ 후보군 프리뷰
                        if SHOW_BAND_PREVIEW and debug is not None:
                            # [OPT] 실서비스/성능 우선 시 비활성화 권장(연산량 큼).
                            head_img = debug["band_head_img"]
                            tail_img = debug["band_tail_img"]
                            overlay = apply_mask_overlay(overlay, head_img, alpha=0.35, color=(255, 0, 255))   # 보라(HEAD 후보)
                            overlay = apply_mask_overlay(overlay, tail_img, alpha=0.35, color=(0, 128, 255))  # 주황/청파(TAIL 후보)

                            # 라벨 텍스트
                            if debug.get("head_ctr") is not None:
                                hx, hy = int(debug["head_ctr"][0]), int(debug["head_ctr"][1])
                                cv2.putText(overlay, "HEAD_CAND", (hx-20, max(hy-8, 15)), FONT, 0.5, (255,0,255), 2, cv2.LINE_AA)
                            if debug.get("tail_ctr") is not None:
                                tx, ty = int(debug["tail_ctr"][0]), int(debug["tail_ctr"][1])
                                cv2.putText(overlay, "TAIL_CAND", (tx-20, max(ty-8, 15)), FONT, 0.5, (0,128,255), 2, cv2.LINE_AA)

                            # 주축 시각화
                            u_major = debug["u_major"]; mean = debug["mean"]
                            tmin = debug["t_min"]; tmax = debug["t_max"]
                            axis_len = 0.5 * (tmax - tmin + 1e-6)
                            p1 = (mean + u_major * (-axis_len)).astype(np.int32)
                            p2 = (mean + u_major * (+axis_len)).astype(np.int32)
                            cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255,255,0), 2)
                            cv2.circle(overlay, (int(mean[0]), int(mean[1])), 3, (255,255,0), -1)

                        if corners is not None:
                            # 각도 평활화
                            if self.smoothed_angle is None:
                                self.smoothed_angle = angle_deg
                            else:
                                diff = angle_deg - self.smoothed_angle
                                if diff > 180: diff -= 360
                                elif diff < -180: diff += 360
                                self.smoothed_angle += SMOOTHING_ALPHA * diff
                                if self.smoothed_angle > 180: self.smoothed_angle -= 360
                                elif self.smoothed_angle < -180: self.smoothed_angle += 360
                            roll_deg = float(self.smoothed_angle)

                            # 최종 OBB/핸들-팁 시각화
                            draw_obb(overlay, corners, color=(0,180,255), thickness=2)
                            if DRAW_HANDLE_TIP:
                                overlay = draw_handle_tip_viz(overlay, handle_ctr, tip_ctr, end_points_tuple, draw_endpoints=DRAW_ENDPOINTS)
                        else:
                            # 마스크 실패 → bbox 폴백
                            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
                            cx, cy = int((x1i+x2i)/2), int((y1i+y2i)/2)
                            depth_m = median_depth_meters_from_center(depth_frame, cx, cy, k=DEPTH_KERNEL, depth_scale=self.depth_scale) if depth_frame is not None else None

                    else:
                        # det-only → bbox 폴백
                        cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
                        cx, cy = int((x1i+x2i)/2), int((y1i+y2i)/2)
                        depth_m = median_depth_meters_from_center(depth_frame, cx, cy, k=DEPTH_KERNEL, depth_scale=self.depth_scale) if depth_frame is not None else None

                    # 3D 중심
                    if masks_np is not None and i < (masks_np.shape[0] if masks_np is not None else 0) and handle_ctr is not None and tip_ctr is not None:
                        center_px = ( (handle_ctr[0] + tip_ctr[0]) * 0.5, (handle_ctr[1] + tip_ctr[1]) * 0.5 )
                    else:
                        center_px = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

                    center_3d = None
                    if depth_m is not None:
                        center_3d = self.pixel_to_3d_point(center_px[0], center_px[1], depth_m)

                    # HUD
                    label = f"{cls_name} {conf:.2f}"
                    if depth_m is not None:
                        label += f" | Depth: {depth_m:.2f}m"
                    if self.smoothed_angle is not None:
                        label += f" | Roll: {self.smoothed_angle:.1f}°"
                    if decide_conf:
                        label += f" | H/T: {decide_conf:.2f}"
                    (tw, th), baseline = cv2.getTextSize(label, FONT, 0.6, 2)
                    px = int(center_px[0]); py = max(int(center_px[1]), th+6)
                    cv2.rectangle(overlay, (px, py - th - 6), (px + tw + 6, py), (0,180,255), -1)
                    cv2.putText(overlay, label, (px + 3, py - 4), FONT, 0.6, (0,0,0), 2, cv2.LINE_AA)

                    # 발행 페이로드 (첫 번째 일치만)
                    if center_3d is not None:
                        payload = {
                            "class_name": cls_name,
                            "confidence": conf,
                            "position": center_3d,                 
                            "roll_deg": float(roll_deg)
                        }
                        found_payload = payload
                        break  # 첫 대상만 사용

            if found_payload is not None:
                self.publish_detections(found_payload)
                self.detection_active = False
                self.target_tool = None

            if SHOW_WINDOW:
                status_text = f"Detection: {'ACTIVE' if self.detection_active else 'INACTIVE'}"
                target_text = f"Target: {self.target_tool if self.target_tool else 'None'}"
                cv2.putText(overlay, status_text, (10, 28), FONT, 0.8,
                            (0,255,0) if self.detection_active else (0,0,255), 2)
                cv2.putText(overlay, target_text, (10, 56), FONT, 0.8, (255,255,255), 2)
                try:
                    cv2.imshow("Tool Detection (seg+obb)", overlay)
                    cv2.waitKey(1)
                except Exception as e:
                    self.get_logger().warn(f"imshow 실패: {e}")

        except Exception as e:
            self.get_logger().error(f"Vision callback 오류: {e}")

    def publish_detections(self, det: dict):
        msg = Float32MultiArray()
        x, y, z = det["position"]
        roll = float(det.get("roll_deg", 0.0))
        msg.data = [float(x), float(y), float(z), roll]
        self.detection_pub.publish(msg)
        self.get_logger().info(
            f"탐지 전송: {det['class_name']} | pos=({x:.3f},{y:.3f},{z:.3f}) m | roll={roll:.1f}° | conf={det['confidence']:.2f}"
        )

    # --- 구독 콜백 ---
    def target_tool_callback(self, msg: String):
        tool_name = msg.data.strip()
        if tool_name and tool_name.lower() not in {"stop", "none"}:
            self.target_tool = tool_name
            self.detection_active = True
            self.get_logger().info(f"도구 탐지 요청 수신: {tool_name}")
            # 각도 평활화 초기화(새 타깃)
            self.smoothed_angle = None
        else:
            self.target_tool = None
            self.detection_active = False
            self.get_logger().info("도구 탐지 중단")

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("중단됨")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    print("ROS2 humble is activated!")
    main()