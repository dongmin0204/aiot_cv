# tool_pca.py  (RealSense + YOLO(seg/det) + PCA OBB + Depth)
# ----------------------------------------------------------
# 필요 패키지:
#   pip install pyrealsense2 opencv-python ultralytics
# ----------------------------------------------------------

import os
# Wayland에서 Qt 플러그인 문제 예방 (cv2 import 전에 설정 권장)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import time
import math
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# =======================
# 사용자 설정
# =======================
WEIGHTS = "../train_results/exp1/weights/best.pt"  # YOLO 가중치(.pt) 경로 (seg 모델 권장: yolov11n-seg.pt 등)
DEVICE  = "cpu"  # CPU 사용으로 변경 # written by DM 20250904
CONF_TH = 0.5                       # confidence threshold
IOU_TH  = 0.5                       # NMS IoU threshold
LABEL_FILTER = None                 # 화면 표시 필터(추론은 아래 TARGET_LABELS로 제한). 예: ["Screwdriver", "드라이버"]
DRAW_MASK_ALPHA = 0.4               # 세그 마스크 오버레이 투명도
DEPTH_KERNEL = 5                    # bbox 중심 주변 (k x k) median depth (디텍션 모델용)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 특정 클래스 제외(불필요하면 빈 리스트)
EXCLUDE_CLASS_IDS = []

# RealSense 해상도 (컬러/깊이 동일 권장, 또는 align 사용)
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30

# Screwdriver만 추론하도록 제한 — 모델 라벨 표기 후보(대소문자 무시)
TARGET_LABELS = {"nipper", "vernier_calipers", "wire_cutter", "wire_stripper"}

# GUI 옵션: 창 표시가 어려운 환경 대비
SHOW_WINDOW = True                  # ← 전역 기본값(실제 제어는 main()의 지역변수로 함)
FALLBACK_SAVE_EVERY_N = 15          # 헤드리스 모드일 때 N틱(0.1초 단위)마다 저장
FALLBACK_SAVE_DIR = "/tmp/tool_pca_frames"


# =======================
# 유틸 함수
# =======================
def apply_mask_overlay(overlay_bgr, mask, alpha=0.4, color=(0, 255, 255)):
    """
    overlay_bgr: 원본 프레임(BGR, uint8, HxWx3)
    mask: 2D (hxw), float/bool/uint8 가능 (값 0/1 또는 0~1)
    alpha: 투명도
    color: (B,G,R)
    """
    H, W = overlay_bgr.shape[:2]

    # 1) 이진화
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)

    # 2) 프레임 크기에 맞게 리사이즈
    if mask.shape[0] != H or mask.shape[1] != W:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # 3) 마스크 영역만 색 오버레이
    out = overlay_bgr.copy()
    # [OPT] 시각화가 불필요하면 복사를 생략하세요. (ex. 헤드리스 성능 측정 시)
    color_img = np.zeros_like(out, dtype=np.uint8)
    color_img[:] = color

    m = mask.astype(bool)
    out[m] = ((1.0 - alpha) * out[m] + alpha * color_img[m]).astype(np.uint8)
    return out


def median_depth_meters_from_center(depth_frame, cx, cy, k=5, depth_scale=0.001):
    """
    깊이 프레임에서 bbox 중심 주변 kxk 패치의 median depth(m)를 계산.
    """
    depth_image = np.asanyarray(depth_frame.get_data())  # uint16
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
    """
    세그 마스크 내부 픽셀의 median depth(m)를 계산.
    mask: 2D (hxw). 깊이 프레임 크기와 다르면 리사이즈.
    """
    depth_image = np.asanyarray(depth_frame.get_data())  # uint16, shape: (H, W)
    H_d, W_d = depth_image.shape[:2]

    # mask → 깊이 크기로 리사이즈
    if mask.shape[0] != H_d or mask.shape[1] != W_d:
        mask = cv2.resize(mask.astype(np.uint8), (W_d, H_d), interpolation=cv2.INTER_NEAREST)
    # [OPT] 동일 마스크를 여러 함수에서 사용할 경우, 이진화/리사이즈를 한 번만 수행해 재사용하세요.

    m = mask.astype(bool)
    if not np.any(m):
        return None

    valid = depth_image[m]
    valid = valid[valid > 0]
    if valid.size == 0:
        return None

    return float(np.median(valid)) * depth_scale


def pca_obb_from_mask(mask, out_size_hw):
    """
    세그멘테이션 마스크 픽셀 좌표로 2D PCA 수행 → OBB(중심, 반축 길이, 각도, 꼭짓점) 계산
    mask: 2D (h x w), 0/1 혹은 0~1
    out_size_hw: (H, W) 최종 그릴 프레임 크기(overlay)
    return:
        corners_int (4,2) int32 시계방향,
        (w, h) 실수(메이저/마이너 축 길이),
        angle_deg (float, 메이저축 각도, +x 기준 CCW),
        center (2,) float
    또는 (None, None, None, None)
    """
    H, W = out_size_hw

    # 이진화 + 리사이즈
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask > 0)
    if xs.size < 10:  # 픽셀 너무 적으면 스킵
        return None, None, None, None

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)  # (N,2), (x,y)
    mean = pts.mean(axis=0)                           # (2,)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)              # (2,2)

    # 고유분해 (고유값 큰 축이 주축)
    vals, vecs = np.linalg.eigh(cov)                  # vecs[:,i]는 eigenvector
    order = np.argsort(vals)[::-1]
    u = vecs[:, order[0]]  # 주축(major)
    v = vecs[:, order[1]]  # 부축(minor)

    # 정규화(수치 안정)
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)

    # 주성분 좌표계로 투영
    R = np.stack([u, v], axis=1)                      # (2,2) [u v]
    proj = centered @ R                               # (N,2)

    # 각 축에서 min/max → 중심 오프셋과 길이
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)

    # 로컬(주성분) 좌표계에서 박스 중심 (비대칭 분포 보정)
    c_local = (mins + maxs) * 0.5                     # (2,)
    # 전역 좌표계로 변환된 실제 중심
    center = mean + R @ c_local

    # 반축 길이
    a = (maxs[0] - mins[0]) * 0.5                     # 주축 반길이
    b = (maxs[1] - mins[1]) * 0.5                     # 부축 반길이

    # 꼭짓점 (시계방향)
    # corners = center ± a*u ± b*v
    c = center
    c1 = c + (+a)*u + (+b)*v
    c2 = c + (+a)*u + (-b)*v
    c3 = c + (-a)*u + (-b)*v
    c4 = c + (-a)*u + (+b)*v
    corners = np.stack([c1, c2, c3, c4], axis=0)      # (4,2)

    # 각도 (주축 u의 각도)
    angle_deg = math.degrees(math.atan2(u[1], u[0]))

    return corners.astype(np.int32), (2*a, 2*b), angle_deg, center


def draw_obb(overlay, corners, color=(0, 180, 255), thickness=2):
    """corners: (4,2) int32 시계방향"""
    cv2.polylines(overlay, [corners], isClosed=True, color=color, thickness=thickness)


# =======================
# 메인
# =======================
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

    # 선택: depth post-processing
    spat_filter = rs.spatial_filter()
    temp_filter = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter(1)

    fps_time = time.time()
    frame_count = 0
    fps = None

    # 헤드리스 저장 준비
    show_window = SHOW_WINDOW          # ← 전역을 복사해서 지역 상태로 관리
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
            # 필터 강도/조합은 SOR/ROR 도입 시 중복되지 않도록 조절 # written by DM 20250904
            # [OPT] 후단에 SOR/ROR 등 점군 outlier 제거를 추가한다면 필터 강도를 낮추거나 일부 생략해 중복 비용을 줄일 수 있습니다.

            color = np.asanyarray(color_frame.get_data())  # BGR uint8 (H x W x 3)
            overlay = color.copy()
            # [OPT] 시각화가 목적이 아니면 overlay 복사 생략 가능. YOLO 입력만 필요 시 RGB 변환만 수행.

            # YOLO 추론: RGB 입력 권장
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            # [OPT] YOLO 입력만 필요하고 overlay를 쓰지 않으면 color.copy()도 생략 가능.
            results = model(
                rgb,
                conf=CONF_TH,
                iou=IOU_TH,
                device=DEVICE,
                verbose=False,
                classes=ALLOWED_CLASS_IDS if ALLOWED_CLASS_IDS else None,  # ← Screwdriver만 추론
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

                # 인스턴스 단위로 처리 (마스크/박스를 동일 인덱스로 사용)
                N = len(boxes)
                for i in range(N):
                    c = int(clses[i])
                    if c in EXCLUDE_CLASS_IDS:
                        continue

                    x1, y1, x2, y2 = boxes[i]
                    cls_name = names.get(c, str(c))
                    conf = float(confs[i])

                    # (선택) 화면 표시만 라벨 필터 (추론은 위 classes로 이미 제한됨)
                    if LABEL_FILTER and cls_name not in LABEL_FILTER:
                        continue

                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

                    # 세그 모델: 마스크 기반 OBB + 거리(마스크 내부 median)
                    if masks_np is not None and i < masks_np.shape[0]:
                        mask = masks_np[i]
                        # 마스크 오버레이
                        overlay = apply_mask_overlay(overlay, mask, alpha=DRAW_MASK_ALPHA, color=(0, 255, 255))

                        # 깊이(median) 계산
                        dist_m = median_depth_meters_from_mask(depth_frame, mask, depth_scale)

                        # === PCA OBB 계산 & 그리기 ===
                        corners, (w_len, h_len), angle_deg, center = pca_obb_from_mask(
                            mask, (overlay.shape[0], overlay.shape[1])
                        )
                        if corners is not None:
                            draw_obb(overlay, corners, color=(0, 180, 255), thickness=2)

                            # 라벨 텍스트
                            label = f"{cls_name} {conf:.2f}"
                            if dist_m is not None:
                                label += f" | {dist_m:.2f}m"
                            # 각도/크기 표기(선택)
                            label2 = f"θ={angle_deg:.1f}°  {w_len:.1f}x{h_len:.1f}px"

                            # 라벨 배경/텍스트
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
                            # 마스크 픽셀 너무 적으면 fallback: AABB 표시
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

                    else:
                        # 디텍션 모델: AABB + 중심 깊이
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
                    cv2.imshow("RealSense YOLO (Screwdriver-only, Seg OBB + Depth)", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except Exception as e:
                    print(f"[WARN] imshow 실패, 헤드리스로 전환합니다: {e}")
                    show_window = False
                    os.makedirs(FALLBACK_SAVE_DIR, exist_ok=True)
            else:
                # 헤드리스: 주기적으로 프레임 저장 (약 0.1초 tick 기준)
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
