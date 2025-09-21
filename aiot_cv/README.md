# aiot_cv: FoundationPose Model-Free Pipeline

YOLOv8-seg → ROI → RGB-D → 6DoF 포즈 추정까지 수행하는 FoundationPose 기반 end-to-end 파이프라인입니다. CAD 없이 레퍼런스 이미지 기반(few-shot)으로 동작하며, 실시간 추적과 포즈 스무딩을 지원합니다.

## 🎯 주요 기능

- **모델 프리 초기화**: 레퍼런스 이미지 세트(6-12장)로 CAD 없이 포즈 초기화
- **실시간 추적**: RGB-D 기반 연속 포즈 추정 및 추적
- **자동 스무딩**: PCA 기반 시간적 스무딩 + 각속도/가속도 클램프
- **표준화된 검출**: YOLOv8-seg → 표준화된 Detection 객체
- **포인트클라우드 처리**: RGB-D 변환, 노이즈 제거, OBB/PCA 계산

## 📁 프로젝트 구조

```
aiot_cv/
├── configs/
│   ├── camera.yaml              # 카메라 내참 (확장 스키마)
│   └── tools/
│       └── screwdriver.json     # 도구 설정 예시
├── pipelines/
│   └── fp_model_free.py         # 메인 파이프라인 스크립트
├── src/
│   ├── detect/
│   │   └── yolo.py              # YOLOv8-seg 래퍼 (표준화된 출력)
│   ├── io/
│   │   └── load_k.py            # 카메라 내참 로딩 (확장/기본 스키마 지원)
│   ├── pc/
│   │   └── pointcloud.py        # RGB-D→포인트클라우드, 필터링, OBB/PCA
│   ├── pose/
│   │   └── foundationpose.py    # FoundationPose 래퍼 (모델 프리)
│   └── track/
│       └── smoother.py          # 실시간 PCA 스무더
└── README.md
```

## ⚙️ 설정

### 1) 카메라 내참 (`configs/camera.yaml`)

확장 스키마를 사용하여 해상도, 정렬, 왜곡, 외부참조계를 모두 포함:

```yaml
version: 1
device_serial: "d435i-sample"
profile:
  color: {width: 640, height: 480, fps: 30}
  depth: {width: 640, height: 480, fps: 30}
align_to_color: true

intrinsics:
  color:
    fx: 615.0; fy: 615.0; cx: 320.0; cy: 240.0
    dist_model: "brown"
    dist: [0.0, 0.0, 0.0, 0.0, 0.0]
  depth:
    fx: 384.1; fy: 384.1; cx: 319.5; cy: 239.5
    dist_model: "inverse_brown"
    dist: [0.0, 0.0, 0.0, 0.0, 0.0]

extrinsics:
  T_color_from_depth:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]

depth_units: 0.001
notes: "D435i @ 640x480; sample schema"
```

### 2) 도구 설정 (`configs/tools/<tool>.json`)

```json
{
  "tool_id": "screwdriver",
  "symmetry": "cyl_90",
  "keypoints_3d": {
    "tip":    [0.0, 0.0,  0.10],
    "handle": [0.0, 0.0, -0.10],
    "sideA":  [0.01, 0.0, 0.0],
    "sideB":  [-0.01, 0.0, 0.0]
  },
  "keypoints_order": ["tip","handle","sideA","sideB"]
}
```

### 3) 레퍼런스 이미지 준비

FoundationPose 모델 프리 모드에서 사용할 레퍼런스 이미지 구조:

```
aiot_cv/data/references/
├── images/
│   ├── ref_001.jpg
│   ├── ref_002.jpg
│   └── ... (6-12장 권장)
├── masks/
│   ├── ref_001.png
│   ├── ref_002.png
│   └── ... (이진 마스크)
└── poses.json (선택사항)  # 레퍼런스 포즈가 있다면
```

## 🚀 사용법

### 설정 템플릿 생성
```bash
python aiot_cv/pipelines/fp_model_free.py --create-config config.yaml
```

### 비디오 파일 처리
```bash
python aiot_cv/pipelines/fp_model_free.py \
  --config config.yaml \
  --video input.mp4 \
  --output output.mp4
```

### RealSense 카메라 (향후 지원)
```bash
python aiot_cv/pipelines/fp_model_free.py \
  --config config.yaml \
  --realsense
```

## 🔧 파이프라인 흐름

1. **카메라 입력**: RGB-D 이미지 수신
2. **YOLO 검출**: YOLOv8-seg로 객체 검출/분할 → 표준화된 Detection 객체
3. **ROI 추출**: 검출된 객체의 RGB/깊이 영역 자동 크롭
4. **FoundationPose 초기화**: 레퍼런스 이미지와 매칭하여 초기 포즈 추정
5. **포즈 추적**: 연속 프레임에서 포즈 추적
6. **스무딩**: PCA 기반 시간적 스무딩 + 모션 제약 조건 적용
7. **결과 출력**: 6DoF 포즈 + FPS/ADD-S 메트릭

## 📊 성능 메트릭

- **FPS**: 실시간 처리 속도
- **ADD-S**: Average Distance of model points for Symmetric objects
- **재투영 오차**: 2D 키포인트 재투영 정확도
- **포즈 안정성**: 시간적 포즈 변화량

## 🛠️ 의존성

### 필수 패키지
```bash
pip install ultralytics opencv-python numpy
pip install open3d scikit-learn  # 포인트클라우드 처리용
pip install pyrealsense2  # RealSense 카메라용 (선택사항)
```

### YOLO 가중치
로컬 가중치를 우선 사용합니다:
- `train_results/exp1/weights/best.pt` (학습된 가중치)
- `tool_seg/yolov8n.pt` (사전학습 가중치)

## 🎛️ 고급 설정

### 포즈 스무딩 파라미터
```yaml
smoothing:
  window_size: 5                    # 스무딩 윈도우 크기
  max_angular_velocity: 2.0         # 최대 각속도 (rad/s)
  max_translation_velocity: 1.0     # 최대 병진속도 (m/s)
  outlier_threshold: 0.1           # 아웃라이어 임계값
```

### YOLO 검출 파라미터
```yaml
yolo:
  weights: null                     # 자동 탐색
  conf: 0.5                        # 신뢰도 임계값
  imgsz: 640                       # 입력 해상도
```

## 🔍 문제 해결

### 일반적인 문제들

1. **포즈 초기화 실패**
   - 레퍼런스 이미지 품질 확인 (조명, 각도, 해상도)
   - 레퍼런스 이미지 수량 (최소 6장 권장)

2. **추적 불안정**
   - 스무딩 파라미터 조정 (`max_angular_velocity`, `outlier_threshold`)
   - 카메라 내참 정확도 확인

3. **YOLO 검출 실패**
   - 신뢰도 임계값 조정 (`conf`)
   - 가중치 파일 경로 확인

### 성능 최적화

- **GPU 사용**: `device: "cuda"` 설정
- **해상도 조정**: 필요에 따라 입력 해상도 조정
- **스무딩 윈도우**: 처리 속도 vs 안정성 트레이드오프

## 📚 참고 자료

- [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose)
- [Ultralytics YOLOv8-Seg 문서](https://docs.ultralytics.com)
- [Open3D RGBD/PCL 튜토리얼](http://www.open3d.org/docs/)
- [Isaac ROS FoundationPose](https://nvidia-isaac-ros.github.io)

## 🤝 기여

이슈 리포트나 기능 제안은 GitHub Issues를 통해 해주세요.

---

**FoundationPose Model-Free Pipeline** - CAD 없는 6DoF 포즈 추정