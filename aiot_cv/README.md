# aiot_cv (A1: Keypoint Head + PnP)

경량 키포인트 Head를 이용해 YOLO(검출/분할) → ROI 크롭 → Heatmap 회귀 → Soft-Argmax → PnP까지 수행하는 파이프라인 스캐폴드입니다. 로컬 YOLO 가중치를 우선 사용합니다.

## 구조
```
aiot_cv/
  configs/
    train_kp.yaml
    tools/
      screwdriver.json
  data/
    images/            # 샘플 이미지 위치(사용자 구성)
    ann.json           # 어노테이션(예시 포맷은 본문 참조)
  src/
    io/load_k.py
    detect/yolo.py
    detect/mask_ops.py
    keypoints/{dataset,model,losses,decoder,train,infer}.py
    pose/{pnp,symmetry}.py
  scripts/
    run_image_kp_pnp.py
  tests/
    test_decoder.py
```

## Config 세팅

1) 카메라 내참(`camera.yaml`)
- 위치: 저장소 내 적절한 곳(예: 프로젝트 루트 `configs/camera.yaml`).
- 형식 예:
```yaml
fx: 600.0
fy: 600.0
cx: 320.0
cy: 240.0
depth_scale: 0.001
```
- 로딩: `aiot_cv/src/io/load_k.py`의 `load_K(path)` 사용.

2) 키포인트 학습 설정(`configs/train_kp.yaml`)
- 위치: `aiot_cv/configs/train_kp.yaml`
- 샘플 값은 이미 포함되어 있음. 필요 시 `image_size/heatmap_size/sigma/epochs` 등을 조정.

3) 도구별 3D 키포인트(`configs/tools/<tool>.json`)
- 위치: `aiot_cv/configs/tools/screwdriver.json`
- 예시:
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

4) 어노테이션(`aiot_cv/data/ann.json`)
```json
[
  {
    "image": "aiot_cv/data/images/0001.png",
    "objects": [
      {
        "tool_id": "screwdriver",
        "bbox": [x1,y1,x2,y2],
        "keypoints": {
          "tip": [u, v, 1],
          "handle": [u, v, 1],
          "sideA": [u, v, 1],
          "sideB": [u, v, 0]
        }
      }
    ]
  }
]
```

## YOLO 가중치 사용(로컬 우선)
- `aiot_cv/src/detect/yolo.py`는 다음 경로 우선 탐색:
  - `train_results/exp1/weights/best.pt`
  - `tool_seg/yolov8n.pt`
- 둘 중 하나를 배치해두면 됩니다. 필요 시 `YoloSeg(weights=...)`로 경로 지정 가능.

## 학습
```bash
python -m aiot_cv.src.keypoints.train \
  --ann aiot_cv/data/ann.json \
  --tool_id screwdriver \
  --cfg aiot_cv/configs/train_kp.yaml \
  --out weights/kp_head.pt
```

## 단일 이미지 추론(PnP 포함)
```bash
python aiot_cv/scripts/run_image_kp_pnp.py
```
- 내부 흐름: YOLO 감지 → ROI 크롭 → 키포인트 히트맵 → soft-argmax → ROI→원본 복원 → PnP → 대칭 후보 선택.
- `weights/kp_head.pt` 경로는 스크립트 내에서 로드하도록 수정 가능.

## 팁
- ROI 스케일 역변환이 틀어지지 않도록, `mask_ops.crop_square_roi` 변환 파라미터 확인.
- Heatmap 품질은 `sigma`, `heatmap_size`와 직접적으로 연관.
- PnP RANSAC 파라미터(재투영 에러 임계) 튜닝으로 안정성 향상.
