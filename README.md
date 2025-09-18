# 공학페스티벌: 도구 분할 · 3D 자세/방향 · YOLOv8 학습 프로젝트

이 저장소는 공학페스티벌 과제를 위한 물체(공구) 검출/분할 데이터와 YOLOv8 학습 파이프라인, 그리고 3D 방향/자세 분석 스크립트를 포함합니다. `tool_seg` 폴더에 학습/추론에 필요한 자료와 코드가 있으며, `train_results` 폴더에는 학습 결과가 저장됩니다.

## 주요 기능
- YOLOv8 기반 공구 검출/분할 모델 학습 및 추론
- 3D PCA/Orientation 기반 자세·방향 분석 유틸리티 스크립트
- 커스텀 데이터셋(`train/`, `valid/`) 관리 및 시각화 결과 확인

## 폴더/파일 구조
```
공학페스티벌/
├─ 1.py
├─ 2.py
├─ README.md   ← 현재 문서
├─ tool_seg/
│  ├─ data.yaml                 # YOLOv8 데이터 설정 파일
│  ├─ README.dataset.txt        # 데이터셋 관련 참고 문서(원본)
│  ├─ README.roboflow.txt       # Roboflow 관련 참고 문서(원본)
│  ├─ tool_3d_orienmtation.py   # 3D 방향/자세 관련 유틸
│  ├─ tool_3dpca .py            # 3D PCA 유틸(파일명에 공백 주의)
│  ├─ tool_3dpca_0819.py        # 3D PCA 유틸(버전)
│  ├─ tool_detect.py            # YOLO 추론/시각화 스크립트(예상)
│  ├─ tool_pca.py               # PCA 관련 유틸
│  ├─ train.py                  # YOLO 학습 스크립트
│  ├─ train/
│  │  ├─ images/                # 학습 이미지 (1377장)
│  │  ├─ labels/                # 학습 라벨 (YOLO txt, 1377개)
│  │  └─ labels.cache
│  ├─ valid/
│  │  ├─ images/                # 검증 이미지 (201장)
│  │  ├─ labels/                # 검증 라벨 (YOLO txt, 201개)
│  │  └─ labels.cache
│  └─ yolov8n.pt                # 사전학습 가중치(예: YOLOv8n)
└─ train_results/
   └─ exp1/
      ├─ args.yaml
      ├─ labels_correlogram.jpg
      ├─ labels.jpg
      ├─ results.csv
      ├─ train_batch*.jpg
      └─ weights/
         ├─ best.pt             # 최고 성능 가중치
         └─ last.pt             # 마지막 에폭 가중치
```

## 요구사항 및 설치
- OS: macOS (darwin 24.6.0 기준)
- Python: 3.9+ 권장
- 필수 패키지(예시):
  - ultralytics (YOLOv8)
  - opencv-python
  - numpy, pandas, matplotlib
  - scipy, scikit-image
  - tqdm
  - (3D 분석 스크립트가 요구할 수 있는) open3d 등

가상환경 생성 및 패키지 설치 예시:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python numpy pandas matplotlib scipy scikit-image tqdm open3d
```

> 참고: 실제 의존성은 각 스크립트 상단의 import를 확인하세요. 필요 시 추가 설치가 필요할 수 있습니다.

## 데이터셋 구조
- YOLO 형식의 `images/`(원본 이미지)와 `labels/`(동명이인의 .txt 라벨)로 구성됩니다.
- `tool_seg/data.yaml`은 학습/검증 경로 및 클래스 정보를 포함합니다.

## 학습(Training)
두 가지 방식 중 편한 방법을 사용하세요.

- 방법 A) 제공 스크립트 사용
```bash
cd tool_seg
python train.py
```
- 방법 B) Ultralytics CLI 직접 사용
```bash
yolo task=detect mode=train \
  model=tool_seg/yolov8n.pt \
  data=tool_seg/data.yaml \
  imgsz=640 epochs=100 batch=16 \
  project=train_results name=exp1
```

학습 결과는 기본적으로 `train_results/exp*/` 하위에 저장되며, 최종/최고 가중치는 `weights/last.pt`, `weights/best.pt`에 위치합니다.

## 추론(Inference)
- 제공 스크립트(예시):
```bash
cd tool_seg
python tool_detect.py --weights ../train_results/exp1/weights/best.pt --source ./valid/images --imgsz 640 --save
```
- 또는 Ultralytics CLI:
```bash
yolo task=detect mode=predict \
  model=train_results/exp1/weights/best.pt \
  source=tool_seg/valid/images \
  imgsz=640 save=True
```

> 스크립트 인자는 실제 구현에 따라 다를 수 있습니다. 사용 전 스크립트 상단/`argparse`를 확인하세요.

## 3D 자세/방향 분석
다음 유틸을 통해 추정/시각화를 수행할 수 있습니다.
- `tool_seg/tool_3d_orienmtation.py`
- `tool_seg/tool_3dpca_0819.py`
- `tool_seg/tool_3dpca .py` (파일명에 공백 존재)

예시 실행(인자명은 스크립트 확인 필요):
```bash
cd tool_seg
python tool_3d_orienmtation.py --input path/to/input --output path/to/output
```

## 결과 확인
- 학습 로그 및 메트릭: `train_results/exp*/results.csv`
- 라벨/배치 시각화: `train_results/exp*/labels.jpg`, `train_results/exp*/train_batch*.jpg`
- 최종 가중치: `train_results/exp*/weights/best.pt`, `last.pt`

## 재현(Quick Start)
```bash
# 1) 환경 준비
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python numpy pandas matplotlib scipy scikit-image tqdm open3d

# 2) 학습
cd tool_seg
python train.py  # 또는 README의 CLI 예시 참고

# 3) 추론
python tool_detect.py --weights ../train_results/exp1/weights/best.pt --source ./valid/images --save
```

## 참고/라이선스
- 데이터 라이선스 및 사용 조건은 `README.dataset.txt`, `README.roboflow.txt`를 참고하세요.
- 모델/코드 기반: Ultralytics YOLOv8 (`ultralytics` 라이브러리)

## 문의
- 이 저장소/코드 관련 이슈는 이슈 트래커 또는 담당자에게 문의해 주세요.

## 파이프라인 (end-to-end)
다음은 데이터 준비부터 3D 방향/자세 후처리까지 전체 흐름입니다. 각 단계는 입력/출력, 주요 스크립트, 생성 산출물을 함께 정의합니다.

### 0) 환경 준비
- **입력**: 없음
- **출력**: 가상환경, 필수 패키지 설치됨
- **명령 예시**:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python numpy pandas matplotlib scipy scikit-image tqdm open3d
```

### 1) 데이터 준비 (YOLO 형식)
- **입력**: 원천 이미지/라벨 또는 제공된 `tool_seg/train/`, `tool_seg/valid/`
- **출력**: YOLO 형식 디렉터리 구조 유지(`images/`, `labels/`), `tool_seg/data.yaml` 설정 확인
- **주요 파일**: `tool_seg/data.yaml`
- **체크리스트**:
  - `data.yaml`의 `path`, `train`, `val`, `names`가 실제 경로/클래스와 일치하는지 확인
  - 각 `labels/*.txt`가 동일 파일명의 `images/*.jpg`(또는 .png 등)를 갖는지 확인

### 2) 학습 (Training)
- **입력**: `tool_seg/data.yaml`, 사전학습 가중치 `tool_seg/yolov8n.pt`
- **출력**: 학습 로그, 가중치(`best.pt`, `last.pt`), 시각화 이미지
- **주요 스크립트/명령**:
  - 스크립트 방식:
    ```bash
    cd tool_seg
    python train.py
    ```
  - CLI 방식:
    ```bash
    yolo task=detect mode=train \
      model=tool_seg/yolov8n.pt \
      data=tool_seg/data.yaml \
      imgsz=640 epochs=100 batch=16 \
      project=../train_results name=exp1
    ```
- **산출물**: `train_results/exp*/`
  - `weights/best.pt`, `weights/last.pt`
  - `results.csv` (에폭별 mAP, Precision, Recall 등)
  - `labels.jpg`, `train_batch*.jpg` (라벨/배치 시각화)

### 3) 검증/평가 (Validation & Evaluation)
- **입력**: 학습 산출물(`train_results/exp*/results.csv`, `best.pt`)
- **출력**: 각종 메트릭(mAP@[.5:.95], P, R, F1), 혼동행렬/라벨 통계 이미지
- **방법**:
  - 학습 중 자동 평가 결과를 `results.csv`로 확인
  - 필요 시 별도 검증 실행:
    ```bash
    yolo task=detect mode=val \
      model=train_results/exp1/weights/best.pt \
      data=tool_seg/data.yaml \
      imgsz=640 \
      project=../train_results name=exp1_val
    ```
- **참고 산출물**: `labels_correlogram.jpg`, `labels.jpg`, 새 `results.csv`

### 4) 추론 (Inference)
- **입력**: 학습된 가중치(`best.pt`), 입력 이미지/폴더
- **출력**: 예측 결과(바운딩박스/마스크) 이미지 및/또는 JSON/TXT(구현에 따름)
- **주요 스크립트/명령**:
  - 스크립트 방식(예시):
    ```bash
    cd tool_seg
    python tool_detect.py \
      --weights ../train_results/exp1/weights/best.pt \
      --source ./valid/images \
      --imgsz 640 --save
    ```
  - CLI 방식:
    ```bash
    yolo task=detect mode=predict \
      model=train_results/exp1/weights/best.pt \
      source=tool_seg/valid/images \
      imgsz=640 save=True
    ```
- **산출물 위치**: Ultralytics 기본값에 따라 `runs/detect/predict*` 또는 스크립트 내부 지정 폴더

### 5) 3D 방향/자세 후처리 (Post-processing)
- **입력**: 추론 결과(감지된 공구의 좌표/마스크), 원본 영상/깊이 또는 포인트 데이터(필요 시)
- **출력**: 공구의 3D 주성분/방향 벡터, 각도(roll/pitch/yaw 유사 파라미터), 시각화 결과
- **주요 스크립트**:
  - `tool_seg/tool_3d_orienmtation.py`
  - `tool_seg/tool_3dpca_0819.py`
  - `tool_seg/tool_3dpca .py` (파일명에 공백 주의)
- **명령 예시(인자 이름은 스크립트 확인 필요)**:
  ```bash
  cd tool_seg
  python tool_3d_orienmtation.py \
    --input path/to/inference_outputs \
    --output path/to/3d_results \
    --mode pca --save_vis
  ```
- **처리 개략**:
  - 감지된 객체의 2D/3D 포인트 집합 추출 → PCA 수행 → 주성분 벡터로 방향 추정
  - 필요 시 좌표계/스케일 정규화, 노이즈 제거, RANSAC 등 적용

### 6) 산출물 정리 및 배포
- **입력**: `train_results/exp*/`, 추론/3D 결과 폴더
- **출력**: 모델 가중치(`best.pt`), 메트릭 보고서, 데모 이미지/영상, 3D 방향 결과물
- **권장 구조**:
  - `train_results/exp*/weights/best.pt` → 배포용 모델
  - `results.csv` → 성능 보고 첨부
  - 데모 샘플: 예측 이미지/영상, 3D 시각화 캡처

### 구성(주요 하이퍼파라미터)
- `imgsz`(입력 해상도), `epochs`, `batch`, `optimizer`, `lr0`, `weight_decay`
- 데이터 증강 옵션(Mosaic, HSV, Flip 등): `ultralytics` 설정 또는 `train.py` 내부 설정 참고
- 클래스/경로: `tool_seg/data.yaml`

### 재현 체크리스트
- [ ] `data.yaml` 경로/클래스 확인
- [ ] 가상환경 및 패키지 설치 완료
- [ ] 학습 실행 후 `best.pt` 생성 확인
- [ ] `results.csv`에서 mAP/PR 확인
- [ ] 추론 산출물(예측 이미지) 생성
- [ ] 3D 방향/자세 스크립트 실행 및 결과 저장
