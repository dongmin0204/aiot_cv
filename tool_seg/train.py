from ultralytics import YOLO

# 세그멘테이션 전용 모델 로드
# (n=나노, s=스몰, m=미디엄, l=라지, x=엑스트라라지)
model = YOLO("yolo11n-seg.pt")  # 사전 학습된 segmentation 모델

# 2 GPU 학습
results = model.train(
    data="../data.yaml",  # 데이터셋 경로
    epochs=30,
    imgsz=640,
    device="0"  # GPU 0과 1번 사용
)
