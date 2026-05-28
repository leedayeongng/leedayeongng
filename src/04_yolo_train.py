
from ultralytics import YOLO
import os

# 저장된 모델 로드
model = YOLO("/home/ubuntu/project/models/yolo/yolov8m.pt")

# 학습 설정 (마운트 후 데이터 경로 수정)
results = model.train(
    data="/home/ubuntu/project/src/ovarian_data.yaml",  # 마운트 후 작성
    epochs=100,
    imgsz=640,
    batch=16,          # V100 32GB라 넉넉
    device=0,          # GPU 사용
    project="/home/ubuntu/project/results",
    name="ovarian_yolo",
    patience=20,       # early stopping
    save=True,
    plots=True
)

# 검증
metrics = model.val()
print(metrics)

# 예측
pred = model.predict(
    source="/home/ubuntu/project/data/ultrasound/",
    save=True,
    conf=0.25
)
