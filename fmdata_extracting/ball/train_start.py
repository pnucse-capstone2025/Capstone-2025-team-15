from ultralytics import YOLO

# 사전 학습된 YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 모델 학습
if __name__ == '__main__':
    results = model.train(data='./dataset/data.yaml', epochs=100, imgsz=640)