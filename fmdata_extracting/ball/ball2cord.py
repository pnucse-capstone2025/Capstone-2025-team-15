from ultralytics import YOLO
import torch
import cv2, os, csv
import numpy as np

model = YOLO('./runs/detect/train/weights/best.pt')

WIDTH =  1499
HEIGHT =  970
input_folder = './video6/'
output_folder = './output/'

ballCords = []
image_files = sorted([f for f in os.listdir(input_folder) if f.startswith("frame5_")])

for i, filename in enumerate(image_files):
    image_path = os.path.join(input_folder, filename)
    results = model(image_path)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        # 1. 객체가 하나 이상 감지되었는지 확인합니다.
        if len(boxes) > 0:
            # 2. 가장 높은 신뢰도를 가진 객체의 인덱스를 찾습니다.
            best_idx = torch.argmax(boxes.conf)
            
            # 3. 해당 인덱스를 사용하여 가장 좋은 바운딩 박스 정보만 추출합니다.
            best_box = boxes[best_idx]
            
            # 바운딩 박스 좌표 [x1, y1, x2, y2]
            xy = best_box.xyxy[0] 
            x1, y1, x2, y2 = xy
            
            # 신뢰도 점수
            conf = best_box.conf[0]
            
            # 중심점 계산
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 좌표 정규화
            coord_x = center_x / WIDTH
            coord_y = center_y / HEIGHT
            
            # 최종 정보 출력 (가장 신뢰도 높은 객체 하나만 출력됨)
            print(f"Frame {i}: Highest Conf: {conf:.2f}, Center: ({coord_x:.5f}, {coord_y:.5f})")
            
            # 4. 가장 신뢰도 높은 객체 하나의 정보만 리스트에 추가합니다.
            ballCords.append({
                "frame": i,
                "x": coord_x.item(), # .item()을 사용하여 숫자만 저장
                "y": coord_y.item()  # .item()을 사용하여 숫자만 저장
            })
        else:
            # 객체가 감지되지 않은 경우, 건너뛰거나 기본값을 추가할 수 있습니다.
            print(f"Frame {i}: No object detected.")
            ballCords.append({"frame": i, "x": np.nan, "y": np.nan})

    csv_name = 'balls3.csv'
    output_path = os.path.join(output_folder, csv_name)
    for i, item in enumerate(ballCords):
        item['time'] = round(i * 0.04, 2)

    with open(output_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["frame", "x", "y", "time"])
        
        # Write the header
        writer.writeheader()
        writer.writerows(ballCords)

print(f"Data has been written to {output_path}")