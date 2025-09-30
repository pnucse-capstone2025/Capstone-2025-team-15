from ultralytics import YOLO
import cv2

# 학습된 모델 로드
model = YOLO('./runs/detect/train/weights/best.pt')

# 이미지로 예측 수행
results = model('./frame4_0002.png')

# 결과 확인
# 1. 결과 이미지 바로 띄워서 보기
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    cv2.imshow('YOLO V8 Detection', im_array)

WIDTH =  1499
HEIGHT =  970
# 1. 탐지된 객체 정보(좌표, 신뢰도 등) 및 '중심점'을 터미널에 출력
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    for box in boxes:
        # 바운딩 박스 좌표 [x1, y1, x2, y2]
        xy = box.xyxy[0] 
        x1, y1, x2, y2 = xy
        
        # 클래스 이름
        cls_name = model.names[int(box.cls[0])]
        
        # 신뢰도 점수
        conf = box.conf[0]
        
        # --- 여기부터 중심점 계산 코드가 추가되었습니다 ---
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # ----------------------------------------------
        
        coord_x = center_x /WIDTH
        coord_y = center_y / HEIGHT
        # 최종 정보 출력 (신뢰도와 중심점은 소수점 2자리까지만 표시)
        print(f"객체: {cls_name}, 신뢰도: {conf:.2f}, 중심점: ({coord_x:.3f}, {coord_y:.3f})")


# 2. 결과 이미지를 창으로 띄워서 보기
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    cv2.imshow('YOLO V8 Detection', im_array)

# 키 입력 대기 및 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
