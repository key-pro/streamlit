from ultralytics import YOLO
import cv2

# YOLOv11をロード
model = YOLO("yolo11n.pt")

# 自分のPCのカメラを取得
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # 物体検出（動画ではtrackメソッドを利用する）
    results = model.track(frame, persist=True)
    # 物体検出（静止画ではdetectメソッドを利用する）
    # results = model.detect("path_to_image.jpg")
    
    # フレームに結果を可視化
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv11トラッキング", annotated_frame)

    # qキーでプログラムを終了する
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
# キャプチャを終了
cap.release()
cv2.destroyAllWindows()
