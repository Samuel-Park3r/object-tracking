from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import cv2

model = YOLO('./yolov8n.pt')
tracker = DeepSort(max_age=5)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if ret:
        bbs = list(map(lambda e: (e.xyxy[0].tolist(), e.conf.tolist()[0], e.cls.tolist()[0]), model.track(source=frame, conf=0.6, iou=0.6)[0].boxes))
        if bbs:
            tracks = tracker.update_tracks(bbs, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 200, 0), 1)
    cv2.imshow('obj_tracking', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break