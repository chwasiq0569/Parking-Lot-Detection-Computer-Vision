import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture("./easy1.mp4")

count = 0


def is_point_inside_polygon(point, polygon):
    # Check if a point is inside a polygon
    x, y = point
    polygon = np.array(polygon)
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


polylines = []
area_names = []
try:
    with open("polylines", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except:
    polylines = []
    area_names = []

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    frame_copy = frame.copy()
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list1 = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        cx = int(x1+x2)//2
        cy = int(y1+y2)//2

        if 'car' in c:
            list1.append([cx, cy])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    for i, polyline in enumerate(polylines):
        cv2.polylines(frame, [polyline], True, (0, 255, 0), 1)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

        for il in list1:
            cx1 = il[0]
            cy1 = il[1]
            result = is_point_inside_polygon((cx1, cy1), polyline)
            if result:
                cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), -1)
                cv2.polylines(frame, [polyline], True, (255, 0, 0), 2)

    cv2.imshow('FRAME', frame)
    key = cv2.waitKey(1) & 0xFF


cap.release()

cv2.destroyAllWindows()
