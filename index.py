import cv2
import numpy as np
import cvzone
import tkinter as tk
from tkinter import simpledialog
import pickle

cap = cv2.VideoCapture('./easy1.mp4')

polylines = []
area_names = []
try:
    with open("polylines", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except:
    polylines = []
    area_names = []

drawing = False
points = []
current_name = ""


def is_point_inside_polygon(point, polygon):
    # Check if a point is inside a polygon
    x, y = point
    polygon = np.array(polygon)
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


def draw(event, x, y, flags, param):
    global points, drawing, polylines, area_names, current_name
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_name = simpledialog.askstring("Input", "Enter area name:")
        if current_name:
            area_names.append(current_name)
            polylines.append(np.array(points, np.int32))

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        for i, polyline in enumerate(polylines):
            if is_point_inside_polygon((x, y), polyline):
                del polylines[i], area_names[i]
                break


root = tk.Tk()
root.withdraw()  # Hide the root window

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAME, 0)
        continue

    frame = cv2.resize(frame, (1020, 500))
    for i, polyline in enumerate(polylines):
        cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME', draw)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('s'):
        with open("polylines", "wb") as f:
            data = {"polylines": polylines, "area_names": area_names}
            pickle.dump(data, f)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
