import cv2
import numpy as np

cap = cv2.VideoCapture('./SLOWED_PARKING.mp4')

drawing = False
points = []
polylines = []


def draw(event, x, y, flags, param):
    global points, drawing
    drawing = True
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        polylines.append(np.array(points, np.int32))
    print(polylines)


while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAME, 0)
        continue

    frame = cv2.resize(frame, (1020, 500))
    for polyline in polylines:
        cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)

    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME',  draw)
    key = cv2.waitKey(100) & 0xFF

cap.release()
cv2.destroyAllWindows()
