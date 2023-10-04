import cv2
import numpy as np

cap = cv2.VideoCapture('./SLOWED_PARKING.mp4')

drawing = False
points = []
polylines = []


def is_point_inside_polygon(point, polygon):
    # Check if a point is inside a polygon
    x, y = point
    polygon = np.array(polygon)
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


def draw(event, x, y, flags, param):
    global points, drawing, polylines
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        polylines.append(np.array(points, np.int32))
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # Check if the user double-clicked inside any polygon and remove it
        for i, polyline in enumerate(polylines):
            if is_point_inside_polygon((x, y), polyline):
                del polylines[i]
                break


while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAME, 0)
        continue

    frame = cv2.resize(frame, (1020, 500))
    for polyline in polylines:
        cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)

    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME', draw)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
