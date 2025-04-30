
import cv2
import numpy as np
import json
import os

# Definições
W, H = 640, 480
VIDEO_FILE = 'video_20250410_072000.avi'
POINTS_FILE = 'warp_points.json'

def nothing(x):
    pass

# Carregar pontos do warp
if not os.path.exists(POINTS_FILE):
    raise FileNotFoundError("Ficheiro warp_points.json não encontrado.")

with open(POINTS_FILE, 'r') as f:
    pts_json = json.load(f)

pts = np.float32([
    [pts_json['x1'], pts_json['y1']],
    [pts_json['x2'], pts_json['y2']],
    [pts_json['x3'], pts_json['y3']],
    [pts_json['x4'], pts_json['y4']]
])

dst = np.float32([
    [0, 0],
    [W, 0],
    [0, H],
    [W, H]
])

matrix = cv2.getPerspectiveTransform(pts, dst)

# Trackbars
cv2.namedWindow("Controlo")
cv2.createTrackbar("H min", "Controlo", 0, 180, nothing)
cv2.createTrackbar("H max", "Controlo", 180, 180, nothing)
cv2.createTrackbar("S min", "Controlo", 0, 255, nothing)
cv2.createTrackbar("S max", "Controlo", 40, 255, nothing)
cv2.createTrackbar("V min", "Controlo", 200, 255, nothing)
cv2.createTrackbar("V max", "Controlo", 255, 255, nothing)
cv2.createTrackbar("Kernel", "Controlo", 3, 20, nothing)

# Vídeo
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    raise IOError("Não foi possível abrir o vídeo.")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (W, H))
    warped = cv2.warpPerspective(frame, matrix, (W, H))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Sliders
    hmin = cv2.getTrackbarPos("H min", "Controlo")
    hmax = cv2.getTrackbarPos("H max", "Controlo")
    smin = cv2.getTrackbarPos("S min", "Controlo")
    smax = cv2.getTrackbarPos("S max", "Controlo")
    vmin = cv2.getTrackbarPos("V min", "Controlo")
    vmax = cv2.getTrackbarPos("V max", "Controlo")
    ksize = cv2.getTrackbarPos("Kernel", "Controlo")
    ksize = max(1, ksize | 1)  # garantir que é ímpar

    # Máscara
    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lower, upper)

    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Resultado com overlay
    overlay = warped.copy()
    overlay[cleaned > 0] = [0, 255, 0]

    cv2.imshow("Warped", warped)
    cv2.imshow("Mask", cleaned)
    cv2.imshow("Resultado", overlay)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
