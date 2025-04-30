
import cv2
import numpy as np
import json
import os

# Definições
W, H = 640, 480
VIDEO_FILE = 'video_20250410_072000.avi'
POINTS_FILE = 'warp_points.json'

# Carregar pontos do ficheiro JSON
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

# Pontos destino (vista de cima)
dst = np.float32([
    [0, 0],
    [W, 0],
    [0, H],
    [W, H]
])

matrix = cv2.getPerspectiveTransform(pts, dst)

# Abrir vídeo
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

    # Converter para HSV
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Intervalo típico para branco puro
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # Visualização
    overlay = warped.copy()
    overlay[clean > 0] = [0, 255, 0]  # pintar linha em verde

    cv2.imshow("Warped", warped)
    cv2.imshow("Mask", clean)
    cv2.imshow("Linha", overlay)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
