
import cv2
import numpy as np
import json
import os

# Configurações
W, H = 640, 480
VIDEO_FILE = 'video.mp4'
POINTS_FILE = 'warp_points.json'
SAVE_DIR = 'masks'

os.makedirs(SAVE_DIR, exist_ok=True)

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

# Criar sliders
cv2.namedWindow("Controlo")
cv2.createTrackbar("H min", "Controlo", 0, 180, nothing)
cv2.createTrackbar("H max", "Controlo", 180, 180, nothing)
cv2.createTrackbar("S min", "Controlo", 0, 255, nothing)
cv2.createTrackbar("S max", "Controlo", 40, 255, nothing)
cv2.createTrackbar("V min", "Controlo", 200, 255, nothing)
cv2.createTrackbar("V max", "Controlo", 255, 255, nothing)
cv2.createTrackbar("Kernel", "Controlo", 3, 20, nothing)
cv2.createTrackbar("Canny 1", "Controlo", 50, 255, nothing)
cv2.createTrackbar("Canny 2", "Controlo", 150, 255, nothing)
cv2.createTrackbar("Use Canny", "Controlo", 0, 1, nothing)

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    raise IOError("Não foi possível abrir o vídeo.")

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (W, H))
    warped = cv2.warpPerspective(frame, matrix, (W, H))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Ler sliders
    hmin = cv2.getTrackbarPos("H min", "Controlo")
    hmax = cv2.getTrackbarPos("H max", "Controlo")
    smin = cv2.getTrackbarPos("S min", "Controlo")
    smax = cv2.getTrackbarPos("S max", "Controlo")
    vmin = cv2.getTrackbarPos("V min", "Controlo")
    vmax = cv2.getTrackbarPos("V max", "Controlo")
    ksize = max(1, cv2.getTrackbarPos("Kernel", "Controlo") | 1)
    c1 = cv2.getTrackbarPos("Canny 1", "Controlo")
    c2 = cv2.getTrackbarPos("Canny 2", "Controlo")
    use_canny = cv2.getTrackbarPos("Use Canny", "Controlo")

    # HSV masking
    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # Canny se ativado
    if use_canny:
        edges = cv2.Canny(clean, c1, c2)
        final_mask = edges
    else:
        final_mask = clean

    overlay = warped.copy()
    overlay[final_mask > 0] = [0, 255, 0]

    cv2.imshow("Warped", warped)
    cv2.imshow("Máscara final", final_mask)
    cv2.imshow("Resultado", overlay)

    key = cv2.waitKey(30)
    if key == 27:
        break
    elif key == ord('s'):
        filename = f"{SAVE_DIR}/mask_{frame_id:05d}.png"
        cv2.imwrite(filename, final_mask)
        print(f"Guardada: {filename}")
        frame_id += 1

cap.release()
cv2.destroyAllWindows()
