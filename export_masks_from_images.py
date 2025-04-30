
import cv2
import numpy as np
import json
import os

# === CONFIGURAÇÕES ===
INPUT_DIR = "images"
OUTPUT_DIR = "masks"
USE_CANNY = False  
HSV_RANGE = {
    'hmin': 0,
    'hmax': 180,
    'smin': 0,
    'smax': 40,
    'vmin': 200,
    'vmax': 255
}
CANNY_THRESHOLDS = (50, 150)
KERNEL_SIZE = 3
W, H = 640, 480

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("warp_points.json", "r") as f:
    pts_json = json.load(f)

pts = np.float32([
    [pts_json["x1"], pts_json["y1"]],
    [pts_json["x2"], pts_json["y2"]],
    [pts_json["x3"], pts_json["y3"]],
    [pts_json["x4"], pts_json["y4"]]
])
dst = np.float32([
    [0, 0],
    [W, 0],
    [0, H],
    [W, H]
])
matrix = cv2.getPerspectiveTransform(pts, dst)

# Processar imagens
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(INPUT_DIR, fname)
    image = cv2.imread(path)
    if image is None:
        print(f"Erro ao ler {fname}")
        continue

    image = cv2.resize(image, (W, H))
    warped = cv2.warpPerspective(image, matrix, (W, H))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # HSV mask
    lower = np.array([HSV_RANGE['hmin'], HSV_RANGE['smin'], HSV_RANGE['vmin']])
    upper = np.array([HSV_RANGE['hmax'], HSV_RANGE['smax'], HSV_RANGE['vmax']])
    mask = cv2.inRange(hsv, lower, upper)

    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # Canny se ativado
    if USE_CANNY:
        c1, c2 = CANNY_THRESHOLDS
        clean = cv2.Canny(clean, c1, c2)

    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, clean)
    print(f"Guardada máscara: {out_path}")
