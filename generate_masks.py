import cv2
import os
import json
import numpy as np

# Função de pipeline com ROI, HSV, Canny e Morph
def apply_pipeline(frame, config):
    # ROI
    roi = frame[config["roi_top"]:config["roi_bottom"], :]

    # HSV Filter
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       (config["hmin"], config["smin"], config["vmin"]),
                       (config["hmax"], config["smax"], config["vmax"]))

    # Canny
    edges = cv2.Canny(mask, config["canny_low"], config["canny_high"])

    # Morphology
    k = config["kernel"]
    kernel = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closed

# Carregar configuração
with open("config_tuner.json", "r") as f:
    config = json.load(f)

# Diretórios
input_dir = "images"
output_dir = "masks"
os.makedirs(output_dir, exist_ok=True)

# Extensões válidas
valid_exts = [".jpg", ".jpeg", ".png"]

# Processar todas as imagens
for fname in sorted(os.listdir(input_dir)):
    if not any(fname.lower().endswith(ext) for ext in valid_exts):
        continue

    path = os.path.join(input_dir, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"[ERRO] Não foi possível ler: {fname}")
        continue

    mask = apply_pipeline(img, config)

    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, mask)
    print(f"[OK] {fname} -> {out_path}")

