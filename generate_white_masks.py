
import os
import cv2
import numpy as np

input_dir = "images"
output_dir = "masks"
os.makedirs(output_dir, exist_ok=True)

def process_image(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Crop opcional: ignorar parte superior da imagem (onde há menos pista)
    roi = image[int(h*0.4):, :]  # manter só parte inferior

    # Conversão para HSV e máscara de branco
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morfologia para remover ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos e manter apenas os grandes e estreitos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = h_box / (w_box + 1e-5)
        if area > 50 and aspect_ratio > 1.5:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # Voltar a encaixar na imagem original (com crop compensado)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    final_mask[int(h*0.4):, :] = clean_mask

    return final_mask

# Processar todas as imagens
for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        mask = process_image(input_path)
        cv2.imwrite(output_path, mask)
        print(f"Salvou: {output_path}")
