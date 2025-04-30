import cv2
import numpy as np
import json

# Inicializar janela
cv2.namedWindow("Tuner")

# Trackbars
def nothing(x): pass

cv2.createTrackbar("ROI Top", "Tuner", 200, 480, nothing)
cv2.createTrackbar("ROI Bottom", "Tuner", 480, 480, nothing)

cv2.createTrackbar("H min", "Tuner", 0, 179, nothing)
cv2.createTrackbar("H max", "Tuner", 179, 179, nothing)
cv2.createTrackbar("S min", "Tuner", 0, 255, nothing)
cv2.createTrackbar("S max", "Tuner", 255, 255, nothing)
cv2.createTrackbar("V min", "Tuner", 200, 255, nothing)
cv2.createTrackbar("V max", "Tuner", 255, 255, nothing)

cv2.createTrackbar("Canny Low", "Tuner", 50, 255, nothing)
cv2.createTrackbar("Canny High", "Tuner", 150, 255, nothing)
cv2.createTrackbar("Kernel Size", "Tuner", 3, 10, nothing)

# Vídeo de exemplo
cap = cv2.VideoCapture("video_20250410_072000.avi")  # usar o último vídeo enviado

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



# Loop de preview
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    config = {
        "roi_top": cv2.getTrackbarPos("ROI Top", "Tuner"),
        "roi_bottom": cv2.getTrackbarPos("ROI Bottom", "Tuner"),
        "hmin": cv2.getTrackbarPos("H min", "Tuner"),
        "hmax": cv2.getTrackbarPos("H max", "Tuner"),
        "smin": cv2.getTrackbarPos("S min", "Tuner"),
        "smax": cv2.getTrackbarPos("S max", "Tuner"),
        "vmin": cv2.getTrackbarPos("V min", "Tuner"),
        "vmax": cv2.getTrackbarPos("V max", "Tuner"),
        "canny_low": cv2.getTrackbarPos("Canny Low", "Tuner"),
        "canny_high": cv2.getTrackbarPos("Canny High", "Tuner"),
        "kernel": max(1, cv2.getTrackbarPos("Kernel Size", "Tuner")),
    }

    processed = apply_pipeline(frame.copy(), config)
    mask_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    combined = np.vstack([frame, mask_color])

    cv2.imshow("combined", mask_color)



    key = cv2.waitKey(30)
    if key == ord('s'):
        with open("config_tuner.json", "w") as f:
            json.dump(config, f, indent=4)
        print("Configuração salva para config_tuner.json")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

