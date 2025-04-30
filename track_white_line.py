
import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture('video_20250410_072000.avi') 
cv2.namedWindow('Trackbars')

# Criar sliders HSV
cv2.createTrackbar('H_min', 'Trackbars', 0, 180, nothing)
cv2.createTrackbar('H_max', 'Trackbars', 180, 180, nothing)
cv2.createTrackbar('S_min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('S_max', 'Trackbars', 30, 255, nothing)
cv2.createTrackbar('V_min', 'Trackbars', 200, 255, nothing)
cv2.createTrackbar('V_max', 'Trackbars', 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    h, w = frame.shape[:2]
    roi = frame[int(h*0.4):, :]  # cortar parte superior

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


    h_min = cv2.getTrackbarPos('H_min', 'Trackbars')
    h_max = cv2.getTrackbarPos('H_max', 'Trackbars')
    s_min = cv2.getTrackbarPos('S_min', 'Trackbars')
    s_max = cv2.getTrackbarPos('S_max', 'Trackbars')
    v_min = cv2.getTrackbarPos('V_min', 'Trackbars')
    v_max = cv2.getTrackbarPos('V_max', 'Trackbars')

    # Criar máscara HSV
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # Contornos filtrados
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(clean)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect = h_box / (w_box + 1e-5)
        if area > 100 and aspect > 1.2:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)

    # Mostrar resultados
    vis = roi.copy()
    vis[filtered > 0] = [0, 255, 0]  # destacar a linha em verde

    cv2.imshow('Original', roi)
    cv2.imshow('HSV', mask)
    cv2.imshow('Filtrada', filtered)
    cv2.imshow('Resultado', vis)

    key = cv2.waitKey(30)
    if key == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
