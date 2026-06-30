import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Previous point
prev_x, prev_y = 0, 0

# Default brush settings
color = (255, 0, 255)   # Purple
brush_size = 5
eraser_size = 25

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Red dot on fingertip
            cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)

            # Start point
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            # Draw line
            thickness = eraser_size if color == (0, 0, 0) else brush_size

            cv2.line(
                canvas,
                (prev_x, prev_y),
                (x, y),
                color,
                thickness
            )

            prev_x, prev_y = x, y

    else:
        prev_x, prev_y = 0, 0

    # Merge canvas and webcam
    img = cv2.add(img, canvas)

    # Instructions
    cv2.putText(img,
                "R=Red G=Green B=Blue P=Purple",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1)

    cv2.putText(img,
                "E=Eraser C=Clear S=Save Q=Quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1)

    cv2.imshow("AI Virtual Air Drawing", img)

    key = cv2.waitKey(1) & 0xFF

    # Colors
    if key == ord('r'):
        color = (0, 0, 255)

    elif key == ord('g'):
        color = (0, 255, 0)

    elif key == ord('b'):
        color = (255, 0, 0)

    elif key == ord('p'):
        color = (255, 0, 255)

    # Eraser
    elif key == ord('e'):
        color = (0, 0, 0)

    # Clear
    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Save
    elif key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print("Saved:", filename)

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()