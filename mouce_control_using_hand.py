import cv2
import mediapipe
import pyautogui

CAPTURE_HANDS = mediapipe.solutions.hands.Hands()
DRAWING_OPTIONS = mediapipe.solutions.drawing_utils

camera = cv2.VideoCapture(0)
x1, y1, x2, y2 = 0, 0, 0, 0

while True:
    _, image = camera.read()
    image = cv2.flip(image, 1)

    # Conver theimage to a RGB image
    RGB_IMAGE = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    OUTPUT_HANDS = CAPTURE_HANDS.process(RGB_IMAGE)
    ALL_HANDS = OUTPUT_HANDS.multi_hand_landmarks
    if ALL_HANDS:
        # Draw the landmarks and capture the position of fingers
        for hand in ALL_HANDS:
            DRAWING_OPTIONS.draw_landmarks(image, hand)
            # Capturing the position of fingers
            one_hand_landmark = hand.landmark
            for id, lm in enumerate(one_hand_landmark):
                height, width, _ = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                # print(id, cx, cy)
                # Moving the pointer finger
                if id == 8:
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    screen_width, screen_height = pyautogui.size()
                    new_x = int((cx / width) * screen_width)
                    new_y = int((cy / height) * screen_height)
                    pyautogui.moveTo(new_x, new_y)
                    x1, y1 = cx, cy
                if id == 4:
                    cv2.circle(image, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
                    x2, y2 = cx, cy
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                # print(dist)
                if dist < 26:
                    pyautogui.click()
                    # pyautogui.sleep(1)

    cv2.imshow("Pointer movement", image)
    key = cv2.waitKey(30)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
