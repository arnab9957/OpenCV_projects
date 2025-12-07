import cv2
import mediapipe
import pyautogui

capture_hands = mediapipe.solutions.hands.Hands()
drawing_option = mediapipe.solutions.drawing_utils

camera = cv2.VideoCapture(0)
x1, y1, x2, y2 = 0, 0, 0, 0

while True:
    _, image = camera.read()
    image = cv2.flip(image, 1)

    # Conver theimage to a RGB image
    RGB_IMAGE = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    outpur_hands = capture_hands.process(RGB_IMAGE)
    all_hands = outpur_hands.multi_hand_landmarks
    if all_hands:
        for hand in all_hands:
            drawing_option.draw_landmarks(image, hand)
            # Capturing the position of fingures
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
