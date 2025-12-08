import cv2
import numpy as np

# load bg inage
bg = cv2.imread("invesible_cloak/bg_image.jpg")


camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if not ret:
        break
    img = cv2.flip(img, 1)

    # Resize background to match current frame size
    if bg.shape[:2] != img.shape[:2]:
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]))

    hsv_frsme = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV range for red color (red wraps around in HSV, so we need two ranges)
    # Lower red range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_frsme, lower_red1, upper_red1)
    
    # Upper red range
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_frsme, lower_red2, upper_red2)
    
    # Combine both red masks
    color_mask = mask1 + mask2

    # Morphology to remove noise
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=10)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    # Show background where the color is detected (makes it invisible)
    part1 = cv2.bitwise_and(bg, bg, mask=color_mask)

    # Invert mask to get non-color areas
    color_mask_inv = cv2.bitwise_not(color_mask)

    # Show current frame where the color is NOT detected (everything else visible)
    part2 = cv2.bitwise_and(img, img, mask=color_mask_inv)

    cv2.imshow("Image", part1+part2)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()