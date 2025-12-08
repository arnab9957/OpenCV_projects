import cv2

camera = cv2.VideoCapture(0)


while True:
    success, img = camera.read()
    img = cv2.flip(img, 1)

    cv2.imshow("Webcam", img)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.imwrite("bg_image.jpg", img)
        break

camera.release()
cv2.destroyAllWindows()