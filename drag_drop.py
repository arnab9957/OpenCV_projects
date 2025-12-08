import cv2
from cvzone.HandTrackingModule import HandDetector

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

CX, CY, W, H = 100, 100 , 200, 200

class Drag_rectangle:
    def __init__(self, posCenter, size=[100, 100]):
        self.posCenter = posCenter
        self.size = size
    
    def check_collision(self, other_rects):
            """Check if this rectangle overlaps with any other rectangle"""
            cx, cy = self.posCenter
            w, h = self.size
            
            for other in other_rects:
                if other is self:
                    continue
                
                ocx, ocy = other.posCenter
                ow, oh = other.size
                
                # Check if rectangles overlap
                if (abs(cx - ocx) < (w + ow) // 2 and 
                    abs(cy - ocy) < (h + oh) // 2):
                    return True
            return False

    def update_position(self, INDEX_FINGURE, other_rects):      
        cx, cy = self.posCenter
        w, h = self.size
        
        finger_x, finger_y = INDEX_FINGURE[0], INDEX_FINGURE[1]
        
        # Check if finger is inside this rectangle
        if finger_x > cx - w//2 and finger_x < cx + w//2 and finger_y > cy - h//2 and finger_y < cy + h//2:
            # Save old position
            old_pos = self.posCenter.copy()
            
            # Try new position
            self.posCenter = [finger_x, finger_y]
            
            # Check for collision
            if self.check_collision(other_rects):
                # Revert to old position if collision detected
                self.posCenter = old_pos
                self.isDragging = False
                return False
            
            self.isDragging = True
            return True
        
        self.isDragging = False
        return False

detector = HandDetector(detectionCon=0.8, maxHands=2)

rectangle_list = []
for rect in range(5):
    rectangle_list.append(Drag_rectangle([100 + rect * 150, 100], [100, 100]))
# rect = Drag_rectangle([200, 200], [200, 200])

while True:
    success, img = camera.read()
    if not success:
        break
    img = cv2.flip(img, 1)  # Flip horizontally to mirror the image
    hands, img = detector.findHands(img, flipType=False)  # Disable cvzone's internal flip
    
    # hands contains all hand information including landmarks
    
    # Draw rectangle first (will be visible even without hands)
    rect_color = (255, 0, 255)  # Default purple color
    
    if hands:
        for hand in hands:
            lm_list = hand["lmList"]  # List of 21 landmarks
            # You can access landmarks like lm_list[8] for index finger tip
            # print(lm_list)

            # Extract only x, y coordinates (exclude z) for findDistance
            point1 = lm_list[8][:2]  # Index finger tip [x, y]
            point2 = lm_list[12][:2]  # Middle finger tip [x, y]
            distance, _, _ = detector.findDistance(point1, point2, img)
            # print(distance)
            if distance < 30:
                INDEX_FINGURE = lm_list[8]  # Landmark for index finger tip
                # if INDEX_FINGURE[0] > CX - W//2 and INDEX_FINGURE[0] < CX + W//2 and INDEX_FINGURE[1] > CY - H//2 and INDEX_FINGURE[1] < CY + H//2:
                #     rect_color = (255, 255, 0)  # Green when finger is inside
                #     CX, CY = INDEX_FINGURE[0], INDEX_FINGURE[1]  # Update position
                # else:
                #     rect_color = (255, 0, 255)  # Purple when finger is outside
                for rect in rectangle_list:
                    rect.update_position(INDEX_FINGURE, rectangle_list)
                    # CX, CY = rect.posCenter

    # Draw the rectangle (always visible)
    for rect in rectangle_list:
        CX, CY = rect.posCenter
        W, H = rect.size
        cv2.rectangle(img, (CX - W//2, CY - H//2), (CX + W//2, CY + H//2), rect_color, cv2.FILLED)

    cv2.imshow("Webcam", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()