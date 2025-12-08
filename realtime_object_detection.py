import cv2
import numpy as np

# # img = cv2.imread(r"D:\Desktop\PYTHON\OpenCV_projects\image.jpg")
# # img = cv2.resize(img, (640, 480))
# camera = cv2.VideoCapture(0)

# while True:
#     ret, img = camera.read()
#     if not ret:
#         break
#     # img = cv2.resize(img, (640, 480))
#     # cv2.imshow("Input", img)
    
#     classNames = []
#     classFile = r"D:\Desktop\PYTHON\OpenCV_projects\coco.names"
#     with open(classFile, "rt") as f:
#         classNames = f.read().rstrip("\n").split("\n")
#     # print(classNames)

#     config_path = (
#         r"D:\Desktop\PYTHON\OpenCV_projects\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
#     )
#     weights_path = r"D:\Desktop\PYTHON\OpenCV_projects\frozen_inference_graph.pb"

#     net = cv2.dnn_DetectionModel(config_path, weights_path)
#     net.setInputSize(320, 320)
#     net.setInputScale(1.0 / 127.5)
#     net.setInputMean((127.5, 127.5, 127.5))
#     net.setInputSwapRB(True)

#     classIds, confs, bbox = net.detect(img, confThreshold=0.5)
#     # print(classIds, bbox)

#     if len(classIds) > 0:  # Checking for empty list
#         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
#             cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            
#             # Ensure classId is within valid range
#             class_name = classNames[classId - 1] if 0 < classId <= len(classNames) else f"Class {classId}"
            
#             cv2.putText(
#                 img,
#                 class_name.upper(),
#                 (box[0] + 10, box[1] + 30),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1,
#                 (255, 0, 0),
#                 2,
#             )
#             cv2.putText(
#                 img,
#                 str(round(confidence * 100, 2)),
#                 (box[0] + 200, box[1] + 30),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1,
#                 (255, 0, 0),
#                 2,
#             )
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break


# # cv2.waitKey(0)
# camera.release()
# cv2.destroyAllWindows()

# Load model
net = cv2.dnn.readNet("yolov8n.onnx")

# COCO class labels
classes = open("coco.names").read().strip().split("\n")

# Generate random colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()

    # Post-processing for YOLOv8
    outputs = outputs[0]  # Shape: (1, 84, 8400) -> (84, 8400)
    outputs = outputs.T   # Transpose to (8400, 84)

    boxes = []
    confidences = []
    class_ids = []

    # Extract detections
    for detection in outputs:
        # First 4 values are [x_center, y_center, width, height]
        # Remaining values are class scores
        scores = detection[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # Confidence threshold
            # Scale coordinates back to original image size
            x_center, y_center, w, h = detection[0:4]
            x_center = int(x_center * width / 640)
            y_center = int(y_center * height / 640)
            w = int(w * width / 640)
            h = int(h * height / 640)

            # Convert to top-left corner coordinates
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label background
            text = f"{label}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width, y), color, -1)

            # Draw label text
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
