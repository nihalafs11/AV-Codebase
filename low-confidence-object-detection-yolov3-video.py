# An example of low confidence object detection from the YOLO network on a using a given video of a Vehicle going through an airport
# From "YOLOv3: An Incremental Improvement" https://pjreddie.com/media/files/papers/YOLOv3.pdf

import cv2
import numpy as np

# Load the COCO class labels and yolo model. Get the output layer names
class_names = open("coco.names").read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the image from a video
cap = cv2.VideoCapture("NMMC_Route.mp4")
while True:
    # Read the next frame from the video
    ret, image = cap.read()

    # Break if we reached the end of the video
    if not ret:
        break
    (H, W) = image.shape[:2]

    # Create the blob with a size of (416, 416), swap red and blue channels, apply scale factor of 1/255
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Feed the input blob to the network, perform inference and get the output:
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    # Get inference time:
    t, _ = net.getPerfProfile()
    print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))

    # Initialization:
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Get class ID and confidence of the current detection:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions:
            if confidence > 0.25:
                # Scale the bounding box coordinates (center, width, height) using the dimensions of the original image:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculate the top-left corner of the bounding box:
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update the information we have for each detection:
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # We can apply non-maxima suppression (eliminate weak and overlapping bounding boxes):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Show the results (if any object is detected after non-maxima suppression):
    if len(indices) > 0:
        for i in indices.flatten():
            # Extract the (previously recalculated) bounding box coordinates:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw label and confidence:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = "{}: {:.4f}".format(class_names[class_ids[i]], confidences[i])
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            y = max(y, labelSize[1])
            cv2.rectangle(image, (x, y - labelSize[1]), (x + labelSize[0], y + 0), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Video", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop on 'q' key press
        break

cv2.destroyAllWindows()
