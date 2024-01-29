import cv2
import numpy as np
import time

# Global variables
camera = None
net = None
output_layers = None
person_class_id = None

def initialize_detector():
    global camera, net, output_layers, person_class_id

    # Load YOLO
    net = cv2.dnn.readNet("/home/solid/Documents/nava/ppl_detection/yolo/yolov3-tiny.weights", 
                          "/home/solid/Documents/nava/ppl_detection/yolo/yolov3-tiny.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load COCO class labels
    labelsPath = "/home/solid/Documents/nava/ppl_detection/yolo/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    person_class_id = LABELS.index("person")

    # Initialize camera
    camera = cv2.VideoCapture(0)  # 0 for the default camera
    print("Detector initialized")

def get_image_size():
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

def close_camera():
    camera.release()
    print("Camera closed")

def get_detections():
    ret, frame = camera.read()
    if not ret:
        return [], None, None

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    start_time = time.time()
    outs = net.forward(output_layers)
    fps = 1 / (time.time() - start_time)

    person_detections = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == person_class_id:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Modified part to include pixel values in the detections array
                pixel_detection = [x, y, w, h, confidence] + list(scores)
                person_detections.append(pixel_detection)  # Store the detection with pixel values

    return person_detections, fps, frame

if __name__ == "__main__":
    # Example usage
    initialize_detector()
    try:
        while True:
            persons, fps, image = get_detections()
            if image is not None:
                cv2.imshow("Frame", image)
                print(f"Persons: {len(persons)}, FPS: {fps:.2f}, {persons}")
            else:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        close_camera()
        cv2.destroyAllWindows()