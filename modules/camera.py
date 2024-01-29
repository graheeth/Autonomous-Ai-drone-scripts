import cv2

def create_camera(camera_index=0):
    # Open a handle to the camera
    return cv2.VideoCapture(camera_index)

def get_image_size(cap):
    # Retrieve width and height from the capture device
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

def get_video(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return None
    return frame

def close_camera(cap):
    cap.release()

if __name__ == "__main__":
    cap = create_camera(0)

    while True:
        img = get_video(cap)
        if img is not None:
            cv2.imshow("camera", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

    close_camera(cap)
    cv2.destroyAllWindows()
