import cv2

def detect_heads(video_path):
    # Load pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Iterate through detected faces and draw bounding boxes
        for (x, y, w, h) in faces:
            # Calculate the center of the detected face
            center_x = x + w // 2
            center_y = y + h // 2

        

            # Draw a circle around the crosshair
            cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 2)

            # Draw a crosshair at the center of the detected face
            cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=30,
                           thickness=2, line_type=cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "squid.mp4"
    detect_heads(video_path)
