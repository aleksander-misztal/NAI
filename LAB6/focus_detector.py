import cv2
import time

class FaceEyeDetection:
    def __init__(self, video_path, camera_index=0):
        """
        Initialize the FaceEyeDetection class.

        Parameters:
        - video_path (str): Path to the video file.
        - camera_index (int): Index of the camera (default is 0 for the default camera).
        """
        # Load Haar cascades for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Open video capture objects for video and camera
        self.cap_video = cv2.VideoCapture(video_path)
        self.cap_camera = cv2.VideoCapture(camera_index)

        # Initialize variables for eye detection status and last detection time
        self.eyes_detected = False
        self.last_detection_time = time.time()

    def detect_faces_eyes(self, frame):
        """
        Detect faces and eyes in the given frame.

        Parameters:
        - frame (numpy.ndarray): Input image frame.
        """
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Loop through detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the region of interest (ROI) for eyes
            roi_gray = gray[y:y + h, x:x + w]

            # Detect eyes in the ROI
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            # Loop through detected eyes and draw rectangles
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                self.eyes_detected = True

    def run_detection(self):
        """Run the face and eye detection process."""
        while True:
            # Read frames from the video and camera
            ret_video, frame_video = self.cap_video.read()
            if not ret_video:
                print("End of video.")
                break

            ret_camera, frame_camera = self.cap_camera.read()
            if not ret_camera:
                print("Error: Couldn't read a frame from the camera.")
                break

            # Detect faces and eyes in the camera frame
            self.detect_faces_eyes(frame_camera)

            # Update the last detection time if eyes are detected
            if self.eyes_detected:
                self.last_detection_time = time.time()

            # Pause the video and display a circle if eyes are not detected for more than 1 second
            if not self.eyes_detected and time.time() - self.last_detection_time > 1:
                height, width, _ = frame_camera.shape
                cv2.circle(frame_camera, (width // 2, height // 2), 30, (0, 0, 255), -1)
                cv2.putText(frame_video, 'Paused', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face and Eye Detection', frame_video)
                self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, self.cap_video.get(cv2.CAP_PROP_POS_FRAMES) - 1)
            else:
                # Display "Playing" if eyes are detected
                cv2.putText(frame_video, 'Playing', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face and Eye Detection', frame_video)

            # Reset the eyes_detected flag
            self.eyes_detected = False

            # Display the camera feed
            cv2.imshow('Camera Feed', frame_camera)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture objects and close all windows
        self.cap_video.release()
        self.cap_camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    video_path = 'vids/All of the Terry Crews Old Spice Commercials.mp4'
    detection = FaceEyeDetection(video_path)
    detection.run_detection()
