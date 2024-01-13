import cv2
import time


class FaceEyeDetection:
    def __init__(self, video_path, camera_index=0):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.cap_video = cv2.VideoCapture(video_path)
        self.cap_camera = cv2.VideoCapture(camera_index)
        self.eyes_detected = False
        self.last_detection_time = time.time()

    def detect_faces_eyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                self.eyes_detected = True

    def run_detection(self):
        while True:
            ret_video, frame_video = self.cap_video.read()

            if not ret_video:
                print("End of video.")
                break

            ret_camera, frame_camera = self.cap_camera.read()
            if not ret_camera:
                print("Error: Couldn't read a frame from the camera.")
                break

            self.detect_faces_eyes(frame_camera)

            if self.eyes_detected:
                self.last_detection_time = time.time()

            time_count = time.time() - self.last_detection_time

            if not self.eyes_detected and time_count > 5:
                height, width, _ = frame_camera.shape
                text = 'Alert: Stay Focused!'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2

                # Calculate text size and position
                text_size, baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = (frame_video.shape[1] - text_size[0]) // 2
                text_y = (frame_video.shape[0] + text_size[1]) // 2

                # Draw filled white rectangle as background
                rectangle_padding = 10  # Adjust this value to control the padding around the text
                cv2.rectangle(frame_video, (text_x - rectangle_padding, text_y - text_size[1] - rectangle_padding),
                            (text_x + text_size[0] + rectangle_padding, text_y + baseline + rectangle_padding),
                            (255, 255, 255), cv2.FILLED)

                # Draw the text on top of the white rectangle
                cv2.putText(frame_video, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

                cv2.imshow('Face and Eye Detection', frame_video)
                self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, self.cap_video.get(
                    cv2.CAP_PROP_POS_FRAMES) - 1)

                # Optionally, you can add some additional actions here, such as playing a sound or triggering an alert.

            elif not self.eyes_detected and time_count > 1:
                height, width, _ = frame_camera.shape
                cv2.circle(frame_camera, (width // 2, height // 2),
                           30, (0, 0, 255), -1)
                cv2.putText(frame_video, 'Paused', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face and Eye Detection', frame_video)
                self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, self.cap_video.get(
                    cv2.CAP_PROP_POS_FRAMES) - 1)
            else:
                cv2.putText(frame_video, 'Playing', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face and Eye Detection', frame_video)

            self.eyes_detected = False
            cv2.imshow('Camera Feed', frame_camera)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap_video.release()
        self.cap_camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = 'vids/All of the Terry Crews Old Spice Commercials.mp4'
    detection = FaceEyeDetection(video_path)
    detection.run_detection()
