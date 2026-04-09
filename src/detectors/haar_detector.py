import cv2

class HaarDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

    def detect_faces(self, gray_frame):
        """Detect frontal faces."""
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return list(faces) if len(faces) > 0 else []

    def detect_profile_faces(self, gray_frame):
        """Detect side-profile faces as secondary fallback."""
        faces = self.profile_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return list(faces) if len(faces) > 0 else []

    def detect_eyes(self, gray_frame, face_roi):
        """Detect eyes within a face region."""
        x, y, w, h = face_roi
        roi_gray = gray_frame[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        return eyes

    def run(self, gray_frame):
        """
        Full Haar chain:
        1. Try frontal detection
        2. If nothing, try profile detection
        """
        faces = self.detect_faces(gray_frame)
        source = "frontal"

        if len(faces) == 0:
            faces = self.detect_profile_faces(gray_frame)
            source = "profile"

        return faces, source