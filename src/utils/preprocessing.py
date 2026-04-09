import os
DB_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "face_database.json")
import cv2
import numpy as np

def resize_frame(frame, width=640):
    """Resize frame while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    ratio = width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h))

def equalize_histogram(gray_frame):
    """Improve contrast for better Haar detection in low light."""
    return cv2.equalizeHist(gray_frame)

def denoise(frame):
    """Remove noise from frame."""
    return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

def crop_face(frame, x, y, w, h, padding=20):
    """
    Crop a face from the frame with optional padding.
    Clamps to frame boundaries.
    """
    img_h, img_w = frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    return frame[y1:y2, x1:x2]

def normalize_face(face_crop, size=(224, 224)):
    """Resize and normalize face for model input."""
    resized = cv2.resize(face_crop, size)
    normalized = resized / 255.0
    return normalized

def align_face(frame, landmarks):
    """
    Basic face alignment using eye landmarks.
    landmarks: dict with 'left_eye' and 'right_eye' (x, y) tuples
    """
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")

    if left_eye is None or right_eye is None:
        return frame

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2
    )

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return aligned

def draw_face_box(frame, x, y, w, h, label="", color=(0, 255, 80), thickness=2):
    """Draw a labeled bounding box on the frame."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x, y - text_h - 10),
            (x + text_w + 5, y),
            color, -1
        )
        cv2.putText(
            frame, label,
            (x + 2, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 2
        )
    return frame