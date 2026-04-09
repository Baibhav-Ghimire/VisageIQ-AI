import os
DB_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "face_database.json")
import cv2
import numpy as np

class SkinSegmenter:
    """
    Skin segmentation using HSV and YCrCb color spaces.
    Inspired by Stanford skin detection logic.
    Useful for filtering false positives in face detection.
    """

    def __init__(self):
        # HSV skin range
        self.hsv_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.hsv_upper = np.array([20, 255, 255], dtype=np.uint8)

        # YCrCb skin range
        self.ycrcb_lower = np.array([0, 135, 85], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, 180, 135], dtype=np.uint8)

    def segment_hsv(self, frame):
        """Detect skin pixels using HSV color space."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return mask

    def segment_ycrcb(self, frame):
        """Detect skin pixels using YCrCb color space (more robust)."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
        return mask

    def combined_mask(self, frame):
        """
        Combine both masks for higher accuracy.
        A pixel is 'skin' only if both spaces agree.
        """
        hsv_mask = self.segment_hsv(frame)
        ycrcb_mask = self.segment_ycrcb(frame)
        combined = cv2.bitwise_and(hsv_mask, ycrcb_mask)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return combined

    def skin_ratio(self, frame, face_box):
        """
        Calculate what % of a detected face region is actually skin.
        Used to validate if a detection is a real face.
        Returns float 0.0 - 1.0
        """
        x, y, w, h = face_box
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            return 0.0
        mask = self.combined_mask(face_crop)
        skin_pixels = np.count_nonzero(mask)
        total_pixels = w * h
        return skin_pixels / total_pixels

    def is_valid_face(self, frame, face_box, threshold=0.15):
        """
        Returns True if the face box contains enough skin pixels.
        Filters out false positives (e.g., objects, shadows).
        """
        ratio = self.skin_ratio(frame, face_box)
        return ratio >= threshold

    def apply_mask(self, frame):
        """Return frame with only skin pixels visible."""
        mask = self.combined_mask(frame)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result