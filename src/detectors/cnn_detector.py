import face_recognition
import numpy as np
from deepface import DeepFace

class CNNDetector:
    def __init__(self):
        self.model = "hog"  # "cnn" for GPU machines

    def detect_faces(self, rgb_frame):
        """
        HOG-based face detection via face_recognition.
        Returns list of (x, y, w, h).
        """
        locations = face_recognition.face_locations(rgb_frame, model=self.model)
        converted = []
        for (top, right, bottom, left) in locations:
            w = right - left
            h = bottom - top
            converted.append((left, top, w, h))
        return converted

    def analyze_emotion(self, bgr_frame, xywh_faces):
        """
        DeepFace emotion analysis per face.
        Returns list of dicts with emotion data.
        """
        results = []
        for (x, y, w, h) in xywh_faces:
            crop = bgr_frame[y:y+h, x:x+w]
            if crop.size == 0:
                results.append({"dominant": "Unknown", "scores": {}})
                continue
            try:
                analysis = DeepFace.analyze(
                    crop,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )
                dominant = analysis[0]["dominant_emotion"]
                scores = analysis[0]["emotion"]
                results.append({"dominant": dominant, "scores": scores})
            except Exception as e:
                results.append({"dominant": "Unknown", "scores": {}})
        return results

    def analyze_demographics(self, bgr_frame, xywh_faces):
        """
        DeepFace age, gender, race analysis per face.
        """
        results = []
        for (x, y, w, h) in xywh_faces:
            crop = bgr_frame[y:y+h, x:x+w]
            if crop.size == 0:
                results.append({})
                continue
            try:
                analysis = DeepFace.analyze(
                    crop,
                    actions=["age", "gender", "race"],
                    enforce_detection=False,
                    silent=True
                )
                results.append({
                    "age": analysis[0]["age"],
                    "gender": analysis[0]["dominant_gender"],
                    "race": analysis[0]["dominant_race"]
                })
            except Exception:
                results.append({})
        return results