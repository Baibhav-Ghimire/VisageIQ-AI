import cv2
import face_recognition
import numpy as np
import pickle
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("visageiq.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VisageIQ_Engine:
    DB_PATH = "models/encodings.pkl"

    def __init__(self):
        # Classical detector
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.known_encodings = []
        self.known_names = []
        self._load_db()

    # ── DATABASE ────────────────────────────────────────────────

    def _load_db(self):
        """Load saved face encodings from disk."""
        if os.path.exists(self.DB_PATH):
            try:
                with open(self.DB_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get("encodings", [])
                    self.known_names = data.get("names", [])
                logger.info(f"Loaded {len(self.known_names)} known face(s) from DB.")
            except Exception as e:
                logger.error(f"Failed to load DB: {e}")
                self.known_encodings = []
                self.known_names = []
        else:
            logger.info("No existing database found. Starting fresh.")

    def _save_db(self):
        """Persist face encodings to disk."""
        try:
            os.makedirs("models", exist_ok=True)
            with open(self.DB_PATH, "wb") as f:
                pickle.dump({
                    "encodings": self.known_encodings,
                    "names": self.known_names
                }, f)
            logger.info("Database saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save DB: {e}")

    # ── DETECTION ───────────────────────────────────────────────

    def detect_haar(self, gray_frame):
        """Fast Haar Cascade detection. Returns list of (x, y, w, h)."""
        faces = self.haar_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces) == 0:
            return []
        return [tuple(f) for f in faces]

    def detect_cnn(self, rgb_frame):
        """HOG/CNN fallback detection. Returns list of (x, y, w, h)."""
        locations = face_recognition.face_locations(rgb_frame, model="hog")
        converted = []
        for (top, right, bottom, left) in locations:
            w = right - left
            h = bottom - top
            converted.append((left, top, w, h))
        return converted

    def process_frame(self, frame):
        """
        Chain logic:
        1. Try Haar (fast)
        2. Fall back to HOG/CNN if Haar finds nothing
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.detect_haar(gray_frame)

            if len(faces) == 0:
                logger.warning("Haar found nothing. Switching to CNN fallback...")
                faces = self.detect_cnn(rgb_frame)
            else:
                logger.info(f"Haar detected {len(faces)} face(s).")

            return faces, rgb_frame

        except Exception as e:
            logger.error(f"process_frame failed: {e}")
            return [], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── EMBEDDINGS & IDENTITY ───────────────────────────────────

    def get_embeddings(self, rgb_frame, xywh_faces):
        """Extract 128-d face embeddings."""
        try:
            locations = []
            for (x, y, w, h) in xywh_faces:
                locations.append((y, x + w, y + h, x))
            encodings = face_recognition.face_encodings(rgb_frame, locations)
            return encodings
        except Exception as e:
            logger.error(f"get_embeddings failed: {e}")
            return []

    def identify_face(self, encoding, tolerance=0.6):
        """Compare encoding against known database."""
        if encoding is None or not self.known_encodings:
            return "Unknown"
        try:
            matches = face_recognition.compare_faces(
                self.known_encodings, encoding, tolerance=tolerance
            )
            if True in matches:
                idx = matches.index(True)
                return self.known_names[idx]
        except Exception as e:
            logger.error(f"identify_face failed: {e}")
        return "Unknown"

    def register_face(self, rgb_frame, name):
        """Add a new person to the attendance database."""
        # Check for duplicate
        if name.strip().lower() in [n.lower() for n in self.known_names]:
            logger.warning(f"{name} is already registered.")
            return "duplicate"

        try:
            locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, locations)
            if encodings:
                self.known_encodings.append(encodings[0])
                self.known_names.append(name.strip())
                self._save_db()
                logger.info(f"Registered: {name}")
                return True
            logger.warning("No face found during registration.")
            return False
        except Exception as e:
            logger.error(f"register_face failed: {e}")
            return False

    def get_registered_names(self):
        """Return list of all registered names."""
        return self.known_names.copy()

    # ── MOOD ────────────────────────────────────────────────────

    def analyze_mood(self, frame, xywh_faces):
        """
        Mood placeholder — returns Neutral for all faces.
        Replace with DeepFace in Phase 5.
        """
        return ["Neutral 😐"] * len(xywh_faces)