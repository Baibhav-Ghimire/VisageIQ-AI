"""
face_register.py  —  Register a new face and save encoding to database.json
Usage:  python face_register.py --name "Alice"
        python face_register.py --name "Alice" --image alice.jpg   (use image file instead of webcam)
"""


import cv2
import face_recognition
import json
import os
DB_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "face_database.json")
import argparse
import numpy as np
from datetime import datetime

DB_FILE = "face_database.json"


# ── Database helpers ──────────────────────────────────────────────────────────

def load_database() -> dict:
    """Load existing face database, or create a fresh one."""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {"people": []}


def save_database(db: dict) -> None:
    """Atomically write database to disk (write → rename so no corruption)."""
    tmp = DB_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp, DB_FILE)          # atomic on all modern OS
    print(f"[✓] Database saved → {DB_FILE}")


# ── Face capture ──────────────────────────────────────────────────────────────

def capture_from_webcam() -> np.ndarray | None:
    """Open webcam, wait for a keypress, return the frame (BGR)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[✗] Could not open webcam.")
        return None

    print("  Webcam open. Press SPACE to capture, Q to quit.")
    frame = None
    while True:
        ret, img = cap.read()
        if not ret:
            print("[✗] Failed to read from webcam.")
            break
        cv2.imshow("Register face — SPACE to capture", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frame = img
            break
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


def load_from_file(path: str) -> np.ndarray | None:
    """Load an image file and return it in BGR format."""
    img = cv2.imread(path)
    if img is None:
        print(f"[✗] Could not read image file: {path}")
    return img


# ── Registration ──────────────────────────────────────────────────────────────

def register_face(name: str, image_path: str | None = None) -> bool:
    """
    Capture or load a frame, extract one face encoding, and store it.
    Returns True on success.
    """
    print(f"\n── Registering '{name}' ──")

    # 1. Get the image (BGR from OpenCV)
    if image_path:
        bgr_frame = load_from_file(image_path)
    else:
        bgr_frame = capture_from_webcam()

    if bgr_frame is None:
        return False

    # ── KEY FIX: convert BGR → RGB before passing to face_recognition ──
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # 2. Detect face locations
    locations = face_recognition.face_locations(rgb_frame, model="hog")
    if not locations:
        print("[✗] No face detected. Tips:")
        print("    • Make sure your face is well-lit and centred.")
        print("    • Avoid strong backlight.")
        print("    • Try moving closer to the camera.")
        return False

    if len(locations) > 1:
        print(f"[!] {len(locations)} faces detected — using the first one only.")

    # 3. Compute 128-d encoding for the first face
    encodings = face_recognition.face_encodings(rgb_frame, [locations[0]])
    if not encodings:
        print("[✗] Could not compute face encoding.")
        return False

    encoding = encodings[0]

    # 4. Load database, add entry, save
    db = load_database()

    # Check for duplicates
    existing_names = [p["name"] for p in db["people"]]
    if name in existing_names:
        print(f"[!] '{name}' already exists. Updating their encoding.")
        db["people"] = [p for p in db["people"] if p["name"] != name]

    db["people"].append({
        "name": name,
        "encoding": encoding.tolist(),   # numpy array → plain list for JSON
        "registered_at": datetime.now().isoformat()
    })

    save_database(db)
    print(f"[✓] '{name}' registered successfully! ({len(db['people'])} total in DB)")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a face into face_database.json")
    parser.add_argument("--name",  required=True, help="Person's name")
    parser.add_argument("--image", default=None,  help="Optional path to an image file")
    args = parser.parse_args()

    success = register_face(args.name, args.image)
    raise SystemExit(0 if success else 1)