import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------
# Paths / setup
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
MONKE_PATH = os.path.join(SCRIPT_DIR, "monke.jpg")

MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}

# --------------------------------------------------------------------------
# Tuning knobs - adjust these if detection feels too sensitive / not enough
# --------------------------------------------------------------------------
CAMERA_INDEX = 0            # change if you have multiple webcams
POSE_ON_FRAMES = 4          # consecutive "pose detected" frames before showing the image
POSE_OFF_FRAMES = 6         # consecutive "no pose" frames before hiding it again

# The "chin/mouth trigger zone" is defined relative to the detected face's
# bounding box. Widening/lowering it makes the gesture easier to trigger.
ZONE_WIDTH_RATIO = 0.9      # zone width as a fraction of face width
ZONE_TOP_RATIO = 0.45       # top edge of zone, as fraction down the face (0=top of head, 1=chin)
ZONE_BOTTOM_RATIO = 1.4     # bottom edge of zone (can extend below the chin)

OVERLAY_WIDTH = 220         # size (px) the monke image is shown at
FINGER_TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips


def ensure_models():
    """Download the MediaPipe model files on first run if they're missing."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}
    for filename, url in MODEL_URLS.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f"[setup] Downloading {filename} (one-time, ~10MB)...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"[setup] ERROR: failed to download {filename}: {e}")
                print(f"[setup] You can download it manually from:\n  {url}")
                print(f"[setup] and place it at:\n  {path}")
                sys.exit(1)
            print(f"[setup] Saved to {path}")
        paths[filename] = path
    return paths


def get_bbox(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def get_chin_zone(face_bbox):
    x0, y0, x1, y1 = face_bbox
    face_w = x1 - x0
    face_h = y1 - y0
    cx = (x0 + x1) / 2
    half_w = (face_w * ZONE_WIDTH_RATIO) / 2
    zone_top = y0 + face_h * ZONE_TOP_RATIO
    zone_bottom = y0 + face_h * ZONE_BOTTOM_RATIO
    return (cx - half_w, zone_top, cx + half_w, zone_bottom)


def point_in_zone(px, py, zone):
    x0, y0, x1, y1 = zone
    return x0 <= px <= x1 and y0 <= py <= y1


def overlay_image(frame, overlay, x, y):
    """Paste a BGR image onto frame at (x, y), clipped to the frame edges."""
    h, w = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    x2, y2 = min(x + ow, w), min(y + oh, h)
    ow_clip, oh_clip = x2 - x, y2 - y
    if ow_clip <= 0 or oh_clip <= 0:
        return
    frame[y:y2, x:x2] = overlay[:oh_clip, :ow_clip]


def main():
    if not os.path.exists(MONKE_PATH):
        print(f"ERROR: monke.jpg not found at {MONKE_PATH}")
        sys.exit(1)

    monke_img = cv2.imread(MONKE_PATH)
    if monke_img is None:
        print("ERROR: failed to load monke.jpg")
        sys.exit(1)

    scale = OVERLAY_WIDTH / monke_img.shape[1]
    overlay_h = int(monke_img.shape[0] * scale)
    monke_resized = cv2.resize(monke_img, (OVERLAY_WIDTH, overlay_h))

    model_paths = ensure_models()

    face_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_paths["hand_landmarker.task"]),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam (index {CAMERA_INDEX})")
        sys.exit(1)

    start_time = time.time()
    pose_on_counter = 0
    pose_off_counter = 0
    pose_active = False
    show_debug = False

    print("Running. Press 'q' to quit, 'd' to toggle debug view.")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.time() - start_time) * 1000)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            pose_detected_this_frame = False

            if face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]
                face_bbox = get_bbox(face_landmarks, w, h)
                zone = get_chin_zone(face_bbox)

                for hand_landmarks in hand_result.hand_landmarks:
                    for tip_id in FINGER_TIP_IDS:
                        lm = hand_landmarks[tip_id]
                        px, py = lm.x * w, lm.y * h
                        if point_in_zone(px, py, zone):
                            pose_detected_this_frame = True
                            break
                    if pose_detected_this_frame:
                        break

                if show_debug:
                    fx0, fy0, fx1, fy1 = map(int, face_bbox)
                    cv2.rectangle(frame, (fx0, fy0), (fx1, fy1), (255, 200, 0), 1)
                    zx0, zy0, zx1, zy1 = map(int, zone)
                    zone_color = (0, 255, 0) if pose_detected_this_frame else (0, 255, 255)
                    cv2.rectangle(frame, (zx0, zy0), (zx1, zy1), zone_color, 2)

            if show_debug:
                for hand_landmarks in hand_result.hand_landmarks:
                    for tip_id in FINGER_TIP_IDS:
                        lm = hand_landmarks[tip_id]
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

            # Debounce so the overlay doesn't flicker on noisy single frames
            if pose_detected_this_frame:
                pose_on_counter += 1
                pose_off_counter = 0
            else:
                pose_off_counter += 1
                pose_on_counter = 0

            if not pose_active and pose_on_counter >= POSE_ON_FRAMES:
                pose_active = True
            elif pose_active and pose_off_counter >= POSE_OFF_FRAMES:
                pose_active = False

            if pose_active:
                margin = 20
                overlay_image(frame, monke_resized, w - OVERLAY_WIDTH - margin, margin)
                cv2.putText(frame, "ERRRRRMMMM...", (margin, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, "q: quit   d: debug view", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Monke Pose Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_debug = not show_debug
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_landmarker.close()
        hand_landmarker.close()


if __name__ == "__main__":
    main()
