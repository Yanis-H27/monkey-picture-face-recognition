# Monke Pose Detector 🐒

Opens your webcam, watches your face and hand, and pops up the
"ERRRRRMMMM..." monke image whenever you do his thinking pose
(hand near your chin/mouth). It disappears again when you stop.

## Setup

1. Make sure you have Python 3.9+ installed.
2. In this folder, install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run it:

   ```
   python monke_pose.py
   ```

On the very first run, it'll automatically download two small MediaPipe
model files (~10MB total) into a `models/` folder — you need an internet
connection for that one time only. After that it works fully offline.

## Controls

- **q** — quit
- **d** — toggle debug view (shows the face box, the trigger zone, and
  your tracked fingertips, so you can see exactly why it is/isn't
  triggering)

## How it works

Each frame, MediaPipe detects your face and hand landmarks. A "trigger
zone" is drawn around your mouth/chin area (sized relative to your face).
If any fingertip lands inside that zone for a few consecutive frames, the
monke image appears in the corner. When your hand leaves the zone for a
few frames, it disappears. The few-frame delay (instead of an instant
on/off) is just to stop it flickering on momentary noise.

## Tuning

If it's not triggering easily enough, or triggers too easily, edit the
constants near the top of `monke_pose.py`:

- `ZONE_WIDTH_RATIO`, `ZONE_TOP_RATIO`, `ZONE_BOTTOM_RATIO` — control the
  size/position of the trigger zone relative to your detected face.
  Bigger numbers = easier to trigger.
- `POSE_ON_FRAMES` / `POSE_OFF_FRAMES` — how many consecutive frames are
  needed to turn the overlay on/off. Lower = more responsive but more
  flicker-prone.
- `OVERLAY_WIDTH` — size of the monke image on screen.
- `CAMERA_INDEX` — change if you have more than one webcam and the
  wrong one opens.

Press `d` while running to see the live debug view — it makes tuning a
lot easier since you can watch the zone and your fingertips directly.

## Troubleshooting

- **Webcam doesn't open**: try changing `CAMERA_INDEX` to `1` (or `2`) in
  `monke_pose.py`.
- **Model download fails**: the script will print the direct URLs — you
  can download them manually and place them in the `models/` folder it
  creates next to the script.
- **`mediapipe` import errors / "module has no attribute 'solutions'"**:
  that error comes from older tutorials using MediaPipe's legacy API,
  which Google removed in late-2025 releases. This script avoids that
  API entirely (it uses the current Tasks API), so a normal
  `pip install -r requirements.txt` should just work.
