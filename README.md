# AeroDraft

Edge AI spatial visualization software for Indian MSME electrical hardware
and lighting retailers. Uses only an ordinary webcam — no LiDAR, no
ARCore, no SLAM, no Unity, no Unreal, no OpenGL.

## Current Status: Phase 1 — Foundational Pipeline

This repository currently implements **Phase 1 only**: camera capture and
MediaPipe-based hand tracking. It intentionally does **not** yet include
spatial mapping, projection, product rendering, or any of the other
core algorithms referenced in the project's design documentation
(Adaptive Spatial Mapping Engine, Relative Pixel Scaling, Lux Check,
Projection Logic, Anchor Logic). Those are future phases and are out of
scope for this code.

### What Phase 1 does

1. Opens a webcam via OpenCV.
2. Runs MediaPipe Hands on each frame to detect hand landmarks.
3. Draws the detected landmarks on screen so the pipeline can be
   visually verified end to end.

That's it. No spatial reasoning, no object placement, no filtering
pipeline beyond what MediaPipe does internally.
## Phase 2: Integrated Pipeline
This phase integrates the core ASME and rendering engines into the main loop:
1. **Hand Tracking**: Extracts wrist coordinates.
2. **DepthEstimation**: Computes the Z-coordinate proxy using the ASME heuristic (relative pixel scaling).
3. **Filtering**: Applies the 1-Euro Filter to stabilize the Z value.
4. **Projection**: Maps the 3D cuboid coordinates into 2D screen space using perspective projection.
5. **Rendering**: Draws the resulting wireframe on the live webcam feed.

## Project Structure

```
AeroDraft/
├── aerodraft/
│   ├── main.py                # Entry point: wires camera + hand tracker together
│   ├── config/
│   │   └── settings.py        # Environment-driven configuration (no hardcoded values)
│   ├── core/
│   │   ├── camera.py          # CameraCapture: webcam lifecycle only
│   │   ├── hand_tracker.py    # HandTracker: MediaPipe Hands wrapper
│   │   ├── data_types.py      # Shared dataclasses (HandDetectionResult, etc.)
│   │   ├── visualization.py   # Landmark drawing for visual verification
│   │   └── exceptions.py      # AeroDraft-specific exception hierarchy
│   └── utils/
│       └── logger.py          # Centralized Loguru configuration
├── tests/                     # Unit tests (pytest)
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

## Running

```bash
python -m aerodraft.main
```

A window will open showing the webcam feed with hand landmarks overlaid.
Press `q` to quit.

### Configuration

All configuration is via environment variables (see
`aerodraft/config/settings.py` for the full list and defaults). Examples:

| Variable | Default | Purpose |
|---|---|---|
| `AERODRAFT_CAMERA_INDEX` | `0` | Which camera device to open |
| `AERODRAFT_FRAME_WIDTH` | `1280` | Requested capture width |
| `AERODRAFT_FRAME_HEIGHT` | `720` | Requested capture height |
| `AERODRAFT_MAX_NUM_HANDS` | `2` | Max hands MediaPipe will track |
| `AERODRAFT_MIN_DETECTION_CONFIDENCE` | `0.5` | MediaPipe detection threshold |
| `AERODRAFT_LOG_DIR` | `logs` | Where log files are written (relative path) |

No paths or device settings are hardcoded in the application code itself.

## Testing

```bash
pytest
```

Camera tests mock `cv2.VideoCapture` and run without a physical webcam.
Hand tracker tests run real MediaPipe inference against deterministic
blank frames (no camera or real hand required).

## Known Limitations (Phase 1)

- `mediapipe` is pinned to `<0.10.15` because this module uses the legacy
  `mediapipe.solutions.hands` API. MediaPipe removed that API in later
  0.10.2x+ releases in favor of the new Tasks API
  (`mediapipe.tasks.python.vision.HandLandmarker`), which requires
  downloading a separate `.task` model file at runtime. Migrating to the
  Tasks API is a reasonable future change but is out of scope for this
  phase; pinning keeps the current code working with `pip install`.
- No spatial mapping, depth estimation, or projection — this phase only
  proves that camera capture + hand detection work together reliably.
- No lighting-adequacy check ("Lux Check") is implemented yet.
- No gesture-to-action mapping is implemented yet; landmarks are detected
  and drawn, but not interpreted as commands.
- Performance (FPS/CPU/memory) on "legacy showroom computers" has not
  been benchmarked in this phase.