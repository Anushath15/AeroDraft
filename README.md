# ✈️ AeroDraft
### AI-Powered Mid-Air CAD Sketching using Computer Vision

> Draw in the air using your hand. AeroDraft transforms natural hand gestures into precise 3D engineering sketches using Computer Vision, Gesture Recognition, and Real-Time Geometry Processing.

---

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-orange)
![License](https://img.shields.io/badge/License-MIT-red)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)

</p>

---

# Overview

AeroDraft is an AI-assisted Human-Computer Interaction (HCI) system that allows engineers and designers to create virtual CAD sketches using only hand gestures in front of a webcam.

Instead of relying on traditional mouse-and-keyboard interaction, AeroDraft interprets hand landmarks, classifies gestures, estimates depth, smooths motion, and renders engineering geometry in real time.

The project combines:

- Computer Vision
- Gesture Recognition
- Real-Time Geometry
- Human Computer Interaction
- Signal Processing
- Engineering Graphics

---

# Key Features

- Real-time hand tracking
- Gesture recognition
- Motion smoothing
- Relative depth estimation
- Perspective projection
- Wireframe CAD rendering
- Modular architecture
- Unit-tested core mathematics
- Extensible gesture pipeline

---

# Current Development Status

| Module | Status |
|---------|--------|
| Camera Pipeline | ✅ Complete |
| MediaPipe Tracking | ✅ Complete |
| Landmark Detection | ✅ Complete |
| One Euro Filter | ✅ Complete |
| Relative Depth Estimation | ✅ Complete |
| Perspective Projection | ✅ Complete |
| Wireframe Renderer | ✅ Complete |
| Gesture Classifier | ✅ Complete |
| Gesture State Machine | ✅ Complete |
| Coordinate Filtering | ✅ Complete |
| Pipeline Integration | ✅ Complete |
| HUD Overlay | 🚧 In Progress |
| Object Manipulation | 🚧 Planned |
| Air Sketch Mode | 🚧 Planned |
| CAD Export | 🚧 Planned |

---

# Working Architecture

```text
                     Webcam
                        │
                        ▼
               OpenCV Video Capture
                        │
                        ▼
             MediaPipe Hand Landmarker
                        │
                        ▼
              21 Hand Landmarks (3D)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 Gesture Classifier            Coordinate Filter
        │                               │
        ▼                               ▼
 Gesture State Machine        One Euro Filter
        │                               │
        └───────────────┬───────────────┘
                        ▼
               Relative Depth Estimator
                        │
                        ▼
              3D Hand Position Estimation
                        │
                        ▼
             Perspective Projection Engine
                        │
                        ▼
             Wireframe CAD Renderer
                        │
                        ▼
                 OpenCV Display Window
```

---

# Repository Structure

```
AeroDraft/
│
├── camera.py
├── config.py
├── hand_tracker.py
├── main.py
├── requirements.txt
│
├── core/
│   ├── coordinate_filter.py
│   ├── depth_estimator.py
│   └── one_euro_filter.py
│
├── engine/
│   ├── projection.py
│   └── wireframe_renderer.py
│
├── gestures/
│   ├── gesture_classifier.py
│   └── state_machine.py
│
├── ui/
│   └── hud_renderer.py
│
├── tests/
│
└── README.md
```

---

# Processing Pipeline

## Step 1 — Video Capture

The webcam continuously captures RGB frames using OpenCV.

↓

## Step 2 — Hand Tracking

MediaPipe Tasks detects:

- 21 landmarks
- Hand confidence
- Handedness

↓

## Step 3 — Gesture Recognition

The Gesture Classifier recognizes:

- Neutral
- Pinch
- Pointing
- Open Palm

↓

## Step 4 — State Machine

The State Machine converts raw frame-level gestures into stable interaction states.

Example:

```
Neutral
   │
Pinch Start
   │
Dragging
   │
Pinch Release
   │
Idle
```

↓

## Step 5 — Coordinate Filtering

Hand coordinates are smoothed using:

- One Euro Filter
- Temporal Filtering

↓

## Step 6 — Depth Estimation

Relative Pixel Scaling estimates approximate hand depth from camera.

↓

## Step 7 — Projection

The Projection Engine converts

```
3D World Coordinates

↓

2D Screen Coordinates
```

using the pinhole camera model.

↓

## Step 8 — Rendering

The Renderer draws

- Cuboids
- Engineering wireframes
- Reference axes
- Interactive objects

using OpenCV.

---

# Gesture Recognition

| Gesture | Purpose |
|----------|----------|
| Neutral | Idle |
| Pinch | Select / Grab |
| Pointing | Cursor |
| Open Palm | Reset / Cancel |

---

# Technologies Used

### Computer Vision

- OpenCV

### AI / Vision

- MediaPipe Tasks API

### Mathematics

- Linear Algebra
- Vector Geometry
- Euclidean Distance
- Perspective Projection

### Signal Processing

- One Euro Filter

### Programming

- Python

### Testing

- PyTest

---

# Installation

Clone the repository

```bash
git clone https://github.com/Anushath15/AeroDraft.git

cd AeroDraft
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# MediaPipe Model

Download the MediaPipe Hand Landmarker model.

Place it in the project root as

```
hand_landmarker.task
```

The directory should look like

```
AeroDraft/

hand_landmarker.task
main.py
camera.py
config.py
...
```

> **Note:** `hand_landmarker.task` is intentionally excluded from Git because it is a large third-party model file.

---

# Running

```bash
python main.py
```

---

# Running Tests

```bash
pytest
```

---

# Engineering Principles

The project follows

- Modular Design
- Separation of Concerns
- Dependency Isolation
- Immutable Configuration
- Test-Driven Development
- Clean Architecture

Each module has a single responsibility and can be independently tested.

---

# Future Roadmap

### Phase 1
- ✅ Hand Tracking

### Phase 2
- ✅ Gesture Recognition

### Phase 3
- ✅ Spatial Coordinate Processing

### Phase 4
- 🚧 Interactive Object Placement

### Phase 5
- 🚧 Mid-Air Sketching

### Phase 6
- 🚧 Engineering Constraints

### Phase 7
- 🚧 CAD File Export

### Phase 8
- 🚧 Multi-Hand Collaboration

### Phase 9
- 🚧 AI-Assisted Sketch Completion

---

# Contributing

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

# License

This project is released under the MIT License.

---

# Author

**Anushath S**

Computer Science & Engineering

AI / Machine Learning Enthusiast

GitHub:
https://github.com/Anushath15

---

## Vision

> **"AeroDraft aims to redefine engineering design by enabling intuitive, touch-free 3D sketching through AI-powered hand interaction—bridging the gap between human creativity and digital CAD systems."**