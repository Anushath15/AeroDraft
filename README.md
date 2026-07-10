# ✈️ AeroDraft

### AI-Powered Spatial Product Visualization for MSMEs

> *Preview electrical hardware in your space before installation — using only a webcam and hand gestures.*

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-FF6F00)
![PyTest](https://img.shields.io/badge/Tests-132%20Passing-success)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue)
![License](https://img.shields.io/badge/License-MIT-red)
![Status](https://img.shields.io/badge/Status-Phase%2012%20Complete-success)

</p>

<p align="center">
<b>Real-Time Edge AI • Computer Vision • Human-Computer Interaction • 3D Spatial Visualization</b>
</p>

---

## 📖 Overview

AeroDraft is an **Edge AI-based Human-Computer Interaction (HCI)** system that enables users to interact with virtual 3D objects using **natural hand gestures captured by a standard webcam**.

Unlike traditional Augmented Reality systems, AeroDraft **does not require LiDAR, depth cameras, ARCore, SLAM, Unity, Unreal Engine, or OpenGL**. Instead, it combines modern Computer Vision techniques with geometric reasoning to estimate spatial information from a monocular RGB camera.

The project demonstrates how **AI-powered perception**, **gesture recognition**, **signal processing**, and **real-time rendering** can work together to create an intuitive touch-free interaction system suitable for engineering visualization and **MSME product demonstrations**.

---

## 🏢 Business Applications

AeroDraft is designed for real-world MSME demonstrations:

| Industry | Use Case |
|----------|----------|
| **Electrical Hardware Shops** | Preview switchboards, distribution boards, and junction boxes on actual walls before purchase |
| **Lighting Retailers** | Show customers exactly how LED ceiling lights will look in their rooms |
| **Home Renovation** | Plan electrical layouts with accurate spatial placement before drilling or wiring |
| **Interior Visualization** | Validate positioning, clearances, and aesthetics in real customer space |
| **Product Placement** | Demonstrate cable management with PVC conduit boxes |
| **Customer Demonstrations** | Interactive AR-style previews that build buyer confidence and reduce returns |

---

## 🎯 Vision

> **"To make spatial interaction accessible on any computer using only AI and Computer Vision, eliminating the need for expensive AR hardware."**

> **"See it before you install it."**

---

## 🚀 Key Features

- 📷 Real-time webcam processing using OpenCV
- ✋ AI-based hand tracking with MediaPipe Tasks API
- 🤏 Gesture recognition using geometric heuristics
- 📐 Relative depth estimation from hand geometry
- 🎯 One Euro Filter for smooth spatial tracking
- 📦 **Interactive 3D wireframe rendering with 7 electrical products**
- 🏷️ **Product catalog with business metadata (names, categories, dimensions)**
- 🧠 Gesture-driven finite state machine with **visual state feedback**
- 🔔 **Transient notification banners** (Object Placed, Hand Lost, Tracking Restored)
- 📊 **Professional HUD** displaying FPS, gesture, state, depth, and product category
- 🖥️ **Optional Demo Mode** with on-screen help panel for judges and visitors
- ⚡ Optional performance benchmarking
- 🧪 **Comprehensive automated test suite (132 tests)**
- 🏗️ Modular and extensible architecture
- 💻 Runs entirely on CPU without dedicated graphics hardware

---

# 📌 Current Project Status

## Phase 12 — MSME Demo Experience ✅

AeroDraft has evolved from a technology demonstration into a **polished MSME product visualization tool**.

| Component | Status |
|------------|--------|
| Camera Pipeline | ✅ Complete |
| MediaPipe Hand Tracking | ✅ Complete |
| Landmark Detection | ✅ Complete |
| Gesture Classification | ✅ Complete |
| Relative Depth Estimation | ✅ Complete |
| One Euro Coordinate Filter | ✅ Complete |
| Coordinate Smoothing | ✅ Complete |
| Gesture State Machine | ✅ Complete |
| Perspective Projection | ✅ Complete |
| **Object Renderer (7 Products)** | ✅ Complete |
| **Product Catalog** | ✅ Complete |
| **Professional HUD with Notifications** | ✅ Complete |
| **Demo Help Panel** | ✅ Complete |
| Performance Benchmarking | ✅ Complete |
| Configuration System | ✅ Complete |
| Integration Pipeline | ✅ Complete |
| Automated Testing | ✅ Complete (132 tests) |

---

## ✅ Verified Results

- ✔ **132 automated tests passing**
- ✔ End-to-end pipeline integrated
- ✔ Stable webcam operation
- ✔ Real-time rendering with state-based colors
- ✔ Clean application shutdown
- ✔ Modular architecture
- ✔ Production-ready code structure
- ✔ Comprehensive documentation

---

## Current Processing Pipeline

1. Capture RGB frames from webcam
2. Detect hand landmarks using MediaPipe (21 landmarks)
3. Estimate relative hand depth
4. Smooth spatial coordinates with One Euro Filter
5. Classify user gestures (Pinch / Fist / Open Palm)
6. Update interaction state machine (IDLE → DRAWING → PLACED → LOCKED)
7. Project 3D product geometry into image space
8. Render specialized wireframe object with state-based colors
9. Display professional HUD with telemetry and notifications
10. Repeat for every frame

---

# 🏗️ System Architecture

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
        ┌───────────────┼───────────────────────────┐
        ▼               ▼                           ▼
 Gesture Classifier  Depth Estimator      Coordinate Filter
        │               │                           │
        ▼               ▼                           ▼
 Gesture Type      Relative Depth             Smoothed (X,Y,Z)
        │               │                           │
        └───────────────┴───────────────────────────┘
                        │
                        ▼
             Gesture State Machine
                        │
                        ▼
            Box State & Spatial Parameters
                        │
                        ▼
          Perspective Projection Engine
                        │
                        ▼
          Object Renderer (Product Catalog)
                        │
                        ▼
             HUD Renderer + Display
```

---

## Core Pipeline

```text
Camera
    │
    ▼
Hand Tracking
    │
    ▼
Landmarks
    │
    ▼
Gesture Recognition
    │
    ▼
Depth Estimation
    │
    ▼
Coordinate Filtering
    │
    ▼
State Machine
    │
    ▼
3D Projection
    │
    ▼
Object Rendering
    │
    ▼
HUD Overlay
    │
    ▼
Display Window
```

---

## Technologies Used

### Computer Vision

- OpenCV
- MediaPipe Tasks API

### Artificial Intelligence

- Hand Landmark Detection
- Gesture Classification
- Spatial Interaction

### Mathematics

- Linear Algebra
- Euclidean Geometry
- Perspective Projection
- Relative Depth Estimation

### Signal Processing

- One Euro Filter
- Temporal Coordinate Smoothing

### Software Engineering

- Python
- PyTest
- Frozen Dataclasses
- Modular Architecture

# 📂 Repository Structure

```text
AeroDraft/
│
├── main.py                      # Application entry point
├── camera.py                    # OpenCV video stream manager
├── hand_tracker.py              # MediaPipe Tasks API wrapper
├── config.py                    # Immutable configuration dataclasses
│
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development dependencies
├── pyproject.toml               # Package metadata
├── README.md                    # Project documentation
│
├── core/
│   ├── __init__.py
│   ├── benchmark.py             # Performance instrumentation
│   ├── coordinate_filter.py     # 3D coordinate smoothing
│   ├── depth_estimator.py       # Relative depth estimation
│   └── one_euro_filter.py       # One Euro signal filter
│
├── engine/
│   ├── __init__.py
│   ├── catalog.py               # Product catalog (Phase 12)
│   ├── object_renderer.py       # Specialized wireframe renderers (Phase 12)
│   ├── projection.py            # Perspective projection engine
│   └── wireframe_renderer.py    # Base wireframe renderer
│
├── gestures/
│   ├── __init__.py
│   ├── gesture_classifier.py    # Gesture recognition logic
│   └── state_machine.py         # Gesture-driven state machine
│
├── ui/
│   ├── __init__.py
│   └── hud_renderer.py          # Professional HUD with notifications (Phase 12)
│
└── tests/
    ├── test_benchmark.py
    ├── test_camera.py
    ├── test_catalog.py            # Product catalog tests (Phase 12)
    ├── test_coordinate_filter.py
    ├── test_depth_estimator.py
    ├── test_gesture_classifier.py
    ├── test_hand_tracker.py
    ├── test_hud_renderer.py       # HUD + notification tests (Phase 12)
    ├── test_object_renderer.py    # Object renderer tests (Phase 12)
    ├── test_one_euro_filter.py
    ├── test_pipeline.py
    ├── test_projection.py
    ├── test_state_machine.py
    └── test_wireframe_renderer.py
```

---

## 📦 Module Responsibilities

| Module | Responsibility |
|---------|----------------|
| `camera.py` | Webcam acquisition and frame capture |
| `hand_tracker.py` | MediaPipe hand landmark detection |
| `gesture_classifier.py` | Converts landmarks into gestures |
| `depth_estimator.py` | Estimates relative hand depth |
| `coordinate_filter.py` | Smooths noisy hand coordinates |
| `state_machine.py` | Maintains interaction states |
| `projection.py` | Projects 3D coordinates into 2D image space |
| `object_renderer.py` | Routes object types to specialized wireframe renderers |
| `catalog.py` | Business metadata registry for all products |
| `hud_renderer.py` | Displays runtime information, notifications, and demo panel |
| `benchmark.py` | Performance monitoring |
| `config.py` | Centralized immutable configuration |

---

## 🧩 Design Principles

AeroDraft follows modern software engineering practices:

- ✅ Modular architecture
- ✅ Single Responsibility Principle (SRP)
- ✅ Immutable configuration
- ✅ Dependency isolation
- ✅ Test-driven development
- ✅ Real-time processing
- ✅ Extensible design
- ✅ Separation of concerns

Each module can be independently tested, maintained, and extended.

---

# 🏷️ Product Catalog

AeroDraft includes **7 industry-standard products** rendered with OpenCV wireframes:

| # | Product | Category | Dimensions (W×H×D) |
|---|---------|----------|-------------------|
| 1 | Wireframe Cube | Demo | 20 × 20 × 20 cm |
| 2 | **Electrical Switchboard** | Electrical | 30 × 30 × 10 cm |
| 3 | **Wall Socket** | Electrical | 8 × 8 × 5 cm |
| 4 | **LED Ceiling Light** | Lighting | 25 × 25 × 5 cm |
| 5 | **Junction Box** | Electrical | 10 × 10 × 5 cm |
| 6 | **PVC Conduit Box** | Conduit | 15 × 15 × 10 cm |
| 7 | **Distribution Board** | Electrical | 40 × 50 × 15 cm |

Each product renders with **category-appropriate detail**:
- **Distribution Board** — breaker row lines on the front face
- **Switchboard** — door seam rectangle on the front face
- **LED Ceiling Light** — cross-pattern panel lines
- **Wall Socket** — vertical center line for outlet holes
- **Junction Box** — lid seam line
- **PVC Conduit Box** — conduit entry point dots on edges

No external 3D models are required. All rendering uses OpenCV primitives.

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Anushath15/AeroDraft.git
cd AeroDraft
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

For development:

```bash
pip install -r requirements-dev.txt
```

Or install the package directly:

```bash
pip install -e .
```

---

## 📥 Download the MediaPipe Model

AeroDraft requires the **MediaPipe Hand Landmarker** model.

Download: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

Place the downloaded file `hand_landmarker.task` inside the project root.

```text
AeroDraft/
├── hand_landmarker.task
├── main.py
├── camera.py
├── config.py
└── ...
```

> **Note:** The model file is intentionally excluded from Git because it is a third-party asset.

---

## ✅ Prerequisites

- Python 3.10+
- Webcam (USB or built-in)
- OpenCV-compatible camera
- MediaPipe Hand Landmarker model

---

# ▶️ Running AeroDraft

Start the application:

```bash
python -m main
```

Or, if installed as a package:

```bash
aerodraft
```

---

## 🖥️ What You'll See

Once the application starts:

- Live webcam feed
- Hand landmarks overlay
- Gesture detection status
- **Specialized 3D wireframe product** (default: cube)
- Current interaction state with **color-coded feedback**
- **Product name and category** in the HUD
- FPS counter and relative depth estimation
- **Transient notification banners** for key events
- Runtime diagnostics

---

## 🎮 Controls

| Input | Action |
|-------|--------|
| **Pinch** (thumb + index) | Place / start drawing the object |
| **Move Hand** | Position the object in 3D space |
| **Fist** | Lock / confirm placement |
| **Open Palm** | Reset to idle |
| **Keys 1–7** | Switch product instantly |
| **Q / ESC** | Exit application |

---

## 🎨 Visual Feedback

| State | Color | Meaning |
|-------|-------|---------|
| **IDLE** | Gray | Ready to place |
| **DRAWING** | Yellow | Object is being positioned |
| **PLACED** | Green | Object positioned, awaiting lock |
| **LOCKED** | Blue | Placement confirmed |

Transient banners appear for key events:
- `✓ OBJECT PLACED` — when placement completes
- `✓ OBJECT LOCKED` — when fist confirms placement
- `⚠ HAND LOST` — when tracking is interrupted
- `✓ TRACKING RESTORED` — when hand re-enters frame

---

## 🖥️ Demo Mode

Enable the **MSME Demo Panel** by setting in `config.py`:

```python
demo: DemoConfig = field(default_factory=lambda: DemoConfig(enabled=True))
```

This displays a persistent help panel showing:
- The demo scenario context
- Control instructions for judges and visitors
- Current product category and status

---

## ⚙️ Configuration

AeroDraft uses immutable configuration dataclasses defined in `config.py`.

Most runtime parameters can be adjusted without modifying application logic.

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Camera Width | 640 | Webcam capture width |
| Camera Height | 480 | Webcam capture height |
| Camera FPS | 30 | Target capture frame rate |
| Focal Length | 500.0 | Perspective projection focal length |
| Pinch Start Threshold | 0.15 | Gesture activation threshold |
| Pinch Release Threshold | 0.30 | Gesture release threshold |
| Lock Hold Duration | 1.0 sec | Time required to lock an object |
| Default Object | cube | Starting product (cube, switchboard, socket, etc.) |

---

## 📊 Benchmark Mode

Enable runtime performance instrumentation:

```bash
AERODRAFT_BENCHMARK=1 python -m main
```

The application records metrics such as:

- FPS
- Frame latency
- Processing time
- Camera performance

Benchmarking is disabled by default and has negligible overhead when not enabled.

---

## 📝 Logging

AeroDraft provides structured runtime logging for:

- Application startup
- Camera initialization
- Model loading
- Gesture transitions
- Performance statistics
- Graceful shutdown

These logs simplify debugging and performance analysis.

# 🧪 Testing

AeroDraft includes a comprehensive automated test suite covering the core Computer Vision pipeline, mathematical computations, gesture recognition, state management, rendering logic, and product catalog.

## Running the Test Suite

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific modules
pytest tests/test_catalog.py -v
pytest tests/test_hud_renderer.py -v
pytest tests/test_object_renderer.py -v

# Generate coverage report
pytest --cov
```

---

## Test Coverage

The automated tests verify:

* Camera pipeline initialization
* MediaPipe integration
* Gesture classification
* State machine transitions
* Relative depth estimation
* Coordinate filtering
* One Euro Filter behavior
* Perspective projection
* **Object rendering (7 products)**
* **Product catalog metadata**
* **HUD rendering and notifications**
* Configuration validation
* Benchmark utilities
* End-to-end pipeline data flow

The test suite is designed so that most tests can run without requiring a physical webcam by using mocked components where appropriate.

---

## Test Summary

| Module | Tests | Focus |
|--------|-------|-------|
| `test_benchmark` | 7 | Performance timing |
| `test_camera` | 5 | Video stream capture |
| `test_catalog` | 11 | Product metadata registry |
| `test_coordinate_filter` | 1 | Spatial smoothing |
| `test_depth_estimator` | 7 | ASME depth calculation |
| `test_gesture_classifier` | 10 | Pinch/fist/palm detection |
| `test_hand_tracker` | 4 | MediaPipe integration |
| `test_hud_renderer` | 35 | Overlay rendering & notifications |
| `test_object_renderer` | 18 | Wireframe primitives |
| `test_one_euro_filter` | 7 | Signal smoothing |
| `test_pipeline` | 1 | End-to-end data flow |
| `test_projection` | 9 | 3D→2D perspective math |
| `test_state_machine` | 11 | Gesture-driven state transitions |
| `test_wireframe_renderer` | 6 | Base cube rendering |
| **Total** | **132** | **Full system coverage** |

---

# 🔍 Troubleshooting

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| Camera cannot be opened | Incorrect camera index | Update `camera.device_index` in `config.py` |
| Model file not found | Missing `hand_landmarker.task` | Download the model and place it in the project root |
| Low FPS | CPU limitations | Reduce camera resolution in `config.py` |
| Gesture flickering | Thresholds too sensitive | Adjust gesture thresholds in `config.py` |
| Box movement is unstable | Filtering disabled or misconfigured | Verify the coordinate filter configuration |
| Application exits immediately | Webcam access denied | Check camera permissions and ensure no other application is using the camera |

---

# 📊 Performance

AeroDraft is designed to run efficiently on standard consumer hardware.

Typical runtime metrics include:

* Real-time webcam processing
* Low-latency gesture recognition
* Stable coordinate filtering
* Lightweight CPU execution
* Optional runtime benchmarking

When benchmarking is enabled, AeroDraft records:

* Average FPS
* Frame latency
* Camera processing time
* Gesture processing time
* Rendering time

These metrics can be used to evaluate deployment on target hardware.

# 👨‍💻 Developer Guide

AeroDraft follows a modular architecture, making it straightforward to extend individual components without affecting the rest of the pipeline.

## Adding a New Product

1. Add a new `ProductInfo` entry to `engine/catalog.py`.
2. Create a specialized renderer class in `engine/object_renderer.py` (or reuse `CubeRenderer`).
3. Register the renderer in `ObjectRenderer._renderers`.
4. Add a selection key in `main.py` (e.g., `elif key == ord("8")`).
5. Add unit tests covering the new renderer.

---

## Adding a New Gesture

1. Add a new value to the `GestureType` enumeration.
2. Implement the detection logic inside `GestureClassifier`.
3. Add priority and conflict resolution if necessary.
4. Update the HUD legend if the gesture should be displayed.
5. Add unit tests covering the new gesture.

---

## Adding a New Interaction State

1. Add a new value to `BoxState`.
2. Implement the corresponding state handler.
3. Define valid transitions.
4. Update rendering behavior and `state_colors` in `config.py` if required.
5. Add unit tests for all new transitions.

---

## Adding a New Renderer

Rendering modules should:

* Receive processed spatial information
* Avoid modifying application state
* Be independent of gesture logic
* Keep rendering separate from perception

This allows multiple renderers (wireframe, mesh, annotation, etc.) to coexist.

---

## Coding Standards

The project follows:

* Modular architecture
* Single Responsibility Principle (SRP)
* Immutable configuration
* Explicit typing
* Comprehensive logging
* Test-driven development
* Clean, readable Python

---

## Project Workflow

```text
Capture Frame
      │
      ▼
Hand Tracking
      │
      ▼
Gesture Recognition
      │
      ▼
Depth Estimation
      │
      ▼
Coordinate Filtering
      │
      ▼
State Machine
      │
      ▼
3D Projection
      │
      ▼
Object Rendering
      │
      ▼
HUD Display
```

---

## Repository Goals

The project prioritizes:

* Maintainability
* Reliability
* Extensibility
* Real-time performance
* Production-quality engineering
* **Business demonstration value**

# 🗺️ Roadmap

The core Computer Vision system is complete. Future work focuses on expanding capabilities rather than rebuilding the existing architecture.

## Current Release (v0.12)

* ✅ Webcam pipeline
* ✅ MediaPipe hand tracking
* ✅ Gesture recognition
* ✅ Relative depth estimation
* ✅ One Euro filtering
* ✅ Coordinate smoothing
* ✅ Gesture state machine
* ✅ Perspective projection
* ✅ **Object renderer with 7 electrical products**
* ✅ **Product catalog with business metadata**
* ✅ **Professional HUD with notifications**
* ✅ **Demo help panel**
* ✅ Performance benchmarking
* ✅ **132 automated tests**

---

## Planned Enhancements

### Interactive Object Manipulation

* Rotate objects using gestures
* Scale objects
* Move objects in 3D space

---

### Mid-Air Sketching

* Continuous freehand drawing
* Line smoothing
* Shape recognition

---

### Engineering Tools

* Measurement utilities
* Alignment guides
* Reference planes
* Grid snapping

---

### CAD Integration

* Export to DXF
* Export to STL
* Export to OBJ
* Export to STEP (future research)

---

### AI-Assisted Interaction

* Intelligent gesture prediction
* Adaptive gesture thresholds
* Personalized interaction profiles
* AI-assisted sketch completion

---

### Multi-Hand Support

* Two-hand interaction
* Collaborative manipulation
* Multi-user experiments

---

### Deployment

* Windows executable
* Cross-platform packaging
* Installer generation
* Continuous Integration (CI)

# 🤝 Contributing

Contributions are welcome.

If you would like to improve AeroDraft:

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/my-feature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push your branch.

```bash
git push origin feature/my-feature
```

5. Open a Pull Request.

---

## Contribution Guidelines

Please ensure:

* Code follows the existing project structure.
* New features include documentation.
* New functionality is covered by tests.
* Existing tests continue to pass.
* Public APIs remain backward compatible where possible.

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for complete license information.

---

# 👤 Author

**Anushath S**

Computer Science & Engineering Student

Artificial Intelligence & Machine Learning Enthusiast

### GitHub

https://github.com/Anushath15

---

## Acknowledgements

This project builds upon several outstanding open-source technologies, including:

* OpenCV
* MediaPipe
* NumPy
* PyTest
* Loguru

Their contributions to the open-source ecosystem have made projects like AeroDraft possible.

# 🌟 Vision

> **"AeroDraft aims to redefine human-computer interaction by enabling intuitive, touch-free spatial visualization through AI-powered Computer Vision, making advanced engineering interaction accessible using only a standard webcam."**

> **"See it before you install it."**

---

# 📌 Project Highlights

* Edge AI architecture
* Real-time Computer Vision
* Human-Computer Interaction (HCI)
* Gesture Recognition
* Relative Depth Estimation
* Signal Processing
* Spatial Computing
* Engineering Visualization
* **MSME Product Demonstration**
* **Professional Demo Experience**
* Modular Software Engineering
* Production-quality Python Architecture

---

## ⭐ Support the Project

If you find AeroDraft useful or interesting:

* ⭐ Star the repository
* 🍴 Fork the project
* 🛠️ Contribute improvements
* 📝 Report issues
* 💡 Share ideas and feedback

Every contribution helps improve AeroDraft and supports continued development.

---

<p align="center">
<b>Built with ❤️ using Python, OpenCV, MediaPipe, and Computer Vision.</b>
</p>
