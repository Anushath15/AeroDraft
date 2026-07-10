"""
Central configuration module for AeroDraft.
All hardcoded values live here as immutable frozen dataclasses.
No other module should define magic numbers.
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass(frozen=True)
class CameraConfig:
    """Settings for OpenCV VideoCapture."""
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class TrackerConfig:
    """Settings for MediaPipe Hands."""
    static_image_mode: bool = False
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7


@dataclass(frozen=True)
class ASMEConfig:
    """Settings for the Adaptive Spatial Mapping Engine (depth + filtering)."""
    reference_distance_px: float = 120.0
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007
    one_euro_d_cutoff: float = 1.0


@dataclass(frozen=True)
class BoxConfig:
    """Default physical dimensions for the placement box (meters)."""
    width: float = 0.2
    height: float = 0.2
    depth: float = 0.2


@dataclass(frozen=True)
class RenderConfig:
    """Settings for the object render engine."""
    focal_length: float = 500.0
    box_color_bgr: Tuple[int, int, int] = (0, 255, 255)
    line_thickness: int = 2
    default_object: str = "cube"
    # Phase 12: State-based colors for visual feedback (BGR format)
    state_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "IDLE": (128, 128, 128),      # Gray
        "DRAWING": (0, 255, 255),      # Yellow
        "PLACED": (0, 255, 0),         # Green
        "LOCKED": (255, 0, 0),         # Blue
    })
    selection_thickness: int = 3
    highlight_color: Tuple[int, int, int] = (0, 165, 255)  # Orange


@dataclass(frozen=True)
class GestureConfig:
    """Settings for gesture classification with hysteresis."""
    pinch_start_threshold: float = 0.15
    pinch_release_threshold: float = 0.30
    frame_history_len: int = 5
    fist_threshold_ratio: float = 0.4
    open_palm_threshold_ratio: float = 0.7
    pinch_threshold_ratio: float = 0.15


@dataclass(frozen=True)
class StateMachineConfig:
    """Settings for the gesture-driven box state machine."""
    lock_hold_duration_s: float = 1.0
    default_box_half_extents: Tuple[float, float, float] = (0.1, 0.1, 0.1)


@dataclass(frozen=True)
class HUDConfig:
    """Layout and styling for the heads-up display overlay."""
    font_face: int = 0
    font_scale: float = 0.5
    font_thickness: int = 1
    text_color: Tuple[int, int, int] = (0, 255, 0)
    alert_color: Tuple[int, int, int] = (0, 0, 255)
    warning_color: Tuple[int, int, int] = (0, 255, 255)
    info_color: Tuple[int, int, int] = (255, 255, 0)
    line_spacing: int = 25
    left_margin: int = 10
    top_margin: int = 30
    indicator_radius: int = 8
    # Phase 12
    notification_duration_frames: int = 60
    demo_panel_width: int = 260


@dataclass(frozen=True)
class DemoConfig:
    """MSME Demo Mode settings."""
    enabled: bool = False
    show_help_panel: bool = True
    scenario_text: str = "Customer wants to preview electrical products before installation"


@dataclass(frozen=True)
class SketchConfig:
    """Settings for mid-air freehand sketching mode."""
    color_bgr: Tuple[int, int, int] = (255, 0, 255)  # Magenta - visually distinct from box colors
    thickness: int = 2
    min_point_distance: float = 0.01  # In virtual projection-space units


@dataclass(frozen=True)
class AppConfig:
    """Global application settings."""
    window_name: str = "AeroDraft"
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    asme: ASMEConfig = field(default_factory=ASMEConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    sketch: SketchConfig = field(default_factory=SketchConfig)


settings = AppConfig()