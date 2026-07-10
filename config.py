"""
Central configuration module for AeroDraft.
All hardcoded values live here as immutable frozen dataclasses.
"""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class GestureConfig:
    """Settings for gesture classification thresholds."""
    pinch_start_threshold: float = 0.15
    pinch_release_threshold: float = 0.30
    fist_threshold_ratio: float = 0.4
    open_palm_threshold_ratio: float = 0.7
    frame_history_len: int = 5
    # Backward compatibility for legacy tests
    pinch_threshold_ratio: float = 0.15 

    def __post_init__(self):
        """Map legacy ratio to start threshold if ratio provided."""
        # Validation for production
        if self.pinch_start_threshold <= 0:
            raise ValueError("Thresholds must be positive.")    
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
    """Settings for the Adaptive Spatial Mapping Engine."""
    reference_distance_px: float = 120.0
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007
    one_euro_d_cutoff: float = 1.0

@dataclass(frozen=True)
class RenderConfig:
    """Settings for the wireframe render engine."""
    focal_length: float = 500.0
    box_color_bgr: Tuple[int, int, int] = (0, 255, 255)
    line_thickness: int = 2

@dataclass(frozen=True)
class BoxConfig:
    """Dimensions for the visualization cuboid (virtual space units)."""
    width: float = 0.5  # half-width
    height: float = 0.5 # half-height
    depth: float = 0.5  # half-depth

@dataclass(frozen=True)
class GestureConfig:
    """Settings for gesture classification thresholds."""
    pinch_start_threshold: float = 0.15
    pinch_release_threshold: float = 0.30
    frame_history_len: int = 5
    fist_threshold_ratio: float = 0.4
    open_palm_threshold_ratio: float = 0.7

@dataclass(frozen=True)
class StateMachineConfig:
    """Settings for the gesture-driven box state machine."""
    lock_hold_duration_s: float = 1.0
    default_box_half_extents: Tuple[float, float, float] = (0.1, 0.1, 0.1)

@dataclass(frozen=True)
class AppConfig:
    """Global application settings."""
    window_name: str = "AeroDraft"
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    asme: ASMEConfig = field(default_factory=ASMEConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)

# Global singleton
settings = AppConfig()