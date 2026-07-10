"""
Central configuration module for AeroDraft.
All hardcoded values live here as immutable frozen dataclasses.
No other module should define magic numbers.
"""
from dataclasses import dataclass, field


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
class AppConfig:
    """Global application settings."""
    window_name: str = "AeroDraft"
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    asme: ASMEConfig = field(default_factory=ASMEConfig)


# Global singleton - import this everywhere
settings = AppConfig()