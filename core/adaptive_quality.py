from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from loguru import logger
import time

class QualityLevel(Enum):
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    RECOVERY = auto()

@dataclass
class QualityConfig:
    camera_width: int
    camera_height: int
    tracking_confidence: float
    skip_frames: int
    enable_smoothing: bool
    enable_hud: bool

QUALITY_PRESETS: dict[QualityLevel, QualityConfig] = {
    QualityLevel.HIGH: QualityConfig(640, 480, 0.7, 1, True, True),
    QualityLevel.MEDIUM: QualityConfig(480, 360, 0.6, 2, True, True),
    QualityLevel.LOW: QualityConfig(320, 240, 0.5, 3, False, True),
    QualityLevel.RECOVERY: QualityConfig(160, 120, 0.4, 5, False, False),
}

class AdaptiveQualityController:
    def __init__(self, target_fps: float = 24.0, min_fps: float = 20.0, upgrade_fps: float = 28.0):
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.upgrade_fps = upgrade_fps
        self.current_level = QualityLevel.HIGH
        self.fps_history: list[float] = []
        self._stable_count = 0
        self._last_check = time.perf_counter()
    
    def update(self, fps: float, frame_time: float) -> Optional[QualityLevel]:
        self.fps_history.append(fps)
        if len(self.fps_history) > 60: self.fps_history.pop(0)
        
        now = time.perf_counter()
        if now - self._last_check < 0.5: return None
        self._last_check = now
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        levels = list(QualityLevel)
        
        if avg_fps < self.min_fps and self.current_level != QualityLevel.RECOVERY:
            idx = levels.index(self.current_level)
            if idx < len(levels) - 1:
                self.current_level = levels[idx + 1]
                logger.warning(f"Downgrading to {self.current_level.name}")
                self._stable_count = 0
                return self.current_level
        elif avg_fps > self.upgrade_fps and self.current_level != QualityLevel.HIGH:
            self._stable_count += 1
            if self._stable_count >= 30:
                idx = levels.index(self.current_level)
                if idx > 0:
                    self.current_level = levels[idx - 1]
                    logger.info(f"Upgrading to {self.current_level.name}")
                    self._stable_count = 0
                    return self.current_level
        else:
            self._stable_count = 0
        return None
    
    @property
    def config(self) -> QualityConfig:
        return QUALITY_PRESETS[self.current_level]
