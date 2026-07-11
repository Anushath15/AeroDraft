from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
import numpy as np
from loguru import logger

from config import AppConfig
from core.error_recovery import ErrorRecoveryManager, FailureType


@dataclass
class PipelineState:
    """Centralized mutable state for the pipeline."""
    current_object_type: str = "cube"
    previous_state: Any = None
    was_tracking: bool = False
    sketch_mode: bool = False
    was_sketch_pinching: bool = False
    fps: float = 0.0
    frame_count: int = 0
    running: bool = True


class Pipeline:
    """
    Orchestrates the AeroDraft processing pipeline.
    Extracted from main.py to enable testing and DI.
    """
    
    def __init__(self, config: AppConfig, camera: Any, tracker: Any, 
                 classifier: Any, state_machine: Any, depth_estimator: Any, 
                 projector: Any, hud_renderer: Any):
        self.config = config
        self.camera = camera
        self.tracker = tracker
        self.classifier = classifier
        self.state_machine = state_machine
        self.depth_estimator = depth_estimator
        self.projector = projector
        self.hud_renderer = hud_renderer
        
        self.state = PipelineState()
        self.notification = _NotificationState(self.config.hud.notification_duration_frames)
        self.renderer_cache = _ObjectRendererCache(self.config.render)
        
        self._last_frame_time = time.perf_counter()
        self._last_fps_update = time.perf_counter()
        self._frame_time_accum = 0.0
        self._frame_count = 0
        
        self._recovery = ErrorRecoveryManager(max_attempts=3)
        self._setup_recovery_handlers()
    
    def _setup_recovery_handlers(self):
        self._recovery.register_handler(FailureType.CAMERA_FAILURE, self._recover_camera)
    
    def _recover_camera(self) -> bool:
        try:
            self.camera.__exit__(None, None, None)
            self.camera.__enter__()
            return True
        except Exception as e:
            logger.error(f"Camera recovery failed: {e}")
            return False
    
    def process_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Process a single frame through the full pipeline."""
        try:
            success, frame = self._capture_frame()
            if not success: return False, None
            
            results = self._track_hand(frame)
            self._update_state(results, frame.shape[:2])
            self._render(frame, results)
            self._update_hud(frame)
            self._update_timing()
            
            return True, frame
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return False, None
    
    def _capture_frame(self) -> Tuple[bool, np.ndarray]:
        try:
            return self.camera.read_frame()
        except Exception as e:
            logger.error(f"Camera error: {e}")
            if self._recovery.report_failure(FailureType.CAMERA_FAILURE):
                if self._recovery.attempt_recovery(FailureType.CAMERA_FAILURE):
                    return self.camera.read_frame()
            raise
    
    def _track_hand(self, frame: np.ndarray) -> Any:
        timestamp_ms = int(time.perf_counter() * 1000)
        return self.tracker.process_frame(frame, timestamp_ms)
    
    def _update_state(self, results: Any, frame_shape: Tuple[int, int]) -> None: pass
    def _render(self, frame: np.ndarray, results: Any) -> None: pass
    def _update_hud(self, frame: np.ndarray) -> None: pass
    
    def _update_timing(self) -> None:
        now = time.perf_counter()
        delta = now - self._last_frame_time
        self._last_frame_time = now
        self._frame_time_accum += delta
        self._frame_count += 1
        
        if now - self._last_fps_update >= 0.5:
            self.state.fps = self._frame_count / self._frame_time_accum
            self._frame_time_accum = 0.0
            self._frame_count = 0
            self._last_fps_update = now
    
    def switch_product(self, product_type: str) -> bool:
        from engine.catalog import ProductCatalog
        if product_type in ProductCatalog.products():
            self.state.current_object_type = product_type
            return True
        return False
    
    def reset(self) -> None:
        self.state_machine.reset()
        self.state.previous_state = None
        self.state.was_tracking = False


# Move these from main.py to their proper modules
class _NotificationState: pass # Move to ui/notification.py
class _ObjectRendererCache: pass # Move to engine/object_renderer.py