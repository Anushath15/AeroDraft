from __future__ import annotations

import dataclasses
import os
import sys
import time
from typing import Any, Tuple, Optional

import cv2
import numpy as np
from loguru import logger

from engine.catalog import ProductCatalog
from config import settings, DemoConfig
from camera import VideoStream
from hand_tracker import HandTracker
from core.depth_estimator import DepthEstimator, NoHandDetectedError
from core.sketch_path import SketchManager
from engine.projection import PerspectiveProjector, InvalidDepthError
from engine.object_renderer import ObjectRenderer
from engine.sketch_renderer import SketchRenderer
from gestures.gesture_classifier import GestureClassifier, GestureType
from gestures.state_machine import GestureStateMachine, BoxState
from ui.hud_renderer import HUDRenderer, HUDData
from core.benchmark import benchmark, print_report
from core.error_recovery import ErrorRecoveryManager, FailureType

# API imports
try:
    from api.server import start_api_server
    from api.shared_state import shared_state, command_queue
    _HAS_API = True
except ImportError:
    _HAS_API = False

try:
    from core.coordinate_filter import CoordinateFilter
    _HAS_COORDINATE_FILTER = True
except ImportError:
    _HAS_COORDINATE_FILTER = False

OBJECT_TYPE_KEYS: dict[int, str] = {
    ord("1"): "cube", ord("2"): "switchboard", ord("3"): "socket",
    ord("4"): "ceiling_light", ord("5"): "junction_box",
    ord("6"): "conduit_box", ord("7"): "distribution_board",
}

WRIST_IDX = 0
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8


class NotificationState:
    """Owns transient HUD notification banner text, color, and countdown."""
    def __init__(self, duration_frames: int) -> None:
        self._duration_frames = duration_frames
        self.text: Optional[str] = None
        self.color: Optional[Tuple[int, int, int]] = None
        self._frames_remaining = 0

    def trigger(self, text: str, color: Tuple[int, int, int]) -> None:
        self.text, self.color, self._frames_remaining = text, color, self._duration_frames

    def tick(self) -> None:
        if self._frames_remaining > 0:
            self._frames_remaining -= 1
        else:
            self.text, self.color = None, None


class ObjectRendererCache:
    """Lazily rebuilds ObjectRenderer only when object type or color changes."""
    def __init__(self, base_config: Any) -> None:
        self._base_config = base_config
        self._current_key: Optional[Tuple[str, Tuple[int, int, int]]] = None
        self._renderer: Optional[ObjectRenderer] = None

    def get(self, object_type: str, color: Tuple[int, int, int]) -> ObjectRenderer:
        key = (object_type, color)
        if key != self._current_key:
            render_cfg = dataclasses.replace(self._base_config, default_object=object_type, box_color_bgr=color)
            self._renderer = ObjectRenderer(render_cfg)
            self._current_key = key
        return self._renderer


def _validate_config() -> None:
    cfg = settings
    if cfg.camera.width <= 0 or cfg.camera.height <= 0:
        raise RuntimeError(f"Invalid camera resolution: {cfg.camera.width}x{cfg.camera.height}")
    if cfg.render.focal_length <= 0:
        raise RuntimeError(f"Invalid focal_length: {cfg.render.focal_length}")
    
    model_path = getattr(HandTracker, "MODEL_PATH", "hand_landmarker.task")
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}. Hand tracking will fail.")


def main() -> None:
    logger.info("AeroDraft starting - Upgraded Phase 13")
    _validate_config()

    if _HAS_API:
        start_api_server(host="0.0.0.0", port=8000)
        logger.info("REST API started — http://localhost:8000")

    # Initialize Components
    hud_renderer = HUDRenderer(settings.hud)
    renderer_cache = ObjectRendererCache(settings.render)
    sketch_renderer = SketchRenderer(color=settings.sketch.color_bgr, thickness=settings.sketch.thickness)
    sketch_manager = SketchManager(min_point_distance=settings.sketch.min_point_distance)
    projector = PerspectiveProjector(focal_length=settings.render.focal_length, frame_width=settings.camera.width, frame_height=settings.camera.height)
    depth_estimator = DepthEstimator(settings.asme)
    gesture_classifier = GestureClassifier(settings.gesture)
    state_machine = GestureStateMachine(settings.state_machine)
    
    if _HAS_COORDINATE_FILTER:
        coordinate_filter = CoordinateFilter(settings.asme)
    else:
        coordinate_filter = None

    # State Variables
    current_object_type = settings.render.default_object
    previous_state = BoxState.IDLE
    was_tracking = False
    demo_config = settings.demo
    sketch_mode = False
    was_sketch_pinching = False
    notification = NotificationState(settings.hud.notification_duration_frames)

    # Timing Variables
    fps, frame_time_accum, frame_count = 0.0, 0.0, 0
    last_frame_time = time.perf_counter()
    last_fps_update = time.perf_counter()

    # Error Recovery
    recovery = ErrorRecoveryManager(max_attempts=3)

    try:
        with VideoStream(settings.camera.device_index, settings.camera.width, settings.camera.height) as stream, \
             HandTracker(config=settings.tracker) as tracker:

            logger.info("Pipeline active. Press Q or ESC to exit.")
            
            while True:
                current_time = time.perf_counter()
                
                # 1. Capture Frame with Error Recovery
                try:
                    success, frame = stream.read_frame()
                except Exception as cam_err:
                    logger.error(f"Camera read error: {cam_err}")
                    if recovery.report_failure(FailureType.CAMERA_FAILURE):
                        if recovery.attempt_recovery(FailureType.CAMERA_FAILURE):
                            continue
                    else:
                        break 
                    continue

                if not success or frame is None:
                    continue

                frame_height, frame_width = frame.shape[:2]
                
                # 2. Process API Commands
                if _HAS_API:
                    while not command_queue.empty():
                        try:
                            cmd = command_queue.get_nowait()
                            if cmd.get("type") == "switch_product": current_object_type = cmd["value"]
                            elif cmd.get("type") == "toggle_demo": demo_config = dataclasses.replace(demo_config, enabled=not demo_config.enabled)
                            elif cmd.get("type") == "reset": state_machine.reset()
                        except Exception: pass

                # 3. Hand Tracking (UPGRADED: Using VIDEO mode timestamp)
                frame_timestamp_ms = int(current_time * 1000)
                results = tracker.process_frame(frame, frame_timestamp_ms)
                
                has_hand = results and results.hand_landmarks and len(results.hand_landmarks) > 0
                
                # 4. Tracking State Notifications
                if has_hand and not was_tracking:
                    notification.trigger("✓ TRACKING RESTORED", (0, 255, 0))
                elif not has_hand and was_tracking:
                    notification.trigger("⚠ HAND LOST", (0, 255, 255))
                was_tracking = has_hand

                # 5. Core Logic Pipeline (if hand is present)
                if has_hand:
                    landmarks = results.hand_landmarks[0]
                    gesture = gesture_classifier.classify(landmarks)
                    
                    try:
                        depth = depth_estimator.estimate(landmarks)
                        wrist_x, wrist_y = landmarks[WRIST_IDX].x, landmarks[WRIST_IDX].y
                        raw_point = np.array([wrist_x, wrist_y, depth])
                        
                        smoothed = coordinate_filter.filter(raw_point) if coordinate_filter else raw_point
                        proj_x, proj_y = projector.project(smoothed)
                        
                        state_machine.update(gesture, current_time)
                        current_state = state_machine.state
                        
                        # State transition notifications
                        if current_state != previous_state:
                            if current_state == BoxState.PLACED: notification.trigger("✓ OBJECT PLACED", (0, 255, 0))
                            elif current_state == BoxState.LOCKED: notification.trigger("✓ OBJECT LOCKED", (255, 0, 0))
                            previous_state = current_state

                        # 6. Render Object
                        color = settings.render.state_colors.get(current_state.name, (255, 255, 255))
                        renderer = renderer_cache.get(current_object_type, color)
                        renderer.render(frame, proj_x, proj_y, smoothed[2])

                    except (NoHandDetectedError, InvalidDepthError):
                        pass

                # 7. Draw Landmarks & HUD
                tracker.draw_landmarks(frame, results)
                
                hud_data = HUDData(
                    fps=fps, gesture=str(gesture) if has_hand else "None",
                    state=state_machine.state.name, depth=smoothed[2] if has_hand else 0.0,
                    product_name=current_object_type, category=ProductCatalog.get(current_object_type).category if ProductCatalog.get(current_object_type) else "N/A"
                )
                hud_renderer.render(frame, hud_data, demo_config, notification.text, notification.color)
                notification.tick()

                # 8. Display & Input
                cv2.imshow(settings.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key in (ord('q'), 27): break
                elif key in OBJECT_TYPE_KEYS: current_object_type = OBJECT_TYPE_KEYS[key]
                elif key == ord('d'): demo_config = dataclasses.replace(demo_config, enabled=not demo_config.enabled)
                elif key == ord('c'): sketch_manager.clear()

                # 9. FPS Calculation
                delta = current_time - last_frame_time
                last_frame_time = current_time
                frame_time_accum += delta
                frame_count += 1
                if current_time - last_fps_update >= 0.5:
                    fps = frame_count / frame_time_accum
                    frame_time_accum, frame_count = 0.0, 0
                    last_fps_update = current_time

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        logger.info("AeroDraft stopped")

if __name__ == "__main__":
    main()
