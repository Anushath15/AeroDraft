"""
AeroDraft - Master entry point.
Orchestrates the complete pipeline:
    Camera -> Hand Tracking -> Gesture Classification -> State Machine
    -> Depth Estimation -> Coordinate Smoothing -> 3D Projection
    -> Object Rendering / Sketch Rendering -> HUD Overlay -> Display
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from typing import Any, List, Optional, Tuple

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

# ═══════════════════════════════════════════════════════════════════
# ADDITION 1 — API imports
# ═══════════════════════════════════════════════════════════════════
from api.server import start_api_server
from api.shared_state import shared_state, command_queue


try:
    from core.coordinate_filter import CoordinateFilter
    _HAS_COORDINATE_FILTER = True
except ImportError:
    _HAS_COORDINATE_FILTER = False
    logger.debug("CoordinateFilter not found - skipping coordinate smoothing.")


OBJECT_TYPE_KEYS: dict[int, str] = {
    ord("1"): "cube",
    ord("2"): "switchboard",
    ord("3"): "socket",
    ord("4"): "ceiling_light",
    ord("5"): "junction_box",
    ord("6"): "conduit_box",
    ord("7"): "distribution_board",
}

WRIST_IDX = 0
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8


def _validate_config() -> None:
    """Validates critical configuration values at startup."""
    cfg = settings

    if cfg.camera.width <= 0 or cfg.camera.height <= 0:
        raise RuntimeError(
            f"Invalid camera resolution: {cfg.camera.width}x{cfg.camera.height}"
        )

    if cfg.render.focal_length <= 0:
        raise RuntimeError(
            f"Invalid focal_length: {cfg.render.focal_length} (must be positive)"
        )

    if cfg.gesture.pinch_start_threshold >= cfg.gesture.pinch_release_threshold:
        logger.warning(
            "pinch_start_threshold >= pinch_release_threshold - "
            "this only affects the unused hysteresis path (is_pinch); "
            "the live pipeline uses classify(), not is_pinch()."
        )

    model_path = getattr(HandTracker, "MODEL_PATH", "hand_landmarker.task")
    if not os.path.exists(model_path):
        logger.warning(
            f"Model file not found: {model_path}. Hand tracking will fail. "
            f"Download from: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker"
        )


def _extract_wrist_position(
    landmarks: Any, frame_width: int, frame_height: int
) -> Tuple[int, int]:
    """Converts normalized wrist landmark (index 0) to pixel coordinates.

    Used as the reference point for object placement - the wrist is a
    stable point that doesn't move much during pinch/fist/palm gestures.
    """
    wrist = landmarks[WRIST_IDX]
    return int(wrist.x * frame_width), int(wrist.y * frame_height)


def _extract_pinch_tip_position(
    landmarks: Any, frame_width: int, frame_height: int
) -> Tuple[int, int]:
    """Converts the midpoint between thumb tip and index tip to pixel coordinates.

    This is the actual visual location where a pinch occurs - used as the
    'pen tip' for sketching, since the wrist does not track where the
    fingers are actually pinching in space.
    """
    thumb_tip = landmarks[THUMB_TIP_IDX]
    index_tip = landmarks[INDEX_TIP_IDX]
    mid_x = (thumb_tip.x + index_tip.x) / 2.0
    mid_y = (thumb_tip.y + index_tip.y) / 2.0
    return int(mid_x * frame_width), int(mid_y * frame_height)


class _NotificationState:
    """Owns transient HUD notification banner text, color, and countdown."""

    def __init__(self, duration_frames: int) -> None:
        self._duration_frames = duration_frames
        self.text: Optional[str] = None
        self.color: Optional[Tuple[int, int, int]] = None
        self._frames_remaining = 0

    def trigger(self, text: str, color: Tuple[int, int, int]) -> None:
        self.text = text
        self.color = color
        self._frames_remaining = self._duration_frames

    def tick(self) -> None:
        if self._frames_remaining > 0:
            self._frames_remaining -= 1
        else:
            self.text = None
            self.color = None


class _ObjectRendererCache:
    """
    Lazily rebuilds ObjectRenderer only when object type or color changes.

    ObjectRenderer bakes its object type and color in at construction time
    (it has no per-call parameters for either), so switching products or
    state-based colors requires constructing a new instance. This cache
    avoids doing that on every single frame when nothing has changed.

    Known limitation: ObjectRenderer has no 'highlight' concept. The
    DRAWING/PLACED-state highlight effect described in early planning
    is not implemented here and is not silently faked.
    """

    def __init__(self, base_config: Any) -> None:
        self._base_config = base_config
        self._current_key: Optional[Tuple[str, Tuple[int, int, int]]] = None
        self._renderer: Optional[ObjectRenderer] = None

    def get(self, object_type: str, color: Tuple[int, int, int]) -> ObjectRenderer:
        key = (object_type, color)
        if key != self._current_key:
            render_cfg = dataclasses.replace(
                self._base_config, default_object=object_type, box_color_bgr=color
            )
            self._renderer = ObjectRenderer(render_cfg)
            self._current_key = key
        return self._renderer


def main() -> None:
    """Runs the integrated AeroDraft application loop."""
    logger.info("AeroDraft starting - Phase 13 (Mid-Air Sketching)")

    # ═══════════════════════════════════════════════════════════════════
    # ADDITION 2 — Start REST API server
    # ═══════════════════════════════════════════════════════════════════
    start_api_server(host="0.0.0.0", port=8000)
    logger.info("REST API started — http://localhost:8000  |  docs: http://localhost:8000/docs")

    _validate_config()

    hud_renderer = HUDRenderer(settings.hud)
    object_renderer_cache = _ObjectRendererCache(settings.render)
    sketch_renderer = SketchRenderer(
        color=settings.sketch.color_bgr, thickness=settings.sketch.thickness
    )
    sketch_manager = SketchManager(min_point_distance=settings.sketch.min_point_distance)
    projector = PerspectiveProjector(
        focal_length=settings.render.focal_length,
        frame_width=settings.camera.width,
        frame_height=settings.camera.height,
    )
    depth_estimator = DepthEstimator(settings.asme)
    gesture_classifier = GestureClassifier(settings.gesture)
    state_machine = GestureStateMachine(settings.state_machine)

    current_object_type = settings.render.default_object
    previous_state = BoxState.IDLE
    was_tracking = False
    demo_config = settings.demo
    sketch_mode = False
    was_sketch_pinching = False

    notification = _NotificationState(settings.hud.notification_duration_frames)

    coordinate_filter: Optional[Any] = None
    if _HAS_COORDINATE_FILTER:
        try:
            coordinate_filter = CoordinateFilter(settings.asme)
        except Exception as e:
            logger.warning(f"Failed to initialize CoordinateFilter: {e}")

    fps = 0.0
    frame_time_accumulator = 0.0
    frame_count_for_fps = 0
    fps_update_interval = 0.5
    last_frame_time = time.perf_counter()
    last_fps_update_time = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════════
    # ADDITION 2 (continued) — API timing & screenshot flag
    # ═══════════════════════════════════════════════════════════════════
    _start_time = time.perf_counter()
    _api_screenshot_pending = False

    try:
        with VideoStream(
            device_index=settings.camera.device_index,
            width=settings.camera.width,
            height=settings.camera.height,
        ) as stream, HandTracker(config=settings.tracker) as tracker:

            logger.info("Pipeline active. Press Q or ESC to exit.")
            logger.info("Keys: 1-7 switch objects | D toggle demo panel | S toggle sketch mode | C clear sketch")

            while True:
                key = 0xFF
                
                with benchmark("camera_read"):
                    success, frame = stream.read_frame()

                if not success or frame is None:
                    logger.warning("Dropped frame - skipping.")
                    continue

                frame_height, frame_width = frame.shape[:2]
                current_time = time.perf_counter()

                # ═══════════════════════════════════════════════════════════
                # ADDITION 3 — Process API commands (from Postman / REST clients)
                # ═══════════════════════════════════════════════════════════
                while not command_queue.empty():
                    try:
                        cmd = command_queue.get_nowait()
                        cmd_type = cmd.get("type")

                        if cmd_type == "switch_product":
                            current_object_type = cmd["value"]
                            logger.info(f"API: switched product to '{current_object_type}'")

                        elif cmd_type == "toggle_demo":
                            demo_config = dataclasses.replace(
                                demo_config, enabled=not demo_config.enabled
                            )
                            logger.info(f"API: demo mode {'ON' if demo_config.enabled else 'OFF'}")

                        elif cmd_type == "screenshot":
                            _api_screenshot_pending = True

                        elif cmd_type == "reset":
                            state_machine.reset()
                            logger.info("API: state machine reset to IDLE")

                    except Exception as _cmd_err:
                        logger.warning(f"API command error: {_cmd_err}")

                frame_delta = current_time - last_frame_time
                last_frame_time = current_time

                if frame_delta > 0:
                    instant_fps = 1.0 / frame_delta
                    frame_time_accumulator += instant_fps
                    frame_count_for_fps += 1

                if current_time - last_fps_update_time >= fps_update_interval:
                    if frame_count_for_fps > 0:
                        fps = frame_time_accumulator / frame_count_for_fps
                    frame_time_accumulator = 0.0
                    frame_count_for_fps = 0
                    last_fps_update_time = current_time

                with benchmark("hand_tracking"):
                    results = tracker.process_frame(frame)

                hand_landmarks = (
                    results.hand_landmarks[0]
                    if results and results.hand_landmarks and len(results.hand_landmarks) > 0
                    else None
                )
                hand_detected = hand_landmarks is not None

                if hand_detected and not was_tracking:
                    notification.trigger("TRACKING RESTORED", (0, 255, 0))
                elif not hand_detected and was_tracking:
                    notification.trigger("HAND LOST", (0, 0, 255))
                was_tracking = hand_detected

                gesture = GestureType.NONE
                current_state = state_machine.current_state
                smoothed_depth: Optional[float] = None
                hand_px, hand_py = 0, 0

                if not hand_detected:
                    if sketch_mode:
                        if was_sketch_pinching:
                            sketch_manager.end_stroke()
                            was_sketch_pinching = False
                    else:
                        if state_machine.current_state != BoxState.IDLE:
                            was_locked = state_machine.current_state == BoxState.LOCKED
                            state_machine.reset()
                            if was_locked:
                                notification.trigger("OBJECT RESET", (128, 128, 128))
                            previous_state = BoxState.IDLE
                        current_state = state_machine.current_state

                    depth_estimator.reset()

                else:
                    with benchmark("gesture_classification"):
                        try:
                            gesture = gesture_classifier.classify(hand_landmarks)
                        except Exception as e:
                            logger.warning(f"Gesture classification failed: {e}")
                            gesture = GestureType.NONE

                    if sketch_mode:
                        hand_px, hand_py = _extract_pinch_tip_position(
                            hand_landmarks, frame_width, frame_height
                        )
                    else:
                        hand_px, hand_py = _extract_wrist_position(
                            hand_landmarks, frame_width, frame_height
                        )

                    with benchmark("depth_estimation"):
                        try:
                            depth = depth_estimator.estimate(
                                hand_landmarks,
                                frame_width=frame_width,
                                frame_height=frame_height,
                                timestamp=current_time,
                            )
                        except NoHandDetectedError as e:
                            logger.debug(f"Depth estimation skipped: {e}")
                            depth = None

                    hand_virtual_x = (hand_px - frame_width / 2) / settings.render.focal_length
                    hand_virtual_y = (hand_py - frame_height / 2) / settings.render.focal_length
                    depth_for_filter = depth if depth is not None else 1.0

                    smoothed_x, smoothed_y, smoothed_depth = (
                        hand_virtual_x, hand_virtual_y, depth_for_filter
                    )
                    if coordinate_filter is not None:
                        with benchmark("coordinate_smoothing"):
                            try:
                                smoothed_x, smoothed_y, smoothed_depth = coordinate_filter(
                                    hand_virtual_x, hand_virtual_y, depth_for_filter, current_time
                                )
                            except Exception as e:
                                logger.debug(f"Coordinate smoothing failed: {e}")

                    if sketch_mode:
                        is_pinching = gesture == GestureType.PINCH
                        if is_pinching:
                            if not was_sketch_pinching:
                                sketch_manager.start_stroke()
                            sketch_manager.add_point(
                                np.array([smoothed_x, smoothed_y, smoothed_depth])
                            )
                        elif was_sketch_pinching:
                            sketch_manager.end_stroke()
                        was_sketch_pinching = is_pinching

                    else:
                        hand_virtual = np.array([smoothed_x, smoothed_y])
                        with benchmark("state_machine"):
                            try:
                                current_state = state_machine.update(
                                    gesture=gesture,
                                    hand_position=hand_virtual,
                                    depth=smoothed_depth,
                                    timestamp=current_time,
                                )
                            except Exception as e:
                                logger.warning(f"State machine update failed: {e}")
                                current_state = state_machine.current_state

                        if current_state != previous_state:
                            if current_state == BoxState.PLACED:
                                notification.trigger("OBJECT PLACED", (0, 255, 0))
                            elif current_state == BoxState.LOCKED:
                                notification.trigger("OBJECT LOCKED", (255, 0, 0))
                            elif current_state == BoxState.IDLE and previous_state == BoxState.LOCKED:
                                notification.trigger("OBJECT RESET", (128, 128, 128))
                            previous_state = current_state

                        with benchmark("projection_rendering"):
                            if (
                                current_state != BoxState.IDLE
                                and state_machine.current_box_center is not None
                            ):
                                try:
                                    center = state_machine.current_box_center
                                    half_extents = (
                                        state_machine.current_box_half_extents
                                        if state_machine.current_box_half_extents is not None
                                        else np.array(settings.state_machine.default_box_half_extents)
                                    )
                                    projected = projector.project_box(center, half_extents)

                                    state_color = settings.render.state_colors.get(
                                        current_state.name, settings.render.box_color_bgr
                                    )
                                    renderer = object_renderer_cache.get(current_object_type, state_color)
                                    renderer.draw(frame, projected)
                                except InvalidDepthError as e:
                                    logger.debug(f"Projection skipped - box behind camera: {e}")
                                except Exception as e:
                                    logger.warning(f"Object rendering failed: {e}")

                if sketch_manager.strokes:
                    with benchmark("sketch_rendering"):
                        projected_strokes: List[np.ndarray] = []
                        for stroke in sketch_manager.strokes:
                            if len(stroke) < 2:
                                continue
                            stroke_2d = []
                            for point in stroke.points:
                                try:
                                    stroke_2d.append(projector.project_point(point))
                                except InvalidDepthError:
                                    continue
                            if len(stroke_2d) >= 2:
                                projected_strokes.append(np.array(stroke_2d))
                        sketch_renderer.render(frame, projected_strokes)

                notification.tick()

                product = ProductCatalog.get(current_object_type)
                category = product.category if product else "Unknown"

                with benchmark("hud_render"):
                    hud_data = HUDData(
                        tracking=hand_detected,
                        gesture=gesture,
                        state=current_state,
                        object_type=current_object_type if not sketch_mode else "sketch",
                        category=category if not sketch_mode else "Sketch Mode",
                        fps=fps,
                        depth=smoothed_depth,
                        hand_position=(hand_px, hand_py) if hand_detected else None,
                        notification=notification.text,
                        notification_color=notification.color,
                        demo_mode=demo_config.enabled,
                    )
                    annotated = hud_renderer.render(frame, hud_data)

                # ═══════════════════════════════════════════════════════════
                # ADDITION 4 — Update shared state for REST API + screenshot
                # ═══════════════════════════════════════════════════════════
                shared_state.update(
                    tracking=hand_detected,
                    gesture=gesture.name if gesture else "NONE",
                    box_state=current_state.name,
                    object_type=current_object_type,
                    category=category,
                    fps=fps,
                    depth=smoothed_depth,
                    hand_position=(hand_px, hand_py) if hand_detected else None,
                    demo_mode=demo_config.enabled,
                    notification=notification.text,
                    uptime_seconds=time.perf_counter() - _start_time,
                )

                if _api_screenshot_pending or key == ord("s"):
                    _fname = f"aerodraft_{int(time.time())}.png"
                    cv2.imwrite(_fname, annotated)
                    logger.info(f"Screenshot saved: {_fname}")
                    notification.trigger("SCREENSHOT SAVED", (0, 255, 0))
                    _api_screenshot_pending = False

                cv2.imshow(settings.window_name, annotated)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    logger.info("Exit signal received.")
                    break
                elif key == ord("d"):
                    demo_config = dataclasses.replace(
                        demo_config, enabled=not demo_config.enabled
                    )
                    logger.info(f"Demo mode: {'ON' if demo_config.enabled else 'OFF'}")
                elif key == ord("s"):
                    sketch_mode = not sketch_mode
                    if was_sketch_pinching:
                        sketch_manager.end_stroke()
                        was_sketch_pinching = False
                    logger.info(f"Sketch mode: {'ON' if sketch_mode else 'OFF'}")
                elif key == ord("c"):
                    sketch_manager.clear()
                    logger.info("Sketch cleared.")
                elif not sketch_mode and key in OBJECT_TYPE_KEYS:
                    current_object_type = OBJECT_TYPE_KEYS[key]

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    except RuntimeError as e:
        logger.critical(f"Hardware error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected crash: {e}")
        sys.exit(1)
    finally:
        print_report()
        cv2.destroyAllWindows()
        logger.info("AeroDraft shutdown complete.")


if __name__ == "__main__":
    main()