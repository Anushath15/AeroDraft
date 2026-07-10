"""
AeroDraft - Master entry point.
Orchestrates the complete pipeline:
    Camera -> Hand Tracking -> Gesture Classification -> State Machine
    -> Depth Estimation -> Coordinate Smoothing -> 3D Projection
    -> Wireframe Rendering -> HUD Overlay -> Display
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config import settings
from camera import VideoStream
from hand_tracker import HandTracker
from core.depth_estimator import DepthEstimator, NoHandDetectedError
from engine.projection import PerspectiveProjector, InvalidDepthError
from engine.wireframe_renderer import WireframeRenderer
from gestures.gesture_classifier import GestureClassifier, GestureType
from gestures.state_machine import GestureStateMachine, BoxState
from ui.hud_renderer import HUDRenderer, HUDData
from core.benchmark import benchmark, print_report

# Optional coordinate filter - may not exist in all repository versions
try:
    from core.coordinate_filter import CoordinateFilter
    _HAS_COORDINATE_FILTER = True
except ImportError:
    _HAS_COORDINATE_FILTER = False
    logger.debug("CoordinateFilter not found - skipping coordinate smoothing.")


def _validate_config() -> None:
    """
    Validates critical configuration values at startup.
    
    Raises RuntimeError if any required value is invalid, preventing
    a late crash deep in the pipeline.
    """
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
            "hysteresis will not function correctly"
        )
    
    # Verify model file exists before opening camera
    model_path = getattr(HandTracker, 'MODEL_PATH', 'hand_landmarker.task')
    if not os.path.exists(model_path):
        logger.warning(
            f"Model file not found: {model_path}. "
            f"Hand tracking will fail. Download from: "
            f"https://developers.google.com/mediapipe/solutions/vision/hand_landmarker"
        )


def _extract_hand_position(
    landmarks: Any, frame_width: int, frame_height: int
) -> Tuple[int, int]:
    """
    Converts normalized wrist landmark to pixel coordinates.
    
    Uses landmark 0 (wrist) as the hand position reference.
    """
    wrist = landmarks[0]
    return int(wrist.x * frame_width), int(wrist.y * frame_height)


def main() -> None:
    """Runs the integrated AeroDraft application loop."""
    logger.info("AeroDraft starting - Phase 10 (Full Integration)")
    
    # Validate configuration before opening hardware
    _validate_config()

    # -- Initialize subsystems --
    hud_renderer = HUDRenderer(settings.hud)
    wireframe_renderer = WireframeRenderer(
        color=settings.render.box_color_bgr,
        thickness=settings.render.line_thickness,
    )
    projector = PerspectiveProjector(
        focal_length=settings.render.focal_length,
        frame_width=settings.camera.width,
        frame_height=settings.camera.height,
    )
    depth_estimator = DepthEstimator(settings.asme)
    gesture_classifier = GestureClassifier(settings.gesture)
    state_machine = GestureStateMachine(settings.state_machine)

    coordinate_filter: Optional[Any] = None
    if _HAS_COORDINATE_FILTER:
        try:
            coordinate_filter = CoordinateFilter(settings.asme)
        except Exception as e:
            logger.warning(f"Failed to initialize CoordinateFilter: {e}")

    # -- Frame timing --
    fps = 0.0
    frame_time_accumulator = 0.0
    frame_count_for_fps = 0
    fps_update_interval = 0.5  # seconds
    last_frame_time = time.perf_counter()

    try:
        with VideoStream(
            device_index=settings.camera.device_index,
            width=settings.camera.width,
            height=settings.camera.height,
        ) as stream, HandTracker(config=settings.tracker) as tracker:

            logger.info("Pipeline active. Press Q or ESC to exit.")

            while True:
                # 1. Read frame
                with benchmark("camera_read"):
                    success, frame = stream.read_frame()
                
                if not success or frame is None:
                    logger.warning("Dropped frame - skipping.")
                    continue

                frame_height, frame_width = frame.shape[:2]
                current_time = time.perf_counter()
                
                # Per-frame FPS calculation
                frame_delta = current_time - last_frame_time
                last_frame_time = current_time
                
                if frame_delta > 0:
                    instant_fps = 1.0 / frame_delta
                    frame_time_accumulator += instant_fps
                    frame_count_for_fps += 1
                
                # Update displayed FPS periodically
                if current_time - last_frame_time >= fps_update_interval:
                    if frame_count_for_fps > 0:
                        fps = frame_time_accumulator / frame_count_for_fps
                    frame_time_accumulator = 0.0
                    frame_count_for_fps = 0

                # 2. Detect hand
                with benchmark("hand_tracking"):
                    results = tracker.process_frame(frame)
                
                hand_landmarks = (
                    results.hand_landmarks[0]
                    if results and results.hand_landmarks and len(results.hand_landmarks) > 0
                    else None
                )

                # 3. No hand detected
                if hand_landmarks is None:
                    # Debounce: only reset if not already IDLE
                    if (hasattr(state_machine, 'reset') and 
                        hasattr(state_machine, 'current_state') and
                        state_machine.current_state != BoxState.IDLE):
                        state_machine.reset()
                    depth_estimator.reset()

                    hud_data = HUDData(
                        tracking=False,
                        fps=fps,
                    )
                    annotated = hud_renderer.render(frame.copy(), hud_data)
                    cv2.imshow(settings.window_name, annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        logger.info("Exit signal received.")
                        break
                    continue

                # 4. Gesture classification
                with benchmark("gesture_classification"):
                    try:
                        gesture = gesture_classifier.classify(hand_landmarks)
                    except Exception as e:
                        logger.warning(f"Gesture classification failed: {e}")
                        gesture = GestureType.NONE

                # Extract hand position for display
                hand_px, hand_py = _extract_hand_position(
                    hand_landmarks, frame_width, frame_height
                )

                # 5. Depth estimation
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

                # 6. Coordinate smoothing (optional)
                hand_virtual_x = (hand_px - frame_width / 2) / settings.render.focal_length
                hand_virtual_y = (hand_py - frame_height / 2) / settings.render.focal_length
                depth_for_filter = depth if depth is not None else 1.0

                smoothed_x, smoothed_y, smoothed_depth = hand_virtual_x, hand_virtual_y, depth_for_filter
                if coordinate_filter is not None:
                    with benchmark("coordinate_smoothing"):
                        try:
                            smoothed_x, smoothed_y, smoothed_depth = coordinate_filter(
                                hand_virtual_x, hand_virtual_y, depth_for_filter, current_time
                            )
                        except Exception as e:
                            logger.debug(f"Coordinate smoothing failed: {e}")

                # 7. State machine update
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

                # 8-9. Project 3D cube if a box exists
                with benchmark("projection_rendering"):
                    if (current_state != BoxState.IDLE and 
                        hasattr(state_machine, 'current_box_center') and 
                        state_machine.current_box_center is not None):
                        try:
                            center = state_machine.current_box_center
                            half_extents = getattr(
                                state_machine, 'current_box_half_extents',
                                settings.state_machine.default_box_half_extents
                            )
                            projected = projector.project_box(center, half_extents)
                            wireframe_renderer.draw_box(frame, projected)
                        except InvalidDepthError as e:
                            logger.debug(f"Projection skipped - box behind camera: {e}")
                        except Exception as e:
                            logger.warning(f"Wireframe rendering failed: {e}")

                # 10. Render HUD
                with benchmark("hud_render"):
                    hud_data = HUDData(
                        tracking=True,
                        gesture=gesture,
                        state=current_state,
                        fps=fps,
                        depth=smoothed_depth,
                        hand_position=(hand_px, hand_py),
                    )
                    annotated = hud_renderer.render(frame, hud_data)

                # 11. Display
                cv2.imshow(settings.window_name, annotated)

                # 12. Exit check
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Exit signal received.")
                    break

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