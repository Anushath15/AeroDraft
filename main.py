"""
AeroDraft — Master entry point.
Orchestrates the complete pipeline:
    Camera → Hand Tracking → Gesture Classification → State Machine
    → Depth Estimation → Coordinate Smoothing → 3D Projection
    → Wireframe Rendering → HUD Overlay → Display
"""
from __future__ import annotations

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

# Optional coordinate filter — may not exist in all repository versions
try:
    from core.coordinate_filter import CoordinateFilter
    _HAS_COORDINATE_FILTER = True
except ImportError:
    _HAS_COORDINATE_FILTER = False
    logger.warning("CoordinateFilter not found — skipping coordinate smoothing.")


def _extract_hand_position(
    landmarks: Any, frame_width: int, frame_height: int
) -> Tuple[int, int]:
    """
    Converts normalized wrist landmark to pixel coordinates.
    
    Uses landmark 0 (wrist) as the hand position reference.
    """
    wrist = landmarks[0]
    return int(wrist.x * frame_width), int(wrist.y * frame_height)


def _compute_fps(frame_count: int, start_time: float) -> float:
    """Computes average FPS over the elapsed interval."""
    elapsed = time.perf_counter() - start_time
    if elapsed <= 0:
        return 0.0
    return frame_count / elapsed


def main() -> None:
    """Runs the integrated AeroDraft application loop."""
    logger.info("AeroDraft starting — Phase 10 (Full Integration)")

    # ── Initialize subsystems ──────────────────────────────────────
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
            coordinate_filter = CoordinateFilter()
        except Exception as e:
            logger.warning(f"Failed to initialize CoordinateFilter: {e}")

    # ── Frame timing ───────────────────────────────────────────────
    frame_count = 0
    fps = 0.0
    fps_update_interval = 0.5  # seconds
    last_fps_update = time.perf_counter()

    try:
        with VideoStream(
            device_index=settings.camera.device_index,
            width=settings.camera.width,
            height=settings.camera.height,
        ) as stream, HandTracker(config=settings.tracker) as tracker:

            logger.info("Pipeline active. Press Q or ESC to exit.")

            while True:
                # 1. Read frame
                success, frame = stream.read_frame()
                if not success or frame is None:
                    logger.warning("Dropped frame — skipping.")
                    continue

                frame_height, frame_width = frame.shape[:2]
                frame_count += 1
                timestamp = time.perf_counter()

                # Update FPS periodically
                if timestamp - last_fps_update >= fps_update_interval:
                    fps = _compute_fps(frame_count, last_fps_update)
                    frame_count = 0
                    last_fps_update = timestamp

                # 2. Detect hand
                results = tracker.process_frame(frame)
                hand_landmarks = (
                    results.hand_landmarks[0]
                    if results and results.hand_landmarks and len(results.hand_landmarks) > 0
                    else None
                )

                # 3. No hand detected
                if hand_landmarks is None:
                    state_machine.reset()
                    depth_estimator.reset()
                    if coordinate_filter is not None and hasattr(coordinate_filter, 'reset'):
                        coordinate_filter.reset()

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
                try:
                    gesture = gesture_classifier.classify(hand_landmarks)
                except Exception as e:
                    logger.warning(f"Gesture classification failed: {e}")
                    gesture = GestureType.NONE

                # Extract hand position for state machine and display
                hand_px, hand_py = _extract_hand_position(
                    hand_landmarks, frame_width, frame_height
                )

                # 6. Depth estimation
                try:
                    depth = depth_estimator.estimate(
                        hand_landmarks,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        timestamp=timestamp,
                    )
                except NoHandDetectedError as e:
                    logger.debug(f"Depth estimation skipped: {e}")
                    depth = None

                # 7. Coordinate smoothing (optional)
                smoothed_pos: Tuple[int, int] = (hand_px, hand_py)
                if coordinate_filter is not None:
                    try:
                        if hasattr(coordinate_filter, 'smooth'):
                            smoothed_pos = coordinate_filter.smooth(hand_px, hand_py)
                        elif callable(coordinate_filter):
                            smoothed_pos = coordinate_filter(hand_px, hand_py)
                    except Exception as e:
                        logger.debug(f"Coordinate smoothing failed: {e}")

                # 5. State machine update (uses gesture, hand position, depth)
                # Convert pixel position to normalized virtual space for state machine
                hand_virtual = np.array([
                    (smoothed_pos[0] - frame_width / 2) / settings.render.focal_length,
                    (smoothed_pos[1] - frame_height / 2) / settings.render.focal_length,
                ])
                depth_for_state = depth if depth is not None else 1.0

                try:
                    current_state = state_machine.update(
                        gesture=gesture,
                        hand_position=hand_virtual,
                        depth=depth_for_state,
                        timestamp=timestamp,
                    )
                except Exception as e:
                    logger.warning(f"State machine update failed: {e}")
                    current_state = state_machine.current_state

                # 8. Project 3D cube if a box exists
                if current_state != BoxState.IDLE and state_machine.current_box_center is not None:
                    try:
                        center = state_machine.current_box_center
                        half_extents = state_machine.current_box_half_extents
                        projected = projector.project_box(center, half_extents)
                        # 9. Render wireframe
                        wireframe_renderer.draw_box(frame, projected)
                    except InvalidDepthError as e:
                        logger.debug(f"Projection skipped — box behind camera: {e}")
                    except Exception as e:
                        logger.warning(f"Wireframe rendering failed: {e}")

                # 10. Render HUD
                hud_data = HUDData(
                    tracking=True,
                    gesture=gesture,
                    state=current_state,
                    fps=fps,
                    depth=depth,
                    hand_position=smoothed_pos,
                )
                annotated = hud_renderer.render(frame, hud_data)

                # 11. Display
                cv2.imshow(settings.window_name, annotated)

                # 12. Exit check
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Exit signal received.")
                    break

    except RuntimeError as e:
        logger.critical(f"Hardware error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected crash: {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        logger.info("AeroDraft shutdown complete.")


if __name__ == "__main__":
    main()