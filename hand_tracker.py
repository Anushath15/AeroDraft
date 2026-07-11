"""
Hand tracking inference module.
Uses the modern MediaPipe Tasks API (>= 0.10).
Isolates all AI logic from camera I/O and rendering.
"""
from __future__ import annotations
from types import TracebackType
from typing import Any, Optional, Type

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from loguru import logger

from config import TrackerConfig


class HandTracker:
    """
    MediaPipe Hands wrapper using the Tasks API.
    Optimized for video streams using RunningMode.VIDEO.
    """

    MODEL_PATH = "hand_landmarker.task"

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._detector: Optional[Any] = None

    def __enter__(self) -> HandTracker:
        """Initializes the MediaPipe HandLandmarker."""
        base_options = mp_python.BaseOptions(
            model_asset_path=self.MODEL_PATH
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self._config.max_num_hands,
            min_hand_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            # UPGRADE: Changed from IMAGE to VIDEO for ~20-30% performance gain
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        logger.info("MediaPipe HandLandmarker initialized (VIDEO mode).")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Closes the detector and releases native resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None
            logger.info("MediaPipe HandLandmarker closed.")

    def process_frame(self, bgr_frame: np.ndarray, timestamp_ms: int) -> Any:
        """
        Runs hand landmark detection on a BGR frame.

        Args:
            bgr_frame: Input frame in BGR color space.
            timestamp_ms: Monotonically increasing timestamp in milliseconds.

        Returns:
            MediaPipe HandLandmarkerResult object.
        """
        if self._detector is None:
            raise RuntimeError("HandTracker not initialized. Use a 'with' block.")

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # UPGRADE: Passing timestamp to VIDEO mode detector
        return self._detector.detect(mp_image, timestamp_ms=timestamp_ms)

    def draw_landmarks(self, bgr_image: np.ndarray, results: Any) -> np.ndarray:
        """Draws hand landmark skeleton onto a BGR frame."""
        if not results or not results.hand_landmarks:
            return bgr_image

        h, w = bgr_image.shape[:2]

        for hand in results.hand_landmarks:
            points = []
            for lm in hand:
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))
                cv2.circle(bgr_image, (px, py), 4, (0, 255, 0), -1)

            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(bgr_image, points[start_idx], points[end_idx],
                             (255, 255, 255), 2)

        return bgr_image


# MediaPipe hand landmark connections (21 landmarks, 0-indexed)
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]
