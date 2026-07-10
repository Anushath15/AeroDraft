"""
Hand tracking inference module.
Wraps MediaPipe Hands and isolates all AI logic
from camera I/O and rendering.
"""
from __future__ import annotations
from types import TracebackType
from typing import Any, Optional, Type

import mediapipe as mp
import numpy as np
from loguru import logger

from config import TrackerConfig


class HandTracker:
    """
    MediaPipe Hands wrapper.

    Handles RGB ingestion, landmark extraction, and
    optional landmark overlay drawing.
    Implements context manager for safe graph resource release.

    Usage:
        with HandTracker(config) as tracker:
            results = tracker.process_frame(rgb_frame)
    """

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands: Optional[Any] = None

    def __enter__(self) -> HandTracker:
        """Initializes the MediaPipe graph."""
        self._hands = self._mp_hands.Hands(
            static_image_mode=self._config.static_image_mode,
            max_num_hands=self._config.max_num_hands,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        logger.info("MediaPipe Hands initialized.")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Closes the MediaPipe graph and releases native resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
            logger.info("MediaPipe Hands graph closed.")

    def process_frame(self, rgb_image: np.ndarray) -> Any:
        """
        Runs hand landmark inference on an RGB frame.

        Args:
            rgb_image: Input frame in RGB color space.

        Returns:
            MediaPipe results NamedTuple (multi_hand_landmarks, etc.)

        Raises:
            RuntimeError: If called before entering the context manager.
        """
        if self._hands is None:
            raise RuntimeError("HandTracker not initialized. Use a 'with' block.")

        rgb_image.flags.writeable = False
        results = self._hands.process(rgb_image)
        rgb_image.flags.writeable = True
        return results

    def draw_landmarks(self, bgr_image: np.ndarray, results: Any) -> np.ndarray:
        """
        Draws landmark skeleton onto a BGR frame.

        Args:
            bgr_image: Original BGR frame to annotate.
            results:   Results object from process_frame().

        Returns:
            Annotated BGR frame.
        """
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    bgr_image,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                )
        return bgr_image