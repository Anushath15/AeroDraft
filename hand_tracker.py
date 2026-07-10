"""Hand tracking module.

Wraps MediaPipe's Hands solution to provide a single-responsibility,
testable interface: given a BGR frame, return structured hand landmark
detections. This module performs no rendering, filtering, or spatial
mapping — those are separate concerns handled by other modules.
"""

from __future__ import annotations

from types import TracebackType
from typing import List, Optional, Type

import mediapipe as mp
import numpy as np
from loguru import logger

from aerodraft.config.settings import HandTrackerSettings
from aerodraft.core.data_types import (
    DetectedHand,
    HandDetectionResult,
    HandLandmarkPoint,
)
from aerodraft.core.exceptions import (
    HandTrackerInitializationError,
    HandTrackerProcessingError,
)

# mediapipe.solutions is bound dynamically inside mediapipe/__init__.py rather
# than being a real importable submodule path, so it must be accessed as an
# attribute of the top-level package (not via `import mediapipe.solutions...`).
_mp_hands = mp.solutions.hands


class HandTracker:
    """Detects hand landmarks in video frames using MediaPipe Hands.

    Attributes:
        settings: Hand tracker configuration (max hands, confidence
            thresholds, etc.).
    """

    def __init__(self, settings: HandTrackerSettings) -> None:
        """Initializes the hand tracker without loading the model.

        Args:
            settings: Hand tracker configuration.
        """
        self.settings = settings
        self._hands: Optional[_mp_hands.Hands] = None

    def open(self) -> None:
        """Loads the underlying MediaPipe Hands model.

        Raises:
            HandTrackerInitializationError: If the model fails to load.
        """
        try:
            logger.debug("Initializing MediaPipe Hands model")
            self._hands = _mp_hands.Hands(
                static_image_mode=self.settings.static_image_mode,
                max_num_hands=self.settings.max_num_hands,
                min_detection_confidence=self.settings.min_detection_confidence,
                min_tracking_confidence=self.settings.min_tracking_confidence,
            )
            logger.info(
                "MediaPipe Hands model initialized "
                "(max_num_hands={}, min_detection_confidence={}, "
                "min_tracking_confidence={})",
                self.settings.max_num_hands,
                self.settings.min_detection_confidence,
                self.settings.min_tracking_confidence,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as domain error
            logger.error("Failed to initialize MediaPipe Hands model: {}", exc)
            raise HandTrackerInitializationError(
                "Could not initialize the MediaPipe Hands model."
            ) from exc

    def process(self, frame: np.ndarray) -> HandDetectionResult:
        """Detects hand landmarks in a single BGR frame.

        Args:
            frame: A BGR image as a ``numpy.ndarray``, typically produced
                by ``CameraCapture.read_frame``.

        Returns:
            A ``HandDetectionResult`` describing every hand detected in
            the frame. The result contains zero hands if none were found.

        Raises:
            HandTrackerProcessingError: If the tracker has not been
                opened, or if the input frame is invalid.
        """
        if self._hands is None:
            raise HandTrackerProcessingError(
                "Hand tracker is not open. Call open() before process()."
            )

        if frame is None or frame.size == 0:
            raise HandTrackerProcessingError(
                "Cannot process an empty or invalid frame."
            )

        frame_height, frame_width = frame.shape[:2]

        # MediaPipe expects RGB input; the rest of the pipeline
        # standardizes on BGR (matching OpenCV's default), so the
        # conversion happens here, at this module's boundary only.
        rgb_frame = frame[:, :, ::-1]

        try:
            results = self._hands.process(rgb_frame)
        except Exception as exc:  # noqa: BLE001 - re-raised as domain error
            logger.error("MediaPipe Hands processing failed: {}", exc)
            raise HandTrackerProcessingError(
                "MediaPipe Hands failed to process the given frame."
            ) from exc

        detected_hands = self._build_detected_hands(results)

        return HandDetectionResult(
            hands=detected_hands,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    @staticmethod
    def _build_detected_hands(results: object) -> List[DetectedHand]:
        """Converts raw MediaPipe results into ``DetectedHand`` objects.

        Args:
            results: The raw output of
                ``mediapipe.solutions.hands.Hands.process``.

        Returns:
            A list of ``DetectedHand`` instances, empty if no hands were
            detected in the frame.
        """
        detected_hands: List[DetectedHand] = []

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return detected_hands

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            landmark_points = [
                HandLandmarkPoint(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks.landmark
            ]
            classification = handedness.classification[0]
            detected_hands.append(
                DetectedHand(
                    handedness=classification.label,
                    confidence=classification.score,
                    landmarks=landmark_points,
                )
            )

        return detected_hands

    def close(self) -> None:
        """Releases the underlying MediaPipe Hands model, if loaded. Safe
        to call multiple times or when the model was never opened."""
        if self._hands is not None:
            self._hands.close()
            logger.info("MediaPipe Hands model released")
            self._hands = None

    def __enter__(self) -> "HandTracker":
        """Opens the hand tracker for use in a ``with`` block."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Releases the hand tracker when leaving a ``with`` block."""
        self.close()