"""
Gesture classification module.

Given a single frame's 21 hand landmarks, classifies which gesture
(if any) the hand is currently making. Pure classification — no memory
of previous frames, no state. Reuses the same scale-invariant technique
as core.depth_estimator: all thresholds are ratios relative to a rigid
hand-size reference (wrist-to-middle-MCP distance), so classification
works identically regardless of hand distance from camera.
"""
from __future__ import annotations
import math
from enum import Enum, auto
from typing import Any, Sequence
from config import GestureConfig
from collections import deque
import numpy as np
from typing import Any


class GestureClassifier:
    def __init__(self, config: GestureConfig):
        self.config = config
        self.history = deque(maxlen=config.frame_history_len)
        self.is_pinched = False

    def is_pinch(self, wrist: Any, index_mcp: Any, index_tip: Any, thumb_tip: Any) -> bool:
        """Public API: Detects pinch gesture with hysteresis."""
        return self._detect_pinch(wrist, index_mcp, index_tip, thumb_tip)

    def _detect_pinch(self, wrist: Any, index_mcp: Any, index_tip: Any, thumb_tip: Any) -> bool:
        """Internal logic for hysteresis pinch detection."""
        pinch_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
        ref_dist = np.linalg.norm(np.array([wrist.x, wrist.y]) - np.array([index_mcp.x, index_mcp.y]))
        
        ratio = pinch_dist / (ref_dist + 1e-6)
        self.history.append(ratio)
        
        if len(self.history) < self.config.frame_history_len:
            return self.is_pinched

        avg_ratio = sum(self.history) / len(self.history)
        
        if not self.is_pinched and avg_ratio < self.config.pinch_start_threshold:
            self.is_pinched = True
        elif self.is_pinched and avg_ratio > self.config.pinch_release_threshold:
            self.is_pinched = False
            
        return self.is_pinched


class GestureType(Enum):
    """Enumerates the gestures this classifier can detect."""
    NONE = auto()
    PINCH = auto()
    OPEN_PALM = auto()
    FIST = auto()


class InvalidLandmarksError(Exception):
    """Raised when classification is attempted with missing/malformed landmarks."""


class GestureClassifier:
    """
    Classifies a single hand's gesture from its 21 landmarks.

    Usage:
        classifier = GestureClassifier(settings.gesture)
        gesture = classifier.classify(landmarks)
    """

    WRIST_IDX = 0
    MIDDLE_MCP_IDX = 9  # Hand-size reference point (rigid across gestures)
    THUMB_TIP_IDX = 4
    INDEX_TIP_IDX = 8

    # Fingertip indices used to determine curled vs. extended fingers
    FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

    REQUIRED_LANDMARK_COUNT = 21

    def __init__(self, config: GestureConfig) -> None:
        """
        Args:
            config: Gesture classification threshold configuration.
        """
        self._config = config

    def classify(self, landmarks: Sequence[Any]) -> GestureType:
        """
        Classifies the gesture being made by a single hand.

        Args:
            landmarks: Sequence of 21 landmark objects (each with .x, .y
                normalized 0-1 attributes), as returned by MediaPipe
                HandLandmarker for one hand.

        Returns:
            The detected GestureType. Returns GestureType.NONE if the
            hand shape doesn't clearly match any known gesture.

        Raises:
            InvalidLandmarksError: If landmarks is empty or has fewer
                than 21 points.
        """
        if not landmarks or len(landmarks) < self.REQUIRED_LANDMARK_COUNT:
            raise InvalidLandmarksError(
                f"Expected {self.REQUIRED_LANDMARK_COUNT} landmarks; "
                f"got {len(landmarks) if landmarks else 0}."
            )

        hand_size = self._distance(
            landmarks[self.WRIST_IDX], landmarks[self.MIDDLE_MCP_IDX]
        )

        if hand_size <= 0:
            raise InvalidLandmarksError(
                "Computed hand-size reference is zero or negative — invalid landmarks."
            )

        # Pinch check takes priority — it's the most specific gesture.
        pinch_distance = self._distance(
            landmarks[self.THUMB_TIP_IDX], landmarks[self.INDEX_TIP_IDX]
        )
        pinch_ratio = pinch_distance / hand_size
        if pinch_ratio < self._config.pinch_threshold_ratio:
            return GestureType.PINCH

        # Average fingertip-to-wrist distance, normalized by hand size,
        # distinguishes an open palm from a closed fist.
        avg_extension_ratio = self._average_fingertip_extension_ratio(
            landmarks, hand_size
        )

        if avg_extension_ratio < self._config.fist_threshold_ratio:
            return GestureType.FIST

        if avg_extension_ratio > self._config.open_palm_threshold_ratio:
            return GestureType.OPEN_PALM

        return GestureType.NONE

    def _average_fingertip_extension_ratio(
        self, landmarks: Sequence[Any], hand_size: float
    ) -> float:
        """Computes mean fingertip-to-wrist distance, normalized by hand size."""
        wrist = landmarks[self.WRIST_IDX]
        distances = [
            self._distance(wrist, landmarks[tip_idx])
            for tip_idx in self.FINGERTIP_INDICES
        ]
        return (sum(distances) / len(distances)) / hand_size

    @staticmethod
    def _distance(point_a: Any, point_b: Any) -> float:
        """Computes 2D Euclidean distance between two normalized landmarks."""
        return math.hypot(point_b.x - point_a.x, point_b.y - point_a.y)