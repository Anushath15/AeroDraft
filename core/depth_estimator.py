"""
Depth estimation module.

Implements the Adaptive Spatial Mapping Engine's (ASME) core technique:
Relative Pixel Scaling. Since we have no LiDAR or stereo camera, we
exploit a physical invariant — the human hand has fixed bone lengths.
As the hand moves closer to the camera, the pixel distance between
rigid anatomical landmarks increases; as it moves away, it shrinks.

We use the wrist (landmark 0) and index finger MCP (landmark 5) as the
reference pair because this segment stays rigid across all common hand
gestures (unlike fingertip-to-fingertip distances, which fold during
pinch/grab gestures).
"""
from __future__ import annotations
import math
from typing import Any, Sequence

from config import ASMEConfig
from core.one_euro_filter import OneEuroFilter


class NoHandDetectedError(Exception):
    """Raised when depth estimation is attempted with no valid landmarks."""


class DepthEstimator:
    """
    Converts 2D hand landmarks into a smoothed pseudo-Z depth value.

    Usage:
        estimator = DepthEstimator(settings.asme)
        depth = estimator.estimate(landmarks, frame_width=640, frame_height=480, timestamp=t)
    """

    WRIST_IDX = 0
    INDEX_MCP_IDX = 5

    def __init__(self, config: ASMEConfig) -> None:
        """
        Args:
            config: ASME configuration (reference distance, filter params).
        """
        self._config = config
        self._filter = OneEuroFilter(
            min_cutoff=config.one_euro_min_cutoff,
            beta=config.one_euro_beta,
            d_cutoff=config.one_euro_d_cutoff,
        )

    def estimate(
        self,
        landmarks: Sequence[Any],
        frame_width: int,
        frame_height: int,
        timestamp: float,
    ) -> float:
        """
        Estimates smoothed pseudo-depth for a single detected hand.

        Args:
            landmarks: Sequence of 21 landmark objects (each with .x, .y
                normalized 0-1 attributes), as returned by MediaPipe
                HandLandmarker for one hand.
            frame_width: Width of the source frame in pixels.
            frame_height: Height of the source frame in pixels.
            timestamp: Monotonic timestamp in seconds for filter continuity.

        Returns:
            Smoothed pseudo-depth value. Larger values indicate the hand
            is closer to the camera; smaller values indicate it is farther.

        Raises:
            NoHandDetectedError: If landmarks is empty or too short to
                contain the required reference points.
        """
        if not landmarks or len(landmarks) <= max(self.WRIST_IDX, self.INDEX_MCP_IDX):
            raise NoHandDetectedError(
                "Landmarks sequence is empty or missing required points "
                f"(need at least {self.INDEX_MCP_IDX + 1} landmarks)."
            )

        wrist = landmarks[self.WRIST_IDX]
        index_mcp = landmarks[self.INDEX_MCP_IDX]

        pixel_dist = self._pixel_distance(
            wrist, index_mcp, frame_width, frame_height
        )

        if pixel_dist <= 0:
            raise NoHandDetectedError(
                "Computed pixel distance is zero or negative — invalid landmarks."
            )

        raw_depth = self._config.reference_distance_px / pixel_dist

        return self._filter(raw_depth, timestamp)

    @staticmethod
    def _pixel_distance(
        point_a: Any, point_b: Any, frame_width: int, frame_height: int
    ) -> float:
        """
        Computes Euclidean pixel distance between two normalized landmarks.

        Args:
            point_a: Landmark with normalized .x, .y attributes.
            point_b: Landmark with normalized .x, .y attributes.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            Euclidean distance in pixels.
        """
        ax, ay = point_a.x * frame_width, point_a.y * frame_height
        bx, by = point_b.x * frame_width, point_b.y * frame_height
        return math.hypot(bx - ax, by - ay)

    def reset(self) -> None:
        """Resets the internal smoothing filter (e.g. on hand loss/regain)."""
        self._filter.reset()