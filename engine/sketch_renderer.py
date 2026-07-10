"""
Sketch rendering module.

Draws freehand sketch strokes as connected polylines from pre-projected
2D points. Pure rendering — no knowledge of 3D space, gestures, or
stroke lifecycle. Mirrors the separation-of-concerns pattern used by
WireframeRenderer and ObjectRenderer.
"""
from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np


class SketchRenderer:
    """
    Draws a set of freehand strokes onto a frame.

    Usage:
        renderer = SketchRenderer(color=(255, 0, 255), thickness=2)
        annotated = renderer.render(frame, projected_strokes)
    """

    def __init__(self, color: Tuple[int, int, int], thickness: int) -> None:
        """
        Args:
            color: BGR color for all sketch strokes.
            thickness: Line thickness in pixels.

        Raises:
            ValueError: If thickness is not positive.
        """
        if thickness <= 0:
            raise ValueError("thickness must be positive.")

        self.color = color
        self.thickness = thickness

    def render(
        self, frame: np.ndarray, projected_strokes: List[np.ndarray]
    ) -> np.ndarray:
        """
        Draws every stroke onto the frame as a connected polyline.

        Args:
            frame: BGR image to annotate (modified in place).
            projected_strokes: A list of (N, 2) arrays, one per stroke,
                each containing 2D pixel points in drawing order.
                Strokes with fewer than 2 points are skipped (nothing
                to connect). An empty list is a safe no-op.

        Returns:
            The annotated frame (same object as input).
        """
        for stroke_points in projected_strokes:
            if stroke_points.shape[0] < 2:
                continue

            points_int = stroke_points.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                frame, [points_int], isClosed=False,
                color=self.color, thickness=self.thickness
            )

        return frame