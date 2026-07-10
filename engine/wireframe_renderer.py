"""
Wireframe rendering module.

Draws a 12-edge cuboid wireframe from 8 projected 2D points using
cv2.polylines. Pure rendering — no projection math, no AI logic.
"""
from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np
from loguru import logger


class WireframeRenderer:
    """
    Draws a wireframe box onto a frame from pre-projected 2D points.

    Usage:
        renderer = WireframeRenderer(color=(0, 255, 255), thickness=2)
        annotated_frame = renderer.draw_box(frame, projected_points)
    """

    # Vertex indices matching PerspectiveProjector.project_box's documented order.
    BOTTOM_FACE = [0, 1, 2, 3]
    TOP_FACE = [4, 5, 6, 7]
    VERTICAL_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]

    def __init__(self, color: Tuple[int, int, int], thickness: int) -> None:
        """
        Args:
            color: BGR color tuple for the wireframe lines.
            thickness: Line thickness in pixels.

        Raises:
            ValueError: If thickness is not positive.
        """
        if thickness <= 0:
            raise ValueError("thickness must be positive.")

        self.color = color
        self.thickness = thickness

    def draw_box(self, frame: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        """
        Draws a 12-edge wireframe cuboid onto the given frame.

        Args:
            frame: BGR frame to draw onto (modified in place and returned).
            projected_points: An (8, 2) array of 2D screen points, in the
                vertex order defined by PerspectiveProjector.project_box.

        Returns:
            The annotated frame (same object as input, modified in place).

        Raises:
            ValueError: If projected_points does not have shape (8, 2).
        """
        if projected_points.shape != (8, 2):
            raise ValueError(
                f"projected_points must have shape (8, 2); got {projected_points.shape}."
            )

        points_int = projected_points.astype(np.int32)

        frame_h, frame_w = frame.shape[:2]
        if self._all_points_offscreen(points_int, frame_w, frame_h):
            logger.trace("Wireframe box is fully outside frame bounds.")

        bottom_loop = points_int[self.BOTTOM_FACE].reshape(-1, 1, 2)
        top_loop = points_int[self.TOP_FACE].reshape(-1, 1, 2)

        cv2.polylines(frame, [bottom_loop], isClosed=True, color=self.color, thickness=self.thickness)
        cv2.polylines(frame, [top_loop], isClosed=True, color=self.color, thickness=self.thickness)

        for start_idx, end_idx in self.VERTICAL_EDGES:
            start_point = tuple(points_int[start_idx])
            end_point = tuple(points_int[end_idx])
            cv2.line(frame, start_point, end_point, self.color, self.thickness)

        return frame

    @staticmethod
    def _all_points_offscreen(points: np.ndarray, frame_w: int, frame_h: int) -> bool:
        """Checks whether every point falls outside the frame bounds."""
        for x, y in points:
            if 0 <= x < frame_w and 0 <= y < frame_h:
                return False
        return True