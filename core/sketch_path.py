"""
Sketch path data management module.

Owns freehand stroke data for mid-air sketching. Pure data management —
no rendering, no OpenCV dependency. A stroke is a sequence of 3D points
in virtual projection space (the same coordinate space used by
PerspectiveProjector), collected while the user holds a pinch gesture
in Sketch Mode.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SketchStroke:
    """A single continuous freehand stroke: an ordered list of 3D points."""
    points: List[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)


class SketchManager:
    """
    Manages the lifecycle of freehand sketch strokes.

    Usage:
        manager = SketchManager(min_point_distance=0.01)
        manager.start_stroke()
        manager.add_point(np.array([0.1, 0.2, 5.0]))
        manager.add_point(np.array([0.11, 0.2, 5.0]))
        manager.end_stroke()
    """

    def __init__(self, min_point_distance: float) -> None:
        """
        Args:
            min_point_distance: Minimum 3D distance a new point must be
                from the last recorded point in the active stroke to be
                accepted. Prevents flooding a stroke with near-duplicate
                points when the hand is nearly still.

        Raises:
            ValueError: If min_point_distance is negative.
        """
        if min_point_distance < 0:
            raise ValueError("min_point_distance must be non-negative.")

        self._min_point_distance = min_point_distance
        self._strokes: List[SketchStroke] = []
        self._active_stroke: Optional[SketchStroke] = None

    def start_stroke(self) -> None:
        """
        Begins a new stroke. If a stroke is already active, it is
        implicitly ended first (its accumulated points are kept).
        """
        if self._active_stroke is not None:
            self.end_stroke()

        new_stroke = SketchStroke()
        self._strokes.append(new_stroke)
        self._active_stroke = new_stroke

    def add_point(self, point: np.ndarray) -> None:
        """
        Appends a point to the active stroke, if far enough from the
        last point. Silently ignored if no stroke is active (call
        start_stroke() first) rather than raising, since this is
        expected to be called every frame during a pinch-and-move
        gesture where timing races against mode toggles are normal.

        Args:
            point: A 3-element array [x, y, z] in virtual projection space.
        """
        if self._active_stroke is None:
            return

        if self._active_stroke.points:
            last = self._active_stroke.points[-1]
            distance = math.dist(point.tolist(), last.tolist())
            if distance < self._min_point_distance:
                return

        self._active_stroke.points.append(point)

    def end_stroke(self) -> None:
        """Finalizes the active stroke. Safe to call with no active stroke."""
        self._active_stroke = None

    def clear(self) -> None:
        """Removes all strokes, active and completed."""
        self._strokes.clear()
        self._active_stroke = None

    @property
    def strokes(self) -> List[SketchStroke]:
        """All strokes, including the active one if present."""
        return self._strokes

    @property
    def active_stroke(self) -> Optional[SketchStroke]:
        """The currently in-progress stroke, or None if not sketching."""
        return self._active_stroke