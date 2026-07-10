"""
Perspective projection module.

Converts 3D points (in a virtual coordinate space) into 2D screen
pixel coordinates using a simple pinhole camera model. No OpenGL,
no matrix libraries beyond NumPy.
"""
from __future__ import annotations
import numpy as np


class InvalidDepthError(ValueError):
    """Raised when a point's depth is at or behind the camera plane."""


class PerspectiveProjector:
    """
    Projects 3D points onto a 2D frame using pinhole perspective projection.

    The projection formula:
        screen_x = (x / z) * focal_length + center_x
        screen_y = (y / z) * focal_length + center_y

    Usage:
        projector = PerspectiveProjector(focal_length=500.0, frame_width=640, frame_height=480)
        screen_point = projector.project_point(np.array([0.0, 0.0, 5.0]))
    """

    def __init__(self, focal_length: float, frame_width: int, frame_height: int) -> None:
        """
        Args:
            focal_length: Virtual focal length in pixels. Larger values
                produce a more zoomed-in, narrower field of view.
            frame_width: Width of the target frame in pixels.
            frame_height: Height of the target frame in pixels.

        Raises:
            ValueError: If focal_length is not positive.
        """
        if focal_length <= 0:
            raise ValueError("focal_length must be positive.")

        self.focal_length = focal_length
        self.center_x = frame_width / 2.0
        self.center_y = frame_height / 2.0

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """
        Projects a single 3D point to 2D screen coordinates.

        Args:
            point_3d: A 3-element array [x, y, z] in virtual camera space.
                z must be strictly positive (in front of the camera).

        Returns:
            A 2-element array [screen_x, screen_y] in pixel coordinates.

        Raises:
            InvalidDepthError: If z <= 0.
        """
        x, y, z = point_3d

        if z <= 0:
            raise InvalidDepthError(
                f"Point depth must be positive (in front of camera); got z={z}."
            )

        screen_x = (x / z) * self.focal_length + self.center_x
        screen_y = (y / z) * self.focal_length + self.center_y

        return np.array([screen_x, screen_y])

    def project_box(self, center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
        """
        Projects an axis-aligned 3D cuboid to 8 screen points.

        Vertex order (fixed, documented contract):
            0: (-x, -y, -z) bottom-face  1: (+x, -y, -z) bottom-face
            2: (+x, +y, -z) bottom-face  3: (-x, +y, -z) bottom-face
            4: (-x, -y, +z) top-face     5: (+x, -y, +z) top-face
            6: (+x, +y, +z) top-face     7: (-x, +y, +z) top-face
        (offsets relative to center; z-offset direction defines 'bottom' vs 'top'
        arbitrarily as the two parallel faces of the cuboid along its depth axis)

        Args:
            center: 3-element array [cx, cy, cz] — box center in virtual space.
            half_extents: 3-element array [hw, hh, hd] — half-width, half-height,
                half-depth of the box.

        Returns:
            An (8, 2) NumPy array of projected 2D screen points, in the
            vertex order documented above.

        Raises:
            InvalidDepthError: If any resulting vertex has z <= 0.
        """
        cx, cy, cz = center
        hw, hh, hd = half_extents

        offsets = np.array([
            [-hw, -hh, -hd],
            [ hw, -hh, -hd],
            [ hw,  hh, -hd],
            [-hw,  hh, -hd],
            [-hw, -hh,  hd],
            [ hw, -hh,  hd],
            [ hw,  hh,  hd],
            [-hw,  hh,  hd],
        ])

        vertices_3d = offsets + np.array([cx, cy, cz])

        projected = np.zeros((8, 2))
        for i, vertex in enumerate(vertices_3d):
            projected[i] = self.project_point(vertex)

        return projected