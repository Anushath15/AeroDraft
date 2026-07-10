"""
Unit tests for the PerspectiveProjector class.
"""
import numpy as np
import pytest
from engine.projection import PerspectiveProjector, InvalidDepthError


@pytest.fixture
def projector() -> PerspectiveProjector:
    return PerspectiveProjector(focal_length=500.0, frame_width=640, frame_height=480)


def test_center_point_projects_to_frame_center(projector: PerspectiveProjector) -> None:
    """A point at (0, 0, z) must project to the exact frame center."""
    result = projector.project_point(np.array([0.0, 0.0, 5.0]))
    np.testing.assert_allclose(result, [320.0, 240.0])


def test_known_offset_point_projects_to_expected_pixel(projector: PerspectiveProjector) -> None:
    """
    Hand-computed reference: point (1.0, 0.0, 5.0) with focal_length=500
    should project to screen_x = (1.0/5.0)*500 + 320 = 420.0
    """
    result = projector.project_point(np.array([1.0, 0.0, 5.0]))
    np.testing.assert_allclose(result, [420.0, 240.0])


def test_farther_depth_shrinks_apparent_offset(projector: PerspectiveProjector) -> None:
    """A point farther from the camera must project closer to center (smaller apparent size)."""
    near_point = projector.project_point(np.array([1.0, 0.0, 2.0]))
    far_point = projector.project_point(np.array([1.0, 0.0, 10.0]))

    near_offset = abs(near_point[0] - projector.center_x)
    far_offset = abs(far_point[0] - projector.center_x)

    assert far_offset < near_offset


def test_zero_depth_raises_invalid_depth_error(projector: PerspectiveProjector) -> None:
    """z = 0 must raise InvalidDepthError, not divide-by-zero silently."""
    with pytest.raises(InvalidDepthError):
        projector.project_point(np.array([0.0, 0.0, 0.0]))


def test_negative_depth_raises_invalid_depth_error(projector: PerspectiveProjector) -> None:
    """z < 0 (behind camera) must raise InvalidDepthError."""
    with pytest.raises(InvalidDepthError):
        projector.project_point(np.array([0.0, 0.0, -1.0]))


def test_invalid_focal_length_raises() -> None:
    """focal_length must be strictly positive."""
    with pytest.raises(ValueError):
        PerspectiveProjector(focal_length=0.0, frame_width=640, frame_height=480)


def test_project_box_returns_eight_points(projector: PerspectiveProjector) -> None:
    """project_box must always return exactly 8 points."""
    result = projector.project_box(
        center=np.array([0.0, 0.0, 5.0]),
        half_extents=np.array([0.5, 0.5, 0.5]),
    )
    assert result.shape == (8, 2)


def test_project_box_vertex_order_is_consistent(projector: PerspectiveProjector) -> None:
    """
    Verifies the documented vertex order: vertex 0 is the
    (-x, -y, -z) corner relative to center, so it must project to
    a smaller x and smaller y than vertex 2 (+x, +y, -z corner).
    """
    result = projector.project_box(
        center=np.array([0.0, 0.0, 5.0]),
        half_extents=np.array([1.0, 1.0, 1.0]),
    )
    vertex_0 = result[0]  # (-x, -y, -z)
    vertex_2 = result[2]  # (+x, +y, -z)

    assert vertex_0[0] < vertex_2[0]
    assert vertex_0[1] < vertex_2[1]


def test_project_box_raises_if_any_vertex_behind_camera(projector: PerspectiveProjector) -> None:
    """If box depth extent puts any vertex at z <= 0, must raise InvalidDepthError."""
    with pytest.raises(InvalidDepthError):
        projector.project_box(
            center=np.array([0.0, 0.0, 0.5]),
            half_extents=np.array([0.5, 0.5, 1.0]),  # half_depth=1.0 pushes near face to z=-0.5
        )