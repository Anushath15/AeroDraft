"""
Unit tests for the WireframeRenderer class.
"""
import numpy as np
import pytest
from engine.wireframe_renderer import WireframeRenderer


@pytest.fixture
def renderer() -> WireframeRenderer:
    return WireframeRenderer(color=(0, 255, 255), thickness=2)


@pytest.fixture
def sample_box_points() -> np.ndarray:
    """A plausible 8-point box roughly centered in a 640x480 frame."""
    return np.array([
        [270, 190], [370, 190], [370, 290], [270, 290],
        [280, 200], [360, 200], [360, 280], [280, 280],
    ], dtype=np.float64)


def test_draw_box_modifies_frame(renderer: WireframeRenderer, sample_box_points: np.ndarray) -> None:
    """Drawing a box onto a blank frame must change some pixel values."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = renderer.draw_box(frame, sample_box_points)
    assert np.any(result != 0)


def test_draw_box_returns_same_frame_object(renderer: WireframeRenderer, sample_box_points: np.ndarray) -> None:
    """draw_box modifies in place and returns the same array object."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = renderer.draw_box(frame, sample_box_points)
    assert result is frame


def test_draw_box_rejects_wrong_point_count(renderer: WireframeRenderer) -> None:
    """draw_box must raise ValueError if given anything other than 8 points."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    wrong_points = np.array([[100, 100], [200, 200]], dtype=np.float64)
    with pytest.raises(ValueError):
        renderer.draw_box(frame, wrong_points)


def test_drawn_line_uses_configured_color(sample_box_points: np.ndarray) -> None:
    """A pixel on a drawn edge should match the configured BGR color."""
    color = (0, 255, 255)
    renderer = WireframeRenderer(color=color, thickness=3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer.draw_box(frame, sample_box_points)

    # Sample the midpoint of the bottom-front edge (vertices 0->1)
    mid_x = int((sample_box_points[0][0] + sample_box_points[1][0]) / 2)
    mid_y = int(sample_box_points[0][1])
    pixel = frame[mid_y, mid_x]

    assert tuple(pixel) == color


def test_invalid_thickness_raises() -> None:
    """thickness must be strictly positive."""
    with pytest.raises(ValueError):
        WireframeRenderer(color=(255, 255, 255), thickness=0)


def test_all_points_offscreen_does_not_crash(renderer: WireframeRenderer) -> None:
    """A box entirely outside frame bounds should not raise, just log/no-op."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    offscreen_points = np.array([
        [-100, -100], [-50, -100], [-50, -50], [-100, -50],
        [-100, -100], [-50, -100], [-50, -50], [-100, -50],
    ], dtype=np.float64)
    result = renderer.draw_box(frame, offscreen_points)
    assert result is not None