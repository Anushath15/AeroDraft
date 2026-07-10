"""
Unit tests for the SketchRenderer class.
"""
import numpy as np
import pytest
from engine.sketch_renderer import SketchRenderer


@pytest.fixture
def renderer() -> SketchRenderer:
    return SketchRenderer(color=(255, 0, 255), thickness=2)


def test_render_draws_single_stroke(renderer: SketchRenderer) -> None:
    """A single multi-point stroke must be drawn onto the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    stroke = np.array([[100, 100], [200, 150], [300, 100]], dtype=np.float64)
    result = renderer.render(frame, [stroke])
    assert np.any(result != 0)


def test_render_draws_multiple_strokes_independently(renderer: SketchRenderer) -> None:
    """Multiple strokes must all be drawn without interfering with each other."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    stroke1 = np.array([[50, 50], [100, 50]], dtype=np.float64)
    stroke2 = np.array([[400, 400], [450, 420]], dtype=np.float64)
    result = renderer.render(frame, [stroke1, stroke2])
    assert np.any(result != 0)


def test_render_empty_stroke_list_is_noop(renderer: SketchRenderer) -> None:
    """An empty list of strokes must not raise and must leave the frame unchanged."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = renderer.render(frame, [])
    assert np.all(result == 0)


def test_render_skips_single_point_stroke(renderer: SketchRenderer) -> None:
    """A stroke with fewer than 2 points has nothing to connect and must be skipped safely."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    single_point_stroke = np.array([[100, 100]], dtype=np.float64)
    result = renderer.render(frame, [single_point_stroke])
    assert np.all(result == 0)  # nothing drawn, no crash


def test_render_returns_same_frame_object(renderer: SketchRenderer) -> None:
    """render() must modify and return the same frame object (in-place)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    stroke = np.array([[10, 10], [20, 20]], dtype=np.float64)
    result = renderer.render(frame, [stroke])
    assert result is frame


def test_invalid_thickness_raises() -> None:
    """thickness must be strictly positive."""
    with pytest.raises(ValueError):
        SketchRenderer(color=(255, 255, 255), thickness=0)