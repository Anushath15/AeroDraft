"""
Unit tests for the SketchManager and SketchStroke classes.
"""
import numpy as np
import pytest
from core.sketch_path import SketchManager, SketchStroke


@pytest.fixture
def manager() -> SketchManager:
    return SketchManager(min_point_distance=0.01)


def test_no_active_stroke_initially(manager: SketchManager) -> None:
    """A freshly constructed manager must have no active stroke and no strokes."""
    assert manager.active_stroke is None
    assert manager.strokes == []


def test_start_stroke_creates_active_stroke(manager: SketchManager) -> None:
    """start_stroke() must create an empty active stroke."""
    manager.start_stroke()
    assert manager.active_stroke is not None
    assert len(manager.active_stroke) == 0
    assert len(manager.strokes) == 1


def test_add_point_appends_to_active_stroke(manager: SketchManager) -> None:
    """add_point() must append to the currently active stroke."""
    manager.start_stroke()
    manager.add_point(np.array([0.1, 0.2, 5.0]))
    assert len(manager.active_stroke) == 1


def test_add_point_without_active_stroke_is_ignored(manager: SketchManager) -> None:
    """add_point() with no active stroke must not raise, just no-op."""
    manager.add_point(np.array([0.1, 0.2, 5.0]))  # no start_stroke() called
    assert manager.strokes == []


def test_points_closer_than_min_distance_are_rejected(manager: SketchManager) -> None:
    """A point too close to the last recorded point must be silently dropped."""
    manager.start_stroke()
    manager.add_point(np.array([0.0, 0.0, 5.0]))
    manager.add_point(np.array([0.001, 0.0, 5.0]))  # distance 0.001 < 0.01 threshold
    assert len(manager.active_stroke) == 1


def test_points_farther_than_min_distance_are_accepted(manager: SketchManager) -> None:
    """A point far enough from the last recorded point must be appended."""
    manager.start_stroke()
    manager.add_point(np.array([0.0, 0.0, 5.0]))
    manager.add_point(np.array([1.0, 0.0, 5.0]))  # distance 1.0 >> 0.01 threshold
    assert len(manager.active_stroke) == 2


def test_end_stroke_clears_active_stroke_reference(manager: SketchManager) -> None:
    """end_stroke() must clear the active stroke pointer but keep the stroke in strokes."""
    manager.start_stroke()
    manager.add_point(np.array([0.0, 0.0, 5.0]))
    manager.end_stroke()
    assert manager.active_stroke is None
    assert len(manager.strokes) == 1
    assert len(manager.strokes[0]) == 1


def test_end_stroke_without_active_stroke_is_safe(manager: SketchManager) -> None:
    """end_stroke() with no active stroke must not raise."""
    manager.end_stroke()  # no start_stroke() called
    assert manager.active_stroke is None


def test_start_stroke_while_active_implicitly_ends_previous(manager: SketchManager) -> None:
    """Calling start_stroke() while a stroke is active must end the previous one first."""
    manager.start_stroke()
    manager.add_point(np.array([0.0, 0.0, 5.0]))
    manager.start_stroke()  # implicitly ends the first stroke
    assert len(manager.strokes) == 2
    assert len(manager.strokes[0]) == 1  # first stroke preserved
    assert len(manager.active_stroke) == 0  # second stroke is fresh


def test_multiple_strokes_coexist_independently(manager: SketchManager) -> None:
    """Multiple completed strokes must retain their own independent point lists."""
    manager.start_stroke()
    manager.add_point(np.array([0.0, 0.0, 5.0]))
    manager.add_point(np.array([1.0, 0.0, 5.0]))
    manager.end_stroke()

    manager.start_stroke()
    manager.add_point(np.array([5.0, 5.0, 5.0]))
    manager.end_stroke()

    assert len(manager.strokes) == 2
    assert len(manager.strokes[0]) == 2
    assert len(manager.strokes[1]) == 1


def test_clear_removes_all_strokes(manager: SketchManager) -> None:
    """clear() must remove all strokes and reset active stroke to None."""
    manager.start_stroke()
    manager.add_point(np.array([0.0, 0.0, 5.0]))
    manager.clear()
    assert manager.strokes == []
    assert manager.active_stroke is None


def test_negative_min_point_distance_raises() -> None:
    """min_point_distance must be non-negative."""
    with pytest.raises(ValueError):
        SketchManager(min_point_distance=-1.0)


def test_zero_min_point_distance_accepts_all_points() -> None:
    """min_point_distance of 0 must accept even identical consecutive points."""
    manager = SketchManager(min_point_distance=0.0)
    manager.start_stroke()
    manager.add_point(np.array([1.0, 1.0, 5.0]))
    manager.add_point(np.array([1.0, 1.0, 5.0]))  # identical point
    assert len(manager.active_stroke) == 2