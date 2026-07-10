"""
Unit tests for the HandTracker module.
No camera hardware required.
"""
import numpy as np
import pytest
from config import TrackerConfig
from hand_tracker import HandTracker


@pytest.fixture
def tracker() -> HandTracker:
    config = TrackerConfig(static_image_mode=True)
    with HandTracker(config) as t:
        yield t


def test_process_frame_returns_result_on_valid_input(tracker: HandTracker) -> None:
    """process_frame must not raise on a valid black BGR frame."""
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(dummy_bgr)
    assert result is not None


def test_process_frame_blank_has_no_hands(tracker: HandTracker) -> None:
    """A blank frame should yield zero hand detections."""
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(dummy_bgr)
    assert len(result.hand_landmarks) == 0


def test_draw_landmarks_handles_no_detection(tracker: HandTracker) -> None:
    """draw_landmarks must return the frame unmodified when no hand is detected."""
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    empty_result = type("Result", (), {"hand_landmarks": []})()
    output = tracker.draw_landmarks(dummy_bgr, empty_result)
    np.testing.assert_array_equal(dummy_bgr, output)


def test_process_frame_raises_without_context() -> None:
    """process_frame must raise RuntimeError if called without 'with' block."""
    config = TrackerConfig(static_image_mode=True)
    tracker = HandTracker(config)
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        tracker.process_frame(dummy_bgr)