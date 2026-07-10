"""
Unit tests for the HandTracker module.
No camera or display hardware required.
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
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(dummy_rgb)
    assert result is not None


def test_draw_landmarks_handles_no_detection(tracker: HandTracker) -> None:
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    empty_result = type('Result', (), {'multi_hand_landmarks': None})()
    output = tracker.draw_landmarks(dummy_bgr, empty_result)
    np.testing.assert_array_equal(dummy_bgr, output)


def test_process_frame_raises_without_context() -> None:
    config = TrackerConfig(static_image_mode=True)
    tracker = HandTracker(config)
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        tracker.process_frame(dummy_rgb)
