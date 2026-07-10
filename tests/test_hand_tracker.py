"""Unit tests for aerodraft.core.hand_tracker.HandTracker.

These tests use real MediaPipe inference on deterministic, hand-free
frames (blank arrays), which is fast, requires no camera hardware, and
exercises the actual integration rather than mocking MediaPipe away
entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerodraft.config.settings import HandTrackerSettings
from aerodraft.core.exceptions import HandTrackerProcessingError
from aerodraft.core.hand_tracker import HandTracker


@pytest.fixture
def hand_tracker_settings() -> HandTrackerSettings:
    """Returns a deterministic HandTrackerSettings instance for tests."""
    return HandTrackerSettings(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=True,
    )


def test_process_without_open_raises(
    hand_tracker_settings: HandTrackerSettings,
) -> None:
    """process() should raise if open() was never called."""
    tracker = HandTracker(hand_tracker_settings)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(HandTrackerProcessingError):
        tracker.process(frame)


def test_process_rejects_empty_frame(
    hand_tracker_settings: HandTrackerSettings,
) -> None:
    """process() should raise on an empty/invalid frame even when open."""
    tracker = HandTracker(hand_tracker_settings)
    tracker.open()
    try:
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(HandTrackerProcessingError):
            tracker.process(empty_frame)
    finally:
        tracker.close()


def test_process_blank_frame_detects_no_hands(
    hand_tracker_settings: HandTrackerSettings,
) -> None:
    """A blank frame with no hand should yield zero detections."""
    tracker = HandTracker(hand_tracker_settings)
    tracker.open()
    try:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process(blank_frame)

        assert result.hand_count == 0
        assert result.frame_width == 640
        assert result.frame_height == 480
    finally:
        tracker.close()


def test_context_manager_opens_and_closes(
    hand_tracker_settings: HandTrackerSettings,
) -> None:
    """Using HandTracker as a context manager should open and close cleanly."""
    with HandTracker(hand_tracker_settings) as tracker:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process(blank_frame)
        assert result.hand_count == 0