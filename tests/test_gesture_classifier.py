"""
Unit tests for the GestureClassifier class.
"""
from __future__ import annotations

import pytest
from config import GestureConfig
from gestures.gesture_classifier import (
    GestureClassifier,
    GestureType,
    InvalidLandmarksError,
)


class FakeLandmark:
    """Minimal stand-in for a MediaPipe landmark (has .x, .y attributes)."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _build_hand(
    wrist=(0.5, 0.9),
    middle_mcp=(0.5, 0.6),
    fingertip_offset_from_wrist: float = 0.3,
    thumb_index_gap: float = 0.3,
    scale: float = 1.0,
) -> list:
    """
    Builds a synthetic 21-landmark hand.

    fingertip_offset_from_wrist controls how far fingertips (4,8,12,16,20)
    sit from the wrist, simulating open (large offset) vs fist (small offset).
    thumb_index_gap controls the distance between landmarks 4 and 8 directly,
    simulating a pinch (small gap) vs not (large gap).
    scale multiplies all coordinates around the wrist to simulate the hand
    being closer/farther from the camera while keeping ratios identical.
    """
    wx, wy = wrist
    mx, my = middle_mcp

    landmarks = [FakeLandmark(wx, wy) for _ in range(21)]
    landmarks[0] = FakeLandmark(wx, wy)
    landmarks[9] = FakeLandmark(mx, my)

    # Thumb tip (4) and index tip (8) positioned to create the desired gap
    landmarks[4] = FakeLandmark(wx - thumb_index_gap / 2, wy - 0.3)
    landmarks[8] = FakeLandmark(wx + thumb_index_gap / 2, wy - 0.3)

    # Other fingertips (12, 16, 20) positioned at fingertip_offset_from_wrist
    for idx in [12, 16, 20]:
        landmarks[idx] = FakeLandmark(wx, wy - fingertip_offset_from_wrist)

    # Apply uniform scale around the wrist (simulates distance-from-camera)
    if scale != 1.0:
        for lm in landmarks:
            lm.x = wx + (lm.x - wx) * scale
            lm.y = wy + (lm.y - wy) * scale

    return landmarks


@pytest.fixture
def config() -> GestureConfig:
    return GestureConfig(
        pinch_threshold_ratio=0.15,
        fist_threshold_ratio=0.4,
        open_palm_threshold_ratio=0.7,
    )


@pytest.fixture
def classifier(config: GestureConfig) -> GestureClassifier:
    return GestureClassifier(config)


def test_hysteresis() -> None:
    """Pinch hysteresis: once active, persists until release threshold is crossed."""
    cfg = GestureConfig(
        pinch_start_threshold=0.1,
        pinch_release_threshold=0.2,
        frame_history_len=5,
        fist_threshold_ratio=0.4,
        open_palm_threshold_ratio=0.7,
    )
    clf = GestureClassifier(cfg)

    # Build realistic landmarks for a clear pinch
    # Wrist at (0.5, 0.8), index MCP at (0.5, 0.6), tips nearly touching
    # hand_scale = 0.2, thumb-index distance = 0.01, ratio = 0.05 < start=0.1
    wrist = FakeLandmark(0.5, 0.8)
    index_mcp = FakeLandmark(0.5, 0.6)
    index_tip = FakeLandmark(0.495, 0.4)
    thumb_tip = FakeLandmark(0.505, 0.4)

    # Feed 5 frames of clear pinch
    for _ in range(5):
        result = clf.is_pinch(wrist, index_mcp, index_tip, thumb_tip)

    assert result is True, "Should detect pinch after history fills"

    # Now move tips apart: distance = 0.08, ratio = 0.08/0.2 = 0.4
    # Above release=0.2, should release
    index_tip_far = FakeLandmark(0.58, 0.4)
    thumb_tip_far = FakeLandmark(0.50, 0.4)

    for _ in range(5):
        result = clf.is_pinch(wrist, index_mcp, index_tip_far, thumb_tip_far)

    assert result is False, "Should release pinch above release threshold"


def test_pinch_gesture_detected(classifier: GestureClassifier) -> None:
    """Thumb and index tip very close together must classify as PINCH."""
    hand = _build_hand(thumb_index_gap=0.02, fingertip_offset_from_wrist=0.3)
    assert classifier.classify(hand) == GestureType.PINCH


def test_open_palm_gesture_detected(classifier: GestureClassifier) -> None:
    """All fingers extended far from wrist must classify as OPEN_PALM."""
    hand = _build_hand(thumb_index_gap=0.35, fingertip_offset_from_wrist=0.5)
    assert classifier.classify(hand) == GestureType.OPEN_PALM


def test_fist_gesture_detected(classifier: GestureClassifier) -> None:
    """All fingers curled close to wrist must classify as FIST."""
    wx, wy = 0.5, 0.9
    mx, my = 0.5, 0.6
    hand = [FakeLandmark(wx, wy) for _ in range(21)]
    hand[0] = FakeLandmark(wx, wy)
    hand[9] = FakeLandmark(mx, my)

    # CRITICAL: thumb (4) and index (8) must be far apart to avoid PINCH.
    # hand_scale = 0.3, so thumb-index distance must be >= 0.045 to stay above 0.15 threshold.
    # Place them at opposite sides: distance = 0.1, ratio = 0.33 > 0.15
    hand[4] = FakeLandmark(0.40, 0.88)   # thumb tip
    hand[8] = FakeLandmark(0.60, 0.88)   # index tip

    # All other fingertips close to wrist (curled)
    hand[12] = FakeLandmark(0.50, 0.86)  # middle
    hand[16] = FakeLandmark(0.50, 0.86)  # ring
    hand[20] = FakeLandmark(0.50, 0.86)  # pinky

    # Verify: distances from wrist (0.5, 0.9)
    # thumb:  sqrt(0.1^2 + 0.02^2) = 0.102, ratio = 0.102/0.3 = 0.34
    # index:  sqrt(0.1^2 + 0.02^2) = 0.102, ratio = 0.34
    # middle: 0.04, ratio = 0.13
    # ring:   0.04, ratio = 0.13
    # pinky:  0.04, ratio = 0.13
    # avg = (0.34 + 0.34 + 0.13 + 0.13 + 0.13) / 5 = 0.214 < 0.4 → FIST

    assert classifier.classify(hand) == GestureType.FIST


def test_ambiguous_shape_returns_none(classifier: GestureClassifier) -> None:
    """A hand shape in between known thresholds must return NONE, not guess."""
    wx, wy = 0.5, 0.9
    mx, my = 0.5, 0.6
    hand = [FakeLandmark(wx, wy) for _ in range(21)]
    hand[0] = FakeLandmark(wx, wy)
    hand[9] = FakeLandmark(mx, my)

    # CRITICAL: thumb and index far apart to avoid PINCH
    hand[4] = FakeLandmark(0.35, 0.88)   # thumb tip
    hand[8] = FakeLandmark(0.65, 0.88)   # index tip

    # Fingertips at medium extension — not curled enough for FIST,
    # not extended enough for OPEN_PALM
    # Target avg_extension_ratio between 0.4 and 0.7
    hand[12] = FakeLandmark(0.50, 0.75)  # middle: dist=0.15, ratio=0.50
    hand[16] = FakeLandmark(0.50, 0.75)  # ring: dist=0.15, ratio=0.50
    hand[20] = FakeLandmark(0.50, 0.75)  # pinky: dist=0.15, ratio=0.50

    # thumb:  sqrt(0.15^2 + 0.02^2) = 0.151, ratio = 0.151/0.3 = 0.504
    # index:  sqrt(0.15^2 + 0.02^2) = 0.151, ratio = 0.504
    # middle: 0.15, ratio = 0.50
    # ring:   0.15, ratio = 0.50
    # pinky:  0.15, ratio = 0.50
    # avg = (0.504 + 0.504 + 0.50 + 0.50 + 0.50) / 5 = 0.502
    # 0.4 < 0.502 < 0.7 → should be NONE

    assert classifier.classify(hand) == GestureType.NONE


def test_pinch_detection_is_scale_invariant(classifier: GestureClassifier) -> None:
    """The same physical pinch gesture must classify identically at different hand scales."""
    hand_near = _build_hand(thumb_index_gap=0.02, fingertip_offset_from_wrist=0.3, scale=1.0)
    hand_far = _build_hand(thumb_index_gap=0.02, fingertip_offset_from_wrist=0.3, scale=0.3)

    assert classifier.classify(hand_near) == GestureType.PINCH
    assert classifier.classify(hand_far) == GestureType.PINCH


def test_fist_detection_is_scale_invariant(classifier: GestureClassifier) -> None:
    """The same physical fist gesture must classify identically at different hand scales."""
    def make_fist(scale: float) -> list:
        wx, wy = 0.5, 0.9
        mx, my = 0.5, 0.6
        hand = [FakeLandmark(wx, wy) for _ in range(21)]
        hand[0] = FakeLandmark(wx, wy)
        hand[9] = FakeLandmark(mx, my)

        # Thumb and index far apart to avoid PINCH
        hand[4] = FakeLandmark(0.40, 0.88)
        hand[8] = FakeLandmark(0.60, 0.88)

        # All other tips close to wrist
        hand[12] = FakeLandmark(0.50, 0.86)
        hand[16] = FakeLandmark(0.50, 0.86)
        hand[20] = FakeLandmark(0.50, 0.86)

        if scale != 1.0:
            for lm in hand:
                lm.x = wx + (lm.x - wx) * scale
                lm.y = wy + (lm.y - wy) * scale
        return hand

    hand_near = make_fist(1.0)
    hand_far = make_fist(0.4)

    assert classifier.classify(hand_near) == GestureType.FIST
    assert classifier.classify(hand_far) == GestureType.FIST


def test_empty_landmarks_raises(classifier: GestureClassifier) -> None:
    """An empty landmark list must raise InvalidLandmarksError."""
    with pytest.raises(InvalidLandmarksError):
        classifier.classify([])


def test_insufficient_landmarks_raises(classifier: GestureClassifier) -> None:
    """Fewer than 21 landmarks must raise InvalidLandmarksError."""
    short_hand = [FakeLandmark(0.5, 0.5) for _ in range(10)]
    with pytest.raises(InvalidLandmarksError):
        classifier.classify(short_hand)


def test_zero_hand_size_raises(classifier: GestureClassifier) -> None:
    """Wrist and middle MCP at identical coordinates must raise (zero hand-size reference)."""
    hand = _build_hand(wrist=(0.5, 0.5), middle_mcp=(0.5, 0.5))
    with pytest.raises(InvalidLandmarksError):
        classifier.classify(hand)