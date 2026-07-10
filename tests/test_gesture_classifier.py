"""
Unit tests for the GestureClassifier class.
"""
import pytest
from config import GestureConfig
from gestures.gesture_classifier import (
    GestureClassifier,
    GestureType,
    InvalidLandmarksError,
)

def test_hysteresis():
    cfg = GestureConfig(pinch_start_threshold=0.1, pinch_release_threshold=0.2)
    clf = GestureClassifier(cfg)
    # Mock data (Wrist, IndexMCP, IndexTip, ThumbTip)
    mock = type('LM', (), {'x': 0.0, 'y': 0.0})
    # Force pinch
    for _ in range(5):
        clf.is_pinch(mock, mock, mock, mock) # Need proper logic for test
    # Verify state consistency...

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

    fingertip_offset_from_wrist controls how far fingertips (8,12,16,20)
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
    hand = _build_hand(thumb_index_gap=0.35, fingertip_offset_from_wrist=0.05)
    assert classifier.classify(hand) == GestureType.FIST


def test_ambiguous_shape_returns_none(classifier: GestureClassifier) -> None:
    """A hand shape in between known thresholds must return NONE, not guess."""
    hand = _build_hand(thumb_index_gap=0.35, fingertip_offset_from_wrist=0.2)
    assert classifier.classify(hand) == GestureType.NONE


def test_pinch_detection_is_scale_invariant(classifier: GestureClassifier) -> None:
    """The same physical pinch gesture must classify identically at different hand scales."""
    hand_near = _build_hand(thumb_index_gap=0.02, fingertip_offset_from_wrist=0.3, scale=1.0)
    hand_far = _build_hand(thumb_index_gap=0.02, fingertip_offset_from_wrist=0.3, scale=0.3)

    assert classifier.classify(hand_near) == GestureType.PINCH
    assert classifier.classify(hand_far) == GestureType.PINCH


def test_fist_detection_is_scale_invariant(classifier: GestureClassifier) -> None:
    """The same physical fist gesture must classify identically at different hand scales."""
    hand_near = _build_hand(thumb_index_gap=0.35, fingertip_offset_from_wrist=0.05, scale=1.0)
    hand_far = _build_hand(thumb_index_gap=0.35, fingertip_offset_from_wrist=0.05, scale=0.4)

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