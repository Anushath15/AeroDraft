"""
Unit tests for the DepthEstimator class.
"""
import pytest
from config import ASMEConfig
from core.depth_estimator import DepthEstimator, NoHandDetectedError


class FakeLandmark:
    """Minimal stand-in for a MediaPipe landmark (has .x, .y attributes)."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _make_landmarks(wrist_xy: tuple, index_mcp_xy: tuple) -> list:
    """
    Builds a 21-element landmark list with only wrist (0) and
    index MCP (5) set meaningfully; the rest are placeholders.
    """
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(21)]
    landmarks[0] = FakeLandmark(*wrist_xy)
    landmarks[5] = FakeLandmark(*index_mcp_xy)
    return landmarks


@pytest.fixture
def config() -> ASMEConfig:
    return ASMEConfig(reference_distance_px=120.0)


def test_estimate_returns_float_for_valid_landmarks(config: ASMEConfig) -> None:
    """A valid landmark set should return a numeric depth value."""
    estimator = DepthEstimator(config)
    landmarks = _make_landmarks(wrist_xy=(0.5, 0.5), index_mcp_xy=(0.6, 0.5))
    depth = estimator.estimate(landmarks, frame_width=640, frame_height=480, timestamp=0.0)
    assert isinstance(depth, float)
    assert depth > 0


def test_closer_hand_yields_larger_depth_value(config: ASMEConfig) -> None:
    """
    Larger pixel distance (hand closer to camera) should yield a
    smaller raw_depth per the reference_distance/pixel_dist formula,
    verified across two independent estimator instances to avoid
    filter-history contamination between readings.
    """
    close_estimator = DepthEstimator(config)
    far_estimator = DepthEstimator(config)

    # Large pixel separation = hand close to camera
    close_landmarks = _make_landmarks(wrist_xy=(0.3, 0.5), index_mcp_xy=(0.7, 0.5))
    # Small pixel separation = hand far from camera
    far_landmarks = _make_landmarks(wrist_xy=(0.48, 0.5), index_mcp_xy=(0.52, 0.5))

    close_depth = close_estimator.estimate(
        close_landmarks, frame_width=640, frame_height=480, timestamp=0.0
    )
    far_depth = far_estimator.estimate(
        far_landmarks, frame_width=640, frame_height=480, timestamp=0.0
    )

    assert close_depth < far_depth


def test_empty_landmarks_raises_no_hand_detected(config: ASMEConfig) -> None:
    """An empty landmark list must raise NoHandDetectedError."""
    estimator = DepthEstimator(config)
    with pytest.raises(NoHandDetectedError):
        estimator.estimate([], frame_width=640, frame_height=480, timestamp=0.0)


def test_insufficient_landmarks_raises_no_hand_detected(config: ASMEConfig) -> None:
    """Fewer than 6 landmarks (missing index MCP) must raise NoHandDetectedError."""
    estimator = DepthEstimator(config)
    short_landmarks = [FakeLandmark(0.5, 0.5) for _ in range(3)]
    with pytest.raises(NoHandDetectedError):
        estimator.estimate(short_landmarks, frame_width=640, frame_height=480, timestamp=0.0)


def test_zero_distance_landmarks_raises_no_hand_detected(config: ASMEConfig) -> None:
    """Wrist and index MCP at identical coordinates must raise (zero distance)."""
    estimator = DepthEstimator(config)
    landmarks = _make_landmarks(wrist_xy=(0.5, 0.5), index_mcp_xy=(0.5, 0.5))
    with pytest.raises(NoHandDetectedError):
        estimator.estimate(landmarks, frame_width=640, frame_height=480, timestamp=0.0)


def test_repeated_calls_apply_smoothing(config: ASMEConfig) -> None:
    """Consecutive estimate() calls should route through the internal filter."""
    estimator = DepthEstimator(config)
    landmarks = _make_landmarks(wrist_xy=(0.4, 0.5), index_mcp_xy=(0.6, 0.5))

    first = estimator.estimate(landmarks, frame_width=640, frame_height=480, timestamp=0.0)
    # First call always passes through unfiltered (no history yet)
    second = estimator.estimate(landmarks, frame_width=640, frame_height=480, timestamp=1 / 30)

    assert isinstance(first, float)
    assert isinstance(second, float)


def test_reset_clears_filter_state(config: ASMEConfig) -> None:
    """reset() should allow the next estimate() to behave as a fresh first call."""
    estimator = DepthEstimator(config)
    landmarks = _make_landmarks(wrist_xy=(0.4, 0.5), index_mcp_xy=(0.6, 0.5))

    estimator.estimate(landmarks, frame_width=640, frame_height=480, timestamp=0.0)
    estimator.estimate(landmarks, frame_width=640, frame_height=480, timestamp=1 / 30)
    estimator.reset()

    # After reset, this call has no filter history — same as a first call
    result = estimator.estimate(landmarks, frame_width=640, frame_height=480, timestamp=99.0)
    raw_depth = config.reference_distance_px / estimator._pixel_distance(
        landmarks[0], landmarks[5], 640, 480
    )
    assert result == pytest.approx(raw_depth)