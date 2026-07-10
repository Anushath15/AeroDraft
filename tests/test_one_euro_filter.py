"""
Unit tests for the OneEuroFilter class.
"""
import pytest
from core.one_euro_filter import OneEuroFilter


def test_first_call_returns_input_unchanged() -> None:
    """With no history, the filter must pass the first value through untouched."""
    f = OneEuroFilter()
    result = f(5.0, timestamp=0.0)
    assert result == 5.0


def test_constant_input_converges_to_input_value() -> None:
    """Feeding the same value repeatedly should converge to that value."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    t = 0.0
    result = 0.0
    for _ in range(50):
        t += 1 / 30  # simulate 30fps
        result = f(10.0, timestamp=t)
    assert result == pytest.approx(10.0, abs=0.01)


def test_smooths_noisy_signal_reduces_variance() -> None:
    """Filtered output variance should be lower than raw noisy input variance."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    raw_values = [10.0, 10.5, 9.5, 10.3, 9.7, 10.6, 9.4, 10.2, 9.8, 10.4]
    filtered_values = []
    t = 0.0
    for v in raw_values:
        t += 1 / 30
        filtered_values.append(f(v, timestamp=t))

    raw_variance = _variance(raw_values)
    filtered_variance = _variance(filtered_values)
    assert filtered_variance < raw_variance


def test_reset_clears_state_so_next_call_acts_as_first() -> None:
    """After reset(), the next call must behave like a fresh first call."""
    f = OneEuroFilter()
    f(5.0, timestamp=0.0)
    f(6.0, timestamp=1 / 30)
    f.reset()
    result = f(99.0, timestamp=5.0)
    assert result == 99.0


def test_non_increasing_timestamp_raises() -> None:
    """The filter must reject a timestamp that does not strictly increase."""
    f = OneEuroFilter()
    f(5.0, timestamp=1.0)
    with pytest.raises(ValueError):
        f(6.0, timestamp=1.0)  # same timestamp, dt = 0


def test_invalid_min_cutoff_raises() -> None:
    """min_cutoff must be strictly positive."""
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0)


def test_invalid_d_cutoff_raises() -> None:
    """d_cutoff must be strictly positive."""
    with pytest.raises(ValueError):
        OneEuroFilter(d_cutoff=-1.0)


def _variance(values: list[float]) -> float:
    """Computes population variance of a list of floats."""
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)