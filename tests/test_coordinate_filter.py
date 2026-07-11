import pytest
import numpy as np
from core.coordinate_filter import CoordinateFilter
from config import ASMEConfig


class TestCoordinateFilter:
    """Comprehensive tests for 3D coordinate smoothing."""

    @pytest.fixture
    def config(self) -> ASMEConfig:
        return ASMEConfig()

    @pytest.fixture
    def filter(self, config: ASMEConfig) -> CoordinateFilter:
        # Reset filter for every test to avoid state leakage
        return CoordinateFilter(config)

    def test_initial_state(self, filter: CoordinateFilter):
        """Filter should instantiate correctly."""
        assert filter is not None

    def test_single_point_returns_same(self, filter: CoordinateFilter):
        """First point should pass through unchanged."""
        point = np.array([0.5, 0.5, 0.3])
        result = filter.filter(point)
        np.testing.assert_array_almost_equal(result, point)

    def test_smoothing_reduces_jitter(self, filter: CoordinateFilter):
        """Filtered signal should have less variance than raw."""
        base = np.array([0.5, 0.5, 0.3])
        np.random.seed(42) # Reproducibility
        raw_points = [base + np.random.normal(0, 0.05, 3) for _ in range(50)]
        filtered_points = [filter.filter(p) for p in raw_points]
        
        raw_var = np.var([p[0] for p in raw_points])
        filtered_var = np.var([p[0] for p in filtered_points])
        
        assert filtered_var < raw_var, "Filter should reduce variance"

    def test_steady_signal_passes_through(self, filter: CoordinateFilter):
        """Constant signal should pass through without distortion."""
        point = np.array([0.5, 0.5, 0.3])
        result = point
        for _ in range(30):
            result = filter.filter(point)
        # After settling, should be very close to original
        np.testing.assert_array_almost_equal(result, point, decimal=2)

    def test_step_response_converges(self, filter: CoordinateFilter):
        """Filter should track step changes with some lag."""
        point_a = np.array([0.2, 0.2, 0.3])
        point_b = np.array([0.8, 0.8, 0.3])
        
        # Stabilize at point_a
        for _ in range(30):
            filter.filter(point_a)
        
        # Step to point_b
        results = [filter.filter(point_b) for _ in range(30)]
        final = results[-1]
        
        # Should converge toward point_b (not still stuck at 0.2)
        assert final[0] > 0.5, f"Should track step change, got {final[0]}"

    def test_depth_channel_filtered(self, filter: CoordinateFilter):
        """Z (depth) channel should also be smoothed."""
        points = [np.array([0.5, 0.5, 0.3 + i * 0.01]) for i in range(30)]
        filtered = [filter.filter(p) for p in points]
        
        raw_z_var = np.var([p[2] for p in points])
        filtered_z_var = np.var([p[2] for p in filtered])
        assert filtered_z_var < raw_z_var, "Depth channel should be smoothed"

    def test_large_jump_dampened(self, filter: CoordinateFilter):
        """Very large jumps should be dampened, not passed through instantly."""
        point_a = np.array([0.1, 0.1, 0.3])
        point_b = np.array([0.9, 0.9, 0.3])
        
        # Stabilize
        for _ in range(20):
            filter.filter(point_a)
        
        # Sudden jump
        result = filter.filter(point_b)
        
        # Should not jump immediately to 0.9 (One Euro filter prevents this)
        assert result[0] < 0.8, f"Large jump should be dampened, got {result[0]}"

    def test_nan_input_no_crash(self, filter: CoordinateFilter):
        """Should handle NaN gracefully without crashing."""
        nan_point = np.array([np.nan, 0.5, 0.3])
        try:
            result = filter.filter(nan_point)
            # If it doesn't crash, it passes. We don't strictly care what it returns
            # as long as the app doesn't die.
            assert result is not None or np.all(np.isnan(result))
        except ValueError:
            # Some strict filters throw ValueError on NaN, which is also acceptable
            pass

    def test_reset_clears_state(self, filter: CoordinateFilter):
        """Reset should clear internal filter state."""
        point = np.array([0.5, 0.5, 0.3])
        for _ in range(20):
            filter.filter(point)
        
        if hasattr(filter, 'reset'):
            filter.reset()
            result = filter.filter(point)
            # After reset, should behave like first frame again
            np.testing.assert_array_almost_equal(result, point)
