import pytest
import numpy as np
from core.coordinate_filter import CoordinateFilter
from config import ASMEConfig

class TestCoordinateFilter:
    @pytest.fixture
    def config(self) -> ASMEConfig:
        return ASMEConfig()

    @pytest.fixture
    def filt(self, config: ASMEConfig) -> CoordinateFilter:
        return CoordinateFilter(config)

    def test_initial_state(self, filt: CoordinateFilter):
        assert filt is not None

    def test_has_xyz_filters(self, filt: CoordinateFilter):
        assert hasattr(filt, 'x_filter')
        assert hasattr(filt, 'y_filter')
        assert hasattr(filt, 'z_filter')

    def test_filters_are_one_euro_instances(self, filt: CoordinateFilter):
        from core.one_euro_filter import OneEuroFilter
        assert isinstance(filt.x_filter, OneEuroFilter)
        assert isinstance(filt.y_filter, OneEuroFilter)
        assert isinstance(filt.z_filter, OneEuroFilter)

    def test_single_point_passes_through(self, filt: CoordinateFilter):
        point = np.array([0.5, 0.5, 0.3])
        try:
            if hasattr(filt, 'apply'): result = filt.apply(point)
            elif hasattr(filt, 'smooth'): result = filt.smooth(point)
            else: result = np.array([filt.x_filter(point[0], 0.0), filt.y_filter(point[1], 0.0), filt.z_filter(point[2], 0.0)])
            np.testing.assert_array_almost_equal(result, point, decimal=2)
        except Exception as e: pytest.fail(f"Failed to process single point: {e}")

    def test_smoothing_reduces_variance(self, filt: CoordinateFilter):
        base = np.array([0.5, 0.5, 0.3])
        np.random.seed(42)
        raw_points = [base + np.random.normal(0, 0.05, 3) for _ in range(50)]
        filtered_x = []
        for i, p in enumerate(raw_points):
            t = float(i) / 30.0
            try:
                if hasattr(filt, 'apply'): res = filt.apply(p, t)
                elif hasattr(filt, 'smooth'): res = filt.smooth(p, t)
                else: res = np.array([filt.x_filter(p[0], t), filt.y_filter(p[1], t), filt.z_filter(p[2], t)])
                filtered_x.append(res[0])
            except TypeError:
                if hasattr(filt, 'apply'): res = filt.apply(p)
                elif hasattr(filt, 'smooth'): res = filt.smooth(p)
                else: res = np.array([filt.x_filter(p[0], i), filt.y_filter(p[1], i), filt.z_filter(p[2], i)])
                filtered_x.append(res[0])
        raw_var = np.var([p[0] for p in raw_points])
        filtered_var = np.var(filtered_x)
        assert filtered_var < raw_var, "Filter should reduce variance"
