"""Multi-dimensional filter for (X, Y, Z) signals."""
from config import ASMEConfig
from core.one_euro_filter import OneEuroFilter

class CoordinateFilter:
    """Wraps three OneEuroFilters for 3D coordinate smoothing."""
    def __init__(self, config: ASMEConfig):
        self.x_filter = OneEuroFilter(config.one_euro_min_cutoff, config.one_euro_beta)
        self.y_filter = OneEuroFilter(config.one_euro_min_cutoff, config.one_euro_beta)
        self.z_filter = OneEuroFilter(config.one_euro_min_cutoff, config.one_euro_beta)

    def __call__(self, x: float, y: float, z: float, timestamp: float) -> tuple[float, float, float]:
        return (
            self.x_filter(x, timestamp),
            self.y_filter(y, timestamp),
            self.z_filter(z, timestamp)
        )