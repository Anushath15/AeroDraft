import pytest
import time
from core.coordinate_filter import CoordinateFilter
from config import ASMEConfig

def test_smoothing():
    cfg = ASMEConfig()
    filt = CoordinateFilter(cfg)
    ts = time.time()
    v1 = filt(0.5, 0.5, 0.5, ts)
    v2 = filt(0.55, 0.55, 0.55, ts + 0.03)
    assert v1 != v2 # Filter should adapt