"""
Integration test for the full pipeline logic.
Ensures modules communicate correctly.
"""
import numpy as np
import pytest
from config import settings
from core.depth_estimator import DepthEstimator
from engine.projection import PerspectiveProjector

# Define a minimal Mock Landmark class for the test scope
class MockLM:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def test_pipeline_data_flow():
    """Verify data flows from landmark -> depth -> projection."""
    # 1. Setup Engine Components
    estimator = DepthEstimator(settings.asme)
    projector = PerspectiveProjector(
        settings.render.focal_length, 
        settings.camera.width, 
        settings.camera.height
    )
    
    # 2. Setup Mock landmarks (Must be valid geometry)
    # Wrist (0) and Index MCP (5) must have non-zero distance 
    # to avoid NoHandDetectedError.
    landmarks = [MockLM(0.5, 0.5) for _ in range(21)]
    landmarks[0] = MockLM(0.5, 0.5) # Wrist
    landmarks[5] = MockLM(0.6, 0.5) # IndexMCP (Distance = 0.1)
    
    # 3. Run Pipeline
    # Estimate depth based on valid geometry
    depth = estimator.estimate(
        landmarks, 
        settings.camera.width, 
        settings.camera.height, 
        1.0
    )
    
    # 4. Construct 3D Center (using Wrist as reference)
    center = np.array([0.0, 0.0, depth])
    extents = np.array([settings.box.width, settings.box.height, settings.box.depth])
    
    # 5. Project and Verify
    projection = projector.project_box(center, extents)
    
    assert projection.shape == (8, 2)
    assert not np.isnan(projection).any()
    assert np.all(projection >= 0) # Basic bounds check