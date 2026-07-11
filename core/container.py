from dataclasses import dataclass, field
from typing import Optional, Protocol, Any
from config import AppConfig


# --- Protocols (Interfaces) ---

class ICamera(Protocol):
    def read_frame(self) -> tuple[bool, Any]: ...
    def __enter__(self) -> "ICamera": ...
    def __exit__(self, *args) -> None: ...


class IHandTracker(Protocol):
    def process_frame(self, frame: Any, timestamp_ms: int) -> Any: ...
    def draw_landmarks(self, frame: Any, results: Any) -> Any: ...
    def __enter__(self) -> "IHandTracker": ...
    def __exit__(self, *args) -> None: ...


class IGestureClassifier(Protocol):
    def classify(self, landmarks: Any) -> Any: ...


# --- DI Container ---

@dataclass
class PipelineContainer:
    """Dependency injection container for the AeroDraft pipeline."""
    
    config: AppConfig
    
    # Lazy-initialized components
    _camera: Optional[ICamera] = field(default=None, init=False, repr=False)
    _tracker: Optional[IHandTracker] = field(default=None, init=False, repr=False)
    _classifier: Optional[IGestureClassifier] = field(default=None, init=False, repr=False)
    
    # Overridable for testing
    _camera_factory: Any = field(default=None, init=False, repr=False)
    _tracker_factory: Any = field(default=None, init=False, repr=False)
    
    def override_camera(self, factory):
        """Override camera for testing."""
        self._camera_factory = factory
        self._camera = None
        
    def override_tracker(self, factory):
        """Override tracker for testing."""
        self._tracker_factory = factory
        self._tracker = None
    
    @property
    def camera(self) -> ICamera:
        if self._camera is None:
            if self._camera_factory:
                self._camera = self._camera_factory()
            else:
                from camera import VideoStream
                self._camera = VideoStream(
                    self.config.camera.device_index,
                    self.config.camera.width,
                    self.config.camera.height
                )
        return self._camera
    
    @property
    def tracker(self) -> IHandTracker:
        if self._tracker is None:
            if self._tracker_factory:
                self._tracker = self._tracker_factory()
            else:
                from hand_tracker import HandTracker
                self._tracker = HandTracker(self.config.tracker)
        return self._tracker
    
    @property
    def classifier(self) -> IGestureClassifier:
        if self._classifier is None:
            from gestures.gesture_classifier import GestureClassifier
            self._classifier = GestureClassifier(self.config.gesture)
        return self._classifier
    
    def create_pipeline(self) -> "Pipeline":
        """Create a fully-wired Pipeline instance."""
        from gestures.state_machine import GestureStateMachine
        from core.depth_estimator import DepthEstimator
        from engine.projection import PerspectiveProjector
        from ui.hud_renderer import HUDRenderer
        from core.pipeline import Pipeline
        
        return Pipeline(
            config=self.config,
            camera=self.camera,
            tracker=self.tracker,
            classifier=self.classifier,
            state_machine=GestureStateMachine(self.config.state_machine),
            depth_estimator=DepthEstimator(self.config.asme),
            projector=PerspectiveProjector(
                focal_length=self.config.render.focal_length,
                frame_width=self.config.camera.width,
                frame_height=self.config.camera.height,
            ),
            hud_renderer=HUDRenderer(self.config.hud),
        )
