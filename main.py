"""Orchestrator for the AeroDraft spatial pipeline."""
import time
import numpy as np
from loguru import logger
import cv2
from config import settings
from camera import VideoStream
from hand_tracker import HandTracker
from core.depth_estimator import DepthEstimator, NoHandDetectedError
from core.coordinate_filter import CoordinateFilter
from engine.projection import PerspectiveProjector, InvalidDepthError
from engine.wireframe_renderer import WireframeRenderer
from gestures.gesture_classifier import GestureClassifier
from gestures.state_machine import StateMachine, InteractionState

class AeroDraftPipeline:
    def __init__(self):
        # Initialize components
        self.tracker = HandTracker(settings.tracker)
        self.depth_estimator = DepthEstimator(settings.asme)
        self.coord_filter = CoordinateFilter(settings.asme)
        self.projector = PerspectiveProjector(
            settings.render.focal_length, settings.camera.width, settings.camera.height
        )
        self.renderer = WireframeRenderer(
            settings.render.box_color_bgr, settings.render.line_thickness
        )
        self.classifier = GestureClassifier(settings.gesture)
        self.state_machine = StateMachine()

    def run(self):
        with VideoStream(settings.camera.device_index, settings.camera.width, settings.camera.height) as stream, \
             self.tracker as tracker:
            
            logger.info("Pipeline active. Press Q or ESC to exit.")
            
            while True:
                success, frame = stream.read_frame()
                if not success: continue

                results = tracker.process_frame(frame)
                
                if results and results.hand_landmarks:
                    try:
                        # 1. Extraction
                        hand = results.hand_landmarks[0]
                        wrist, index_mcp, index_tip, thumb_tip = hand[0], hand[5], hand[8], hand[4]
                        
                        # 2. Gesture and State
                        is_pinched = self.classifier.is_pinch(wrist, index_mcp, index_tip, thumb_tip)
                        self.state_machine.update(is_pinched, (wrist.x, wrist.y))
                        
                        # 3. Coordinate Filtering
                        ts = time.time()
                        pseudo_depth = self.depth_estimator.estimate(hand, settings.camera.width, settings.camera.height, ts)
                        f_x, f_y, f_z = self.coord_filter(wrist.x, wrist.y, pseudo_depth, ts)
                        
                        # 4. State-based positioning
                        if self.state_machine.state == InteractionState.ANCHORED:
                            cx, cy, cz = self.state_machine.anchor_pos[0], self.state_machine.anchor_pos[1], f_z
                        else:
                            cx, cy, cz = (f_x - 0.5) * 2.0, (f_y - 0.5) * 2.0, f_z

                        # 5. Projection & Rendering
                        points_2d = self.projector.project_box(
                            np.array([cx, cy, cz]), 
                            np.array([settings.box.width, settings.box.height, settings.box.depth])
                        )
                        frame = self.renderer.draw_box(frame, points_2d)
                        
                    except (NoHandDetectedError, InvalidDepthError) as e:
                        logger.trace(f"Pipeline skip: {e}")
                    except Exception as e:
                        logger.error(f"Pipeline error: {e}")

                # 6. Skeleton Visualization (Debugging Overlay)
                frame = self.tracker.draw_landmarks(frame, results)
                cv2.imshow(settings.window_name, frame)

                if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                    break
        cv2.destroyAllWindows()