"""
AeroDraft - Master entry point.
Orchestrates the camera stream, hand tracking inference,
and frame rendering. No business logic lives here.
"""
import sys
import cv2
from loguru import logger
from config import settings
from camera import VideoStream
from hand_tracker import HandTracker


def main() -> None:
    """Runs the core application loop."""
    logger.info("AeroDraft starting - Phase 1 (Camera + Hand Tracking)")

    with VideoStream(
        device_index=settings.camera.device_index,
        width=settings.camera.width,
        height=settings.camera.height,
    ) as stream, HandTracker(config=settings.tracker) as tracker:

        logger.info("Pipeline active. Press Q or ESC to exit.")

        while True:
            success, frame = stream.read_frame()
            if not success:
                logger.warning("Dropped frame - skipping.")
                continue

            # process_frame handles BGR->RGB internally
            results = tracker.process_frame(frame)
            annotated_frame = tracker.draw_landmarks(frame, results)
            cv2.imshow(settings.window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("Exit signal received.")
                break

    cv2.destroyAllWindows()
    logger.info("AeroDraft shutdown complete.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        logger.critical(f"Hardware error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected crash: {e}")
        sys.exit(1)