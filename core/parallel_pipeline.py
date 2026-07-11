import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
from loguru import logger

@dataclass
class TrackedFrame:
    frame: np.ndarray
    results: Any
    timestamp: float
    processing_time: float

class ParallelPipeline:
    """Decouples tracking from rendering using producer-consumer pattern."""
    
    def __init__(self, tracker: Any, max_queue_size: int = 2):
        self.tracker = tracker
        self._frame_queue: queue.Queue[Optional[TrackedFrame]] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._tracking_thread: Optional[threading.Thread] = None
        self._latest_result: Optional[TrackedFrame] = None
        
    def start(self, camera: Any):
        self._stop_event.clear()
        self._tracking_thread = threading.Thread(target=self._tracking_loop, args=(camera,), daemon=True)
        self._tracking_thread.start()
        
    def stop(self):
        self._stop_event.set()
        if self._tracking_thread: self._tracking_thread.join(timeout=2.0)
    
    def _tracking_loop(self, camera: Any):
        while not self._stop_event.is_set():
            try:
                success, frame = camera.read_frame()
                if not success: continue
                
                start = time.perf_counter()
                timestamp_ms = int(start * 1000)
                results = self.tracker.process_frame(frame, timestamp_ms)
                
                tracked = TrackedFrame(frame, results, start, time.perf_counter() - start)
                
                try:
                    self._frame_queue.put_nowait(tracked)
                except queue.Full:
                    try: self._frame_queue.get_nowait()
                    except queue.Empty: pass
                    self._frame_queue.put_nowait(tracked)
            except Exception as e:
                logger.error(f"Tracking error: {e}")
        self._frame_queue.put(None)
    
    def get_tracked_frame(self, timeout: float = 0.1) -> Optional[TrackedFrame]:
        try:
            result = self._frame_queue.get(timeout=timeout)
            if result is not None: self._latest_result = result
            return result
        except queue.Empty:
            return self._latest_result
