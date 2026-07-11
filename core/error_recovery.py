from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable
from loguru import logger
import time


class FailureType(Enum):
    CAMERA_FAILURE = auto()
    TRACKER_FAILURE = auto()
    RENDER_FAILURE = auto()
    PIPELINE_FAILURE = auto()


@dataclass
class RecoveryState:
    failure_type: FailureType
    failure_count: int
    last_failure_time: float
    recovery_attempt: int
    max_recovery_attempts: int = 3
    backoff_base_seconds: float = 1.0
    
    @property
    def is_exhausted(self) -> bool:
        return self.recovery_attempt >= self.max_recovery_attempts
    
    @property
    def backoff_seconds(self) -> float:
        return self.backoff_base_seconds * (2 ** self.recovery_attempt)


class ErrorRecoveryManager:
    """Manages graceful recovery from pipeline failures."""
    
    def __init__(self, max_attempts: int = 3):
        self._states: dict[FailureType, RecoveryState] = {}
        self._max_attempts = max_attempts
        self._recovery_handlers: dict[FailureType, Callable] = {}
        
    def register_handler(self, failure_type: FailureType, handler: Callable[[], bool]):
        """Register a recovery handler that returns True on success."""
        self._recovery_handlers[failure_type] = handler
        
    def report_failure(self, failure_type: FailureType) -> bool:
        """Report a failure. Returns True if recovery should be attempted."""
        now = time.time()
        
        if failure_type not in self._states:
            self._states[failure_type] = RecoveryState(
                failure_type=failure_type,
                failure_count=1,
                last_failure_time=now,
                recovery_attempt=0,
                max_recovery_attempts=self._max_attempts
            )
        else:
            state = self._states[failure_type]
            state.failure_count += 1
            state.last_failure_time = now
        
        state = self._states[failure_type]
        
        if state.is_exhausted:
            logger.error(f"Recovery exhausted for {failure_type.name}")
            return False
        
        logger.warning(
            f"Failure #{state.failure_count} for {failure_type.name}, "
            f"attempting recovery #{state.recovery_attempt + 1}"
        )
        return True
    
    def attempt_recovery(self, failure_type: FailureType) -> bool:
        """Attempt recovery. Returns True if successful."""
        state = self._states.get(failure_type)
        if not state:
            return False
            
        time.sleep(state.backoff_seconds)
        
        handler = self._recovery_handlers.get(failure_type)
        if not handler:
            logger.error(f"No recovery handler for {failure_type.name}")
            return False
        
        try:
            success = handler()
            if success:
                logger.info(f"Recovery successful for {failure_type.name}")
                self._states.pop(failure_type, None)
                return True
            else:
                state.recovery_attempt += 1
                return False
        except Exception as e:
            logger.error(f"Recovery handler failed: {e}")
            state.recovery_attempt += 1
            return False
    
    def reset(self, failure_type: Optional[FailureType] = None):
        """Reset recovery state."""
        if failure_type:
            self._states.pop(failure_type, None)
        else:
            self._states.clear()
