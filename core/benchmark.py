"""
Lightweight performance benchmarking module.

Provides context-manager and decorator-based timing instrumentation
with negligible overhead when disabled.

Usage:
    from core.benchmark import benchmark
    
    with benchmark("inference"):
        results = tracker.process_frame(frame)
    
    @benchmark("classify")
    def classify_hand(landmarks):
        return classifier.classify(landmarks)

Enable via environment variable:
    AERODRAFT_BENCHMARK=1 python -m main
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

from loguru import logger


# Benchmarking is disabled by default for zero overhead
_BENCHMARK_ENABLED = os.environ.get("AERODRAFT_BENCHMARK", "0") == "1"


class BenchmarkCollector:
    """
    Accumulates timing samples and computes statistics.
    
    Thread-safe for single-threaded use (GIL-protected).
    Not safe for concurrent multi-threaded writes.
    """
    
    def __init__(self) -> None:
        self._samples: defaultdict[str, list[float]] = defaultdict(list)
        self._call_counts: defaultdict[str, int] = defaultdict(int)
    
    def record(self, name: str, duration_ms: float) -> None:
        """Records a single timing sample."""
        self._samples[name].append(duration_ms)
        self._call_counts[name] += 1
    
    def report(self) -> dict[str, dict[str, float]]:
        """
        Computes statistics for all collected samples.
        
        Returns:
            Dict mapping benchmark name to stats dict containing:
            - count: number of samples
            - total_ms: cumulative time
            - mean_ms: average duration
            - min_ms: fastest sample
            - max_ms: slowest sample
        """
        stats: dict[str, dict[str, float]] = {}
        for name, samples in self._samples.items():
            if not samples:
                continue
            count = len(samples)
            total = sum(samples)
            stats[name] = {
                "count": count,
                "total_ms": round(total, 3),
                "mean_ms": round(total / count, 3),
                "min_ms": round(min(samples), 3),
                "max_ms": round(max(samples), 3),
            }
        return stats
    
    def reset(self) -> None:
        """Clears all collected samples."""
        self._samples.clear()
        self._call_counts.clear()


# Global singleton collector
_collector = BenchmarkCollector()


@contextmanager
def benchmark(name: str) -> None:
    """
    Context manager that times a code block when benchmarking is enabled.
    
    Args:
        name: Identifier for this benchmark segment.
    
    Usage:
        with benchmark("frame_pipeline"):
            process_frame()
    """
    if not _BENCHMARK_ENABLED:
        yield
        return
    
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        _collector.record(name, duration_ms)


def benchmark_func(name: Optional[str] = None):
    """
    Decorator that times function execution when benchmarking is enabled.
    
    Args:
        name: Override for the benchmark identifier. Defaults to function name.
    
    Usage:
        @benchmark_func()
        def expensive_operation():
            ...
    """
    def decorator(func):
        bench_name = name or func.__name__
        
        def wrapper(*args, **kwargs):
            if not _BENCHMARK_ENABLED:
                return func(*args, **kwargs)
            
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                _collector.record(bench_name, duration_ms)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def get_report() -> dict[str, dict[str, float]]:
    """Returns current benchmark statistics."""
    return _collector.report()


def print_report() -> None:
    """Logs benchmark statistics at INFO level."""
    stats = _collector.report()
    if not stats:
        logger.info("No benchmark data collected.")
        return
    
    logger.info("=" * 50)
    logger.info("BENCHMARK REPORT")
    logger.info("=" * 50)
    logger.info(f"{'Name':<25} {'Count':>8} {'Mean(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
    logger.info("-" * 65)
    for name, data in sorted(stats.items()):
        logger.info(
            f"{name:<25} {data['count']:>8} {data['mean_ms']:>10.3f} "
            f"{data['min_ms']:>10.3f} {data['max_ms']:>10.3f}"
        )
    logger.info("=" * 50)


def reset_benchmarks() -> None:
    """Clears all benchmark data."""
    _collector.reset()