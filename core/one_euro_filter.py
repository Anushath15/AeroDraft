"""
1-Euro Filter implementation.

Adaptive low-pass filter for eliminating jitter from noisy real-time
signals while preserving responsiveness during fast motion.

Reference: Casiez, G., Roussel, N., and Vogel, D. (2012).
'1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input
in Interactive Systems.' CHI 2012.
"""
from __future__ import annotations
import math
from typing import Optional


class OneEuroFilter:
    """
    Smooths a noisy scalar signal using an adaptive cutoff frequency.

    At low speeds the filter smooths aggressively (removing jitter).
    At high speeds the filter relaxes (avoiding perceptible lag).

    Usage:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        smoothed = f(raw_value, timestamp)
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        """
        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Controls smoothing
                at rest. Lower = smoother but more lag.
            beta: Speed coefficient. Higher = less lag during fast motion,
                but more jitter allowed through.
            d_cutoff: Cutoff frequency for the derivative estimate.
                Rarely needs tuning; 1.0 is standard.
        """
        if min_cutoff <= 0:
            raise ValueError("min_cutoff must be positive.")
        if d_cutoff <= 0:
            raise ValueError("d_cutoff must be positive.")

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        """Computes the smoothing factor for a given cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _low_pass(x: float, x_prev: float, alpha: float) -> float:
        """Applies exponential smoothing between current and previous value."""
        return alpha * x + (1.0 - alpha) * x_prev

    def __call__(self, x: float, timestamp: float) -> float:
        """
        Filters a new sample.

        Args:
            x: The raw signal value at this timestep.
            timestamp: Time in seconds (monotonic, strictly increasing).

        Returns:
            The filtered (smoothed) value.

        Raises:
            ValueError: If timestamp does not advance from the previous call.
        """
        if self._t_prev is None:
            # First call — no history to filter against.
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return x

        dt = timestamp - self._t_prev
        if dt <= 0:
            raise ValueError(
                f"timestamp must strictly increase; got dt={dt}."
            )

        # Estimate and smooth the derivative (speed of change).
        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_hat = self._low_pass(dx, self._dx_prev, alpha_d)

        # Adaptive cutoff — widens as speed increases.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(dt, cutoff)
        x_hat = self._low_pass(x, self._x_prev, alpha)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = timestamp

        return x_hat

    def reset(self) -> None:
        """Clears all internal state. Next call behaves as the first call."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None