"""The one clock every timestamp in a trace comes from."""

from __future__ import annotations

import time


def now_nanoseconds() -> int:
    """Return a monotonic time in nanoseconds; only differences between two values mean anything."""
    return time.monotonic_ns()
