"""Percentiles of time to first audio, computed from trace records."""

from __future__ import annotations

from typing import Sequence

from .trace import stage_durations_milliseconds, time_to_first_audio_milliseconds


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the percentile by linear interpolation; fraction 0.5 is the median."""
    if not values:
        raise ValueError("Cannot take a percentile of no values.")

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("The fraction must be between 0 and 1.")

    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index

    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def summarize_time_to_first_audio(records: Sequence[dict]) -> dict[str, float]:
    """Return the turn count and the p50 and p95 time to first audio in milliseconds."""
    values = [time_to_first_audio_milliseconds(record) for record in records]

    return {
        "turn_count": float(len(values)),
        "p50_milliseconds": percentile(values, 0.5),
        "p95_milliseconds": percentile(values, 0.95),
    }


def summarize_stages(records: Sequence[dict]) -> dict[str, dict[str, float]]:
    """Return the p50 and p95 in milliseconds of every stage and of the time to first audio.

    The stages are asr, llm_start, speech_start and llm_remaining (see STAGES in the trace module).
    The p50 of the stages do not add up to the p50 of the time to first audio, because the median turn
    of one stage is rarely the median turn of another.
    """
    durations = [stage_durations_milliseconds(record) for record in records]
    names = list(durations[0]) if durations else []

    return {
        name: {
            "p50_milliseconds": percentile([turn[name] for turn in durations], 0.5),
            "p95_milliseconds": percentile([turn[name] for turn in durations], 0.95),
        }
        for name in names
    }
