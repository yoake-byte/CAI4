"""Audio shim for the voice agent: replays a recording as the user and captures the agent's speech.

The agent talks to audio only through this package. The shim stamps two of the five timestamps
itself (end of user speech and TTS first audio) on one monotonic clock, and writes one record
per turn to a JSONL trace.
"""

from .audio import DEFAULT_CHUNK_MILLISECONDS, DEFAULT_ENERGY_THRESHOLD
from .clock import now_nanoseconds
from .metrics import percentile, summarize_stages, summarize_time_to_first_audio
from .session import ReplaySession, Turn
from .trace import (
    ASR_FINAL,
    END_OF_USER_SPEECH,
    LLM_COMPLETE,
    LLM_FIRST_TOKEN,
    REQUIRED_EVENTS,
    STAGES,
    TTS_FIRST_AUDIO,
    TraceWriter,
    event_problems,
    events_are_complete_and_consistent,
    extras_problems,
    read_trace,
    record_problems,
    stage_durations_milliseconds,
    time_to_first_audio_milliseconds,
)

__all__ = [
    "ASR_FINAL",
    "DEFAULT_CHUNK_MILLISECONDS",
    "DEFAULT_ENERGY_THRESHOLD",
    "END_OF_USER_SPEECH",
    "LLM_COMPLETE",
    "LLM_FIRST_TOKEN",
    "REQUIRED_EVENTS",
    "STAGES",
    "ReplaySession",
    "TTS_FIRST_AUDIO",
    "TraceWriter",
    "Turn",
    "event_problems",
    "events_are_complete_and_consistent",
    "extras_problems",
    "now_nanoseconds",
    "percentile",
    "read_trace",
    "record_problems",
    "stage_durations_milliseconds",
    "summarize_stages",
    "summarize_time_to_first_audio",
    "time_to_first_audio_milliseconds",
]
