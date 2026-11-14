"""One JSONL record per turn, holding the event timestamps of the pipeline."""

from __future__ import annotations

import json
import threading
from pathlib import Path

END_OF_USER_SPEECH = "end_of_user_speech"
ASR_FINAL = "asr_final"
LLM_FIRST_TOKEN = "llm_first_token"
LLM_COMPLETE = "llm_complete"
TTS_FIRST_AUDIO = "tts_first_audio"

# The five timestamps the assignment requires, in the order they must happen.
REQUIRED_EVENTS = (
    END_OF_USER_SPEECH,
    ASR_FINAL,
    LLM_FIRST_TOKEN,
    LLM_COMPLETE,
    TTS_FIRST_AUDIO,
)

CLOCK_NAME = "monotonic_ns"


class TraceWriter:
    """Appends one JSON line per turn; safe to call from several threads."""

    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path
        self._lock = threading.Lock()
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    def append_turn(
        self,
        turn_id: str,
        events: dict[str, int],
        scenario_node: str | None = None,
        extras: dict | None = None,
    ) -> None:
        """Write one turn record; events map an event name to nanoseconds on the shim clock."""
        record = {
            "turn_id": turn_id,
            "scenario_node": scenario_node,
            "clock": CLOCK_NAME,
            "events": events,
            "extras": extras or {},
        }

        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(record, sort_keys=True) + "\n")


def read_trace(trace_path: Path) -> list[dict]:
    """Return every turn record of a trace file."""
    with trace_path.open(encoding="utf-8") as trace_file:
        return [json.loads(line) for line in trace_file if line.strip()]


def event_problems(record: dict) -> list[str]:
    """Return what is wrong with the five events of a record; an empty list means they are fine.

    Streaming stages overlap, so TTS first audio may come before the LLM finishes. Only two things
    are checked beyond completeness: no event is earlier than the end of user speech, and the LLM
    first token is not later than the LLM completion.
    """
    events = record.get("events", {})
    problems = [
        f"event {name} is missing" for name in REQUIRED_EVENTS if name not in events
    ]

    if problems:
        return problems

    end_of_speech = events[END_OF_USER_SPEECH]

    for name in REQUIRED_EVENTS:
        if events[name] < end_of_speech:
            problems.append(f"event {name} is earlier than end_of_user_speech")

    if events[LLM_FIRST_TOKEN] > events[LLM_COMPLETE]:
        problems.append("llm_first_token is later than llm_complete")

    return problems


def events_are_complete_and_consistent(record: dict) -> bool:
    """Return True when all five required events exist and the few orderings that always hold are met."""
    return not event_problems(record)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def extras_problems(record: dict) -> list[str]:
    """Return what is wrong with the extras of a record; an empty list means they are fine.

    The shim writes the recording name and the feed lag. The agent has to write the transcript and
    the reply, as non-empty text. The first reasoning token time is optional, because only a model
    that reasons produces one; when it is there, it is a time on the shim clock that falls after the
    end of user speech and no later than the first answer token.
    """
    extras = record.get("extras")

    if not isinstance(extras, dict):
        return ["extras is missing"]

    problems = []

    if not isinstance(extras.get("recording"), str) or not extras["recording"]:
        problems.append("extras.recording is missing")

    if not _is_number(extras.get("max_feed_lag_milliseconds")):
        problems.append("extras.max_feed_lag_milliseconds is missing or not a number")

    for name in ("transcript", "reply"):
        value = extras.get(name)

        if not isinstance(value, str) or not value.strip():
            problems.append(f"extras.{name} is missing or empty")

    if "first_reasoning_token_nanoseconds" in extras:
        reasoning = extras["first_reasoning_token_nanoseconds"]
        events = record.get("events", {})

        if not _is_integer(reasoning):
            problems.append(
                "extras.first_reasoning_token_nanoseconds is not an integer"
            )

        elif END_OF_USER_SPEECH in events and reasoning < events[END_OF_USER_SPEECH]:
            problems.append(
                "extras.first_reasoning_token_nanoseconds is earlier than end_of_user_speech"
            )

        elif LLM_FIRST_TOKEN in events and reasoning > events[LLM_FIRST_TOKEN]:
            problems.append(
                "extras.first_reasoning_token_nanoseconds is later than llm_first_token"
            )

    return problems


def record_problems(record: dict) -> list[str]:
    """Return every problem of a turn record, events first; an empty list means the turn passes."""
    return event_problems(record) + extras_problems(record)


def time_to_first_audio_milliseconds(record: dict) -> float:
    """Return the time from end of user speech to the first audio, in milliseconds."""
    events = record["events"]
    elapsed_nanoseconds = events[TTS_FIRST_AUDIO] - events[END_OF_USER_SPEECH]

    return elapsed_nanoseconds / 1_000_000


# The stages of the latency waterfall, as (name, later event, earlier event). The first three add up to
# the time to first audio. The last one runs after the first token and is not on that path.
STAGES = (
    ("asr", ASR_FINAL, END_OF_USER_SPEECH),
    ("llm_start", LLM_FIRST_TOKEN, ASR_FINAL),
    ("speech_start", TTS_FIRST_AUDIO, LLM_FIRST_TOKEN),
    ("llm_remaining", LLM_COMPLETE, LLM_FIRST_TOKEN),
)


def stage_durations_milliseconds(record: dict) -> dict[str, float]:
    """Return the duration of each stage of a record in milliseconds, plus the time to first audio."""
    events = record["events"]
    durations = {
        name: (events[later] - events[earlier]) / 1_000_000
        for name, later, earlier in STAGES
    }
    durations["time_to_first_audio"] = time_to_first_audio_milliseconds(record)

    return durations
