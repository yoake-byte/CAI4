"""Ties the replay source, the capture sink and the trace together, one turn at a time."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .audio import DEFAULT_CHUNK_MILLISECONDS, DEFAULT_ENERGY_THRESHOLD
from .clock import now_nanoseconds
from .sink import CaptureSink
from .source import Recording, ReplaySource, load_recording
from .trace import END_OF_USER_SPEECH, TTS_FIRST_AUDIO, TraceWriter

NANOSECONDS_PER_MILLISECOND = 1_000_000

# Piper's usual output rate; pass the rate of your own TTS to ReplaySession.
DEFAULT_SINK_SAMPLE_RATE_HZ = 22_050


class Turn:
    """One user utterance and everything the agent does in answer to it."""

    def __init__(
        self,
        turn_id: str,
        recording: Recording,
        trace: TraceWriter,
        sink_path: Path,
        sink_sample_rate_hz: int,
        chunk_milliseconds: int,
        energy_threshold: float,
    ) -> None:
        self.turn_id = turn_id
        self.recording = recording
        self.events: dict[str, int] = {}
        self.extras: dict = {}
        self.end_of_speech_reached = threading.Event()
        self._trace = trace
        self._source = ReplaySource(recording, chunk_milliseconds)
        self.sink = CaptureSink(
            sink_path,
            sink_sample_rate_hz,
            on_first_audio=lambda stamp: self.stamp(TTS_FIRST_AUDIO, stamp),
            energy_threshold=energy_threshold,
        )

    def start(self, on_chunk: Callable[[bytes], None]) -> None:
        """Start feeding the recording to the agent's ASR and return at once."""
        self._source.start(on_chunk, self._on_end_of_speech)

    def stamp(self, event: str, time_nanoseconds: int | None = None) -> None:
        """Record an event once per turn; the time defaults to now on the shim clock."""
        if event in self.events:
            raise ValueError(f"Event {event!r} was already stamped in {self.turn_id}.")

        self.events[event] = (
            now_nanoseconds() if time_nanoseconds is None else time_nanoseconds
        )

    def finish(self) -> None:
        """Wait for the feed to end, save the captured audio and write the turn record."""
        self._source.wait()
        self.sink.close()

        self._trace.append_turn(
            self.turn_id,
            self.events,
            scenario_node=self.recording.scenario_node,
            extras={
                **self.extras,
                "recording": self.recording.wav_path.name,
                "max_feed_lag_milliseconds": self._source.max_feed_lag_nanoseconds
                / NANOSECONDS_PER_MILLISECOND,
            },
        )

    def _on_end_of_speech(self, stamp_nanoseconds: int) -> None:
        self.stamp(END_OF_USER_SPEECH, stamp_nanoseconds)
        self.end_of_speech_reached.set()


class ReplaySession:
    """Replays a list of recordings as consecutive turns and writes one trace."""

    def __init__(
        self,
        recordings: Sequence[Path],
        output_directory: Path,
        trace_path: Path | None = None,
        sink_sample_rate_hz: int = DEFAULT_SINK_SAMPLE_RATE_HZ,
        chunk_milliseconds: int = DEFAULT_CHUNK_MILLISECONDS,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    ) -> None:
        self.recordings = list(recordings)
        self.output_directory = output_directory
        self.sink_sample_rate_hz = sink_sample_rate_hz
        self.chunk_milliseconds = chunk_milliseconds
        self.energy_threshold = energy_threshold
        self.trace_path = trace_path or output_directory / "trace.jsonl"
        self.trace = TraceWriter(self.trace_path)

    def turns(self) -> Iterator[Turn]:
        """Yield one Turn per recording; call finish() on each before asking for the next."""
        for index, wav_path in enumerate(self.recordings, start=1):
            turn_id = f"turn_{index:04d}"
            recording = load_recording(
                wav_path, self.chunk_milliseconds, self.energy_threshold
            )

            yield Turn(
                turn_id,
                recording,
                self.trace,
                self.output_directory / "tts" / f"{turn_id}.wav",
                self.sink_sample_rate_hz,
                self.chunk_milliseconds,
                self.energy_threshold,
            )
