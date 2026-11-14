"""Replays a recorded utterance as the user, in real time, and stamps the end of user speech."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .audio import (
    DEFAULT_CHUNK_MILLISECONDS,
    DEFAULT_ENERGY_THRESHOLD,
    SAMPLE_RATE_HZ,
    SAMPLE_WIDTH_BYTES,
    chunk_byte_count,
    detect_speech_end_seconds,
    read_wav,
)
from .clock import now_nanoseconds

BYTES_PER_SECOND = SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES
NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class Recording:
    """One recorded utterance and where its speech ends."""

    wav_path: Path
    pcm_bytes: bytes
    speech_end_seconds: float
    scenario_node: str | None


def load_recording(
    wav_path: Path,
    chunk_milliseconds: int = DEFAULT_CHUNK_MILLISECONDS,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
) -> Recording:
    """Load a WAV file; a JSON file with the same name may give speech_end_seconds and scenario_node."""
    pcm_bytes = read_wav(wav_path)
    duration_seconds = len(pcm_bytes) / BYTES_PER_SECOND

    sidecar_path = wav_path.with_suffix(".json")
    sidecar = {}

    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    # A sidecar value wins because a person marked it; the energy rule is the default.
    if "speech_end_seconds" in sidecar:
        speech_end_seconds = float(sidecar["speech_end_seconds"])

    else:
        speech_end_seconds = detect_speech_end_seconds(
            pcm_bytes, chunk_milliseconds, energy_threshold
        )

    if not 0.0 < speech_end_seconds <= duration_seconds:
        raise ValueError(
            f"{wav_path}: speech_end_seconds {speech_end_seconds} is outside "
            f"the recording, which lasts {duration_seconds:.3f} seconds."
        )

    return Recording(
        wav_path=wav_path,
        pcm_bytes=pcm_bytes,
        speech_end_seconds=speech_end_seconds,
        scenario_node=sidecar.get("scenario_node"),
    )


def sleep_until(deadline_nanoseconds: int) -> None:
    """Block until the shim clock reaches the deadline; return at once if it already has."""
    remaining_seconds = (
        deadline_nanoseconds - now_nanoseconds()
    ) / NANOSECONDS_PER_SECOND

    if remaining_seconds > 0:
        time.sleep(remaining_seconds)


class ReplaySource:
    """Feeds a recording to the agent in chunks, paced at real time from its own thread."""

    def __init__(
        self,
        recording: Recording,
        chunk_milliseconds: int = DEFAULT_CHUNK_MILLISECONDS,
    ) -> None:
        self.recording = recording
        self.chunk_milliseconds = chunk_milliseconds
        self.max_feed_lag_nanoseconds = 0
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(
        self,
        on_chunk: Callable[[bytes], None],
        on_end_of_speech: Callable[[int], None],
    ) -> None:
        """Start feeding and return at once; on_end_of_speech gets the stamp after the last speech chunk was delivered."""
        self._thread = threading.Thread(
            target=self._run, args=(on_chunk, on_end_of_speech), daemon=True
        )
        self._thread.start()

    def wait(self) -> None:
        """Wait until the whole recording has been fed, and re-raise any error from the feeding thread."""
        if self._thread is not None:
            self._thread.join()

        if self._error is not None:
            raise self._error

    def _run(
        self,
        on_chunk: Callable[[bytes], None],
        on_end_of_speech: Callable[[int], None],
    ) -> None:
        try:
            self._feed(on_chunk, on_end_of_speech)

        except BaseException as error:  # noqa: BLE001
            # The error is stored so that wait() raises it in the thread that called start().
            self._error = error

    def _feed(
        self,
        on_chunk: Callable[[bytes], None],
        on_end_of_speech: Callable[[int], None],
    ) -> None:
        pcm_bytes = self.recording.pcm_bytes
        step = chunk_byte_count(self.chunk_milliseconds)
        speech_end_seconds = self.recording.speech_end_seconds
        end_of_speech_stamped = False
        start_nanoseconds = now_nanoseconds()

        for offset in range(0, len(pcm_bytes), step):
            chunk = pcm_bytes[offset : offset + step]
            chunk_end_seconds = (offset + len(chunk)) / BYTES_PER_SECOND

            # A chunk is released when its last sample would have been captured, as a microphone does.
            release_nanoseconds = start_nanoseconds + round(
                chunk_end_seconds * NANOSECONDS_PER_SECOND
            )
            sleep_until(release_nanoseconds)

            released_nanoseconds = now_nanoseconds()
            lag_nanoseconds = released_nanoseconds - release_nanoseconds
            self.max_feed_lag_nanoseconds = max(
                self.max_feed_lag_nanoseconds, lag_nanoseconds
            )

            # The stamp is taken now but announced after the chunk is delivered, so everything the speaker
            # said has reached the agent when it learns that speech ended, as with a push-to-talk release.
            # It is moved back by the part of the chunk that came after the speech ended.
            pending_stamp_nanoseconds = None

            if not end_of_speech_stamped and chunk_end_seconds >= speech_end_seconds:
                overshoot_nanoseconds = round(
                    (chunk_end_seconds - speech_end_seconds) * NANOSECONDS_PER_SECOND
                )
                pending_stamp_nanoseconds = released_nanoseconds - overshoot_nanoseconds
                end_of_speech_stamped = True

            on_chunk(chunk)

            if pending_stamp_nanoseconds is not None:
                on_end_of_speech(pending_stamp_nanoseconds)
