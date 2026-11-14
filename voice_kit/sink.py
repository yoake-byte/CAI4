"""Captures the agent's speech to a WAV file and stamps the first audible chunk."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from .audio import DEFAULT_ENERGY_THRESHOLD, root_mean_square, write_wav
from .clock import now_nanoseconds


class CaptureSink:
    """Receives 16-bit mono PCM chunks from the TTS; no sound card is involved."""

    def __init__(
        self,
        output_path: Path,
        sample_rate_hz: int,
        on_first_audio: Callable[[int], None],
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    ) -> None:
        self.output_path = output_path
        self.sample_rate_hz = sample_rate_hz
        self.energy_threshold = energy_threshold
        self._on_first_audio = on_first_audio
        self._captured = bytearray()
        self._first_audio_seen = False
        self._lock = threading.Lock()

    def write(self, pcm_bytes: bytes) -> None:
        """Take one chunk; the first chunk at or above the energy threshold is the TTS first audio."""
        arrival_nanoseconds = now_nanoseconds()

        with self._lock:
            self._captured.extend(pcm_bytes)

            if self._first_audio_seen:
                return

            if root_mean_square(pcm_bytes) >= self.energy_threshold:
                self._first_audio_seen = True
                self._on_first_audio(arrival_nanoseconds)

    def close(self) -> None:
        """Write everything captured so far to the output WAV file."""
        with self._lock:
            write_wav(self.output_path, bytes(self._captured), self.sample_rate_hz)
