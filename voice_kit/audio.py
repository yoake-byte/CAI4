"""Reading and writing WAV files and measuring loudness, using only the standard library."""

from __future__ import annotations

import array
import math
import sys
import wave
from pathlib import Path

SAMPLE_RATE_HZ = 16_000
SAMPLE_WIDTH_BYTES = 2
CHANNEL_COUNT = 1

DEFAULT_CHUNK_MILLISECONDS = 20

# Root mean square of 16-bit samples below which a chunk counts as silence (about -44 dBFS).
DEFAULT_ENERGY_THRESHOLD = 200.0


def root_mean_square(pcm_bytes: bytes) -> float:
    """Return the root mean square of little-endian 16-bit mono samples."""
    usable_length = len(pcm_bytes) - len(pcm_bytes) % SAMPLE_WIDTH_BYTES
    samples = array.array("h")
    samples.frombytes(pcm_bytes[:usable_length])

    # WAV data is little-endian, so a big-endian machine has to swap the bytes.
    if sys.byteorder == "big":
        samples.byteswap()

    if not samples:
        return 0.0

    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def read_wav(wav_path: Path) -> bytes:
    """Return the PCM bytes of a 16 kHz mono 16-bit WAV file, or raise ValueError for any other format."""
    with wave.open(str(wav_path), "rb") as wav_file:
        actual_format = (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
        )
        expected_format = (SAMPLE_RATE_HZ, CHANNEL_COUNT, SAMPLE_WIDTH_BYTES)

        if actual_format != expected_format:
            raise ValueError(
                f"{wav_path} must be 16 kHz mono 16-bit PCM; "
                f"got rate, channels, sample width = {actual_format}."
            )

        return wav_file.readframes(wav_file.getnframes())


def write_wav(wav_path: Path, pcm_bytes: bytes, sample_rate_hz: int) -> None:
    """Write mono 16-bit PCM bytes to a WAV file."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNEL_COUNT)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm_bytes)


def chunk_byte_count(chunk_milliseconds: int) -> int:
    """Return how many bytes one chunk holds at the shim's input format."""
    samples_per_chunk = SAMPLE_RATE_HZ * chunk_milliseconds // 1000

    return samples_per_chunk * SAMPLE_WIDTH_BYTES


def detect_speech_end_seconds(
    pcm_bytes: bytes,
    chunk_milliseconds: int = DEFAULT_CHUNK_MILLISECONDS,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
) -> float:
    """Return the end of the last chunk whose energy reaches the threshold, in seconds."""
    step = chunk_byte_count(chunk_milliseconds)
    last_loud_end = None

    for start in range(0, len(pcm_bytes), step):
        chunk = pcm_bytes[start : start + step]

        if root_mean_square(chunk) >= energy_threshold:
            last_loud_end = start + len(chunk)

    if last_loud_end is None:
        raise ValueError("No speech found: every chunk is below the energy threshold.")

    return last_loud_end / (SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES)
