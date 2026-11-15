"""Replay recordings through local ASR, chat completion, and TTS."""

from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.config import PiperConfig

from voice_kit import (
    ASR_FINAL,
    LLM_COMPLETE,
    LLM_FIRST_TOKEN,
    ReplaySession,
    record_problems,
    summarize_stages,
)

ASR_SAMPLE_RATE_HZ = 16_000
TTS_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
DEFAULT_TTS_MODEL = Path(
    "/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache/hub/"
    "models--rhasspy--piper-voices/snapshots/"
    "cf910fb59f1d79ab3b625a63993a108751c6c25d/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--asr-model", default="Systran/faster-whisper-small.en")
    parser.add_argument("--tts-model", type=Path, default=DEFAULT_TTS_MODEL)
    parser.add_argument("--knowledge-base", type=Path, default=Path("toy_kb/knowledge_base.json"))
    parser.add_argument("--cpu-count", type=int, default=1)
    return parser.parse_args()


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def transcribe(asr: WhisperModel, pcm: bytes) -> str:
    segments, _ = asr.transcribe(
        pcm_to_float32(pcm),
        language="en",
        beam_size=1,
        vad_filter=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def request_stream(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
) -> Iterator[str]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 180,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        response = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM server returned HTTP {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not connect to LLM server: {error.reason}") from error

    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices", [])
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if isinstance(content, str):
                yield content


def complete_sentences(buffer: str) -> tuple[list[str], str]:
    parts = TTS_SENTENCE_END.split(buffer)
    if len(parts) == 1:
        return [], buffer
    complete = [part.strip() for part in parts[:-1] if part.strip()]
    return complete, parts[-1]


def speak(voice: PiperVoice, text: str, sink) -> None:
    for chunk in voice.synthesize(text):
        pcm = chunk.audio_int16_bytes
        if pcm:
            sink.write(pcm)


def speak_worker(voice: PiperVoice, sentences: queue.Queue, sink, errors: list) -> None:
    """speak each sentence from the queue on its own thread until None arrives."""
    while True:
        sentence = sentences.get()
        if sentence is None:
            return
        try:
            speak(voice, sentence, sink)
        except Exception as error:  # noqa: BLE001
            errors.append(error)


def generate_reply(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    voice: PiperVoice,
    turn,
) -> str:
    fragments: list[str] = []
    pending = ""
    first_token_seen = False

    # sentences go to a queue; the speaker thread speaks them while the LLM keeps streaming.
    to_speak: queue.Queue = queue.Queue()
    speak_errors: list = []
    speaker = threading.Thread(
        target=speak_worker, args=(voice, to_speak, turn.sink, speak_errors), daemon=True
    )
    speaker.start()

    try:
        for fragment in request_stream(base_url, model, messages):
            if not first_token_seen and fragment.strip():
                turn.stamp(LLM_FIRST_TOKEN)
                first_token_seen = True
            fragments.append(fragment)
            pending += fragment
            sentences, pending = complete_sentences(pending)
            for sentence in sentences:
                to_speak.put(sentence)

        reply = "".join(fragments).strip()
        if not first_token_seen or not reply:
            raise RuntimeError("LLM returned an empty answer.")
        turn.stamp(LLM_COMPLETE)
        if pending.strip():
            to_speak.put(pending.strip())
    finally:
        # Wait until everything queued has been spoken before the turn ends.
        to_speak.put(None)
        speaker.join()

    if speak_errors:
        raise speak_errors[0]
    return reply


def make_system_prompt(knowledge_base: dict) -> str:
    return (
        "You are a concise spoken assistant for the assignment's toy community center. "
        "Use only facts in the provided knowledge base; never invent facts or availability. "
        "If a fact is absent, say you do not know. Handle timers/reminders conversationally, "
        "repeat back important values, and ask one short question when required information "
        "is missing. For room requests, collect day, time, party size, and required feature; "
        "only propose rooms whose capacity, features, and free_hour_starts all satisfy the "
        "request. Ask for confirmation before booking, then repeat the room, day, time, and "
        "user's name. Keep answers to one or two short sentences. Preserve relevant context "
        "from earlier turns.\n\nKnowledge base:\n"
        + json.dumps(knowledge_base, ensure_ascii=False)
    )


def run(args: argparse.Namespace) -> None:
    if args.cpu_count < 1:
        raise ValueError("--cpu-count must be at least 1")
    if not args.recordings_directory.is_dir():
        raise FileNotFoundError(f"Recordings directory not found: {args.recordings_directory}")
    if not args.tts_model.is_file():
        raise FileNotFoundError(f"Piper model not found: {args.tts_model}")
    if not args.knowledge_base.is_file():
        raise FileNotFoundError(f"Knowledge base not found: {args.knowledge_base}")

    knowledge_base = json.loads(args.knowledge_base.read_text(encoding="utf-8"))
    asr = WhisperModel(
        args.asr_model,
        device="cuda",
        compute_type="float16",
        cpu_threads=args.cpu_count,
        num_workers=1,
        # load from the shared cache only; never try to reach the Hugging Face website.
        local_files_only=True,
    )
    # PiperVoice.load cannot set a thread count, so ONNX Runtime would size itself to the
    # whole node. Build the voice by hand and give it only the CPUs this job has.
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.cpu_count
    options.inter_op_num_threads = 1
    config_path = Path(f"{args.tts_model}.json")
    voice = PiperVoice(
        config=PiperConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8"))),
        session=onnxruntime.InferenceSession(
            str(args.tts_model), sess_options=options, providers=["CPUExecutionProvider"]
        ),
    )
    sample_rate = voice.config.sample_rate

    recordings = sorted(args.recordings_directory.glob("*.wav"))
    if not recordings:
        raise FileNotFoundError(f"No .wav recordings found in {args.recordings_directory}")

    # Warm each stage before collecting timed turns.
    transcribe(asr, np.zeros(ASR_SAMPLE_RATE_HZ, dtype="<i2").tobytes())
    for _ in voice.synthesize("Warming up the voice."):
        pass
    list(request_stream(args.llm_base_url, args.llm_model, [
        {"role": "user", "content": "Reply with the word ready."}
    ]))

    session = ReplaySession(
        recordings,
        args.output_directory,
        sink_sample_rate_hz=sample_rate,
    )
    messages = [{"role": "system", "content": make_system_prompt(knowledge_base)}]
    conversation: list[tuple[str, str]] = []

    for turn in session.turns():
        try:
            received_audio = bytearray()
            turn.start(on_chunk=received_audio.extend)
            turn.end_of_speech_reached.wait()

            transcript = transcribe(asr, bytes(received_audio))
            if not transcript:
                raise RuntimeError(f"No speech recognized in {turn.recording.wav_path.name}")
            turn.stamp(ASR_FINAL)
            turn.extras["transcript"] = transcript

            turn_messages = messages + [
                {"role": role, "content": content}
                for pair in conversation[-8:]
                for role, content in (("user", pair[0]), ("assistant", pair[1]))
            ]
            turn_messages.append({"role": "user", "content": transcript})
            reply = generate_reply(
                args.llm_base_url,
                args.llm_model,
                turn_messages,
                voice,
                turn,
            )
            turn.extras["reply"] = reply
            conversation.append((transcript, reply))
        except Exception as error:  # noqa: BLE001
            turn.extras["error"] = repr(error)
            print(f"{turn.turn_id} failed: {error!r}")
        finally:
            turn.finish()

    from voice_kit import read_trace

    # print p50 and p95 of every stage over the passing turns
    records = read_trace(session.trace_path)
    passing = [record for record in records if not record_problems(record)]
    print(f"{len(passing)} of {len(records)} turns pass")
    if passing:
        for stage, values in summarize_stages(passing).items():
            print(f"  {stage:>20}: p50 {values['p50_milliseconds']:7.0f} ms   p95 {values['p95_milliseconds']:7.0f} ms")

    problems = [
        (record["turn_id"], problem)
        for record in read_trace(session.trace_path)
        for problem in record_problems(record)
    ]
    if problems:
        for turn_id, problem in problems:
            print(f"{turn_id}: {problem}")
        raise RuntimeError(f"Trace validation found {len(problems)} problem(s).")
    print(f"Processed {len(recordings)} turns; trace: {session.trace_path}")


if __name__ == "__main__":
    run(parse_args())
