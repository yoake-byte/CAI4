"""Replay recordings through local ASR, chat completion, and TTS."""

from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import time
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
        "from earlier turns. Say times in words (two in the afternoon, not 14:00) and never use "
        "lists. When a ROOM CHECK or TIMERS line is given, trust it over your own reading of the "
        "knowledge base, and never read those lines out.\n\nKnowledge base:\n"
        + json.dumps(knowledge_base, ensure_ascii=False)
    )


# Compute room availability and timer state deterministically and keep in state for LLM to use

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
FEATURES = {"projector": "projector", "whiteboard": "whiteboard", "tv": "tv screen", "screen": "tv screen",
            "video call": "video call kit", "microphone": "microphone", "mic": "microphone",
            "stage": "stage", "quiet": "quiet"}
NUMBERS = {"forty five": 45, "fifteen": 15, "twelve": 12, "eleven": 11, "ten": 10, "nine": 9, "eight": 8,
           "seven": 7, "six": 6, "five": 5, "four": 4, "three": 3, "two": 2, "one": 1, "twenty": 20,
           "thirty": 30, "forty": 40}
NOT_NAMES = {"the", "a", "an", "me", "us", "it", "my", "our", "just", "only", "now", "today", "tonight",
             "tomorrow", "noon", "people", "person", "everyone", "please", "that", "this"}


def plain(text: str) -> str:
    """Lowercase, no punctuation, number words as digits."""
    t = re.sub(r"[^\w\s]", " ", text.lower())
    t = re.sub(r"\b([ap]) m\b", r"\1m", t)
    t = t.replace("quietly", "quiet")
    for word, number in NUMBERS.items():
        t = re.sub(rf"\b{word}\b", str(number), t)
    return t


def say_hour(hour: int) -> str:
    return "noon" if hour == 12 else f"{hour % 12} {'am' if hour < 12 else 'pm'}"


def read_hour(t: str) -> int | None:
    """The hour asked for, 24-hour clock. A bare 'at 2' means 2 pm at this venue."""
    if re.search(r"\bnoon\b", t):          # not "afternoon"
        return 12
    m = re.search(r"\b(\d{1,2})\s*(?:(am|pm)|in the (morning|afternoon|evening))", t)
    if m:
        return int(m[1]) % 12 + (12 if m[2] == "pm" or m[3] in ("afternoon", "evening") else 0)
    m = re.search(r"\bat (\d{1,2})\b(?! (?:people|minutes|hours))", t)
    if m:
        return int(m[1]) + 12 if 1 <= int(m[1]) <= 6 else int(m[1])
    return None


class Booking:
    """The current booking request; the newest value of each constraint wins."""

    def __init__(self) -> None:
        self.active, self.day, self.hour, self.afternoon = False, None, None, False
        self.people, self.room, self.features, self.equipment_asked = None, None, set(), False
        self.proposed, self.confirmed, self.name, self.done, self.long = False, False, None, False, False

    def update(self, transcript: str, last_reply: str, kb: dict) -> None:
        t = plain(transcript)
        if self.done:                # the last turn finished a booking
            self.__init__()
        mentions = re.search(r"\b(rooms?|book|booking|reserve|hall)\b", t)
        if mentions and (not self.active or re.match(r"\s*(i need|i want|can i|do you have|a room)", t)):
            self.__init__()          # a new booking request
            self.active = True
        elif not self.active:
            return
        elif not mentions and re.search(r"\b(remind|timers?|cost|where|what time does)\b", t):
            self.__init__()          # the user moved on to something else
            return

        self.equipment_asked |= "equipment" in last_reply.lower()
        self.long = bool(re.search(r"\b([2-9]|\d\d) hours\b", t))
        if self.proposed:            # a room was proposed: listen for a yes and a name
            self.confirmed |= bool(re.search(r"\b(yes|yeah|yep|sure|ok|okay|go ahead|book it|fine)\b", t))
            m = re.search(r"\b(?:name(?: is)?|under|for)\s+(?:the name\s+)?([A-Za-z]+)", transcript, re.I)
            if m and m[1].lower() not in NOT_NAMES and m[1].lower() not in NUMBERS and m[1].lower() not in DAYS:
                self.name = m[1][0].upper() + m[1][1:]
            elif self.confirmed and "name" in last_reply.lower() and re.fullmatch(r"\W*([A-Za-z]+)\W*", transcript):
                self.name = re.search(r"[A-Za-z]+", transcript)[0].title()
        self.day = next((d for d in DAYS if d in t), self.day)
        hour = read_hour(t)
        if hour is not None:
            self.hour, self.afternoon = hour, False
        elif "afternoon" in t:
            self.hour, self.afternoon = None, True
        m = re.search(r"\b(\d+) (?:people|person)|\bfor (\d+)\b(?! (?:pm|am|in the|minutes|hours))", t)
        if m:
            self.people = int(m[1] or m[2])
        elif re.search(r"\b(just me|only me|myself)\b", t):
            self.people = 1
        elif "how many" in last_reply.lower() and re.fullmatch(r"\D*\d+\D*", t):
            self.people = int(re.search(r"\d+", t)[0])
        found = {name for word, name in FEATURES.items() if re.search(rf"\b{word}\b", t)}
        if found:
            self.features = found if "instead" in t else self.features | found
        for room in kb["rooms"]:
            if room["name"].split()[0].lower() in t:
                self.room = room
                if room["name"] in last_reply:   # accepting the room we offered as the alternative
                    self.people = min(self.people or 1, room["capacity"])
                    self.features &= set(room["features"])

    def check(self, kb: dict) -> str:
        """Which rooms meet every constraint so far, worked out from the knowledge base."""
        people = self.people or 1
        note = " Say that rooms can only be booked for 1 hour at a time." if self.long else ""
        fits = [r for r in ([self.room] if self.room else kb["rooms"])
                if r["capacity"] >= people and self.features <= set(r["features"])]
        if not fits:
            big = [r for r in kb["rooms"] if r["capacity"] >= people]
            if not big:
                largest = max(kb["rooms"], key=lambda r: r["capacity"])
                return (f"No room holds {people}. The largest is {largest['name']}, which holds "
                        f"{largest['capacity']}. Offer it for a smaller group." + note)
            # Fix 5: say what the alternative lacks, and when it is free on the day asked for.
            alt = max(big, key=lambda r: (len(self.features & set(r["features"])), -r["capacity"]))
            missing = ", ".join(sorted(self.features - set(alt["features"])))
            when = ""
            if self.day:
                hours = alt["free_hour_starts"].get(self.day)
                when = (f" It is free on {self.day} at {', '.join(say_hour(int(h[:2])) for h in hours)}." if hours
                        else f" It is not free on {self.day}; it is free on {', '.join(alt['free_hour_starts'])}.")
            return (f"No room holds {people} and has {', '.join(sorted(self.features))}. Closest: {alt['name']}, "
                    f"holds {alt['capacity']}, but has no {missing}.{when} Offer it instead." + note)
        # Fix 4: while asking, the LLM must not name a room (it named a wrong one).
        if self.people is None and not self.room:
            return "Ask how many people. Do not name a room yet." + note
        if not self.features and not self.equipment_asked and not self.room:
            return "Ask if they need any equipment, like a projector or a whiteboard. Do not name a room yet." + note
        if not self.day or (self.hour is None and not self.afternoon):
            return f"Rooms that fit: {', '.join(r['name'] for r in fits)}. Ask for the missing day or time." + note
        for r in fits:
            hours = [int(h[:2]) for h in r["free_hour_starts"].get(self.day, [])]
            match = [h for h in hours if h == self.hour or (self.afternoon and 12 <= h < 17)]
            if match:
                when = f"{self.day.title()} at {say_hour(match[0])}"
                if self.confirmed and self.name:
                    self.done = True
                    return f"The user confirmed. Say: Booked. {r['name']}, {when}, for {self.name}." + note
                if self.confirmed:
                    return f"The user said yes to {r['name']}, {when}. Ask for the name to book it under." + note
                self.proposed = True
                return (f"{r['name']} on floor {r['floor']} is free on {when}, holds {r['capacity']}, has "
                        f"{', '.join(r['features'])}. Propose it and ask if they want to book it." + note)
        # Nothing free then: other times that day if there are any, otherwise the days each room is free.
        same_day = {r["name"]: r["free_hour_starts"][self.day] for r in fits if r["free_hour_starts"].get(self.day)}
        if same_day:
            free = "; ".join(f"{name} at {', '.join(say_hour(int(h[:2])) for h in hours)}" for name, hours in same_day.items())
            return f"No room that fits is free then. Free on {self.day}: {free}. Offer these." + note
        free = "; ".join(f"{r['name']} on {', '.join(r['free_hour_starts'])}" for r in fits)
        return f"No room that fits is free on {self.day}. {free}. Offer one of those days." + note


class Timers:
    """Timers kept in code, so 'how long is left' is real arithmetic."""

    def __init__(self) -> None:
        self.items: dict[str, list[float]] = {}   # label -> [length in seconds, start time]
        self.last, self.pending, self.recent = None, None, False
        self.focus, self.cancelled = [], None     # what this turn's TIMERS line talks about

    def update(self, transcript: str) -> bool:
        """Apply a timer request; return True when this turn was about timers."""
        t = plain(transcript)
        self.focus, self.cancelled = [], None
        seconds = sum(int(n) * (3600 if u.startswith("h") else 60) for n, u in re.findall(r"(\d+) (minutes?|hours?)", t))
        named = (re.search(r"\bfor (?:the |my )?([a-z]+)", t) or re.search(r"\b(?:the|this|my) ([a-z]+) timer\b", t)
                 or re.search(r"\bleft on the ([a-z]+)", t))
        label = named[1] if named else self.last
        asks = re.search(r"\b(timers?|on the clock)\b", t) and "remind" not in t
        follow_up = self.items and (re.search(r"\b(is left|time left|remaining)\b", t)
                                    or (self.recent and re.search(r"\b(make it|add|cancel)\b", t)))

        if self.pending and seconds:             # the length for a timer asked for without one
            self.items[self.pending] = [seconds, time.monotonic()]
            self.last, self.pending = self.pending, None
            self.focus = [self.last]
        elif not (asks or follow_up):
            self.recent, self.pending = False, None
            return False
        elif re.search(r"\b(cancel|stop)\b", t):
            if self.items.pop(label, None) is not None:
                self.cancelled = label
        elif re.search(r"\b(add|extend)\b", t) and seconds and label in self.items:
            self.items[label][0] += seconds
            self.focus = [label]
        elif re.search(r"\b(make it|change)\b", t) and label in self.items and re.search(r"\d+", t):
            self.items[label][0] = seconds or 60 * int(re.search(r"\d+", t)[0])   # keeps the start time
            self.focus = [label]
        elif seconds:
            for part in t.split(" and "):        # "a 5 minute timer for X and a 2 minute timer for Y"
                part_seconds = sum(int(n) * (3600 if u.startswith("h") else 60)
                                   for n, u in re.findall(r"(\d+) (minutes?|hours?)", part))
                part_label = re.search(r"\bfor (?:the |my )?([a-z]+)", part)
                if part_seconds:
                    self.last = part_label[1] if part_label else f"timer {len(self.items) + 1}"
                    self.items[self.last] = [part_seconds, time.monotonic()]
                    self.focus.append(self.last)
        elif asks and not follow_up and not re.search(r"\b(what|which|list)\b", t):
            self.pending = label or "timer"      # no length given: the LLM should ask for one
        elif re.search(r"\b(what|which|list)\b", t) and not follow_up:
            self.focus = list(self.items)        # "what timers do I have?"
        else:                                    # "how long is left (on the laundry)?"
            self.focus = [label] if label in self.items else list(self.items)
        self.recent = True
        return True

    def summary(self) -> str:
        if self.pending:
            return f"A timer for {self.pending} needs a length: ask how long."
        now = time.monotonic()
        parts = [f"The {self.cancelled} timer is cancelled."] if self.cancelled else []
        for label in self.focus:
            length, start = self.items[label]
            left = max(0, round((length - (now - start)) / 60))
            name = "The timer" if label.startswith("timer ") else f"The {label} timer"
            parts.append(f"{name} is set for {round(length / 60)} minutes and has about {left} minutes left.")
        if self.items and not self.focus and not self.cancelled:
            parts.append(f"Timers running: {', '.join(self.items)}.")
        return " ".join(parts) or "No timers are running."


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
    booking, timers = Booking(), Timers()

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
            # Facts worked out in code for this turn, added to this user message only.
            booking.update(transcript, conversation[-1][1] if conversation else "", knowledge_base)
            facts = ["ROOM CHECK: " + booking.check(knowledge_base)] if booking.active else []
            if timers.update(transcript):
                facts.append("TIMERS: " + timers.summary())
            turn.extras["facts"] = facts
            turn_messages.append({"role": "user", "content": "\n\n".join([transcript] + facts)})
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
