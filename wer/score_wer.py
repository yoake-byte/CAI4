"""Word error rate of the transcripts in a trace, against your own reference transcripts.

Run it from the kit folder:

    python3 wer/score_wer.py --trace run_output/trace.jsonl --references references.json

The references file maps a recording's file name to the words that were said. For one speaker it is a flat
mapping. For two speakers, group the recordings under each speaker's name, and the scorer reports each
speaker on their own:

    {"alex": {"alex_01.wav": "set a timer for five minutes"},
     "sam":  {"sam_01.wav": "what time does the cafe close"}}

How words are compared (the same steps run on the reference and on the ASR transcript):

* Case is ignored, and punctuation is dropped. A hyphen counts as a space. An apostrophe inside a word stays
  ("o'clock", "it's"), so "it's" and "it is" are different words: write the reference as you said it.
* Digits are turned into the words a person says, on both sides, so "5" matches "five" and "25" matches
  "twenty five". Numbers are written without "and": "105" becomes "one hundred five".
* A time with a colon is read as a time: "3:30" becomes "three thirty", "3:05" becomes "three oh five",
  and "3:00" becomes "three o'clock". A number with a dot is a decimal: "3.5" becomes "three point five".
* "1st", "22nd", "40th" and so on become "first", "twenty second", "fortieth". "$5" becomes
  "five dollars", and "20%" becomes "twenty percent". An amount with cents, such as "$2.50", is read as a decimal.
* Years are not special: "2026" becomes "two thousand twenty six", so a reference that says "twenty twenty
  six" will not match it.

Write "am" and "pm" as those two letters ("p.m." is also read as "pm").

The word error rate of an utterance is the number of word substitutions, deletions and insertions that
turn the ASR transcript into the reference, divided by the number of words in the reference. The pooled
rate adds the errors and the reference words of all utterances first, so a long utterance counts for more.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"))
IRREGULAR_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}  # fmt: skip


def number_to_words(digits: str) -> str:
    """Return how a person says a whole number written in digits.

    A number with a leading zero ("007") is read digit by digit, and so is a number of a trillion or more.
    """
    if len(digits) > 1 and digits.startswith("0") or len(digits) > 12:
        return " ".join(ONES[int(digit)] for digit in digits)

    number = int(digits)

    if number == 0:
        return "zero"

    parts = []

    for size, name in SCALES:
        if number >= size:
            parts.append(_below_one_thousand(number // size) + " " + name)
            number %= size

    if number:
        parts.append(_below_one_thousand(number))

    return " ".join(parts)


def _below_one_thousand(number: int) -> str:
    words = []

    if number >= 100:
        words.append(ONES[number // 100] + " hundred")
        number %= 100

    if number >= 20:
        words.append(TENS[number // 10])
        number %= 10

    if number:
        words.append(ONES[number])

    return " ".join(words)


def _digits_one_by_one(digits: str) -> str:
    return " ".join(ONES[int(digit)] for digit in digits)


def _time_to_words(match: re.Match) -> str:
    hour, minute = match.group(1), match.group(2)
    spoken_hour = number_to_words(str(int(hour)))

    if minute == "00":
        return f" {spoken_hour} o'clock "

    if minute.startswith("0"):
        return f" {spoken_hour} oh {ONES[int(minute)]} "

    return f" {spoken_hour} {number_to_words(minute)} "


def _dollars_to_words(match: re.Match) -> str:
    amount = match.group(1)
    unit = "dollar" if amount == "1" else "dollars"

    return f" {number_to_words(amount)} {unit} "


def _ordinal_to_words(match: re.Match) -> str:
    *head, last = number_to_words(match.group(1)).split()

    if last in IRREGULAR_ORDINALS:
        last = IRREGULAR_ORDINALS[last]

    elif last.endswith("y"):
        last = last[:-1] + "ieth"

    else:
        last += "th"

    return " " + " ".join(head + [last]) + " "


def normalize_words(text: str) -> list[str]:
    """Return the words of a text as the scorer compares them; see the top of this file."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("&", " and ")

    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    text = re.sub(r"\$\s*(\d+)(?!\d|\.\d)", _dollars_to_words, text)
    text = re.sub(r"(\d+)\s*%", r" \1 percent ", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time_to_words, text)
    text = re.sub(
        r"\b(\d+)\.(\d+)\b",
        lambda match: f" {number_to_words(match.group(1))} point {_digits_one_by_one(match.group(2))} ",
        text,
    )
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", _ordinal_to_words, text)
    text = re.sub(r"\d+", lambda match: f" {number_to_words(match.group(0))} ", text)

    text = re.sub(r"[-‐-―]", " ", text)
    text = text.replace(".", "")
    text = re.sub(r"[^a-z' ]+", " ", text)

    return [word.strip("'") for word in text.split() if word.strip("'")]


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Return the fewest word substitutions, deletions and insertions that turn one list into the other."""
    previous = list(range(len(hypothesis) + 1))

    for row, reference_word in enumerate(reference, start=1):
        current = [row]

        for column, hypothesis_word in enumerate(hypothesis, start=1):
            cost = 0 if reference_word == hypothesis_word else 1
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + cost,
                )
            )

        previous = current

    return previous[-1]


def load_references(path: Path) -> dict[str, dict[str, str]]:
    """Return {speaker: {recording name: reference text}} from a flat or a grouped references file."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or not data:
        raise ValueError(
            "The references file must be a JSON object with at least one entry."
        )

    if all(isinstance(value, str) for value in data.values()):
        grouped = {"all": data}

    elif all(isinstance(value, dict) for value in data.values()):
        grouped = data

    else:
        raise ValueError(
            "Give either one mapping of recording to text, or one such mapping per speaker, not a mix."
        )

    seen: dict[str, str] = {}

    for speaker, references in grouped.items():
        for recording, text in references.items():
            if not isinstance(text, str) or not normalize_words(text):
                raise ValueError(f"The reference of {recording} has no words.")

            if recording in seen:
                raise ValueError(
                    f"{recording} is listed for both {seen[recording]} and {speaker}."
                )

            seen[recording] = speaker

    return grouped


def read_transcripts(trace_path: Path) -> list[tuple[str, str]]:
    """Return (recording name, transcript) for every turn of a trace; a missing transcript is empty."""
    turns = []

    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            extras = json.loads(line).get("extras", {})
            transcript = extras.get("transcript")
            turns.append(
                (
                    extras.get("recording", ""),
                    transcript if isinstance(transcript, str) else "",
                )
            )

    return turns


def score(trace_path: Path, references_path: Path) -> dict:
    """Return the per-speaker results, plus the references that no turn of the trace answers."""
    grouped = load_references(references_path)
    speaker_of = {
        recording: speaker
        for speaker, speaker_references in grouped.items()
        for recording in speaker_references
    }
    results: dict[str, dict] = {
        speaker: {"utterances": [], "errors": 0, "reference_words": 0}
        for speaker in grouped
    }
    scored = set()

    for recording, transcript in read_transcripts(trace_path):
        if recording not in speaker_of:
            continue

        speaker = speaker_of[recording]

        reference = normalize_words(grouped[speaker][recording])

        errors = word_errors(reference, normalize_words(transcript))

        scored.add(recording)

        results[speaker]["utterances"].append(
            {
                "recording": recording,
                "reference": " ".join(reference),
                "transcript": " ".join(normalize_words(transcript)),
                "errors": errors,
                "reference_words": len(reference),
                "word_error_rate": errors / len(reference),
            }
        )

        results[speaker]["errors"] += errors
        results[speaker]["reference_words"] += len(reference)

    for result in results.values():
        utterances = result["utterances"]
        result["utterance_count"] = len(utterances)
        result["pooled_word_error_rate"] = (
            result["errors"] / result["reference_words"] if utterances else None
        )
        result["mean_utterance_word_error_rate"] = (
            sum(item["word_error_rate"] for item in utterances) / len(utterances)
            if utterances
            else None
        )

    return {
        "speakers": results,
        "not_in_trace": sorted(set(speaker_of) - scored),
    }


def format_report(result: dict, details: bool) -> str:
    """Return the text report of a result of score()."""
    lines = []

    for speaker, speaker_result in result["speakers"].items():
        lines.append(
            f"{speaker}: {speaker_result['utterance_count']} utterances, "
            f"{speaker_result['reference_words']} reference words"
        )

        if not speaker_result["utterance_count"]:
            lines.append("  nothing scored")
            continue

        lines.append(
            f"  pooled word error rate {speaker_result['pooled_word_error_rate']:.4f}"
            f"   mean over utterances {speaker_result['mean_utterance_word_error_rate']:.4f}"
        )
        ranked = sorted(
            speaker_result["utterances"],
            key=lambda item: item["word_error_rate"],
            reverse=True,
        )

        for item in ranked if details else ranked[:3]:
            if item["errors"] or details:
                lines.append(
                    f"  {item['recording']}: {item['word_error_rate']:.3f} "
                    f"({item['errors']} errors in {item['reference_words']} words)"
                )

                if item["errors"]:
                    lines.append(f"    said : {item['reference']}")
                    lines.append(f"    heard: {item['transcript']}")

    if result["not_in_trace"]:
        lines.append("")
        lines.append("In the references but not in the trace, so not scored:")
        lines.extend(f"  {recording}" for recording in result["not_in_trace"])

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Word error rate of a trace against reference transcripts."
    )
    parser.add_argument(
        "--trace", type=Path, required=True, help="the trace.jsonl of your run"
    )
    parser.add_argument(
        "--references", type=Path, required=True, help="JSON file of what was said"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="list every utterance, not only the worst",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    arguments = parser.parse_args(argv)

    try:
        result = score(arguments.trace, arguments.references)

    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps(result, indent=2))

    else:
        print(format_report(result, arguments.details))

    return 1 if result["not_in_trace"] else 0


if __name__ == "__main__":
    sys.exit(main())
