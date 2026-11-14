# Assignment 4: Voice Agent and Latency Waterfall

Build a spoken assistant. Then account for every millisecond between end of your question and start of its answer.

| | | | |
|---|---|---|---|
| **Due** | Wed, November 18, 11:59 pm ET | **Released** | Tue, November 10 |
| **Work mode** | Paired (teams of 2; same teams as A3) | **Effort** | ≈25 hours |
| **Code** | Canvas (zip) | **Video** | Canvas (≤3 minutes) |
| **Report** | Gradescope (PDF, ≤3 pages) | | |

## Why this big one exists

Lecture 12: humans respond in about 200 ms. Every voice system cares a lot about response speed. A4 grades instrumentation over polish: a modest agent whose latency you can account for beats a slick one whose delays are a mystery. **This artifact may be extended into your final project — rather build one good thing than build two half-things.**

## What you build

A push-to-talk spoken agent: ASR ▶️ LLM ▶️ TTS, all streaming where you can make them stream. Barge-in is optional (bonus). It must complete three scripted task families against the provided toy knowledge base: a timer or reminder task, a lookup task, and a two-constraint booking roleplay. `tasks/task_families.md` describes them and gives example dialogues; the knowledge base is `toy_kb/knowledge_base.json`.

> [!IMPORTANT]
> Everything runs locally. No hosted ASR, TTS or LLM service is allowed, so that every pair can be measured fairly on a cluster node.

- **LLM:** pick one chat model from professor's shared Hugging Face cache and serve it yourself: Llama 3.2 1B Instruct, Llama 3.2 3B Instruct, Llama 3.1 8B Instruct, or gpt-oss-20b. The model name and the server URL are settings of your agent, never hard-coded.
- **ASR:** faster-whisper or an equivalent. Shared cache holds `faster-whisper-small.en`.
- **TTS:** Piper, Kokoro or an equivalent. Shared cache holds Piper `en_US` lessac medium voice.
- TAs run are offline (`HF_HUB_OFFLINE=1`), so the weights you use must be available there. Clear any model that is not in shared cache with TAs before you rely on it.

Our shared cache is `/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache`, the same one A2 and A3 use.

## The audio shim

PACE-ICE nodes have no microphone or speaker, so your agent never opens an audio device. It reaches audio only through `voice_kit/`, which is standard library only. The shim:

- Replays a recorded utterance to your agent as the user, at real time, in 20 ms chunks,
- Captures the audio your TTS produces and writes it to a WAV file,
- Stamps two of the five timestamps itself (end of user speech and TTS first audio) on one monotonic clock, so they are not self-reported,
- Writes one record per turn to a trace file.

A recording is a 16 kHz, mono, 16-bit WAV file. Next to it, you may put a JSON file with the same name that holds `speech_end_seconds` (where the speech ends) and `scenario_node` (a label for the turn). Without `speech_end_seconds`, the shim finds the end of speech with an energy rule. **The end of user speech is the end of the speech in the recording, never the end of the file.**

> [!WARNING]
> TTS first audio is the first chunk your agent writes to the shim whose energy is above a fixed threshold, so write only the chunks your TTS actually produced, with no padding silence in front.

Record your own utterances to test with. The own-voice task below already gives you 40 between the two of you. When TAs measure your agent on PACE-ICE they replay their own recordings of the task scripts; those are not released.

### Using the shim

Put the kit folder on the Python path (`PYTHONPATH=<kit folder> python3 your_agent.py`). The skeleton of an agent:

```python
"""A skeleton of an agent: replay recordings, time every stage and write the trace."""

from pathlib import Path

from voice_kit import ASR_FINAL, LLM_COMPLETE, LLM_FIRST_TOKEN, ReplaySession

# Replace this with the sample rate of your TTS, in hertz.
TTS_SAMPLE_RATE_HZ = 22050


def run_session(recordings_directory: Path, output_directory: Path) -> None:
    """Replay every recording as one turn and write the trace."""
    session = ReplaySession(
        sorted(recordings_directory.glob("*.wav")),
        output_directory,
        sink_sample_rate_hz=TTS_SAMPLE_RATE_HZ,
    )

    for turn in session.turns():
        received_audio = bytearray()

        # The shim feeds recording at real time.
        turn.start(on_chunk=received_audio.extend)

        # Stands in for releasing push-to-talk key.
        turn.end_of_speech_reached.wait()

        transcript = your_asr(bytes(received_audio))

        # The moment the final transcript is available.
        turn.stamp(ASR_FINAL)

        turn.extras["transcript"] = transcript

        # Call your LLM here: stamp LLM_FIRST_TOKEN when the first token of the answer arrives,
        # and LLM_COMPLETE when the answer is finished.
        # Hand each sentence to your TTS as soon as it is complete, and pass every chunk it
        # produces to turn.sink.write(chunk).
        # Here your_asr and reply_text stand for your own code.

        turn.extras["reply"] = reply_text

        # Waits for replay to end and writes the record.
        turn.finish()
```

Each event can be stamped once per turn. If a turn fails, still call `turn.finish()`; the record is written without the missing events and the checker rejects that turn.

### The trace

`run_output/trace.jsonl` has one JSON object per turn:

```json
{
  "turn_id": "turn_0001",
  "scenario_node": "timer.set",
  "clock": "monotonic_ns",
  "events": {
    "end_of_user_speech": 0,
    "asr_final": 0,
    "llm_first_token": 0,
    "llm_complete": 0,
    "tts_first_audio": 0
  },
 "extras": {
   "recording": "utterance_001.wav",
   "max_feed_lag_milliseconds": 0.0,
   "transcript": "...",
   "reply": "...",
   "first_reasoning_token_nanoseconds": 0
 }
}
```

| Field | Written by | Rule |
|---|---|---|
| `turn_id`, `clock`, `scenario_node` | shim | `scenario_node` is `null` for a recording without a JSON file |
| `events.end_of_user_speech`, `events.tts_first_audio` | shim | always |
| `events.asr_final`, `events.llm_first_token`, `events.llm_complete` | your agent, with `turn.stamp` | required |
| `extras.recording`, `extras.max_feed_lag_milliseconds` | shim | always |
| `extras.transcript` | your agent | required; the own-voice WER reads it |
| `extras.reply` | your agent | required; your peers and the TAs read it |
| `extras.first_reasoning_token_nanoseconds` | your agent | required only if your model produces reasoning tokens before its answer |

Every event is nanoseconds on the shim's clock, so use `voice_kit.now_nanoseconds()` for any extra time you record.

Rules that make numbers comparable:

- **`asr_final`** is the time your final transcript is available. You may transcribe the whole utterance once the speaker stops, or transcribe while the audio arrives; either way the time is counted honestly.
- **`llm_first_token`** is the first token of the answer you will speak. For a model that reasons first (gpt-oss-20b), reasoning tokens are never spoken, so the first answer token is the event. You may lower the reasoning effort; say so in your report.
- **No event may come before `end_of_user_speech`**, and `llm_first_token` may not come after `llm_complete`. Streaming makes `tts_first_audio` come before `llm_complete` in a good agent, and that is fine.
- **Warm up before the first timed turn.** Load every model and run each stage once, so that start-up cost never lands inside a turn.

To check and summarize a run:

```python
"""Check a trace and print the numbers for the report."""

from pathlib import Path

from voice_kit import (
    read_trace,
    record_problems,
    summarize_stages,
    summarize_time_to_first_audio,
)

records = read_trace(Path("run_output/trace.jsonl"))

for record in records:
    for problem in record_problems(record):
        # Nothing is printed when every turn passes.
        print(record["turn_id"], problem)

print(summarize_time_to_first_audio(records))

# p50 and p95 of each stage, for the table in your report.
print(summarize_stages(records))
```

`record_problems` is the same check TAs run: the five events are present and in order, and the extras in the table above are filled in. A turn with a problem does not count, so fix every line it prints before you count your turns.

## Graded core: latency waterfall

Instrument every stage: end of user speech, ASR final, LLM first token, LLM completion, TTS first audio. Report the p50 and p95 time to first audio (TTFA, measured from end of user speech) over at least 30 turns, decomposed as a stacked waterfall figure. Put the value of each stage on its bar, and say how you treat stages that overlap.

> [!IMPORTANT]
> Honesty over speed: numbers that can't be recomputed from your raw logs earn zero. Ship the logs.

How these 40 points are earned:

- **20 points: your report.** The p50 and p95 in your report and on your figure must match the raw logs you ship, over at least 30 turns. TAs recompute them from your trace, and the figure is read back against it. A mismatch is referred to a TA, who decides; it is not an automatic zero.
- **20 points: performance.** TAs run your agent on PACE-ICE on common hardware and measure the p50 and p95 from that run. Full credit at p50 TTFA of 1.5 s or less. There is no partial credit: a p50 above 1.5 s earns none of these 20 points.

You still develop on your own machine. Your own numbers are how you check your work; the PACE-ICE run sets the performance points.

## Your own WER

Each partner records 20 utterances (16 kHz, mono, 16-bit WAV) and types the reference transcript of each. Your ASR transcribes them through the shim, and `wer/score_wer.py` scores the `extras.transcript` of the trace against your reference transcripts. Compare the two partners' results in your report: the same ASR can behave very differently on two voices.

```
python3 wer/score_wer.py --trace run_output/trace.jsonl --references references.json
```

`references.json` maps each recording's file name to the words you said. To score both partners in one run, group the recordings under each partner's name (`{"alex": {"alex_01.wav": "..."}, "sam": {...}}`) and the scorer reports each one separately. It ignores case and punctuation and turns digits into words on both sides, so "5" matches "five" and "3:30" matches "three thirty"; write the reference as you said it, and read the header of `score_wer.py` for the few things it does not normalize.

## Peer cross-testing

Each pair is assigned another pair's agent. You test it the same way TAs measure latency: record your own 10 dialogues across the three task families (`tasks/task_families.md`; reuse your practice dialogues if they fit), then feed them to the other pair's agent through the audio shim — not live with a microphone. You build their image yourself, with `apptainer build --fakeroot` from their submitted `recipe/agent.def`, on PACE-ICE; this is the test of whether a stranger can actually run their recipe, so do not ask them to build it for you or hand you a built image.

File your 10-line test report with `report_template/peer_test_report_template.md`: task success, where it broke, one surprise, per dialogue. Your agent's task-success grade comes from the other pair's report plus TA spot checks, never from self-grading. Writing a fair, useful test report is itself graded (10 points).

If the agent you were assigned will not build or run at all, say so in your report with the build or run log attached; a TA independently attempts it before any points are affected. This never costs the other pair their task-success points on your report alone.

TAs assign the pairings on **Fri, Nov 20**, by random assignment with no pair testing the pair that tests them back. Peer test reports are due **Wed, Nov 25**.

## Running on PACE-ICE

PACE-ICE does not allow Docker, so containers run with Apptainer, and `recipe/` holds a template (`agent.def` and `run_agent.sbatch`).

> [!WARNING]
> Three things cost people hours:
>
> - **Apptainer mounts your home directory.** Packages in your own `~/.local` then shadow the ones in your image. Run with `--env PYTHONNOUSERSITE=1` (or `--no-home`).
> - **Thread pools size themselves from the machine, not from your job.** A node has many more CPUs than your job may use, and libraries such as ONNX Runtime, CTranslate2 and PyTorch default to the machine's count. Set thread counts explicitly from the CPUs your job was given (`len(os.sched_getaffinity(0))`) and check their effect on your stage timings.
> - **Offline.** Set `HF_HOME` to the shared cache and `HF_HUB_OFFLINE=1`; nothing may download at run time.

Request a GPU with `--gres=gpu:H100:1` and pass `--nv` to Apptainer. `run_agent.sbatch` serves your LLM on the node with vLLM from the shared A4 environment, gives your job its own port (jobs on one node share the network, so a fixed port can reach someone else's server), waits for the server, and checks that it serves your model before the agent starts. Build the image once with `apptainer build --fakeroot` (the exact commands are at the top of the file), then `sbatch run_agent.sbatch`. The Llama and gpt-oss models above have been run through this path; load time is a few minutes, and it is not part of your trace.

## Grading

| Component | What full credit looks like | Points |
|---|---|---|
| Latency waterfall | Correct instrumentation, p50 and p95 over at least 30 turns, an honest decomposition, raw logs shipped, and a good PACE-ICE run (see above) | 40 |
| Task success (peer-tested) | At least 7 of 10 peer-run dialogues succeed; graceful failure on the rest | 20 |
| Peer test report you write | Fair, specific, reproducible observations about the other pair's system | 10 |
| Demo video | At most 3 minutes, an unedited single take of the three tasks | 10 |
| Bottleneck analysis | Your report names the dominant latency stage, why, and the fix you would ship next | 15 |
| AI-use disclosure | Complete and candid | 5 |
| Total | | 100 |

Bonus:

- Barge-in that works (interrupt TTS mid-utterance, correctly, with echo handled): +5. It is judged from your demo video.
- p50 TTFA of 800 ms or less on PACE-ICE, with the waterfall to prove it: +3.

## What you submit

- **Canvas, code.** Your repository as a single `.zip` file, as in A2 and A3.
- **Canvas, video.** At most 3 minutes, one unedited take. If Canvas rejects the upload for size, compress it (lower bitrate or resolution) rather than editing the take; if a compressed, unedited take still will not upload, tell TAs right away.
- **Gradescope, report.** A PDF of at most 3 pages. The half-page AI-use disclosure appendix comes on top of those 3 pages. Use `report_template/report_template.md`.

> [!NOTE]
> Your repository must run from a clean clone: a README with the exact commands, pinned dependencies and fixed seeds. If a TA cannot reproduce your headline number within tolerance in 15 minutes, the metric part is graded from what does reproduce. Include your `trace.jsonl` and the captured TTS audio of the run behind your figure.

## Policies

- **AI tools.** You may use AI tools for anything except writing your analysis for you. Every submission includes a half-page AI-use disclosure appendix: which tools, for what, and how you verified the results. The appendix is graded. An empty or evasive disclosure on work that plainly used AI is an honor-code matter; a candid disclosure never costs you points on the work itself.
- **Late work.** Five slip days for the semester, at most two per assignment, applied automatically in whole days. Beyond slip days: minus 10 points per day. For pair assignments, slip days are charged to both partners.
- **Regrades.** Written regrade requests within 7 days of grade release, on Gradescope. The whole component is regraded, in either direction.

## What is in this kit

```
README.md                         this file
voice_kit/                        the audio shim
toy_kb/knowledge_base.json        the knowledge base of the task families
tasks/task_families.md            the three task families and example dialogues
report_template/                  the report template and the peer test report template
wer/score_wer.py                  the WER scorer
recipe/                           the Apptainer run recipe template
```
#   C A I 4  
 