# Assignment 4 report: [name your agent]

> How to use this template: write your report in any tool and submit a PDF to Gradescope, at most 3 pages, not counting the half-page AI-use disclosure appendix at the end. Delete every line that starts with ">" before you submit. Numbers in the report must be the numbers in the `trace.jsonl` you ship.

**Partners:** [name, Georgia Tech username] and [name, Georgia Tech username]
**Models:** LLM [name, and reasoning effort if you lowered it]; ASR [name and size]; TTS [name and voice]
**Run behind the figures:** [laptop or PACE-ICE job id]; [CPU and GPU]; [number of turns]; trace file [path inside your zip]

## 1. The system

> Five or six lines. The pipeline in one sentence; which stages stream and which wait for the previous one; the settings you chose that affect time (thread counts, model sizes, how the LLM reply is cut into pieces for the TTS, how much of the knowledge base goes into each prompt); how you warm the models up before the first timed turn.

## 2. The latency waterfall

> Time to first audio (TTFA) is the time from `end_of_user_speech` to `tts_first_audio`. Cut it into the stages below, and draw them as one stacked waterfall figure with the value written on each bar. Use at least 30 turns.

[Figure: stacked waterfall of the median stages, ending at the TTFA. Value on every bar.]

| | p50 (ms) | p95 (ms) |
|---|---|---|
| **TTFA, all turns** | | |
| ASR: `asr_final` minus `end_of_user_speech` | | |
| LLM start: `llm_first_token` minus `asr_final` | | |
| Speech start: `tts_first_audio` minus `llm_first_token` | | |
| Not on the TTFA path: `llm_complete` minus `llm_first_token` | | |

Turns counted: [n] of [n].

> `summarize_stages` in `voice_kit` prints every number of this table from your trace. The p50 and p95 of the TTFA row must match the trace. The percentiles of the stages need not add up to the TTFA percentile, because the p95 turn of one stage is rarely the p95 turn of another. Say which one the figure shows.

**How I treat stages that overlap:** [one or two sentences. For example, whether your TTS starts before the LLM finishes, and so which stage the overlap is counted in.]

**How to recompute these numbers:** [the command or script in your repository]

## 3. Word error rate on our own voices

| Speaker | Utterances | Reference words | Pooled WER | Mean WER per utterance |
|---|---|---|---|---|
| [partner 1] | | | | |
| [partner 2] | | | | |

> Two or three sentences: which voice does the ASR handle worse, and what does the difference look like in the transcripts? Quote one utterance as said and as heard. Mention anything about the recording (room, microphone, speed) that could explain it.

## 4. The bottleneck

> This section is graded on whether the evidence supports the claim.

**The dominant stage:** [which stage takes the largest share of the TTFA, with its p50 and its share of the total]

**Why:** [the cause, backed by something you measured in the trace or in a separate timing. Not a guess.]

**The fix I would ship next:** [one change, and the effect you expect on the p50, with the reasoning]

**One thing I tried that did not help:** [what, and what it showed]

## 5. Reproducing this

> What a stranger types, from a clean clone or the unzipped submission, on PACE-ICE: image build, `sbatch` command, where the trace appears. Name the seeds and where the dependency versions are pinned. The reproduction should finish within 15 minutes of the models loading.

## Appendix: AI-use disclosure

> Half a page, candid. It is graded for completeness, never against the quality of your work.

**Tools used:** [which AI tools, which models]
**What for:** [for example: scaffolding the agent, debugging a stage, explaining a library, drafting text. Say which parts of the code and of this report it touched.]
**What I wrote or decided myself:** [the design choices and the analysis in sections 2 to 4]
**How I checked it:** [what you ran or measured to confirm the AI's output was right, rather than trusting it]
