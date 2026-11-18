# Assignment 4 report: [name your agent]

> How to use this template: write your report in any tool and submit a PDF to Gradescope, at most 3 pages, not counting the half-page AI-use disclosure appendix at the end. Delete every line that starts with ">" before you submit. Numbers in the report must be the numbers in the `trace.jsonl` you ship.

**Partners:** [name, Georgia Tech username] and [name, Georgia Tech username]
**Models:** LLM [name, and reasoning effort if you lowered it]; ASR [name and size]; TTS [name and voice]
**Run behind the figures:** [laptop or PACE-ICE job id]; [CPU and GPU]; [number of turns]; trace file [path inside your zip]

## 1. The system

> Five or six lines. The pipeline in one sentence; which stages stream and which wait for the previous one; the settings you chose that affect time (thread counts, model sizes, how the LLM reply is cut into pieces for the TTS, how much of the knowledge base goes into each prompt); how you warm the models up before the first timed turn.

The pipeline is ASR → LLM → TTS. 

The LLM streams its response, and completed sentences are queued to Piper’s voice on CPU while generation continues. ASR waits for the full utterance, and TTS starts once the first complete sentence is available. We manage to reduce latency by buffering LLM text and sending the sentence to Piper when sentence boundary detected, which can begin speaking while the LLM continues generating. Before timing, the agent warms up ASR, TTS, and the LLM with a dummy synthesized audio and asks the LLM for a short “ready” response, so the initialization is outside the measured turns. 

 
## 2. The latency waterfall

> Time to first audio (TTFA) is the time from `end_of_user_speech` to `tts_first_audio`. Cut it into the stages below, and draw them as one stacked waterfall figure with the value written on each bar. Use at least 30 turns.

[Figure: stacked waterfall of the median stages, ending at the TTFA. Value on every bar.]

[figure:TODO]

| | p50 (ms) | p95 (ms) |
|---|---|---|
| **TTFA, all turns** |227 |336 |
| ASR: `asr_final` minus `end_of_user_speech` |47|63 |
| LLM start: `llm_first_token` minus `asr_final` |21 |38 |
| Speech start: `tts_first_audio` minus `llm_first_token` |156 |263 |
| Not on the TTFA path: `llm_complete` minus `llm_first_token` |62 |176 |

Turns counted: [61] of [61].

> `summarize_stages` in `voice_kit` prints every number of this table from your trace. The p50 and p95 of the TTFA row must match the trace. The percentiles of the stages need not add up to the TTFA percentile, because the p95 turn of one stage is rarely the p95 turn of another. Say which one the figure shows.

**How I treat stages that overlap:** [one or two sentences. For example, whether your TTS starts before the LLM finishes, and so which stage the overlap is counted in.]

[TODO]
**How to recompute these numbers:** [the command or script in your repository]
[TODO]

## 3. Word error rate on our own voices

| Speaker | Utterances | Reference words | Pooled WER | Mean WER per utterance |
|---|---|---|---|---|
| [partner 1] | 41| 343| 0.0641| 0.0764|
| [partner 2] | 20| 139|0.0072 |0.0050 |

> Two or three sentences: which voice does the ASR handle worse, and what does the difference look like in the transcripts? Quote one utterance as said and as heard. Mention anything about the recording (room, microphone, speed) that could explain it.

On these recordings, ASR handled Yu’s voice worse, especially words like "the" and people's names. 

 In Yu Wang 's recording: 

    said : what time does the makerspace open today 

    heard: what time does the makerspace open the day 


The reason might be the accent and the speed of speaking. "to" in "today" sounds very light and fast, so ASR transcribed incorrectly. 


## 4. The bottleneck

> This section is graded on whether the evidence supports the claim.

**The dominant stage:** [which stage takes the largest share of the TTFA, with its p50 and its share of the total]

Speech start stage takes the largest share, TTFA(p50) = 227. Its own p50 is 156, and its share of the total is 156/227 = 0.687 

 

**Why:** [the cause, backed by something you measured in the trace or in a separate timing. Not a guess.]

For the same run, we also measured p50 and p95 for other stages (47 ms for ASR and 21 ms for LLM start), so both speech recognition and LLM initialization are completed in a short time. Also we have observed shorter response took shorter time than longer response. For example, speech start time of the response "Booked." Is 49ms, but a longer response listing available rooms does not start speech until 200ms. 

Then the reason for taking too much time between LLM first token and TTS is that LLM is generating response tokens one by one like we usually observe. 
 

**The fix I would ship next:** [one change, and the effect you expect on the p50, with the reasoning] 


Modify the boundary definition of sentence so it triggers the TTS on commas and semicolons as well. This can further reduce the time before starting the first tts speech and p50 decreases, because we observed LLM tends to use long compound sentences.
 

**One thing I tried that did not help:** [what, and what it showed] 
[TODO, test further] We set max lengths of buffering text, but it broke the natural rhythm and intonation, generating very unnatural audio responses. 

## 5. Reproducing this

> What a stranger types, from a clean clone or the unzipped submission, on PACE-ICE: image build, `sbatch` command, where the trace appears. Name the seeds and where the dependency versions are pinned. The reproduction should finish within 15 minutes of the models loading.


The seed is the defualt 42. 

The dependency versions are under /path/to/CAI_Assignment_4/recipe/requirements.txt 

 

When a stranger tries to reproduce, he should go to the assignment directory, run commands to build the image, the agent image would be built under ~/scratch under current setting. Then run sbatch, and the output trace would be under /path/to/CAI_Assignment_4/run_output/<JOB_ID>/trace.jsonl 

 
Commands in bash:
````bash

cd /path/to/CAI_Assignment_4 

export SCRATCH_DIR="$HOME/scratch" 

mkdir -p "$SCRATCH_DIR/cache" "$SCRATCH_DIR/tmp" 

export APPTAINER_CACHEDIR="$SCRATCH_DIR/cache" 

export APPTAINER_TMPDIR="$SCRATCH_DIR/tmp" 
  
apptainer build --fakeroot "$SCRATCH_DIR/agent.sif" recipe/agent.def 

sbatch recipe/run_agent.sbatch 

```` 

Once the job is completed, the output trace would be under /path/to/CAI_Assignment_4/run_output/<JOB_ID>/trace.jsonl 

## Appendix: AI-use disclosure 

 

> Half a page, candid. It is graded for completeness, never against the quality of your work. 

 

**Tools used:** Team member Anusha Saha used [], and Team member Yu Wang used Gemini3.1 and Copolit.

**What for:** [for example: scaffolding the agent, debugging a stage, explaining a library, drafting text. Say which parts of the code and of this report it touched.]
Team member Anusha Saha []
Team member Yu Wang used Copilot to build the outline of function for agent, and used Gemini3.1 to generate recording transcripts.

**What I wrote or decided myself:** 


Team member Yu Wang wrote report questions (1,5, parts of 2,3,4)

**How I checked it:** [what you ran or measured to confirm the AI's output was right, rather than trusting it] 

For the recording transcipts, Yu Wang checked if the number and quality of transcipts satisfy the tasks requirements.
We compared the results of traces for different runs after modifying the agent code. We checked the response and transcripts recoded in the traces, and looked into the traces that failed carefully.

 

 
