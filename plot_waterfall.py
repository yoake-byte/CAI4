"""Recompute the report's latency numbers from a trace and draw the waterfall figure.

    PYTHONPATH=. python3 plot_waterfall.py run_output/7147192/trace.jsonl

Prints the same p50/p95 the TAs compute (voice_kit's checker and summarize_stages), then writes
waterfall.png next to the trace. The figure shows the p50 of each stage on the TTFA path, and the
p50 TTFA as the last bar. Medians do not add up, so the stage bars end close to, not exactly at, it.
"""

import sys
from pathlib import Path

from voice_kit import read_trace, record_problems, summarize_stages, summarize_time_to_first_audio

trace_path = Path(sys.argv[1])
records = read_trace(trace_path)
passing = [record for record in records if not record_problems(record)]
stages = summarize_stages(passing)
ttfa = summarize_time_to_first_audio(passing)

print(f"Turns counted: {len(passing)} of {len(records)}")
for name, values in stages.items():
    print(f"{name:>20}: p50 {values['p50_milliseconds']:6.0f} ms   p95 {values['p95_milliseconds']:6.0f} ms")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is not installed: the numbers above are complete, the figure was skipped.")

STAGE_COLOR, TOTAL_COLOR = "#2a78d6", "#52514e"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

rows = [
    ("ASR", stages["asr"]["p50_milliseconds"]),
    ("LLM start", stages["llm_start"]["p50_milliseconds"]),
    ("Speech start (TTS)", stages["speech_start"]["p50_milliseconds"]),
]
total = ttfa["p50_milliseconds"]

fig, ax = plt.subplots(figsize=(7, 2.8), dpi=200)
start = 0.0
for index, (label, value) in enumerate(rows):
    ax.barh(index, value, left=start, height=0.6, color=STAGE_COLOR, edgecolor="white", linewidth=2)
    ax.text(start + value + 3, index, f"{value:.0f} ms", va="center", fontsize=9, color=TEXT)
    start += value
ax.barh(len(rows), total, height=0.6, color=TOTAL_COLOR, edgecolor="white", linewidth=2)
ax.text(total + 3, len(rows), f"{total:.0f} ms", va="center", fontsize=9, color=TEXT, fontweight="bold")

ax.set_yticks(range(len(rows) + 1), [label for label, _ in rows] + ["Time to first audio"])
ax.invert_yaxis()
ax.set_xlim(0, total * 1.18)
ax.set_xlabel("Milliseconds after end of user speech (p50)", fontsize=9, color=MUTED)
ax.set_title(f"Latency waterfall, p50 over {len(passing)} turns", fontsize=11, color=TEXT, loc="left")
ax.tick_params(colors=MUTED, labelsize=9, length=0)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
fig.text(0.01, 0.01, f"Stage bars are each stage's own p50 (sum {start:.0f} ms); the last bar is the p50 TTFA.",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))

output = trace_path.parent / "waterfall.png"
fig.savefig(output)
print(f"figure: {output}")
