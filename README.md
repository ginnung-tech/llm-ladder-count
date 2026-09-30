# Counting ladder rungs in one photo

Can general-purpose multimodal LLMs count something simple in a photo, and do more reasoning, a
different prompt or a precise description help? This repository holds the scripts, the raw
answers and the metadata for every call made on **2026-09-30**: **218 calls to 10 models** via
OpenRouter, for a total of **$6.48**.

It accompanies the post *"Worth more than a thousand &lt;tokens&gt;? Are LLMs legally blind?"*
([ginnung.tech](https://ginnung.tech/worth-more-than-a-thousand-tokens-179020/)). This is a
small, informal experiment, not a benchmark. Read the limitations before drawing conclusions.

**Report:** [`report/ladder-count-report.pdf`](report/ladder-count-report.pdf) documents the design reasoning,
every prompt, full results tables and every answer to the guided prompt. It is generated from `data/`.

## The task

One photo (`data/image/`, 1024 × 768, public domain / CC0) of a replica pirate ship. The question
is how many red horizontal rope steps are on the front-most rope ladder, which runs diagonally
from the deck up to the front mast.

**The author counts 11.** Some readers count 12 by including the white anchoring bar at the
bottom. Answers within **10–12 (11 ± 1)** are scored as hits, so both readings pass. This
tolerance was chosen after the first round of testing, not fixed in advance.

## Experiments

All calls use OpenRouter's chat completions API with the prompt text first and the image second,
in one user message. Temperature and other sampling settings are left at provider defaults.
Each condition was run 5 times.

| # | What | Models | Prompts | Reasoning | Image detail | Output cap |
|---|---|---|---|---|---|---|
| 1 | Reasoning level × example number | GPT-6 Astra, Claude Opus 5.5 | `example_2`, `example_21`, `example_11` | low, medium, high (+ none for Astra, which the API rejects) | high | 4k / 12k / 24k |
| 2 | The phase-1 prompt, typos included | GPT-6 Astra, Claude Opus 5.5 | `original` | low, medium, high | default (field omitted) | 4k / 12k / 24k |
| 3 | Output cap | GPT-6 Astra, Claude Opus 5.5 | `example_11` | low | default | 2,000 / 1,000 / 100 |
| 4 | Every help: a precise description, maximum reasoning | 10 models (below) | `guided` | `xhigh` effort; for Claude a 24,000-token thinking budget | high | 32k (Gemini retry 64k) |

The exact prompt texts are in `scripts/run_reasoning_probe.py` (`PROMPTS`) and in
`data/manifest.json`. `original` is imported verbatim from the phase-1 script.

## Results

### Experiment 4: guided prompt, maximum reasoning

| Model (OpenRouter id) | Counts | Within 11 ± 1 | Mean reasoning tokens |
|---|---|---|---|
| openai/gpt-6-astra | 12, 12, 12, 12, 12 | 5 / 5 | ~3,800 |
| minimax/minimax-m3 | 10, 10, 10, 12, 14 | 4 / 5 | ~660 |
| openai/gpt-6.1-sol | 12, 12, 13, 13, 13 | 2 / 5 | ~4,200 |
| xiaomi/mimo-v2.6-pro | 11, 12, 18, 19, 20 | 2 / 5 | ~3,300 |
| anthropic/claude-opus-5.5 | 10, 19, 20, 20, 22 | 1 / 5 | ~550 |
| anthropic/claude-fable-5.1 | 11, 13, 17, 17, 20 | 1 / 5 | ~860 |
| deepseek/deepseek-v4.1-flash | 12, 13, 15, 16, 20 | 1 / 5 | ~10,800 |
| google/gemini-3.8-flash | 9, 13, 13, 13 *(4 valid of 8 attempts)* | 0 / 4 | ~31,700 |
| x-ai/grok-4.7 | 15, 16, 16, 22, 23 | 0 / 5 | ~8,700 |
| meta/muse-spark-1.3 | 23, 24, 26, 28, 35 | 0 / 5 | ~8,800 |

**Overall, 16 of 49 valid answers (33%) are within 11 ± 1.** Four Gemini attempts failed on the
provider side (`finish_reason: "error"`, not billed); they are kept in the raw data.

### Experiments 1–3: GPT-6 Astra and Claude Opus 5.5

- **GPT-6 Astra** answered 12–14 in every valid call, apart from a single 8. It never answered 11,
  including when the prompt's example was `11`. Its most common answer was 13.
- **Claude Opus 5.5** answered anywhere from 8 to 25. With the `2` example its answers cluster
  lower (mostly 8–13); with `21` they cluster higher (mostly 17–25).
- **More reasoning** did not make either model more consistent or more accurate.
- **`reasoning: none`** is rejected for GPT-6 Astra ("Reasoning is mandatory for this endpoint").
- **With an output cap of 100 tokens**, neither model answers: all of the cap goes to reasoning
  (`finish_reason: "length"`, empty answer).

Per-condition answers can be recomputed from `data/calls.csv`.

## Limitations

- **One image, one question.** Nothing here generalises beyond counting thin, repeated structures
  in a single 1024 × 768 photo.
- **5 runs per condition** show the spread of answers, not precise rates.
- **The ground truth is a human count** (11, or 12 if you include the white bar). The ± 1 scoring
  covers both readings.
- **Reasoning levels are provider-specific.** OpenRouter maps `low` / `medium` / `high` / `xhigh`
  to each vendor's own settings, so the same label does not mean the same effort across models.
  Compare using the logged `reasoning_tokens`, not the labels.
- **Image detail** (`detail: "high"`) is an OpenAI-style field. Whether other providers honour it
  is not documented.
- **Models change over time.** The same GPT-6 Astra prompt and settings gave mostly 12 in phase 1
  (2026-09-22/23) and mostly 13 here, a week later. OpenRouter does not expose a model version, so
  a re-run may differ.
- **The models' own explanations** (experiment 4) describe a careful count even when the count is
  far off. Treat them as text, not as evidence of how the number was produced.

## Repository layout

```
scripts/
  run_reasoning_probe.py   the runner used for every call in data/raw
  run_image_probe.py       phase-1 script; imported for the API URL, the image path and the original prompt
  build_metadata.py        rebuilds data/manifest.json and data/calls.csv from data/raw
  build_report.py          rebuilds report/ladder-count-report.html from data/ (print it to PDF with Chrome/Edge)
report/
  ladder-count-report.pdf  the written report; .html is its source
data/
  image/                   the photo (CC0)
  raw/*.jsonl              one JSON line per call: settings, full answer or error, usage, cost, latency
  manifest.json            per-file description, call and error counts, cost, SHA-256; prompts; image checksum
  calls.csv                one row per call with the parsed count and a within-11±1 flag
```

A raw line holds `model`, `prompt`, `level`, `detail`, `max_tokens`, `repeat`, `seed`, then
either `answer` + `finish_reason` + `provider` + token usage + `cost` + `latency_s`, or `error`.
`value` and `deviation` are the parsed count and any format deviation.

## Replicating

Requires Python 3.10+ and only the standard library. Set an OpenRouter key first:

```
export OPENROUTER_API_KEY=sk-or-...        # PowerShell: $env:OPENROUTER_API_KEY = 'sk-or-...'
```

Every command is a dry run (plan + rough cost, nothing sent) until you add `--run`. New results
are written to `data/raw/`.

```
# Experiment 1 (Astra shown; use --model anthropic/claude-opus-5.5 for Opus)
python scripts/run_reasoning_probe.py --model openai/gpt-6-astra --prompts example_2,example_21,example_11 --levels low,medium,high --run

# Experiment 2
python scripts/run_reasoning_probe.py --model openai/gpt-6-astra --prompts original --detail default --run

# Experiment 3 (repeat with --max-tokens 1000 and 100)
python scripts/run_reasoning_probe.py --model openai/gpt-6-astra --prompts example_11 --levels low --detail default --max-tokens 2000 --run

# Experiment 4 (xhigh for most models; for Claude models use --levels max)
python scripts/run_reasoning_probe.py --model minimax/minimax-m3 --prompts guided --levels xhigh --workers 2 --run
python scripts/run_reasoning_probe.py --model anthropic/claude-opus-5.5 --prompts guided --levels max --workers 2 --run

# Rebuild the metadata from data/raw
python scripts/build_metadata.py

# Rebuild the report, then print it to PDF
python scripts/build_report.py
chrome --headless --no-pdf-header-footer --print-to-pdf=report/ladder-count-report.pdf report/ladder-count-report.html
```

`none` was dropped from the script's levels after GPT-6 Astra rejected it. The 15 rejected calls
are still in `data/raw/reasoning-20260930-042404.jsonl`.

Calls run 5 at a time by default (`--workers`), with the first five started 2 s apart, in random
order (the seed is recorded on every line). The runner stops starting new calls on HTTP 401
(bad key) or 402 (out of credits).

## License

Everything in this repository (the scripts, the raw results, the metadata and the photo) is
dedicated to the public domain under CC0 1.0 Universal (see `LICENSE`). No attribution is
required.
