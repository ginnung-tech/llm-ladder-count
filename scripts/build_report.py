"""Build the written report (report/ladder-count-report.html) from data/.

    python scripts/build_report.py

Every table and every number in the text is computed from data/raw and
data/manifest.json; the prose around them is fixed. Print the HTML to PDF with
any Chromium browser, e.g.

    msedge --headless --no-pdf-header-footer --print-to-pdf=report/ladder-count-report.pdf report/ladder-count-report.html
"""

import html
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_reasoning_probe import MAX_TOKENS, PROMPTS, REASONING, parse_answer  # noqa: E402

OUT = ROOT / "report" / "ladder-count-report.html"
REPO = "https://github.com/ginnung-tech/llm-ladder-count"
ARTICLE = "https://ginnung.tech/worth-more-than-a-thousand-tokens-179020/"

NAMES = {
    "openai/gpt-6-astra": "GPT-6 Astra",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
    "anthropic/claude-opus-5.5": "Claude Opus 5.5",
    "anthropic/claude-fable-5.1": "Claude Fable 5.1",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "x-ai/grok-4.7": "Grok 4.7",
    "meta/muse-spark-1.3": "Muse Spark 1.3",
    "minimax/minimax-m3": "MiniMax-M3",
    "xiaomi/mimo-v2.6-pro": "MiMo-V2.6-Pro",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
}
PAIR = ["openai/gpt-6-astra", "anthropic/claude-opus-5.5"]
PROMPT_ORDER = ["example_2", "example_21", "example_11", "original"]
LEVEL_ORDER = ["none", "low", "medium", "high"]
LEVEL_CAPS = {MAX_TOKENS[lv] for lv in ("low", "medium", "high")}
e = html.escape


# ---------------------------------------------------------------- data

def load() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "data" / "raw").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["file"] = path.name
                r["ok"] = "answer" in r and r.get("finish_reason") != "error"
                r["count"] = parse_answer(r["answer"])[0] if r["ok"] else None
                rows.append(r)
    return rows


def experiment(r: dict) -> int:
    if r["prompt"] == "guided":
        return 4
    if r["max_tokens"] not in LEVEL_CAPS and r["level"] != "none":
        return 3
    return 2 if r["prompt"] == "original" else 1


def hit(n: int | None) -> bool:
    return n is not None and 10 <= n <= 12


def stats(rs: list[dict]) -> dict:
    counts = sorted(r["count"] for r in rs if r["count"] is not None)
    rt = [r["reasoning_tokens"] for r in rs if r["ok"] and r.get("reasoning_tokens") is not None]
    lat = [r["latency_s"] for r in rs if r["ok"]]
    return {
        "calls": len(rs), "valid": len(counts), "counts": counts,
        "errors": sum(not r["ok"] for r in rs),
        "empty": sum(r["ok"] and r["count"] is None for r in rs),
        "hits": sum(hit(n) for n in counts),
        "median": statistics.median(counts) if counts else None,
        "min": counts[0] if counts else None, "max": counts[-1] if counts else None,
        "rt": round(statistics.mean(rt)) if rt else None,
        "cost": sum(float(r.get("cost") or 0) for r in rs),
        "lat": statistics.median(lat) if lat else None,
    }


# ---------------------------------------------------------------- formatting

def num(x, dash="–") -> str:
    if x is None:
        return dash
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return f"{x:,}" if isinstance(x, int) else f"{x:,.1f}"


def counts_cell(s: dict) -> str:
    parts = [f"<b>{n}</b>" if hit(n) else str(n) for n in s["counts"]]
    missing = s["calls"] - s["valid"]
    if missing:
        parts.append(f"<span class=muted>+{missing}&nbsp;none</span>")
    return ", ".join(parts)


def table(head: list[str], body: list[list[str]], cls: str = "", widths: list[str] | None = None) -> str:
    cols = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    th = "".join(f"<th>{h}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body)
    return f'<table class="{cls}"><colgroup>{cols}</colgroup><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def pct(a: int, b: int) -> str:
    return f"{a} / {b}" + (f" ({round(100 * a / b)}%)" if b else "")


def usd(x: float) -> str:
    return f"${x:.2f}" if x >= 0.01 or x == 0 else f"${x:.4f}"


# ---------------------------------------------------------------- sections

def section_prompts() -> str:
    labels = {
        "original": "Phase-1 prompt, verbatim (typos kept on purpose). Used in experiment 2.",
        "example_2": "Cleaned stem, example <code>2</code>. Experiment 1.",
        "example_21": "Cleaned stem, example <code>21</code>. Experiment 1.",
        "example_11": "Cleaned stem, example <code>11</code> (the author's count). Experiments 1 and 3.",
        "guided": "Guided prompt: location, exclusions, explanation and a <code>COUNT:</code> line. Experiment 4.",
    }
    out = []
    for key in ["original", "example_2", "example_21", "example_11", "guided"]:
        out.append(f'<div class="prompt"><div class="plabel"><code>{key}</code> — {labels[key]}</div>'
                   f'<pre>{e(PROMPTS[key])}</pre></div>')
    return "\n".join(out)


def section_exp12(rows: list[dict]) -> str:
    out = []
    for model in PAIR:
        body = []
        for p in PROMPT_ORDER:
            for lv in LEVEL_ORDER:
                rs = [r for r in rows if r["model"] == model and r["prompt"] == p and r["level"] == lv
                      and experiment(r) in (1, 2)]
                if not rs:
                    continue
                s = stats(rs)
                if s["valid"] == 0:
                    body.append([f"<code>{p}</code>", lv, rs[0]["detail"], f"<span class=muted>all {s['calls']} "
                                 "calls rejected: HTTP 400 “Reasoning is mandatory”</span>", "–", "–", "–", "–"])
                    continue
                body.append([f"<code>{p}</code>", lv, rs[0]["detail"], counts_cell(s), num(s["median"]),
                             pct(s["hits"], s["valid"]), num(s["rt"]), usd(s["cost"])])
        out.append(f"<h4>{NAMES[model]}</h4>")
        out.append(table(["Prompt", "Reasoning", "Detail", "Counts (sorted; bold = within 11 ± 1)", "Median",
                          "Within ± 1", "Mean reasoning tokens", "Cost"], body, "data",
                         ["13%", "9%", "8%", "30%", "7%", "12%", "11%", "10%"]))
    return "\n".join(out)


def section_pooled(rows: list[dict]) -> str:
    base = [r for r in rows if experiment(r) in (1, 2) and r["level"] != "none"]
    body_p, body_l = [], []
    for p in PROMPT_ORDER:
        row = [f"<code>{p}</code>"]
        for model in PAIR:
            s = stats([r for r in base if r["model"] == model and r["prompt"] == p])
            row += [f"{num(s['median'])} ({s['min']}–{s['max']})", pct(s["hits"], s["valid"])]
        body_p.append(row)
    for lv in ["low", "medium", "high"]:
        row = [lv]
        for model in PAIR:
            s = stats([r for r in base if r["model"] == model and r["level"] == lv])
            row += [f"{num(s['median'])} ({s['min']}–{s['max']})", pct(s["hits"], s["valid"]), num(s["rt"])]
        body_l.append(row)
    a, o = (NAMES[m] for m in PAIR)
    return (
        "<h4>By prompt (reasoning levels pooled, 15 calls per cell)</h4>"
        + table(["Prompt", f"{a}: median (range)", f"{a}: within ± 1", f"{o}: median (range)", f"{o}: within ± 1"],
                body_p, "data")
        + "<h4>By reasoning level (all four prompts pooled, 20 calls per cell)</h4>"
        + table(["Reasoning", f"{a}: median (range)", f"{a}: within ± 1", f"{a}: mean reasoning tokens",
                 f"{o}: median (range)", f"{o}: within ± 1", f"{o}: mean reasoning tokens"], body_l, "data")
    )


def section_exp3(rows: list[dict]) -> str:
    body = []
    for model in PAIR:
        for cap in (2000, 1000, 100):
            rs = [r for r in rows if experiment(r) == 3 and r["model"] == model and r["max_tokens"] == cap]
            s = stats(rs)
            fr = Counter(r.get("finish_reason") for r in rs)
            body.append([NAMES[model], f"{cap:,}", counts_cell(s) if s["valid"] else
                         f"<span class=muted>no answer in {s['calls']} of {s['calls']}</span>",
                         pct(s["hits"], s["valid"]) if s["valid"] else "–", num(s["rt"]),
                         ", ".join(f"{k} × {v}" for k, v in fr.items()), usd(s["cost"])])
    return table(["Model", "Output cap", "Counts (sorted)", "Within ± 1", "Mean reasoning tokens",
                  "finish_reason", "Cost"], body, "data", ["16%", "9%", "26%", "11%", "12%", "14%", "12%"])


def section_exp4(rows: list[dict]) -> tuple[str, dict]:
    g = [r for r in rows if experiment(r) == 4]
    per = []
    for model in sorted({r["model"] for r in g}):
        per.append((model, stats([r for r in g if r["model"] == model])))
    per.sort(key=lambda ms: (-ms[1]["hits"] / max(ms[1]["valid"], 1), ms[1]["median"] or 99))
    body = []
    for model, s in per:
        lv = next(r["level"] for r in g if r["model"] == model)
        reasoning = "thinking budget 24,000" if lv == "max" else "effort xhigh"
        body.append([f"{NAMES[model]}<br><span class=id>{model}</span>", reasoning, counts_cell(s),
                     num(s["median"]), pct(s["hits"], s["valid"]), num(s["rt"]), num(s["lat"]),
                     usd(s["cost"])])
    total = stats(g)
    body.append(["<b>All ten models</b>", "", f"{total['valid']} valid of {total['calls']} calls",
                 num(total["median"]), f"<b>{pct(total['hits'], total['valid'])}</b>", num(total["rt"]),
                 num(total["lat"]), usd(total["cost"])])
    return table(["Model", "Reasoning", "Counts (sorted; bold = within 11 ± 1)", "Median", "Within ± 1",
                  "Mean reasoning tokens", "Median latency (s)", "Cost"], body, "data",
                 ["20%", "11%", "24%", "7%", "11%", "10%", "9%", "8%"]), total


def section_totals(rows: list[dict]) -> str:
    names = {1: "Reasoning level × example number", 2: "Original phase-1 prompt", 3: "Output cap",
             4: "Guided prompt, maximum reasoning"}
    body = []
    for x in (1, 2, 3, 4):
        s = stats([r for r in rows if experiment(r) == x])
        body.append([str(x), names[x], str(s["calls"]), str(s["valid"]), str(s["errors"]), str(s["empty"]),
                     pct(s["hits"], s["valid"]), usd(s["cost"])])
    s = stats(rows)
    body.append(["", "<b>Total</b>", f"<b>{s['calls']}</b>", f"<b>{s['valid']}</b>", f"<b>{s['errors']}</b>",
                 f"<b>{s['empty']}</b>", f"<b>{pct(s['hits'], s['valid'])}</b>", f"<b>{usd(s['cost'])}</b>"])
    return table(["#", "Experiment", "Calls", "Valid counts", "API errors", "Empty answers", "Within ± 1",
                  "Cost"], body, "data", ["4%", "32%", "8%", "10%", "10%", "10%", "15%", "11%"])


def section_errors(rows: list[dict]) -> str:
    body = []
    groups = defaultdict(list)
    for r in rows:
        if not r["ok"]:
            reason = r.get("error") or ""
            if "Reasoning is mandatory" in reason:
                reason = "HTTP 400: Reasoning is mandatory for this endpoint and cannot be disabled."
            elif r.get("finish_reason") == "error" or "mid-generation" in reason:
                reason = "Provider error mid-generation (finish_reason “error”); not billed."
            groups[(r["model"], r["prompt"], r["level"], r["max_tokens"], reason)].append(r)
        elif r["count"] is None:
            groups[(r["model"], r["prompt"], r["level"], r["max_tokens"],
                    f"Empty answer, finish_reason “{r.get('finish_reason')}”: the whole cap went to reasoning.")].append(r)
    for (m, p, lv, cap, reason), rs in groups.items():
        body.append([NAMES[m], f"<code>{p}</code>", lv, f"{cap:,}", str(len(rs)), e(reason)])
    return table(["Model", "Prompt", "Reasoning", "Output cap", "Calls", "What happened"], body, "data",
                 ["16%", "13%", "10%", "10%", "6%", "45%"])


def section_explanations(rows: list[dict]) -> str:
    g = [r for r in rows if experiment(r) == 4]
    out = []
    by_model = defaultdict(list)
    for r in g:
        by_model[r["model"]].append(r)
    order = sorted(by_model, key=lambda m: list(NAMES).index(m))
    for model in order:
        out.append(f'<h3 class="appx">{NAMES[model]} <span class=id>{model}</span></h3>')
        for r in sorted(by_model[model], key=lambda r: (r["file"], r["repeat"])):
            if r["ok"]:
                head = (f"Count <b>{num(r['count'])}</b>{' (within ± 1)' if hit(r['count']) else ''} · "
                        f"{num(r.get('reasoning_tokens'))} reasoning tokens · {num(r['latency_s'])} s · "
                        f"cap {r['max_tokens']:,} · repeat {r['repeat']}")
                body = e(r["answer"]) if r["answer"] else "<span class=muted>(empty)</span>"
            else:
                head = f"No answer · cap {r['max_tokens']:,} · repeat {r['repeat']}"
                body = e(r.get("error") or "provider error mid-generation")
            out.append(f'<div class="answer"><div class="ahead">{head}</div><pre>{body}</pre></div>')
    return "\n".join(out)


def section_files() -> str:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    body = [[f"<code>{Path(f['file']).name}</code>", e(f["experiment"]), str(f["calls"]), str(f["errors"]),
             usd(f["cost_usd"]), f"<span class=hash>{f['sha256']}</span>"] for f in manifest["files"]]
    img = manifest["image"]
    return (table(["File (data/raw/)", "Contents", "Calls", "Errors", "Cost", "SHA-256"], body, "data small",
                  ["19%", "31%", "6%", "6%", "7%", "31%"])
            + f'<p class="small">Image <code>{e(img["file"])}</code>, {img["size_px"][0]} × {img["size_px"][1]} px, '
              f'SHA-256 <span class=hash>{img["sha256"]}</span>.</p>')


# ---------------------------------------------------------------- document

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
  @bottom-center { content: counter(page) " / " counter(pages); font: 8.5pt 'Segoe UI', sans-serif; color: #777; }
  @top-right { content: "Counting ladder rungs in one photo"; font: 8.5pt 'Segoe UI', sans-serif; color: #999; } }
@page :first { @top-right { content: none; } }
html { font: 10.5pt/1.45 Georgia, 'Times New Roman', serif; color: #1a1a1a; background: #fff; }
body { margin: 0; }
h1 { font: 600 22pt/1.2 'Segoe UI', sans-serif; margin: 0 0 4pt; }
h2 { font: 600 14pt/1.3 'Segoe UI', sans-serif; margin: 22pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt;
     break-after: avoid; }
h3 { font: 600 11.5pt/1.3 'Segoe UI', sans-serif; margin: 14pt 0 4pt; break-after: avoid; }
h4 { font: 600 10pt/1.3 'Segoe UI', sans-serif; margin: 12pt 0 4pt; break-after: avoid; color: #333; }
p, li { margin: 0 0 6pt; } ul, ol { padding-left: 16pt; margin: 0 0 8pt; }
.subtitle { font: 12pt/1.4 'Segoe UI', sans-serif; color: #444; margin-bottom: 10pt; }
.meta { font: 9pt/1.5 'Segoe UI', sans-serif; color: #555; margin-bottom: 14pt; }
.abstract { background: #f5f3ee; border-left: 3px solid #8a7f6a; padding: 8pt 11pt; margin: 10pt 0 6pt; }
.abstract p:last-child { margin-bottom: 0; }
code, pre, .id, .hash { font-family: Consolas, 'Courier New', monospace; }
code { font-size: 9pt; background: #f2f2f2; padding: 0 2pt; border-radius: 2pt; }
.id { font-size: 7.5pt; color: #777; } .hash { font-size: 6.8pt; word-break: break-all; color: #444; }
.muted { color: #888; } .small, .small td, .small th { font-size: 8pt; }
.small code, p.small code { font-size: 7.3pt; background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; table-layout: fixed;
        font: 8.6pt/1.35 'Segoe UI', sans-serif; }
th { text-align: left; font-weight: 600; border-bottom: 1.2px solid #444; padding: 3pt 4pt; vertical-align: bottom; }
td { border-bottom: 0.5px solid #ddd; padding: 3pt 4pt; vertical-align: top; }
tr { break-inside: avoid; } thead { display: table-header-group; }
.params td:first-child { width: 28%; font-weight: 600; }
.prompt { margin: 6pt 0 10pt; break-inside: avoid; }
.plabel { font: 9pt 'Segoe UI', sans-serif; color: #444; margin-bottom: 2pt; }
pre { white-space: pre-wrap; word-wrap: break-word; font-size: 8.6pt; line-height: 1.4; margin: 0;
      background: #fafafa; border: 0.5px solid #ddd; padding: 5pt 7pt; }
.answer { margin: 0 0 7pt; break-inside: avoid; }
.ahead { font: 8.3pt 'Segoe UI', sans-serif; color: #444; margin-bottom: 1.5pt; }
.answer pre { font-size: 7.9pt; }
h3.appx { border-top: 1px solid #ccc; padding-top: 6pt; }
.pagebreak { break-before: page; }
.note { font: 8.8pt/1.45 'Segoe UI', sans-serif; color: #444; }
"""


def build() -> str:
    rows = load()
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    all_s = stats(rows)
    pair = {m: stats([r for r in rows if r["model"] == m and experiment(r) in (1, 2) and r["level"] != "none"])
            for m in PAIR}
    astra, opus = pair[PAIR[0]], pair[PAIR[1]]
    astra_elevens = sum(n == 11 for r in rows if r["model"] == PAIR[0] for n in [r["count"]])
    exp4_html, exp4 = section_exp4(rows)
    models = sorted({r["model"] for r in rows})
    opus_by = {p: stats([r for r in rows if r["model"] == PAIR[1] and r["prompt"] == p and experiment(r) in (1, 2)])
               for p in PROMPT_ORDER}

    params = table(["Setting", "Value"], [
        ["Endpoint", "OpenRouter chat completions, <code>POST https://openrouter.ai/api/v1/chat/completions</code>"],
        ["Message", "One user message: the prompt text first, then the image as a base64 JPEG data URL. "
                    "No system prompt, no conversation history."],
        ["Reasoning", "<code>reasoning: {\"effort\": level}</code> for "
                      + ", ".join(f"<code>{lv}</code>" for lv in ("low", "medium", "high", "xhigh"))
                      + f"; for Claude models in experiment 4, <code>reasoning: {json.dumps(REASONING['max'])}"
                        "</code> (a 24,000-token thinking budget)."],
        ["Output cap (<code>max_tokens</code>)", ", ".join(f"{lv} {MAX_TOKENS[lv]:,}" for lv in
                                                         ("low", "medium", "high", "xhigh"))
         + "; experiment 3 overrides it with 2,000 / 1,000 / 100; the Gemini retry in experiment 4 used 64,000."],
        ["Image detail", "<code>detail: \"high\"</code> for the cleaned and guided prompts; the field omitted "
                         "(provider default) for the original prompt and in experiment 3, as in phase 1."],
        ["Sampling", "Temperature, top-p and every other sampling setting left at provider defaults."],
        ["Accounting", "<code>usage: {\"include\": true}</code>: the reply carries prompt, completion and "
                       "reasoning tokens and the billed cost, which are stored per call."],
        ["Concurrency", "5 parallel workers by default (fewer in some experiment-4 runs), the first requests started 2 s "
                        "apart, conditions in a seeded random order. The seed is stored on every row."],
        ["Repeats", "5 per condition. Each call is independent; nothing is cached or reused between calls."],
    ], "data params", ["24%", "76%"])

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Counting ladder rungs in one photo</title><style>{CSS}</style></head><body>

<h1>Counting ladder rungs in one photo</h1>
<div class="subtitle">Ten multimodal language models, three reasoning levels, five prompts and one rope ladder</div>
<div class="meta">Ginnung · runs made 2026-09-30 · {all_s['calls']} API calls to {len(models)} models ·
total cost {usd(all_s['cost'])}<br>Data and scripts: <a href="{REPO}">{REPO}</a> (CC0) · Background article:
<a href="{ARTICLE}">ginnung.tech</a></div>

<div class="abstract">
<p><b>Question.</b> Can current general-purpose multimodal language models count the red rope steps on one
rope ladder in a single photo, and does more reasoning, a different prompt, or a precise description help?</p>
<p><b>Design.</b> One 1024 × 768 photo; the author counts 11 steps (12 if the white anchoring bar at the bottom
is included), and any answer from 10 to 12 is scored as a hit. Experiments 1–3 ran GPT-6 Astra and Claude Opus 5.5
over reasoning levels, the number used as a format example in the prompt, the original phase-1 wording and the
output cap. Experiment 4 gave ten models a long description of exactly which ladder to count and what to leave
out, with maximum reasoning, high image detail and a large output budget. Each condition ran 5 times.</p>
<p><b>Results.</b> GPT-6 Astra is consistent but too high: its median answer is {num(astra['median'])} and
{pct(astra['hits'], astra['valid'])} of its answers in experiments 1–2 are within 11 ± 1; it answered 11 in
{astra_elevens} of all its calls. Claude Opus 5.5 answers between {opus['min']} and {opus['max']}, and its answers
move with the example number in the prompt (median {num(opus_by['example_2']['median'])} with <code>2</code>,
{num(opus_by['example_21']['median'])} with <code>21</code>). Raising reasoning effort did not make either model
more accurate. In experiment 4, {pct(exp4['hits'], exp4['valid'])} valid answers across ten models are within
11 ± 1, and only GPT-6 Astra is within tolerance in all five runs (answering 12 each time). The models' written
explanations describe careful counts whether the count is right or far off.</p>
</div>

<h2>1. Background</h2>
<p>A first round of testing (phase 1, 2026-09-22/23), described in the background article, asked GPT-6 Astra the
question in the original, informally written prompt reproduced in section 4. GPT-6 Astra
answered mostly 12. Readers asked for a more rigorous follow-up: more models, several reasoning levels, prompt
variations, repeated runs, and an answer to the objection that the prompt was ambiguous or sloppy. This report
documents that follow-up (phase 2). All of its calls were made on 2026-09-30.</p>
<p>This is a small, informal experiment about one image and one question. It is not a benchmark and does not
support claims about visual ability in general.</p>

<h2>2. Test object and ground truth</h2>
<p>The photo (<code>data/image/</code>, public domain) shows a replica pirate ship moored at a wooden pier, seen
from the pier, with the bow and bowsprit to the left. Several rope ladders (shrouds with ratlines) run from the
deck up to the masts; at least five of them have red horizontal rope steps. The ladder in question is the
front-most one, closest to the camera, running diagonally from the deck at the lower right, just right of the
“PIRAT” name board, up to the left towards the front mast.</p>
<p><b>Ground truth.</b> The author counts <b>11</b> red horizontal steps between a white anchoring bar at the
bottom and the knot where the ladder narrows to a point at the top. Some readers count 12 by including the white
bar. The scoring therefore accepts <b>10–12 (11 ± 1)</b> as a hit, which covers both readings and a one-step
slip. This tolerance was chosen after phase 1, not fixed before any data was seen. The ground truth is a
single human count.</p>

<h2>3. Method</h2>
<h3>3.1 Request settings</h3>
{params}
<h3>3.2 Parsing and scoring</h3>
<p>Every identifiable answer is counted, and any deviation from the requested format is recorded. The rules, in
order (from <code>parse_answer</code> in <code>scripts/run_reasoning_probe.py</code>):</p>
<ol>
<li>If the answer contains a line <code>COUNT: n</code> (the guided prompt asks for one), the last such line is
the count.</li>
<li>A bare integer, optionally in backticks or followed by a full stop, is the count.</li>
<li>Otherwise, if the text contains exactly one distinct number, or one number written as an English word
(zero to thirty), that is the count, noted as a deviation.</li>
<li>An empty answer or an answer with several different numbers and no <code>COUNT</code> line has no count.</li>
</ol>
<p>In practice every non-empty answer in this data was either a bare integer or had a <code>COUNT</code> line;
rules 3 and 4 were never needed except for empty answers. A count is a <b>hit</b> if it lies in 10–12.
Calls that failed at the API or returned no text are reported separately and are not counted as misses.</p>

<h3>3.3 Experiments</h3>
{table(["#", "Question", "Models", "Prompts", "Reasoning", "Detail", "Output cap", "Calls"], [
        ["1", "Does reasoning effort help, and does the example number in the prompt leak into the answer?",
         "GPT-6 Astra, Claude Opus 5.5", "<code>example_2</code>, <code>example_21</code>, <code>example_11</code>",
         "low, medium, high (+ none for Astra, rejected)", "high", "4k / 12k / 24k", "105"],
        ["2", "How do the models answer the original phase-1 prompt, typos included?",
         "GPT-6 Astra, Claude Opus 5.5", "<code>original</code>", "low, medium, high", "default",
         "4k / 12k / 24k", "30"],
        ["3", "Does a tight output budget change the answer?", "GPT-6 Astra, Claude Opus 5.5",
         "<code>example_11</code>", "low", "default", "2,000 / 1,000 / 100", "30"],
        ["4", "With every help we could give, can any model count the ladder?", "10 models",
         "<code>guided</code>", "xhigh effort, or 24k thinking budget for Claude", "high", "32k (Gemini retry 64k)",
         "53"],
    ], "data", ["4%", "27%", "14%", "15%", "14%", "7%", "10%", "6%"])}

<h2>4. Design reasoning</h2>
<p>The design was drafted in response to criticism of phase 1 and reviewed before any call was made. The choices,
and the reason for each:</p>
<ul>
<li><b>A cleaned prompt stem.</b> The original prompt has typos (“horizoltal”, “outlook basket”) and no question
mark, which critics could blame for the answers. The stem used in experiment 1 fixes these, names the ladder as
“front-most”, uses “lookout basket”, and asks a proper question. Everything else is kept short, as in phase 1.</li>
<li><b>The original prompt, unchanged.</b> Kept verbatim, typos included, with the image detail left at the
default, so experiment 2 is directly comparable with phase 1. It is imported from the phase-1 script rather than
copied, so it cannot drift.</li>
<li><b>Three example numbers.</b> The phrase “return only an int like `2`” shows the format, but it also puts a
number next to the question. Three otherwise identical prompts use <code>2</code> (far below the true count),
<code>21</code> (far above) and <code>11</code> (the true count). A model that reads the count off the image should
give the same answers to all three. If answers move with the example, the example is leaking into the count. The
<code>11</code> variant also asks whether handing the model the right number is enough to get it back.</li>
<li><b>Reasoning levels.</b> Low, medium and high effort ask whether more thinking improves the count. “None” was
planned too, but GPT-6 Astra rejects it (“Reasoning is mandatory for this endpoint”), so it was dropped after the
first file; the 15 rejected calls are kept in the data.</li>
<li><b>Highest image detail.</b> For the cleaned and guided prompts the image is sent with
<code>detail: "high"</code>, the highest resolution setting OpenRouter exposes, so that low resolution cannot
explain a wrong count. The image itself is never modified; an upscaled copy was considered and rejected.</li>
<li><b>Output cap.</b> Experiment 3 asks whether a tight token budget changes the answer, and at what point the
answer disappears entirely.</li>
<li><b>Five repeats.</b> Five runs per condition show the spread of answers. They are not enough to estimate
rates precisely, and the report does not attempt significance tests.</li>
<li><b>The guided prompt.</b> To answer the objection that the task was ambiguous, experiment 4 describes the
ladder by several independent cues (bow direction, the name board, the rear deckhouse, the front mast, “closest to
the camera”), says exactly what not to count (the white bar, the top knot), invites the model to explain how it
counted, and asks for a machine-readable last line. The draft was corrected three times during review: an early
claim that it was “the only red rope ladder” was removed, because at least five ladders have red steps; the top
was described as a knot where the ladder comes to a point; and a description of the ladder as leading to the
basket was reworded to position only, since these ladders are not access ladders. Every model got its maximum
reasoning setting, high image detail and a 32,000-token output cap.</li>
<li><b>Model choice.</b> Experiment 4 covers current frontier models from OpenAI, Anthropic, Google, xAI and
Meta, and three open-weight models (MiniMax-M3, MiMo-V2.6-Pro, DeepSeek V4.1 Flash). A GLM model was considered,
but the top GLM-5.3 model takes no image input and was left out.</li>
</ul>
<p><b>What the design does not control:</b> sampling temperature (provider defaults), the provider's internal
image preprocessing, the meaning of “low” / “high” across vendors, and model version (OpenRouter does not expose
one). There is one image and one human counter, and no blinding.</p>

<h2>5. Prompts</h2>
<p>Exact texts, as sent. The image follows the text in the same message.</p>
{section_prompts()}

<h2 class="pagebreak">6. Results</h2>
<h3>6.1 Overview</h3>
{section_totals(rows)}
<p class="note">“Valid counts” excludes API errors and empty answers. “Within ± 1” is out of the valid counts.</p>

<h3>6.2 Experiments 1 and 2: reasoning level × prompt</h3>
<p>Every condition, 5 calls each. Counts are sorted; bold marks a hit (10–12).</p>
{section_exp12(rows)}
{section_pooled(rows)}

<h3>6.3 Experiment 3: output cap</h3>
<p>Prompt <code>example_11</code>, reasoning low, image detail default, 5 calls per cap.</p>
{section_exp3(rows)}

<h3>6.4 Experiment 4: guided prompt, maximum reasoning</h3>
<p>Sorted by share of hits, then by median. Gemini 3.8 Flash failed mid-generation in 4 of 8 attempts (5 at a
32k cap, 3 retried at 64k); only its valid answers are shown. Full answer texts are in Appendix A.</p>
{exp4_html}

<h3>6.5 Failed and empty calls</h3>
{section_errors(rows)}

<h2>7. Observations</h2>
<ul>
<li><b>Consistent is not correct.</b> GPT-6 Astra's answers sit in a narrow band (12–14, one outlier of 8) under
every prompt and reasoning level, with a median of {num(astra['median'])}. In phase 1 the same prompt gave mostly
12; here, a week later, mostly 13. Its only fully correct condition is the guided prompt, where it answered 12 five
times, which is within tolerance but still one more than the author's count.</li>
<li><b>The example number leaks into Claude Opus 5.5's count.</b> Its median is
{num(opus_by['example_2']['median'])} with <code>2</code>, {num(opus_by['example_21']['median'])} with
<code>21</code> and {num(opus_by['example_11']['median'])} with <code>11</code>. The <code>2</code> and
<code>21</code> examples pull the answers towards themselves. Giving it the true count as the example did not
bring it back: with <code>11</code> its median is {num(opus_by['example_11']['median'])}. GPT-6 Astra's answers
barely move with the example.</li>
<li><b>More reasoning did not help.</b> For both models, going from low to high effort roughly doubles or
quadruples the reasoning tokens (see the pooled table in 6.2) without raising the share of hits or narrowing the
spread. In experiment 4 the models that reasoned longest (Gemini 3.8 Flash, Muse Spark 1.3, Grok 4.7, DeepSeek
V4.1 Flash) are not the most accurate.</li>
<li><b>The guided prompt did not settle it.</b> With a precise description and the maximum settings,
{pct(exp4['hits'], exp4['valid'])} valid answers are within 11 ± 1. Several models still answer in the high teens
or twenties.</li>
<li><b>Explanations are not evidence.</b> In Appendix A, answers of 20 and more describe the same kind of
careful, step-by-step count as answers of 11 or 12. The explanation text does not reveal which counts are
wrong.</li>
<li><b>A 100-token cap gives no answer at all.</b> Both models spend the whole budget on reasoning and return an
empty answer with <code>finish_reason: "length"</code>, while still being billed for the tokens.</li>
</ul>

<h2>8. Limitations</h2>
<ul>
<li>One image and one question. Nothing here generalises beyond counting thin, repeated structures in a single
1024 × 768 photo.</li>
<li>Five runs per condition show spread, not precise rates. No significance tests were run.</li>
<li>The ground truth is one person's count, and the ± 1 tolerance was set after phase 1.</li>
<li>Reasoning levels are mapped by OpenRouter to each vendor's own settings, so the same label is not the same
effort across models. Compare the logged reasoning tokens instead.</li>
<li><code>detail: "high"</code> is an OpenAI-style field; whether other providers honour it is not documented.</li>
<li>Models change behind a fixed name. OpenRouter does not expose a model version, so a re-run may differ, as
GPT-6 Astra's shift from 12 to 13 between phases shows.</li>
<li>Only the final answers were stored, not the models' hidden reasoning. Appendix A shows the explanation each
model chose to write, which is not a trace of how the number was produced.</li>
</ul>

<h2>9. Replication</h2>
<p>The repository holds the scripts, the photo and every raw response. With Python 3.10+ and an OpenRouter key in
<code>OPENROUTER_API_KEY</code>, each experiment is one command (see the README); every command is a dry run until
<code>--run</code> is added. <code>scripts/build_metadata.py</code> rebuilds <code>data/manifest.json</code> and
<code>data/calls.csv</code>, and <code>scripts/build_report.py</code> rebuilds this report from the same data.</p>

<h2 class="pagebreak">Appendix A. Answers to the guided prompt</h2>
<p>Every experiment-4 call, as returned by the model: the count, the explanation it gave, and the settings. These
are the visible answers, not the hidden reasoning, which the API did not return. Grouped by model; within a model,
in the order the calls were made.</p>
{section_explanations(rows)}

<h2 class="pagebreak">Appendix B. Raw files</h2>
<p>SHA-256 checksums of the raw files in <code>data/raw/</code>, as listed in <code>data/manifest.json</code>
(total {manifest['total_calls']} calls, {manifest['total_errors']} API errors, {usd(manifest['total_cost_usd'])}).</p>
{section_files()}
</body></html>
"""


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build(), encoding="utf-8", newline="\n")
    print(f"-> {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
