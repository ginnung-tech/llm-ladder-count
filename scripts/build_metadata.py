"""Rebuild data/manifest.json and data/calls.csv from the raw JSONL results.

    python scripts/build_metadata.py

The raw files are the source of truth; these two are derived views:
  - manifest.json: one entry per raw file (what it contains, call and error
    counts, cost, SHA-256), plus the prompts and the image checksum.
  - calls.csv: one row per call, with the parsed count and whether it falls
    within 11 +/- 1.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_reasoning_probe import PROMPTS, parse_answer  # noqa: E402

RAW = ROOT / "data" / "raw"
IMAGE = ROOT / "data" / "image" / "ship-the-ship-haven-travel-vacation-62f2b5-1024.jpg"
AUTHOR_COUNT = 11

# Which experiment each raw file belongs to, keyed on its timestamp prefix.
EXPERIMENTS = [
    ("042404", "1: GPT-6 Astra, prompts example_2/21/11 x reasoning none/low/medium/high (none is rejected by the API)"),
    ("043124", "2: GPT-6 Astra, original phase-1 prompt, default image detail, low/medium/high"),
    ("043624", "1: Claude Opus 5.5, prompts example_2/21/11 x reasoning low/medium/high"),
    ("043804", "2: Claude Opus 5.5, original phase-1 prompt, default image detail, low/medium/high"),
    ("0442", "3: output-cap test, example_11, low reasoning, default detail"),
    ("0443", "3: output-cap test, example_11, low reasoning, default detail"),
    ("0444", "3: output-cap test, example_11, low reasoning, default detail"),
    ("05", "4: guided prompt, maximum reasoning, image detail high"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files, calls = [], []
    for path in sorted(RAW.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        stamp = path.stem.split("-")[-1]
        experiment = next(desc for prefix, desc in EXPERIMENTS if stamp.startswith(prefix))
        errors = 0
        for r in rows:
            ok = "answer" in r and r.get("finish_reason") != "error"
            errors += not ok
            count, deviation = parse_answer(r["answer"]) if ok else (None, None)
            calls.append({
                "file": path.name, "model": r["model"], "prompt": r["prompt"], "level": r["level"],
                "detail": r.get("detail"), "max_tokens": r.get("max_tokens"), "repeat": r.get("repeat"),
                "status": "ok" if ok else "error", "count": count,
                "within_11_pm_1": (AUTHOR_COUNT - 1 <= count <= AUTHOR_COUNT + 1) if count is not None else None,
                "format_deviation": deviation, "finish_reason": r.get("finish_reason"),
                "provider": r.get("provider"), "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"), "reasoning_tokens": r.get("reasoning_tokens"),
                "cost_usd": r.get("cost"), "latency_s": r.get("latency_s"), "seed": r.get("seed"),
            })
        files.append({
            "file": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "experiment": experiment,
            "calls": len(rows), "errors": errors,
            "models": sorted({r["model"] for r in rows}), "prompts": sorted({r["prompt"] for r in rows}),
            "levels": sorted({r["level"] for r in rows}), "detail": sorted({str(r.get("detail")) for r in rows}),
            "max_tokens": sorted({r.get("max_tokens") for r in rows}),
            "cost_usd": round(sum(float(r.get("cost") or 0) for r in rows), 4),
        })

    with (ROOT / "data" / "calls.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(calls[0]))
        writer.writeheader()
        writer.writerows(calls)

    manifest = {
        "title": "Counting ladder rungs in one photo: 10 multimodal LLMs, reasoning levels and prompts",
        "date": "2026-09-30",
        "api": "OpenRouter chat completions, https://openrouter.ai/api/v1",
        "image": {"file": IMAGE.relative_to(ROOT).as_posix(), "sha256": sha256(IMAGE),
                  "size_px": [1024, 768], "license": "CC0 (public domain)"},
        "prompts": PROMPTS,
        "author_count": AUTHOR_COUNT,
        "scoring": "a parsed count within 10-12 (11 +/- 1) is a hit",
        "total_calls": len(calls),
        "total_errors": sum(f["errors"] for f in files),
        "total_cost_usd": round(sum(float(c["cost_usd"] or 0) for c in calls), 4),
        "files": files,
    }
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(f"{len(calls)} calls, {manifest['total_errors']} errors, ${manifest['total_cost_usd']} -> "
          f"data/manifest.json, data/calls.csv")


if __name__ == "__main__":
    main()
