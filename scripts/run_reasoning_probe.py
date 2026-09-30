"""Reasoning-level x prompt probe on one model (phase 2 of the ladder test).

Every combination of PROMPTS x LEVELS is queried REPEATS times, in a random
order, by a small pool of parallel workers whose first requests are staggered.

Default is a DRY RUN: checks the model against the live OpenRouter catalogue
and prints the plan and a rough cost estimate. Nothing is sent.

    python run_reasoning_probe.py                      # plan + estimate only
    python run_reasoning_probe.py --run                # 3 prompts x 4 levels x 5 = 60 calls
    python run_reasoning_probe.py --run --levels low,high --repeats 10

Settings: image sent with detail "high"; reasoning effort per LEVELS;
temperature and everything else left at provider defaults (same as phase 1).
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from run_image_probe import API, IMAGE, PROMPTS as PHASE1_PROMPTS, RESULTS_DIR, fetch_catalogue, per_mtok

MODEL = "openai/gpt-6-astra"

_STEM = ("I can see several rope ladders leading to the outlooks on the masts. On the front-most "
         "rope ladder how many horizontal red steps are there to the lookout basket? "
         "return only an int like ")
PROMPTS = {
    # Phase 1's ladder prompt, verbatim (typos included), imported so it cannot drift.
    "original": PHASE1_PROMPTS["ladder_steps"],
    "example_2": _STEM + "`2`",
    "example_21": _STEM + "`21`",
    "example_11": _STEM + "`11`",
    # Phase 3 (arbirk 2026-09-30): give the model every help — unambiguous
    # location, what not to count, room to explain. Parsed from the COUNT line.
    "guided": (
        "Here is a photo of a pirate ship moored at a wooden pier, seen from the pier. The bow, with the "
        "carved figurehead and the long bowsprit, points to the left side of the photo. I want you to count "
        "the steps (rungs) of one rope ladder: the front-most rope ladder, the one closest to the camera. It "
        "runs diagonally from the lower right, where it is fastened to the deck in front of the raised rear "
        "deckhouse, just to the right of the \"PIRAT\" name board on the hull, up to the left towards the "
        "front mast, the mast nearest the bow. Its upper end is high on that mast, below the lookout basket. "
        "Its steps are red horizontal ropes. At the bottom of this ladder there is a white horizontal "
        "anchoring bar. Do not count that one. At the top, the ladder narrows and comes to a point in a knot. "
        "Do not count that knot either. Only count the red horizontal rope steps in between. I am bad at "
        "counting. Give me the step count and a short description of how you counted and reasoned. End your "
        "answer with a final line in exactly this form: COUNT: <number>"
    ),
}

# "none" is not offered: Astra rejects it ("Reasoning is mandatory for this endpoint").
LEVELS = ["low", "medium", "high", "xhigh", "max"]

# How each level is sent. "xhigh" is OpenRouter's top effort (OpenAI models);
# "max" is an explicit thinking budget (Anthropic takes a token budget, and it
# must stay below max_tokens).
REASONING = {lv: {"effort": lv} for lv in ("low", "medium", "high", "xhigh")}
REASONING["max"] = {"max_tokens": 24000}

# Output-token cap per level. OpenRouter reserves credit for max_tokens up
# front (phase 1 hit HTTP 402 on that reserve), so caps stay as low as each
# level plausibly needs. A capped call shows finish_reason "length".
MAX_TOKENS = {"low": 4000, "medium": 12000, "high": 24000, "xhigh": 32000, "max": 32000}

# Rough output tokens per call, for the dry-run estimate only. Phase 1 measured
# ~180 on Astra at low; the rest are guesses and the run will replace them.
# Measured on Astra 2026-09-30: mean reasoning tokens ~250 / ~550 / ~1130.
EST_OUT_TOKENS = {"low": 260, "medium": 560, "high": 1150, "xhigh": 6000, "max": 6000}
EST_IN_TOKENS = 1000

WORKERS = 5
STAGGER_S = 2.0


def ask(key: str, model_id: str, prompt: str, level: str, image_b64: str, detail: str) -> dict:
    image_url = {"url": f"data:image/jpeg;base64,{image_b64}"}
    if detail != "default":  # "default" omits the field, exactly like phase 1
        image_url["detail"] = detail
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_url},
        ]}],
        "reasoning": REASONING[level],
        "max_tokens": MAX_TOKENS[level],
        "usage": {"include": True},
    }
    req = urllib.request.Request(
        f"{API}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}",
                "latency_s": round(time.monotonic() - t0, 2)}
    except Exception as e:  # network/timeout: record it, the run carries on
        return {"error": repr(e), "latency_s": round(time.monotonic() - t0, 2)}
    if "error" in resp:
        return {"error": json.dumps(resp["error"])[:500], "latency_s": round(time.monotonic() - t0, 2)}
    choice = resp["choices"][0]
    usage = resp.get("usage") or {}
    if choice.get("finish_reason") == "error":  # provider failed mid-generation (seen on Gemini); not billed
        return {"error": f"provider error mid-generation ({resp.get('provider')}); "
                         f"completion_tokens={usage.get('completion_tokens')}",
                "provider": resp.get("provider"), "latency_s": round(time.monotonic() - t0, 2)}
    return {
        "answer": (choice["message"].get("content") or "").strip(),
        "finish_reason": choice.get("finish_reason"),
        "provider": resp.get("provider"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost": usage.get("cost"),
        "latency_s": round(time.monotonic() - t0, 2),
    }


NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({f"twenty{w}": 20 + i for w, i in list(NUMBER_WORDS.items())[1:10]})
NUMBER_WORDS["thirty"] = 30


def parse_answer(answer: str) -> tuple[int | None, str | None]:
    """Count every identifiable answer; note how it deviates from the asked-for bare int.

    Returns (value, deviation). deviation is None for a bare int (optionally in
    backticks / with a trailing full stop), otherwise a short note. value is None
    only when no single number can be identified.
    """
    text = answer.strip()
    if not text:
        return None, "empty answer"
    count_lines = re.findall(r"COUNT:\s*\**\s*(\d+)", text)
    if count_lines:  # the guided prompt asks for a final "COUNT: <n>" line; the last one wins
        return int(count_lines[-1]), None
    m = re.fullmatch(r"`?\s*(\d+)\s*`?\.?", text)
    if m:
        return int(m.group(1)), None
    nums = {int(n) for n in re.findall(r"\d+", text)}
    if not nums:
        joined = re.sub(r"twenty[\s-]+(?=[a-z])", "twenty", text.lower())  # "twenty-one" -> "twentyone"
        words = {w: NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", joined) if w in NUMBER_WORDS}
        if len(set(words.values())) == 1:
            return next(iter(words.values())), "number written as a word"
        nums = set(words.values())
    if len(nums) == 1:
        return nums.pop(), "number inside extra text"
    if not nums:
        return None, "no number"
    return None, f"several numbers ({', '.join(map(str, sorted(nums)))})"


def parse_int(answer: str) -> int | None:
    return parse_answer(answer)[0]


def estimate(model: dict, levels: list[str], prompts: list[str], repeats: int) -> float:
    pin, pout = per_mtok(model, "prompt") / 1e6, per_mtok(model, "completion") / 1e6
    n = len(prompts) * repeats
    return sum(n * (EST_IN_TOKENS * pin + EST_OUT_TOKENS[lv] * pout) for lv in levels)


def summarise(rows: list[dict]) -> None:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["prompt"], r["level"])].append(r)
    print(f"\n{'prompt':<12}{'level':<8}{'answers':<34}{'modal share':>12}{'deviations':>11}"
          f"{'reason tok':>12}{'$':>9}")
    for (p, lv), rs in sorted(by_cell.items(), key=lambda kv: (kv[0][0], LEVELS.index(kv[0][1]))):
        answers = Counter(str(r["value"]) if "answer" in r else "ERR" for r in rs)
        modal = answers.most_common(1)[0][1] / len(rs)
        devs = sum(1 for r in rs if r.get("deviation"))
        rt = [r["reasoning_tokens"] for r in rs if r.get("reasoning_tokens") is not None]
        cost = sum(float(r.get("cost") or 0) for r in rs)
        dist = ", ".join(f"{a}x{c}" for a, c in answers.most_common())
        print(f"{p:<12}{lv:<8}{dist:<34}{modal:>11.0%}{devs:>11}{(sum(rt) / len(rt) if rt else 0):>12.0f}{cost:>9.4f}")
    notes = Counter(r["deviation"] for r in rows if r.get("deviation"))
    if notes:
        print("\nFormat deviations (counted when a single number was identifiable):")
        for note, n in notes.most_common():
            print(f"  {n:>3} x {note}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually call the model (default: dry run)")
    ap.add_argument("--model", default=MODEL, help="OpenRouter model id")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--levels", default="low,medium,high")
    ap.add_argument("--prompts", default="example_2,example_21,example_11")
    ap.add_argument("--detail", default="high", choices=["default", "low", "high", "auto"],
                    help='"default" sends no detail field (phase 1 setting)')
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="output cap for every level (overrides MAX_TOKENS); reasoning counts against it")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--image", default=str(IMAGE), help="image file to send (default: the phase-1 photo)")
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed (default: random, recorded)")
    args = ap.parse_args()

    levels = [x.strip() for x in args.levels.split(",")]
    if args.max_tokens:
        for lv in MAX_TOKENS:
            MAX_TOKENS[lv] = args.max_tokens
    prompts = [x.strip() for x in args.prompts.split(",")]
    for lv in levels:
        if lv not in MAX_TOKENS:
            sys.exit(f"unknown level {lv!r}; choose from {LEVELS}")
    for p in prompts:
        if p not in PROMPTS:
            sys.exit(f"unknown prompt {p!r}; choose from {list(PROMPTS)}")

    model_id = args.model
    model = fetch_catalogue().get(model_id)
    if model is None:
        sys.exit(f"{model_id} is not in the OpenRouter catalogue")
    if "reasoning" not in (model.get("supported_parameters") or []):
        sys.exit(f"{model_id} does not list the reasoning parameter")

    jobs = [(p, lv, rep) for p in prompts for lv in levels for rep in range(args.repeats)]
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    random.Random(seed).shuffle(jobs)
    # Highest reasoning first (arbirk): it is the most expensive and the least
    # known, so its real token use shows before the cheaper levels spend budget.
    # Order within a level stays shuffled.
    jobs.sort(key=lambda j: -LEVELS.index(j[1]))

    print(f"model      {model_id}  (${per_mtok(model, 'prompt'):.2f} in / ${per_mtok(model, 'completion'):.2f} out per M)")
    print(f"prompts    {', '.join(prompts)}")
    print(f"levels     {', '.join(levels)}   max_tokens {', '.join(f'{lv}={MAX_TOKENS[lv]}' for lv in levels)}")
    print(f"calls      {len(jobs)}  ({len(prompts)} prompts x {len(levels)} levels x {args.repeats} repeats)")
    print(f"workers    {args.workers}, first requests staggered {STAGGER_S}s apart; shuffle seed {seed}")
    print(f"image      {Path(args.image).name}, detail={args.detail}")
    print(f"estimate   ~${estimate(model, levels, prompts, args.repeats):.2f} (rough; high-reasoning output is a guess)")
    for p in prompts:
        print(f"\n[{p}] {PROMPTS[p]}")
    if not args.run:
        print("\nDry run. Re-run with --run to query.")
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")

    image_b64 = base64.b64encode(Path(args.image).read_bytes()).decode()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"reasoning-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    print(f"\nWriting {out}\n")

    lock = threading.Lock()
    stop = threading.Event()  # set on 401 (bad key) or 402 (out of credits): no new calls start
    rows: list[dict] = []
    started = [0]

    def work(job: tuple[str, str, int]) -> None:
        p, lv, rep = job
        with lock:
            slot = started[0]
            started[0] += 1
        if slot < args.workers:
            time.sleep(slot * STAGGER_S)  # stagger only the first wave
        if stop.is_set():
            return
        for attempt in range(3):
            r = ask(key, model_id, PROMPTS[p], lv, image_b64, args.detail)
            if not r.get("error", "").startswith("HTTP 429"):
                break
            time.sleep(10 * (attempt + 1))
        err = r.get("error", "")
        if err.startswith(("HTTP 401", "HTTP 402")):
            stop.set()
        row = {"model": model_id, "prompt": p, "level": lv, "repeat": rep, "detail": args.detail,
               "image": Path(args.image).name,
               "max_tokens": MAX_TOKENS[lv], "seed": seed, **r}
        if "answer" in r:
            row["value"], row["deviation"] = parse_answer(r["answer"])
            if r.get("finish_reason") == "length":
                row["deviation"] = "hit max_tokens" + (f"; {row['deviation']}" if row["deviation"] else "")
        with lock:
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            shown = r.get("answer", err).replace("\n", " ")
            note = f"  [{row['deviation']}]" if row.get("deviation") else ""
            print(f"[{len(rows):>3}/{len(jobs)}] {p:<11} {lv:<7} {r['latency_s']:>6}s "
                  f"rt={r.get('reasoning_tokens')!s:<6} ${float(r.get('cost') or 0):.4f}  "
                  f"{row.get('value')!s:<5} {shown[:80]}{note}")

    with out.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, jobs))

    if stop.is_set():
        print("\nStopped early: API key rejected or credits exhausted (see the last error above).")
    summarise(rows)
    print(f"\nTotal cost: ${sum(float(r.get('cost') or 0) for r in rows):.4f}   {len(rows)}/{len(jobs)} calls recorded")


if __name__ == "__main__":
    main()
