"""Image-reasoning probe across OpenRouter multimodal models.

Default is a DRY RUN: verifies every model ID against the live OpenRouter
catalogue and prints price + reasoning support, sends nothing.

    python run_image_probe.py                    # verify list, no calls
    python run_image_probe.py --run --max-tier 3 # run tiers 1..3 (cheapest first)
    python run_image_probe.py --run --models google/gemma-3-4b-it,openai/gpt-5-mini

Settings: reasoning effort "low" where the model supports the reasoning
parameter; everything else (temperature, max_tokens, image detail/media
resolution) left at provider defaults.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
# Repository layout: scripts/ next to data/image/ and data/raw/.
IMAGE = HERE.parent / "data" / "image" / "ship-the-ship-haven-travel-vacation-62f2b5-1024.jpg"
RESULTS_DIR = HERE.parent / "data" / "raw"
API = "https://openrouter.ai/api/v1"
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"

# (tier, model_id) — weak -> strong, roughly cheapest -> most expensive.
MODELS = [
    (1, "google/gemma-3-4b-it"),
    (1, "qwen/qwen3-vl-8b-instruct"),
    (1, "google/gemini-2.5-flash-lite"),
    (1, "meta/muse-spark-1.3-contributor"),   # cheap tier; prompts may be used for training
    (1, "deepseek/deepseek-v4.1-flash"),
    (1, "z-ai/glm-5.3-flash"),
    (1, "qwen/qwen3.8-flash"),
    (2, "meta-llama/llama-4-maverick"),
    (2, "perceptron/perceptron-mk1"),
    (2, "qwen/qwen3-vl-235b-a22b-instruct"),
    (2, "stepfun/step-3.7-flash"),
    (2, "openai/gpt-5-mini"),
    (2, "minimax/minimax-m3"),
    (3, "xiaomi/mimo-v2.6-pro"),
    (3, "bytedance-seed/seed-2-1-turbo"),
    (3, "google/gemini-3.8-flash"),
    (3, "anthropic/claude-haiku-4.5"),
    (3, "x-ai/grok-4.7"),
    (4, "qwen/qwen3.8-max-0902"),
    (4, "moonshotai/kimi-k3"),
    (4, "anthropic/claude-sonnet-5"),
    (4, "google/gemini-3.1-pro-preview"),
    (4, "openai/gpt-5.6-terra"),
    (5, "anthropic/claude-opus-5.5"),
    (5, "openai/gpt-6-astra"),
    (5, "anthropic/claude-fable-5.1"),
    (5, "gemini-api/gemini-robotics-er-2-preview"),  # Gemini API only; needs GEMINI_API_KEY
]

PROMPTS = {
    "rescue_devices": (
        "is there sea rescue devices in the image. if yes give me the center pixel "
        "localtion of each device. coordinate system starts at (0,0) upper left and "
        "end (1024,768) lower left. Return and array like this: `[{10,20}]` if there "
        "are those devices"
    ),
    "ladder_steps": (
        "I can see several rope ladders leading to the outlooks on the masts. on the "
        "front most rope ladder how many horizoltal red steps are there to the outlook "
        "basket. return only an int like `2`"
    ),
}


def fetch_catalogue() -> dict:
    with urllib.request.urlopen(f"{API}/models", timeout=30) as r:
        cat = {m["id"]: m for m in json.load(r)["data"]}
    # Models only served by Google's Gemini API (e.g. Robotics-ER) get a "gemini-api/" prefix.
    if gkey := os.environ.get("GEMINI_API_KEY"):
        req = urllib.request.Request(f"{GEMINI_API}/models?pageSize=1000", headers={"x-goog-api-key": gkey})
        with urllib.request.urlopen(req, timeout=30) as r:
            for g in json.load(r)["models"]:
                mid = "gemini-api/" + g["name"].removeprefix("models/")
                cat[mid] = {"id": mid, "pricing": {}, "architecture": {"input_modalities": ["image"]},
                            "supported_parameters": ["reasoning"] if g.get("thinking") else []}
    return cat


def per_mtok(m: dict, key: str) -> float:
    return float(m["pricing"].get(key) or 0) * 1e6


def verify(selected: list[tuple[int, str]], catalogue: dict) -> list[tuple[int, str, dict]]:
    ok, missing = [], []
    print(f"{'tier':<5}{'model':<40}{'$in/M':>8}{'$out/M':>8}  reasoning  image")
    for tier, mid in selected:
        m = catalogue.get(mid)
        if m is None:
            missing.append(mid)
            print(f"{tier:<5}{mid:<40}  MISSING FROM CATALOGUE")
            continue
        has_img = "image" in m["architecture"].get("input_modalities", [])
        has_rsn = "reasoning" in (m.get("supported_parameters") or [])
        print(f"{tier:<5}{mid:<40}{per_mtok(m, 'prompt'):>8.3f}{per_mtok(m, 'completion'):>8.3f}"
              f"  {'low' if has_rsn else '-':<9}  {'yes' if has_img else 'NO'}")
        if has_img:
            ok.append((tier, mid, m))
        else:
            missing.append(mid)
    if missing:
        print(f"\nWARNING: skipping {len(missing)} model(s): {', '.join(missing)}", file=sys.stderr)
    return ok


def query_gemini(model: dict, prompt: str, image_b64: str, max_tokens: int | None) -> dict:
    """Same request shape as the OpenRouter path: prompt text then image, one user turn, low thinking."""
    gen: dict = {}
    if "reasoning" in model["supported_parameters"]:
        gen["thinkingConfig"] = {"thinkingLevel": "low"}
    if max_tokens:
        gen["maxOutputTokens"] = max_tokens
    body = {"contents": [{"role": "user", "parts": [
        {"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]}],
        "generationConfig": gen}
    req = urllib.request.Request(
        f"{GEMINI_API}/models/{model['id'].removeprefix('gemini-api/')}:generateContent",
        data=json.dumps(body).encode(),
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}",
                "latency_s": round(time.monotonic() - t0, 2)}
    except Exception as e:  # network/timeout: record and keep going with the next model
        return {"error": repr(e), "latency_s": round(time.monotonic() - t0, 2)}
    cand = (resp.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
    u = resp.get("usageMetadata") or {}
    return {
        "answer": text.strip(),
        "finish_reason": cand.get("finishReason"),
        "provider": "google-gemini-api",
        "usage": {"prompt_tokens": u.get("promptTokenCount"),
                  "completion_tokens": (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0),
                  "reasoning_tokens": u.get("thoughtsTokenCount"), "cost": None},
        "latency_s": round(time.monotonic() - t0, 2),
    }


def query(key: str, model: dict, prompt: str, image_b64: str, max_tokens: int | None) -> dict:
    if model["id"].startswith("gemini-api/"):
        return query_gemini(model, prompt, image_b64, max_tokens)
    body = {
        "model": model["id"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        "usage": {"include": True},
    }
    if "reasoning" in (model.get("supported_parameters") or []):
        body["reasoning"] = {"effort": "low"}
    if max_tokens:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(
        f"{API}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}",
                "latency_s": round(time.monotonic() - t0, 2)}
    except Exception as e:  # network/timeout: record and keep going with the next model
        return {"error": repr(e), "latency_s": round(time.monotonic() - t0, 2)}
    if "error" in resp:
        return {"error": json.dumps(resp["error"])[:500], "latency_s": round(time.monotonic() - t0, 2)}
    choice = resp["choices"][0]
    return {
        "answer": (choice["message"].get("content") or "").strip(),
        "finish_reason": choice.get("finish_reason"),
        "provider": resp.get("provider"),
        "usage": resp.get("usage"),
        "latency_s": round(time.monotonic() - t0, 2),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually call the models (default: dry run)")
    ap.add_argument("--max-tier", type=int, default=5, help="only tiers <= N")
    ap.add_argument("--models", help="comma-separated IDs, overrides the built-in list")
    ap.add_argument("--max-tokens", type=int, help="cap output tokens (default: provider default)")
    ap.add_argument("--repeat", type=int, default=1, help="query each model/prompt N times; stops when credits run out")
    ap.add_argument("--prompts", default=",".join(PROMPTS), help="comma-separated prompt keys")
    args = ap.parse_args()

    selected = ([(0, m.strip()) for m in args.models.split(",")] if args.models
                else [(t, m) for t, m in MODELS if t <= args.max_tier])
    prompt_keys = [p.strip() for p in args.prompts.split(",")]
    for p in prompt_keys:
        if p not in PROMPTS:
            sys.exit(f"unknown prompt key {p!r}; choose from {list(PROMPTS)}")

    catalogue = fetch_catalogue()
    models = sorted(verify(selected, catalogue), key=lambda x: (x[0], per_mtok(x[2], "prompt")))
    if not args.run:
        print("\nDry run. Re-run with --run to query.")
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")

    image_b64 = base64.b64encode(IMAGE.read_bytes()).decode()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    total_cost = 0.0
    print(f"\nWriting {out}\n")
    with out.open("w", encoding="utf-8") as f:
        for rep in range(args.repeat):
            for tier, mid, m in models:
                for pk in prompt_keys:
                    r = query(key, m, PROMPTS[pk], image_b64, args.max_tokens)
                    if r.get("error", "").startswith("HTTP 401"):
                        sys.exit(f"Aborting: API key rejected ({r['error'][:120]}). Every call would fail.")
                    if r.get("error", "").startswith("HTTP 402"):
                        print(f"\nOut of credits during repeat #{rep}. Total cost: ${total_cost:.4f}")
                        return
                    cost = float((r.get("usage") or {}).get("cost") or 0)
                    total_cost += cost
                    row = {"tier": tier, "model": mid, "prompt": pk, "repeat": rep,
                           "max_tokens": args.max_tokens, **r}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    shown = r.get("answer", r.get("error", "")).replace("\n", " ")
                    print(f"[{mid}] #{rep} {pk:<15} ${cost:.5f} {r['latency_s']:>6}s  {shown[:160]}")
    print(f"\nTotal cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
