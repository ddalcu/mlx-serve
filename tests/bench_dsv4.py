#!/usr/bin/env python3
"""DSV4 single-prompt bench harness — speed-loop instrument.

Measures decode tok/s and prefill tok/s on a canonical prompt against either:
  - mlx-serve  (via HTTP /v1/chat/completions, stream=false)
  - mlx-lm     (direct via `mlx_lm.generate`/`stream_generate`, no server)

Same prompt for both engines, same temperature (0.0), same max_tokens.
Throws out the first run (cold JIT), reports the median of the rest.

Usage:
  # mlx-serve (server must already be running at --port)
  python3 tests/bench_dsv4.py --engine mlx-serve --port 8095 --runs 5

  # mlx-lm (loads model fresh; first call is the warm-up that is discarded)
  /private/tmp/mlx-lm-dsv4/bin/python tests/bench_dsv4.py --engine mlx-lm --runs 5

  # Append a row to BenchmarkLog.md with --log-row "label"
  python3 tests/bench_dsv4.py --engine mlx-serve --port 8095 --log-row "phase-1.1 padded-rope-on"

Output (stdout, key=value):
  engine=mlx-serve runs=5 prompt_tokens=29 completion_tokens=256
  prefill_tps_median=212.4 decode_tps_median=18.7 wall_median_s=14.0
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

CANONICAL_PROMPT = (
    "How do i build a sveltekit blog app with prisma orm ? "
    "Tell me the files i need to create"
)

DEFAULT_MODEL = str(
    Path.home() / ".lmstudio/models/mlx-community/DeepSeek-V4-Flash-2bit-DQ"
)


def bench_mlx_serve(host: str, port: int, runs: int, max_tokens: int, model: str, mtp: bool = False) -> list[dict]:
    url = f"http://{host}:{port}/v1/chat/completions"
    body_template = {
        "model": model,
        "messages": [{"role": "user", "content": CANONICAL_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "enable_mtp": mtp,
    }
    rows: list[dict] = []
    for i in range(runs):
        payload = json.dumps(body_template).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            sys.stderr.write(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}\n")
            raise
        wall = time.monotonic() - t0
        usage = data.get("usage", {})
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        # We don't know split of prefill vs decode wall-clock from the JSON alone;
        # the server logs `prefill: X tok/s` and `decode: X tok/s` lines, but we
        # don't have them here. Approximate: decode dominates for long-ish
        # completions, so report ct/wall as decode_tps. For exact split, parse
        # server logs externally and pass via --server-log.
        decode_tps = ct / wall if wall > 0 else 0.0
        prefill_tps = pt / wall if wall > 0 else 0.0  # rough; mainly informational
        rows.append({
            "run": i,
            "wall_s": wall,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "decode_tps": decode_tps,
            "prefill_tps_approx": prefill_tps,
        })
        sys.stderr.write(
            f"  run {i}: wall={wall:.2f}s pt={pt} ct={ct} decode={decode_tps:.2f} tok/s\n"
        )
    return rows


def bench_mlx_lm(model_path: str, runs: int, max_tokens: int) -> list[dict]:
    try:
        import mlx.core as mx  # type: ignore
        from mlx_lm import load, stream_generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore
    except ImportError as e:
        sys.stderr.write(
            f"mlx_lm not importable in this Python ({sys.executable}). "
            f"Use /private/tmp/mlx-lm-dsv4/bin/python. {e}\n"
        )
        sys.exit(2)

    sys.stderr.write(f"Loading {model_path}...\n")
    t_load = time.monotonic()
    model, tokenizer = load(model_path)
    sys.stderr.write(f"Loaded in {time.monotonic() - t_load:.1f}s\n")

    sampler = make_sampler(temp=0.0)
    messages = [{"role": "user", "content": CANONICAL_PROMPT}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    rows: list[dict] = []
    for i in range(runs):
        # Stream generation so we can extract per-segment timing from mlx-lm's
        # built-in counters (prompt_tps, gen_tps).
        t0 = time.monotonic()
        last_response = None
        for response in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
        ):
            last_response = response
        wall = time.monotonic() - t0
        if last_response is None:
            raise RuntimeError("stream_generate returned no tokens")
        pt = int(getattr(last_response, "prompt_tokens", 0))
        ct = int(getattr(last_response, "generation_tokens", 0))
        prefill_tps = float(getattr(last_response, "prompt_tps", 0.0))
        decode_tps = float(getattr(last_response, "generation_tps", 0.0))
        rows.append({
            "run": i,
            "wall_s": wall,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "decode_tps": decode_tps,
            "prefill_tps": prefill_tps,
        })
        sys.stderr.write(
            f"  run {i}: wall={wall:.2f}s pt={pt} ct={ct} "
            f"prefill={prefill_tps:.2f} decode={decode_tps:.2f} tok/s\n"
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    # Discard the first run (cold JIT) when we have >1 run.
    sample = rows[1:] if len(rows) > 1 else rows
    decode_tps = [r["decode_tps"] for r in sample]
    walls = [r["wall_s"] for r in sample]
    return {
        "runs_total": len(rows),
        "runs_used": len(sample),
        "decode_tps_median": statistics.median(decode_tps),
        "decode_tps_min": min(decode_tps),
        "decode_tps_max": max(decode_tps),
        "wall_median_s": statistics.median(walls),
        "prompt_tokens": rows[-1]["prompt_tokens"],
        "completion_tokens": rows[-1]["completion_tokens"],
    }


def append_benchmark_log(repo_root: Path, label: str, engine: str, summary: dict) -> None:
    log = repo_root / "BenchmarkLog.md"
    line = (
        f"\n### bench_dsv4 — {time.strftime('%Y-%m-%d %H:%M')} — {label}\n"
        f"- engine={engine} runs={summary['runs_used']}/{summary['runs_total']} "
        f"decode_tps median={summary['decode_tps_median']:.2f} "
        f"(min={summary['decode_tps_min']:.2f} max={summary['decode_tps_max']:.2f}) "
        f"wall_median={summary['wall_median_s']:.2f}s "
        f"pt={summary['prompt_tokens']} ct={summary['completion_tokens']}\n"
    )
    with log.open("a") as f:
        f.write(line)
    sys.stderr.write(f"Logged to {log}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mlx-serve", "mlx-lm"], required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--log-row", default=None, help="If set, append a summary to BenchmarkLog.md with this label.")
    ap.add_argument("--mtp", action="store_true", help="Set enable_mtp=true on mlx-serve requests (no effect for mlx-lm — they don't ship MTP for DSV4).")
    args = ap.parse_args()

    if args.engine == "mlx-serve":
        rows = bench_mlx_serve(args.host, args.port, args.runs, args.max_tokens, args.model, mtp=args.mtp)
    else:
        rows = bench_mlx_lm(args.model, args.runs, args.max_tokens)

    summary = summarize(rows)
    print(
        f"engine={args.engine} runs={summary['runs_used']}/{summary['runs_total']} "
        f"prompt_tokens={summary['prompt_tokens']} "
        f"completion_tokens={summary['completion_tokens']} "
        f"decode_tps_median={summary['decode_tps_median']:.2f} "
        f"decode_tps_min={summary['decode_tps_min']:.2f} "
        f"decode_tps_max={summary['decode_tps_max']:.2f} "
        f"wall_median_s={summary['wall_median_s']:.2f}"
    )

    if args.log_row:
        repo_root = Path(__file__).resolve().parent.parent
        append_benchmark_log(repo_root, args.log_row, args.engine, summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
