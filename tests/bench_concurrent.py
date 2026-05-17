#!/usr/bin/env python3
"""Multi-slot concurrent throughput bench.

Fires N concurrent /v1/chat/completions requests against a running server and
measures total wall-clock + per-request decode tok/s. The total-throughput
metric (sum of completion_tokens / total wall) is the headline for "did
batching help?" — should approach N × single-request rate if the scheduler
batches efficiently.

Usage:
  ./bench_concurrent.py --port 8095 --concurrency 2 --runs 3
"""
import argparse, json, statistics, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

CANONICAL_PROMPT = (
    "How do i build a sveltekit blog app with prisma orm ? "
    "Tell me the files i need to create"
)

def one_request(host: str, port: int, idx: int, max_tokens: int, seed: int | None) -> dict:
    url = f"http://{host}:{port}/v1/chat/completions"
    body = {
        "messages": [{"role": "user", "content": CANONICAL_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "enable_mtp": False,
    }
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    wall = time.monotonic() - t0
    usage = d.get("usage", {})
    return {
        "idx": idx,
        "wall_s": wall,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "decode_tps": int(usage.get("completion_tokens", 0)) / wall if wall > 0 else 0.0,
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default="concurrent-bench")
    args = ap.parse_args()

    print(f"Warming up with 1 single request...", file=sys.stderr)
    one_request(args.host, args.port, 0, args.max_tokens, None)

    # Single-request baseline (for reference; throughput baseline)
    print(f"\n=== Single-request baseline (concurrency=1) ===", file=sys.stderr)
    single_runs = []
    for i in range(args.runs):
        r = one_request(args.host, args.port, i, args.max_tokens, None)
        single_runs.append(r)
        print(f"  run {i}: wall={r['wall_s']:.2f}s ct={r['completion_tokens']} decode={r['decode_tps']:.2f} tok/s", file=sys.stderr)
    single_decode_median = statistics.median([r["decode_tps"] for r in single_runs])
    single_wall_median = statistics.median([r["wall_s"] for r in single_runs])
    single_total_tps = single_decode_median  # by definition

    # Concurrent N
    print(f"\n=== Concurrent {args.concurrency}-way ===", file=sys.stderr)
    concurrent_rounds = []
    for round_i in range(args.runs):
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(one_request, args.host, args.port, j, args.max_tokens, None)
                       for j in range(args.concurrency)]
            results = [f.result() for f in futures]
        total_wall = time.monotonic() - t0
        total_ct = sum(r["completion_tokens"] for r in results)
        total_tps = total_ct / total_wall
        per_req_decode_avg = statistics.mean([r["decode_tps"] for r in results])
        concurrent_rounds.append({
            "round": round_i, "wall_s": total_wall, "total_ct": total_ct,
            "total_tps": total_tps, "per_req_decode_avg": per_req_decode_avg,
        })
        print(f"  round {round_i}: wall={total_wall:.2f}s "
              f"total_ct={total_ct} total_tps={total_tps:.2f} "
              f"per_req_avg={per_req_decode_avg:.2f} tok/s",
              file=sys.stderr)

    total_tps_median = statistics.median([r["total_tps"] for r in concurrent_rounds])
    per_req_decode_median = statistics.median([r["per_req_decode_avg"] for r in concurrent_rounds])
    speedup = total_tps_median / single_total_tps

    print()
    print(f"label={args.label} concurrency={args.concurrency}")
    print(f"single_decode_tps_median={single_decode_median:.2f}")
    print(f"single_wall_median_s={single_wall_median:.2f}")
    print(f"concurrent_total_tps_median={total_tps_median:.2f}")
    print(f"concurrent_per_req_decode_avg_median={per_req_decode_median:.2f}")
    print(f"throughput_speedup={speedup:.2f}x (ideal={args.concurrency}.0x)")
    print(f"per_req_latency_ratio={single_decode_median / per_req_decode_median:.2f}x slowdown (ideal=1.0x)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
