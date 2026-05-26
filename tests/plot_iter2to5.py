#!/usr/bin/env python3
"""plot_iter2to5.py — render the Iteration 2 + 3-5 perf chart from the
focused bench CSV (`tests/bench_iter2to5.sh`).

CSV columns:
    label|run|prompt_n|cached_n|tokenize_ms|prompt_ms|predicted_per_second|hw

Three panels arranged in one figure:

  1. **Iteration 2 — Gemma 4 E4B MLX warm tokenize cache.** Stacked
     bar per run (cold=run 1, warm=runs 2-5). Bottom = tokenize_ms,
     top = prompt_ms. Demonstrates the 236 ms tokenize cost vanishing
     on the warm path while prompt_ms drops to the KV-cache-hit floor.

  2. **Iteration 3-5 — Qwen3.5-4B GGUF cached_n on alternating prompts.**
     Side-by-side bars: multi-session (`--llama-cache-entries 2`) vs
     single-session (`--llama-cache-entries 1`), for each of the 5
     alternating-A/B requests. With multi-session, the second visit to
     each prompt hits at 40/41 cached_n; single-session stays at 0.

  3. **Iteration 3-5 — same workload, decode tok/s.** Multi-session
     warm hits lift decode from ~19 tok/s to ~29 tok/s because the
     no-prefix-rebuild path skips llama.cpp's sync overhead.

Usage: ./tests/plot_iter2to5.py [csv_path] [out_png]
Defaults: latest `docs/perf-csvs/bench_iter2to5-*.csv` → PNG sibling.
"""
import csv
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            row["run"] = int(row["run"])
            row["prompt_n"] = int(row["prompt_n"])
            row["cached_n"] = int(row["cached_n"])
            row["tokenize_ms"] = float(row["tokenize_ms"])
            row["prompt_ms"] = float(row["prompt_ms"])
            row["predicted_per_second"] = float(row["predicted_per_second"])
            rows.append(row)
    return rows


def find_latest_csv() -> Path:
    candidates = sorted(glob.glob("docs/perf-csvs/bench_iter2to5-*.csv"))
    if not candidates:
        print("No bench_iter2to5-*.csv found under docs/perf-csvs/", file=sys.stderr)
        sys.exit(1)
    return Path(candidates[-1])


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_csv()
    out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.with_suffix(".png")
    rows = load_csv(csv_path)
    hw = rows[0]["hw"] if rows else "unknown-hw"

    # ── Iteration 2 data ──
    gemma = [r for r in rows if r["label"] == "gemma4-e4b-mlx"]
    gemma.sort(key=lambda r: r["run"])

    # ── Iteration 3-5 data ──
    multi = [r for r in rows if r["label"].startswith("qwen35-4b-gguf-multi-")]
    single = [r for r in rows if r["label"].startswith("qwen35-4b-gguf-single-")]
    multi.sort(key=lambda r: r["run"])
    single.sort(key=lambda r: r["run"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Overnight perf push (Iterations 2 + 3-5) on " + hw,
        fontsize=14, fontweight="bold",
    )

    # ─── Panel 1: Gemma 4 stacked TTFT bars ───
    ax = axes[0]
    runs = [r["run"] for r in gemma]
    tok = [r["tokenize_ms"] for r in gemma]
    prm = [r["prompt_ms"] for r in gemma]
    bars_tok = ax.bar(runs, tok, color="#d62728", label="tokenize_ms")
    bars_prm = ax.bar(runs, prm, bottom=tok, color="#1f77b4", label="prompt_ms (Metal prefill)")
    for i, (t, p) in enumerate(zip(tok, prm)):
        total = t + p
        ax.text(runs[i], total + max(prm) * 0.02, f"{total:.0f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Iteration 2 — Gemma 4 E4B MLX (1813-tok)\nstacked TTFT components", fontsize=11)
    ax.set_xlabel("Run (1 = cold, 2+ = warm)")
    ax.set_ylabel("Milliseconds")
    ax.set_xticks(runs)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max((tok[0] + prm[0]) * 1.15, 100))
    ax.grid(axis="y", alpha=0.3)
    # Annotation arrow showing the win
    cold_total = tok[0] + prm[0]
    warm_total = tok[1] + prm[1]
    ax.annotate(
        f"{cold_total / warm_total:.0f}× faster warm\n({cold_total:.0f} → {warm_total:.1f} ms)",
        xy=(2, warm_total + 30), xytext=(3, cold_total * 0.65),
        fontsize=10, ha="center", fontweight="bold", color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
    )

    # ─── Panel 2: cached_n alternating ───
    ax = axes[1]
    runs2 = list(range(1, 6))
    multi_c = [r["cached_n"] for r in multi]
    single_c = [r["cached_n"] for r in single]
    x = np.arange(len(runs2))
    w = 0.4
    bars_m = ax.bar(x - w / 2, multi_c, w,
                    color="#2ca02c", label="--llama-cache-entries 2 (multi-session LRU)")
    bars_s = ax.bar(x + w / 2, single_c, w,
                    color="#7f7f7f", label="--llama-cache-entries 1 (single-session, baseline)")
    # Annotate labels: A/B sequence
    seq = ["A1", "B1", "A2", "B2", "A3"]
    ax.set_xticks(x)
    ax.set_xticklabels(seq)
    ax.set_xlabel("Alternating request (A then B then A …)")
    ax.set_ylabel("cached_n (out of 41 prompt tokens)")
    for i, v in enumerate(multi_c):
        ax.text(i - w / 2, v + 1, str(v), ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(single_c):
        ax.text(i + w / 2, v + 1, str(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Iteration 3-5 — Qwen3.5-4B GGUF\nKV reuse on alternating A/B prompts", fontsize=11)
    ax.set_ylim(0, max(multi_c) * 1.25 if multi_c else 50)
    ax.axhline(41, ls="--", color="black", alpha=0.3, lw=0.8)
    ax.text(0, 41.5, "prompt_n = 41", fontsize=8, color="black", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ─── Panel 3: decode tok/s ───
    ax = axes[2]
    multi_d = [r["predicted_per_second"] for r in multi]
    single_d = [r["predicted_per_second"] for r in single]
    bars_m = ax.bar(x - w / 2, multi_d, w, color="#2ca02c",
                    label="--llama-cache-entries 2")
    bars_s = ax.bar(x + w / 2, single_d, w, color="#7f7f7f",
                    label="--llama-cache-entries 1")
    ax.set_xticks(x)
    ax.set_xticklabels(seq)
    ax.set_xlabel("Alternating request")
    ax.set_ylabel("Decode tokens / sec")
    for i, v in enumerate(multi_d):
        ax.text(i - w / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(single_d):
        ax.text(i + w / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Iteration 3-5 — same workload\ndecode tok/s (warm hits skip llama.cpp sync overhead)", fontsize=11)
    ax.set_ylim(0, max(multi_d + single_d) * 1.2)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
