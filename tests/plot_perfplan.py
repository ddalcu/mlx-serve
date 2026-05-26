#!/usr/bin/env python3
"""plot_perfplan.py — render the performance-plan.md comparison chart from a
CSV produced by `tests/bench_perfplan_compare.sh`.

Two panels per chart:
  - Cold prefill (left): wall_ms median per (label, kind)
  - Multi-turn warm  (right): median wall_ms across warm_repeat2..N

Each label has two bars: mlx-serve vs lmstudio. Labels above each bar:
  - mlx-serve: raw value
  - lmstudio:  raw value + percent vs mlx-serve (red if behind us, green if ahead)

Usage:
  python3 tests/plot_perfplan.py <csv_in> <png_out>
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Returns nested dict: label -> kind -> scenario -> [wall_ms values]."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 12 or parts[0] == "label":
                continue
            label, kind, _model, scenario, _run, _pn, _cn, _pms, _tps, wall, _hw, _notes = parts[:12]
            try:
                wall_v = float(wall)
            except ValueError:
                continue
            if wall_v <= 0:
                continue
            out[label][kind][scenario].append(wall_v)
    return out


def median_safe(xs):
    return statistics.median(xs) if xs else 0.0


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: plot_perfplan.py <csv_in> <png_out>")
    csv_path = Path(sys.argv[1])
    png_out = Path(sys.argv[2])
    data = load(csv_path)
    if not data:
        sys.exit(f"empty CSV: {csv_path}")

    labels = sorted(data.keys())
    # Aggregations per label:
    #   cold = median(cold runs, both run 1+2)
    #   warm = median(warm_repeat2..N) — first is the cold within the warm batch
    cold = {l: {} for l in labels}
    warm = {l: {} for l in labels}
    for l in labels:
        for kind in data[l]:
            cold_runs = data[l][kind].get("cold", [])
            warm_repeats = []
            for scen, vals in data[l][kind].items():
                if scen.startswith("warm_repeat"):
                    warm_repeats += vals
            cold[l][kind] = median_safe(cold_runs)
            warm[l][kind] = median_safe(warm_repeats)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        "mlx-serve vs LM Studio — wall TTFT after Phase 1 (SSM checkpointing)\n"
        "Apple M4 / 16 GB · same model files · temperature=0 · max_tokens=4",
        fontsize=13, fontweight="bold",
    )

    label_human = {
        "qwen35-4b-mlx": "Qwen3.5-4B MLX\n(hybrid SSM)",
        "gemma4-e4b": "Gemma 4 E4B MLX\n(plain attn)",
        "qwen35-4b-gguf": "Qwen3.5-4B GGUF\n(IQ4_NL, llama.cpp)",
    }

    x = np.arange(len(labels))
    width = 0.35
    kinds = ["mlx-serve", "lmstudio"]
    kind_label = {"mlx-serve": "mlx-serve", "lmstudio": "LM Studio"}
    kind_color = {"mlx-serve": "#3b82f6", "lmstudio": "#888888"}

    panels = [
        ("cold prefill (ms, lower is better)", cold),
        ("multi-turn warm TTFT (ms, lower is better)", warm),
    ]

    for panel_idx, (title, source) in enumerate(panels):
        ax = axes[panel_idx]
        for i, kind in enumerate(kinds):
            vals = [source[l].get(kind, 0) for l in labels]
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                label=kind_label[kind], color=kind_color[kind],
                edgecolor="black", linewidth=0.6,
            )
            for j, (bar, v) in enumerate(zip(bars, vals)):
                if v <= 0:
                    continue
                if kind == "mlx-serve":
                    txt = f"{v:.0f}"
                    col = "#111"
                else:
                    ours = source[labels[j]].get("mlx-serve", 0) or 1e-9
                    ratio = v / ours
                    pct = (ratio - 1) * 100
                    # We want LMS ≥ us by margin (we're FASTER means lower ms).
                    # Green when LMS is much slower (we win); red when LMS is faster.
                    col = "#15803d" if pct >= 20 else ("#b91c1c" if pct <= -5 else "#525252")
                    txt = f"{v:.0f}\n{'+' if pct >= 0 else ''}{pct:.0f}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.012,
                    txt,
                    ha="center", va="bottom", fontsize=9, color=col,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([label_human.get(l, l) for l in labels], fontsize=9)
        ax.set_ylabel("wall time (ms)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if panel_idx == 0:
            ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
        # Headroom for label
        cur = ax.get_ylim()
        ax.set_ylim(0, cur[1] * 1.22)

    # Tagline subtitle
    fig.text(
        0.5, 0.005,
        "(Phase 1: per-position SSM/conv state checkpoints during prefill — "
        "hybrid SSM models can now reuse prompt prefixes across turns. "
        "Green +% = mlx-serve is decisively faster.)",
        ha="center", fontsize=9, color="#555", style="italic",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    png_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    print(f"Wrote {png_out}")


if __name__ == "__main__":
    main()
