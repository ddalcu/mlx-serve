#!/usr/bin/env python3
"""plot_decode.py — render the decode tok/s comparison from
tests/bench_decode.sh. Single bar group per (label, kind), median of N runs.
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 7 or parts[0] == "label":
                continue
            label, kind, _run, _ct, dec, _wall, _hw = parts[:7]
            try:
                v = float(dec)
            except ValueError:
                continue
            if v > 0:
                out[(label, kind)].append(v)
    return out


LABEL_HUMAN = {
    "qwen35-4b-mlx": "Qwen3.5-4B MLX\n(hybrid SSM)",
    "gemma4-e4b": "Gemma 4 E4B MLX\n(plain attn)",
    "qwen35-4b-gguf": "Qwen3.5-4B GGUF\n(IQ4_NL, llama.cpp)",
}


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: plot_decode.py <csv> <png>")
    data = load(Path(sys.argv[1]))
    if not data:
        sys.exit("empty CSV")

    labels = sorted({k[0] for k in data.keys()})
    medians = defaultdict(dict)
    for (label, kind), vs in data.items():
        medians[label][kind] = statistics.median(vs)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "Decode tok/s · mlx-serve vs LM Studio · Apple M4 / 16 GB\n"
        "Same model files · max_tokens=128 · temperature=0 · median of 3 runs",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(labels))
    width = 0.35
    kinds = ["mlx-serve", "lmstudio"]
    kind_label = {"mlx-serve": "mlx-serve (this work)", "lmstudio": "LM Studio"}
    kind_color = {"mlx-serve": "#3b82f6", "lmstudio": "#888888"}

    for i, kind in enumerate(kinds):
        vals = [medians[l].get(kind, 0) for l in labels]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=kind_label[kind],
                      color=kind_color[kind], edgecolor="black", linewidth=0.6)
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v <= 0:
                continue
            if kind == "mlx-serve":
                txt = f"{v:.1f}"
                col = "#111"
            else:
                ours = medians[labels[j]].get("mlx-serve", 0) or 1e-9
                pct = (ours / v - 1) * 100  # higher tok/s = better, so ours/theirs > 1 = good
                col = "#15803d" if pct >= 10 else ("#b91c1c" if pct <= -5 else "#525252")
                txt = f"{v:.1f}\n{'+' if pct >= 0 else ''}{pct:.0f}%"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals + [1]) * 0.015,
                    txt, ha="center", va="bottom", fontsize=10, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_HUMAN.get(l, l) for l in labels], fontsize=10)
    ax.set_ylabel("decode tok/s (higher is better)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    cur = ax.get_ylim()
    ax.set_ylim(0, cur[1] * 1.18)

    fig.text(0.5, 0.005,
             "Green +% on the mlx-serve bar shows the relative throughput win vs LM Studio on the same model file.",
             ha="center", fontsize=9, color="#555", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
