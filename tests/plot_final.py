#!/usr/bin/env python3
"""plot_final.py — render the final mlx-serve vs LM Studio chart from a CSV
produced by `tests/bench_final.sh`.

CSV columns: label|kind|run|prompt_n|cached_n|prompt_ms|wall_ms|hw

Each (label, kind) has N runs against an identical prompt. Run 1 is cold;
runs 2..N are warm. Median of warm runs = warm number. Run 1 = cold number.

Two panels:
  - Cold TTFT (left)
  - Warm TTFT median (right)

Bars labelled with raw ms + relative %; mlx-serve in blue, LM Studio in grey.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    out = defaultdict(lambda: defaultdict(list))  # (label, kind) -> [(run, wall_ms)]
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 8 or parts[0] == "label":
                continue
            label, kind, run, _pn, _cn, _pms, wall, _hw = parts[:8]
            try:
                run_i = int(run)
                wall_v = float(wall)
            except ValueError:
                continue
            out[(label, kind)] = out[(label, kind)]
            out[(label, kind)][run_i] = wall_v if not isinstance(out[(label, kind)], dict) else wall_v
    # Re-shape into dict[(label,kind)] -> dict[run]->wall_ms
    fixed: dict = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 8 or parts[0] == "label":
                continue
            label, kind, run, _pn, _cn, _pms, wall, _hw = parts[:8]
            try:
                fixed[(label, kind)][int(run)] = float(wall)
            except ValueError:
                continue
    return fixed


LABEL_HUMAN = {
    "qwen35-4b-mlx": "Qwen3.5-4B MLX\n(hybrid SSM)",
    "gemma4-e4b": "Gemma 4 E4B MLX\n(plain attn)",
    "qwen35-4b-gguf": "Qwen3.5-4B GGUF\n(IQ4_NL, llama.cpp)",
}


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: plot_final.py <csv> <png>")
    csv_path = Path(sys.argv[1])
    png_out = Path(sys.argv[2])
    data = load(csv_path)
    if not data:
        sys.exit(f"empty CSV: {csv_path}")

    labels = sorted({k[0] for k in data.keys()})
    cold = {l: {} for l in labels}
    warm = {l: {} for l in labels}
    for (label, kind), runs in data.items():
        # run 1 = cold
        if 1 in runs and runs[1] > 0:
            cold[label][kind] = runs[1]
        # runs 2..N = warm
        warm_vals = [v for r, v in runs.items() if r >= 2 and v > 0]
        if warm_vals:
            warm[label][kind] = statistics.median(warm_vals)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "mlx-serve vs LM Studio · wall TTFT (lower is better) · Apple M4 / 16 GB\n"
        "Same model files · temperature=0 · max_tokens=8 · identical-prompt warm-up",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(len(labels))
    width = 0.35
    kinds = ["mlx-serve", "lmstudio"]
    kind_label = {"mlx-serve": "mlx-serve (this work, +Phase 1)", "lmstudio": "LM Studio"}
    kind_color = {"mlx-serve": "#3b82f6", "lmstudio": "#888888"}

    panels = [("Cold (1st request, KV empty)", cold), ("Warm (subsequent identical-prompt, median)", warm)]

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
                    # We're FASTER means lower ms ⇒ ours < theirs ⇒ +% on their bar.
                    pct = (v / ours - 1) * 100
                    col = "#15803d" if pct >= 20 else ("#b91c1c" if pct <= -5 else "#525252")
                    txt = f"{v:.0f}\n{'+' if pct >= 0 else ''}{pct:.0f}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals + [1]) * 0.012,
                    txt,
                    ha="center", va="bottom", fontsize=9.5, color=col,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_HUMAN.get(l, l) for l in labels], fontsize=10)
        ax.set_ylabel("wall time (ms)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if panel_idx == 0:
            ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
        cur = ax.get_ylim()
        ax.set_ylim(0, cur[1] * 1.22)

    fig.text(
        0.5, 0.005,
        "Phase 1 (this work): per-position SSM/conv state checkpoints during prefill — "
        "hybrid SSM (Qwen3.5) now reuses prompt prefixes across turns. "
        "Green +% = mlx-serve is decisively faster.",
        ha="center", fontsize=9, color="#555", style="italic",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    png_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    print(f"Wrote {png_out}")


if __name__ == "__main__":
    main()
