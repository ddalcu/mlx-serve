#!/usr/bin/env python3
"""plot_gemma_e4b_5way.py — render the 5-way Gemma 4 E4B chart.

CSV (long-form): engine|format|metric|value|prompt_n|hw
Metrics: ttft_cold_ms, ttft_warm_median_ms, decode_tps, prefill_tps_cold

4 grouped-bar panels:
  1. Cold TTFT (ms, lower is better)
  2. Warm TTFT (ms, lower is better; mlx-lm CLI cold-only so omitted)
  3. Cold prefill tok/s (higher is better)
  4. Decode tok/s (higher is better)

Colors by engine (mlx-serve / lmstudio / mlx-lm); patterns by format
(MLX vs GGUF). Best-in-panel bar gets a gold edge.
"""
import csv
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_csv(path: Path) -> dict:
    # data[(engine, format)][metric] = float
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            k = (row["engine"], row["format"])
            data.setdefault(k, {})[row["metric"]] = float(row["value"])
    return data


def find_latest() -> Path:
    candidates = sorted(glob.glob("docs/perf-csvs/gemma_e4b_5way-*.csv"))
    if not candidates:
        print("No gemma_e4b_5way-*.csv found", file=sys.stderr); sys.exit(1)
    return Path(candidates[-1])


# Order matters: dictates bar order left-to-right within each panel.
CONFIGS = [
    ("mlx-serve", "MLX-4bit"),
    ("mlx-serve", "GGUF-Q4_K_M"),
    ("lmstudio",  "MLX-4bit"),
    ("lmstudio",  "GGUF-Q4_K_M"),
    ("mlx-lm",    "MLX-4bit"),
]
LABELS = ["mlx-serve\nMLX-4bit", "mlx-serve\nGGUF-Q4_K_M",
          "LM Studio\nMLX-4bit", "LM Studio\nGGUF-Q4_K_M",
          "mlx-lm CLI\nMLX-4bit"]
COLORS = {"mlx-serve": "#1f77b4", "lmstudio": "#7f7f7f", "mlx-lm": "#2ca02c"}
HATCHES = {"MLX-4bit": "", "GGUF-Q4_K_M": "//"}


def panel(ax, data, metric, title, ylabel, higher_better, sub=None):
    vals = []
    for engine, fmt in CONFIGS:
        v = data.get((engine, fmt), {}).get(metric, 0) or 0
        if sub is True and v == 0:
            v = float("nan")
        vals.append(v)
    x = np.arange(len(CONFIGS))
    colors = [COLORS[e] for e, _ in CONFIGS]
    hatches = [HATCHES[f] for _, f in CONFIGS]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    # Highlight best: gold edge.
    clean = [v for v in vals if v and not np.isnan(v)]
    if clean:
        best_val = max(clean) if higher_better else min(clean)
        for i, v in enumerate(vals):
            if v == best_val:
                bars[i].set_edgecolor("#d4a017")
                bars[i].set_linewidth(3)
    for i, v in enumerate(vals):
        if v == 0 or np.isnan(v):
            ax.text(x[i], 0.5, "n/a", ha="center", va="bottom", fontsize=9, color="grey", style="italic")
            continue
        # ms vs tok/s formatting
        if "ms" in metric:
            label = f"{v:,.0f} ms" if v >= 100 else f"{v:.1f} ms"
        else:
            label = f"{v:.1f}"
        ax.text(x[i], v + max(vals) * 0.015, label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel(ylabel)
    arrow = "↓ lower is better" if not higher_better else "↑ higher is better"
    ax.set_title(f"{title}\n({arrow})", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    if clean:
        ax.set_ylim(0, max(clean) * 1.20)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest()
    out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.with_suffix(".png")
    data = load_csv(csv_path)
    # Read hw tag from any row
    hw = "unknown-hw"
    with open(csv_path) as f:
        rd = csv.DictReader(f, delimiter="|")
        for row in rd:
            hw = row.get("hw", hw); break

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Gemma 4 E4B — 5-way bench on {hw}  (1222-token prompt, temp=0)",
        fontsize=14, fontweight="bold",
    )

    panel(axes[0][0], data, "ttft_cold_ms",        "Cold TTFT", "Milliseconds", higher_better=False)
    panel(axes[0][1], data, "ttft_warm_median_ms", "Warm TTFT (median of runs 2-5)", "Milliseconds", higher_better=False, sub=True)
    panel(axes[1][0], data, "prefill_tps_cold",    "Cold prefill rate", "Tokens / sec", higher_better=True)
    panel(axes[1][1], data, "decode_tps",          "Decode rate (max_tokens=128)", "Tokens / sec", higher_better=True)

    # Legend
    handles = []
    for engine, color in COLORS.items():
        handles.append(mpatches.Patch(color=color, label=engine))
    for fmt, hatch in HATCHES.items():
        handles.append(mpatches.Patch(facecolor="white", edgecolor="black", hatch=hatch or " ", label=fmt))
    handles.append(mpatches.Patch(facecolor="white", edgecolor="#d4a017", linewidth=3, label="best in panel"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.005))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
