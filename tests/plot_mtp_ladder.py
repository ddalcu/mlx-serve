#!/usr/bin/env python3
"""plot_mtp_ladder.py — render the native-MTP context ladder (LM Studio vs
MTPLX vs MLX-serve) from a pipe-delimited CSV.

Input CSV columns (header row required):
  context|lms_prefill|mtplx_prefill|ours_prefill|lms_decode|mtplx_decode|ours_decode

One row per context rung (0.5k … 64k). All three engines load the IDENTICAL
checkpoint (Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed — 4-bit trunk + its
calibrated MTP adapter); LM Studio has no MTP support and decodes plain AR.
The numbers come from a manual head-to-head (fresh loads, AC power, best-of-N
per cell on every engine) — there is no bench.sh mode that produces this CSV.
LM Studio prefill is measured client-side (prompt_tokens / TTFT); the MTP
engines report engine-side prefill (the difference is ~1-3% at these sizes).

COLD-PROMPT RULE (2026-07-12): the ladder prompts are NESTED (each rung
shares 92-99% of the previous rung's bytes as a prefix), and LM Studio
prompt-caches across requests at 2048-token-chunk granularity — an ascending
same-session ladder inflates its 8K+ client pp ~1.5-1.7x (311/359/368/324
where cold measures 239/227/215/189; verified: fresh load + 32K first = 214.7,
immediate repeat = 0.87 s TTFT). Every LM Studio 8K+ cell here is COLD
(fresh model load per rung, or its vendored `mlx_lm generate` CLI, which
matches its server cold rate). Sub-8K rungs can't hit its cache (shared
prefix < one 2048 chunk) so ascending-session values are already cold.

Usage:
  python3 tests/plot_mtp_ladder.py docs/perf-csvs/mtp-ladder-26.7.6.csv \
      docs/perf-mtp-ladder-26.7.6.png

Requires matplotlib; style matches tests/plot_vs_lmstudio_omlx.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SUBTITLE = ("Identical checkpoint on all three engines: Qwen3.6-27B-MTPLX-Optimized-Speed "
            "(4-bit + calibrated MTP adapter) · Apple M4 Max · coding-agent prompts · "
            "temp 0.6 · fresh loads, COLD prompts, best-of-N per cell · MTPLX 2.0.2 · LM Studio 0.4.15 (no MTP)")

# Colors match the family charts: LM Studio = baseline gray, MTPLX = purple
# (comparison engine), MLX-serve native MTP = the MLXS-MTP orange.
ENGINES = [
    ("lms",   "LM Studio (no MTP, AR decode)", "#9ca3af", "#6b7280", True),
    ("mtplx", "MTPLX (native MTP)",            "#a78bfa", "#1f2937", False),
    ("ours",  "MLX-serve (native MTP)",        "#ea580c", "#1f2937", False),
]


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("|")
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != len(header) or not parts[0]:
                continue
            rows.append(dict(zip(header, parts)))
    if not rows:
        sys.exit(f"No data rows in {path}")
    return rows


def render(csv_path: Path, png_out: Path) -> None:
    rows = load_csv(csv_path)
    contexts = [r["context"] for r in rows]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": "#374151",
        "xtick.color": "#374151",
        "ytick.color": "#6b7280",
        "axes.titlecolor": "#111827",
    })

    fig, axes = plt.subplots(1, 2, figsize=(20, 6.6))
    fig.suptitle("Native MTP head-to-head — LM Studio vs MTPLX vs MLX-serve, 0.5K to 64K context",
                 fontsize=15, fontweight="bold", color="#111827", y=0.99)
    fig.text(0.5, 0.925, SUBTITLE, ha="center", fontsize=9.5, color="#4b5563")

    panels = [
        ("prefill", "Prefill (tok/s)", "prefill tok/s"),
        ("decode", "Decode (tok/s)", "decode tok/s"),
    ]

    x = np.arange(len(contexts))
    width = 0.27
    for ax, (key, panel_title, ylab) in zip(axes, panels):
        series = {eng: [float(r[f"{eng}_{key}"]) for r in rows] for eng, *_ in ENGINES}
        top = max(v for vals in series.values() for v in vals)
        for e_idx, (eng, label, color, edge, light) in enumerate(ENGINES):
            vals = series[eng]
            offset = (e_idx - 1) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=color,
                          edgecolor=edge, linewidth=0.5, zorder=2)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 0.5, f"{val:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="#111827" if light else "#ffffff",
                        fontweight="bold", rotation=90)
            # Percent delta above OUR bar, vs MTPLX at the same rung — the
            # MTP-runtime race; LM Studio is the no-MTP floor.
            if eng == "ours":
                for bar, val, base in zip(bars, vals, series["mtplx"]):
                    if base <= 0:
                        continue
                    gain = (val / base - 1) * 100
                    gcolor = ("#15803d" if gain >= 5 else
                              "#b91c1c" if gain <= -5 else "#525252")
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + top * 0.015,
                            f"{gain:+.0f}%", ha="center", va="bottom",
                            fontsize=8.5, color=gcolor, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(contexts, fontsize=10, fontweight="medium")
        ax.set_xlabel("prompt context", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(panel_title, fontsize=12, fontweight="semibold", pad=8)
        ax.grid(True, axis="y", alpha=0.35, linestyle="--", color="#d1d5db", zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, top * 1.14)

    axes[1].legend(loc="upper right", fontsize=9.5, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(png_out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {png_out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render the LM Studio / MTPLX / MLX-serve MTP-ladder chart.")
    p.add_argument("csv", type=Path, help="input CSV path")
    p.add_argument("png", type=Path, help="output PNG path")
    args = p.parse_args()
    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")
    render(args.csv, args.png)


if __name__ == "__main__":
    main()
