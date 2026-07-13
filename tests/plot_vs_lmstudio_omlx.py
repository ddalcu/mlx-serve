#!/usr/bin/env python3
"""plot_vs_lmstudio_omlx.py — render the MLX-serve vs LM Studio vs oMLX
comparison chart from a CSV produced by tests/bench.sh.

Three panels per chart:
  - "Echo (verbatim recitation)"       → echo prompt — PLD's home turf
  - "Code completion (drafter's turf)" → code prompt — fresh code, no n-gram lookup
  - "Free-form writing (parity)"       → decode (creative essay) prompt

Bar layout depends on --family:
  gemma   → 6 bars: LM Studio MLX | LM Studio GGUF | oMLX | MLX-serve {--no-pld, --pld, --drafter}
  qwen36  → 5 bars: LM Studio MLX (baseline) | LM Studio GGUF | oMLX | MLX-serve {--no-pld, --pld}

Usage:
  python3 tests/plot_vs_lmstudio_omlx.py <csv> <png_out> --family <gemma|qwen36>

Requires matplotlib; install with `pip3 install --user matplotlib`
(or `--break-system-packages` on PEP-668 systems).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Family-specific layout. Each variant tuple is:
#   (variant_filter, spec, label, color, is_baseline, show_delta, short)
# - variant_filter matches the second '/' segment in the CSV `label` column
#   ("lmstudio-baseline", "lmstudio-alt", "omlx", "mlx-serve-gguf", "mlx-serve")
# - spec matches the third segment ("none", "pld", "drafter")
# - is_baseline: this row is the reference for percentage deltas (one per family)
# - show_delta: render a "+X%" label above the bar. False for comparison engines
#   to halve label density; the bar height already shows the comparison visually.
# - short: 3-9 char label rendered inside each bar under the tok/s value, so
#   bars self-identify without the reader chasing the top-of-figure legend.
FAMILIES = {
    "gemma": {
        "title": "MLX-serve vs LM Studio — Gemma 4 (Apple Silicon, decode tok/s)",
        "x_label": lambda key: {
            "gemma4-e2b-4bit":              "E2B (4bit)",
            "gemma4-e4b-4bit":              "E4B (4bit)",
            "gemma4-31b-4bit":              "31B (4bit)",
            "gemma4-26b-a4b-moe-4bit":      "26B-A4B-MoE (4bit)",
            "gemma4-26b-a4b-moe-qat-4bit":  "26B-A4B-MoE (QAT 4bit)",
        }.get(key, key),
        "model_order": [
            "gemma4-e2b-4bit",
            "gemma4-e4b-4bit",
            "gemma4-31b-4bit",
            "gemma4-26b-a4b-moe-qat-4bit",
        ],
        # Visual order: comparison engines (muted grays/cool) → mlx-serve
        # variants (vivid). Percentage deltas only on the mlx-serve rows so
        # the labels above tiny bars don't pile up.
        "variants": [
            ("lmstudio-baseline", "none",    "LM Studio (MLX, baseline)",   "#9ca3af", True,  False, "LM-MLX"),
            ("lmstudio-alt",      "none",    "LM Studio (GGUF)",            "#d1d5db", False, False, "LM-GG"),
            ("mlx-serve-gguf",    "none",    "MLX-serve (GGUF / llama.cpp)", "#a78bfa", False, False, "MLXS-GG"),
            ("omlx",              "none",    "oMLX",                        "#10b981", False, False, "oMLX"),
            ("mlx-serve",         "none",    "MLX-serve (MLX, --no-pld)",   "#2563eb", False, True,  "MLXS-NPLD"),
            ("mlx-serve",         "pld",     "MLX-serve (MLX, --pld)",      "#16a34a", False, True,  "MLXS-PLD"),
            ("mlx-serve",         "drafter", "MLX-serve (MLX, --drafter)",  "#ea580c", False, True,  "MLXS-DRFT"),
        ],
    },
    "qwen36": {
        "title": "MLX-serve vs LM Studio — Qwen 3.6 (Apple Silicon, decode tok/s)",
        "x_label": lambda key: {
            "qwen36-27b":      "27B (4bit)",
            "qwen36-35b-a3b":  "35B-A3B (4bit)",
        }.get(key, key),
        "model_order": [
            "qwen36-27b",
            "qwen36-35b-a3b",
        ],
        "variants": [
            ("lmstudio-baseline", "none", "LM Studio (MLX, baseline)",   "#9ca3af", True,  False, "LM-MLX"),
            ("lmstudio-alt",      "none", "LM Studio (GGUF)",            "#d1d5db", False, False, "LM-GG"),
            ("mlx-serve-gguf",    "none", "MLX-serve (GGUF / llama.cpp)", "#a78bfa", False, False, "MLXS-GG"),
            ("omlx",              "none", "oMLX",                        "#10b981", False, False, "oMLX"),
            ("mlx-serve",         "none", "MLX-serve (MLX, --no-pld)",   "#2563eb", False, True,  "MLXS-NPLD"),
            ("mlx-serve",         "pld",  "MLX-serve (MLX, --pld)",      "#16a34a", False, True,  "MLXS-PLD"),
            ("mlx-serve",         "mtp",  "MLX-serve (MLX, native MTP)", "#ea580c", False, True,  "MLXS-MTP"),
        ],
    },
}


def load_csv(path: Path, hardware_filter: str | None = None) -> tuple[dict, set[str]]:
    """Returns ({(model_logical, variant, spec): {prefill,decode,echo}}, hardware_seen).

    Schema is `label|engine|model|spec|prompt|prefill|decode|pt|ct|hardware|notes`.
    Older CSVs (pre-Phase-B, no hardware column) are accepted as-is and tagged
    `unknown` for grouping. When `hardware_filter` is set, rows whose hardware
    tag does not match are dropped before aggregation.
    """
    data: dict = defaultdict(dict)
    hardware_seen: set[str] = set()
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 9 or parts[0] in ("label", ""):
                continue
            label, _engine, _model, _spec, prompt, pf, dc, _pt, _ct, *rest = parts
            # Phase B added a hardware column before the trailing notes column.
            # Old CSVs: rest = [notes]. New CSVs: rest = [hardware, notes].
            if len(rest) >= 2:
                hardware = rest[0] or "unknown"
            else:
                hardware = "unknown"
            hardware_seen.add(hardware)
            if hardware_filter and hardware != hardware_filter:
                continue
            bits = label.split("/")
            if len(bits) < 3:
                continue
            logical, variant, spec = bits[0], bits[1], bits[2]
            try:
                pf_v = float(pf or 0)
                dc_v = float(dc or 0)
            except ValueError:
                continue
            key = (logical, variant, spec)
            if prompt == "prefill":
                data[key]["prefill"] = pf_v
            elif prompt == "decode":
                data[key]["decode"] = dc_v
            elif prompt == "echo":
                data[key]["echo"] = dc_v
            elif prompt == "code":
                data[key]["code"] = dc_v
    return data, hardware_seen


def render(csv_path: Path, png_out: Path, family: str,
           hardware: str | None = None) -> None:
    if family not in FAMILIES:
        sys.exit(f"Unknown family '{family}'; pick one of: {', '.join(FAMILIES)}")
    cfg = FAMILIES[family]
    data, hardware_seen = load_csv(csv_path, hardware_filter=hardware)
    # Reject mixed-hardware CSVs without explicit --hardware: combining M1 Pro
    # and M4 Max numbers in one chart is exactly the bug Phase B fixed.
    real_hardware = {h for h in hardware_seen if h != "unknown"}
    if hardware is None and len(real_hardware) > 1:
        sys.exit(
            f"CSV contains {len(real_hardware)} hardware tags: "
            f"{sorted(real_hardware)}. Pick one with --hardware <tag>."
        )
    title_hw = hardware or (next(iter(real_hardware)) if real_hardware else None)
    title = cfg["title"]
    if title_hw:
        title = f"{title} · {title_hw}"

    # Style: clean grid, sans-serif, axis lines hidden except baseline.
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

    # Wider figure + extra room above for a top-of-figure legend.
    fig, axes = plt.subplots(1, 3, figsize=(28, 9.2))
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#111827", y=0.985)

    # Drop variants with no data anywhere in this CSV (e.g. oMLX not run, or
    # a drafter-less family) so absent engines don't leave empty bar slots
    # and dead legend entries.
    variants = [
        v for v in cfg["variants"]
        if any(data.get((m, v[0], v[1])) for m in cfg["model_order"])
    ]

    n_models = len(cfg["model_order"])
    n_variants = len(variants)
    # Spread model groups further apart so 7 bars per group breathe.
    group_step = 1.6
    x = np.arange(n_models) * group_step
    # Bar width: use ~85% of the per-group slot, divided evenly across variants.
    width = (group_step * 0.85) / n_variants

    panels = [
        ("echo",   "Echo (verbatim recitation — PLD's home turf)"),
        ("code",   "Code completion (drafter's intended turf)"),
        ("decode", "Free-form writing (creative essay, parity case)"),
    ]

    # Resolve baseline (variant, spec) once.
    base_variant_spec = next(
        ((var, sp) for (var, sp, _l, _c, base, _d, _s) in cfg["variants"] if base),
        None,
    )

    legend_handles = None
    for panel_idx, (workload_key, workload_label) in enumerate(panels):
        ax = axes[panel_idx]
        # Alternating very-faint background bands per model group make it
        # easier to see which bars belong together.
        for i in range(n_models):
            if i % 2 == 0:
                ax.axvspan(
                    x[i] - group_step / 2, x[i] + group_step / 2,
                    color="#f9fafb", zorder=0,
                )
        for v_idx, (variant, spec, label, color, is_baseline, show_delta, short) in enumerate(variants):
            values, baselines = [], []
            for logical in cfg["model_order"]:
                cell = data.get((logical, variant, spec), {})
                base_cell = (data.get((logical,) + base_variant_spec, {})
                             if base_variant_spec else {})
                values.append(cell.get(workload_key, 0))
                baselines.append(base_cell.get(workload_key, 0))
            offset = (v_idx - (n_variants - 1) / 2) * width
            bars = ax.bar(
                x + offset, values, width,
                label=label, color=color,
                edgecolor="#1f2937" if not is_baseline else "#6b7280",
                linewidth=0.5, zorder=2,
            )
            for bar, val, base in zip(bars, values, baselines):
                if val <= 0:
                    continue
                # Value (tok/s) and engine short-name stack INSIDE the bar
                # near the bottom. White text reads cleanly on saturated
                # colors; muted gray comparison bars get dark text.
                light_bar = is_baseline or color in ("#d1d5db", "#9ca3af")
                text_color = "#111827" if light_bar else "#ffffff"
                # Stack with the tok/s value above the engine short name.
                # Vertical offsets are picked to clear the bar baseline even
                # on the 31B tiny-bar case (~20 tok/s with y-range 0-150).
                value_str = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height() * 0.18, 1.2),
                    value_str,
                    ha="center", va="bottom",
                    fontsize=9, color=text_color, fontweight="bold",
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height() * 0.04, 0.3),
                    short,
                    ha="center", va="bottom",
                    fontsize=6.5, color=text_color, alpha=0.85,
                )
                # Percent delta: horizontal, ABOVE the bar. Only on mlx-serve
                # rows (`show_delta=True`) so the labels above don't collide.
                # Suppress noise-grade deltas (<3%) — bar height shows it.
                if show_delta and base > 0:
                    gain = (val / base - 1) * 100
                    if abs(gain) >= 3:
                        gcolor = ("#15803d" if gain >= 5 else
                                  "#b91c1c" if gain <= -5 else "#525252")
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (bar.get_height() * 0.015 + 0.4),
                            f"{gain:+.0f}%",
                            ha="center", va="bottom",
                            fontsize=9, color=gcolor, fontweight="bold",
                        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_xticks(x)
        ax.set_xticklabels([cfg["x_label"](m) for m in cfg["model_order"]],
                           fontsize=10, fontweight="medium")
        ax.set_ylabel("decode tok/s", fontsize=10)
        ax.set_title(workload_label, fontsize=12, fontweight="semibold", pad=8)
        ax.grid(True, axis="y", alpha=0.35, linestyle="--", color="#d1d5db", zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    # Single shared legend, positioned ~30px below the title (in 140-dpi
    # output, ~0.03 figure-fraction). Title sits at y=0.985, legend at 0.92.
    fig.legend(legend_handles, legend_labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.915),
               ncol=len(variants), fontsize=10,
               frameon=False, columnspacing=1.5, handlelength=1.6)

    # Headroom for the percent labels above bars.
    for ax in axes:
        cur_ylim = ax.get_ylim()
        ax.set_ylim(0, cur_ylim[1] * 1.18)
        ax.set_xlim(x[0] - group_step / 2, x[-1] + group_step / 2)

    # Leave room above for the title + legend stack (title at 0.985, legend
    # band around 0.915 → reserve top 12% of the figure).
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(png_out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {png_out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render MLX-serve vs LM Studio vs oMLX comparison chart from a "
                    "CSV produced by tests/bench.sh.",
    )
    p.add_argument("csv", type=Path, help="input CSV path")
    p.add_argument("png", type=Path, help="output PNG path")
    p.add_argument("--family", required=True, choices=list(FAMILIES.keys()),
                   help="model family (matches --family of tests/bench.sh)")
    p.add_argument("--hardware", default=None,
                   help="hardware tag to filter on (e.g. Apple-M1-Pro-32gb). "
                        "Required when the CSV mixes multiple machines.")
    args = p.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")
    render(args.csv, args.png, args.family, hardware=args.hardware)


if __name__ == "__main__":
    main()
