#!/usr/bin/env python3
"""plot_vs_lmstudio_omlx.py — render the MLX-serve vs LM Studio vs oMLX
comparison chart from a CSV produced by tests/bench_vs_lmstudio_omlx.sh.

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


# Family-specific layout. Each entry:
#   (variant_filter, label, color)
# variant_filter is matched against the second '/' segment in the CSV `label`
# column (e.g. "lmstudio-baseline", "lmstudio-alt", "omlx", "mlx-serve") AND
# the spec (third segment: "none", "pld", "drafter").
FAMILIES = {
    "gemma": {
        "title": "MLX-serve vs LM Studio vs oMLX — Gemma 4 (Apple Silicon, decode tok/s)",
        "x_label": lambda key: {
            "gemma4-e2b-4bit":          "E2B (4bit)",
            "gemma4-e4b-4bit":          "E4B (4bit)",
            "gemma4-31b-4bit":          "31B (4bit)",
            "gemma4-26b-a4b-moe-4bit":  "26B-A4B-MoE (4bit)",
        }.get(key, key),
        "model_order": [
            "gemma4-e2b-4bit",
            "gemma4-e4b-4bit",
            "gemma4-31b-4bit",
            "gemma4-26b-a4b-moe-4bit",
        ],
        # Order: LM Studio MLX baseline | LM Studio GGUF | mlx-serve GGUF
        # (head-to-head with the LMS GGUF row) | oMLX | mlx-serve MLX
        # {--no-pld, --pld, --drafter}.
        "variants": [
            ("lmstudio-baseline", "none",    "LM Studio (MLX, baseline)", "#888888", True),
            ("lmstudio-alt",      "none",    "LM Studio (GGUF)",          "#cccccc", False),
            ("mlx-serve-gguf",    "none",    "MLX-serve (GGUF, llama.cpp)", "#a855f7", False),
            ("omlx",              "none",    "oMLX",                      "#2ca02c", False),
            ("mlx-serve",         "none",    "MLX-serve --no-pld",        "#3b82f6", False),
            ("mlx-serve",         "pld",     "MLX-serve --pld",           "#22c55e", False),
            ("mlx-serve",         "drafter", "MLX-serve --drafter",       "#f97316", False),
        ],
    },
    "qwen36": {
        "title": "MLX-serve vs LM Studio vs oMLX — Qwen 3.6 (Apple Silicon, decode tok/s)",
        "x_label": lambda key: {
            "qwen36-27b":      "27B (4bit)",
            "qwen36-35b-a3b":  "35B-A3B (4bit)",
        }.get(key, key),
        "model_order": [
            "qwen36-27b",
            "qwen36-35b-a3b",
        ],
        "variants": [
            ("lmstudio-baseline", "none", "LM Studio (MLX, baseline)", "#888888", True),
            ("lmstudio-alt",      "none", "LM Studio (GGUF)",          "#cccccc", False),
            ("omlx",              "none", "oMLX",                      "#2ca02c", False),
            ("mlx-serve",         "none", "MLX-serve --no-pld",        "#3b82f6", False),
            ("mlx-serve",         "pld",  "MLX-serve --pld",           "#22c55e", False),
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
        title = f"{title} [{title_hw}]"

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    x = np.arange(len(cfg["model_order"]))
    n_variants = len(cfg["variants"])
    width = 0.8 / n_variants

    panels = [
        ("echo",   "Echo (verbatim recitation — PLD's home turf)"),
        ("code",   "Code completion (drafter's intended turf)"),
        ("decode", "Free-form writing (creative essay, parity case)"),
    ]

    for panel_idx, (workload_key, workload_label) in enumerate(panels):
        ax = axes[panel_idx]
        for v_idx, (variant, spec, label, color, is_baseline) in enumerate(cfg["variants"]):
            values, baselines = [], []
            for logical in cfg["model_order"]:
                cell = data.get((logical, variant, spec), {})
                # Find the row marked as baseline within the family, look up
                # its value for this prompt as the comparison point.
                base_variant_spec = next(
                    ((var, sp) for (var, sp, _l, _c, base) in cfg["variants"] if base),
                    None,
                )
                if base_variant_spec:
                    base_cell = data.get((logical,) + base_variant_spec, {})
                else:
                    base_cell = {}
                values.append(cell.get(workload_key, 0))
                baselines.append(base_cell.get(workload_key, 0))
            offset = (v_idx - (n_variants - 1) / 2) * width
            bars = ax.bar(
                x + offset, values, width,
                label=label, color=color, edgecolor="black", linewidth=0.6,
            )
            for bar, val, base in zip(bars, values, baselines):
                if val <= 0:
                    continue
                if is_baseline:
                    txt = f"{val:.1f}"
                    color_txt = "#222"
                else:
                    gain = (val / base - 1) * 100 if base > 0 else 0
                    color_txt = (
                        "#15803d" if gain >= 5 else
                        "#b91c1c" if gain <= -5 else
                        "#525252"
                    )
                    txt = f"{val:.1f}\n{gain:+.0f}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    txt,
                    ha="center", va="bottom", fontsize=8, color=color_txt,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([cfg["x_label"](m) for m in cfg["model_order"]], fontsize=9)
        ax.set_ylabel("decode tok/s", fontsize=10)
        ax.set_title(workload_label, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if panel_idx == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Headroom for the +X% labels above tall bars.
    for ax in axes:
        cur_ylim = ax.get_ylim()
        ax.set_ylim(0, cur_ylim[1] * 1.18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(png_out, dpi=140, bbox_inches="tight")
    print(f"Wrote {png_out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render MLX-serve vs LM Studio vs oMLX comparison chart from a "
                    "CSV produced by tests/bench_vs_lmstudio_omlx.sh.",
    )
    p.add_argument("csv", type=Path, help="input CSV path")
    p.add_argument("png", type=Path, help="output PNG path")
    p.add_argument("--family", required=True, choices=list(FAMILIES.keys()),
                   help="model family (matches --family of bench_vs_lmstudio_omlx.sh)")
    p.add_argument("--hardware", default=None,
                   help="hardware tag to filter on (e.g. Apple-M1-Pro-32gb). "
                        "Required when the CSV mixes multiple machines.")
    args = p.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")
    render(args.csv, args.png, args.family, hardware=args.hardware)


if __name__ == "__main__":
    main()
