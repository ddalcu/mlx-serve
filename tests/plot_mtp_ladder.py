#!/usr/bin/env python3
"""plot_mtp_ladder.py — render the context ladder for one model from the probe
CSV `tests/bench.sh` writes.

The ladder comes out of the SAME llmprobe run as the headline chart (its
`contextScaling` block), so there is no second protocol to keep in sync: one
boot per engine, a discarded warmup, and one rung per prompt size (median of
three per rung under `--full`, which also climbs to 32k/64k).

llmprobe's ladder measures agent work — a synthetic TypeScript codebase and a
task that must use a constant planted mid-corpus — so a rung whose answer never
references the constant is reported as such rather than counted. That matters
here: timing generation with a large irrelevant prefix attached is not
long-context work, and an echo-shaped ladder silently collects a speculation
boost at every rung.

Each engine is booted with its shipping defaults, so the lanes answer "how does
this engine hold up as the KV cache grows", not "how does this flag combination
score".

Usage:
  python3 tests/plot_mtp_ladder.py docs/perf-csvs/probe-<tag>.csv \
      docs/perf-pngs/perf-mtp-ladder-<tag>.png --model qwen36-27b

Requires matplotlib; style matches tests/plot_vs_lmstudio_omlx.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from bench_engines import label_with_version, parse_engine_versions  # noqa: E402
import numpy as np

SUBTITLE = ("Context ladder from the same llmprobe run as the headline chart · "
            "one boot per engine on shipping defaults, discarded warmup, "
            "agent-shaped coding task per rung")

# Same gray ramp as the headline chart: comparison engines muted, ours the one
# saturated lane. Keys are the CSV's `engine` column.
ENGINES = [
    ("lmstudio-alt",      "LM Studio (GGUF)", "#d1d5db", "#6b7280", True),
    ("lmstudio-baseline", "LM Studio (MLX)",  "#9ca3af", "#6b7280", True),
    ("omlx",              "oMLX",             "#6b7280", "#1f2937", False),
    ("mtplx",             "MTPLX",            "#4b5563", "#1f2937", False),
    ("mlx-serve",         "MLX-serve",        "#2563eb", "#1f2937", False),
]

DEFAULT_TITLE = "Context ladder — prefill and decode"


def ladder_span_label(rows) -> str:
    """`"0.5k to 16k"` from the rungs actually measured.

    The title used to hardcode "0.5K to 64K" while a default-depth run stops at
    16k — a chart claiming a reach its data does not have, same class as naming
    an engine that never ran. Derived, so it cannot drift from the CSV.
    """
    ctxs = [r.get("context") for r in rows if r.get("context")]
    if not ctxs:
        return ""
    return f"{ctxs[0]} to {ctxs[-1]}" if len(ctxs) > 1 else str(ctxs[0])
# Percent-delta annotation: (engine whose bar gets the label, engine it is
# compared against).
DEFAULT_DELTA = ("mlx-serve", "omlx")


def parse_engines(spec: str) -> list[tuple]:
    """`key:Label:#color[:light]` comma-separated → ENGINES-shaped tuples.
    `light=1` renders the in-bar value dark-on-light (pale bar colors)."""
    out = []
    for item in spec.split(","):
        parts = item.split(":")
        if len(parts) < 3:
            sys.exit(f"--engines spec needs key:Label:#color[:light], got: {item}")
        key, label, color = parts[0], parts[1], parts[2]
        light = len(parts) > 3 and parts[3] == "1"
        out.append((key, label, color, "#6b7280" if light else "#1f2937", light))
    return out


def load_csv(path: Path, model: str | None = None) -> tuple[list[dict], set[str], str, dict]:
    """Probe CSV → ([{context, <engine>_prefill, <engine>_decode}], engines, note).

    Ladder rows only (the `headline` rows belong to the other chart). Rungs keep
    the order they were measured in — ascending prompt size, and a rung the
    engine rejected ends the ladder, so a short lane means "not attempted", not
    "zero".
    """
    by_rung: dict[str, dict] = {}
    order: list[str] = []
    engines: set[str] = set()
    models_seen: set[str] = set()
    note = ""
    with open(path) as f:
        engine_versions = parse_engine_versions(f)
        f.seek(0)
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                note = note or line.lstrip("# ").strip()
                continue
            parts = line.split("|")
            if len(parts) < 12 or parts[0] in ("model", ""):
                continue
            row_model, engine, _spec, context, prefill, decode = parts[:6]
            if context == "headline":
                continue
            models_seen.add(row_model)
            if model and row_model != model:
                continue
            if context not in by_rung:
                by_rung[context] = {"context": context}
                order.append(context)
            if prefill or decode:
                engines.add(engine)
            if prefill:
                by_rung[context][f"{engine}_prefill"] = prefill
            if decode:
                by_rung[context][f"{engine}_decode"] = decode

    if not by_rung:
        hint = f" (models in CSV: {sorted(models_seen)})" if models_seen else ""
        sys.exit(f"No ladder rows for model '{model}' in {path}{hint}")
    # A ladder is per model. Without --model on a multi-model CSV the rungs of
    # six different checkpoints land in one lane, last row winning per rung —
    # a chart that renders perfectly and means nothing.
    if model is None and len(models_seen) > 1:
        sys.exit(f"{path} holds {len(models_seen)} models: {sorted(models_seen)}. "
                 f"Pick one with --model <name>.")
    return [by_rung[c] for c in order], engines, note, engine_versions


def render(csv_path: Path, png_out: Path, engines: list[tuple] = ENGINES,
           title: str = DEFAULT_TITLE, subtitle: str | None = None,
           delta: tuple = DEFAULT_DELTA, model: str | None = None) -> None:
    rows, present, note, engine_versions = load_csv(csv_path, model=model)
    span = ladder_span_label(rows)
    if span and "to" not in title:
        title = f"{title} ({span})"

    # Name the BUILD each lane came from — see tests/bench_engines.py.
    engines = [(key, label_with_version(label, key, engine_versions), *rest)
               for (key, label, *rest) in engines]
    contexts = [r["context"] for r in rows]
    # Drop lanes the CSV has nothing for, so a partial run renders cleanly
    # instead of laying a flat zero bar over every rung.
    engines = [e for e in engines if e[0] in present]
    if not engines:
        sys.exit(f"None of the known engine lanes are in {csv_path}: found {sorted(present)}")
    if subtitle is None:
        subtitle = f"{model} · {note}" if (model and note) else (note or SUBTITLE)

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
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#111827", y=0.99)
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=9.5, color="#4b5563")

    panels = [
        ("prefill", "Prefill (tok/s)", "prefill tok/s"),
        ("decode", "Decode (tok/s)", "decode tok/s"),
    ]

    x = np.arange(len(contexts))
    width = 0.81 / len(engines)
    for ax, (key, panel_title, ylab) in zip(axes, panels):
        # A rung a lane never reached is 0 here, and the bar-label loop below
        # skips it — an absent rung must not be drawn as a measured floor.
        series = {eng: [float(r.get(f"{eng}_{key}") or 0) for r in rows]
                  for eng, *_ in engines}
        top = max(v for vals in series.values() for v in vals)
        for e_idx, (eng, label, color, edge, light) in enumerate(engines):
            vals = series[eng]
            offset = (e_idx - (len(engines) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=color,
                          edgecolor=edge, linewidth=0.5, zorder=2)
            for bar, val in zip(bars, vals):
                if val <= 0:
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 0.5, f"{val:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="#111827" if light else "#ffffff",
                        fontweight="bold", rotation=90)
            # Percent delta above the `delta[0]` bar vs `delta[1]` at the same
            # rung — the head-to-head race the chart is about.
            if eng == delta[0] and delta[1] in series:
                for bar, val, base in zip(bars, vals, series[delta[1]]):
                    if base <= 0 or val <= 0:
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
        description="Render the context-ladder chart for one model from the probe "
                    "CSV produced by tests/bench.sh.")
    p.add_argument("csv", type=Path, help="input CSV path")
    p.add_argument("png", type=Path, help="output PNG path")
    p.add_argument("--model", default=None,
                   help="which model's ladder to plot (required when the CSV holds several)")
    p.add_argument("--engines", help="key:Label:#color[:light],… — engine lanes to plot")
    p.add_argument("--title", default=None)
    p.add_argument("--subtitle", default=None,
                   help="methodology line; defaults to the CSV's own '#' header")
    p.add_argument("--delta", default=":".join(DEFAULT_DELTA),
                   help="annotated:baseline engine keys for the percent labels")
    args = p.parse_args()
    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")
    engines = parse_engines(args.engines) if args.engines else ENGINES
    title = args.title or (f"{DEFAULT_TITLE} — {args.model}" if args.model else DEFAULT_TITLE)
    render(args.csv, args.png, engines=engines, title=title,
           subtitle=args.subtitle, delta=tuple(args.delta.split(":", 1)),
           model=args.model)


if __name__ == "__main__":
    main()
