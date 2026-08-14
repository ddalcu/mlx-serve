#!/usr/bin/env python3
"""plot_version_ab.py — the same engine, two builds, one chart.

Renders the CSV `tests/bench_versions.sh` folds out of llmprobe reports (via
bench_csv.py): the ENGINE is held fixed and the BUILD varies.

Two things shape the layout, both learned the hard way:

1. The win is a CONTEXT curve, not a number. Both builds are byte-identical at
   short context, so a headline bar chart would render the whole story as
   noise. The deepest common rung leads, and the ladder panels show the gap
   opening.

2. Decode and prefill are different wins on different rows. A sliding trim on a
   SERIAL arch (laguna, inkling) pays in prefill chunks and leaves decode flat;
   on a speculative arch (muse, gemma) it pays in the verify block and leaves
   prefill flat. Charting only one of them silently zeroes half the matrix.

Controls are drawn in grey below a rule. They are supposed to be flat — a
control that moves is the finding, so it must be visible, not hidden.

Usage: python3 tests/plot_version_ab.py <csv> <out.png>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BASE_ARM, NEW_ARM = "shipped", "dev"

# Row order is the story order: sliding archs (where the trim lives) first,
# controls last. A model absent from the CSV is dropped, never zero-filled.
ORDER = [
    ("muse-30b-4bit",        "Muse-Glimmer 30B",   "4-bit · sw 2048 · DFlash", False),
    ("gemma4-26b-a4b-4bit",  "Gemma 4 26B-A4B",    "4-bit · sw 1024 · PLD",    False),
    ("gemma4-e4b-4bit",      "Gemma 4 E4B",        "4-bit · sw 512 · drafter", False),
    ("laguna-xs-nvfp4",      "Laguna XS 2.1",      "NVFP4 · sw 512 · serial",  False),
    ("inkling-small-2bit",   "Inkling Small",      "2-bit · sw 512 · serial",  False),
    ("qwen36-27b-4bit",      "Qwen 3.6 27B",       "4-bit · no sliding · MTP", True),
    ("qwen36-35b-a3b-oq4",   "Qwen 3.6 35B-A3B",   "oQ4 · no sliding · MTP",   True),
    ("lfm2-2.6b-nvfp4",      "LFM2.5 2.6B",        "NVFP4 · no sliding",       True),
]
RUNG_ORDER = ["0.5k", "1k", "2k", "4k", "8k", "16k", "32k", "64k"]

C_DECODE, C_PREFILL = "#2563eb", "#0d9488"
C_CTRL, C_GRID, C_ZERO = "#9ca3af", "#e5e7eb", "#6b7280"


def read_csv(path):
    """-> {(model, engine, context): {metric: float}}, plus the header notes."""
    cells, notes = {}, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            notes.append(line.lstrip("# ").strip())
            continue
        f = line.split("|")
        if len(f) < 7 or f[0] == "model":
            continue
        model, engine, _spec, ctx = f[0], f[1], f[2], f[3]

        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        cells[(model, engine, ctx)] = {"prefill": num(f[4]), "decode": num(f[5])}
    return cells, notes


def pair(cells, model, ctx, metric):
    """(shipped, dev) at one rung, or None unless BOTH arms measured it — a
    missing arm is a gap, and a gap rendered as 0% reads as 'no change'."""
    b = cells.get((model, BASE_ARM, ctx), {}).get(metric)
    n = cells.get((model, NEW_ARM, ctx), {}).get(metric)
    if not b or not n or b <= 0:
        return None
    return (b, n)


def pct(cells, model, ctx, metric):
    """dev vs shipped at one rung, in percent."""
    p = pair(cells, model, ctx, metric)
    return None if p is None else (p[1] / p[0] - 1.0) * 100.0


def fmt_tps(v):
    """Absolute rates, at the precision the number deserves: decode lands in
    the tens, prefill in the thousands."""
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def deepest_common(cells, model):
    """The deepest rung both arms measured — where the trim has the most to
    show. Falls back to nothing rather than to `headline`, which by design
    cannot separate these builds."""
    for rung in reversed(RUNG_ORDER):
        if (model, BASE_ARM, rung) in cells and (model, NEW_ARM, rung) in cells:
            return rung
    return None


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-1])
    csv_path, out_path = sys.argv[1], sys.argv[2]
    cells, notes = read_csv(csv_path)

    rows = []
    for key, label, sub, is_ctrl in ORDER:
        rung = deepest_common(cells, key)
        if rung is None:
            continue
        rows.append({
            "key": key, "label": label, "sub": sub, "ctrl": is_ctrl, "rung": rung,
            "decode": pct(cells, key, rung, "decode"),
            "prefill": pct(cells, key, rung, "prefill"),
            "decode_ab": pair(cells, key, rung, "decode"),
            "prefill_ab": pair(cells, key, rung, "prefill"),
        })
    if not rows:
        sys.exit("nothing to plot: no model has both arms at a common rung")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": "#374151",
        "xtick.color": "#374151",
        "ytick.color": "#6b7280",
        "axes.titlecolor": "#111827",
    })
    fig = plt.figure(figsize=(17.0, 4.2 + 0.62 * len(rows)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.52, wspace=0.16,
                          left=0.20, right=0.975, top=0.82, bottom=0.135)
    ax = fig.add_subplot(gs[0, :])

    fig.suptitle("mlx-serve — shipped build vs current tree",
                 fontsize=16, fontweight="bold", color="#111827", y=0.982)
    fig.text(0.5, 0.912,
             "Same engine, same models, same flags — only the binary changes. "
             "Speedup at the deepest measured context.",
             ha="center", fontsize=9.5, color="#4b5563")

    # ── Panel A: delta at the deepest rung ──
    ys = list(range(len(rows)))[::-1]
    h = 0.34
    vals = [r[m] for r in rows for m in ("decode", "prefill") if r[m] is not None]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    span = max(hi - lo, 1.0)
    # Headroom on the side the labels sit, and always include 0 — a chart of
    # deltas that crops the zero line misreads at a glance.
    ax.set_xlim(lo - span * 0.46, hi + span * 0.60)
    ax.set_ylim(-0.72, len(rows) - 0.28)
    pad = span * 0.018

    for y, r in zip(ys, rows):
        for off, metric, colour in ((h / 2, "decode", C_DECODE), (-h / 2, "prefill", C_PREFILL)):
            v = r[metric]
            if v is None:
                continue
            c = C_CTRL if r["ctrl"] else colour
            ax.barh(y + off, v, height=h, color=c, edgecolor="#1f2937",
                    linewidth=0.4, zorder=3)
            # Label outside the bar tip, on the side the bar points. The
            # absolute pair rides along: a percentage with no rate behind it is
            # not a performance claim, it is a ratio.
            ab = r[metric + "_ab"]
            txt = f"{v:+.1f}%"
            if ab:
                txt = f"{fmt_tps(ab[0])} -> {fmt_tps(ab[1])} tok/s   {v:+.1f}%"
            ax.text(v + (pad if v >= 0 else -pad), y + off, txt, va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=8.2, fontweight="bold",
                    color="#111827" if not r["ctrl"] else "#6b7280", zorder=4)

    ax.axvline(0, color=C_ZERO, linewidth=1.1, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']}\n{r['sub']}  ·  @{r['rung']}" for r in rows],
                       fontsize=9, linespacing=1.5)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color("#6b7280" if r["ctrl"] else "#111827")
    ax.set_xlabel("faster than the shipped build  (%)", fontsize=9.5)
    ax.grid(axis="x", color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # Rule between the sliding archs and the controls, so a reader never has to
    # infer which rows are supposed to move.
    ctrl_ys = [y for y, r in zip(ys, rows) if r["ctrl"]]
    if ctrl_ys and len(ctrl_ys) < len(rows):
        ax.axhline(max(ctrl_ys) + 0.5, color="#cbd5e1", linewidth=1, linestyle=(0, (4, 3)))
        ax.text(0.995, (max(ctrl_ys) + 0.62 + 0.72) / (len(rows) + 0.44),
                "controls — no sliding layers, expected flat",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, style="italic", color="#9ca3af")

    ax.legend(handles=[
        Line2D([], [], marker="s", linestyle="", markersize=9, color=C_DECODE, label="decode"),
        Line2D([], [], marker="s", linestyle="", markersize=9, color=C_PREFILL, label="prefill"),
        Line2D([], [], marker="s", linestyle="", markersize=9, color=C_CTRL, label="control"),
    ], loc="lower right", bbox_to_anchor=(1.0, 1.005), frameon=False,
        fontsize=9, ncol=3)

    # ── Panels B/C: the delta as a function of context ──
    for col, (metric, colour, title) in enumerate((
            ("decode", C_DECODE, "Decode speedup vs context"),
            ("prefill", C_PREFILL, "Prefill speedup vs context"))):
        axl = fig.add_subplot(gs[1, col])
        rungs = [r for r in RUNG_ORDER
                 if any((r_["key"], BASE_ARM, r) in cells for r_ in rows)]
        xs = list(range(len(rungs)))
        drew = False
        for r in rows:
            series = [pct(cells, r["key"], rung, metric) for rung in rungs]
            if all(v is None for v in series):
                continue
            pts = [(x, v) for x, v in zip(xs, series) if v is not None]
            if len(pts) < 2:
                continue
            drew = True
            axl.plot([p[0] for p in pts], [p[1] for p in pts],
                     marker="o", markersize=3.4, linewidth=1.9 if not r["ctrl"] else 1.1,
                     color=C_CTRL if r["ctrl"] else None,
                     alpha=0.55 if r["ctrl"] else 1.0,
                     linestyle="--" if r["ctrl"] else "-",
                     label=r["label"], zorder=3 if not r["ctrl"] else 2)
        axl.axhline(0, color=C_ZERO, linewidth=1.0, zorder=1)
        axl.set_xticks(xs)
        axl.set_xticklabels(rungs, fontsize=8.5)
        axl.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
        axl.set_xlabel("prompt context", fontsize=9)
        if col == 0:
            axl.set_ylabel("faster than shipped (%)", fontsize=9)
        axl.grid(color=C_GRID, linewidth=0.7, zorder=0)
        axl.set_axisbelow(True)
        for s in ("top", "right"):
            axl.spines[s].set_visible(False)
        if drew:
            axl.legend(frameon=False, fontsize=7.4, loc="upper left", ncol=2)

    footer = "  ·  ".join(n for n in notes[:2] if n)
    if footer:
        fig.text(0.5, 0.022, footer, ha="center", fontsize=7.4, color="#9ca3af")

    fig.savefig(out_path, dpi=170, facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
