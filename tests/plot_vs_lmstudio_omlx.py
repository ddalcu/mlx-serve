#!/usr/bin/env python3
"""plot_vs_lmstudio_omlx.py — render the MLX-serve vs LM Studio vs oMLX
comparison chart from a CSV produced by tests/bench.sh.

The canonical `all` chart is a SINGLE code-completion panel (2026-07-14: the
echo panel was retired with the bench's echo cell — verbatim recitation is
spec-decode's synthetic best case; code completion is the honest workload).
Its baseline is LM Studio GGUF (what LM Studio users actually run), with
LM Studio MLX kept as a second comparison bar, and the four mlx-serve spec
cells COLLAPSED into one "best config" bar per model (the winning spec is
named in the delta label). The legacy families keep the old two-panel
echo+code layout for old CSVs.

Families:
  all     → THE canonical chart: Gemma 4 + Qwen 3.6, one code panel.
  gemma   → legacy per-family layout (side-by-side echo+code panels)
  qwen36  → legacy per-family layout

Usage:
  python3 tests/plot_vs_lmstudio_omlx.py <csv> <png_out> --family <all|gemma|qwen36>

The `all` CSV is produced by `tests/bench.sh --family all` (or by
concatenating a gemma and a qwen36 CSV). Engines with no rows in the CSV are
dropped automatically, so partial runs render cleanly.

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
    "all": {
        "title": "MLX-serve vs LM Studio · oMLX · MTPLX — Gemma 4 + Qwen 3.6 (Apple Silicon, decode tok/s)",
        # Panels stack vertically (2 rows x 1 col) so every model group gets
        # the full figure width — bars are wide enough to carry their engine
        # short-label legibly.
        "stacked": True,
        "x_label": lambda key: {
            "gemma4-e4b-4bit":              "Gemma 4 E4B (4bit)",
            "gemma4-31b-4bit":              "Gemma 4 31B (4bit)",
            "gemma4-26b-a4b-moe-qat-4bit":  "Gemma 4 26B-A4B MoE (QAT 4bit)",
            "qwen36-27b":                   "Qwen 3.6 27B (4bit)",
            "qwen36-35b-a3b":               "Qwen 3.6 35B-A3B MoE (4bit)",
            "qwen36-27b-mtplxopt":          "Qwen 3.6 27B (MTPLX-opt 4bit)",
        }.get(key, key),
        "model_order": [
            "gemma4-e4b-4bit",
            "gemma4-31b-4bit",
            "gemma4-26b-a4b-moe-qat-4bit",
            "qwen36-27b",
            "qwen36-35b-a3b",
            "qwen36-27b-mtplxopt",
        ],
        # Two stacked family rows instead of one super-wide strip: Gemma on
        # top, Qwen below. Each model group gets a bordered box so the last
        # bar of one group can't read as the first bar of the next.
        "model_rows": [
            {"title": "Gemma 4", "models": [
                "gemma4-e4b-4bit",
                "gemma4-31b-4bit",
                "gemma4-26b-a4b-moe-qat-4bit",
            ]},
            {"title": "Qwen 3.6", "models": [
                "qwen36-27b",
                "qwen36-35b-a3b",
                "qwen36-27b-mtplxopt",
            ]},
        ],
        # One code panel only (echo retired 2026-07-14 with the bench cell).
        "panels": [
            ("code", "Code completion (decode tok/s)"),
        ],
        # Engine set. Baseline = LM Studio GGUF (the llama.cpp path LM Studio
        # users actually run); LM Studio MLX stays as a comparison bar. The
        # four mlx-serve spec cells collapse into ONE "best config" bar per
        # model (spec "best" = max over none/pld/drafter/mtp; the winning
        # spec is named in the delta label). The layout PACKS each model
        # group — absent cells take no space; only MTPLX keeps a labeled
        # zero-slot. The mtplxopt row has no GGUF counterpart, so its deltas
        # fall back to the LM Studio MLX bar (see baseline_fallback).
        "variants": [
            # Competitors ride ONE neutral gray ramp (light → dark, in the order
            # they get harder to beat: LM-GGUF → LM-MLX → oMLX → MTPLX) so the
            # only saturated bar in the chart is ours. Tailwind gray 300/400/500/600.
            ("lmstudio-alt",      "none", "LM Studio (GGUF, baseline)", "#d1d5db", True,  False, "LM-GGUF"),
            ("lmstudio-baseline", "none", "LM Studio (MLX)",            "#9ca3af", False, False, "LM-MLX"),
            ("omlx",              "none", "oMLX",                       "#6b7280", False, False, "oMLX"),
            ("mtplx",             "auto", "MTPLX",                      "#4b5563", False, False, "MTPLX"),
            ("mlx-serve",         "best", "MLX-serve (best config)",    "#2563eb", False, True,  "MLX-Serve"),
        ],
        "baseline_fallback": [("lmstudio-baseline", "none")],
    },
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
            ("omlx",              "none",    "oMLX",                        "#6b7280", False, False, "oMLX"),
            ("mtplx",             "auto",    "MTPLX (auto)",                "#4b5563", False, False, "MTPLX"),
            ("mlx-serve",         "none",    "MLX-serve (MLX, --no-pld)",   "#2563eb", False, True,  "MLXS-NPLD"),
            ("mlx-serve",         "pld",     "MLX-serve (MLX, --pld)",      "#16a34a", False, True,  "MLXS-PLD"),
            ("mlx-serve",         "drafter", "MLX-serve (MLX, --drafter)",  "#ea580c", False, True,  "MLXS-DRFT"),
        ],
    },
    "qwen36": {
        "title": "MLX-serve vs LM Studio — Qwen 3.6 (Apple Silicon, decode tok/s)",
        "x_label": lambda key: {
            "qwen36-27b":          "27B (4bit)",
            "qwen36-35b-a3b":      "35B-A3B (4bit)",
            "qwen36-27b-mtplxopt": "27B (MTPLX-opt 4bit)",
        }.get(key, key),
        "model_order": [
            "qwen36-27b",
            "qwen36-35b-a3b",
            "qwen36-27b-mtplxopt",
        ],
        "variants": [
            ("lmstudio-baseline", "none", "LM Studio (MLX, baseline)",   "#9ca3af", True,  False, "LM-MLX"),
            ("lmstudio-alt",      "none", "LM Studio (GGUF)",            "#d1d5db", False, False, "LM-GG"),
            ("mlx-serve-gguf",    "none", "MLX-serve (GGUF / llama.cpp)", "#a78bfa", False, False, "MLXS-GG"),
            ("omlx",              "none", "oMLX",                        "#6b7280", False, False, "oMLX"),
            ("mtplx",             "auto", "MTPLX (auto)",                "#4b5563", False, False, "MTPLX"),
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

    # Panels are family-driven: the canonical "all" chart is a single code
    # panel; the legacy families keep the old echo+code pair for old CSVs.
    panels = cfg.get("panels", [
        ("echo",   "Echo (verbatim recitation — prompt-lookup's home turf)"),
        ("code",   "Code completion (spec-decode's real workload)"),
    ])

    # Axis plan. With "model_rows" the (single) workload panel is split
    # across stacked axes — one per model-family row (Gemma on top, Qwen
    # below) — so the chart is tall rather than super-wide. Without it,
    # one axis per panel (legacy echo+code layouts).
    stacked = bool(cfg.get("stacked", False) or cfg.get("model_rows"))
    if cfg.get("model_rows"):
        wk, wl = panels[0]
        axis_specs = [(wk, f'{row["title"]} — {wl}', row["models"])
                      for row in cfg["model_rows"]]
        fig, axes = plt.subplots(len(axis_specs), 1,
                                 figsize=(14.5, 5.3 * len(axis_specs)))
    else:
        axis_specs = [(wk, wl, cfg["model_order"]) for (wk, wl) in panels]
        if stacked:
            fig, axes = plt.subplots(len(axis_specs), 1,
                                     figsize=(24, 6.8 * len(axis_specs)))
        else:
            fig, axes = plt.subplots(1, len(axis_specs), figsize=(22, 9.2))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#111827", y=0.98)

    # mlx-serve spec cells a spec="best" variant collapses over, and the
    # display name the delta label uses for the winning spec.
    BEST_SPECS = {"none": "AR", "pld": "PLD", "drafter": "drafter", "mtp": "MTP"}
    SPEC_DISPLAY_TO_KEY = {disp: s for s, disp in BEST_SPECS.items()}

    def cell_val(logical: str, variant: str, spec: str, key: str):
        """(value, winning_spec_display) for one cell; spec 'best' takes the
        max over the mlx-serve spec cells and names the winner."""
        if spec == "best":
            best_v, best_s = 0.0, None
            for s, disp in BEST_SPECS.items():
                v = data.get((logical, variant, s), {}).get(key, 0)
                if v > best_v:
                    best_v, best_s = v, disp
            return best_v, best_s
        return data.get((logical, variant, spec), {}).get(key, 0), None

    def cell_has_data(logical: str, v: tuple) -> bool:
        if v[1] == "best":
            return any(data.get((logical, v[0], s)) for s in BEST_SPECS)
        return bool(data.get((logical, v[0], v[1])))

    # Drop variants with no data anywhere in this CSV (e.g. oMLX not run, or
    # a drafter-less family) so absent engines don't leave empty bar slots
    # and dead legend entries.
    variants = [
        v for v in cfg["variants"]
        if any(cell_has_data(m, v) for m in cfg["model_order"])
    ]

    # Spread model groups further apart so the bars per group breathe.
    group_step = 1.6

    def slot_present(logical: str, v: tuple) -> bool:
        # MTPLX keeps a labeled zero-slot on rows it can't run (its absence
        # is a result, not a layout artifact); every OTHER absent
        # (variant, spec) cell is packed away entirely — no empty gaps where
        # the other family's spec slot would sit.
        if v[0] == "mtplx":
            return True
        return cell_has_data(logical, v)

    # Packed per-group layout. Each model group lays out only the variants
    # that actually ran, split into two sub-clusters — comparison engines |
    # MLX-serve — with a small gap between them and a tinted band behind the
    # MLX-serve cluster so the family reads as one unit.
    CLUSTER_GAP = 0.55   # gap between the two sub-clusters, in bar-widths
    BAND_PAD = 0.18      # band padding around the mlx-serve cluster, in bar-widths

    def group_slots(logical: str) -> tuple[list[int], float]:
        present = [vi for vi, v in enumerate(variants) if slot_present(logical, v)]
        has_split = (any(variants[vi][0] != "mlx-serve" for vi in present)
                     and any(variants[vi][0] == "mlx-serve" for vi in present))
        return present, len(present) + (CLUSTER_GAP if has_split else 0.0)

    # Bar width is FIXED across every axis and group (sized by the fullest
    # group anywhere); sparser groups just get breathing room, centered.
    width = (group_step * 0.92) / max(
        group_slots(m)[1] for m in cfg["model_order"])

    def compute_layout(models: list[str]):
        """Per-axis packed geometry: x centers, per-group slot lists, bar
        positions, and the mlx-serve cluster extent per group."""
        x = np.arange(len(models)) * group_step
        layout: list[list[int]] = []
        positions: dict[tuple[int, int], float] = {}
        mlx_spans: list[tuple[float, float] | None] = []
        for mi, logical in enumerate(models):
            present, eff = group_slots(logical)
            layout.append(present)
            cur = x[mi] - (eff * width) / 2
            span = None
            gap_spent = False
            for vi in present:
                if (variants[vi][0] == "mlx-serve" and not gap_spent
                        and eff > len(present)):
                    cur += CLUSTER_GAP * width
                    gap_spent = True
                center = cur + width / 2
                positions[(mi, vi)] = center
                if variants[vi][0] == "mlx-serve":
                    lo = center - width / 2 - BAND_PAD * width
                    hi = center + width / 2 + BAND_PAD * width
                    span = (min(span[0], lo), max(span[1], hi)) if span else (lo, hi)
                cur += width
            mlx_spans.append(span)
        return x, layout, positions, mlx_spans

    # Baseline resolution: the marked baseline (variant, spec) first, then
    # the cfg's fallback chain — e.g. the mtplxopt row has no GGUF build, so
    # its delta compares against the LM Studio MLX bar instead of vanishing.
    baseline_chain = [
        (var, sp) for (var, sp, _l, _c, base, _d, _s) in cfg["variants"] if base
    ] + list(cfg.get("baseline_fallback", []))

    def baseline_val(logical: str, key: str) -> float:
        for var, sp in baseline_chain:
            v = data.get((logical, var, sp), {}).get(key, 0)
            if v > 0:
                return v
        return 0.0

    legend_handles = None
    for ax_idx, (workload_key, ax_title, models) in enumerate(axis_specs):
        ax = axes[ax_idx]
        x, group_layout, positions, mlx_spans = compute_layout(models)
        # Fixed label geometry: the tok/s value and the engine short-name sit
        # at the SAME two heights on every bar (fractions of the panel's
        # y-range), independent of bar height — uniform, never squished.
        panel_max = max(
            (cell_val(m, v[0], v[1], workload_key)[0]
             for v in variants for m in models),
            default=0,
        )
        ymax = panel_max * 1.18 if panel_max > 0 else 1.0
        short_y = ymax * 0.015
        value_y = ymax * 0.055
        inside_min = ymax * 0.11  # bar must reach here for in-bar white text
        ax.set_ylim(0, ymax)
        ax.set_xlim(x[0] - group_step * 0.55, x[-1] + group_step * 0.55)
        # A bordered box around each model group keeps neighbours visually
        # separate — the last bar of one group can't read as the first bar
        # of the next benchmark.
        for gx in x:
            ax.add_patch(plt.Rectangle(
                (gx - group_step * 0.48, 0), group_step * 0.96, ymax,
                facecolor="none", edgecolor="#cbd5e1", linewidth=1.1,
                zorder=1.6,
            ))
        # Tinted band behind each MLX-serve sub-cluster: the spec variants
        # read as ONE engine family (the bars all say "MLX-Serve"; the
        # legend names the winning-config semantics).
        for span in mlx_spans:
            if span:
                ax.axvspan(span[0], span[1], color="#dbeafe",
                           alpha=0.55, zorder=0.6)
        # Declutter: the "+X%"-vs-baseline label goes only on the BEST
        # mlx-serve bar of each model group (per panel) — one headline
        # number per group.
        best_delta: dict[int, int] = {}
        for mi, logical in enumerate(models):
            best_vi, best_val = None, 0.0
            for vi in group_layout[mi]:
                v = variants[vi]
                if v[0] != "mlx-serve" or not v[5]:
                    continue
                val = cell_val(logical, v[0], v[1], workload_key)[0]
                if val > best_val:
                    best_vi, best_val = vi, val
            if best_vi is not None:
                best_delta[mi] = best_vi
        for v_idx, (variant, spec, label, color, is_baseline, show_delta, short) in enumerate(variants):
            xs, values, metas = [], [], []
            for mi, logical in enumerate(models):
                if (mi, v_idx) not in positions:
                    continue
                val, win = cell_val(logical, variant, spec, workload_key)
                xs.append(positions[(mi, v_idx)])
                values.append(val)
                metas.append((mi, baseline_val(logical, workload_key), win))
            if not xs:
                continue
            bars = ax.bar(
                xs, values, width,
                label=label, color=color,
                edgecolor="#1f2937" if not is_baseline else "#6b7280",
                linewidth=0.5, zorder=2,
            )
            for bar, val, (mi, base, win) in zip(bars, values, metas):
                cx = bar.get_x() + bar.get_width() / 2
                if val <= 0:
                    # MTPLX slots stay labeled with an explicit 0 where the
                    # engine couldn't run (non-MTPLX artifacts) so its absence
                    # reads as "measured: can't", not a rendering gap.
                    if variant == "mtplx":
                        ax.text(cx, value_y, "0", ha="center", va="bottom",
                                fontsize=10 if stacked else 9,
                                color="#6b7280", fontweight="bold")
                        ax.text(cx, short_y, short, ha="center", va="bottom",
                                fontsize=8 if stacked else 6.5,
                                color="#6b7280", alpha=0.9)
                    continue
                # White text reads cleanly inside saturated bars; bars too
                # short to contain the fixed-position labels (and the muted
                # gray comparison bars) get dark text instead.
                light_bar = is_baseline or color in ("#d1d5db", "#9ca3af")
                inside = bar.get_height() >= inside_min
                text_color = "#111827" if (light_bar or not inside) else "#ffffff"
                value_str = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
                ax.text(cx, value_y, value_str, ha="center", va="bottom",
                        fontsize=10 if stacked else 9,
                        color=text_color, fontweight="bold")
                ax.text(cx, short_y, short, ha="center", va="bottom",
                        fontsize=8 if stacked else 6.5,
                        color=text_color, alpha=0.9)
                # Percent delta: horizontal, ABOVE the bar — only on the best
                # mlx-serve bar per group, naming the winning spec when the
                # bar is a "best config" collapse. Suppress noise (<3%).
                if show_delta and base > 0 and best_delta.get(mi) == v_idx:
                    gain = (val / base - 1) * 100
                    if abs(gain) >= 3:
                        gcolor = ("#15803d" if gain >= 5 else
                                  "#b91c1c" if gain <= -5 else "#525252")
                        dlabel = f"{gain:+.0f}%"
                        if win:
                            dlabel += f" · {win}"
                        ax.text(
                            cx,
                            bar.get_height() + ymax * 0.012,
                            dlabel,
                            ha="center", va="bottom",
                            fontsize=9, color=gcolor, fontweight="bold",
                        )
        # Embedded top-left prefill mini-chart per model group: a small
        # horizontal reference chart (own x-scale, own group's engines only)
        # instead of a second bar sharing the decode axis — prefill tok/s is
        # 5-15x decode tok/s, so a shared axis would squash the decode bars.
        # Pinned to a strip above panel_max (no real bar can reach it, since
        # ymax = panel_max * 1.18) so it never overlaps a decode bar.
        for mi, logical in enumerate(models):
            rows = []  # (color, short, prefill_value), top-to-bottom = variant order
            for vi in group_layout[mi]:
                variant, spec, _label, color, _is_baseline, _show_delta, short = variants[vi]
                val, win = cell_val(logical, variant, spec, workload_key)
                if spec == "best" and win is not None:
                    pf_val = data.get((logical, variant, SPEC_DISPLAY_TO_KEY[win]), {}).get("prefill", 0)
                else:
                    pf_val = cell_val(logical, variant, spec, "prefill")[0]
                if pf_val > 0:
                    rows.append((color, short, pf_val))
            if not rows:
                continue
            gx = x[mi]
            box_left = gx - group_step * 0.48
            box_w = group_step * 0.96
            inset_ax = ax.inset_axes(
                [box_left + box_w * 0.03, ymax * 0.855, box_w * 0.40, ymax * 0.12],
                transform=ax.transData,
            )
            pf_max = max(v for _, _, v in rows)
            ys = np.arange(len(rows))[::-1]
            for (rcolor, rshort, pf_val), y in zip(rows, ys):
                inset_ax.barh(y, pf_val, height=0.6, color=rcolor,
                               edgecolor="#1f2937", linewidth=0.3, zorder=2)
                inset_ax.text(pf_val + pf_max * 0.06, y, f"{pf_val:.0f}",
                              va="center", ha="left", fontsize=4.6, color="#374151")
            inset_ax.set_xlim(0, pf_max * 1.5)
            inset_ax.set_ylim(-0.6, len(rows) - 0.4)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            for spine in inset_ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#e5e7eb")
                spine.set_linewidth(0.5)
            inset_ax.set_facecolor("white")
            inset_ax.patch.set_alpha(0.95)
            inset_ax.set_title("prefill tok/s", fontsize=4.8, color="#6b7280",
                               pad=1.2, loc="left")
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_xticks(x)
        ax.set_xticklabels([cfg["x_label"](m) for m in models],
                           fontsize=10, fontweight="medium")
        ax.set_ylabel("decode tok/s", fontsize=10)
        ax.set_title(ax_title, fontsize=12, fontweight="semibold", pad=10)
        ax.grid(True, axis="y", alpha=0.35, linestyle="--", color="#d1d5db", zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    # Header stack with clear air: title on top (y=0.98), legend a band
    # below it (0.925), axes reserved under 0.9 — nothing touches.
    fig.legend(legend_handles, legend_labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.925),
               ncol=len(variants), fontsize=10,
               frameon=False, columnspacing=1.5, handlelength=1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
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
