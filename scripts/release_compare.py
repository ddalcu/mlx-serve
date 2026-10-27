#!/usr/bin/env python3
"""release_compare.py: one release-vs-release view across machines and models, from llmprobe reports.

  release_compare.py --old 26.10.3 --new 26.11.1 \
      --dir "M5 Ultra=~/claude-tmp/bench-26.11.1/m5#24e3646b" --dir "M4 Max=~/claude-tmp/bench-26.11.1/m4max" \
      [--csv rows.csv] [--noise 3] --md compare.md --html compare.html

A `--dir NAME=PATH[#BUILD]` holds llmprobe JSON reports (any depth, `.bb/scripts/final/perf.sh` or `tests/bench.sh`
output). A report is the old or new release by its `run.label`: `<side>-<arm>`, where side is `--old-label`
(default `baseline`) or `--new-label` (default `final`) and arm the cell (`default`, `nospec`, `plain`); its model is
`target.model`. Repeated reports of one cell take their median; a later `--dir` of the same machine wins
where it has the new side.
A `--csv` adds rows measured elsewhere: machine,model,arm,build,decode_old,decode_new,prefill_old,prefill_new,
agg_old,agg_new,note (empty = not measured), and an optional spread (% between repeated passes of one side).
Status: every measured delta at or above -noise% = ok; any below = regression; no old side at all = new (a capability
the old release lacks); the new side missing = pending; repeated reports of a cell more than 2 x noise apart = noisy
(rerun before reading it).
Tests: python3 -m unittest scripts/release_compare.py"""
import argparse
import csv
import html
import json
import os
import statistics
import sys

METRICS = [("decode", "Decode"), ("prefill", "Prefill"), ("agg", "4-way aggregate")]


def report_values(r):
    """(decode, prefill, 4-way aggregate) medians of one llmprobe report."""
    b = r.get("bench") or {}

    def med(k):
        v = b.get(k)
        return v.get("median") if isinstance(v, dict) else None

    return {"decode": med("decodeTokPerSec"), "prefill": med("prefillTokPerSec"),
            "agg": (b.get("batching") or {}).get("aggregateTokPerSec")}


def read_dir(path, old_label, new_label):
    """{(model, arm): {"old": [values...], "new": [values...]}} from every llmprobe report under path."""
    cells = {}
    for root, _, files in os.walk(os.path.expanduser(path)):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            try:
                r = json.load(open(os.path.join(root, fn)))
            except (OSError, ValueError):
                continue
            label = (r.get("run") or {}).get("label") or ""
            model = (r.get("target") or {}).get("model")
            if not model or "bench" not in r or "-" not in label:
                continue
            side, arm = label.split("-", 1)
            side = {old_label: "old", new_label: "new"}.get(side)
            if side:
                cells.setdefault((model, arm), {"old": [], "new": []})[side].append(report_values(r))
    return cells


def median_of(values, key):
    xs = [v[key] for v in values if v.get(key) is not None]
    return statistics.median(xs) if xs else None


def spread(values, key):
    """Max over min of one cell's repeated reports, in percent; 0 for a single report."""
    xs = [v[key] for v in values if v.get(key)]
    return (max(xs) / min(xs) - 1) * 100 if len(xs) > 1 else 0.0


def rows_from_dirs(dirs, old_label, new_label):
    """Rows keyed (machine, model, arm); a later dir of the same machine replaces its cells."""
    rows = {}
    for spec in dirs:
        name, _, rest = spec.partition("=")
        path, _, build = rest.partition("#")
        for (model, arm), sides in read_dir(path, old_label, new_label).items():
            row = {"machine": name, "model": model, "arm": arm, "build": build, "note": "", "spread": 0.0}
            for key, _ in METRICS:
                row[key + "_old"] = median_of(sides["old"], key)
                row[key + "_new"] = median_of(sides["new"], key)
                row["spread"] = max(row["spread"], spread(sides["old"], key), spread(sides["new"], key))
            if not sides["new"] and (name, model, arm) in rows:
                continue  # a run still in progress keeps the earlier measured row
            rows[(name, model, arm)] = row
    return rows


def rows_from_csv(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        row = {k: (r.get(k) or "").strip() for k in ("machine", "model", "arm", "build", "note")}
        for key, _ in METRICS:
            for side in ("old", "new"):
                v = (r.get(f"{key}_{side}") or "").strip()
                row[f"{key}_{side}"] = float(v) if v else None
        row["spread"] = float(r.get("spread") or 0)
        rows[(row["machine"], row["model"], row["arm"])] = row
    return rows


def delta(row, key):
    o, n = row[key + "_old"], row[key + "_new"]
    return (n / o - 1) * 100 if o and n else None


def status(row, noise):
    ds = [delta(row, k) for k, _ in METRICS]
    if all(row[k + "_old"] is None for k, _ in METRICS) and any(row[k + "_new"] is not None for k, _ in METRICS):
        return "new"
    if all(d is None for d in ds):
        return "pending"
    if row.get("spread", 0.0) > 2 * noise:
        return "noisy"
    return "regression" if any(d is not None and d < -noise for d in ds) else "ok"


SYMBOL = {"ok": "✅", "regression": "❌", "pending": "⏳", "noisy": "⚠ reps disagree, rerun", "new": "🆕 new"}


def num(v):
    return "—" if v is None else (f"{v:.0f}" if v >= 1000 else f"{v:.1f}")


def pct(d):
    return "—" if d is None else f"{d:+.1f}%"


def markdown(rows, old, new, noise):
    head = ["Machine", "Model", "Arm", "Build"]
    for _, title in METRICS:
        head += [f"{title} {old} → {new}", "Δ"]
    head.append("Status")
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        cells = [row["machine"], row["model"], row["arm"], row["build"] or "—"]
        for key, _ in METRICS:
            d = delta(row, key)
            mark = f"**{pct(d)}**" if d is not None and d < -noise else pct(d)
            cells += [f"{num(row[key + '_old'])} → {num(row[key + '_new'])}", mark]
        st = status(row, noise)
        cells.append(SYMBOL[st] + (" " + row["note"] if row["note"] else ""))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append(f"tok/s; Δ = {new} vs {old}; ✅ every Δ ≥ −{noise:g}%, ❌ one below, ⏳ not measured, 🆕 {old} cannot run "
               f"it, ⚠ the repeated reports of a cell differ by more than {2 * noise:g}%.")
    return "\n".join(out) + "\n"


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
:root{{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e7e6e1;--band:#f0efec;
--gain:#2a78d6;--loss:#e34948;--flat:#898781}}
@media (prefers-color-scheme:dark){{:root{{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--band:#383835;--gain:#3987e5;--loss:#e66767;--flat:#898781}}}}
body{{background:var(--page);color:var(--ink);font:13px/1.4 -apple-system,system-ui,sans-serif;margin:24px}}
h1{{font-size:18px;margin:0 0 4px}} p{{color:var(--ink2);margin:0 0 16px}}
svg{{background:var(--surface);border-radius:8px}} text{{fill:var(--ink2)}} .lab{{fill:var(--ink)}}
table{{border-collapse:collapse;margin-top:24px;background:var(--surface)}} td,th{{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right}}
td:nth-child(-n+4),th:nth-child(-n+4),td:last-child,th:last-child{{text-align:left}} th{{color:var(--ink2);font-weight:600}}
#tip{{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--grid);
border-radius:6px;padding:6px 8px;display:none;box-shadow:0 2px 8px rgba(0,0,0,.15)}}
.key span{{display:inline-block;width:10px;height:10px;border-radius:5px;margin:0 4px 0 12px}}
</style></head><body><h1>{title}</h1><p>{sub}</p><div class="key">
<span style="background:var(--gain)"></span>faster<span style="background:var(--flat)"></span>within ±{noise:g}%
<span style="background:var(--loss)"></span>slower than {old}</div>
{svg}<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table><div id="tip"></div>
<script>const tip=document.getElementById('tip');document.querySelectorAll('[data-tip]').forEach(e=>{{
e.addEventListener('mousemove',v=>{{tip.style.display='block';tip.textContent=e.dataset.tip;
tip.style.left=(v.clientX+12)+'px';tip.style.top=(v.clientY+12)+'px'}});
e.addEventListener('mouseleave',()=>tip.style.display='none')}})</script></body></html>"""


def chart_svg(rows, old, new, noise, span):
    """Three small multiples (one per metric) of Δ% dots per row, one shared x range, the noise band shaded."""
    row_h, top, panel_w, gap = 22, 36, 260, 64
    names = [f'{r["machine"]} · {r["model"]} · {r["arm"]}' for r in rows]
    label_w = 24 + 7 * max((len(n) for n in names), default=10)  # 13px sans averages under 7px a character
    h = top + row_h * len(rows) + 28
    w = label_w + len(METRICS) * (panel_w + gap)
    p = [f'<svg width="{w}" height="{h}" role="img" aria-label="{html.escape(new)} vs {html.escape(old)}">']
    for i, row in enumerate(rows):
        y = top + i * row_h + row_h / 2
        p.append(f'<text class="lab" x="8" y="{y + 4}">{html.escape(names[i])}</text>')
    for j, (key, title) in enumerate(METRICS):
        x0 = label_w + j * (panel_w + gap)

        def xs(d):
            return x0 + (max(-span, min(span, d)) + span) / (2 * span) * panel_w

        p.append(f'<text x="{x0 + panel_w / 2}" y="18" text-anchor="middle" font-weight="600">{title} Δ%</text>')
        p.append(f'<rect x="{xs(-noise)}" y="{top - 6}" width="{xs(noise) - xs(-noise)}" '
                 f'height="{row_h * len(rows) + 6}" fill="var(--band)"/>')
        for t in (-span, 0, span):
            p.append(f'<line x1="{xs(t)}" x2="{xs(t)}" y1="{top - 6}" y2="{top + row_h * len(rows)}" '
                     f'stroke="var(--grid)" stroke-width="{2 if t == 0 else 1}"/>')
            p.append(f'<text x="{xs(t)}" y="{h - 10}" text-anchor="middle" fill="var(--muted)">{t:+g}%</text>')
        for i, row in enumerate(rows):
            d = delta(row, key)
            if d is None:
                continue
            y = top + i * row_h + row_h / 2
            color = "var(--loss)" if d < -noise else "var(--gain)" if d > noise else "var(--flat)"
            tipt = f'{row["machine"]} {row["model"]} {row["arm"]}: {title} {num(row[key + "_old"])} → ' \
                   f'{num(row[key + "_new"])} tok/s ({pct(d)})'
            clip = " (clipped)" if abs(d) > span else ""
            p.append(f'<g data-tip="{html.escape(tipt + clip)}"><rect x="{x0}" y="{y - row_h / 2}" width="{panel_w}" '
                     f'height="{row_h}" fill="transparent"/><line x1="{xs(0)}" x2="{xs(d)}" y1="{y}" y2="{y}" '
                     f'stroke="{color}" stroke-width="2"/><circle cx="{xs(d)}" cy="{y}" r="5" fill="{color}" '
                     f'stroke="var(--surface)" stroke-width="2"/></g>')
            if abs(d) > noise:
                anchor, dx = ("start", 8) if d >= 0 else ("end", -8)
                p.append(f'<text x="{xs(d) + dx}" y="{y + 4}" text-anchor="{anchor}">{pct(d)}</text>')
    p.append("</svg>")
    return "".join(p)


def page(rows, old, new, noise, span):
    md = markdown(rows, old, new, noise).splitlines()
    head = [c.strip() for c in md[0].strip("|").split("|")]
    body = ["<tr>" + "".join(f"<td>{html.escape(c.strip().replace('**', ''))}</td>" for c in line.strip("|").split("|"))
            + "</tr>" for line in md[2:] if line.startswith("|")]
    machines = sorted({r["machine"] for r in rows})
    return PAGE.format(title=f"mlx-serve {html.escape(new)} vs {html.escape(old)}", noise=noise, old=html.escape(old),
                       sub=f"{len(rows)} cells on {', '.join(map(html.escape, machines))}; Δ clipped at ±{span:g}%",
                       svg=chart_svg(rows, old, new, noise, span),
                       thead="".join(f"<th>{html.escape(c)}</th>" for c in head), tbody="".join(body))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--dir", action="append", default=[])
    ap.add_argument("--csv", action="append", default=[])
    ap.add_argument("--old-label", default="baseline")
    ap.add_argument("--new-label", default="final")
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--span", type=float, default=30.0)
    ap.add_argument("--md")
    ap.add_argument("--html")
    a = ap.parse_args(argv)
    rows = rows_from_dirs(a.dir, a.old_label, a.new_label)
    for c in a.csv:
        rows.update(rows_from_csv(c))
    ordered = [rows[k] for k in sorted(rows, key=lambda k: (k[0], k[1].lower(), k[2]))]
    md = markdown(ordered, a.old, a.new, a.noise)
    if a.md:
        open(a.md, "w").write(md)
    else:
        sys.stdout.write(md)
    if a.html:
        open(a.html, "w").write(page(ordered, a.old, a.new, a.noise, a.span))
    # A noisy cell may hide a regression: it fails the gate until a rerun settles it.
    return 1 if any(status(r, a.noise) in ("regression", "noisy") for r in ordered) else 0


if __name__ == "__main__":
    sys.exit(main())


import unittest  # noqa: E402


class Test(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name

        def rep(path, label, model, dec, pf, agg):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump({"run": {"label": label}, "target": {"model": model},
                       "bench": {"decodeTokPerSec": {"median": dec}, "prefillTokPerSec": {"median": pf},
                                 "batching": {"aggregateTokPerSec": agg}}}, open(path, "w"))
        rep(f"{d}/perf/q/baseline.default.r1.json", "baseline-default", "Qwen", 100, 1000, 200)
        rep(f"{d}/perf/q/final.default.r1.json", "final-default", "Qwen", 99, 1010, 180)
        rep(f"{d}/perf/q/final.default.r2.json", "final-default", "Qwen", 101, 1020, 182)
        rep(f"{d}/perf/g/baseline.nospec.r1.json", "baseline-nospec", "Gemma", 50, 500, None)
        open(f"{d}/rows.csv", "w").write("machine,model,arm,build,decode_old,decode_new,prefill_old,prefill_new,"
                                         "agg_old,agg_new,note\nM4,Flash,default,abc,87.8,89.9,793,790,98.5,99.6,"
                                         "bytes identical\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_medians_deltas_and_status(self):
        rows = rows_from_dirs([f"M5={self.tmp.name}#24e3"], "baseline", "final")
        q = rows[("M5", "Qwen", "default")]
        self.assertEqual((q["decode_new"], q["prefill_new"], q["agg_new"], q["build"]), (100, 1015, 181, "24e3"))
        self.assertAlmostEqual(delta(q, "agg"), -9.5)
        self.assertEqual(status(q, 3), "regression")
        q2 = dict(q, spread=12.0)
        self.assertEqual(status(q2, 3), "noisy")
        self.assertEqual(status(rows[("M5", "Gemma", "nospec")], 3), "pending")
        self.assertEqual(status(rows_from_csv(f"{self.tmp.name}/rows.csv")[("M4", "Flash", "default")], 3), "ok")

    def test_csv_new_and_noisy_rows(self):
        p = f"{self.tmp.name}/more.csv"
        open(p, "w").write("machine,model,arm,build,decode_old,decode_new,prefill_old,prefill_new,agg_old,agg_new,note,"
                           "spread\nM5,Pack,stream,abc,,13.1,,587.3,,9.1,new only,\nM5,Repack,stream,abc,47.7,44.5,825.6,"
                           "1256.3,32.3,31.2,one pass,11\n")
        rows = rows_from_csv(p)
        self.assertEqual(status(rows[("M5", "Pack", "stream")], 3), "new")
        self.assertEqual(status(rows[("M5", "Repack", "stream")], 3), "noisy")

    def test_outputs(self):
        md, h = f"{self.tmp.name}/o.md", f"{self.tmp.name}/o.html"
        rc = main(["--old", "26.10.3", "--new", "26.11.1", "--dir", f"M5={self.tmp.name}",
                   "--csv", f"{self.tmp.name}/rows.csv", "--md", md, "--html", h])
        self.assertEqual(rc, 1)
        text = open(md).read()
        self.assertIn("Decode 26.10.3 → 26.11.1", text)
        self.assertIn("| M5 | Qwen | default | — | 100.0 → 100.0 | +0.0% | 1000 → 1015 | +1.5% | 200.0 → 181.0 | "
                      "**-9.5%** | ❌ |", text)
        self.assertIn("✅ bytes identical", text)
        self.assertIn("<svg", open(h).read())

    def test_a_noisy_regression_fails_the_gate(self):
        d = f"{self.tmp.name}/noisy"
        for name, label, dec in (("baseline.default.r1", "baseline-default", 100), ("final.default.r1", "final-default", 80),
                                 ("final.default.r2", "final-default", 100)):
            os.makedirs(f"{d}/perf/q", exist_ok=True)
            json.dump({"run": {"label": label}, "target": {"model": "Qwen"},
                       "bench": {"decodeTokPerSec": {"median": dec}, "prefillTokPerSec": {"median": 1000}}},
                      open(f"{d}/perf/q/{name}.json", "w"))
        self.assertEqual(status(rows_from_dirs([f"M5={d}"], "baseline", "final")[("M5", "Qwen", "default")], 3), "noisy")
        self.assertEqual(main(["--old", "a", "--new", "b", "--dir", f"M5={d}", "--md", f"{d}/o.md"]), 1)
