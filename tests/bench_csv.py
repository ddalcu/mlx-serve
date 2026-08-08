#!/usr/bin/env python3
"""bench_csv.py — turn a directory of llmprobe `--save` reports into the one
CSV both charts render from.

llmprobe is the measurement layer (tests/bench.sh drives it). One `--bench-only`
run per engine per model produces headline decode/prefill/TTFT medians AND the
context-scaling ladder, so a release needs ONE bench run and ONE CSV where it
used to need two protocols and two files.

Input:  a directory of `<model>__<engine>__<spec>.json` llmprobe reports.
Output: pipe-delimited CSV, one row per (model, engine, context):

  model|engine|spec|context|prefill_tps|decode_tps|ttft_ms|tok_per_step|spec_ratio|checkpoint|hardware|notes

`context` is `headline` for the top-level bench numbers (what the bar chart
plots) and a rung label — 0.5k, 4k, 8k, 16k, 32k, 64k — for each ladder point
(what the ladder chart plots). A rung the engine rejected is written with empty
rates and llmprobe's own error in `notes`, never as a zero.

Usage:
  python3 tests/bench_csv.py <json_dir> --out docs/perf-csvs/probe-<ver>.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_engines import format_engines_note  # noqa: E402

HEADER = ("model|engine|spec|context|prefill_tps|decode_tps|ttft_ms|"
          "tok_per_step|spec_ratio|checkpoint|hardware|notes")


def rung_label(tokens: int) -> str:
    """512 → 0.5k, 4096 → 4k, 65536 → 64k. Matches the historical ladder CSVs."""
    k = tokens / 1024
    return f"{k:g}k"


def hardware_tag(machine: dict) -> str:
    """`Apple M4 Max` + 128 GB → `Apple-M4-Max-128gb`, the tag the charts group
    on. A CSV mixing two machines is rejected by the plot scripts, not merged."""
    cpu = (machine.get("cpu") or machine.get("arch") or "unknown").strip()
    mem = machine.get("memGB")
    tag = "-".join(cpu.split())
    return f"{tag}-{int(mem)}gb" if mem else tag


def med(stat) -> str:
    """A BenchStat's median, or empty. Empty means "not measured" — never 0,
    which a chart would draw as a real bar sitting on the floor."""
    if not stat or stat.get("median") is None:
        return ""
    return f"{stat['median']:.1f}"


def num(value, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def clean(text) -> str:
    """Notes ride in a pipe-delimited field and are read by humans, not parsed."""
    if not text:
        return ""
    return " ".join(str(text).split()).replace("|", "/")[:160]


def rows_for(report: dict, model: str, engine: str, spec: str) -> list[str]:
    bench = report.get("bench")
    if not bench:
        return []
    target = report.get("target") or {}
    checkpoint = target.get("model", "")
    hw = hardware_tag(bench.get("machine") or {})

    drift = bench.get("loadDrift") or {}
    notes = []
    if drift.get("verdict") and drift["verdict"] != "steady":
        notes.append(f"load {drift['verdict']} {num(drift.get('driftPct'), 1)}%")
    if bench.get("streamCaveat"):
        notes.append(clean(bench["streamCaveat"]))
    spec_probe = bench.get("speculative") or {}

    out = [
        "|".join([
            model, engine, spec, "headline",
            med(bench.get("prefillTokPerSec")),
            med(bench.get("decodeTokPerSec")),
            med(bench.get("ttftMs")),
            num(spec_probe.get("tokensPerStep")),
            num(spec_probe.get("ratio")),
            checkpoint, hw, clean("; ".join(notes)),
        ])
    ]

    for point in bench.get("contextScaling") or []:
        rung_spec = point.get("speculative") or {}
        out.append("|".join([
            model, engine, spec, rung_label(point["targetTokens"]),
            num(point.get("prefillTokPerSec"), 1),
            num(point.get("decodeTokPerSec"), 1),
            num(point.get("ttftMs"), 0),
            num(rung_spec.get("tokensPerStep")),
            num(rung_spec.get("ratio")),
            checkpoint, hw, clean(point.get("note")),
        ]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("json_dir", type=Path, help="directory of llmprobe --save reports")
    p.add_argument("--out", type=Path, required=True, help="CSV to write")
    p.add_argument("--note", default="", help="run note recorded in the CSV header comment")
    p.add_argument(
        "--engines", default="",
        help="engine versions for the charts, e.g. "
             "'mlx-serve=26.8.3 omlx=0.5.2 mtplx=2.5.3 lmstudio=0.4.19+2'. "
             "Written as a SECOND '#' line; readers take the run note from the "
             "first '#' only, so this is additive and old CSVs still parse.")
    args = p.parse_args()

    reports = sorted(args.json_dir.glob("*__*__*.json"))
    if not reports:
        sys.exit(f"No llmprobe reports (<model>__<engine>__<spec>.json) in {args.json_dir}")

    rows: list[str] = []
    skipped: list[str] = []
    for path in reports:
        model, engine, spec = path.stem.split("__", 2)
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            skipped.append(f"{path.name}: unreadable ({exc})")
            continue
        got = rows_for(report, model, engine, spec)
        if not got:
            # A conformance-only run has no bench block. Saying so beats a
            # silently short CSV that reads as "this engine wasn't run".
            skipped.append(f"{path.name}: no bench block (run llmprobe with --bench-only)")
            continue
        rows.extend(got)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        if args.note:
            f.write(f"# {clean(args.note)}\n")
        if args.engines:
            # A chart that names oMLX has to name WHICH oMLX, or the claim
            # ages badly. llmprobe's reports carry no engine version, so it
            # rides here and both plot scripts read it off the CSV.
            versions = dict(
                tok.split("=", 1) for tok in args.engines.split() if "=" in tok)
            note = format_engines_note(versions)
            if note:
                f.write(f"# {clean(note)}\n")
        f.write(HEADER + "\n")
        f.write("\n".join(rows) + "\n")

    print(f"Wrote {args.out} ({len(rows)} rows from {len(reports) - len(skipped)} reports)")
    for line in skipped:
        print(f"  skipped {line}", file=sys.stderr)


if __name__ == "__main__":
    main()
