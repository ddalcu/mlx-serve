#!/usr/bin/env python3
"""Engine versions for the bench charts — parsed from the CSV, not hardcoded.

A comparison chart names the engines it beat, so it has to name WHICH BUILD it
beat: "+23% vs oMLX" ages into a claim about whatever oMLX is today. llmprobe's
saved reports carry no engine version (only the target's baseUrl and model), so
`tests/bench.sh`/`bench_csv.py` records them as a second `#` header line:

    # 2026-08-07 · llmprobe --bench-only (one run/rung, to 16k) · shipping defaults
    # engines: mlx-serve=26.8.3 omlx=0.5.2 mtplx=2.5.3 lmstudio=0.4.19+2

The FIRST `#` line stays the run note (both charts already render it as the
subtitle) — every existing reader takes `run_note` from the first `#` and
ignores the rest, so this line is additive and old CSVs keep working.

Shared by both plot scripts so the two charts cannot disagree about what ran.
"""
from __future__ import annotations

ENGINES_PREFIX = "engines:"


def parse_engine_versions(lines) -> dict:
    """`{engine_key: version}` from the CSV's `# engines:` line.

    Accepts any iterable of raw CSV lines. Unknown/malformed tokens are skipped
    rather than guessed at — a wrong version on a public chart is worse than an
    absent one.
    """
    out: dict = {}
    for line in lines:
        s = line.strip()
        if not s.startswith("#"):
            continue
        s = s.lstrip("#").strip()
        if not s.lower().startswith(ENGINES_PREFIX):
            continue
        for token in s[len(ENGINES_PREFIX):].split():
            key, sep, version = token.partition("=")
            if sep and key and version:
                out[key.strip()] = version.strip()
    return out


def label_with_version(label: str, engine_key: str, versions: dict) -> str:
    """`"oMLX"` → `"oMLX 0.5.2"`, leaving the label alone when unknown.

    Matches the exact engine key first, then the longest declared key the
    engine key starts with — so one `lmstudio=` entry covers both the
    `lmstudio-baseline` and `lmstudio-alt` bars, which are one product.
    """
    if not versions:
        return label
    v = versions.get(engine_key)
    if v is None:
        candidates = [k for k in versions if engine_key.startswith(k)]
        if candidates:
            v = versions[max(candidates, key=len)]
    return f"{label} {v}" if v else label


def format_engines_note(versions: dict) -> str:
    """The `# engines: …` line body, stably ordered so CSVs diff cleanly."""
    if not versions:
        return ""
    return ENGINES_PREFIX + " " + " ".join(
        f"{k}={versions[k]}" for k in sorted(versions))
