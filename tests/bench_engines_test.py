#!/usr/bin/env python3
"""Pure-layer test for chart engine versions (`python3 tests/bench_engines_test.py`).

A comparison chart that names oMLX without naming WHICH oMLX is a claim with a
shelf life. These pin the parse and the labelling, including the two cases that
would silently put a wrong version on a public chart: a missing entry (label
must be left alone, never guessed) and the two LM Studio bars, which are one
product and must share one declared version.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_engines import (  # noqa: E402
    format_engines_note,
    label_with_version,
    parse_engine_versions,
)

CSV = [
    "# 2026-08-07 · llmprobe --bench-only (one run/rung, to 16k) · shipping defaults",
    "# engines: mlx-serve=26.8.3 omlx=0.5.2 mtplx=2.5.3 lmstudio=0.4.19+2",
    "model|engine|spec|context|prefill_tps|decode_tps|ttft_ms|x|y|z|hw|n",
    "gemma4-e4b-4bit|mlx-serve|default|headline|2594.1|117.9|58|1|1|c|hw|",
]

failed = 0


def check(name, got, want):
    global failed
    ok = got == want
    failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  got {got!r} want {want!r}"))


v = parse_engine_versions(CSV)
check("parses every declared engine", v,
      {"mlx-serve": "26.8.3", "omlx": "0.5.2", "mtplx": "2.5.3", "lmstudio": "0.4.19+2"})
check("a CSV with no engines line yields nothing",
      parse_engine_versions(CSV[:1] + CSV[2:]), {})
check("the run note is NOT mistaken for an engines line",
      parse_engine_versions([CSV[0]]), {})

check("exact key", label_with_version("oMLX", "omlx", v), "oMLX 0.5.2")
check("mlx-serve", label_with_version("MLX-serve", "mlx-serve", v), "MLX-serve 26.8.3")
# One product, two bars, one declared version — the prefix case.
check("lmstudio-baseline inherits the lmstudio version",
      label_with_version("LM Studio (MLX)", "lmstudio-baseline", v), "LM Studio (MLX) 0.4.19+2")
check("lmstudio-alt inherits it too",
      label_with_version("LM Studio (GGUF, baseline)", "lmstudio-alt", v),
      "LM Studio (GGUF, baseline) 0.4.19+2")
# An engine that ran but was never declared must keep its plain label rather
# than borrowing someone else's number.
check("undeclared engine is left alone",
      label_with_version("vLLM", "vllm", v), "vLLM")
check("no versions at all is a no-op",
      label_with_version("oMLX", "omlx", {}), "oMLX")
# A version containing '=' (build metadata) must survive the split.
check("build metadata survives",
      parse_engine_versions(["# engines: lmstudio=0.4.19+2"])["lmstudio"], "0.4.19+2")

check("note round-trips through the parser",
      parse_engine_versions(["# " + format_engines_note(v)]), v)
check("empty map writes no note", format_engines_note({}), "")

print(f"\n{10 - failed}/10 passed")
sys.exit(1 if failed else 0)
