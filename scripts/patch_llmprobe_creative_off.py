#!/usr/bin/env python3
"""Make llmprobe 0.6.7 timed/eval chat requests explicitly thinking-off.

The published `--reasoning off` omits `reasoning_effort` on eval requests but
does not send `enable_thinking=false`. Timed requests also bypass MTPLX's
prefix cache; mlx-serve ignores that header and disables its cache at launch.
Patch a disposable bundle, not npm's installation. Conformance probes retain
their original request bodies.
"""

import hashlib
import json
import sys
from pathlib import Path


def patch_bundle(source: str) -> str:
    timed_start = source.index("async function timedRun(ctx, surface, text, maxTokens, extra, system) {")
    timed_end = source.index("\n  let timed;", timed_start)
    timed = source[timed_start:timed_end]
    old_timed = "    stream: true\n  };"
    if timed.count(old_timed) != 1:
        raise ValueError("llmprobe timedRun request shape changed")
    timed = timed.replace(
        old_timed,
        '    stream: true,\n'
        '    ...(surface === "chat" ? { enable_thinking: false } : {})\n'
        '  };',
    )
    source = source[:timed_start] + timed + source[timed_end:]

    timed_start = source.index("async function timedRun(ctx, surface, text, maxTokens, extra, system) {")
    timed_end = source.index("\n}", timed_start)
    timed = source[timed_start:timed_end]
    old_headers = "      adapter.headers(ctx.config),\n      null"
    if timed.count(old_headers) != 1:
        raise ValueError("llmprobe timedRun headers shape changed")
    timed = timed.replace(
        old_headers,
        '      { ...adapter.headers(ctx.config), "X-MTPLX-Cache-Mode": "bypass" },\n'
        '      null',
    )
    source = source[:timed_start] + timed + source[timed_end:]

    eval_start = source.index("async function runReasoning(ctx, opts) {")
    eval_end = source.index("\n        const sendOpts", eval_start)
    reasoning = source[eval_start:eval_end]
    old_eval = "          allowReasoning: false\n        };"
    if reasoning.count(old_eval) != 1:
        raise ValueError("llmprobe runReasoning request shape changed")
    reasoning = reasoning.replace(
        old_eval,
        '          allowReasoning: false,\n'
        '          ...(surface === "chat" ? { extra: { enable_thinking: false } } : {})\n'
        '        };',
    )
    return source[:eval_start] + reasoning + source[eval_end:]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_llmprobe_creative_off.py INPUT_BUNDLE OUTPUT_BUNDLE")
    source_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    metadata = json.loads((source_path.parents[2] / "package.json").read_text())
    if (metadata.get("name"), metadata.get("version")) != ("llmprobe", "0.6.7"):
        raise RuntimeError("expected pinned llmprobe 0.6.7")
    source = source_path.read_text()
    patched = patch_bundle(source)
    output_path.write_text(patched)
    print(f"source_sha256={hashlib.sha256(source.encode()).hexdigest()}")
    print(f"patched_sha256={hashlib.sha256(patched.encode()).hexdigest()}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
