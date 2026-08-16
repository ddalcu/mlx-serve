#!/usr/bin/env python3
"""Per-input-channel activation statistics (an "imatrix") for Qwen3.8-27B.

Collected on the EXACT bf16 weights the pack is converted from — an imatrix is
only valid for the weights it was collected on, so there is no GGUF detour and
no tensor-name mapping to get wrong. `mlx_lm` loads the source checkpoint
directly (its `qwen3_5.py` sanitize drops the vision tower and the MTP head,
neither of which this build calibrates), every `nn.Linear` in the loaded tree is
wrapped, and the accumulated `sum(x**2)` per input channel is written keyed by
the SOURCE weight name.

The objective the numbers serve: for a linear layer an error dW_j in input
channel j contributes E[(dW_j x_j)^2] = dW_j^2 * E[x_j^2] to the output, so the
per-channel weight IS the mean-squared activation — no square rooting. That is
the same contract `tests/dsv4_imatrix.py` parses out of llama.cpp's format, so
`weighted_affine_quant` consumes these values unchanged.

Corpus (all local, seeded, reproducible — recorded in the output metadata):
  agent  real captured tool traffic (src/fixtures/tool_traffic.jsonl) rendered
         through the model's OWN chat template with its tool schemas, paired
         with a real task description as the user turn and the captured model
         output as the assistant turn
  code   this repo's sources (zig/python/swift/js) and real SWE-bench patches
  prose  this repo's docs and SWE-bench problem statements
  math   generated worked arithmetic/algebra (synthetic — it is here for the
         digit-token channels, and the metadata says so)

Nothing else large may be resident: the bf16 checkpoint is ~54 GB.

  python3 tests/qwen38_imatrix_collect.py --out ~/claude-tmp/qwen38-iq/imatrix.safetensors
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen38_iq_allocate import classify, read_headers  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = "/Volumes/G Drive SSD/models-src/Qwen3.8-27B"
SWEBENCH_GLOB = str(Path.home() / ".cache/huggingface/datasets/princeton-nlp___swe-bench_lite"
                    / "**/swe-bench_lite-*.arrow")

# One rule, both sides: every tenth rendered document is HELD OUT. The imatrix is
# fit on the calibration half, so a token-agreement number measured on the same
# documents flatters the pack it is judging. `tests/qwen38_iq_battery.py` calls
# `split_docs(..., holdout=True)` and gets exactly what this file never saw.
HOLDOUT_EVERY = 10


def split_docs(docs, holdout):
    return [d for i, d in enumerate(docs) if (i % HOLDOUT_EVERY == 0) == holdout]


# ============================================================
# Source-name mapping
# ============================================================

def source_weight_name(module_path: str) -> str:
    """mlx_lm module path -> the weight key in the SOURCE checkpoint.

    Exactly the inverse of `mlx_lm.models.qwen3_5.Model.sanitize`, which is also
    what `tests/convert_qwen38_iq.py`'s `rename()` reproduces forwards. Keying
    the file by source names keeps it a description of the CHECKPOINT rather
    than of our converter's naming choices."""
    key = module_path + ".weight"
    if key.startswith("language_model.model."):
        return "model.language_model." + key[len("language_model.model."):]
    if key.startswith("language_model.lm_head."):
        return key[len("language_model."):]
    return key


# ============================================================
# Collection
# ============================================================

class Collector:
    """Accumulates sum(x**2) per input channel for every wrapped nn.Linear."""

    def __init__(self):
        self.by_id = {}      # id(module) -> source weight name
        self.acc = {}        # source weight name -> mx.array f32 [in_dim]
        self.rows = {}       # source weight name -> tokens seen

    def attach(self, model):
        for path, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                name = source_weight_name(path)
                self.by_id[id(mod)] = name
                self.acc[name] = None
                self.rows[name] = 0
        return len(self.by_id)

    def observe(self, name, x):
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        s = (flat * flat).sum(axis=0)
        cur = self.acc[name]
        self.acc[name] = s if cur is None else cur + s
        self.rows[name] += flat.shape[0]

    def flush(self):
        vals = [v for v in self.acc.values() if v is not None]
        if vals:
            mx.eval(vals)


COLLECTOR = Collector()
_ORIG_LINEAR_CALL = nn.Linear.__call__


def _patched_linear_call(self, x):
    name = COLLECTOR.by_id.get(id(self))
    if name is not None:
        COLLECTOR.observe(name, x)
    return _ORIG_LINEAR_CALL(self, x)


nn.Linear.__call__ = _patched_linear_call


# ============================================================
# Corpus
# ============================================================

def swebench_rows(limit=300):
    """Real GitHub issue text + patches from the locally cached SWE-bench Lite."""
    files = sorted(glob.glob(SWEBENCH_GLOB, recursive=True))
    if not files:
        return []
    import pyarrow as pa
    rows = []
    for f in files:
        with pa.memory_map(f) as src:
            table = pa.ipc.open_stream(src).read_all()
        rows.extend(table.select(["problem_statement", "patch", "test_patch"]).to_pylist())
        if len(rows) >= limit:
            break
    return rows[:limit]


def repo_files(patterns, limit):
    out = []
    for pat in patterns:
        for p in sorted(REPO.glob(pat)):
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            if len(text) > 200:
                out.append((str(p.relative_to(REPO)), text))
            if len(out) >= limit:
                return out
    return out


def tool_traffic(limit):
    path = REPO / "src/fixtures/tool_traffic.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("tools") and rec.get("raw"):
                out.append(rec)
            if len(out) >= limit:
                break
    return out


def render(tok, messages, tools=None):
    """The model's OWN chat template. Thinking/effort are left at the template's
    defaults on purpose — the 3.8 template raises on several of the values our
    server passes, and none of them change the activation distribution."""
    kw = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kw["tools"] = tools
    return tok.apply_chat_template(messages, **kw)


def slice_agent(tok, rng, traffic, sweb, n):
    """Real tool schemas + a real task description + the captured model output."""
    docs = []
    for i in range(n):
        rec = traffic[i % len(traffic)]
        task = (sweb[i % len(sweb)]["problem_statement"] if sweb
                else "Investigate the failure and fix it.")
        tools = [t for t in rec["tools"] if isinstance(t, dict)]
        prompt = render(tok, [{"role": "user", "content": task[:6000]}], tools=tools)
        docs.append(prompt + rec["raw"])
    return docs


def slice_code(tok, rng, sweb, n):
    files = repo_files(["src/*.zig", "tests/*.py", "app/**/*.swift", "website/**/*.js",
                        "scripts/*.sh"], limit=n)
    docs = []
    for i in range(n):
        if i % 2 == 0 and files:
            path, text = files[i % len(files)]
            start = rng.randrange(0, max(1, len(text) - 8000))
            body = text[start:start + 8000]
            docs.append(render(tok, [
                {"role": "user", "content": f"Explain what this does:\n\n```\n{body}\n```"}]))
        elif sweb:
            row = sweb[i % len(sweb)]
            patch = (row["patch"] or "") + "\n" + (row["test_patch"] or "")
            docs.append(render(tok, [
                {"role": "user", "content": "Review this patch."},
            ]) + patch[:8000])
    return docs


def slice_prose(tok, rng, sweb, n):
    files = repo_files(["docs/*.md", "docs/**/*.md", "*.md"], limit=n)
    docs = []
    for i in range(n):
        if i % 2 == 0 and files:
            path, text = files[i % len(files)]
            start = rng.randrange(0, max(1, len(text) - 8000))
            docs.append(render(tok, [{"role": "user", "content": "Summarise this."}])
                        + text[start:start + 8000])
        elif sweb:
            docs.append(render(tok, [
                {"role": "user", "content": sweb[i % len(sweb)]["problem_statement"][:8000]}]))
    return docs


def slice_math(tok, rng, n):
    """Generated worked arithmetic/algebra. Synthetic by construction — it is in
    the corpus for the digit-token channels, not as a claim about math data."""
    docs = []
    for _ in range(n):
        lines = []
        for _ in range(24):
            kind = rng.randrange(4)
            if kind == 0:
                a, b = rng.randrange(1000, 999999), rng.randrange(1000, 999999)
                lines.append(f"{a} * {b} = {a * b}\n{a} + {b} = {a + b}\n{a} - {b} = {a - b}")
            elif kind == 1:
                a, b, c = (rng.randrange(2, 40) for _ in range(3))
                lines.append(f"Solve {a}x + {b} = {c * a + b}.\n"
                             f"{a}x = {c * a}\nx = {c}")
            elif kind == 2:
                n1 = rng.randrange(2, 60)
                lines.append(f"{n1}^2 = {n1 ** 2}, {n1}^3 = {n1 ** 3}, "
                             f"sqrt({n1 ** 2}) = {n1}")
            else:
                vals = [rng.randrange(1, 500) for _ in range(8)]
                tot = sum(vals)
                lines.append("mean of " + ", ".join(map(str, vals)) +
                             f" is {tot}/8 = {tot / 8:.4f}")
        body = "\n".join(lines)
        docs.append(render(tok, [{"role": "user", "content": "Work these out step by step."}])
                    + body)
    return docs


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=300_000,
                    help="total tokens forwarded across all slices")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--head-rows", type=int, default=64,
                    help="positions per forward projected through lm_head")
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--holdout", action="store_true",
                    help="collect on the HELD-OUT half instead (diagnostic only)")
    args = ap.parse_args()

    from mlx_lm.utils import load

    t0 = time.time()
    print(f"loading {args.src} (bf16, ~54 GB) …", flush=True)
    model, tok = load(args.src)
    model.eval()
    n_linear = COLLECTOR.attach(model)
    trunk = model.language_model.model
    head = getattr(model.language_model, "lm_head", None)
    print(f"  loaded in {time.time() - t0:.0f}s, wrapped {n_linear} nn.Linear modules",
          flush=True)

    # Coverage is a property of the CHECKPOINT, not of the module tree we happen
    # to walk: every weight the converter will quantize as a matmul read has to
    # have an entry, or the allocator silently falls back to uncalibrated error
    # for it. (embed_tokens is gather-read and legitimately absent.)
    want = {n for n in read_headers(args.src)
            if classify(n)[1] and not n.startswith("mtp.") and "embed_tokens" not in n}
    missing_names = want - set(COLLECTOR.acc)
    if missing_names:
        raise SystemExit(f"{len(missing_names)} source weights have no wrapped module: "
                         f"{sorted(missing_names)[:5]}")

    rng = random.Random(args.seed)
    sweb = swebench_rows()
    traffic = tool_traffic(600)
    print(f"  corpus sources: {len(sweb)} swe-bench rows, {len(traffic)} traffic records",
          flush=True)

    # Roughly equal token shares; document counts are generous and the token
    # budget is what actually stops each slice.
    slices = {name: split_docs(docs, args.holdout) for name, docs in (
        ("agent", slice_agent(tok, rng, traffic, sweb, 400)),
        ("code", slice_code(tok, rng, sweb, 400)),
        ("prose", slice_prose(tok, rng, sweb, 400)),
        ("math", slice_math(tok, rng, 200)),
    )}
    per_slice_budget = args.max_tokens // len(slices)

    composition, total = {}, 0
    for name, docs in slices.items():
        rng.shuffle(docs)
        used, forwards = 0, 0
        for doc in docs:
            if used >= per_slice_budget:
                break
            ids = tok.encode(doc)
            for off in range(0, len(ids), args.seq_len):
                window = ids[off:off + args.seq_len]
                if len(window) < 16:
                    continue
                # The trunk returns the FINAL normed hidden — exactly lm_head's
                # input. Projecting all 2048 rows through a 248320-wide head is a
                # 2.6 TFLOP GEMM and a 2 GB logits array for statistics that do
                # not need it; serving feeds the head one row per decode step, so
                # sample positions instead.
                h = trunk(mx.array(window)[None])
                if head is not None:
                    pos = rng.sample(range(h.shape[1]), min(args.head_rows, h.shape[1]))
                    head(h[:, mx.array(pos), :])
                mx.eval(h)
                COLLECTOR.flush()
                mx.clear_cache()
                used += len(window)
                forwards += 1
                if used >= per_slice_budget:
                    break
        composition[name] = {"tokens": used, "forwards": forwards, "documents": len(docs)}
        total += used
        print(f"  {name:6s} {used:8d} tokens in {forwards} forwards "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    missing = [n for n, v in COLLECTOR.acc.items() if v is None]
    if missing:
        raise SystemExit(f"{len(missing)} wrapped linears never fired: {missing[:5]}")

    arrays = {}
    for name, s in COLLECTOR.acc.items():
        rows = COLLECTOR.rows[name]
        arrays[name] = (s / float(rows)).astype(mx.float32)   # mean-squared per channel
    mx.eval(list(arrays.values()))

    meta = {
        "source": args.src,
        "seed": str(args.seed),
        "seq_len": str(args.seq_len),
        "total_tokens": str(total),
        "composition": json.dumps(composition),
        "rows_per_weight": json.dumps(COLLECTOR.rows),
        "values": "mean-squared activation per INPUT channel (sum(x^2)/rows)",
        "keys": "SOURCE checkpoint weight names",
        "note": "math slice is generated, not sampled from a dataset",
        "holdout": "calibrated on split_docs(holdout=False); every 10th document withheld",
    }
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), arrays, metadata=meta)
    print(f"\nwrote {out} — {len(arrays)} entries, {total} tokens, "
          f"{time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    sys.exit(main())
