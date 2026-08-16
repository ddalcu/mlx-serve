#!/usr/bin/env python3
"""Quality battery for the Qwen3.8-27B packs — the gate the iQ-MLX build has to clear.

Run every cell on the SHIPPED 4-bit and 8-bit packs first. The bar is those two
points, not an invented threshold: a number is only meaningful next to what the
packs we already ship score on the same corpus.

Cells:

  reference  Held-out windows (`split_docs(holdout=True)` — the documents the
             imatrix collection deliberately never saw) forwarded through the
             bf16 source, saving the full log-probs at sampled positions.
  tokens     The same windows through a pack: top-1 agreement and mean
             KL(bf16 || pack), per corpus slice.
  agent      A fixed multi-round tool loop against a booted server, scoring the
             longest run of consecutive IDENTICAL tool calls. This is the cell
             that matters: uniform low-bit quantization causes TURN-level agent
             loops that token batteries are structurally blind to.
  perf       Decode tok/s with and without MTP, plus the `[spec-stats]` line —
             sub-4-bit weights fall out of `verifyQmmLane` (4/5/6-bit only), so
             the pack may give back in lost lanes what it gains in bandwidth.

  python3 tests/qwen38_iq_battery.py reference --out ~/claude-tmp/qwen38-iq/ref.npz
  python3 tests/qwen38_iq_battery.py tokens --ref ...ref.npz --pack <dir>
  python3 tests/qwen38_iq_battery.py agent  --pack <dir>
  python3 tests/qwen38_iq_battery.py perf   --pack <dir>
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_imatrix_collect as corpus  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "zig-out/bin/mlx-serve"
DEFAULT_SRC = corpus.DEFAULT_SRC



# ============================================================
# mlx-lm as an oracle for OUR packs
# ============================================================

def _install_sanitize_fix():
    """mlx-lm folds the `+1` into Qwen3.8's delta-encoded norms when it sees
    EITHER an MTP head or an untransposed conv1d. Our packs ship the MTP head
    and have already had both transforms applied by the converter, so mlx-lm
    folds a second time: every norm off by one and the output is noise. It is
    silent — the shipped, production-serving 4-bit pack scored 0.2% top-1
    agreement against its own bf16 source through an unpatched mlx-lm, i.e.
    exactly what a dead model scores, which is how this was found.

    The honest signal is the conv1d layout, not the head: HF's `[C, 1, K]` means
    a raw checkpoint that still needs both transforms, `[C, K, 1]` means it has
    already been through a converter. Dropping the MTP keys in the second case
    leaves mlx-lm's own rule pointing the right way."""
    from mlx_lm.models import qwen3_5
    original = qwen3_5.TextModel.sanitize

    def sanitize(self, weights):
        converted = not any(k.endswith("conv1d.weight") and v.shape[-1] != 1
                            for k, v in weights.items())
        if converted:
            weights = {k: v for k, v in weights.items() if "mtp." not in k}
        return original(self, weights)

    qwen3_5.TextModel.sanitize = sanitize


# ============================================================
# Held-out windows
# ============================================================

def holdout_windows(tok, seed, seq_len, per_slice):
    """(slice, token-id window) pairs the imatrix collection never touched."""
    rng = random.Random(seed)
    sweb = corpus.swebench_rows()
    traffic = corpus.tool_traffic(600)
    slices = {name: corpus.split_docs(docs, True) for name, docs in (
        ("agent", corpus.slice_agent(tok, rng, traffic, sweb, 400)),
        ("code", corpus.slice_code(tok, rng, sweb, 400)),
        ("prose", corpus.slice_prose(tok, rng, sweb, 400)),
        ("math", corpus.slice_math(tok, rng, 200)),
    )}
    out = []
    for name, docs in slices.items():
        taken = 0
        for doc in docs:
            ids = tok.encode(doc)
            if len(ids) < seq_len // 2:
                continue
            out.append((name, ids[:seq_len]))
            taken += 1
            if taken >= per_slice:
                break
    return out


def forward_logprobs(model, window, positions):
    import mlx.core as mx
    logits = model(mx.array(window)[None])[0]
    lp = logits[mx.array(positions)].astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    mx.eval(lp)
    return np.array(lp, copy=True)


def sampled_positions(n_tokens, k, rng):
    # Never position 0: with no context its distribution says nothing about the
    # model, only about the embedding table.
    return sorted(rng.sample(range(1, n_tokens), min(k, n_tokens - 1)))


def cmd_reference(args):
    from mlx_lm.utils import load
    _install_sanitize_fix()
    print(f"loading reference {args.src} …", flush=True)
    model, tok = load(args.src)
    model.eval()
    wins = holdout_windows(tok, args.seed, args.seq_len, args.per_slice)
    rng = random.Random(args.seed + 1)
    names, ids, poss, refs = [], [], [], []
    t0 = time.time()
    for i, (name, window) in enumerate(wins, 1):
        pos = sampled_positions(len(window), args.positions, rng)
        refs.append(forward_logprobs(model, window, pos).astype(np.float16))
        names.append(name)
        ids.append(np.array(window, dtype=np.int32))
        poss.append(np.array(pos, dtype=np.int32))
        if i % 8 == 0 or i == len(wins):
            print(f"  {i}/{len(wins)} windows, {time.time()-t0:.0f}s", flush=True)
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    # Windows are ragged (a short document is forwarded at its own length, not
    # padded — padding would put the model in a state serving never puts it in),
    # so the ids ride flat with an offset table. Positions are uniform.
    np.savez_compressed(
        out, slices=np.array(names), logprobs=np.stack(refs),
        ids_flat=np.concatenate(ids), ids_len=np.array([len(a) for a in ids]),
        positions=np.stack(poss),
        meta=np.array(json.dumps({"src": args.src, "seed": args.seed,
                                  "seq_len": args.seq_len,
                                  "positions": args.positions})))
    print(f"wrote {out} — {len(wins)} windows x {args.positions} positions")


def cmd_tokens(args):
    from mlx_lm.utils import load
    _install_sanitize_fix()
    ref = np.load(os.path.expanduser(args.ref), allow_pickle=False)
    names, poss = ref["slices"], ref["positions"]
    bounds = np.concatenate([[0], np.cumsum(ref["ids_len"])])
    ids = [ref["ids_flat"][bounds[i]:bounds[i + 1]] for i in range(len(names))]
    ref_lp = ref["logprobs"]
    print(f"loading pack {args.pack} …", flush=True)
    model, _ = load(args.pack)
    model.eval()

    per_slice = {}
    for i in range(len(names)):
        got = forward_logprobs(model, ids[i].tolist(), poss[i].tolist())
        p = np.exp(ref_lp[i].astype(np.float32))
        kl = float((p * (ref_lp[i].astype(np.float32) - got)).sum(axis=-1).mean())
        agree = float((ref_lp[i].argmax(-1) == got.argmax(-1)).mean())
        d = per_slice.setdefault(str(names[i]), {"kl": [], "agree": []})
        d["kl"].append(kl)
        d["agree"].append(agree)

    rows = {k: {"top1_agreement": float(np.mean(v["agree"])),
                "mean_kl": float(np.mean(v["kl"])),
                "windows": len(v["kl"])} for k, v in sorted(per_slice.items())}
    overall = {
        "top1_agreement": float(np.mean([r["top1_agreement"] for r in rows.values()])),
        "mean_kl": float(np.mean([r["mean_kl"] for r in rows.values()])),
    }
    result = {"pack": args.pack, "per_slice": rows, "overall": overall}
    print(json.dumps(result, indent=1))
    if args.json:
        Path(os.path.expanduser(args.json)).write_text(json.dumps(result, indent=1))


# ============================================================
# Live-server cells
# ============================================================

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory to list"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File to read"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite.",
        "parameters": {"type": "object", "properties": {
            "filter": {"type": "string", "description": "Optional name filter"}}}}},
]

# A small fake repository the tools answer FROM. Results must depend on the
# arguments: with one canned answer per tool name, re-reading is rational rather
# than degenerate, and the metric stops measuring what it is for — the shipped
# 8-bit pack scored WORSE than the 4-bit one on the first version of this cell
# for exactly that reason. The bug is findable: `parse` never handles the unary
# minus, so negative numbers come back wrong.
FILES = {
    "README.md": "# tinylang\n\nA toy expression parser. See src/ for the lexer and parser.\n",
    "src/lexer.py": (
        "TOKENS = ('num', 'op', 'lparen', 'rparen')\n\n"
        "def lex(text):\n"
        "    out = []\n"
        "    for part in text.split():\n"
        "        if part.lstrip('-').isdigit():\n"
        "            out.append(Token('num', part))\n"
        "        else:\n"
        "            out.append(Token('op', part))\n"
        "    return out\n"),
    "src/parser.py": (
        "def parse(tokens):\n"
        "    out = []\n"
        "    for t in tokens:\n"
        "        if t.kind == 'num':\n"
        "            out.append(int(t.text.lstrip('-')))\n"
        "        elif t.kind == 'op':\n"
        "            out.append(t.text)\n"
        "    return out\n"),
    "tests/test_parser.py": (
        "def test_negative_numbers():\n"
        "    assert parse(lex('-3 + 4')) == [-3, '+', 4]\n"),
}
DIRS = {
    ".": "src/\ntests/\nREADME.md\n",
    "src": "lexer.py\nparser.py\n",
    "tests": "test_parser.py\n",
}
TEST_OUTPUT = ("1 failed, 3 passed\n"
               "FAILED tests/test_parser.py::test_negative_numbers\n"
               "  assert [3, '+', 4] == [-3, '+', 4]\n")


def scripted(name, arguments):
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return "Error: arguments were not valid JSON."
    if name == "read_file":
        path = str(args.get("path", "")).lstrip("./")
        return FILES.get(path, f"Error: no such file: {args.get('path')}")
    if name == "list_dir":
        path = str(args.get("path", ".")).strip("./") or "."
        return DIRS.get(path, f"Error: no such directory: {args.get('path')}")
    if name == "run_tests":
        return TEST_OUTPUT
    return f"Error: unknown tool {name}"


TASK = ("The test `test_negative_numbers` fails. Look around the repository, read the "
        "relevant source, and tell me in one short paragraph what the bug is. Use the "
        "tools rather than guessing.")


def boot(pack, port, extra=(), log_name="battery"):
    log = open(os.path.expanduser(f"~/claude-tmp/qwen38-iq/{log_name}.log"), "w")
    proc = subprocess.Popen(
        [str(BIN), "--model", pack, "--serve", "--port", str(port), "--host", "127.0.0.1",
         "--prefix-cache-entries", "0", "--ctx-size", "32768", "--log-level", "info", *extra],
        cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise SystemExit(f"server died (rc={proc.returncode}) — see {log.name}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=2)
            return proc, log
        except Exception:
            time.sleep(1)
    raise SystemExit("server never came up")


def shutdown(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    time.sleep(4)


def post(port, path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def agent_run(port, rounds, temp, seed_note):
    messages = [{"role": "user", "content": TASK + seed_note}]
    calls = []
    for _ in range(rounds):
        out = post(port, "/v1/chat/completions", {
            "model": "x", "messages": messages, "tools": TOOLS, "temperature": temp,
            "top_p": 0.95, "max_tokens": 2048, "stream": False})
        msg = out["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tcs})
        for tc in tcs:
            fn = tc["function"]
            calls.append((fn["name"], fn.get("arguments", "")))
            messages.append({"role": "tool", "tool_call_id": tc.get("id", "call_0"),
                             "content": scripted(fn["name"], fn.get("arguments", ""))})
    best, run = 0, 0
    for i, c in enumerate(calls):
        run = run + 1 if i and c == calls[i - 1] else 1
        best = max(best, run)
    return {"calls": len(calls), "max_repeat_run": best,
            "sequence": [c[0] for c in calls]}


def cmd_agent(args):
    proc, log = boot(args.pack, args.port, log_name="agent-" + Path(args.pack).name)
    try:
        runs = [agent_run(args.port, args.rounds, args.temp, f"\n\n(run {i+1})")
                for i in range(args.reps)]
    finally:
        shutdown(proc, log)
    result = {"pack": args.pack, "runs": runs,
              "worst_repeat_run": max(r["max_repeat_run"] for r in runs)}
    print(json.dumps(result, indent=1))
    if args.json:
        Path(os.path.expanduser(args.json)).write_text(json.dumps(result, indent=1))


def cmd_perf(args):
    cells = {}
    for label, extra in (("mtp", ["--mtp"]), ("no-mtp", ["--no-mtp", "--no-pld"])):
        name = f"perf-{Path(args.pack).name}-{label}"
        proc, log = boot(args.pack, args.port, extra, log_name=name)
        try:
            rates = []
            for rep in range(args.reps):
                out = post(args.port, "/v1/chat/completions", {
                    "model": "x", "temperature": 0, "max_tokens": args.tokens,
                    "messages": [{"role": "user", "content":
                                  f"(rep {rep}) Write a detailed explanation of how a "
                                  f"B-tree stays balanced on insertion."}],
                    "stream": False})
                t = out.get("timings", {})
                if t.get("predicted_per_second"):
                    rates.append(t["predicted_per_second"])
            text = Path(os.path.expanduser(f"~/claude-tmp/qwen38-iq/{name}.log")).read_text()
            spec = [ln.strip() for ln in text.splitlines() if "[spec-stats]" in ln]
            cells[label] = {"decode_tok_s": sorted(rates)[len(rates) // 2] if rates else None,
                            "samples": rates, "spec_stats": spec[-3:]}
        finally:
            shutdown(proc, log)
    result = {"pack": args.pack, "cells": cells}
    print(json.dumps(result, indent=1))
    if args.json:
        Path(os.path.expanduser(args.json)).write_text(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reference")
    r.add_argument("--src", default=DEFAULT_SRC)
    r.add_argument("--out", required=True)
    r.add_argument("--seq-len", type=int, default=1024)
    r.add_argument("--per-slice", type=int, default=8)
    r.add_argument("--positions", type=int, default=16)
    r.add_argument("--seed", type=int, default=20260815)
    r.set_defaults(fn=cmd_reference)

    t = sub.add_parser("tokens")
    t.add_argument("--ref", required=True)
    t.add_argument("--pack", required=True)
    t.add_argument("--json", default=None)
    t.set_defaults(fn=cmd_tokens)

    a = sub.add_parser("agent")
    a.add_argument("--pack", required=True)
    a.add_argument("--port", type=int, default=11311)
    a.add_argument("--rounds", type=int, default=12)
    a.add_argument("--reps", type=int, default=3)
    a.add_argument("--temp", type=float, default=0.7)
    a.add_argument("--json", default=None)
    a.set_defaults(fn=cmd_agent)

    p = sub.add_parser("perf")
    p.add_argument("--pack", required=True)
    p.add_argument("--port", type=int, default=11312)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--json", default=None)
    p.set_defaults(fn=cmd_perf)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
