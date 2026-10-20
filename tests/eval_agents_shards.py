#!/usr/bin/env python3
"""Do agents follow AGENTS.md pointers into the shard files?

Exports two git refs (arms), runs each probe in cases through headless `claude -p` inside the
exported tree, and scores from the stream-json transcript:
  correct   every answer regex matches the final answer text
  shard     a shard in the case's `shard` list was Read (or Grep'd by path) before the answer
  control   a control case needs neither a shard read nor over-reading
Gate: the new arm's correct rate is at least the old arm's, and its shard-read rate on non-control
probes is at least 90%.

Usage: eval_agents_shards.py [--old REF] [--new REF] [--reps N] [--jobs N] [--case ID ...]
Needs an authenticated `claude` CLI (CLAUDE_BIN names another binary, e.g. a profile wrapper). Each run
makes about 60 paid model calls at the default --reps 3. Writes <tmp>/shard-eval-results/<ts>/ and
prints a table.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 300
EXPORT = ["AGENTS.md", "CLAUDE.md", "docs", "src"]


def export_arm(ref):
    """Tree of `ref` in a temp dir: CLAUDE.md loads from every ancestor, so keep it outside the repo."""
    dest = Path(tempfile.mkdtemp(prefix="shard-eval-"))
    arc = subprocess.run(["git", "archive", ref, *EXPORT], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arc.stdout, check=True)
    return dest


def claude_cmd(prompt, model):
    return [
        os.environ.get("CLAUDE_BIN", "claude"), "-p", prompt,
        "--model", model, "--effort", "medium",
        "--output-format", "stream-json", "--verbose",
        "--no-session-persistence",
        "--permission-mode", "dontAsk",
        "--allowedTools", "Read", "Glob", "Grep",
        "--disallowedTools", "Agent", "WebFetch", "WebSearch", "Bash", "Edit", "Write",
        "--strict-mcp-config",
        "--settings", json.dumps({"disableAllHooks": True}),
    ]


def events(transcript):
    for line in transcript.splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def read_paths(transcript):
    """Every path a Read or path-scoped Grep touched, in order."""
    paths = []
    for ev in events(transcript):
        if ev.get("type") != "assistant":
            continue
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") != "tool_use":
                continue
            inp = block.get("input", {})
            if block["name"] == "Read":
                paths.append(inp.get("file_path", ""))
            elif block["name"] == "Grep":
                paths.append(inp.get("path", ""))
    return paths


def result_texts(transcript):
    """Text of every tool_result: a Grep over a directory shows a shard when its file:line prefix appears."""
    texts = []
    for ev in events(transcript):
        if ev.get("type") != "user":
            continue
        content = ev.get("message", {}).get("content", [])
        for block in content if isinstance(content, list) else []:
            body = block.get("content") if block.get("type") == "tool_result" else None
            if isinstance(body, list):
                body = " ".join(part.get("text", "") for part in body)
            if isinstance(body, str):
                texts.append(body)
    return texts


def final_answer(transcript):
    for ev in reversed(list(events(transcript))):
        if ev.get("type") == "result":
            return ev.get("result", "") or ""
    return ""


def score(case, transcript):
    answer = final_answer(transcript)
    paths = read_paths(transcript)
    shard_read = (any(s in p for p in paths for s in case.get("shard", []))
                  or any(re.search(re.escape(s) + r"[\w.\-/]*?[:\-]\d+[:\-]", t)
                         for t in result_texts(transcript) for s in case.get("shard", [])))
    correct = all(re.search(rx, answer, re.S | re.I) for rx in case["answer"])
    return {"correct": correct, "shard": shard_read, "reads": len(paths), "answer_len": len(answer)}


def run_one(arm, tree, case, rep, out_dir, model):
    t0 = time.time()
    try:
        p = subprocess.run(claude_cmd(case["prompt"], model), cwd=tree, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=TIMEOUT_S)
        transcript = p.stdout.decode(errors="replace")
        err = p.stderr.decode(errors="replace")[-300:] if p.returncode else None
    except subprocess.TimeoutExpired as e:
        transcript, err = (e.stdout or b"").decode(errors="replace"), "timeout"
    (out_dir / f"{arm}-{case['id']}-{rep}.jsonl").write_text(transcript)
    return {"arm": arm, "case": case["id"], "rep": rep, "control": case.get("control", False),
            "secs": round(time.time() - t0), "error": err, **score(case, transcript)}


def rate(rs, key):
    return f"{sum(r[key] for r in rs)}/{len(rs)}" if rs else "-"


def table(results, cases, arms):
    print(f"{'case':<20}" + "".join(f"{a + ' ok':>10}{a + ' shard':>12}" for a in arms))
    for c in cases:
        cells = ""
        for a in arms:
            rs = [r for r in results if r["arm"] == a and r["case"] == c["id"]]
            cells += f"{rate(rs, 'correct'):>10}{rate(rs, 'shard'):>12}"
        print(f"{c['id'] + (' (ctl)' if c.get('control') else ''):<20}{cells}")


def gate(results, arms):
    old, new = arms
    pick = lambda a, ctl: [r for r in results if r["arm"] == a and r["control"] == ctl and not r["error"]]
    ok = lambda rs: sum(r["correct"] for r in rs) / max(len(rs), 1)
    new_probes = pick(new, False)
    shard = sum(r["shard"] for r in new_probes) / max(len(new_probes), 1)
    checks = {
        f"correct {new} >= {old}": ok(pick(new, False)) >= ok(pick(old, False)),
        f"controls {new} >= {old}": ok(pick(new, True)) >= ok(pick(old, True)),
        f"shard-read {new} >= 0.90": shard >= 0.90,
    }
    for name, passed in checks.items():
        print(("PASS " if passed else "FAIL ") + name)
    print(f"correct: {old} {ok(pick(old, False)):.2f}  {new} {ok(pick(new, False)):.2f}  shard-read {new} {shard:.2f}")
    return all(checks.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="origin/main")
    ap.add_argument("--new", default="HEAD")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--cases", type=Path, default=HERE / "agents_shard_cases.json")
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "shard-eval-results")
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text())
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    out_dir = args.out / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    trees = {"old": export_arm(args.old), "new": export_arm(args.new)}
    jobs = [(a, trees[a], c, r) for a in ("old", "new") for c in cases for r in range(args.reps)]
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda j: run_one(*j, out_dir, args.model), jobs))
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    table(results, cases, ["old", "new"])
    passed = gate(results, ["old", "new"])
    print(f"\n{out_dir}  errors: {sum(bool(r['error']) for r in results)}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
