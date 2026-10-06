#!/usr/bin/env python3
"""Score pi agent runs against mlx-serve: early ends, leaks, tool-call health, prefix reuse.

  analyze_pi_sessions.py CELL [CELL...] [--json out.json]

A CELL is a directory (or one session .jsonl). In a directory every *.jsonl under it is a pi
session, `server.log` is the server log, `rawdump.txt` the MLX_SERVE_RAW_DUMP_FILE capture (its
tools JSON is the schema tool arguments are checked against) and `cell.json` / `outcome.json`
are what tests/glm_pi_matrix.sh writes. Missing files only drop their checks. Stdlib only.
"""

import glob
import json
import os
import re
import sys
from collections import Counter

MARKUP = re.compile(r"<think>|</think>|<tool_call>|</tool_call>|<arg_key>|<arg_value>|<\|")
RAW_HDR = re.compile(rb"\n===MLX_RAW_DUMP tools=(\d+) raw=(\d+)===\n")
REUSED = re.compile(r"\[hot-cache\] reused (\d+)/(\d+) tokens")
LOG_COUNTS = {
    "loop_stop": r"\[loop-stop\]",
    "jinja_error": r"jinja error:|jinja render failed",
    "mlx_error": r"\[mlx\] error|error latched|panic|Segmentation",
    "admission_refused": r"\[admission\].*verdict=(?!admit)",
    "hybrid_miss": r"\[hot-cache\] hybrid miss",
}


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def tool_schemas(rawdump):
    """name -> JSON schema, from the last tools JSON the server saw."""
    if not os.path.exists(rawdump):
        return {}
    data = open(rawdump, "rb").read()
    tools = b""
    for m in RAW_HDR.finditer(data):
        t = int(m.group(1))
        if t:
            tools = data[m.end():m.end() + t]
    try:
        return {t["function"]["name"]: t["function"].get("parameters") or {} for t in json.loads(tools)}
    except (ValueError, KeyError, TypeError):
        return {}


JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
              "array": list, "object": dict}


def schema_errors(args, schema):
    """Shallow check: required keys present, no undeclared keys, top-level types match."""
    errs = []
    props = schema.get("properties") or {}
    for k in schema.get("required") or []:
        if k not in args:
            errs.append(f"missing {k}")
    for k, v in args.items():
        if k not in props:
            if schema.get("additionalProperties") is False:
                errs.append(f"undeclared {k}")
            continue
        t = props[k].get("type")
        py = JSON_TYPES.get(t) if isinstance(t, str) else None
        if py and (not isinstance(v, py) or (t in ("integer", "number") and isinstance(v, bool))):
            errs.append(f"{k}: {type(v).__name__} for {t}")
    return errs


def analyze_session(path, schemas):
    entries = load_jsonl(path)
    levels = [e.get("thinkingLevel") for e in entries if e.get("type") == "thinking_level_change"]
    msgs = [e["message"] for e in entries if e.get("type") == "message" and isinstance(e.get("message"), dict)]
    r = {"session": path, "thinking_level": levels[-1] if levels else None, "turns": 0,
         "stop": Counter(), "empty_turns": [], "think_only_turns": [], "thinking_turns": 0,
         "thinking_chars": 0, "text_chars": 0, "args_chars": 0, "leaks": [], "tool_calls": 0,
         "tool_arg_errors": [], "tool_results": 0, "tool_errors": 0, "cache_turns": 0,
         "cache_hit_turns": 0, "user_turns": 0}
    asst_i = 0
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "user":
            r["user_turns"] += 1
        elif role == "toolResult":
            r["tool_results"] += 1
            r["tool_errors"] += bool(m.get("isError"))
        if role != "assistant":
            continue
        asst_i += 1
        r["turns"] += 1
        stop = m.get("stopReason") or "?"
        r["stop"][stop] += 1
        content = m.get("content") or []
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        thinking = "".join(c.get("thinking", "") for c in content if c.get("type") == "thinking")
        calls = [c for c in content if c.get("type") == "toolCall"]
        where = f"turn {asst_i} (msg {i}, stop={stop})"
        if not text.strip() and not calls and stop not in ("error", "aborted"):
            (r["think_only_turns"] if thinking.strip() else r["empty_turns"]).append(where)
        if thinking.strip():
            r["thinking_turns"] += 1
        r["thinking_chars"] += len(thinking)
        r["text_chars"] += len(text)
        for label, s in (("text", text), ("thinking", thinking)):
            hit = MARKUP.search(s)
            # A thought may quote the format it reasons about; only the split tags are a leak there.
            if hit and (label == "text" or hit.group(0) in ("<think>", "</think>")):
                r["leaks"].append(f"{where}: {label} has {hit.group(0)!r} near {s[max(0, hit.start() - 40):hit.end() + 40]!r}")
        for c in calls:
            r["tool_calls"] += 1
            args = c.get("arguments")
            blob = json.dumps(args) if not isinstance(args, str) else args
            r["args_chars"] += len(blob)
            hit = MARKUP.search(blob)
            # File content legitimately carries markup only when the task is about it; report, don't judge.
            if hit:
                r["leaks"].append(f"{where}: {c.get('name')} args have {hit.group(0)!r}")
            if not isinstance(args, dict):
                r["tool_arg_errors"].append(f"{where}: {c.get('name')} args not an object: {blob[:120]}")
            elif c.get("name") not in schemas and schemas:
                r["tool_arg_errors"].append(f"{where}: undeclared tool {c.get('name')!r}")
            elif c.get("name") in schemas:
                for e in schema_errors(args, schemas[c["name"]]):
                    r["tool_arg_errors"].append(f"{where}: {c['name']} {e}")
        usage = m.get("usage") or {}
        if asst_i >= 2:
            r["cache_turns"] += 1
            r["cache_hit_turns"] += (usage.get("cacheRead") or 0) > 0
    last = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
    r["final_stop"] = (last or {}).get("stopReason")
    r["stop"] = dict(r["stop"])
    return r


def analyze_cell(cell):
    if os.path.isfile(cell):
        files, root = [cell], os.path.dirname(cell)
    else:
        files = sorted(glob.glob(os.path.join(cell, "**", "*.jsonl"), recursive=True))
        root = cell
    schemas = tool_schemas(os.path.join(root, "rawdump.txt"))
    sessions = [analyze_session(f, schemas) for f in files]
    out = {"cell": cell, "sessions": sessions, "schemas_from_dump": sorted(schemas)}
    for key, name in (("meta", "cell.json"), ("outcome", "outcome.json")):
        p = os.path.join(root, name)
        if os.path.exists(p):
            try:
                out[key] = json.load(open(p))
            except ValueError:
                out[key] = {"error": f"unreadable {name}"}
    log = os.path.join(root, "server.log")
    if os.path.exists(log):
        text = open(log, errors="replace").read()
        out["log"] = {k: len(re.findall(p, text)) for k, p in LOG_COUNTS.items()}
        reused = [(int(a), int(b)) for a, b in REUSED.findall(text)]
        out["log"]["requests_reused"] = sum(1 for a, _ in reused if a > 0)
        out["log"]["requests_logged"] = len(reused)
        out["log"]["reused_tokens"] = sum(a for a, _ in reused)
        out["log"]["prompt_tokens"] = sum(b for _, b in reused)
    out["summary"] = summarize(out)
    return out


def summarize(c):
    s = c["sessions"]
    tot = lambda k: sum(x[k] if isinstance(x[k], int) else len(x[k]) for x in s)
    stop = Counter()
    for x in s:
        stop.update(x["stop"])
    chars = tot("thinking_chars") + tot("text_chars") + tot("args_chars")
    level = (c.get("meta") or {}).get("thinking") or next((x["thinking_level"] for x in s if x["thinking_level"]), None)
    sm = {
        "sessions": len(s), "thinking_level": level, "turns": tot("turns"), "stop": dict(stop),
        "empty_turns": tot("empty_turns"), "think_only_turns": tot("think_only_turns"),
        "thinking_turns": tot("thinking_turns"),
        "thinking_share": round(tot("thinking_chars") / chars, 3) if chars else 0.0,
        "leaks": tot("leaks"), "tool_calls": tot("tool_calls"), "tool_arg_errors": tot("tool_arg_errors"),
        "tool_error_rate": round(tot("tool_errors") / tot("tool_results"), 3) if tot("tool_results") else 0.0,
        "cache_hit_rate": round(tot("cache_hit_turns") / tot("cache_turns"), 3) if tot("cache_turns") else None,
        "final_stops": [x["final_stop"] for x in s],
    }
    # Thinking at level off is a protocol finding; a thinking level whose turns all close the
    # thought at once is the checkpoint's choice (GLM at low effort), reported, not judged.
    if level:
        sm["thinking_separation_ok"] = level != "off" or sm["thinking_turns"] == 0
    sm["outcome_pass"] = (c.get("outcome") or {}).get("pass")
    if "log" in c:
        sm.update({f"log_{k}": v for k, v in c["log"].items() if k in LOG_COUNTS})
    bad = []
    if sm["empty_turns"]: bad.append("empty turns")
    if sm["think_only_turns"]: bad.append("think-only turns")
    if sm["leaks"]: bad.append("markup leak")
    if sm["tool_arg_errors"]: bad.append("tool arg errors")
    if stop.get("length"): bad.append("length stops")
    if stop.get("error"): bad.append("error stops")
    if sm.get("thinking_separation_ok") is False: bad.append("thinking separation")
    cache_off = "--prefix-cache-entries 0" in ((c.get("meta") or {}).get("flags") or "")
    if sm["cache_hit_rate"] is not None and sm["cache_hit_rate"] < 0.5 and not cache_off: bad.append("low prefix reuse")
    for k in ("log_jinja_error", "log_mlx_error", "log_admission_refused"):
        if sm.get(k): bad.append(k[4:])
    if sm["outcome_pass"] is False: bad.append("task test failed")
    sm["findings"] = bad
    return sm


def table(cells):
    cols = ["cell", "think", "turns", "stops", "empty", "thinkOnly", "think%", "leaks", "calls",
            "argErr", "toolErr", "cache", "loop", "jinja", "mlx", "task", "findings"]
    rows = []
    for c in cells:
        s = c["summary"]
        rows.append([os.path.basename(c["cell"].rstrip("/")), s["thinking_level"] or "-", s["turns"],
                     " ".join(f"{k}:{v}" for k, v in sorted(s["stop"].items())), s["empty_turns"],
                     s["think_only_turns"], f"{100 * s['thinking_share']:.0f}", s["leaks"], s["tool_calls"],
                     s["tool_arg_errors"], f"{100 * s['tool_error_rate']:.0f}%",
                     "-" if s["cache_hit_rate"] is None else f"{100 * s['cache_hit_rate']:.0f}%",
                     s.get("log_loop_stop", "-"), s.get("log_jinja_error", "-"), s.get("log_mlx_error", "-"),
                     {True: "pass", False: "FAIL", None: "-"}[s["outcome_pass"]],
                     ", ".join(s["findings"]) or "ok"])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    cells = [analyze_cell(c) for c in args]
    print(table(cells))
    for c in cells:
        for x in c["sessions"]:
            for k in ("empty_turns", "think_only_turns", "leaks", "tool_arg_errors"):
                for item in x[k]:
                    print(f"  {os.path.basename(c['cell'].rstrip('/'))} {k}: {item}  [{x['session']}]")
    if out:
        with open(out, "w") as f:
            json.dump(cells, f, indent=1, default=str)


if __name__ == "__main__":
    main()
