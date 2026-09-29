#!/usr/bin/env python3
"""Run one pi task headless, compacting between turns like a person at the TUI would.

  pi_rpc.py PROMPT_FILE [pi args...] > final.txt

`pi -p` only checks compaction after the whole task, so a long task dies at the context window.
This drives `pi --mode rpc` and compacts after any turn past contextWindow - reserveTokens, then
tells the agent to continue. Both numbers come from the config in $PI_CODING_AGENT_DIR (written by
`mlx-serve launch pi`), the same values pi itself compacts on. Prints the final assistant text;
exits 1 if the run ends on an error.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

MAX_COMPACTIONS = int(os.environ.get("PI_MAX_COMPACTIONS", 8))
CONTINUE = "Context was compacted. Continue the task from where you left off."
SETTLE_S = 5  # after agent_end, how long pi's own compaction gets to start before the run is done


def compact_threshold(model):
    """contextWindow - reserveTokens for `model`, from the pi config that run.sh generated."""
    cfg = os.environ["PI_CODING_AGENT_DIR"]
    with open(os.path.join(cfg, "models.json")) as f:
        models = [m for p in json.load(f)["providers"].values() for m in p["models"]]
    with open(os.path.join(cfg, "settings.json")) as f:
        reserve = json.load(f)["compaction"]["reserveTokens"]
    return next(m["contextWindow"] for m in models if m["id"] == model) - reserve


def context_tokens(usage):
    return usage.get("totalTokens") or sum(usage.get(k, 0) for k in ("input", "output", "cacheRead", "cacheWrite"))


def final_text(messages):
    for m in reversed(messages):
        if m.get("role") == "assistant":
            text = "".join(c.get("text", "") for c in m.get("content", []) if c.get("type") == "text")
            return text, m.get("stopReason", "")
    return "", ""


def main():
    prompt = open(sys.argv[1]).read()
    threshold = compact_threshold(sys.argv[sys.argv.index("--model") + 1])
    proc = subprocess.Popen(["pi", "--mode", "rpc", *sys.argv[2:]], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(cmd):
        proc.stdin.write(json.dumps(cmd) + "\n")
        proc.stdin.flush()

    send({"type": "prompt", "message": prompt})
    ours = 0
    compacting = False  # one of ours is in flight; the aborted run's agent_end is not the end
    final = ("", "")
    idle_since = None
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)], daemon=True).start()
    while True:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            if idle_since and time.time() - idle_since > SETTLE_S:
                break
            continue
        if line is None:
            break
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        t = ev.get("type")
        if t == "turn_end":
            usage = (ev.get("message") or {}).get("usage") or {}
            if (not compacting and ev.get("toolResults") and ours < MAX_COMPACTIONS
                    and context_tokens(usage) > threshold):
                compacting = True
                ours += 1
                print(f"[pi_rpc] compacting at {context_tokens(usage)} tokens ({ours})", file=sys.stderr)
                send({"type": "compact"})
        elif t == "compaction_start":
            idle_since = None
        elif t == "compaction_end":
            if compacting:
                compacting = False
                send({"type": "prompt", "message": CONTINUE})
            elif not ev.get("willRetry"):
                idle_since = time.time()
        elif t == "agent_start":
            idle_since = None
        elif t == "agent_end" and not compacting:
            final = final_text(ev.get("messages") or [])
            idle_since = time.time()
    proc.stdin.close()
    proc.terminate()
    text, stop = final
    print(text)
    print(f"[pi_rpc] compactions={ours} stop={stop}", file=sys.stderr)
    return 1 if stop in ("error", "aborted") else 0


if __name__ == "__main__":
    sys.exit(main())
