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


class State:
    def __init__(self, threshold):
        self.threshold = threshold
        self.ours = 0  # compactions this script followed with CONTINUE
        self.compacting = False  # a compaction is running; the aborted run's agent_end is not the end
        self.final = ("", "")
        self.idle_since = None


def on_event(st, ev):
    """The commands one pi event calls for. pi compacts on its own too, and a manual compact aborts
    whatever runs, so only a compaction that FINISHED is followed by CONTINUE."""
    t = ev.get("type")
    if t == "turn_end":
        usage = (ev.get("message") or {}).get("usage") or {}
        if (not st.compacting and ev.get("toolResults") and st.ours < MAX_COMPACTIONS
                and context_tokens(usage) > st.threshold):
            st.compacting = True
            print(f"[pi_rpc] compacting at {context_tokens(usage)} tokens ({st.ours + 1})", file=sys.stderr)
            return [{"type": "compact"}]
    elif t == "compaction_start":
        st.compacting = True
        st.idle_since = None
    elif t == "compaction_end":
        if ev.get("aborted"):
            return []  # superseded by the compaction that aborted it
        st.compacting = False
        if ev.get("willRetry"):
            return []  # pi resumes the turn itself
        if ev.get("errorMessage") or st.ours >= MAX_COMPACTIONS:
            print(f"[pi_rpc] compaction ended: {ev.get('errorMessage') or 'limit reached'}", file=sys.stderr)
            st.idle_since = time.time()
            return []
        st.ours += 1
        return [{"type": "prompt", "message": CONTINUE}]
    elif t == "agent_start":
        st.idle_since = None
    elif t == "agent_end" and not st.compacting:
        st.final = final_text(ev.get("messages") or [])
        st.idle_since = time.time()
    return []


def main():
    prompt = open(sys.argv[1]).read()
    st = State(compact_threshold(sys.argv[sys.argv.index("--model") + 1]))
    proc = subprocess.Popen(["pi", "--mode", "rpc", *sys.argv[2:]], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(cmd):
        proc.stdin.write(json.dumps(cmd) + "\n")
        proc.stdin.flush()

    send({"type": "prompt", "message": prompt})
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)], daemon=True).start()
    while True:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            if st.idle_since and time.time() - st.idle_since > SETTLE_S:
                break
            continue
        if line is None:
            break
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        for cmd in on_event(st, ev):
            send(cmd)
    proc.stdin.close()
    proc.terminate()
    text, stop = st.final
    print(text)
    print(f"[pi_rpc] compactions={st.ours} stop={stop}", file=sys.stderr)
    return 1 if stop in ("error", "aborted") else 0


if __name__ == "__main__":
    sys.exit(main())
