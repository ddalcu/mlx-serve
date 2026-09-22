#!/usr/bin/env python3
"""Record src/fixtures/mtp_accept_traces.txt: per request, accepted drafts per round at forced depth 6.

Boot the server first, one request at a time, nothing else on the GPU:
  MLX_SERVE_MTP_FORCE_DEPTH=6 zig-out/bin/mlx-serve --model <27B pack> --serve --port 11260 \
      --mtp --prefix-cache-entries 0 --log-level debug > server.log 2>&1
usage: record_mtp_accept_traces.py server.log out.txt [url]
The replay's `hitProb` table came from aligning forced-depth 1 and 3 runs of the code prompts
with the depth-6 run ([mtp-round] t1/drafts/accepted) over their common text prefix.
"""
import json, os, re, sys, urllib.request

CODE = [
    "Write a Python function that parses an ISO 8601 duration string like P3DT4H12M into total seconds. Include tests.",
    "Implement an LRU cache in TypeScript with get, put and a max size, O(1) operations. Explain briefly.",
    "Write a Go HTTP handler that accepts a JSON body with a list of integers and returns their median and mode.",
    "Write a Rust function that merges overlapping intervals in a Vec<(i64, i64)> and returns the result sorted.",
    "Implement a trie in Java with insert, search and startsWith. Include a small main method that exercises it.",
    "Write a C function that reverses a singly linked list in place, plus a test harness that prints the list.",
    "Write a Python class for a token bucket rate limiter with a thread-safe acquire method. Include tests.",
    "Write a SQL schema and three queries for a library lending system: books, members, loans, overdue report.",
]
PROSE = [
    "Write a short essay about the history of quantum computing.",
    "Explain how a compiler turns source code into machine code.",
    "Describe the water cycle for a graduate seminar.",
    "Summarize the causes of the French revolution.",
    "Write a short essay about the rise and fall of the Roman republic.",
    "Explain how vaccines train the immune system, for a general audience.",
    "Describe how the printing press changed European society.",
    "Discuss the economic arguments for and against free trade.",
]
PASSAGE = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
           "measures the throughput of a speculative decoder on a laptop that is "
           "plugged into the wall and not running on battery power.")
GCD = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\ndef fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n < 2: return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]\n\ndef reverse_string(s):\n    out = ''\n    for ch in s:\n        out = ch + out\n    return out"
LOG = "".join("Section %d. In this section the maintenance log records routine checks of the cooling loop, the firmware revision of each sensor node, the calibration offsets applied on the last inspection, and the names of the technicians who signed off on the work before the shift change.\n" % i for i in range(6))
ECHO = [
    "Repeat the following text back to me exactly, three times in a row:\n\n" + PASSAGE,
    "Repeat the following text back to me exactly, twenty times in a row:\n\n" + PASSAGE,
    "Repeat the following code block back EXACTLY as written, no commentary: " + GCD,
    "Repeat the following code block back EXACTLY as written, five times, no commentary:\n\n" + GCD,
    "Copy the following log back to me verbatim, with no changes and no commentary:\n\n" + LOG,
    "Copy the following log back to me verbatim twice, with no changes and no commentary:\n\n" + LOG,
    "Here is a function. Return it unchanged except rename the variable `out` to `result`:\n\n" + GCD,
    "Translate nothing; just output this paragraph ten times, one per line:\n\n" + PASSAGE,
]
CLASSES = {"code": CODE, "prose": PROSE, "echo": ECHO}

LOG, OUT = sys.argv[1], sys.argv[2]
URL = (sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:11260") + "/v1/chat/completions"


def ask(prompt, max_tokens=1024, echo=False):
    body = {"model": "mlx-serve", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "stream": False}
    if not echo:
        body["reasoning_effort"] = "medium"
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


ask("Say hi.", 32)
rx = re.compile(r"\[mtp-round\].* m=(\d+)/(\d+) .*accepted=(\d+)")
with open(OUT, "w") as out:
    for cls, prompts in CLASSES.items():
        for p in prompts:
            off = os.path.getsize(LOG)
            r = ask(p, echo=(cls == "echo"))
            with open(LOG, "rb") as f:
                f.seek(off)
                chunk = f.read().decode("utf-8", "replace")
            digits = "".join(m.group(3) for m in rx.finditer(chunk) if m.group(1) == "6")
            print(cls, r["usage"]["completion_tokens"], "tok", len(digits), "rounds", flush=True)
            out.write(f"{cls} {digits}\n")
