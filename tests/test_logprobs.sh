#!/bin/bash
# test_logprobs.sh — the logprobs wire contract, on every surface that serves it.
#
# Three defects fixed 2026-08-05 (temperature-scaled logits, float-equality id
# recovery, a one-token offset) all reduce to ONE observable bar: at temp 0 the
# emitted token IS the argmax, so rank 1 of top_logprobs must equal it. A broken
# instrument is worse than none — an early "-0.004, therefore 99.6% confident"
# read off this field sent a quantization hunt down the wrong path for a day.
#
# Section [4] pins the class those three did NOT cover: STREAMING chat accepted
# `logprobs`, disabled speculation to honor it (paying the throughput), had the
# generator compute every entry — and then dropped them, because the SSE chunk
# template had no `logprobs` field at all. Silently-ignored-field class. It is
# invisible to llmprobe (which probes logprobs non-streaming only, and scored
# this server 100% on `Logprob consistency` while streaming returned nothing),
# and invisible to any output-equality test, since the deltas are unchanged.
#
#   LOGPROBS_TEST_MODEL=<dir> ./tests/test_logprobs.sh [port]
#
# Any chat model works; defaults to LFM2.5-2.6B-8bit. SKIPs without one.
set -uo pipefail

MODEL="${LOGPROBS_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/LFM2.5-2.6B-8bit}"
PORT="${1:-11293}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: no model at $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first"; exit 1; }

ID="$(basename "$MODEL")"
LOG="$(mktemp)"
pkill -f "mlx-serve --serve.*port $PORT" 2>/dev/null; sleep 1
"$BIN" --serve --model "$MODEL" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f "$LOG"' EXIT

for _ in $(seq 1 180); do curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 || { echo "fail: server never came up"; tail -20 "$LOG"; exit 1; }

python3 - "$BASE" "$ID" <<'PY'
import json, sys, urllib.request
BASE, MODEL = sys.argv[1], sys.argv[2]
fails, checks = [], 0

def ck(name, cond, detail=""):
    global checks
    checks += 1
    print(("  \033[32mPASS\033[0m  " if cond else "  \033[31mFAIL\033[0m  ") + name + ("  " + detail if not cond else ""))
    if not cond: fails.append(name)

# A token is a BPE fragment, so it can carry HALF a multi-byte character — and
# the token STRINGS go out in the JSON. Raw bytes there make the whole body
# invalid UTF-8, i.e. unparseable, not merely degraded. Checked on EVERY
# response this script makes rather than in one place, since which request
# happens to draw a split candidate into its top-5 is luck.
utf8_bad = []

def post(path, body, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    if stream:
        return r
    raw = r.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as e:
        utf8_bad.append((path, e.start, bytes(raw[e.start:e.start + 2])))
        return json.loads(raw.decode("utf-8", "replace"))

MSG = [{"role": "user", "content": "Count from one to eight in words."}]
REQ = {"model": MODEL, "messages": MSG, "max_tokens": 48, "temperature": 0,
       "logprobs": True, "top_logprobs": 5}

def entries_ok(label, content):
    bad_rank1 = [(i, e["token"], e["top_logprobs"][0]["token"])
                 for i, e in enumerate(content)
                 if e.get("top_logprobs") and e["top_logprobs"][0]["token"] != e["token"]]
    ck(f"[{label}] rank 1 == emitted token at temp 0", not bad_rank1,
       f"{len(bad_rank1)}/{len(content)}: {bad_rank1[:3]}")
    bad_ord = [i for i, e in enumerate(content)
               if (v := [t["logprob"] for t in e.get("top_logprobs") or []]) != sorted(v, reverse=True)]
    ck(f"[{label}] top_logprobs descend", not bad_ord, f"{bad_ord[:3]}")
    ck(f"[{label}] logprobs are <= 0",
       all(e["logprob"] <= 1e-6 and all(t["logprob"] <= 1e-6 for t in e.get("top_logprobs") or [])
           for e in content))
    ck(f"[{label}] every entry carries bytes", all("bytes" in e for e in content))

print("── [1/7] chat non-streaming ──")
r = post("/v1/chat/completions", {**REQ, "stream": False})
ns = (r["choices"][0].get("logprobs") or {}).get("content") or []
ck("[non-stream] logprobs present", bool(ns), f"got {r['choices'][0].get('logprobs')}")
if ns:
    entries_ok("non-stream", ns)
    ck("[non-stream] one entry per completion token",
       len(ns) == r["usage"]["completion_tokens"],
       f"{len(ns)} entries vs {r['usage']['completion_tokens']} tokens")

print("── [2/7] temperature must not move the model's own distribution ──")
a = post("/v1/chat/completions", {**REQ, "max_tokens": 1, "stream": False})["choices"][0]["logprobs"]["content"][0]
b = post("/v1/chat/completions", {**REQ, "max_tokens": 1, "temperature": 2.0, "top_k": 1,
                                  "stream": False})["choices"][0]["logprobs"]["content"][0]
ck("same token drawn", a["token"] == b["token"], f"{a['token']!r} vs {b['token']!r}")
ck("its logprob is temperature-INDEPENDENT", abs(a["logprob"] - b["logprob"]) < 1e-3,
   f"temp0={a['logprob']} temp2={b['logprob']}")
ck("distribution not saturated to 0.0 at temp 0",
   any(abs(t["logprob"]) > 1e-4 for t in a["top_logprobs"][1:]),
   f"{[round(t['logprob'],6) for t in a['top_logprobs']]}")

print("── [3/7] /v1/completions: integer logprobs, four parallel arrays ──")
r = post("/v1/completions", {"model": MODEL, "prompt": "Count from one to five:",
                             "max_tokens": 24, "temperature": 0, "logprobs": 5})
lp = r["choices"][0].get("logprobs")
ck("[completions] logprobs present", lp is not None, f"got {lp}")
if lp:
    keys = ("tokens", "token_logprobs", "top_logprobs", "text_offset")
    ck("[completions] all four arrays present", all(lp.get(k) is not None for k in keys), f"{list(lp)}")
    if all(lp.get(k) is not None for k in keys):
        n = len(lp["tokens"])
        ck("[completions] arrays are parallel", all(len(lp[k]) == n for k in keys),
           str({k: len(lp[k]) for k in keys}))
        ck("[completions] top_logprobs is a text-keyed map", isinstance(lp["top_logprobs"][0], dict))
        ck("[completions] text_offset ascends, in range",
           lp["text_offset"] == sorted(lp["text_offset"])
           and all(0 <= o <= len(r["choices"][0]["text"]) for o in lp["text_offset"]))
        bad = [(i, v, max(m.values())) for i, (v, m) in
               enumerate(zip(lp["token_logprobs"], lp["top_logprobs"]))
               if m and abs(max(m.values()) - v) > 1e-4]
        ck("[completions] emitted token is the max of its map at temp 0", not bad, f"{bad[:3]}")

print("── [4/7] STREAMING chat must carry the same logprobs ──")
resp = post("/v1/chat/completions", {**REQ, "stream": True}, stream=True)
st = []
for raw in resp:
    line = raw.decode().strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    for ch in json.loads(line[6:]).get("choices", []):
        c = (ch.get("logprobs") or {}).get("content") or []
        st.extend(c)
ck("[stream] logprobs present", bool(st), "streaming emitted NO logprobs")
if st:
    entries_ok("stream", st)
    # The decisive one: same greedy request, so streaming must agree with
    # non-streaming entry for entry. A partial or duplicated drain fails here.
    ck("[stream] same entry count as non-streaming", len(st) == len(ns), f"{len(st)} vs {len(ns)}")
    n = min(len(st), len(ns))
    tok_mismatch = [i for i in range(n) if st[i]["token"] != ns[i]["token"]]
    ck("[stream] tokens match non-streaming", not tok_mismatch, f"{tok_mismatch[:5]}")
    lp_mismatch = [i for i in range(n) if abs(st[i]["logprob"] - ns[i]["logprob"]) > 1e-6]
    ck("[stream] logprob VALUES match non-streaming", not lp_mismatch, f"{lp_mismatch[:5]}")

print("── [5/7] STREAMING /v1/completions carries the legacy shape ──")
CREQ = {"model": MODEL, "prompt": "Count from one to five:", "max_tokens": 24,
        "temperature": 0, "logprobs": 5}
ns = post("/v1/completions", {**CREQ, "stream": False})["choices"][0]
resp = post("/v1/completions", {**CREQ, "stream": True}, stream=True)
toks, tlp, tops, offs, sawkey, text = [], [], [], [], 0, ""
for raw in resp:
    line = raw.decode().strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    for ch in json.loads(line[6:]).get("choices", []):
        if "logprobs" in ch:
            sawkey += 1
        text += ch.get("text") or ""
        lp = ch.get("logprobs")
        if lp:
            toks += lp["tokens"]; tlp += lp["token_logprobs"]
            tops += lp["top_logprobs"]; offs += lp["text_offset"]
ck("[cmpl-stream] every chunk carries a logprobs key", sawkey > 0, f"{sawkey} chunks had it")
ck("[cmpl-stream] entries delivered", bool(toks), "none")
if toks:
    ck("[cmpl-stream] arrays parallel", len(tlp) == len(toks) == len(tops) == len(offs),
       f"{len(toks)} {len(tlp)} {len(tops)} {len(offs)}")
    # text_offset indexes the WHOLE completion, so it must ascend across chunks —
    # a per-chunk reset (each starting at 0) is the bug this catches.
    ck("[cmpl-stream] text_offset ascends across chunks", offs == sorted(offs) and len(set(offs)) == len(offs),
       f"{offs[:8]}")
    ck("[cmpl-stream] text_offset stays within the text", all(0 <= o <= len(text) for o in offs),
       f"max={max(offs)} len(text)={len(text)}")
    nlp = ns.get("logprobs") or {}
    if not nlp.get("tokens"):
        ck("[cmpl-stream] non-streaming reference available", False,
           "non-streaming completions returned no logprobs to compare against")
    else:
        ck("[cmpl-stream] tokens match non-streaming", toks == nlp["tokens"],
           f"{toks[:5]} vs {nlp['tokens'][:5]}")
        ck("[cmpl-stream] offsets match non-streaming", offs == nlp["text_offset"],
           f"{offs[:6]} vs {nlp['text_offset'][:6]}")
        ck("[cmpl-stream] logprob VALUES match non-streaming",
           len(tlp) == len(nlp["token_logprobs"])
           and all(abs(a - b) < 1e-6 for a, b in zip(tlp, nlp["token_logprobs"])),
           "values differ")

print("── [6/7] logprobs describe message.content, not the reasoning we strip ──")
# OpenAI defines `logprobs.content` as the tokens of the message CONTENT. We
# built it from the whole generation, so on every model that thinks the array
# described the reasoning block — text the client never receives — and entry 0
# was the first token of the thought. Measured on Qwen3.6-27B (3 builds),
# Qwen3.6-35B-A3B, gemma-4-31b, gemma-4-e4b, Qwen3-4B and LFM2.5: 37-186
# entries against 8 characters of content.
TMSG = [{"role": "user", "content": "The capital of Australia is which city? Reply with just the city."}]
TREQ = {"model": MODEL, "messages": TMSG, "max_tokens": 300, "temperature": 0,
        "logprobs": True, "top_logprobs": 5, "enable_thinking": True}
r = post("/v1/chat/completions", {**TREQ, "stream": False})
tch = r["choices"][0]
tcontent = tch["message"].get("content") or ""
treasoning = tch["message"].get("reasoning_content") or ""
tents = (tch.get("logprobs") or {}).get("content") or []
tjoined = "".join(e["token"] for e in tents)
ck("[think non-stream] logprobs present", bool(tents))
if tents:
    ck("[think non-stream] entries reconstruct message.content",
       tjoined.strip() == tcontent.strip(),
       f"{len(tents)} entries -> {tjoined[:44]!r} vs content {tcontent[:44]!r}")
    # Red-on-revert: pre-fix this equalled completion_tokens exactly, because
    # every generated token got an entry whether or not it survived the split.
    if treasoning.strip():
        ck("[think non-stream] reasoning tokens are NOT in the array",
           len(tents) < r["usage"]["completion_tokens"],
           f"{len(tents)} entries vs {r['usage']['completion_tokens']} generated"
           f" ({len(treasoning)} chars of reasoning stripped)")
    else:
        print("  \033[33mNOTE\033[0m  model emitted no reasoning; count check skipped")

sc, sents = "", []
for raw in post("/v1/chat/completions",
                {**TREQ, "stream": True, "stream_options": {"include_usage": True}}, stream=True):
    line = raw.decode().strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    for ch in json.loads(line[6:]).get("choices", []):
        sc += (ch.get("delta") or {}).get("content") or ""
        sents.extend((ch.get("logprobs") or {}).get("content") or [])
sjoined = "".join(e["token"] for e in sents)
ck("[think stream] logprobs present", bool(sents), "streaming emitted NO logprobs")
if sents:
    # Streaming has its own content (the gate trims differently), so it is held
    # to ITS deltas — the invariant is that entries describe what was sent.
    ck("[think stream] entries reconstruct the content deltas",
       sjoined.strip() == sc.strip(),
       f"{len(sents)} entries -> {sjoined[:44]!r} vs deltas {sc[:44]!r}")

print("── [7/7] every response body is valid UTF-8 ──")
ck("no response carried a split multi-byte token as raw bytes", not utf8_bad,
   f"{len(utf8_bad)} bodies failed to decode: {utf8_bad[:2]}")

print(f"\n{checks - len(fails)}/{checks} passed")
if fails:
    print("FAILED:"); [print("  -", f) for f in fails]
sys.exit(1 if fails else 0)
PY
rc=$?
[ $rc -eq 0 ] && echo "✅ logprobs contract holds on all surfaces" || echo "❌ logprobs contract broken"
exit $rc
