#!/usr/bin/env python3
"""Stream-measured decode rate for bench.sh.

Usage: printf '%s' "$body" | python3 _decode_stream.py <chat_completions_url> [timeout_s]

Reads an OpenAI chat-completions request body on stdin, forces stream:true,
and measures decode tok/s from the SSE stream itself: counts content +
reasoning delta pieces and times first->last delta. This does NOT depend on
the server's usage/token accounting (LM Studio reports those inconsistently
once reasoning is involved) and it excludes prefill entirely. Engines stream
one delta per token on the content path (PLD/drafter accepted tokens also
stream one delta each), so speculative speedups show up correctly.

Output (single line, pipe-separated): rate_tps|n_pieces|n_reasoning_pieces
  rate_tps            (n_pieces-1) / (t_last - t_first), 1 decimal
  n_pieces            content + reasoning delta count (~completion tokens)
  n_reasoning_pieces  reasoning-only delta count (>0 = thinking leaked when
                      the request asked for thinking off)
On any failure prints: ERR|0|0
"""
import json
import sys
import time
import urllib.request


def main():
    url = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0

    body = json.loads(sys.stdin.read())
    body["stream"] = True
    # Ask for usage on the final chunk (OpenAI stream_options). Engines that
    # batch several tokens into one SSE delta (stream-interval > 1) would
    # otherwise read low from piece-counting alone; usage_ct below corrects
    # that. Engines that ignore stream_options keep the piece-count path.
    body.setdefault("stream_options", {"include_usage": True})

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
    )

    first = last = None
    n = 0
    rn = 0
    usage_ct = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            u = obj.get("usage") or {}
            if u.get("completion_tokens"):
                usage_ct = int(u["completion_tokens"])
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            got = False
            if delta.get("content"):
                n += 1
                got = True
            if delta.get("reasoning_content") or delta.get("reasoning"):
                rn += 1
                got = True
            if got:
                now = time.monotonic()
                if first is None:
                    first = now
                last = now

    total = n + rn
    # Prefer the engine's own completion_tokens when it exceeds the piece
    # count (deltas were batched); keep the piece count as the floor (LM
    # Studio's usage undercounts once reasoning is involved).
    if usage_ct > total:
        total = usage_ct
    if first is None or last is None or total < 2 or last <= first:
        print("0|%d|%d" % (total, rn))
        return
    rate = (total - 1) / (last - first)
    print("%.1f|%d|%d" % (rate, total, rn))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ERR|0|0")


