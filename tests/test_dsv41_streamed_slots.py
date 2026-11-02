#!/usr/bin/env python3
"""Check two streamed DeepSeek owners, a queued third, and HTTP prefix reuse.

Run against an otherwise idle EXL3 server started by the caller with --metrics,
--max-concurrent 2, --prefill-chunk 8192 and --prefill-decode-share 0:
  python3 tests/test_dsv41_streamed_slots.py --url http://127.0.0.1:11353 \
      --model mlx-serve --output "$HOME/claude-tmp/dsv41-slots.json"
Does not start or stop a server. Serial decode ticks are expected, not batching.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid

from lib.server import wait_health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--request-url", help="generation base URL; defaults to --url")
    parser.add_argument("--api-key-env", default="MLX_SERVE_API_KEY",
                        help="environment variable containing the generation bearer token")
    parser.add_argument("--header", action="append", default=[], metavar="NAME: VALUE",
                        help="additional generation header; repeatable")
    parser.add_argument("--model", default="mlx-serve")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    request_base = (args.request_url or args.url).rstrip("/")
    request_headers = {"Content-Type": "application/json"}
    for header in args.header:
        name, separator, value = header.partition(":")
        if not separator or not name.strip() or "\n" in header or "\r" in header:
            parser.error("--header requires NAME: VALUE without line breaks")
        request_headers[name.strip()] = value.strip()
    token = os.environ.get(args.api_key_env)
    if token:
        request_headers["Authorization"] = "Bearer " + token
    evidence = {"observations": [], "responses": [], "passed": False}

    def get_metrics():
        with urllib.request.urlopen(base + "/metrics.json", timeout=10) as response:
            return json.load(response)

    def live(snapshot):
        return {row["request_id"]: row for row in snapshot["sessions"]
                if row["phase"] in ("prefill", "decode") and row["request_id"] != 0}

    def observe(predicate, description, futures=()):
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            snapshot = get_metrics()
            evidence["observations"].append(snapshot)
            assert len(live(snapshot)) <= 2, "more than two live owners"
            if predicate(snapshot):
                print("PASS:", description, flush=True)
                return snapshot
            for future in futures:
                if future.done():
                    future.result()  # Surface HTTP/model errors immediately.
                    raise AssertionError("request finished before " + description)
            time.sleep(0.25)
        raise AssertionError("timed out: " + description)

    def complete(body):
        request = urllib.request.Request(request_base + "/v1/completions",
                                         json.dumps(body).encode(),
                                         request_headers)
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        assert "error" not in result, result
        choice = result["choices"][0]
        assert choice["finish_reason"] in ("stop", "length"), result
        assert choice["text"].strip(), result
        assert result["usage"]["completion_tokens"] > 0, result
        return result

    try:
        wait_health(base, 30)
        initial = get_metrics()
        assert not live(initial), "server must be idle"
        assert initial["gauges"]["requests_waiting"] == 0, "queue must be empty"
        tag = uuid.uuid4().hex
        bodies = [{"model": args.model,
                   "prompt": "<｜begin▁of▁sentence｜><｜User｜>"
                             f"Request {tag}-{i}. List integers 1 through 500, "
                             "one per line, without skipping any."
                             "<｜Assistant｜></think>",
                   "max_tokens": args.max_tokens, "temperature": 0,
                   "enable_mtp": False, "enable_pld": False}
                  for i in range(3)]
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(complete, bodies[0])
            observe(lambda s: any(r["generated_tokens"] > 0 for r in live(s).values()),
                    "first owner decoded", (first,))
            second = executor.submit(complete, bodies[1])
            pair = observe(lambda s: len(live(s)) == 2 and
                           all(r["generated_tokens"] > 0 for r in live(s).values()),
                           "two distinct owners decoded", (first, second))
            owners = set(live(pair))
            third = executor.submit(complete, bodies[2])
            observe(lambda s: set(live(s)) == owners and
                    s["gauges"]["requests_waiting"] >= 1 and not third.done(),
                    "third waits while both owners remain live", (first, second, third))
            evidence["responses"] = [f.result(timeout=args.timeout)
                                     for f in (first, second, third)]
        observe(lambda s: not live(s) and s["gauges"]["requests_waiting"] == 0,
                "all three requests completed and drained")
        repeated = complete(bodies[2])
        evidence["repeat"] = repeated
        cached = repeated["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert cached > 0, "repeated prompt did not reuse prefix state"
        assert cached <= repeated["usage"]["prompt_tokens"], repeated
        assert repeated["choices"][0]["text"] == evidence["responses"][2]["choices"][0]["text"], \
            "repeated greedy output diverged"
        print("PASS: repeated prompt reused state and retained greedy output", flush=True)
        evidence["passed"] = True
    except Exception as error:
        evidence["error"] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
