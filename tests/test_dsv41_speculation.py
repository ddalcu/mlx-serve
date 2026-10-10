#!/usr/bin/env python3
"""Check speculation configuration and API arming against an already-running server."""

import argparse
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlencode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", required=True)
parser.add_argument("--log", required=True, type=Path)
parser.add_argument("--expect-enabled", required=True, choices=("0", "1"))
parser.add_argument("--model")
parser.add_argument("--native", action="store_true", help="require mlx-stream native metadata")
args = parser.parse_args()
base = args.url.rstrip("/")
enabled = args.expect_enabled == "1"


def get(path):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return json.load(response)


model = args.model or get("/v1/models")["data"][0]["id"]
mtp = get("/props?" + urlencode({"model": model}))["settings"]["mtp"]
assert mtp["default_on"] is enabled, mtp
if args.native or "native" in mtp:
    native = mtp["native"]
    assert mtp["loaded"] and native["block_size"] > 0, mtp
    assert "typical" in native["lane"], native
    assert native["sampled_acceptance"] == "stochastic", native
    for key in ("acceptance", "acceptance_param", "greedy_tail", "depth", "adaptive"):
        assert mtp[key] is None, mtp

prompt = "List five rivers and a fact about each."
bodies = {
    "/v1/completions": {"prompt": prompt},
    "/v1/chat/completions": {"messages": [{"role": "user", "content": prompt}]},
    "/v1/messages": {"messages": [{"role": "user", "content": prompt}]},
    "/v1/responses": {"input": prompt, "max_output_tokens": 32},
}
for path, body in bodies.items():
    for stream in (False, True):
        for flag in (None, True, False, True):
            request = dict(body, model=model, max_tokens=32, temperature=0,
                           stream=stream, enable_pld=True, enable_thinking=False)
            if flag is not None:
                request["enable_mtp"] = flag
            before = args.log.read_text().count("spec=dspark (")
            req = urllib.request.Request(base + path, json.dumps(request).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as response:
                result = response.read().decode()
            if stream:
                events = [json.loads(line[6:]) for line in result.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                assert events and all(event.get("error") is None and
                                      event.get("type") not in ("error", "response.failed")
                                      for event in events), result
                if path == "/v1/responses":
                    terminal = [event["response"] for event in events
                                if event.get("type") in ("response.completed", "response.incomplete")]
                    assert terminal and terminal[-1].get("error") is None, result
                    assert terminal[-1]["status"] in ("completed", "incomplete"), result
                elif path == "/v1/messages":
                    assert any(event.get("type") == "message_stop" for event in events), result
                    assert any(event.get("type") == "message_delta" and
                               event.get("delta", {}).get("stop_reason") in
                               ("end_turn", "max_tokens", "tool_use", "stop_sequence", "pause_turn", "refusal")
                               for event in events), result
                else:
                    assert "data: [DONE]" in result.splitlines(), result
                    assert any(choice.get("finish_reason") in ("stop", "length", "tool_calls")
                               for event in events for choice in event.get("choices", [])), result
            else:
                payload = json.loads(result)
                assert payload.get("error") is None, result
                if path == "/v1/responses":
                    assert payload["status"] in ("completed", "incomplete"), result
            after = args.log.read_text().count("spec=dspark (")
            if enabled and flag is not False:
                assert after > before, (path, stream, flag, "native lane did not arm")
            else:
                assert after == before, (path, stream, flag, "opt-out drafted")
            print(f"PASS: {path} stream={stream} enable_mtp={flag}")
print("PASS: speculation metadata and all API controls")
