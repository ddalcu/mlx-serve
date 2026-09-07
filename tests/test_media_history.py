"""Live Qwen multimodal cache regression against an already-running server.

No downloads, server restarts, or model changes. Logs contain synthetic data.
Run: python3 tests/test_media_history.py --url http://127.0.0.1:11429 --output DIR
"""
import argparse
import base64
import copy
import json
from pathlib import Path
import re
import struct
import time
import urllib.error
import urllib.request
import zlib


def image(rgb):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0)
    pixels = (b"\0" + bytes(rgb) * 128) * 128
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def media_turn(url, question):
    return {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": question},
    ]}


def ask(base, model, messages, mtp=True):
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": 24, "enable_thinking": False, "enable_mtp": mtp,
            "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.monotonic()
    first = None
    text = ""
    usage = {}
    done = False
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            value = line[6:].strip()
            if value == b"[DONE]":
                done = True
                break
            event = json.loads(value)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                part = delta.get("content") or ""
                if first is None and (part or delta.get("reasoning_content")):
                    first = time.monotonic() - start
                text += part
    assert done, "SSE ended without DONE"
    assert usage.get("prompt_tokens"), "Missing usage"
    return {"answer": text, "ttft_seconds": first, "seconds": time.monotonic() - start,
            "prompt": usage["prompt_tokens"], "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="mlx-serve")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-mtp", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    with (args.output / "results.jsonl").open("x") as log:
        def run(name, messages, color=None, min_cached=None, max_cached=None):
            result = ask(args.url.rstrip("/"), args.model, messages, not args.no_mtp)
            result["case"] = name
            checks = {}
            if color:
                checks["pixels"] = bool(re.search(r"\b" + color + r"\b", result["answer"], re.I))
            if min_cached is not None:
                checks["reuse"] = result["cached"] >= min_cached
            if max_cached is not None:
                checks["invalidation"] = result["cached"] <= max_cached
            result["checks"] = checks
            if not all(checks.values()):
                failures.append(name)
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(json.dumps(result), flush=True)
            return result

        red, blue, yellow = image((255, 0, 0)), image((0, 0, 255)), image((255, 255, 0))
        messages = [{"role": "system", "content": "Inspect the actual images. Answer with the requested color only. Do not reason."}]
        messages.append(media_turn(red, "What color is this image?"))
        first = run("first-image", messages, "red")
        run("exact-repeat", messages, "red", min_cached=max(1, first["prompt"] - 1024))
        messages.append({"role": "assistant", "content": "Received."})
        messages.append({"role": "user", "content": "What color was the first image?"})
        run("ordinary-followup", messages, "red")
        for part in range(3):
            call = f"archive-{part}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": call, "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
            archive = "".join(f"Record {part}.{i}: An ordinary district recorded a routine inventory delivery.\n" for i in range(400))
            messages.append({"role": "tool", "tool_call_id": call, "content": archive})
            messages.append({"role": "user", "content": "Stop reading the archive. Look at the FIRST attached image and answer with its color only."})
            previous = run(f"tool-expansion-{part}", messages, "red")
        messages.append(media_turn(blue, "What color is the NEWEST image?"))
        second = run("append-second-image", messages, "blue", min_cached=previous["prompt"] - 2048)
        messages.append({"role": "assistant", "content": "Received."})
        messages.append({"role": "user", "content": "What color was the FIRST image?"})
        run("long-historical-recall", messages, "red", min_cached=second["prompt"] - 2048)
        edited = copy.deepcopy(messages)
        edited[1]["content"][0]["image_url"]["url"] = yellow
        run("edit-first-image", edited, "yellow", max_cached=first["prompt"])
        run("return-to-original", messages, "red")
        # Multiple images in one message require separate M-RoPE blocks.
        multi = [messages[0], media_turn(red, "What color is the second image?")]
        multi[1]["content"].insert(1, {"type": "image_url", "image_url": {"url": blue}})
        run("two-images-one-turn", multi, "blue")
        multi[1]["content"][0], multi[1]["content"][1] = multi[1]["content"][1], multi[1]["content"][0]
        run("reordered-images", multi, "red")
        multi[1]["content"].pop(0)
        multi[1]["content"][-1]["text"] = "What color is the only image?"
        run("removed-image", multi, "red")
        empty = [messages[0], media_turn(red, "Inspect this."), {"role": "assistant", "content": ""},
                 {"role": "user", "content": "What color was the image?"}]
        run("empty-assistant-boundary", empty, "red")
        for name, bad in (("invalid-base64", media_turn("data:image/png;base64,not-an-image", "Inspect this.")),
                          ("missing-image-url", {"role": "user", "content": [{"type": "image_url", "image_url": {}}]})):
            try:
                ask(args.url.rstrip("/"), args.model, [bad], not args.no_mtp)
            except urllib.error.HTTPError as error:
                body = json.loads(error.read())
                ok = error.code == 400 and bool(body.get("error"))
            else:
                ok = False
            result = {"case": name, "checks": {"rejected": ok}}
            if not ok:
                failures.append(name)
            log.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
    assert not failures, "Failed cases: " + ", ".join(failures)


if __name__ == "__main__":
    main()
