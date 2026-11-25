#!/usr/bin/env python3
"""Llama-engine vision end-to-end regression against an ALREADY RUNNING server.

Stdlib-only HTTP client (urllib + struct/zlib/base64 PNGs, no Pillow, no
artifacts written). The server is NOT started, stopped or polled by this
script — it must already be up (mlx-serve directly, or a gateway fronting it).

What it checks, per case (run all by default, or a subset with --case):
  models          /v1/models lists --model with engine "llama", capability
                  "vision" and input modality "image"
  red_blue        same textual prompt, red image then blue image (temperature 0,
                  enable_thinking false, max_tokens 128): the blue answer must
                  name blue, not red — a poisoned image prefix cache would answer red
  stream          green image over streaming SSE: chunks assemble green content,
                  a finish reason arrives, the stream terminates with [DONE]
  two_images      red then blue in one message, "list the colors in order":
                  both colors named, red before blue
  parallel        red and blue concurrently (ThreadPoolExecutor, identical
                  prompts): no cross-contamination under batching
  text_after_image pure-text arithmetic after an image turn still answers 4
  anthropic       /v1/messages base64 image block (green); WARN-skip if the
                  endpoint rejects the block format with a 4xx
  usage           image prompt_tokens exceed the text-only baseline for the same
                  prompt; cached token counts, where provided, are never negative
  malformed       invalid data-URL image bytes get a 4xx (not a hang/crash), and
                  a valid image request still succeeds afterwards

Usage:
  python3 tests/test_llama_vision.py --model <id-from-/v1/models> [--base-url http://127.0.0.1:8000]
                                     [--api-key ...] [--timeout 120] [--case red_blue,stream]

--api-key defaults to $BENCH_API_KEY. The key and image bytes are never
printed; error bodies are truncated and base64-looking runs redacted.
Exit 0 = all checks pass, 1 = any failure.
"""
import argparse
import base64
import concurrent.futures
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
import zlib

COLOR_PROMPT = "What is the main color of this image? Answer with one word only."
LIST_PROMPT = "List the colors of these two images in order, first to last. Answer with just the two color words."
MATH_PROMPT = "What is 2 + 2? Reply with only the number, nothing else."

REDACT_RE = re.compile(r"[A-Za-z0-9+/_=]{80,}")


def solid_png(r, g, b, size=128):
    """Solid-color RGB PNG built from struct/zlib primitives (no Pillow)."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit truecolor RGB
    raw = b"".join(b"\x00" + bytes((r, g, b)) * size for _ in range(size))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


class HttpError(Exception):
    def __init__(self, status, detail):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class Report:
    def __init__(self):
        self.passed = self.failed = 0

    def ok(self, name, detail=""):
        self.passed += 1
        print(f"  PASS: {name}" + (f" — {detail}" if detail else ""))

    def bad(self, name, detail=""):
        self.failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name, detail=""):
        print(f"  WARN: {name} — {detail}")


def clean_text(blob, limit=300):
    """Truncated, base64-scrubbed text safe to print (never keys or image data)."""
    return REDACT_RE.sub("<redacted>", blob)[:limit].replace("\n", " ")


class Client:
    def __init__(self, base_url, api_key, timeout, report):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.report = report
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def request(self, path, body=None):
        """Single JSON request/response. Raises HttpError on 4xx/5xx."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers=self.headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise HttpError(e.code, clean_text(e.read().decode("utf-8", "replace"))) from None
        except urllib.error.URLError as e:
            raise HttpError(0, f"connection failed: {e.reason}") from None
        except (TimeoutError, OSError) as e:
            raise HttpError(0, f"timed out or broken after {self.timeout}s: {e.__class__.__name__}") from None

    def chat(self, content, max_tokens=128, stream=False, **extra):
        body = {"model": ARGS.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "enable_thinking": False, "max_tokens": max_tokens,
                "stream": stream, **extra}
        if stream:
            body["stream_options"] = {"include_usage": True}
        return self.request("/v1/chat/completions", body)

    def chat_stream(self, content, max_tokens=128):
        """SSE chat: returns (text, finish_reason, usage, saw_done)."""
        body = {"model": ARGS.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "enable_thinking": False, "max_tokens": max_tokens,
                "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     headers=self.headers, method="POST")
        text, finish, usage, saw_done = [], None, None, False
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        saw_done = True
                        break
                    obj = json.loads(payload)
                    if "error" in obj:
                        raise HttpError(0, clean_text(json.dumps(obj["error"])))
                    for choice in obj.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            text.append(delta["content"])
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                    if obj.get("usage"):
                        usage = obj["usage"]
        except urllib.error.HTTPError as e:
            raise HttpError(e.code, clean_text(e.read().decode("utf-8", "replace"))) from None
        except (TimeoutError, OSError) as e:
            raise HttpError(0, f"stream broke after {self.timeout}s: {e.__class__.__name__}") from None
        return "".join(text), finish, usage, saw_done

    def check_usage(self, where, usage):
        """Cached counts, where provided, must be sane (never negative, <= prompt)."""
        if not usage:
            return
        prompt = usage.get("prompt_tokens")
        if prompt is not None and prompt < 0:
            self.report.bad(f"{where}: prompt_tokens non-negative", f"got {prompt}")
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if cached is not None and cached < 0:
            self.report.bad(f"{where}: cached_tokens non-negative", f"got {cached}")
        if cached is not None and prompt is not None and cached > prompt:
            self.report.bad(f"{where}: cached_tokens <= prompt_tokens", f"{cached} > {prompt}")


def image_content(png, prompt):
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": prompt}]


def mentions(text, word):
    return re.search(rf"\b{word}\b", text or "", re.IGNORECASE) is not None


def chat_text(response):
    return (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""


# ── cases ────────────────────────────────────────────────────────────────────

def case_models(client, images):
    data = client.request("/v1/models").get("data") or []
    entry = next((m for m in data if m.get("id") == ARGS.model), None)
    if entry is None:
        client.report.bad("model listed in /v1/models",
                          f"{ARGS.model!r} not among {len(data)} entries")
        return
    client.report.ok("model listed in /v1/models")
    meta = entry.get("meta") or {}
    if meta.get("engine") == "llama":
        client.report.ok("meta.engine is llama")
    else:
        client.report.bad("meta.engine is llama", f"got {meta.get('engine')!r}")
    caps = entry.get("capabilities") or []
    if "vision" in caps:
        client.report.ok("capabilities include vision")
    else:
        client.report.bad("capabilities include vision", f"got {caps}")
    mods = entry.get("input_modalities") or []
    if "image" in mods:
        client.report.ok("input_modalities include image")
    else:
        client.report.bad("input_modalities include image", f"got {mods}")


def case_red_blue(client, images):
    red_resp = client.chat(image_content(images["red"], COLOR_PROMPT))
    red = chat_text(red_resp)
    client.check_usage("red_blue red", red_resp.get("usage"))
    if mentions(red, "red"):
        client.report.ok("red image answers red", f"→ {clean_text(red, 80)}")
    else:
        client.report.bad("red image answers red", f"→ {clean_text(red, 120)}")
    resp = client.chat(image_content(images["blue"], COLOR_PROMPT))
    blue = chat_text(resp)
    client.check_usage("red_blue blue", resp.get("usage"))
    if mentions(blue, "blue"):
        client.report.ok("blue image answers blue (identical prompt, cache not poisoned)")
    else:
        client.report.bad("blue image answers blue", f"→ {clean_text(blue, 120)}")
    if mentions(blue, "red"):
        client.report.bad("blue answer free of 'red' (cache-pollution signal)", f"→ {clean_text(blue, 120)}")
    else:
        client.report.ok("blue answer free of 'red'")


def case_stream(client, images):
    text, finish, usage, saw_done = client.chat_stream(image_content(images["green"], COLOR_PROMPT))
    if saw_done:
        client.report.ok("stream terminated with [DONE]")
    else:
        client.report.bad("stream terminated with [DONE]", "no terminator seen")
    if mentions(text, "green"):
        client.report.ok("streamed green content names green", f"→ {clean_text(text, 80)}")
    else:
        client.report.bad("streamed green content names green", f"→ {clean_text(text, 120)}")
    if finish == "stop":
        client.report.ok("stream finish_reason is stop")
    else:
        client.report.bad("stream finish_reason is stop", f"got {finish!r}")
    client.check_usage("stream", usage)


def case_two_images(client, images):
    content = image_content(images["red"], LIST_PROMPT)
    content.insert(1, {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(images["blue"]).decode()}})
    resp = client.chat(content)
    text = chat_text(resp)
    client.check_usage("two_images", resp.get("usage"))
    if mentions(text, "red") and mentions(text, "blue"):
        client.report.ok("two images: both colors named", f"→ {clean_text(text, 80)}")
    else:
        client.report.bad("two images: both colors named", f"→ {clean_text(text, 120)}")
        return
    low = text.lower()
    if low.find("red") < low.find("blue"):
        client.report.ok("two images: red listed before blue")
    else:
        client.report.bad("two images: red listed before blue", f"→ {clean_text(text, 120)}")


def case_parallel(client, images):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        red_future = pool.submit(client.chat, image_content(images["red"], COLOR_PROMPT))
        blue_future = pool.submit(client.chat, image_content(images["blue"], COLOR_PROMPT))
        red = chat_text(red_future.result())
        blue = chat_text(blue_future.result())
    for name, text, want, other in (("red", red, "red", "blue"), ("blue", blue, "blue", "red")):
        if mentions(text, want) and not mentions(text, other):
            client.report.ok(f"parallel {name} answers {want}, no cross-contamination")
        else:
            client.report.bad(f"parallel {name} answers {want}, no cross-contamination",
                             f"→ {clean_text(text, 120)}")


def case_text_after_image(client, images):
    client.chat(image_content(images["red"], COLOR_PROMPT))  # an image turn precedes the text turn
    text = chat_text(client.chat(MATH_PROMPT, max_tokens=32))
    if re.search(r"\b4\b", text):
        client.report.ok("pure text after image: 2 + 2 = 4", f"→ {clean_text(text, 80)}")
    else:
        client.report.bad("pure text after image: 2 + 2 = 4", f"→ {clean_text(text, 120)}")


def case_anthropic(client, images):
    body = {"model": ARGS.model, "max_tokens": 128,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(images["green"]).decode()}},
                {"type": "text", "text": COLOR_PROMPT}]}]}
    try:
        resp = client.request("/v1/messages", body)
    except HttpError as e:
        if 400 <= e.status < 500:
            client.report.warn("anthropic base64 image block",
                               f"format not supported here (HTTP {e.status}); skipping")
        else:
            client.report.bad("anthropic /v1/messages green image", str(e))
        return
    content = resp.get("content")
    text = "".join(b.get("text", "") for b in content if isinstance(b, dict)) if isinstance(content, list) else str(content)
    if mentions(text, "green"):
        client.report.ok("anthropic /v1/messages green image answers green", f"→ {clean_text(text, 80)}")
    else:
        client.report.bad("anthropic /v1/messages green image answers green", f"→ {clean_text(text, 120)}")
    usage = resp.get("usage") or {}
    if usage.get("input_tokens") is not None and usage["input_tokens"] < 0:
        client.report.bad("anthropic input_tokens non-negative", f"got {usage['input_tokens']}")


def case_usage(client, images):
    baseline = client.chat(COLOR_PROMPT, max_tokens=32).get("usage") or {}
    client.check_usage("usage baseline", baseline)
    image_resp = client.chat(image_content(images["red"], COLOR_PROMPT))
    client.check_usage("usage image", image_resp.get("usage"))
    base_tokens = baseline.get("prompt_tokens")
    img_tokens = (image_resp.get("usage") or {}).get("prompt_tokens")
    if not isinstance(base_tokens, int) or not isinstance(img_tokens, int):
        client.report.bad("usage prompt_tokens reported for text and image requests",
                          f"text={base_tokens!r} image={img_tokens!r}")
        return
    if img_tokens > base_tokens:
        client.report.ok("image prompt_tokens exceed text-only baseline",
                         f"{img_tokens} > {base_tokens}")
    else:
        client.report.bad("image prompt_tokens exceed text-only baseline",
                          f"image={img_tokens} baseline={base_tokens}")


def case_budget_fill(client, images):
    """A client (or the server's own auto budget) may ask for more generation
    than remains once the image expands. The prompt fits, so the answer is a
    clamped budget — never a failed request."""
    models = client.request("/v1/models")["data"]
    ctx = next((m.get("context_length") or 8192 for m in models if m.get("id") == ARGS.model), 8192)
    filler = "Please look at the image and answer. " * 8
    resp = client.chat(image_content(images["red"], filler + COLOR_PROMPT), max_tokens=ctx)
    text = chat_text(resp)
    if "red" in text.lower():
        client.report.ok("window-filling budget clamped, not refused", f"answered: {text.strip()[:60]!r}")
    else:
        client.report.bad("window-filling budget clamped, not refused", f"unexpected answer: {text[:80]!r}")


def case_malformed(client, images):
    bad_content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@not-valid-base64@@"}},
                   {"type": "text", "text": COLOR_PROMPT}]
    try:
        client.chat(bad_content)
        client.report.bad("malformed image rejected with 4xx", "server accepted it (HTTP 200)")
    except HttpError as e:
        if 400 <= e.status < 500:
            client.report.ok("malformed image rejected with 4xx", f"HTTP {e.status}")
        else:
            client.report.bad("malformed image rejected with 4xx", str(e))
    text = chat_text(client.chat(image_content(images["red"], COLOR_PROMPT)))
    if mentions(text, "red"):
        client.report.ok("valid image request still works after malformed input")
    else:
        client.report.bad("valid image request still works after malformed input",
                          f"→ {clean_text(text, 120)}")


CASES = [
    ("models", case_models),
    ("red_blue", case_red_blue),
    ("stream", case_stream),
    ("two_images", case_two_images),
    ("parallel", case_parallel),
    ("text_after_image", case_text_after_image),
    ("anthropic", case_anthropic),
    ("usage", case_usage),
    ("budget_fill", case_budget_fill),
    ("malformed", case_malformed),
]


def main():
    parser = argparse.ArgumentParser(description="Llama-engine vision e2e regression against a running server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="model id as listed by /v1/models")
    parser.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", ""),
                        help="bearer key (default: $BENCH_API_KEY; never logged)")
    parser.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")
    parser.add_argument("--case", default="", help="comma-separated subset, e.g. red_blue,stream (default: all)")
    global ARGS
    ARGS = parser.parse_args()

    selected = CASES if not ARGS.case else [c for c in CASES if c[0] in {s.strip() for s in ARGS.case.split(",")}]
    unknown = ({s.strip() for s in ARGS.case.split(",")} - {name for name, _ in CASES}) if ARGS.case else set()
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}. Valid: {', '.join(n for n, _ in CASES)}")
    if not selected:
        parser.error("no cases selected")

    images = {name: solid_png(*rgb) for name, rgb in
              (("red", (237, 28, 36)), ("green", (34, 177, 76)), ("blue", (0, 112, 192)))}
    report = Report()
    client = Client(ARGS.base_url, ARGS.api_key, ARGS.timeout, report)
    print(f"=== llama vision e2e — {ARGS.base_url} model={ARGS.model} "
          f"cases={','.join(name for name, _ in selected)} timeout={ARGS.timeout}s ===")
    for name, fn in selected:
        print(f"── {name} ──")
        try:
            fn(client, images)
        except HttpError as e:
            report.bad(f"{name}: request failed", str(e))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            report.bad(f"{name}: malformed response", f"{e.__class__.__name__}: {clean_text(str(e), 160)}")
    print(f"=== {report.passed} passed, {report.failed} failed ===")
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
