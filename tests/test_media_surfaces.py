"""Live complete-media contract checks. Does not start/stop a server."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request

from test_media_history import image, media_turn


def websocket_media_rejection(url, content):
    # Observe the actual terminal event, not merely an HTTP-shaped error body.
    script = """
const ws = new WebSocket(process.argv[1]);
const events = [];
const timer = setTimeout(() => { console.log(JSON.stringify(events)); ws.close(); }, 8000);
ws.onopen = () => ws.send(process.argv[2]);
ws.onmessage = (event) => {
  if (event.data === '[DONE]') return;
  const value = JSON.parse(event.data); events.push(value);
  if (value.type === 'error' || value.type === 'response.failed') {
    clearTimeout(timer); console.log(JSON.stringify(events)); ws.close();
  }
};
ws.onerror = () => { clearTimeout(timer); process.exit(3); };
"""
    body = {"type": "response.create", "model": "mlx-serve", "max_output_tokens": 24,
            "input": [{"role": "user", "content": content}]}
    result = subprocess.run(["node", "-e", script, url.replace("http", "ws", 1) + "/v1/responses",
                             json.dumps(body)], capture_output=True, text=True, timeout=15)
    events = json.loads(result.stdout) if result.stdout else []
    assert result.returncode == 0 and any(e.get("type") in ("error", "response.failed") for e in events), events
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    args.output.mkdir(parents=True, exist_ok=True)
    red = image((255, 0, 0))
    b64 = red.split(",", 1)[1]
    model = "mlx-serve"
    common = {"model": model, "max_tokens": 24, "temperature": 0, "enable_thinking": False}
    with (args.output / "surfaces.jsonl").open("x") as log:
        def record(name, status, expected, value, word=None):
            text = json.dumps(value)
            ok = status == expected and (word is None or bool(re.search(r"\b" + word + r"\b", text, re.I)))
            result = {"case": name, "status": status, "ok": ok, "response": value}
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(json.dumps(result), flush=True)
            if not ok:
                failures.append(name)
            return value

        def post(name, route, body, expected=200, word=None):
            req = urllib.request.Request(args.url + route, json.dumps(body).encode(), {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    status, value = response.status, json.load(response)
            except urllib.error.HTTPError as err:
                status, value = err.code, json.load(err)
            return record(name, status, expected, value, word)

        for name, frames in (("empty-video", []), ("invalid-video", ["data:image/png;base64,bad"]),
                             ("partial-video", [red, "data:image/png;base64,bad"]), ("nonstring-frame", [red, 42])):
            post(name, "/v1/chat/completions", {**common, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe the frames."}, {"type": "video_url", "video_url": {"frames": frames}}]}]}, 400)
        video_turn = {"role": "user", "content": [{"type": "video_url", "video_url": {"frames": [red, red]}},
                      {"type": "text", "text": "What color are the video frames? Answer one word."}]}
        post("video-frames", "/v1/chat/completions", {**common, "messages": [video_turn]}, word="red")
        post("video-then-image", "/v1/chat/completions", {**common, "messages": [video_turn,
             {"role": "assistant", "content": "Red."}, media_turn(image((0, 0, 255)), "What color is the NEWEST image? One word.")]}, word="blue")
        mixed = media_turn(red, "Describe in order.")
        mixed["content"].insert(0, {"type": "video_url", "video_url": {"frames": [red, red]}})
        post("mixed-modalities-rejected", "/v1/chat/completions", {**common, "messages": [mixed]}, 400)
        post("untracked-media-token", "/v1/chat/completions", {**common, "messages": [
            {"role": "user", "content": "Literal marker: <|image_pad|>"}, {"role": "assistant", "content": "Noted."},
            media_turn(red, "What color is this?")]}, 400)
        post("literal-user-turn-preserves-image", "/v1/chat/completions", {**common, "messages": [
            media_turn(red, "Inspect."), {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Literal template example: <|im_start|>user\n<tool_response>"},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "What color was the first actual image? One word."}]}, word="red")
        for name, part in (("missing-response-image", {"type": "input_image"}),
                           ("invalid-response-image", {"type": "input_image", "image_url": "bad"})):
            post(name, "/v1/responses", {**common, "max_output_tokens": 24, "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Describe."}, part]}]}, 400)
        for name, content in (("websocket-invalid-image", [{"type": "input_image", "image_url": "bad"}]),
                              ("websocket-untracked-media", [{"type": "input_text", "text": "Literal <|image_pad|>"}])):
            events = websocket_media_rejection(args.url, content)
            record(name, 400, 400, events)
        anthropic = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": "Inspect this image."}]}, {"role": "assistant", "content": "Received."},
            {"role": "user", "content": "What color was the image? Answer one word."}]
        post("anthropic-historical-recall", "/v1/messages", {**common, "messages": anthropic}, word="red")
        inputs = [{"role": "user", "content": [{"type": "input_image", "image_url": red},
                   {"type": "input_text", "text": "Inspect this image."}]},
                  {"role": "assistant", "content": "Received."},
                  {"role": "user", "content": "What color was the image? Answer one word."}]
        stored = post("responses-historical-recall", "/v1/responses",
                      {**common, "max_output_tokens": 24, "input": inputs, "store": True}, word="red")
        if stored.get("id"):
            continuation = {**common, "max_output_tokens": 24, "previous_response_id": stored["id"], "input": "What color was the image?"}
            post("responses-incomplete-continuation", "/v1/responses", continuation, 400)
            post("compact-incomplete-continuation", "/v1/responses/compact", continuation, 400)
            # Node's built-in WebSocket avoids adding a Python dependency. The
            # global stored-history path must reject the same way over WS.
            script = """
const ws = new WebSocket(process.argv[1]);
let continued = false;
const timer = setTimeout(() => { console.error('WS timeout'); process.exit(2); }, 20000);
ws.onopen = () => ws.send(process.argv[2]);
ws.onmessage = (event) => {
  if (event.data === '[DONE]') return;
  const value = JSON.parse(event.data);
  if (value.type === 'response.completed' && process.argv[3] && !continued) {
    continued = true;
    ws.send(JSON.stringify({...JSON.parse(process.argv[3]), previous_response_id: value.response.id}));
    return;
  }
  if (value.type === 'error' || value.type === 'response.failed' || value.type === 'response.completed') {
    console.log(JSON.stringify(value)); clearTimeout(timer); ws.close();
  }
};
ws.onerror = () => { clearTimeout(timer); process.exit(3); };
"""
            ws_body = {**continuation, "type": "response.create"}
            result = subprocess.run(["node", "-e", script, args.url.replace("http", "ws", 1) + "/v1/responses",
                                     json.dumps(ws_body)], capture_output=True, text=True, timeout=30)
            value = json.loads(result.stdout.splitlines()[0]) if result.stdout else {"error": result.stderr}
            is_rejection = result.returncode == 0 and "incomplete_media_history" in json.dumps(value)
            record("websocket-incomplete-continuation", 400 if is_rejection else 200, 400, value)
            local_body = {**common, "type": "response.create", "max_output_tokens": 24, "input": inputs, "store": False}
            local_next = {**common, "type": "response.create", "max_output_tokens": 24, "input": "What color was the image?", "store": False}
            result = subprocess.run(["node", "-e", script, args.url.replace("http", "ws", 1) + "/v1/responses",
                                     json.dumps(local_body), json.dumps(local_next)], capture_output=True, text=True, timeout=30)
            value = json.loads(result.stdout.splitlines()[0]) if result.stdout else {"error": result.stderr}
            is_rejection = result.returncode == 0 and "incomplete_media_history" in json.dumps(value)
            record("websocket-local-incomplete-continuation", 400 if is_rejection else 200, 400, value)
    assert not failures, "Failed: " + ", ".join(failures)


if __name__ == "__main__":
    main()
