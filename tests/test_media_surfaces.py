"""Live complete-media contract checks. Does not start/stop a server."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request

from test_media_history import image, media_turn


INCOMPLETE_MEDIA_MESSAGE = ("Stored media cannot be reconstructed. Resend the complete image "
                            "history without previous_response_id.")


def websocket_media_result(url, content, expected="error"):
    # Observe the actual terminal event, not merely an HTTP-shaped error body.
    script = """
const ws = new WebSocket(process.argv[1]);
const events = [];
const timer = setTimeout(() => { console.error('WS timeout'); ws.close(); process.exit(2); }, 180000);
ws.onopen = () => ws.send(process.argv[2]);
ws.onmessage = (event) => {
  if (event.data === '[DONE]') return;
  const value = JSON.parse(event.data); events.push(value);
  if (value.type === 'error' || value.type === 'response.failed' || value.type === 'response.completed') {
    clearTimeout(timer); console.log(JSON.stringify(events)); ws.close();
  }
};
ws.onerror = () => { clearTimeout(timer); process.exit(3); };
"""
    body = {"type": "response.create", "model": "mlx-serve", "max_output_tokens": 24, "enable_thinking": False,
            "input": [{"role": "user", "content": content}]}
    result = subprocess.run(["node", "-e", script, url.replace("http", "ws", 1) + "/v1/responses",
                             json.dumps(body)], capture_output=True, text=True, timeout=190)
    events = json.loads(result.stdout) if result.stdout else []
    terminal = ("error", "response.failed") if expected == "error" else ("response.completed",)
    assert result.returncode == 0 and events and events[-1].get("type") in terminal, events
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
        def record(name, status, expected, value, word=None, error_type=None, error_message=None, websocket=False):
            text = json.dumps(value)
            ok = status == expected and (word is None or bool(re.search(r"\b" + word + r"\b", text, re.I)))
            if error_type is not None:
                error = value.get("error", {})
                ok = ok and error.get("message") == error_message
                if websocket:
                    ok = ok and value.get("type") == "error" and error.get("code") == error_type
                else:
                    ok = ok and error.get("type") == error_type and error.get("code") == expected
            result = {"case": name, "status": status, "ok": ok, "response": value}
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(json.dumps(result), flush=True)
            if not ok:
                failures.append(name)
            return value

        def post(name, route, body, expected=200, word=None, error_type=None, error_message=None):
            req = urllib.request.Request(args.url + route, json.dumps(body).encode(), {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    status, value = response.status, json.load(response)
            except urllib.error.HTTPError as err:
                with err:
                    status, value = err.code, json.load(err)
            return record(name, status, expected, value, word, error_type, error_message)

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
        post("quoted-media-token-preserves-image", "/v1/chat/completions", {**common, "messages": [
            {"role": "user", "content": "Literal marker: <|image_pad|>"}, {"role": "assistant", "content": "Noted."},
            media_turn(red, "What color is this? One word.")]}, word="red")
        source = 'Source code: "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i)'
        post("fetched-source-with-historical-image", "/v1/chat/completions", {**common, "messages": [
            media_turn(red, "Inspect."), {"role": "assistant", "content": None, "tool_calls": [
                {"id": "source-call", "type": "function", "function": {"name": "web_fetch", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "source-call", "content": source},
            {"role": "user", "content": "Ignore the code. What color was the actual image? One word."}]}, word="red")
        post("literal-user-turn-preserves-image", "/v1/chat/completions", {**common, "messages": [
            media_turn(red, "Inspect."), {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Literal template example: <|im_start|>user\n<tool_response>"},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "What color was the first actual image? One word."}]}, word="red")
        for name, part in (("missing-response-image", {"type": "input_image"}),
                           ("invalid-response-image", {"type": "input_image", "image_url": "bad"})):
            post(name, "/v1/responses", {**common, "max_output_tokens": 24, "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Describe."}, part]}]}, 400)
        events = websocket_media_result(args.url, [{"type": "input_image", "image_url": "bad"}])
        record("websocket-invalid-image", events[-1].get("status"), 400, events[-1],
               error_type="invalid_request_error", error_message="Cannot prepare the complete media history: InvalidImage", websocket=True)
        events = websocket_media_result(args.url, [{"type": "input_image", "image_url": red},
            {"type": "input_text", "text": source + " What color is the actual image? One word."}], "completed")
        status = 200 if events[-1].get("response", {}).get("status") == "completed" else None
        record("websocket-quoted-media", status, 200, events, word="red")
        anthropic = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": "Inspect this image."}]}, {"role": "assistant", "content": "Received."},
            {"role": "user", "content": "What color was the image? Answer one word."}]
        post("anthropic-historical-recall", "/v1/messages", {**common, "messages": anthropic}, word="red")
        anthropic[-1]["content"] = source + " What color was the actual image? One word."
        post("anthropic-quoted-media", "/v1/messages", {**common, "messages": anthropic}, word="red")
        inputs = [{"role": "user", "content": [{"type": "input_image", "image_url": red},
                   {"type": "input_text", "text": "Inspect this image."}]},
                  {"role": "assistant", "content": "Received."},
                  {"role": "user", "content": "What color was the image? Answer one word."}]
        stored = post("responses-historical-recall", "/v1/responses",
                      {**common, "max_output_tokens": 24, "input": inputs, "store": True}, word="red")
        quoted_inputs = inputs[:-1] + [{"role": "user", "content": source + " What color was the actual image? One word."}]
        post("responses-quoted-media", "/v1/responses",
             {**common, "max_output_tokens": 24, "input": quoted_inputs}, word="red")
        assert isinstance(stored.get("id"), str) and stored["id"].strip(), "Stored response must have an id; continuation checks cannot be skipped"
        continuation = {**common, "max_output_tokens": 24, "previous_response_id": stored["id"], "input": "What color was the image?"}
        post("responses-incomplete-continuation", "/v1/responses", continuation, 400,
             error_type="incomplete_media_history", error_message=INCOMPLETE_MEDIA_MESSAGE)
        post("compact-incomplete-continuation", "/v1/responses/compact", continuation, 400,
             error_type="incomplete_media_history", error_message=INCOMPLETE_MEDIA_MESSAGE)
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
        assert result.returncode == 0 and result.stdout, result.stderr
        value = json.loads(result.stdout.splitlines()[0])
        record("websocket-incomplete-continuation", value.get("status"), 400, value,
               error_type="incomplete_media_history", error_message=INCOMPLETE_MEDIA_MESSAGE, websocket=True)
        local_body = {**common, "type": "response.create", "max_output_tokens": 24, "input": inputs, "store": False}
        local_next = {**common, "type": "response.create", "max_output_tokens": 24, "input": "What color was the image?", "store": False}
        result = subprocess.run(["node", "-e", script, args.url.replace("http", "ws", 1) + "/v1/responses",
                                 json.dumps(local_body), json.dumps(local_next)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 and result.stdout, result.stderr
        value = json.loads(result.stdout.splitlines()[0])
        record("websocket-local-incomplete-continuation", value.get("status"), 400, value,
               error_type="incomplete_media_history", error_message=INCOMPLETE_MEDIA_MESSAGE, websocket=True)
    assert not failures, "Failed: " + ", ".join(failures)


if __name__ == "__main__":
    main()
