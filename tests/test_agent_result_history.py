#!/usr/bin/env python3
"""Run with TESTING_MODE=1, a test-only chat, and the inference server stopped."""
import json
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request("http://127.0.0.1:8090" + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def main():
    prompt = "Read the three fixture files and report their values."
    history = request("/test/history")["messages"]
    assert all(m["role"] != "user" or m["content"] == prompt for m in history), "Select a fresh test chat"
    prefix = "history-" + uuid.uuid4().hex
    observed = []
    with tempfile.TemporaryDirectory(prefix="agent-history-") as directory:
        files = [Path(directory) / f"part-{i}.txt" for i in range(3)]
        markers = [f"VALUE_{i}={prefix}-{i}" for i in range(3)]
        for path, marker in zip(files, markers):
            path.write_text(("padding" * 10 + "\n") * 75 + marker + "\n")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, value, content_type="application/json"):
                data = value.encode() if isinstance(value, str) else json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.send({"status": "ok", "data": []})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                results = {m["tool_call_id"]: m["content"] for m in body["messages"]
                           if m.get("tool_call_id", "").startswith(prefix)}
                stage = len(results)
                if stage < 3:
                    delta = {"tool_calls": [{"index": 0, "id": f"{prefix}-{stage}", "type": "function",
                             "function": {"name": "readFile", "arguments": json.dumps({"path": str(files[stage])})}}]}
                    reason = "tool_calls"
                else:
                    observed.extend(marker in results[f"{prefix}-{i}"] for i, marker in enumerate(markers))
                    delta = {"content": "PASS" if all(observed) else "FAIL: earlier tool payload was cut"}
                    reason = "stop"
                chunks = [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}]
                self.send("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n",
                          "text/event-stream")

        server = ThreadingHTTPServer(("127.0.0.1", 11234), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request("/test/agent", {"message": prompt,
                                    "working_directory": directory, "max_rounds": 4})
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = request("/test/agent/status")
                if status["status"] != "running":
                    break
                time.sleep(0.1)
            print("Tool payloads retained across three rounds:", observed)
            assert observed == [True, True, True], status
        finally:
            server.shutdown()
            server.server_close()
            request("/test/stop", {})


if __name__ == "__main__":
    main()
