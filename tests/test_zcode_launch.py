#!/usr/bin/env python3
"""CPU-only launch contract test; optionally exercise the real ZCode CLI.

python3 tests/test_zcode_launch.py --bin <mlx-serve or launch_cli binary>
  [--zcode <built ZCode .cjs>]
No model weights or GPU are used. Fake model identities test discovery, not
model quality. The real-client leg verifies split SSE tool arguments, tool
execution, reasoning, tool-result replay, and the final answer.
"""
import argparse
import http.server
import json
import os
import shlex
from pathlib import Path
import subprocess
import tempfile
import threading

MODELS = [
    {"id": "arbitrary/org/model\"quoted", "capabilities": ["chat", "vision"], "context_length": 98304},
    {"id": "glm-native-fixture", "capabilities": ["chat"], "meta": {"context_length": 65536}},
    {"id": "embedding-only", "capabilities": ["embeddings"]},
]
REQUESTS = []


class Fixture(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        payload = {"data": MODELS} if self.path == "/v1/models" else {"status": "ok"}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(request)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(delta, finish=None):
            body = {"id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 1,
                    "model": request["model"], "choices": [{"index": 0, "delta": delta,
                    "finish_reason": finish}]}
            self.wfile.write(("data: " + json.dumps(body) + "\n\n").encode())
            self.wfile.flush()

        send({"role": "assistant"})
        if len(REQUESTS) == 1:
            send({"reasoning_content": "Read both fixture files before answering."})
            tools = {t["function"]["name"] for t in request["tools"]}
            assert "Read" in tools, tools
            for i, name in enumerate(["a.txt", "b.txt"]):
                args = json.dumps({"file_path": str(self.server.fixture_dir / name)})
                cut = len(args) // 2
                send({"tool_calls": [{"index": i, "id": f"call_{i}", "type": "function",
                                      "function": {"name": "Read", "arguments": args[:cut]}}]})
                send({"tool_calls": [{"index": i, "function": {"arguments": args[cut:]}}]})
            send({}, "tool_calls")
        else:
            send({"content": "ZCODE_FIXTURE_OK"})
            send({}, "stop")
        self.wfile.write(b"data: [DONE]\n\n")


def run(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, timeout=90, **kwargs)
    assert result.returncode == 0, (command, result.stdout[-6000:], result.stderr[-6000:])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", required=True)
    parser.add_argument("--zcode", help="Absolute path to built ZCode .cjs; adds real-client smoke")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="zcode-launch-") as temp:
        root = Path(temp)
        env = dict(os.environ, HOME=str(root))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
        server.fixture_dir = root
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            command = [str(Path(options.bin).resolve()), "launch", "zcode", "--no-start", "--url", base]
            result = run(command + ["--model", MODELS[0]["id"], "--print", "--", "--prompt", "it's a prompt"], env=env)
            config_file = root / ".mlx-serve/zcode/provider_config.json"
            config = json.loads(config_file.read_text())["config"]
            assert config["defaultModelSelection"]["modelId"] == MODELS[0]["id"]
            provider = config["providerConfigRules"]["providerRules"][0]["config"]
            assert provider["api"] == {"type": "openai-chat-completions", "baseUrl": base + "/v1"}
            assert provider["personalModelIds"] == [m["id"] for m in MODELS[:2]]
            rules = config["modelConfigRules"]["providerModelRules"]
            assert rules[0]["config"]["properties"]["contextWindow"] == 98304
            assert rules[0]["config"]["properties"]["inputFormat"]["supportsImage"] is True
            assert rules[1]["config"]["optionSpecs"]["maxOutputTokens"]["max"] == 32768
            assert "ZCODE_PERSONAL_PROVIDER_CONFIG_FILE" in result.stdout
            assert "zcode '--prompt' 'it'\\''s a prompt'" in result.stdout
            failed = subprocess.run(command + ["--model", "embedding-only", "--print"], env=env, capture_output=True)
            assert failed.returncode != 0
            print("PASS: arbitrary IDs, chat filtering, advertised limits, isolated config, passthrough, unknown model")
            if options.zcode:
                # Invoke the printed production script with a function named
                # zcode so no global installation or user PATH mutation is needed.
                (root / "a.txt").write_text("ALPHA")
                (root / "b.txt").write_text("BETA")
                result = run(command + ["--model", "glm-native-fixture", "--print", "--", "--prompt",
                             "Read a.txt and b.txt and report ZCODE_FIXTURE_OK", "--cwd", str(root),
                             "--mode", "build", "--output-format", "json"], env=env)
                script = "zcode() { node "+shlex.quote(str(Path(options.zcode).resolve()))+" \"$@\"; }\n" + result.stdout
                result = run(["/bin/zsh", "-c", script], cwd=root, env=env)
                assert "ZCODE_FIXTURE_OK" in result.stdout, result.stdout
                assert len(REQUESTS) == 2, REQUESTS
                assert all(r["model"] == "glm-native-fixture" and r["stream"] for r in REQUESTS)
                assert REQUESTS[0]["reasoning_effort"] == "medium"
                assert 0 < REQUESTS[0]["max_tokens"] <= 32768
                messages = REQUESTS[1]["messages"]
                results = [m for m in messages if m["role"] == "tool"]
                assert {m["tool_call_id"] for m in results} == {"call_0", "call_1"}, results
                assert any("ALPHA" in str(m["content"]) for m in results), results
                assert any("BETA" in str(m["content"]) for m in results), results
                print("PASS: real ZCode streaming reasoning, fragmented two-tool calls, execution, replay, final answer")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
