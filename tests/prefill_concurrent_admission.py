#!/usr/bin/env python3
"""Small-model HTTP sanity; memory admission decisions are pinned by Zig unit tests."""
import argparse
import concurrent.futures
import contextlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import time


class Client:
    def __init__(self, port):
        self.port = port

    def post(self, prompt, count=8, stream=False):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        payload = dict(model="mlx-serve", prompt=prompt, max_tokens=count,
                       temperature=0, stream=stream, ignore_eos=True)
        conn.request("POST", "/v1/completions", json.dumps(payload),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        if not stream:
            body = json.loads(response.read())
            conn.close()
            return response.status, body
        if response.status != 200:
            detail = response.read().decode()
            conn.close()
            raise AssertionError(f"stream HTTP {response.status}: {detail}")
        return conn, response

    def finish(self, response, first=None):
        events = [] if first is None else [first]
        done = False
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            assert "error" not in event, event
            events.append(event)
        assert done and events, "stream did not finish with [DONE]"
        assert any(c.get("finish_reason") for e in events for c in e.get("choices", [])), events

    def first(self, response):
        for line in response:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                event = json.loads(line[6:])
                assert "error" not in event, event
                return event
        raise AssertionError("stream closed before first event")

    def stream(self, prompt):
        conn, response = self.post(prompt, stream=True)
        try:
            self.finish(response)
        finally:
            conn.close()


def success(result):
    status, body = result
    assert status == 200 and "error" not in body, (status, body)
    assert body.get("choices"), body
    return body


@contextlib.contextmanager
def server(args, concurrency, timeout, label):
    # Validate the resolved HOME before the server can create configuration or caches.
    home = args.scratch / (label + "-home")
    home.mkdir()
    env = os.environ.copy()
    env.update(HOME=str(home), TMPDIR=str(args.scratch))
    resolved = subprocess.check_output(
        ["python3", "-c", "from pathlib import Path; print(Path.home().resolve())"],
        env=env, text=True).strip()
    assert Path(resolved) == home.resolve() and home.is_relative_to(args.scratch)
    log_path = args.scratch / (label + ".log")
    with log_path.open("w") as log:
        proc = subprocess.Popen([
            str(args.binary), "--serve", "--model", str(args.model), "--port", str(args.port),
            "--ctx-size", "8192", "--max-concurrent", str(concurrency),
            "--prefill-chunk", "512", "--prefix-cache-entries", "8",
            "--no-pld", "--no-drafter", "--no-mtp", "--timeout", str(timeout),
            "--log-level", "debug"], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                assert proc.poll() is None, f"server exited; see {log_path}"
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=1)
                    conn.request("GET", "/health")
                    response = conn.getresponse()
                    healthy = response.status == 200
                    response.read()
                    conn.close()
                    if healthy:
                        break
                except OSError:
                    pass
                time.sleep(0.2)
            else:
                raise AssertionError(f"server health deadline; see {log_path}")
            yield Client(args.port), log_path
            assert proc.poll() is None, "server crashed"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.scratch = args.scratch.resolve()
    assert 1024 <= args.port <= 65535 and args.port != 11234
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    prefix = "The archive describes a quiet blue lake. " * 120
    with server(args, 4, 60, "concurrent") as (client, log_path):
        success(client.post(prefix))
        conn, response = client.post(prefix + " Continue listing observations.", 512, True)
        try:
            first = client.first(response)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                leader = pool.submit(client.finish, response, first)
                neighbors = [pool.submit(client.stream, p) for p in
                             (prefix + " Summarize.", "Red desert dunes. " * 120, "Hello")]
                for neighbor in neighbors:
                    neighbor.result(timeout=90)
                leader.result(timeout=90)
        finally:
            conn.close()
        conn, response = client.post(prefix + " List more observations.", 4096, True)
        client.first(response)
        wire = conn.sock or response.fp.raw._sock
        wire.shutdown(socket.SHUT_RDWR)
        response.close()
        conn.close()
        success(client.post("Reply OK after cancellation."))
        status, body = client.post("Oversized unique text. " * 20_000)
        assert status == 400 and "error" in body, (status, body)
        success(client.post("Reply OK after refusal."))
        assert "reused" in log_path.read_text(), "warm cache did not engage"
        print("PASS concurrent warm/miss/short SSE, disconnect cancellation, context refusal")
    print("SKIP HTTP server queue timeout: --max-concurrent does not serialize "
          "this batchable model; deterministic held-request timeout is covered by Zig tests")
    print("SKIP HTTP deterministic warm-memory hold and standalone memory refusal: "
          "no process-local prefill budget override; Zig regressions cover these decisions")


if __name__ == "__main__":
    main()
