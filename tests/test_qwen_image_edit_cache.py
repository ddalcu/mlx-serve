#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Real-pack edit cache A/B: same binary, weights, input, seed and step count.

Run after building: uv run tests/test_qwen_image_edit_cache.py --model <pack>
Needs the vision tower in text_encoder/. Uses a private server, generated benign
references, bounded requests and owned-process cleanup. Saves PNGs, SSE timings,
sampled memory and a JSON report; visual inspection is still required.
"""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request

from PIL import Image, ImageChops, ImageDraw, ImageStat


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def request(url, body=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url, body=None, timeout=10):
    with request(url, body, timeout) as response:
        return json.load(response)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def guard(url):
    if url:
        rows = get_json(url.rstrip("/") + "/v1/models")["data"]
        if any(row.get("loaded") or row.get("state") in ("loading", "ready") for row in rows):
            raise RuntimeError("Guarded server has a loaded model; refusing concurrent inference")


def references(folder, count, opaque=False):
    encoded = []
    for index in range(count):
        im = Image.new("RGBA", (256 + 64 * (index % 2), 256), (255, 255, 255, 255 if opaque else 0))
        draw = ImageDraw.Draw(im)
        draw.rectangle((60, 110, 196, 225), fill=(225, 201, 160, 255))
        draw.polygon(((40, 115), (128, 35), (216, 115)), fill=(180, 35 + index * 12, 45, 255))
        draw.rectangle((111, 165, 148, 225), fill=(60, 50, 40, 255))
        draw.rectangle((76, 135, 98, 158), fill=(70, 170, 220, 255))
        path = folder / f"reference-{index + 1}.png"
        im.save(path)
        encoded.append(base64.b64encode(path.read_bytes()).decode())
    return encoded


def generate(api, process, body, folder, name, timeout, guard_url):
    guard(guard_url)
    done = threading.Event()
    samples, errors = [], []
    start, wall = time.monotonic(), time.time()

    def sample():
        while not done.wait(0.5):
            try:
                guard(guard_url)
                metrics = get_json(api + "/metrics.json", timeout=3)["gauges"]
                samples.append({"elapsed_s": time.monotonic() - start,
                                **{k: metrics.get(k) for k in ("memory_mb", "mlx_active_bytes")}})
            except Exception as error:
                errors.append(str(error))
                process.kill()
                return

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    timer = threading.Timer(timeout, process.kill)
    timer.start()
    events, complete = [], None
    try:
        with request(api + "/v1/images/generations", body, timeout) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("type") == "error":
                    raise RuntimeError(event)
                if event.get("type") == "complete":
                    complete = event
                else:
                    events.append({"elapsed_s": time.monotonic() - start, **event})
        elapsed = time.monotonic() - start
        assert abs((time.time() - wall) - elapsed) < 2, "sleep/clock discontinuity invalidates timing"
        assert not errors, errors
        assert complete is not None, "missing completion event"
        png = base64.b64decode(complete["data"][0]["b64_json"], validate=True)
        im = Image.open(io.BytesIO(png))
        im.load()
        assert im.size == tuple(map(int, body["size"].split("x"))), im.size
        assert im.mode in ("RGB", "RGBA"), im.mode
        (folder / f"{name}.png").write_bytes(png)
        steps = [e["elapsed_s"] for e in events if e.get("stage") == "Generating"]
        assert len(steps) == (body["steps"] or 40), events
        result = {"elapsed_s": elapsed, "first_step_including_encode_s": steps[0],
                  "steady_step_s": statistics.median(b - a for a, b in zip(steps, steps[1:])) if len(steps) > 1 else None,
                  "sha256": hashlib.sha256(png).hexdigest(), "events": events, "memory_samples": samples}
        save(folder / f"{name}.json", result)
        print(f"{name}: {elapsed:.3f}s; steady step {result['steady_step_s']}", flush=True)
        return result
    finally:
        timer.cancel()
        done.set()
        watcher.join(timeout=10)


def cancel_edit(api, body, log_path, guard_url):
    offset = log_path.stat().st_size
    with request(api + "/v1/images/generations", body | {"steps": 40}, 60) as response:
        for line in response:
            if line.startswith(b"data: "):
                event = json.loads(line[6:])
                if event.get("stage") == "Generating" and event.get("step") == 1:
                    break
                assert event.get("type") not in ("error", "complete"), event
    for _ in range(30):
        guard(guard_url)
        with log_path.open("rb") as log:
            log.seek(offset)
            if b"generation cancelled" in log.read():
                return
        time.sleep(1)
    raise AssertionError("disconnect did not cancel the cached edit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "zig-out/bin/mlx-serve")
    parser.add_argument("--refs", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--ref-resolution", type=int, choices=(256, 512, 1024), default=512)
    parser.add_argument("--steps", type=int, default=20, help="Denoise steps; 0 tests the server's 40-step default")
    parser.add_argument("--cfg", type=float, default=1)
    parser.add_argument("--opaque", action="store_true", help="Use white rather than transparent reference backgrounds")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--guard-url", help="Abort if this separate server loads any model")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    assert args.steps >= 0 and args.trials >= 1 and args.timeout > 0
    resolved_steps = args.steps or 40
    assert args.binary.is_file() and (args.model / "config.json").is_file()
    folder = args.out or Path(tempfile.mkdtemp(prefix="qwen-edit-cache-"))
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "report.json").exists(), "use a fresh output directory"
    refs = references(folder, args.refs, args.opaque)
    body = dict(model=args.model.name, mode="edit", image=refs[0], ref_images=refs[1:],
                prompt="Change the roof of the house in <image1> to blue. Keep its windows and door.",
                size=args.size, ref_resolution=args.ref_resolution, steps=args.steps, seed=42,
                guidance_scale=args.cfg, negative_prompt="blurry, distorted" if args.cfg != 1 else "",
                stream=True)
    report = {"status": "running", "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "arms": {}, "memory_note": "Sampled process footprint MB and MLX active bytes, not exact peaks.",
              "timing_note": "One cold/warmup request excluded per arm; subsequent requests timed including conditioning and VAE."}
    print(f"Evidence: {folder}", flush=True)
    try:
        for enabled in (False, True):
            label = "cached" if enabled else "full"
            assert hashlib.sha256(args.binary.read_bytes()).hexdigest() == report["binary_sha256"], "binary changed between arms"
            guard(args.guard_url)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            api = f"http://127.0.0.1:{port}"
            log_path = folder / f"{label}.log"
            env = os.environ | {"MLX_SERVE_QWEN_IMAGE_KV_CACHE": str(int(enabled))}
            with log_path.open("w") as log:
                process = subprocess.Popen([str(args.binary.resolve()), "--serve", "--metrics", "--model-dir", str(folder),
                                            "--port", str(port)], stdout=log, stderr=subprocess.STDOUT, env=env)
                try:
                    for attempt in range(60):
                        assert process.poll() is None, log_path.read_text()[-4000:]
                        try:
                            get_json(api + "/health", timeout=1)
                            break
                        except (OSError, ValueError):
                            if attempt == 59:
                                raise
                            time.sleep(1)
                    loaded = get_json(api + "/v1/load-model", {"model": str(args.model.resolve())}, args.timeout)
                    assert loaded.get("error") is None, loaded
                    rows = get_json(api + "/v1/models")["data"]
                    ready = [r for r in rows if r["state"] == "ready"]
                    assert len(ready) == 1, rows
                    body["model"] = ready[0]["id"]
                    results = []
                    for trial in range(args.trials + 1):
                        results.append(generate(api, process, body, folder, f"{label}-{trial}", args.timeout, args.guard_url))
                    text = log_path.read_text()
                    assert f"steps={resolved_steps} guidance=" in text, "admission did not resolve the step count"
                    expected = enabled and resolved_steps > 1
                    assert f"edit prefix cache enabled={str(expected).lower()}" in text, "cache dispatch not engaged"
                    if expected:
                        branches = 2 if args.cfg != 1 else 1
                        assert f"32 layers, {branches} branch(es); remaining steps target-only" in text
                        cancel_edit(api, body, log_path, args.guard_url)
                    get_json(api + "/v1/unload-model", {"model": body["model"]}, 120)
                    for _ in range(30):
                        active = get_json(api + "/metrics.json")["gauges"]["mlx_active_bytes"]
                        if active < 128 * 1024**2:
                            break
                        time.sleep(1)
                    assert active < 128 * 1024**2, f"unload retained {active} MLX bytes"
                    report["arms"][label] = {"cold": results[0], "hot": results[1:], "unloaded_active_bytes": active,
                                             "cancel_verified": expected,
                                             "median_s": statistics.median(r["elapsed_s"] for r in results[1:])}
                finally:
                    stop(process)
        comparisons = []
        for trial in range(args.trials + 1):
            before = Image.open(folder / f"full-{trial}.png").convert("RGBA")
            after = Image.open(folder / f"cached-{trial}.png").convert("RGBA")
            diff = ImageChops.difference(before, after)
            stat = ImageStat.Stat(diff)
            comparisons.append({"trial": trial, "pixel_identical": before.tobytes() == after.tobytes(),
                                "mae_rgba": stat.mean, "max_error_rgba": [p[1] for p in stat.extrema]})
            assert max(stat.mean) < 1, comparisons[-1]
        report.update(status="passed_numeric_visual_review_required", comparisons=comparisons,
                      speedup=report["arms"]["full"]["median_s"] / report["arms"]["cached"]["median_s"] if resolved_steps > 1 else None)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(folder / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k in ("status", "speedup", "error")}), flush=True)


if __name__ == "__main__":
    main()
