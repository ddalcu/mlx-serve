#!/usr/bin/env python3
"""Two media responses within a closeness bar: the mean absolute pixel difference of the images
(or the videos' middle frames) must stay under --bar (default 1.0 of 255). For the H3 LoRA stack,
whose adapters are summed as separate fused multiply-adds, A@1+B@1 and A@2 differ by rounding.

Usage: lora_close.py <a.json> <b.json> [--bar N]
"""
import base64
import io
import json
import sys

try:
    import numpy as np
    from PIL import Image
except ImportError as e:  # never silently vanish — the caller prints this
    print(f"numpy+PIL required for the closeness check ({e})")
    sys.exit(3)


def pixels(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("format") == "rgb8":  # video: raw frames, take the middle one
        F, H, W = d["frames"], d["height"], d["width"]
        raw = np.frombuffer(base64.b64decode(d["data"]), dtype=np.uint8)
        return raw.reshape(F, H, W, 3)[F // 2]
    png = base64.b64decode(d["data"][0]["b64_json"])
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def main():
    args = sys.argv[1:]
    bar, paths = 1.0, []
    while args:
        a = args.pop(0)
        if a == "--bar":
            bar = float(args.pop(0))
        else:
            paths.append(a)
    if len(paths) != 2:
        print("usage: lora_close.py <a.json> <b.json> [--bar N]")
        return 2
    try:
        a, b = pixels(paths[0]), pixels(paths[1])
    except Exception as e:
        print(f"undecodable ({type(e).__name__}: {e})")
        return 1
    if a.shape != b.shape:
        print(f"shape {a.shape} vs {b.shape}")
        return 1
    m = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())
    print(f"{m:.3f}")
    return 0 if m < bar else 1


if __name__ == "__main__":
    sys.exit(main())
