#!/usr/bin/env python3
"""Is a generation a usable render, or is it noise?

The multi-LoRA suite compares BYTES: a delta is real if the output changed, and
adapters sum if two at 1.0 are byte-identical to one at 2.0. **Noise satisfies
both.** The first time a real community LoRA was pointed at the server every
byte check passed and the image was pure static — the adapter ran 8x too strong
because its alpha lived in the safetensors metadata and we read only kohya's
per-module `.alpha` tensor.

Metric: mean absolute difference between HORIZONTALLY ADJACENT pixels. Real
images are locally smooth because the world is; static is not. Measured on the
artifacts of that session (`~/claude-tmp/h3-ref2va/lora-visuals/`):

    baseline photo, no adapter           7.5
    a working style LoRA                 4.4
    the same LoRA at its correct alpha   6.8
    8x overdriven (noise)               50.4
    stacked, both overdriven (noise)    44.3

Bar at 20 — the middle of a ~6x gap with nothing in it. This is a TRIPWIRE, not
a substitute for opening the file: it catches static, not a bad image.

Usage: lora_noise.py <response.json> [--bar N] [--save out.png]
Exit:  0 usable · 1 noise · 3 numpy/PIL missing (caller should SKIP)
Accepts either response shape — an image (`data[0].b64_json`, PNG) or a video
(`format: "rgb8"` raw frames, of which the middle one is measured).
"""
import base64
import io
import json
import sys

try:
    import numpy as np
    from PIL import Image
except ImportError as e:  # never silently vanish — the caller prints this
    print(f"numpy+PIL required for the noise check ({e})")
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
    bar, save, src = 20.0, None, None
    while args:
        a = args.pop(0)
        if a == "--bar":
            bar = float(args.pop(0))
        elif a == "--save":
            save = args.pop(0)
        else:
            src = a
    if src is None:
        print("usage: lora_noise.py <response.json> [--bar N] [--save out.png]")
        return 2
    try:
        px = pixels(src)
    except Exception as e:
        print(f"undecodable ({type(e).__name__}: {e})")
        return 1
    if save:  # keep the artifact — the metric is a tripwire, the eyeball decides
        Image.fromarray(px).save(save)
    m = float(np.abs(np.diff(px.astype(np.float64), axis=1)).mean())
    print(f"{m:.2f}")
    return 0 if m < bar else 1


if __name__ == "__main__":
    sys.exit(main())
