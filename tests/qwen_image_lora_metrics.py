#!/usr/bin/env python3
"""Pixel metrics + contact sheet for tests/test_qwen_image_lora.sh.

  valid <resp.json>                 decodable PNG, not static, not flat. exit 0/1
  diff <a.json> <b.json>            mean |a-b| per channel on the 0..255 scale
  expected <lora.safetensors>       module-attachments a full attach must report
  sheet <out.png> <label=resp.json>...   labelled grid, 4 tiles per row

`valid` reuses lora_noise.py's tripwire (mean adjacent-pixel delta, bar 20) and
adds a flat-image floor (pixel std-dev) — a LoRA that zeroes the residual stream
renders a uniform colour, which is smooth and so passes the noise check alone.

Exit: 0 ok · 1 failed · 3 numpy/PIL missing (caller SKIPs)
"""
import base64
import io
import json
import struct
import sys

try:
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError as e:
    print(f"numpy+PIL required ({e})")
    sys.exit(3)

NOISE_BAR = 20.0  # lora_noise.py's bar
FLAT_FLOOR = 8.0  # std-dev of the pixel values below which an image is a flat fill


def pixels(path):
    with open(path) as f:
        d = json.load(f)
    png = base64.b64decode(d["data"][0]["b64_json"])
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def cmd_valid(path):
    try:
        px = pixels(path)
    except Exception as e:  # an undecodable PNG is a failed render
        print(f"undecodable ({type(e).__name__}: {e})")
        return 1
    f = px.astype(np.float64)
    adj = float(np.abs(np.diff(f, axis=1)).mean())
    std = float(f.std())
    print(f"adj={adj:.2f} std={std:.1f}")
    return 0 if adj < NOISE_BAR and std >= FLAT_FLOOR else 1


def cmd_diff(a, b):
    pa, pb = pixels(a), pixels(b)
    if pa.shape != pb.shape:
        print(f"shape mismatch {pa.shape} vs {pb.shape}")
        return 1
    print(f"{float(np.abs(pa.astype(np.float64) - pb.astype(np.float64)).mean()):.3f}")
    return 0


def cmd_expected(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    keys = [k for k in header if k != "__metadata__"]
    mods = [k for k in keys if k.endswith(("lora_A.weight", "lora_down.weight"))]
    # A fused `gate_up` module fans out into the runtime's gate_layer + proj.
    print(sum(2 if ".gate_up." in k else 1 for k in mods))
    return 0


def cmd_sheet(out, items):
    tiles = []
    for item in items:
        label, _, path = item.partition("=")
        tiles.append((label, Image.fromarray(pixels(path))))
    side, bar, cols = 384, 22, 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * side, rows * (side + bar)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(tiles):
        x, y = (i % cols) * side, (i // cols) * (side + bar)
        sheet.paste(im.resize((side, side)), (x, y + bar))
        draw.text((x + 6, y + 5), label, fill=(235, 235, 235))
    sheet.save(out)
    return 0


def main():
    a = sys.argv[1:]
    try:
        if a[0] == "valid":
            return cmd_valid(a[1])
        if a[0] == "diff":
            return cmd_diff(a[1], a[2])
        if a[0] == "expected":
            return cmd_expected(a[1])
        if a[0] == "sheet":
            return cmd_sheet(a[1], a[2:])
    except IndexError:
        pass
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
