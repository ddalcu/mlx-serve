#!/usr/bin/env python3
"""Reference image / video embeddings for EmbeddingGemma 2, and the check of a running mlx-serve against them.

Reference = Hugging Face's `EmbeddingGemma2Model` in float32 on CPU behind its own `EmbeddingGemma2Processor`
(what sentence-transformers runs): the prompt is the model's chat template (text and `<|image|>`/`<|video|>`
placeholders concatenated, no separator), the image/video processors pick each size and token count, and the
embedding is the mean over every token of the projected states, L2-normalized. Images are the repo's test fixtures,
re-encoded as PNG for the request so both sides decode the same pixels (JPEG and WebP are sent as-is too).

Needs transformers >= 5.19, torch, torchvision and pillow:
    uv run --with "transformers>=5.19" --with torch --with torchvision --with pillow --with numpy \
        python3 tests/dump_embeddinggemma2_media.py dump [--model google/embeddinggemma-2] [--out /tmp/eg2_media_ref.json]

Then, against a server with the same checkpoint (or a quantized pack of it) loaded:
    python3 tests/dump_embeddinggemma2_media.py compare http://127.0.0.1:11297 [--ref /tmp/eg2_media_ref.json]
Expected: token counts equal, per-request cosine >= 0.9998 on the bf16 pack, 0.9995 on 8-bit (4-bit packs lose more
to their own quantization; `--floor` sets the bar, 0.999 by default).
"""

import argparse
import base64
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def sliding_crops(im, n):
    """n windows, two thirds of the width each, sliding left to right: a clip that moves."""
    w, h = im.size
    step = (w // 3) / max(n - 1, 1)
    return [im.crop((int(i * step), 0, int(i * step) + 2 * w // 3, h)) for i in range(n)]


def cases():
    """Name -> list of ('text', str) | ('image', PIL | data URL) | ('video', [PIL frames])."""
    from PIL import Image

    house = Image.open(FIXTURES / "house.jpeg")
    signs = Image.open(FIXTURES / "street-name-signs.jpg")
    robot = Image.open(FIXTURES / "robot.png")  # palette PNG with transparency: alpha is dropped, not composited
    chart = Image.open(ROOT / "docs" / "perf-vs-engines.png")  # 2015x1468: a real downscale
    for im in (house, signs, robot, chart):
        im.load()
    long_clip = [Image.new("RGB", (64, 48), (i * 5, 255 - i * 5, 90)) for i in range(40)]  # more than the 32-frame cap
    house_jpeg = "data:image/jpeg;base64," + base64.b64encode((FIXTURES / "house.jpeg").read_bytes()).decode()
    return {
        "image_house": [("image", house)],
        "image_signs": [("image", signs)],
        "image_robot_alpha": [("image", robot)],
        "image_chart": [("image", chart)],
        "text_then_image": [("text", "a photo of a house: "), ("image", house)],
        "image_then_text": [("image", signs), ("text", " street signs")],
        "two_images": [("image", house), ("image", signs)],
        "video_5_frames": [("video", sliding_crops(house, 5))],
        "video_40_frames_capped": [("video", long_clip)],
        "text_video_image": [("text", "clip: "), ("video", sliding_crops(house, 3)), ("text", " then "), ("image", robot)],
        "jpeg_as_sent": [("image", house_jpeg)],
    }


def png_url(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def body(parts):
    content = []
    for kind, v in parts:
        if kind == "text":
            content.append({"type": "text", "text": v})
        elif kind == "image":
            content.append({"type": "image_url", "image_url": {"url": v if isinstance(v, str) else png_url(v)}})
        else:
            content.append({"type": "video_url", "video_url": {"frames": [png_url(f) for f in v]}})
    return {"model": "google/embeddinggemma-2", "messages": [{"role": "user", "content": content}]}


def dump(model_id, out_path):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, EmbeddingGemma2Model

    proc = AutoProcessor.from_pretrained(model_id)
    model = EmbeddingGemma2Model.from_pretrained(model_id, dtype=torch.float32).eval()

    def reference(parts):
        text, images, videos = "", [], []
        for kind, v in parts:
            if kind == "text":
                text += v
            elif kind == "image":
                text += "<|image|>"
                if isinstance(v, str):  # a data URL, decoded here as the server decodes it
                    v = Image.open(io.BytesIO(base64.b64decode(v.split(",", 1)[1])))
                images.append(v.convert("RGB"))
            else:
                text += "<|video|>"
                videos.append(np.stack([np.asarray(f.convert("RGB")) for f in v]))
        kw = {**({"images": [images]} if images else {}), **({"videos": videos} if videos else {})}
        inputs = proc(text=[text], return_tensors="pt", **kw)
        with torch.no_grad():
            pooled = model(**inputs).last_hidden_state[0].mean(0)
        return (pooled / pooled.norm()).tolist(), int(inputs["input_ids"].shape[1])

    out = {}
    for name, parts in cases().items():
        emb, n = reference(parts)
        out[name] = {"body": body(parts), "embedding": emb, "tokens": n}
        print(f"  {name}: {n} tokens")
    Path(out_path).write_text(json.dumps({"model": model_id, "cases": out}))
    print(f"wrote {len(out)} reference embeddings to {out_path}")


def compare(server, ref_path, floor):
    ref = json.loads(Path(ref_path).read_text())["cases"]
    worst, bad = 1.0, 0
    for name, case in ref.items():
        req = urllib.request.Request(server.rstrip("/") + "/v1/embeddings", data=json.dumps(case["body"]).encode(), headers={"Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                got = json.load(r)
        except urllib.error.HTTPError as e:
            print(f"  {name}: HTTP {e.code} {e.read()[:200]!r}")
            bad += 1
            continue
        a, b = case["embedding"], got["data"][0]["embedding"]
        cos = sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))
        tokens = got["usage"]["prompt_tokens"]
        worst = min(worst, cos)
        bad += tokens != case["tokens"]
        print(f"  {name:24s} cos {cos:.5f}  tokens {tokens} (reference {case['tokens']}{'' if tokens == case['tokens'] else ' MISMATCH'})  {time.time() - t:.2f}s")
    print(f"worst cosine: {worst:.5f}")
    if bad or worst < floor:
        print(f"FAIL: {bad} request(s) failed or miscounted, or a cosine fell below {floor}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--model", default="google/embeddinggemma-2")
    d.add_argument("--out", default="/tmp/eg2_media_ref.json")
    c = sub.add_parser("compare")
    c.add_argument("server")
    c.add_argument("--ref", default="/tmp/eg2_media_ref.json")
    c.add_argument("--floor", type=float, default=0.999)
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args.model, args.out)
    else:
        sys.exit(compare(args.server, args.ref, args.floor))
