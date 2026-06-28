#!/usr/bin/env python3
"""Dump NSFW-classifier parity fixtures for the Zig oracle in src/nsfw.zig.

Runs the HF reference (Falconsai/nsfw_image_detection, Apache-2.0) on a fixed
synthetic image and writes the preprocessed pixel tensor + the reference logits.
The Zig oracle feeds the SAME preprocessed pixels through its pure-MLX ViT port
and compares logits (isolating the model port from the resize). Needs
`pip install torch transformers pillow` (CPU torch is fine).

Usage: python3 tests/dump_nsfw_fixtures.py [OUT_DIR]
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification

OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath("nsfw_fixtures")
os.makedirs(OUT, exist_ok=True)
REPO = "Falconsai/nsfw_image_detection"

model = AutoModelForImageClassification.from_pretrained(REPO).eval()

# Deterministic synthetic image (RGB gradient, 256² → resized to 224). Content is
# irrelevant: parity is about matching the reference on identical preprocessed
# pixels. Preprocess manually (PIL BILINEAR resize + rescale + normalize 0.5/0.5)
# to match ViTImageProcessor without needing torchvision.
g = np.linspace(0, 255, 256).astype(np.uint8)
img = np.zeros((256, 256, 3), np.uint8)
img[:, :, 0] = g[None, :]
img[:, :, 1] = g[:, None]
img[:, :, 2] = 128
pil = Image.fromarray(img).resize((224, 224), Image.BILINEAR)
arr = (np.asarray(pil, np.float32) / 255.0 - 0.5) / 0.5  # [224,224,3] → [-1,1]
px = torch.from_numpy(arr.transpose(2, 0, 1)[None].copy())  # [1,3,224,224]
with torch.no_grad():
    logits = model(px).logits[0]  # [2]
probs = torch.softmax(logits, -1)

np.asarray(px.numpy(), np.float32).tofile(os.path.join(OUT, "nsfw_pixels.raw"))
np.asarray(logits.numpy(), np.float32).tofile(os.path.join(OUT, "nsfw_logits.raw"))
print(f"[dump] logits={logits.tolist()}  P(normal)={float(probs[0]):.4f}  P(nsfw)={float(probs[1]):.4f}")
print(f"[dump] wrote {OUT}/nsfw_pixels.raw + nsfw_logits.raw")
print("\n# run the Zig oracle:")
print(f"NSFW_TEST_MODEL=$HOME/.mlx-serve/models/Falconsai/nsfw_image_detection \\")
print(f"NSFW_PIXELS={OUT}/nsfw_pixels.raw NSFW_LOGITS={OUT}/nsfw_logits.raw \\")
print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="nsfw"')
