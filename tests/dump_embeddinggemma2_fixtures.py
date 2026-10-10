#!/usr/bin/env python3
"""Tiny random EmbeddingGemma 2 checkpoint + the reference's own outputs, for the hermetic parity tests.

The reference is Hugging Face's `EmbeddingGemma2Model` (transformers >= 5.19), run in float32 on CPU. The
checkpoint is a toy-geometry model in the release's layout (`language_model.*` keys, per-layer geometry
overrides on the full-attention layers, `ple_block.*`, projection-only per-layer inputs, a Gemma 4 vision tower
under `vision_tower.*` + `embed_vision.*`), so the Zig forward is checked against code it did not write, with
every code path of the real model exercised: the 5:1-style sliding/full pattern, GQA at two widths, the
inclusive bidirectional band, PLE, layer scalars, the embedding projection, padded batches, and image and video
soft tokens spliced into the sequence.

    uv run --with "transformers>=5.19" --with torch --with torchvision --with safetensors --with numpy \
        --with pillow python3 tests/dump_embeddinggemma2_fixtures.py tiny [--out src/fixtures]

For the real checkpoint, compare a running server against sentence-transformers with
`tests/dump_embeddinggemma_fixtures.py --model google/embeddinggemma-2` (same script, same bar).
"""

import argparse
import base64
import json
import os

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from safetensors.torch import save_file
from torchvision.transforms import InterpolationMode
from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model
from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import EmbeddingGemma2VideoProcessor
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor, get_aspect_ratio_preserving_size

HIDDEN, INTER, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 16, 32, 6, 4, 2, 8
GLOBAL_HEAD_DIM, GLOBAL_KV_HEADS, PATTERN = 16, 1, 3
PLE_DIM, EMBED_DIM, VOCAB, WINDOW = 8, 12, 48, 2
IMAGE, VIDEO, AUDIO, BOI, EOI = 40, 41, 42, 43, 44
# The vision tower keeps the release's patch size and pooling kernel (so the processors' arithmetic is the real
# one) at a toy width; 70 is the smallest soft-token budget the processors accept.
V_HIDDEN, V_INTER, V_LAYERS, V_HEADS, V_HEAD_DIM, V_POS, V_BUDGET = 24, 48, 2, 3, 8, 96, 70
ROPE = {
    "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
}
# bos=2 ... eos=1, right-padded with 0: lengths cross the sliding band (radius 2) in three of four rows.
SEQUENCES = [[2, 7, 1], [2, 5, 9, 13, 21, 8, 30, 4, 17, 6, 11, 1], [2, 33, 12, 1, 25, 3, 19], [2, 10, 20, 30, 15, 25, 35, 5, 9, 14, 24, 1]]


def text_config() -> dict:
    return dict(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
        hidden_activation="gelu_pytorch_tanh", max_position_embeddings=64, rms_norm_eps=1e-6,
        pad_token_id=0, eos_token_id=1, bos_token_id=2, sliding_window=WINDOW,
        hidden_size_per_layer_input=PLE_DIM, embedding_dim=EMBED_DIM, rope_parameters=ROPE,
        sliding_window_pattern=PATTERN, global_head_dim=GLOBAL_HEAD_DIM, num_global_key_value_heads=GLOBAL_KV_HEADS,
    )


def vision_config() -> dict:
    return dict(
        hidden_size=V_HIDDEN, intermediate_size=V_INTER, num_hidden_layers=V_LAYERS, num_attention_heads=V_HEADS,
        num_key_value_heads=V_HEADS, head_dim=V_HEAD_DIM, patch_size=16, pooling_kernel_size=3,
        position_embedding_size=V_POS, rms_norm_eps=1e-6, use_clipped_linears=False, standardize=False,
        rope_parameters={"rope_theta": 100.0, "rope_type": "axial"},
    )


def make(text_overrides: dict | None = None) -> EmbeddingGemma2Model:
    config = EmbeddingGemma2Config(
        text_config=text_config() | (text_overrides or {}), vision_config=vision_config(), audio_config=None,
        image_token_id=IMAGE, video_token_id=VIDEO, audio_token_id=AUDIO, boi_token_id=BOI, eoi_token_id=EOI,
    )
    return EmbeddingGemma2Model(config).eval()


def build(seed: int) -> EmbeddingGemma2Model:
    torch.manual_seed(seed)
    model = make()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn_like(p))
            elif "norm" in name:
                p.copy_(0.5 + torch.rand_like(p))
            elif "vision_tower" in name:
                p.copy_(0.12 * torch.randn_like(p))
            else:
                p.copy_(0.35 * torch.randn_like(p))
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.6 + 0.8 * torch.rand_like(b))
    return model


def reference(model: EmbeddingGemma2Model, ids: list[int]):
    """Final-normed token states (what the projection reads) and the pooled, normalized embedding of ONE unpadded row."""
    seen = {}
    hook = model.language_model.norm.register_forward_hook(lambda m, i, o: seen.setdefault("norm", o.detach()))
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state[0]
    hook.remove()
    pooled = out.mean(dim=0)
    return seen["norm"][0].tolist(), (pooled / pooled.norm()).tolist()


def cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def tensors(model: EmbeddingGemma2Model) -> dict:
    sd = {k: v.detach().clone().contiguous().float() for k, v in model.state_dict().items()}
    assert all(k.startswith(("language_model.", "vision_tower.", "embed_vision.")) for k in sd), sorted(sd)[:5]
    return sd


def release_config() -> dict:
    """config.json in the release's shape (per_layer_config keyed by two-digit layer index)."""
    tc = text_config()
    kinds = ["sliding_attention" if (i + 1) % PATTERN else "full_attention" for i in range(LAYERS)]
    for k in ("sliding_window_pattern", "global_head_dim", "num_global_key_value_heads"):
        tc.pop(k)
    tc.update(
        layer_types=kinds, model_type="embedding_gemma2_text", dtype="float32",
        per_layer_config={f"{i:02d}": {"head_dim": GLOBAL_HEAD_DIM, "num_key_value_heads": GLOBAL_KV_HEADS} for i, k in enumerate(kinds) if k == "full_attention"},
    )
    vc = vision_config() | {"model_type": "gemma4_vision", "default_output_length": V_BUDGET, "global_head_dim": V_HEAD_DIM}
    return {
        "architectures": ["EmbeddingGemma2Model"], "model_type": "embedding_gemma2", "dtype": "float32", "text_config": tc,
        "vision_config": vc, "image_token_id": IMAGE, "video_token_id": VIDEO, "audio_token_id": AUDIO,
        "boi_token_id": BOI, "eoi_token_id": EOI, "vision_soft_tokens_per_image": V_BUDGET,
    }


# ── Media ──────────────────────────────────────────────────────────────────────────────────────────────────


def synth_rgb(h: int, w: int, seed: int) -> np.ndarray:
    """A gradient with flat blocks and mild noise: smooth enough to survive a resize, busy enough to see one."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x + y) * 255 // max(h + w - 2, 1)], -1).astype(np.int32)
    for _ in range(3):
        y0, x0 = int(rng.integers(0, h // 2)), int(rng.integers(0, w // 2))
        img[y0 : y0 + h // 3, x0 : x0 + w // 3] = rng.integers(0, 255, 3)
    img += rng.integers(-12, 13, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def rnd(x):
    """Seven decimals: far below any tolerance the tests use, and a third of the JSON."""
    return np.round(np.asarray(x, dtype=np.float64), 7).tolist()


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def pattern_rgb(h: int, w: int) -> np.ndarray:
    """Closed-form source for the resize cases, so the Zig test rebuilds it instead of embedding it."""
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([(x * 7 + y * 13 + c * 61 + ((x * y) % 29) * 5) % 256 for c in range(3)], -1).astype(np.uint8)


def resize_cases() -> list[dict]:
    """torchvision's bicubic antialiased uint8 resize (what both processors call), on closed-form sources."""
    cases = []
    for (sh, sw), (th, tw) in [((120, 200), (48, 96)), ((20, 30), (48, 72))]:
        t = torch.from_numpy(pattern_rgb(sh, sw).copy()).permute(2, 0, 1)
        out = tvF.resize(t, [th, tw], interpolation=InterpolationMode.BICUBIC, antialias=True)
        cases.append({"src": [sh, sw], "dst": [th, tw], "rgb": b64(out.permute(1, 2, 0).numpy())})
    return cases


def mm_embedding(model: EmbeddingGemma2Model, ids: list[int], **kwargs) -> list[float]:
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long), **kwargs).last_hidden_state[0]
    pooled = out.mean(dim=0)
    return rnd(pooled / pooled.norm())


def media(model: EmbeddingGemma2Model) -> dict:
    ip = Gemma4ImageProcessor(patch_size=16, max_soft_tokens=V_BUDGET, pooling_kernel_size=3)
    vp = EmbeddingGemma2VideoProcessor(patch_size=16, max_soft_tokens=V_BUDGET, pooling_kernel_size=3, do_sample_frames=False)

    img = synth_rgb(50, 70, 11)
    io = ip(images=[Image.fromarray(img)], return_tensors="pt")
    n_img = int(io["num_soft_tokens_per_image"][0])
    with torch.no_grad():
        soft = model.get_image_features(io["pixel_values"], io["image_position_ids"], return_dict=True).pooler_output[0]
    img_kw = {"pixel_values": io["pixel_values"], "image_position_ids": io["image_position_ids"]}
    # The same image on a budget twice the checkpoint's `default_output_length`: a grid larger than the one the
    # tower was configured around (1260 patches against 630).
    big = Gemma4ImageProcessor(patch_size=16, max_soft_tokens=2 * V_BUDGET, pooling_kernel_size=3)(images=[Image.fromarray(img)], return_tensors="pt")
    with torch.no_grad():
        big_soft = model.get_image_features(big["pixel_values"], big["image_position_ids"], return_dict=True).pooler_output[0]
    block = lambda kind, n: [BOI] + [kind] * n + [EOI]  # noqa: E731
    image_ids = [2] + block(IMAGE, n_img) + [1]
    text_image_ids = [2, 7, 5] + block(IMAGE, n_img) + [9, 1]

    frames = np.stack([synth_rgb(40, 56, 20 + i) for i in range(3)])
    vo = vp(videos=[frames], return_tensors="pt")
    n_vid = int(vo["num_soft_tokens_per_video"][0])
    vid_kw = {"pixel_values_videos": vo["pixel_values_videos"], "video_position_ids": vo["video_position_ids"], "num_frames_per_video": vo["num_frames_per_video"]}
    with torch.no_grad():
        video_soft = model.get_video_features(vo["pixel_values_videos"], vo["video_position_ids"], vo["num_frames_per_video"], return_dict=True).pooler_output[0]
    video_ids = [2] + sum([block(VIDEO, n_vid) for _ in range(len(frames))], []) + [1]
    # The video comes first in the prompt, the image second: the soft rows must follow the prompt, not the kind.
    mixed_ids = [2, 7] + sum([block(VIDEO, n_vid) for _ in range(len(frames))], []) + [5] + block(IMAGE, n_img) + [9, 1]

    return {
        "image": {"h": 50, "w": 70, "rgb": b64(img), "target": list(get_aspect_ratio_preserving_size(50, 70, 16, V_BUDGET * 9, 3)), "tokens": n_img, "soft": rnd(soft)},
        "frames": {"h": 40, "w": 56, "rgb": [b64(f) for f in frames], "target": list(get_aspect_ratio_preserving_size(40, 56, 16, V_BUDGET * 9, 3)), "tokens": n_vid, "soft": rnd(video_soft)},
        "image_big": {"budget": 2 * V_BUDGET, "target": list(get_aspect_ratio_preserving_size(50, 70, 16, 2 * V_BUDGET * 9, 3)), "tokens": int(big["num_soft_tokens_per_image"][0]), "soft": rnd(big_soft)},
        "image_ids": image_ids, "image_embedding": mm_embedding(model, image_ids, **img_kw),
        "text_image_ids": text_image_ids, "text_image_embedding": mm_embedding(model, text_image_ids, **img_kw),
        "video_ids": video_ids, "video_embedding": mm_embedding(model, video_ids, **vid_kw),
        "mixed_ids": mixed_ids, "mixed_embedding": mm_embedding(model, mixed_ids, **img_kw, **vid_kw),
        "resize": resize_cases(),
    }


def tiny(out_dir: str, seed: int) -> None:
    model = build(seed)
    rows = [reference(model, ids) for ids in SEQUENCES]
    # The fixture must be able to see the band's edge: with the exclusive radius the long rows change.
    alt = make({"sliding_window": WINDOW - 1})
    alt.load_state_dict(model.state_dict())
    gap = min(cos(r[1], reference(alt, ids)[1]) for r, ids in zip(rows, SEQUENCES) if len(ids) > 2 * WINDOW + 2)
    assert gap < 0.9999, f"band edge invisible to this fixture (inclusive vs exclusive cosine {gap})"

    mm = media(model)
    # Image rows must reach the embedding: a text-only run of the same ids (soft slots left as the image token's
    # own embedding) lands far from the spliced one.
    blind = mm_embedding(model, mm["image_ids"])
    assert cos(blind, mm["image_embedding"]) < 0.98, "the image rows do not move the embedding"

    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors(model), os.path.join(out_dir, "embedding_gemma2_tiny.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "embedding_gemma2_tiny_config.json"), "w") as f:
        json.dump(release_config(), f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "embedding_gemma2_tiny_expected.json"), "w") as f:
        json.dump({"sequences": SEQUENCES, "norm": [rnd(r[0]) for r in rows], "embedding": [rnd(r[1]) for r in rows], "media": mm}, f, separators=(",", ":"))
    print(f"wrote {out_dir}: {sum(t.numel() for t in tensors(model).values())} params, band-edge cosine {gap:.5f}, "
          f"image {mm['image']['tokens']} soft tokens, video {mm['frames']['tokens']} per frame, blind-vs-spliced cosine {cos(blind, mm['image_embedding']):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiny")
    t.add_argument("--out", default="src/fixtures")
    t.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    tiny(args.out, args.seed)
