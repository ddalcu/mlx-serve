#!/usr/bin/env python3
"""Dump MiniCPM-V 4.7 (minicpmv4_7) VISION parity fixtures: the reference EXECUTES.

Runs transformers' own `MiniCPMV4_6ImageProcessorPil` (the processor the checkpoint maps to),
`MiniCPMV4_7Processor`, `MiniCPMV4_7VisionModel` + `MiniCPMV4_7Merger` (fp32, CPU) on the
checkpoint's dense tower weights, and `get_rope_index` (canvas M-RoPE), saving every stage
`src/minicpm_vision.zig` has to reproduce. Both sides read the same bf16 numbers, so a diff is a
layout/math bug. Preprocessing cases start from RGB bytes decoded here (our server decodes with
stb, the reference with libjpeg; the decoders are not under test):
  house  the repo's house.jpeg: sliced into a thumbnail plus a grid of detail crops
  noise  a random 61x97 image: below the slice threshold, so UPSCALED to one 448-class view
  wide   a random 900x2600 image: the slice grid is capped by `max_slice_nums`
The tower runs on `house` and `noise` in both downsample modes; the rope table on prompts with
one image, two images, text between them and the capped grid.

Usage:
    venv/bin/python -I tests/dump_minicpmv4_7_vision_fixtures.py \
        --model <HF checkpoint dir> --image tests/fixtures/house.jpeg \
        --out ~/claude-tmp/minicpm-vision/minicpm_vision_fixture.safetensors
"""

import argparse
import json
import os
import types

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import AutoProcessor
from transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6 import MiniCPMV4_6ImageProcessorPil
from transformers.models.minicpmv4_7 import modeling_minicpmv4_7 as M
from transformers.models.minicpmv4_7.configuration_minicpmv4_7 import MiniCPMV4_7Config

TOWER_CASES = ("noise", "house")
PROMPT_HEAD = "<|im_start|>user\n"
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_tower(model_dir, cfg):
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    tower, merger = {}, {}
    for key, shard in index.items():
        if not key.startswith(("model.vision_tower.", "model.merger.")):
            continue
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            v = f.get_tensor(key).float()
        (tower if key.startswith("model.vision_tower.") else merger)[key.split(".", 2)[2]] = v
    vm = M.MiniCPMV4_7VisionModel._from_config(cfg.vision_config, attn_implementation="eager").float().eval()
    mg = M.MiniCPMV4_7Merger(cfg).float().eval()
    for module, state in ((vm, tower), (mg, merger)):
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"missing {missing[:5]} unexpected {unexpected[:5]}")
    return vm, mg


def rope_for(cfg, input_ids, mm_type, target_sizes, mode):
    """`get_rope_index` on a bare namespace: it reads only `config` and its sibling helpers."""
    host = types.SimpleNamespace(config=cfg)
    for name in ("_group_visual_frames", "get_vision_position_ids"):
        setattr(host, name, types.MethodType(getattr(M.MiniCPMV4_7Model, name), host))
    host._frame_end_idx = M.MiniCPMV4_7Model._frame_end_idx
    pos, deltas = M.MiniCPMV4_7Model.get_rope_index(
        host, input_ids, mm_token_type_ids=mm_type, target_sizes=target_sizes, downsample_mode=mode)
    return pos[:, 0].to(torch.int32).contiguous(), deltas[0].to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the HF checkpoint directory")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rope-json", help="also write the canvas M-RoPE cases as the hermetic test fixture")
    args = ap.parse_args()

    cfg = MiniCPMV4_7Config.from_pretrained(args.model)
    proc_cfg = json.load(open(os.path.join(args.model, "preprocessor_config.json")))
    ip = MiniCPMV4_6ImageProcessorPil(**{k: v for k, v in proc_cfg.items()
                                         if k not in ("mrope_mode", "image_processor_type", "processor_class")})
    processor = AutoProcessor.from_pretrained(args.model)

    rng = np.random.default_rng(0x4707)
    sources = {
        "house": np.asarray(Image.open(args.image).convert("RGB")),
        "noise": rng.integers(0, 256, size=(61, 97, 3), dtype=np.uint8),
        "wide": rng.integers(0, 256, size=(900, 2600, 3), dtype=np.uint8),
    }

    out = {}
    packed = {}
    for name, rgb in sources.items():
        feat = ip(images=Image.fromarray(rgb), return_tensors="pt")
        pv, ts = feat["pixel_values"], feat["target_sizes"]
        out[f"{name}_rgb"] = torch.from_numpy(np.ascontiguousarray(rgb))
        out[f"{name}_pixel_values"] = pv.float().contiguous()
        out[f"{name}_target_sizes"] = ts.to(torch.int32).contiguous()
        out[f"{name}_grid"] = torch.tensor(feat["grids"][0], dtype=torch.int32)
        packed[name] = (pv.float(), ts)
        print(f"[{name}] {rgb.shape[0]}x{rgb.shape[1]} -> grid {feat['grids'][0]} pieces {ts.tolist()}")

    vm, mg = load_tower(args.model, cfg)
    stages = {}
    vm.embeddings.register_forward_hook(lambda _m, _i, o: stages.__setitem__("embeddings", o))
    vm.encoder.layers[cfg.insert_layer_id].register_forward_hook(lambda _m, _i, o: stages.__setitem__("layer_insert", o))
    vm.vit_merger.register_forward_hook(lambda _m, _i, o: stages.__setitem__("vit_merger", o))
    for name in TOWER_CASES:
        pv, ts = packed[name]
        for mode in ("16x", "4x"):
            with torch.no_grad():
                res = vm(pv, target_sizes=ts, use_vit_merger=(mode == "16x"))
                feats = mg(res.last_hidden_state, ts // 2 if mode == "16x" else ts)
            tag = f"{name}_{mode}"
            out[f"{tag}_embeddings"] = stages["embeddings"][0].contiguous()
            out[f"{tag}_layer_insert"] = stages["layer_insert"][0].contiguous()
            if mode == "16x":
                out[f"{tag}_vit_merger"] = stages["vit_merger"][0].contiguous()
            out[f"{tag}_post_ln"] = res.last_hidden_state[0].contiguous()
            out[f"{tag}_features"] = torch.cat(feats, dim=0).contiguous()
            print(f"[{tag}] features {tuple(out[f'{tag}_features'].shape)}")

    prompts = {
        "one": [("house", "Describe this image.")],
        "two": [("noise", "Compare with"), ("house", "Which is bigger?")],
        "capped": [("wide", "Read the text.")],
    }
    canvas_cases = []
    for pname, parts in prompts.items():
        text = PROMPT_HEAD
        images = []
        for src, words in parts:
            text += "<|image_pad|>\n" + words + "\n"
            images.append(Image.fromarray(sources[src]))
        text += PROMPT_TAIL
        for mode in ("16x", "4x"):
            enc = processor(images=images, text=[text], downsample_mode=mode, return_tensors="pt")
            ids = enc["input_ids"]
            mm_type = enc["mm_token_type_ids"]
            pos, delta = rope_for(cfg, ids, mm_type, enc["target_sizes"], mode)
            tag = f"{pname}_{mode}"
            out[f"{tag}_input_ids"] = ids[0].to(torch.int32).contiguous()
            out[f"{tag}_target_sizes"] = enc["target_sizes"].to(torch.int32).contiguous()
            out[f"{tag}_rope_pos"] = pos
            out[f"{tag}_rope_delta"] = delta
            print(f"[{tag}] {ids.shape[1]} tokens, rope delta {int(delta)}")
            pieces = [ip(images=img, return_tensors="pt") for img in images]
            canvas_cases.append({
                "name": tag, "merge": 2 if mode == "4x" else 4, "ids": ids[0].tolist(),
                "pre_ids": processor.tokenizer(text, add_special_tokens=False)["input_ids"],
                "images": [{"rows": f["grids"][0][0], "cols": f["grids"][0][1], "pieces": f["target_sizes"].tolist()} for f in pieces],
                "grids": enc["target_sizes"].tolist(), "pos": pos.tolist(), "delta": int(delta)})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out)
    print(f"wrote {args.out} ({len(out)} tensors)")
    if args.rope_json:
        markers = {k: getattr(cfg, k) for k in ("image_start_id", "image_end_id", "slice_start_id", "slice_end_id", "newline_id")}
        tk = processor.tokenizer
        tags = {"image_id_start": tk.convert_tokens_to_ids("<image_id>"), "image_id_end": tk.convert_tokens_to_ids("</image_id>"),
                "digits": [tk.encode(str(d), add_special_tokens=False)[0] for d in range(10)]}
        doc = {"image_token": cfg.image_token_id, "video_token": cfg.video_token_id, **markers, **tags, "cases": canvas_cases}
        with open(args.rope_json, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        print(f"wrote {args.rope_json}")


if __name__ == "__main__":
    main()
