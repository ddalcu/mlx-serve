#!/usr/bin/env python3
"""Execute NVIDIA's RADIO tower, projector and EVS on a tiny decoded video.

  python -I tests/dump_radio_vision_fixtures.py --reference <Bolt checkpoint>

Requires torch 2.10.0, torchvision 0.25.0, transformers 5.8.1, safetensors 0.8.0,
numpy 2.5.4 and Pillow 12.3.0. The checkpoint's modeling_radio.py executes
unchanged; only the independent vision classes/functions are imported from
modeling_nemotron_h_omni.py, avoiding its CUDA-only Mamba import. No numerical
reference forward is reimplemented here. A small 2-head, 2-layer tower keeps
the checked-in fp32 fixture hermetic. The source MOV has five distinct frames.
"""

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import warnings

import numpy as np
from PIL import Image
import torch
from torch import nn
from safetensors.torch import save_file
from transformers.activations import ACT2FN


def reference_modules(path):
    package = types.ModuleType("bolt_radio_reference")
    package.__path__ = [str(path)]
    sys.modules[package.__name__] = package
    radio = importlib.import_module(package.__name__ + ".modeling_radio")
    config = importlib.import_module(package.__name__ + ".configuration_radio")
    processor = importlib.import_module(package.__name__ + ".image_processing_nemotron_h_omni")
    wanted = {"NemotronH_Omni_RMSNorm", "NemotronH_Omni_Reasoning_V3MLP",
              "NemotronH_Omni_Reasoning_V3VisionProjector", "compute_retention_mask"}
    tree = ast.parse((path / "modeling_nemotron_h_omni.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    for node in nodes:
        node.decorator_list = []
    ns = {"torch": torch, "nn": nn, "ACT2FN": ACT2FN, "warnings": warnings}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path / "modeling_nemotron_h_omni.py"), "exec"), ns)
    return radio, config, processor, ns


def decode_clip(clip, output):
    # AVFoundation is only a fixture-generation dependency; the engine receives RGB frames.
    swift = r'''
import AVFoundation
import AppKit
let asset = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))
let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter = .zero
for i in 0..<5 {
    let image = try generator.copyCGImage(at: CMTime(value: Int64(i), timescale: 2), actualTime: nil)
    let data = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])!
    try data.write(to: URL(fileURLWithPath: CommandLine.arguments[2]).appendingPathComponent("\(i).png"))
}
'''
    script = output / "decode.swift"
    script.write_text(swift)
    subprocess.run(["swift", str(script), str(clip), str(output)], check=True)
    return [Image.open(output / f"{i}.png").convert("RGB") for i in range(5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "src/fixtures/radio_tiny.safetensors")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0xB017)
    radio, config, processor, omni = reference_modules(args.reference.resolve())
    cfg = config.RadioConfig(hidden_size=8, num_attention_heads=2, num_hidden_layers=2,
                             num_cls_tokens=1, num_registers=1, summary_idxs=[0],
                             mlp_ratio=2.0, patch_size=16, max_img_size=32,
                             video_temporal_patch_size=2, layerscale_value=1.0)
    cfg._attn_implementation = "eager"
    tower = radio.RadioModel(cfg).float().eval()
    tower.make_preprocessor_external()
    pcfg = types.SimpleNamespace(downsample_ratio=0.5, video_temporal_patch_size=2,
                                 ps_version="v2", vit_hidden_size=8, projector_hidden_size=16,
                                 llm_config=types.SimpleNamespace(hidden_size=6),
                                 vision_final_layernorm=False)
    projector = omni["NemotronH_Omni_Reasoning_V3VisionProjector"](pcfg).float().eval()
    with torch.no_grad():
        for name, value in list(tower.named_parameters()) + list(projector.named_parameters()):
            if "layer_scale" in name:
                value.fill_(1)
            elif "norm" in name and name.endswith("weight"):
                value.uniform_(0.8, 1.2)
            else:
                value.uniform_(-0.2, 0.2)
    clip = Path(__file__).resolve().parents[1] / "src/fixtures/radio_clip.mov"
    with tempfile.TemporaryDirectory() as tmp:
        images = decode_clip(clip, Path(tmp))
        proc = processor.NemotronH_Omni_Reasoning_V3ImageProcessor(
            norm_mean=cfg.norm_mean, norm_std=cfg.norm_std,
            min_num_patches=8, max_num_patches=8, video_target_num_patches=8)
        image = proc._preprocess([images[0]])["pixel_values"]
        proc._is_video_mode = True
        video = proc._preprocess(images)["pixel_values"]
        out = {"image_pixels": image, "video_pixels": video,
               "video_rgb": torch.from_numpy(np.stack([np.asarray(im).copy() for im in images]))}
        with torch.no_grad():
            out["image_features"] = projector(image, tower)
            feats = projector.forward_video(video, tower)
            out["video_features"] = feats.reshape(1, -1, 6)
            out["video_mask"] = omni["compute_retention_mask"](
                video_embeds=feats.reshape(-1, 6), thw=torch.tensor([3, 1, 2]), spatial_merge_size=1, q=0.3)
            out["image_positions"] = tower.embeddings._interpolate_position_embedding((2, 4), torch.float32)
            evs = feats.reshape(-1, 6).repeat(4, 1)[:20].contiguous()
            out["evs_features"] = evs.reshape(1, 20, 6)
            for name, rate in (("10", 0.1), ("30", 0.3), ("70", 0.7)):
                out["evs_mask_" + name] = omni["compute_retention_mask"](
                    video_embeds=evs, thw=torch.tensor([5, 2, 2]), spatial_merge_size=1, q=rate)
        for name, value in tower.state_dict().items():
            if name.startswith("embeddings."):
                name = name.replace("embeddings.patch_projection", "patch_generator.embedder").replace(
                    "embeddings.video_patch_projection", "patch_generator.video_embedder").replace(
                    "embeddings.position_embedding", "patch_generator.pos_embed").replace(
                    "embeddings.cls_register_token", "patch_generator.cls_token.token")
                out["vision_model.radio_model.model." + name] = value
        for i, layer in enumerate(tower.encoder.layer):
            base = f"vision_model.radio_model.model.blocks.{i}."
            for name in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"):
                out[base + name] = layer.state_dict()[name]
            for suffix in ("weight", "bias"):
                out[base + "attn.qkv." + suffix] = torch.cat([getattr(getattr(layer.attention.attention, n), suffix) for n in ("query", "key", "value")])
                out[base + "attn.proj." + suffix] = getattr(layer.attention.output.dense, suffix)
        for source, target in (("norm.weight", "0.weight"), ("linear1.weight", "1.weight"), ("linear2.weight", "3.weight")):
            out["mlp1." + target] = projector.mlp1.state_dict()[source]
        # The same float interpolation path is tested in both directions, with overshoot.
        rgb = np.random.default_rng(0xB017).integers(0, 256, (8, 9, 3), dtype=np.uint8)
        out["resize_rgb"] = torch.from_numpy(rgb)
        for name, size in (("down", (3, 4)), ("up", (11, 13))):
            raw = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
            out["resize_" + name] = torch.nn.functional.interpolate(raw, size=size, mode="bicubic", align_corners=False, antialias=True)[0]
    save_file({k: v.detach().contiguous() for k, v in out.items()}, args.out)
    metadata = {"source": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16", "dtype": "float32",
                "clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                "files_sha256": {name: hashlib.sha256((args.reference / name).read_bytes()).hexdigest()
                                  for name in ("configuration_radio.py", "modeling_radio.py", "modeling_nemotron_h_omni.py", "image_processing_nemotron_h_omni.py")},
                "versions": {name: importlib.import_module(name).__version__ for name in ("torch", "transformers", "numpy", "safetensors", "PIL")},
                "geometry": {"hidden_size": 8, "heads": 2, "layers": 2, "patch": 16, "temporal_patch": 2,
                             "image_grid": [2, 4], "real_video_frames": 5, "projector_hidden": 16, "output_hidden": 6}}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.out}: image {tuple(image.shape)}, video {tuple(video.shape)}, features {tuple(feats.shape)}")


if __name__ == "__main__":
    main()
