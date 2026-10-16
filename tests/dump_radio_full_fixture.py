#!/usr/bin/env python3
"""Optional full-checkpoint RADIO oracle; saves outputs, never the model weights.

  python -I tests/dump_radio_full_fixture.py --reference <Bolt pack> --out <work/fixture.safetensors>
  RADIO_VISION_MODEL=<Bolt pack> RADIO_VISION_FIXTURE=<work/fixture.safetensors> \
    zig build test -Doptimize=ReleaseFast -Dtest-filter='radio full checkpoint'

Uses the same pinned dependencies and NVIDIA source loader as dump_radio_vision_fixtures.py.
Only the vision shards are loaded. All reference computation runs in fp32 on CPU.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import types

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("radio_fixture", Path(__file__).with_name("dump_radio_vision_fixtures.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.set_num_threads(8)
    radio, config, processor, omni = helper.reference_modules(args.reference.resolve())
    raw = json.loads((args.reference / "config.hf.json").read_text())
    cfg = config.RadioConfig(**raw["vision_config"])
    cfg.dtype = torch.float32
    cfg._attn_implementation = "sdpa"
    tower = radio.RadioModel(cfg).float().eval()
    tower.make_preprocessor_external()
    pcfg = types.SimpleNamespace(**{**raw, "llm_config": types.SimpleNamespace(**raw["llm_config"]), "vision_config": cfg})
    projector = omni["NemotronH_Omni_Reasoning_V3VisionProjector"](pcfg).float().eval()
    index = json.loads((args.reference / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({v for k, v in index.items() if k.startswith(("vision_model.", "mlp1."))})
    state, projection = {}, {}
    for file in files:
        with safe_open(str(args.reference / file), framework="pt") as shard:
            for key in shard.keys():
                value = shard.get_tensor(key).float()
                if key.startswith("mlp1."):
                    name = key[5:].replace("0.weight", "norm.weight").replace("1.weight", "linear1.weight").replace("3.weight", "linear2.weight")
                    projection[name] = value
                elif key.startswith("vision_model.radio_model.model."):
                    name = key.removeprefix("vision_model.radio_model.model.")
                    name = name.replace("patch_generator.video_embedder", "embeddings.video_patch_projection").replace(
                        "patch_generator.embedder", "embeddings.patch_projection").replace(
                        "patch_generator.pos_embed", "embeddings.position_embedding").replace(
                        "patch_generator.cls_token.token", "embeddings.cls_register_token").replace("blocks.", "encoder.layer.")
                    if ".attn.qkv." in name:
                        for suffix, part in zip(("query", "key", "value"), value.chunk(3, 0)):
                            state[name.replace("attn.qkv", "attention.attention." + suffix)] = part
                    else:
                        state[name.replace("attn.proj", "attention.output.dense")] = value
    missing, extra = tower.load_state_dict(state, strict=False)
    assert not extra and all("layer_scale" in name or name == "summary_idxs" for name in missing), (missing, extra)
    projector.mlp1.load_state_dict(projection)
    del state, projection
    clip = Path(__file__).resolve().parents[1] / "src/fixtures/radio_clip.mov"
    with tempfile.TemporaryDirectory() as tmp:
        images = helper.decode_clip(clip, Path(tmp))
        proc = processor.NemotronH_Omni_Reasoning_V3ImageProcessor(norm_mean=cfg.norm_mean, norm_std=cfg.norm_std)
        image = proc._preprocess([images[0]])["pixel_values"]
        proc._is_video_mode = True
        video = proc._preprocess(images)["pixel_values"]
        with torch.inference_mode():
            image_features = projector(image, tower)
            video_features = projector.forward_video(video, tower).reshape(1, -1, raw["llm_config"]["hidden_size"])
            thw = torch.tensor([3, video.shape[-2] // 32, video.shape[-1] // 32])
            mask = omni["compute_retention_mask"](video_embeds=video_features[0], thw=thw, spatial_merge_size=1, q=0.3)
    save_file({"image_pixels": image.contiguous(), "video_pixels": video.contiguous(),
               "video_rgb": torch.from_numpy(np.stack([np.asarray(im).copy() for im in images])),
               "image_features": image_features.contiguous(), "video_features": video_features.contiguous(),
               "video_mask": mask.contiguous()}, args.out)
    print(f"wrote {args.out}: image={tuple(image_features.shape)} video={tuple(video_features.shape)}", flush=True)


if __name__ == "__main__":
    main()
