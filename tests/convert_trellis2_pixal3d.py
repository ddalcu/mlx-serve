#!/usr/bin/env python3
"""Repackage `microsoft/TRELLIS.2-4B` or `TencentARC/Pixal3D` as an 8-bit mlx-serve pack.

USER-RUN (needs mlx + the official weights + the DINOv3 checkpoint). One file at a time. The pack keeps
the release's layout and key names (what src/trellis2.zig and src/pixal3d.zig load) with these changes:

  (a) the DiT block projections (`blocks.N.*.weight`, 2-D, input width a multiple of the group) are
      affine-quantized (group 64). The loaders decide dense-vs-quantized PER TENSOR from a `.scales`
      sibling, and read the group size from the DiT json's mlx-lm style `quantization` block;
  (b) the DiT json gains that block;
  (c) the shape/texture ENCODERS are dropped (image-to-3D never runs them).

NOT quantized: the decoders and the sparse-structure decoder (convolutions, under 1 GB together), the
DiTs' input/output layers, time embedder and AdaLN modulation, every norm gamma and the RoPE phases.
DINOv3 (the image encoder, Meta-gated upstream) rides in the pack as float16, or 8-bit with --dino-bits 8,
with its licence; the NAF upsampler (Pixal3D) rides with its Apache licence. A source without a LICENSE is refused.

    python3 tests/convert_trellis2_pixal3d.py --model trellis2 --src <TRELLIS.2-4B> --ss-dec <dir with ss_dec_conv3d_16l8_fp16.*> \\
        --dinov3 <dinov3 dir> --license <LICENSE> --out <dir>
    python3 tests/convert_trellis2_pixal3d.py --model pixal3d --src <Pixal3D> --naf <dir with model.safetensors> \\
        --dinov3 <dinov3 dir> --out <dir>
    python3 tests/convert_trellis2_pixal3d.py --self-test
"""

import argparse
import glob
import json
import os
import shutil
import sys

GROUP_SIZE = 64
SS_DEC = "ss_dec_conv3d_16l8_fp16"

MODELS = {
    "trellis2": {
        "base": "microsoft/TRELLIS.2-4B",
        "title": "TRELLIS.2-4B",
        "blurb": "Microsoft's image-to-3D model: an image with a transparent background becomes a textured PBR GLB.",
        "ss_dec_in_ckpts": False,
    },
    "pixal3d": {
        "base": "TencentARC/Pixal3D",
        "title": "Pixal3D",
        "blurb": (
            "Tencent's image-to-3D model, built on the TRELLIS.2 backbone with pixel-aligned conditioning: an image "
            "with a transparent background becomes a textured PBR GLB. Single-view and multi-view (`*_mv`) models."
        ),
        "ss_dec_in_ckpts": True,
    },
}

NAF_NOTE = "\nThe NAF feature upsampler in `naf/` is Apache-2.0 (`naf/LICENSE`)."

README = """\
---
base_model: {base}
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: mit-with-bundled-dinov3-license
license_link: LICENSE
pipeline_tag: image-to-3d
tags:
- mlx
- mlx-serve
- image-to-3d
- 3d
- glb
---

# {title} for mlx-serve (8-bit)

[{base}](https://huggingface.co/{base}) repackaged for [mlx-serve](https://github.com/ddalcu/mlx-serve)
on Apple Silicon. {blurb}

- The flow-model (DiT) block projections are affine-quantized to 8 bit, group size 64; everything else is as released.
- Includes the DINOv3 image encoder it conditions on ({dino}); its licence is in `dinov3/LICENSE.md`.
- No background remover is included: send an image with a transparent background (alpha is the subject).

## Licences

The {title} weights are MIT (`LICENSE`). The bundled DINOv3 encoder is Meta's, under the DINOv3 License
(`dinov3/LICENSE.md`), which travels with any redistribution of it.{naf_note}

Run: `mlx-serve --model-dir <folder holding this repo>`, then `POST /v1/3d/generations`.
Converted by `tests/convert_trellis2_pixal3d.py` in the mlx-serve repository.
"""


def copy_licence(src_dir, dst_path, what):
    """Copies the licence file found in `src_dir`; a pack never ships without the licence of what it redistributes."""
    names = sorted(n for n in os.listdir(src_dir) if n.upper().startswith(("LICENSE", "LICENCE")))
    if not names:
        raise SystemExit(f"{what}: no LICENSE file in {src_dir}; refusing to build a pack that redistributes it without one")
    shutil.copy(os.path.join(src_dir, names[0]), dst_path)


def is_dit(stem):
    return "_dit_" in stem


def is_encoder(stem):
    return "_enc_" in stem


def should_quantize(name, shape):
    return name.startswith("blocks.") and name.endswith(".weight") and len(shape) == 2 and shape[1] % GROUP_SIZE == 0


def should_quantize_dino(name, shape):
    return name.startswith(("layer.", "model.layer.")) and name.endswith(".weight") and "norm" not in name and len(shape) == 2 and shape[1] % GROUP_SIZE == 0


def quantize_file(src, dst, bits, rule):
    import mlx.core as mx

    out = {}
    n_q = 0
    for name, tensor in mx.load(src).items():
        if rule(name, tensor.shape):
            base = name[: -len(".weight")]
            # scales and biases in bf16: the blocks compute in bf16
            q, s, b = mx.quantize(tensor.astype(mx.float32), group_size=GROUP_SIZE, bits=bits)
            new = {name: q, base + ".scales": s.astype(mx.bfloat16), base + ".biases": b.astype(mx.bfloat16)}
            n_q += 1
        else:
            new = {name: tensor}
        mx.eval(*new.values())
        out.update(new)
    mx.save_safetensors(dst, out)
    mx.clear_cache()
    return n_q


def to_f16(src, dst):
    import mlx.core as mx

    t = {k: (v.astype(mx.float16) if v.dtype == mx.float32 else v) for k, v in mx.load(src).items()}
    mx.eval(*t.values())
    mx.save_safetensors(dst, t)
    mx.clear_cache()


def convert_ckpts(src, out, bits, keep_ss_dec):
    ckpts = os.path.join(out, "ckpts")
    os.makedirs(ckpts, exist_ok=True)
    total = 0
    for path in sorted(glob.glob(os.path.join(src, "ckpts", "*.safetensors"))):
        stem = os.path.basename(path)[: -len(".safetensors")]
        if is_encoder(stem) or (stem == SS_DEC and not keep_ss_dec):
            continue
        dst = os.path.join(ckpts, stem + ".safetensors")
        cfg_path = os.path.join(src, "ckpts", stem + ".json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        if is_dit(stem):
            n = quantize_file(path, dst, bits, should_quantize)
            cfg["quantization"] = cfg["quantization_config"] = {"group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}
            print(f"  {stem}: {n} projections -> {bits}-bit, {os.path.getsize(dst) / 1e9:.2f} GB")
        else:
            shutil.copy(path, dst)
            print(f"  {stem}: copied, {os.path.getsize(dst) / 1e9:.2f} GB")
        with open(os.path.join(ckpts, stem + ".json"), "w") as f:
            json.dump(cfg, f, indent=2)
        total += os.path.getsize(dst)
    return total


def convert_dinov3(src, out, dino_bits):
    dst = os.path.join(out, "dinov3")
    os.makedirs(dst, exist_ok=True)
    weights = os.path.join(src, "model.safetensors")
    target = os.path.join(dst, "model.safetensors")
    if dino_bits == 8:
        n = quantize_file(weights, target, 8, should_quantize_dino)
        print(f"  dinov3: {n} projections -> 8-bit, {os.path.getsize(target) / 1e9:.2f} GB")
    else:
        to_f16(weights, target)
        print(f"  dinov3: float16, {os.path.getsize(target) / 1e9:.2f} GB")
    cfg = json.load(open(os.path.join(src, "config.json")))
    if dino_bits == 8:
        cfg["quantization"] = cfg["quantization_config"] = {"group_size": GROUP_SIZE, "bits": 8, "mode": "affine"}
    else:
        cfg["torch_dtype"] = "float16"
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    copy_licence(src, os.path.join(dst, "LICENSE.md"), "DINOv3")
    return os.path.getsize(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--src", help="the official repo: ckpts/ and pipeline.json")
    ap.add_argument("--ss-dec", help="trellis2: a folder holding the TRELLIS v1 sparse-structure decoder")
    ap.add_argument("--naf", help="pixal3d: a folder holding NAF's model.safetensors")
    ap.add_argument("--dinov3", help="facebook/dinov3-vitl16-pretrain-lvd1689m, as safetensors")
    ap.add_argument("--dino-bits", type=int, choices=(8, 16), default=16)
    ap.add_argument("--license", help="the model's LICENSE file, when the source folder has none")
    ap.add_argument("--out")
    ap.add_argument("--bits", type=int, choices=(4, 8), default=8)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.model and args.src and args.dinov3 and args.out):
        ap.error("--model, --src, --dinov3 and --out are required")
    spec = MODELS[args.model]
    os.makedirs(args.out, exist_ok=True)

    total = convert_ckpts(args.src, args.out, args.bits, spec["ss_dec_in_ckpts"])
    if not spec["ss_dec_in_ckpts"]:
        if not args.ss_dec:
            ap.error("--ss-dec is required for trellis2")
        dst = os.path.join(args.out, "microsoft", "TRELLIS-image-large", "ckpts")
        os.makedirs(dst, exist_ok=True)
        for ext in (".safetensors", ".json"):
            shutil.copy(os.path.join(args.ss_dec, SS_DEC + ext), os.path.join(dst, SS_DEC + ext))
    total += convert_dinov3(args.dinov3, args.out, args.dino_bits)
    if args.model == "pixal3d":
        if not args.naf:
            ap.error("--naf is required for pixal3d")
        os.makedirs(os.path.join(args.out, "naf"), exist_ok=True)
        shutil.copy(os.path.join(args.naf, "model.safetensors"), os.path.join(args.out, "naf", "model.safetensors"))
        copy_licence(args.naf, os.path.join(args.out, "naf", "LICENSE"), "NAF")
    for name in ("pipeline.json", "pipeline_mv.json", "LICENSE", "NOTICE"):
        path = os.path.join(args.src, name)
        if os.path.exists(path):
            shutil.copy(path, os.path.join(args.out, name))
    if args.license and not os.path.exists(os.path.join(args.out, "LICENSE")):
        shutil.copy(args.license, os.path.join(args.out, "LICENSE"))
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(README.format(base=spec["base"], title=spec["title"], blurb=spec["blurb"],
                              dino="float16" if args.dino_bits == 16 else "8-bit",
                              naf_note=NAF_NOTE if args.model == "pixal3d" else ""))
    print(f"done: {total / 1e9:.2f} GB of weights -> {args.out}")


def self_test():
    q = should_quantize
    assert q("blocks.0.self_attn.to_qkv.weight", (4608, 1536))
    assert q("blocks.29.mlp.mlp.2.weight", (1536, 8192))
    assert q("blocks.3.cross_attn.cross_attn_block.to_kv.weight", (3072, 1024))
    assert q("blocks.3.cross_attn.proj_linear.weight", (1536, 2048))
    assert not q("blocks.0.self_attn.q_rms_norm.gamma", (12, 128))
    assert not q("blocks.0.modulation", (9216,))
    assert not q("input_layer.weight", (1536, 32))
    assert not q("out_layer.weight", (32, 1536))
    assert not q("t_embedder.mlp.0.weight", (1536, 256))
    assert not q("adaLN_modulation.1.weight", (9216, 1536))
    assert should_quantize_dino("layer.3.attention.q_proj.weight", (1024, 1024))
    assert should_quantize_dino("model.layer.3.mlp.up_proj.weight", (4096, 1024))
    assert not should_quantize_dino("layer.3.norm1.weight", (1024,))
    assert not should_quantize_dino("layer.3.layer_scale1.lambda1", (1024,))
    assert not should_quantize_dino("embeddings.patch_embeddings.weight", (1024, 3, 16, 16))
    assert is_dit("slat_flow_img2shape_dit_1_3B_1024_bf16_mv") and not is_dit("shape_dec_next_dc_f16c32_fp16")
    assert is_encoder("shape_enc_next_dc_f16c32_fp16") and not is_encoder("ss_flow_img_dit_1_3B_64_bf16")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        try:
            copy_licence(d, os.path.join(d, "out"), "X")
            raise AssertionError("a source without a LICENSE must be refused")
        except SystemExit:
            pass
        open(os.path.join(d, "LICENSE"), "w").write("terms")
        copy_licence(d, os.path.join(d, "out"), "X")
        assert open(os.path.join(d, "out")).read() == "terms"
    assert "license: other" in README and "{naf_note}" in README
    import mlx.core as mx

    w = mx.random.normal((256, 1536)) * 0.02
    qw, s, b = mx.quantize(w, group_size=GROUP_SIZE, bits=8)
    err = mx.sqrt(mx.mean((mx.dequantize(qw, s, b, group_size=GROUP_SIZE, bits=8) - w) ** 2)) / mx.sqrt(mx.mean(w**2))
    assert float(err) < 0.01, float(err)
    print("ok")


if __name__ == "__main__":
    main()
