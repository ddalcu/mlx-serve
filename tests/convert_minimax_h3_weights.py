#!/usr/bin/env python3
"""Convert MiniMax-H3 bf16 releases into an mlx-serve 8-bit model directory.

Produces the layout `minimax_h3.GenPaths` wants, as ONE self-contained model
dir — which also fixes the upstream split where the weights live in
`Comfy-Org/MiniMax-H3` but the tokenizer only exists in `MiniMaxAI/MiniMax-H3`:

    <out>/
      config.json                 model_type minimax_h3, so discovery can
                                  classify it WITHOUT a configless-repo probe
      transformer.safetensors     8-bit affine
      text_encoder.safetensors    8-bit affine
      video_vae.safetensors       copied verbatim (fp16, ~5 GB)
      audio_vae.safetensors       copied verbatim (fp32, ~0.6 GB)
      tokenizer.json / vocab.json / merges.txt / tokenizer_config.json
      README.md                   card with base_model_relation: quantized

Quantization is MLX affine (group_size 64), i.e. exactly what `MfLinear`
already solves geometry for, so the output loads through the SAME path as the
bf16 files with no loader change.

WHAT IS NOT QUANTIZED, and why it is an explicit list rather than a shape rule:
a shape-based "is this a linear?" test happily packs `embed_tokens.weight` and
`visual.pos_embed.weight`, which are read with `take_axis` — the gather then
returns packed uint32 garbage. Lookup tables are the one case shape cannot
decide. The checkpoint's fp32 islands (patch projections, output heads, time
embedder, rope inv_freq) are also kept dense: they are small, and narrowing a
distilled model's conditioning path is the MageFlow precision class.

Memory: source tensors are read LAZILY one at a time, so the 62 GB DiT never
lands in RAM; peak is roughly the size of the output being accumulated.

MiniMax ships TWO checkpoints — FL2VA (first/last-frame conditioning) and
REF2VA (reference images / videos / audio). They share this layout and differ
only in the DiT file and what `config.json` declares, so `--partition` selects
which one is being built; nothing downstream branches on the file NAME.

Usage:
    uv run --with mlx --with safetensors --with numpy \
        tests/convert_minimax_h3_weights.py \
        --src ~/claude-tmp/h3-build/src \
        --tokenizer ~/claude-tmp/h3-build/src_orig/FL2VA/processor \
        --out ~/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit

    # the reference-conditioning checkpoint
    uv run --with mlx --with safetensors --with numpy \
        tests/convert_minimax_h3_weights.py --partition ref2va \
        --src ~/claude-tmp/h3-build/src \
        --tokenizer ~/claude-tmp/h3-build/src_orig/REF2VA/processor \
        --out ~/.mlx-serve/models/ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit

    # FastH3 Omni: FastVideo's 8-step PDD student of the REF2VA DiT, read from
    # its diffusers release; everything else is cloned from an H3 pack
    uv run --with mlx tests/convert_minimax_h3_weights.py --partition omni \
        --src <FastVideo/FastVideo-FastH3-Omni-8-Step-V1 snapshot> \
        --reuse-from ~/.mlx-serve/models/ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit \
        --out ~/.mlx-serve/models/ddalcu/FastH3-Omni-MLX-Serve-8bit
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import time

import mlx.core as mx

GROUP_SIZE = 64
BITS = 8  # default; --bits 4 builds the small pack (compute-neutral under
          # dq-gemm — the DiT is GEMM-bound — so 4-bit buys FOOTPRINT: DiT
          # resident ~11 GB after AdaLN precompute, pack ~38 GB vs 69)

# Substrings that must NEVER be quantized. Two classes:
#   * GATHERED tables (read with take_axis, not a matmul) — packing them is
#     silent garbage, and shape cannot tell them from a linear.
#   * the checkpoint's own fp32 islands and learned token buffers.
NEVER_QUANTIZE = (
    "embed_tokens",          # gathered
    "pos_embed",             # gathered
    "register_tokens",       # learned buffer
    "mask_token",            # learned buffer
    "rope.inv_freq",         # fp32 island
    "time_embedder",         # fp32 island
    "video_patch_proj",      # fp32 island
    "audio_patch_proj",      # fp32 island
    "final_layer.video_out",  # fp32 island
    "final_layer.audio_out",  # fp32 island
    "latents_mean",
    "latents_std",
)


def should_quantize(name, shape, dtype_str):
    if any(k in name for k in NEVER_QUANTIZE):
        return False
    if len(shape) != 2:
        return False
    if dtype_str not in ("BF16", "F16", "F32"):
        return False
    out_f, in_f = shape
    # group_size must divide the contraction dim, and a tiny matrix is not
    # worth the scale/bias overhead.
    if in_f % GROUP_SIZE != 0:
        return False
    return in_f >= 256 and out_f >= 256


DTYPE_NAME = {
    mx.bfloat16: "BF16", mx.float16: "F16", mx.float32: "F32",
}


# diffusers -> ComfyUI module roots, as FastVideo's own Comfy export renames them
# (FastVideo/FastH3-Omni-Comfy `conversion/convert_omniref_comfy.py`).
FASTVIDEO_ROOTS = {
    "proj_in": "video_patch_proj", "audio_proj_in": "audio_patch_proj",
    "context_embedder": "condition_proj", "proj_out": "final_layer.video_out",
    "audio_proj_out": "final_layer.audio_out", "norm_out.linear": "final_layer.adaln_proj.linear",
    "norm_out.norm": "final_layer.norm", "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
}


def comfy_name(name):
    for before in sorted(FASTVIDEO_ROOTS, key=len, reverse=True):
        if name.startswith(before + "."):
            return FASTVIDEO_ROOTS[before] + name[len(before):]
    name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    if name.startswith("transformer_blocks."):
        name = "blocks." + name[len("transformer_blocks."):]
    return (name.replace(".attn.norm_q.", ".attn.q_norm.").replace(".attn.norm_k.", ".attn.k_norm.")
            .replace(".attn.to_out.0.", ".attn.out_proj.").replace(".ff.net.2.", ".mlp.fc2.")
            .replace(".ff.net.0.proj.", ".mlp.fc1."))


def fastvideo_dit(dit_dir):
    """A FastVideo diffusers DiT as the lazy Comfy-named tensors this engine
    reads: Q/K/V fused in that order, SwiGLU halves swapped from (value, gate)
    to (gate, value) and RoPE inverse frequencies generated. Every other tensor
    is consumed, the PDD head banks and the VSA `to_gate_compress` projections
    included."""
    cfg = json.load(open(os.path.join(dit_dir, "config.json")))
    src = {}
    for shard in sorted(glob.glob(os.path.join(dit_dir, "*.safetensors"))):
        src.update(mx.load(shard))
    out = {}
    for k in sorted(src):
        if ".attn.to_k." in k or ".attn.to_v." in k:
            continue
        if ".attn.to_q." in k:
            qkv = [src[k.replace(".to_q.", f".to_{p}.")] for p in "qkv"]
            out[comfy_name(k).replace(".attn.to_q.", ".attn.qkv_proj.")] = mx.concatenate(qkv, axis=0)
        elif ".ff.net.0.proj." in k:
            half = src[k].shape[0] // 2
            out[comfy_name(k)] = mx.concatenate([src[k][half:], src[k][:half]], axis=0)
        else:
            out[comfy_name(k)] = src[k]
    gates = sum(".to_gate_compress." in k for k in src)
    fused_away = 2 * (cfg["num_layers"] + cfg["num_refiner_layers"])  # to_k, to_v
    if gates != cfg["num_layers"] or len(src) != len(out) + fused_away:
        raise SystemExit(f"unexpected FastVideo DiT key set ({len(src)} source, {len(out)} mapped)")
    half = cfg["rope_freq_dim"]
    out["rope.inv_freq"] = 1.0 / (cfg["rope_theta"] ** (mx.arange(0, 2 * half, 2, dtype=mx.float32) / (2 * half)))
    return out


def convert_file(src_path, dst_path, label):
    t0 = time.time()
    out = {}
    n_q = n_d = 0
    q_bytes = d_bytes = 0
    # mx.load mmaps and returns LAZY arrays, so the 62 GB source never lands in
    # RAM as a whole. (safetensors' numpy framework cannot represent bfloat16 at
    # all — `data type 'bfloat16' not understood` — so this is not just faster,
    # it is the only path that reads these files.)
    src = fastvideo_dit(src_path) if os.path.isdir(src_path) else mx.load(src_path)
    keys = list(src.keys())
    for i, k in enumerate(keys):
        arr = src.pop(k)
        dtype_str = DTYPE_NAME.get(arr.dtype, str(arr.dtype))
        if should_quantize(k, list(arr.shape), dtype_str):
            # mx.quantize returns (packed uint32, scales, biases) — the exact
            # affine triple MfLinear solves bits/group_size from.
            w, scales, biases = mx.quantize(arr, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(w, scales, biases)
            base = k.rsplit(".", 1)[0]
            out[k] = w
            out[base + ".scales"] = scales
            out[base + ".biases"] = biases
            n_q += 1
            q_bytes += w.nbytes + scales.nbytes + biases.nbytes
        else:
            mx.eval(arr)
            out[k] = arr
            n_d += 1
            d_bytes += arr.nbytes
        del arr
        if (i + 1) % 100 == 0:
            print(f"  [{label}] {i+1}/{len(keys)}  "
                  f"{(q_bytes+d_bytes)/1e9:.1f} GB so far", flush=True)
    mx.save_safetensors(dst_path, out, metadata={
        "quantization": f"affine {BITS}-bit g{GROUP_SIZE}",
        "quantized_tensors": str(n_q),
        "dense_tensors": str(n_d),
        # Section III.2 of the MiniMax H3 Community License: modified files must
        # carry a prominent notice that they were modified. The sidecar
        # MODIFICATIONS.md is the human-readable copy; this travels INSIDE the
        # file, so it survives being moved out of the directory.
        "modified_from": PARTITIONS[CONFIG["partition"]]["upstream"],
        "modified_by": "mlx-serve tests/convert_minimax_h3_weights.py",
        "modification": f"quantized to MLX affine {BITS}-bit group size {GROUP_SIZE}",
        "license": "MiniMax H3 Community License Agreement",
    })
    total = (q_bytes + d_bytes) / 1e9
    print(f"  [{label}] {n_q} quantized / {n_d} dense -> {total:.1f} GB "
          f"in {time.time()-t0:.0f}s", flush=True)
    return total


# Per-partition facts. The DiT filename and the declared task list are the ONLY
# things that differ; the text encoder, both VAEs and every geometry number are
# shared, which is why one converter builds both.
PARTITIONS = {
    "fl2va": {
        "dit_file": "minimax_h3_fl2va_bf16.safetensors",
        "tasks": ["t2va", "fl2va"],
        "upstream": "MiniMaxAI/MiniMax-H3 (FL2VA)",
        "title": "MiniMax-H3 FL2VA",
        "blurb": "Text-to-audio-video with optional first/last-frame conditioning.",
    },
    "ref2va": {
        "dit_file": "minimax_h3_ref2va_bf16.safetensors",
        "tasks": ["t2va", "ref2va"],
        "upstream": "MiniMaxAI/MiniMax-H3 (REF2VA)",
        "title": "MiniMax-H3 REF2VA",
        "blurb": ("Text-to-audio-video conditioned on reference images, videos "
                  "and audio, for character / style / scene continuity."),
    },
    "omni": {
        "dit_file": "transformer_ref",
        "tasks": ["ref2va"],
        "upstream": "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
        "base_model": "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
        "title": "FastH3 Omni",
        "blurb": ("FastVideo's 8-step distillation of MiniMax-H3 REF2VA: video with "
                  "a matching stereo soundtrack from a prompt and ordered image, "
                  "video and audio references, in eight transformer forwards. It needs "
                  "at least one image or video reference and runs exactly 8 steps."),
        "tags": "- image-text-to-video",
        "extra": ("* The DiT is read from FastVideo's diffusers release: modules "
                  "renamed to ComfyUI's MiniMax-H3 names, Q/K/V fused and the SwiGLU "
                  "input halves reordered to (gate, value). All 32 PDD output heads "
                  "and the sparse-attention `to_gate_compress` projections are kept.\n"),
    },
}

CONFIG = {
    "model_type": "minimax_h3",
    "partition": "fl2va",
    "tasks": ["t2va", "fl2va"],
    "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
    "fps": 24,
    "quantization": {"group_size": GROUP_SIZE, "bits": BITS, "mode": "affine"},
    # Geometry the engine asserts against; recorded so a future release with
    # different dims fails loudly at load instead of half-loading.
    "transformer": {
        "hidden_size": 5376, "num_layers": 50, "num_attention_heads": 56,
        "attention_head_dim": 128, "ffn_hidden_size": 14336,
        "latents_dim": 24, "audio_latents_dim": 32, "text_dim": 5120,
        "time_embed_dim": 2688, "rope_inv_freq_len": 16,
    },
    "text_encoder": {"hidden": 5120, "layers": 50, "heads": 64, "kv_heads": 8,
                     "head_dim": 128, "intermediate": 25600, "theta": 5000000.0},
}

NOTICE = (
    "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
    "Copyright \u00a9 2026 MiniMax. All Rights Reserved.\n"
)

# Section III.2 requires modified files to carry PROMINENT notices saying so.
# Quantizing the weights modifies them, so this ships alongside them and the
# same statement goes into each safetensors file's own metadata.
MODIFICATIONS = """# Modifications to MiniMax H3

These files are MODIFIED versions of the MiniMax H3 Works ({upstream}),
redistributed under the MiniMax H3 Community License Agreement (see LICENSE and
NOTICE).

Modified by: mlx-serve (https://github.com/ddalcu/mlx-serve)

What changed:

* `transformer.safetensors` and `text_encoder.safetensors` are QUANTIZED from
  the original bfloat16 releases to MLX affine {bits}-bit, group size 64. Gathered
  embedding tables and the checkpoint's fp32 islands (patch projections, output
  heads, time embedder, rope inverse frequencies) are left dense.
* `video_vae.safetensors` and `audio_vae.safetensors` are byte-for-byte copies
  of the originals, unmodified.
* The tokenizer files are byte-for-byte copies from `MiniMaxAI/MiniMax-H3`
  (`FL2VA/processor/`), relocated into this directory so the model is
  self-contained.
{extra}* `config.json` is new, written by mlx-serve's converter to describe the
  layout above. It is not from the original release.

No weights were retrained, distilled, pruned or otherwise altered beyond the
numeric quantization described above.
"""

README = """---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: {base_model}
base_model_relation: quantized
library_name: mlx
tags:
- mlx-serve
{tags}
- audio-video-generation
---

# {title} — MLX-Serve {bits}-bit

{bits}-bit affine (group size 64) conversion of {upstream} for
[mlx-serve](https://github.com/ddalcu/mlx-serve), running natively on Apple
Silicon. The DiT denoises video and stereo audio jointly in one packed sequence.

{blurb}

Self-contained: weights, both VAEs and the tokenizer in one directory. Upstream
splits these across `Comfy-Org/MiniMax-H3` (weights, no tokenizer) and
`MiniMaxAI/MiniMax-H3` (tokenizer).

Quantized: the DiT and text-encoder matmul weights. Kept dense: gathered
embedding tables, the checkpoint's fp32 islands (patch projections, output
heads, time embedder) and both VAEs.

Note that quantization here buys FOOTPRINT, not speed — the workload is
compute-bound at roughly 192,000 FLOPs per weight byte.

## Modifications

These are MODIFIED files. The transformer and text encoder are quantized to
{bits}-bit; see MODIFICATIONS.md for the full list. The VAEs and tokenizer are
unmodified copies.

## License

Powered by MiniMax H3. Licensed under the MiniMax H3 Community License
Agreement -- see LICENSE and NOTICE, both included here.

**Territorial restriction.** The Agreement defines the Applicable Territory as
worldwide EXCLUDING the European Union, the United Kingdom, the Republic of
Korea and the United States of America, and Section V.4 prohibits use,
reproduction, modification, distribution and display outside it. Check whether
your jurisdiction permits you to use these files before downloading them.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="fl2va", choices=tuple(PARTITIONS),
                    help="which MiniMax checkpoint is being converted (default fl2va)")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8),
                    help="affine bits for every quantized linear (default 8)")
    ap.add_argument("--cpu", action="store_true",
                    help="quantize on the CPU stream (lets a conversion run while the GPU is busy)")
    ap.add_argument("--src", required=True, help="dir holding the Comfy-Org bf16 files (omni: the FastVideo snapshot)")
    ap.add_argument("--tokenizer", help="dir holding tokenizer.json (default: --reuse-from)")
    ap.add_argument("--reuse-from", help="an H3 pack to clone the text encoder, VAEs, tokenizer and LICENSE from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    global BITS
    BITS = args.bits
    CONFIG["quantization"]["bits"] = BITS
    part = PARTITIONS[args.partition]
    CONFIG["partition"] = args.partition
    CONFIG["tasks"] = part["tasks"]
    if args.cpu:
        mx.set_default_device(mx.cpu)

    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    if args.reuse_from:
        reuse = os.path.expanduser(args.reuse_from)
        for fn in ("text_encoder.safetensors", "video_vae.safetensors", "audio_vae.safetensors"):
            if not os.path.exists(os.path.join(out, fn)):
                print(f"[{fn}] cloning from {reuse}", flush=True)
                # APFS copy-on-write clone: instant, no space; a real copy elsewhere.
                if subprocess.run(["cp", "-c", os.path.join(reuse, fn), os.path.join(out, fn)]).returncode != 0:
                    shutil.copyfile(os.path.join(reuse, fn), os.path.join(out, fn))
    if not (args.tokenizer or args.reuse_from):
        raise SystemExit("--tokenizer or --reuse-from is required")
    if args.partition == "omni":
        contract = json.load(open(os.path.join(src, "fastvideo_inference.json")))
        if (contract.get("video_scheduler_shift"), contract.get("audio_scheduler_shift")) != (12.0, 3.0):
            raise SystemExit("contract shifts differ from H3's 12/3; the engine samples on those")
        for key in ("pdd_steps", "pdd_step_indices", "grid_max_t", "attention_backend",
                    "vsa_sparsity", "vsa_tile_size", "vsa_ref_policy", "vsa_ref_keep_rate"):
            CONFIG[key] = contract[key]

    jobs = [
        ("transformer", os.path.join(src, part["dit_file"]) if args.partition == "omni"
         else f"{src}/diffusion_models/{part['dit_file']}",
         f"{out}/transformer.safetensors"),
        ("text_encoder", f"{src}/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
         f"{out}/text_encoder.safetensors"),
    ]
    total = 0.0
    for label, s, d in jobs:
        # Destination first: the fetch driver DELETES each source once it has
        # been converted, so demanding a source for an output that already
        # exists makes every resumed run fail on the work it already finished.
        if args.skip_existing and os.path.exists(d):
            print(f"[{label}] exists, skipping", flush=True)
            total += os.path.getsize(d) / 1e9
            continue
        if not os.path.exists(s):
            raise SystemExit(f"missing source: {s}")
        print(f"[{label}] {s} -> {d}", flush=True)
        total += convert_file(s, d, label)

    # VAEs are copied verbatim: 5.8 GB combined, and the visual decoder is the
    # component whose output IS the image — not where to spend quality.
    for label, s, d in (
        ("video_vae", f"{src}/vae/minimax_h3_video_vae_fp16.safetensors", f"{out}/video_vae.safetensors"),
        ("audio_vae", f"{src}/vae/minimax_h3_audio_vae_fp32.safetensors", f"{out}/audio_vae.safetensors"),
    ):
        if os.path.exists(d):
            print(f"[{label}] exists, skipping", flush=True)
        else:
            print(f"[{label}] copying", flush=True)
            shutil.copyfile(s, d)
        total += os.path.getsize(d) / 1e9

    tok = os.path.expanduser(args.tokenizer or args.reuse_from)
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        p = os.path.join(tok, fn)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(out, fn))
    lic_src = os.path.join(tok, "LICENSE")
    if not os.path.exists(lic_src):
        lic_src = os.path.normpath(os.path.join(os.path.dirname(tok.rstrip("/")), "..", "LICENSE"))
    if not os.path.exists(lic_src):
        lic_src = os.path.join(os.path.dirname(os.path.dirname(tok.rstrip("/"))), "LICENSE")
    if not os.path.exists(lic_src):
        raise SystemExit(
            "LICENSE not found next to the tokenizer. Section III.1 of the MiniMax H3 "
            "Community License requires a copy of the Agreement to accompany any "
            "distribution, so refusing to write a directory that cannot legally be "
            "shared.\n  Fetch it with: hf download MiniMaxAI/MiniMax-H3 LICENSE "
            "--local-dir <staging>"
        )
    shutil.copyfile(lic_src, os.path.join(out, "LICENSE"))
    with open(os.path.join(out, "NOTICE"), "w") as f:
        f.write(NOTICE)
    def fill(t):
        return (t.replace("{bits}", str(BITS))
                 .replace("{extra}", part.get("extra", ""))
                 .replace("{base_model}", part.get("base_model", "MiniMaxAI/MiniMax-H3"))
                 .replace("{upstream}", part["upstream"])
                 .replace("{title}", part["title"])
                 .replace("{blurb}", part["blurb"])
                 .replace("{tags}", part.get("tags", "- text-to-video\n- image-text-to-video")))

    with open(os.path.join(out, "MODIFICATIONS.md"), "w") as f:
        f.write(fill(MODIFICATIONS))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(fill(README))

    print(f"\nDONE -> {out}  ({total:.1f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
