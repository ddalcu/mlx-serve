#!/usr/bin/env python3
"""Convert MiniMaxAI/MiniMax-Music3 into the layout mlx-serve's music engine loads.

USER-RUN (needs mlx + safetensors + the upstream repo). Not run in CI. No torch:
every component we consume ships as .safetensors, so `mx.load` reads the source
directly and the two .pth files are never touched.

Upstream ships the SAME weights twice. `modular_model_index.json` names seven
diffusers components; `qwen_7B/qwen_7B/` (18.5 GB) is the global LLM + depth
decoder again in the raw SGLang release format, and `flowmatching_vae.pth` /
`dav.pth` are the raw forms of `transformer/` + `vocoder/`. Only the diffusers
set is converted.

Produces:

    <out>/config.json                    {"model_type":"minimax_music3", ...}
    <out>/language_model.safetensors     Global LLM 8B (Qwen3, 36L, vocab 200k)
    <out>/rvq_depth_decoder.safetensors  Local LLM 0.6B (4L, 8 codebooks)
    <out>/transformer.safetensors        Flow-matching DiT (36 blocks, hidden 2048)
    <out>/condition_encoder.safetensors  8-layer hidden fuse + Conv1d 4096->2048
    <out>/vocoder.safetensors            Flow-VAE decoder (dense, precision-critical)
    <out>/tokenizer/                     text tokenizer (prompt + lyrics)
    <out>/music_tokenizer/               music-token vocab
    <out>/LICENSE                        MiniMax-Music3 Community License (+ Exhibit A AUP)

Quantization (affine, group 64, packed uint32 .weight + bf16 .scales/.biases):
eligible 2-D linears only. Deliberately NOT quantized:

  * `model.embed_tokens.weight` and the depth decoder's codebook tables — those
    are GATHER reads, not matmuls (root CLAUDE.md: "Quantize only what a MATMUL
    reads"). `lm_head` is a real matmul and IS quantized.
  * the vocoder — it is the Flow-VAE decoder, and VAE precision is load-bearing
    across our media engines.
  * the condition encoder — 100 MB of fp32 whose Conv1d output conditions EVERY
    DiT step; not worth the risk for 50 MB.

DELIBERATELY NOT DONE HERE — structural transforms the engine still owes:

  * The vocoder is a weight-norm'd Snake conv stack (the same shape as
    ACE-Step's Oobleck VAE): `conv*.weight_g` / `.weight_v` pairs need fusing
    (w = g * v / ||v||), Conv1d needs PT [out, in, K] -> MLX [out, K, in],
    ConvTranspose1d needs [in, out, K] -> [out, K, in], and `snake*.alpha`
    is [1, C, 1] and must stay fp32 (exp headroom). Which of the two conv
    kinds a tensor is can be read off the bias length, but none of it is
    checkable without a parity fixture, so this pack ships the vocoder
    VERBATIM in the upstream layout and the transform lands with the engine
    against a cos/rms_ratio oracle (cf. tests/dump_acestep_fixtures.py).
    Same for `preprocess_conv`/`postprocess_conv` in the DiT and the
    condition encoder's Conv1d `proj.weight`.

Usage:
    python3 scripts/convert_music3_weights.py <src> <out> [--bits {4,8,16}]
    python3 scripts/convert_music3_weights.py --self-test   # no ckpt/mlx needed
"""

import argparse
import json
import os
import shutil
import re
import struct
import sys
import time

GROUP_SIZE = 64

# HF model card. `base_model_relation: quantized` is load-bearing: HF defaults
# a missing relation to `finetune` (tests/test_model_card_frontmatter.sh class).
README = '---\nlicense: other\nlicense_name: minimax-music3-community-license\nlicense_link: LICENSE\nbase_model: MiniMaxAI/MiniMax-Music3\nbase_model_relation: quantized\nlibrary_name: mlx-serve\ntags:\n  - mlx\n  - mlx-serve\n  - music\n  - text-to-music\n  - minimax\npipeline_tag: text-to-audio\n---\n\n# MiniMax Music 3 for mlx-serve (8-bit)\n\n[MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) converted for\n[mlx-serve](https://github.com/ddalcu/mlx-serve)\'s native Zig + MLX engine.\nFull songs with sung lyrics at 44.1 kHz stereo, generated locally on Apple\nSilicon.\n\nQuantization: affine 8-bit, group 64, on every real matmul (LLM, depth\ndecoder, DiT, lm_head). Kept dense on purpose: the embedding tables (gather\nreads), the condition encoder and the whole vocoder (VAE-class precision).\nWorst per-tensor reconstruction error 1.48% RMS. 13 GB on disk instead of the\n57 GB upstream repo (which ships its weights twice).\n\n| File | Contents |\n|---|---|\n| `language_model.safetensors` | Qwen3 8B global LLM, 36L, vocab 200k |\n| `rvq_depth_decoder.safetensors` | 0.6B local LLM, 7 residual codebooks |\n| `transformer.safetensors` | 2.4B flow-matching DiT, 36 blocks |\n| `condition_encoder.safetensors` | hidden-state mix + resampler, f32 |\n| `vocoder.safetensors` | Flow-VAE / DAC decoder, f32 |\n\nEngine parity against the fp32 reference on these exact weights: prefill\ncosine 0.9999, condition encoder 0.999999, DiT velocity 0.999, vocoder\n1.000000. The autoregressive stage runs about 44 ms per frame on an M-series\nMac, so a one-minute song takes roughly a minute of LLM time plus the\ndiffusion pass.\n\n## Run it\n\nDownload **[MLX Core.app](https://github.com/ddalcu/mlx-serve/releases/latest)**,\nopen the Music tab and pick **MiniMax Music 3**. Style prompt + lyrics in,\nWAV out.\n\nOver HTTP:\n\n```bash\nmlx-serve --serve\ncurl http://127.0.0.1:11234/v1/audio/music-generations \\\n  -H \'Content-Type: application/json\' -o song.wav -d \'{\n    "model": "MiniMax-Music3-MLX-Serve-8bit",\n    "prompt": "upbeat synthwave with driving bass and dreamy pads",\n    "lyrics": "[verse]\\nneon lights across the bay\\n[chorus]\\nwe run all night",\n    "duration_seconds": 60\n  }\'\n```\n\nLyrics are required (the model is lyric-conditioned) and structure tags like\n`[verse]` or `[chorus]` go on their own lines. `duration_seconds` (1-360) is\nan upper bound, the model may end the song earlier. ACE-Step style fields\n(bpm, keyscale, timesignature, vocal_language) do not exist on this model.\n\nRebuild from the upstream repo with\n[`scripts/convert_music3_weights.py`](https://github.com/ddalcu/mlx-serve/blob/main/scripts/convert_music3_weights.py).\n\nWeights are covered by the MiniMax-Music3 Community License (see `LICENSE`,\nacceptable-use policy included as Exhibit A).\n'


# Gather-read tables: dequantizing a row on every lookup buys nothing and costs
# fidelity on the one table the whole autoregressive path indexes.
NEVER_QUANTIZE = (
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "audio_embeddings",
    "codebook",
)


def should_quantize(name, shape, bits):
    """A .weight is quantized iff 2-D, in-features % GROUP_SIZE == 0, and
    min(out, in) >= 512 — and it is not a gather-read table.

    Excludes by construction: norms and biases (1-D), the pre/post-process convs
    and the condition-encoder proj (3-D), and small projections such as
    time_embed.linear_1 (in=256)."""
    if bits not in (4, 8):
        return False
    if not name.endswith(".weight"):
        return False
    if any(t in name for t in NEVER_QUANTIZE):
        return False
    if len(shape) != 2:
        return False
    out_f, in_f = shape
    if in_f % GROUP_SIZE != 0:
        return False
    return min(out_f, in_f) >= 512


# In a 4-bit build the timestep-embedding family stays 8-bit: its output
# modulates every DiT block at every flow step, so error there compounds over
# the whole denoise. Costs ~10 MB. The Zig loader infers (bits, group) per
# tensor from packed geometry, so a mixed-precision pack loads unchanged.
SENSITIVE_8BIT = ("time_embed.",)


def quant_bits_for(name, shape, bits):
    """Effective quantization for one tensor: None = dense bf16, else 4 or 8."""
    if not should_quantize(name, shape, bits):
        return None
    if bits == 4 and any(t in name for t in SENSITIVE_8BIT):
        return 8
    return bits


def quant_label(bits):
    return {4: "4bit", 8: "8bit"}.get(bits, "bf16")


# (out file, src subdir, shard glob, quantize?) — the five diffusers components
# the engine needs. `scheduler` is config-only and folds into config.json.
COMPONENTS = (
    ("language_model.safetensors",    "language_model",    "model-*.safetensors", True),
    ("rvq_depth_decoder.safetensors", "rvq_depth_decoder", "diffusion_pytorch_model*.safetensors", True),
    ("transformer.safetensors",       "transformer",       "diffusion_pytorch_model*.safetensors", True),
    # Dense by policy: the Flow-VAE decoder (precision load-bearing) and the
    # condition encoder (100 MB feeding every DiT step).
    ("condition_encoder.safetensors", "condition_encoder", "diffusion_pytorch_model*.safetensors", False),
    ("vocoder.safetensors",           "vocoder",           "diffusion_pytorch_model*.safetensors", False),
)

TOKENIZER_SRC = "tokenizer"
MUSIC_TOKENIZER_SRC = os.path.join("qwen_7B", "qwen3-8B-tokenizer-music")


def header(path):
    """Tensor inventory without reading the payload."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def shards(src, subdir, pattern):
    import glob as _glob
    found = sorted(_glob.glob(os.path.join(src, subdir, pattern)))
    if not found:
        raise SystemExit(f"no weights matching {subdir}/{pattern} under {src}")
    return found


def convert_component(src, out, spec, bits):
    """Quantize (or copy through) one component into a single safetensors file.

    Shards are loaded one at a time and released before the next, so peak
    residency is one source shard plus the accumulating (smaller) output.
    """
    import mlx.core as mx

    out_name, subdir, pattern, quantize = spec
    dst = os.path.join(out, out_name)
    if os.path.exists(dst):
        print(f"[{subdir}] exists, skipping", flush=True)
        return None

    packed, n_quant, n_dense = {}, 0, 0
    for shard in shards(src, subdir, pattern):
        loaded = mx.load(shard)
        for name, arr in loaded.items():
            eff = quant_bits_for(name, arr.shape, bits) if quantize else None
            if eff is None:
                packed[name] = arr           # source dtype preserved
                n_dense += 1
                continue
            # Quantize at bf16, NOT fp32-then-round-the-scales. mx.quantize
            # returns scales/biases in the input dtype, and the runtime needs
            # them in the activation dtype (bf16) -- so quantizing in fp32 and
            # casting afterwards encodes each weight against a scale the
            # decoder no longer has. Measured on this checkpoint's DiT, that
            # mismatch costs ~10% of the reconstruction error for nothing
            # (1.17% -> 1.04% RMS relative on transformer_blocks.35.ff_out).
            wq, scales, biases = mx.quantize(
                arr.astype(mx.bfloat16), group_size=GROUP_SIZE, bits=eff
            )
            base = name[: -len(".weight")]
            packed[f"{base}.weight"] = wq
            packed[f"{base}.scales"] = scales
            packed[f"{base}.biases"] = biases
            n_quant += 1
        mx.eval(*packed.values())
        del loaded

    mx.save_safetensors(dst, packed)
    nbytes = os.path.getsize(dst)
    print(
        f"[{subdir}] -> {out_name}: {n_quant} quantized + {n_dense} dense, "
        f"{nbytes / 1e9:.2f} GB",
        flush=True,
    )
    return nbytes


def write_config(src, out, bits):
    """Flatten the upstream per-component configs into one config.json.

    Read from the source rather than transcribed, so a checkpoint revision that
    changes a dimension cannot silently disagree with the pack.
    """
    def cfg(name):
        with open(os.path.join(src, name, "config.json")) as f:
            return json.load(f)

    lm = cfg("language_model")
    dd = cfg("rvq_depth_decoder")
    tr = cfg("transformer")
    ce = cfg("condition_encoder")
    vo = cfg("vocoder")
    with open(os.path.join(src, "scheduler", "scheduler_config.json")) as f:
        sch = json.load(f)

    rope = lm.get("rope_parameters") or {}
    out_cfg = {
        "model_type": "minimax_music3",
        "quant": quant_label(bits),
        "group_size": GROUP_SIZE,
        # Pipeline-level facts. The LLM emits RVQ frames at 25 Hz; the Flow-VAE
        # latent runs at output_sampling_rate / output_hop_length = 86.13 Hz, so
        # the condition encoder is also the resampler between the two rates.
        "frame_rate": ce["input_sampling_rate"] / ce["input_hop_length"],
        "latent_rate": ce["output_sampling_rate"] / ce["output_hop_length"],
        "sample_rate": vo["sampling_rate"],
        "max_frames": 9000,
        "max_text_tokens": 5000,
        "num_codebooks": dd["num_codebooks"],
        "semantic_vocab_size": 16384,
        "audio_vocab_size": dd["audio_vocab_size"],
        "language_model": {
            "hidden_size": lm["hidden_size"],
            "num_hidden_layers": lm["num_hidden_layers"],
            "num_attention_heads": lm["num_attention_heads"],
            "num_key_value_heads": lm["num_key_value_heads"],
            "head_dim": lm["head_dim"],
            "intermediate_size": lm["intermediate_size"],
            "vocab_size": lm["vocab_size"],
            "rms_norm_eps": lm["rms_norm_eps"],
            "rope_theta": rope.get("rope_theta", lm.get("rope_theta")),
            "max_position_embeddings": lm["max_position_embeddings"],
        },
        "rvq_depth_decoder": {
            "hidden_size": dd["hidden_size"],
            "intermediate_size": dd["intermediate_size"],
            "num_attention_heads": dd["num_attention_heads"],
            "num_layers": dd["num_layers"],
            "max_position_embeddings": dd["max_position_embeddings"],
        },
        "transformer": {
            "num_layers": tr["num_layers"],
            "num_attention_heads": tr["num_attention_heads"],
            "attention_head_dim": tr["attention_head_dim"],
            "hidden_size": tr["num_attention_heads"] * tr["attention_head_dim"],
            "ff_inner_dim": tr["ff_inner_dim"],
            "in_channels": tr["in_channels"],
            "condition_dim": tr["condition_dim"],
            "fourier_embedding_dim": tr["fourier_embedding_dim"],
            "rotary_dim": tr["rotary_dim"],
        },
        "condition_encoder": {
            "condition_hidden_dim": ce["condition_hidden_dim"],
            "out_dim": ce["out_dim"],
            "num_condition_layers": ce["num_condition_layers"],
            "input_hop_length": ce["input_hop_length"],
            "input_sampling_rate": ce["input_sampling_rate"],
            "output_hop_length": ce["output_hop_length"],
            "output_sampling_rate": ce["output_sampling_rate"],
        },
        "vocoder": {
            "latent_channels": vo["latent_channels"],
            "decoder_input_dim": vo["decoder_input_dim"],
            "decoder_hidden_dim": vo["decoder_hidden_dim"],
            "upsampling_ratios": vo["upsampling_ratios"],
        },
        "scheduler": {
            "type": "flow_match_euler_discrete",
            "num_train_timesteps": sch["num_train_timesteps"],
            "shift": sch["shift"],
            "invert_sigmas": sch["invert_sigmas"],
        },
    }
    path = os.path.join(out, "config.json")
    with open(path, "w") as f:
        json.dump(out_cfg, f, indent=2)
        f.write("\n")
    print(f"[config] {path}", flush=True)


def copy_tree(src, out, rel, dst_name, files=None):
    s = os.path.join(src, rel)
    if not os.path.isdir(s):
        print(f"[copy] skip missing {rel}", flush=True)
        return
    d = os.path.join(out, dst_name)
    os.makedirs(d, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(s)):
        if files is not None and name not in files:
            continue
        sp = os.path.join(s, name)
        if os.path.isfile(sp):
            shutil.copyfile(sp, os.path.join(d, name))
            n += 1
    print(f"[copy] {dst_name}: {n} files", flush=True)


def inspect(src):
    for _, subdir, pattern, _ in COMPONENTS:
        try:
            found = shards(src, subdir, pattern)
        except SystemExit as e:
            print(e)
            continue
        merged = {}
        for p in found:
            merged.update(header(p))
        grouped = {}
        for k, v in merged.items():
            gk = re.sub(r"\.\d+\.", ".*.", k)
            sh, dt, c = grouped.get(gk, (v["shape"], v["dtype"], 0))
            grouped[gk] = (sh, dt, c + 1)
        print(f"### {subdir}  ({len(merged)} tensors, {len(found)} shard(s))")
        for k, (sh, dt, c) in grouped.items():
            print(f"  x{c:<4d} {str(sh):24s} {dt:8s} {k}")
        print()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="upstream MiniMax-Music3 dir")
    ap.add_argument("out", nargs="?", help="destination pack dir")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8, 16))
    ap.add_argument("--inspect", action="store_true",
                    help="print the tensor inventory and exit")
    ap.add_argument("--gpu", action="store_true",
                    help="quantize on the GPU (default CPU, so this runs beside "
                         "a live server or the app)")
    args = ap.parse_args()

    if args.inspect:
        return inspect(args.src)

    import mlx.core as mx
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    if not args.out:
        ap.error("out is required unless --inspect")

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    total = 0
    for spec in COMPONENTS:
        n = convert_component(args.src, args.out, spec, args.bits)
        total += n or 0

    write_config(args.src, args.out, args.bits)
    copy_tree(args.src, args.out, TOKENIZER_SRC, "tokenizer")
    copy_tree(args.src, args.out, MUSIC_TOKENIZER_SRC, "music_tokenizer")

    # The MiniMax-Music3 Community License incorporates its Acceptable Use
    # Policy as Exhibit A in the same file, so copying LICENSE carries both.
    lic = os.path.join(args.src, "LICENSE")
    if os.path.exists(lic):
        shutil.copyfile(lic, os.path.join(args.out, "LICENSE"))
        print("[copy] LICENSE", flush=True)

    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(README)
    print("[card] README.md", flush=True)

    print(f"\ndone in {time.time() - t0:.0f}s -> {args.out}", flush=True)
    return 0


def self_test():
    Q, N = should_quantize, lambda *a: not should_quantize(*a)

    # Global LLM: every projection is a real matmul.
    assert Q("model.layers.0.self_attn.q_proj.weight", (4096, 4096), 8)
    assert Q("model.layers.0.self_attn.k_proj.weight", (1024, 4096), 8)
    assert Q("model.layers.0.mlp.gate_proj.weight", (12288, 4096), 8)
    assert Q("model.layers.0.mlp.down_proj.weight", (4096, 12288), 8)
    assert Q("lm_head.weight", (200000, 4096), 8), "lm_head is a matmul"

    # ...but the embedding table is a GATHER, and stays dense.
    assert N("model.embed_tokens.weight", (200000, 4096), 8), \
        "embed_tokens is gather-read, must not quantize"

    # Norms / biases are 1-D.
    assert N("model.layers.0.self_attn.q_norm.weight", (128,), 8)
    assert N("model.norm.weight", (4096,), 8)
    assert N("transformer_blocks.0.ff_in.bias", (8192,), 8)

    # Flow DiT block linears. ff_in is a FUSED gate+up: 16384 = 2 x ff_inner_dim.
    assert Q("transformer_blocks.0.attn.to_q.weight", (2048, 2048), 8)
    assert Q("transformer_blocks.0.attn.to_out.0.weight", (2048, 2048), 8)
    assert Q("transformer_blocks.0.ff_in.weight", (16384, 2048), 8)
    assert Q("transformer_blocks.0.ff_out.weight", (2048, 8192), 8)

    # The DiT's in/out projections carry the 128-channel latent, and the
    # min(out, in) bar keeps both dense — proj_out is the tensor the whole
    # denoise resolves to.
    assert N("proj_in.weight", (2048, 128), 8)
    assert N("proj_out.weight", (128, 2048), 8)

    # Depth decoder: the 7 acoustic heads are matmuls, its two tables are not.
    assert Q("audio_heads.0.weight", (1024, 4096), 8)
    assert Q("projection.weight", (4096, 4096), 8)
    assert Q("layers.0.attn.to_out.weight", (4096, 4096), 8)
    assert N("audio_embeddings.weight", (7168, 4096), 8), \
        "7 x 1024 acoustic codebooks, gather-read"
    assert N("pos_embedding.weight", (16, 4096), 8), "16-position table"

    # Convs are 3-D and never eligible, whatever their size.
    assert N("preprocess_conv.weight", (2048, 128, 3), 8)
    assert N("proj.weight", (2048, 4096, 3), 8), "condition-encoder Conv1d"

    # Small projections: the fourier head is in=256, under the min(out,in) bar.
    assert N("time_embed.linear_1.weight", (2048, 256), 8)

    # A contraction dim that does not divide the group size cannot be packed.
    assert N("odd.weight", (2048, 2050), 8)

    # bits=16 means dense everywhere.
    assert N("model.layers.0.mlp.gate_proj.weight", (12288, 4096), 16)

    # 4-bit build: the timestep family holds 8 bits, everything else drops.
    assert quant_bits_for("time_embed.linear_2.weight", (2048, 2048), 4) == 8
    assert quant_bits_for("transformer_blocks.0.ff_in.weight", (8192, 2048), 4) == 4
    assert quant_bits_for("transformer_blocks.0.ff_in.weight", (8192, 2048), 8) == 8
    assert quant_bits_for("model.embed_tokens.weight", (200000, 4096), 4) is None

    print("self-test OK")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    raise SystemExit(main())
