#!/usr/bin/env python3
"""Convert HeartMuLa-oss-3B + HeartCodec-oss into the pack mlx-serve's `heartmula` plugin loads.

USER-RUN (needs mlx + safetensors + the upstream checkpoints). Not run in CI.

Source: heartlib's `ckpt/` layout, i.e. `HeartMuLa/HeartMuLaGen` (tokenizer.json, gen_config.json) with
`HeartMuLa/HeartMuLa-oss-3B-happy-new-year` in `HeartMuLa-oss-3B/` and `HeartMuLa/HeartCodec-oss-20260123` in
`HeartCodec-oss/`. `--lm` / `--codec` point elsewhere (e.g. the RL checkpoint, same arch).

Produces:

    <out>/config.json         {"model_type":"heartmula", ...} geometry of the LM, the codec and the gen ids
    <out>/model.safetensors   the LM (backbone + depth decoder + embeddings + heads)
    <out>/codec.safetensors   the HeartCodec decoder path (RVQ, flow DiT, scalar decoder); the completion marker
    <out>/tokenizer.json, <out>/README.md, <out>/LICENSE

LM names follow mlx-audio's Sesame CSM port (mlx-lm Llama names under `backbone.` / `decoder.`); q/k rows stay in
torchtune's INTERLEAVED RoPE order (the engine runs `rope(traditional=true)`). `audio_head` [7, D, V] is split into
seven Linear-layout `audio_head.N.weight` [V, D] so it can be quantized.

Codec: weight norm folded (w = g * v / ||v||), Conv1d PT [out, in, K] -> MLX [out, K, in], ConvTranspose1d PT
[in, out, K] -> [out, K, in], RVQ codebooks squeezed to [8192, 32]. DiT matrices bf16, everything else f32. The
encoder, the RVQ EMA state, `project_in` and `zero_cond_embedding1` (never reached: a decode window is always full)
are dropped by name.

Presets (`--bits`): 16 = bf16 everywhere (the quality reference); 8 = every LM matmul + both gather tables affine
8-bit g64; 4 = the backbone trunk 4-bit, depth decoder + heads + tables 8-bit.

Usage:
    python3 scripts/convert_heartmula_weights.py <ckpt> <out> --bits {16,8,4}
    python3 scripts/convert_heartmula_weights.py --self-test
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys

GROUP_SIZE = 64
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"

# torchtune flavor -> geometry (heartmula/modeling_heartmula.py FLAVORS).
FLAVORS = {
    "llama-3B": dict(layers=28, heads=24, kv_heads=8, hidden=3072, ffn=8192, max_seq_len=8192),
    "llama-300M": dict(layers=3, heads=8, kv_heads=4, hidden=3072, ffn=8192, max_seq_len=2048),
    "llama-7B": dict(layers=32, heads=32, kv_heads=8, hidden=4096, ffn=14336, max_seq_len=8192),
    "llama-400M": dict(layers=4, heads=8, kv_heads=4, hidden=3072, ffn=8192, max_seq_len=2048),
}
ROPE = dict(rope_theta=500000.0, rms_norm_eps=1e-5,
            rope_scaling=dict(rope_type="llama3", factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0,
                              original_max_position_embeddings=8192))

LM_RENAMES = (
    (".attn.q_proj.", ".self_attn.q_proj."),
    (".attn.k_proj.", ".self_attn.k_proj."),
    (".attn.v_proj.", ".self_attn.v_proj."),
    (".attn.output_proj.", ".self_attn.o_proj."),
    (".mlp.w1.", ".mlp.gate_proj."),
    (".mlp.w2.", ".mlp.down_proj."),
    (".mlp.w3.", ".mlp.up_proj."),
    (".sa_norm.scale", ".input_layernorm.weight"),
    (".mlp_norm.scale", ".post_attention_layernorm.weight"),
    (".norm.scale", ".norm.weight"),
)

LM_DROP = ("muq_linear.weight",)  # the reference-audio slot is refused; text-only reads the bias alone


def lm_name(k):
    for a, b in LM_RENAMES:
        k = k.replace(a, b)
    return k


def lm_bits(name, preset):
    """Affine width for one LM tensor (None = bf16)."""
    if preset == 16:
        return None
    if name in ("unconditional_text_embedding.weight", "muq_linear.bias") or not name.endswith(".weight"):
        return None
    if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
        return None
    if preset == 4 and name.startswith("backbone.layers."):
        return 4
    return 8


CODEC_DROP_RE = re.compile(
    r"^encoder\.|\._codebook\.(cluster_size|embed_avg|initted)$|^flow_matching\.vq_embed\.project_in\."
    r"|^flow_matching\.zero_cond_embedding1$|^scalar_model\.encoder\.")


def is_conv_transpose(name):
    return bool(re.match(r"^scalar_model\.decoder\.\d+\.up_conv\.layer\.", name))


def fold_weight_norm(g, v):
    """torch.nn.utils.parametrizations.weight_norm, dim=0: g * v / ||v|| over every axis but 0."""
    import mlx.core as mx
    norm = mx.sqrt(mx.sum(v * v, axis=tuple(range(1, v.ndim)), keepdims=True))
    return g * v / norm


def codec_tensors(raw):
    """Upstream codec state dict (f32 mx arrays) -> {name: array} in the pack's layout."""
    import mlx.core as mx
    out = {}
    for k in sorted(raw):
        if CODEC_DROP_RE.search(k):
            continue
        if k.endswith("parametrizations.weight.original1"):
            continue
        v = raw[k]
        if k.endswith("parametrizations.weight.original0"):
            base = k[: -len("parametrizations.weight.original0")]
            v = fold_weight_norm(v, raw[base + "parametrizations.weight.original1"])
            k = base + "weight"
        if k.endswith("._codebook.embed"):
            k = k.replace("._codebook.embed", ".codebook")
            v = v.reshape(v.shape[-2], v.shape[-1])
        if v.ndim == 3 and k.endswith(".weight"):
            v = v.transpose(1, 2, 0) if is_conv_transpose(k) else v.transpose(0, 2, 1)
        dit_matrix = k.startswith("flow_matching.estimator.") and v.ndim >= 2 and "scale_shift_table" not in k
        out[k] = v.astype(mx.bfloat16 if dit_matrix else mx.float32)
    return out


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def shard_files(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    return ["model.safetensors"]


def convert_lm(lm_dir, out, preset):
    import mlx.core as mx
    packed, n_q = {}, 0
    for shard in shard_files(lm_dir):
        for k, v in mx.load(os.path.join(lm_dir, shard)).items():
            if k in LM_DROP:
                continue
            items = [(f"audio_head.{i}.weight", v[i].T) for i in range(v.shape[0])] if k == "audio_head" else [(lm_name(k), v)]
            for name, arr in items:
                arr = arr.astype(mx.bfloat16)
                bits = lm_bits(name, preset)
                if bits is None:
                    packed[name] = arr
                    continue
                # quantized AT bf16 so scales/biases are the ones the bf16 runtime dequantizes with
                wq, sc, bi = mx.quantize(arr, group_size=GROUP_SIZE, bits=bits)
                base = name[: -len(".weight")]
                packed[base + ".weight"], packed[base + ".scales"], packed[base + ".biases"] = wq, sc, bi
                n_q += 1
            mx.eval(*packed.values())
    mx.save_safetensors(os.path.join(out, "model.safetensors"), packed)
    print(f"[lm] {len(packed)} tensors, {n_q} quantized", flush=True)


def convert_codec(codec_dir, out):
    import mlx.core as mx
    raw = {}
    for shard in shard_files(codec_dir):
        raw.update({k: v for k, v in mx.load(os.path.join(codec_dir, shard)).items() if not k.startswith("encoder.")})
    packed = codec_tensors(raw)
    mx.eval(*packed.values())
    tmp = os.path.join(out, "codec.safetensors.partial.safetensors")
    mx.save_safetensors(tmp, packed)
    os.replace(tmp, os.path.join(out, "codec.safetensors"))  # the completion marker lands last
    print(f"[codec] {len(packed)} tensors", flush=True)


def build_config(lm_cfg, codec_cfg, gen_cfg, preset):
    def stack(flavor):
        f = FLAVORS[flavor]
        return dict(num_hidden_layers=f["layers"], hidden_size=f["hidden"], num_attention_heads=f["heads"],
                    num_key_value_heads=f["kv_heads"], head_dim=f["hidden"] // f["heads"],
                    intermediate_size=f["ffn"], max_position_embeddings=f["max_seq_len"], **ROPE)

    keep = ("dim", "codebook_size", "codebook_dim", "num_quantizers", "attention_head_dim", "in_channels",
            "num_attention_heads", "num_layers", "num_layers_2", "out_channels", "sample_rate", "num_samples",
            "upsample_factors", "upsample_kernel_sizes", "latent_hidden_dim", "delay_kernel_size", "init_channel",
            "res_kernel_size", "default_kernel_size")
    return {
        "model_type": "heartmula",
        "quant": {16: "bf16", 8: "8bit", 4: "4bit"}[preset],
        "group_size": GROUP_SIZE,
        "audio_num_codebooks": lm_cfg["audio_num_codebooks"],
        "audio_vocab_size": lm_cfg["audio_vocab_size"],
        "text_vocab_size": lm_cfg["text_vocab_size"],
        "backbone": stack(lm_cfg["backbone_flavor"]),
        "decoder": stack(lm_cfg["decoder_flavor"]),
        "codec": {k: codec_cfg[k] for k in keep},
        **{k: gen_cfg[k] for k in ("text_bos_id", "text_eos_id", "audio_eos_id", "empty_id")},
    }


README = """---
license: apache-2.0
base_model:
  - HeartMuLa/HeartMuLa-oss-3B-happy-new-year
  - HeartMuLa/HeartCodec-oss-20260123
base_model_relation: quantized
library_name: mlx-serve
tags:
  - mlx
  - mlx-serve
  - music
  - text-to-music
  - heartmula
pipeline_tag: text-to-audio
---

# HeartMuLa-oss-3B for mlx-serve ({quant})

[HeartMuLa-oss-3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year) with
[HeartCodec](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123), converted for
[mlx-serve](https://github.com/ddalcu/mlx-serve)'s native Zig + MLX engine: lyrics + style tags in, a 48 kHz stereo
song out.

{quant_line}

Run it in [MLX-Serve.app](https://github.com/ddalcu/mlx-serve/releases/latest) (Music tab) or over HTTP. Tags are
lowercase and comma-separated without spaces; structure tags like `[Verse]` go on their own lines in the lyrics.

```bash
curl http://127.0.0.1:11234/v1/audio/music-generations -H 'Content-Type: application/json' -o song.wav -d '{{
  "model": "{name}", "prompt": "pop,piano,female vocal,upbeat",
  "lyrics": "[Verse]\\nneon lights across the bay\\n[Chorus]\\nwe run all night", "duration_seconds": 60}}'
```

Rebuild with `scripts/convert_heartmula_weights.py`. Apache-2.0, as upstream.
"""

QUANT_LINES = {
    16: "Everything bf16 (the quality reference); the codec's scalar decoder and tables stay f32.",
    8: "LM: affine 8-bit, group 64, on every matmul and both embedding tables. Codec: DiT bf16, the rest f32.",
    4: "LM: the 28-layer backbone affine 4-bit (group 64); the depth decoder, heads and embedding tables 8-bit. "
       "Codec: DiT bf16, the rest f32.",
}


def convert(src, out, preset, lm_dir=None, codec_dir=None):
    lm_dir = lm_dir or os.path.join(src, "HeartMuLa-oss-3B")
    codec_dir = codec_dir or os.path.join(src, "HeartCodec-oss")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(lm_dir, "config.json")) as f:
        lm_cfg = json.load(f)
    with open(os.path.join(codec_dir, "config.json")) as f:
        codec_cfg = json.load(f)
    with open(os.path.join(src, "gen_config.json")) as f:
        gen_cfg = json.load(f)
    if lm_cfg.get("model_type") != "heartmula" or codec_cfg.get("model_type") != "heartcodec":
        raise SystemExit("not a HeartMuLa + HeartCodec pair")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(build_config(lm_cfg, codec_cfg, gen_cfg, preset), f, indent=2)
    shutil.copy(os.path.join(src, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    name = os.path.basename(os.path.normpath(out))
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(README.format(quant=name.rsplit("-", 1)[-1],
                              quant_line=QUANT_LINES[preset], name=name))
    lic = os.path.join(os.path.dirname(os.path.normpath(src)), "LICENSE")
    if os.path.exists(lic):
        shutil.copy(lic, os.path.join(out, "LICENSE"))
    convert_lm(lm_dir, out, preset)
    convert_codec(codec_dir, out)


def self_test():
    import mlx.core as mx
    assert lm_name("backbone.layers.3.attn.output_proj.weight") == "backbone.layers.3.self_attn.o_proj.weight"
    assert lm_name("decoder.layers.0.mlp.w2.weight") == "decoder.layers.0.mlp.down_proj.weight"
    assert lm_name("backbone.layers.0.sa_norm.scale") == "backbone.layers.0.input_layernorm.weight"
    assert lm_name("decoder.norm.scale") == "decoder.norm.weight"
    assert lm_bits("backbone.layers.0.mlp.up_proj.weight", 4) == 4
    assert lm_bits("decoder.layers.0.mlp.up_proj.weight", 4) == 8
    assert lm_bits("audio_head.6.weight", 4) == 8 and lm_bits("text_embeddings.weight", 8) == 8
    assert lm_bits("backbone.norm.weight", 8) is None and lm_bits("projection.weight", 16) is None
    # weight norm folds to unit rows times g; conv axes land in MLX order
    g = mx.array([[[2.0]], [[3.0]]])
    v = mx.random.normal((2, 4, 5))
    raw = {
        "scalar_model.decoder.0.parametrizations.weight.original0": g,
        "scalar_model.decoder.0.parametrizations.weight.original1": v,
        "scalar_model.decoder.1.up_conv.layer.parametrizations.weight.original0": mx.ones((4, 1, 1)),
        "scalar_model.decoder.1.up_conv.layer.parametrizations.weight.original1": mx.random.normal((4, 2, 6)),
        "flow_matching.vq_embed.layers.0._codebook.embed": mx.zeros((1, 16, 8)),
        "flow_matching.vq_embed.layers.0._codebook.embed_avg": mx.zeros((1, 16, 8)),
        "flow_matching.estimator.transformer_blocks.0.attn.q_proj.weight": mx.zeros((8, 8)),
        "flow_matching.estimator.scale_shift_table": mx.zeros((2, 8)),
        "flow_matching.zero_cond_embedding1": mx.zeros((8,)),
    }
    out = codec_tensors(raw)
    w = out["scalar_model.decoder.0.weight"]
    assert w.shape == (2, 5, 4), w.shape
    norms = mx.sqrt(mx.sum(w * w, axis=(1, 2)))
    assert mx.allclose(norms, mx.array([2.0, 3.0]), atol=1e-5).item()
    assert out["scalar_model.decoder.1.up_conv.layer.weight"].shape == (2, 6, 4)
    assert out["flow_matching.vq_embed.layers.0.codebook"].shape == (16, 8)
    assert out["flow_matching.estimator.transformer_blocks.0.attn.q_proj.weight"].dtype == mx.bfloat16
    assert out["flow_matching.estimator.scale_shift_table"].dtype == mx.float32
    assert not any("embed_avg" in k or "zero_cond" in k for k in out)
    print("self-test ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?")
    p.add_argument("out", nargs="?")
    p.add_argument("--bits", type=int, choices=(16, 8, 4), default=8)
    p.add_argument("--lm")
    p.add_argument("--codec")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not (a.src and a.out):
        p.error("src and out are required")
    convert(a.src, a.out, a.bits, a.lm, a.codec)


if __name__ == "__main__":
    sys.exit(main())
