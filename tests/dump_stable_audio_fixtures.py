#!/usr/bin/env python3
"""Dump Stable Audio 3 (small) parity fixtures (.raw f32 / i32) for the Zig
oracle tests in `src/stable_audio.zig`.

USER-RUN (needs mlx + numpy + sentencepiece). Runs Stability's own MLX
reference (github.com/Stability-AI/stable-audio-3, `optimized/mlx/models/defs`)
in fp32 over the OFFICIAL checkpoint layout (the `stabilityai/stable-audio-3-small-*`
repo: PyTorch-named model.safetensors + t5gemma-b-b-ul2/), so the oracle reads
the same file the engine loads.

Oracle taps (env prefix SA3_*):
  TOK    : PROMPT -> SentencePiece ids (no BOS/EOS, the HF tokenizer_config)
  T5     : those ids -> T5Gemma last_hidden_state [n,768]
  CROSS  : padded prompt rows + seconds row [257,768]; GLOBAL: seconds row [768]
  DIT    : noise X [1,256,T] at t=0.7 -> velocity [1,256,T]
  LAT    : the full seeded 8-step pingpong sample (SEED) -> latents [1,256,T]
  PATCH  : SAME-S decode of LAT (the reference's chunked dispatch) -> [1,512,T*16]

Usage:
    python3 tests/dump_stable_audio_fixtures.py --ref <stable-audio-3>/optimized/mlx \
        --model ~/.mlx-serve/models/stabilityai/stable-audio-3-small-sfx [--out DIR]
"""

import argparse
import math
import os
import sys

import mlx.core as mx
import numpy as np

PROMPT = "Dog barking next to a waterfall"
SECONDS = 5.0
SEED = 42
STEPS = 8
T_DIT = 0.7


def torch_dit_to_ref(w):
    out = {}
    for k, v in w.items():
        if not k.startswith("model.model."):
            continue
        k = k[len("model.model."):]
        if k in ("preprocess_conv.weight", "postprocess_conv.weight"):
            v = v.transpose(0, 2, 1)
        if k.endswith(".gamma"):
            k = k[: -len(".gamma")] + ".weight"
        k = k.replace(".to_local_embed.0.", ".to_local_embed.seq.0.").replace(".to_local_embed.2.", ".to_local_embed.seq.2.")
        out[k] = v
    return out


def torch_dec_to_ref(w):
    p = "pretransform.model.decoder.layers."
    out = {
        "project_in.weight": w[p + "1.weight"],
        "project_in.bias": w[p + "1.bias"],
        "new_tokens": w[p + "3.new_tokens"],
        "running_std": w["pretransform.model.bottleneck.running_std"],
        "mapping.bias": w[p + "3.mapping.bias"],
    }
    g, v = w[p + "3.mapping.weight_g"], w[p + "3.mapping.weight_v"]
    norm = mx.sqrt((v * v).sum(axis=(1, 2), keepdims=True))
    out["mapping.weight"] = (g * v / norm).transpose(0, 2, 1)  # [out, k, in]
    for i in range(6):
        t = f"{p}3.transformers.{i}."
        b = f"blocks.{i}."
        for n in ("pre_norm", "ff_norm"):
            for f in ("alpha", "gamma", "beta"):
                out[f"{b}{n}.{f}"] = w[f"{t}{n}.{f}"]
        for n in ("q_norm", "k_norm"):
            for f in ("alpha", "gamma", "beta"):
                out[f"{b}attn.{n}.{f}"] = w[f"{t}self_attn.{n}.{f}"]
        out[f"{b}attn.to_qkv.weight"] = w[f"{t}self_attn.to_qkv.weight"]
        out[f"{b}attn.to_out.weight"] = w[f"{t}self_attn.to_out.weight"]
        out[f"{b}ff.glu_proj.weight"] = w[f"{t}ff.ff.0.proj.weight"]
        out[f"{b}ff.glu_proj.bias"] = w[f"{t}ff.ff.0.proj.bias"]
        out[f"{b}ff.proj_out.weight"] = w[f"{t}ff.ff.2.weight"]
        out[f"{b}ff.proj_out.bias"] = w[f"{t}ff.ff.2.bias"]
    return out


def t5_encode_f32(model_dir, ids):
    from models.defs.t5gemma_mlx import T5GemmaConfig, _Encoder, _rms_norm, _rope_cos_sin

    w = mx.load(os.path.join(model_dir, "t5gemma-b-b-ul2", "model.safetensors"))
    cfg = T5GemmaConfig()
    enc = _Encoder(cfg)
    pre = "model.encoder."
    enc.embed_tokens.weight = w[pre + "embed_tokens.weight"].astype(mx.float32)
    enc.norm = w[pre + "norm.weight"].astype(mx.float32)
    for i, layer in enumerate(enc.layers):
        lp = f"{pre}layers.{i}."
        for f in ("pre_self_attn_layernorm", "post_self_attn_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            setattr(layer, f, w[lp + f + ".weight"].astype(mx.float32))
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(layer.self_attn, n).weight = w[f"{lp}self_attn.{n}.weight"].astype(mx.float32)
        for n in ("gate_proj", "up_proj", "down_proj"):
            getattr(layer.mlp, n).weight = w[f"{lp}mlp.{n}.weight"].astype(mx.float32)
    # The reference forward hard-casts the embedding to fp16; this is the same
    # loop in fp32 (unpadded: the n real tokens attend only to each other).
    x = enc.embed_tokens(mx.array([ids], dtype=mx.int32)) * math.sqrt(cfg.hidden_size)
    cos, sin = _rope_cos_sin(len(ids), cfg.head_dim, cfg.rope_theta)
    for layer in enc.layers:
        x = layer(x, cos, sin, None)
    return _rms_norm(x, enc.norm, cfg.rms_norm_eps)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="stable-audio-3/optimized/mlx")
    ap.add_argument("--model", required=True, help="official stable-audio-3-small-* repo dir")
    ap.add_argument("--out", default=os.path.expanduser("~/claude-tmp/sa3/fixtures"))
    args = ap.parse_args()
    sys.path.insert(0, args.ref)
    from models.defs import dit_mlx, same_s_decoder
    from models.defs.sa3_pipeline import (SecondsTotalEmbedder, apply_prompt_padding,
                                          build_pingpong_schedule, sample_flow_pingpong, patched_decode)
    import sentencepiece as spm

    os.makedirs(args.out, exist_ok=True)

    def save(name, arr, dt=np.float32):
        path = os.path.join(args.out, name + ".raw")
        np.asarray(arr).astype(dt).tofile(path)
        return path

    env = {}
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.model, "t5gemma-b-b-ul2", "tokenizer.model"))
    ids = sp.Encode(PROMPT)
    env["SA3_TOK"] = save("tok", ids, np.int32)

    hidden = t5_encode_f32(args.model, ids)
    env["SA3_T5"] = save("t5", hidden)

    w = mx.load(os.path.join(args.model, "model.safetensors"))
    cp = "conditioner.conditioners."
    secs = SecondsTotalEmbedder(w[cp + "seconds_total.embedder.embedding.1.weight"],
                                w[cp + "seconds_total.embedder.embedding.1.bias"])
    embeds = mx.zeros((1, 256, 768))
    embeds[:, : len(ids), :] = hidden[None]
    mask = mx.array([[1] * len(ids) + [0] * (256 - len(ids))], dtype=mx.int32)
    padded = apply_prompt_padding(embeds, mask, w[cp + "prompt.padding_embedding"])
    sec = secs(SECONDS)
    cross = mx.concatenate([padded, sec], axis=1)
    glob = sec[:, 0, :]
    env["SA3_CROSS"] = save("cross", cross)
    env["SA3_GLOBAL"] = save("global", glob)

    T = max(1, math.ceil(SECONDS * 44100 / 4096))
    dit = dit_mlx.DiT(T_lat=T)
    dit.load_weights(list(torch_dit_to_ref(w).items()), strict=False)
    x = mx.random.normal((1, 256, T), key=mx.random.key(7))
    v = dit(x, mx.array([T_DIT]), cross, glob)
    env["SA3_X"] = save("x", x)
    env["SA3_DIT"] = save("dit", v)

    noise = mx.random.normal((1, 256, T), dtype=mx.float32, key=mx.random.key(SEED))
    sigmas = build_pingpong_schedule(STEPS, sigma_max=1.0, use_logsnr_shift=True)
    lat = sample_flow_pingpong(lambda xx, tt: dit(xx, tt, cross, glob), noise, sigmas, seed=SEED + 1)
    env["SA3_LAT"] = save("lat", lat)

    dec = same_s_decoder.SAMESDecoder()
    dec.load_weights(list(torch_dec_to_ref(w).items()), strict=True)
    patches = same_s_decoder.decode_chunked(dec, lat, 8, 2)
    env["SA3_PATCH"] = save("patch", patches)
    audio = patched_decode(patches)
    print(f"# T_lat={T} ids={len(ids)} audio={tuple(audio.shape)} peak={float(mx.abs(audio).max()):.3f}")

    print(f"export SA3_TEST_MODEL={args.model}")
    for k, p in env.items():
        print(f"export {k}={p}")


if __name__ == "__main__":
    main()
