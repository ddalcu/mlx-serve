#!/usr/bin/env python3
"""Dump YuE2 parity fixtures (.raw f32 / i32) for the Zig oracle tests in `src/yue2.zig`.

USER-RUN (needs mlx + numpy + tiktoken). Runs the ahmadw/YuE2-3B-MLX Python reference
(its yue2_model.py / yue2_vae.py / generate.py) over the SAME pack the engine loads.

Oracle taps (env prefix YUE2_*):
  TOK_TEXT / TOK_IDS : a mixed-script string -> the reference tiktoken ids
  AR_PREFIX          : a cot=off prefix; AR_LOGITS0 its last-row logits [V],
                       AR_NEXT one codec id fed after it; AR_LOGITS1 the logits after it
  NAR_AR / NAR_STATE / NAR_V : AR ids + a bf16 state [24,64] at raw t -> velocity [24,64]
  NAR_LAT            : the seeded 4-step midpoint solve over those 24 frames [24,64]
  VAE_LAT / VAE_AUDIO: 1100 random latents (two decode tiles) -> audio [N,2]

Usage:
    python3 tests/dump_yue2_fixtures.py --ref <YuE2-3B-MLX checkout> --model <pack dir> [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

STYLE = "English, indie pop, bright acoustic guitar, soft drums, warm lead vocal"
LYRICS = "[Verse]\nSoft morning light is touching the window.\n[Chorus]\n留在节奏里 let it carry us home."
TOK_TEXT = "Hello, world! 你好，世界。\nline two\t'tabs' don't — 12345 𝄞 café"
SEED = 7
RAW_T = 1.0986123  # logit(0.75)


def dump(out: Path, name: str, arr, dtype):
    np.asarray(arr).astype(dtype).tofile(out / name)
    print(f"export YUE2_{name.split('.')[0].upper()}={out / name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/yue2_fixtures"))
    args = ap.parse_args()
    sys.path.insert(0, str(args.ref))
    import generate as g
    from yue2_model import KVCache, load_model
    from yue2_vae import load_vae

    args.out.mkdir(parents=True, exist_ok=True)
    tok = g.Tokenizer(args.model / "qwen.tiktoken")
    (args.out / "tok_text.txt").write_text(TOK_TEXT, encoding="utf-8")
    print(f"export YUE2_TOK_TEXT={args.out / 'tok_text.txt'}")
    dump(args.out, "tok_ids.i32", tok.encode(TOK_TEXT), np.int32)

    model = load_model(args.model)
    prefix = g.token_prefix(tok, STYLE, LYRICS, "off")
    dump(args.out, "ar_prefix.i32", prefix, np.int32)
    caches = [KVCache() for _ in model.model.layers]
    dump(args.out, "ar_logits0.f32", model.ar_step(mx.array([prefix]), caches).astype(mx.float32), np.float32)
    nxt = g.CODEC_OFFSET + 123
    dump(args.out, "ar_next.i32", [nxt], np.int32)
    dump(args.out, "ar_logits1.f32", model.ar_step(mx.array([[nxt]]), caches).astype(mx.float32), np.float32)

    rng = np.random.default_rng(SEED)
    codec = [int(c) for c in rng.integers(0, g.CODEC_SIZE, 24)]
    ar_ids = prefix + [c + g.CODEC_OFFSET for c in codec] + [g.MUSIC_END]
    dump(args.out, "nar_ar.i32", ar_ids, np.int32)
    state = mx.random.normal((24, 64), key=mx.random.key(SEED)).astype(mx.bfloat16)
    dump(args.out, "nar_state.f32", state.astype(mx.float32), np.float32)
    cache = model.nar_prefill(ar_ids)
    v = model.nar_velocity(state, RAW_T, cache, len(ar_ids))
    dump(args.out, "nar_v.f32", v.astype(mx.float32), np.float32)
    lat = g.synthesize(model, prefix, codec, SEED, steps=4)
    dump(args.out, "nar_lat.f32", lat, np.float32)
    np.asarray([len(prefix)], dtype=np.int32).tofile(args.out / "nar_prefix_len.i32")
    print(f"export YUE2_NAR_PREFIX_LEN={args.out / 'nar_prefix_len.i32'}")
    np.asarray(codec, dtype=np.int32).tofile(args.out / "nar_codec.i32")
    print(f"export YUE2_NAR_CODEC={args.out / 'nar_codec.i32'}")

    vae = load_vae(args.model)
    z = mx.random.normal((1100, 64), key=mx.random.key(SEED + 1)) * 0.7
    dump(args.out, "vae_lat.f32", z, np.float32)
    dump(args.out, "vae_audio.f32", mx.clip(vae.decode_tiled(z), -1, 1), np.float32)


if __name__ == "__main__":
    main()
