#!/usr/bin/env python3
"""Dump a Qwen3-TTS voice fixture (CustomVoice speaker or VoiceDesign description): the reference EXECUTES.

Runs mlx-audio's own `qwen3_tts` model on one request and writes, as raw little-endian files in --out:
  input_ids.i32     the text turn's token ids
  instruct_ids.i32  the instruct turn's token ids (only with --instruct)
  speaker.txt       the speaker's name (only with --speaker)
  prefill.f32       the talker's first input, `_prepare_generation_inputs` [L, hidden]
  codes.i32         greedy codes [frames, 16] (temperature 0, no repetition penalty), the reference's own samplers

`src/plugins/qwen3_tts/tts.zig` "a voice prefix matches the reference" reads them:
    TTS_VOICE_MODEL=<pack> TTS_VOICE_FIXTURE=<out> zig build test -Dtest-filter="a voice prefix matches"

Usage:
    venv/bin/python -I tests/dump_qwen3_tts_voice_fixture.py \
        --model <models>/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
        --speaker ryan --instruct "Speak very angrily and loudly." --out ~/claude-tmp/qwen3-tts/custom_voice
"""

import argparse
import os

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--speaker")
    ap.add_argument("--instruct")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.model)
    tok = model.tokenizer
    ids = tok.encode(f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n")
    np.array(ids, dtype="<i4").tofile(f"{args.out}/input_ids.i32")
    if args.instruct:
        np.array(tok.encode(f"<|im_start|>user\n{args.instruct}<|im_end|>\n"), dtype="<i4").tofile(f"{args.out}/instruct_ids.i32")
    if args.speaker:
        with open(f"{args.out}/speaker.txt", "w") as f:
            f.write(args.speaker)

    embeds, trailing, tts_pad = model._prepare_generation_inputs(
        text=args.text, language=args.language, speaker=args.speaker, instruct=args.instruct
    )
    np.array(embeds[0].astype(mx.float32), dtype="<f4").tofile(f"{args.out}/prefill.f32")

    cfg = model.config.talker_config
    eos = cfg.codec_eos_token_id
    suppress = [i for i in range(cfg.vocab_size - 1024, cfg.vocab_size) if i != eos]
    cache = model.talker.make_cache()
    frames = []
    x = embeds
    for _ in range(args.frames):
        logits, hidden = model.talker(x, cache=cache)
        first = model._sample_token(logits, temperature=0.0, repetition_penalty=1.0, suppress_tokens=suppress)
        if int(first[0, 0]) == eos:
            break
        tokens, codes = model._predict_code_tokens(first, hidden, temperature=0.0, top_k=0, top_p=1.0)
        frames.append(np.array(codes[0], dtype="<i4"))
        x = tts_pad + model._codec_embeds_for_tokens(tokens)
        mx.eval(x)
    np.stack(frames).tofile(f"{args.out}/codes.i32")
    print(f"{args.out}: prefill {tuple(embeds.shape)}, {len(frames)} greedy frames, first {frames[0].tolist()}")


if __name__ == "__main__":
    main()
