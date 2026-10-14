#!/usr/bin/env python3
"""Dump Qwen3-ASR parity fixtures — the reference EXECUTES.

Runs mlx-audio's own `qwen3_asr` model (the layout our pack loader matches) on a
16 kHz mono WAV and saves every stage `src/qwen_audio.zig` and the transcription
endpoint have to reproduce:
  pcm           the 16 kHz samples the feature extractor read (f32)
  features      WhisperFeatureExtractor log-mel [128, frames] (f32)
  audio_embeds  the audio tower's soft tokens [n, text_hidden] (f32)
  prompt_ids    the reference prompt, language auto-detected (i32)
  gen_ids       greedy generated ids, EOS excluded (i32)

Usage:
    venv/bin/python -I tests/dump_qwen3_asr_fixtures.py \
        --model ~/.mlx-serve/models/mlx-community/Qwen3-ASR-0.6B-8bit \
        --audio ~/claude-tmp/qwen3-asr/speech.wav \
        --out ~/claude-tmp/qwen3-asr/speech_fixture.safetensors
"""

import argparse

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load
from mlx_audio.stt.utils import load_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    model = load(args.model)._model
    pcm = np.array(load_audio(args.audio), dtype=np.float32)

    feats, mask, n_audio = model._preprocess_audio(pcm)
    embeds = model.get_audio_features(feats, mask)
    prompt = model._build_prompt(n_audio, None, None)

    gen = []
    for tok, _ in model.stream_generate(pcm, max_tokens=args.max_tokens):
        gen.append(int(tok))
    print("frames", feats.shape[-1], "mask", int(mask.sum()), "audio tokens", n_audio, "embeds", embeds.shape)
    print("text", repr(model._tokenizer.decode(gen, skip_special_tokens=True)))

    mx.save_safetensors(args.out, {
        "pcm": mx.array(pcm),
        "features": feats[0].astype(mx.float32),
        "audio_embeds": embeds.astype(mx.float32),
        "prompt_ids": prompt[0].astype(mx.int32),
        "gen_ids": mx.array(np.array(gen, dtype=np.int32)),
    })


if __name__ == "__main__":
    main()
