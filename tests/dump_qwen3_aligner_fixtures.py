#!/usr/bin/env python3
"""Dump Qwen3-ForcedAligner parity fixtures — the reference EXECUTES.

Runs mlx-audio's own `qwen3_forced_aligner` (the layout our pack loader
matches) on a 16 kHz mono WAV and its transcript, and saves every stage
`src/plugins/qwen3_asr/forced_align.zig` has to reproduce:
  text          the transcript, UTF-8 (u8)
  pcm           the 16 kHz samples the feature extractor read (f32)
  features      WhisperFeatureExtractor log-mel [128, frames] (f32)
  audio_embeds  the aligner's audio tower rows [n, text_hidden] (f32)
  input_ids     the aligner prompt, its audio placeholder expanded (i32)
  classes       the head's argmax at every <timestamp> (i32)
  times_ms      the repaired times, a start and an end per unit (i32)

Usage:
    venv/bin/python -I tests/dump_qwen3_aligner_fixtures.py \
        --model ~/.mlx-serve/models/mlx-community/Qwen3-ForcedAligner-0.6B-8bit \
        --audio speech.wav --text "The quick brown fox." --language English \
        --out aligner_fixture.safetensors
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
    ap.add_argument("--text", required=True)
    ap.add_argument("--language", default="English")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = load(args.model)._model
    pcm = np.array(load_audio(args.audio), dtype=np.float32)

    feats, mask, n_audio = model._preprocess_audio(pcm)
    embeds = model.get_audio_features(feats, mask)
    words, prompt = model.aligner_processor.encode_timestamp(args.text, args.language)
    prompt = prompt.replace("<|audio_pad|>", "<|audio_pad|>" * n_audio)
    ids = model._tokenizer.encode(prompt, add_special_tokens=False)
    logits = model(mx.array([ids]), input_features=feats, feature_attention_mask=mask)
    at = np.array(ids) == model.config.timestamp_token_id
    classes = np.array(mx.argmax(logits, axis=-1))[0][at]
    times = model.aligner_processor.fix_timestamp(classes * model.config.timestamp_segment_time)
    print("audio tokens", n_audio, "prompt", len(ids), "units", len(words))
    for k, w in enumerate(words):
        print(f"  {w} {times[2 * k]} {times[2 * k + 1]}")

    mx.save_safetensors(args.out, {
        "text": mx.array(np.frombuffer(args.text.encode(), dtype=np.uint8)),
        "pcm": mx.array(pcm),
        "features": feats[0].astype(mx.float32),
        "audio_embeds": embeds.astype(mx.float32),
        "input_ids": mx.array(np.array(ids, dtype=np.int32)),
        "classes": mx.array(classes.astype(np.int32)),
        "times_ms": mx.array(np.array(times, dtype=np.int32)),
    })


if __name__ == "__main__":
    main()
