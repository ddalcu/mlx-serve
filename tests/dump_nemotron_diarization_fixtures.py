#!/usr/bin/env python3
"""Dump Nemotron 3 Diarization parity fixtures — the references EXECUTE.

Runs mlx-audio's `nemotron_diarization` (the pack layout our loader matches)
and the official model in 🤗 Transformers (float32, CPU) on a 16 kHz mono WAV,
and saves what `src/plugins/nemotron_diarization/diarization.zig` has to reproduce:
  pcm              the 16 kHz samples (f32)
  features         mlx-audio's log-mel of the first window [128, frames] (f32)
  mlx_probs        mlx-audio's speaker activity [frames, 8] with the pack (f32)
  hf_probs         the official model's speaker activity [frames, 8] (f32)
  aosc_probs_<i>   the probabilities of each speaker-cache compression the
                   official model ran [n, 8] (f32)
  aosc_frames_<i>  the frames that compression kept, -1 for a silence slot (i32)

Usage:
    HF_HOME=<scratch> venv/bin/python -I tests/dump_nemotron_diarization_fixtures.py \
        --model ~/.mlx-serve/models/mlx-community/Nemotron-3-Diarization-8bit \
        --audio conversation.wav --out diarization_fixture.safetensors
"""

import argparse

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch
from mlx_audio.vad import load
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.models.nemotron3_diarization import modeling_nemotron3_diarization as hf

SOURCE = "nvidia/Nemotron-3-Diarization"


def official(pcm):
    """The official forward, recording which frames every compression keeps."""
    processor = AutoProcessor.from_pretrained(SOURCE)
    model = AutoModelForAudioFrameClassification.from_pretrained(SOURCE, dtype=torch.float32).eval()
    cases = []
    compress = hf.Nemotron3DiarizationSpeakerCache._compress

    def recording(cache, embeds, probs, silence_embeds):
        # Frame ids as one-wide embeddings reveal the selection; silence slots read -1.
        ids = torch.arange(probs.shape[1], dtype=torch.float32).view(1, -1, 1)
        kept, _ = compress(cache, ids, probs.float(), torch.tensor([-1.0]))
        cases.append((probs[0].float().numpy().copy(), kept[0, :, 0].numpy().astype(np.int32)))
        return compress(cache, embeds, probs, silence_embeds)

    hf.Nemotron3DiarizationSpeakerCache._compress = recording
    inputs = processor(pcm, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    hf.Nemotron3DiarizationSpeakerCache._compress = compress
    return torch.sigmoid(logits)[0].numpy(), cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pcm, rate = sf.read(args.audio, dtype="float32")
    assert rate == 16000 and pcm.ndim == 1, "a 16 kHz mono WAV"

    model = load(args.model, strict=True)
    proc, mod = model.config.processor_config, model.config.modules_config
    window = (mod.chunk_len + mod.chunk_right_context) * model.config.encoder_config.subsampling_factor
    count = min(window, len(pcm) // proc.hop_length)
    features = np.array(model.preprocessor(mx.array(pcm), start=0, count=count)[0])
    mlx_probs = np.array(model.generate(pcm).speaker_probs, dtype=np.float32)

    hf_probs, cases = official(pcm)
    hf_probs = hf_probs[: len(mlx_probs)]
    print(f"frames {len(mlx_probs)}, compressions {len(cases)}")
    agree = ((mlx_probs > 0.5) == (hf_probs > 0.5)).mean()
    print(f"mlx-audio vs official: max |diff| {np.abs(mlx_probs - hf_probs).max():.4f}, active agreement {agree:.4%}")

    out = {
        "pcm": mx.array(pcm),
        "features": mx.array(features),
        "mlx_probs": mx.array(mlx_probs),
        "hf_probs": mx.array(hf_probs.astype(np.float32)),
    }
    for i, (probs, frames) in enumerate(cases):
        out[f"aosc_probs_{i}"] = mx.array(probs)
        out[f"aosc_frames_{i}"] = mx.array(frames)
    mx.save_safetensors(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
