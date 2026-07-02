#!/usr/bin/env python3
"""Dump LTX-2.3 audio VAE ENCODER parity fixtures (.raw) for the Zig oracle in
src/ltx_audio.zig ("audio VAE encoder matches the reference latent").

Runs the ltx-2-mlx REFERENCE encode path (AudioProcessor mel + AudioVAEEncoder)
on a fixed deterministic stereo waveform and writes the patchified DiT-shaped
latent the Zig port must reproduce (cosine parity). This is the exact path
a2vid_two_stage.py uses to build the frozen audio conditioning tokens.

USER-RUN (not CI): needs the ltx-2-mlx checkout's environment, e.g.

    cd ~/projects/agents/ltx-2-mlx && uv run python \
      ~/projects/agents/mlx-serve/tests/dump_ltx_audio_encoder_fixtures.py \
      ~/.mlx-serve/models/dgrauet/ltx-2.3-mlx-q4 /tmp/ltx_audio_enc_fixtures

Writes (flat little-endian f32):
    enc_wave.raw    [S, 2]     interleaved L,R stereo waveform @ 16 kHz — the
                               EXACT input encodeAudioCond() expects
    enc_latent.raw  [1, T, 128] reference patchified audio latent (normalized)

Then paste the printed exports and run:
    LTX_AUDIO_TEST_MODEL=<model_dir> LTX_AUDIO_ENC_WAVE=…/enc_wave.raw \
      LTX_AUDIO_ENC_LATENT=…/enc_latent.raw \
      zig build test -Dtest-filter="encoder matches the reference"
"""

import os
import sys

import numpy as np

SEED = 0
DURATION_S = 1.5
SR = 16000


def build_waveform() -> np.ndarray:
    """Deterministic stereo test signal: tones + band noise, (1, 2, S) f32."""
    rng = np.random.default_rng(SEED)
    t = np.arange(int(DURATION_S * SR), dtype=np.float64) / SR
    left = 0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 1760.0 * t)
    right = 0.4 * np.sin(2 * np.pi * 330.0 * t) + 0.15 * np.sin(2 * np.pi * 987.0 * t)
    noise = 0.05 * rng.standard_normal((2, t.shape[0]))
    wav = np.stack([left, right], axis=0) + noise
    # AM envelope so the mel has temporal structure (speech-ish dynamics)
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    wav = (wav * env).astype(np.float32)
    return wav[None, :, :]  # (1, 2, S)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    model_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./ltx_audio_enc_fixtures"
    os.makedirs(out_dir, exist_ok=True)

    import mlx.core as mx
    from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder
    from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

    encoder = AudioVAEEncoder()
    path = os.path.join(model_dir, "audio_vae.safetensors")
    encoder_weights = load_split_safetensors(path, prefix="audio_vae.encoder.")
    all_audio = load_split_safetensors(path, prefix="audio_vae.")
    for k, v in all_audio.items():
        if k.startswith("per_channel_statistics."):
            encoder_weights[k] = v
    encoder_weights = remap_audio_vae_keys(encoder_weights)
    encoder.load_weights(list(encoder_weights.items()))

    processor = AudioProcessor(sample_rate=SR)
    wav = build_waveform()  # (1, 2, S)
    mel = processor.waveform_to_mel(mx.array(wav))  # (1, 2, T', 64)
    latent = encoder.encode(mel)  # (1, 8, T, 16)
    tokens = latent.transpose(0, 2, 1, 3).reshape(1, latent.shape[2], 128)  # (1, T, 128)
    mx.eval(tokens)

    tokens_np = np.asarray(tokens.astype(mx.float32))
    wave_interleaved = np.ascontiguousarray(wav[0].T)  # (S, 2) L,R interleaved

    wave_p = os.path.join(out_dir, "enc_wave.raw")
    lat_p = os.path.join(out_dir, "enc_latent.raw")
    wave_interleaved.astype("<f4").tofile(wave_p)
    tokens_np.astype("<f4").tofile(lat_p)

    print(f"mel: {mel.shape}  latent: {latent.shape}  tokens: {tokens_np.shape}")
    print(f"tokens mean|x|={np.abs(tokens_np).mean():.4f}")
    print("\nexport LTX_AUDIO_TEST_MODEL=" + model_dir)
    print("export LTX_AUDIO_ENC_WAVE=" + wave_p)
    print("export LTX_AUDIO_ENC_LATENT=" + lat_p)
    print('zig build test -Doptimize=ReleaseFast -Dtest-filter="encoder matches the reference"')


if __name__ == "__main__":
    main()
