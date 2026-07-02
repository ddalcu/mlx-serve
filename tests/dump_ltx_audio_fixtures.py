#!/usr/bin/env python3
"""Dump LTX-2 AUDIO parity fixtures (.raw) for the Zig oracle tests in src/ltx_audio.zig.

Runs the Lightricks/LTX-2 PyTorch REFERENCE audio decode path (audio VAE +
BigVGAN vocoder) on a fixed random latent and writes byte-exact golden tensors
that `zig build test` compares the Zig port against (cosine parity).

USER-RUN: you accept LTX-2's license, download the model, and install the
reference package (`pip install -e packages/ltx-core packages/ltx-pipelines`
from a Lightricks/LTX-2 checkout, plus torch). This script is NOT run in CI and
is NOT exercised by the always-on tests — the hermetic + weights-gated
structural tests in src/ltx_audio.zig cover the port without it.

Usage:
    python3 tests/dump_ltx_audio_fixtures.py <AUDIO_VAE.safetensors> [OUT_DIR]

Writes (flat little-endian f32):
    latent.raw   [1, Na, 128]   the DiT-shaped (patchified) audio latent — the
                                EXACT input src/ltx_audio.zig decodeAudio expects
    mel.raw      [1, 2, T, 64]  reference audio_vae.decoder(latent) output
    wave.raw     [L, 2]         reference vocoder(mel), INTERLEAVED L,R,L,R… to
                                match the Zig vocode() output layout

Then it prints the `export LTX_AUDIO_*` block to paste before:
    LTX_AUDIO_TEST_MODEL=<AUDIO_VAE.safetensors> LTX_AUDIO_LATENT=…/latent.raw \
      LTX_AUDIO_MEL=…/mel.raw LTX_AUDIO_WAVE=…/wave.raw \
      zig build test -Dtest-filter="reference"

Oracles validated: VAE mel (cos>0.99), vocoder waveform (cos>0.9).
"""

import os
import sys

import numpy as np
import torch
import einops

# Fixed, reproducible fixture parameters. Na audio tokens → mel T = 4*Na-3.
NA = 16
SEED = 0


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ckpt = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./ltx_audio_fixtures"
    os.makedirs(out_dir, exist_ok=True)

    # The reference builds the decoder + vocoder from the checkpoint via the
    # AudioDecoder block's two Builders (it owns the comfy-key filters + the
    # configurators that read the safetensors `__metadata__["config"]`).
    from ltx_pipelines.utils.blocks import AudioDecoder as RefAudioDecoder
    from ltx_core.model.audio_vae.audio_vae import decode_audio  # noqa: F401  (sanity import)

    device = torch.device("cpu")
    dtype = torch.bfloat16
    block = RefAudioDecoder(checkpoint_path=ckpt, dtype=dtype, device=device)
    decoder = block._decoder_builder.build(device=device, dtype=dtype).eval()
    vocoder = block._vocoder_builder.build(device=device, dtype=dtype).eval()

    # A fixed random latent in the decoder's input shape [1, 8(z), Na, 16(mel)].
    torch.manual_seed(SEED)
    latent_8 = torch.randn(1, 8, NA, 16, dtype=torch.float32).to(dtype)

    with torch.no_grad():
        mel = decoder(latent_8)                 # [1, 2, T, 64]
        wave = vocoder(mel).squeeze(0).float()  # [2, L]

    # The Zig pipeline takes the PATCHIFIED latent [1, Na, 128] (= the DiT output
    # shape). patchify is "b c t f -> b t (c f)" (c outer, f inner) — exactly the
    # 128-channel layout decodeAudio denormalizes + unpatchifies.
    latent_patched = einops.rearrange(latent_8.float(), "b c t f -> b t (c f)").contiguous()
    wave_interleaved = wave.transpose(0, 1).contiguous()  # [L, 2] L,R,L,R…

    def save(name, t):
        arr = t.detach().cpu().float().numpy().astype("<f4").ravel()
        path = os.path.join(out_dir, name)
        arr.tofile(path)
        return path, t.shape

    lat_p, lat_s = save("latent.raw", latent_patched)
    mel_p, mel_s = save("mel.raw", mel)
    wav_p, wav_s = save("wave.raw", wave_interleaved)

    print(f"latent {tuple(lat_s)} -> {lat_p}")
    print(f"mel    {tuple(mel_s)} -> {mel_p}")
    print(f"wave   {tuple(wav_s)} -> {wav_p}")
    print()
    print("Paste this to run the Zig parity oracles:")
    print(
        f'  LTX_AUDIO_TEST_MODEL="{ckpt}" \\\n'
        f'  LTX_AUDIO_LATENT="{lat_p}" LTX_AUDIO_MEL="{mel_p}" LTX_AUDIO_WAVE="{wav_p}" \\\n'
        f'  zig build test -Dtest-filter="reference"'
    )


if __name__ == "__main__":
    main()
