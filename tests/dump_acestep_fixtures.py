#!/usr/bin/env python3
"""Dump ACE-Step v1.5 XL Turbo parity fixtures (.raw f32) for the Zig oracle
tests in `src/acestep.zig`.

USER-RUN (needs torch + transformers + diffusers + vector-quantize-pytorch +
einops + the HF checkpoints). Runs the REFERENCE PyTorch modeling code and
writes byte-exact golden tensors that `zig build test` compares the Zig port
against (cosine parity), mirroring `tests/dump_hunyuan3d_fixtures.py`.

MEMORY: the XL checkpoint is 5B params fp32 (~20 GB resident) — run this on the
128 GB Mac, NOT the 16 GB dev machine. Per the CLAUDE.md fixture-corruption
gotcha (deep nets decorrelate at fp16 on MPS), everything is dumped from an
fp32 model; --device mps runs fp32-on-MPS (fine numerically), --device cpu is
the paranoid fallback.

Oracle taps (env prefix ACESTEP_*):

  1 TEXT  : fixed caption prompt -> Qwen3-Embedding-0.6B token ids + last_hidden_state
            (+ LYRIC: lyric token ids + embed-table lookups)
  2 COND  : text/lyric embeds + silence-latent timbre -> packed encoder_hidden_states
  3 DIT   : injected noise [1,250,64] + conditioning + t=1.0 -> one velocity pred
  4 E2E   : full 8-step euler (shift=3.0) + DCW haar "double" (0.05/0.02) from
            oracle 3's noise -> final latents  (sampler loop mirrors the pipeline's
            MLX reference dit_generate.py, which is the shipping default path)
  5 VAEDEC: fixed latent [1,50,64] -> waveform [1,2,96000]
  6 VAEENC: fixed audio [1,2,96000] -> latent MEAN [1,64,50] (deterministic; the
            reference samples the posterior at runtime — M3 may add that noise)

Usage:
    python3 tests/dump_acestep_fixtures.py --src-xl <acestep-v15-xl-turbo dir> \
        --src-main <Ace-Step1.5 dir> [--out DIR] [--device auto|mps|cpu] \
        [--test-model CONVERTED_DIR]

Prints the exact `export ACESTEP_*` block to paste before running the Zig oracles.
"""

import argparse
import math
import os
import sys

import numpy as np

SEED = 0
DURATION_S = 10
FRAMES = DURATION_S * 25          # 250 latent frames
TIMBRE_FRAMES = 750               # timbre_fix_frame
NUM_STEPS = 8
SHIFT = 3.0
DCW_LOW = 0.05
DCW_HIGH = 0.02
VAEDEC_FRAMES = 50                # 2 s
HOP = 1920

# ── exact conditioning strings (music-dossier.md §2.4; the Zig side must build
# these byte-identically — the fixture pins them) ─────────────────────────────
INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
CAPTION = "upbeat synthwave with driving bass, dreamy pads and a catchy lead melody"
LYRICS = "[Instrumental]"
LANGUAGE = "en"

SFT_GEN_PROMPT = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n"


def build_meta_string(bpm=None, keyscale="", timesignature="", duration=None):
    parts = [
        f"- bpm: {bpm if bpm else 'N/A'}",
        f"- timesignature: {timesignature if timesignature.strip() else 'N/A'}",
        f"- keyscale: {keyscale if keyscale.strip() else 'N/A'}",
    ]
    s = "\n".join(parts) + "\n"
    if duration is not None:
        s = s[:-1] + f"\n- duration: {int(duration)} seconds\n"
    return s


def build_text_prompt():
    metas = build_meta_string(duration=DURATION_S)
    return SFT_GEN_PROMPT.format(INSTRUCTION, CAPTION, metas)


def build_lyric_text():
    return f"# Languages\n{LANGUAGE}\n\n# Lyric\n{LYRICS}<|endoftext|>"


# ── DCW haar (mirrors acestep/models/mlx/dcw_correction_mlx.py, torch) ────────
def haar_dwt(x):
    T = x.shape[1]
    if T % 2 == 1:
        import torch
        x = torch.cat([x, torch.zeros_like(x[:, :1, :])], dim=1)
    even, odd = x[:, 0::2, :], x[:, 1::2, :]
    inv = 1.0 / math.sqrt(2.0)
    return (even + odd) * inv, (even - odd) * inv


def haar_idwt(low, high, out_T):
    import torch
    inv = 1.0 / math.sqrt(2.0)
    even, odd = (low + high) * inv, (low - high) * inv
    stacked = torch.stack([even, odd], dim=2)      # [B, T//2, 2, C]
    rec = stacked.reshape(even.shape[0], -1, even.shape[2])
    return rec[:, :out_T, :]


def apply_dcw_double(x_next, denoised, t_curr):
    low_s = t_curr * DCW_LOW
    high_s = (1.0 - t_curr) * DCW_HIGH
    T = x_next.shape[1]
    xL, xH = haar_dwt(x_next)
    yL, yH = haar_dwt(denoised)
    if low_s != 0.0:
        xL = xL + low_s * (xL - yL)
    if high_s != 0.0:
        xH = xH + high_s * (xH - yH)
    return haar_idwt(xL, xH, T)


def timestep_schedule(num_steps, shift):
    raw = [1.0 - i / num_steps for i in range(num_steps)]
    if shift != 1.0:
        raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
    return raw


def pick_device(requested):
    import torch
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="Dump ACE-Step parity fixtures.")
    ap.add_argument("--src-xl", required=True, help="dir of ACE-Step/acestep-v15-xl-turbo snapshot")
    ap.add_argument("--src-main", required=True, help="dir of ACE-Step/Ace-Step1.5 snapshot")
    ap.add_argument("--out", default=os.path.join("tests", "fixtures", "acestep"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-dit", action="store_true",
                    help="skip oracles 3/4 (the 20 GB DiT) — text/cond/VAE only")
    ap.add_argument("--test-model",
                    default=os.path.expanduser("~/.mlx-serve/models/local/acestep-v15-xl-turbo-bf16"),
                    help="converted mlx-serve model dir (echoed as ACESTEP_TEST_MODEL; "
                         "use the bf16 parity build for oracle runs)")
    args = ap.parse_args()

    import torch

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(args.device)
    print(f"[dump] src-xl={args.src_xl}\n[dump] src-main={args.src_main}")
    print(f"[dump] out={out_dir}\n[dump] device={device} (fp32)")

    def save_f32(name, arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().float().cpu().numpy()
        np.asarray(arr, dtype="<f4").ravel().tofile(os.path.join(out_dir, name))
        return list(np.asarray(arr).shape)

    def save_i32(name, arr):
        np.asarray(arr, dtype="<i4").ravel().tofile(os.path.join(out_dir, name))
        return list(np.asarray(arr).shape)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── oracle 1: Qwen3-Embedding text encoder ────────────────────────────────
    from transformers import AutoModel, AutoTokenizer
    te_dir = os.path.join(args.src_main, "Qwen3-Embedding-0.6B")
    tok = AutoTokenizer.from_pretrained(te_dir)
    text_encoder = AutoModel.from_pretrained(te_dir, torch_dtype=torch.float32).to(device).eval()

    prompt = build_text_prompt()
    lyric_text = build_lyric_text()
    with open(os.path.join(out_dir, "acestep_prompt.txt"), "w") as f:
        f.write(prompt)
    with open(os.path.join(out_dir, "acestep_lyrics.txt"), "w") as f:
        f.write(lyric_text)

    enc = tok(prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt")
    text_ids = enc.input_ids.to(device)
    with torch.inference_mode():
        text_hidden = text_encoder(input_ids=text_ids).last_hidden_state  # [1,T,1024]
    print(f"[o1] text ids {save_i32('acestep_text_ids.raw', text_ids.cpu().numpy())} "
          f"hidden {save_f32('acestep_text_hidden.raw', text_hidden)}")

    lenc = tok(lyric_text, padding="longest", truncation=True, max_length=2048, return_tensors="pt")
    lyric_ids = lenc.input_ids.to(device)
    with torch.inference_mode():
        lyric_embeds = text_encoder.embed_tokens(lyric_ids)               # [1,T,1024]
    print(f"[o1] lyric ids {save_i32('acestep_lyric_ids.raw', lyric_ids.cpu().numpy())} "
          f"embeds {save_f32('acestep_lyric_embeds.raw', lyric_embeds)}")

    text_encoder.to("cpu")
    del text_encoder
    if device == "mps":
        torch.mps.empty_cache()

    # ── load the XL model (fp32; needs vector_quantize_pytorch + einops) ─────
    sys.path.insert(0, args.src_xl)
    from modeling_acestep_v15_xl_turbo import AceStepConditionGenerationModel

    # transformers' meta-device init breaks ResidualFSQ's constructor (it
    # calls .item() during __init__, illegal on meta tensors). None of our
    # oracles touch the FSQ path (cover-mode only, dropped from the converted
    # model too), so stub it out before instantiation; its checkpoint weights
    # land as ignorable "unexpected keys".
    import modeling_acestep_v15_xl_turbo as _m

    class _FsqStub(torch.nn.Module):
        def __init__(self, **kwargs):  # noqa: ARG002 — signature-compatible
            super().__init__()

    _m.ResidualFSQ = _FsqStub
    print("[load] XL-turbo fp32 (~20 GB) ...")
    model = AceStepConditionGenerationModel.from_pretrained(
        args.src_xl, torch_dtype=torch.float32)
    model.config._attn_implementation = "sdpa"
    model = model.to(device).eval()

    # transformers ≥5.x meta-device init leaves the NON-PERSISTENT rotary
    # `inv_freq` buffers of custom-code models as ZEROS (they're not in the
    # checkpoint and the re-init hook only covers transformers' own classes)
    # → every custom rotary runs IDENTITY rope and the fixtures silently
    # decorrelate from the true (transformers 4.57 / official-MLX) semantics.
    # Symptom that caught it: the Zig DiT oracle "passed" with rope REMOVED.
    # Recompute each buffer from its module's config.
    fixed = 0
    for _name, _mod in model.named_modules():
        if _name.endswith("rotary_emb") and hasattr(_mod, "inv_freq"):
            if bool((_mod.inv_freq == 0).all()):
                # A FRESH instance (real-device init) computes the correct
                # buffer from the same config; copy it over.
                _fresh = type(_mod)(config=_mod.config)
                _mod.inv_freq = _fresh.inv_freq.to(device)
                _mod.attention_scaling = _fresh.attention_scaling
                fixed += 1
    if fixed:
        print(f"[fix] re-initialized {fixed} zeroed rotary inv_freq buffer(s) "
              "(transformers 5.x meta-init artifact)")

    silence = torch.load(os.path.join(args.src_xl, "silence_latent.pt"),
                         map_location="cpu", weights_only=True)
    silence = silence.transpose(1, 2).to(device=device, dtype=torch.float32)  # [1,15000,64]
    assert silence.shape[0] == 1 and silence.shape[2] == 64, silence.shape

    # ── oracle 2: condition encoder (silence timbre = the text2music path) ───
    ones = torch.ones
    with torch.inference_mode():
        enc_hidden, enc_mask = model.encoder(
            text_hidden_states=text_hidden.to(device),
            text_attention_mask=ones(1, text_hidden.shape[1], device=device),
            lyric_hidden_states=lyric_embeds.to(device),
            lyric_attention_mask=ones(1, lyric_embeds.shape[1], device=device),
            refer_audio_acoustic_hidden_states_packed=silence[:, :TIMBRE_FRAMES, :],
            refer_audio_order_mask=torch.zeros(1, dtype=torch.long, device=device),
        )
    assert enc_mask.all(), "batch=1 pack must be all-valid"
    print(f"[o2] encoder_hidden_states {save_f32('acestep_cond.raw', enc_hidden)} "
          f"(= lyric {lyric_embeds.shape[1]} + timbre 1 + text {text_hidden.shape[1]})")

    if not args.skip_dit:
        # ── oracle 3: one DiT velocity at t=1.0 ──────────────────────────────
        torch.manual_seed(SEED + 1)
        noise = torch.randn(1, FRAMES, 64, device=device, dtype=torch.float32)
        src_latents = silence[:, :FRAMES, :]
        chunk = ones(1, FRAMES, 64, device=device)
        ctx = torch.cat([src_latents, chunk], dim=-1)                      # [1,250,128]
        t1 = torch.full((1,), 1.0, device=device)
        with torch.inference_mode():
            v1, _ = model.decoder(
                hidden_states=noise, timestep=t1, timestep_r=t1,
                attention_mask=ones(1, FRAMES, device=device),
                encoder_hidden_states=enc_hidden, encoder_attention_mask=enc_mask,
                context_latents=ctx, use_cache=False,
            )
        save_f32("acestep_dit_noise.raw", noise)
        save_f32("acestep_dit_v1.raw", v1)
        print(f"[o3] DiT velocity [1,{FRAMES},64] at t=1.0")

        # ── oracle 4: full 8-step euler + DCW from the SAME noise ────────────
        # Mirrors the pipeline's MLX dit_generate loop (the shipping default):
        # euler ODE, final step x0 = x - v*t, DCW haar "double" after each update.
        sched = timestep_schedule(NUM_STEPS, SHIFT)
        xt = noise.clone()
        with torch.inference_mode():
            for i, t_curr in enumerate(sched):
                t_arr = torch.full((1,), t_curr, device=device)
                vt, _ = model.decoder(
                    hidden_states=xt, timestep=t_arr, timestep_r=t_arr,
                    attention_mask=ones(1, FRAMES, device=device),
                    encoder_hidden_states=enc_hidden, encoder_attention_mask=enc_mask,
                    context_latents=ctx, use_cache=False,
                )
                x_before = xt
                if i == NUM_STEPS - 1:
                    xt = xt - vt * t_curr
                else:
                    dt = t_curr - sched[i + 1]
                    xt = xt - vt * dt
                denoised = x_before - vt * t_curr
                xt = apply_dcw_double(xt, denoised, t_curr)
        save_f32("acestep_e2e_latents.raw", xt)
        print(f"[o4] e2e latents [1,{FRAMES},64] ({NUM_STEPS} steps, shift={SHIFT}, DCW double)")
    else:
        print("[o3/o4] skipped (--skip-dit)")

    model.to("cpu")
    del model
    if device == "mps":
        torch.mps.empty_cache()

    # ── oracles 5/6: Oobleck VAE ──────────────────────────────────────────────
    from diffusers import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(
        os.path.join(args.src_main, "vae"), torch_dtype=torch.float32).to(device).eval()

    torch.manual_seed(SEED + 2)
    dec_lat = torch.randn(1, 64, VAEDEC_FRAMES, device=device, dtype=torch.float32)  # [B,C,T]
    with torch.inference_mode():
        wav = vae.decode(dec_lat).sample                                   # [1,2,96000]
    assert wav.shape == (1, 2, VAEDEC_FRAMES * HOP), tuple(wav.shape)
    save_f32("acestep_vaedec_lat.raw", dec_lat)
    save_f32("acestep_vaedec_wav.raw", wav)
    print(f"[o5] VAE decode [1,64,{VAEDEC_FRAMES}] -> [1,2,{VAEDEC_FRAMES*HOP}]")

    # deterministic-ish audio: stereo chirp + tone, avoids degenerate silence
    t = np.arange(VAEDEC_FRAMES * HOP, dtype=np.float32) / 48000.0
    left = 0.5 * np.sin(2 * np.pi * (220.0 + 440.0 * t) * t)
    right = 0.5 * np.sin(2 * np.pi * 330.0 * t)
    audio = torch.from_numpy(np.stack([left, right])[None]).to(device)     # [1,2,96000]
    with torch.inference_mode():
        enc_out = vae.encode(audio).latent_dist
        lat_mean = enc_out.mean                                            # [1,64,50]
    assert lat_mean.shape == (1, 64, VAEDEC_FRAMES), tuple(lat_mean.shape)
    save_f32("acestep_vaeenc_audio.raw", audio)
    save_f32("acestep_vaeenc_mean.raw", lat_mean)
    print(f"[o6] VAE encode mean [1,2,{VAEDEC_FRAMES*HOP}] -> [1,64,{VAEDEC_FRAMES}]")

    # ── env block ─────────────────────────────────────────────────────────────
    o = out_dir
    env = {
        "ACESTEP_TEST_MODEL": os.path.abspath(os.path.expanduser(args.test_model)),
        "ACESTEP_TEXT_IDS": f"{o}/acestep_text_ids.raw",
        "ACESTEP_TEXT_HIDDEN": f"{o}/acestep_text_hidden.raw",
        "ACESTEP_TEXT_LEN": str(text_ids.shape[1]),
        "ACESTEP_LYRIC_IDS": f"{o}/acestep_lyric_ids.raw",
        "ACESTEP_LYRIC_EMBEDS": f"{o}/acestep_lyric_embeds.raw",
        "ACESTEP_LYRIC_LEN": str(lyric_ids.shape[1]),
        "ACESTEP_COND": f"{o}/acestep_cond.raw",
        "ACESTEP_VAEDEC_LAT": f"{o}/acestep_vaedec_lat.raw",
        "ACESTEP_VAEDEC_WAV": f"{o}/acestep_vaedec_wav.raw",
        "ACESTEP_VAEENC_AUDIO": f"{o}/acestep_vaeenc_audio.raw",
        "ACESTEP_VAEENC_MEAN": f"{o}/acestep_vaeenc_mean.raw",
    }
    if not args.skip_dit:
        env.update({
            "ACESTEP_DIT_NOISE": f"{o}/acestep_dit_noise.raw",
            "ACESTEP_DIT_V1": f"{o}/acestep_dit_v1.raw",
            "ACESTEP_E2E_LATENTS": f"{o}/acestep_e2e_latents.raw",
        })
    print("\n# ── paste this to run the Zig oracle tests ──")
    print(" \\\n".join(f"{k}={v}" for k, v in env.items()) + " \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="acestep"')


if __name__ == "__main__":
    main()
