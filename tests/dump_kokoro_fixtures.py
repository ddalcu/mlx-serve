#!/usr/bin/env python3
"""Dump reference Kokoro-82M activations for the `KOKORO_*` parity oracles.

USER-RUN (torch + transformers + the upstream hexgrad/kokoro source). Writes raw
little-endian f32/i32 blobs that `src/kokoro.zig`'s env-gated tests read back.

    python3 tests/dump_kokoro_fixtures.py --kokoro-src <path to hexgrad/kokoro> \\
        --out /tmp/kokoro-fixtures

    KOKORO_TEST_MODEL=~/.mlx-serve/models/hexgrad/Kokoro-82M-mlx-serve \\
    KOKORO_FIXTURES=/tmp/kokoro-fixtures \\
    zig build test -Doptimize=ReleaseFast -Dtest-filter=kokoro

WHAT IS COMPARED, AND WHY IT IS SPLIT THIS WAY
----------------------------------------------
The generator is STOCHASTIC — SineGen draws a random initial phase per harmonic
and adds Gaussian noise — so a bit-exact waveform comparison is impossible
across two different PRNGs. Everything BEFORE the generator is fully
deterministic, so the oracle is split:

  durations  (i32)  EXACT equality. Covers ALBERT, bert_encoder, the whole
                    DurationEncoder (3 BiLSTM + 3 AdaLayerNorm), the duration
                    BiLSTM and duration_proj. If these match to the frame, the
                    entire prosody trunk is right — this is the strongest
                    single check in the file.
  f0 / n     (f32)  cosine. Covers the shared BiLSTM and both AdainResBlk1d
                    stacks including the ×2 upsample and the AdaIN path.
  asr        (f32)  cosine. Covers TextEncoder (conv stack + LayerNorm(gamma,
                    beta) + BiLSTM) and the duration-driven frame expansion.
  audio      (f32)  cosine, LOOSE. Only meaningful as "the generator is not
                    producing garbage"; the phase noise genuinely decorrelates
                    fine structure between runs.

The reference is imported from the upstream source tree rather than the `kokoro`
pip package: that package pulls misaki → spacy, which does not build here, and
none of it is needed to drive `KModel` from a phoneme string.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def write(out_dir, name, arr, dtype):
    p = os.path.join(out_dir, name)
    np.asarray(arr, dtype=dtype).tofile(p)
    print(f"  {name}: {np.asarray(arr).shape} {dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kokoro-src", required=True, help="checkout of hexgrad/kokoro")
    ap.add_argument("--repo", help="local Kokoro-82M repo (default: download)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phonemes", default="həlˈoʊ wˈɜːld")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    # Import `kokoro.model` WITHOUT executing `kokoro/__init__.py`, which pulls
    # in KPipeline → misaki → spacy (does not build here, and none of it is
    # needed to drive KModel from a phoneme string). Pre-registering the package
    # with a __path__ makes the submodule import resolve on its own.
    import types

    sys.path.insert(0, args.kokoro_src)
    pkg = types.ModuleType("kokoro")
    pkg.__path__ = [os.path.join(args.kokoro_src, "kokoro")]
    sys.modules["kokoro"] = pkg
    from kokoro.model import KModel  # noqa: E402

    if args.repo:
        cfg = os.path.join(args.repo, "config.json")
        ckpt = os.path.join(args.repo, "kokoro-v1_0.pth")
        voice_path = os.path.join(args.repo, "voices", f"{args.voice}.pt")
    else:
        from huggingface_hub import hf_hub_download

        cfg = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
        ckpt = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
        voice_path = hf_hub_download("hexgrad/Kokoro-82M", f"voices/{args.voice}.pt")

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    model = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg, model=ckpt).eval()
    pack = torch.load(voice_path, weights_only=True)

    ps = args.phonemes
    ids = [i for i in (model.vocab.get(p) for p in ps) if i is not None]
    input_ids = torch.LongTensor([[0, *ids, 0]])
    ref_s = pack[len(ps) - 1]
    print(f"phonemes={ps!r} -> {len(ids)} ids, pack row {len(ps)-1}")

    # Mirror forward_with_tokens, capturing each stage.
    with torch.no_grad():
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s_pred, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / args.speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s_pred)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()

    print("writing fixtures:")
    write(args.out, "durations.i32", pred_dur.numpy(), np.int32)
    write(args.out, "f0.f32", F0_pred.squeeze().numpy(), np.float32)
    write(args.out, "n.f32", N_pred.squeeze().numpy(), np.float32)
    # asr is [1, 512, F] channel-first; the engine holds NLC, so transpose here
    # and keep the engine free of a fixture-shaped special case.
    write(args.out, "asr.f32", asr.squeeze(0).transpose(0, 1).contiguous().numpy(), np.float32)
    write(args.out, "audio.f32", audio.numpy(), np.float32)

    meta = {
        "phonemes": ps,
        "voice": args.voice,
        "speed": args.speed,
        "n_tokens": int(input_ids.shape[1]),
        "n_frames": int(pred_dur.sum()),
        "audio_samples": int(audio.shape[0]),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
