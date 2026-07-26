#!/usr/bin/env python3
"""Repackage hexgrad/Kokoro-82M as an mlx-serve model dir.

USER-RUN (needs torch + safetensors + numpy + huggingface_hub, ~330 MB
download; safetensors serialises through numpy, so it is NOT optional). Not
run in CI. Produces the dir `src/kokoro.zig` loads:

    <out>/config.json            upstream config + "model_type": "kokoro"
    <out>/model.safetensors      f32 trunk, weight-norm FOLDED
    <out>/voices.safetensors     54 voice packs, one tensor each [510, 1, 256]
    <out>/g2p/{us_gold,us_silver,gb_gold}.json   pronunciation dictionaries

Usage:
    python3 tests/convert_kokoro_weights.py --out ~/.mlx-serve/models/hexgrad/Kokoro-82M-mlx-serve

Published as `ddalcu/Kokoro-82M-MLX-Serve` (f32, ~345 MB, `g2p/` included) —
what `AudioModelPreset.kokoro82M` downloads. `--upload` ignores `.DS_Store`;
a hand-rolled `hf upload` must pass `--exclude ".DS_Store"` itself.

WHY EACH TRANSFORM EXISTS
-------------------------

(a) WEIGHT-NORM FOLDING. Every Conv1d/ConvTranspose1d in the text encoder,
    the AdaIN residual blocks and the iSTFTNet generator is wrapped in
    `weight_norm`, which stores the weight factored as magnitude `g` and
    direction `v` (`parametrizations.weight.original0/original1` on modern
    torch, `weight_g`/`weight_v` on old). The engine wants the product, so we
    fold once here: `w = g · v/‖v‖`, norm taken over every axis EXCEPT 0
    (torch's `weight_norm` default `dim=0`, for ConvTranspose1d too — where
    axis 0 is C_IN, not C_out; folding is dim-agnostic so this is safe, but do
    not "fix" the axis on the assumption that axis 0 means output channels).

    Detection is by KEY SHAPE, not a hardcoded module list — a list would
    silently miss a module if upstream adds one, and an unfolded conv produces
    quiet, wrong audio rather than an error.

(b) NO LAYOUT TRANSPOSE. Conv weights stay in PyTorch layout; the engine
    transposes at use through `ltx_audio.zig`'s conv helpers, which is the
    convention that file already established. Do not pre-transpose here or the
    two halves disagree.

(c) f32 THROUGHOUT — and this was MEASURED, not assumed. Do not "optimize" it.

    71% of the weights (57.7M of 81.1M) are 3-D conv tensors, and mlx has NO
    quantized conv kernel (only quantized_matmul / gather_qmm). So quantizing
    them buys a smaller download and NOTHING else: they must be dequantized to
    run, and RAM stays at 325 MB either way.

    Quality, via the KOKORO_FIXTURES oracle on the same prompt and voice:

        variant              durations   f0        asr       audio
        f32 (this script)    EXACT       1.000000  1.000000  0.996831
        8-bit everything     EXACT       0.999998  0.999996  0.973787
        8-bit matmuls only   EXACT       0.999999  0.999997  0.915636
        4-bit everything     EXACT       0.999655  0.998833  0.069956

    The reference's own seed-to-seed spread is 0.9941–0.9960, so even 8-bit is
    measurably outside it. Note how the INTERMEDIATES stay at ~0.99999 while the
    waveform collapses at 4-bit: per-stage cosines say almost nothing here,
    because the vocoder loop compounds small weight errors over hundreds of
    steps. Same trap as the MageFlow bf16 finding.

    Caveat on the metric: 8-bit-matmuls-only scoring WORSE than
    8-bit-everything is not a real quality ordering — waveform cosine is
    phase-sensitive, so it partly measures "phase shifted". At 4-bit (0.07) it
    is unambiguous. In a listening test 8-bit was indistinguishable from f32.

    Verdict: ship f32. 325 MB is already small, RAM does not improve, and the
    generator is the one place precision demonstrably matters. If a download-size
    build is ever wanted, the honest option is an 8-bit STORAGE mirror that
    dequantizes at load (86 MB pull, same RAM, same speed) — not a runtime win.

WHAT IS PRUNED
--------------
  - `bert.pooler.*` — `CustomAlbert.forward` returns `last_hidden_state` and
    never builds the pooled output, so these weights are dead. A wrong prune
    fails LOUDLY at load (the engine errors on a missing key), so this is safe
    to be aggressive about.

PRONUNCIATION DICTIONARIES
--------------------------
Fetched verbatim from hexgrad/misaki (Apache-2.0, same owner and licence as
Kokoro itself) into `<out>/g2p/`. ~9 MB for the three English tables.

This is the one asset with a LICENCE TRAP attached: upstream Kokoro's default
phonemizer path falls back to espeak-ng, which is GPLv3 and would contaminate a
shipped closed app. misaki's dictionaries carry no such term. Never add an
espeak fallback to the Zig side.

Shape: 90k entries per table, values are either a plain IPA string or, for the
~790 heteronyms, an object keyed by part of speech
(`{"DEFAULT": "ˈæbz", "NOUN": null}`) — so the loader needs a POS tagger to
resolve those, and a null means "no pronunciation for this POS, fall through".

VOICE PACKS
-----------
A voice is NOT a single 256-float style vector. Each `.pt` is `[510, 1, 256]`
and the reference indexes it by the phoneme count of the utterance
(`pack[len(ps)-1]`). We keep the whole table per voice so the engine can do the
same; collapsing it to one row is a silent quality regression.
"""

import argparse
import json
import os
import sys

try:
    import numpy  # noqa: F401 - safetensors serialises through it
    import torch
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - user-run script
    sys.exit(
        "needs torch + safetensors + numpy: "
        "pip install torch safetensors numpy huggingface_hub"
    )

REPO_ID = "hexgrad/Kokoro-82M"
CHECKPOINT = "kokoro-v1_0.pth"

# Model card written into the output dir, so `--upload` publishes a repo that
# renders correctly instead of a bare file listing.
#
# `base_model_relation` is LOAD-BEARING: Hugging Face defaults a missing value to
# `finetune`, which would publish this as a finetune of Kokoro-82M and land it in
# the wrong list on hexgrad's page. HF's vocabulary has no "format conversion"
# value, so `quantized` is the closest non-training relation — the body says
# plainly that the weights are f32 and unquantized, because the frontmatter
# cannot.
README = """---
license: apache-2.0
base_model: hexgrad/Kokoro-82M
base_model_relation: quantized
library_name: mlx-serve
tags:
  - mlx
  - mlx-serve
  - tts
  - text-to-speech
  - kokoro
pipeline_tag: text-to-speech
---

# Kokoro-82M for mlx-serve

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) repacked for
[mlx-serve](https://github.com/ddalcu/mlx-serve)'s native Zig + MLX engine.

Same weights, same precision — **f32, not quantized**. The repack folds
weight-norm into `.weight` (so the engine has no `weight_g`/`weight_v` to
resolve), drops the unused pooler tensors, converts `.pth` to safetensors, and
bundles the 54 voice packs plus the English pronunciation dictionaries the
phonemizer needs.

| File | Contents |
|---|---|
| `model.safetensors` | 457 tensors, 81.1M params, 325 MB f32 |
| `voices.safetensors` | 54 voice packs, `[510, 1, 256]` each |
| `g2p/*.json` | misaki `us_gold` / `us_silver` / `gb_gold` |

Verified against the torch reference: per-phoneme durations match **exactly**,
F0 / noise / text-encoder outputs at cosine 1.000000, and the waveform at 0.9968
— which is inside the reference's own seed-to-seed spread (0.9941–0.9960), since
its vocoder is stochastic.

About **17x realtime** on an M-series Mac, ~350 MB resident.

## Run it

Download **[MLX Core.app](https://github.com/ddalcu/mlx-serve/releases/latest)**,
open Settings ▸ Voice, and pick **Kokoro** as the voice engine. 54 voices, and
naming several separated by commas blends them into a new one.

Over HTTP:

```bash
curl -X POST http://localhost:11234/v1/audio/speech \\
  -H 'content-type: application/json' \\
  -d '{"model":"kokoro","input":"Hello there.","voice":"af_bella,af_sky"}' \\
  --output out.wav
```

## Credit

Kokoro-82M and the misaki G2P dictionaries are both by
[hexgrad](https://huggingface.co/hexgrad), Apache-2.0. This repo only changes
the packaging. No espeak-ng anywhere in the pipeline — the dictionaries make it
unnecessary, which keeps the whole path Apache-2.0.
"""

MISAKI_DATA = "https://raw.githubusercontent.com/hexgrad/misaki/main/misaki/data"
DICTS = ["us_gold.json", "us_silver.json", "gb_gold.json"]

# The 54 published voices. Prefix = language (a=American, b=British, e=Spanish,
# f=French, h=Hindi, i=Italian, j=Japanese, p=Portuguese, z=Chinese), second
# letter = gender.
VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]


def fold_weight_norm(state):
    """Replace every (g, v) parametrization pair with the folded weight.

    Handles both spellings:
      modern: `<p>.parametrizations.weight.original0` / `.original1`
      legacy: `<p>.weight_g` / `<p>.weight_v`
    """
    out = {}
    consumed = set()
    folded = 0

    for key in state:
        if key.endswith(".parametrizations.weight.original0"):
            prefix = key[: -len(".parametrizations.weight.original0")]
            g_key = key
            v_key = prefix + ".parametrizations.weight.original1"
            target = prefix + ".weight"
        elif key.endswith(".weight_g"):
            prefix = key[: -len(".weight_g")]
            g_key = key
            v_key = prefix + ".weight_v"
            target = prefix + ".weight"
        else:
            continue

        if v_key not in state:
            sys.exit(f"weight-norm pair broken: {g_key} has no matching {v_key}")

        g, v = state[g_key].float(), state[v_key].float()
        # norm over every axis except 0 (torch weight_norm default dim=0)
        dims = tuple(range(1, v.dim()))
        norm = v.norm(2, dim=dims, keepdim=True)
        out[target] = g * v / norm
        consumed.update({g_key, v_key})
        folded += 1

    for key, val in state.items():
        if key in consumed:
            continue
        if key in out:
            sys.exit(f"folded weight {key} collides with a stored tensor")
        out[key] = val.float()

    print(f"  folded {folded} weight-norm pairs")
    return out


def prune(state):
    """Drop tensors nothing in the engine reads."""
    dropped = [k for k in state if k.startswith("bert.pooler.")]
    for k in dropped:
        del state[k]
    if dropped:
        print(f"  pruned {len(dropped)} dead pooler tensors")
    return state


def convert_trunk(checkpoint_path):
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # The checkpoint is a dict of per-module state dicts, not one flat dict.
    flat = {}
    for module, sd in raw.items():
        for k, v in sd.items():
            # Some releases carry a stale "module." DataParallel prefix.
            k = k[7:] if k.startswith("module.") else k
            flat[f"{module}.{k}"] = v
    print(f"  {len(flat)} tensors across {len(raw)} modules")
    return prune(fold_weight_norm(flat))


def convert_voices(fetch):
    packs = {}
    missing = []
    for name in VOICES:
        try:
            path = fetch(f"voices/{name}.pt")
        except Exception as e:  # noqa: BLE001 - report and continue
            missing.append(f"{name}: {e}")
            continue
        t = torch.load(path, map_location="cpu", weights_only=True).float()
        if t.dim() != 3 or t.shape[1] != 1 or t.shape[2] != 256:
            sys.exit(f"voice {name} has unexpected shape {tuple(t.shape)}; expected [N,1,256]")
        packs[name] = t.contiguous()
    if missing:
        print(f"  WARNING: {len(missing)} voices unavailable:")
        for m in missing:
            print(f"    {m}")
    print(f"  {len(packs)} voice packs, shape {tuple(next(iter(packs.values())).shape)}")
    return packs


def convert_dicts(out_dir):
    """Copy the misaki pronunciation tables into `<out>/g2p/`."""
    import urllib.request

    g2p_dir = os.path.join(out_dir, "g2p")
    os.makedirs(g2p_dir, exist_ok=True)
    for name in DICTS:
        dest = os.path.join(g2p_dir, name)
        with urllib.request.urlopen(f"{MISAKI_DATA}/{name}") as r:
            body = r.read()
        # Parse before writing: a truncated download is otherwise a load-time
        # failure much later, with nothing pointing back at this step.
        table = json.loads(body)
        n_het = sum(1 for v in table.values() if isinstance(v, dict))
        with open(dest, "wb") as f:
            f.write(body)
        print(f"  {name}: {len(table)} entries ({n_het} POS-dependent)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output model dir")
    ap.add_argument("--src", help="local Kokoro-82M repo dir (default: download from HF)")
    ap.add_argument("--upload", metavar="REPO_ID",
                    help="after converting, publish to this HF repo (e.g. ddalcu/Kokoro-82M-MLX-Serve). "
                         "Needs `huggingface-cli login`.")
    args = ap.parse_args()

    if args.src:
        def fetch(rel):
            p = os.path.join(args.src, rel)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            return p
    else:
        from huggingface_hub import hf_hub_download

        def fetch(rel):
            return hf_hub_download(repo_id=REPO_ID, filename=rel)

    os.makedirs(args.out, exist_ok=True)

    print("config…")
    with open(fetch("config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Discovery and gen.peekModelType dispatch on this; upstream has no
    # model_type at all, so we add it rather than special-casing the shape.
    cfg["model_type"] = "kokoro"
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("trunk…")
    trunk = convert_trunk(fetch(CHECKPOINT))
    save_file(trunk, os.path.join(args.out, "model.safetensors"))

    print("voices…")
    packs = convert_voices(fetch)
    save_file(packs, os.path.join(args.out, "voices.safetensors"))

    print("dictionaries…")
    convert_dicts(args.out)

    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)
    print("  README.md")

    total = sum(t.numel() for t in trunk.values())
    print(f"\nwrote {args.out}")
    print(f"  {len(trunk)} tensors, {total/1e6:.1f}M params, {total*4/1e6:.0f} MB f32")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload, repo_type="model", exist_ok=True)
        # Finder writes .DS_Store into any dir you open, and upload_folder takes
        # the tree verbatim — the first publish of this repo shipped one.
        api.upload_folder(folder_path=args.out, repo_id=args.upload, repo_type="model",
                          ignore_patterns=[".DS_Store", "**/.DS_Store"])
        print(f"published https://huggingface.co/{args.upload}")
        print("Remember to point AudioModelPreset.kokoro82M.repo at it. It stays OUT "
              "of AudioModelPreset.all (voice mode only — the media panes send "
              "ref_audio, which this backend 400s); the browser lists it via "
              "allIncludingVoiceOnly.")


if __name__ == "__main__":
    main()
