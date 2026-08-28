#!/usr/bin/env python3
"""Convert circlestone-labs/Anima ComfyUI split_files into the flat layout
mlx-serve's `src/anima.zig` `Engine` loads.

USER-RUN. Pure stdlib (no torch/mlx/safetensors) — every source file is
already a plain safetensors file with the exact key names `src/anima.zig`
reads (`model.diffusion_model.*` for the DiT+adapter, `model.layers.*` for
the Qwen3-0.6B TE, `decoder.*`/`conv2.*` for the VAE), so conversion is a
straight file copy plus a small `config.json`. The engine loader upcasts
every weight to float32 at load time regardless of on-disk dtype (see
`anima.zig`'s module doc on why f32 end-to-end, for now), so no numeric
transform is needed here either. Quantization is a documented follow-up
(mirrors every other convert_*_weights.py's "DELIBERATELY NOT DONE HERE").

Anima ships several variants with the SAME architecture but different
recommended sampling (comfy `supported_models.Anima.sampling_settings` is
fixed at shift=3.0 for all of them; steps/cfg are a per-checkpoint authoring
choice, not read from the checkpoint itself — see notes/anima-implementation.md):

    base / aesthetic / aesthetic-v1.0b / aesthetic-v1.1 / preview*   30-50 steps, CFG 4-5
    turbo / turbo-v1.1                                                8-12 steps, CFG 1 (skips
                                                                       the uncond forward entirely)

Produces:

    <out>/config.json               {"model_type":"anima","recommended_steps":N,"recommended_cfg":F}
    <out>/text_encoder.safetensors  Qwen3-0.6B (verbatim)
    <out>/tokenizer/                Qwen BPE tokenizer (vocab.json + merges.txt, verbatim)
    <out>/vae.safetensors           Qwen-Image VAE (verbatim — ships the encoder weights too,
                                     unused until img2img/edit lands; harmless, just extra bytes)
    <out>/t5_tokenizer/             T5 SentencePiece unigram vocab (tokenizer.json, verbatim)
    <out>/transformer.safetensors   DiT + llm_adapter (verbatim; the required-media marker,
                                     so it is written LAST — see model_discovery.requiredMediaMarker)

Usage:
    python3 scripts/convert_anima_weights.py \\
        --dit anima-turbo-v1.1.safetensors \\
        --te qwen_3_06b_base.safetensors \\
        --vae qwen_image_vae.safetensors \\
        --qwen-tokenizer-dir path/to/comfy/text_encoders/qwen25_tokenizer \\
        --t5-tokenizer-dir path/to/comfy/text_encoders/t5_tokenizer \\
        --variant turbo \\
        <out_dir>

    python3 scripts/convert_anima_weights.py --self-test   # no checkpoint needed
"""

import argparse
import json
import os
import shutil
import sys

VARIANT_DEFAULTS = {
    "base": (32, 4.5),
    "aesthetic": (32, 4.5),
    "preview": (32, 4.5),
    "turbo": (10, 1.0),
}


def variant_defaults(variant: str) -> tuple[int, float]:
    for prefix, defaults in VARIANT_DEFAULTS.items():
        if variant.startswith(prefix):
            return defaults
    raise ValueError(
        f"unknown variant {variant!r}; expected one starting with "
        f"{sorted(VARIANT_DEFAULTS)} (e.g. 'turbo-v1.1', 'aesthetic-v1.0b')"
    )


def convert(args: argparse.Namespace) -> None:
    steps, cfg = variant_defaults(args.variant)
    if args.steps is not None:
        steps = args.steps
    if args.cfg is not None:
        cfg = args.cfg

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    config = {
        "model_type": "anima",
        "variant": args.variant,
        "recommended_steps": steps,
        "recommended_cfg": cfg,
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"wrote {out}/config.json ({config})")

    print(f"copying text encoder -> {out}/text_encoder.safetensors")
    shutil.copyfile(args.te, os.path.join(out, "text_encoder.safetensors"))

    tok_out = os.path.join(out, "tokenizer")
    os.makedirs(tok_out, exist_ok=True)
    for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        src = os.path.join(args.qwen_tokenizer_dir, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(tok_out, name))
            print(f"copied {name} -> {tok_out}/")
        elif name != "tokenizer_config.json":
            raise FileNotFoundError(f"missing required Qwen tokenizer file: {src}")

    print(f"copying VAE -> {out}/vae.safetensors")
    shutil.copyfile(args.vae, os.path.join(out, "vae.safetensors"))

    t5_out = os.path.join(out, "t5_tokenizer")
    os.makedirs(t5_out, exist_ok=True)
    t5_src = os.path.join(args.t5_tokenizer_dir, "tokenizer.json")
    shutil.copyfile(t5_src, os.path.join(t5_out, "tokenizer.json"))
    print(f"copied tokenizer.json -> {t5_out}/")

    # Written LAST: model_discovery.requiredMediaMarker("anima") gates
    # discovery/completeness on this file's presence.
    print(f"copying DiT+adapter -> {out}/transformer.safetensors")
    shutil.copyfile(args.dit, os.path.join(out, "transformer.safetensors"))

    print(f"done: {out} ({args.variant}, {steps} steps, cfg {cfg})")


def self_test() -> None:
    assert variant_defaults("turbo-v1.1") == (10, 1.0)
    assert variant_defaults("turbo") == (10, 1.0)
    assert variant_defaults("aesthetic-v1.0b") == (32, 4.5)
    assert variant_defaults("base-v1.0") == (32, 4.5)
    assert variant_defaults("preview3-base") == (32, 4.5)
    try:
        variant_defaults("nonsense")
        raise AssertionError("expected ValueError for an unknown variant")
    except ValueError:
        pass
    print("self-test OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dit", help="path to anima-<variant>.safetensors (DiT + llm_adapter)")
    p.add_argument("--te", help="path to qwen_3_06b_base.safetensors")
    p.add_argument("--vae", help="path to qwen_image_vae.safetensors")
    p.add_argument("--qwen-tokenizer-dir", help="comfy/text_encoders/qwen25_tokenizer")
    p.add_argument("--t5-tokenizer-dir", help="comfy/text_encoders/t5_tokenizer")
    p.add_argument("--variant", help="e.g. 'turbo-v1.1', 'aesthetic-v1.0b', 'base-v1.0'")
    p.add_argument("--steps", type=int, default=None, help="override the variant's recommended steps")
    p.add_argument("--cfg", type=float, default=None, help="override the variant's recommended cfg")
    p.add_argument("out_dir", nargs="?")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["dit", "te", "vae", "qwen_tokenizer_dir", "t5_tokenizer_dir", "variant", "out_dir"]
    missing = [f"--{r.replace('_', '-')}" if r != "out_dir" else "out_dir" for r in required if getattr(args, r) is None]
    if missing:
        p.error(f"missing required arguments: {', '.join(missing)}")

    convert(args)


if __name__ == "__main__":
    sys.exit(main())
