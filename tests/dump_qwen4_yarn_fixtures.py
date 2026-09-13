#!/usr/bin/env python3
"""Dump the YaRN oracle for qwen4_exp (Qwen3.8-Flash-Next) context extension.

Answers one question with numbers the Zig engine is checked against: what does
the REFERENCE produce for this checkpoint's rope geometry when its context is
scaled past the trained window? Nothing here is mlx-serve's own math — the
frequencies come out of `transformers.modeling_rope_utils` (the same library
that wrote the checkpoint's config.json), and every value is cross-checked
against a transcription of vLLM's `YaRNScalingRotaryEmbedding` before the
fixture is written, so the two references have to agree with each other first.

Why the geometry is interesting: qwen4_exp rotates only 64 of its 256 head dims
(`partial_rotary_factor` 0.25) at theta 1e7, so the 262144-token window lives in
just 32 frequencies and `mrope_section` [11,11,10] splits exactly those 32
across the (t,h,w) axes. YaRN's ramp therefore lands at dims [14, 22] and the
extension is a 64-of-256-dim surgery, not a whole-head one.

HF `attention_factor` REPLACES the computed 0.1·ln(factor)+1; vLLM `attn_factor`
MULTIPLIES it. Default dump `--attn_factor 1.0` so the two references (and the
Zig host-spectrum oracle) stay on the computed mscale.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# Positions the cos/sin tables are dumped at: inside the trained window, at its
# exact edge, one past it (the length a naive --ctx-size would already corrupt),
# and through the extended range up to the last position of a 1M window.
PROBE_POSITIONS = [
    0, 1, 2, 3, 31, 32, 33, 1000, 8192, 65536,
    262143, 262144, 262145, 300000, 524288, 786432, 1048575,
]


def vllm_yarn_inv_freq(head_size, rotary_dim, orig_max, base, factor,
                       beta_fast=32, beta_slow=1, extrapolation_factor=1.0,
                       attn_factor=1.0, truncate=True):
    """Transcription of vLLM's YaRNScalingRotaryEmbedding._compute_inv_freq /
    _compute_cos_sin_cache (vllm/model_executor/layers/rotary_embedding/
    yarn_scaling_rope.py + common.py), in float64. Kept here so the fixture
    proves HF and vLLM agree before mlx-serve is compared to either."""

    def find_correction_dim(num_rotations, dim, b, max_pos):
        return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(b))

    def find_correction_range(low_rot, high_rot, dim, b, max_pos, trunc):
        low = find_correction_dim(low_rot, dim, b, max_pos)
        high = find_correction_dim(high_rot, dim, b, max_pos)
        if trunc:
            low, high = math.floor(low), math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_mask(low, high, dim):
        if low == high:
            high += 0.001
        return np.clip((np.arange(dim, dtype=np.float64) - low) / (high - low), 0, 1)

    def get_mscale(scale=1):
        return 1.0 if scale <= 1 else 0.1 * math.log(scale) + 1.0

    pos_freqs = np.power(base, np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)
    low, high = find_correction_range(beta_fast, beta_slow, rotary_dim, base, orig_max, truncate)
    mask = (1 - linear_ramp_mask(low, high, rotary_dim // 2)) * extrapolation_factor
    inv_freq = inv_interp * (1 - mask) + inv_extrap * mask
    mscale = float(get_mscale(factor) * attn_factor)
    return inv_freq, mscale, low, high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser(
        "~/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit/config.json"))
    ap.add_argument("--factor", type=float, default=4.0,
                    help="YaRN scaling factor (4.0 turns 262144 into 1048576)")
    ap.add_argument("--attn_factor", type=float, default=1.0,
                    help="vLLM attn_factor (multiplies computed mscale; 1.0 keeps the Zig oracle valid)")
    ap.add_argument("--out", default="/tmp/qwen4_yarn.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    tc = cfg.get("text_config", cfg)
    theta = float(tc["rope_theta"] if "rope_theta" in tc
                  else tc["rope_parameters"]["rope_theta"])
    partial = float(tc["rope_parameters"]["partial_rotary_factor"])
    head_dim = int(tc["head_dim"])
    # If the config ALREADY declares YaRN, its `original_max_position_embeddings`
    # is the trained window — read it from there so re-running this on an
    # extended pack describes the same extension instead of compounding it.
    orig_max = int(tc["rope_parameters"].get("original_max_position_embeddings")
                   or tc["max_position_embeddings"])
    mrope_section = list(tc["rope_parameters"]["mrope_section"])
    rotary_dim = int(head_dim * partial)

    # ---- the reference: HF's own YaRN, on a config that looks like the
    # checkpoint's with the vLLM recipe applied -------------------------------
    rp = {
        "rope_type": "yarn",
        "rope_theta": theta,
        "partial_rotary_factor": partial,
        "factor": args.factor,
        "original_max_position_embeddings": orig_max,
        # carried through because the checkpoint ships them; HF is told to
        # ignore them for validation, exactly as vLLM's config layer does.
        "mrope_section": mrope_section,
        "mrope_interleaved": bool(tc["rope_parameters"].get("mrope_interleaved", False)),
    }
    conf = PretrainedConfig(
        model_type=tc.get("model_type", "qwen4_exp_text"),
        head_dim=head_dim,
        hidden_size=int(tc["hidden_size"]),
        num_attention_heads=int(tc["num_attention_heads"]),
        max_position_embeddings=int(round(orig_max * args.factor)),
        rope_parameters=rp,
        partial_rotary_factor=partial,
    )
    conf.ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}
    conf.standardize_rope_params()
    hf_inv_freq, hf_attention_factor = ROPE_INIT_FUNCTIONS["yarn"](conf)
    hf_inv_freq = hf_inv_freq.double().numpy()  # HF computes in f32; widen for the dump

    # ---- cross-check: vLLM's transcription of the same method ---------------
    vl_inv_freq, vl_mscale, low, high = vllm_yarn_inv_freq(
        head_dim, rotary_dim, orig_max, theta, args.factor,
        attn_factor=args.attn_factor)
    rel = np.max(np.abs(hf_inv_freq - vl_inv_freq) / np.maximum(np.abs(vl_inv_freq), 1e-30))
    print(f"rotary_dim={rotary_dim} ({rotary_dim // 2} freqs)  ramp=[{low}, {high}]  "
          f"attention_factor: HF={hf_attention_factor!r} vLLM={vl_mscale!r}")
    print(f"HF vs vLLM inv_freq max relative difference: {rel:.3e}")
    assert rel < 1e-6, f"the two references disagree ({rel})"
    assert abs(hf_attention_factor - vl_mscale) < 1e-9, "mscale disagreement"

    # ---- the tables the engine reproduces ----------------------------------
    # HF's construction (Qwen3-Next-style rotary): emb = cat(freqs, freqs) with
    # cos/sin multiplied by attention_factor. For TEXT positions t==h==w, so the
    # interleaved 3-D selection collapses onto one scalar position — the exact
    # property mlx-serve's mrope.zig relies on.
    #
    # The rows are built from the float64 spectrum on purpose. HF and vLLM both
    # compute inv_freq in float32 and then multiply it by the position, so at a
    # position near 1M an f32 frequency's ~6e-8 relative error is already ~0.04
    # radians of ANGLE error — comparing an f64 engine against an f32 cache at
    # that depth measures the reference's rounding, not the port. The f32-vs-f64
    # spread is printed below so the size of that effect is on the record.
    rows = {}
    rows_f32_skew = 0.0
    for p in PROBE_POSITIONS:
        ang = vl_inv_freq * float(p)
        cos = np.cos(ang).astype(np.float32)
        sin = np.sin(ang).astype(np.float32)
        rows[str(p)] = {
            "cos": cos.tolist(),
            "sin": sin.tolist(),
            "cos_scaled": (cos * np.float32(hf_attention_factor)).tolist(),
            "sin_scaled": (sin * np.float32(hf_attention_factor)).tolist(),
        }
        ang32 = hf_inv_freq * float(p)
        skew = max(float(np.max(np.abs(np.cos(ang32) - cos))), float(np.max(np.abs(np.sin(ang32) - sin))))
        rows_f32_skew = max(rows_f32_skew, skew)

    plain = np.power(theta, -np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    out = {
        "source": "transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS['yarn'] "
                  f"(transformers {__import__('transformers').__version__}), "
                  "cross-checked against vLLM's YaRNScalingRotaryEmbedding",
        "config_path": args.config,
        "head_dim": head_dim,
        "partial_rotary_factor": partial,
        "rotary_dim": rotary_dim,
        "rope_theta": theta,
        "factor": args.factor,
        "original_max_position_embeddings": orig_max,
        "extended_max_position_embeddings": int(round(orig_max * args.factor)),
        "mrope_section": mrope_section,
        "mrope_interleaved": bool(tc["rope_parameters"].get("mrope_interleaved", False)),
        "attention_factor": float(hf_attention_factor),
        "beta_fast": 32,
        "beta_slow": 1,
        "truncate": True,
        "correction_low": low,
        "correction_high": high,
        "inv_freq": hf_inv_freq.tolist(),
        "plain_inv_freq": plain.tolist(),
        "vllm_inv_freq": vl_inv_freq.tolist(),
        "rows": rows,
        # The widest |cos|/|sin| gap between HF's float32 cache and the float64
        # spectrum at the dumped positions: how much of the reference is rounding.
        "hf_f32_row_skew_max": rows_f32_skew,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: {len(hf_inv_freq)} freqs, {len(rows)} position rows")

    # A plain statement of what the extension buys, for the run log.
    beyond = [p for p in PROBE_POSITIONS if p > orig_max]
    print(f"positions dumped past the trained {orig_max}: {beyond}")
    print(f"max |row(f32 freqs) - row(f64 freqs)| over those positions: {rows_f32_skew:.2e}"
          "  <- HF/vLLM float32 cache error, grows with position")
    print(f"interpolated band (i>={high}) is plain/factor exactly: "
          f"{np.max(np.abs(vl_inv_freq[high:] / plain[high:] - 1.0 / args.factor)):.2e}")
    print(f"extrapolated band (i<={low}) is plain exactly:         "
          f"{np.max(np.abs(vl_inv_freq[: low + 1] - plain[: low + 1])):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
