#!/usr/bin/env python3
"""Convert / repack a Krea-2-Turbo checkpoint into an mlx-serve pack.

    python tests/convert_krea.py \
        --src ~/krea2-turbo-bf16 \
        --out ~/.mlx-serve/models/<you>/Krea-2-Turbo-MLX-Serve-mixed_3_8 \
        --precision mixed_3_8

Krea is the one image backend that never had a converter: the packs were
ASSEMBLED by hand from an already-quantized upstream transformer plus a
Qwen3-VL-4B encoder, a Qwen-Image VAE, a tokenizer and a synthesized
`config.json`. This script is that assembly written down, with the
quantization step added.

`--precision` (a mixed name reads bulk_sensitive):

  4         every quantizable projection at 4-bit. What the shipped pack runs.
  8         every quantizable projection at 8-bit. The reference point.
  mixed     bulk 4-bit, the attention out-GATE 8-bit, encoder 8-bit.
  mixed_3_8 bulk 3-bit, gate 8-bit, encoder 4-bit. The small one.

UNVERIFIED BELOW 4 BITS. Ideogram 4's DiT renders correctly at 3-bit and turns
into a woven grid at 2 — but that is a 9.3B model, and Krea-2-Turbo is both
smaller and DISTILLED, which historically means less quantization headroom
(MageFlow Turbo will not even run in f32). `mixed_3_8` is offered because the
engine loads it, not because anyone has looked at its output. Render a prompt
before publishing one; a structured artifact at every seed is the tell, and the
check that settles it in a minute is the same prompt on a pack at another width
(`docs/gotchas/models-media.md`).

`--bulk-bits` / `--sensitive-bits` / `--te-bits` override any tier of a mixed
policy. 2-bit on the bulk is refused outright — see `MIN_BULK_BITS`.

WHAT STAYS DENSE, AND WHY IT IS NOT A JUDGEMENT CALL
----------------------------------------------------
`src/krea.zig` loads most of the DiT through `MixedLinear`, which reads a
`.scales` sibling and infers the width. But three families are loaded by
`normBf16` / `kreaNorm`, which call `ownWeight` and cast — no quantized path at
all. Quantizing one of those does not produce a worse image, it produces a pack
that loads packed u32 as if it were bf16. They are listed in `ENGINE_DENSE` and
the self-test pins them.

The rest of the dense set is a size decision the shipped pack already made: of
the DiT's linears only the 28 transformer blocks are quantized (224 tensors,
7.6 GB of a 12 GB pack). `first`, `tmlp.*`, `tproj.*`, `txtfusion.*`, `txtmlp.*`
and `last.linear` together are a rounding error, and leaving them dense removes
them as suspects — the same reasoning as Ideogram's `DENSE_MODULES`.

Two of them could not be quantized anyway: `first` takes 64 input features and
`txtfusion.projector` takes 12, and a group of 64 does not fit in either.

THE SOURCE MAY ALREADY BE QUANTIZED
-----------------------------------
`--src` may be a dense bf16/f32 checkpoint or an existing mlx-serve pack. A
quantized source is dequantized first and the result is a REPACK, which is
lossy twice over — fine for trying a width out locally, not fine for something
you publish. It warns, loudly, and `--allow-requantize` is required.

Dequantizing needs `(bits, group_size)`, and those are NOT recoverable from the
tensor shapes: `[out, in*bits/32]` and `[out, in/gs]` are satisfied by (8, 32)
and (4, 64) alike. This is exactly why `MixedLinear.load` takes `in_features`
as a parameter instead of solving for it, and why `IN_FEATURES` below mirrors
`krea.KreaConfig` rather than guessing.

`config.json` is written LAST. Discovery classifies a media dir by its root
`model_type`, so an interrupted conversion stays invisible to `list` and to the
loader instead of half-registering. (Krea has no `requiredMediaMarker` entry;
writing the file that IS the marker last gets the same property for free.)
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

# ── the checkpoint's geometry ─────────────────────────────────────────────
#
# Mirrors `krea.KreaConfig`, which is a struct of literal defaults — the engine
# does NOT derive these from the checkpoint (`loadDit` opens with `d.cfg = .{}`),
# so a Krea DiT that differs in any of them is a different model and this script
# is not it. `--self-test` re-derives every in_features from these.

FEATURES = 6144
TDIM = 256
TXTDIM = 2560
TXTHEADS = 20
MULTIPLIER = 4
LAYERS = 28
PATCH = 2
CHANNELS = 16
TXTLAYERS = 12

# Text encoder (Qwen3-VL-4B).
TE_HIDDEN = 2560
TE_LAYERS = 36
TE_HEADS = 32
TE_HEAD_DIM = 128
TE_INTER = 9728


def swiglu_dim(features: int, multiplier: int) -> int:
    """`krea.swigluDim`: roundup((2*features/3)*multiplier, 128), integer div."""
    base = (2 * features // 3) * multiplier
    return ((base + 127) // 128) * 128


MLP_DIM = swiglu_dim(FEATURES, MULTIPLIER)        # 16384
TXT_MLP_DIM = swiglu_dim(TXTDIM, MULTIPLIER)      # 6912


def in_features_for(module: str) -> int | None:
    """Input width of a DiT/encoder linear, or None if the name is not one.

    The single source of truth for both dequantizing a quantized source and
    deciding what can be quantized at all. Every arm mirrors the corresponding
    `MixedLinear.load(...)` call site in `src/krea.zig`.
    """
    # ── DiT: the 28 transformer blocks (the bulk) ──
    if module.startswith("blocks."):
        rest = module.split(".", 2)[2] if module.count(".") >= 2 else ""
        if rest in ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate"):
            return FEATURES
        if rest in ("mlp.gate", "mlp.up"):
            return FEATURES
        if rest == "mlp.down":
            return MLP_DIM
        return None
    # ── DiT: text fusion (4 blocks: 2 layerwise + 2 refiner) ──
    if module.startswith("txtfusion.layerwise_blocks.") or module.startswith(
        "txtfusion.refiner_blocks."
    ):
        rest = module.split(".", 3)[3] if module.count(".") >= 3 else ""
        if rest in ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate"):
            return TXTDIM
        if rest in ("mlp.gate", "mlp.up"):
            return TXTDIM
        if rest == "mlp.down":
            return TXT_MLP_DIM
        return None
    # ── DiT: the small stuff ──
    fixed = {
        "first": CHANNELS * PATCH * PATCH,   # 64
        "tmlp.0": TDIM,
        "tmlp.2": FEATURES,
        "tproj.1": FEATURES,
        "txtfusion.projector": TXTLAYERS,    # 12
        "txtmlp.1": TXTDIM,
        "txtmlp.3": FEATURES,
        "last.linear": FEATURES,
    }
    if module in fixed:
        return fixed[module]
    # ── Text encoder (Qwen3-VL-4B), `loadTextEncoder`'s call sites ──
    if ".layers." in module:
        leaf = module.rsplit(".", 1)[-1]
        if leaf in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            return TE_HIDDEN
        if leaf == "o_proj":
            return TE_HEADS * TE_HEAD_DIM    # 4096
        if leaf == "down_proj":
            return TE_INTER
    return None


# ── what the ENGINE cannot read quantized ─────────────────────────────────
#
# `normBf16` / `kreaNorm` call `ownWeight` and cast. There is no `.scales`
# branch, so a quantized tensor here is read as bf16 garbage — a pack that
# loads and renders noise, which is the failure this repo has now paid for
# twice.
#
# This is belt-and-braces, and it is worth being precise about which brace is
# load-bearing today: `in_features_for` is an ALLOWLIST, so nothing it does not
# name can be quantized at all, and it does not name these. What `ENGINE_DENSE`
# buys is the FORWARD case — `blocks.N.mod.lin` is a real linear with a real
# width (features -> 6*features), so "let us quantize the modulation too" is a
# one-line change to the geometry map away, and this is what refuses it. The
# self-test asserts both mechanisms SEPARATELY for that reason; asserting only
# `bits_for` would pass on either one alone and catch neither regression.
ENGINE_DENSE = (
    ".mod.lin",              # blocks.N.mod.lin      — per-block modulation
    "last.modulation.lin",   # the final modulation
    ".scale",                # every kreaNorm/normF32 RMSNorm vector
)


def engine_dense(module: str) -> bool:
    """True when the ENGINE has no quantized read path for this tensor."""
    return any(module.endswith(s) or module == s for s in ENGINE_DENSE)

# Quantized in the shipped packs: the 28 blocks and nothing else.
#
# `.mlp.gate` is spelled out rather than a bare ".gate" because that suffix also
# matches `.attn.gate`. The list being unambiguous is not what SAVES it — the
# tier resolution below tests `sensitive` first, so an overlap resolves to the
# sensitive tier either way — but a reader should not have to know that to trust
# the list, and the self-test pins the precedence separately from the spelling.
BULK_SUFFIXES = (
    ".attn.wq",
    ".attn.wk",
    ".attn.wv",
    ".attn.wo",
    ".mlp.gate",
    ".mlp.up",
    ".mlp.down",
)
# The attention out-gate: 6144x6144 whose output is a sigmoid multiplier over
# the whole attention result, so its error is multiplicative over the residual
# stream rather than averaged into one projection. Same argument that keeps
# Ideogram's `adaln_modulation` at 8-bit, and it costs ~28 x 6144 x 6144 = 1.06B
# params... which is NOT cheap here, unlike Ideogram's tier. See `--sensitive-bits`.
SENSITIVE_SUFFIXES = (".attn.gate",)


# ── quantization policy ───────────────────────────────────────────────────

BASE_POLICIES: dict[str, dict[str, int]] = {
    "mixed": {"bulk": 4, "sensitive": 8, "te": 8},
    "mixed_3_8": {"bulk": 3, "sensitive": 8, "te": 4},
}
MIXED_POLICIES: dict[str, dict[str, int]] = {k: dict(v) for k, v in BASE_POLICIES.items()}

# 3 bits is the floor. Not a Krea measurement — a DiT one: Ideogram 4's
# `mixed_2_8` pack rendered a woven grid at every prompt, seed and resolution
# while its 3-bit sibling rendered the same prompts correctly. Krea's own 3-bit
# behaviour is unmeasured (see the module docstring); 2-bit is the width we
# have a verdict on, and the verdict is that it does not render.
MIN_BULK_BITS = 3

# Widths `transformer.affineParamsFromGeometry` can solve back out of a packed
# tensor. Anything else fails at LOAD, not at convert time, which is the worst
# place to find out.
AFFINE_WIDTHS = (2, 3, 4, 5, 6, 8)


def apply_overrides(
    precision: str, bulk: int | None, sensitive: int | None, te: int | None
) -> dict[str, int]:
    """Fold per-tier CLI overrides into the named policy. Mixed policies only —
    a flat precision has no tiers, and silently ignoring the flags would ship a
    pack that does not match what was asked for."""
    policy = MIXED_POLICIES.get(precision)
    if policy is None:
        if bulk or sensitive or te:
            sys.exit(
                f"--bulk-bits/--sensitive-bits/--te-bits need a mixed policy; "
                f"--precision {precision} quantizes everything at one width"
            )
        return {}
    if bulk is not None:
        if bulk < MIN_BULK_BITS:
            sys.exit(
                f"--bulk-bits {bulk}: a {bulk}-bit DiT bulk renders noise, not a "
                f"softer image (measured on Ideogram 4; {MIN_BULK_BITS}-bit is the floor)"
            )
        policy["bulk"] = bulk
    if sensitive is not None:
        policy["sensitive"] = sensitive
    if te is not None:
        policy["te"] = te
    for k, v in policy.items():
        if v not in AFFINE_WIDTHS:
            sys.exit(f"{k} width {v} is not an affine width mlx-serve can read back")
    return policy


def bits_for(module: str, precision: str, default_bits: int) -> int | None:
    """Bit width for a DiT module path, or None to keep it dense bf16."""
    if engine_dense(module):
        return None
    width = in_features_for(module)
    if width is None:
        return None
    policy = MIXED_POLICIES.get(precision)
    # SENSITIVE wins an overlap: `.attn.gate` is a gate first and a projection
    # second, and folding it into the bulk is how `--sensitive-bits` becomes a
    # silent no-op.
    sensitive = any(module.endswith(s) for s in SENSITIVE_SUFFIXES)
    bulk = any(module.endswith(s) for s in BULK_SUFFIXES)
    if not (sensitive or bulk):
        # The small stuff: dense in every policy, flat precisions included.
        # It is ~2% of the pack and its absence from the quantized set is what
        # the shipped 4-bit pack already looks like on disk.
        return None
    if policy is None:
        return default_bits
    return policy["sensitive"] if sensitive else policy["bulk"]


def te_bits_for(precision: str, default_bits: int) -> int:
    """Bit width for the Qwen3-VL-4B text encoder.

    Its own tier because it is a THIRD of the pack (4.3 GB of 12): shrinking
    only the DiT does not produce a meaningfully smaller download, which is the
    trap `mixed_3_6` fell into on the Ideogram side.
    """
    policy = MIXED_POLICIES.get(precision)
    return default_bits if policy is None else policy["te"]


def quantizable(width: int, bits: int, group_size: int) -> bool:
    """Whether `mx.quantize` can pack this width, and mlx-serve read it back.

    Two conditions, both exact-division: the group must tile the row, and the
    packed row must land on a whole number of u32 words. `first` (64 inputs)
    and `txtfusion.projector` (12) fail the first at any sane group size, which
    is why they are dense on disk in every pack that exists.
    """
    if width % group_size:
        return False
    return (width * bits) % 32 == 0


# Parameter counts read off the shipped 4-bit pack's tensor shapes, used ONLY
# for the size estimate printed at plan time and the ordering assertion in the
# self-test.
DIT_BULK_PARAMS = 28 * (4 * FEATURES * FEATURES + 2 * FEATURES * MLP_DIM + MLP_DIM * FEATURES)
DIT_GATE_PARAMS = 28 * FEATURES * FEATURES
DIT_DENSE_PARAMS = 0.30e9  # txtfusion + txtmlp + first/tmlp/tproj/last
TE_PARAMS = 4.0e9
VAE_BYTES = 0.48e9


def bytes_per_param(bits: int | None, group_size: int = 64) -> float:
    """Affine layout: `bits/8` packed, plus a bf16 scale and bias per group."""
    if bits is None:
        return 2.0
    return bits / 8 + 4 / group_size


def estimate_pack_bytes(precision: str, group_size: int = 64) -> float:
    default_bits = {"3": 3, "4": 4, "8": 8}.get(precision, 8)
    bulk = bits_for("blocks.0.attn.wq", precision, default_bits)
    gate = bits_for("blocks.0.attn.gate", precision, default_bits)
    dit = (
        DIT_BULK_PARAMS * bytes_per_param(bulk, group_size)
        + DIT_GATE_PARAMS * bytes_per_param(gate, group_size)
        + DIT_DENSE_PARAMS * 2.0
    )
    te = TE_PARAMS * bytes_per_param(te_bits_for(precision, default_bits), group_size)
    return dit + te + VAE_BYTES


# ── weight I/O ────────────────────────────────────────────────────────────


def dequantize_source(tensors: dict[str, mx.array]) -> tuple[dict[str, mx.array], int]:
    """Fold any existing affine quantization back to bf16, in place of the
    packed keys. Returns (tensors, count) and CONSUMES the input dict.

    `(bits, group_size)` are solved per module from `IN_FEATURES`, never from
    the shapes: `[out, in*bits/32]` alone is satisfied by (8,32) and (4,64)
    alike, and picking the wrong one dequantizes to plausible-looking noise.
    """
    out: dict[str, mx.array] = {}
    n = 0
    for k in list(tensors.keys()):
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        if not k.endswith(".weight"):
            out[k] = tensors.pop(k)
            continue
        module = k[: -len(".weight")]
        scales = tensors.get(module + ".scales")
        if scales is None:
            out[k] = tensors.pop(k)
            continue
        width = in_features_for(module)
        if width is None:
            sys.exit(
                f"{module} is quantized but has no known input width — this "
                f"checkpoint is not the Krea-2-Turbo geometry convert_krea.py mirrors"
            )
        w = tensors.pop(k)
        biases = tensors.pop(module + ".biases")
        tensors.pop(module + ".scales")
        packed_cols = w.shape[1]
        bits = (32 * packed_cols) // width
        gs = width // scales.shape[1]
        if bits not in AFFINE_WIDTHS or width % gs:
            sys.exit(f"{module}: solved ({bits}-bit, group {gs}) is not a layout mlx wrote")
        deq = mx.dequantize(w, scales, biases, group_size=gs, bits=bits)
        mx.eval(deq)
        out[k] = deq.astype(mx.bfloat16)
        n += 1
        del w, biases, deq
        if n % 64 == 0:
            gc.collect()
    out.update({k: v for k, v in tensors.items()})
    tensors.clear()
    return out, n


def emit_linear(
    dst: dict[str, mx.array],
    module: str,
    w: mx.array,
    bits: int | None,
    group_size: int,
) -> int | None:
    """Write one linear, quantized or dense. Returns the width actually used.

    A width the geometry cannot carry is written DENSE rather than silently
    rounded — `MixedLinear` solves `(bits, group_size)` back out of the packed
    shape, so a tensor that does not divide exactly is unreadable, not merely
    imprecise.
    """
    if bits is None or not quantizable(w.shape[1], bits, group_size):
        dst[module + ".weight"] = w.astype(mx.bfloat16)
        return None
    q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=group_size, bits=bits)
    mx.eval(q, s, b)
    dst[module + ".weight"] = q
    dst[module + ".scales"] = s
    dst[module + ".biases"] = b
    return bits


def convert_component(
    src_file: Path,
    dst_file: Path,
    precision: str,
    default_bits: int,
    group_size: int,
    is_text_encoder: bool,
    allow_requantize: bool,
) -> None:
    log(f"[load] {src_file}")
    tensors = dict(mx.load(str(src_file)))
    quantized_in = sum(1 for k in tensors if k.endswith(".scales"))
    if quantized_in:
        if not allow_requantize:
            sys.exit(
                f"{src_file.name} is already quantized ({quantized_in} linears). "
                f"Re-quantizing loses precision twice and the result is not "
                f"something to publish — pass --allow-requantize to do it anyway, "
                f"or point --src at a bf16 checkpoint."
            )
        log(f"[dequant] {quantized_in} quantized linears -> bf16 (LOSSY REPACK)")
        tensors, n = dequantize_source(tensors)
        log(f"[dequant] restored {n} tensors")

    out: dict[str, mx.array] = {}
    counts: dict[int | None, int] = {}
    for k in sorted(tensors.keys()):
        v = tensors[k]
        if not k.endswith(".weight"):
            out[k] = v
            continue
        module = k[: -len(".weight")]
        bits = (
            te_bits_for(precision, default_bits)
            if is_text_encoder and in_features_for(module) is not None
            else bits_for(module, precision, default_bits)
        )
        if is_text_encoder and in_features_for(module) is None:
            bits = None
        used = emit_linear(out, module, v, bits, group_size)
        counts[used] = counts.get(used, 0) + 1
    log(f"[quant] {dst_file.name}: " + ", ".join(
        f"{'dense' if b is None else str(b) + '-bit'}x{n}" for b, n in sorted(
            counts.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)
        )
    ))
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst_file), out)
    del tensors, out
    gc.collect()


def log(msg: str) -> None:
    print(msg, flush=True)


# ── self-test (no mlx weights, no checkpoint) ─────────────────────────────


def self_test() -> None:
    """Unit-test the policy, the geometry map and the engine-dense contract."""
    global BULK_SUFFIXES
    bulk = "blocks.0.attn.wq"
    mlp = "blocks.0.mlp.down"
    gate = "blocks.0.attn.gate"
    mod = "blocks.0.mod.lin"

    # The geometry mirrors krea.KreaConfig's literal defaults.
    assert in_features_for(bulk) == 6144
    assert in_features_for("blocks.0.mlp.gate") == 6144
    assert in_features_for(mlp) == 16384, in_features_for(mlp)
    assert in_features_for("txtfusion.refiner_blocks.1.mlp.down") == 6912
    assert in_features_for("txtfusion.layerwise_blocks.0.attn.wq") == 2560
    assert in_features_for("first") == 64
    assert in_features_for("txtfusion.projector") == 12
    assert in_features_for("last.linear") == 6144
    assert in_features_for("model.layers.7.self_attn.o_proj") == 4096
    assert in_features_for("model.layers.7.mlp.down_proj") == 9728
    assert in_features_for("blocks.0.prenorm.scale") is None
    print("[self-test] geometry map OK")

    # `.attn.gate` and `.mlp.gate` are DIFFERENT tiers.
    assert bits_for(gate, "mixed_3_8", 8) == 8, "attn.gate must take the sensitive tier"
    assert bits_for("blocks.0.mlp.gate", "mixed_3_8", 8) == 3, "mlp.gate is bulk"
    # The spelling is unambiguous...
    assert ".gate" not in BULK_SUFFIXES, "a bare .gate suffix also matches .attn.gate"
    # ...AND the precedence holds even if it stops being. Pinned separately
    # because either one alone makes the two assertions above pass.
    _saved = BULK_SUFFIXES
    try:
        BULK_SUFFIXES = _saved + (".gate",)
        assert bits_for(gate, "mixed_3_8", 8) == 8, "sensitive must win an overlap"
    finally:
        BULK_SUFFIXES = _saved

    # Flat precisions: one width across the quantized set.
    for p_name, want in (("3", 3), ("4", 4), ("8", 8)):
        for m in (bulk, mlp, gate):
            assert bits_for(m, p_name, want) == want, (p_name, m)
        assert te_bits_for(p_name, want) == want

    # Mixed policies: bulk / sensitive / text encoder.
    for p_name, b, sens, te in (("mixed", 4, 8, 8), ("mixed_3_8", 3, 8, 4)):
        assert bits_for(bulk, p_name, 8) == b, p_name
        assert bits_for(mlp, p_name, 8) == b, p_name
        assert bits_for(gate, p_name, 8) == sens, p_name
        assert te_bits_for(p_name, 8) == te, p_name

    # The engine has no quantized path for these. A pack that quantizes one
    # loads packed u32 as bf16 and renders noise. TWO mechanisms keep them
    # dense and each is asserted on its own — `bits_for` alone would stay green
    # with either of them deleted.
    for m in (mod, "last.modulation.lin", "blocks.3.prenorm.scale",
              "blocks.3.postnorm.scale", "blocks.3.attn.qknorm.qnorm.scale"):
        assert engine_dense(m), f"{m} is not refused by ENGINE_DENSE"
        assert in_features_for(m) is None, f"{m} is named by the geometry allowlist"
        for p_name in ("3", "4", "8", "mixed", "mixed_3_8"):
            assert bits_for(m, p_name, 8) is None, (p_name, m)

    # The small stuff stays dense in EVERY policy, flat ones included.
    for m in ("first", "tmlp.0", "tproj.1", "txtfusion.projector",
              "txtmlp.1", "last.linear"):
        for p_name in ("3", "4", "8", "mixed", "mixed_3_8"):
            assert bits_for(m, p_name, 8) is None, (p_name, m)
    print("[self-test] tier assignment OK")

    # Overrides reach every tier without minting another named policy.
    try:
        apply_overrides("mixed_3_8", bulk=4, sensitive=None, te=8)
        assert bits_for(bulk, "mixed_3_8", 8) == 4
        assert te_bits_for("mixed_3_8", 8) == 8
    finally:
        MIXED_POLICIES["mixed_3_8"] = dict(BASE_POLICIES["mixed_3_8"])
    assert bits_for(bulk, "mixed_3_8", 8) == 3

    # A 2-bit DiT bulk is refused: measured on Ideogram 4, where it rendered a
    # woven grid at every prompt, seed and resolution.
    for bad in (1, 2):
        try:
            apply_overrides("mixed_3_8", bulk=bad, sensitive=None, te=None)
        except SystemExit as e:
            assert "renders noise" in str(e), str(e)
        else:
            raise AssertionError(f"--bulk-bits {bad} was accepted")
        finally:
            MIXED_POLICIES["mixed_3_8"] = dict(BASE_POLICIES["mixed_3_8"])
    print("[self-test] quantization policy OK")

    # Quantizability is exact division, both ways. These two are why `first`
    # and the projector are dense on disk in every pack that exists.
    for width in (6144, 16384, 2560, 6912):
        for bits in AFFINE_WIDTHS:
            assert quantizable(width, bits, 64), (width, bits)
    assert quantizable(64, 4, 64), "64 inputs tile exactly at group 64"
    assert not quantizable(12, 4, 64), "the 12-wide projector cannot be grouped"
    # Tiles the row, but the packed row misses the u32 boundary.
    assert not quantizable(16, 3, 16), "16*3 is not a multiple of 32"
    print("[self-test] quantizability OK")

    # A smaller policy must produce a smaller PACK — the text encoder is a
    # third of the download, so a DiT-only shrink is what fails this.
    sizes = {p: estimate_pack_bytes(p) for p in ("8", "4", "3", "mixed", "mixed_3_8")}
    assert sizes["mixed"] < sizes["8"], sizes
    assert sizes["mixed_3_8"] < sizes["mixed"], sizes
    assert sizes["mixed_3_8"] < sizes["4"], sizes
    print("[self-test] pack size ordering OK: "
          + ", ".join(f"{k}={v / 1e9:.1f}GB" for k, v in sizes.items()))


# ── driver ────────────────────────────────────────────────────────────────


def find_transformer(src: Path) -> Path:
    cands = sorted(
        p for p in src.glob("*.safetensors")
        if p.name.startswith("transformer") or p.stem in ("turbo", "model")
    )
    if not cands:
        sys.exit(f"no transformer *.safetensors directly under {src}")
    if len(cands) > 1:
        sys.exit(f"more than one transformer file under {src}: {[p.name for p in cands]}")
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", help="assembled Krea dir (transformer + text_encoder/ + vae/ + tokenizer/)")
    ap.add_argument("--out", help="destination pack directory")
    ap.add_argument("--precision", choices=("3", "4", "8", "mixed", "mixed_3_8"), default="mixed")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bulk-bits", type=int, help="override a mixed policy's attention/MLP width")
    ap.add_argument("--sensitive-bits", type=int, help="override a mixed policy's attention-gate width")
    ap.add_argument("--te-bits", type=int, help="override a mixed policy's text-encoder width")
    ap.add_argument(
        "--allow-requantize",
        action="store_true",
        help="permit an already-quantized --src (lossy; local experiments only)",
    )
    ap.add_argument("--self-test", action="store_true", help="run the policy unit tests and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.src or not args.out:
        ap.error("--src and --out are required (or pass --self-test)")

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    default_bits = {"3": 3, "4": 4, "8": 8}.get(args.precision, 8)
    apply_overrides(args.precision, args.bulk_bits, args.sensitive_bits, args.te_bits)
    te_bits = te_bits_for(args.precision, default_bits)

    if args.precision == "mixed_3_8" or (args.bulk_bits or default_bits) == 3:
        log("[warn] a 3-bit Krea bulk is UNVERIFIED — render a prompt before "
            "publishing this pack (see the module docstring)")

    log(f"[plan] {src} -> {out}  precision={args.precision} group_size={args.group_size} "
        f"text_encoder={te_bits}-bit  (estimated pack ~ "
        f"{estimate_pack_bytes(args.precision, args.group_size) / 1e9:.1f} GB)")
    out.mkdir(parents=True, exist_ok=True)

    convert_component(
        find_transformer(src),
        out / f"transformer_{args.precision}.safetensors",
        args.precision, default_bits, args.group_size,
        is_text_encoder=False, allow_requantize=args.allow_requantize,
    )
    te_src = src / "text_encoder"
    te_files = sorted(te_src.glob("*.safetensors"))
    if not te_files:
        sys.exit(f"no text encoder weights under {te_src}")
    if len(te_files) > 1:
        sys.exit(f"sharded text encoders are not handled yet: {[p.name for p in te_files]}")
    convert_component(
        te_files[0], out / "text_encoder" / "model.safetensors",
        args.precision, default_bits, args.group_size,
        is_text_encoder=True, allow_requantize=args.allow_requantize,
    )

    # The VAE runs in f32 and is 0.48 GB — copied, never quantized. Tokenizer
    # and licences ride along verbatim.
    for sub in ("vae", "tokenizer"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, out / sub, dirs_exist_ok=True)
            log(f"[copy] {sub}/")
    for f in ("LICENSE", "NOTICE"):
        if (src / f).is_file():
            shutil.copy2(src / f, out / f)

    # LAST: discovery classifies a media dir by this file, so an interrupted
    # conversion stays invisible instead of half-registering.
    (out / "config.json").write_text(json.dumps({"model_type": "krea2_turbo"}))
    log(f"[done] {out}")


if __name__ == "__main__":
    main()
