#!/usr/bin/env python3
"""Imatrix-calibrated affine quantization for the DSV4-Flash mirror rebuild.

Two pieces, both consumed by tests/convert_dsv4_weights.py --imatrix:

1. A parser for antirez's llama.cpp-legacy imatrix binary (format decoded from
   lib/ds4/gguf-tools/deepseek4-quantize.c imatrix_load/imatrix_find):
   `i32 n_entries; per entry: i32 name_len, name, i32 ncall, i32 nval,
   f32[nval]`; values/ncall = mean-squared activation per input channel.
   Entries are per-expert-concatenated: nval = in_dim * n_experts, expert e's
   channels at [e*in_dim, (e+1)*in_dim).

2. `weighted_affine_quant`: an activation-weighted (s, b) search (the
   llama.cpp make_qkx2_quants pattern — multi-start iscale sweep + alternating
   weighted-least-squares refit) whose output triples are byte-compatible with
   mx.quantize's affine layout, so the mirror stays engine-native. Scales and
   biases are rounded to bf16 BEFORE the final q derivation (the stored dtype
   is the arithmetic the engine dequants with).

Objective: for a linear layer, an error dW_j in input channel j contributes
E[(dW_j x_j)^2] = dW_j^2 * E[x_j^2] to the output — so the per-channel weight
IS the imatrix value (mean-squared activation), no square rooting.

Pure numpy in the quantization path (multiprocessing-safe, never touches the
GPU); mlx is imported only by the self-test to verify the packing layout
against mx.dequantize.

Usage:
  python3 tests/dsv4_imatrix.py --self-test
"""

import argparse
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import bf16_to_f32, f32_to_bf16_u16, mx  # noqa: E402

# Collected 2026-08-01 on the 0731 weights themselves (full chat-v2 corpus,
# 2.9M tokens through antirez's official 0731 GGUF via `lib/ds4/ds4
# --imatrix-dataset`). The May preview-checkpoint .dat lives beside it —
# never calibrate with it again: 0731 moved the statistics (median r 0.66).
IMATRIX_DEFAULT = os.path.expanduser(
    "~/.mlx-serve/staging/dsv4-ref/imatrix/DeepSeek-V4-Flash-0731-chat-v2-routed-moe-mlxserve.dat")
N_EXPERTS = 256
# Converter proj -> gguf tensor stem (blk.{L}.{stem}.weight):
#   w1 = gate, w3 = up (both read hidden), w2 = down (reads moe_inter).
GGUF_PROJ = {"w1": "ffn_gate_exps", "w2": "ffn_down_exps", "w3": "ffn_up_exps"}


def gguf_expert_entry(layer, proj):
    return f"blk.{layer}.{GGUF_PROJ[proj]}.weight"


# ============================================================
# Imatrix parsing
# ============================================================

def load_imatrix(path):
    """Returns {name: (ncall, values_f32)} with values already divided by
    ncall (matching imatrix_load) — i.e. mean-squared activation per input
    channel. Trailing chunks/dataset metadata is tolerated and ignored."""
    entries = {}
    with open(path, "rb") as f:
        def i32(label):
            b = f.read(4)
            assert len(b) == 4, f"short read: {label}"
            return struct.unpack("<i", b)[0]

        n = i32("entry count")
        assert 0 < n < 1_000_000, f"implausible entry count {n}"
        for _ in range(n):
            ln = i32("name length")
            assert 0 < ln <= 4096, f"bad name length {ln}"
            name = f.read(ln).decode()
            ncall = i32("ncall")
            nval = i32("nval")
            assert nval > 0, f"{name}: bad nval {nval}"
            raw = f.read(nval * 4)
            assert len(raw) == nval * 4, f"{name}: short value read"
            vals = np.frombuffer(raw, dtype=np.float32).copy()
            if ncall > 0:
                vals /= float(ncall)
            assert np.isfinite(vals).all(), f"{name}: non-finite imatrix value"
            assert (vals >= 0).all(), f"{name}: negative mean-squared activation"
            entries[name] = (ncall, vals)
    return entries


def expert_channel_weights(im, layer, proj, eid, in_dim, n_experts=N_EXPERTS):
    """Per-input-channel f32 weights for one expert's projection."""
    name = gguf_expert_entry(layer, proj)
    assert name in im, f"missing imatrix entry {name}"
    _, vals = im[name]
    assert vals.size == in_dim * n_experts, \
        f"{name}: nval {vals.size} != {in_dim} x {n_experts}"
    return vals[eid * in_dim:(eid + 1) * in_dim]


# ============================================================
# MLX affine packing (byte-compatible with mx.quantize's layout)
# ============================================================

def pack_bits(q, bits):
    """q: integer array [..., in_dim], values in [0, 2^bits-1] -> uint32 array
    matching mx.quantize's packed layout. 2/4/8-bit pack 32/bits consecutive
    values little-endian into each uint32; 3-bit packs 8 values into a 24-bit
    little-endian group emitted as 3 bytes, byte stream viewed as uint32
    (mlx/backend/cpu/quantized.cpp quantize<T,U>)."""
    lead = q.shape[:-1]
    n = q.shape[-1]
    if bits in (2, 4, 8):
        per = 32 // bits
        assert n % per == 0, f"in_dim {n} not divisible by {per}"
        g = q.reshape(-1, per).astype(np.uint32)
        word = np.zeros(g.shape[0], dtype=np.uint32)
        for k in range(per):
            word |= g[:, k] << np.uint32(k * bits)
        return word.reshape(*lead, n // per)
    if bits == 3:
        assert n % 8 == 0 and (n * 3 // 8) % 4 == 0, f"in_dim {n} unpackable at 3 bits"
        g = q.reshape(-1, 8).astype(np.uint32)
        w24 = np.zeros(g.shape[0], dtype=np.uint32)
        for k in range(8):
            w24 |= g[:, k] << np.uint32(3 * k)
        b = np.empty((g.shape[0], 3), dtype=np.uint8)
        b[:, 0] = w24 & 0xFF
        b[:, 1] = (w24 >> 8) & 0xFF
        b[:, 2] = (w24 >> 16) & 0xFF
        return np.ascontiguousarray(b.reshape(*lead, n * 3 // 8)).view(np.uint32)
    if bits in (5, 6):
        # mx.quantize packs DENSELY: element i at bit offset i*bits of a little-endian
        # byte stream (straddling words), viewed as uint32.
        per = 32 // math.gcd(bits, 32)          # values per whole number of words
        assert n % per == 0, f"in_dim {n} not divisible by {per}"
        g = q.reshape(-1, per).astype(np.uint16)
        nbytes = per * bits // 8
        acc = np.zeros((g.shape[0], nbytes + 1), dtype=np.uint16)
        for k in range(per):
            off = k * bits
            v = g[:, k] << np.uint16(off % 8)
            acc[:, off // 8] |= v & 0xFF
            acc[:, off // 8 + 1] |= v >> 8
        b = acc[:, :nbytes].astype(np.uint8)
        return np.ascontiguousarray(b.reshape(*lead, n * bits // 8)).view(np.uint32)
    raise ValueError(f"unsupported bits {bits}")


def unpack_bits(packed_u32, bits, in_dim):
    """Inverse of pack_bits: uint32 [..., cols] -> uint8 q [..., in_dim]."""
    lead = packed_u32.shape[:-1]
    if bits in (2, 4, 8):
        per = 32 // bits
        w = packed_u32.reshape(-1, 1)
        shifts = np.uint32(bits) * np.arange(per, dtype=np.uint32)
        q = (w >> shifts) & np.uint32((1 << bits) - 1)
        return q.reshape(*lead, in_dim).astype(np.uint8)
    if bits == 3:
        by = np.ascontiguousarray(packed_u32).view(np.uint8).reshape(-1, 3)
        w24 = by[:, 0].astype(np.uint32) | (by[:, 1].astype(np.uint32) << 8) \
            | (by[:, 2].astype(np.uint32) << 16)
        shifts = np.uint32(3) * np.arange(8, dtype=np.uint32)
        q = (w24.reshape(-1, 1) >> shifts) & np.uint32(7)
        return q.reshape(*lead, in_dim).astype(np.uint8)
    raise ValueError(f"unsupported bits {bits}")


def dequant_np(packed_u32, scales_u16, biases_u16, bits, group_size):
    """Numpy dequant of our packed triple: q*s + b in f32, s/b upcast from
    bf16. Reference arm for the packing round-trip and the pilot."""
    q = unpack_bits(packed_u32, bits, scales_u16.shape[-1] * group_size).astype(np.float32)
    s = bf16_to_f32(scales_u16)
    b = bf16_to_f32(biases_u16)
    return q.reshape(*s.shape, group_size) * s[..., None] + b[..., None]


# ============================================================
# Weighted affine quantization
# ============================================================

def _mlx_minmax_sb(X, n_bins):
    """MLX's own affine (s, b) per group (quantize<T,U> in
    mlx/backend/cpu/quantized.cpp): sign-flipped scale, edge snapped to an
    exact grid point. X: [..., gs] f32 -> (s, b) each [...]."""
    xmin = X.min(-1)
    xmax = X.max(-1)
    mask = np.abs(xmin) > np.abs(xmax)
    scale = np.maximum((xmax - xmin) / n_bins, np.float32(1e-7))
    scale = np.where(mask, scale, -scale)
    edge = np.where(mask, xmin, xmax)
    q0 = np.rint(edge / scale)
    nz = q0 != 0
    scale = np.where(nz, edge / np.where(nz, q0, 1.0), scale)
    bias = np.where(nz, edge, np.float32(0.0))
    return scale.astype(np.float32), bias.astype(np.float32)


def _bf16_round(x):
    return bf16_to_f32(f32_to_bf16_u16(x.astype(np.float32)))


def weighted_affine_quant(w_f32, bits, group_size, ch_weights,
                          nstep=14, refine=3, return_stats=False):
    """f32 [out, in] + per-input-channel weights [in] -> ("U32"/"BF16" triples
    ready for write_safetensors_raw, same contract as mlx_affine_quant).

    Search: MLX's own minmax (s, b) as candidate 0 (so uniform weights can
    never do worse than mx.quantize), an iscale multi-start sweep, and for
    every candidate labeling q a weighted-least-squares refit of (s, b);
    then alternating q/(s, b) refinement from the best. Scales/biases are
    bf16-rounded before the final q so the stored arithmetic is what was
    optimized."""
    out_dim, in_dim = w_f32.shape
    assert in_dim % group_size == 0
    G = in_dim // group_size
    n_bins = float((1 << bits) - 1)

    X = np.ascontiguousarray(w_f32, dtype=np.float32).reshape(out_dim, G, group_size)
    om = np.asarray(ch_weights, dtype=np.float32)
    assert om.shape == (in_dim,)
    # Relative floor so all-zero channels (never activated in calibration)
    # keep a small vote instead of letting WLS go singular.
    floor = float(om.mean()) * 1e-4 + 1e-30
    W = (om + floor).reshape(1, G, group_size)

    sw = W.sum(-1)                    # [1, G]
    swx = (W * X).sum(-1)             # [out, G]
    swx2 = (W * X * X).sum(-1)        # [out, G]

    best_err = np.full((out_dim, G), np.inf, dtype=np.float32)
    best_s = np.ones((out_dim, G), dtype=np.float32)
    best_b = np.zeros((out_dim, G), dtype=np.float32)

    def q_sums(q):
        Wq = W * q
        return Wq.sum(-1), (Wq * q).sum(-1), (Wq * X).sum(-1)

    def closed_err(s, b, sl, sl2, sxl):
        return (swx2 + s * s * sl2 + b * b * sw + 2 * s * b * sl
                - 2 * s * sxl - 2 * b * swx)

    def consider(s, b, sl, sl2, sxl):
        nonlocal best_err, best_s, best_b
        err = closed_err(s, b, sl, sl2, sxl)
        upd = err < best_err
        if upd.any():
            best_err = np.where(upd, err, best_err)
            best_s = np.where(upd, s, best_s)
            best_b = np.where(upd, b, best_b)

    def refit(sl, sl2, sxl):
        D = sw * sl2 - sl * sl
        ok = D > 0
        Dsafe = np.where(ok, D, 1.0)
        s = np.where(ok, (sw * sxl - swx * sl) / Dsafe, np.nan)
        b = np.where(ok, (sl2 * swx - sl * sxl) / Dsafe, np.nan)
        return s, b, ok

    def consider_refit(sl, sl2, sxl):
        s, b, ok = refit(sl, sl2, sxl)
        if ok.any():
            err = closed_err(s, b, sl, sl2, sxl)
            upd = ok & (err < best_err)
            if upd.any():
                nonlocal_update(upd, err, s, b)

    def nonlocal_update(upd, err, s, b):
        nonlocal best_err, best_s, best_b
        best_err = np.where(upd, err, best_err)
        best_s = np.where(upd, s, best_s)
        best_b = np.where(upd, b, best_b)

    # Candidate 0: MLX's own minmax (s, b) + its labeling, plus a WLS refit
    # of that labeling.
    s0, b0 = _mlx_minmax_sb(X, n_bins)
    q = np.clip(np.rint((X - b0[..., None]) / s0[..., None]), 0, n_bins)
    sl, sl2, sxl = q_sums(q)
    consider(s0, b0, sl, sl2, sxl)
    consider_refit(sl, sl2, sxl)

    # Multi-start iscale sweep (make_qkx2_quants shape): label from the
    # min-anchored grid at a spread of scales, refit (s, b) per labeling.
    xmin = X.min(-1)
    span = X.max(-1) - xmin
    nondeg = span > 0
    span_safe = np.where(nondeg, span, 1.0)
    for k in range(nstep):
        iscale = (n_bins - 1.0 + 0.25 * k) / span_safe
        q = np.clip(np.rint((X - xmin[..., None]) * iscale[..., None]), 0, n_bins)
        sl, sl2, sxl = q_sums(q)
        consider_refit(sl, sl2, sxl)

    # Alternating refinement from the incumbent.
    for _ in range(refine):
        q = np.clip(np.rint((X - best_b[..., None]) / best_s[..., None]), 0, n_bins)
        sl, sl2, sxl = q_sums(q)
        consider_refit(sl, sl2, sxl)

    # bf16 storage rounding, then the final q under the ROUNDED params.
    sb = _bf16_round(best_s)
    bb = _bf16_round(best_b)
    dead = (sb == 0) | ~np.isfinite(sb) | ~np.isfinite(bb)
    if dead.any():
        # Constant/degenerate groups: q = 0, b = weighted mean.
        sb = np.where(dead, np.float32(1.0), sb)
        bb = np.where(dead, _bf16_round(swx / sw), bb)
    q = np.clip(np.rint((X - bb[..., None]) / sb[..., None]), 0, n_bins)
    q = q.astype(np.uint8).reshape(out_dim, in_dim)

    packed = pack_bits(q, bits)
    triples = (
        ("U32", packed.shape, packed.tobytes()),
        ("BF16", (out_dim, G), f32_to_bf16_u16(sb).tobytes()),
        ("BF16", (out_dim, G), f32_to_bf16_u16(bb).tobytes()),
    )
    if not return_stats:
        return triples
    qf = q.reshape(out_dim, G, group_size).astype(np.float32)
    resid = sb[..., None] * qf + bb[..., None] - X
    werr = float((W * resid * resid).sum())
    wnorm = float(swx2.sum())
    return triples, {"weighted_err": werr, "weighted_rel_err": werr / max(wnorm, 1e-30)}


def weighted_rel_err(w_f32, w_hat_f32, ch_weights):
    """Sum_j om_j (w_hat - w)^2 / Sum_j om_j w^2 over the tensor — the pilot's
    comparison metric (proportional to expected output-error energy)."""
    om = np.asarray(ch_weights, dtype=np.float32)[None, :]
    d = (w_hat_f32 - w_f32).astype(np.float32)
    num = float((om * d * d).sum())
    den = float((om * w_f32.astype(np.float32) ** 2).sum())
    return num / max(den, 1e-30)


# ============================================================
# Self-test
# ============================================================

def self_test():
    failures = []

    def check(cond, label):
        print(("PASS " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    rng = np.random.default_rng(11)
    m = mx()

    # --- packing round-trip: our packed triples through mx.dequantize match
    # our own numpy dequant, and the raw q survives exactly (s=1, b=0).
    for bits in (2, 3, 4, 8):
        for gs in (64, 128):
            in_dim, out_dim = 256, 8
            q = rng.integers(0, 1 << bits, size=(out_dim, in_dim), dtype=np.uint8)
            packed = pack_bits(q, bits)
            ones = np.ones((out_dim, in_dim // gs), dtype=np.float32)
            zeros = np.zeros_like(ones)
            wq = m.array(packed)
            s = m.array(f32_to_bf16_u16(ones)).view(m.bfloat16)
            b = m.array(f32_to_bf16_u16(zeros)).view(m.bfloat16)
            back = np.array(m.dequantize(wq, s, b, group_size=gs, bits=bits)
                            .astype(m.float32), copy=False)
            check(np.array_equal(back, q.astype(np.float32)),
                  f"pack/mx.dequantize identity q round-trip bits={bits} gs={gs}")
            check(np.array_equal(unpack_bits(packed, bits, in_dim), q),
                  f"unpack_bits inverts pack_bits bits={bits} gs={gs}")
            # with real scales/biases: mx.dequantize == our numpy dequant
            sc = rng.standard_normal((out_dim, in_dim // gs)).astype(np.float32) * 0.1
            bi = rng.standard_normal((out_dim, in_dim // gs)).astype(np.float32) * 0.05
            sc_u16, bi_u16 = f32_to_bf16_u16(sc), f32_to_bf16_u16(bi)
            ref = dequant_np(packed, sc_u16, bi_u16, bits, gs)
            got = np.array(m.dequantize(
                wq, m.array(sc_u16).view(m.bfloat16), m.array(bi_u16).view(m.bfloat16),
                group_size=gs, bits=bits).astype(m.float32), copy=False)
            close = np.allclose(got, ref.reshape(got.shape), rtol=2e-2, atol=1e-3)
            check(close, f"pack + scales/biases matches mx.dequantize bits={bits} gs={gs}")

    # --- weighted quant vs mx.quantize, uniform weights: never worse.
    for bits in (2, 3):
        w = rng.standard_normal((32, 512)).astype(np.float32)
        uni = np.ones(512, dtype=np.float32)
        tri = weighted_affine_quant(w, bits, 128, uni)
        ours = dequant_np(np.frombuffer(tri[0][2], np.uint32).reshape(tri[0][1]),
                          np.frombuffer(tri[1][2], np.uint16).reshape(tri[1][1]),
                          np.frombuffer(tri[2][2], np.uint16).reshape(tri[2][1]),
                          bits, 128).reshape(w.shape)
        wb = m.array(f32_to_bf16_u16(w)).view(m.bfloat16).reshape(w.shape)
        mq, msc, mbi = m.quantize(wb, group_size=128, bits=bits)
        theirs = np.array(m.dequantize(mq, msc, mbi, group_size=128, bits=bits)
                          .astype(m.float32), copy=False)
        e_ours = float(((ours - w) ** 2).sum())
        e_mlx = float(((theirs - w) ** 2).sum())
        check(e_ours <= e_mlx * 1.0001,
              f"uniform weights: search <= mx.quantize ({e_ours:.4f} vs {e_mlx:.4f}, {bits}b)")

    # --- skewed weights: weighted error strictly below minmax's, and hot
    # channels reconstruct better than they do under minmax.
    for bits in (2, 3):
        w = rng.standard_normal((48, 512)).astype(np.float32)
        om = np.full(512, 0.01, dtype=np.float32)
        hot = rng.choice(512, size=32, replace=False)
        om[hot] = 10.0
        tri = weighted_affine_quant(w, bits, 128, om)
        ours = dequant_np(np.frombuffer(tri[0][2], np.uint32).reshape(tri[0][1]),
                          np.frombuffer(tri[1][2], np.uint16).reshape(tri[1][1]),
                          np.frombuffer(tri[2][2], np.uint16).reshape(tri[2][1]),
                          bits, 128).reshape(w.shape)
        wb = m.array(f32_to_bf16_u16(w)).view(m.bfloat16).reshape(w.shape)
        mq, msc, mbi = m.quantize(wb, group_size=128, bits=bits)
        theirs = np.array(m.dequantize(mq, msc, mbi, group_size=128, bits=bits)
                          .astype(m.float32), copy=False)
        we_ours = weighted_rel_err(w, ours, om)
        we_mlx = weighted_rel_err(w, theirs, om)
        check(we_ours < we_mlx * 0.98,
              f"skewed weights: weighted err beats minmax ({we_ours:.5f} vs {we_mlx:.5f}, {bits}b)")
        hot_ours = float(((ours - w)[:, hot] ** 2).sum())
        hot_mlx = float(((theirs - w)[:, hot] ** 2).sum())
        check(hot_ours < hot_mlx,
              f"skewed weights: hot channels reconstruct better ({hot_ours:.4f} vs {hot_mlx:.4f}, {bits}b)")
        # The discriminating check: an implementation that IGNORES ch_weights
        # (uniform-weight search evaluated under the skew) still beats raw
        # minmax via the refit alone, so the two checks above cannot pin the
        # weighting. Aware must strictly beat blind under the skew.
        trib = weighted_affine_quant(w, bits, 128, np.ones(512, dtype=np.float32))
        blind = dequant_np(np.frombuffer(trib[0][2], np.uint32).reshape(trib[0][1]),
                           np.frombuffer(trib[1][2], np.uint16).reshape(trib[1][1]),
                           np.frombuffer(trib[2][2], np.uint16).reshape(trib[2][1]),
                           bits, 128).reshape(w.shape)
        we_blind = weighted_rel_err(w, blind, om)
        check(we_ours < we_blind * 0.98,
              f"skewed weights: aware beats weight-blind search ({we_ours:.5f} vs {we_blind:.5f}, {bits}b)")

    # --- degenerate groups survive: constant group, zero-weight channels.
    w = np.zeros((4, 256), dtype=np.float32)
    w[:, :128] = 3.25
    w[2, 130] = 1.0
    om = np.zeros(256, dtype=np.float32)
    om[130] = 5.0
    tri = weighted_affine_quant(w, 2, 128, om)
    back = dequant_np(np.frombuffer(tri[0][2], np.uint32).reshape(tri[0][1]),
                      np.frombuffer(tri[1][2], np.uint16).reshape(tri[1][1]),
                      np.frombuffer(tri[2][2], np.uint16).reshape(tri[2][1]),
                      2, 128).reshape(w.shape)
    check(np.isfinite(back).all() and abs(back[0, 0] - 3.25) < 0.05
          and abs(back[2, 130] - 1.0) < 0.05,
          "degenerate groups: constant + zero-weight channels stay finite/accurate")

    # --- parser on the REAL .dat (skips with a note if absent).
    if os.path.exists(IMATRIX_DEFAULT):
        im = load_imatrix(IMATRIX_DEFAULT)
        check(len(im) == 129, f"real .dat: 129 entries (got {len(im)})")
        want = {gguf_expert_entry(l, p) for l in range(43) for p in ("w1", "w2", "w3")}
        check(set(im.keys()) == want, "real .dat: names cover 43 layers x gate/up/down")
        ok_shapes = all(
            im[gguf_expert_entry(l, p)][1].size == (4096 if p in ("w1", "w3") else 2048) * N_EXPERTS
            for l in range(43) for p in ("w1", "w2", "w3"))
        check(ok_shapes, "real .dat: nval == in_dim x 256 per entry")
        cw = expert_channel_weights(im, 21, "w2", 255, 2048)
        check(cw.shape == (2048,) and np.isfinite(cw).all() and (cw >= 0).all()
              and cw.max() > 0,
              "real .dat: per-expert slice is finite, nonnegative, nonzero")
    else:
        print(f"SKIP real .dat parse ({IMATRIX_DEFAULT} not present)")

    print(f"\n{len(failures)} failures" if failures else "\nALL SELF-TESTS PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    ap.print_help()
