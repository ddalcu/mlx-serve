#!/usr/bin/env python3
"""Faithful MLX reference for DeepSeek-V4-Flash, transcribed from the release's
own inference/{model,kernel}.py (torch/tilelang, start_pos==0 path ONLY).

This is the ORACLE for the Zig port (the mlx-lm PR draft diverges from the
reference in several places and is not usable as ground truth). Decode is
token-by-token FULL re-forward — O(n^2) but stateless, so none of the
reference's incremental ring-buffer machinery needs transcribing.

Runs the converted mirror (tests/convert_dsv4_weights.py output: affine-
quantized, stacked expert banks, bare inference-style names).

Usage:
  python3 tests/dsv4_mlx_ref.py --model ~/.mlx-serve/models/ddalcu/DeepSeek-V4-Flash-0731-MLX-Serve-mixed-2-3-8bit \
      --prompt "The capital of France is" --max-tokens 20 [--raw] [--layers N]

Fidelity notes (each pinned to the reference):
  - window idxs: last 128 raw positions per query (get_window_topk_idxs)
  - compressed slots appended AFTER raw kv at start_pos==0 (offset = seqlen)
  - ratio-4 layers: indexer top-512 over its own Hadamard+fp4-sim compressed
    keys; other ratios: ALL visible compressed slots (get_compress_topk_idxs)
  - kv non-rope dims fp8-simulated (e4m3 gs64, ue8m0 scale 2^ceil(log2(amax/448)))
  - attn_sink joins the softmax denominator only
  - inverse (conjugate) rope on the attention output's rope dims
  - hc: pre/post/comb via Sinkhorn (softmax+eps, colnorm, then 19x row/col)
  - hc_post residual mix is comb^T (sum over SOURCE copy j of comb[j,k]·res[j])
  - gate: sqrt(softplus) scores f32, bias for selection only, hash layers route
    by token id, weights sum-normalized then ×1.5
  - clipped SwiGLU: up=clip(±10), gate=min(+10) — shared expert INCLUDED
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx


# ---------------- fp8/fp4 QAT simulation (numpy, host) ----------------

def _round_half_even_to_grid(y, mant_bits, min_exp, max_val):
    """Round |y| to a float grid with `mant_bits` mantissa bits, subnormal
    floor at 2^min_exp, saturating clamp at max_val. Ties to even."""
    a = np.abs(y)
    a = np.minimum(a, max_val)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** min_exp)))
    e = np.maximum(e, min_exp)
    quantum = np.exp2(e - mant_bits)
    q = np.round(a / quantum) * quantum  # np.round is half-to-even
    # rounding can carry to the next binade (e.g. 1.9999 -> 2.0); that's fine
    return np.copysign(np.minimum(q, max_val), y)


def fp8_sim(x, group=64):
    """act_quant(..., round_scale=True, inplace=True): e4m3 quant-dequant with
    ue8m0 (power-of-2) per-group scales along the last dim."""
    orig_dtype = x.dtype
    xs = x.astype(np.float32)
    shp = xs.shape
    g = xs.reshape(-1, group)
    amax = np.maximum(np.abs(g).max(axis=-1, keepdims=True), 1e-4)
    scale = np.exp2(np.ceil(np.log2(amax / 448.0)))
    y = np.clip(g / scale, -448.0, 448.0)
    y = _round_half_even_to_grid(y, mant_bits=3, min_exp=-6, max_val=448.0)
    return (y * scale).reshape(shp).astype(orig_dtype)


def fp4_sim(x, group=32):
    """fp4_act_quant(inplace=True): e2m1 quant-dequant, ue8m0 scales."""
    orig_dtype = x.dtype
    xs = x.astype(np.float32)
    shp = xs.shape
    g = xs.reshape(-1, group)
    amax = np.maximum(np.abs(g).max(axis=-1, keepdims=True), 6.0 * 2.0 ** -126)
    scale = np.exp2(np.ceil(np.log2(amax / 6.0)))
    y = np.clip(g / scale, -6.0, 6.0)
    y = _round_half_even_to_grid(y, mant_bits=1, min_exp=0, max_val=6.0)
    return (y * scale).reshape(shp).astype(orig_dtype)


def hadamard_matrix(n):
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    assert h.shape[0] == n
    return h * (n ** -0.5)


# ---------------- rope (interleaved complex pairs = traditional) ----------------

def precompute_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """Returns (cos, sin) [seqlen, dim//2], YaRN-interpolated when original_seq_len>0."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if original_seq_len > 0:
        def corr_dim(nr):
            return dim * np.log(original_seq_len / (nr * 2 * np.pi)) / (2 * np.log(base))
        low = max(int(np.floor(corr_dim(beta_fast))), 0)
        high = min(int(np.ceil(corr_dim(beta_slow))), dim - 1)
        if low == high:
            high += 1
        ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = np.arange(seqlen, dtype=np.float64)
    ang = np.outer(t, freqs)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x, cos, sin, inverse=False):
    """x [..., P, rd] with cos/sin [P, rd//2]; interleaved pairs."""
    orig_dtype = x.dtype
    xf = x.astype(np.float32)
    xr = xf[..., 0::2]
    xi = xf[..., 1::2]
    s = -sin if inverse else sin
    if xr.ndim == cos.ndim + 1:
        cos = cos[:, None, :]  # [P, rd/2] -> [P, 1(heads), rd/2]
        s = s[:, None, :]
    assert cos.shape[0] == xr.shape[0], f"rope pos axis {cos.shape} vs {xr.shape}"
    yr = xr * cos - xi * s
    yi = xr * s + xi * cos
    y = np.empty_like(xf)
    y[..., 0::2] = yr
    y[..., 1::2] = yi
    return y.astype(orig_dtype)


# ---------------- sinkhorn hyper-connections ----------------

def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    """mixes [T, (2+hc)*hc] f32 -> pre [T,hc], post [T,hc], comb [T,hc,hc]."""
    pre = 1.0 / (1.0 + np.exp(-(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]))) + eps
    post = 2.0 / (1.0 + np.exp(-(mixes[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])))
    comb = mixes[:, 2 * hc:].reshape(-1, hc, hc) * hc_scale[2] + hc_base[2 * hc:].reshape(hc, hc)
    m = comb.max(axis=-1, keepdims=True)
    comb = np.exp(comb - m)
    comb = comb / comb.sum(axis=-1, keepdims=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


# ---------------- quantized linear helpers (mlx) ----------------

class QLinear:
    def __init__(self, weights, base, qcfg):
        self.w = weights[base + ".weight"]
        self.s = weights[base + ".scales"]
        self.b = weights[base + ".biases"]
        c = qcfg.get(base, qcfg)
        self.bits = c["bits"]
        self.gs = c["group_size"]

    def __call__(self, x):
        return mx.quantized_matmul(x, self.w, self.s, self.b, transpose=True,
                                   group_size=self.gs, bits=self.bits)

    def dequant(self):
        return mx.dequantize(self.w, self.s, self.b, group_size=self.gs, bits=self.bits)


class BF16Linear:
    def __init__(self, weights, base):
        self.w = weights[base + ".weight"]

    def __call__(self, x):
        return x @ self.w.T


def rms_norm(x, weight, eps=1e-6):
    xf = x.astype(mx.float32)
    xn = xf * mx.rsqrt((xf * xf).mean(axis=-1, keepdims=True) + eps)
    return (weight.astype(mx.float32) * xn).astype(x.dtype)


# ---------------- model ----------------

class Dsv4Ref:
    def __init__(self, model_dir, max_layers=None):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        self.qcfg = cfg["quantization"]
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        shard_names = sorted(set(idx["weight_map"].values()))
        weights = {}
        for sn in shard_names:
            weights.update(mx.load(os.path.join(model_dir, sn)))
        self.weights = weights

        self.n_layers = cfg["num_hidden_layers"] if max_layers is None else min(max_layers, cfg["num_hidden_layers"])
        self.dim = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["head_dim"]
        self.rd = cfg["qk_rope_head_dim"]
        self.window = cfg["sliding_window"]
        self.hc = cfg["hc_mult"]
        self.hc_iters = cfg["hc_sinkhorn_iters"]
        self.hc_eps = cfg["hc_eps"]
        self.eps = cfg["rms_norm_eps"]
        self.ratios = cfg["compress_ratios"]
        self.o_groups = cfg["o_groups"]
        self.o_lora = cfg["o_lora_rank"]
        self.topk = cfg["num_experts_per_tok"]
        self.n_hash = cfg["num_hash_layers"]
        self.route_scale = cfg["routed_scaling_factor"]
        self.swiglu_limit = cfg["swiglu_limit"]
        self.idx_topk = cfg["index_topk"]
        self.idx_heads = cfg["index_n_heads"]
        self.idx_hd = cfg["index_head_dim"]

        rs = cfg.get("rope_scaling") or {}
        self.yarn = (cfg["compress_rope_theta"], rs.get("original_max_position_embeddings", 0),
                     rs.get("factor", 1.0), rs.get("beta_fast", 32), rs.get("beta_slow", 1))
        self.plain_theta = cfg["rope_theta"]
        self.H128 = hadamard_matrix(self.idx_hd)

        # DSpark (0731): stage count = ratio-table entries past the trunk (the
        # release config carries no n_mtp_layers; num_nextn_predict_layers is a
        # stale HF leftover reading 1 against 3 shipped stages).
        self.dspark_block = cfg.get("dspark_block_size", 0)
        self.dspark_noise = cfg.get("dspark_noise_token_id", 0)
        self.dspark_targets = cfg.get("dspark_target_layer_ids", [])
        self.dspark_rank = cfg.get("dspark_markov_rank", 0)
        self.n_mtp = max(len(self.ratios) - cfg["num_hidden_layers"], 0) if self.dspark_block else 0

        # embed dequantized once (gather-read)
        self.embed = QLinear(weights, "embed", self.qcfg).dequant()
        self.head = QLinear(weights, "head", self.qcfg)
        self._freqs_cache = {}

    def freqs(self, kind, seqlen):
        key = (kind, seqlen)
        if key not in self._freqs_cache:
            if kind == "plain":
                self._freqs_cache[key] = precompute_freqs(self.rd, seqlen, 0, self.plain_theta, 1, 32, 1)
            else:
                theta, orig, factor, bf, bs = self.yarn
                self._freqs_cache[key] = precompute_freqs(self.rd, seqlen, orig, theta, factor, bf, bs)
        return self._freqs_cache[key]

    # ---- components (numpy in/out for the fiddly parts) ----

    def compressor(self, x_np, pfx, ratio, head_dim, rotate, cos, sin):
        """x_np [s, dim] f32 -> compressed [s//ratio, head_dim] f32 (or empty)."""
        w = self.weights
        s = x_np.shape[0]
        cutoff = (s // ratio) * ratio
        if cutoff == 0:
            return np.zeros((0, head_dim), dtype=np.float32)
        overlap = ratio == 4
        coff = 2 if overlap else 1
        wkv = np.array(w[pfx + ".wkv.weight"].astype(mx.float32))
        wgate = np.array(w[pfx + ".wgate.weight"].astype(mx.float32))
        ape = np.array(w[pfx + ".ape"])
        norm_w = np.array(w[pfx + ".norm.weight"].astype(mx.float32))
        kv = x_np @ wkv.T          # [s, coff*d]
        score = x_np @ wgate.T
        kv = kv[:cutoff].reshape(-1, ratio, coff * head_dim)
        score = score[:cutoff].reshape(-1, ratio, coff * head_dim) + ape
        if overlap:
            d = head_dim
            nb = kv.shape[0]
            kv_ov = np.zeros((nb, 2 * ratio, d), dtype=np.float32)
            kv_ov[:, ratio:] = kv[:, :, d:]
            kv_ov[1:, :ratio] = kv[:-1, :, :d]
            sc_ov = np.full((nb, 2 * ratio, d), -np.inf, dtype=np.float32)
            sc_ov[:, ratio:] = score[:, :, d:]
            sc_ov[1:, :ratio] = score[:-1, :, :d]
            kv, score = kv_ov, sc_ov
        m = score.max(axis=1, keepdims=True)
        e = np.exp(score - m)
        wts = e / e.sum(axis=1, keepdims=True)
        out = (kv * wts).sum(axis=1)  # [nb, d]
        # norm in bf16 like the reference (kv.to(dtype) then RMSNorm)
        ob = mx.array(out).astype(mx.bfloat16)
        ob = rms_norm(ob, mx.array(norm_w), self.eps)
        out = np.array(ob.astype(mx.float32))
        # rope at block-start raw positions 0, ratio, 2*ratio, ...
        out[:, -self.rd:] = apply_rope(out[:, -self.rd:], cos[0:cutoff:ratio], sin[0:cutoff:ratio])
        if rotate:
            out = (out @ self.H128.T) if head_dim == self.idx_hd else out
            out = fp4_sim(out, 32)
        else:
            out[:, :-self.rd] = fp8_sim(out[:, :-self.rd], 64)
        return out

    def indexer(self, x_np, qr_np, li, s, cos, sin):
        """Top-k compressed-slot indices per position. Returns [s, k] with -1
        padding, already offset by +s (compressed slots sit after raw kv)."""
        w = self.weights
        pfx = f"layers.{li}.attn.indexer"
        ratio = 4
        n_slots = s // ratio
        if n_slots == 0:
            return np.zeros((s, 0), dtype=np.int64)
        qlin = QLinear(w, pfx + ".wq_b", self.qcfg)
        q = np.array(qlin(mx.array(qr_np).astype(mx.bfloat16)).astype(mx.float32))
        q = q.reshape(s, self.idx_heads, self.idx_hd)
        q[..., -self.rd:] = apply_rope(q[..., -self.rd:], cos[:s], sin[:s])
        q = q @ self.H128.T
        q = fp4_sim(q, 32)
        ck = self.compressor(x_np, pfx + ".compressor", ratio, self.idx_hd, True, cos, sin)
        wp = np.array(w[pfx + ".weights_proj.weight"].astype(mx.float32))
        wts = (x_np @ wp.T) * (self.idx_hd ** -0.5) * (self.idx_heads ** -0.5)  # [s, h]
        scores = np.einsum("shd,td->sht", q, ck)
        scores = np.maximum(scores, 0.0) * wts[:, :, None]
        scores = scores.sum(axis=1)  # [s, t]
        pos = np.arange(s)
        visible = np.arange(n_slots)[None, :] < ((pos[:, None] + 1) // ratio)
        scores = np.where(visible, scores, -np.inf)
        k = min(self.idx_topk, n_slots)
        top = np.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
        # invalid (masked) selections -> -1, else +offset s
        sel_vis = np.take_along_axis(visible, top, axis=1)
        return np.where(sel_vis, top + s, -1)

    def attention(self, x, li, ratio, input_len):
        """x [s, dim] bf16 mlx. Returns [s, dim] bf16 mlx."""
        w = self.weights
        pfx = f"layers.{li}.attn"
        s = x.shape[0]
        kind = "yarn" if ratio else "plain"
        cos, sin = self.freqs(kind, s)

        qr = rms_norm(QLinear(w, pfx + ".wq_a", self.qcfg)(x), w[pfx + ".q_norm.weight"], self.eps)
        q = QLinear(w, pfx + ".wq_b", self.qcfg)(qr)
        q_np = np.array(q.astype(mx.float32)).reshape(s, self.n_heads, self.head_dim)
        q_np = q_np * (1.0 / np.sqrt((q_np ** 2).mean(axis=-1, keepdims=True) + self.eps))
        q_np[..., -self.rd:] = apply_rope(q_np[..., -self.rd:], cos[:s], sin[:s])

        kv = rms_norm(QLinear(w, pfx + ".wkv", self.qcfg)(x), w[pfx + ".kv_norm.weight"], self.eps)
        kv_np = np.array(kv.astype(mx.float32))
        kv_np[:, -self.rd:] = apply_rope(kv_np[:, -self.rd:], cos[:s], sin[:s])
        kv_np[:, :-self.rd] = fp8_sim(kv_np[:, :-self.rd], 64)

        x_np = np.array(x.astype(mx.float32))
        pos = np.arange(s)
        # window idxs: last `window` raw positions
        wsz = min(s, self.window)
        widx = np.clip(pos[:, None] - self.window + 1, 0, None) + np.arange(wsz)[None, :]
        widx = np.where(widx > pos[:, None], -1, widx)

        all_kv = kv_np
        idxs = widx
        if ratio:
            comp = self.compressor(x_np, pfx + ".compressor", ratio, self.head_dim, False, cos, sin)
            if comp.shape[0] > 0:
                all_kv = np.concatenate([kv_np, comp], axis=0)
                if ratio == 4:
                    qr_np = np.array(qr.astype(mx.float32))
                    cidx = self.indexer(x_np, qr_np, li, s, cos, sin)
                else:
                    n_slots = comp.shape[0]
                    ci = np.arange(n_slots)[None, :].repeat(s, axis=0)
                    vis = ci < ((pos[:, None] + 1) // ratio)
                    cidx = np.where(vis, ci + s, -1)
                idxs = np.concatenate([widx, cidx], axis=1)

        # gathered attention with sink in the denominator
        sink = np.array(w[pfx + ".attn_sink"]).astype(np.float32)  # [h]
        safe = np.maximum(idxs, 0)
        gk = all_kv[safe]                       # [s, t, d]
        scores = np.einsum("shd,std->sht", q_np, gk) * (self.head_dim ** -0.5)
        scores = np.where((idxs != -1)[:, None, :], scores, -np.inf)
        m = scores.max(axis=-1, keepdims=True)
        m = np.maximum(m, sink[None, :, None])  # sink participates in the max
        e = np.exp(scores - m)
        denom = e.sum(axis=-1) + np.exp(sink[None, :] - m[..., 0])
        o = np.einsum("sht,std->shd", e, gk) / denom[..., None]
        o[..., -self.rd:] = apply_rope(o[..., -self.rd:], cos[:s], sin[:s], inverse=True)

        # grouped low-rank O in BF16 (the torch reference einsums wo_a in
        # bf16 — "using BF16 for simplicity" — and the Zig engine matches it)
        og, ol = self.o_groups, self.o_lora
        gin = (self.n_heads // og) * self.head_dim
        o = o.reshape(s, og, gin).transpose(1, 0, 2)  # [og, s, gin]
        wo_a = QLinear(w, pfx + ".wo_a", self.qcfg).dequant().astype(mx.bfloat16)
        wo_a = wo_a.reshape(og, ol, gin).transpose(0, 2, 1)  # [og, gin, ol]
        ob = mx.array(o.astype(np.float32)).astype(mx.bfloat16)
        red = mx.matmul(ob, wo_a)  # [og, s, ol]
        o = np.array(red.astype(mx.float32)).transpose(1, 0, 2).reshape(s, og * ol)
        ob2 = mx.array(o).astype(mx.bfloat16)
        return QLinear(w, pfx + ".wo_b", self.qcfg)(ob2)

    def moe(self, x, li, input_ids, pfx=None):
        w = self.weights
        if pfx is None:
            pfx = f"layers.{li}.ffn"
        s = x.shape[0]
        xf = np.array(x.astype(mx.float32))
        gate_w = np.array(w[pfx + ".gate.weight"].astype(mx.float32))
        logits = xf @ gate_w.T
        scores = np.sqrt(np.log1p(np.exp(-np.abs(logits))) + np.maximum(logits, 0.0))
        if li < self.n_hash:
            tid2eid = np.array(w[pfx + ".gate.tid2eid"])
            indices = tid2eid[np.array(input_ids)].astype(np.int64)
        else:
            bias = np.array(w[pfx + ".gate.bias"])
            sel = scores + bias[None, :]
            indices = np.argpartition(-sel, kth=self.topk - 1, axis=-1)[:, :self.topk]
        wts = np.take_along_axis(scores, indices, axis=-1)
        wts = wts / wts.sum(axis=-1, keepdims=True)
        wts = wts * self.route_scale

        qc = self.qcfg[pfx + ".experts.w1"]
        qc2 = self.qcfg[pfx + ".experts.w2"]
        xe = mx.expand_dims(x, (1, 2))          # [s, 1, 1, d]
        ind = mx.array(indices.astype(np.int32))
        lim = self.swiglu_limit
        gate = mx.gather_qmm(xe, w[pfx + ".experts.w1.weight"], w[pfx + ".experts.w1.scales"],
                             w[pfx + ".experts.w1.biases"], rhs_indices=ind, transpose=True,
                             group_size=qc["group_size"], bits=qc["bits"])
        up = mx.gather_qmm(xe, w[pfx + ".experts.w3.weight"], w[pfx + ".experts.w3.scales"],
                           w[pfx + ".experts.w3.biases"], rhs_indices=ind, transpose=True,
                           group_size=qc["group_size"], bits=qc["bits"])
        gate = gate.astype(mx.float32)
        up = mx.clip(up.astype(mx.float32), -lim, lim)
        gate = mx.minimum(gate, lim)
        act = (gate * mx.sigmoid(gate) * up).astype(x.dtype)
        down = mx.gather_qmm(act, w[pfx + ".experts.w2.weight"], w[pfx + ".experts.w2.scales"],
                             w[pfx + ".experts.w2.biases"], rhs_indices=ind, transpose=True,
                             group_size=qc2["group_size"], bits=qc2["bits"])
        down = down.squeeze(2)                  # [s, k, d]
        routed = (down.astype(mx.float32) * mx.array(wts[..., None].astype(np.float32))).sum(axis=1)

        # shared expert (clipped swiglu too — the reference passes the limit)
        sg = QLinear(w, pfx + ".shared_experts.w1", self.qcfg)(x).astype(mx.float32)
        su = mx.clip(QLinear(w, pfx + ".shared_experts.w3", self.qcfg)(x).astype(mx.float32), -lim, lim)
        sg = mx.minimum(sg, lim)
        sact = (sg * mx.sigmoid(sg) * su).astype(x.dtype)
        shared = QLinear(w, pfx + ".shared_experts.w2", self.qcfg)(sact)
        return (routed + shared.astype(mx.float32)).astype(x.dtype)

    def hc_pre(self, h_np, fn, scale, base):
        """h_np [s, hc, d] f32 -> (y [s,d] f32, post, comb)."""
        s = h_np.shape[0]
        flat = h_np.reshape(s, -1)
        rsq = 1.0 / np.sqrt((flat ** 2).mean(axis=-1, keepdims=True) + self.eps)
        mixes = (flat @ fn.T) * rsq
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, self.hc, self.hc_iters, self.hc_eps)
        y = (pre[..., None] * h_np).sum(axis=1)
        return y, post, comb

    @staticmethod
    def hc_post(out_np, residual_np, post, comb):
        """out [s,d], residual [s,hc,d], comb[j,k]: y[k] = post[k]*out + sum_j comb[j,k]*res[j]."""
        return post[..., None] * out_np[:, None, :] + np.einsum("sjk,sjd->skd", comb, residual_np)

    trace = None  # dict to collect per-component activations (fixture dump)

    def _tr(self, key, arr):
        if self.trace is not None:
            self.trace[key] = np.asarray(arr, dtype=np.float32).tolist()

    def forward(self, token_ids, capture_main=False):
        """capture_main: also return main_hidden [s, n_targets*dim] — the
        stream averaged over the hc copies at each dspark target layer,
        concatenated (the reference forward's third return)."""
        w = self.weights
        s = len(token_ids)
        ids = mx.array(np.asarray(token_ids, dtype=np.int32))
        h = self.embed[ids].astype(mx.bfloat16)          # [s, d]
        h_np = np.array(h.astype(mx.float32))
        stream = np.repeat(h_np[:, None, :], self.hc, axis=1)  # [s, hc, d]
        main_hiddens = []

        for li in range(self.n_layers):
            ratio = self.ratios[li]
            fn_a = np.array(w[f"layers.{li}.hc_attn_fn"])
            base_a = np.array(w[f"layers.{li}.hc_attn_base"])
            scale_a = np.array(w[f"layers.{li}.hc_attn_scale"])
            y, post, comb = self.hc_pre(stream, fn_a, scale_a, base_a)
            self._tr(f"l{li}.hc_attn.y", y)
            self._tr(f"l{li}.hc_attn.post", post)
            self._tr(f"l{li}.hc_attn.comb", comb)
            yb = rms_norm(mx.array(y).astype(mx.bfloat16), w[f"layers.{li}.attn_norm.weight"], self.eps)
            attn_out = self.attention(yb, li, ratio, s)
            self._tr(f"l{li}.attn_out", np.array(attn_out.astype(mx.float32)))
            stream = self.hc_post(np.array(attn_out.astype(mx.float32)), stream, post, comb)

            fn_f = np.array(w[f"layers.{li}.hc_ffn_fn"])
            base_f = np.array(w[f"layers.{li}.hc_ffn_base"])
            scale_f = np.array(w[f"layers.{li}.hc_ffn_scale"])
            y, post, comb = self.hc_pre(stream, fn_f, scale_f, base_f)
            yb = rms_norm(mx.array(y).astype(mx.bfloat16), w[f"layers.{li}.ffn_norm.weight"], self.eps)
            ffn_out = self.moe(yb, li, ids)
            self._tr(f"l{li}.ffn_out", np.array(ffn_out.astype(mx.float32)))
            stream = self.hc_post(np.array(ffn_out.astype(mx.float32)), stream, post, comb)
            self._tr(f"l{li}.stream", stream)
            if capture_main and li in self.dspark_targets:
                main_hiddens.append(stream.mean(axis=1))  # mean over hc copies
            mx.clear_cache()

        # hyper-head collapse: sigmoid weights only
        flat = stream.reshape(s, -1)
        rsq = 1.0 / np.sqrt((flat ** 2).mean(axis=-1, keepdims=True) + self.eps)
        fn = np.array(w["hc_head_fn"])
        base = np.array(w["hc_head_base"])
        scale = np.array(w["hc_head_scale"])
        mixes = (flat @ fn.T) * rsq
        pre = 1.0 / (1.0 + np.exp(-(mixes * scale[0] + base))) + self.hc_eps
        hout = (pre[..., None] * stream).sum(axis=1)     # [s, d]

        hb = rms_norm(mx.array(hout[-1:]).astype(mx.bfloat16), w["norm.weight"], self.eps)
        logits = self.head(hb.astype(mx.float32))
        mx.eval(logits)
        if capture_main:
            return np.array(logits)[0], np.concatenate(main_hiddens, axis=-1)
        return np.array(logits)[0]

    # ---- DSpark (block-parallel draft stages, transcribed from forward_spec) ----

    def _wo(self, o_np, apfx):
        """Grouped low-rank O tail shared with trunk attention: [s,h,hd] f32 ->
        [s,dim] bf16 mlx (wo_a einsum in bf16, then wo_b)."""
        w = self.weights
        s = o_np.shape[0]
        og, ol = self.o_groups, self.o_lora
        gin = (self.n_heads // og) * self.head_dim
        o = o_np.reshape(s, og, gin).transpose(1, 0, 2)  # [og, s, gin]
        wo_a = QLinear(w, apfx + ".wo_a", self.qcfg).dequant().astype(mx.bfloat16)
        wo_a = wo_a.reshape(og, ol, gin).transpose(0, 2, 1)  # [og, gin, ol]
        ob = mx.array(o.astype(np.float32)).astype(mx.bfloat16)
        red = mx.matmul(ob, wo_a)  # [og, s, ol]
        o = np.array(red.astype(mx.float32)).transpose(1, 0, 2).reshape(s, og * ol)
        ob2 = mx.array(o).astype(mx.bfloat16)
        return QLinear(w, apfx + ".wo_b", self.qcfg)(ob2)

    def dspark_attention(self, x, apfx, main_x_win, lo, start_pos):
        """DSparkAttention start_pos>0 (stateless): x [B, dim] bf16 = the
        attn-normed draft block at positions start_pos+1..start_pos+B;
        main_x_win [w, dim] bf16 = main_x for ring positions lo..start_pos.
        Ring semantics collapse statelessly: every draft position attends the
        SAME set — all ring slots plus the whole draft block (block-parallel,
        NO causal mask inside the block). Plain rope (ratio 0)."""
        w = self.weights
        B = x.shape[0]
        cos, sin = self.freqs("plain", start_pos + 1 + B)

        # ring: this stage's main_kv at absolute positions lo..start_pos
        mkv = rms_norm(QLinear(w, apfx + ".wkv", self.qcfg)(main_x_win), w[apfx + ".kv_norm.weight"], self.eps)
        mkv_np = np.array(mkv.astype(mx.float32))
        mkv_np[:, -self.rd:] = apply_rope(mkv_np[:, -self.rd:], cos[lo:start_pos + 1], sin[lo:start_pos + 1])
        mkv_np[:, :-self.rd] = fp8_sim(mkv_np[:, :-self.rd], 64)

        dcos = cos[start_pos + 1:start_pos + 1 + B]
        dsin = sin[start_pos + 1:start_pos + 1 + B]
        qr = rms_norm(QLinear(w, apfx + ".wq_a", self.qcfg)(x), w[apfx + ".q_norm.weight"], self.eps)
        q = QLinear(w, apfx + ".wq_b", self.qcfg)(qr)
        q_np = np.array(q.astype(mx.float32)).reshape(B, self.n_heads, self.head_dim)
        q_np = q_np * (1.0 / np.sqrt((q_np ** 2).mean(axis=-1, keepdims=True) + self.eps))
        q_np[..., -self.rd:] = apply_rope(q_np[..., -self.rd:], dcos, dsin)

        kv = rms_norm(QLinear(w, apfx + ".wkv", self.qcfg)(x), w[apfx + ".kv_norm.weight"], self.eps)
        kv_np = np.array(kv.astype(mx.float32))
        kv_np[:, -self.rd:] = apply_rope(kv_np[:, -self.rd:], dcos, dsin)
        kv_np[:, :-self.rd] = fp8_sim(kv_np[:, :-self.rd], 64)

        all_kv = np.concatenate([mkv_np, kv_np], axis=0)  # [w+B, hd]
        sink = np.array(w[apfx + ".attn_sink"]).astype(np.float32)
        scores = np.einsum("bhd,td->bht", q_np, all_kv) * (self.head_dim ** -0.5)
        m = scores.max(axis=-1, keepdims=True)
        m = np.maximum(m, sink[None, :, None])
        e = np.exp(scores - m)
        denom = e.sum(axis=-1) + np.exp(sink[None, :] - m[..., 0])
        o = np.einsum("bht,td->bhd", e, all_kv) / denom[..., None]
        o[..., -self.rd:] = apply_rope(o[..., -self.rd:], dcos, dsin, inverse=True)
        return self._wo(o, apfx)

    def dspark_spec(self, trunk_tok, main_hidden, start_pos):
        """forward_spec at start_pos>0, greedy (temperature 0). main_hidden
        [start_pos+1, n_targets*dim] covers EVERY position 0..start_pos so the
        stateless ring is just its last min(window, start_pos+1) rows.
        Returns (output_ids [B+1], logits [B, V] f32, confidence [B] f32)."""
        w = self.weights
        B = self.dspark_block
        n = start_pos + 1
        lo = max(0, n - self.window)
        last = self.n_mtp - 1

        # stage-0 main projection + norm, once, shared by every stage's wkv
        mh = mx.array(main_hidden[lo:].astype(np.float32)).astype(mx.bfloat16)
        main_x = rms_norm(QLinear(w, "mtp.0.main_proj", self.qcfg)(mh), w["mtp.0.main_norm.weight"], self.eps)

        # draft ids: [trunk_token, noise, noise, ...] -> embed -> hc copies
        draft_ids = np.full(B, self.dspark_noise, dtype=np.int64)
        draft_ids[0] = trunk_tok
        x = self.embed[mx.array(draft_ids.astype(np.int32))].astype(mx.bfloat16)
        x_np = np.array(x.astype(mx.float32))
        stream = np.repeat(x_np[:, None, :], self.hc, axis=1)  # [B, hc, d]
        ids_mx = mx.array(draft_ids.astype(np.int32))  # scored routing ignores it

        for st in range(self.n_mtp):
            p = f"mtp.{st}"
            fn_a = np.array(w[f"{p}.hc_attn_fn"])
            base_a = np.array(w[f"{p}.hc_attn_base"])
            scale_a = np.array(w[f"{p}.hc_attn_scale"])
            y, post, comb = self.hc_pre(stream, fn_a, scale_a, base_a)
            yb = rms_norm(mx.array(y).astype(mx.bfloat16), w[f"{p}.attn_norm.weight"], self.eps)
            attn_out = self.dspark_attention(yb, p + ".attn", main_x, lo, start_pos)
            stream = self.hc_post(np.array(attn_out.astype(mx.float32)), stream, post, comb)

            fn_f = np.array(w[f"{p}.hc_ffn_fn"])
            base_f = np.array(w[f"{p}.hc_ffn_base"])
            scale_f = np.array(w[f"{p}.hc_ffn_scale"])
            y, post, comb = self.hc_pre(stream, fn_f, scale_f, base_f)
            yb = rms_norm(mx.array(y).astype(mx.bfloat16), w[f"{p}.ffn_norm.weight"], self.eps)
            ffn_out = self.moe(yb, self.n_layers + st, ids_mx, pfx=p + ".ffn")
            stream = self.hc_post(np.array(ffn_out.astype(mx.float32)), stream, post, comb)

        # forward_head: last stage's OWN hc collapse -> norm -> SHARED trunk head
        flat = stream.reshape(B, -1)
        rsq = 1.0 / np.sqrt((flat ** 2).mean(axis=-1, keepdims=True) + self.eps)
        fn = np.array(w[f"mtp.{last}.hc_head_fn"])
        base = np.array(w[f"mtp.{last}.hc_head_base"])
        scale = np.array(w[f"mtp.{last}.hc_head_scale"])
        mixes = (flat @ fn.T) * rsq
        pre = 1.0 / (1.0 + np.exp(-(mixes * scale[0] + base))) + self.hc_eps
        hout = (pre[..., None] * stream).sum(axis=1)  # [B, d]  (pre-norm: confidence reads THIS)

        hb = rms_norm(mx.array(hout).astype(mx.bfloat16), w[f"mtp.{last}.norm.weight"], self.eps)
        logits = np.array(self.head(hb.astype(mx.float32)))  # [B, V] f32

        # sequential Markov bigram bias + greedy sample
        w1 = np.array(w[f"mtp.{last}.markov_head.markov_w1.weight"].astype(mx.float32))  # [V, rank]
        w2 = np.array(w[f"mtp.{last}.markov_head.markov_w2.weight"].astype(mx.float32))
        out_ids = np.empty(B + 1, dtype=np.int64)
        out_ids[0] = trunk_tok
        membeds = []
        for i in range(B):
            emb = w1[out_ids[i]]          # embed of the PREVIOUS emitted id
            logits[i] += emb @ w2.T
            membeds.append(emb)
            out_ids[i + 1] = int(np.argmax(logits[i]))

        conf_w = np.array(w[f"mtp.{last}.confidence_head.proj.weight"].astype(mx.float32))  # [1, d+rank]
        hidden = np.concatenate([hout, np.stack(membeds)], axis=-1)  # [B, d+rank]
        confidence = (hidden @ conf_w.T)[:, 0]
        return out_ids, logits, confidence


def fabricate(out_dir, seed=11):
    """Miniature random dsv4 checkpoint exercising every code path: one hash
    layer, ratio-4 overlap compressor + indexer, a plain non-overlap ratio,
    hc machinery, sink. All dims chosen so the fp8(gs64)/fp4(gs32) sims and
    affine gs32 divide cleanly."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    D, H, HD, RD = 64, 4, 96, 32
    QL, OL, OG = 32, 16, 2
    IH, IHD = 2, 32
    E, K, MI = 8, 2, 32
    L, V = 4, 64
    N_MTP, DS_BLOCK, DS_RANK = 3, 3, 16
    DS_TARGETS = [1, 3]  # two targets, not three — the concat must not hardcode 3
    # Stage count rides the ratio table like the real mirror (trunk + N_MTP
    # zeros; the release ships NO n_mtp_layers key and a stale
    # num_nextn_predict_layers=1 against 3 stages).
    ratios = [0, 4, 16, 4] + [0] * N_MTP
    cfg = {
        "model_type": "deepseek_v4", "hidden_size": D, "num_hidden_layers": L,
        "num_attention_heads": H, "num_key_value_heads": 1, "head_dim": HD,
        "qk_rope_head_dim": RD, "q_lora_rank": QL, "o_lora_rank": OL, "o_groups": OG,
        "sliding_window": 8, "compress_ratios": ratios, "compress_rope_theta": 160000.0,
        "rope_theta": 10000.0,
        "rope_scaling": {"factor": 16, "original_max_position_embeddings": 64,
                         "beta_fast": 32, "beta_slow": 1, "type": "yarn"},
        "index_n_heads": IH, "index_head_dim": IHD, "index_topk": 4,
        "n_routed_experts": E, "num_experts_per_tok": K, "num_hash_layers": 1,
        "n_shared_experts": 1, "moe_intermediate_size": MI,
        "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "norm_topk_prob": True,
        "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "rms_norm_eps": 1e-6,
        "vocab_size": V, "max_position_embeddings": 4096,
        "bos_token_id": 0, "eos_token_id": 1,
        "dspark_block_size": DS_BLOCK, "dspark_noise_token_id": V - 1,
        "dspark_target_layer_ids": DS_TARGETS, "dspark_markov_rank": DS_RANK,
    }
    qcfg = {"group_size": 32, "bits": 8, "mode": "affine"}
    W = {}

    def q(name, out_d, in_d, scale=0.05):
        w = (rng.standard_normal((out_d, in_d)) * scale).astype(np.float32)
        wq, s, b = mx.quantize(mx.array(w).astype(mx.bfloat16), group_size=32, bits=8)
        W[name + ".weight"] = wq
        W[name + ".scales"] = s
        W[name + ".biases"] = b
        qcfg[name] = {"group_size": 32, "bits": 8, "mode": "affine"}

    def raw(name, arr, dtype=mx.float32):
        W[name] = mx.array(arr.astype(np.float32)).astype(dtype)

    q("embed", V, D)
    q("head", V, D)
    raw("norm.weight", np.ones(D), mx.bfloat16)
    mix = (2 + 4) * 4
    raw("hc_head_fn", rng.standard_normal((4, 4 * D)) * 0.02)
    raw("hc_head_base", rng.standard_normal(4) * 0.1)
    raw("hc_head_scale", np.array([0.5]))
    for li in range(L):
        p = f"layers.{li}"
        for nm in ("attn_norm", "ffn_norm"):
            raw(f"{p}.{nm}.weight", np.ones(D), mx.bfloat16)
        for tag in ("attn", "ffn"):
            raw(f"{p}.hc_{tag}_fn", rng.standard_normal((mix, 4 * D)) * 0.02)
            raw(f"{p}.hc_{tag}_base", rng.standard_normal(mix) * 0.1)
            raw(f"{p}.hc_{tag}_scale", np.abs(rng.standard_normal(3)) * 0.5 + 0.2)
        a = f"{p}.attn"
        q(f"{a}.wq_a", QL, D)
        q(f"{a}.wq_b", H * HD, QL)
        q(f"{a}.wkv", HD, D)
        q(f"{a}.wo_a", OG * OL, H * HD // OG)
        q(f"{a}.wo_b", D, OG * OL)
        raw(f"{a}.q_norm.weight", np.ones(QL), mx.bfloat16)
        raw(f"{a}.kv_norm.weight", np.ones(HD), mx.bfloat16)
        raw(f"{a}.attn_sink", rng.standard_normal(H) * 0.5)
        ratio = ratios[li]
        if ratio:
            coff = 2 if ratio == 4 else 1
            c = f"{a}.compressor"
            raw(f"{c}.wkv.weight", rng.standard_normal((coff * HD, D)) * 0.05, mx.bfloat16)
            raw(f"{c}.wgate.weight", rng.standard_normal((coff * HD, D)) * 0.05, mx.bfloat16)
            raw(f"{c}.ape", rng.standard_normal((ratio, coff * HD)) * 0.1)
            raw(f"{c}.norm.weight", np.ones(HD), mx.bfloat16)
            if ratio == 4:
                ix = f"{a}.indexer"
                q(f"{ix}.wq_b", IH * IHD, QL)
                raw(f"{ix}.weights_proj.weight", rng.standard_normal((IH, D)) * 0.05, mx.bfloat16)
                ic = f"{ix}.compressor"
                raw(f"{ic}.wkv.weight", rng.standard_normal((2 * IHD, D)) * 0.05, mx.bfloat16)
                raw(f"{ic}.wgate.weight", rng.standard_normal((2 * IHD, D)) * 0.05, mx.bfloat16)
                raw(f"{ic}.ape", rng.standard_normal((ratio, 2 * IHD)) * 0.1)
                raw(f"{ic}.norm.weight", np.ones(IHD), mx.bfloat16)
        f = f"{p}.ffn"
        raw(f"{f}.gate.weight", rng.standard_normal((E, D)) * 0.05, mx.bfloat16)
        if li < 1:
            W[f"{f}.gate.tid2eid"] = mx.array(rng.integers(0, E, size=(V, K)).astype(np.int64))
        else:
            raw(f"{f}.gate.bias", rng.standard_normal(E) * 0.1)
        for proj, (od, idim) in (("w1", (MI, D)), ("w3", (MI, D)), ("w2", (D, MI))):
            bank = (rng.standard_normal((E, od, idim)) * 0.05).astype(np.float32)
            wq, s, b = mx.quantize(mx.array(bank).astype(mx.bfloat16), group_size=32, bits=8)
            W[f"{f}.experts.{proj}.weight"] = wq
            W[f"{f}.experts.{proj}.scales"] = s
            W[f"{f}.experts.{proj}.biases"] = b
            qcfg[f"{f}.experts.{proj}"] = {"group_size": 32, "bits": 8, "mode": "affine"}
            q(f"{f}.shared_experts.{proj}", od, idim)
    # DSpark stages (mtp.{i}.*): trunk-layer-shaped (ratio 0 -> no compressor)
    # plus stage-0 main_proj/main_norm and last-stage head extras — the exact
    # namespace the 0731 mirror ships. Scored MoE only (layer_id >= n_hash).
    for st in range(N_MTP):
        p = f"mtp.{st}"
        for nm in ("attn_norm", "ffn_norm"):
            raw(f"{p}.{nm}.weight", np.ones(D), mx.bfloat16)
        for tag in ("attn", "ffn"):
            raw(f"{p}.hc_{tag}_fn", rng.standard_normal((mix, 4 * D)) * 0.02)
            raw(f"{p}.hc_{tag}_base", rng.standard_normal(mix) * 0.1)
            raw(f"{p}.hc_{tag}_scale", np.abs(rng.standard_normal(3)) * 0.5 + 0.2)
        a = f"{p}.attn"
        q(f"{a}.wq_a", QL, D)
        q(f"{a}.wq_b", H * HD, QL)
        q(f"{a}.wkv", HD, D)
        q(f"{a}.wo_a", OG * OL, H * HD // OG)
        q(f"{a}.wo_b", D, OG * OL)
        raw(f"{a}.q_norm.weight", np.ones(QL), mx.bfloat16)
        raw(f"{a}.kv_norm.weight", np.ones(HD), mx.bfloat16)
        raw(f"{a}.attn_sink", rng.standard_normal(H) * 0.5)
        f = f"{p}.ffn"
        raw(f"{f}.gate.weight", rng.standard_normal((E, D)) * 0.05, mx.bfloat16)
        raw(f"{f}.gate.bias", rng.standard_normal(E) * 0.1)
        for proj, (od, idim) in (("w1", (MI, D)), ("w3", (MI, D)), ("w2", (D, MI))):
            bank = (rng.standard_normal((E, od, idim)) * 0.05).astype(np.float32)
            wq, s, b = mx.quantize(mx.array(bank).astype(mx.bfloat16), group_size=32, bits=8)
            W[f"{f}.experts.{proj}.weight"] = wq
            W[f"{f}.experts.{proj}.scales"] = s
            W[f"{f}.experts.{proj}.biases"] = b
            qcfg[f"{f}.experts.{proj}"] = {"group_size": 32, "bits": 8, "mode": "affine"}
            q(f"{f}.shared_experts.{proj}", od, idim)
        if st == 0:
            q(f"{p}.main_proj", D, D * len(DS_TARGETS))
            raw(f"{p}.main_norm.weight", np.ones(D), mx.bfloat16)
        if st == N_MTP - 1:
            raw(f"{p}.norm.weight", np.ones(D), mx.bfloat16)
            # bf16 like the mirror; the reference upcasts at load/run
            raw(f"{p}.markov_head.markov_w1.weight", rng.standard_normal((V, DS_RANK)) * 0.1, mx.bfloat16)
            raw(f"{p}.markov_head.markov_w2.weight", rng.standard_normal((V, DS_RANK)) * 0.1, mx.bfloat16)
            raw(f"{p}.confidence_head.proj.weight", rng.standard_normal((1, D + DS_RANK)) * 0.1, mx.bfloat16)
            raw(f"{p}.hc_head_fn", rng.standard_normal((4, 4 * D)) * 0.02)
            raw(f"{p}.hc_head_base", rng.standard_normal(4) * 0.1)
            raw(f"{p}.hc_head_scale", np.array([0.5]))
    mx.eval(list(W.values()))
    shard = "model-mini.safetensors"
    mx.save_safetensors(os.path.join(out_dir, shard), W, metadata={"format": "mlx"})
    json.dump({"metadata": {}, "weight_map": {k: shard for k in W}},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    cfg["quantization"] = qcfg
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    print(f"fabricated miniature at {out_dir} ({len(W)} tensors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--fabricate", type=str, help="write a miniature random checkpoint here")
    ap.add_argument("--smoke", action="store_true", help="run a forward on the miniature")
    ap.add_argument("--dump-fixtures", type=str, help="run s=17 on --model, dump per-component activations as JSON")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--layers", type=int, default=None, help="truncate to N layers (smoke)")
    ap.add_argument("--raw", action="store_true", help="no chat encoding, just BOS + text")
    ap.add_argument("--encoding-dir", default=os.path.expanduser("~/.mlx-serve/staging/dsv4-ref/encoding"))
    args = ap.parse_args()

    if args.fabricate:
        fabricate(os.path.expanduser(args.fabricate))
        if args.smoke:
            model = Dsv4Ref(os.path.expanduser(args.fabricate))
            rng = np.random.default_rng(0)
            for s in (3, 5, 17, 33):  # crosses window=8, ratio 4 and ratio 16 boundaries
                ids = rng.integers(0, model.cfg["vocab_size"], size=s).tolist()
                logits = model.forward(ids)
                ok = np.isfinite(logits).all()
                print(f"smoke s={s}: logits[{logits.shape}] finite={ok} "
                      f"argmax={int(np.argmax(logits))} norm={np.linalg.norm(logits):.3f}")
                assert ok, f"non-finite logits at s={s}"
            # DSpark: pre- and post-window-wrap start positions
            for i in (6, 9):
                ids = rng.integers(0, model.cfg["vocab_size"], size=i + 1).tolist()
                logits, mh = model.forward(ids, capture_main=True)
                assert mh.shape == (i + 1, model.dim * len(model.dspark_targets)), mh.shape
                out_ids, dlog, conf = model.dspark_spec(int(np.argmax(logits)), mh, i)
                ok = np.isfinite(dlog).all() and np.isfinite(conf).all()
                print(f"smoke dspark i={i}: out_ids={out_ids.tolist()} conf={np.round(conf, 3).tolist()} finite={ok}")
                assert ok and out_ids.shape == (model.dspark_block + 1,)
            print("SMOKE PASS")
        return

    assert args.model, "--model required"
    model_dir = os.path.expanduser(args.model)

    if args.dump_fixtures:
        model = Dsv4Ref(model_dir, max_layers=args.layers)
        rng = np.random.default_rng(0)
        ids = rng.integers(0, model.cfg["vocab_size"], size=17).tolist()
        model.trace = {"input_ids": ids}
        logits = model.forward(ids)
        model.trace["logits_last"] = logits.astype(np.float32).tolist()
        trace = model.trace
        model.trace = None  # the dspark re-forwards must not clobber the trunk trace
        if model.n_mtp:
            # start positions: ring not full (6), just past the window wrap
            # (9), and a ratio-16 boundary crossing prefix (12)
            for i in (6, 9, 12):
                li, mh = model.forward(ids[:i + 1], capture_main=True)
                tok = int(np.argmax(li))
                out_ids, dlog, conf = model.dspark_spec(tok, mh, i)
                trace[f"dspark.{i}.trunk_tok"] = tok
                trace[f"dspark.{i}.main_hidden_last"] = mh[-1].astype(np.float32).tolist()
                trace[f"dspark.{i}.out_ids"] = out_ids.tolist()
                trace[f"dspark.{i}.logits"] = dlog.astype(np.float32).tolist()
                trace[f"dspark.{i}.confidence"] = conf.astype(np.float32).tolist()
        with open(os.path.expanduser(args.dump_fixtures), "w") as f:
            json.dump(trace, f)
        print(f"dumped {len(trace)} fixture entries to {args.dump_fixtures}")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    if args.raw:
        prompt_ids = tok.encode(args.prompt)
    else:
        sys.path.insert(0, args.encoding_dir)
        from encoding_dsv4 import encode_messages
        text = encode_messages([{"role": "user", "content": args.prompt}], thinking_mode="chat")
        prompt_ids = tok.encode(text)
    print(f"prompt tokens: {len(prompt_ids)}", flush=True)

    t0 = time.time()
    model = Dsv4Ref(model_dir, max_layers=args.layers)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    ids = list(prompt_ids)
    out = []
    for i in range(args.max_tokens):
        t1 = time.time()
        logits = model.forward(ids)
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        out.append(nxt)
        piece = tok.decode(out)
        print(f"[{i}] {time.time()-t1:.1f}s token={nxt} text={piece!r}", flush=True)
        if nxt == tok.eos_token_id:
            break
    print("---")
    print(tok.decode(out))


if __name__ == "__main__":
    main()
