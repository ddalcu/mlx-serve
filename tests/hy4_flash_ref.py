#!/usr/bin/env python3
"""NumPy oracle for the hy_v4_flash TEXT path, transcribed from the reference
modeling code in huggingface/transformers PR #50903 (src/transformers/models/
hy_v4_flash/modeling_hy_v4_flash.py, HYV4Flash* text classes).

Writes a tiny converter-layout pack (bf16, `model.layers.N.*` names) and a
fixture with f32 logits plus per-layer hidden streams. The Zig port reads both.

Usage:
  temp/hy4venv/bin/python -I tests/hy4_flash_ref.py --out /tmp/hy4tiny
"""

import argparse
import json
import math
import os

import numpy as np
from safetensors.numpy import save_file

F = np.float32


def rms(x, w, eps):
    return (x * F(1.0) / np.sqrt((x.astype(F) ** 2).mean(-1, keepdims=True) + eps)).astype(x.dtype) * w


def rms_unweighted(x, eps):
    return (F(1.0) / np.sqrt((x.astype(F) ** 2).mean(-1, keepdims=True) + eps)).astype(x.dtype)


def rms_norm_nw(x, eps):
    """nn.RMSNorm with elementwise_affine=False: x / sqrt(mean(x^2)+eps)."""
    return (x.astype(F) / np.sqrt((x.astype(F) ** 2).mean(-1, keepdims=True) + eps)).astype(x.dtype)


def rope_tables(base, rope_dim, max_pos=4096):
    inv = 1.0 / (base ** (np.arange(0, rope_dim, 2, dtype=F) / rope_dim))
    pos = np.arange(max_pos, dtype=F)[:, None]
    freqs = pos * inv[None, :]
    return np.cos(freqs), np.sin(freqs)


def mrope_cos_sin(cos, sin, mrope_section, positions, rope_dim):
    """Interleaved 3-D M-RoPE for a 1-D text position sequence: all three
    planes read the same positions (the reference expands a 1-plane text step
    to three identical planes)."""
    half = rope_dim // 2
    c = np.array(cos[positions])          # [T, half]
    s = np.array(sin[positions])
    for dim, off in ((1, 1), (2, 2)):
        length = mrope_section[dim] * 3
        idx = np.arange(off, length, 3)
        c[:, idx] = cos[positions][:, idx]
        s[:, idx] = sin[positions][:, idx]
    return np.concatenate((c, c), -1), np.concatenate((s, s), -1)


def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack((-x2, x1), -1).reshape(x.shape)


def apply_rope(q, k, cos, sin):
    # cos/sin [T, rope], q [T, heads, rope] -> [T, heads, rope]
    half = cos.shape[-1] // 2
    c = np.repeat(cos[..., :half], 2, axis=-1)
    s = np.repeat(sin[..., :half], 2, axis=-1)
    kc, ks = (c[:, None, :], s[:, None, :]) if k.ndim == 3 else (c, s)
    return (q * c[:, None, :] + rotate_half(q) * s[:, None, :]).astype(q.dtype), (k * kc + rotate_half(k) * ks).astype(k.dtype)


def apply_partial_rope(x, cos, sin, rope_dim):
    passthrough, rotary = x[..., :-rope_dim], x[..., -rope_dim:]
    both = apply_rope(rotary, rotary, cos, sin)
    r = both[0] if x.ndim == 3 else both[1]
    return np.concatenate((passthrough, r), -1)


class Ref:
    def __init__(self, cfg, weights):
        self.cfg = cfg
        self.w = weights
        self.d = cfg["hidden"]
        self.hc = cfg["hc_mult"]
        self.n = cfg["layers"]
        self.eps = cfg["eps"]
        self.heads_i = cfg["indexed_heads"]
        self.heads_s = cfg["swa_heads"]
        self.rope_i = rope_tables(cfg["indexed_theta"], cfg["rope_dim"])
        self.rope_s = rope_tables(cfg["swa_theta"], cfg["rope_dim"])

    def lin(self, x, key):
        w = self.w[key]
        return x.astype(F) @ w.astype(F).T

    def layer(self, li, streams, positions, cache):
        c, w, d = self.cfg, self.w, self.d
        pfx = f"model.layers.{li}."
        dtype = streams.dtype
        T = streams.shape[0]

        # Engram adds to the streams before attention.
        if li in c["engram_layers"]:
            streams = streams + self.engram(li, streams, c["ids"][:T])

        def hc(prefix):
            fn = w[pfx + prefix + ".hc_fn"].astype(F)        # [8, hc*d]
            base = w[pfx + prefix + ".hc_base"].astype(F)    # [8]
            scale = w[pfx + prefix + ".hc_scale"].astype(F)  # [2]
            flat = streams.astype(F).reshape(T, -1)
            inv = rms_unweighted(flat, c["eps"]).astype(F)
            mixes = (flat @ fn.T) * inv
            pre_l, post_l = mixes[:, :self.hc], mixes[:, self.hc:]
            pre = 1.0 / (1.0 + np.exp(-(pre_l * scale[0] + base[:self.hc]))) + self.eps
            post = c["hc_magnitude"] / (1.0 + np.exp(-(post_l * scale[1] + base[self.hc:]))) + self.eps
            out = (pre[:, :, None] * streams.astype(F)).sum(1)
            return post.astype(dtype), out.astype(dtype)

        # Attention sublayer.
        residual = streams
        post, x = hc("hc_attn_layer.hc_pre")
        x = rms(x, w[pfx + "input_layernorm.weight"], self.eps)
        x = self.attention(li, x, positions, cache)
        x = x.astype(F)
        streams = (post.astype(F)[:, :, None] * x[:, None, :] + residual.astype(F)).astype(dtype)

        # FFN / MoE sublayer.
        residual = streams
        post, x = hc("hc_mlp_layer.hc_pre")
        x = rms(x, w[pfx + "post_attention_layernorm.weight"], self.eps)
        x = self.mlp(li, x).astype(F)
        streams = (post.astype(F)[:, :, None] * x[:, None, :] + residual.astype(F)).astype(dtype)
        return streams

    def attention(self, li, x, positions, cache):
        c, w, d = self.cfg, self.w, self.d
        pfx = f"model.layers.{li}.self_attn."
        indexed = c["layer_types"][li] == "indexed_attention"
        heads = self.heads_i if indexed else self.heads_s
        nope = c["qk_nope"] if indexed else c["swa_qk_nope"]
        rope_dim = c["rope_dim"]
        vd = c["v_dim"] if indexed else c["swa_v_dim"]
        qd = nope + rope_dim
        scale = qd ** -0.5
        T = x.shape[0]
        st = cache.setdefault(li, {})

        q_resid = rms(self.lin(x, pfx + "q_a_proj.weight"), w[pfx + "q_a_layernorm.weight"], self.eps)
        q = self.lin(q_resid, pfx + "q_b_proj.weight").reshape(T, heads, qd)
        qp, qr = q[..., :nope], q[..., nope:]
        gate = self.lin(x, pfx + "linear_gate.weight").reshape(T, heads, vd)

        compressed = self.lin(x, pfx + "kv_a_proj_with_mqa.weight")
        kv_pass, kr = compressed[..., :c["kv_lora"]], compressed[..., c["kv_lora"]:]
        kp = rms(kv_pass, w[pfx + "kv_a_layernorm.weight"], self.eps)

        cos, sin = mrope_cos_sin(*(self.rope_i if indexed else self.rope_s), c["mrope"], positions, rope_dim)
        qr, kr2 = apply_rope(qr[..., :rope_dim], kr[..., :rope_dim], cos, sin)

        # Cache the compressed latents; the reference expands them to full K/V per head.
        if "kp" not in st:
            st["kp"], st["kr"] = kp, kr2
        else:
            st["kp"] = np.concatenate((st["kp"], kp), 0)
            st["kr"] = np.concatenate((st["kr"], kr2), 0)
        kp_all, kr_all = st["kp"], st["kr"]
        S = kp_all.shape[0]
        # kv_b_proj: [heads*(nope+vd), kv_lora]
        kv = kp_all.astype(F) @ w[pfx + "kv_b_proj.weight"].astype(F).T
        kv = kv.reshape(S, heads, nope + vd)
        kn, v = kv[..., :nope], kv[..., nope:]
        k = np.concatenate((kn, np.repeat(kr_all[:, None, :], heads, 1)), -1)

        q_full = np.concatenate((qp, qr), -1)
        # [T, heads, qd] x [S, heads, qd] -> scores [T, heads, S]
        scores = np.einsum("thd,shd->hts", q_full.astype(F), k.astype(F)) * scale
        # Causal mask over cache positions; queries are the last T positions.
        qpos = np.arange(S - T, S)[:, None]
        kpos = np.arange(S)[None, :]
        scores = np.where(kpos <= qpos, scores, -np.inf)
        if not indexed:
            scores = np.where(kpos >= qpos - c["sliding"] + 1, scores, -np.inf)
        else:
            # Compressed indexer selection once the context outgrows the recall budget.
            if S > c["candidate_topk"]:
                scores = self.index_mask(li, x, q_resid, S, T, scores, qpos, cache)
        sinks = w[pfx + "learnable_sink_param"].astype(F)  # [heads]
        mx_ = np.maximum(scores.max(-1, keepdims=True), sinks[:, None, None])
        ex = np.exp(scores - mx_)
        es = np.exp(sinks[:, None, None] - mx_)
        probs = ex / (ex.sum(-1, keepdims=True) + es)
        o = np.einsum("hts,shv->thv", probs, v.astype(F))
        o = o * (1.0 / (1.0 + np.exp(-gate.astype(F))))
        o = (o.reshape(T, -1) @ w[pfx + "o_proj.weight"].astype(F).T)
        o = rms(o, w[pfx + "post_proj_norm.weight"], self.eps)
        return o.astype(x.dtype)

    def index_mask(self, li, x, q_resid, S, T, scores, qpos, cache):
        c, w = self.cfg, self.w
        ip = f"model.layers.{li}.self_attn.compressed_indexer."
        ratio, topk, heads, hd = c["ratio"], c["group_topk"], c["idx_heads"], c["idx_head_dim"]
        tk = rms(self.lin(x, ip + "token_key_projection.weight"), w[ip + "token_key_norm.weight"], self.eps)
        st = cache.setdefault(li, {})
        if "tk" not in st:
            st["tk"] = tk
        else:
            st["tk"] = np.concatenate((st["tk"], tk), 0)
        tk_all = st["tk"]
        g = tk_all[: S // ratio * ratio].reshape(-1, ratio, hd).astype(F)
        val = g @ w[ip + "key_value_projection.weight"].astype(F).T
        gat = g @ w[ip + "key_gate_projection.weight"].astype(F).T + w[ip + "intra_group_ape"].astype(F)
        pooled = (val * np.exp(gat - gat.max(1, keepdims=True)) / np.exp(gat - gat.max(1, keepdims=True)).sum(1, keepdims=True)).sum(1)
        pooled = rms(pooled, w[ip + "key_norm.weight"], self.eps).astype(F)
        q = self.lin(q_resid, ip + "query_projection.weight").reshape(T, heads, hd)
        qcos, qsin = mrope_cos_sin(*self.rope_i, c["mrope"], np.arange(S - T, S), c["rope_dim"])
        kcos, ksin = mrope_cos_sin(*self.rope_i, c["mrope"], np.arange(0, S, ratio)[: pooled.shape[0]], c["rope_dim"])
        q = apply_partial_rope(q, qcos, qsin, c["rope_dim"])
        keys = apply_partial_rope(pooled, kcos, ksin, c["rope_dim"])
        hw = self.lin(x, ip + "head_weight_projection.weight").astype(F) * (heads ** -0.5 * hd ** -0.5)
        sc = np.maximum(q @ keys.T, 0.0)  # [T, heads, G]
        logits = np.einsum("thg,th->tg", sc, hw)
        gid = np.arange(pooled.shape[0])
        logits = np.where(gid[None, :] >= (qpos // ratio), -np.inf, logits)
        recalled = np.argsort(-logits, axis=-1)[:, : min(topk - 1, pooled.shape[0])]
        own = (qpos // ratio).clip(max=pooled.shape[0] - 1)
        groups = np.concatenate((recalled, own), -1)
        # The reference builds a [queries, slots] mask and scatters False into
        # each query's recalled slots; a 1-D union mask would let every query
        # attend to every other query's recalled tokens (and, with T small, to
        # everything). Keep it strictly per query.
        slots = (groups[:, :, None] * ratio + np.arange(ratio)).reshape(T, -1).clip(max=S - 1)
        keep = np.zeros((T, S), bool)
        keep[(np.arange(T)[:, None], slots)] = True
        return np.where(keep[None, :, :], scores, -np.inf)

    def mlp(self, li, x):
        c, w, d = self.cfg, self.w, self.d
        pfx = f"model.layers.{li}.mlp."
        if c["mlp_types"][li] == "dense":
            h = self.lin(x, pfx + "gate_proj.weight")
            u = self.lin(x, pfx + "up_proj.weight")
            y = self.lin((h / (1.0 + np.exp(-h))) * u, pfx + "down_proj.weight")
            return rms(y, w[pfx + "post_fc2_norm.weight"], self.eps)
        # Latent routed MoE.
        scores = self.lin(x, pfx + "gate.weight").astype(F)
        sc = 1.0 / (1.0 + np.exp(-scores)) + w[pfx + "gate.e_score_correction_bias"].astype(F)
        top = np.argsort(-sc, axis=-1)[:, : c["top_k"]]
        tw = np.take_along_axis(1.0 / (1.0 + np.exp(-scores)), top, 1)
        tw = tw / tw.sum(-1, keepdims=True) * c["route_scale"]
        latent = self.lin(x, pfx + "routed_expert_down_proj.weight")
        out = self.moe_experts(li, latent, top, tw) + self.mlp_shared(li, x)
        return rms_norm_nw(out, self.cfg["eps"]) * w[pfx + "combine_output_norm.weight"].astype(F)

    def moe_experts(self, li, latent, top, tw):
        c, w = self.cfg, self.w
        pfx = f"model.layers.{li}.mlp.experts."
        gu_bank = w[pfx + "gate_up_proj"].astype(F)     # [E, 2*inter, latent]
        dn_bank = w[pfx + "down_proj"].astype(F)        # [E, latent, inter]
        out = np.zeros((latent.shape[0], c["latent"]), F)
        lim = c["swiglu_limit"]
        for t in range(latent.shape[0]):
            for j, e in enumerate(top[t]):
                gu = latent[t] @ gu_bank[e].T
                g, u = gu[: c["moe_inter"]], gu[c["moe_inter"] :]
                g, u = np.minimum(g, lim), np.clip(u, -lim, lim)
                act = (g / (1.0 + np.exp(-g))) * u
                out[t] += (act @ dn_bank[e].T) * tw[t, j]
        return self.lin(out, f"model.layers.{li}.mlp.routed_expert_up_proj.weight")

    def mlp_shared(self, li, x):
        c, w, pfx_ = self.cfg, self.w, f"model.layers.{li}.mlp.shared_experts."
        h = self.lin(x, pfx_ + "gate_proj.weight")
        u = self.lin(x, pfx_ + "up_proj.weight")
        return self.lin((h / (1.0 + np.exp(-h))) * u, pfx_ + "down_proj.weight")

    def engram(self, li, streams, ids):
        c, w = self.cfg, self.w
        pfx = f"model.layers.{li}.engram_layer."
        T = streams.shape[0]
        # ngram tokens: [T, max_ngram]; past-start positions use pad.
        ngs = np.stack([np.concatenate((np.full(k, c["pad"]), ids[: T - k])) if k else ids[:T] for k in range(c["max_ngram"])], 1)
        table = w[pfx + "lookup_table"]
        compressed = table[np.clip(ngs, 0, len(table) - 1)]
        mult = w[pfx + "hash_multipliers"]
        mixed = compressed.astype(np.int64) * mult[None, :]
        sizes = w[pfx + "head_vocab_sizes"]  # [orders, heads]
        hid = []
        offset = 0
        for order in range(2, c["max_ngram"] + 1):
            mix = mixed[..., 0].copy()
            for o in range(1, order):
                mix = np.bitwise_xor(mix, mixed[..., o])
            mod = sizes[order - 2]
            for m in mod:
                hid.append((mix % m) + offset)
                offset += int(m)
        hash_ids = np.stack(hid, -1)  # [T, heads]
        emb = w[pfx + "multi_head_embedding.embedding.weight"][hash_ids]  # [T, H, dim]
        if c.get("engram_fp8", False):
            amax = np.abs(emb.astype(F)).max(-1, keepdims=True).clip(min=1e-12)
            q = (emb.astype(F) / amax * 448.0)
            q = np.clip(np.round(q), -448, 448) / 448.0 * amax
            emb = (emb.astype(F) + (q - emb.astype(F))).astype(emb.dtype)
        emb = np.stack([rms(emb[:, h], w[pfx + f"lookup_per_head_norms.{h}.weight"], c["eps"]) for h in range(emb.shape[1])], 1)
        emb = emb.reshape(T, -1)
        value = rms_norm_nw(self.lin(emb, pfx + "value_proj.weight"), c["eps"])
        value = value * w[pfx + "unified_pre_gamma"]
        gates = []
        for hc_i in range(self.hc):
            key = rms(self.lin(emb, pfx + f"key_projs.{hc_i}.weight"), w[pfx + f"norm_key.{hc_i}.weight"], c["eps"])
            query = rms(streams[:, hc_i], w[pfx + f"norm_query.{hc_i}.weight"], c["eps"])
            g = (key.astype(F) * query.astype(F)).sum(-1) * (self.d ** -0.5)
            g = np.sign(g) * np.sqrt(np.abs(g).clip(min=1e-6))
            gates.append(1.0 / (1.0 + np.exp(-g)))
        gates = np.stack(gates, 1)[:, :, None]
        branch = np.stack([value * w[pfx + f"unified_per_hc_scales.{i}"] for i in range(self.hc)], 1)
        return (gates * value[:, None, :] + branch).astype(streams.dtype)

    def forward(self, ids):
        c = self.cfg
        T = len(ids)
        streams = np.repeat(self.w["model.embed_tokens.weight"][ids][:, None, :], self.hc, 1)
        cache = {}
        layers = []
        for li in range(self.n):
            streams = self.layer(li, streams, np.arange(T), cache)
            layers.append(streams.copy())
        fn = self.w["model.hc_head.hc_head_fn"].astype(F)
        base = self.w["model.hc_head.hc_head_base"].astype(F)
        scale = self.w["model.hc_head.hc_head_scale"].astype(F)
        flat = streams.astype(F).reshape(T, -1)
        mixes = (flat @ fn.T) * rms_unweighted(flat, c["eps"]).astype(F)
        pre = 1.0 / (1.0 + np.exp(-(mixes * scale[0] + base))) + self.eps
        out = (pre[:, :, None] * streams.astype(F)).sum(1)
        out = rms(out, self.w["model.norm.weight"], self.eps)
        logits = out.astype(F) @ self.w["lm_head.weight"].astype(F).T
        return logits, layers


def make_weights(c, rng):
    w = {}
    d, T = c["hidden"], c["vocab"]

    def put(name, shape, dtype=F, scale=0.02):
        w[name] = (rng.standard_normal(shape).astype(dtype) * scale).astype(dtype)

    def lin(name, out, inp, scale=0.02):
        put(name, (out, inp), F, scale)

    put("model.embed_tokens.weight", (T, d), F, 0.02)
    put("lm_head.weight", (T, d), F, 0.02)
    put("model.norm.weight", (d,), F, 1.0)
    put("model.hc_head.hc_head_fn", (c["hc_mult"], c["hc_mult"] * d), F, 0.02)
    put("model.hc_head.hc_head_base", (c["hc_mult"],), F, 0.1)
    put("model.hc_head.hc_head_scale", (1,), F, 1.0)
    for li in range(c["layers"]):
        p = f"model.layers.{li}."
        indexed = c["layer_types"][li] == "indexed_attention"
        heads = c["indexed_heads"] if indexed else c["swa_heads"]
        vd = c["v_dim"] if indexed else c["swa_v_dim"]
        put(p + "input_layernorm.weight", (d,), F, 1.0)
        put(p + "post_attention_layernorm.weight", (d,), F, 1.0)
        put(p + "self_attn.q_a_proj.weight", (c["q_lora"], d), F)
        put(p + "self_attn.q_a_layernorm.weight", (c["q_lora"],), F, 1.0)
        put(p + "self_attn.q_b_proj.weight", (heads * (c["qk_nope"] + c["rope_dim"]), c["q_lora"]), F)
        put(p + "self_attn.kv_a_proj_with_mqa.weight", (c["kv_lora"] + c["rope_dim"], d), F)
        put(p + "self_attn.kv_a_layernorm.weight", (c["kv_lora"],), F, 1.0)
        put(p + "self_attn.kv_b_proj.weight", (heads * (c["qk_nope"] + vd), c["kv_lora"]), F)
        put(p + "self_attn.linear_gate.weight", (heads * vd, d), F)
        put(p + "self_attn.o_proj.weight", (d, heads * vd), F)
        put(p + "self_attn.post_proj_norm.weight", (d,), F, 1.0)
        put(p + "self_attn.learnable_sink_param", (heads,), F, 0.1)
        for hc in ("hc_attn_layer.hc_pre", "hc_mlp_layer.hc_pre"):
            put(p + hc + ".hc_fn", (2 * c["hc_mult"], c["hc_mult"] * d), F)
            put(p + hc + ".hc_base", (2 * c["hc_mult"],), F, 0.1)
            put(p + hc + ".hc_scale", (2,), F, 1.0)
        if indexed:
            ip = p + "self_attn.compressed_indexer."
            put(ip + "query_projection.weight", (c["idx_heads"] * c["idx_head_dim"], c["q_lora"]), F)
            put(ip + "head_weight_projection.weight", (c["idx_heads"], d), F)
            put(ip + "token_key_projection.weight", (c["idx_head_dim"], d), F)
            put(ip + "token_key_norm.weight", (c["idx_head_dim"],), F, 1.0)
            put(ip + "key_value_projection.weight", (c["idx_head_dim"], c["idx_head_dim"]), F)
            put(ip + "key_gate_projection.weight", (c["idx_head_dim"], c["idx_head_dim"]), F)
            put(ip + "key_norm.weight", (c["idx_head_dim"],), F, 1.0)
            put(ip + "intra_group_ape", (c["ratio"], c["idx_head_dim"]), F)
        if c["mlp_types"][li] == "dense":
            put(p + "mlp.gate_proj.weight", (c["inter"], d), F)
            put(p + "mlp.up_proj.weight", (c["inter"], d), F)
            put(p + "mlp.down_proj.weight", (d, c["inter"]), F)
            put(p + "mlp.post_fc2_norm.weight", (d,), F, 1.0)
        else:
            put(p + "mlp.gate.weight", (c["experts"], d), F)
            put(p + "mlp.gate.e_score_correction_bias", (c["experts"],), F, 0.01)
            put(p + "mlp.routed_expert_down_proj.weight", (c["latent"], d), F)
            put(p + "mlp.routed_expert_up_proj.weight", (d, c["latent"]), F)
            put(p + "mlp.experts.gate_up_proj", (c["experts"], 2 * c["moe_inter"], c["latent"]), F)
            put(p + "mlp.experts.down_proj", (c["experts"], c["latent"], c["moe_inter"]), F)
            put(p + "mlp.shared_experts.gate_proj.weight", (c["moe_inter"], d), F)
            put(p + "mlp.shared_experts.up_proj.weight", (c["moe_inter"], d), F)
            put(p + "mlp.shared_experts.down_proj.weight", (d, c["moe_inter"]), F)
            put(p + "mlp.combine_output_norm.weight", (d,), F, 1.0)
        if li in c["engram_layers"]:
            ep = p + "engram_layer."
            sizes = np.array(c["engram_sizes"], np.int64)
            put(ep + "lookup_table", (c["lookup_size"],), np.int32, 1.0)
            w[ep + "lookup_table"] = np.arange(c["lookup_size"], dtype=np.int32)
            put(ep + "hash_multipliers", (c["max_ngram"],), np.int64, 1.0)
            w[ep + "hash_multipliers"] = np.array(c["hash_mult"], np.int64)
            put(ep + "head_vocab_sizes", sizes.shape, np.int64, 1.0)
            w[ep + "head_vocab_sizes"] = sizes
            nh = int(sizes.size)
            table_rows = int(sizes.sum())
            put(ep + "multi_head_embedding.embedding.weight", (table_rows, c["head_dim"]), F)
            for h in range(nh):
                put(ep + f"lookup_per_head_norms.{h}.weight", (c["head_dim"],), F, 1.0)
            put(ep + "value_proj.weight", (d, nh * c["head_dim"]), F)
            for h in range(c["hc_mult"]):
                put(ep + f"key_projs.{h}.weight", (d, nh * c["head_dim"]), F)
                put(ep + f"norm_key.{h}.weight", (d,), F, 1.0)
                put(ep + f"norm_query.{h}.weight", (d,), F, 1.0)
                put(ep + f"unified_per_hc_scales.{h}", (d,), F, 1.0)
            put(ep + "unified_pre_gamma", (d,), F, 1.0)
            put(ep + "unified_rmsnorm", (d,), F, 1.0)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--no-indexer", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = dict(
        model_type="hy_v4_flash",
        hidden=32, vocab=64, layers=4, eps=1e-4, hc_mult=2, hc_magnitude=2.0,
        indexed_heads=2, swa_heads=4, qk_nope=8, rope_dim=8, v_dim=8,
        swa_qk_nope=8, swa_v_dim=8, kv_lora=12, q_lora=12,
        indexed_theta=10000.0, swa_theta=1000.0, mrope=[2, 1, 1],
        layer_types=["indexed_attention", "sliding_attention", "indexed_attention", "sliding_attention"],
        mlp_types=["dense", "sparse", "sparse", "sparse"],
        experts=4, top_k=2, moe_inter=16, latent=12, inter=32, route_scale=1.5, swiglu_limit=1.0,
        sliding=4, ratio=2, idx_heads=2, idx_head_dim=16, group_topk=3, candidate_topk=8,
        engram_layers=[1], max_ngram=3, head_dim=8, lookup_size=80, pad=2,
        engram_sizes=[[11, 13], [17, 19]], hash_mult=[3, 5, 7],
    )
    if args.no_engram:
        cfg["engram_layers"] = []
    if args.no_indexer:
        cfg["candidate_topk"] = 100000
    rng = np.random.default_rng(args.seed)
    cfg["ids"] = rng.integers(3, cfg["vocab"] - 1, size=args.seq).tolist()
    w = make_weights(cfg, rng)
    ref = Ref(cfg, w)
    logits, layers = ref.forward(cfg["ids"])

    # The pack must be loadable by the Zig side: cast fp32 to bf16 by truncation (numpy has no bf16).
    model_weights = {
        (k + ".weight" if k.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")) else k): v.astype(np.float32)
        for k, v in w.items()
    }
    save_file(model_weights, os.path.join(args.out, "model.safetensors"))
    fx = {"ids": np.array(cfg["ids"], np.int32), "logits": logits.astype(F)}
    for i, h in enumerate(layers):
        fx[f"layer.{i}"] = h.astype(F)
    save_file(fx, os.path.join(args.out, "fixture.safetensors"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {args.out}: {len(w)} tensors, logits {logits.shape}")


if __name__ == "__main__":
    main()
