#!/usr/bin/env python3
"""Dump Hy4-flash numeric oracle fixtures for the Zig block tests in `src/plugins/hy_v4/`.

USER-RUN (needs torch + a transformers checkout carrying the Hy4-flash modeling PR,
huggingface/transformers#50903). It builds TINY random modules with the REFERENCE PyTorch code,
runs each block on CPU, and writes the weights, inputs and outputs as little-endian f32/i32
`.bin` files beside their dims JSON under `src/plugins/hy_v4/fixtures/num/`. The Zig tests load
them and pin each block at a cos/rms bar, as the vision tower dump does.

The full HF bf16 checkpoint is ~440 GB (trunk ~313 GB + Engram ~127 GB) and cannot be held here;
this script never touches it. Tiny widths are enough to catch a sign, a scale or an ordering bug
in iHC, gated MLA with sinks, the compressed indexer, the latent MoE and the Engram gate.

Usage:
    HYV4_MODELING_DIR=<transformers>/models/hy_v4_flash python tests/dump_hyv4_fixtures.py
or point PYTHONPATH at a transformers install whose `models/hy_v4_flash` carries the PR.
"""

import json
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

# The modeling PR is newer than many installed transformers releases; fill in the few symbols the
# module imports at load time when the installed tree predates them. None of them is exercised by
# the tiny cases below.
import transformers.configuration_utils as cu
if not hasattr(cu, "SubConfigSpec"):
    class SubConfigSpec:
        def __init__(self, **kw):
            self.kw = kw
    cu.SubConfigSpec = SubConfigSpec
if not hasattr(cu, "remap_legacy_layer_types"):
    cu.remap_legacy_layer_types = lambda keys, *a, **k: keys
import transformers.utils.generic as generic
if not hasattr(generic, "get_max_seqlen"):
    generic.get_max_seqlen = lambda *a, **k: None
import transformers.vision_utils as vision_utils
if not hasattr(vision_utils, "get_vision_attention_seqlens"):
    vision_utils.get_vision_attention_seqlens = lambda *a, **k: None

try:
    import transformers.models.hy_v4_flash.modeling_hy_v4_flash as hyref
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig
except ModuleNotFoundError:
    ref = os.environ.get("HYV4_MODELING_DIR")
    if ref is None:
        raise SystemExit("set HYV4_MODELING_DIR to the transformers models/hy_v4_flash folder")
    import importlib.util
    spec = importlib.util.spec_from_file_location("hy_v4_flash.modeling_hy_v4_flash",
                                                  os.path.join(ref, "modeling_hy_v4_flash.py"))
    hyref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hyref)
    cfg_spec = importlib.util.spec_from_file_location("hy_v4_flash.configuration_hy_v4_flash",
                                                      os.path.join(ref, "configuration_hy_v4_flash.py"))
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg_mod)
    HYV4FlashTextConfig = cfg_mod.HYV4FlashTextConfig

# The strict dataclass validator does not know the PR's `indexed_attention` layer type yet.
HYV4FlashTextConfig.validate = lambda self: None

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "plugins", "hy_v4", "fixtures", "num")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(4)


def save(name, tensor):
    if tensor.dtype == torch.int64 or tensor.dtype == torch.int32:
        tensor.detach().cpu().numpy().astype("<i4").tofile(os.path.join(OUT, name + ".bin"))
    else:
        tensor.detach().cpu().float().numpy().astype("<f4").tofile(os.path.join(OUT, name + ".bin"))


def write_json(name, obj):
    json.dump(obj, open(os.path.join(OUT, name + ".json"), "w"), indent=1)


def cfg(**kw):
    base = dict(
        hidden_size=8, intermediate_size=8, moe_intermediate_size=4, num_hidden_layers=2,
        num_attention_heads=2, q_lora_rank=2, kv_lora_rank=2, qk_nope_head_dim=2,
        qk_rope_head_dim=2, v_head_dim=2, hc_mult=4, hc_eps=1e-6, hc_magnitude=2.0,
        rms_norm_eps=1e-5, learnable_sink_init=0.1, swa_num_attention_heads=2,
        swa_qk_nope_head_dim=2, swa_qk_rope_head_dim=2, swa_v_head_dim=2,
        sliding_window=8, max_position_embeddings=16,
        layer_types=["sliding_attention", "sliding_attention"],
        mlp_layer_types=["dense", "sparse"], n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=1, routed_expert_hidden_size=4, routed_scaling_factor=1.5,
        norm_topk_prob=True, n_group=1, topk_group=1, swiglu_limit=10.0,
        compressed_indexer_compress_ratio=2, compressed_indexer_num_attention_heads=2,
        compressed_indexer_head_dim=2, compressed_indexer_group_topk=3,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 0, 0]},
            "indexed_attention": {"rope_type": "default", "rope_theta": 10000000.0, "mrope_section": [1, 0, 0]},
        },
    )
    base.update(kw)
    c = HYV4FlashTextConfig(**base)
    c.layer_types = ["sliding_attention", "indexed_attention"]
    return c


def rand(*shape, lo=-0.5, hi=0.5):
    return torch.rand(*shape) * (hi - lo) + lo


def dense(module):
    """A reference linear as its `[out, in]` weight (what the Zig test transposes)."""
    return module.weight.detach().float()


# ── iHC: the sublayer mix/post gate and the head collapse ──
def dump_ihc():
    c = cfg(hc_mult=4, hidden_size=5)
    torch.manual_seed(11)
    hc = hyref.HYV4FlashHyperConnection(c)
    for p in hc.parameters():
        p.data.uniform_(-0.5, 0.5)
    head = hyref.HYV4FlashHyperHead(cfg(hc_mult=4, hidden_size=5))
    for p in head.parameters():
        p.data.uniform_(-0.5, 0.5)
    stream = rand(1, 2, 4, 5)
    post, mixed = hc(stream)
    out = head(stream)
    save("ihc_fn", hc.fn)
    save("ihc_base", hc.base)
    save("ihc_scale", hc.scale)
    save("ihc_streams", stream)
    save("ihc_post", post)
    save("ihc_mixed", mixed)
    save("ihc_head_fn", head.hc_fn)
    save("ihc_head_base", head.hc_base)
    save("ihc_head_scale", head.hc_scale)
    save("ihc_head_out", out)
    write_json("ihc", {"lanes": 4, "hidden": 5, "rows": 2, "magnitude": c.hc_magnitude,
                       "eps": c.hc_eps, "rms_eps": c.rms_norm_eps})


# ── gated MLA with sinks ──
def dump_mla():
    c = cfg(num_hidden_layers=2)
    torch.manual_seed(12)
    att = hyref.HYV4FlashAttention(c, 0)
    for p in att.parameters():
        p.data.uniform_(-0.5, 0.5)
    att.post_proj_norm.weight.data.uniform_(0.8, 1.2)
    att.q_a_layernorm.weight.data.uniform_(0.8, 1.2)
    att.kv_a_layernorm.weight.data.uniform_(0.8, 1.2)
    x = rand(1, 4, 8)
    pos = torch.arange(4).unsqueeze(0)
    rotary = hyref.HYV4FlashRotaryEmbedding(c)
    pe = rotary(x, pos, "sliding_attention")
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0].masked_fill_(torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1), torch.finfo(torch.float32).min)
    out, _ = att(x, pe, mask)
    save("mla_q_a", dense(att.q_a_proj))
    save("mla_q_a_norm", att.q_a_layernorm.weight)
    save("mla_q_b", dense(att.q_b_proj))
    save("mla_kv_a", dense(att.kv_a_proj_with_mqa))
    save("mla_kv_a_norm", att.kv_a_layernorm.weight)
    save("mla_kv_b", dense(att.kv_b_proj))
    save("mla_gate", dense(att.gate_proj))
    save("mla_o", dense(att.o_proj))
    save("mla_post_norm", att.post_proj_norm.weight)
    save("mla_sink", att.sinks)
    save("mla_x", x)
    save("mla_out", out)
    write_json("mla", {
        "hidden": 8, "heads": att.num_heads, "nope": att.qk_nope_head_dim, "rope": att.qk_rope_head_dim,
        "v_dim": att.v_head_dim, "rank": att.kv_lora_rank, "q_lora": att.q_lora_rank, "rows": 4,
        "theta": c.rope_parameters["sliding_attention"]["rope_theta"], "eps": c.rms_norm_eps,
    })


# ── compressed indexer: pooled groups + recall selection ──
def dump_indexer():
    c = cfg(num_hidden_layers=2)
    torch.manual_seed(13)
    idx = hyref.HYV4FlashCompressedIndexer(c)
    for p in idx.parameters():
        p.data.uniform_(-0.5, 0.5)
    idx.token_key_norm.weight.data.uniform_(0.8, 1.2)
    idx.key_norm.weight.data.uniform_(0.8, 1.2)
    kv_len, rows = 8, 3
    hidden_all = rand(1, kv_len, 8)
    hidden_rows = hidden_all[:, kv_len - rows:]
    token_keys = idx.token_key_norm(idx.token_key_projection(hidden_all))
    group_keys = idx.pool_groups(token_keys)
    q_resid = rand(1, rows, 2)
    pos_full = torch.arange(kv_len).unsqueeze(0)
    pos_rows = torch.arange(kv_len - rows, kv_len).unsqueeze(0)
    gpos = pos_full[..., : group_keys.shape[1] * idx.compress_ratio : idx.compress_ratio]
    qgi = (torch.arange(kv_len - rows, kv_len) // idx.compress_ratio).expand(1, -1)
    recalled = idx(hidden_rows, q_resid, group_keys, gpos, pos_rows, qgi)
    save("idx_query", dense(idx.query_projection))
    save("idx_head_weight", dense(idx.head_weight_projection))
    save("idx_token_key", dense(idx.token_key_projection))
    save("idx_token_key_norm", idx.token_key_norm.weight)
    save("idx_value", idx.key_value_projection.weight)
    save("idx_gate", idx.key_gate_projection.weight)
    save("idx_key_norm", idx.key_norm.weight)
    save("idx_ape", idx.intra_group_ape)
    save("idx_hidden_all", hidden_all)
    save("idx_hidden_rows", hidden_rows)
    save("idx_q_resid", q_resid)
    save("idx_token_keys_all", token_keys)
    save("idx_positions", pos_rows)
    save("idx_recalled", recalled)
    write_json("indexer", {
        "hidden": 8, "q_lora": 2, "heads": idx.num_heads, "dim": idx.head_dim,
        "ratio": idx.compress_ratio, "group_topk": idx.group_topk, "rope": idx.rope_dim,
        "theta": c.rope_parameters["indexed_attention"]["rope_theta"], "eps": c.rms_norm_eps,
        "kv_len": kv_len, "rows": rows, "base_pos": 0,
    })


# ── latent MoE: router, latent experts, shared expert, combine norm ──
def dump_moe():
    c = cfg(num_hidden_layers=2)
    torch.manual_seed(14)
    moe = hyref.HYV4FlashMoE(c)
    for p in moe.parameters():
        p.data.uniform_(-0.5, 0.5)
    moe.gate.e_score_correction_bias.data.uniform_(-0.1, 0.1)
    moe.combine_output_norm.weight.data.uniform_(0.8, 1.2)
    x = rand(1, 3, 8)
    out = moe(x)
    save("moe_gate", moe.gate.weight)
    save("moe_bias", moe.gate.e_score_correction_bias)
    save("moe_routed_down", dense(moe.routed_expert_down_proj))
    save("moe_routed_up", dense(moe.routed_expert_up_proj))
    save("moe_gate_up", moe.experts.gate_up_proj)
    save("moe_down", moe.experts.down_proj)
    save("moe_shared_gate", dense(moe.shared_experts.gate_proj))
    save("moe_shared_up", dense(moe.shared_experts.up_proj))
    save("moe_shared_down", dense(moe.shared_experts.down_proj))
    save("moe_combine_norm", moe.combine_output_norm.weight)
    save("moe_x", x)
    save("moe_out", out)
    write_json("moe", {
        "hidden": 8, "latent": c.routed_expert_hidden_size, "mi": c.moe_intermediate_size,
        "experts": c.n_routed_experts, "top_k": c.num_experts_per_tok, "rows": 3,
        "scale": c.routed_scaling_factor, "swiglu_limit": c.swiglu_limit, "eps": c.rms_norm_eps,
        "shared_mi": c.moe_intermediate_size * c.n_shared_experts,
    })


# ── Engram gate: the memory signal and the per-lane gate/branch ──
def dump_engram():
    c = cfg(num_hidden_layers=2, engram_layer_ids=[1],
            engram_max_ngram_size=3, engram_n_head_per_ngram=2, engram_n_embed_per_ngram=4,
            engram_layer_head_vocab_sizes={"1": [[5, 7], [11, 13]]},
            engram_pad_id=2, engram_lookup_table_size=16, engram_fp8_embedding=False)
    torch.manual_seed(15)
    layer = hyref.HYV4FlashEngramLayer(c, 1)
    layer.to(torch.bfloat16)
    with torch.no_grad():
        for p in layer.parameters():
            p.uniform_(-0.5, 0.5)
        layer.unified_pre_gamma.uniform_(0.8, 1.2)
        for p in layer.unified_per_hc_scales:
            p.uniform_(-0.3, 0.3)
        layer.unified_rmsnorm.eps = None
    rows, lanes, hidden = 3, c.hc_mult, c.hidden_size
    emb_width = layer.value_proj.in_features
    emb = rand(rows, emb_width).to(torch.bfloat16)
    streams = rand(1, rows, lanes, hidden).to(torch.bfloat16)
    with torch.no_grad():
        value = layer.value_proj(emb.unsqueeze(0))
        signal = layer.unified_pre_gamma * layer.unified_rmsnorm(value)
        gates = []
        for k in range(lanes):
            key = layer.norm_key[k](layer.key_projs[k](emb.unsqueeze(0)))
            query = layer.norm_query[k](streams[:, :, k])
            gate = (key * query).sum(dim=-1) * hidden ** -0.5
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates = torch.stack(gates, dim=2).unsqueeze(-1)
        branch = torch.stack([scale * signal for scale in layer.unified_per_hc_scales], dim=2)
        out = gates * signal.unsqueeze(2) + branch
    save("eng_emb", emb)
    save("eng_streams", streams)
    save("eng_gate", out)
    for k in range(lanes):
        save(f"eng_key_w{k}", layer.key_projs[k].weight)
        save(f"eng_key_b{k}", layer.key_projs[k].bias)
        save(f"eng_norm_key{k}", layer.norm_key[k].weight)
        save(f"eng_norm_query{k}", layer.norm_query[k].weight)
    save("eng_value_w", layer.value_proj.weight)
    save("eng_value_b", layer.value_proj.bias)
    save("eng_head_norm", torch.stack([n.weight for n in layer.lookup_per_head_norms]))
    save("eng_pre_gamma", layer.unified_pre_gamma)
    save("eng_hc_scales", torch.stack([s for s in layer.unified_per_hc_scales]))
    write_json("engram", {"hidden": hidden, "lanes": lanes, "heads": layer.num_hash_heads,
                          "head_dim": layer.multi_head_embedding.embedding.embedding_dim,
                          "emb_width": emb_width, "rows": rows})


if __name__ == "__main__":
    dump_ihc()
    dump_mla()
    dump_indexer()
    dump_moe()
    dump_engram()
    print("wrote", OUT)
