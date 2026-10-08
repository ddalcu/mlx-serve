#!/usr/bin/env python3
"""Oracle for deepseek_v41 on a TINY random model: DeepSeek's own
inference/model.py (rev dba1be0a), run in fp32 on the CPU.

The reference's TileLang kernels are replaced by PipeNetwork's CPU stub
(deepseek-v41-mlx tests/kernel_stub.py, MIT), with the three adaptations its
parity test declares: the Engram table stays fp32, the head returns every
position, and an index-key owner reads its own key cache (the reference's
decode otherwise scores ratio-2 owners against the LAST owner's keys).

Writes two packs from one set of random weights, each in a published layout,
with the reference run over that pack's dequantized values:
  <out>/pipe/    pipenetwork's MLX layout: affine 8-bit linears, 4-bit stacked
                 experts (gate/up/down_proj), bf16 wo_a, a 4-bit Engram table
                 in its own shard, engram_token_map.json; no DSpark heads
  <out>/repack/  OpensourceWTF's repack layout: mxfp8 linears (wo_a too), bf16
                 embed/head, flat mxfp8 Engram records under engram/ with the
                 manifest and residents, engram-token-map.u32, DSpark heads with
                 per-expert mxfp4 experts. Its routed experts are stacked affine
                 here (the real pack's EXL3 bank has its own tests).
  <out>/qat.safetensors
                 the three QAT round-trips over inputs with exact midpoints
  <out>/<pack>/fixture.safetensors (and fixture_noqat.safetensors, the same run
                 with the QAT round-trips off on both sides: plain math, tight bar)
                 ids [S+D]; logits_full [S+D, V] (one prefill); logits_decode
                 [D, V] (prefill S, then one token at a time); for the repack
                 also draft_ids [D, B+1], draft_logits [D, B, V], draft_conf [D, B]
                 (the DSpark head after each decode step, greedy).

  venv/bin/python -I tests/dump_dsv41_fixtures.py \\
      --reference <DeepSeek-V4.1-Flash>/inference --kernel-stub <deepseek-v41-mlx>/tests/kernel_stub.py \\
      --out ~/claude-tmp/dsv41-tiny
  DSV41_TINY=~/claude-tmp/dsv41-tiny zig build test -Doptimize=ReleaseFast -Dtest-filter="dsv41 fixture"
"""

import argparse
import importlib.util
import json
import os
import struct
import sys
import types

import numpy as np
import torch
from safetensors.torch import save_file

S, D = 40, 6
VOCAB = 96
STAGES = 3
CFG = dict(
    vocab_size=VOCAB, hidden_size=128, moe_intermediate_size=128, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=1, head_dim=64, qk_rope_head_dim=16,
    q_lora_rank=64, o_lora_rank=32, o_groups=2, swiglu_limit=1.0, rms_norm_eps=1e-20,
    max_position_embeddings=128, rope_theta=10000.0,
    rope_scaling=dict(rope_type="yarn", factor=4, beta_fast=32, beta_slow=1, original_max_position_embeddings=64),
    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=3, scoring_func="sqrtsoftplus",
    topk_method="noaux_tc", norm_topk_prob=True, routed_scaling_factor=1.5, sliding_window=8,
    compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1] + [0] * STAGES, compress_rope_theta=40000.0,
    kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 7], index_n_heads=8, index_head_dim=32,
    index_topk=4, candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, engram_layer_ids=[1, 2], engram_max_ngram_size=4,
    engram_vocab_size=50, engram_n_heads=2, engram_head_dim=64, engram_pad_token_id=2,
    engram_compressed_vocab_size=VOCAB, num_nextn_predict_layers=STAGES, dspark_block_size=4,
    dspark_noise_token_id=VOCAB - 1, dspark_target_layer_ids=[5, 6, 7], dspark_markov_rank=32,
    dspark_n_routed_experts=4, dspark_num_experts_per_tok=2,
)
E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def load_reference(ref_dir, stub_path):
    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace()
    pil.ImageOps = types.SimpleNamespace()
    sys.modules.setdefault("PIL", pil)
    spec = importlib.util.spec_from_file_location("kernel", stub_path)
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)
    sys.modules["kernel"] = stub
    global KERNEL_STUB
    KERNEL_STUB = stub
    sys.path.insert(0, ref_dir)
    import engram as ref_engram
    import model as ref
    ref_engram.build_compressed_token_map = lambda tok: (list(range(VOCAB)), VOCAB)

    def engram_fp32(self, indices):
        v = torch.nn.functional.embedding(indices, self.weight)
        sc = torch.nn.functional.embedding(indices, self.scale)
        return (v.float().unflatten(-1, (-1, self.block_size)) * sc.float().unsqueeze(-1)).flatten(-2)

    ref.ParallelEngramEmbedding.forward = engram_fp32
    head_fwd = ref.ParallelHead.forward
    ref.ParallelHead.forward = lambda self, x, full_logits=False: head_fwd(self, x, True)
    idx_fwd = ref.Indexer.forward

    def own_keys(self, x, qr, latent, start_pos, offset):
        if self.owns_k:
            ref.shared_attn.index_k = self.k_cache
        return idx_fwd(self, x, qr, latent, start_pos, offset)

    ref.Indexer.forward = own_keys
    return ref, ref_engram


def ref_args(ref, ref_engram, stages):
    c = CFG
    rs = c["rope_scaling"]
    args = ref.ModelArgs(
        max_batch_size=1, max_seq_len=c["max_position_embeddings"], temperature=0, dtype="bf16",
        expert_dtype=None, vocab_size=VOCAB, dim=c["hidden_size"], moe_inter_dim=c["moe_intermediate_size"],
        n_layers=c["num_hidden_layers"], n_mtp_layers=stages, n_heads=c["num_attention_heads"],
        n_routed_experts=c["n_routed_experts"], n_activated_experts=c["num_experts_per_tok"],
        route_scale=c["routed_scaling_factor"], swiglu_limit=c["swiglu_limit"], q_lora_rank=c["q_lora_rank"],
        head_dim=c["head_dim"], rope_head_dim=c["qk_rope_head_dim"], norm_eps=c["rms_norm_eps"],
        o_groups=c["o_groups"], o_lora_rank=c["o_lora_rank"], window_size=c["sliding_window"],
        compress_ratios=tuple(c["compress_ratios"]), kv_source_layers=tuple(c["kv_source_layer_ids"]),
        index_source_layers=tuple(c["index_source_layer_ids"]), compress_rope_theta=c["compress_rope_theta"],
        original_seq_len=rs["original_max_position_embeddings"], rope_theta=c["rope_theta"],
        rope_factor=rs["factor"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
        index_n_heads=c["index_n_heads"], index_head_dim=c["index_head_dim"], index_topk=c["index_topk"],
        candidate_source_layer=c["candidate_source_layer_id"], candidate_topk_blocks=c["candidate_topk_blocks"],
        candidate_block_size=c["candidate_block_size"], hc_mult=c["hc_mult"],
        hc_sinkhorn_iters=c["hc_sinkhorn_iters"], hc_eps=c["hc_eps"],
        engram_layer_ids=tuple(c["engram_layer_ids"]), engram_max_ngram_size=c["engram_max_ngram_size"],
        engram_vocab_size=c["engram_vocab_size"], engram_n_heads=c["engram_n_heads"],
        engram_head_dim=c["engram_head_dim"], engram_pad_id=c["engram_pad_token_id"],
        engram_compressed_vocab_size=VOCAB, vision_n_layers=0,
        dspark_block_size=c["dspark_block_size"] if stages else 0, dspark_noise_token_id=c["dspark_noise_token_id"],
        dspark_target_layer_ids=tuple(c["dspark_target_layer_ids"]), dspark_markov_rank=c["dspark_markov_rank"],
        dspark_n_routed_experts=c["dspark_n_routed_experts"],
        dspark_n_activated_experts=c["dspark_num_experts_per_tok"],
    )
    layout = ref_engram.EngramLayout.from_args(types.SimpleNamespace(**{**c, "engram_num_embeddings": (0, 0)}))
    rows = tuple(sum(p for per in layer for p in per) for layer in layout.primes)
    args.engram_num_embeddings = rows
    return args, rows


class Pack:
    """Stored tensors (a pack's names and dtypes) and the f32 values the
    reference sees for them."""

    def __init__(self, rng, fmt):
        self.rng, self.fmt = rng, fmt
        self.stored, self.deq = {}, {}

    def bf16(self, a):
        return torch.from_numpy(np.asarray(a, np.float32)).to(torch.bfloat16)

    def dense(self, name, a, dtype=torch.bfloat16):
        t = torch.from_numpy(np.asarray(a, np.float32)).to(dtype)
        self.stored[name] = t
        self.deq[name] = t.float().numpy()

    def affine(self, base, out, inp, bits, gs=64, gathered=False):
        """`gathered`: the engine reads rows through MLX `dequantize`, which
        rounds to the bf16 of the scales even when asked for f32."""
        rng = self.rng
        q = rng.integers(0, 2 ** bits, size=(*out, inp)).astype(np.uint32)
        a = np.sqrt(3.0) * inp ** -0.5
        ng = inp // gs
        scale = self.bf16(rng.uniform(0.6, 1.4, (*out, ng)) * 2 * a / (2 ** bits - 1))
        bias = self.bf16(-a * rng.uniform(0.6, 1.4, (*out, ng)))
        w = (q.astype(np.float64).reshape(*out, ng, gs) * scale.double().numpy()[..., None]
             + bias.double().numpy()[..., None]).reshape(*out, inp).astype(np.float32)
        packed = np.zeros((*out, inp * bits // 32), np.uint32)
        for j in range(inp):
            bit = j * bits
            packed[..., bit // 32] |= (q[..., j] << (bit % 32)) & 0xFFFFFFFF
            if bit % 32 + bits > 32:
                packed[..., bit // 32 + 1] |= q[..., j] >> (32 - bit % 32)
        self.stored[base + ".weight"] = torch.from_numpy(packed.view(np.int32)).view(torch.uint32)
        self.stored[base + ".scales"] = scale
        self.stored[base + ".biases"] = bias
        self.deq[base + ".weight"] = self.bf16(w).float().numpy() if gathered else w

    def fp(self, base, out, inp, fmt):
        rng = self.rng
        if fmt == "mxfp8":
            # Normal draws in e4m3, like a real checkpoint (uniform codes reach +-448).
            codes = torch.from_numpy(rng.standard_normal((*out, inp)).astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            vals = E4M3[codes]
            packed = codes.reshape(*out, inp // 4, 4).copy().view(np.uint32)[..., 0]
            mag = 1.0
        else:
            codes = rng.integers(0, 16, size=(*out, inp)).astype(np.uint32)
            vals = E2M1[codes]
            packed = np.zeros((*out, inp // 8), np.uint32)
            for j in range(8):
                packed |= codes[..., j::8] << (4 * j)
            mag = 2.0
        e = np.round(np.log2(inp ** -0.5 / mag)).astype(np.int64)
        se = (127 + e + rng.integers(-1, 2, size=(*out, inp // 32))).astype(np.uint8)
        w = (vals.reshape(*out, inp // 32, 32) * (2.0 ** (se.astype(np.float64) - 127))[..., None]).reshape(*out, inp)
        self.stored[base + ".weight"] = torch.from_numpy(packed.view(np.int32)).view(torch.uint32)
        self.stored[base + ".scales"] = torch.from_numpy(se)
        self.deq[base + ".weight"] = w.astype(np.float32)

    def linear(self, base, out, inp, bits=8):
        """A quantizable linear in this pack's trunk format."""
        if self.fmt == "pipe":
            self.affine(base, (out,), inp, bits)
        else:
            self.fp(base, (out,), inp, "mxfp8")


def gauss(rng, *shape, s):
    return rng.standard_normal(shape) * s


def build_weights(fmt, args, rows):
    c = CFG
    rng = np.random.default_rng(7)
    p = Pack(rng, fmt)
    d, hd, ql = c["hidden_size"], c["head_dim"], c["q_lora_rank"]
    H, og, ol = c["num_attention_heads"], c["o_groups"], c["o_lora_rank"]
    inter, E = c["moe_intermediate_size"], c["n_routed_experts"]
    hc, mix = c["hc_mult"], (2 + c["hc_mult"]) * c["hc_mult"]
    norm = lambda n: 1 + 0.1 * rng.standard_normal(n)
    if fmt == "pipe":
        p.affine("embed", (VOCAB,), d, 8, gathered=True)
        p.affine("head", (VOCAB,), d, 8)
    else:
        p.dense("embed.weight", gauss(rng, VOCAB, d, s=0.5))
        p.dense("head.weight", gauss(rng, VOCAB, d, s=0.3))
    p.dense("norm.weight", norm(d))

    def block(pfx, li, n_exp, stage):
        for nm in ("attn", "ffn"):
            p.dense(f"{pfx}.hc_{nm}_fn", gauss(rng, mix, hc * d, s=0.05), torch.float32)
            p.dense(f"{pfx}.hc_{nm}_base", gauss(rng, mix, s=0.2), torch.float32)
            p.dense(f"{pfx}.hc_{nm}_scale", 0.5 + gauss(rng, 3, s=0.2), torch.float32)
        p.dense(f"{pfx}.attn_norm.weight", norm(d))
        p.dense(f"{pfx}.ffn_norm.weight", norm(d))
        a = f"{pfx}.attn"
        p.linear(f"{a}.wq_a", ql, d)
        p.dense(f"{a}.q_norm.weight", norm(ql))
        p.linear(f"{a}.wq_b", H * hd, ql)
        p.linear(f"{a}.wkv", hd, d)
        p.dense(f"{a}.kv_norm.weight", norm(hd))
        if fmt == "pipe":
            p.dense(f"{a}.wo_a.weight", gauss(rng, og * ol, H * hd // og, s=(H * hd // og) ** -0.5))
        else:
            p.fp(f"{a}.wo_a", (og * ol,), H * hd // og, "mxfp8")
        p.linear(f"{a}.wo_b", d, og * ol)
        p.dense(f"{a}.attn_sink", gauss(rng, H, s=0.5), torch.float32)
        if not stage and li in c["kv_source_layer_ids"]:
            p.dense(f"{a}.compressor.wkv.weight", gauss(rng, hd, d, s=d ** -0.5))
            if c["compress_ratios"][li] > 1:
                p.dense(f"{a}.compressor.wgate.weight", gauss(rng, hd, d, s=d ** -0.5))
            p.dense(f"{a}.compressor.norm.weight", norm(hd))
        if not stage and li in c["index_source_layer_ids"]:
            ih, ihd = c["index_n_heads"], c["index_head_dim"]
            p.linear(f"{a}.indexer.wq_b", ih * ihd, ql)
            p.dense(f"{a}.indexer.weights_proj.weight", gauss(rng, ih, d, s=d ** -0.5))
            if li in c["kv_source_layer_ids"]:
                p.dense(f"{a}.indexer.wk.weight", gauss(rng, ihd, hd, s=hd ** -0.5))
                p.dense(f"{a}.indexer.k_norm.weight", norm(ihd))
        f = f"{pfx}.ffn"
        p.dense(f"{f}.gate.weight", gauss(rng, n_exp, d, s=d ** -0.5))
        p.dense(f"{f}.gate.bias", gauss(rng, n_exp, s=0.3), torch.float32)
        p.linear(f"{f}.shared_experts.w1", inter, d)
        p.linear(f"{f}.shared_experts.w3", inter, d)
        p.linear(f"{f}.shared_experts.w2", d, inter)
        if stage:
            for e in range(n_exp):
                p.fp(f"{f}.experts.{e}.w1", (inter,), d, "mxfp4")
                p.fp(f"{f}.experts.{e}.w3", (inter,), d, "mxfp4")
                p.fp(f"{f}.experts.{e}.w2", (d,), inter, "mxfp4")
        else:
            p.affine(f"{f}.experts.gate_proj", (n_exp, inter), d, 4)
            p.affine(f"{f}.experts.up_proj", (n_exp, inter), d, 4)
            p.affine(f"{f}.experts.down_proj", (n_exp, d), inter, 4)

    for li in range(c["num_hidden_layers"]):
        block(f"layers.{li}", li, E, False)
    if fmt == "repack":
        for si in range(STAGES):
            block(f"mtp.{si}", si, c["dspark_n_routed_experts"], True)
        p.linear("mtp.0.main_proj", d, d * len(c["dspark_target_layer_ids"]))
        p.dense("mtp.0.main_norm.weight", norm(d))
        last = f"mtp.{STAGES - 1}"
        r = c["dspark_markov_rank"]
        p.dense(f"{last}.norm.weight", norm(d))
        p.dense(f"{last}.markov_head.embed.weight", gauss(rng, VOCAB, r, s=0.5))
        p.dense(f"{last}.markov_head.head.weight", gauss(rng, VOCAB, r, s=r ** -0.5))
        p.dense(f"{last}.confidence_head.proj.weight", gauss(rng, 1, d + r, s=(d + r) ** -0.5))

    ehd = c["engram_head_dim"]
    cols = (c["engram_max_ngram_size"] - 1) * c["engram_n_heads"]
    tables = {}
    for k, li in enumerate(c["engram_layer_ids"]):
        e = f"layers.{li}.engram"
        p.linear(f"{e}.wkv", d * (hc + 1), cols * ehd)
        p.dense(f"{e}.q_weight", 1 + gauss(rng, hc, d, s=0.3))
        p.dense(f"{e}.k_weight", 1 + gauss(rng, hc, d, s=0.3))
        if fmt == "pipe":
            p.affine(f"{e}.embed", (rows[k],), ehd, 4, gathered=True)
        else:
            p.fp(f"{e}.embed", (rows[k],), ehd, "mxfp8")
        tables[li] = p.deq[f"{e}.embed.weight"]
    return p, tables


def write_pack(out, fmt, p, rows):
    os.makedirs(out, exist_ok=True)
    cfg = {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "bos_token_id": 0,
           "eos_token_id": 1, "text_config": {"model_type": "deepseek_v41_text", **CFG,
                                              "engram_num_embeddings": list(rows)}}
    cfg["quantization"] = {"group_size": 64, "bits": 8} if fmt == "pipe" else {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    shards = {"model-00001.safetensors": {}, "model-engram.safetensors": {}}
    residents = {}
    for name, t in p.stored.items():
        if ".engram.embed." in name:
            if fmt == "pipe":
                shards["model-engram.safetensors"][name] = t
        elif ".engram." in name and fmt == "repack":
            residents[name] = t
        else:
            shards["model-00001.safetensors"][name] = t
    weight_map = {}
    for fname, tensors in shards.items():
        if not tensors:
            continue
        save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out, fname), metadata={"format": "mlx"})
        weight_map.update({k: fname for k in tensors})
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    if fmt == "pipe":
        with open(os.path.join(out, "engram_token_map.json"), "w") as f:
            json.dump(list(range(VOCAB)), f)
        return
    with open(os.path.join(out, "engram-token-map.u32"), "wb") as f:
        f.write(np.arange(VOCAB, dtype="<u4").tobytes())
    os.makedirs(os.path.join(out, "engram"), exist_ok=True)
    save_file({k: v.contiguous() for k, v in residents.items()}, os.path.join(out, "engram", "engram-residents.safetensors"),
              metadata={"format": "mlx"})
    ehd = CFG["engram_head_dim"]
    layers = []
    for k, li in enumerate(CFG["engram_layer_ids"]):
        codes = p.stored[f"layers.{li}.engram.embed.weight"].view(torch.int32).numpy().view(np.uint8).reshape(rows[k], ehd)
        sc = p.stored[f"layers.{li}.engram.embed.scales"].numpy().reshape(rows[k], ehd // 32)
        name = f"engram-L{li}.bin"
        with open(os.path.join(out, "engram", name), "wb") as f:
            f.write(np.concatenate([codes, sc], axis=1).tobytes())
        layers.append({"layer_id": li, "file": name, "rows": rows[k], "record_bytes": ehd + ehd // 32,
                       "quant": {"bits": 8, "group_size": 32, "mode": "mxfp8", "head_dim": ehd}})
    with open(os.path.join(out, "engram", "engram-manifest.json"), "w") as f:
        json.dump({"format": "mtplx-engram-manifest-v1", "layers": layers}, f, indent=1)


def load_into(rm, p, tables):
    c = CFG
    filled = set()

    def put(param, arr):
        with torch.no_grad():
            param.copy_(torch.from_numpy(np.asarray(arr, np.float32)).reshape(param.shape))
        filled.add(id(param))

    deq = p.deq
    for name, param in rm.named_parameters(remove_duplicate=True):
        if ".engram.embed." in name:
            li = int(name.split(".")[1])
            put(param, tables[li] if name.endswith(".weight") else np.ones(param.shape))
            continue
        m = name.split(".")
        if ".ffn.experts." in name and name.startswith("layers."):
            e, proj = int(m[4]), m[5]
            stacked = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[proj]
            put(param, deq[f"{m[0]}.{m[1]}.ffn.experts.{stacked}.weight"][e])
        elif name in deq:
            put(param, deq[name])
        elif name.endswith(".scale") and name[: -len(".scale")] + ".weight" in deq:
            put(param, np.ones(param.shape))
        else:
            raise SystemExit(f"no value for reference parameter {name}")
    del c


KERNEL_STUB = None


def run(ref, ref_engram, out, fmt, qat=True):
    dspark = fmt == "repack"
    args, rows = ref_args(ref, ref_engram, STAGES if dspark else 0)
    p, tables = build_weights(fmt, args, rows)
    write_pack(out, fmt, p, rows)
    KERNEL_STUB.DISABLE_FAKE_QUANT = not qat
    torch.manual_seed(0)
    with torch.no_grad():
        rm = ref.Transformer(args, tokenizer=None).float()
    load_into(rm, p, tables)
    rng = np.random.default_rng(11)
    ids = rng.integers(3, VOCAB - 1, size=S + D)
    ids[0] = 2
    ids[9] = 2
    t = torch.from_numpy(ids[None]).long()
    res = {"ids": torch.from_numpy(ids.astype(np.int32))}
    # Per-layer captures of the full prefill: the bisect ladder.
    hooks = []
    calls = {"attn": 0, "split": 0}
    orig_attn, orig_split = ref.sparse_attn, ref.hc_split_sinkhorn

    def attn_spy(q, kv, sink, idxs, scale):
        o = orig_attn(q, kv, sink, idxs, scale)
        if calls["attn"] == 0:
            res.update(sa_q_0=q[0].float().clone(), sa_kv_0=kv[0].float().clone(), sa_o_0=o[0].float().clone())
        calls["attn"] += 1
        return o

    def split_spy(mixes, scale, base, *a):
        out = orig_split(mixes, scale, base, *a)
        if calls["split"] == 0:
            res.update(hc_pre_0=out[0][0].float().clone(), hc_post_0=out[1][0].float().clone(), hc_comb_0=out[2][0].float().clone())
        calls["split"] += 1
        return out

    ref.sparse_attn, ref.hc_split_sinkhorn = attn_spy, split_spy
    hooks.append(rm.layers[0].attn.register_forward_pre_hook(lambda mod, inp: res.__setitem__("attn_in_0", inp[0][0].float().clone())))
    for li, blk in enumerate(rm.layers):
        hooks.append(blk.register_forward_hook(lambda mod, inp, out, li=li: res.__setitem__(f"stream_{li}", out[0][0].float().clone())))
        hooks.append(blk.attn.register_forward_hook(lambda mod, inp, out, li=li: res.__setitem__(f"attn_{li}", out[0].float().clone())))
        hooks.append(blk.ffn.register_forward_hook(lambda mod, inp, out, li=li: res.__setitem__(f"ffn_{li}", out[0].float().clone())))
        if blk.engram is not None:
            hooks.append(blk.engram.register_forward_hook(lambda mod, inp, out, li=li: res.__setitem__(f"engram_{li}", out[0].float().clone())))
        if blk.attn.indexer is not None:
            # Selected compressed entries (the window offset removed), sorted.
            hooks.append(blk.attn.indexer.register_forward_hook(lambda mod, inp, out, li=li: res.__setitem__(
                f"topk_{li}", torch.where(out[0] >= 0, out[0] - inp[4], -1).sort(-1).values.int())))
    with torch.no_grad():
        _, full, _ = rm(t, 0)
        res["logits_full"] = full[0].float()
        for h in hooks:
            h.remove()
        ref.sparse_attn, ref.hc_split_sinkhorn = orig_attn, orig_split
        n = S + D
        for li, blk in enumerate(rm.layers):
            r = CFG["compress_ratios"][li]
            if blk.attn.is_kv_source:
                res[f"comp_{li}"] = blk.attn.compress_kv_cache[0, : n // r].float().clone()
            if blk.attn.indexer is not None and blk.attn.indexer.owns_k:
                res[f"idxk_{li}"] = blk.attn.indexer.k_cache[0, : n // r].float().clone()
        out_ids, _, mh = rm(t[:, :S], 0)
        if dspark:
            rm.forward_spec(out_ids[:, -1], mh, 0)
        dec, dids, dlog, dconf = [], [], [], []
        for i in range(S, S + D):
            out_ids, lg, mh = rm(t[:, i:i + 1], i)
            dec.append(lg[0, -1].float())
            if dspark:
                o, l, cf = rm.forward_spec(out_ids[:, -1], mh, i)
                dids.append(o[0].int())
                dlog.append(l[0].float())
                dconf.append(cf[0].float())
    res["logits_decode"] = torch.stack(dec)
    if dspark:
        res["draft_ids"] = torch.stack(dids)
        res["draft_logits"] = torch.stack(dlog)
        res["draft_conf"] = torch.stack(dconf)
    name = "fixture.safetensors" if qat else "fixture_noqat.safetensors"
    save_file({k: v.contiguous() for k, v in res.items()}, os.path.join(out, name))
    print(f"{fmt}: {out} ({len(p.stored)} tensors, engram rows {rows})")


def write_qat(out):
    """The three QAT round-trips on inputs that include exact rounding
    midpoints and power-of-two amax boundaries: <out>/qat.safetensors."""
    rng = np.random.default_rng(3)
    mid = np.array([0.033203125, -0.033203125, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -5.0, 448.0, 6.0,
                    1e-9, 0.0, -0.0, 2.0] * 8, np.float32).reshape(4, 32) * (2.0 ** rng.integers(-8, 9, size=(4, 1))).astype(np.float32)
    x = np.concatenate([(rng.standard_normal((60, 32)) * np.exp(rng.standard_normal((60, 1)))).astype(np.float32), mid]).reshape(32, 64)
    st = KERNEL_STUB
    st.DISABLE_FAKE_QUANT = False
    t = torch.from_numpy(x.copy())
    res = {
        "x": t.clone(),
        "fp8_ue8m0_32": st.act_quant(t.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True),
        "fp4_ue8m0_32": st.fp4_act_quant(t.clone(), 32, True),
        "fp4_e4m3_16": st.fp4_act_quant(t.clone(), 16, True, torch.float8_e4m3fn),
    }
    os.makedirs(out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in res.items()}, os.path.join(out, "qat.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="DeepSeek-V4.1-Flash inference/ directory")
    ap.add_argument("--kernel-stub", required=True, help="deepseek-v41-mlx tests/kernel_stub.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    ref, ref_engram = load_reference(os.path.abspath(a.reference), os.path.abspath(a.kernel_stub))
    write_qat(os.path.expanduser(a.out))
    for fmt in ("pipe", "repack"):
        for qat in (True, False):
            run(ref, ref_engram, os.path.join(os.path.expanduser(a.out), fmt), fmt, qat)


if __name__ == "__main__":
    main()
