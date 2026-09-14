#!/usr/bin/env python3
"""Per-input-channel activation statistics (an "imatrix") for Qwen3.8-Flash-Next
(model_type qwen4_exp), collected on the EXACT bf16 checkpoint.

The 360 GB checkpoint never has to be resident: the corpus is embedded once,
then every decoder layer is built on the meta device, its weights read from the
HF shards, run over every sequence's stored hyper-connection stream (CPU bf16,
4 x hidden per token), and freed before the next layer. The 102 GB n-gram table
is never loaded either — the PLE layer's `nn.Embedding` is swapped for a gather
over numpy memmaps of the 128 checkpoint shards.

Statistics, keyed by SOURCE weight name (same contract as
tests/qwen38_imatrix_collect.py, consumed by dsv4_imatrix.weighted_affine_quant):
  <name>.weight                  every nn.Linear: mean(x^2) per input channel
  ...mlp.experts.gate_up_proj    per-expert, concatenated [E * hidden] (expert e's
  ...mlp.experts.down_proj       channels at [e*in, (e+1)*in)), sum(x^2) over the
                                 tokens routed to e divided by the LAYER's token
                                 count — a busy expert keeps its larger vote
  ...mlp.experts.gate_up_proj.rows  [E] tokens routed to each expert
Not covered: embed_tokens (gather-read), lm_head and the final mixer (8-bit,
bytes-heavy, no calibration payoff), the MTP head (4-bit, pinned).

Corpus = the 27B collector's (agent traffic, code, prose, math), rendered with
this model's own template; same held-out split.

`--reference OUT` runs the HELD-OUT half instead and writes bf16 next-token logits at
sampled positions (`ids`, `pos`, `logits` f16, `slice`) for tests/qwen38_flash_next_score.py,
which scores a served pack's greedy pick against them. No statistics are collected.

  venv/bin/python tests/qwen38_flash_next_imatrix_collect.py \
      --src "/Volumes/G Drive SSD/models-src/Qwen3.8-Flash-Next" \
      --out ~/claude-tmp/qwen38-flash-next-23/imatrix.safetensors
"""

import argparse
import json
import math
import os
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_imatrix_collect as corpus  # noqa: E402  (pure corpus builders)

PREFIX = "model.language_model."


def shard_of(index, key):
    return index["weight_map"][key]


class NgramGather(torch.nn.Module):
    """Row gather over the checkpoint's n-gram shards, memmapped in place."""

    def __init__(self, src, index, layer_prefix):
        super().__init__()
        keys = sorted((k for k in index["weight_map"] if k.startswith(layer_prefix + "ple.ple_embedding.ngram_embedding.shard_")),
                      key=lambda k: int(k.rsplit("shard_", 1)[1].split(".")[0]))
        self.maps, self.starts, total = [], [], 0
        for k in keys:
            path = Path(src) / shard_of(index, k)
            with open(path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
            meta = hdr[k]
            assert meta["dtype"] == "BF16", (k, meta["dtype"])
            b, e = meta["data_offsets"]
            mm = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + hlen + b, shape=tuple(meta["shape"]))
            self.maps.append(mm)
            self.starts.append(total)
            total += meta["shape"][0]
        self.rows, self.dim = total, self.maps[0].shape[1]
        self.starts = np.array(self.starts + [total])

    @property
    def weight(self):
        return torch.empty(0, self.dim)  # device placement hook only reads .device

    def forward(self, ids):
        flat = ids.reshape(-1).cpu().numpy().astype(np.int64)
        assert flat.max() < self.rows, "n-gram id past the table"
        out = np.empty((flat.size, self.dim), dtype=np.uint16)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        for s in np.unique(shard):
            sel = shard == s
            out[sel] = self.maps[s][flat[sel] - self.starts[s]]
        t = torch.from_numpy(out.view(np.int16)).view(torch.bfloat16)
        return t.reshape(*ids.shape, self.dim).to(ids.device)


class Stats:
    def __init__(self, device):
        self.device = device
        self.acc, self.rows = {}, {}

    def observe(self, name, x):
        flat = x.reshape(-1, x.shape[-1]).float()
        s = (flat * flat).sum(0)
        self.acc[name] = s if name not in self.acc else self.acc[name] + s
        self.rows[name] = self.rows.get(name, 0) + flat.shape[0]


def hook_linears(layer, layer_prefix, stats):
    handles = []
    for path, mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            name = layer_prefix + path + ".weight"
            handles.append(mod.register_forward_pre_hook(lambda m, args, n=name: stats.observe(n, args[0])))
    return handles


def experts_forward_with_stats(experts, gu_acc, dn_acc, rows):
    """Qwen4ExpTextExperts.forward, plus per-expert sum(x^2) on both inputs."""
    def forward(hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=experts.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = int(e[0])
            top_k_pos, token_idx = torch.where(mask[e])
            x = hidden_states[token_idx]
            xf = x.float()
            gu_acc[e] += (xf * xf).sum(0)
            rows[e] += x.shape[0]
            gate, up = torch.nn.functional.linear(x, experts.gate_up_proj[e]).chunk(2, dim=-1)
            h = experts.act_fn(gate) * up
            hf = h.float()
            dn_acc[e] += (hf * hf).sum(0)
            h = torch.nn.functional.linear(h, experts.down_proj[e])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h.to(final.dtype))
        return final
    return forward


def qsa_indexer_forward_vectorized(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """Qwen4ExpTextQSAIndexer.forward for ONE unpadded causal sequence with no cache:
    the reference loops over every query position in Python (50 min per layer at 133k
    tokens); this computes the same block scores for all queries at once. Mask-exact
    against the reference (--verify-indexer) except on exact-zero score ties at the
    budget edge, where the two topk calls pick a different zero block (14 rows in 3001)."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb
    assert past_key_values is None and hidden_states.shape[0] == 1 and attention_mask.dtype == torch.bool
    L = hidden_states.shape[1]
    R, d = self.compress_ratio, self.index_head_dim
    full_cos, full_sin = position_embeddings
    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(qk, [self.index_n_heads * d, self.index_kv_heads * d], dim=-1)
    q = self.q_layernorm(q.reshape(1, L, -1, d))
    q = apply_rotary_pos_emb(q, cos=full_cos[:, -L:, :], sin=full_sin[:, -L:, :], unsqueeze_dim=2)[0]  # [L, nh, d]
    raw_keys = token_k.reshape(1, L, -1, d).squeeze(2)[0]                                           # [L, d]
    nb = L // R
    dev = hidden_states.device
    pos = torch.arange(L, device=dev)
    n_complete = (pos + 1) // R                                                                       # [L]
    if nb > 0:
        pooled = raw_keys[: nb * R].view(nb, R, d).float().mean(dim=1).to(raw_keys.dtype)
        pooled = self.k_layernorm(pooled)
        starts = torch.arange(nb, device=dev) * R
        kb = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=full_cos[0].index_select(0, starts),
                                  sin=full_sin[0].index_select(0, starts)).squeeze(1)              # [nb, d]
        scores = torch.einsum("lhd,bd->lhb", q.float(), kb.float())
        scores = torch.relu(scores).sum(dim=1) / math.sqrt(d)                                        # [L, nb]
        block_idx = torch.arange(nb, device=dev)
        valid = block_idx[None, :] < n_complete[:, None]
        scores = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
        k = min(self.block_topk, nb)
        top = scores.topk(k, dim=1).indices                                                           # [L, k]
        top_valid = torch.gather(valid, 1, top)
        sel_blocks = torch.zeros(L, nb, dtype=torch.bool, device=dev).scatter(1, top, top_valid)
        sel_tokens = sel_blocks[:, :, None].expand(L, nb, R).reshape(L, nb * R)
        sel = torch.zeros(L, L, dtype=torch.bool, device=dev)
        sel[:, : nb * R] = sel_tokens
    else:
        sel = torch.zeros(L, L, dtype=torch.bool, device=dev)
    tail = (pos[None, :] >= (n_complete * R)[:, None]) & (pos[None, :] <= pos[:, None])
    return (sel | tail)[None, None]


def write_reference(args, tcfg, index, seqs, hidden, slice_of, dev, M):
    """Mixer + lm_head on sampled rows of every held-out window -> npz of bf16 logits."""
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        mixer = M.Qwen4ExpTextGatedResidual(tcfg, use_combine=False)
    torch.set_default_dtype(torch.float32)
    mixer.to_empty(device=dev)
    mp = PREFIX + "hyper_connection_mixer."
    missing, unexpected = mixer.load_state_dict(load_layer_state(args.src, index, mp, dev), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    with safe_open(str(Path(args.src) / shard_of(index, "lm_head.weight")), "pt", device="cpu") as fh:
        head = fh.get_tensor("lm_head.weight").to(dev)
    rng = random.Random(args.seed + 1)
    ids_out, pos_out, logits_out, slice_out = [], [], [], []
    with torch.no_grad():
        for i, h in enumerate(hidden):
            L = h.shape[1]
            if L < 96:
                continue
            for p in sorted(rng.sample(range(64, L), min(args.positions, L - 64))):
                x = mixer(h[:, p:p + 1].to(dev))
                logits = (x.reshape(1, -1) @ head.T).float().cpu().numpy()[0]
                ids_out.append(seqs[i][:p + 1].numpy().astype(np.int32))   # row p predicts token p+1
                pos_out.append(p)
                logits_out.append(logits.astype(np.float16))
                slice_out.append(slice_of[i])
    out = Path(os.path.expanduser(args.reference))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array(ids_out, dtype=object), pos=np.array(pos_out), logits=np.stack(logits_out),
             slice=np.array(slice_out))
    print(f"wrote {out}: {len(pos_out)} positions from {len(hidden)} held-out windows", flush=True)


def load_layer_state(src, index, layer_prefix, device):
    sd, by_file = {}, {}
    for k, f in index["weight_map"].items():
        if k.startswith(layer_prefix) and ".ngram_embedding.shard_" not in k:
            by_file.setdefault(f, []).append(k)
    for f, keys in by_file.items():
        with safe_open(str(Path(src) / f), "pt", device="cpu") as fh:
            for k in keys:
                sd[k[len(layer_prefix):]] = fh.get_tensor(k).to(device)
    return sd


def build_corpus(tok, args, holdout=False):
    rng = random.Random(args.seed)
    sweb = corpus.swebench_rows()
    traffic = corpus.tool_traffic(600)
    slices = {name: corpus.split_docs(docs, holdout) for name, docs in (
        ("agent", corpus.slice_agent(tok, rng, traffic, sweb, 400)),
        ("code", corpus.slice_code(tok, rng, sweb, 400)),
        ("prose", corpus.slice_prose(tok, rng, sweb, 400)),
        ("math", corpus.slice_math(tok, rng, 200)),
    )}
    budget = args.max_tokens // len(slices)
    seqs, composition, slice_of = [], {}, []
    for name, docs in slices.items():
        rng.shuffle(docs)
        used = 0
        for doc in docs:
            if used >= budget:
                break
            ids = tok.encode(doc)
            for off in range(0, len(ids), args.seq_len):
                w = ids[off:off + args.seq_len]
                if len(w) < 16:
                    continue
                seqs.append(torch.tensor(w, dtype=torch.long))
                slice_of.append(name)
                used += len(w)
                if used >= budget:
                    break
        composition[name] = used
    return seqs, composition, slice_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None, help="imatrix safetensors (required unless --reference)")
    ap.add_argument("--max-tokens", type=int, default=160_000)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--layers", type=int, default=None, help="stop after N layers (smoke)")
    ap.add_argument("--reference", default=None, help="held-out bf16 next-token logits (npz) instead of statistics")
    ap.add_argument("--positions", type=int, default=4, help="--reference: scored positions per window")
    ap.add_argument("--dump-hidden", default=None, help="write sequence 0's final stream (smoke parity)")
    ap.add_argument("--verify-indexer", type=int, default=0,
                    help="on the first attention layer, check the vectorized QSA indexer against the reference loop on N sequences")
    ap.add_argument("--clamp-vocab", action="store_true", help="ids modulo the embedding rows (tiny-fixture smoke only)")
    args = ap.parse_args()
    if not args.out and not args.reference:
        ap.error("--out or --reference is required")
    from transformers import AutoConfig, AutoTokenizer
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    t0 = time.time()
    dev = torch.device(args.device)
    cfg = AutoConfig.from_pretrained(args.src)
    tcfg = getattr(cfg, "text_config", cfg)
    tcfg._attn_implementation = "sdpa"
    index = json.loads((Path(args.src) / "model.safetensors.index.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.src)
    seqs, composition, slice_of = build_corpus(tok, args, holdout=args.reference is not None)
    total = sum(len(s) for s in seqs)
    print(f"corpus: {len(seqs)} sequences, {total} tokens {composition} ({time.time()-t0:.0f}s)", flush=True)

    with safe_open(str(Path(args.src) / shard_of(index, PREFIX + "embed_tokens.weight")), "pt", device="cpu") as fh:
        embed = fh.get_tensor(PREFIX + "embed_tokens.weight")
    if args.clamp_vocab:
        seqs = [s % embed.shape[0] for s in seqs]
    hidden = [torch.nn.functional.embedding(s, embed)[None].repeat(1, 1, tcfg.hc_count) for s in seqs]
    del embed
    rotary = M.Qwen4ExpTextRotaryEmbedding(tcfg).to(dev)

    def per_seq(i):
        ids = seqs[i][None].to(dev)
        L = ids.shape[1]
        pos = (torch.arange(L, device=dev)).view(1, 1, -1).expand(4, 1, -1)
        probe = torch.empty(1, L, tcfg.hidden_size, device=dev, dtype=torch.bfloat16)
        mask = create_causal_mask(config=tcfg, inputs_embeds=probe, attention_mask=None, past_key_values=None,
                                  position_ids=pos[0], allow_is_causal_skip=False)
        conv_mask = create_recurrent_attention_mask(config=tcfg, inputs_embeds=probe, attention_mask=None,
                                                    past_key_values=None, position_ids=pos[0])
        return ids, rotary(probe, pos[1:]), mask, conv_mask

    arrays, rows_meta, per_layer, verified = {}, {}, [], {"done": False}
    n_layers = tcfg.num_hidden_layers if args.layers is None else min(args.layers, tcfg.num_hidden_layers)
    for li in range(n_layers):
        tl = time.time()
        lp = f"{PREFIX}layers.{li}."
        # bf16 params like the checkpoint; the default dtype goes back to f32 before the
        # forward so the reference's own f32 scratch (GDN chunk math, masks) stays f32.
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            layer = M.Qwen4ExpTextDecoderLayer(tcfg, li)
        torch.set_default_dtype(torch.float32)
        ng = None
        if layer.ple is not None:
            ng = layer.ple.ple_embedding
            ng.ngram_embedding = NgramGather(args.src, index, lp)
        layer.to_empty(device=dev)
        missing, unexpected = layer.load_state_dict(load_layer_state(args.src, index, lp, dev), strict=False)
        assert not unexpected, unexpected
        assert not missing, missing
        if ng is not None:
            ng.ngram_heads_vocab_sizes.copy_(torch.tensor(ng.head_vocab_sizes))
            ng.ngram_heads_offsets.copy_(torch.tensor(ng.head_offsets))
        layer.eval()
        if hasattr(layer, "self_attn"):
            ix = layer.self_attn.indexer
            if args.verify_indexer and not verified["done"]:
                ref_fwd = ix.forward
                with torch.no_grad():
                    for i in range(min(args.verify_indexer, len(seqs))):
                        ids, pe, mask, _ = per_seq(i)
                        x = layer.attn_hyper_connection(hidden[i].to(dev))[0]
                        a = ref_fwd(x, pe, mask, None)
                        b = qsa_indexer_forward_vectorized(ix, x, pe, mask, None)
                        diff = (a != b).sum().item()
                        print(f"indexer verify seq {i} L={ids.shape[1]}: mask mismatches {diff} of {a.numel()}", flush=True)
                        assert diff == 0, "vectorized QSA indexer disagrees with the reference"
                verified["done"] = True
            ix.forward = lambda *a, _ix=ix, **kw: qsa_indexer_forward_vectorized(_ix, *a, **kw)
        stats = Stats(dev)
        ex = layer.mlp.experts
        E, H, I = ex.num_experts, ex.hidden_dim, ex.intermediate_dim
        gu_acc = torch.zeros(E, H, device=dev, dtype=torch.float32)
        dn_acc = torch.zeros(E, I, device=dev, dtype=torch.float32)
        erows = [0] * E
        handles = [] if args.reference else hook_linears(layer, lp, stats)
        if not args.reference:
            ex.forward = experts_forward_with_stats(ex, gu_acc, dn_acc, erows)
        with torch.no_grad():
            for i in range(len(seqs)):
                ids, pe, mask, conv_mask = per_seq(i)
                out = layer(hidden[i].to(dev), position_embeddings=pe, attention_mask=mask, conv_mask=conv_mask,
                            past_key_values=None, ple_input_ids=ids)
                hidden[i] = out.to("cpu")
                del out
                if args.dump_hidden and i == 0:
                    per_layer.append(hidden[0].clone())
                if dev.type == "mps" and i % 8 == 7:
                    torch.mps.empty_cache()   # varying L keeps the caching allocator growing; swap otherwise
        for h in handles:
            h.remove()
        for name, s in stats.acc.items():
            arrays[name] = (s / float(stats.rows[name])).cpu().numpy()
            rows_meta[name] = stats.rows[name]
        arrays[lp + "mlp.experts.gate_up_proj"] = (gu_acc / float(total)).reshape(-1).cpu().numpy()
        arrays[lp + "mlp.experts.down_proj"] = (dn_acc / float(total)).reshape(-1).cpu().numpy()
        arrays[lp + "mlp.experts.gate_up_proj.rows"] = np.array(erows, dtype=np.float32)
        del layer, stats, gu_acc, dn_acc, ex
        if dev.type == "mps":
            torch.mps.empty_cache()
        print(f"layer {li:2d} {time.time()-tl:5.0f}s  elapsed {(time.time()-t0)/60:.1f} min  "
              f"experts hit {sum(1 for r in erows if r)}/{E}", flush=True)

    if args.dump_hidden:
        torch.save({"ids": seqs[0], "hidden": hidden[0], "per_layer": per_layer}, args.dump_hidden)
    if args.reference:
        return write_reference(args, tcfg, index, seqs, hidden, slice_of, dev, M)
    for name, a in arrays.items():
        assert np.isfinite(a).all(), name
    meta = {
        "source": args.src, "seed": str(args.seed), "seq_len": str(args.seq_len), "total_tokens": str(total),
        "layers": str(n_layers), "composition": json.dumps(composition), "rows_per_weight": json.dumps(rows_meta),
        "values": "mean-squared activation per INPUT channel; experts: sum(x^2)/layer tokens, per expert concatenated",
        "keys": "SOURCE checkpoint weight names",
        "holdout": "calibrated on split_docs(holdout=False); every 10th document withheld",
    }
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(arrays, str(out), metadata=meta)
    print(f"wrote {out}: {len(arrays)} entries, {total} tokens, {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
