#!/usr/bin/env python3
"""Oracle for the MiMo-V2.6 MTP heads (`model.mtp.layers.{k}`) on the TINY fixture model.

The HF reference skips `model.mtp.*`, so the head is rendered here from the HF
reference's OWN modules (MiMoV2Attention with the SWA geometry and its sinks,
MiMoV2MLP, MiMoV2RMSNorm, the SWA rotary table) composed exactly as the
vLLM/SGLang MiMo-V2 MTP layer does (XiaomiMiMo, Apache-2.0):

    x   = eh_proj(cat[enorm(embed(token)), hnorm(target_hidden)])
    x   = x + self_attn(input_layernorm(x))          # sliding window, sinks
    x   = x + mlp(pre_mlp_layernorm(x))              # dense SwiGLU
    out = final_layernorm(x)  -> shared lm_head

Head k's row p pairs the trunk's FINAL-NORMED hidden at position p with token
p+k+1 at rope position p (SGLang multi-layer EAGLE: every head reads the
target's hidden, only the token shifts), and predicts token p+k+2.

Run AFTER tests/dump_mimo_v2_fixtures.py OUT_DIR (same --seed): rebuilds the
same tiny trunk, adds three random heads to OUT_DIR as `model_mtp.safetensors`
(index updated; fused qkv stored [q | k | v]) and writes `mtp_fixture.safetensors`:
  input_ids [T] i32, target_hidden [T, H], mtp{k}_out [T-1-k, H] (final-normed),
  mtp{k}_logits [T-1-k, V]  (all f32).
Exit 2 = SKIP (missing torch/transformers/numpy/safetensors).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dump_mimo_v2_fixtures as base  # noqa: E402

HEADS = 3


def build_heads(mm, cfg, seed):
    import torch
    from torch import nn
    heads = []
    g = torch.Generator().manual_seed(seed + 101)
    for _ in range(HEADS):
        h = nn.Module()
        h.enorm = mm.MiMoV2RMSNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        h.hnorm = mm.MiMoV2RMSNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        h.eh_proj = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        h.input_layernorm = mm.MiMoV2RMSNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        h.self_attn = mm.MiMoV2Attention(cfg, True, 0, projection_layout="fused_qkv")
        h.pre_mlp_layernorm = mm.MiMoV2RMSNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        h.mlp = mm.MiMoV2MLP(cfg)
        h.final_layernorm = mm.MiMoV2RMSNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        h = h.float().eval()
        with torch.no_grad():
            for name, p in h.named_parameters():
                if name.endswith("attention_sink_bias"):
                    p.copy_(torch.randn(p.shape, generator=g))
                elif "norm" in name:
                    p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.1)
                else:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        heads.append(h)
    return heads


def sliding_mask(rows, window):
    """Additive [1, 1, rows, rows] mask: query i sees keys (i - window, i]."""
    import torch
    i = torch.arange(rows)[:, None]
    j = torch.arange(rows)[None, :]
    keep = (j <= i) & (j > i - window)
    m = torch.zeros(rows, rows)
    m[~keep] = float("-inf")
    return m[None, None]


def head_forward(model, head, cfg, target_hidden, ids, k):
    import torch
    T = ids.shape[0]
    rows = T - 1 - k
    emb = model.model.embed_tokens(torch.tensor(ids[k + 1: k + 1 + rows], dtype=torch.long))[None]
    prev = target_hidden[:, :rows]
    x = head.eh_proj(torch.cat([head.enorm(emb), head.hnorm(prev)], dim=-1))
    pos = torch.arange(rows)[None]
    cos_sin = model.model.swa_rotary_emb(x, pos)
    attn, _ = head.self_attn(
        hidden_states=head.input_layernorm(x),
        position_embeddings=cos_sin,
        attention_mask=sliding_mask(rows, cfg.sliding_window),
        position_ids=pos,
    )
    x = x + attn
    x = x + head.mlp(head.pre_mlp_layernorm(x))
    out = head.final_layernorm(x)
    return out[0], model.lm_head(out)[0]


def run(out_dir, seed, ref, cache_dir, offline):
    try:
        import numpy as np
        import torch
        from safetensors.numpy import save_file
    except ImportError as e:
        base._die_skip(f"python deps unavailable ({e})")
    base.resolve_reference(ref, cache_dir, offline)
    mm = __import__("mimo_v2_ref.modeling_mimo_v2", fromlist=["modeling_mimo_v2"])
    out_dir = Path(out_dir).expanduser()
    T = base.tiny_dims()["t_total"]
    cfg, model = base.build_model(mm, seed)
    ids = np.random.default_rng(seed).integers(2, base.TINY["vocab_size"], size=T).astype(np.int32)
    from safetensors.numpy import load_file
    trunk = load_file(str(out_dir / "model.safetensors"))
    sd = model.state_dict()
    for k, v in sd.items():
        if not np.array_equal(trunk[k], v.detach().numpy().astype(np.float32)):
            raise base.FixtureError(f"{k}: the trunk in {out_dir} is not this seed's trunk")
    heads = build_heads(mm, cfg, seed)
    fx = {"input_ids": ids}
    with torch.no_grad():
        target = model.model(input_ids=torch.tensor(ids, dtype=torch.long)[None]).last_hidden_state
        fx["target_hidden"] = target[0].numpy().astype(np.float32)
        weights = {}
        for k, head in enumerate(heads):
            out, logits = head_forward(model, head, cfg, target, ids, k)
            fx[f"mtp{k}_out"] = out.numpy().astype(np.float32)
            fx[f"mtp{k}_logits"] = logits.numpy().astype(np.float32)
            for name, p in head.state_dict().items():
                weights[f"model.mtp.layers.{k}.{name}"] = p.numpy().astype(np.float32)
    save_file(weights, str(out_dir / "model_mtp.safetensors"))
    index_path = out_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"] = {k: v for k, v in index["weight_map"].items() if not k.startswith("model.mtp.")}
    index["weight_map"].update({k: "model_mtp.safetensors" for k in weights})
    index_path.write_text(json.dumps(index, indent=2))
    save_file({k: np.ascontiguousarray(v) for k, v in fx.items()}, str(out_dir / "mtp_fixture.safetensors"))
    print(f"wrote {out_dir}: {HEADS} heads, T={T}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="a tests/dump_mimo_v2_fixtures.py dump of the same seed")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--cache-dir", default=str(base.DEFAULT_CACHE))
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    try:
        return run(a.out_dir, a.seed, a.ref, a.cache_dir, a.offline)
    except base.FixtureError as e:
        sys.stderr.write(f"FAIL: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
