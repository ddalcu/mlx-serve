#!/usr/bin/env python3
"""Reference-parity oracle for the MiMo-V2.6 engine arm (model_type "mimo_v2").

The full 309B checkpoint cannot run in torch here, so this builds a TINY
random-weight MiMoV2Config model from the HF reference modeling code
(XiaomiMiMo/MiMo-V2.6-Flash-RL, Apache-2.0) and dumps per-layer tensors the
Zig fixture tests compare against (cosine / max-abs bars).  The tiny config
keeps the REAL structural geometry: qk head_dim 192 / v head_dim 128 (the
v-width asymmetry), partial_rotary_factor 0.334 (rope_dim = int(192*0.334) =
64, rotate_half over the first 64 dims), attention_value_scale 0.707,
distinct GA vs SWA rope thetas, sliding_window 128 with a per-head
attention_sink_bias on SWA layers only, sigmoid MoE scoring +
e_score_correction_bias + norm_topk_prob.  Two passes run over the same
weights and prompt: (a) full-sequence prefill and (b) token-by-token with
cache; both are dumped and the script asserts the reference is self-
consistent (prefill-vs-cache margin bound 1e-3) and records the observed
margins in the fixture metadata.

Usage:
  tests/dump_mimo_v2_fixtures.py OUT_DIR [--seed N] [--ref PATH]
      [--cache-dir PATH] [--offline]
  tests/dump_mimo_v2_fixtures.py --verify OUT_DIR

OUT_DIR receives (schema v1; see fixture_schema()/weight_schema()):
  config.json                    the tiny MiMoV2Config (model_type mimo_v2)
  model.safetensors(+index)      tiny weights, HF checkpoint naming, f32
  fixture.safetensors            activations + routing detail + rope tables,
                                 metadata key "mimo_v2_fixture" = JSON
                                 {schema_version, seed, dims, margins, ...}

Exit codes: 0 ok; 1 failure (invariant or environment, stderr detail);
2 SKIP (missing torch/transformers/numpy/safetensors — nothing is faked).

The reference pair (modeling_mimo_v2.py + configuration_mimo_v2.py) is
downloaded once into --cache-dir and imported as a package (its relative
`from .configuration_mimo_v2 import` needs one).  --offline refuses the
network and requires the cache (or --ref); once cached every run is
hermetic.  The oracle runs the reference's OWN eager attention
(`_attn_implementation = "eager"` resolves to the remote module's
eager_attention_forward — "eager" is not registered in transformers'
AttentionInterface — so the sink path's `is` identity check fires and every
layer runs the same math).  Captures wrap that function and the MoE gate and
are validated against the reference's own outputs on EVERY call.  Reference
gotchas pinned here: the sink logit enters the softmax as one extra column
(constant per head) and is dropped before the value mix; value_states are
scaled by attention_value_scale BEFORE the KV cache update; the SWA window
admits keys [pos-sliding_window+1, pos] (128 incl. self) and the SWA cache
trims to sliding_window-1 history keys; MoE SELECTION uses
scores + e_score_correction_bias while the weights gathered are the RAW
sigmoid scores (norm_topk_prob divides by sum + 1e-20);
config.moe_router_dtype is IGNORED by the reference (routing is f32).
"""
import argparse
import json
import os
import shutil
import struct
import sys
import urllib.request
from pathlib import Path

# The reference's use_kernel_forward_from_hub("RMSNorm") decorator must stay a
# no-op: this oracle is hermetic once the reference pair is cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REF_BASE = "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/raw/main/"
REF_FILES = ("configuration_mimo_v2.py", "modeling_mimo_v2.py")
DEFAULT_CACHE = Path("~/.cache/sushi/mimo-v2-ref").expanduser()
BOUND = 1e-3  # prefill-vs-cache self-consistency bound (f32 reference)

TINY = dict(
    vocab_size=128,
    hidden_size=384,
    intermediate_size=1536,          # 4x hidden (the real ratio)
    num_hidden_layers=4,
    num_attention_heads=4,           # GA layers
    num_key_value_heads=2,
    head_dim=192,                    # REAL dims: the v-width asymmetry is under test
    v_head_dim=128,
    swa_num_attention_heads=6,       # SWA layers: own head counts + own rope theta
    swa_num_key_value_heads=2,
    swa_head_dim=192,
    swa_v_head_dim=128,
    swa_rope_theta=10000.0,
    sliding_window=128,
    sliding_window_size=128,
    add_swa_attention_sink_bias=True,
    add_full_attention_sink_bias=False,
    hybrid_layer_pattern=[0, 1, 1, 0],   # 0 = GA/global, 1 = SWA
    moe_layer_freq=[0, 1, 1, 1],          # layer 0 = the dense SwiGLU
    n_routed_experts=16,
    num_experts_per_tok=4,
    moe_intermediate_size=192,       # 0.5x hidden (the real ratio)
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=None,
    partial_rotary_factor=0.334,
    rope_theta=10000000.0,
    attention_value_scale=0.707,
    attention_projection_layout="fused_qkv",
    tie_word_embeddings=False,
    max_position_embeddings=4096,
    layernorm_epsilon=1e-6,
    hidden_act="silu",
    eos_token_id=1,
    pad_token_id=0,
)

T_PREFILL = 160   # crosses the 128 window: 32 fully-windowed rows + 6 decode rows
T_DECODE = 6


class FixtureError(Exception):
    """An invariant of the dump or its self-verify is violated."""


def rope_dim_for(head_dim, partial_rotary_factor):
    """The reference's partial-rotary width: int(head_dim * factor), even."""
    d = int(head_dim * partial_rotary_factor)
    if d <= 0 or d % 2 != 0:
        raise FixtureError(
            f"rope_dim must be positive and even, got {d} from "
            f"head_dim={head_dim} and partial_rotary_factor={partial_rotary_factor}")
    return d


def tiny_dims():
    """Every dimension the fixture schema is parameterized on."""
    return dict(
        t_total=T_PREFILL + T_DECODE,
        t_prefill=T_PREFILL,
        t_decode=T_DECODE,
        hidden=TINY["hidden_size"],
        vocab=TINY["vocab_size"],
        layers=TINY["num_hidden_layers"],
        ga_heads=TINY["num_attention_heads"],
        ga_kv=TINY["num_key_value_heads"],
        swa_heads=TINY["swa_num_attention_heads"],
        swa_kv=TINY["swa_num_key_value_heads"],
        head_dim=TINY["head_dim"],
        v_head_dim=TINY["v_head_dim"],
        experts=TINY["n_routed_experts"],
        topk=TINY["num_experts_per_tok"],
        rope_dim=rope_dim_for(TINY["head_dim"], TINY["partial_rotary_factor"]),
        pattern=list(TINY["hybrid_layer_pattern"]),
        moe=[bool(x) for x in TINY["moe_layer_freq"]],
    )


def fixture_schema(d):
    """fixture.safetensors contract: key -> (safetensors dtype, shape).

    Naming: stream_i = residual INTO layer i (stream_0 = embed_out),
    l{i}_* = layer-i sub-module captures of pass (a) full prefill,
    cache_* = pass (b) token-by-token with cache (rows assembled in
    position order), *_dec = the cached-decode rows only.  Every
    l{i}_attn_probs row is the POST-drop key-probabilities (what multiplies
    the values); l{i}_attn_sink carries the dropped sink column so the
    Zig side can check the same softmax-with-sink normalization.
    """
    T, H, V, TD = d["t_total"], d["hidden"], d["vocab"], d["t_decode"]
    s = {
        "input_ids": ("I32", (T,)),
        "embed_out": ("F32", (T, H)),
        "logits_full": ("F32", (T, V)),
        "logit_margin": ("F32", (T,)),
        "final_norm": ("F32", (T, H)),
        "moe_route_gap": ("F32", (T,)),
        "rope_cos_ga": ("F32", (T, d["rope_dim"])),
        "rope_sin_ga": ("F32", (T, d["rope_dim"])),
        "rope_cos_swa": ("F32", (T, d["rope_dim"])),
        "rope_sin_swa": ("F32", (T, d["rope_dim"])),
        "cache_logits": ("F32", (T, V)),
        "cache_final_norm": ("F32", (T, H)),
        "stream_4": ("F32", (T, H)),
        "cache_stream_4": ("F32", (T, H)),
    }
    for i in range(d["layers"]):
        heads = d["swa_heads"] if d["pattern"][i] == 1 else d["ga_heads"]
        s[f"stream_{i}"] = ("F32", (T, H))
        s[f"l{i}_attn_out"] = ("F32", (T, H))
        s[f"l{i}_mlp_out"] = ("F32", (T, H))
        s[f"l{i}_attn_probs"] = ("F32", (heads, T, T))
        s[f"l{i}_attn_sink"] = ("F32", (heads, T))
        s[f"l{i}_vis_from"] = ("I32", (T,))
        s[f"cache_stream_{i}"] = ("F32", (T, H))
        s[f"cache_l{i}_attn_out"] = ("F32", (T, H))
        s[f"cache_l{i}_mlp_out"] = ("F32", (T, H))
        s[f"cache_l{i}_attn_probs_dec"] = ("F32", (heads, TD, T))
        s[f"cache_l{i}_attn_sink_dec"] = ("F32", (heads, TD))
        s[f"cache_l{i}_vis_from_dec"] = ("I32", (TD,))
        if d["moe"][i]:
            E, K = d["experts"], d["topk"]
            s[f"l{i}_moe_scores"] = ("F32", (T, E))
            s[f"l{i}_moe_topk_idx"] = ("I32", (T, K))
            s[f"l{i}_moe_topk_w_pre"] = ("F32", (T, K))
            s[f"l{i}_moe_topk_w_post"] = ("F32", (T, K))
            s[f"l{i}_moe_rank_gaps"] = ("F32", (T, E - 1))
            s[f"l{i}_moe_route_gap"] = ("F32", (T,))
            s[f"cache_l{i}_moe_scores_dec"] = ("F32", (TD, E))
            s[f"cache_l{i}_moe_topk_idx_dec"] = ("I32", (TD, K))
            s[f"cache_l{i}_moe_topk_w_pre_dec"] = ("F32", (TD, K))
            s[f"cache_l{i}_moe_topk_w_post_dec"] = ("F32", (TD, K))
    return s


def weight_schema(d):
    """model.safetensors contract: HF checkpoint naming -> shape (f32)."""
    H, V, E = d["hidden"], d["vocab"], d["experts"]
    inter = TINY["intermediate_size"]
    me = TINY["moe_intermediate_size"]
    w = {
        "model.embed_tokens.weight": (V, H),
        "model.norm.weight": (H,),
        "lm_head.weight": (V, H),
    }
    for i in range(d["layers"]):
        swa = d["pattern"][i] == 1
        heads = d["swa_heads"] if swa else d["ga_heads"]
        kv = d["swa_kv"] if swa else d["ga_kv"]
        hd, vhd = d["head_dim"], d["v_head_dim"]
        w[f"model.layers.{i}.input_layernorm.weight"] = (H,)
        w[f"model.layers.{i}.post_attention_layernorm.weight"] = (H,)
        # fused_qkv: one projection carrying q(head_dim) + k(head_dim) + v(v_head_dim)
        w[f"model.layers.{i}.self_attn.qkv_proj.weight"] = (heads * hd + kv * hd + kv * vhd, H)
        w[f"model.layers.{i}.self_attn.o_proj.weight"] = (H, heads * vhd)
        if TINY["add_swa_attention_sink_bias"] if swa else TINY["add_full_attention_sink_bias"]:
            w[f"model.layers.{i}.self_attn.attention_sink_bias"] = (heads,)
        if not d["moe"][i]:
            w[f"model.layers.{i}.mlp.gate_proj.weight"] = (inter, H)
            w[f"model.layers.{i}.mlp.up_proj.weight"] = (inter, H)
            w[f"model.layers.{i}.mlp.down_proj.weight"] = (H, inter)
        else:
            w[f"model.layers.{i}.mlp.gate.weight"] = (E, H)
            w[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] = (E,)
            for e in range(E):
                w[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"] = (me, H)
                w[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"] = (me, H)
                w[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"] = (H, me)
    return w


def logit_margin_np(logits):
    """Per-row top1-top2 gap (the argmax tie measure)."""
    import numpy as np
    top2 = np.sort(logits, axis=-1)[:, -2:][:, ::-1]
    return (top2[:, 0] - top2[:, 1]).astype(np.float32)


def routing_metrics(scores, bias, top_k):
    """(rank_gaps, route_gap) over the SELECTION scores (raw sigmoid +
    e_score_correction_bias): the tie measure the Zig test acquits by."""
    import numpy as np
    choice = scores + bias[None, :]
    srt = np.sort(choice, axis=-1)[:, ::-1]
    gaps = (srt[:, :-1] - srt[:, 1:]).astype(np.float32)
    rg = ((srt[:, top_k - 1] - srt[:, top_k]) / np.maximum(srt[:, 0], np.float32(1e-9))).astype(np.float32)
    return gaps, rg


def read_header(path):
    """safetensors header as (tensor-info dict, metadata dict) — stdlib only."""
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return ({k: v for k, v in hdr.items() if k != "__metadata__"},
            dict(hdr.get("__metadata__", {})))


def _die_skip(msg):
    sys.stderr.write(f"SKIP: {msg}\n")
    raise SystemExit(2)


def resolve_reference(ref, cache_dir, offline):
    """The HF reference pair as an importable package under cache_dir."""
    pkg = Path(cache_dir).expanduser() / "mimo_v2_ref"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").touch()
    if ref:
        src = Path(ref).expanduser()
        src = src.parent if src.is_file() else src
        for f in REF_FILES:
            if not (src / f).is_file():
                raise FixtureError(f"--ref {src} lacks {f}")
            shutil.copyfile(src / f, pkg / f)
    else:
        for f in REF_FILES:
            dst = pkg / f
            if dst.is_file():
                continue
            if offline:
                raise FixtureError(
                    f"--offline and {f} is not cached under {pkg}; pass --ref or run once online")
            tmp = dst.with_suffix(".part")
            try:
                with urllib.request.urlopen(REF_BASE + f, timeout=30) as r, open(tmp, "wb") as out:
                    shutil.copyfileobj(r, out)
            except Exception as e:
                tmp.unlink(missing_ok=True)
                raise FixtureError(f"downloading {REF_BASE + f} failed: {e}")
            tmp.rename(dst)
    sys.path.insert(0, str(pkg.parent))
    return pkg


def build_model(mm, seed):
    """TINY MiMoV2ForCausalLM with deterministic per-parameter draws.  The
two torch.empty parameters the reference leaves uninitialized
    (attention_sink_bias, e_score_correction_bias) MUST be filled here."""
    import torch
    cfg = mm.MiMoV2Config(**TINY)
    # "eager" is unregistered in transformers' AttentionInterface, so
    # get_interface falls back to the remote module's own eager_attention_forward
    # and the sink branch's `is` identity check fires — uniform math per layer.
    cfg._attn_implementation = "eager"
    model = mm.MiMoV2ForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("attention_sink_bias"):
                p.copy_(torch.randn(p.shape, generator=g))
            elif name.endswith("e_score_correction_bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
            elif name.endswith("mlp.gate.weight"):
                # modest router logits: the correction bias spreads selection
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            elif "norm" in name:
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.1)
            elif p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * (0.08 if ".experts." in name else 0.05))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return cfg, model


def run_passes(mm, model, ids_np):
    """Pass (a) full prefill + pass (b) cached prefill/decode with captures.

    Returns rec = {"a": {...}, "b": {...}} of per-call records.  The eager
capture recomputes the reference's own softmax to obtain the probs BEFORE
    the sink column is dropped and asserts torch.equal against the reference's
    outputs on every call; the gate captures are validated the same way."""
    import torch
    import torch.nn.functional as F

    ids = torch.tensor(ids_np, dtype=torch.long)[None, :]
    T = ids.shape[1]
    rec = {"a": _new_pass(), "b": _new_pass()}
    state = {"phase": "a"}

    def cur():
        return rec[state["phase"]]

    orig_eager = mm.eager_attention_forward

    def eager_capture(module, query, key, value, attention_mask, scaling, dropout=0.0, sinks=None, **kw):
        out = orig_eager(module, query, key, value, attention_mask, scaling,
                         dropout=dropout, sinks=sinks, **kw)
        key_states = mm.repeat_kv(key, module.num_key_value_groups)
        value_states = mm.repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        kv = key_states.shape[-2]
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, :kv]
        if sinks is not None:
            col = module.attention_sink_bias.reshape(1, -1, 1, 1).expand(
                query.shape[0], -1, query.shape[-2], -1)
            attn_weights = torch.cat([attn_weights, col], dim=-1)
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
        probs_full = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        probs_keys = probs_full[..., :-1] if sinks is not None else probs_full
        check = torch.matmul(probs_keys, value_states).transpose(1, 2).contiguous()
        if not (torch.equal(check, out[0]) and torch.equal(probs_keys, out[1])):
            raise FixtureError(
                f"capture recompute diverged from the reference eager path (layer {module.layer_idx})")
        if attention_mask is None:
            raise FixtureError("no additive mask observed — visibility capture needs it")
        pos = kw.get("position_ids")
        if pos is None:
            raise FixtureError("no position_ids observed — absolute key mapping needs it")
        key_pos0 = int(pos.reshape(-1)[-1]) + 1 - kv
        vis = (attention_mask[0, 0, :, :kv] == 0)
        if not bool(vis.any(dim=-1).all()):
            raise FixtureError(f"a query row saw no visible key (layer {module.layer_idx})")
        first_col = vis.to(torch.int64).argmax(dim=-1)
        cur()["attn"].append(dict(
            layer=module.layer_idx,
            is_swa=module.is_swa,
            probs=probs_full.detach().cpu()[0],   # batch 1 -> [h, q, kv(+1 with sink)]
            q_pos=[int(x) for x in pos.reshape(-1)],
            key_pos0=key_pos0,
            vis=[int(x) + key_pos0 for x in first_col],
            has_sink=sinks is not None,
        ))
        return out

    hooks = []
    with torch.no_grad():
        mm.eager_attention_forward = eager_capture
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_pre_hook(
                lambda mod, args, i=i: cur()["layer_in"].append((i, args[0].detach().cpu()))))
            hooks.append(layer.register_forward_hook(
                lambda mod, args, out, i=i: cur()["layer_out"].append((i, out.detach().cpu()))))
            hooks.append(layer.self_attn.register_forward_hook(
                lambda mod, args, out, i=i: cur()["attn_out"].append((i, out[0].detach().cpu()))))
            hooks.append(layer.mlp.register_forward_hook(
                lambda mod, args, out, i=i: cur()["mlp_out"].append((i, out.detach().cpu()))))
            if hasattr(layer.mlp, "gate"):
                hooks.append(layer.mlp.gate.register_forward_pre_hook(
                    lambda mod, args, i=i: cur()["gate"].append(
                        dict(layer=i, h=args[0].detach().cpu()))))
                hooks.append(layer.mlp.gate.register_forward_hook(
                    lambda mod, args, out, i=i: cur()["gate"][-1].update(
                        idx=out[0].detach().cpu(), w_post=out[1].detach().cpu())))
        hooks.append(model.model.norm.register_forward_hook(
            lambda mod, args, out: cur()["final"].append(out.detach().cpu())))
        hooks.append(model.model.embed_tokens.register_forward_hook(
            lambda mod, args, out: cur()["embed"].append(out.detach().cpu())))
        for tag, rope in (("ga", model.model.rotary_emb), ("swa", model.model.swa_rotary_emb)):
            hooks.append(rope.register_forward_hook(
                lambda mod, args, out, tag=tag: cur()["rope"].update(
                    {tag: (out[0].detach().cpu(), out[1].detach().cpu())})))
        try:
            state["phase"] = "a"
            out_a = model(input_ids=ids, use_cache=False)
            state["phase"] = "b"
            out_b = model(input_ids=ids[:, :T_PREFILL], use_cache=True)
            pkv = out_b.past_key_values
            dec_logits = []
            for t in range(T_PREFILL, T):
                step = model(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
                pkv = step.past_key_values
                dec_logits.append(step.logits[0, -1].detach().cpu())
        finally:
            mm.eager_attention_forward = orig_eager
            for h in hooks:
                h.remove()
    rec["a"]["logits"] = out_a.logits[0].detach().cpu()
    rec["b"]["logits"] = torch.cat(
        [out_b.logits[0].detach().cpu()] + [x[None, :] for x in dec_logits], dim=0)
    return rec


def _new_pass():
    return dict(attn=[], gate=[], layer_in=[], layer_out=[], attn_out=[], mlp_out=[],
                final=[], embed=[], rope={}, logits=None)


def gate_metrics(model, g):
    """Validate one gate call against the reference's own outputs and return
    the numpy routing detail (scores = RAW sigmoid; selection = scores + bias)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    i = g["layer"]
    gate = model.model.layers[i].mlp.gate
    h = g["h"].reshape(-1, g["h"].shape[-1])
    with torch.no_grad():
        logits = F.linear(h.type(torch.float32), gate.weight.type(torch.float32), None)
        scores = logits.sigmoid()
        choice = scores + gate.e_score_correction_bias.unsqueeze(0)
        k = gate.top_k
        w_pre = scores.gather(1, g["idx"])
        w_post = w_pre / (w_pre.sum(dim=-1, keepdim=True) + 1e-20)
        w_post = w_post * gate.routed_scaling_factor
        chosen = choice.gather(1, g["idx"])
        rejected = choice.masked_fill(
            F.one_hot(g["idx"], choice.shape[-1]).any(dim=1), float("-inf"))
    if not torch.equal(w_post, g["w_post"]):
        raise FixtureError(f"layer {i} MoE: reference topk_weight != raw scores gathered then norm_topk_prob")
    if bool((chosen.min(dim=-1).values < rejected.max(dim=-1).values).any()):
        raise FixtureError(f"layer {i} MoE: reference topk ids are not the top-{k} of scores + e_score_correction_bias")
    scores_np = scores.numpy().astype(np.float32)
    bias_np = gate.e_score_correction_bias.detach().numpy().astype(np.float32)
    gaps, rg = routing_metrics(scores_np, bias_np, k)
    return dict(
        scores=scores_np,
        idx=g["idx"].numpy().astype(np.int32),
        w_pre=w_pre.numpy().astype(np.float32),
        w_post=g["w_post"].numpy().astype(np.float32),
        rank_gaps=gaps,
        route_gap=rg,
    )


def assemble(model, rec, d):
    """Captures -> the fixture arrays named by fixture_schema()."""
    import numpy as np
    T, H, TD = d["t_total"], d["hidden"], d["t_decode"]
    fx = {}

    def rows(pass_rec, key, want, label):
        items = sorted(pass_rec[key], key=lambda t: t[0])
        out = []
        for i in range(d["layers"]):
            mine = [t for t in items if t[0] == i]
            cat = np.concatenate([t[1].numpy().reshape(-1, H) for t in mine], axis=0)
            if cat.shape[0] != want:
                raise FixtureError(f"{label} layer {i}: {cat.shape[0]} rows != {want}")
            out.append(cat)
        return out

    a, b = rec["a"], rec["b"]
    if len(a["embed"]) != 1 or len(a["final"]) != 1 or \
            len(b["embed"]) != 1 + TD or len(b["final"]) != 1 + TD:
        raise FixtureError("embed/final-norm hook counts off")
    fx["embed_out"] = a["embed"][0][0].numpy().astype(np.float32)
    fx["final_norm"] = a["final"][0][0].numpy().astype(np.float32)
    fx["cache_final_norm"] = np.concatenate(
        [x[0].numpy().reshape(-1, H) for x in b["final"]], axis=0).astype(np.float32)

    a_in = rows(a, "layer_in", T, "layer_in a")
    a_out = rows(a, "layer_out", T, "layer_out a")
    a_attn = rows(a, "attn_out", T, "attn_out a")
    a_mlp = rows(a, "mlp_out", T, "mlp_out a")
    b_in = rows(b, "layer_in", T, "layer_in b")
    b_out = rows(b, "layer_out", T, "layer_out b")
    b_attn = rows(b, "attn_out", T, "attn_out b")
    b_mlp = rows(b, "mlp_out", T, "mlp_out b")
    for i in range(d["layers"]):
        fx[f"stream_{i}"] = a_in[i].astype(np.float32)      # layer i's input residual
        fx[f"l{i}_attn_out"] = a_attn[i].astype(np.float32)
        fx[f"l{i}_mlp_out"] = a_mlp[i].astype(np.float32)
        fx[f"cache_stream_{i}"] = b_in[i].astype(np.float32)
        fx[f"cache_l{i}_attn_out"] = b_attn[i].astype(np.float32)
        fx[f"cache_l{i}_mlp_out"] = b_mlp[i].astype(np.float32)
    fx["stream_4"] = a_out[3].astype(np.float32)            # last layer's output
    fx["cache_stream_4"] = b_out[3].astype(np.float32)
    fx["logits_full"] = a["logits"].numpy().astype(np.float32)
    fx["cache_logits"] = b["logits"].numpy().astype(np.float32)

    for tag in ("ga", "swa"):
        cos, sin = a["rope"][tag]
        fx[f"rope_cos_{tag}"] = cos[0].numpy().astype(np.float32)
        fx[f"rope_sin_{tag}"] = sin[0].numpy().astype(np.float32)

    for rec_attn, prefill in ((a["attn"], True), (b["attn"], False)):
        by_layer = {}
        for r in rec_attn:
            by_layer.setdefault(r["layer"], []).append(r)
        for i, rs in by_layer.items():
            if prefill:
                if len(rs) != 1 or len(rs[0]["q_pos"]) != T:
                    raise FixtureError(f"attn calls layer {i} (pass a): {len(rs)} calls != 1 full-sequence")
                keys = rs[0]["probs"].numpy().astype(np.float32)
                if rs[0]["has_sink"]:
                    fx[f"l{i}_attn_sink"] = keys[:, :, -1]
                    keys = keys[:, :, :-1]
                else:
                    fx[f"l{i}_attn_sink"] = np.zeros(keys.shape[:2], dtype=np.float32)
                fx[f"l{i}_attn_probs"] = keys
                fx[f"l{i}_vis_from"] = np.asarray(rs[0]["vis"], dtype=np.int32)
            else:
                if len(rs) != TD + 1:
                    raise FixtureError(f"attn calls layer {i} (pass b): {len(rs)} != {TD + 1}")
                dec = sorted((r for r in rs if len(r["q_pos"]) == 1), key=lambda r: r["q_pos"][0])
                heads = dec[0]["probs"].shape[0]
                probs = np.zeros((heads, TD, T), dtype=np.float32)
                sink = np.zeros((heads, TD), dtype=np.float32)
                vis = np.zeros(TD, dtype=np.int32)
                for s, r in enumerate(dec):
                    p = r["probs"].numpy().astype(np.float32)
                    k0, kv = r["key_pos0"], p.shape[-1] - (1 if r["has_sink"] else 0)
                    if r["has_sink"]:
                        sink[:, s] = p[:, 0, -1]
                        p = p[:, :, :-1]
                    probs[:, s, k0:k0 + kv] = p[:, 0, :]
                    vis[s] = r["vis"][0]
                fx[f"cache_l{i}_attn_probs_dec"] = probs
                fx[f"cache_l{i}_attn_sink_dec"] = sink
                fx[f"cache_l{i}_vis_from_dec"] = vis

    gaps_all = []
    for g in a["gate"]:
        i = g["layer"]
        if g["h"].shape[1] != T:
            raise FixtureError(f"MoE gate layer {i} (pass a): {g['h'].shape[1]} rows != {T}")
        m = gate_metrics(model, g)
        fx[f"l{i}_moe_scores"] = m["scores"]
        fx[f"l{i}_moe_topk_idx"] = m["idx"]
        fx[f"l{i}_moe_topk_w_pre"] = m["w_pre"]
        fx[f"l{i}_moe_topk_w_post"] = m["w_post"]
        fx[f"l{i}_moe_rank_gaps"] = m["rank_gaps"]
        fx[f"l{i}_moe_route_gap"] = m["route_gap"]
        gaps_all.append(m["route_gap"])
    for i in range(d["layers"]):
        if not d["moe"][i]:
            continue
        decs = [g for g in b["gate"] if g["layer"] == i and g["h"].shape[1] == 1]
        if len(decs) != TD:
            raise FixtureError(f"MoE gate layer {i} (pass b): {len(decs)} decode calls != {TD}")
        ms = [gate_metrics(model, g) for g in decs]
        fx[f"cache_l{i}_moe_scores_dec"] = np.concatenate([m["scores"] for m in ms], axis=0)
        fx[f"cache_l{i}_moe_topk_idx_dec"] = np.concatenate([m["idx"] for m in ms], axis=0)
        fx[f"cache_l{i}_moe_topk_w_pre_dec"] = np.concatenate([m["w_pre"] for m in ms], axis=0)
        fx[f"cache_l{i}_moe_topk_w_post_dec"] = np.concatenate([m["w_post"] for m in ms], axis=0)
    fx["moe_route_gap"] = np.minimum.reduce(gaps_all).astype(np.float32)
    fx["logit_margin"] = logit_margin_np(fx["logits_full"])
    fx["input_ids"] = np.asarray([], dtype=np.int32)  # filled by the caller
    return fx


def verify_out(out):
    """Reload OUT_DIR and re-run the whole invariant battery.  Returns a list
of problem strings (empty = pass).  numpy + safetensors only."""
    try:
        import numpy as np
        from safetensors.numpy import load_file
    except ImportError as e:
        _die_skip(f"numpy/safetensors unavailable ({e})")
    problems = []

    def fail(msg):
        problems.append(msg)

    d = tiny_dims()
    out = Path(out)
    fx_path = out / "fixture.safetensors"
    wt_path = out / "model.safetensors"
    for p in (fx_path, wt_path, out / "config.json", out / "model.safetensors.index.json"):
        if not p.is_file():
            return [f"{p.name}: missing from {out}"]
    fx = load_file(str(fx_path))
    wt = load_file(str(wt_path))
    for label, tensors in (("fixture", fx), ("model", wt)):
        for name, tensor in tensors.items():
            if np.issubdtype(tensor.dtype, np.floating):
                bad = int(np.count_nonzero(~np.isfinite(tensor)))
                if bad:
                    fail(f"{label} tensor {name}: contains {bad} non-finite values")
    _, meta = read_header(fx_path)
    try:
        info = json.loads(meta["mimo_v2_fixture"])
    except (KeyError, ValueError) as e:
        return [f"fixture metadata: {e}"]

    # -- shape math and the exact schema -----------------------------------
    for key, (dt, shp) in fixture_schema(d).items():
        want = "int32" if dt == "I32" else "float32"
        if key not in fx:
            fail(f"{key}: missing from fixture")
            continue
        if str(fx[key].dtype) != want:
            fail(f"{key}: dtype {fx[key].dtype} != {want}")
        if tuple(fx[key].shape) != tuple(shp):
            fail(f"{key}: shape {tuple(fx[key].shape)} != {tuple(shp)}")
    for key in fx:
        if key not in fixture_schema(d):
            fail(f"{key}: unexpected fixture key")
    for key, shp in weight_schema(d).items():
        if key not in wt:
            fail(f"{key}: missing from model.safetensors")
        elif tuple(wt[key].shape) != tuple(shp):
            fail(f"{key}: shape {tuple(wt[key].shape)} != {tuple(shp)}")
    for key in wt:
        if key not in weight_schema(d):
            fail(f"{key}: unexpected weight key")
    if info.get("schema_version") != 1:
        fail(f"metadata schema_version {info.get('schema_version')!r} != 1")
    if info.get("dims") != json.loads(json.dumps(d)):
        fail("metadata dims != tiny_dims()")
    for name, value in info.get("margins", {}).items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            fail(f"fixture metadata margins.{name}: non-finite value")

    cfg = json.loads((out / "config.json").read_text())
    if cfg.get("model_type") != "mimo_v2":
        fail(f"config.json model_type {cfg.get('model_type')!r} != 'mimo_v2'")
    if (cfg.get("head_dim"), cfg.get("v_head_dim")) != (d["head_dim"], d["v_head_dim"]):
        fail("config.json head_dim/v_head_dim asymmetry not preserved")
    if rope_dim_for(cfg["head_dim"], cfg["partial_rotary_factor"]) != d["rope_dim"]:
        fail("config.json rope_dim shape math")

    # -- softmax-with-sink + visibility per layer ---------------------------
    T, TD = d["t_total"], d["t_decode"]
    W = TINY["sliding_window"]
    for i in range(d["layers"]):
        swa = d["pattern"][i] == 1
        for suffix, rows_, vis_key in (("", range(T), f"l{i}_vis_from"),
                                       ("_dec", range(TD), f"cache_l{i}_vis_from_dec")):
            pre = "l" if not suffix else f"cache_l"
            keys = fx[f"{pre}{i}_attn_probs{'_dec' if suffix else ''}"]
            sink = fx[f"{pre}{i}_attn_sink{'_dec' if suffix else ''}"]
            vis = fx[vis_key]
            sums = keys.astype(np.float64).sum(-1) + sink.astype(np.float64)
            err = float(np.abs(sums - 1.0).max())
            if err > 1e-5:
                fail(f"{pre}{i}_attn_probs/{pre}{i}_attn_sink: softmax-with-sink rows must sum to 1 "
                     f"before the sink column is dropped (max |sum-1| = {err:.2e})")
            if swa and float(sink.min()) <= 0.0:
                fail(f"{pre}{i}_attn_sink: SWA sink column must carry mass on every row")
            if not swa and bool(np.any(sink != 0)):
                fail(f"{pre}{i}_attn_sink: GA layers have no sink")
            offs = set()
            for s in rows_:
                pos = s if not suffix else d["t_prefill"] + s
                row = keys[:, s, :]
                if bool(np.any(row[:, :int(vis[s])] != 0)):
                    fail(f"{pre}{i}_attn_probs: mass left of vis_from at row {pos}")
                if bool(np.any(row[:, pos + 1:] != 0)):
                    fail(f"{pre}{i}_attn_probs: mass right of the causal diagonal at row {pos}")
                if swa:
                    want = max(0, pos - W + 1)
                    if int(vis[s]) != want:
                        fail(f"{vis_key}: row {pos} sees from {int(vis[s])}, expected {want} "
                             f"(keys [pos-{W}+1, pos] incl. self)")
                    if int(vis[s]) > 0:
                        offs.add(int(vis[s]) - pos + W)
                elif int(vis[s]) != 0:
                    fail(f"{vis_key}: GA rows must see the full prefix")
            if swa:
                if not offs:
                    fail(f"{vis_key}: the {W}-token window never engages")
                elif offs != {1}:
                    fail(f"{vis_key}: inconsistent window offsets {sorted(offs)}")
                if info.get("swa_window_offset") != 1:
                    fail("metadata swa_window_offset != 1")

    # -- MoE routing detail -------------------------------------------------
    covered = set()
    for i in range(d["layers"]):
        if not d["moe"][i]:
            continue
        for pre, sfx in (("l", ""), ("cache_l", "_dec")):
            scores = fx[f"{pre}{i}_moe_scores{sfx}"]
            idx = fx[f"{pre}{i}_moe_topk_idx{sfx}"]
            w_pre = fx[f"{pre}{i}_moe_topk_w_pre{sfx}"]
            w_post = fx[f"{pre}{i}_moe_topk_w_post{sfx}"]
            covered.update(int(x) for x in idx.reshape(-1))
            row_ix = np.arange(scores.shape[0])[:, None]
            if not np.array_equal(scores[row_ix, idx], w_pre):
                fail(f"{pre}{i}_moe_topk_w_pre{sfx}: must be the RAW sigmoid scores gathered at topk_idx")
            norm = w_pre / (w_pre.sum(-1, keepdims=True) + 1e-20)
            if float(np.abs(norm - w_post).max()) > 1e-6:
                fail(f"{pre}{i}_moe_topk_w_post{sfx}: norm_topk_prob must be w / (sum + 1e-20)")
            if not sfx:
                bias = wt[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]
                gaps, rg = routing_metrics(scores, bias.astype(np.float32), d["topk"])
                if not np.array_equal(gaps, fx[f"l{i}_moe_rank_gaps"]):
                    fail(f"l{i}_moe_rank_gaps: must be the sorted selection-score gaps")
                if not np.array_equal(rg, fx[f"l{i}_moe_route_gap"]):
                    fail(f"l{i}_moe_route_gap: must be (srt[k-1]-srt[k]) / max(srt[0], 1e-9)")
    if covered and covered != set(range(d["experts"])):
        fail(f"MoE routing covers experts {sorted(covered)}, not all {d['experts']} — "
             "the fixture would not exercise expert-index mapping")

    # -- prefill-vs-cache self-consistency ----------------------------------
    margins = info.get("margins", {})
    lm = float(np.abs(fx["logits_full"].astype(np.float64) - fx["cache_logits"].astype(np.float64)).max())
    pairs = [(f"stream_{i}", f"cache_stream_{i}") for i in range(d["layers"] + 1)]
    pairs += [(f"l{i}_attn_out", f"cache_l{i}_attn_out") for i in range(d["layers"])]
    pairs += [(f"l{i}_mlp_out", f"cache_l{i}_mlp_out") for i in range(d["layers"])]
    pairs += [("final_norm", "cache_final_norm")]
    sm = max(float(np.abs(fx[x].astype(np.float64) - fx[y].astype(np.float64)).max()) for x, y in pairs)
    if abs(lm - float(margins.get("logits_max_abs", -1))) > 1e-12:
        fail(f"margins.logits_max_abs {margins.get('logits_max_abs')} != recomputed {lm}")
    if abs(sm - float(margins.get("streams_max_abs", -1))) > 1e-12:
        fail(f"margins.streams_max_abs {margins.get('streams_max_abs')} != recomputed {sm}")
    bound = float(margins.get("bound", 0))
    if lm > bound or sm > bound:
        fail(f"reference prefill-vs-cache margins (logits {lm:.2e}, streams {sm:.2e}) exceed bound {bound:.0e}")

    # -- tie margins + rope tables ------------------------------------------
    if not np.array_equal(logit_margin_np(fx["logits_full"]), fx["logit_margin"]):
        fail("logit_margin: must be the per-row top1-top2 gap")
    gaps_all = [fx[f"l{i}_moe_route_gap"] for i in range(d["layers"]) if d["moe"][i]]
    if not np.array_equal(np.minimum.reduce(gaps_all), fx["moe_route_gap"]):
        fail("moe_route_gap: must be the min over MoE layers")
    if abs(float(fx["logit_margin"].min()) - float(margins.get("logit_margin_min", "nan"))) > 1e-12:
        fail("margins.logit_margin_min != recomputed")
    if abs(float(fx["moe_route_gap"].min()) - float(margins.get("moe_route_gap_min", "nan"))) > 1e-12:
        fail("margins.moe_route_gap_min != recomputed")
    for name in ("rope_cos_ga", "rope_sin_ga", "rope_cos_swa", "rope_sin_swa"):
        a = fx[name]
        half = a.shape[1] // 2
        if not np.array_equal(a[:, :half], a[:, half:]):
            fail(f"{name}: emb = cat((freqs, freqs)) ⇒ both halves identical")
    for kind in ("cos", "sin"):
        if float(np.abs(fx[f"rope_{kind}_ga"] - fx[f"rope_{kind}_swa"]).max()) <= 1e-3:
            fail(f"rope_{kind}_ga vs rope_{kind}_swa: distinct rope thetas must give distinct tables")
    return problems


def run_dump(out_dir, seed, ref, cache_dir, offline):
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from safetensors.numpy import save_file
    except ImportError as e:
        _die_skip(f"python deps unavailable ({e}) — run under "
                  "`uv run --with torch --with transformers --with numpy --with safetensors`")
    try:
        import transformers  # noqa: F401  (the reference imports its APIs)
    except ImportError as e:
        _die_skip(f"transformers unavailable ({e})")

    resolve_reference(ref, cache_dir, offline)
    mm = __import__("mimo_v2_ref.modeling_mimo_v2", fromlist=["modeling_mimo_v2"])
    d = tiny_dims()
    T = d["t_total"]
    cfg, model = build_model(mm, seed)
    ids = np.random.default_rng(seed).integers(2, TINY["vocab_size"], size=T).astype(np.int32)

    sd = model.state_dict()
    w_schema = weight_schema(d)
    if set(sd) != set(w_schema):
        raise FixtureError(f"state-dict naming drift: only-in-model {sorted(set(sd) - set(w_schema))}, "
                           f"only-in-schema {sorted(set(w_schema) - set(sd))}")
    for k, v in sd.items():
        if tuple(v.shape) != tuple(w_schema[k]):
            raise FixtureError(f"{k}: model shape {tuple(v.shape)} != schema {w_schema[k]}")

    rec = run_passes(mm, model, ids)
    fx = assemble(model, rec, d)
    fx["input_ids"] = ids

    if set(fx) != set(fixture_schema(d)):
        raise FixtureError(f"fixture assembly drift: extra {sorted(set(fx) - set(fixture_schema(d)))}, "
                           f"missing {sorted(set(fixture_schema(d)) - set(fx))}")

    lm = float(np.abs(fx["logits_full"].astype(np.float64) - fx["cache_logits"].astype(np.float64)).max())
    pairs = [(f"stream_{i}", f"cache_stream_{i}") for i in range(d["layers"] + 1)]
    pairs += [(f"l{i}_attn_out", f"cache_l{i}_attn_out") for i in range(d["layers"])]
    pairs += [(f"l{i}_mlp_out", f"cache_l{i}_mlp_out") for i in range(d["layers"])]
    pairs += [("final_norm", "cache_final_norm")]
    sm = max(float(np.abs(fx[x].astype(np.float64) - fx[y].astype(np.float64)).max()) for x, y in pairs)
    if lm > BOUND or sm > BOUND:
        raise FixtureError(
            f"reference not self-consistent: prefill-vs-cache logits {lm:.2e} streams {sm:.2e} > {BOUND:.0e}")

    info = {
        "schema_version": 1,
        "seed": seed,
        "ref_dtype": "f32",
        "attention_projection_layout": TINY["attention_projection_layout"],
        "dims": d,
        "tiny": TINY,
        "swa_window_offset": 1,
        "margins": {
            "bound": BOUND,
            "logits_max_abs": lm,
            "streams_max_abs": sm,
            "logit_margin_min": float(fx["logit_margin"].min()),
            "moe_route_gap_min": float(fx["moe_route_gap"].min()),
        },
    }

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}
    save_file(weights, str(out_dir / "model.safetensors"))
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}}, indent=2))
    (out_dir / "config.json").write_text(json.dumps(
        dict(TINY, model_type="mimo_v2", architectures=["MiMoV2ForCausalLM"]), indent=2))
    save_file({k: np.ascontiguousarray(v) for k, v in fx.items()},
              str(out_dir / "fixture.safetensors"),
              metadata={"mimo_v2_fixture": json.dumps(info)})
    print(f"wrote {out_dir}: {len(fx)} fixture tensors, T={T}; "
          f"prefill-vs-cache max|d| logits {lm:.2e} streams {sm:.2e} (bound {BOUND:.0e})")
    return verify_out(out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="dump target (or the dump to check with --verify)")
    ap.add_argument("--seed", type=int, default=1234, help="weights + prompt ids seed")
    ap.add_argument("--ref", default=os.environ.get("MIMO_V2_REF"),
                    help="dir (or modeling file) holding the HF reference pair; "
                         "copied into the cache package (env MIMO_V2_REF)")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                    help="reference cache dir (default %(default)s)")
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network; require the cached reference or --ref")
    ap.add_argument("--verify", action="store_true",
                    help="only reload OUT_DIR and re-run the invariant battery")
    a = ap.parse_args()
    try:
        if a.verify:
            problems = verify_out(a.out_dir)
        else:
            problems = run_dump(a.out_dir, a.seed, a.ref, a.cache_dir, a.offline)
    except FixtureError as e:
        sys.stderr.write(f"FAIL: {e}\n")
        return 1
    for p in problems:
        sys.stderr.write(f"FAIL: {p}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())