#!/usr/bin/env python3
"""Engine oracle for glm5_next (GLM-5.3-Flash) on a TINY random model.

Builds mlx-vlm's own `glm5_next` language model (the reference this port follows)
with random weights, runs it in bf16 (the engine's numerics), and writes:
  <out>/pack/                 the tiny checkpoint in the PUBLISHED layout (TensorFold's
                              GLM-5.3-Flash-MLX packs: separate q/k/v + conv1d, f_a/b/g_a,
                              gate/up, q_a + kv_a_proj_with_mqa, embed_q/unembed_out),
                              bf16 except layer 0's q/k/v (5/8/5-bit, `mix_kda_quant`),
                              plus the MTP head (`language_model.mtp.0.*`)
  <out>/fixture.safetensors   input_ids, logits_full (one forward), logits_step
                              (prefill then one-token decode through the reference cache),
                              cap_ids + cap_l<i>: the mean of the four hyper-connection streams
                              after each tapped layer, what a DFlash2 drafter reads
  <out>/mtp_fixture.safetensors  the trunk's final-normed rows and the MTP drafter's logits for the same sequence

The pack round-trips through the reference's own `sanitize` (asserted), so the
layout our loader reads is the one mlx-vlm reads. Every expert runs (top-k = all)
so a near-tie in expert choice cannot fail a comparison; the indexer selects 2 of
up to 10 pools, so the sparse path is exercised, and `sel_gap` records how decided each
row's pool pick was.

  venv/bin/python tests/dump_glm5_next_fixtures.py --out ~/claude-tmp/glm-tiny
  GLM5_MODEL=~/claude-tmp/glm-tiny/pack GLM5_FIXTURE=~/claude-tmp/glm-tiny/fixture.safetensors \\
      zig build test -Dtest-filter="glm5_next fixture"
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next import language as glm_language
from mlx_vlm.models.glm5_next.language import LanguageModel

# Per query row, the smallest relative gap between the last pool the indexer keeps and
# the first it drops, over every DSA layer of the recorded forward (inf = nothing
# dropped). bf16 flips a pick where that gap is noise, and the flip reaches every later
# row through the KDA state, so the test scores only the rows before the first such tie.
_SEL_GAPS = []
# The same gap relative to the row's TOP score: a k-th pick of 0.0005 over exact zeros is a
# tie bf16 flips, which the k-th-relative gap reads as 1.0.
_SEL_GAPS_TOP = []
_exact_pool_select = glm_language._exact_pool_select


def _recording_pool_select(q, pool_keys, weights, pool_ends, pool_valid, query_positions,
                           select_k, scale, chunk_size=512):
    out = _exact_pool_select(q, pool_keys, weights, pool_ends, pool_valid, query_positions,
                             select_k, scale, chunk_size)
    if select_k < pool_keys.shape[1]:
        cand = pool_valid[:, None] & (pool_ends[:, None] <= query_positions[None, :, None])
        sc = np.array(glm_language._score_index_keys(q, pool_keys, weights, scale))[0]
        cand = np.array(cand)[0]
        gap = np.full(sc.shape[0], np.inf, dtype=np.float32)
        gap_top = np.full(sc.shape[0], np.inf, dtype=np.float32)
        for r in range(sc.shape[0]):
            c = np.sort(sc[r][cand[r]])[::-1]
            if len(c) > select_k:
                gap[r] = (c[select_k - 1] - c[select_k]) / max(abs(c[select_k - 1]), 1e-6)
                gap_top[r] = (c[select_k - 1] - c[select_k]) / max(abs(c[0]), 1e-6)
        _SEL_GAPS.append(gap)
        _SEL_GAPS_TOP.append(gap_top)
    return out


glm_language._exact_pool_select = _recording_pool_select

TEXT = dict(
    model_type="glm5_next_text", vocab_size=512, hidden_size=128, intermediate_size=256,
    moe_intermediate_size=64, num_hidden_layers=8, num_nextn_predict_layers=1,
    num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8,
    routed_scaling_factor=2.5, kv_lora_rank=64, q_lora_rank=64, qk_rope_head_dim=0,
    v_head_dim=32, qk_nope_head_dim=32, qk_head_dim=32, n_group=1, topk_group=1,
    num_experts_per_tok=8, norm_topk_prob=True, max_position_embeddings=4096,
    rms_norm_eps=1e-5, pad_token_id=0, eos_token_id=[1], tie_word_embeddings=False,
    mlp_layer_types=["dense"] + ["sparse"] * 7,
    layer_types=(["linear_attention"] * 3 + ["deepseek_sparse_attention"]) * 2,
    indexer_types=["full"] * 8, index_topk=8, index_head_dim=32, index_n_heads=4,
    swiglu_limit=10.0, hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20, index_kpool=4,
    index_kpool_always_select_tail=True, index_kpool_compress=True, mla_use_nope=True,
    first_k_dense_replace=1, scoring_func="sigmoid", topk_method="noaux_tc",
    linear_attn_config={"num_heads": 4, "head_dim": 32, "gate_lower_bound": -5.0,
                        "short_conv_kernel_size": 4},
)


# Layers whose stream mean the DFlash capture test reads: both KDA and DSA kinds, the last one before the head.
CAP_IDS = (1, 3, 5, 6)

SEL_TIE = 0.10  # the test's bar; keep the two in step
GAIN = 0.5


def init_params(lm, rng):
    out = {}
    for name, p in tree_flatten(lm.parameters()):
        shape = p.shape
        last = name.rsplit(".", 1)[-1]
        if name.endswith("A_log"):
            v = np.log(rng.uniform(1.0, 16.0, shape))
        elif name.endswith(("dt_bias", "e_score_correction_bias", "k_norm.bias")):
            v = 0.1 * rng.standard_normal(shape)
        elif name.endswith(("_hc.fn",)):
            v = 0.02 * rng.standard_normal(shape)
        elif name.endswith(("_hc.base",)):
            v = 0.5 * rng.standard_normal(shape)
        elif name.endswith(("_hc.scale",)):
            v = rng.uniform(0.5, 1.5, shape)
        elif "index_kpool_compress" in name:
            v = 0.3 * rng.standard_normal(shape)
        elif last == "weight" and len(shape) == 1:
            v = 1.0 + 0.1 * rng.standard_normal(shape)
        elif name.endswith("embed_tokens.weight"):
            v = 0.5 * rng.standard_normal(shape)
        elif len(shape) >= 2:
            # Gain < 1 keeps each layer's branch small beside the residual: a random
            # model at gain 1 amplifies bf16 rounding layer over layer.
            fan_in = shape[-1]
            v = GAIN * rng.standard_normal(shape) / np.sqrt(fan_in)
        else:
            v = 0.1 * rng.standard_normal(shape)
        # The pack stores bf16; the oracle runs f32 on those exact values.
        out[name] = mx.array(v.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
    return out


def published_layout(params, cfg):
    """Inverse of the reference sanitize: the names and splits the published pack stores."""
    out = {}
    hd, nh = cfg.linear_head_dim, cfg.linear_num_heads
    proj = hd * nh
    for name, v in params.items():
        key = "language_model." + name
        if ".self_attn.qkv_proj." in key:
            for part, piece in zip(("q_proj", "k_proj", "v_proj"), mx.split(v, 3, axis=0)):
                out[key.replace("qkv_proj", part)] = piece
        elif ".self_attn.qkv_conv.conv.weight" in key:
            # [C, K, 1] (mlx conv1d) -> per-part [C/3, 1, K] as published.
            for part, piece in zip(("q_conv1d", "k_conv1d", "v_conv1d"), mx.split(v, 3, axis=0)):
                out[key.replace("qkv_conv.conv", part)] = piece.moveaxis(1, 2)
        elif ".self_attn.fbg_a_proj." in key:
            f_a, b, g_a = mx.split(v, [hd, hd + nh], axis=0)
            out[key.replace("fbg_a_proj", "f_a_proj")] = f_a
            out[key.replace("fbg_a_proj", "b_proj")] = b
            out[key.replace("fbg_a_proj", "g_a_proj")] = g_a
        elif ".gate_up_proj." in key:
            g, u = mx.split(v, 2, axis=0)
            out[key.replace("gate_up_proj", "gate_proj")] = g
            out[key.replace("gate_up_proj", "up_proj")] = u
        elif ".self_attn.qkv_a_proj." in key:
            q, kv = mx.split(v, [cfg.q_lora_rank], axis=0)
            out[key.replace("qkv_a_proj", "q_a_proj")] = q
            out[key.replace("qkv_a_proj", "kv_a_proj_with_mqa")] = kv
        else:
            out[key] = v
    del proj
    return out


def mix_kda_quant(params, published):
    """Layer 0's q/k/v quantized at 5/8/5 bits, as TensorFold's pack ships layer 6: they
    cannot join into one quantized matrix. The reference runs the dequantized values."""
    pre = "language_model.model.layers.0.self_attn."
    deq = []
    for proj, bits in (("q_proj", 5), ("k_proj", 8), ("v_proj", 5)):
        w = published.pop(pre + proj + ".weight").astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
        published[pre + proj + ".weight"] = wq
        published[pre + proj + ".scales"] = sc
        published[pre + proj + ".biases"] = bi
        deq.append(mx.dequantize(wq, sc, bi, group_size=64, bits=bits).astype(mx.float32))
    params["model.layers.0.self_attn.qkv_proj.weight"] = mx.concatenate(deq, axis=0)


def layer_means(lm, ids, cap_ids):
    """The mean of the hyper-connection streams after each tapped layer, in one forward without a
    cache: SGLang's `hc_contract` of the target hiddens (the reference model itself exposes only the
    final normed hidden, so this walks its own layers the way `Glm5NextTextModel.__call__` does)."""
    m = lm.model
    h = mx.repeat(m.embed_tokens(ids)[:, :, None], m.config.hc_mult, axis=2)
    topk = None
    out = {}
    for i, layer in enumerate(m.layers):
        h, topk = layer(h, None, None, topk)
        if i in cap_ids:
            out[i] = h.mean(axis=2).astype(mx.float32)
    return out


def check_round_trip(lm, params, published):
    stripped = {k[len("language_model."):] if not k.startswith("language_model.model.") else k: v
                for k, v in published.items() if ".mtp." not in k}
    sanitized = lm.sanitize(dict(stripped))
    for name, v in params.items():
        key = "language_model." + name
        got = sanitized.get(key, sanitized.get(name))
        assert got is not None, f"sanitize lost {name}"
        assert got.shape == v.shape and mx.array_equal(got, v).item(), f"round trip differs: {name}"


def mtp_params(rng, cfg, lm_params):
    """The drafter's own weights, named as the published pack (`mtp.0.*`)."""
    from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
    from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel

    draft = Glm5NextMTPDraftModel(Glm5NextMTPConfig(text_config=cfg))
    p = init_params(draft, rng)
    draft.update(tree_unflatten(list(p.items())))
    pub = {}
    for name, v in p.items():
        key = name.replace("mtp_block.", "block.").replace("shared_head_norm", "norm")
        if ".gate_up_proj." in key:
            g, u = mx.split(v, 2, axis=0)
            pub["language_model.mtp.0." + key.replace("gate_up_proj", "gate_proj")] = g
            pub["language_model.mtp.0." + key.replace("gate_up_proj", "up_proj")] = u
        elif ".qkv_a_proj." in key:
            q, kv = mx.split(v, [cfg.q_lora_rank], axis=0)
            pub["language_model.mtp.0." + key.replace("qkv_a_proj", "q_a_proj")] = q
            pub["language_model.mtp.0." + key.replace("qkv_a_proj", "kv_a_proj_with_mqa")] = kv
        else:
            pub["language_model.mtp.0." + key] = v
    return draft, pub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--len", type=int, default=40)
    ap.add_argument("--prefill", type=int, default=30)
    ap.add_argument("--seed", type=lambda v: int(v, 0), default=1)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    (out / "pack").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cfg = TextConfig.from_dict(TEXT)
    lm = LanguageModel(cfg)
    params = init_params(lm, rng)
    lm.update(tree_unflatten(list(params.items())))
    published = published_layout(params, cfg)
    check_round_trip(lm, params, published)
    mix_kda_quant(params, published)
    lm.update(tree_unflatten(list(params.items())))
    # Serve-time numerics: bf16 weights and activations, as the engine runs. An f32
    # oracle moves the indexer's pool scores enough to flip selections our bf16 makes.
    lm.update(tree_unflatten([(k, v.astype(mx.bfloat16)) for k, v in params.items()]))

    ids = mx.array(rng.integers(2, cfg.vocab_size, size=(1, args.len)).astype(np.int32))
    _SEL_GAPS.clear()
    logits_full = lm(ids).logits.astype(mx.float32)
    sel_gap = np.min(np.stack(_SEL_GAPS), axis=0) if _SEL_GAPS else np.full(args.len, np.inf)
    _SEL_GAPS.clear()
    caps = layer_means(lm, ids, CAP_IDS)
    _SEL_GAPS.clear()

    cache = lm.make_cache()
    steps = [lm(ids[:, : args.prefill], cache=cache).logits]
    for t in range(args.prefill, args.len):
        steps.append(lm(ids[:, t : t + 1], cache=cache).logits)
    logits_step = mx.concatenate(steps, axis=1).astype(mx.float32)

    # MTP: head row p drafts x_{p+2} from (x_{p+1}, final-normed hidden h_p).
    draft, mtp_pub = mtp_params(rng, cfg, params)
    hidden = lm.model(ids)
    emb = lm.model.embed_tokens(ids[:, 1:])
    h = draft.eh_proj(mx.concatenate([draft.enorm(emb), draft.hnorm(hidden[:, :-1])], axis=-1))
    _SEL_GAPS_TOP.clear()
    h = draft.mtp_block(h, None)
    # One DSA layer and no recurrence: a near-tie pick perturbs its own row only, so the
    # test skips rows by gap instead of cutting at the first tie.
    mtp_sel_gap = np.min(np.stack(_SEL_GAPS_TOP), axis=0) if _SEL_GAPS_TOP else np.full(args.len - 1, np.inf)
    _SEL_GAPS_TOP.clear()
    mtp_logits = lm.lm_head(draft.shared_head_norm(h)).astype(mx.float32)

    published.update(mtp_pub)
    mx.save_safetensors(str(out / "pack" / "model.safetensors"),
                        {k: v if v.dtype == mx.uint32 else v.astype(mx.bfloat16)
                         for k, v in published.items()})
    config = {
        "architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next",
        "tie_word_embeddings": False, "text_config": {**TEXT, "dtype": "bfloat16"},
        "eos_token_id": [1], "pad_token_id": 0,
    }
    (out / "pack" / "config.json").write_text(json.dumps(config, indent=2))
    mx.save_safetensors(str(out / "fixture.safetensors"), {
        "input_ids": ids, "logits_full": logits_full, "logits_step": logits_step,
        "prefill": mx.array([args.prefill], dtype=mx.int32),
        "sel_gap": mx.array(sel_gap.astype(np.float32)),
        "cap_ids": mx.array(list(CAP_IDS), dtype=mx.int32),
        **{f"cap_l{i}": v for i, v in caps.items()},
    })
    mx.save_safetensors(str(out / "mtp_fixture.safetensors"),
                        {"input_ids": ids, "hidden": hidden, "mtp_logits": mtp_logits,
                         "sel_gap": mx.array(mtp_sel_gap.astype(np.float32))})
    d = mx.abs(logits_full - logits_step).max().item()
    first_tie = int(np.argmax(sel_gap < SEL_TIE)) if (sel_gap < SEL_TIE).any() else args.len
    print(f"wrote {out}: T={args.len}, full vs cached-step max |diff| {d:.2e}, "
          f"first pool pick within {SEL_TIE:.0%} at row {first_tie}")


if __name__ == "__main__":
    main()
