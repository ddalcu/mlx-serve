#!/usr/bin/env python3
"""Oracle for the qwen4_exp (Qwen3.8-Flash-Next) port, on a TINY random model.

Two phases, run from the torch venv (transformers main carries the arch):

  1. `build`  — a random Qwen4ExpForCausalLM at toy geometry, saved in the
     real checkpoint's naming (model.language_model.*, sharded n-gram table),
     so `tests/convert_qwen38_flash_next.py --src` converts it verbatim.
  2. `dump`   — the reference forward on OUR dequantized pack (mx.dequantize
     of every quantized tensor written back into the torch model), so the
     fixture measures the ENGINE, not the quantizer. Writes input_ids, the
     full-prefill logits and a stepwise-decode logits block past the QSA
     budget, plus the per-layer residual stream for bisecting.

  venv/bin/python tests/dump_qwen4_exp_fixtures.py build --out ~/claude-tmp/qwen4-tiny/hf
  python3 tests/convert_qwen38_flash_next.py --src ~/claude-tmp/qwen4-tiny/hf --dst ~/claude-tmp/qwen4-tiny/pack
  venv/bin/python tests/dump_qwen4_exp_fixtures.py dump --hf ~/claude-tmp/qwen4-tiny/hf \
      --pack ~/claude-tmp/qwen4-tiny/pack --out ~/claude-tmp/qwen4-tiny/fixture.safetensors
  QWEN4_TEST_MODEL=~/claude-tmp/qwen4-tiny/pack QWEN4_FIXTURE=~/claude-tmp/qwen4-tiny/fixture.safetensors \
      zig build test -Dtest-filter="qwen4 fixture"

`--vision` on both phases builds Qwen4ExpForConditionalGeneration with a tiny
tower (the converter packs it via `--add-vision --src`) and dumps ONE image
prompt: pixel_values + grid, the tower's pooler_output, the 3-D position ids
+ rope delta, full-prefill logits, prefill + stepwise decode, per-layer
streams and the layer-3 QSA mask (the prompt crosses the tiny budget so the
mask depends on the M-RoPE angles of the pooled block keys).
"""

import argparse
import os
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# QWEN4_REF_DTYPE=bf16 renders the reference in the shipped dtype (bf16 residual
# streams); f32 is the tighter oracle for the math but not what the checkpoint runs.
REF_DTYPE = torch.bfloat16 if os.environ.get("QWEN4_REF_DTYPE", "f32") == "bf16" else torch.float32
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_qwen38_flash_next import read_header, read_raw  # noqa: E402


def load_raw(path):
    """safetensors -> raw numpy (bf16 stays uint16; safetensors.numpy can't)."""
    hdr, off = read_header(path)
    return {k: read_raw(path, off, m) for k, m in hdr.items()}

TINY = dict(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=8,
    full_attention_interval=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    hc_count=4,
    hc_lowrank=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_conv_kernel_dim=4,
    num_experts=8,
    # Every expert active: a random tiny MoE routes near ties everywhere
    # (8 experts x 8 layers x T rows), and our kernels' bf16-class noise flips
    # one pick per prompt whatever the router scale or seed (the tie RATE is
    # scale-invariant). With k = E the weighting + gather + expert mapping
    # are still measured; top-k SELECTION is covered by the live pack.
    num_experts_per_tok=8,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    ple_layer_ids=[2],
    ple_embed_dim=128,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=100,
    make_ngram_vocab_size_divisible_by=8,
    seed=1234,
    split_ngram_parts=2,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=4,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    output_gate_type="sigmoid",
    partial_rotary_factor=0.25,
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 0.25,
                     "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    max_position_embeddings=4096,
    bos_token_id=1,
    eos_token_id=1,
    tie_word_embeddings=False,
    attention_bias=False,
    norm_topk_prob=True,
)

T_PREFILL = 14
T_DECODE = 6

TINY_VISION = dict(
    depth=2, hidden_size=64, num_heads=4, intermediate_size=128, in_channels=3,
    patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
    num_position_embeddings=16, out_hidden_size=64, hidden_act="gelu_pytorch_tanh",
)
VISION_IDS = dict(image_token_id=122, video_token_id=123, vision_start_token_id=120, vision_end_token_id=121)
IMAGE_GRID = (1, 6, 8)  # patches (t, h, w) -> 3x4 = 12 merged tokens
V_PREFILL = 22  # 3 text + start + 12 image + end + 5 text, then T_DECODE steps


def lm(m):
    return m.model.language_model if hasattr(m.model, "language_model") else m.model


def make_model(vision=False):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    cfg = Qwen4ExpTextConfig(**TINY)
    cfg._attn_implementation = "eager"
    torch.manual_seed(7)
    if vision:
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig, Qwen4ExpVisionConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
        full = Qwen4ExpConfig(text_config=cfg, vision_config=Qwen4ExpVisionConfig(**TINY_VISION), **VISION_IDS)
        full._attn_implementation = "eager"
        full.vision_config._attn_implementation = "eager"
        m = Qwen4ExpForConditionalGeneration(full).float().eval()
    else:
        m = Qwen4ExpForCausalLM(cfg).float().eval()  # built f32 for the random init; cast after loading
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("mlp.gate.weight"):
                p.normal_(0, 0.5)  # spread the softmax weights (k = E, see TINY)
            elif p.dim() >= 2:
                p.normal_(0, 0.05 if "experts" not in n else 0.08)
            elif n.endswith("A_log"):
                p.uniform_(-2.0, 0.5)
            elif n.endswith("dt_bias"):
                p.normal_(0, 0.5)
            elif "norm" in n:
                p.normal_(0, 0.2)
            else:
                p.normal_(0, 0.2)
    return cfg, m


def hf_name(k):
    if k == "lm_head.weight" or k.startswith("model.language_model.") or k.startswith("model.visual."):
        return k
    assert k.startswith("model."), k
    return "model.language_model." + k[len("model."):]


def build(out, vision=False):
    cfg, m = make_model(vision)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd = {}
    for k, v in m.state_dict().items():
        nk = hf_name(k)
        if nk.endswith(".ple.ple_embedding.ngram_embedding.weight"):
            parts = TINY["split_ngram_parts"]
            assert v.shape[0] % parts == 0
            rows = v.shape[0] // parts
            base = nk[:-len(".weight")]
            for i in range(parts):
                sd[f"{base}.shard_{i}.weight"] = v[i * rows:(i + 1) * rows].contiguous().to(torch.bfloat16)
            continue
        if v.dtype in (torch.float32, torch.bfloat16, torch.float16):
            v = v.to(torch.bfloat16)
        sd[nk] = v.contiguous()
    # Synthetic MTP head in the checkpoint's naming: layer 3's tensors as
    # mtp.layers.0.*, plus random fc/pre-norm and the trunk mixer's tensors.
    torch.manual_seed(11)
    for k, v in list(sd.items()):
        if k.startswith("model.language_model.layers.3."):
            sd["mtp.layers.0." + k[len("model.language_model.layers.3."):]] = v.clone()
        elif k.startswith("model.language_model.hyper_connection_mixer."):
            sd["mtp.hyper_connection_mixer." + k[len("model.language_model.hyper_connection_mixer."):]] = v.clone()
    H = TINY["hidden_size"]
    sd["mtp.fc_embedding.weight"] = (torch.randn(H, H) * 0.05).to(torch.bfloat16)
    sd["mtp.fc_hidden.weight"] = (torch.randn(H, H) * 0.05).to(torch.bfloat16)
    sd["mtp.pre_fc_norm_embedding.weight"] = (torch.randn(H) * 0.2).to(torch.bfloat16)
    sd["mtp.pre_fc_norm_hidden.weight"] = (torch.randn(H * TINY["hc_count"]) * 0.2).to(torch.bfloat16)
    save_torch(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    text_cfg = dict(TINY)
    text_cfg["model_type"] = "qwen4_exp_text"
    text_cfg["layer_types"] = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                               for i in range(TINY["num_hidden_layers"])]
    full = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
            "text_config": text_cfg, "tie_word_embeddings": False}
    if vision:
        full["vision_config"] = dict(TINY_VISION, model_type="qwen4_exp_vision")
        full.update(VISION_IDS)
    (out / "config.json").write_text(json.dumps(full, indent=2))
    print(f"wrote {len(sd)} tensors to {out}")


def dequant_pack(pack):
    """Every tensor of the pack in HF naming, quantized ones dequantized."""
    import mlx.core as mx
    pack = Path(pack)
    idx = json.loads((pack / "model.safetensors.index.json").read_text())
    raw = {}
    for f in sorted(set(idx["weight_map"].values())):
        raw.update(load_raw(str(pack / f)))
    out = {}
    for k, v in raw.items():
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        base = k[:-len(".weight")] if k.endswith(".weight") else k
        if v.dtype == np.uint32 and base + ".scales" in raw:
            sc = raw[base + ".scales"]
            bi = raw[base + ".biases"]
            # widths the converter uses, by name (see its docstring)
            bits = 4 if (".switch_mlp." in base or base.endswith("embed_tokens")) else 8
            gs = 64
            wq = mx.array(v)
            s_ = mx.array(sc).view(mx.bfloat16)
            b_ = mx.array(bi).view(mx.bfloat16)
            deq = mx.dequantize(wq, s_, b_, group_size=gs, bits=bits)
            out[base + ".weight"] = np.array(deq.astype(mx.float32))
        else:
            arr = v
            if arr.dtype == np.uint16:  # bf16 raw
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            out[k] = arr
    # merged n-gram table
    from struct import unpack
    with open(pack / "ngram_table.bin", "rb") as f:
        hlen = unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        data_off = 8 + hlen
        bits = int(hdr["__metadata__"]["bits"])
        gs = int(hdr["__metadata__"]["group_size"])
        def rd(name, dt):
            b, e = hdr[name]["data_offsets"]
            f.seek(data_off + b)
            return np.frombuffer(f.read(e - b), dtype=dt).reshape(hdr[name]["shape"])
        wq = rd("weight", np.uint32)
        sc = rd("scales", np.uint16)
        bi = rd("biases", np.uint16)
    deq = mx.dequantize(mx.array(wq), mx.array(sc).view(mx.bfloat16), mx.array(bi).view(mx.bfloat16), group_size=gs, bits=bits)
    out["__ngram__"] = np.array(deq.astype(mx.float32))
    return out


def pack_to_torch_key(k, vision=False):
    # language_model.model.X -> model.X (model.language_model.X under the
    # conditional-generation wrapper); language_model.lm_head -> lm_head
    if k.startswith("language_model.model."):
        return ("model.language_model." if vision else "model.") + k[len("language_model.model."):]
    if k == "language_model.lm_head.weight":
        return "lm_head.weight"
    return k


def dump(hf, pack, out, vision=False):
    cfg, m = make_model(vision)
    ten = dequant_pack(pack)
    sd = m.state_dict()
    with torch.no_grad():
        for k, v in ten.items():
            if k == "__ngram__" or k.startswith("language_model.mtp."):
                continue
            tk = pack_to_torch_key(k, vision)
            if tk.endswith(".mlp.switch_mlp.gate_proj.weight"):
                base = tk[:-len("switch_mlp.gate_proj.weight")]
                gate = torch.from_numpy(v)
                up = torch.from_numpy(ten[k.replace("gate_proj", "up_proj")])
                sd[base + "experts.gate_up_proj"].copy_(torch.cat([gate, up], dim=1))
                continue
            if tk.endswith(".mlp.switch_mlp.up_proj.weight"):
                continue
            if tk.endswith(".mlp.switch_mlp.down_proj.weight"):
                sd[tk[:-len("switch_mlp.down_proj.weight")] + "experts.down_proj"].copy_(torch.from_numpy(v))
                continue
            t = torch.from_numpy(np.ascontiguousarray(v))
            if tk.endswith("conv1d.weight") and t.dim() == 3:
                t = t.transpose(1, 2).contiguous()  # back to HF [C,1,K]
            if tk not in sd:
                if not tk.startswith("model.visual."):
                    print("skip", tk)
                continue
            if t.shape != sd[tk].shape:
                raise SystemExit(f"{tk}: {tuple(t.shape)} vs {tuple(sd[tk].shape)}")
            # the converter folded +1 into every (1+w) norm: undo for torch
            if t.dim() == 1 and any(tk.endswith(sfx) for sfx in (
                    "hc_norm.weight", "q_norm.weight", "k_norm.weight", "q_layernorm.weight",
                    "k_layernorm.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight")):
                t = t - 1.0
            sd[tk].copy_(t.to(sd[tk].dtype))
        ng = torch.from_numpy(ten["__ngram__"])
        for k in sd:
            if k.endswith("ngram_embedding.weight"):
                assert sd[k].shape == ng.shape, (sd[k].shape, ng.shape)
                sd[k].copy_(ng)
    m.load_state_dict(sd)
    m.float().eval()

    if vision:
        return dump_vision(m, ids_out=out)
    T = T_PREFILL + T_DECODE

    torch.manual_seed(3)
    ids = torch.randint(2, TINY["vocab_size"], (1, T))
    ids[0, 5] = TINY["eos_token_id"]  # an eos inside the prompt: PLE shift reset
    m.to(REF_DTYPE)
    l0 = m.model.layers[0]
    cap = {}
    def hook(name):
        def f(mod, inp, out):
            cap[name] = out
        return f
    hooks = [l0.attn_hyper_connection.register_forward_hook(hook("hc_attn")),
             (l0.linear_attn if hasattr(l0, "linear_attn") else l0.self_attn).register_forward_hook(hook("attn")),
             l0.mlp_hyper_connection.register_forward_hook(hook("hc_mlp")),
             l0.mlp.register_forward_hook(hook("mlp")),
             m.model.layers[1].ple.ple_embedding.register_forward_hook(hook("ple_emb")),
             m.model.layers[1].ple.register_forward_hook(hook("ple_out")),
             m.model.layers[1].ple.ple_embedding.ngram_embedding.register_forward_hook(
                 lambda mod, inp, out: cap.__setitem__("ngram_ids", inp[0])),
             m.model.layers[3].self_attn.indexer.register_forward_hook(hook("qsa_mask")),
             m.model.layers[3].self_attn.register_forward_hook(hook("attn3")),
             m.model.hyper_connection_mixer.register_forward_hook(lambda mod, inp, out: cap.__setitem__("pre_mixer", inp[0]))]
    hooks += hook_indexers(lm(m), cap)
    with torch.no_grad():
        full = m(input_ids=ids, use_cache=False, output_hidden_states=True)
        for h in hooks:
            h.remove()
        logits_full = full.logits[0].float().numpy()
        hs = [h[0].float().numpy() for h in full.hidden_states]  # residual streams per layer
        # stepwise past the QSA budget
        pre = m(input_ids=ids[:, :T_PREFILL], use_cache=True)
        pkv = pre.past_key_values
        dec = []
        for t in range(T_PREFILL, T):
            step = m(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = step.past_key_values
            dec.append(step.logits[0, -1].float().numpy())
    fixture = {
        "mtp_logits": mtp_reference(m, ten, ids, T),
        "input_ids": ids[0].numpy().astype(np.int32),
        "logit_margin": logit_margin(logits_full),
        "qsa_gap": qsa_rel_gaps(lm(m), cap, T),
        "logits_full": logits_full.astype(np.float32),
        "logits_prefill_last": pre.logits[0, -1].float().numpy().astype(np.float32),
        "logits_decode": np.stack(dec).astype(np.float32),
    }
    for i, h in enumerate(hs):
        fixture[f"stream_{i}"] = h.astype(np.float32)
    def flat(x):
        return x.reshape(T, -1).float().numpy()
    fixture["l0_mixed_attn"] = flat(cap["hc_attn"][0])
    fixture["l0_inj_attn"] = flat(cap["hc_attn"][2])
    attn_out = cap["attn"][0] if isinstance(cap["attn"], tuple) else cap["attn"]
    fixture["l0_attn_out"] = flat(attn_out)
    fixture["l0_mixed_mlp"] = flat(cap["hc_mlp"][0])
    fixture["l0_inj_mlp"] = flat(cap["hc_mlp"][2])
    fixture["l0_mlp_out"] = flat(cap["mlp"])
    fixture["l1_ple_emb"] = flat(cap["ple_emb"])
    fixture["l1_ple_out"] = flat(cap["ple_out"])
    fixture["l1_ngram_ids"] = cap["ngram_ids"].reshape(T, -1).numpy().astype(np.int32)
    qm = cap["qsa_mask"]
    qm = (qm == 0) if qm.is_floating_point() else qm  # eager float mask: 0 = visible
    fixture["l3_qsa_mask"] = qm.reshape(T, -1).float().numpy()
    fixture["l3_attn_out"] = flat(cap["attn3"][0])
    fixture["stream_pre_mixer"] = flat(cap["pre_mixer"])
    save_file(fixture, os.path.expanduser(out))
    # sanity: stepwise == full at the decode positions (the reference's own consistency)
    ref_gap = np.abs(np.stack(dec) - logits_full[T_PREFILL:]).max()
    print(f"wrote {out}; reference stepwise-vs-full max|Δ| = {ref_gap:.2e}")


def logit_margin(logits):
    top2 = torch.from_numpy(logits).float().topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).numpy().astype(np.float32)


def hook_indexers(text, cap):
    """Pre-hooks capturing every QSA indexer's (hidden, position_embeddings)."""
    def pre(l):
        def f(mod, args):
            cap[("qsa", l)] = (args[0], args[1])
        return f
    return [layer.self_attn.indexer.register_forward_pre_hook(pre(i))
            for i, layer in enumerate(text.layers) if hasattr(getattr(layer, "self_attn", None), "indexer")]


def qsa_rel_gaps(text, cap, T):
    """Per query row, the smallest (over QSA layers) RELATIVE margin between
    the last selected block score and the first rejected one — the reference's
    own tie measure. relu leaves exact-zero scores, so a near-zero q·k that
    flips sign under 1% kernel noise selects a different block; the Zig test
    acquits such rows by this number. 1.0 where no selection happens."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb
    out = np.ones(T, dtype=np.float32)
    for (tag, l), (h, (cos, sin)) in [(k, v) for k, v in cap.items() if isinstance(k, tuple) and k[0] == "qsa"]:
        idx = text.layers[l].self_attn.indexer
        with torch.no_grad():
            qk = idx.index_qk_proj(h)
            hd = idx.index_head_dim
            q, k = torch.split(qk, [idx.index_n_heads * hd, idx.index_kv_heads * hd], -1)
            q = apply_rotary_pos_emb(idx.q_layernorm(q.reshape(1, T, -1, hd)), cos=cos, sin=sin, unsqueeze_dim=2)[0]
            raw = k.reshape(T, hd)
            r = idx.compress_ratio
            nb = T // r
            pooled = idx.k_layernorm(raw[:nb * r].view(nb, r, -1).float().mean(1).to(raw.dtype))
            starts = torch.arange(nb) * r
            kb = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[0].index_select(0, starts), sin=sin[0].index_select(0, starts)).squeeze(1)
            for row in range(T):
                ncb = (row + 1) // r
                if ncb <= idx.block_topk:
                    continue
                sc = torch.relu(q[row].float() @ kb[:ncb].float().T).sum(0)
                srt = sc.sort(descending=True).values
                gap = float(srt[idx.block_topk - 1] - srt[idx.block_topk]) / max(float(srt[0]), 1e-9)
                out[row] = min(out[row], gap)
    return out


def dump_vision(m, ids_out):
    """One image prompt through Qwen4ExpForConditionalGeneration."""
    T = V_PREFILL + T_DECODE
    gt, gh, gw = IMAGE_GRID
    n_img = gt * gh * gw // (TINY_VISION["spatial_merge_size"] ** 2)
    feat = TINY_VISION["in_channels"] * TINY_VISION["temporal_patch_size"] * TINY_VISION["patch_size"] ** 2
    thw = torch.tensor([[gt, gh, gw]])
    text = lm(m)
    torch.manual_seed(5)
    ids = torch.randint(2, VISION_IDS["vision_start_token_id"], (1, T))
    ids[0, 2] = TINY["eos_token_id"]
    ids[0, 3] = VISION_IDS["vision_start_token_id"]
    ids[0, 4:4 + n_img] = VISION_IDS["image_token_id"]
    ids[0, 4 + n_img] = VISION_IDS["vision_end_token_id"]
    mm = (ids == VISION_IDS["image_token_id"]).int()
    pv = torch.rand(gt * gh * gw, feat) * 2 - 1
    cap = {}
    def hook(name):
        def f(mod, inp, out):
            cap[name] = out
        return f
    m.to(REF_DTYPE)
    hooks = [text.layers[3].self_attn.indexer.register_forward_hook(hook("qsa_mask")),
             text.layers[3].self_attn.register_forward_hook(hook("attn3")),
             text.hyper_connection_mixer.register_forward_hook(lambda mod, inp, out: cap.__setitem__("pre_mixer", inp[0]))]
    hooks += hook_indexers(text, cap)
    with torch.no_grad():
        pos_ids, deltas = m.model.get_rope_index(ids, mm, image_grid_thw=thw)
        cos, sin = text.rotary_emb(torch.zeros(1, T, 1, dtype=REF_DTYPE), pos_ids)
        embeds = torch.cat(m.model.get_image_features(pv, thw, return_dict=True).pooler_output, dim=0)
        full = m(input_ids=ids, pixel_values=pv, image_grid_thw=thw, mm_token_type_ids=mm,
                 use_cache=False, output_hidden_states=True)
        for h in hooks:
            h.remove()
        hs = [h[0].float().numpy() for h in full.hidden_states]
        pre = m(input_ids=ids[:, :V_PREFILL], pixel_values=pv, image_grid_thw=thw,
                mm_token_type_ids=mm[:, :V_PREFILL], use_cache=True)
        pkv = pre.past_key_values
        dec = []
        for t in range(V_PREFILL, T):
            step = m(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = step.past_key_values
            dec.append(step.logits[0, -1].float().numpy())
    qm = cap["qsa_mask"]
    qm = (qm == 0) if qm.is_floating_point() else qm
    fixture = {
        "input_ids": ids[0].numpy().astype(np.int32),
        "logit_margin": logit_margin(full.logits[0].float().numpy()),
        "qsa_gap": qsa_rel_gaps(text, cap, T),
        "pixel_values": pv.numpy().astype(np.float32),
        "image_grid_thw": thw[0].numpy().astype(np.int32),
        "image_embeds": embeds.float().numpy().astype(np.float32),
        "position_ids": pos_ids[:, 0].numpy().astype(np.int32),
        "rope_delta": deltas.reshape(-1).numpy().astype(np.int32),
        "logits_full": full.logits[0].float().numpy().astype(np.float32),
        "logits_prefill_last": pre.logits[0, -1].float().numpy().astype(np.float32),
        "logits_decode": np.stack(dec).astype(np.float32),
        "l3_qsa_mask": qm.reshape(T, -1).float().numpy(),
        "rope_cos": cos[0].float().numpy().astype(np.float32),
        "rope_sin": sin[0].float().numpy().astype(np.float32),
        "l3_attn_out": cap["attn3"][0].reshape(T, -1).float().numpy().astype(np.float32),
        "stream_pre_mixer": cap["pre_mixer"].reshape(T, -1).float().numpy().astype(np.float32),
    }
    for i, h in enumerate(hs):
        fixture[f"stream_{i}"] = h.astype(np.float32)
    save_file(fixture, os.path.expanduser(ids_out))
    ref_gap = np.abs(np.stack(dec) - fixture["logits_full"][V_PREFILL:]).max()
    print(f"wrote {ids_out}; {n_img} image tokens, delta {int(deltas[0, 0])}, "
          f"reference stepwise-vs-full max|Δ| = {ref_gap:.2e}")


def mtp_reference(m, ten, ids, T):
    """vLLM/SGLang `residual_linear_shared` MTP on the trunk's pre-mixer stream:
    row t (stream at t, token t+1, position t+1) predicts token t+2."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, Qwen4ExpTextRMSNorm)
    cfg = m.config
    import copy
    c9 = copy.deepcopy(cfg)
    c9.num_hidden_layers = 9
    c9.layer_types = list(cfg.layer_types) + ["qwen_sparse_attention"]
    layer = Qwen4ExpTextDecoderLayer(c9, 8).float().eval()
    mixer = Qwen4ExpTextGatedResidual(c9, use_combine=False).float().eval()  # cast to REF_DTYPE after loading
    H, hc = cfg.hidden_size, cfg.hc_count
    norm_e = Qwen4ExpTextRMSNorm(H, eps=cfg.rms_norm_eps)
    norm_h = Qwen4ExpTextRMSNorm(H * hc, eps=cfg.rms_norm_eps)
    fc_e = torch.nn.Linear(H, H, bias=False)
    fc_h = torch.nn.Linear(H, H, bias=False)
    folded = ("hc_norm.weight", "q_norm.weight", "k_norm.weight", "q_layernorm.weight", "k_layernorm.weight",
              "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")
    def get(name):
        t = torch.from_numpy(np.ascontiguousarray(ten["language_model.mtp." + name]))
        if t.dim() == 1 and name.endswith(folded):
            t = t - 1.0
        return t
    lsd = layer.state_dict()
    with torch.no_grad():
        for k in lsd:
            if k == "mlp.experts.gate_up_proj":
                lsd[k].copy_(torch.cat([get("layers.0.mlp.switch_mlp.gate_proj.weight"), get("layers.0.mlp.switch_mlp.up_proj.weight")], dim=1))
            elif k == "mlp.experts.down_proj":
                lsd[k].copy_(get("layers.0.mlp.switch_mlp.down_proj.weight"))
            else:
                lsd[k].copy_(get("layers.0." + k))
        layer.load_state_dict(lsd)
        msd = mixer.state_dict()
        for k in msd:
            msd[k].copy_(get("hyper_connection_mixer." + k))
        mixer.load_state_dict(msd)
        norm_e.weight.copy_(get("pre_fc_norm_embedding.weight"))
        norm_h.weight.copy_(get("pre_fc_norm_hidden.weight"))
        fc_e.weight.copy_(get("fc_embedding.weight"))
        fc_h.weight.copy_(get("fc_hidden.weight"))
        for mod in (layer, mixer, norm_e, norm_h, fc_e, fc_h):
            mod.to(REF_DTYPE)
        cap = {}
        hk = m.model.hyper_connection_mixer.register_forward_hook(lambda mod, inp, out: cap.__setitem__("pre", inp[0]))
        m(input_ids=ids, use_cache=False)
        hk.remove()
        stream = cap["pre"][:, :T - 1]  # [1, T-1, hc*H]
        tok = ids[:, 1:T]
        e = fc_e(norm_e(m.model.embed_tokens(tok)))
        hh = fc_h(norm_h(stream).view(1, T - 1, hc, H))
        x = (hh + e.unsqueeze(-2)).flatten(-2)
        pos = torch.arange(1, T, dtype=torch.long)[None]
        pe = m.model.rotary_emb(x, pos[None].expand(3, 1, -1))
        mask = torch.full((T - 1, T - 1), torch.finfo(REF_DTYPE).min, dtype=REF_DTYPE).triu(1)[None, None]
        out = layer(x, position_embeddings=pe, attention_mask=mask, conv_mask=None, past_key_values=None)
        logits = m.lm_head(mixer(out))
    return logits[0].float().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--vision", action="store_true")
    d = sub.add_parser("dump")
    d.add_argument("--hf", required=True)
    d.add_argument("--pack", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--vision", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(os.path.expanduser(a.out), a.vision)
    else:
        dump(os.path.expanduser(a.hf), os.path.expanduser(a.pack), a.out, a.vision)


if __name__ == "__main__":
    sys.exit(main())
