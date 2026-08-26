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
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
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
    num_experts_per_tok=2,
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


def make_model():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    cfg = Qwen4ExpTextConfig(**TINY)
    cfg._attn_implementation = "eager"
    torch.manual_seed(7)
    m = Qwen4ExpForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() >= 2:
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
    if k == "lm_head.weight":
        return k
    assert k.startswith("model."), k
    return "model.language_model." + k[len("model."):]


def build(out):
    cfg, m = make_model()
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


def pack_to_torch_key(k):
    # language_model.model.X -> model.X ; language_model.lm_head -> lm_head
    if k.startswith("language_model.model."):
        return "model." + k[len("language_model.model."):]
    if k == "language_model.lm_head.weight":
        return "lm_head.weight"
    return k


def dump(hf, pack, out):
    cfg, m = make_model()
    ten = dequant_pack(pack)
    sd = m.state_dict()
    with torch.no_grad():
        for k, v in ten.items():
            if k == "__ngram__" or k.startswith("language_model.mtp."):
                continue
            tk = pack_to_torch_key(k)
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
    m.eval()

    torch.manual_seed(3)
    T = T_PREFILL + T_DECODE
    ids = torch.randint(2, TINY["vocab_size"], (1, T))
    ids[0, 5] = TINY["eos_token_id"]  # an eos inside the prompt: PLE shift reset
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
             m.model.layers[3].self_attn.register_forward_hook(hook("attn3"))]
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
    save_file(fixture, os.path.expanduser(out))
    # sanity: stepwise == full at the decode positions (the reference's own consistency)
    ref_gap = np.abs(np.stack(dec) - logits_full[T_PREFILL:]).max()
    print(f"wrote {out}; reference stepwise-vs-full max|Δ| = {ref_gap:.2e}")


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
    mixer = Qwen4ExpTextGatedResidual(c9, use_combine=False).float().eval()
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
        mask = torch.full((T - 1, T - 1), torch.finfo(torch.float32).min).triu(1)[None, None]
        out = layer(x, position_embeddings=pe, attention_mask=mask, conv_mask=None, past_key_values=None)
        logits = m.lm_head(mixer(out))
    return logits[0].float().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--hf", required=True)
    d.add_argument("--pack", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(os.path.expanduser(a.out))
    else:
        dump(os.path.expanduser(a.hf), os.path.expanduser(a.pack), a.out)


if __name__ == "__main__":
    sys.exit(main())
