#!/usr/bin/env python3
"""Reference fixtures for the inkling2 plugin, from mlx-lm's Inkling 2 (`mlx_lm/models/inkling2`, commit 0c4c991).

Every reference run follows transformers' declared bf16 policy instead of mlx-lm's: the static short convolutions run
in f32 and every weighted RMSNorm multiplies in f32 (`policy`). Both are no-ops in f32; the script asserts that, and
asserts that each patch changes a bf16 run.

`--out .../tiny` (CPU f32 arm): a tiny random `inkling2_loop` model (8 layers, a 3-pass core, 16 visits, sliding and
global attention, NVFP4-sized widths) with bf16-rounded weights stored as F32, and its logits over a 32-token prompt
and 4 decode steps through mlx-lm's own caches.

`--quant .../tiny_q` (GPU arm): the same shape at a smaller weight scale, the trunk affine 4-bit g32, layers 0-6's
experts NVFP4 with a per-expert global scale, layer 7's MXFP8, stored in the pack's dtypes. The truth is a CPU f32
forward over the dequantized weights; the other arm is mlx-lm on the GPU over the quantized weights, kernels off.

`--pack PACK --pack-out DIR` (not committed): a real pack's chat prompt, greedy steps and teacher-forced logits.

  <venv with the mlx-lm inkling2 PR>/bin/python -I tests/dump_inkling2_fixtures.py \\
      --out src/plugins/inkling2/fixtures/tiny --quant src/plugins/inkling2/fixtures/tiny_q
  <venv with the mlx-lm inkling2 PR>/bin/python -I tests/dump_inkling2_fixtures.py \\
      --pack <Inkling-2-Mini pack dir> --pack-out <dir>
"""
import argparse
import copy
import json
import math
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.inkling2 import Model, ModelArgs, layers
from mlx_lm.models.inkling2.model import TextModel
from mlx_lm.utils import load_model

CONFIG = {
    "model_type": "inkling2_loop",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "eos_token_id": 2,
    "text_config": {
        "hidden_size": 32,
        "num_hidden_layers": 8,
        "vocab_size": 128,
        "unpadded_vocab_size": 120,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 8,
        "d_rel": 4,
        # The band engages within the prompt, and the global extent differs from the window.
        "sliding_window_size": 8,
        "rel_extent": 6,
        "local_layer_ids": [0, 2, 4, 5, 6],
        "n_layers_per_attn_res_block": 2,
        "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 4, "is_varloop": False},
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "qk_l2_clip_threshold": 2.0,
        "rel_l2_clip_threshold": 1.0,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "intermediate_size": 32,
        "route_scale": 8.0,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "final_logit_softcapping": 30.0,
        "sconv_kernel_size": 4,
        "rms_norm_eps": 1e-6,
        "n_shared_experts": 0,
        "use_gate_bias": True,
        "attn_res_normalize_values": True,
        "use_value_norm_no_gamma": True,
        "q_bias": False,
        "o_bias": False,
    },
}
# 32 tokens x top-2 = 64 expert rows, so the prefill takes the sorted gather; decode takes the unsorted one.
PROMPT = [102, 21, 3, 76, 43, 56, 9, 44, 77, 42, 99, 94, 84, 108, 86, 21,
          103, 78, 11, 35, 19, 116, 87, 110, 33, 76, 72, 90, 14, 61, 77, 99]
DECODE = [78, 53, 55, 40]

# Weight scales: N(0, scale) for weights, 1 + N(0, norm) for norm weights and clip scales.
CPU = {"seed": 7, "scale": 0.3, "norm": 0.2, "bias": 0.1, "seeds": 20}
# At 0.3 mlx-lm's own bf16 run is chaotic on this model; at 0.05 the clips need lower thresholds to engage.
# Expert widths of 64: at an inner dimension of 32, MLX's sorted gather_qmm intermittently returns non-finite rows.
QUANT = {"seed": 7, "seeds": 20, "scale": 0.05, "norm": 0.05, "bias": 0.3, "qk": 0.5, "rel": 0.25,
         "hidden_size": 64, "intermediate_size": 64}
ROUTER_MARGIN = 1e-4
MIN_DRIFT_COS = 0.99
CLIP_SENSITIVITY = 1e-3
NVFP4_LAYERS = 7  # layers 0..6; layer 7 is MXFP8
PACK_PROMPT = "Explain why the sky is blue."
PACK_STEPS = 16


def check(ok, what):
    if not ok:
        raise SystemExit(f"check failed: {what}")


def raw(path, arr):
    path.write_bytes(np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).tobytes())


def conv_f32(self, hidden_states, conv_state, conv_mask):
    K = self.kernel_size
    residual = hidden_states.astype(mx.float32)
    window = layers._conv_input(hidden_states, conv_state, conv_mask, K)
    w = self.conv1d.weight.astype(mx.float32)
    out = (mx.conv1d(window, w, groups=w.shape[0]) + residual) / math.sqrt(K + 1)
    return out.astype(hidden_states.dtype), window


def rms_f32(self, x):
    return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps)


def embed_f32(self, inputs):
    e = self.embed_tokens(inputs)
    return self.embed_norm(e).astype(e.dtype)


CONV = [(layers.ShortConvolution, "__call__", conv_f32)]
NORM = [(nn.RMSNorm, "__call__", rms_f32), (TextModel, "embed", embed_f32)]


@contextmanager
def policy(*groups):
    """transformers' declared bf16 rounding (module docstring); `policy()` applies all of it."""
    patches = [p for g in (groups or (CONV, NORM)) for p in g]
    saved = [(cls, name, cls.__dict__[name]) for cls, name, _ in patches]
    for cls, name, fn in patches:
        setattr(cls, name, fn)
    try:
        yield
    finally:
        for cls, name, fn in saved:
            setattr(cls, name, fn)


@contextmanager
def kernels_off():
    saved = layers._use_kernels
    layers._use_kernels = lambda: False
    try:
        yield
    finally:
        layers._use_kernels = saved


@contextmanager
def router_margins():
    """Collects, per router call (one per visit), the smallest gap between the k-th and (k+1)-th biased score."""
    gaps, saved = [], layers.TopKRouter.__call__

    def probe(self, hidden_states):
        logits = hidden_states.astype(mx.float32) @ self.weight.T
        biased = mx.sort(mx.sigmoid(logits) + self.e_score_correction_bias, axis=-1)
        gaps.append((biased[..., -self.top_k] - biased[..., -self.top_k - 1]).min())
        return saved(self, hidden_states)

    layers.TopKRouter.__call__ = probe
    try:
        yield gaps
    finally:
        layers.TopKRouter.__call__ = saved


def quant_config():
    config = copy.deepcopy(CONFIG)
    config["text_config"].update(qk_l2_clip_threshold=QUANT["qk"], rel_l2_clip_threshold=QUANT["rel"],
                                 hidden_size=QUANT["hidden_size"], intermediate_size=QUANT["intermediate_size"])
    config["quantization"] = config["quantization_config"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    return config


def build(config):
    return Model(ModelArgs.from_dict(config))


def randomize(model, arm):
    params = dict(tree_flatten(model.parameters()))
    for name, v in params.items():
        if name.endswith(("norm.weight", "l2_clip.scale")):
            params[name] = 1.0 + arm["norm"] * mx.random.normal(v.shape)
        elif name.endswith("e_score_correction_bias"):
            params[name] = arm["bias"] * mx.random.normal(v.shape)
        else:
            # Queries too: a zero query makes the depth softmax uniform.
            params[name] = arm["scale"] * mx.random.normal(v.shape)
    model.load_weights(list(params.items()))
    mx.eval(model.parameters())


def pack_dtypes(model, stored=None):
    """Rounds to the packs' dtypes: bf16, except the tensors they keep F32 (queries, router, message-gate up
    projections). `stored` then re-types every tensor without changing its value."""
    keep = model.cast_predicate

    def cast(k, v):
        v = v.astype(mx.bfloat16) if keep(k) else v.astype(mx.float32)
        return v if stored is None else v.astype(stored)

    model.update(tree_unflatten([(k, cast(k, v)) for k, v in tree_flatten(model.parameters())]))
    mx.eval(model.parameters())


def run(model):
    cache = model.make_cache()
    prefill = model(mx.array([PROMPT]), cache=cache)[0]
    steps = [model(mx.array([[t]]), cache=cache)[0, -1] for t in DECODE]
    out = (prefill.astype(mx.float32), mx.stack(steps).astype(mx.float32))
    mx.eval(out)
    return out


def set_clips(model, qk, rel):
    for layer in model.layers:
        attn = layer.self_attn
        attn.q_l2_clip.threshold = attn.k_l2_clip.threshold = qk
        attn.rel_l2_clip.threshold = rel


def check_clips_engage(model, base, config, what):
    t = config["text_config"]
    set_clips(model, 1e9, 1e9)
    moved = mx.abs(run(model)[0] - base[0]).max().item()
    set_clips(model, t["qk_l2_clip_threshold"], t["rel_l2_clip_threshold"])
    check(moved > CLIP_SENSITIVITY,
          f"{what}: lifting the clip thresholds moves the logits by {moved:.3g} <= {CLIP_SENSITIVITY}")
    return moved


def save(path, model):
    mx.save_safetensors(str(path), dict(tree_flatten(model.parameters())), metadata={"format": "mlx"})


def same(a, b):
    return all(mx.array_equal(x, y).item() for x, y in zip(a, b))


def min_cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1])
    b = np.asarray(b, dtype=np.float64).reshape(-1, b.shape[-1])
    return float(np.min((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))))


def dump_cpu(out):
    out.mkdir(parents=True, exist_ok=True)
    mx.set_default_device(mx.cpu)
    for seed in range(CPU["seed"], CPU["seed"] + CPU["seeds"]):
        mx.random.seed(seed)
        model = build(CONFIG)
        randomize(model, CPU)
        # bf16-rounded values stored as F32: the reference then runs in f32 at every weight-dtype cast, as the plugin.
        pack_dtypes(model, mx.float32)
        with policy(), router_margins() as gaps:
            logits = run(model)
        margin = min(g.item() for g in gaps)
        if margin > ROUTER_MARGIN:
            break
        print(f"seed {seed}: router margin {margin:.3g} <= {ROUTER_MARGIN}, trying seed {seed + 1}")
    else:
        check(False, f"no seed in {CPU['seeds']} gives a router margin above {ROUTER_MARGIN}")

    check(same(run(model), logits), "the reference policy is not a no-op in f32")
    with policy():
        moved = check_clips_engage(model, logits, CONFIG, "tiny")

    save(out / "model.safetensors", model)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=1))
    reloaded, _ = load_model(out)
    with policy():
        check(same(run(reloaded), logits), "a fresh model loading the saved tiny pack does not reproduce the logits")

    raw(out / "prefill_logits.f32", logits[0])
    raw(out / "decode_logits.f32", logits[1])
    (out / "tokens.json").write_text(json.dumps({"prompt": PROMPT, "decode": DECODE}))
    print(f"wrote {out}: seed {seed}, prefill {tuple(logits[0].shape)}, decode {tuple(logits[1].shape)}, "
          f"router margin {margin:.3g}, clips move the logits by {moved:.3g}")


def quantize_experts(model):
    """Layers 0-6 NVFP4 with a per-expert global scale of 2688 x U(0.5, 2), so each expert's scale (gs / 2688) is
    far from 1 and differs per expert; layer 7 MXFP8 without one."""
    for i, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            lin = getattr(experts, name)
            if i >= NVFP4_LAYERS:
                setattr(experts, name, lin.to_quantized(32, 8, mode="mxfp8"))
                continue
            ql = lin.to_quantized(16, 4, mode="nvfp4", global_scale=True)
            gs = 2688.0 * mx.random.uniform(0.5, 2.0, (lin.weight.shape[0],))
            qs = [mx.quantize(lin.weight[e], 16, 4, mode="nvfp4", global_scale=gs[e]) for e in range(gs.shape[0])]
            ql.weight, ql.scales, ql.global_scale = mx.stack([q for q, _ in qs]), mx.stack([s for _, s in qs]), gs
            setattr(experts, name, ql)
    mx.eval(model.parameters())


def dequantized(model, config):
    """A dense f32 copy of a quantized model. MLX's CPU backend has no global-scale dequantize: those run on the GPU."""
    dense, skip = {}, set()
    for path, m in model.named_modules():
        if not hasattr(m, "scales"):
            continue
        skip |= {f"{path}.{p}" for p in ("weight", "scales", "biases", "global_scale")}
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            w = mx.dequantize(m.weight, m.scales, m.biases, m.group_size, m.bits, mode=m.mode, dtype=mx.float32)
        else:
            gs = m.global_scale
            w = mx.stack([mx.dequantize(m.weight[e], m.scales[e], None, m.group_size, m.bits, mode=m.mode,
                                        global_scale=None if gs is None else gs[e], dtype=mx.float32, stream=mx.gpu)
                          for e in range(m.weight.shape[0])])
        dense[f"{path}.weight"] = w
    for k, v in tree_flatten(model.parameters()):
        if k not in skip:
            dense[k] = v.astype(mx.float32)
    truth = build(config)
    truth.load_weights(list(dense.items()))
    mx.eval(truth.parameters())
    return truth


def quantized_model(config):
    """The tiny model in its pack form: weights rounded to the pack dtypes, the trunk affine, the experts fp4/fp8."""
    model = build(config)
    randomize(model, QUANT)
    pack_dtypes(model)
    trunk = model.quant_predicate
    nn.quantize(model, group_size=32, bits=4, mode="affine",
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and ".mlp.experts." not in p and trunk(p, m))
    quantize_experts(model)
    return model


def dump_quant(out):
    out.mkdir(parents=True, exist_ok=True)
    config = quant_config()
    for seed in range(QUANT["seed"], QUANT["seed"] + QUANT["seeds"]):
        mx.set_default_device(mx.gpu)
        mx.random.seed(seed)
        model = quantized_model(config)
        truth = dequantized(model, config)
        mx.set_default_device(mx.cpu)
        with policy():
            want = run(truth)
        mx.set_default_device(mx.gpu)
        with kernels_off(), policy():
            gpu = run(model)
        # A router near-tie that bf16 flips sinks one row; such a seed is no fixture for a 0.999 bar.
        drift = min(min_cos(gpu[0], want[0]), min_cos(gpu[1], want[1]))
        if drift >= MIN_DRIFT_COS:
            break
        print(f"seed {seed}: mlx-lm GPU vs the f32 truth min cos {drift:.5f} < {MIN_DRIFT_COS}, trying seed {seed + 1}")
    else:
        check(False, f"no seed in {QUANT['seeds']} keeps mlx-lm's GPU run within min cos {MIN_DRIFT_COS} of the truth")

    mx.set_default_device(mx.cpu)
    with policy():
        moved = check_clips_engage(truth, want, config, "tiny_q truth")
    mx.set_default_device(mx.gpu)
    with kernels_off():
        bare = run(model)
        for name, group in (("f32 static conv", CONV), ("f32 weighted norm", NORM)):
            with policy(group):
                delta = mx.abs(run(model)[0] - bare[0]).max().item()
            check(delta > 0, f"the {name} patch does not change the bf16 GPU logits")

    save(out / "model.safetensors", model)
    (out / "config.json").write_text(json.dumps(config, indent=1))
    reloaded, _ = load_model(out)
    with kernels_off(), policy():
        check(same(run(reloaded), gpu), "a fresh model loading the saved tiny_q pack does not reproduce the GPU logits")

    raw(out / "truth_prefill.f32", want[0])
    raw(out / "truth_decode.f32", want[1])
    raw(out / "gpu_prefill.f32", gpu[0])
    raw(out / "gpu_decode.f32", gpu[1])
    print(f"wrote {out}: seed {seed}, GPU vs truth min cos {drift:.6f}, clips move the truth logits by {moved:.3g}")


def dump_pack(pack, out):
    from mlx_lm import load

    out.mkdir(parents=True, exist_ok=True)
    mx.set_default_device(mx.gpu)
    model, tok = load(pack)
    prompt = list(tok.apply_chat_template([{"role": "user", "content": PACK_PROMPT}], add_generation_prompt=True))
    with policy():
        cache = model.make_cache()
        last = model(mx.array([prompt]), cache=cache)[0, -1].astype(mx.float32)
        steps, rows, logits = [], [], last
        for _ in range(PACK_STEPS):
            steps.append(mx.argmax(logits).item())
            logits = model(mx.array([[steps[-1]]]), cache=cache)[0, -1].astype(mx.float32)
            rows.append(logits)
        mx.eval(last, rows)
    (out / "prompt.json").write_text(json.dumps(prompt))
    (out / "steps.json").write_text(json.dumps(steps))
    raw(out / "prefill_last.f32", last)
    raw(out / "decode_logits.f32", mx.stack(rows))
    print(f"wrote {out}: {len(prompt)} prompt tokens, steps {steps}: {tok.decode(steps)!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--quant", type=Path)
    ap.add_argument("--pack", type=Path)
    ap.add_argument("--pack-out", type=Path)
    a = ap.parse_args()
    if not (a.out or a.quant or a.pack):
        ap.error("nothing to do: pass --out, --quant or --pack")
    if bool(a.pack) != bool(a.pack_out):
        ap.error("--pack and --pack-out go together")
    if a.out:
        dump_cpu(a.out)
    if a.quant:
        dump_quant(a.quant)
    if a.pack:
        dump_pack(a.pack, a.pack_out.expanduser())


if __name__ == "__main__":
    main()
