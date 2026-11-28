#!/usr/bin/env python3
"""thinkingmachines/Inkling-2-Mini (model_type inkling2_loop) -> an MLX pack in
mlx-lm's layout (ml-explore/mlx-lm#2086, the mlx-community/Inkling-2-Mini-4bit
pack), so it loads in mlx-serve and mlx-lm alike.

The release is already quantized where it matters: the routed experts are
NVFP4 with one global scale per expert (layers 0-19) and MXFP8 (layers 20-23).
Those bytes are kept (lossless): `w13_weight` -> `gate_up_proj`, `w2_weight`
-> `down_proj`, codes viewed as u32, block scales as stored, and
`global_scale = amax * 6 * 448 / outer_scale_denominator`.

`--bits` sets the rest of the language model, as `mlx_lm.convert -q --q-bits`
does: the embedding, the head, the attention projections and the dynamic
short convs' weight projections go affine at that width (group 64); 16 keeps
them bf16. The router, the depth queries and the loop gates stay float; the
vision tower stays bf16. The DSpark draft (`mtp.safetensors`) is copied as it is.

`--embed-bits` and `--head-bits` set those two off the `--bits` width (the attention's).

`--experts` re-quantizes the routed experts too, for a Mac the release's ~15.7 GB of expert bytes do not fit (a 16 GB
Mac: about 13 GB resident). The release ships no bf16 experts, so each bank is dequantized (NVFP4 / MXFP4 / MXFP8, the
per-expert global scale applied) and quantized again as mlx-lm's affine: errors compound, score the pack against the
release before using it. SPEC is `BITS:GROUP` for every layer, then `LAYERS=BITS:GROUP` or `LAYERS=keep` overrides
(`LAYERS` = `7` or `20-23`), e.g. `3:64,0-1=4:64,20-23=keep`. A bank off the `--bits` block gets its own entry in
`quantization`, keyed by its module path, as `mlx_lm.convert` writes a mixed-width pack.

Names follow mlx-lm's `models/inkling2/checkpoint.py` sanitize.

  python tests/convert_inkling2.py --src ~/.mlx-serve/models/thinkingmachines/Inkling-2-Mini \
      --dst ~/.mlx-serve/models/ddalcu/Inkling-2-Mini-MLX-8bit
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx

MAX_SHARD_BYTES = 5 << 30
GROUP_SIZE = 64

RENAMES = [
    (r"^model\.llm\.unembed\.", "lm_head."),
    (r"^model\.llm\.embed\.", "model.language_model.embed_tokens."),
    (r"^model\.llm\.", "model.language_model."),
    (r"\.attn_res\.mlp_queries\.(\d+)$", r".layers.\1.mlp_input.query"),
    (r"\.attn_res\.output_query$", ".final_input.query"),
    (r"\.attn\.wq_du\.", ".self_attn.q_proj."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.r_proj."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj."),
    (r"\.attn\.gate_attn\.", ".self_attn.gate_proj."),
    (r"\.attn\.attn_sink_proj\.", ".self_attn.sink_proj."),
    (r"\.controllers\.attn\.", ".q_message_gate."),
    (r"\.controllers\.(k|v|mlp)\.", r".\1_message_gate."),
    (r"_message_gate\.down_weight$", "_message_gate.down_proj.weight"),
    (r"_message_gate\.forget_weight$", "_message_gate.forget_gate_proj.weight"),
    (r"_message_gate\.gain_weight$", "_message_gate.message_gate_proj.weight"),
    (r"\.attn\.", ".self_attn."),
    (r"\.attn_sconv\.", ".post_attention_convolution."),
    (r"\.mlp_sconv\.", ".post_mlp_convolution."),
    (r"sconv\.weight$", "sconv.conv1d.weight"),
    (r"convolution\.weight$", "convolution.conv1d.weight"),
    (r"sconv\.static_w$", "sconv.static_weight"),
    (r"sconv\.weight_proj$", "sconv.weight_proj.weight"),
    (r"\.mlp_norm\.", ".post_attention_layernorm."),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.(q|k|v)_input_norm\.", r".\1_input_layernorm."),
    (r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
    (r"^model\.visual\.backbone\.ln_pre\.", "model.vision_tower.pre_layernorm."),
    (r"^model\.visual\.backbone\.", "model.vision_tower."),
    (r"^model\.visual\.proj_in\.", "model.vision_tower.merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "model.vision_tower.merger.linear_2."),
    (r"embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"(vision_tower\.layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"(vision_tower\.layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"(vision_tower\.layers\.\d+)\.ls_(\d)\.gamma$", r"\1.lambda_\2"),
]

EXPERTS = re.compile(r"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$")
ATTN_QUERY = re.compile(r"^model\.llm\.attn_res\.attn_queries\.(\d+)$")
FLOAT32 = (".mlp.gate.weight", ".forget_gate_proj.weight", ".message_gate_proj.weight")
# What `nn.quantize` reaches under mlx-lm's predicate: the Linear and Embedding modules outside the vision tower and
# the loop gates.
QUANTIZED = re.compile(r"^(lm_head|model\.language_model\.(embed_tokens|layers\.\d+\.self_attn\.(\w+_proj|[kv]_sconv\.weight_proj)))\.weight$")


def rename(key: str) -> str:
    if m := ATTN_QUERY.match(key):
        # Layer 0 has no attention input, so query i is layer i + 1's.
        return f"model.language_model.layers.{int(m.group(1)) + 1}.attention_input.query"
    for pattern, replacement in RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


def sanitize(weights: dict) -> dict:
    out, amax, denominator, queries = {}, {}, {}, {}
    for k, v in weights.items():
        k = rename(k)
        if k.endswith("conv1d.weight"):
            v = v.swapaxes(1, 2)
        if k.endswith(("cross_pass_queries", "cross_pass_kv_queries")):
            queries["kv" if "_kv_" in k else "q"] = v
            continue
        if k.endswith(FLOAT32):
            v = v.astype(mx.float32)
        elif m := EXPERTS.match(k):
            prefix, which, suffix = m.groups()
            name = f"{prefix}.{'gate_up_proj' if which == 'w13' else 'down_proj'}"
            if suffix == "_amax":
                amax[name] = v
            elif suffix == "_outer_scale_denominator":
                denominator[name] = v
            elif suffix == "_scale":
                out[f"{name}.scales"] = v.view(mx.uint8)
            elif suffix == "":
                out[f"{name}.weight"] = v.view(mx.uint32)
            else:
                raise SystemExit(f"unknown expert tensor {k}")
            continue
        out[k] = v
    # The release holds [query, mlp] and [key, value] rows per looped layer.
    dims = queries["q"].shape[-1]
    q = queries["q"].reshape(-1, 2, dims)
    kv = queries["kv"].reshape(-1, 2, dims)
    out["model.language_model.message_input.query"] = mx.concatenate([q[:, :1], kv, q[:, 1:]], axis=1)
    if amax.keys() != denominator.keys():
        raise SystemExit("an expert bank has an amax without its denominator")
    for name in amax:
        scale = 6.0 * 448.0 / denominator[name].astype(mx.float32)
        out[f"{name}.global_scale"] = amax[name].astype(mx.float32) * scale
    return out


EXPERT_BANK = re.compile(r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj))\.weight$")
# Last-axis ratio of the packed words to the block scales: NVFP4 (e2m1, e4m3 per 16), MXFP4 and MXFP8 (e8m0 per 32).
RELEASE_EXPERTS = {2: ("nvfp4", 4, 16), 4: ("mxfp4", 4, 32), 8: ("mxfp8", 8, 32)}
NVFP4_GLOBAL_UNIT = 6.0 * 448.0


def parse_experts(spec: str | None, layers: int):
    """`BITS:GROUP[,LAYERS=BITS:GROUP|keep ...]` -> one (bits, group) or None (keep the release bytes) per layer."""
    plan = [None] * layers
    if not spec:
        return plan
    for i, part in enumerate(spec.split(",")):
        where, _, what = part.rpartition("=") if "=" in part else ("", "", part)
        if bool(where) == (i == 0):
            raise SystemExit(f"--experts: the first entry is BITS:GROUP for every layer, the rest LAYERS=...: {part!r}")
        want = None if what == "keep" else tuple(int(x) for x in what.split(":"))
        if want is not None and (len(want) != 2 or want[0] not in (2, 3, 4, 5, 6, 8) or want[1] not in (32, 64, 128)):
            raise SystemExit(f"--experts: {what!r} is not BITS:GROUP with bits 2-6 or 8 and group 32, 64 or 128")
        lo, _, hi = where.partition("-") if where else ("0", "", str(layers - 1))
        for n in range(int(lo), int(hi or lo) + 1):
            plan[n] = want
    return plan


def requantize_bank(w, scales, global_scale, bits: int, group: int):
    """A release expert bank `[E, out, in]` as mlx-lm's affine at `bits`, `group`: (weight, scales, biases)."""
    mode, rbits, rgroup = RELEASE_EXPERTS[w.shape[-1] // scales.shape[-1]]
    parts = []
    for e in range(0, w.shape[0], 8):  # a few experts at a time: the dequantized bank is 2.7 GB in f32
        x = mx.dequantize(w[e:e + 8], scales[e:e + 8], group_size=rgroup, bits=rbits, mode=mode).astype(mx.float32)
        if global_scale is not None:
            x = x * (global_scale[e:e + 8].astype(mx.float32) / NVFP4_GLOBAL_UNIT)[:, None, None]
        x = x.astype(mx.bfloat16)
        q = mx.quantize(x, group_size=group, bits=bits)
        mx.eval(q)
        parts.append(q)
    return tuple(mx.concatenate([p[i] for p in parts], axis=0) for i in range(3))


def quantize(weights: dict, bits: int, plan=(), overrides=None, ends=None) -> dict:
    out, banks = {}, {}
    for k in weights:
        if (m := EXPERT_BANK.match(k)) and plan and plan[int(m.group(2))]:
            banks[m.group(1)] = plan[int(m.group(2))]
    for k, v in weights.items():
        base = k.rsplit(".", 1)[0]
        if base in banks and k.endswith((".scales", ".global_scale")):
            continue
        if k.endswith(".weight") and base in banks:
            width = banks[base]
            w, sc, bi = requantize_bank(v, weights[f"{base}.scales"], weights.get(f"{base}.global_scale"), *width)
            out[k], out[f"{base}.scales"], out[f"{base}.biases"] = w, sc, bi
            if overrides is not None and width != (bits, GROUP_SIZE):
                overrides[base] = {"group_size": width[1], "bits": width[0], "mode": "affine"}
            continue
        if bits < 16 and QUANTIZED.match(k):
            # The embedding and the head may sit off the `--bits` width (`ends`), listed per module path.
            width = (ends or {}).get(base, bits)
            if width != bits and overrides is not None:
                overrides[base] = {"group_size": GROUP_SIZE, "bits": width, "mode": "affine"}
            w, scales, biases = mx.quantize(v, group_size=GROUP_SIZE, bits=width)
            out[k], out[f"{base}.scales"], out[f"{base}.biases"] = w, scales, biases
        else:
            out[k] = v
    return out


def config_for(src: dict, bits: int, overrides: dict) -> dict:
    cfg = {k: v for k, v in src.items() if k not in ("dtype", "quantization_config")}
    cfg["text_config"] = {k: v for k, v in src["text_config"].items() if k != "model_type"}
    if bits < 16 or overrides:
        block = {"group_size": GROUP_SIZE, "bits": bits if bits < 16 else 4, "mode": "affine", **overrides}
        cfg["quantization"] = block
        cfg["quantization_config"] = dict(block)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--bits", type=int, default=8, choices=(4, 5, 6, 8, 16))
    ap.add_argument("--embed-bits", type=int, choices=(4, 5, 6, 8), help="the embedding's width when it is not --bits")
    ap.add_argument("--head-bits", type=int, choices=(4, 5, 6, 8), help="the LM head's width when it is not --bits")
    ap.add_argument("--experts", help="re-quantize the routed experts: BITS:GROUP[,LAYERS=BITS:GROUP|keep ...] (see the header)")
    ap.add_argument("--header-only", action="store_true", help="write the layout (name, dtype, shape) as JSON to --dst and stop")
    args = ap.parse_args()

    # Quantized on the default device (the GPU), as mlx-lm's convert: the CPU rounds a few codes per thousand the
    # other way, and the pack is meant to equal mlx-lm's byte for byte.
    shards = sorted(args.src.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no model-*.safetensors under {args.src}")
    weights = {}
    for shard in shards:
        weights.update(mx.load(str(shard)))
    weights = sanitize(weights)
    src_cfg = json.loads((args.src / "config.json").read_text())
    plan = parse_experts(args.experts, src_cfg["text_config"]["num_hidden_layers"])
    overrides = {}
    if args.bits < 16 or any(plan):
        ends = {"model.language_model.embed_tokens": args.embed_bits or args.bits, "lm_head": args.head_bits or args.bits}
        weights = quantize(weights, args.bits, plan, overrides, ends)

    if args.header_only:
        layout = {k: [str(v.dtype).split(".")[-1], list(v.shape)] for k, v in weights.items()}
        args.dst.write_text(json.dumps(layout, indent=0, sort_keys=True))
        print(f"{args.dst}: {len(layout)} tensors")
        return

    args.dst.mkdir(parents=True, exist_ok=True)
    groups, size = [{}], 0
    for k in sorted(weights):
        n = weights[k].nbytes
        if groups[-1] and size + n > MAX_SHARD_BYTES:
            groups.append({})
            size = 0
        groups[-1][k] = weights[k]
        size += n
    weight_map, total = {}, 0
    for i, group in enumerate(groups):
        name = f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors"
        mx.eval(*group.values())
        mx.save_safetensors(str(args.dst / name), group, metadata={"format": "mlx"})
        for k, v in group.items():
            weight_map[k] = name
            total += v.nbytes
        print(f"{name}: {len(group)} tensors", flush=True)
        group.clear()
        mx.clear_cache()
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (args.dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    cfg = config_for(src_cfg, args.bits, overrides)
    (args.dst / "config.json").write_text(json.dumps(cfg, indent=4) + "\n")
    for name in ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "processor_config.json", "mtp.safetensors"):
        if (args.src / name).exists():
            shutil.copy2(args.src / name, args.dst / name)
    print(f"{args.dst}: {len(weight_map)} tensors, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
