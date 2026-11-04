#!/usr/bin/env python3
"""Verify the converted Super LiLiCorr head with NVIDIA's executed full head.

CPU only, with synthetic embeddings and hidden states. This checks the actual
converted quantized head, explicit in_proj names, flattened embeddings, and
materialized attention bias. No target-model weights or personal media are used.

  python tests/verify_nemotron_lilicorr.py --source <NVFP4 checkpoint> \
    --converted <MLX bf16 checkpoint> --nvfp4-reference <nvfp4_tensor.py> \
    --head-reference <modeling_lilicorr.py> --out <verification.json>

The two reference files come from NVIDIA/Model-Optimizer at the revision pinned
in convert_nemotron_lilicorr.py; their full SHA256 hashes are validated.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import types

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

HEAD_SHA256 = "67819e3760d8078d5a010c63271fb130fd40478470dd82b361756f28aea1ca6d"


def head_reference(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != HEAD_SHA256:
        raise ValueError("LiLiCorr reference does not match the pinned NVIDIA SHA256")
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ("LiLiCorrLayer", "LiLiCorrHead")]
    scope = dict(torch=torch, nn=nn, F=F, _NORM_CLS=Qwen3RMSNorm, _NUM_CANDIDATE_FEATURES=5)
    exec(compile(tree, "NVIDIA LiLiCorr reference", "exec"), scope)
    return scope["LiLiCorrHead"]


def original_name(name):
    name = name.removeprefix("lilicorr.")
    name = name.replace("feature_norm.", "feature_mlp.0.")
    name = name.replace("feature_mlp.up_proj.", "feature_mlp.1.").replace("feature_mlp.down_proj.", "feature_mlp.3.")
    name = name.replace(".mlp.up_proj.", ".mlp.0.").replace(".mlp.down_proj.", ".mlp.2.")
    return name.replace(".attn.in_proj.weight", ".attn.in_proj_weight").replace(".attn.in_proj.bias", ".attn.in_proj_bias")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "converted", "nvfp4-reference", "head-reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("converter", Path(__file__).with_name("convert_nemotron_lilicorr.py"))
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    dequantize = converter.modelopt_reference(args.nvfp4_reference)
    cls = head_reference(args.head_reference)
    config = json.loads((args.source / "config.json").read_text())
    d = config["dflash_config"]
    kwargs = dict(model_hidden_size=config["hidden_size"], hidden_size=d["lilicorr_hidden_size"],
                  num_layers=d["lilicorr_num_layers"], num_heads=d["lilicorr_num_heads"],
                  mlp_ratio=d["lilicorr_mlp_ratio"], block_size=config["block_size"],
                  candidate_topk=d["lilicorr_candidate_topk"], factor_dim=d["lilicorr_factor_dim"],
                  rms_norm_eps=config["rms_norm_eps"], vector_eps=d["lilicorr_vector_eps"],
                  logit_scale=d["lilicorr_logit_scale"])
    torch.set_num_threads(8)
    torch.manual_seed(870)
    native = cls(**kwargs).to(dtype=torch.bfloat16).eval()
    converted = cls(**kwargs).to(dtype=torch.bfloat16).eval()
    state = {}
    with safe_open(str(args.source / "model.safetensors"), framework="pt") as source:
        keys = set(source.keys())
        for name in sorted(keys):
            if not name.startswith("lilicorr.") or name.endswith((".weight_scale", ".weight_scale_2")):
                continue
            value = source.get_tensor(name)
            if value.dtype == torch.uint8:
                base = name.removesuffix(".weight")
                value = dequantize(value, source.get_tensor(base + ".weight_scale"), source.get_tensor(base + ".weight_scale_2")).to(torch.bfloat16)
            state[original_name(name)] = value
    native.load_state_dict(state)
    del state
    index = json.loads((args.converted / "model.safetensors.index.json").read_text())["weight_map"]
    state, materialized = {}, None
    for shard in sorted({file for name, file in index.items() if name.startswith("lilicorr.")}):
        with safe_open(str(args.converted / shard), framework="pt") as weights:
            for name in weights.keys():
                if not name.startswith("lilicorr."):
                    continue
                value = weights.get_tensor(name)
                if name == "lilicorr.attention_bias":
                    materialized = value
                    continue
                if name.endswith("slot_embedding"):
                    value = value.reshape(1, 1, config["block_size"] - 1, 1, d["lilicorr_hidden_size"])
                elif name.endswith("rank_embedding"):
                    value = value.reshape(1, 1, 1, d["lilicorr_candidate_topk"], d["lilicorr_hidden_size"])
                state[original_name(name)] = value
    missing, unexpected = converted.load_state_dict(state, strict=False)
    assert set(missing) == {"relative_slot_bias", "same_slot_bias"} and not unexpected, (missing, unexpected)
    del state
    expected_bias = native._attention_bias(1, device=torch.device("cpu"), dtype=torch.bfloat16, topk=d["lilicorr_candidate_topk"])
    assert torch.equal(materialized, expected_bias), "Materialized attention bias differs from NVIDIA"

    def fixed_bias(self, batch_blocks, *, device, dtype, topk):
        assert topk == d["lilicorr_candidate_topk"]
        return materialized[None].expand(batch_blocks, -1, -1, -1).reshape(batch_blocks * materialized.shape[0], *materialized.shape[1:]).to(device=device, dtype=dtype)
    converted._attention_bias = types.MethodType(fixed_bias, converted)
    cases = []
    for seed in (870, 871):
        torch.manual_seed(seed)
        slots, hidden = config["block_size"] - 1, config["hidden_size"]
        embeddings = nn.Embedding(97, hidden).to(dtype=torch.bfloat16).eval()
        logits = torch.randn(1, 1, slots, 97)
        _, ids = logits.topk(d["lilicorr_candidate_topk"], dim=-1)
        log_probs = logits.log_softmax(dim=-1).gather(-1, ids)
        inputs = dict(candidate_token_ids=ids, candidate_log_probs=log_probs,
                      pass_hidden=torch.randn(1, 1, slots, hidden).to(torch.bfloat16),
                      embed_tokens=embeddings, anchor_hidden=torch.randn(1, 1, 1, hidden).to(torch.bfloat16),
                      anchor_valid=None)
        with torch.inference_mode():
            a = native.compose_log_factors(**dict(zip(("start_scores", "pair_scores"), native(**inputs))))
            b = converted.compose_log_factors(**dict(zip(("start_scores", "pair_scores"), converted(**inputs))))
        differences = [float((x - y).abs().max()) for x, y in zip(a, b)]
        assert differences == [0., 0.], differences
        cases.append({"seed": seed, "start_max_abs_difference": differences[0], "pair_max_abs_difference": differences[1]})
    report = {"reference_revision": converter.REFERENCE_REVISION, "head_reference_sha256": HEAD_SHA256,
              "dtype": "bfloat16", "geometry": kwargs, "attention_bias_equal": True, "cases": cases}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
