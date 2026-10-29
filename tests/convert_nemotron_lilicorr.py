#!/usr/bin/env python3
"""Convert NVIDIA's W4A16 NVFP4 LiLiCorr checkpoint (Nemotron 3.5 Super) to dense MLX bf16, the target's drafter/.

  python tests/convert_nemotron_lilicorr.py --source <download> --out <new dir> --reference <nvfp4_tensor.py>

Needs torch + safetensors and NVIDIA Model Optimizer's nvfp4_tensor.py at the pinned commit:
  https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/6bf36980e1d73a46750c947b7ff4c3cc3fa9c34d/modelopt/torch/quantization/qtensor/nvfp4_tensor.py
Every packed matrix is checked against that unmodified CPU dequantizer (E4M3 block scales, fp32 global scale), every
source tensor is consumed, layout changes go in the manifest, shards are hashed. The checkpoint's own LM head is
kept; no new quantization. An existing output dir is refused; install only a completed output as `drafter/`.
CPU regressions: python tests/test_convert_nemotron_lilicorr.py.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE_REPO = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr"
SOURCE_REVISION = "7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2"
SOURCE_MODEL_SHA256 = "eec5efdf1b838020c2842f52d560c187515e84889c51f99d70b75c773c286599"
SOURCE_CONFIG_GIT_BLOB = "7ec9439a8faa16087e42a6d0ddd8da2eae35db03"
REFERENCE_REVISION = "6bf36980e1d73a46750c947b7ff4c3cc3fa9c34d"
REFERENCE_SHA256 = "e128fd62f0d0c4aa179e057b53a615bccf42f30f8ed9ef7c8f7e3dc84c5eb61e"
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.])


def dequantize_nvfp4(packed, scales, global_scale):
    """Native row-major U8 E2M1 (low nibble first), block-16 E4M3FN scales."""
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[-1] % 8:
        raise ValueError("NVFP4 weights must be U8 [rows, logical_columns/2], block size 16")
    expected = (packed.shape[0], packed.shape[1] // 8)
    if scales.dtype != torch.float8_e4m3fn or tuple(scales.shape) != expected:
        raise ValueError(f"NVFP4 scales must be E4M3FN with shape {expected}")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise ValueError("NVFP4 global scale must be one float32 value")
    global_scale = global_scale.reshape(())
    block_scales = scales.float()
    if not bool(torch.isfinite(global_scale)) or not bool(global_scale > 0):
        raise ValueError("NVFP4 global scale must be finite and positive")
    if not bool(torch.isfinite(block_scales).all()) or bool((block_scales < 0).any()):
        raise ValueError("NVFP4 block scales must be finite and nonnegative")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    values = E2M1[codes.long()].reshape(packed.shape[0], -1, 16)
    # Match NVIDIA's rounding order: multiply the two scales before the values.
    out = values * (block_scales * global_scale).unsqueeze(-1)
    if not bool(torch.isfinite(out).all()):
        raise ValueError("NVFP4 packet dequantizes to nonfinite weights")
    return out.reshape(packed.shape[0], -1)


def attention_bias(relative, same, block_size, top_k):
    """Materialize NVIDIA LiLiCorr's query-minus-key slot bias once."""
    if relative.ndim != 2 or relative.shape[1] != 2 * block_size - 1 or same.shape != relative.shape[:1]:
        raise ValueError("LiLiCorr relative/same-slot bias geometry is inconsistent")
    slots = torch.arange(block_size - 1).repeat_interleave(top_k)
    distances = slots[:, None] - slots[None, :]
    return relative[:, distances + block_size - 1] + (distances == 0)[None].to(relative.dtype) * same[:, None, None]


def modelopt_reference(path):
    """Execute only the pinned, unmodified CPU dequantizer and its lookup helper."""
    text = Path(path).read_bytes()
    if hashlib.sha256(text).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Model Optimizer source does not match the pinned reference SHA256")
    tree = ast.parse(text)
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NVFP4QTensor")
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in ("get_e2m1_values", "dequantize")]
    original.bases = []
    original.body = ast.parse("e2m1_values_on_device = {}").body + methods
    table = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "e2m1_values" for target in node.targets))
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[table, original], type_ignores=[])), "NVIDIA NVFP4 reference", "exec"), scope)
    cls = scope["NVFP4QTensor"]

    def run(packed, scales, global_scale):
        ref = cls()
        ref._quantized_data = packed
        ref.metadata = {"shape": (packed.shape[0], packed.shape[1] * 2), "dtype": torch.float32}
        return ref.dequantize(dtype=torch.float32, scale=scales, double_scale=global_scale, block_sizes={-1: 16})
    return run


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor):
    digest = hashlib.sha256()
    raw = tensor.contiguous().view(torch.uint8).reshape(-1)
    for chunk in raw.split(8 * 1024 * 1024):
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def required_shapes(config):
    """Source geometry for the released DFlash2/LiLiCorr contract."""
    d = config["dflash_config"]
    h, hd = config["hidden_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * hd, config["num_key_value_heads"] * hd
    mlp, lh = config["intermediate_size"], d["lilicorr_hidden_size"]
    shapes = {"fc.weight": (h, h * len(d["target_layer_ids"])), "hidden_norm.weight": (h,), "norm.weight": (h,)}
    for i in range(config["num_hidden_layers"]):
        base = f"layers.{i}."
        shapes.update({base + name: shape for name, shape in {
            "input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
            "self_attn.q_norm.weight": (hd,), "self_attn.k_norm.weight": (hd,),
            "self_attn.q_proj.weight": (q, h), "self_attn.k_proj.weight": (kv, h),
            "self_attn.v_proj.weight": (kv, h), "self_attn.o_proj.weight": (h, q),
            "mlp.gate_proj.weight": (mlp, h), "mlp.up_proj.weight": (mlp, h), "mlp.down_proj.weight": (h, mlp),
        }.items()})
        kernel = d.get("conv_kernel_size", 0)
        if kernel:
            group = d["conv_group_size"]
            for name in ("attention_conv", "mlp_conv"):
                shapes[base + name + ".base_kernel"] = (2, kernel, h)
                shapes[base + name + ".kernel_projection.weight"] = (2 * kernel * h // group, h)
    if config.get("has_own_lm_head"):
        shapes["lm_head.weight"] = (config["draft_vocab_size"], h)
    for name in ("context_proj", "token_proj", "pass_hidden_proj"):
        shapes[f"lilicorr.{name}.weight"] = (lh, h)
        shapes[f"lilicorr.{name}.bias"] = (lh,)
    for name, width in (("feature_norm", 5), ("output_norm", lh), ("anchor_norm", lh)):
        shapes[f"lilicorr.{name}.weight"] = (width,)
    shapes["lilicorr.feature_norm.bias"] = (5,)
    for name, dims in (("feature_mlp.up_proj", (lh, 5)), ("feature_mlp.down_proj", (lh, lh)), ("factor_input_proj", (lh, lh * 3))):
        shapes[f"lilicorr.{name}.weight"] = dims
        shapes[f"lilicorr.{name}.bias"] = dims[:1]
    for name in ("in_head", "out_head", "anchor_out_head"):
        factor = d["lilicorr_factor_dim"]
        shapes[f"lilicorr.{name}.weight"] = (factor, lh)
        shapes[f"lilicorr.{name}.bias"] = (factor,)
    for i in range(d["lilicorr_num_layers"]):
        base = f"lilicorr.layers.{i}."
        for name in ("attn_norm", "mlp_norm"):
            shapes[base + name + ".weight"] = (lh,)
        mid = int(lh * d["lilicorr_mlp_ratio"])
        for name, dims in (("attn.in_proj_weight", (lh * 3, lh)), ("attn.out_proj.weight", (lh, lh)),
                           ("mlp.up_proj.weight", (mid, lh)), ("mlp.down_proj.weight", (lh, mid))):
            shapes[base + name] = dims
            bias_name = name.replace("in_proj_weight", "in_proj_bias").replace(".weight", ".bias")
            shapes[base + bias_name] = dims[:1]
    block, k, heads = config["block_size"], d["lilicorr_candidate_topk"], d["lilicorr_num_heads"]
    shapes.update({"lilicorr.slot_embedding": (1, 1, block - 1, 1, lh), "lilicorr.rank_embedding": (1, 1, 1, k, lh),
                   "lilicorr.relative_slot_bias": (heads, 2 * block - 1), "lilicorr.same_slot_bias": (heads,)})
    return shapes


def convert_checkpoint(source, destination, reference, source_info=None, expected_sha256=None, shard_bytes=512 * 1024 * 1024):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output must be a new directory outside the source checkpoint")
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if quant.get("quant_algo") != "W4A16_NVFP4" or quant.get("group_size") != 16:
        raise ValueError("Expected native W4A16_NVFP4 block-16 source configuration")
    if config.get("architectures") != ["LiLiCorrDraftModel"] or config["dflash_config"].get("projector_type") != "lilicorr":
        raise ValueError("Expected the native LiLiCorr DFlash checkpoint contract")
    source_sha256 = sha256_file(source / "model.safetensors")
    if expected_sha256 and source_sha256 != expected_sha256:
        raise ValueError("Source checkpoint does not match the pinned NVIDIA revision SHA256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + "-", dir=destination.parent))
    manifest = {"source_safetensors_sha256": source_sha256,
                "modelopt_revision": REFERENCE_REVISION, "modelopt_source_sha256": REFERENCE_SHA256,
                "output_dtype": "bfloat16", "quantized_matrices": 0, "tensors": {}, "layout_changes": []}
    if source_info:
        manifest.update(source_info)
    pending, pending_bytes, shard_paths, weight_map = {}, 0, [], {}

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        path = temporary / f"part-{len(shard_paths):05}.safetensors"
        save_file(pending, str(path))
        shard_paths.append((path, list(pending)))
        pending, pending_bytes = {}, 0

    def add(name, value, operation, sources):
        nonlocal pending_bytes
        if name in manifest["tensors"]:
            raise ValueError(f"Output tensor collision: {name}")
        value = value.to(torch.bfloat16).contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite output tensor: {name}")
        size = value.numel() * value.element_size()
        if pending_bytes + size > shard_bytes:
            flush()
        pending[name] = value
        pending_bytes += size
        manifest["tensors"][name] = {"shape": list(value.shape), "sha256": tensor_digest(value), "operation": operation, "sources": sources}
        if pending_bytes >= shard_bytes:
            flush()

    try:
        with safe_open(str(source / "model.safetensors"), framework="pt") as weights:
            keys, consumed = set(weights.keys()), set()
            for name, expected in required_shapes(config).items():
                if name not in keys:
                    raise ValueError(f"Missing required source tensor: {name}")
                shape = tuple(weights.get_slice(name).get_shape())
                if name.endswith(".weight") and name.removesuffix(".weight") + ".weight_scale" in keys:
                    shape = shape[:-1] + (shape[-1] * 2,)
                if shape != expected:
                    raise ValueError(f"Invalid source tensor shape: {name} is {shape}, expected {expected}")
            for name in sorted(keys):
                if name in consumed or name.endswith((".weight_scale", ".weight_scale_2")) or name in ("lilicorr.relative_slot_bias", "lilicorr.same_slot_bias"):
                    continue
                value = weights.get_tensor(name)
                target = name.replace(".attn.in_proj_weight", ".attn.in_proj.weight").replace(".attn.in_proj_bias", ".attn.in_proj.bias")
                sources, operation = [name], "copy_bf16"
                if value.dtype == torch.uint8:
                    if not name.endswith(".weight"):
                        raise ValueError(f"Unexpected U8 tensor: {name}")
                    base = name.removesuffix(".weight")
                    sources += [base + ".weight_scale", base + ".weight_scale_2"]
                    if not all(key in keys for key in sources):
                        raise ValueError(f"Incomplete NVFP4 packet: {name}")
                    scales, global_scale = (weights.get_tensor(key) for key in sources[1:])
                    dense = torch.empty((value.shape[0], value.shape[1] * 2), dtype=torch.bfloat16)
                    for start in range(0, value.shape[0], 1024):
                        block, scale = value[start:start + 1024], scales[start:start + 1024]
                        decoded = dequantize_nvfp4(block, scale, global_scale)
                        expected = reference(block, scale, global_scale)
                        if not torch.equal(decoded.view(torch.int32), expected.view(torch.int32)):
                            raise ValueError(f"NVIDIA fp32 dequantization mismatch: {name}")
                        dense[start:start + 1024] = decoded.to(torch.bfloat16)
                    value, operation = dense, "nvfp4_to_bf16_verified_exact"
                    manifest["quantized_matrices"] += 1
                    print(f"verified NVFP4 {name}", flush=True)
                elif value.dtype not in (torch.bfloat16, torch.float32, torch.float16):
                    raise ValueError(f"Unexpected source dtype: {name} ({value.dtype})")
                if name in ("lilicorr.slot_embedding", "lilicorr.rank_embedding"):
                    value = value.reshape(-1, config["dflash_config"]["lilicorr_hidden_size"])
                    operation = "flatten_embedding"
                if target != name or operation == "flatten_embedding":
                    manifest["layout_changes"].append({"source": name, "target": target, "operation": operation})
                add(target, value, operation, sources)
                consumed.update(sources)
            relative_key, same_key = "lilicorr.relative_slot_bias", "lilicorr.same_slot_bias"
            bias = attention_bias(weights.get_tensor(relative_key), weights.get_tensor(same_key), config["block_size"], config["dflash_config"]["lilicorr_candidate_topk"])
            add("lilicorr.attention_bias", bias, "materialize_relative_and_same_slot_bias", [relative_key, same_key])
            manifest["layout_changes"].append({"sources": [relative_key, same_key], "target": "lilicorr.attention_bias", "operation": "materialize_attention_bias"})
            consumed.update((relative_key, same_key))
            if consumed != keys:
                raise ValueError(f"Unconsumed source tensors: {sorted(keys - consumed)}")
            manifest["source_tensor_count"] = len(keys)
        flush()
        total_bytes = 0
        for i, (path, names) in enumerate(shard_paths, 1):
            final = f"model-{i:05}-of-{len(shard_paths):05}.safetensors"
            path.rename(temporary / final)
            with safe_open(str(temporary / final), framework="pt") as check:
                if set(check.keys()) != set(names):
                    raise ValueError("Output shard tensor inventory mismatch")
                for name in names:
                    value = check.get_tensor(name)
                    if value.dtype != torch.bfloat16 or tensor_digest(value) != manifest["tensors"][name]["sha256"]:
                        raise ValueError(f"Serialized output verification failed: {name}")
                    weight_map[name] = final
                    total_bytes += value.numel() * value.element_size()
        config.pop("quantization_config")
        config["torch_dtype"] = "bfloat16"
        (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (temporary / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2) + "\n")
        manifest["output_tensor_count"], manifest["output_weight_bytes"] = len(weight_map), total_bytes
        (temporary / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if (source / "README.md").exists():
            shutil.copyfile(source / "README.md", temporary / "SOURCE_README.md")
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    raw_config = (args.source / "config.json").read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw_config)).encode() + b"\0" + raw_config).hexdigest()
    if blob != SOURCE_CONFIG_GIT_BLOB:
        raise ValueError("Source config does not match the pinned NVIDIA revision")
    manifest = convert_checkpoint(args.source, args.out, modelopt_reference(args.reference),
                                  source_info={"source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION},
                                  expected_sha256=SOURCE_MODEL_SHA256)
    print(json.dumps({key: manifest[key] for key in ("source_revision", "quantized_matrices", "source_tensor_count", "output_tensor_count", "output_weight_bytes")}, indent=2))


if __name__ == "__main__":
    main()
