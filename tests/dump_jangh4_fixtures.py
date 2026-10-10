"""vMLX reference fixtures for the JANGH4 load path, from vMLX's own qwen4_exp code (v1.6.77, MLX 0.32.3) on the real
bundle, reading only the rows and layer it needs. `<out>/ple/` holds the n-gram PLE rows: a token sequence replayed
chunk by chunk through PLELayer._embed over the file-backed table (prefetch ticket at decode widths), the row ids its
hasher produced, and table rows at every shard edge, the padding rows and random rows (FileBackedQuantizedNGramTable
.gather_mlx). `<out>/moe-block/` holds layer 0's SparseMoeBlock with TQSwitchGLU installed as vMLX's loader installs
it, run in mlx-serve's dtypes (bf16 activations, the shared expert's F16 scales narrowed to BF16), plus vMLX's routing
and its routed sum. Regenerate with `PYTHONPATH=<vmlx checkout> python tests/dump_jangh4_fixtures.py --bundle <JANGH4
dir> --out <dir>`; then `JANGH4_PLE_FIXTURE=<dir>/ple`, and `QWEN4_JANGH_TEST_MODEL=<bundle>
JANGH4_MOE_BLOCK_FIXTURE=<dir>/moe-block`, run the gated tests (`zig build test-build -Dtest-filter=jangh`).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

ARGS = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ARGS.add_argument("--bundle", required=True, help="the JANGH4 bundle (JANGQ-AI/Qwen3.8-Flash-Next-JANGH4)")
ARGS.add_argument("--out", required=True, help="fixture directory to write (ple/ and moe-block/ inside)")
OPTS = ARGS.parse_args()
# Read when vmlx_engine is imported: decode-width reads take the prefetch ticket with pread; never warm the table.
os.environ["VMLX_QWEN4_PLE_PREFETCH"] = "1"
os.environ["VMLX_QWEN4_PLE_PREFETCH_PREAD"] = "1"
os.environ["VMLX_QWEN4_PLE_WARM"] = "0"

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from vmlx_engine.jangh import kernels as K  # noqa: E402
from vmlx_engine.jangh.runtime_identity import QWEN4_PREFILL_FUSED, qwen4_decode_max_tokens  # noqa: E402
from vmlx_engine.jangh.switch import TQSwitchGLU  # noqa: E402
from vmlx_engine.models.qwen4_exp import loader  # noqa: E402
from vmlx_engine.models.qwen4_exp.language import PLELayer, Qwen4ExpTextArgs, SparseMoeBlock  # noqa: E402
from vmlx_engine.models.qwen4_exp.table_reader import FileBackedQuantizedNGramTable  # noqa: E402

BUNDLE = Path(OPTS.bundle).resolve()
OUT = Path(OPTS.out).resolve()

TEXT = [
    "The quick brown fox jumps over the lazy dog. Hyper-connections widen the residual stream into four "
    "parallel copies, and the per-layer embedding reads hashed bigram and trigram rows from a 51B table.",
    "def ngram_rows(history, multipliers, primes):\n    mixed = history[-1] * multipliers[0]\n"
    "    for k, tok in enumerate(reversed(history[:-1]), 1):\n        mixed ^= tok * multipliers[k]\n"
    "    return [mixed % p for p in primes]\n",
    "长上下文推理需要在固态硬盘上按行读取量化的 n-gram 表，每个 token 读取十六行。",
    "1234567890 !@#$%^&*() <tool_call>{\"name\": \"read\", \"arguments\": {\"path\": \"/tmp/x\"}}</tool_call>",
    "Le cache de préfixe garde l'historique du n-gramme : deux jetons suffisent pour reprendre un fragment. "
    "Αυτό ισχύει και για τα ελληνικά. Это верно и для русского текста. 🙂🚀 ∑ x² dx",
]


def safetensors_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        return n, json.loads(handle.read(n))


def read_small_tensor(weight_map: dict, name: str) -> mx.array:
    path = BUNDLE / weight_map[name]
    n, header = safetensors_header(path)
    info = header[name]
    assert info["dtype"] == "I64", info
    start, end = info["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(8 + n + start)
        raw = handle.read(end - start)
    return mx.array(np.frombuffer(raw, dtype="<i8").reshape(info["shape"]))


def runtime_dtype_from_headers(weight_map: dict):
    """vMLX's own dtype resolver over stand-ins carrying every non-table tensor's checkpoint dtype."""
    dtypes = {"F16": mx.float16, "BF16": mx.bfloat16, "F32": mx.float32, "U32": mx.uint32, "I64": mx.int64,
              "I32": mx.int32, "U8": mx.uint8}
    stand_ins = {}
    for fname in sorted(set(weight_map.values())):
        _, header = safetensors_header(BUNDLE / fname)
        for key, info in header.items():
            if key == "__metadata__" or loader._classify_ple_tensor(key) is not None:
                continue
            stand_ins[key] = mx.zeros((1,), dtype=dtypes[info["dtype"]])
    return loader._resolve_jang_runtime_compute_dtype(stand_ins)


def token_sequence(args: Qwen4ExpTextArgs, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    tok = Tokenizer.from_file(str(BUNDLE / "tokenizer.json"))
    eos = args.eos_token_id
    parts = [tok.encode(TEXT[0]).ids, [eos], tok.encode(TEXT[1]).ids, [eos, eos], tok.encode(TEXT[2]).ids]
    real = [t for p in parts for t in p]
    rand = rng.integers(0, args.vocab_size, 300).tolist()
    edges = [0, args.vocab_size - 1, eos, 248045, 248046, 0, 0, eos, args.vocab_size - 1, 1]
    tail = tok.encode(TEXT[3]).ids
    seq = np.array(real + rand[:150] + edges + rand[150:] + [eos] + tail + tok.encode(TEXT[4]).ids, dtype=np.int32)
    # Prefill chunks (> 64 tokens, plain host gather), decode/verify widths (<= 64, prefetch ticket), an eos on a
    # chunk boundary (the 3-token chunk ends right before the [eos, eos] pair) and a long final prefill.
    first = len(tok.encode(TEXT[0]).ids) + 1 + len(tok.encode(TEXT[1]).ids)
    chunks = [first - 5, 1, 1, 3, 2, 64, 65, 1]
    chunks.append(len(seq) - sum(chunks))
    assert all(c > 0 for c in chunks) and sum(chunks) == len(seq)
    return seq, chunks


def boundary_rows(per: int, n_shards: int, total_vocab: int, padded: int, rng: np.random.Generator) -> np.ndarray:
    rows = []
    for s in range(n_shards):
        rows += [s * per, s * per + 1, (s + 1) * per - 2, (s + 1) * per - 1]
    rows += [total_vocab - 1, total_vocab, padded - 2, padded - 1]  # the last hashed row, the padding rows
    rows += rng.integers(0, padded, 512).tolist()
    rows += [rows[7], rows[3], rows[-1]]  # repeats inside one call
    return np.array(rows, dtype=np.int64)


def ple_fixture(config: dict, bit_map: dict, weight_map: dict) -> None:
    out = OUT / "ple"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261009)
    args = Qwen4ExpTextArgs.from_dict(config)
    ple = PLELayer(args, args.ple_layer_ids.index(2))
    n_shards = int(args.split_ngram_parts)
    key_format = loader._resolve_ple_module_key_format(weight_map, n_shards)
    head_dim = args.ple_embed_dim // ple.hasher.ngram_heads
    table = FileBackedQuantizedNGramTable(BUNDLE, key_format, n_shards, expected_head_dim=head_dim, bit_map=bit_map)
    runtime_dtype = runtime_dtype_from_headers(weight_map)
    ple.ngram_embedding.set_file_backed(table, output_dtype=runtime_dtype)

    buffers = {}
    for name in weight_map:
        cls = loader._classify_ple_tensor(name)
        if cls is not None and cls[0] == "buffer":
            buffers[cls[1]] = read_small_tensor(weight_map, name)
    fake_model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(
        layers=[None, SimpleNamespace(ple=ple)])))
    loader._validate_ple_hash_buffers(fake_model, buffers)

    seq, chunks = token_sequence(args, rng)
    # The served path: chunk by chunk through the PLE cache slot, the prefetch ticket at decode widths.
    cache = [None, None, None, None]
    embs, rows_all, prefetched, pos = [], [], 0, 0
    for c in chunks:
        ids = mx.array(seq[None, pos:pos + c])
        prev = None if cache[2] is None else np.asarray(cache[2], dtype=np.int64)
        rows_all.append(ple.hasher.hash_tokens(np.asarray(ids, dtype=np.int64), prev)[0])
        # Qwen4ExpTextModel.__call__ prepares a ticket only for 0 < B*S <= _PLE_PREFETCH_MAX_TOKENS (64).
        ticket = ple.prepare_read(ids, cache) if c <= 64 else None
        prefetched += ticket is not None
        emb = ple._embed(ids, cache, prepared=ticket)
        mx.eval(emb)
        embs.append(emb[0])
        pos += c
    emb = mx.concatenate(embs, axis=0)
    rows = np.concatenate(rows_all, axis=0)
    mx.eval(emb)
    assert table.prefetch_stats["consumed"] == prefetched == sum(1 for c in chunks if c <= 64), table.prefetch_stats
    # The reference agrees with itself: one fresh pass over the whole sequence gives the same bits.
    whole = ple._embed(mx.array(seq[None]), [None, None, None, None])[0]
    mx.eval(whole)
    assert np.array_equal(np.array(whole).view(np.uint16), np.array(emb).view(np.uint16))
    assert emb.dtype == runtime_dtype == mx.float16 and emb.shape == (len(seq), args.ple_embed_dim)

    per = table.per
    probe = boundary_rows(per, n_shards, ple.hasher.total_vocab_size, ple.hasher.padded_vocab_size, rng)
    probe_vals = table.gather_mlx(probe)
    mx.eval(probe_vals)
    assert probe_vals.dtype == mx.float16 and probe_vals.shape == (len(probe), head_dim)
    mx.save_safetensors(str(out / "ple_fixture.safetensors"), {
        "tokens": mx.array(seq[None, :]),
        "chunks": mx.array(np.array(chunks, dtype=np.int32)[None, :]),
        "row_ids": mx.array(rows.astype(np.int64)),
        "emb": emb,
        "probe_rows": mx.array(probe[None, :]),
        "probe_vals": probe_vals,
    })
    manifest = {"bundle": str(BUNDLE), "fixture": "ple_fixture.safetensors", "mlx": mx.__version__,
                "key_format": key_format, "n_shards": n_shards, "rows_per_shard": int(per),
                "total_rows": int(table.total_rows), "head_dim": int(head_dim), "tokens": int(len(seq)),
                "chunks": chunks, "probe_rows": int(len(probe)), "hash_buffers": sorted(buffers)}
    (out / "ple_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest), flush=True)
    table.close()


def layer_tensors(weight_map: dict, layer: int) -> dict[str, mx.array]:
    dense = f"language_model.layers.{layer}.mlp."
    banks = f"model.layers.{layer}.mlp.switch_mlp."
    shards: dict[str, list[str]] = {}
    for n in weight_map:
        if n.startswith(dense) or n.startswith(banks):
            shards.setdefault(weight_map[n], []).append(n)
    out = {}
    for shard, names in shards.items():
        loaded = mx.load(str(BUNDLE / shard))
        for n in names:
            out[n[len(dense):] if n.startswith(dense) else "switch_mlp." + n[len(banks):]] = loaded[n]
    return out


def quantized(w: dict[str, mx.array], name: str) -> nn.QuantizedLinear:
    weight = w[f"{name}.weight"]
    out_dims, packed = weight.shape
    lin = nn.QuantizedLinear(packed * 32 // 8, out_dims, bias=False, group_size=64, bits=8)
    lin.weight = weight
    lin.scales = w[f"{name}.scales"].astype(mx.bfloat16)
    lin.biases = w[f"{name}.biases"].astype(mx.bfloat16)
    return lin


def moe_block_fixture(config: dict, raw: dict, weight_map: dict, layer: int = 0) -> None:
    """SparseMoeBlock.__call__ (precise softmax router, argpartition top-k, renormalized scores, TQSwitchGLU.routed,
    sigmoid-gated shared expert) at 1 (one decode row), 7, 97 (the first sorted-GEMM width) and 300 tokens."""
    out = OUT / "moe-block"
    out.mkdir(parents=True, exist_ok=True)
    args = Qwen4ExpTextArgs.from_dict(config)
    w = layer_tensors(weight_map, layer)
    entry = lambda p: raw["quantization"][f"model.layers.{layer}.mlp.switch_mlp.{p}_proj"]  # noqa: E731
    limit = float(raw.get("text_config", raw).get("swiglu_limit", 0.0) or 0.0)
    blk = SparseMoeBlock(args)
    sw = TQSwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts, entry("gate")["bits"],
                     entry("down")["bits"], limit, rotation_gate_up=entry("gate")["rotation"],
                     rotation_down=entry("down")["rotation"])
    sw.is_jangh = sw.is_jangtq2 = True
    sw.decode_max_tokens = qwen4_decode_max_tokens()
    if QWEN4_PREFILL_FUSED == "1":
        sw.use_weighted_unsort = True
        for lin in (sw.gate_proj, sw.up_proj, sw.down_proj):
            lin.use_h32_rows = True
    for p, lin in (("gate", sw.gate_proj), ("up", sw.up_proj), ("down", sw.down_proj)):
        lin.tq2_packed = w[f"switch_mlp.{p}_proj.tq2_packed"]
        lin.tq2_scales = w[f"switch_mlp.{p}_proj.tq2_scales"]
    blk.switch_mlp = sw
    blk.gate.weight = w["gate.weight"].astype(mx.bfloat16)
    blk.shared_expert_gate.weight = w["shared_expert_gate.weight"].astype(mx.bfloat16)
    se = blk.shared_expert
    se.gate_proj = quantized(w, "shared_expert.gate_proj")
    se.up_proj = quantized(w, "shared_expert.up_proj")
    se.down_proj = quantized(w, "shared_expert.down_proj")
    assert se.prepare_runtime(), "vMLX declined its joined shared gate/up projection"

    rng = np.random.default_rng(20261009)
    # The sorted GEMM's arm (NAX or steel) decides the prefill cases' bits: the test needs the same GPU class.
    manifest = {"bundle": str(BUNDLE), "layer": layer, "mlx": mx.__version__, "nax": bool(K.nax_available()), "cases": []}
    for t in (1, 7, 97, 300):
        # A stand-in for the mixed, hc-normed stream: unit-variance rows times the layer's ~1.37 norm mean.
        x = mx.array(rng.standard_normal((1, t, args.hidden_size)).astype(np.float32) * 1.37).astype(mx.bfloat16)
        gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-blk.top_k, axis=-1)[..., -blk.top_k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if blk.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        case = {"x": x, "gates": gates, "inds": inds.astype(mx.uint32), "scores": scores,
                "routed": sw.routed(x, inds.astype(mx.uint32), scores), "out": blk(x)}
        mx.eval(case)
        assert case["out"].dtype == mx.bfloat16
        name = f"L{layer}_S{t}.safetensors"
        mx.save_safetensors(str(out / name), case)
        manifest["cases"].append({"file": name, "tokens": t})
        print(name, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def main() -> None:
    if mx.__version__ != "0.32.3":
        raise SystemExit(f"expected MLX 0.32.3 (the mlx-serve pin), got {mx.__version__}")
    config, _affine, bit_map, sanitized = loader._load_runtime_config(BUNDLE)
    assert sanitized and bit_map is not None
    weight_map = json.loads((BUNDLE / "model.safetensors.index.json").read_text())["weight_map"]
    ple_fixture(config, bit_map, weight_map)
    moe_block_fixture(config, json.loads((BUNDLE / "config.json").read_text()), weight_map)


if __name__ == "__main__":
    sys.exit(main())
