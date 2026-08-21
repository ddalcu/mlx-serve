#!/usr/bin/env python3
"""Bring a published ddalcu Qwen3.8 pack up to the house standard, in place-ish
(reads one pack, writes a new directory — the source is never modified).

Three transforms, each independently switchable, each a no-op when the pack is
already in the target state:

  1. `--quant-embed`   dense bf16 `embed_tokens.weight` -> affine quantized at
                       the pack's own width/group. Measured cost: nothing
                       resolvable (ddalcu-8bit 95.5%/KL 0.0136 vs the
                       byte-identical-except-this-tensor mlxc-8bit
                       96.3%/0.0158; same picture at 4-bit). Measured saving:
                       1.8 GB at 4-bit, 1.2 GB at 8-bit. The engine reads the
                       table through `gatherQuantizedRows`, so this is a size
                       decision, not a correctness one.

  2. `--fold-mtp-norms` the 7 MTP-head RMSNorm gammas from Qwen's zero-centered
                       (delta) convention to the MLX `+1` convention every other
                       vendor publishes. Detection is per tensor and by VALUE
                       (a delta gamma carries negatives, a folded one is
                       strictly positive), so re-running is a no-op.

  3. `--dwq <dir>`     graft a DWQ-distilled pack's LEARNED tensors (`.scales`,
                       `.biases`, and the un-quantized `conv1d.weight`) onto our
                       codes. Only legal because our packed codes are
                       byte-identical to the pack the DWQ run started from —
                       which this script VERIFIES per tensor and refuses to
                       graft where they differ, rather than trusting the claim.

Usage:
  python3 tests/restandardize_qwen38_pack.py --src <pack> --dst <out> \
      [--quant-embed] [--fold-mtp-norms] [--dwq <dwq-pack>] [--verify]
"""

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import bf16_to_f32, write_safetensors_raw  # noqa: E402

SHARD_BYTES = 5 * 1024**3

# The head's RMSNorm gammas — the only tensors the fold touches.
MTP_NORM_SUFFIXES = (
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)

# What a DWQ run learns: the dequantization parameters and the un-quantized
# conv path. NEVER the packed codes (grafting those would just be adopting the
# other pack wholesale) and never a norm (a norm difference means the two packs
# disagree about something we have not verified).
DWQ_GRAFT_SUFFIXES = (".scales", ".biases", "conv1d.weight")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def weight_map(pack: Path):
    idx = pack / "model.safetensors.index.json"
    if idx.exists():
        return json.load(open(idx))["weight_map"]
    out = {}
    for f in sorted(p.name for p in pack.glob("*.safetensors")):
        h, _ = read_header(pack / f)
        for k in h:
            if k != "__metadata__":
                out[k] = f
    return out


class Reader:
    """Raw (dtype, shape, bytes) reads out of a sharded pack, one open file."""

    def __init__(self, pack: Path, wmap: dict):
        self.pack, self.wmap, self._h = pack, wmap, {}

    def _hdr(self, shard):
        if shard not in self._h:
            self._h[shard] = read_header(self.pack / shard)
        return self._h[shard]

    def raw(self, key):
        shard = self.wmap[key]
        h, base = self._hdr(shard)
        info = h[key]
        s, e = info["data_offsets"]
        with open(self.pack / shard, "rb") as f:
            f.seek(base + s)
            return info["dtype"], tuple(info["shape"]), f.read(e - s)


def quantize_bf16_rows(raw_bf16: bytes, shape, bits, group, chunk_rows=16384):
    """Affine-quantize a bf16 table straight from its stored bytes, in row
    chunks (rows are independent — the group divides a row). Byte-identical to
    what mlx_lm's convert produces: verified against
    mlx-community/Qwen3.8-27B-4bit's embed_tokens. Going through an f32 round
    trip does NOT reproduce it, so the raw bytes stay raw."""
    import mlx.core as mx
    rows, cols = shape
    qs, ss, bs = [], [], []
    flat = np.frombuffer(raw_bf16, dtype=np.uint16)
    for r0 in range(0, rows, chunk_rows):
        r1 = min(r0 + chunk_rows, rows)
        blk = mx.array(flat[r0 * cols:r1 * cols].copy()).view(mx.bfloat16).reshape(r1 - r0, cols)
        q, s, b = mx.quantize(blk, group_size=group, bits=bits)
        mx.eval(q, s, b)
        qs.append(np.array(q, copy=False).tobytes())
        ss.append(np.array(s.view(mx.uint16), copy=False).tobytes())
        bs.append(np.array(b.view(mx.uint16), copy=False).tobytes())
        del blk, q, s, b
    qcols = cols * bits // 32
    gcols = cols // group
    return (("U32", (rows, qcols), b"".join(qs)),
            ("BF16", (rows, gcols), b"".join(ss)),
            ("BF16", (rows, gcols), b"".join(bs)))


def is_delta_norm(raw_bf16: bytes) -> bool:
    """A delta-encoded gamma carries negatives; a folded one is positive by
    construction. Same evidence the engine's loader reads (mtp.zig)."""
    v = np.frombuffer(raw_bf16, dtype=np.uint16)
    return bool((v >> 15).any())


def fold_plus_one(raw_bf16: bytes) -> bytes:
    """+1 in f32, back to bf16 — bit-for-bit what mlx-serve's foldNormPlusOne
    and every published folded pack produce (verified against
    mlx-community/Qwen3.8-27B-MTP-bf16)."""
    f = bf16_to_f32(np.frombuffer(raw_bf16, dtype=np.uint16)) + 1.0
    import mlx.core as mx
    a = mx.array(f).astype(mx.bfloat16)
    mx.eval(a)
    return np.array(a.view(mx.uint16), copy=False).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--quant-embed", action="store_true")
    ap.add_argument("--embed-bits", type=int, default=0, help="0 = the pack's own declared width")
    ap.add_argument("--embed-group", type=int, default=0, help="0 = the pack's own declared group")
    ap.add_argument("--fold-mtp-norms", action="store_true")
    ap.add_argument("--dwq", default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    wmap = weight_map(src)
    rd = Reader(src, wmap)
    cfg = json.load(open(src / "config.json"))
    q = cfg.get("quantization", {}) or {}
    bits = args.embed_bits or int(q.get("bits", 4))
    group = args.embed_group or int(q.get("group_size", 64))

    dwq_map = dwq_rd = None
    if args.dwq:
        dwq = Path(args.dwq)
        dwq_map = weight_map(dwq)
        dwq_rd = Reader(dwq, dwq_map)

    stats = {"embed": 0, "folded": 0, "grafted": 0, "graft_refused": 0, "copied": 0}
    out_tensors, shard_files, cur, cur_bytes = {}, [], {}, 0

    def flush():
        nonlocal cur, cur_bytes
        if not cur:
            return
        name = f"model-{len(shard_files) + 1:05d}-of-XXXXX.safetensors"
        shard_files.append((name, list(cur.keys())))
        write_safetensors_raw(str(dst / name), cur, {"format": "mlx"})
        cur, cur_bytes = {}, 0

    def emit(key, triple):
        nonlocal cur_bytes
        cur[key] = triple
        cur_bytes += len(triple[2])
        if cur_bytes >= SHARD_BYTES:
            flush()

    embed_key = next((k for k in wmap if k.endswith("embed_tokens.weight")), None)

    for key in wmap:
        dt, shape, raw = rd.raw(key)

        # 1. embedding
        if args.quant_embed and key == embed_key and dt == "BF16":
            wq, sc, bi = quantize_bf16_rows(raw, shape, bits, group)
            # A DWQ pack's learned embed params are only ours to take when its
            # codes came out identical to the ones we just produced.
            if dwq_map is not None and key in dwq_map and dwq_rd.raw(key)[2] == wq[2]:
                sk, bk = key.replace(".weight", ".scales"), key.replace(".weight", ".biases")
                if sk in dwq_map and bk in dwq_map:
                    sc, bi = dwq_rd.raw(sk), dwq_rd.raw(bk)
                    stats["grafted"] += 2
            elif dwq_map is not None and key in dwq_map:
                print("  embed codes differ from the DWQ pack — its embed scales NOT grafted")
            emit(key, wq)
            emit(key.replace(".weight", ".scales"), sc)
            emit(key.replace(".weight", ".biases"), bi)
            stats["embed"] = 1
            continue

        # 2. head norms
        if (args.fold_mtp_norms and ".mtp." in key and dt == "BF16"
                and any(key.endswith(s) for s in MTP_NORM_SUFFIXES)
                and is_delta_norm(raw)):
            emit(key, (dt, shape, fold_plus_one(raw)))
            stats["folded"] += 1
            continue

        # 3. DWQ graft — only where OUR codes match the DWQ pack's codes.
        if dwq_map is not None and any(key.endswith(s) for s in DWQ_GRAFT_SUFFIXES) and key in dwq_map:
            base_key = key
            for s in (".scales", ".biases"):
                if key.endswith(s):
                    base_key = key[: -len(s)] + ".weight"
            codes_match = True
            if base_key != key:
                if base_key not in wmap or base_key not in dwq_map:
                    codes_match = False
                else:
                    codes_match = rd.raw(base_key)[2] == dwq_rd.raw(base_key)[2]
            d_dt, d_shape, d_raw = dwq_rd.raw(key)
            if codes_match and (d_dt, d_shape) == (dt, shape):
                emit(key, (d_dt, d_shape, d_raw))
                stats["grafted"] += 1
                continue
            stats["graft_refused"] += 1

        emit(key, (dt, shape, raw))
        stats["copied"] += 1

    flush()

    # Rename shards to the final of-N and write the index.
    total = len(shard_files)
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for i, (name, keys) in enumerate(shard_files, 1):
        final = f"model-{i:05d}-of-{total:05d}.safetensors"
        os.replace(dst / name, dst / final)
        index["metadata"]["total_size"] += os.path.getsize(dst / final)
        for k in keys:
            index["weight_map"][k] = final
    json.dump(index, open(dst / "model.safetensors.index.json", "w"), indent=1)

    # config: declare the embedding's own geometry when it differs from the base
    if stats["embed"] and (bits, group) != (int(q.get("bits", 0)), int(q.get("group_size", 0))):
        cfg.setdefault("quantization", {})[embed_key.rsplit(".", 1)[0]] = {
            "group_size": group, "bits": bits}
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)

    for f in src.iterdir():
        if f.is_file() and f.suffix not in (".safetensors",) and f.name not in (
                "config.json", "model.safetensors.index.json"):
            shutil.copy2(f, dst / f.name)

    print(f"  embed quantized: {stats['embed']} ({bits}b/g{group})   norms folded: {stats['folded']}   "
          f"dwq grafted: {stats['grafted']} (refused {stats['graft_refused']})   copied: {stats['copied']}")
    print(f"  shards: {total}   size: {index['metadata']['total_size'] / 1e9:.2f} GB")

    if args.verify:
        verify(dst, src)


def verify(dst: Path, src: Path):
    """Post-conditions that a wrong run would violate: the head's norms are
    strictly positive, the embedding solves to a sane affine geometry, and no
    tensor was lost."""
    import mlx.core as mx
    wmap, swmap = weight_map(dst), weight_map(src)
    rd = Reader(dst, wmap)
    missing = [k for k in swmap if k not in wmap and not k.endswith("embed_tokens.weight")]
    assert not missing, f"lost tensors: {missing[:5]}"
    for k in wmap:
        if ".mtp." in k and any(k.endswith(s) for s in MTP_NORM_SUFFIXES):
            dt, shape, raw = rd.raw(k)
            assert not is_delta_norm(raw), f"{k} still delta-encoded"
    ek = next(k for k in wmap if k.endswith("embed_tokens.weight"))
    dt, shape, _ = rd.raw(ek)
    if dt == "U32":
        sk = ek.replace(".weight", ".scales")
        _, sshape, _ = rd.raw(sk)
        hidden = json.load(open(dst / "config.json")).get(
            "text_config", {}).get("hidden_size") or 5120
        b = shape[1] * 32 // hidden
        g = hidden // sshape[1]
        assert b in (2, 3, 4, 5, 6, 8) and g in (32, 64, 128), f"embed geometry {b}b/g{g}"
        print(f"  verify OK — embed {b}b/g{g}, head norms folded, {len(wmap)} tensors")
    else:
        print(f"  verify OK — embed left dense, head norms folded, {len(wmap)} tensors")


if __name__ == "__main__":
    main()
