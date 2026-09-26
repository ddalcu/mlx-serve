#!/usr/bin/env python3
"""Graft the Qwen3-VL vision tower into a towerless Qwen-Image-2.1 MLX pack.

The published ddalcu packs (`Qwen-Image-2.1-MLX-Serve-{8,4}bit`) shipped without
`model.visual.*`, so they serve text-to-image only. This tool rebuilds
`text_encoder/` from the original `Qwen/Qwen-Image-2.1` checkpoint — language
tensors AND the tower under `tests/convert_qwen_image21_weights.py`'s exact
rules — and swaps the new directory into the pack, giving it the edit path
without a full re-conversion. All quant logic is IMPORTED from that converter:
zero duplicated rules.

    python3 tests/graft_qwen_image21_vision.py --pack <pack dir> --src <Qwen-Image-2.1 dir> [--bits N]
    python3 tests/graft_qwen_image21_vision.py --self-test

`--bits` overrides the te_bits read from the pack's root `config.json`
quantization block; a mismatch against the pack's language tensors is legal
(the engine resolves quant per weight) but WARNs. The new `text_encoder/` is
built in `text_encoder.graft/` and atomically swapped in; the old one moves to
`text_encoder.bak-<unix ts>/` (path printed). `transformer/`, `vae/`,
`processor/` and `config.json` are untouched. The Zig loader globs
`{pack}/text_encoder/*.safetensors`, so the converter's no-index shard scheme
loads unchanged.

Language tensors are re-quantized from the same source at the same bits rather
than copied from the pack, so a pack quantized by any converter build rebuilds
coherently; against a pack this converter produced, they come out byte-identical.
The tower half is deterministic (mx.quantize, group 64, fixed params).
Apache-2.0 upstream.
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "tests"))
import convert_qwen_image21_weights as conv  # noqa: E402  (the one rules source)


def shard_keys(path):
    """Key set of one safetensors file, read straight from its header."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k for k in header if k != "__metadata__"}


def all_pack_keys(te_dir):
    keys = set()
    for shard in sorted(glob.glob(os.path.join(te_dir, "*.safetensors"))):
        keys |= shard_keys(shard)
    return keys


def source_tower_weight_keys(src):
    """The source's model.visual.*.weight key set, for the completeness check."""
    keys = set()
    for shard in sorted(glob.glob(os.path.join(src, "text_encoder", "*.safetensors"))):
        keys |= {k for k in shard_keys(shard) if k.startswith("model.visual.") and k.endswith(".weight")}
    return keys


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_language_bits(pack):
    """Bits the pack's language linears were quantized at, from packed geometry:
    in_features = scales_cols * group_size, bits = packed_u32_cols * 32 / in_features."""
    import mlx.core as mx
    with open(os.path.join(pack, "config.json")) as f:
        q = (json.load(f).get("quantization") or {})
        group = int(q.get("group_size", 64))
    for shard in sorted(glob.glob(os.path.join(pack, "text_encoder", "*.safetensors"))):
        loaded = mx.load(shard)
        for name, t in loaded.items():
            if name.startswith("model.language_model.layers.") and name.endswith(".scales"):
                packed = loaded.get(name[: -len(".scales")] + ".weight")
                if packed is None:
                    continue
                in_features = int(t.shape[1]) * group
                return int(round(packed.shape[1] * 32 / in_features))
    return None


def warn_bits_mismatch(pack, new_bits):
    pack_bits = pack_language_bits(pack)
    if pack_bits is not None and pack_bits != new_bits:
        print(
            f"WARNING: pack's language linears are {pack_bits}-bit but the graft writes "
            f"{new_bits}-bit. Mixed quantization is legal (the engine resolves per weight "
            f"from geometry); the tower just quantizes at {new_bits}-bit."
        )


def run_graft(pack, src, bits_override):
    """Returns the backup dir path. Exits nonzero with a named message on bad input."""
    if not os.path.isdir(os.path.join(pack, "text_encoder")):
        sys.exit(f"error: {pack}/text_encoder not found (is this a converted mlx-serve pack?)")
    if not glob.glob(os.path.join(src, "text_encoder", "*.safetensors")):
        sys.exit(f"error: no safetensors under {src}/text_encoder")
    tower_weights = source_tower_weight_keys(src)
    if not tower_weights:
        sys.exit(
            "error: source has no model.visual.* tensors — this checkpoint cannot supply "
            "the vision tower (need the full Qwen/Qwen-Image-2.1 text_encoder shards)"
        )

    if bits_override:
        bits = bits_override
    else:
        with open(os.path.join(pack, "config.json")) as f:
            q = json.load(f).get("quantization") or {}
        bits = q.get("te_bits")
        if bits is None:
            sys.exit(f"error: no quantization.te_bits in {pack}/config.json (use --bits)")
        bits = int(bits)
    warn_bits_mismatch(pack, bits)
    print(f"grafting vision tower into {pack}/text_encoder at {bits}-bit from {src}")

    # convert_component's `out` is a PACK ROOT (it writes out/text_encoder/),
    # so stage a fake pack root and swap the built text_encoder/ into place.
    stage_root = os.path.join(pack, "text_encoder.graft")
    if os.path.exists(stage_root):
        shutil.rmtree(stage_root)
    os.makedirs(stage_root)
    conv.convert_component(src, stage_root, "text_encoder", bits)
    stage = os.path.join(stage_root, "text_encoder")

    ts = int(time.time())
    backup = os.path.join(pack, f"text_encoder.bak-{ts}")
    while os.path.exists(backup):
        ts += 1
        backup = os.path.join(pack, f"text_encoder.bak-{ts}")
    os.rename(os.path.join(pack, "text_encoder"), backup)
    try:
        os.rename(stage, os.path.join(pack, "text_encoder"))
    except OSError:
        os.rename(backup, os.path.join(pack, "text_encoder"))
        raise
    shutil.rmtree(stage_root, ignore_errors=True)

    # convert_component writes only the shards; text_encoder/config.json (the
    # vision_config the Zig loader reads) must ride along — the old pack's,
    # else the source's.
    new_cfg = os.path.join(pack, "text_encoder", "config.json")
    if not os.path.exists(new_cfg):
        cfg_src = os.path.join(backup, "config.json")
        if not os.path.exists(cfg_src):
            cfg_src = os.path.join(src, "text_encoder", "config.json")
        shutil.copy2(cfg_src, new_cfg)
    print(f"old text_encoder moved to: {backup}")

    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(pack, "text_encoder", "*.safetensors")))
    keys = all_pack_keys(os.path.join(pack, "text_encoder"))
    tower = sum(1 for k in keys if k.startswith("model.visual."))
    print(f"summary: {len(keys)} tensors, {total / 1e9:.2f} GB, tower keys: {tower}")
    print("done: pack now carries instruction-edit capability")
    return backup


def self_test():
    import mlx.core as mx

    tmp = tempfile.mkdtemp(prefix="graft-selftest-")
    try:
        # -- synthetic source: language + tower in the real naming --
        src = os.path.join(tmp, "src")
        os.makedirs(os.path.join(src, "text_encoder"))
        tensors = {
            # dense: embed table (prefix not a layer linear)
            "model.language_model.embed_tokens.weight": mx.random.normal((32, 16), key=mx.random.key(1)),
            # dense: 16 % 64 != 0 exercises the shape fallback
            "model.language_model.layers.0.attn.qkv.weight": mx.random.normal((33, 16), key=mx.random.key(2)),
            # quantized: 2D layer linear, 64 % 64 == 0
            "model.language_model.layers.0.mlp.down_proj.weight": mx.random.normal((32, 64), key=mx.random.key(3)),
            "lm_head.weight": mx.random.normal((4, 16), key=mx.random.key(4)),  # converter drops
            "model.visual.pos_embed.weight": mx.random.normal((6, 16), key=mx.random.key(5)),  # dense
            "model.visual.blocks.0.attn.qkv.weight": mx.random.normal((32, 64), key=mx.random.key(6)),  # quantized
        }
        mx.save_safetensors(os.path.join(src, "text_encoder", "model-00001-of-00001.safetensors"), tensors)
        with open(os.path.join(src, "text_encoder", "config.json"), "w") as f:
            f.write("{}")

        # -- synthetic towerless pack: what the converter itself produces from a
        #    tower-less source, so byte-identity of language tensors is exact --
        src_no_tower = os.path.join(tmp, "src-no-tower")
        os.makedirs(os.path.join(src_no_tower, "text_encoder"))
        mx.save_safetensors(
            os.path.join(src_no_tower, "text_encoder", "model-00001-of-00001.safetensors"),
            {k: v for k, v in tensors.items() if not k.startswith("model.visual.")},
        )
        with open(os.path.join(src_no_tower, "text_encoder", "config.json"), "w") as f:
            f.write("{}")
        pack = os.path.join(tmp, "pack")
        os.makedirs(pack)
        with open(os.path.join(pack, "config.json"), "w") as f:
            json.dump({"quantization": {"group_size": 64, "dit_bits": 4, "te_bits": 4}}, f)
        conv.convert_component(src_no_tower, pack, "text_encoder", 4)
        orig_shard = os.path.join(pack, "text_encoder", "model-00001-of-00001.safetensors")
        orig_keys = all_pack_keys(os.path.join(pack, "text_encoder"))
        assert not any(k.startswith("model.visual.") for k in orig_keys)

        backup = run_graft(pack, src, None)
        new_shard = os.path.join(pack, "text_encoder", "model-00001-of-00001.safetensors")

        # convert_component writes only the shards; the loader reads the tower
        # geometry from text_encoder/config.json, so the rebuilt dir MUST carry
        # it (the old pack's, else the source's).
        assert os.path.exists(os.path.join(pack, "text_encoder", "config.json"))

        # 1. tower keys present, quantized triplets, pos_embed dense, lm_head dropped
        new_keys = all_pack_keys(os.path.join(pack, "text_encoder"))
        for suffix in (".weight", ".scales", ".biases"):
            assert "model.visual.blocks.0.attn.qkv" + suffix in new_keys, f"missing qkv{suffix}"
        assert "model.visual.pos_embed.weight" in new_keys
        assert "model.visual.pos_embed.scales" not in new_keys
        assert "lm_head.weight" not in new_keys
        # odd-dim language linear stays dense
        assert "model.language_model.layers.0.attn.qkv.scales" not in new_keys
        # quantized language linear carries the triplet
        assert "model.language_model.layers.0.mlp.down_proj.scales" in new_keys

        # 2. language tensors byte-identical to the pack's originals
        a, b = mx.load(orig_shard), mx.load(new_shard)
        for k in sorted(orig_keys):
            assert k in b, f"key lost in graft: {k}"
            assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, f"shape/dtype drift on {k}"
            assert mx.array_equal(a[k], b[k]).item(), f"tensor mismatch on {k}"

        # 3. the swap kept the originals: backup dir exists with the old shard
        assert os.path.isdir(backup)
        assert os.path.exists(os.path.join(backup, os.path.basename(orig_shard)))

        # 4. missing --src model.visual keys -> nonzero exit with a named message
        try:
            run_graft(pack, src_no_tower, None)
            raise AssertionError("graft over a tower-less source should have exited nonzero")
        except SystemExit as e:
            assert "model.visual" in str(e), f"unexpected exit message: {e}"

        print("self-test ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", help="converted Qwen-Image-2.1 pack dir (towerless)")
    ap.add_argument("--src", help="original Qwen/Qwen-Image-2.1 checkpoint dir")
    ap.add_argument("--bits", type=int, help="override te_bits from the pack config")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.pack or not args.src:
        ap.error("--pack and --src are required")
    run_graft(args.pack, args.src, args.bits)


if __name__ == "__main__":
    main()
