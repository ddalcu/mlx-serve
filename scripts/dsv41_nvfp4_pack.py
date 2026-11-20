#!/usr/bin/env python3
"""NVIDIA's DeepSeek-V4.1-Flash-1017-NVFP4 as a pack mlx-serve's deepseek_v41 reads, exact.

Every converted tensor keeps NVIDIA's bytes; only the safetensors headers change, plus one side shard of row scales:

- The fp8 linears (attention, shared experts, the indexer's wq_b, Engram's wkv): F8_E4M3 `.weight` [out, in] with an
  F8_E8M0 `.scale` grid of one scale per 32x32 block. MLX's mxfp8 at group 32 reads the same codes as U32 words
  [out, in/4] beside one e8m0 scale per row and 32 columns: each block's scale repeated over its 32 rows
  (`<base>.scales` U8 [out, in/32], written to `model-mxfp8-scales.safetensors`). The block grid stays in its shard,
  renamed `<base>.block_scale`, so the shard's bytes are untouched.
- The DSpark (mtp.*) routed experts: I8 e2m1 pairs [out, in/2] with an F8_E8M0 `.scale` [out, in/32] are MLX's mxfp4
  at group 32 as stored: the codes as U32 [out, in/8], the scale renamed `.scales` (U8).
- The NVFP4 routed experts (U8 codes, F8_E4M3 `weight_scale` per 16, F32 `weight_scale_2`, `input_scale`) and the
  Engram tables (U8 e2m1 rows, F8_E8M0 `scale` per 32) keep their names; only their 8-bit float dtypes are relabelled
  U8 (the same bytes), which every safetensors reader loads.

Each shard is an APFS clone of NVIDIA's (`cp -c`: no data copied) whose header is rewritten in place at the same
length (padded with spaces), so every tensor keeps its offsets; a shard with no change is a symlink. The index is
written new. Runs on the box that holds the pack, never on a laptop:

    python3 -I dsv41_nvfp4_pack.py SRC_DIR DST_DIR [--check] [--experts-mxfp4]

`--check` re-reads DST and proves each converted tensor against SRC (codes byte for byte, every row scale against its
block).

`--experts-mxfp4` (numpy) writes the NVFP4 routed experts as MLX mxfp4 at group 32, which the model serves itself with
no quant plugin. NVIDIA converted them from mxfp4 losslessly: in every 32-group, each 16-half that holds a nonzero
code has e4m3 scale x `weight_scale_2` = the same power of two 2^k, so the codes stay as stored (U32 [out, in/8]) and
`<base>.scales` is k + 127 (e8m0, U8 [out, in/32], in `model-mxfp4-scales-<shard>.safetensors`). A group that breaks
this stops the conversion by name. `weight_scale`, `weight_scale_2` and `input_scale` leave the header (their bytes
stay in the clone, unread). `--check` then dequantizes every expert both ways and requires every value equal.
"""

import json
import os
import random
import shutil
import struct
import subprocess
import sys

SIDE = "model-mxfp8-scales.safetensors"
EXPERT_RE = __import__("re").compile(r"^(layers\.\d+\.ffn\.experts\.\d+\.w[123])\.(weight|weight_scale|weight_scale_2|input_scale)$")
BLOCK = 32
DTYPE_BYTES = {"F8_E4M3": 1, "F8_E8M0": 1, "U8": 1, "I8": 1, "U32": 4, "F32": 4, "BF16": 2, "F16": 2, "I32": 4, "I64": 8}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return n, h


def nbytes(t):
    n = DTYPE_BYTES[t["dtype"]]
    for d in t["shape"]:
        n *= d
    return n


def nvfp4_expert(t, k):
    """The expert base `k` names (its codes or a scale) when the shard holds an NVFP4 expert there, else None."""
    m = EXPERT_RE.match(k)
    return m.group(1) if m and m.group(1) + ".weight_scale_2" in t else None


def plan_shard(h, mx=False):
    """The shard's new header, the side scales it owes ([(base, out, in, block-scale tensor)]) and, with `mx`, its
    NVFP4 experts ([(base, out, in)])."""
    meta = h.get("__metadata__")
    t = {k: v for k, v in h.items() if k != "__metadata__"}
    new = {}
    side = []
    experts = []
    changed = False
    for k, v in t.items():
        v = dict(v)
        if mx and nvfp4_expert(t, k):
            if k.endswith(".weight"):
                out, half = v["shape"]
                if v["dtype"] != "U8" or half % 16:
                    raise SystemExit(f"{k}: NVFP4 codes other than U8 pairs in whole 32-groups")
                v["dtype"], v["shape"] = "U32", [out, half // 4]
                new[k] = v
                experts.append((k[: -len(".weight")], out, half * 2))
            changed = True
            continue
        if k.endswith(".scale") and k[: -len(".scale")] + ".weight" in t and ".engram.embed." not in k:
            base = k[: -len(".scale")]
            w = t[base + ".weight"]
            if w["dtype"] == "F8_E4M3":
                out, inn = w["shape"]
                if v["dtype"] != "F8_E8M0" or v["shape"] != [-(-out // BLOCK), -(-inn // BLOCK)] or inn % BLOCK:
                    raise SystemExit(f"{k}: an fp8 scale grid other than one e8m0 per 32x32 block: {v}")
                v["dtype"] = "U8"
                new[base + ".block_scale"] = v
                side.append((base, out, inn, base + ".block_scale"))
                changed = True
                continue
            if w["dtype"] in ("I8", "U8"):
                out, half = w["shape"]
                if v["dtype"] != "F8_E8M0" or v["shape"] != [out, half * 2 // BLOCK] or (half * 2) % BLOCK:
                    raise SystemExit(f"{k}: an fp4 scale other than one e8m0 per 32 along the row: {v}")
                v["dtype"] = "U8"
                new[base + ".scales"] = v
                changed = True
                continue
            raise SystemExit(f"{k}: a scale beside a {w['dtype']} weight this converter does not read")
        if k.endswith(".weight") and k[: -len(".weight")] + ".scale" in t and ".engram.embed." not in k:
            out, cols = v["shape"]
            if v["dtype"] == "F8_E4M3":
                if cols % 4:
                    raise SystemExit(f"{k}: fp8 columns not a multiple of 4")
                v["dtype"], v["shape"] = "U32", [out, cols // 4]
            elif v["dtype"] in ("I8", "U8"):
                if cols % 4:
                    raise SystemExit(f"{k}: fp4 bytes not a multiple of 4")
                v["dtype"], v["shape"] = "U32", [out, cols // 4]
            changed = True
            new[k] = v
            continue
        if v["dtype"] in ("F8_E4M3", "F8_E8M0"):
            # NVFP4 expert scales, Engram scales: the same bytes as U8.
            v["dtype"] = "U8"
            changed = True
        new[k] = v
    if meta is not None:
        new["__metadata__"] = meta
    return new, side, changed, experts


def write_header_in_place(path, n, header):
    raw = json.dumps(header, separators=(",", ":")).encode()
    if len(raw) > n:
        raise SystemExit(f"{path}: the new header ({len(raw)} B) is longer than the old ({n} B)")
    raw += b" " * (n - len(raw))
    with open(path, "r+b") as f:
        f.seek(8)
        f.write(raw)


def expand(grid, out, inn):
    """One e8m0 per 32x32 block -> one per row and 32 columns."""
    gc = -(-inn // BLOCK)
    rows = [grid[(r // BLOCK) * gc : (r // BLOCK + 1) * gc] for r in range(out)]
    return b"".join(rows)


def read_tensor(path, n, t):
    """A tensor's bytes, read past the page cache (F_NOCACHE): the experts are hundreds of GB read once."""
    a, b = t["data_offsets"]
    fd = os.open(path, os.O_RDONLY)
    try:
        if sys.platform == "darwin":
            __import__("fcntl").fcntl(fd, 48, 1)
        out = bytearray(b - a)
        mv = memoryview(out)
        got = 0
        while got < len(out):
            k = os.preadv(fd, [mv[got : got + (1 << 30)]], 8 + n + a + got)
            if k <= 0:
                raise SystemExit(f"{path}: short read")
            got += k
        return bytes(out)
    finally:
        os.close(fd)


E2M1_ZERO = (0, 8)


def mx_scales(job):
    """One NVFP4 expert's mxfp4 scales: k + 127 per 32-group, where every used 16-half's e4m3 scale x global is 2^k."""
    import numpy as np

    path, n, base, codes_t, scale_t, s2_t, out, inn = job
    c = np.frombuffer(read_tensor(path, n, codes_t), np.uint8).reshape(out, inn // 32, 16)
    s = np.frombuffer(read_tensor(path, n, scale_t), np.uint8).reshape(out, inn // 32, 2)
    s2 = np.frombuffer(read_tensor(path, n, s2_t), np.float32)[0]
    S = E4M3[s] * np.float64(s2)
    m, e = np.frexp(S)
    pow2 = m == 0.5
    # a half is used when any of its 16 codes is nonzero (0 and 8 are +-0)
    nz = ((c & 7) != 0) | ((c >> 4 & 7) != 0)
    used = nz.reshape(out, inn // 32, 2, 8).any(-1)
    k = e - 1
    bad = used & ~pow2
    both = used[..., 0] & used[..., 1]
    bad_pair = both & (k[..., 0] != k[..., 1])
    if bad.any() or bad_pair.any():
        raise SystemExit(f"{base}: {int(bad.sum())} halves off a power of two, {int(bad_pair.sum())} pairs at two scales: not mxfp4")
    kk = np.where(used[..., 0], k[..., 0], np.where(used[..., 1], k[..., 1], np.where(pow2[..., 0], k[..., 0], 0)))
    if (kk < -127).any() or (kk > 127).any():
        raise SystemExit(f"{base}: a scale outside e8m0")
    return base, (kk + 127).astype(np.uint8).tobytes()


def e4m3_lut():
    import numpy as np

    out = np.zeros(256)
    for b in range(256):
        e, mt = (b >> 3) & 15, b & 7
        v = np.nan if (e == 15 and mt == 7) else (mt / 8 * 2.0**-6 if e == 0 else (1 + mt / 8) * 2.0 ** (e - 7))
        out[b] = -v if b & 0x80 else v
    return out


E4M3 = None


def mx_init():
    global E4M3
    E4M3 = e4m3_lut()


def write_side(path, tensors):
    hdr = {}
    off = 0
    for k, data, shape in tensors:
        hdr[k] = {"dtype": "U8", "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    hdr["__metadata__"] = {"format": "pt"}
    raw = json.dumps(hdr, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for _, data, _ in tensors:
            f.write(data)
    return off


def convert(src, dst, mx=False):
    os.makedirs(dst, exist_ok=True)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    for name in shards:
        if not os.path.exists(os.path.join(src, name)):
            raise SystemExit(f"{name}: missing in {src}")
    weight_map = {}
    side_tensors = []  # (key, bytes, shape)
    total = 0
    for name in shards:
        sp = os.path.join(src, name)
        dp = os.path.join(dst, name)
        n, h = read_header(sp)
        new, side, changed, experts = plan_shard(h, mx)
        if os.path.lexists(dp):
            os.remove(dp)
        if changed:
            subprocess.run(["cp", "-c", sp, dp], check=True)
            write_header_in_place(dp, n, new)
        else:
            os.symlink(sp, dp)
        for base, out, inn, bk in side:
            grid = read_tensor(sp, n, h[base + ".scale"])
            side_tensors.append((base + ".scales", expand(grid, out, inn), [out, inn // BLOCK]))
        for k, v in new.items():
            if k == "__metadata__":
                continue
            weight_map[k] = name
            total += nbytes(v)
        if experts:
            from multiprocessing import Pool

            jobs = [(sp, n, b, h[b + ".weight"], h[b + ".weight_scale"], h[b + ".weight_scale_2"], o, i) for b, o, i in experts]
            shapes = {b: [o, i // BLOCK] for b, o, i in experts}
            with Pool(24, initializer=mx_init) as pool:
                got = pool.map(mx_scales, jobs, chunksize=8)
            side_name = "model-mxfp4-scales-" + name.split("-")[1] + ".safetensors"
            total += write_side(os.path.join(dst, side_name), [(b + ".scales", data, shapes[b]) for b, data in got])
            for b, _ in got:
                weight_map[b + ".scales"] = side_name
        print(f"{name}: {'header rewritten' if changed else 'linked'}, {len(side)} fp8 scale grids, {len(experts)} NVFP4 experts as mxfp4", flush=True)
    # The side shard: every fp8 linear's row scales.
    off = sum(len(d) for _, d, _ in side_tensors)
    for k, data, shape in side_tensors:
        weight_map[k] = SIDE
        total += len(data)
    write_side(os.path.join(dst, SIDE), side_tensors)
    print(f"{SIDE}: {len(side_tensors)} row-scale tensors, {off} B", flush=True)
    meta = dict(idx.get("metadata", {}))
    meta["total_size"] = total
    meta["converted_by"] = "mlx-serve scripts/dsv41_nvfp4_pack.py"
    json.dump({"metadata": meta, "weight_map": dict(sorted(weight_map.items()))}, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    # Everything else (config, tokenizer, the reference code) as links.
    for e in os.listdir(src):
        if e in shards or e == "model.safetensors.index.json" or e.startswith("."):
            continue
        dp = os.path.join(dst, e)
        if not os.path.lexists(dp):
            os.symlink(os.path.join(src, e), dp)


def mx_same(job):
    """Every value of one expert: NVIDIA's e2m1 x e4m3 x global against the mxfp4 e2m1 x 2^(scale - 127)."""
    import numpy as np

    sp, on, dp, dn, xp, xn, base, codes_t, scale_t, s2_t, mx_t, out, inn = job
    c = np.frombuffer(read_tensor(sp, on, codes_t), np.uint8).reshape(out, inn // 32, 16)
    if read_tensor(dp, dn, codes_t) != c.tobytes():
        return base, "codes differ"
    s = np.frombuffer(read_tensor(sp, on, scale_t), np.uint8).reshape(out, inn // 32, 2)
    s2 = np.float64(np.frombuffer(read_tensor(sp, on, s2_t), np.float32)[0])
    x = np.frombuffer(read_tensor(xp, xn, mx_t), np.uint8).reshape(out, inn // 32)
    e2m1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    v = e2m1[np.stack([c & 15, c >> 4], -1).reshape(out, inn // 32, 32)]
    nv = v * np.repeat(E4M3[s] * s2, 16, axis=-1)
    mv = v * np.ldexp(1.0, x.astype(np.int64) - 127)[..., None]
    bad = int((nv != mv).sum())
    return base, f"{bad} values differ" if bad else None


def check(src, dst, mx=False):
    idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
    sidx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    sn, sh = read_header(os.path.join(dst, SIDE))
    rng = random.Random(1234)
    n_fp8 = n_fp4 = n_relabel = 0
    mx_jobs = []
    sides = {}
    for name in sorted(set(sidx["weight_map"].values())):
        on, oh = read_header(os.path.join(src, name))
        nn, nh = read_header(os.path.join(dst, name))
        if on != nn:
            raise SystemExit(f"{name}: header length moved")
        for k, v in oh.items():
            if k == "__metadata__":
                continue
            if mx and nvfp4_expert(oh, k):
                if not k.endswith(".weight"):
                    if k in nh or k in idx["weight_map"]:
                        raise SystemExit(f"{k}: still named")
                    continue
                base = k[: -len(".weight")]
                if nh[k]["data_offsets"] != v["data_offsets"] or nh[k]["dtype"] != "U32":
                    raise SystemExit(f"{k}: codes moved")
                xp = os.path.join(dst, idx["weight_map"][base + ".scales"])
                if xp not in sides:
                    sides[xp] = read_header(xp)
                xn, xh = sides[xp]
                out, half = v["shape"]
                if xh[base + ".scales"]["shape"] != [out, half * 2 // BLOCK]:
                    raise SystemExit(f"{base}.scales: shape")
                mx_jobs.append((os.path.join(src, name), on, os.path.join(dst, name), nn, xp, xn, base, v,
                                oh[base + ".weight_scale"], oh[base + ".weight_scale_2"], xh[base + ".scales"], out, half * 2))
                continue
            if k.endswith(".scale") and ".engram.embed." not in k and k[:-6] + ".weight" in oh:
                base = k[:-6]
                if oh[base + ".weight"]["dtype"] == "F8_E4M3":
                    nv = nh[base + ".block_scale"]
                    if nv["data_offsets"] != v["data_offsets"]:
                        raise SystemExit(f"{k}: block grid moved")
                    out, inn = oh[base + ".weight"]["shape"]
                    grid = read_tensor(os.path.join(src, name), on, v)
                    rows = read_tensor(os.path.join(dst, SIDE), sn, sh[base + ".scales"])
                    gc = -(-inn // BLOCK)
                    if sh[base + ".scales"]["shape"] != [out, inn // BLOCK] or len(rows) != out * gc:
                        raise SystemExit(f"{base}.scales: shape")
                    for r in range(out):
                        if rows[r * gc : (r + 1) * gc] != grid[(r // BLOCK) * gc : (r // BLOCK + 1) * gc]:
                            raise SystemExit(f"{base}.scales: row {r} is not its block's scale")
                    n_fp8 += 1
                else:
                    nv = nh[base + ".scales"]
                    if nv["data_offsets"] != v["data_offsets"] or nv["dtype"] != "U8":
                        raise SystemExit(f"{k}: fp4 scale moved")
                    n_fp4 += 1
                continue
            nv = nh[k]
            if nv["data_offsets"] != v["data_offsets"] or nbytes(nv) != nbytes(v):
                raise SystemExit(f"{k}: moved or resized")
            if nv["dtype"] != v["dtype"]:
                n_relabel += 1
            if idx["weight_map"].get(k) != name:
                raise SystemExit(f"{k}: the index does not name {name}")
        # A sample of tensors byte for byte (clones share the blocks; this proves the header kept every offset).
        keys = [k for k in oh if k != "__metadata__" and not k.endswith(".scale") and not (mx and nvfp4_expert(oh, k))]
        for k in rng.sample(keys, min(4, len(keys))):
            a = read_tensor(os.path.join(src, name), on, oh[k])
            b = read_tensor(os.path.join(dst, name), nn, nh[k])
            if a != b:
                raise SystemExit(f"{k}: bytes differ")
    if mx_jobs:
        from multiprocessing import Pool

        with Pool(24, initializer=mx_init) as pool:
            bad = [r for r in pool.imap_unordered(mx_same, mx_jobs, chunksize=8) if r[1]]
        if bad:
            raise SystemExit(f"{len(bad)} experts differ, first {bad[0]}")
    print(f"check: {n_fp8} fp8 grids expanded exactly, {n_fp4} fp4 scales renamed, {n_relabel} tensors relabelled, "
          f"{len(mx_jobs)} NVFP4 experts equal as mxfp4 value for value, offsets all kept")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        raise SystemExit(__doc__)
    mx = "--experts-mxfp4" in sys.argv
    if mx:
        mx_init()
    if "--check" not in sys.argv:
        convert(args[0], args[1], mx)
    check(args[0], args[1], mx)
