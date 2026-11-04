#!/usr/bin/env python3
"""glm53_pack.py: GLM-5.3 (`model_type` glm_moe_dsa) snapshots into the streaming packs mlx-serve reads, bytes copied,
never converted. The routed experts go to `experts.bin` (one 4096-aligned record per expert, or per tensor-parallel
mini-expert on EXL3) beside a manifest; every other tensor stays a resident safetensors shard.

  glm53_pack.py affine --src <MLX snapshot> --dst <pack> [--bits auto|3|4] [--shard-bytes 5GiB]
      writes the resident shards (no `.mlp.switch_mlp.` tensors), experts.bin, expert-manifest-affine-v1.json
  glm53_pack.py exl3 --src <EXL3 snapshot> --dst <pack> --residents-from <affine pack>
      hard-links the affine pack's resident shards, writes experts.bin (mixed K3/K4 trellis minis, the MTP layer's
      too), expert-manifest-exl3-v1.json and mtp-residents.safetensors
  glm53_pack.py mtp --dst <affine pack>/mtp (--from-pack <EXL3 pack> | --src <EXL3 snapshot>)
      the MTP layer alone for an affine pack: mtp-residents.safetensors, mtp-experts.bin, mtp-manifest-exl3-v1.json
Common flags: [--resume] [--verify N|all] [--source-repo R --source-revision SHA] [--stop-after-layer K (bank layer)].
`convert-progress.json` is written after each bank layer, `convert-report.json` at the end; `--resume` keeps the bank
layers whose first and last records still match their sha256; `--verify` re-reads records against the source.
Exit 0; 1: a verified record differs; 2: refused (one line naming the cause).
Tests: python3 -m unittest scripts/glm53_pack.py"""
import argparse
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import sys
import time

ALIGN = 4096
SIDECAR = "experts.bin"
PROGRESS = "convert-progress.json"
REPORT = "convert-report.json"
INDEX = "model.safetensors.index.json"
TIER = "tier_bitmap.json"
MTP_RESIDENTS = "mtp-residents.safetensors"
MTP_SIDECAR = "mtp-experts.bin"
MTP_MANIFEST = "mtp-manifest-exl3-v1.json"
AFFINE = {"format": "mlx-stream-expert-manifest-affine-v1", "manifest": "expert-manifest-affine-v1.json"}
EXL3 = {"format": "mlx-stream-expert-manifest-exl3-v1", "manifest": "expert-manifest-exl3-v1.json"}
COPIED = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE"]
SWITCH = ".mlp.switch_mlp."
EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.")
SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
               "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
PROJS = ["gate_proj", "up_proj", "down_proj"]
AFFINE_COMPONENTS = ["%s.%s" % (p, c) for p in ["gate", "up", "down"] for c in ["weight", "scales", "biases"]]
EXL3_COMPONENTS = ["%s.%s" % (p, c) for p in PROJS for c in ["code", "rout", "rin"]]
EXL3_PART = {"code": "trellis", "rout": "svh", "rin": "suh"}
CHUNK = 64 << 20


class Refused(Exception):
    pass


def nbytes(dtype, shape):
    n = DTYPE_BYTES[dtype]
    for d in shape:
        n *= d
    return n


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(path + ".tmp", path)


def need(src, *files):
    for f in files:
        if not os.path.exists(os.path.join(src, f)):
            raise Refused("%s has no %s" % (src, f))


def same_device(path, dst):
    near = os.path.abspath(dst)
    while not os.path.exists(near):
        near = os.path.dirname(near)
    return os.stat(path).st_dev == os.stat(near).st_dev


class Source:
    """The snapshot's safetensors tensors: name -> (file, dtype, shape, absolute begin, absolute end)."""

    def __init__(self, src):
        idx = os.path.join(src, INDEX)
        names = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
        files = sorted(set(names.values())) if names else sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
        if not files:
            raise Refused("no safetensors shards in %s" % src)
        self.src, self.tensors, self.order, self.maps = src, {}, [], {}
        for fn in files:
            with open(os.path.join(src, fn), "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            for name, t in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                a, b = t["data_offsets"]
                self.tensors[name] = (fn, t["dtype"], list(t["shape"]), 8 + n + a, 8 + n + b)
                self.order.append(name)
        for name in sorted(set(names or []) - set(self.tensors)):
            raise Refused("index names %s but no shard holds it" % name)

    def view(self, fn):
        if fn not in self.maps:
            f = open(os.path.join(self.src, fn), "rb")
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.maps[fn] = (f, m, memoryview(m))
        return self.maps[fn][2]

    def close(self):
        for f, m, v in self.maps.values():
            v.release()
            m.close()
            f.close()
        self.maps = {}

    def write(self, path, members):
        """The tensors `members` as one safetensors file, bytes unchanged; returns its data bytes."""
        header, off = {"__metadata__": {"format": "mlx"}}, 0
        for n in members:
            _, dtype, shape, a, b = self.tensors[n]
            header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + b - a]}
            off += b - a
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb)
            for n in members:
                fn, _, _, a, b = self.tensors[n]
                for lo in range(a, b, CHUNK):
                    piece = self.view(fn)[lo:min(b, lo + CHUNK)]
                    f.write(piece)
                    piece.release()
        return off

    def write_shards(self, dst, names, shard_bytes):
        """`names` in shards of at most `shard_bytes` and their index (other model-* shards in dst removed)."""
        shards, cur, cur_n = [], [], 0
        for n in names:
            size = self.tensors[n][4] - self.tensors[n][3]
            if cur and cur_n + size > shard_bytes:
                shards.append(cur)
                cur, cur_n = [], 0
            cur.append(n)
            cur_n += size
        shards += [cur] if cur else []
        files = ["model-%05d-of-%05d.safetensors" % (k + 1, len(shards)) for k in range(len(shards))]
        for f in os.listdir(dst):
            if SHARD_RE.match(f) and f not in files:
                os.remove(os.path.join(dst, f))
        weight_map, total = {}, 0
        for fname, members in zip(files, shards):
            total += self.write(os.path.join(dst, fname), members)
            weight_map.update({n: fname for n in members})
        write_json(os.path.join(dst, INDEX), {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))})
        return total


# ── geometry: a bank is a list of bank layers, each a run of equal records (one per expert x rank) ──

def bank_layer(g, layer, experts, segs, base, **extra):
    logical = segs[-1]["offset"] + segs[-1]["length"]
    record = -(-logical // ALIGN) * ALIGN
    return dict({"bank_layer": len(g["bank"]), "layer": layer, **extra, "n_minis": g["tp"] * len(experts),
                 "record_bytes": record, "logical_bytes": logical, "base_offset": base, "experts": experts,
                 "segments": segs})


def layout(parts):
    """Segments of a record from (component, dtype, shape) in order."""
    segs, off = [], 0
    for c, dtype, shape in parts:
        segs.append({"component": c, "dtype": dtype, "shape": shape, "offset": off, "length": nbytes(dtype, shape)})
        off += segs[-1]["length"]
    return segs


def affine_geometry(cfg, bits_arg):
    if cfg.get("model_type") != "glm_moe_dsa":
        raise Refused("model_type %r is not glm_moe_dsa" % cfg.get("model_type"))
    for k in ["mlp_layer_types", "n_routed_experts", "hidden_size", "moe_intermediate_size", "quantization"]:
        if k not in cfg:
            raise Refused("config.json has no %s" % k)
    q = cfg["quantization"]
    if q.get("mode", "affine") != "affine" or q.get("group_size") != 64 or "bits" not in q:
        raise Refused("quantization %s is not affine with group_size 64" % json.dumps(q, sort_keys=True))
    bits = q["bits"] if bits_arg == "auto" else int(bits_arg)
    if bits not in (3, 4):
        raise Refused("expert bits %d is not 3 or 4" % bits)
    hidden, inter, n_exp = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"]
    if hidden % 64 or inter % 64:
        raise Refused("hidden %d / inter %d do not pack at group 64" % (hidden, inter))
    shapes = {"gate": (inter, hidden), "up": (inter, hidden), "down": (hidden, inter)}
    segs = layout([(c, "U32", [shapes[c.split(".")[0]][0], shapes[c.split(".")[0]][1] * bits // 32])
                   if c.endswith(".weight") else (c, "BF16", [shapes[c.split(".")[0]][0], shapes[c.split(".")[0]][1] // 64])
                   for c in AFFINE_COMPONENTS])
    g = {"kind": "affine", "bits": bits, "hidden": hidden, "inter": inter, "n_experts": n_exp, "tp": 1, "bank": []}
    g["layers"] = [i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "sparse"]
    if not g["layers"]:
        raise Refused("mlp_layer_types has no sparse layer")
    base = 0
    for L in g["layers"]:
        g["bank"].append(bank_layer(g, L, list(range(n_exp)), segs, base))
        base += n_exp * g["bank"][-1]["record_bytes"]
    g["size"] = base
    return g


def exl3_shapes(g, proj, k):
    h, m = g["hidden"], g["mini_inter"]
    out, inn = (m, h) if proj != "down_proj" else (h, m)
    return {"trellis": ("I16", [inn // 16, out // 16, 16 * k]), "suh": ("F16", [inn]), "svh": ("F16", [out]),
            "mcg": ("I32", [])}


def exl3_geometry(cfg, tier):
    if cfg.get("model_type") != "glm_moe_dsa":
        raise Refused("model_type %r is not glm_moe_dsa" % cfg.get("model_type"))
    for k in ["mlp_layer_types", "n_routed_experts", "hidden_size", "moe_intermediate_size", "num_hidden_layers",
              "hybrid_tr3_tail"]:
        if k not in cfg:
            raise Refused("config.json has no %s" % k)
    tail = cfg["hybrid_tr3_tail"]
    if tail.get("codebook") != "mcg" or "mcg_multiplier" not in tail or "tp" not in tail:
        raise Refused("hybrid_tr3_tail is not an mcg codebook with mcg_multiplier and tp")
    k_values = tail.get("k_values", [])
    if not k_values or not set(k_values) <= {3, 4} or k_values != sorted(set(k_values)):
        raise Refused("hybrid_tr3_tail.k_values %s is not an ascending subset of [3, 4]" % k_values)
    hidden, inter, n_exp, tp = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"], tail["tp"]
    if inter % tp or inter // tp % 16 or hidden % 16:
        raise Refused("hidden %d / inter %d / tp %d do not split in 16-wide trellis tiles" % (hidden, inter, tp))
    n_main = cfg["num_hidden_layers"]
    mtp = list(range(n_main, n_main + cfg.get("num_nextn_predict_layers", 0)))
    layers = [i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "sparse"] + mtp
    if not layers:
        raise Refused("config.json has no routed layer")
    if sorted(tier, key=int) != [str(L) for L in layers]:
        raise Refused("%s layers %s differ from the routed layers %s" % (TIER, sorted(map(int, tier)), layers))
    g = {"kind": "exl3", "hidden": hidden, "inter": inter, "mini_inter": inter // tp, "tp": tp, "n_experts": n_exp,
         "layers": layers, "mtp": mtp, "k_values": k_values, "multiplier": tail["mcg_multiplier"], "bank": [],
         "experts": {}}
    segs = {k: layout([(c, *exl3_shapes(g, c.split(".")[0], k)[EXL3_PART[c.split(".")[1]]]) for c in EXL3_COMPONENTS])
            for k in k_values}
    base = 0
    for L in layers:
        ks = tier[str(L)]["k"]
        if len(ks) != n_exp:
            raise Refused("%s layer %d has %d experts, want %d" % (TIER, L, len(ks), n_exp))
        for e, k in enumerate(ks):
            if k not in k_values:
                raise Refused("%s layer %d expert %d has K %r, not in k_values %s" % (TIER, L, e, k, k_values))
        g["experts"][str(L)] = [None] * n_exp
        for k in k_values:
            members = [e for e in range(n_exp) if ks[e] == k]
            if members:
                for local, e in enumerate(members):
                    g["experts"][str(L)][e] = [k, local]
                g["bank"].append(bank_layer(g, L, members, segs[k], base, k=k, mtp=L in mtp))
                base += g["bank"][-1]["n_minis"] * g["bank"][-1]["record_bytes"]
    g["size"] = base
    return g


def mtp_only(g):
    """The MTP layers' part of bank `g`, renumbered from bank layer 0 at offset 0."""
    bank, base = [], 0
    for b in g["bank"]:
        if b["mtp"]:
            bank.append(dict(b, bank_layer=len(bank), base_offset=base))
            base += b["n_minis"] * b["record_bytes"]
    if not bank:
        raise Refused("the source has no MTP layer (num_nextn_predict_layers is 0)")
    return dict(g, layers=list(g["mtp"]), bank=bank, size=base, experts={str(L): g["experts"][str(L)] for L in g["mtp"]})


def minis(g, b):
    """(mini, expert, rank) of bank layer b in record order."""
    return [(local * g["tp"] + r, e, r) for local, e in enumerate(b["experts"]) for r in range(g["tp"])]


def source_name(g, layer, expert, rank, component):
    p, part = component.split(".")
    if g["kind"] == "affine":
        return "model.layers.%d.mlp.switch_mlp.%s_proj.%s" % (layer, p, part)
    return "model.layers.%d.mlp.experts.%d.%s.rank%d.%s" % (layer, expert, p, rank, EXL3_PART[part])


def source_slices(g, b, expert, rank):
    """(name, offset into the tensor) of each segment of a record."""
    return [(source_name(g, b["layer"], expert, rank, s["component"]), expert * s["length"] if g["kind"] == "affine" else 0)
            for s in b["segments"]]


def check_affine(srcs, g):
    sparse = set(g["layers"])
    for name in srcs.tensors:
        if SWITCH in name and int(name.split(".")[2]) not in sparse:
            raise Refused("%s is not on a sparse layer" % name)
    for b in g["bank"]:
        for s in b["segments"]:
            name = source_name(g, b["layer"], 0, 0, s["component"])
            if name not in srcs.tensors:
                raise Refused("%s is missing" % name)
            _, dtype, shape, a, z = srcs.tensors[name]
            want = [g["n_experts"]] + s["shape"]
            if dtype != s["dtype"] or shape != want or z - a != g["n_experts"] * s["length"]:
                raise Refused("%s is %s %s, want %s %s" % (name, dtype, shape, s["dtype"], want))


def check_exl3(srcs, g, only=None):
    """Refuses a missing, unknown or misshapen expert tensor and mixed K; returns the shared mcg scalar."""
    want = {"model.layers.%d.mlp.experts.%d.%s.rank%d.%s" % (L, e, p, r, part) for L in g["layers"]
            for e in range(g["n_experts"]) for p in PROJS for r in range(g["tp"]) for part in exl3_shapes(g, p, 3)}
    for n in sorted(want - set(srcs.tensors)):
        raise Refused("%s is missing" % n)
    for n in sorted(n for n in srcs.tensors if EXPERT_RE.match(n) and n not in want
                    and (only is None or int(n.split(".")[2]) in only)):
        raise Refused("%s is not a routed expert tensor of the pack format" % n)
    mcg = None
    for L in g["layers"]:
        for e in range(g["n_experts"]):
            k = g["experts"][str(L)][e][0]
            ks = {srcs.tensors["model.layers.%d.mlp.experts.%d.%s.rank%d.trellis" % (L, e, p, r)][2][-1] // 16
                  for p in PROJS for r in range(g["tp"])}
            if len(ks) > 1:
                raise Refused("model.layers.%d.mlp.experts.%d has trellis K %s across its ranks and projections"
                              % (L, e, sorted(ks)))
            for p in PROJS:
                for r in range(g["tp"]):
                    for part, (wd, ws) in exl3_shapes(g, p, k).items():
                        n = "model.layers.%d.mlp.experts.%d.%s.rank%d.%s" % (L, e, p, r, part)
                        fn, dtype, shape, a, z = srcs.tensors[n]
                        if dtype != wd or shape != ws or z - a != nbytes(wd, ws):
                            raise Refused("%s is %s %s, want %s %s" % (n, dtype, shape, wd, ws))
                        if part == "mcg":
                            (v,) = struct.unpack("<i", srcs.view(fn)[a:z])
                            if mcg is None:
                                mcg = (v, n)
                            elif v != mcg[0]:
                                raise Refused("%s is %d, %s is %d: the mcg scalars differ" % (mcg[1], mcg[0], n, v))
    return mcg[0]


# ── the bank file ──

def reader(srcs, g):
    """read(b, mini, expert, rank, buf): fills buf[:logical] with the record's source bytes; returns their sha256."""
    def read(b, mini, expert, rank, buf):
        h = hashlib.sha256()
        for s, (name, off) in zip(b["segments"], source_slices(g, b, expert, rank)):
            fn, _, _, a, _ = srcs.tensors[name]
            piece = srcs.view(fn)[a + off:a + off + s["length"]]
            buf[s["offset"]:s["offset"] + s["length"]] = piece
            h.update(piece)
            piece.release()
        return h.hexdigest()
    return read


def record_at(fd, b, mini):
    return os.pread(fd, b["logical_bytes"], b["base_offset"] + mini * b["record_bytes"])


def write_bank(path, g, read, resume=False, prog_path=None, stop_after=None, want_sha=None):
    """Writes every bank layer's records; resumes the layers `prog_path` records whose first and last records still
    match. Returns ({bank_layer: [sha256 per mini]}, bytes written, records written, stopped early)."""
    prog = json.load(open(prog_path)) if resume and prog_path and os.path.exists(prog_path) else {"bank_layers": []}
    if not resume and os.path.lexists(path):
        os.remove(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != g["size"]:
        os.ftruncate(fd, g["size"])
    done = {}
    for e in prog["bank_layers"]:
        i = e["bank_layer"]
        if i >= len(g["bank"]) or (e["layer"], e["experts"], len(e["sha256"])) != (
                g["bank"][i]["layer"], g["bank"][i]["experts"], g["bank"][i]["n_minis"]):
            continue
        b = g["bank"][i]
        if all(hashlib.sha256(record_at(fd, b, m)).hexdigest() == e["sha256"][m] for m in (0, b["n_minis"] - 1)):
            done[i] = e
        else:
            print("bank layer %d: first or last record differs from its sha256, converting it again" % i)
    written = n_written = 0
    stopped = False
    try:
        for i, b in enumerate(g["bank"]):
            if i in done:
                continue
            t0, buf, shas = time.time(), bytearray(b["record_bytes"]), [None] * b["n_minis"]
            for mini, e, r in minis(g, b):
                shas[mini] = read(b, mini, e, r, buf)
                if want_sha is not None and shas[mini] != want_sha(b, mini):
                    raise Refused("pack record of layer %d K%d mini %d differs from its sha256 in %s" % (
                        b["layer"], b["k"], mini, EXL3["manifest"]))
                if os.pwrite(fd, buf, b["base_offset"] + mini * b["record_bytes"]) != len(buf):
                    raise OSError("short write to %s at bank layer %d mini %d" % (path, i, mini))
            os.fsync(fd)
            written += b["n_minis"] * b["record_bytes"]
            n_written += b["n_minis"]
            done[i] = {"bank_layer": i, "layer": b["layer"], "experts": b["experts"], "sha256": shas}
            if prog_path:
                write_json(prog_path, dict(prog, bank_layers=[done[k] for k in sorted(done)]))
            dt = max(time.time() - t0, 1e-9)
            print("bank layer %d (layer %d): %d records, %.1f s, %.2f GB/s" % (
                i, b["layer"], b["n_minis"], dt, b["n_minis"] * b["record_bytes"] / dt / 1e9))
            if stop_after is not None and i >= stop_after:
                stopped = True
                break
    except BaseException:
        os.close(fd)
        if prog_path is None:
            os.remove(path)
        raise
    os.close(fd)
    return {i: d["sha256"] for i, d in done.items()}, written, n_written, stopped


def records(g, shas):
    out = []
    for i, b in enumerate(g["bank"]):
        for mini, e, r in minis(g, b):
            out.append({"bank_layer": i, "mini": mini, "expert": e, "rank": r,
                        "sidecar_offset": b["base_offset"] + mini * b["record_bytes"], "sha256": shas[i][mini]})
    return out


def verify(path, g, read, recs, which):
    if which == "all":
        picks = range(len(recs))
    else:
        n = max(2, min(int(which), len(recs)))
        picks = sorted({round(k * (len(recs) - 1) / (n - 1)) for k in range(n)})
    bad = 0
    fd = os.open(path, os.O_RDONLY)
    for k in picks:
        r = recs[k]
        b = g["bank"][r["bank_layer"]]
        buf = bytearray(b["logical_bytes"])
        got = record_at(fd, b, r["mini"])
        sha = read(b, r["mini"], r["expert"], r["rank"], buf)
        if got != bytes(buf) or hashlib.sha256(got).hexdigest() != r["sha256"] or sha != r["sha256"]:
            if g["kind"] == "affine":
                print("verify: layer %d expert %d differs" % (b["layer"], r["expert"]))
            else:
                print("verify: bank layer %d mini %d (layer %d expert %d rank %d) differs" % (
                    r["bank_layer"], r["mini"], b["layer"], r["expert"], r["rank"]))
            bad += 1
    os.close(fd)
    return {"all_pass": bad == 0, "checked": len(picks), "total": len(recs), "method": "bytes-equal-source"}


# ── manifests: the contracts plugins/glm_moe_dsa/bank.zig and exl3_bank.zig read ──

def affine_manifest(g, recs, parity, repo, revision):
    layers = [{"layer": b["layer"], "index": i, "record_bytes": b["record_bytes"], "logical_bytes": b["logical_bytes"],
               "base_offset": b["base_offset"], "segments": b["segments"]} for i, b in enumerate(g["bank"])]
    out = [{"layer": g["bank"][r["bank_layer"]]["layer"], "index": r["bank_layer"], "expert": r["expert"],
            "sidecar_offset": r["sidecar_offset"], "record_bytes": g["bank"][r["bank_layer"]]["record_bytes"],
            "logical_bytes": g["bank"][r["bank_layer"]]["logical_bytes"], "sha256": r["sha256"]} for r in recs]
    return {"format": AFFINE["format"], "model_type": "glm_moe_dsa", "source": {"repo": repo, "revision": revision},
            "quantization": {"mode": "affine", "bits": g["bits"], "group_size": 64},
            "dims": {"hidden": g["hidden"], "inter": g["inter"], "n_experts": g["n_experts"], "n_layers": len(layers)},
            "components": AFFINE_COMPONENTS, "layers": layers,
            "sidecar": {"file": SIDECAR, "alignment": ALIGN, "size": g["size"]}, "records": out, "parity": parity}


def exl3_manifest(g, mcg, recs, parity, repo, revision, sidecar=SIDECAR):
    bank = [{k: v for k, v in b.items() if k != "kind"} for b in g["bank"]]
    return {"format": EXL3["format"], "model_type": "glm_moe_dsa", "source": {"repo": repo, "revision": revision},
            "quantization": {"mode": "exl3", "codebook": "mcg", "codebook_multiplier": g["multiplier"], "mcg_scalar": mcg,
                             "k_values": g["k_values"], "tp_ranks": g["tp"]},
            "dims": {"hidden": g["hidden"], "inter": g["inter"], "mini_inter": g["mini_inter"], "n_experts": g["n_experts"],
                     "n_model_layers": len(g["layers"]), "n_bank_layers": len(bank)},
            "components": EXL3_COMPONENTS, "layers": bank, "experts": g["experts"],
            "sidecar": {"file": sidecar, "alignment": ALIGN, "size": g["size"]}, "records": recs, "parity": parity}


# ── the three commands ──

def copy_files(src, dst, files):
    for f in files:
        if os.path.exists(os.path.join(src, f)):
            if os.path.lexists(os.path.join(dst, f)):
                os.remove(os.path.join(dst, f))
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))


def link_residents(res, dst):
    """Hard links of the affine pack's config, index and shards (same filesystem)."""
    files = ["config.json", INDEX]
    for f in files:
        if not os.path.isfile(os.path.join(res, f)):
            raise Refused("residents pack %s has no %s" % (res, f))
    files += sorted(set(json.load(open(os.path.join(res, INDEX)))["weight_map"].values()))
    for f in files[2:]:
        if not os.path.isfile(os.path.join(res, f)):
            raise Refused("residents pack %s has no %s" % (res, f))
    if not same_device(res, dst):
        raise Refused("residents pack %s is on another filesystem than %s" % (res, dst))
    return files


def mtp_names(srcs, g):
    prefix = tuple("model.layers.%d." % L for L in g["mtp"])
    return [n for n in srcs.order if n.startswith(prefix) and not EXPERT_RE.match(n)]


def convert(a):
    """`affine` and `exl3`: the whole pack."""
    t0 = time.time()
    if a.cmd == "affine":
        need(a.src, "config.json")
        cfg = json.load(open(os.path.join(a.src, "config.json")))
        g = affine_geometry(cfg, a.bits)
        srcs = Source(a.src)
        check_affine(srcs, g)
        mcg, res_files = None, None
    else:
        if a.residents_from is None:
            raise Refused("exl3 takes --residents-from <affine pack>")
        need(a.src, "config.json", TIER)
        g = exl3_geometry(json.load(open(os.path.join(a.src, "config.json"))), json.load(open(os.path.join(a.src, TIER))))
        res_files = link_residents(a.residents_from, a.dst)
        srcs = Source(a.src)
        mcg = check_exl3(srcs, g)
    os.makedirs(a.dst, exist_ok=True)
    prog_path = os.path.join(a.dst, PROGRESS)
    res_written = 0
    try:
        shas, exp_written, n_written, stopped = write_bank(os.path.join(a.dst, SIDECAR), g, reader(srcs, g), a.resume,
                                                           prog_path, a.stop_after_layer)
        if stopped:
            print("stopped after bank layer %d" % a.stop_after_layer)
            return 0
        prog = json.load(open(prog_path))
        if not prog.get("residents"):
            if a.cmd == "affine":
                res_written = srcs.write_shards(a.dst, [n for n in srcs.order if SWITCH not in n], a.shard_bytes)
                write_json(os.path.join(a.dst, "config.json"), {k: v for k, v in cfg.items() if k != "model_file"})
                copy_files(a.src, a.dst, COPIED)
            else:
                for f in os.listdir(a.dst):
                    if SHARD_RE.match(f) and f not in res_files:
                        os.remove(os.path.join(a.dst, f))
                for f in res_files:
                    s, d = os.path.join(a.residents_from, f), os.path.join(a.dst, f)
                    if os.path.lexists(d) and not os.path.samefile(s, d):
                        os.remove(d)
                    if not os.path.lexists(d):
                        os.link(s, d)
                res_written = srcs.write(os.path.join(a.dst, MTP_RESIDENTS), mtp_names(srcs, g))
                copy_files(a.src, a.dst, COPIED + [TIER])
            write_json(prog_path, dict(prog, residents=True))
        recs = records(g, shas)
        parity = {"all_pass": False, "checked": 0, "total": len(recs), "method": "bytes-equal-source"}
        if a.verify:
            parity = verify(os.path.join(a.dst, SIDECAR), g, reader(srcs, g), recs, a.verify)
        if a.cmd == "affine":
            write_json(os.path.join(a.dst, AFFINE["manifest"]), affine_manifest(g, recs, parity, a.source_repo, a.source_revision))
        else:
            write_json(os.path.join(a.dst, EXL3["manifest"]), exl3_manifest(g, mcg, recs, parity, a.source_repo, a.source_revision))
    finally:
        srcs.close()
    wall = max(time.time() - t0, 1e-9)
    written = exp_written + res_written
    write_json(os.path.join(a.dst, REPORT), {
        "bytes_written": written, "expert_bytes_written": exp_written, "resident_bytes_written": res_written,
        "records": parity["total"], "records_written": n_written, "wall_s": round(wall, 3),
        "gb_per_s": round(written / wall / 1e9, 3), "verify": parity})
    print("records %d (written %d), bank layers %d, %.1f s, %.2f GB/s, parity all_pass=%s checked=%d/%d" % (
        parity["total"], n_written, len(g["bank"]), wall, written / wall / 1e9, str(parity["all_pass"]).lower(),
        parity["checked"], parity["total"]))
    return 0 if (not a.verify or parity["all_pass"]) else 1


def convert_mtp(a):
    """`mtp`: the MTP layer alone, from an EXL3 pack (its records checked against its manifest) or a snapshot."""
    if (a.from_pack is None) == (a.src is None):
        raise Refused("mtp takes one of --from-pack and --src")
    srcs, pfd, want = None, None, None
    if a.from_pack is not None:
        need(a.from_pack, EXL3["manifest"], SIDECAR, MTP_RESIDENTS)
        pm = json.load(open(os.path.join(a.from_pack, EXL3["manifest"])))
        if pm.get("format") != EXL3["format"]:
            raise Refused("%s format %r is not %s" % (EXL3["manifest"], pm.get("format"), EXL3["format"]))
        if os.path.getsize(os.path.join(a.from_pack, SIDECAR)) != pm["sidecar"]["size"]:
            raise Refused("%s is not the %d bytes the manifest gives" % (SIDECAR, pm["sidecar"]["size"]))
        if not same_device(a.from_pack, a.dst):
            raise Refused("EXL3 pack %s is on another filesystem than %s" % (a.from_pack, a.dst))
        q, d = pm["quantization"], pm["dims"]
        full = {"kind": "exl3", "hidden": d["hidden"], "inter": d["inter"], "mini_inter": d["mini_inter"],
                "tp": q["tp_ranks"], "n_experts": d["n_experts"], "k_values": q["k_values"],
                "multiplier": q["codebook_multiplier"], "bank": pm["layers"], "experts": pm["experts"],
                "layers": sorted({b["layer"] for b in pm["layers"]}), "mtp": sorted({b["layer"] for b in pm["layers"] if b["mtp"]})}
        g, mcg = mtp_only(full), q["mcg_scalar"]
        repo = a.source_repo if a.source_repo is not None else pm["source"]["repo"]
        revision = a.source_revision if a.source_revision is not None else pm["source"]["revision"]
        old = {(b["layer"], b["k"]): b for b in full["bank"]}
        pack_sha = {(r["bank_layer"], r["mini"]): r["sha256"] for r in pm["records"]}
        pfd = os.open(os.path.join(a.from_pack, SIDECAR), os.O_RDONLY)

        def read(b, mini, e, r, buf):
            piece = record_at(pfd, old[(b["layer"], b["k"])], mini)
            buf[:b["logical_bytes"]] = piece
            return hashlib.sha256(piece).hexdigest()

        def want(b, mini):
            return pack_sha[(old[(b["layer"], b["k"])]["bank_layer"], mini)]
    else:
        need(a.src, "config.json", TIER)
        g = mtp_only(exl3_geometry(json.load(open(os.path.join(a.src, "config.json"))),
                                   json.load(open(os.path.join(a.src, TIER)))))
        srcs = Source(a.src)
        mcg, repo, revision = check_exl3(srcs, g, set(g["mtp"])), a.source_repo, a.source_revision
        read = reader(srcs, g)
    os.makedirs(a.dst, exist_ok=True)
    try:
        shas, written, _, _ = write_bank(os.path.join(a.dst, MTP_SIDECAR), g, read, want_sha=want)
        path = os.path.join(a.dst, MTP_RESIDENTS)
        if os.path.lexists(path):
            os.remove(path)
        if srcs is None:
            os.link(os.path.join(a.from_pack, MTP_RESIDENTS), path)
        else:
            written += srcs.write(path, mtp_names(srcs, g))
        recs = records(g, shas)
        parity = {"all_pass": False, "checked": 0, "total": len(recs), "method": "bytes-equal-source"}
        if a.verify:
            parity = verify(os.path.join(a.dst, MTP_SIDECAR), g, read, recs, a.verify)
        write_json(os.path.join(a.dst, MTP_MANIFEST), exl3_manifest(g, mcg, recs, parity, repo, revision, MTP_SIDECAR))
    finally:
        if srcs is not None:
            srcs.close()
        if pfd is not None:
            os.close(pfd)
    print("mtp: records %d, bank layers %d, parity all_pass=%s checked=%d/%d" % (
        parity["total"], len(g["bank"]), str(parity["all_pass"]).lower(), parity["checked"], parity["total"]))
    return 0 if (not a.verify or parity["all_pass"]) else 1


def size_arg(s):
    m = re.fullmatch(r"(\d+)\s*(|B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("bad size %r" % s)
    return int(m.group(1)) * {"": 1, "B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40,
                              "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "TB": 10 ** 12}[m.group(2)]


def verify_arg(s):
    if s != "all" and not (s.isdigit() and int(s) > 0):
        raise argparse.ArgumentTypeError("--verify takes N > 0 or all")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ["affine", "exl3", "mtp"]:
        p = sub.add_parser(cmd)
        p.add_argument("--src", required=cmd != "mtp")
        p.add_argument("--dst", required=True)
        p.add_argument("--verify", type=verify_arg)
        p.add_argument("--source-repo")
        p.add_argument("--source-revision")
        if cmd == "mtp":
            p.add_argument("--from-pack")
            continue
        p.add_argument("--resume", action="store_true")
        p.add_argument("--stop-after-layer", type=int)
        if cmd == "affine":
            p.add_argument("--bits", default="auto", choices=["auto", "3", "4"])
            p.add_argument("--shard-bytes", type=size_arg, default=5 << 30)
        else:
            p.add_argument("--residents-from")
    a = ap.parse_args(argv)
    try:
        return convert_mtp(a) if a.cmd == "mtp" else convert(a)
    except Refused as e:
        print("glm53_pack: refused: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())


# ── tests: synthetic snapshots of random bytes (the packer copies bytes, never decodes them) ──

import unittest  # noqa: E402


class _Fixture:
    HIDDEN, INTER, N_EXP, TP, VOCAB = 128, 64, 4, 4, 256

    @staticmethod
    def write_st(path, tensors):
        header, off = {}, 0
        for n, (dtype, shape, b) in tensors.items():
            header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(b)]}
            off += len(b)
        hb = json.dumps(header).encode()
        hb += b" " * (-len(hb) % 8)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + b"".join(v[2] for v in tensors.values()))

    @staticmethod
    def read_st(path):
        data = open(path, "rb").read()
        (n,) = struct.unpack("<Q", data[:8])
        h = json.loads(data[8:8 + n])
        h.pop("__metadata__", None)
        return {k: (v["dtype"], v["shape"], data[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]])
                for k, v in h.items()}

    @classmethod
    def snapshot(cls, path, t, cfg, split):
        """Two shards (`split(name)` picks the first), an index, config.json and the copied files."""
        os.makedirs(path, exist_ok=True)
        wm = {}
        for k, pred in enumerate([split, lambda n: not split(n)]):
            fn = "model-%05d-of-00002.safetensors" % (k + 1)
            part = {n: v for n, v in reversed(list(t.items())) if pred(n)}
            cls.write_st(os.path.join(path, fn), part)
            wm.update({n: fn for n in part})
        json.dump({"metadata": {"total_size": sum(len(v[2]) for v in t.values())}, "weight_map": wm},
                  open(os.path.join(path, INDEX), "w"))
        json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
        for fn in COPIED:
            open(os.path.join(path, fn), "w").write("source " + fn)
        open(os.path.join(path, "README.md"), "w").write("readme")


def _rand(rng, dtype, shape):
    return rng.randbytes(nbytes(dtype, shape))


def _run(*args):
    import subprocess
    return subprocess.run([sys.executable, "-I", os.path.abspath(__file__)] + [str(x) for x in args],
                          capture_output=True, text=True)


class AffineTest(unittest.TestCase):
    TYPES = ["dense", "dense", "sparse", "sparse", "sparse"]

    def setUp(self):
        import random
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        F, rng = _Fixture, random.Random(0)
        self.src, self.dst = os.path.join(self.tmp.name, "src"), os.path.join(self.tmp.name, "dst")
        t = {"model.embed_tokens.weight": ("BF16", [F.VOCAB, F.HIDDEN], None), "model.norm.weight": ("BF16", [F.HIDDEN], None)}
        shapes = {"gate": (F.INTER, F.HIDDEN), "up": (F.INTER, F.HIDDEN), "down": (F.HIDDEN, F.INTER)}
        for i, kind in enumerate(self.TYPES):
            t["model.layers.%d.input_layernorm.weight" % i] = ("BF16", [F.HIDDEN], None)
            if kind == "sparse":
                t["model.layers.%d.mlp.gate.weight" % i] = ("BF16", [F.N_EXP, F.HIDDEN], None)
                for c in AFFINE_COMPONENTS:
                    p, part = c.split(".")
                    out, inn = shapes[p]
                    t["model.layers.%d.mlp.switch_mlp.%s_proj.%s" % (i, p, part)] = (
                        ("U32", [F.N_EXP, out, inn * 4 // 32], None) if part == "weight" else ("BF16", [F.N_EXP, out, inn // 64], None))
        self.t = {n: (d, s, _rand(rng, d, s)) for n, (d, s, _) in t.items()}
        cfg = {"model_type": "glm_moe_dsa", "hidden_size": F.HIDDEN, "moe_intermediate_size": F.INTER,
               "n_routed_experts": F.N_EXP, "mlp_layer_types": self.TYPES, "model_file": "glm_moe_dsa.py",
               "quantization": {"group_size": 64, "bits": 4}}
        # layer 2's gate_proj sits in shard 1 and the rest of layer 2 in shard 2: a layer spans two files.
        F.snapshot(self.src, self.t, cfg, lambda n: not n.startswith("model.layers.") or int(n.split(".")[2]) < 2
                   or ".switch_mlp.gate_proj." in n and n.startswith("model.layers.2."))

    def tearDown(self):
        self.tmp.cleanup()

    def pack(self, dst, *args):
        r = _run("affine", "--src", self.src, "--dst", dst, "--shard-bytes", "64KiB", "--source-repo", "test/glm",
                 "--source-revision", "0" * 40, *args)
        return r

    def test_records_manifest_residents(self):
        r = self.pack(self.dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(os.path.join(self.dst, AFFINE["manifest"])))
        self.assertEqual((m["format"], m["quantization"]), (AFFINE["format"], {"mode": "affine", "bits": 4, "group_size": 64}))
        self.assertEqual(m["components"], ["gate.weight", "gate.scales", "gate.biases", "up.weight", "up.scales",
                                           "up.biases", "down.weight", "down.scales", "down.biases"])
        self.assertEqual([(l["layer"], l["index"]) for l in m["layers"]], [(2, 0), (3, 1), (4, 2)])
        bank = open(os.path.join(self.dst, SIDECAR), "rb").read()
        self.assertEqual(len(bank), m["sidecar"]["size"])
        for rec in m["records"]:
            layer = m["layers"][rec["index"]]
            want = b"".join(self.t["model.layers.%d.mlp.switch_mlp.%s_proj.%s" % (rec["layer"], *s["component"].split("."))][2]
                            [rec["expert"] * s["length"]:(rec["expert"] + 1) * s["length"]] for s in layer["segments"])
            off = rec["sidecar_offset"]
            self.assertEqual(off, layer["base_offset"] + rec["expert"] * layer["record_bytes"])
            self.assertEqual(bank[off:off + rec["logical_bytes"]], want)
            self.assertEqual(rec["sha256"], hashlib.sha256(want).hexdigest())
        got = {}
        idx = json.load(open(os.path.join(self.dst, INDEX)))
        for f in set(idx["weight_map"].values()):
            got.update(_Fixture.read_st(os.path.join(self.dst, f)))
        self.assertEqual(got, {n: v for n, v in self.t.items() if SWITCH not in n})
        self.assertNotIn("model_file", json.load(open(os.path.join(self.dst, "config.json"))))
        self.assertFalse(os.path.exists(os.path.join(self.dst, "README.md")))

    def test_resume_and_verify(self):
        part = os.path.join(self.tmp.name, "part")
        r = self.pack(part, "--stop-after-layer", "0")
        self.assertIn("stopped after bank layer 0", r.stdout)
        self.assertEqual(self.pack(part, "--resume").returncode, 0)
        self.assertEqual(self.pack(self.dst).returncode, 0)
        for f in [SIDECAR, AFFINE["manifest"], INDEX]:
            self.assertEqual(open(os.path.join(part, f), "rb").read(), open(os.path.join(self.dst, f), "rb").read(), f)
        m = json.load(open(os.path.join(part, AFFINE["manifest"])))
        b = m["layers"][1]
        with open(os.path.join(part, SIDECAR), "r+b") as f:
            f.seek(b["base_offset"] + b["record_bytes"] + b["logical_bytes"] // 2)
            v = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([v[0] ^ 0xFF]))
        r = self.pack(part, "--resume", "--verify", "all")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("verify: layer 3 expert 1 differs", r.stdout)

    def test_refuses_model_type(self):
        cfg = json.load(open(os.path.join(self.src, "config.json")))
        json.dump(dict(cfg, model_type="deepseek_v41"), open(os.path.join(self.src, "config.json"), "w"))
        r = self.pack(self.dst)
        self.assertEqual((r.returncode, r.stderr.strip()), (2, "glm53_pack: refused: model_type 'deepseek_v41' is not glm_moe_dsa"))


class Exl3Test(unittest.TestCase):
    TYPES, MTP, ROUTED = ["dense", "dense", "sparse", "sparse"], 4, [2, 3, 4]
    TIER = {"2": [3, 4, 3, 4], "3": [4, 3, 3, 3], "4": [4, 4, 3, 4]}
    MCG = struct.unpack("<i", struct.pack("<I", 0xCBAC1FED))[0]

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.src, self.res, self.dst = (os.path.join(self.tmp.name, d) for d in ["src", "res", "dst"])
        self.t = self.source(self.src)
        F = _Fixture
        os.makedirs(self.res)
        wm = {}
        for k in range(2):
            fn = "model-%05d-of-00002.safetensors" % (k + 1)
            F.write_st(os.path.join(self.res, fn), {"resident.%d" % k: ("U32", [4], bytes(range(16)))})
            wm["resident.%d" % k] = fn
        json.dump({"weight_map": wm}, open(os.path.join(self.res, INDEX), "w"))
        json.dump({"model_type": "glm_moe_dsa"}, open(os.path.join(self.res, "config.json"), "w"))

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, path, tweak=None):
        import random
        F, rng = _Fixture, random.Random(0)
        g = {"hidden": F.HIDDEN, "mini_inter": F.INTER // F.TP}
        t = {"model.embed_tokens.weight": ("BF16", [F.VOCAB, F.HIDDEN], _rand(rng, "BF16", [F.VOCAB, F.HIDDEN]))}
        for L in range(len(self.TYPES) + 1):
            t["model.layers.%d.input_layernorm.weight" % L] = ("BF16", [F.HIDDEN], _rand(rng, "BF16", [F.HIDDEN]))
            if L not in self.ROUTED:
                continue
            for e in range(F.N_EXP):
                for p in PROJS:
                    for r in range(F.TP):
                        for part, (d, s) in exl3_shapes(g, p, self.TIER[str(L)][e]).items():
                            t["model.layers.%d.mlp.experts.%d.%s.rank%d.%s" % (L, e, p, r, part)] = (
                                d, s, struct.pack("<i", self.MCG) if part == "mcg" else _rand(rng, d, s))
        t["model.layers.4.eh_proj.weight"] = ("BF16", [F.HIDDEN, 2 * F.HIDDEN], _rand(rng, "BF16", [F.HIDDEN, 2 * F.HIDDEN]))
        if tweak:
            tweak(t)
        cfg = {"model_type": "glm_moe_dsa", "hidden_size": F.HIDDEN, "moe_intermediate_size": F.INTER,
               "n_routed_experts": F.N_EXP, "num_hidden_layers": len(self.TYPES), "num_nextn_predict_layers": 1,
               "mlp_layer_types": self.TYPES, "hybrid_tr3_tail": {"codebook": "mcg", "mcg_multiplier": 0xCBAC1FED,
                                                                  "k_values": [3, 4], "tp": F.TP}}
        F.snapshot(path, t, cfg, lambda n: not n.startswith("model.layers.") or int(n.split(".")[2]) < 3
                   or n.startswith(("model.layers.3.mlp.experts.0.", "model.layers.3.mlp.experts.1.")))
        json.dump({L: {"k": ks} for L, ks in self.TIER.items()}, open(os.path.join(path, TIER), "w"))
        return t

    def pack(self, src, dst, *args):
        return _run("exl3", "--src", src, "--dst", dst, "--residents-from", self.res, "--source-repo", "test/x",
                    "--source-revision", "0" * 40, *args)

    def test_records_manifest_residents(self):
        r = self.pack(self.src, self.dst, "--verify", "all")
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(os.path.join(self.dst, EXL3["manifest"])))
        self.assertEqual([(b["layer"], b["k"], b["mtp"]) for b in m["layers"]],
                         [(2, 3, False), (2, 4, False), (3, 3, False), (3, 4, False), (4, 3, True), (4, 4, True)])
        self.assertEqual(m["experts"]["3"], [[4, 0], [3, 0], [3, 1], [3, 2]])
        self.assertEqual(m["quantization"]["mcg_scalar"], self.MCG)
        self.assertEqual(m["parity"], {"all_pass": True, "checked": 48, "total": 48, "method": "bytes-equal-source"})
        bank = open(os.path.join(self.dst, SIDECAR), "rb").read()
        for rec in m["records"]:
            b = m["layers"][rec["bank_layer"]]
            want = b"".join(self.t["model.layers.%d.mlp.experts.%d.%s.rank%d.%s" % (
                b["layer"], rec["expert"], s["component"].split(".")[0], rec["rank"], EXL3_PART[s["component"].split(".")[1]])][2]
                for s in b["segments"])
            self.assertEqual(bank[rec["sidecar_offset"]:rec["sidecar_offset"] + b["logical_bytes"]], want)
        for f in ["config.json", INDEX, "model-00001-of-00002.safetensors"]:
            self.assertEqual(os.stat(os.path.join(self.dst, f)).st_ino, os.stat(os.path.join(self.res, f)).st_ino, f)
        self.assertEqual(sorted(_Fixture.read_st(os.path.join(self.dst, MTP_RESIDENTS))),
                         ["model.layers.4.eh_proj.weight", "model.layers.4.input_layernorm.weight"])

    def test_mtp_from_pack_equals_from_snapshot(self):
        self.assertEqual(self.pack(self.src, self.dst).returncode, 0)
        a, b = os.path.join(self.tmp.name, "a"), os.path.join(self.tmp.name, "b")
        r = _run("mtp", "--dst", a, "--from-pack", self.dst, "--verify", "all")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = _run("mtp", "--dst", b, "--src", self.src, "--source-repo", "test/x", "--source-revision", "0" * 40)
        self.assertEqual(r.returncode, 0, r.stderr)
        for f in [MTP_RESIDENTS, MTP_SIDECAR]:
            self.assertEqual(open(os.path.join(a, f), "rb").read(), open(os.path.join(b, f), "rb").read(), f)
        ma, mb = (json.load(open(os.path.join(d, MTP_MANIFEST))) for d in (a, b))
        self.assertEqual((ma["sidecar"]["file"], ma["dims"]["n_bank_layers"]), (MTP_SIDECAR, 2))
        self.assertEqual(ma["records"], mb["records"])

    def test_refuses_mixed_k(self):
        name = "model.layers.3.mlp.experts.1.down_proj.rank2.trellis"
        src = os.path.join(self.tmp.name, "bad")
        self.source(src, lambda t: t.__setitem__(name, ("I16", [1, 8, 64], bytes(nbytes("I16", [1, 8, 64])))))
        r = self.pack(src, self.dst)
        self.assertEqual((r.returncode, r.stderr.strip()), (2, "glm53_pack: refused: model.layers.3.mlp.experts.1 has "
                                                               "trellis K [3, 4] across its ranks and projections"))
