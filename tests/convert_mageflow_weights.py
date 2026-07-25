#!/usr/bin/env python3
"""Repackage a released MageFlow repo as an 8-bit mlx-serve mirror.

USER-RUN (needs mlx + the ~16 GB HF checkpoint). Not run in CI. Produces the
diffusers-shaped dir `src/mage_flow.zig` loads, with two changes:

  (a) DiT + text-encoder linears affine-quantized (default 8-bit, group 64), so
      the DiT/TE roughly halve. `MfLinear` in the engine picks dense-vs-quantized
      PER TENSOR from the presence of a `.scales` sibling, so a partially
      quantized checkpoint (see --keep-bf16) loads with no flag anywhere.
  (b) Everything the engine never reads is dropped — see PRUNING below.

    <out>/model_index.json                    verbatim (classification keys on it)
    <out>/scheduler/scheduler_config.json     verbatim
    <out>/transformer/config.json             verbatim
    <out>/transformer/diffusion_pytorch_model.safetensors    8-bit
    <out>/text_encoder/{config,preprocessor_config}.json     verbatim
    <out>/text_encoder/tokenizer.json         verbatim
    <out>/text_encoder/model.safetensors      8-bit (one file, no index)
    <out>/vae/config.json                     verbatim
    <out>/vae/diffusion_pytorch_model.safetensors            bf16, UNQUANTIZED
    <out>/{LICENSE,README.md}

WHAT IS NOT QUANTIZED, and why:

  - The whole VAE. Its precision is load-bearing (the engine runs it in f32) and
    it is only ~345 MB, so there is nothing to win and a real risk to take.
  - Lookup tables (`NEVER_QUANTIZE`). `embed_tokens.weight` and
    `visual.pos_embed.weight` are read with `mlx_take_axis`, NOT a matmul — a
    packed uint32 table would gather garbage rows, so these stay bf16 even
    though they are 2-D and pass every other test. This is the one rule that
    cannot be derived from a tensor's shape.
  - Anything with min(out_features, in_features) < 512. A row covering only 2 or
    4 groups is the lossiest thing to quantize and these are tiny anyway (img_in,
    proj_out, timestep_embedder.linear_1 — ~3 MB combined in the DiT).
  - Any tensor whose input dim is not a multiple of the group size (mlx cannot
    quantize it), and anything that is not rank 2.

PRUNING (a wrong prune fails LOUDLY at load — `ownWeight` errors on a missing
key — so this is safe to be aggressive about):

  - `pipeline.y_embedder.encoder.*` (~69 MB): the VAE's training-time encoder
    half. Nothing in src/mage_flow.zig loads it, in either mode.
  - `model.visual.*` (~830 MB) and `student.dconv_encoder.*` (~132 MB): loaded
    only when the engine is in EDIT mode (`VisionTower`/`VaeEncoder`), so they
    are dropped unless --edit.
  - The 8 repo files the engine never opens: chat_template.json (prompt templates
    are hardcoded constants), tokenizer_config.json, generation_config.json,
    video_preprocessor_config.json, vocab.json + merges.txt (`loadTokenizerAny`
    prefers tokenizer.json), .gitattributes, and assets/ (~41 MB of README
    images).

The output dir NAME decides edit capability (`dirIsEdit` matches a case-insensitive
`mage-flow-edit`), so keep "Mage-Flow-Edit" in the name of an --edit build.

Usage:
    python3 tests/convert_mageflow_weights.py --src <repo dir> --out <dir> [--edit]
    python3 tests/convert_mageflow_weights.py --src <dir> --dry-run   # manifest only, no mlx
    python3 tests/convert_mageflow_weights.py --self-test             # no ckpt/mlx needed

MIT-licensed upstream (microsoft/Mage-Flow-Turbo, microsoft/Mage-Flow-Edit-Turbo).
"""

import argparse
import glob
import json
import os
import shutil
import struct
import sys

GROUP_SIZE = 64
DEFAULT_BITS = 8
MIN_DIM = 512

# Read with mlx_take_axis, never a matmul — see the module docstring.
NEVER_QUANTIZE = (
    "embed_tokens.weight",
    "visual.pos_embed.weight",
    "patch_embed.proj.weight",
)

# Dead in both modes: the VAE's training-time encoder half.
DROP_ALWAYS = ("pipeline.y_embedder.encoder.",)
# Loaded only by VisionTower / VaeEncoder, i.e. only in edit mode.
DROP_UNLESS_EDIT = ("model.visual.", "student.dconv_encoder.")

# Everything the engine opens, per component dir (see src/mage_flow.zig:145-227,
# src/model_discovery.zig:153, tokenizer.zig loadTokenizerAny).
KEEP_FILES = {
    "": ["model_index.json"],
    "scheduler": ["scheduler_config.json"],
    "transformer": ["config.json"],
    "vae": ["config.json"],
    "text_encoder": ["config.json", "preprocessor_config.json", "tokenizer.json"],
}
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE")


def should_quantize(name, shape, bits, keep_bf16=()):
    """A `.weight` is quantized iff it is a real 2-D linear: rank 2, input dim a
    multiple of GROUP_SIZE, both dims >= MIN_DIM, not a lookup table, and not
    held back by --keep-bf16."""
    if bits == 16:
        return False
    if not name.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    if any(t in name for t in NEVER_QUANTIZE):
        return False
    if any(t and t in name for t in keep_bf16):
        return False
    out_f, in_f = shape[0], shape[1]
    if in_f % GROUP_SIZE != 0:
        return False
    return min(out_f, in_f) >= MIN_DIM


def should_drop(name, edit):
    """True when the engine never loads this tensor in the target mode."""
    if any(name.startswith(p) for p in DROP_ALWAYS):
        return True
    if not edit and any(name.startswith(p) for p in DROP_UNLESS_EDIT):
        return True
    return False


def quantized_nbytes(shape, bits):
    """Packed weight + bf16 scales + bf16 biases, in bytes."""
    out_f, in_f = shape
    return out_f * in_f * bits // 8 + 2 * 2 * out_f * (in_f // GROUP_SIZE)


# ── safetensors header reading (dry-run needs no mlx and no big memory) ───────
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def component_files(src, component):
    """The weight shards of one component dir, in a stable order."""
    d = os.path.join(src, component)
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise SystemExit(f"[error] no .safetensors under {d}")
    return files


def plan_component(src, component, bits, edit, keep_bf16):
    """Decide keep/drop/quantize for every tensor, without loading any data."""
    quantize_here = component in ("transformer", "text_encoder")
    rows = []
    for path in component_files(src, component):
        for name, meta in read_header(path).items():
            shape = meta["shape"]
            nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
            if should_drop(name, edit):
                rows.append((name, shape, nbytes, "drop", 0))
            elif quantize_here and should_quantize(name, shape, bits, keep_bf16):
                rows.append((name, shape, nbytes, "quant", quantized_nbytes(shape, bits)))
            else:
                rows.append((name, shape, nbytes, "keep", nbytes))
    return rows


def print_manifest(component, rows):
    kinds = {}
    for _, _, src_b, kind, out_b in rows:
        n, s, o = kinds.get(kind, (0, 0, 0))
        kinds[kind] = (n + 1, s + src_b, o + out_b)
    src_total = sum(v[1] for v in kinds.values())
    out_total = sum(v[2] for v in kinds.values())
    print(f"  {component or '(root)'}:")
    for kind in ("quant", "keep", "drop"):
        if kind in kinds:
            n, s, o = kinds[kind]
            print(f"    {kind:<6} {n:5d} tensors  {s/1e9:7.3f} GB → {o/1e9:7.3f} GB")
    print(f"    {'TOTAL':<6} {len(rows):5d} tensors  {src_total/1e9:7.3f} GB → {out_total/1e9:7.3f} GB")
    return src_total, out_total


# ── conversion ───────────────────────────────────────────────────────────────
def convert_component(src, out, component, bits, edit, keep_bf16, out_name):
    import mlx.core as mx

    rows = plan_component(src, component, bits, edit, keep_bf16)
    decision = {r[0]: r[3] for r in rows}
    packed = {}
    for path in component_files(src, component):
        loaded = mx.load(path)
        for name, arr in loaded.items():
            kind = decision[name]
            if kind == "drop":
                continue
            if kind == "quant":
                wq, scales, biases = mx.quantize(arr, group_size=GROUP_SIZE, bits=bits)
                base = name[: -len(".weight")]
                packed[f"{base}.weight"] = wq
                packed[f"{base}.scales"] = scales
                packed[f"{base}.biases"] = biases
            else:
                packed[name] = arr
    mx.eval(*packed.values())
    dst_dir = os.path.join(out, component)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, out_name)
    mx.save_safetensors(dst, packed)
    written = os.path.getsize(dst)
    print(f"  wrote {dst} ({written/1e9:.3f} GB, {len(packed)} tensors)")
    return written


def copy_support_files(src, out):
    # Finder litters .DS_Store into any browsed dir and `upload-large-folder`
    # ships whatever it finds, so sweep before writing. Cheap insurance against
    # a model card whose file list has junk in it.
    for junk in glob.glob(os.path.join(out, "**", ".DS_Store"), recursive=True):
        os.remove(junk)

    for component, names in KEEP_FILES.items():
        dst_dir = os.path.join(out, component) if component else out
        os.makedirs(dst_dir, exist_ok=True)
        for name in names:
            s = os.path.join(src, component, name) if component else os.path.join(src, name)
            if not os.path.exists(s):
                # preprocessor_config.json is optional (the engine falls back to
                # its own VLM constants); everything else is required.
                if name == "preprocessor_config.json":
                    print(f"  [skip] {name} absent in source")
                    continue
                raise SystemExit(f"[error] required file missing from source: {s}")
            shutil.copy2(s, os.path.join(dst_dir, name))
    for name in LICENSE_NAMES:
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out, name))


README = """---
license: mit
base_model: {base}
base_model_relation: quantized
library_name: mlx-serve
tags:
  - mlx
  - mlx-serve
  - quantized
  - {pipeline}
pipeline_tag: {pipeline}
---

# {title}

8-bit mirror of [{base}](https://huggingface.co/{base}) for
[mlx-serve](https://github.com/ddalcu/mlx-serve). Half the download, half the
memory, same distilled 4-step schedule.

{sizes}

Judged against bf16 at the same seed on text-to-image, single-reference edits
and multi-reference composition: no visible quality difference.

## Run it

Download **[MLX Core.app](https://github.com/ddalcu/mlx-serve/releases/latest)**,
open the Image tab, and pick **{preset}** from the model menu.
It downloads with a progress bar and generates in the same window. No terminal,
nothing to configure.

Prefer Homebrew? It is a third-party tap, so tap it first:

```bash
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve
brew install --cask mlx-core
```

{use_line}

## mlx-serve

A native Zig server for Apple Silicon. No Python, no cloud, no Electron. One
9 MB binary.

- **One server, every modality.** Chat, images, video, music, speech with voice
  cloning, and 3D, all running natively on MLX.
- **Points at what you already use.** OpenAI- *and* Anthropic-compatible APIs on
  `http://localhost:11234`, so Claude Code, the OpenAI SDK, Continue, Cursor and
  Open WebUI just work.
- **Any LLM, not just these.** Every MLX model and every GGUF on Hugging Face,
  with speculative decoding built in.
- **MLX Core.app included.** Signed macOS menu-bar app: chat, agent mode with
  MCP tools, model downloads, and every generator above, no terminal needed.

[mlxserve.com](https://mlxserve.com/) · [GitHub](https://github.com/ddalcu/mlx-serve)

If it is useful to you, a star on
[GitHub](https://github.com/ddalcu/mlx-serve) genuinely helps.

## Recipe

DiT and text-encoder linears affine-quantized at {bits}-bit, group size {gs}, via
`mlx.core.quantize`. Left at bf16: the whole VAE (its precision is load-bearing
for a distilled 4-step model), the token and position embedding tables (they are
gathered, not matmul'd), and any linear with a dimension under {mindim}.

Dropped because the engine never loads them: the VAE's training-time encoder
half.{extra_drop}

Dropped too, the repo files it never opens: chat_template, tokenizer_config,
generation_config, video_preprocessor_config, vocab.json + merges.txt, assets.

Built by `tests/convert_mageflow_weights.py` in the mlx-serve repo. Original
model and weights by Microsoft, MIT licensed.
"""

USE_T2I = (
    "Driving it from code instead? The app runs the server on\n"
    "`http://localhost:11234`, so POST to `/v1/images/generations`."
)
USE_EDIT = (
    "Drop in a source image, type an instruction, generate. Add more references and\n"
    "it composes across them.\n"
    "\n"
    "Driving it from code instead? The app runs the server on\n"
    "`http://localhost:11234`, so POST to `/v1/images/edits`. That is the OpenAI\n"
    "image-edit shape, so the official SDK's `client.images.edit()` works as-is,\n"
    "with repeated `image[]` for multi-reference."
)


def write_readme(out, src_repo, edit, bits, sizes):
    title = os.path.basename(out.rstrip("/"))
    (open(os.path.join(out, "README.md"), "w")).write(
        README.format(
            base=src_repo,
            title=title,
            bits=bits,
            gs=GROUP_SIZE,
            mindim=MIN_DIM,
            sizes=sizes,
            use_line=USE_EDIT if edit else USE_T2I,
            pipeline="image-to-image" if edit else "text-to-image",
            preset="Mage-Flow Edit Turbo 8-bit" if edit else "Mage-Flow Turbo 8-bit",
            extra_drop="" if edit else "\nThis is the text-to-image build, so the vision tower and the\nreference-image encoder went with it.",
        )
    )


# ── self-test ────────────────────────────────────────────────────────────────
def self_test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("[self-test] should_quantize on the real DiT tensors")
    # (name, shape, expected) — shapes read from the released Turbo checkpoint.
    dit = [
        ("transformer_blocks.0.img_mod.1.weight", [18432, 3072], True),
        ("transformer_blocks.0.txt_mod.1.weight", [18432, 3072], True),
        ("transformer_blocks.0.img_mlp.net.0.proj.weight", [12288, 3072], True),
        ("transformer_blocks.0.img_mlp.net.2.weight", [3072, 12288], True),
        ("transformer_blocks.0.attn.to_q.weight", [3072, 3072], True),
        ("norm_out.linear.weight", [6144, 3072], True),
        ("txt_in.weight", [3072, 2560], True),
        ("time_text_embed.timestep_embedder.linear_2.weight", [3072, 3072], True),
        # too narrow to quantize well, and tiny
        ("img_in.weight", [3072, 128], False),
        ("proj_out.weight", [128, 3072], False),
        ("time_text_embed.timestep_embedder.linear_1.weight", [3072, 256], False),
        # 1-D norms and biases
        ("transformer_blocks.0.attn.to_q.bias", [3072], False),
        ("txt_norm.weight", [2560], False),
    ]
    for name, shape, want in dit:
        check(should_quantize(name, shape, 8) == want, f"{name} {shape} → {want}")

    print("[self-test] should_quantize on the real text-encoder tensors")
    te = [
        ("model.language_model.layers.0.self_attn.q_proj.weight", [4096, 2560], True),
        ("model.language_model.layers.0.self_attn.o_proj.weight", [2560, 4096], True),
        ("model.language_model.layers.0.mlp.gate_proj.weight", [9728, 2560], True),
        ("model.language_model.layers.0.mlp.down_proj.weight", [2560, 9728], True),
        ("model.visual.blocks.0.attn.qkv.weight", [3072, 1024], True),
        ("model.visual.merger.linear_fc1.weight", [4096, 4096], True),
        ("model.visual.merger.linear_fc2.weight", [2560, 4096], True),
        # LOOKUP TABLES — gathered with mlx_take_axis, must stay dense
        ("model.language_model.embed_tokens.weight", [151936, 2560], False),
        ("model.visual.pos_embed.weight", [2304, 1024], False),
        # rank-5 Conv3d patch embed
        ("model.visual.patch_embed.proj.weight", [1024, 3, 2, 16, 16], False),
        ("model.language_model.layers.0.input_layernorm.weight", [2560], False),
    ]
    for name, shape, want in te:
        check(should_quantize(name, shape, 8) == want, f"{name} {shape} → {want}")

    print("[self-test] --keep-bf16 holds back matching tensors")
    check(
        not should_quantize("transformer_blocks.3.img_mod.1.weight", [18432, 3072], 8, ("img_mod.1",)),
        "img_mod.1 held back by substring",
    )
    check(
        should_quantize("transformer_blocks.3.attn.to_q.weight", [3072, 3072], 8, ("img_mod.1",)),
        "unrelated tensor still quantized",
    )
    check(not should_quantize("transformer_blocks.0.attn.to_q.weight", [3072, 3072], 16), "bits=16 quantizes nothing")

    print("[self-test] should_drop matches the engine's conditional loads")
    drops = [
        ("pipeline.y_embedder.encoder.conv_in.weight", False, True),
        ("pipeline.y_embedder.encoder.conv_in.weight", True, True),   # dead in BOTH modes
        ("pipeline.y_embedder.decoder.conv_in.weight", False, False),
        ("student.dconv_encoder.proj_down.weight", False, True),
        ("student.dconv_encoder.proj_down.weight", True, False),      # VaeEncoder, edit only
        ("model.visual.blocks.0.attn.qkv.weight", False, True),
        ("model.visual.blocks.0.attn.qkv.weight", True, False),       # VisionTower, edit only
        ("model.language_model.layers.0.mlp.gate_proj.weight", False, False),
    ]
    for name, edit, want in drops:
        check(should_drop(name, edit) == want, f"{name} edit={edit} → drop={want}")

    print("[self-test] quantized_nbytes math")
    # [3072,3072] at 8 bits: 9.44 MB packed + bf16 scales/biases over 48 groups.
    n = quantized_nbytes([3072, 3072], 8)
    check(n == 3072 * 3072 + 2 * 2 * 3072 * 48, f"8-bit [3072,3072] = {n} bytes")
    check(quantized_nbytes([3072, 3072], 4) < n, "4-bit is smaller than 8-bit")
    check(n < 3072 * 3072 * 2, "8-bit beats bf16")

    print("\n[self-test] " + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="released MageFlow repo dir (model_index.json at its root)")
    ap.add_argument("--out", help="output dir; keep 'Mage-Flow-Edit' in the name for an --edit build")
    ap.add_argument("--bits", type=int, default=DEFAULT_BITS, choices=[4, 8, 16],
                    help="16 = repackage/prune only, no quantization")
    ap.add_argument("--edit", action="store_true",
                    help="keep the vision tower + reference-image encoder (edit checkpoints)")
    ap.add_argument("--keep-bf16", default="",
                    help="comma-separated substrings to leave dense, e.g. img_mod.1,txt_mod.1")
    ap.add_argument("--dry-run", action="store_true", help="print the manifest and exit (no mlx needed)")
    ap.add_argument("--self-test", action="store_true", help="run the hermetic unit tests")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.src:
        ap.error("--src is required (or use --self-test)")
    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")

    src = os.path.abspath(os.path.expanduser(args.src))
    if not os.path.exists(os.path.join(src, "model_index.json")):
        raise SystemExit(f"[error] {src} has no model_index.json — not a MageFlow repo")
    keep_bf16 = tuple(t.strip() for t in args.keep_bf16.split(",") if t.strip())

    mode = "EDIT" if args.edit else "text-to-image"
    print(f"[convert] {src}\n[convert] {mode}, {args.bits}-bit, group {GROUP_SIZE}"
          + (f", holding back {list(keep_bf16)}" if keep_bf16 else ""))

    src_total = out_total = 0
    for component in ("transformer", "text_encoder", "vae"):
        rows = plan_component(src, component, args.bits, args.edit, keep_bf16)
        s, o = print_manifest(component, rows)
        src_total += s
        out_total += o
    print(f"  WEIGHTS {src_total/1e9:.3f} GB → {out_total/1e9:.3f} GB "
          f"({100*out_total/src_total:.0f}%)")

    if args.dry_run:
        return 0

    out = os.path.abspath(os.path.expanduser(args.out))
    if args.edit and "mage-flow-edit" not in os.path.basename(out).lower().replace("_", "-"):
        print(f"[warn] --edit but '{os.path.basename(out)}' does not contain 'Mage-Flow-Edit' — "
              "the engine gates edit capability on the DIRECTORY NAME, so this build "
              "will come up in text-to-image mode")
    os.makedirs(out, exist_ok=True)

    written = 0
    written += convert_component(src, out, "transformer", args.bits, args.edit, keep_bf16,
                                 "diffusion_pytorch_model.safetensors")
    written += convert_component(src, out, "text_encoder", args.bits, args.edit, keep_bf16,
                                 "model.safetensors")
    # bits=16 for the VAE: precision is load-bearing there (the engine runs it f32).
    written += convert_component(src, out, "vae", 16, args.edit, keep_bf16,
                                 "diffusion_pytorch_model.safetensors")
    copy_support_files(src, out)

    src_repo = "microsoft/Mage-Flow-Edit-Turbo" if args.edit else "microsoft/Mage-Flow-Turbo"
    write_readme(out, src_repo, args.edit, args.bits,
                 f"Weights: {written/1e9:.1f} GB (upstream bf16: {src_total/1e9:.1f} GB).")
    print(f"[convert] done → {out} ({written/1e9:.3f} GB of weights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
