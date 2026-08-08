#!/usr/bin/env python3
"""Disk-safe fetch+convert driver for the MiniMax-H3 REF2VA checkpoint.

Ref2VA is the reference-conditioning partition: images, videos and audio that
the generation follows for character / style / scene continuity. It shares the
text encoder, both VAEs and every geometry number with FL2VA — only the DiT
differs — so the conversion is `convert_minimax_h3_weights.py --partition
ref2va` and this script exists purely to keep the SOURCE off the disk.

Sizes (8-bit output):
    DiT bf16 source     ~66 GB   ->  transformer.safetensors  ~35 GB
    text encoder bf16   ~57 GB   ->  text_encoder.safetensors ~28 GB
    VAEs                ~5.8 GB  copied verbatim

Both sources and both outputs together do not fit on a box with ~170 GB free,
so each stage downloads, converts, then DELETES its source before the next
begins. Peak source-on-disk is one file.

Resumable in both directions: `hf download` skips what is present, and the
converter skips an output that already exists (`--skip-existing`). Re-run the
same command after any interruption.

The tokenizer and, critically, the LICENSE come from the MiniMaxAI repo — the
converter treats a missing LICENSE as FATAL, because Section III.1 of the
MiniMax H3 Community License requires the Agreement to accompany any
distribution, and a directory that cannot legally be shared should not be
written in the first place.

    python3 tests/fetch_convert_minimax_h3_ref2va.py \
        --src ~/.mlx-serve/staging/minimax-h3-ref2va \
        --out ~/.mlx-serve/models/ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit

Before it can run, the REF2VA release must actually be published under the
filenames in COMFY_FILES below; check `PARTITIONS` in the converter if
upstream renames them.

TERRITORY: the Agreement's Applicable Territory EXCLUDES the EU, UK, South
Korea and the USA (Section V.4 covers use, reproduction, modification,
distribution AND display). This script downloads and modifies the Works, so
check your jurisdiction before running it.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

COMFY_REPO = "Comfy-Org/MiniMax-H3"
MINIMAX_REPO = "MiniMaxAI/MiniMax-H3"

# (label, repo-relative path, ~GiB) — converted one at a time, source deleted
# after each so two 60 GB files are never on disk together.
COMFY_FILES = [
    ("transformer", "diffusion_models/minimax_h3_ref2va_bf16.safetensors", 66),
    ("text_encoder", "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", 52),
]
VAE_FILES = [
    ("video_vae", "vae/minimax_h3_video_vae_fp16.safetensors", 5.2),
    ("audio_vae", "vae/minimax_h3_audio_vae_fp32.safetensors", 0.6),
]
# The tokenizer lives ONLY in the MiniMaxAI repo; the weights live ONLY in the
# Comfy-Org one. LICENSE must come with the weights it licenses.
#
# MiniMaxAI ships BOTH partitions' processors (`FL2VA/` and `Ref2VA/` — note the
# capitalization, which a case-sensitive listing filter will miss), and they are
# BYTE-IDENTICAL: `tokenizer.json` sha256 a5d85b6d… and `chat_template.json`
# 5c72a170… on both sides, verified 2026-08-05. That is what one text-encoder
# file serving both partitions implies, but it is checked rather than assumed —
# so this pulls FL2VA's for both and the choice costs nothing either way.
MINIMAX_FILES = ["FL2VA/processor/tokenizer.json",
                 "FL2VA/processor/tokenizer_config.json",
                 "FL2VA/processor/vocab.json",
                 "FL2VA/processor/merges.txt",
                 "LICENSE"]

# Everything except the DiT is byte-identical between the FL2VA and REF2VA
# packs, so an already-converted pack can hand them over instead of spending a
# 52 GB download and an hour of quantization to re-derive them. Reconversion is
# NOT byte-stable (the DSV4 mirror round measured 59/149 tensors drifting at
# rounding level between converter sessions), so copying a proven shard is the
# correct move, not merely the fast one.
SHARED_WITH_FL2VA = ["text_encoder.safetensors", "video_vae.safetensors",
                     "audio_vae.safetensors"]

MIN_FREE_GIB = 20.0  # abort rather than wedge the machine


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gib(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1 << 30)


def need_space(path, gib, what):
    have = free_gib(path)
    if have < gib + MIN_FREE_GIB:
        raise SystemExit(
            f"refusing to start {what}: {have:.1f} GiB free, need ~{gib:.0f} + "
            f"{MIN_FREE_GIB:.0f} GiB headroom. Free space and re-run — the "
            "script is resumable."
        )


def hf_download(repo, patterns, dest):
    subprocess.run(["hf", "download", repo, *patterns, "--local-dir", dest],
                   check=True, stdout=subprocess.DEVNULL)


def clone_file(src, dst):
    """APFS copy-on-write clone, falling back to a real copy off APFS.

    A clone is instant and costs no space, which is what makes reusing a 28 GB
    text encoder from a neighbouring pack free rather than merely fast.
    """
    if subprocess.run(["cp", "-c", src, dst]).returncode != 0:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="staging dir for downloads")
    ap.add_argument("--out", required=True, help="model dir to write")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--keep-src", action="store_true",
                    help="do not delete each source after converting it (needs ~130 GB free)")
    ap.add_argument("--reuse-from", default=None,
                    help="an already-converted H3 pack to clone the shared text "
                         "encoder and VAEs from (only the DiT differs between "
                         "partitions), skipping their download and conversion")
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)
    os.makedirs(src, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    # `hf download --local-dir` reproduces the repo's own layout, which here is
    # already the flat `{diffusion_models,text_encoders,vae}/` the converter
    # reads — this repo has no `split_files/` prefix.
    flat = src

    if args.reuse_from:
        reuse = os.path.expanduser(args.reuse_from)
        for fn in SHARED_WITH_FL2VA:
            s, d = os.path.join(reuse, fn), os.path.join(out, fn)
            if os.path.exists(d):
                log(f"{fn}: present")
            elif os.path.exists(s):
                log(f"cloning {fn} from {os.path.basename(reuse)}")
                clone_file(s, d)
            else:
                raise SystemExit(f"--reuse-from given but {s} is missing")

    log("fetching tokenizer + LICENSE from " + MINIMAX_REPO)
    hf_download(MINIMAX_REPO, MINIMAX_FILES, src)
    lic = os.path.join(src, "LICENSE")
    if not os.path.exists(lic):
        raise SystemExit(
            "LICENSE did not download. The converter refuses to write a model "
            "dir without it (Section III.1), so stop here rather than produce "
            "something that cannot be shared."
        )

    for label, path, gib in VAE_FILES:
        # A VAE already sitting in the OUTPUT (cloned by --reuse-from) makes its
        # source irrelevant — the converter copies it verbatim either way.
        if os.path.exists(os.path.join(out, f"{label}.safetensors")):
            log(f"{label}: already in the output")
            continue
        if os.path.exists(os.path.join(flat, path)):
            log(f"{label}: present")
            continue
        need_space(src, gib, label)
        log(f"fetching {label} (~{gib} GiB)")
        hf_download(COMFY_REPO, [path], src)

    # One heavy file at a time: download -> convert -> delete the source. The
    # converter itself skips an output that already exists, so a re-run after an
    # interruption re-downloads only what it still needs.
    for label, path, gib in COMFY_FILES:
        produced = os.path.join(out, f"{label}.safetensors")
        if os.path.exists(produced):
            log(f"{label}: already converted")
            continue
        staged = os.path.join(flat, path)
        if not os.path.exists(staged):
            need_space(src, gib, label)
            log(f"fetching {label} (~{gib} GiB)")
            hf_download(COMFY_REPO, [path], src)

        log(f"converting {label}")
        rc = subprocess.run([
            sys.executable, os.path.join(os.path.dirname(__file__),
                                         "convert_minimax_h3_weights.py"),
            "--partition", "ref2va",
            "--bits", str(args.bits),
            "--src", flat,
            "--tokenizer", os.path.join(src, "FL2VA", "processor"),
            "--out", out,
        ]).returncode
        if rc != 0:
            raise SystemExit(f"converter failed on {label} (exit {rc})")

        if not args.keep_src and os.path.exists(staged):
            log(f"deleting source {os.path.basename(staged)} "
                f"({os.path.getsize(staged)/1e9:.0f} GB)")
            os.remove(staged)

    # The converter looks for LICENSE one/two levels above the tokenizer dir;
    # the staged tree puts it at <src>/LICENSE, which is exactly two up from
    # <src>/FL2VA/processor. Assert rather than assume — a missing LICENSE in
    # the OUTPUT is the thing that must never ship.
    for required in ("LICENSE", "NOTICE", "MODIFICATIONS.md", "config.json", "README.md"):
        if not os.path.exists(os.path.join(out, required)):
            raise SystemExit(f"{required} missing from {out} — do not publish this directory")

    if not args.keep_src:
        shutil.rmtree(os.path.join(src, "FL2VA"), ignore_errors=True)

    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)
                if f.endswith(".safetensors")) / 1e9
    log(f"DONE -> {out} ({total:.1f} GB of weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
