#!/usr/bin/env python3
"""Dump golden VISION-PRESENTATION values for MiniMax H3 from the REFERENCE.

Phase 1 of the ref2va work needs four things that are pure math and easy to get
subtly wrong in ways that produce a model which runs and conditions on the WRONG
thing:

  1. the Qwen3-VL image/video resize policy (`process_qwen2vl_images` /
     `process_video_block`) -- how many vision tokens a reference contributes,
  2. `token_tags_from_embeds_info` -- the adaLN modality tags, where the vision
     span widens by ONE on each side to cover the flanking vision_start/end,
  3. `qwen2vl_mrope_position_ids` -- the [3, seq] T/H/W position ids, which are
     NOT a plain arange once a vision block is present,
  4. `precompute_freqs_cis(..., interleaved_mrope=True)` -- Qwen3-VL's
     interleaved M-RoPE, where H and W replace every 3rd frequency.

All four are EXECUTED here, not transcribed: ComfyUI's own functions produce the
golden values, with its packages stubbed by an object that raises on any
attribute access so nothing can be produced by a mock.

Usage:
    uv run --with torch --with numpy --with tqdm \
        tests/dump_minimax_h3_vision_fixtures.py \
        --ref ~/claude-tmp/h3-ref --out src/fixtures/minimax_h3_vision.json

Refresh the sources with:
    for f in comfy/text_encoders/minimax.py comfy/text_encoders/qwen_vl.py \
             comfy/text_encoders/llama.py comfy/text_encoders/qwen3vl.py; do
      curl -sL "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/$f" \
        -o ~/claude-tmp/h3-ref/$(basename $f)
    done
"""

import argparse
import importlib.util
import json
import os
import sys
import types


class _Exploding(types.ModuleType):
    """Stub that raises on ANY attribute access (see dump_minimax_h3_layout.py)."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AssertionError(
            f"reference reached stubbed {self.__name__}.{name} -- that code path "
            "is not pure presentation math, so its output is not a valid golden value"
        )


def _stub(*names):
    for n in names:
        sys.modules.setdefault(n, _Exploding(n))


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _exec_prefix(path, name, package, required):
    """Exec a module's source up to its first `class`, inside a package context.

    minimax.py's `from .qwen3vl import Qwen3VL, ...` needs a package to resolve
    against, and Qwen3VL must be a real class for the FIRST class statement to
    even compile -- but everything we want (VISION_START, process_video_block,
    token_tags_from_embeds_info) sits above it. Cut there rather than faking a
    class hierarchy whose only job is to be discarded.
    """
    with open(path) as f:
        src = f.read()
    cut = src.index("\nclass ")
    mod = types.ModuleType(name)
    mod.__dict__["__name__"] = name
    mod.__dict__["__package__"] = package
    exec(compile(src[:cut], path, "exec"), mod.__dict__)
    missing = [r for r in required if not hasattr(mod, r)]
    if missing:
        raise SystemExit(
            f"{os.path.basename(path)}: {missing} are no longer above the first "
            "class -- the prefix cut is wrong, do not guess the values"
        )
    return mod


def load_reference(ref_dir):
    _stub("comfy", "comfy.ldm", "comfy.ldm.modules", "comfy.model_management",
          "comfy.ops", "comfy.sd1_clip", "comfy.rmsnorm", "comfy.float",
          "comfy.model_patcher", "comfy.utils", "comfy.clip_model",
          "comfy.ldm.common_dit", "transformers", "torchaudio")
    attn = types.ModuleType("comfy.ldm.modules.attention")
    attn.optimized_attention_for_device = None
    attn.optimized_attention = None
    sys.modules["comfy.ldm.modules.attention"] = attn

    for p in ("qwen_vl.py", "llama.py", "minimax.py"):
        full = os.path.join(ref_dir, p)
        if not os.path.exists(full):
            raise SystemExit(f"missing reference source: {full} (see the header)")

    # llama.py and minimax.py both use relative imports, so they need a package
    # to resolve against. Register the siblings they reach for under it.
    pkg = types.ModuleType("h3ref")
    pkg.__path__ = []
    sys.modules["h3ref"] = pkg

    qwen_vl = _load_module(os.path.join(ref_dir, "qwen_vl.py"), "h3ref.qwen_vl")
    llama_mod = _load_module(os.path.join(ref_dir, "llama.py"), "h3ref.llama")

    # minimax.py's `from .qwen3vl import ...` -- both names are only touched by
    # the class bodies below the cut, so binding them to a bare object binds
    # something that cannot compute a value (unlike a mock).
    q3 = types.ModuleType("h3ref.qwen3vl")
    q3.Qwen3VL = object
    q3.Qwen3VLSDTokenizer = object
    sys.modules["h3ref.qwen3vl"] = q3

    minimax = _exec_prefix(
        os.path.join(ref_dir, "minimax.py"), "h3ref.minimax", "h3ref",
        ["VISION_START", "VISION_END", "process_video_block",
         "token_tags_from_embeds_info", "QWEN_IMAGE_MEAN", "QWEN_IMAGE_STD"],
    )
    return qwen_vl, llama_mod, minimax


# ── cases ───────────────────────────────────────────────────────────────────

# (label, height, width) -- an aspect that is NOT a multiple of 32, one that is,
# one above the pixel cap and one below the floor, so every branch of the
# resize policy is exercised and no case can pass by round-number luck.
IMAGE_CASES = [
    ("square_512", 512, 512),
    ("wide_480x864", 480, 864),
    ("odd_453x611", 453, 611),
    ("tiny_40x30", 40, 30),
    ("huge_3024x4032", 3024, 4032),
]


def dump_image_grids(qwen_vl, minimax):
    """`process_qwen2vl_images` at H3's patch_size 16 / [-1,1] normalization."""
    import torch
    out = []
    for label, h, w in IMAGE_CASES:
        # A deterministic non-degenerate image: a constant would leave the
        # normalization invisible and a gradient makes the patch ORDER matter.
        img = torch.arange(h * w * 3, dtype=torch.float32).reshape(1, h, w, 3)
        img = (img % 251.0) / 251.0
        flat, grid = qwen_vl.process_qwen2vl_images(
            img, patch_size=16, image_mean=minimax.QWEN_IMAGE_MEAN,
            image_std=minimax.QWEN_IMAGE_STD)
        g = [int(v) for v in grid[0]]
        out.append({
            "label": label, "in_h": h, "in_w": w,
            "grid_thw": g,
            "n_patches": int(flat.shape[0]),
            "patch_dim": int(flat.shape[1]),
            # merged tokens are what actually enter the LM sequence
            "merged_tokens": g[0] * g[1] * g[2] // 4,
        })
    return out


def dump_video_blocks(minimax):
    """`process_video_block`: a 2-frame pair fills the temporal patch."""
    import torch
    out = []
    for label, h, w in [("pair_480x864", 480, 864), ("pair_240x240", 240, 240)]:
        frames = torch.arange(2 * h * w * 3, dtype=torch.float32).reshape(2, h, w, 3)
        frames = (frames % 251.0) / 251.0
        flat, grid = minimax.process_video_block(frames)
        g = [int(v) for v in grid[0]]
        out.append({
            "label": label, "in_h": h, "in_w": w,
            "grid_thw": g,
            "n_patches": int(flat.shape[0]),
            "patch_dim": int(flat.shape[1]),
            "merged_tokens": g[0] * g[1] * g[2] // 4,
        })
    return out


def _embeds_info(spans):
    """spans: list of (index, size, grid_thw) -> the reference's embeds_info shape."""
    import torch
    return [
        {"type": "image", "index": i, "size": n,
         "extra": {"grid": torch.tensor([list(g)], dtype=torch.long)}}
        for i, n, g in spans
    ]


# (label, seq_len, [(index, size, (t, gh, gw)), ...])
# index is the FIRST expanded vision row, i.e. one past the <|vision_start|>.
TAG_CASES = [
    ("no_vision", 24, []),
    ("one_image", 60, [(9, 30, (1, 10, 12))]),
    ("two_images", 120, [(6, 30, (1, 10, 12)), (45, 48, (1, 12, 16))]),
    ("vision_at_zero", 40, [(1, 20, (1, 8, 10))]),
]


def dump_token_tags(minimax):
    out = []
    for label, seq, spans in TAG_CASES:
        tags = minimax.token_tags_from_embeds_info(seq, _embeds_info(spans))
        lst = [int(v) for v in tags]
        # runs, because the DiT consumes runs and a per-position list hides an
        # off-by-one at a boundary in a wall of identical numbers
        runs = []
        start = 0
        for i in range(1, seq + 1):
            if i == seq or lst[i] != lst[start]:
                runs.append([start, i, lst[start]])
                start = i
        out.append({"label": label, "seq_len": seq,
                    "spans": [[i, n, list(g)] for i, n, g in spans],
                    "runs": runs, "tag_sum": sum(lst)})
    return out


def dump_mrope_position_ids(qwen_vl):
    out = []
    for label, seq, spans in TAG_CASES:
        pos = qwen_vl.qwen2vl_mrope_position_ids(_embeds_info(spans), seq, "cpu")
        if pos is None:
            out.append({"label": label, "seq_len": seq, "present": False})
            continue
        rows = [[float(v) for v in pos[i]] for i in range(3)]
        out.append({
            "label": label, "seq_len": seq, "present": True,
            "spans": [[i, n, list(g)] for i, n, g in spans],
            "rows": rows,
            "row_sums": [sum(r) for r in rows],
            # order-sensitive companion: a permutation-invariant checksum cannot
            # see a permutation (the layout fixture already paid for this).
            "row_weighted": [sum((j + 1) * v for j, v in enumerate(r)) for r in rows],
        })
    return out


def dump_interleaved_rope(llama_mod, qwen_vl):
    """Qwen3-VL interleaved M-RoPE: T-freqs by default, H/W replace every 3rd."""
    import torch
    head_dim = 128
    theta = 5000000.0
    rope_dims = [24, 20, 20]
    out = []
    for label, seq, spans in TAG_CASES:
        pos = qwen_vl.qwen2vl_mrope_position_ids(_embeds_info(spans), seq, "cpu")
        if pos is None:
            pos = torch.arange(0, seq).unsqueeze(0).float()
        cos, sin_lo, nsin_hi = llama_mod.precompute_freqs_cis(
            head_dim, pos, theta, rope_dims=rope_dims, interleaved_mrope=True)
        c = cos.reshape(-1, head_dim)
        # The reference hands back (cos, sin[:half], -sin[half:]); sin is the
        # duplicated half so sin[:half] IS the angle sine.
        s = sin_lo.reshape(-1, head_dim // 2)
        out.append({
            "label": label, "seq_len": seq, "head_dim": head_dim,
            "spans": [[i, n, list(g)] for i, n, g in spans],
            "mrope": pos.shape[0] > 1,
            "cos_first": [float(v) for v in c[0]],
            "cos_last": [float(v) for v in c[-1]],
            "cos_sum": float(c.sum()),
            "sin_sum": float(s.sum()),
            "cos_weighted": float((c.sum(dim=-1) *
                                   torch.arange(1, c.shape[0] + 1, dtype=c.dtype)).sum()),
            "sin_weighted": float((s.sum(dim=-1) *
                                   torch.arange(1, s.shape[0] + 1, dtype=s.dtype)).sum()),
        })
    return out


def dump_interleave_map(llama_mod):
    """Which axis feeds each of the head_dim/2 frequency slots.

    Recovered by RUNNING the reference against per-axis position ids that are
    distinguishable by construction (axis a gets position 1<<a), so the slot's
    source is read off the output rather than re-derived from the slice
    arithmetic this fixture exists to pin.
    """
    import torch
    head_dim = 128
    rope_dims = [24, 20, 20]
    half = head_dim // 2
    pos = torch.tensor([[1.0], [2.0], [4.0]])  # one position, T=1 H=2 W=4
    cos, _, _ = llama_mod.precompute_freqs_cis(
        head_dim, pos, 10000.0, rope_dims=rope_dims, interleaved_mrope=True)
    row = cos.reshape(-1, head_dim)[0][:half]
    # reference angles per axis at the same frequencies
    inv = [1.0 / (10000.0 ** (2 * j / head_dim)) for j in range(half)]
    axis_of = []
    for j in range(half):
        vals = [torch.cos(torch.tensor(p * inv[j])).item() for p in (1.0, 2.0, 4.0)]
        diffs = [abs(row[j].item() - v) for v in vals]
        axis_of.append(int(min(range(3), key=lambda a: diffs[a])))
    return {"head_dim": head_dim, "rope_dims": rope_dims, "axis_of_freq": axis_of,
            "counts": [axis_of.count(a) for a in range(3)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.expanduser("~/claude-tmp/h3-ref"))
    ap.add_argument("--out", default="src/fixtures/minimax_h3_vision.json")
    args = ap.parse_args()

    qwen_vl, llama_mod, minimax = load_reference(args.ref)

    data = {
        "_source": "ComfyUI comfy/text_encoders/{minimax,qwen_vl,llama}.py",
        "_generator": "tests/dump_minimax_h3_vision_fixtures.py",
        "constants": {
            "VISION_START": int(minimax.VISION_START),
            "VISION_END": int(minimax.VISION_END),
            "QWEN_IMAGE_MEAN": list(minimax.QWEN_IMAGE_MEAN),
            "QWEN_IMAGE_STD": list(minimax.QWEN_IMAGE_STD),
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "min_pixels": 3136,
            "max_pixels": 12845056,
            "rope_theta": 5000000.0,
            "rope_dims": [24, 20, 20],
            "vit_hidden": 1152,
            "vit_depth": 27,
            "vit_intermediate": 4304,
            "vit_heads": 16,
            "vit_deepstack_indexes": [8, 16, 24],
            "vit_out_hidden": 5120,
            "vit_num_position_embeddings": 2304,
        },
        "image_grids": dump_image_grids(qwen_vl, minimax),
        "video_blocks": dump_video_blocks(minimax),
        "token_tags": dump_token_tags(minimax),
        "mrope_position_ids": dump_mrope_position_ids(qwen_vl),
        "interleaved_rope": dump_interleaved_rope(llama_mod, qwen_vl),
        "interleave_map": dump_interleave_map(llama_mod),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out}")
    for c in data["image_grids"]:
        print(f"  image {c['label']}: {c['in_w']}x{c['in_h']} -> grid {c['grid_thw']} "
              f"= {c['merged_tokens']} merged tokens")
    for c in data["video_blocks"]:
        print(f"  video {c['label']}: grid {c['grid_thw']} = {c['merged_tokens']} tokens")
    print(f"  interleave_map counts (T,H,W): {data['interleave_map']['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
