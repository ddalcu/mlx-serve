#!/usr/bin/env python3
"""Dump golden packed-layout / schedule values for MiniMax H3 from the REFERENCE.

The values in `src/fixtures/minimax_h3_layout.json` are produced by calling
ComfyUI's own `comfy/ldm/minimax/model.py` and `comfy_extras/nodes_minimax_h3.py`
functions, not by re-deriving them here -- a transcription checked against its
own transcription proves nothing. ComfyUI's package imports are stubbed because
the functions we need (frame grid, position grids, PackedLayout, sigma schedule)
touch only torch and math; anything that reaches a stub raises rather than
silently returning a plausible number.

This layer is weightless, so the Zig side is a hermetic test with no env gate
and no checkpoint -- see `layout fixtures` in src/minimax_h3.zig.

Usage:
    uv run --with torch --with numpy tests/dump_minimax_h3_layout.py \
        --ref ~/claude-tmp/h3-ref --out src/fixtures/minimax_h3_layout.json

Refresh the sources with:
    for f in comfy/ldm/minimax/model.py comfy_extras/nodes_minimax_h3.py; do
      curl -sL "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/$f" \
        -o ~/claude-tmp/h3-ref/$(basename $f)
    done
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import types


class _Exploding(types.ModuleType):
    """Stub that raises on ANY attribute access.

    A stub returning MagicMock would let a golden value be produced by a mock
    instead of by the reference, which is exactly the failure this file exists
    to prevent.
    """

    def __getattr__(self, name):
        # Dunders must behave like an ordinary missing attribute: torch's own
        # import machinery walks sys.modules probing `__file__`/`__path__`, and
        # exploding there says nothing about the reference reaching our stub.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AssertionError(
            f"reference reached stubbed {self.__name__}.{name} -- that code path "
            "is not pure layout math, so its output is not a valid golden value"
        )


def _stub(*names):
    for n in names:
        sys.modules.setdefault(n, _Exploding(n))


# Everything in nodes_minimax_h3.py that we need (the 17k+5 ladder, the canvas
# rule, the fps/latent-fps constants) is module-level and sits ABOVE the first
# `class`. Below it is pure ComfyUI node-registration scaffolding, which needs a
# subclassable `io.ComfyNode` and a working `io.Schema` to import at all. Rather
# than fake a metaclass hierarchy whose only job is to be discarded, exec the
# prefix. This is still the reference's own source text -- the thing we must not
# do is retype the formulas.
def _exec_module_prefix(path, name, required):
    with open(path) as f:
        src = f.read()
    cut = src.index("\nclass ")
    mod = types.ModuleType(name)
    mod.__dict__["__name__"] = name
    exec(compile(src[:cut], path, "exec"), mod.__dict__)
    missing = [r for r in required if not hasattr(mod, r)]
    if missing:
        raise SystemExit(
            f"{os.path.basename(path)}: {missing} are no longer above the first "
            "class -- the prefix cut is wrong, do not guess the values"
        )
    return mod


def load_reference(ref_dir):
    """Import the two reference modules with ComfyUI's packages stubbed out."""
    _stub(
        "comfy", "comfy.ldm", "comfy.ldm.common_dit", "comfy.ldm.modules",
        "comfy.model_management", "comfy.model_prefetch", "comfy.ops",
        "comfy.patcher_extension", "comfy.quant_ops", "comfy.model_sampling",
        "comfy.nested_tensor", "comfy.utils", "nodes", "node_helpers",
        "torchaudio",
    )
    # `from comfy.ldm.modules.attention import optimized_attention` needs a real
    # attribute on a real module object, not an exploding one.
    attn = types.ModuleType("comfy.ldm.modules.attention")
    attn.optimized_attention = None
    sys.modules["comfy.ldm.modules.attention"] = attn
    # `from comfy_api.latest import ComfyExtension, io` sits in the node file's
    # import block, i.e. inside the prefix we exec. Both names are only ever
    # touched by the class bodies below the cut, so binding them to None binds
    # something that cannot compute a value -- unlike a mock, which could.
    api = types.ModuleType("comfy_api.latest")
    api.ComfyExtension = None
    api.io = None
    sys.modules["comfy_api"] = types.ModuleType("comfy_api")
    sys.modules["comfy_api.latest"] = api

    model_path = os.path.join(ref_dir, "model.py")
    nodes_path = os.path.join(ref_dir, "nodes_minimax_h3.py")
    for p in (model_path, nodes_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing reference source: {p}")

    spec = importlib.util.spec_from_file_location("h3_model", model_path)
    model = importlib.util.module_from_spec(spec)
    sys.modules["h3_model"] = model
    spec.loader.exec_module(model)

    nodes = _exec_module_prefix(nodes_path, "h3_nodes", [
        "align_frame_count", "video_latent_t", "temporal_shape", "adapt_canvas",
        "CANVAS_MULTIPLE", "BASE_SHORT_EDGE", "MAX_PIXELS", "FPS", "AUDIO_LATENT_FPS",
        "REF_IMAGE_SHORT_EDGE",
    ])
    return model, nodes


def dump_frame_grid(N):
    """The 17k+5 frame ladder and its latent-frame count.

    The ladder is the whole reason a duration cannot be chosen freely: 56 frames
    (2.33 s) is legal, 48 is not.
    """
    cases = []
    for n in [1, 5, 6, 22, 39, 40, 56, 73, 90, 107, 124, 141, 200, 362, 363]:
        aligned = N.align_frame_count(max(5, n))
        cases.append({
            "requested": n,
            "aligned": aligned,
            "latent_t": N.video_latent_t(aligned),
        })
    return cases


def dump_temporal(N):
    out = []
    for n in [5, 56, 124, 362]:
        frame_count, latent_t, audio_t = N.temporal_shape(n)
        out.append({"length": n, "frame_count": frame_count,
                    "latent_t": latent_t, "audio_t": audio_t})
    return out


def dump_canvas(N):
    """768-short-edge canvas with the 768*1344 area cap, per-axis /32."""
    out = []
    for w, h in [(1344, 768), (768, 1344), (1920, 1080), (1080, 1920),
                 (1024, 1024), (2560, 1080), (4032, 3024), (640, 480)]:
        cw, ch = N.adapt_canvas(w, h)
        out.append({"in_w": w, "in_h": h, "out_w": cw, "out_h": ch})
    return out


def dump_sigma_schedule(M):
    """Video->audio sigma map and the velocity slope, at shift 12.0 / 3.0.

    The slope is what makes a flat ODE integrated on the video schedule equal
    the audio stream's true ODE on its own schedule; getting it wrong desyncs
    the soundtrack rather than erroring.
    """
    out = []
    for sigma in [1.0, 0.95, 0.75, 0.5, 0.25, 0.1, 0.02, 1e-3, 1e-6]:
        out.append({
            "sigma_v": sigma,
            "sigma_a": float(M.time_shift_sigma(_t(sigma), 12.0, 3.0)),
            "slope": float(M.time_shift_slope(_t(sigma), 12.0, 3.0)),
        })
    return out


def _t(x):
    import torch
    return torch.tensor(x, dtype=torch.float64)


def dump_axis_and_frame_grid(M):
    """Area-normalized (h, w) coordinates -- the RoPE position grid.

    Not a plain arange: each axis is a linspace over (1-ratio)/2 .. (1+ratio)/2
    scaled by 32, where ratio = dim / sqrt(h*w). Aspect ratio therefore moves
    EVERY position, so a transcription that uses raw indices produces a model
    that runs and generates the wrong framing.
    """
    out = []
    for h, w in [(48, 84), (30, 54), (24, 42), (64, 64)]:
        frame, w_grid = M._frame_grid(h, w)
        out.append({
            "latent_h": h, "latent_w": w,
            "rows": int(frame.shape[0]),
            "frame_first": [float(v) for v in frame[0]],
            "frame_last": [float(v) for v in frame[-1]],
            "frame_sum": float(frame.sum()),
            "w_grid_first": float(w_grid[0]),
            "w_grid_last": float(w_grid[-1]),
        })
    return out


def dump_t_grid(M):
    """Per-latent-frame temporal origins: FRAME_PER_TOKEN (1,4,4,4,4) * 5/3."""
    out = []
    for n, origin in [(2, 0.0), (7, 0.0), (17, 12.0), (37, 100.0)]:
        g = M._video_t_grid(n, origin)
        out.append({
            "n": n, "origin": origin,
            "grid": [float(v) for v in g],
            "spans_sum": float(sum(M._video_t_spans(n))),
        })
    return out


def _row_weights(layout):
    """1..S, so a checksum weighted by it is sensitive to row ORDER."""
    import torch
    return torch.arange(1, layout.seq_len + 1, dtype=torch.float64)


def dump_packed_layout(M, N):
    """Full segment tables for the shapes we actually intend to serve.

    dev  = 56 frames @ 864x480  (the ~13 min iteration config)
    spec = 124 frames @ 1344x768 (on-spec, the acceptance run)

    latent_t / audio_t come from the reference's own temporal_shape -- deriving
    them here would pin our arithmetic against itself.
    """
    cases = []
    specs = [
        # (name, text_len, frames, width, height, keyframe anchors)
        ("dev_t2va", 512, 56, 864, 480, None),
        ("spec_t2va", 512, 124, 1344, 768, None),
        ("dev_fl2va_first", 512, 56, 864, 480, ["first"]),
        ("dev_fl2va_both", 512, 56, 864, 480, ["first", "last"]),
    ]
    for name, text_len, frames, width, height, kf_at in specs:
        frame_count, latent_t, audio_t = N.temporal_shape(frames)
        lat_h, lat_w = height // 16, width // 16
        keyframes = None
        if kf_at is not None:
            keyframes = [{"resolved_frame_index": 0 if a == "first" else frame_count - 1}
                         for a in kf_at]
        layout = M.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                keyframes=keyframes, frame_count=frame_count)
        cases.append({
            "name": name,
            "text_len": text_len,
            "frame_count": frame_count,
            "width": width, "height": height,
            "latent_t": latent_t, "latent_h": lat_h, "latent_w": lat_w,
            "audio_t": audio_t,
            "seq_len": int(layout.seq_len),
            "segments": [[int(a), int(b), k] for a, b, k in layout.segments],
            # position_ids is [S, 3] float64 -- spot-check ends plus a checksum,
            # since embedding all of it would be megabytes
            "pos_first": [float(v) for v in layout.position_ids[0]],
            "pos_last": [float(v) for v in layout.position_ids[-1]],
            "pos_sum": [float(layout.position_ids[:, i].sum()) for i in range(3)],
            # A plain column sum is permutation-INVARIANT: swapping the two
            # stereo channels' pinned w coordinates, or reversing a frame's
            # rows, leaves it untouched. Weighting by row index makes the
            # checksum order-sensitive so those slips cannot pass.
            "pos_weighted": [
                float((layout.position_ids[:, i] * _row_weights(layout)).sum())
                for i in range(3)
            ],
            # Per-segment first row, so a failure localizes to a segment instead
            # of reporting one wrong number for the whole sequence.
            "seg_first_pos": [
                [float(v) for v in layout.position_ids[a]]
                for a, _, _ in layout.segments
            ],
            "img_update_true": int(layout.img_update.sum()),
            "audio_update_true": int(layout.audio_update.sum()),
        })
    return cases


def dump_ref_sizing(N):
    """How a reference IMAGE and a reference VIDEO are sized before encoding.

    Executed from the node's own module-level constants and `adapt_canvas`.
    Two independent rules that are easy to conflate:

      image: `match` scales (DOWN only) to the generation's pixel AREA, `max`
             to a 2048 short edge — then each axis rounds to 32. `max` is
             several times slower because reference tokens ride through every
             sampling step.
      video: `adapt_canvas` normalizes to the 768 short edge, which UPSCALES a
             small clip; the node then refuses that (`if vw*vh < cw*ch`) and
             falls back to the plain /32 rounding, so a reference video is
             never upscaled.
    """
    def size_image(w, h, gen_w, gen_h, mode):
        if mode == "match":
            scale = min(1.0, math.sqrt((gen_w * gen_h) / (w * h)))
        else:
            scale = min(1.0, N.REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(N.CANVAS_MULTIPLE, round(w * scale / N.CANVAS_MULTIPLE) * N.CANVAS_MULTIPLE)
        th = max(N.CANVAS_MULTIPLE, round(h * scale / N.CANVAS_MULTIPLE) * N.CANVAS_MULTIPLE)
        return tw, th

    def size_video(vw, vh):
        cw, ch = N.adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(N.CANVAS_MULTIPLE, round(vw / N.CANVAS_MULTIPLE) * N.CANVAS_MULTIPLE)
            ch = max(N.CANVAS_MULTIPLE, round(vh / N.CANVAS_MULTIPLE) * N.CANVAS_MULTIPLE)
        return cw, ch

    images = []
    for w, h in [(4032, 3024), (1024, 1024), (640, 480), (3000, 500), (100, 100)]:
        for gen_w, gen_h in [(864, 480), (1344, 768)]:
            for mode in ("match", "max"):
                tw, th = size_image(w, h, gen_w, gen_h, mode)
                images.append({"in_w": w, "in_h": h, "gen_w": gen_w, "gen_h": gen_h,
                               "mode": mode, "out_w": tw, "out_h": th})
    videos = []
    for vw, vh in [(1920, 1080), (640, 480), (256, 144), (1344, 768), (3000, 500)]:
        cw, ch = size_video(vw, vh)
        videos.append({"in_w": vw, "in_h": vh, "out_w": cw, "out_h": ch})

    # Frame counts snap DOWN for references (a reference is trimmed, never
    # padded with frames the user did not supply), with a 5-frame floor.
    frames = []
    for n in [4, 5, 6, 21, 22, 23, 100, 124, 400]:
        v = n
        ok = v >= 5
        if ok:
            while v % 17 != 5:
                v -= 1
        frames.append({"requested": n, "aligned": v if ok else 0})

    return {"REF_IMAGE_SHORT_EDGE": int(N.REF_IMAGE_SHORT_EDGE),
            "images": images, "videos": videos, "frames": frames}


def dump_ref_layout(M, N):
    """ref2va: the `refs` branch of PackedLayout, block by block.

    Four things here are silent when wrong and only a fixture can see them:
      - the cursor advances by a DIFFERENT amount per kind (1.0 per image,
        ref_audio_t per standalone audio, max(audio_t, sum of video spans) per
        video block),
      - a video block's soundtrack rows pack IMMEDIATELY BEFORE its video rows
        and share the block's cursor origin,
      - that soundtrack's w coordinates are pinned to the BLOCK's own grid
        extremes, while a standalone audio uses the TARGET's,
      - the target streams still start at the cursor the refs left behind.
    """
    cases = []
    frames, width, height, text_len = 56, 864, 480, 128
    frame_count, latent_t, audio_t = N.temporal_shape(frames)
    lat_h, lat_w = height // 16, width // 16

    def img(h, w):
        return {"kind": "image", "latent_h": h // 16, "latent_w": w // 16}

    def vid(h, w, vt, rt=0):
        return {"kind": "video_audio" if rt else "video", "latent_t": vt,
                "latent_h": h // 16, "latent_w": w // 16, "ref_audio_t": rt}

    def aud(rt):
        return {"kind": "audio", "ref_audio_t": rt}

    # Every reference here is deliberately a DIFFERENT aspect from the 864x480
    # target. A reference that happens to match the target cannot distinguish
    # "the block's own w extremes" from "the target's", so a case built that way
    # passes whichever the port picks — verified by red-on-revert, which stayed
    # green until these aspects diverged.
    specs = [
        ("ref_one_image", [img(512, 512)]),
        # two images: the cursor advances 1.0 apiece, so the second block's t
        # differs from the first by exactly one
        ("ref_two_images", [img(512, 512), img(320, 576)]),
        ("ref_one_video", [vid(320, 576, 7)]),
        # a soundtracked video: its <Audio j> rows precede its <Video k> rows
        # and take the BLOCK's w extremes, not the target's
        ("ref_video_audio", [vid(512, 512, 7, rt=12)]),
        ("ref_standalone_audio", [aud(20)]),
        # the reference's fixed order: images, then videos, then standalone audio
        ("ref_mixed", [img(512, 512), img(320, 576), vid(512, 512, 7, rt=12), aud(20)]),
    ]
    for name, refs in specs:
        layout = M.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                refs=refs, frame_count=frame_count)
        cases.append({
            "name": name,
            "text_len": text_len,
            "frame_count": frame_count,
            "width": width, "height": height,
            "latent_t": latent_t, "latent_h": lat_h, "latent_w": lat_w,
            "audio_t": audio_t,
            "refs": refs,
            "seq_len": int(layout.seq_len),
            "segments": [[int(a), int(b), k] for a, b, k in layout.segments],
            "pos_first": [float(v) for v in layout.position_ids[0]],
            "pos_last": [float(v) for v in layout.position_ids[-1]],
            "pos_sum": [float(layout.position_ids[:, i].sum()) for i in range(3)],
            "pos_weighted": [
                float((layout.position_ids[:, i] * _row_weights(layout)).sum())
                for i in range(3)
            ],
            "seg_first_pos": [
                [float(v) for v in layout.position_ids[a]]
                for a, _, _ in layout.segments
            ],
            # the LAST row of each segment too: a block whose interior grid is
            # right at its first row can still run off the end of its own span
            "seg_last_pos": [
                [float(v) for v in layout.position_ids[b - 1]]
                for _, b, _ in layout.segments
            ],
            "img_update_true": int(layout.img_update.sum()),
            "audio_update_true": int(layout.audio_update.sum()),
            "img_rows": int(layout.img_update.shape[0]),
            "audio_rows": int(layout.audio_update.shape[0]),
        })
    return cases


def dump_rope_freqs(M):
    """[S,3] positions -> [S,96] angles: per-axis 16 freqs, concat(t,h,w), duplicated."""
    import torch
    # inv_freq is a checkpoint buffer; use a synthetic but non-degenerate one so
    # the Zig side is pinned on the ARRANGEMENT (concat order, duplication),
    # which is the part that is easy to get wrong.
    inv = torch.tensor([1.0 / (10000.0 ** (i / 16.0)) for i in range(16)], dtype=torch.float32)
    pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 5.0, 7.0], [100.0, 12.5, 33.25]],
                       dtype=torch.float32)
    per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    ang = torch.cat((half, half), dim=-1)
    return {
        "inv_freq": [float(v) for v in inv],
        "positions": [[float(v) for v in row] for row in pos],
        "angles": [[float(v) for v in row] for row in ang],
        "rot_dim": int(ang.shape[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.expanduser("~/claude-tmp/h3-ref"),
                    help="directory holding the reference model.py / nodes_minimax_h3.py")
    ap.add_argument("--out", default="src/fixtures/minimax_h3_layout.json")
    args = ap.parse_args()

    M, N = load_reference(args.ref)

    # Constants the Zig side hardcodes: assert them against the reference so a
    # future ComfyUI change shows up as a fixture diff, not as silent drift.
    consts = {
        "FRAME_PER_TOKEN": list(M.FRAME_PER_TOKEN),
        "FRAME_RESCALE": float(M.FRAME_RESCALE),
        "VISUAL_COND_TIMESTEP": float(M.VISUAL_COND_TIMESTEP),
        "AUDIO_COND_TIMESTEP": float(M.AUDIO_COND_TIMESTEP),
        "CANVAS_MULTIPLE": int(N.CANVAS_MULTIPLE),
        "BASE_SHORT_EDGE": int(N.BASE_SHORT_EDGE),
        "MAX_PIXELS": int(N.MAX_PIXELS),
        "FPS": int(N.FPS),
        "AUDIO_LATENT_FPS": int(N.AUDIO_LATENT_FPS),
        "sigma_shift_video": 12.0,
        "sigma_shift_audio": 3.0,
    }

    data = {
        "_source": "ComfyUI comfy/ldm/minimax/model.py + comfy_extras/nodes_minimax_h3.py",
        "_generator": "tests/dump_minimax_h3_layout.py",
        "constants": consts,
        "frame_grid": dump_frame_grid(N),
        "temporal_shape": dump_temporal(N),
        "adapt_canvas": dump_canvas(N),
        "sigma_schedule": dump_sigma_schedule(M),
        "frame_position_grid": dump_axis_and_frame_grid(M),
        "video_t_grid": dump_t_grid(M),
        "rope_freqs": dump_rope_freqs(M),
        "packed_layout": dump_packed_layout(M, N),
        "ref_layout": dump_ref_layout(M, N),
        "ref_sizing": dump_ref_sizing(N),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out}")
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} cases")
    for c in data["ref_layout"]:
        kinds = "+".join(k for _, _, k in c["segments"])
        print(f"  {c['name']}: seq_len={c['seq_len']} [{kinds}]")
    # A layout whose sequence length we cannot explain is a layout we cannot port.
    for c in data["packed_layout"]:
        print(f"  {c['name']}: seq_len={c['seq_len']} "
              f"(video {c['latent_t']}x{c['latent_h']//2}x{c['latent_w']//2}"
              f"={c['latent_t']*(c['latent_h']//2)*(c['latent_w']//2)}, "
              f"audio {c['audio_t']*2}, text {c['text_len']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
