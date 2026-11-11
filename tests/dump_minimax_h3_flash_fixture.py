#!/usr/bin/env python3
"""HyperFlow H3 Flash parity fixture: one stable-diffusion.cpp DiT call, in mlx-serve's packed-row layout.

The reference is stable-diffusion.cpp with Video Rebirth's `hyperflow-two-time.patch`. To make the dump, have
`MiniMaxH3Runner::compute` write one call's inputs and output as raw float32 into a directory (`x`, `timestep`,
`context`, `scalars` = [audio L, video shift, audio shift, sigma_next], `out`; FL2VA adds `cond<i>`, `keyframes`, and
`tags` per text row), each `<name>.bin`, with `meta.txt` holding `<name> <dims in ggml order>` per line.
This turns that dump into the inputs and expected outputs of `minimax h3 flash parity` in
`src/plugins/minimax_h3/h3.zig`.

Usage: uv run --with numpy --with safetensors tests/dump_minimax_h3_flash_fixture.py <dump dir> <fixture.safetensors>
"""

import sys

import numpy as np
from safetensors.numpy import save_file

VIDEO_SHIFT, AUDIO_SHIFT = 12.0, 3.0


def shift(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    shapes = {}
    for line in open(f"{src}/meta.txt"):
        name, *dims = line.split()
        shapes[name] = [int(d) for d in dims]
    raw = {n: np.fromfile(f"{src}/{n}.bin", dtype=np.float32) for n in ("x", "timestep", "context", "scalars", "out")}
    audio_len, vshift, ashift, sigma_next = raw["scalars"].tolist()
    if (vshift, ashift) != (VIDEO_SHIFT, AUDIO_SHIFT):
        raise SystemExit(f"shifts {vshift}/{ashift} are not H3's 12/3")
    audio_len = int(audio_len)
    # The packed latent is [W, H, T, channels] in ggml order: 24 video channels, then the audio carrier and padding.
    w, h, t = shapes["x"][:3]
    c = 24
    sigma = min(max(float(raw["timestep"][0]) / 1000.0, 1e-6), 1.0)
    sigma_a = shift(sigma, VIDEO_SHIFT, AUDIO_SHIFT)
    k = VIDEO_SHIFT / AUDIO_SHIFT
    nv = c * t * h * w
    na = 32 * 2 * audio_len

    def video_rows(flat):
        v = flat[:nv].reshape(c, t, h // 2, 2, w // 2, 2)
        return np.ascontiguousarray(v.transpose(1, 2, 4, 0, 3, 5).reshape(t * (h // 2) * (w // 2), c * 4))

    def audio_rows(flat):
        a = flat[nv:nv + na].reshape(32, 2, audio_len)
        return np.ascontiguousarray(a.transpose(1, 2, 0).reshape(2 * audio_len, 32))

    # sd.cpp samples the audio as the carrier c_a = (sigma_v / sigma_a) x_a and returns d(c_a)/d(sigma_v):
    # (1 - k) x_a + (1 + (k - 1) sigma_a) v_a.
    x_a = audio_rows(raw["x"]) * (sigma_a / sigma)
    v_a = (audio_rows(raw["out"]) - (1.0 - k) * x_a) / (1.0 + (k - 1.0) * sigma_a)
    ctx_w = shapes["context"][0]
    out = {
        "video_rows": video_rows(raw["x"]),
        "audio_rows": x_a.astype(np.float32),
        "context": np.ascontiguousarray(raw["context"].reshape(-1, ctx_w)),
        "video_v": video_rows(raw["out"]),
        "audio_v": v_a.astype(np.float32),
        "meta": np.array([sigma, sigma_next, t, h, w, audio_len, (t - 2) // 5 * 17 + 5], dtype=np.float32),
    }
    # FL2VA keyframes: each condition latent ([W, H, T, 24] in ggml order) as patch rows, and its anchor (0 first, 1 last).
    conds = sorted((n for n in shapes if n.startswith("cond")), key=lambda n: int(n[4:]))
    if conds:
        rows = []
        for n in conds:
            cw, ch, ct, cc = shapes[n][:4]
            v = np.fromfile(f"{src}/{n}.bin", dtype=np.float32).reshape(cc, ct, ch // 2, 2, cw // 2, 2)
            rows.append(v.transpose(1, 2, 4, 0, 3, 5).reshape(ct * (ch // 2) * (cw // 2), cc * 4))
        out["cond_rows"] = np.ascontiguousarray(np.concatenate(rows))
        kf = np.fromfile(f"{src}/keyframes.bin", dtype=np.int32) if "keyframes" in shapes else np.arange(len(conds))
        out["keyframes"] = (kf != 0).astype(np.float32)
    # The prompt's AdaLN modality per text row: sd.cpp splices the keyframes into the prompt as vision rows (tag 0).
    if "tags" in shapes:
        tags = np.fromfile(f"{src}/tags.bin", dtype=np.int32)
        if np.any(tags != 1):
            out["text_tags"] = tags.astype(np.float32)
    save_file(out, dst)
    for n, a in out.items():
        print(n, a.shape, float(np.abs(a).mean()))


if __name__ == "__main__":
    main()
