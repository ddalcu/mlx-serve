#!/usr/bin/env python3
"""
SeedVR2 parity oracle — dumps reference activations + the true weight manifest.

THIS SCRIPT EXECUTES THE REFERENCE. Unlike the transcription-only fixtures in
this repo, everything it emits is produced by running ByteDance's own code, so
when it disagrees with `docs/seedvr2-arch.md` or `src/seedvr2_*.zig`, IT WINS
and the transcription is the bug.

Requirements (none are vendored, and none are installed by the repo):
    pip install torch diffusers safetensors einops omegaconf rotary-embedding-torch
    git clone https://github.com/ByteDance-Seed/SeedVR  # for models/, common/
    huggingface-cli download ByteDance-Seed/SeedVR2-3B  # ema_vae.pth, seedvr2_ema_3b.pth

The reference imports `flash_attn` and `apex` at module scope. Neither builds on
macOS and neither is needed for CPU fp32 reference math, so `--stub-cuda`
installs import stubs that RAISE on use rather than silently returning wrong
numbers — an oracle that cannot execute a path must fail loudly, not fake it.

Usage
-----
    # 1. The cheap one. No checkpoint needed for the manifest of a built model,
    #    but the REAL key list needs the real file:
    python3 tests/dump_seedvr2_fixtures.py manifest \\
        --dit /path/to/seedvr2_ema_3b.pth --out tests/fixtures/seedvr2/

    # 2. Window partitions — pins src/seedvr2_window.zig. No checkpoint at all.
    python3 tests/dump_seedvr2_fixtures.py windows \\
        --seedvr-src /path/to/SeedVR --out tests/fixtures/seedvr2/

    # 3. VAE round-trip activations (needs the 1 GB VAE only, not the DiT).
    python3 tests/dump_seedvr2_fixtures.py vae \\
        --seedvr-src /path/to/SeedVR --vae /path/to/ema_vae.pth \\
        --out tests/fixtures/seedvr2/

Then, on the Zig side:
    SEEDVR2_FIXTURES=tests/fixtures/seedvr2 zig build test -Dtest-filter="seedvr2"

Conventions this repo already enforces, and this script follows
---------------------------------------------------------------
* fp32 on CPU, always. MPS fp16 quietly decorrelates deep conv stacks, and a
  parity number dumped on MPS is not evidence of anything.
* Raw little-endian f32 blobs + a JSON sidecar of shapes. No pickles in
  fixtures — a fixture you cannot read without torch is not a fixture.
* Every dump asserts its inputs are non-zero and finite BEFORE writing. A
  silently-zeroed rotary buffer (transformers >= 5.x does this to custom models)
  produces a beautiful cosine of 1.0 between two piles of zeros.
"""

import argparse
import json
import os
import sys


# --------------------------------------------------------------------------
# Import stubs. These RAISE. A stub that returns a plausible tensor turns an
# unrunnable path into a wrong number, which is worse than a crash.
# --------------------------------------------------------------------------

def _varlen_sdpa(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None,
                 max_seqlen_q=None, max_seqlen_k=None, **kw):
    """Exact stand-in for `flash_attn_varlen_func`.

    flash-attn is an IO-aware reordering of exact attention, not an
    approximation, so replacing it with SDPA over the same segments is
    numerically faithful rather than a convenient fudge. This is the ONE
    substitution in the DiT path and it is recorded in the fixture sidecar.

    q/k/v are `[total_tokens, heads, head_dim]` with segment boundaries in
    `cu_seqlens_*`. SeedVR2's DiT is bidirectional — `flash_attn_varlen_func`
    defaults to `causal=False` — so no mask is applied. If a caller ever asks
    for causal, refuse rather than silently returning the wrong thing.
    """
    import torch
    import torch.nn.functional as F

    if kw.get("causal", False):
        raise RuntimeError("_varlen_sdpa: causal=True is not implemented; "
                           "SeedVR2's DiT does not use it, so this would be "
                           "a silent wrong answer.")
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    assert len(cu_q) == len(cu_k), "mismatched segment counts"
    outs = []
    for i in range(len(cu_q) - 1):
        qs, qe = cu_q[i], cu_q[i + 1]
        ks, ke = cu_k[i], cu_k[i + 1]
        if qe == qs:
            continue
        # [seq, h, d] -> [1, h, seq, d]
        qi = q[qs:qe].permute(1, 0, 2).unsqueeze(0).float()
        ki = k[ks:ke].permute(1, 0, 2).unsqueeze(0).float()
        vi = v[ks:ke].permute(1, 0, 2).unsqueeze(0).float()
        o = F.scaled_dot_product_attention(qi, ki, vi)
        outs.append(o.squeeze(0).permute(1, 0, 2))
    return torch.cat(outs, dim=0).to(q.dtype)


def install_cuda_stubs():
    import types

    def _raise(name):
        def f(*a, **k):
            raise RuntimeError(
                f"{name} was called, but this oracle runs CPU fp32 only. "
                f"The code path you are dumping needs a real {name}; dump it "
                f"on a CUDA box instead of trusting a stub."
            )
        return f

    import importlib.machinery

    # A stub module MUST carry a __spec__. diffusers probes optional deps with
    # `importlib.util.find_spec("flash_attn")`, which raises
    # `ValueError: flash_attn.__spec__ is None` on a bare ModuleType — so a
    # stub without one turns "dependency absent" into a hard crash three
    # imports deep, nowhere near the thing that registered it.
    fa = types.ModuleType("flash_attn")
    fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    fa.__version__ = "0.0.0-stub"
    fa.flash_attn_varlen_func = _varlen_sdpa
    sys.modules.setdefault("flash_attn", fa)

    apex = types.ModuleType("apex")
    apexnorm = types.ModuleType("apex.normalization")
    # apex's fused norms are numerically plain RMS/LayerNorm; substituting the
    # torch versions is the ONE substitution that is defensible, and it is
    # recorded in the sidecar so a reader knows it happened.
    import torch.nn as nn

    class FusedRMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
            super().__init__()
            import torch
            self.eps = eps
            self.weight = (
                nn.Parameter(torch.ones(normalized_shape)) if elementwise_affine else None
            )

        def forward(self, x):
            import torch
            v = x.float().pow(2).mean(-1, keepdim=True)
            out = x.float() * torch.rsqrt(v + self.eps)
            if self.weight is not None:
                out = out * self.weight.float()
            return out.type_as(x)

    apexnorm.FusedRMSNorm = FusedRMSNorm
    apexnorm.FusedLayerNorm = nn.LayerNorm
    import importlib.machinery as _im
    apex.__spec__ = _im.ModuleSpec("apex", None)
    apexnorm.__spec__ = _im.ModuleSpec("apex.normalization", None)
    apex.normalization = apexnorm
    sys.modules.setdefault("apex", apex)
    sys.modules.setdefault("apex.normalization", apexnorm)
    return {"flash_attn": "torch-sdpa-varlen (exact)", "apex.normalization": "torch-equivalent"}


def write_blob(store, name, tensor, meta, extra=None):
    """Stage an f32 tensor for the fixture safetensors. Asserts non-degenerate.

    Fixtures are safetensors, not raw blobs — that is the convention the rest
    of this repo's live parity tests use (see `minimax_h3_vae.zig`), so the Zig
    side reads them with the loader it already has instead of hand-rolling
    file IO and a shape sidecar.
    """
    import torch

    # .clone() is load-bearing: DiagonalGaussianDistribution.mean is a CHUNK
    # VIEW of .parameters, and safetensors refuses to write aliased storage.
    t = tensor.detach().to(torch.float32).contiguous().cpu().clone()
    assert torch.isfinite(t).all(), f"{name} contains non-finite values"
    # A dump that is all-zero is almost always a zeroed buffer, not a result.
    assert t.abs().max() > 0, f"{name} is entirely zero — check for a zeroed buffer"
    store[name] = t
    entry = {"shape": list(t.shape), "dtype": "f32"}
    if extra:
        entry.update(extra)
    meta[name] = entry
    print(f"  staged {name} {list(t.shape)}")


# --------------------------------------------------------------------------
# manifest — the TRUE key list, which is what reconciles src/seedvr2_manifest.zig
# --------------------------------------------------------------------------
def cmd_manifest(args):
    import torch

    os.makedirs(args.out, exist_ok=True)
    print(f"loading {args.dit} (weights only, no model construction)")
    sd = load_state_dict_any(args.dit)

    keys = sorted(sd.keys())
    entries = {k: list(sd[k].shape) for k in keys}

    # The assertion this whole file exists for: report the split/shared
    # boundary as OBSERVED, not as configured.
    split, shared = set(), set()
    for k in keys:
        parts = k.split(".")
        if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
            i = int(parts[1])
            if ".vid." in k or ".txt." in k:
                split.add(i)
            if ".all." in k:
                shared.add(i)
    overlap = sorted(split & shared)
    boundary = min(shared) if shared else None
    print(f"\n  split-weight layers : {sorted(split)}")
    print(f"  shared-weight layers: {sorted(shared)}")
    print(f"  boundary            : {boundary}")
    if overlap:
        print(f"  !! layers with BOTH namings: {overlap}")

    out = {
        "tensors": entries,
        "count": len(keys),
        "split_layers": sorted(split),
        "shared_layers": sorted(shared),
        "shared_boundary": boundary,
        "source": os.path.basename(args.dit),
    }
    with open(os.path.join(args.out, "dit_manifest.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"\nwrote {args.out}/dit_manifest.json  ({len(keys)} tensors)")
    print("Reconcile against src/seedvr2_manifest.zig — the checkpoint wins.")


# --------------------------------------------------------------------------
# windows — pins src/seedvr2_window.zig against the reference partitioner
# --------------------------------------------------------------------------
def cmd_windows(args):
    sys.path.insert(0, args.seedvr_src)
    from models.dit_v2.window import (
        make_720Pwindows_bysize,
        make_shifted_720Pwindows_bysize,
    )

    os.makedirs(args.out, exist_ok=True)
    cases = [
        (1, 45, 80), (1, 68, 120), (5, 45, 80), (1, 32, 32),
        (1, 30, 53), (9, 90, 160), (1, 17, 17), (3, 45, 80),
        (1, 90, 160), (17, 45, 80), (1, 135, 240), (33, 68, 120),
    ]
    out = {}
    for size in cases:
        for label, fn in (("plain", make_720Pwindows_bysize),
                          ("shifted", make_shifted_720Pwindows_bysize)):
            wins = fn(size, (4, 3, 3))
            key = f"{label}_{size[0]}x{size[1]}x{size[2]}"
            out[key] = [[s.start, s.stop] for w in wins for s in w]
            print(f"  {key}: {len(wins)} windows")
    with open(os.path.join(args.out, "windows.json"), "w") as f:
        json.dump({"num_windows": [4, 3, 3], "cases": out}, f, indent=1)
    print(f"\nwrote {args.out}/windows.json")


# --------------------------------------------------------------------------
# vae — encoder/decoder activations for the round-trip parity test
# --------------------------------------------------------------------------
def load_state_dict_any(path):
    """Accept either a torch .pth pickle or a .safetensors file.

    The official ByteDance release ships .pth; the community fp16 mirrors ship
    safetensors. Both carry the same tensors, so the oracle accepts both rather
    than forcing a conversion step that could itself introduce the error we are
    trying to measure.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "module", "ema", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            return sd[key]
    return sd


def cmd_vae(args):
    import torch

    subs = install_cuda_stubs()
    sys.path.insert(0, args.seedvr_src)
    from omegaconf import OmegaConf
    from common.config import create_object

    os.makedirs(args.out, exist_ok=True)
    cfg = OmegaConf.load(
        os.path.join(args.seedvr_src,
                     "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml")
    )
    # `freeze_encoder` is required by VideoAutoencoderKLWrapper but lives in
    # configs_3b/main.yaml's `vae.model` override rather than in the standalone
    # VAE yaml, which only reaches it via `__inherit__`. Loading the VAE yaml
    # directly therefore misses it. Inference-only, so the value is inert —
    # it only picks `torch.no_grad()` vs `nullcontext()` in `forward`, and we
    # call `encode`/`decode` directly.
    cfg.freeze_encoder = False
    vae = create_object(cfg)
    sd = load_state_dict_any(args.vae)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    assert not missing, f"VAE missing {len(missing)} tensors, first: {missing[:5]}"
    vae = vae.float().eval()

    meta = {"_substitutions": subs, "_note": "CPU fp32; scaling_factor NOT applied"}
    store = {}
    torch.manual_seed(0)

    # Frame counts are 4k+1 by construction — the causal head replication means
    # frame 0 stands alone. A non-4k+1 input is a different code path and is
    # deliberately not dumped here.
    # DO NOT dump `vae.encode(x).latent`. VideoAutoencoderKLWrapper.encode
    # returns `p.sample()` — a STOCHASTIC draw from the posterior. As a parity
    # target it is worthless: a correct port scores an imperfect cosine against
    # it forever, with no bug to find, because the difference is the reference's
    # own RNG. Go through the base class and take the DISTRIBUTION.
    from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKL

    for (t, h, w) in [(1, 64, 64), (5, 64, 64), (1, 128, 128), (5, 128, 96)]:
        x = torch.randn(1, 3, t, h, w)
        tag = f"{t}x{h}x{w}"
        with torch.no_grad():
            post = VideoAutoencoderKL.encode(vae, x).latent_dist
            # `parameters` is the raw 32-channel conv_out result (16 mean then
            # 16 logvar) — exactly what our `encode` returns, so it is the
            # tightest possible comparison point.
            moments = post.parameters
            mean = post.mean
            # Decode from the MEAN, not a sample, so the reconstruction is
            # deterministic too.
            rec = vae.decode(mean)
            rec = rec.sample if hasattr(rec, "sample") else rec
        write_blob(store, f"in_{tag}", x, meta)
        write_blob(store, f"moments_{tag}", moments, meta,
                   extra={"note": "B C T H W, 16 mean + 16 logvar, pre-scaling_factor"})
        write_blob(store, f"mean_{tag}", mean, meta,
                   extra={"note": "B C T H W, posterior mean, pre-scaling_factor"})
        write_blob(store, f"recon_{tag}", rec, meta,
                   extra={"note": "decoded from the MEAN, deterministic"})
        print(f"  {tag}: moments {list(moments.shape)}  recon {list(rec.shape)}")

    from safetensors.torch import save_file

    st = os.path.join(args.out, "vae_fixture.safetensors")
    save_file(store, st)
    with open(os.path.join(args.out, "vae_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote {st} ({len(store)} tensors) + vae_meta.json")



# --------------------------------------------------------------------------
# dit — a TINY random-weight NaDiT plus its activations.
#
# The 3B checkpoint is 12 GB as f32 and this box has 24 GB, so a full-size CPU
# reference forward is not on. A small model with the SAME structure is the
# better test anyway: it exercises every decision the port can get wrong —
# window partitioning, the text broadcast/mean-pool, mmrope3d, AdaSingle's
# (d l g) packing, SwiGLU's gate/up naming, and the split/shared weight
# boundary — in seconds, and the fixture carries the weights so the Zig side
# is self-contained. The REAL checkpoint's layout is pinned separately by the
# `manifest` command (635/635 tensors).
# --------------------------------------------------------------------------
def cmd_dit(args):
    import torch

    subs = install_cuda_stubs()
    sys.path.insert(0, args.seedvr_src)
    from models.dit_v2.nadit import NaDiT
    from models.dit_v2 import na

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    # Sized so the committed fixture stays near the repo's existing toy-geometry
    # precedent (src/fixtures/minimax_h3_dit.safetensors, 1.8 MB) while still
    # exercising every structural decision: 4 layers straddling the split/shared
    # boundary at 2, multi-window in BOTH space and time, and a rope that leaves
    # part of head_dim unrotated.
    dim, heads, head_dim, layers, mm_layers = 64, 2, 32, 4, 2
    model = NaDiT(
        vid_in_channels=33,
        vid_out_channels=16,
        vid_dim=dim,
        txt_in_dim=80,
        txt_dim=dim,
        emb_dim=6 * dim,
        heads=heads,
        head_dim=head_dim,
        expand_ratio=4,
        norm="fusedrms",
        norm_eps=1e-5,
        ada="single",
        qk_bias=False,
        qk_norm="fusedrms",
        patch_size=(1, 2, 2),
        num_layers=layers,
        block_type=["mmdit_sr"] * layers,
        mm_layers=mm_layers,
        mlp_type="swiglu",
        rope_type="mmrope3d",
        # rope_dim must be divisible by 3 with an EVEN quotient:
        # rotary_embedding_torch allocates its cache at `dim` but fills
        # `2*(dim//2)` values, so an odd per-axis dim throws. 24//3 = 8 -> 4
        # freqs -> 8 values/axis -> 24 of the 32 head dims rotated, mirroring
        # the real model (126 of 128) rather than rotating all of them.
        rope_dim=24,
        window=[(4, 3, 3)] * layers,
        window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * (layers // 2),
        vid_out_norm="fusedrms",
    ).float().eval()

    # Randomise every parameter: a checkpoint of zeros or ones makes half the
    # possible bugs invisible (a transposed AdaSingle view over constant data
    # is indistinguishable from a correct one).
    with torch.no_grad():
        for p_ in model.parameters():
            p_.normal_(0.0, 0.25)

    # 48x48 latent -> 24x24 tokens: big enough that the 720p-normalised window
    # size (20 tokens) splits it 2x2, and T=2 splits 2x1, so the partition is
    # genuinely multi-window on BOTH the plain and shifted layers. A grid that
    # fits in one window would exercise none of the windowing.
    T, H, W = 2, 48, 48
    vid_shape = torch.tensor([[T, H, W]])
    txt_len = 7
    txt_shape = torch.tensor([[txt_len]])
    vid = torch.randn(T * H * W, 33)
    txt = torch.randn(txt_len, 80)
    timestep = torch.tensor([734.0])

    meta = {"_substitutions": subs,
            "_note": "tiny random-weight NaDiT; CPU fp32",
            "_config": {"vid_dim": dim, "heads": heads, "head_dim": head_dim,
                        "num_layers": layers, "mm_layers": mm_layers,
                        "txt_in_dim": 80, "patch": [1, 2, 2],
                        "vid_shape": [T, H, W], "txt_len": txt_len,
                        "timestep": 734.0}}
    store = {}

    # Capture the per-block outputs: a final-output-only fixture tells you the
    # port is wrong but not WHERE, and with 4 blocks the bisect is free.
    caught = {}

    def mk_hook(i):
        def hook(_mod, _inp, out):
            caught[f"block{i}_vid"] = out[0].detach().clone()
            caught[f"block{i}_txt"] = out[1].detach().clone()
        return hook

    def mk_attn_hook(i):
        def hook(_mod, _inp, out):
            caught[f"block{i}_attn_vid"] = out[0].detach().clone()
            caught[f"block{i}_attn_txt"] = out[1].detach().clone()
        return hook

    def mk_mlp_hook(i):
        def hook(_mod, _inp, out):
            caught[f"block{i}_mlp_vid"] = out[0].detach().clone()
        return hook

    for i, blk in enumerate(model.blocks):
        blk.register_forward_hook(mk_hook(i))
        # Sub-block probes so a divergence localises to attention vs MLP
        # instead of just "block 0 is off".
        blk.attn.register_forward_hook(mk_attn_hook(i))
        blk.mlp.register_forward_hook(mk_mlp_hook(i))
    model.emb_in.register_forward_hook(
        lambda _m, _i, o: caught.__setitem__("emb", o.detach().clone()))
    model.vid_in.register_forward_hook(
        lambda _m, _i, o: caught.__setitem__("vid_in", o[0].detach().clone()))

    with torch.no_grad():
        # disable_cache=False IS LOAD-BEARING, not a performance choice.
        # AdaSingle keys its repeated-embedding cache on
        # f"emb_repeat_{idx}_{branch_tag}". vid_out_ada has layers=["out"], so
        # its idx is 0 and its branch_tag is "vid" — the SAME key the blocks'
        # attn ada already populated. The cache therefore HITS and vid_out_ada
        # silently reuses the attn modulation slice. With the cache disabled the
        # lambda actually runs, computes a (L, 2*dim) slice against a (L, dim)
        # hidden, and raises. Every implementation (ByteDance, numz, ComfyUI
        # core) shares this, so the collision IS the semantics.
        out = model(vid=vid, txt=txt, vid_shape=vid_shape, txt_shape=txt_shape,
                    timestep=timestep, disable_cache=False)

    for k, t in model.state_dict().items():
        # No prefix: the Zig loader looks up the checkpoint's own key names, and
        # state-dict keys (vid_in./txt_in./emb_in./blocks./vid_out) cannot
        # collide with the in_/act_/out_ activation names.
        store[k] = t.detach().float().clone()
    write_blob(store, "in_vid", vid, meta)
    write_blob(store, "in_txt", txt, meta)
    for k in sorted(caught):
        write_blob(store, "act_" + k, caught[k], meta)
    write_blob(store, "out_vid", out.vid_sample, meta)

    from safetensors.torch import save_file
    st = os.path.join(args.out, "seedvr2_dit_tiny.safetensors")
    save_file(store, st)
    with open(os.path.join(args.out, "seedvr2_dit_tiny.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote {st} ({len(store)} tensors) + seedvr2_dit_tiny.json")
    print(f"  output {list(out.vid_sample.shape)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="dump the true DiT weight key list")
    m.add_argument("--dit", required=True, help="seedvr2_ema_3b.pth")
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_manifest)

    w = sub.add_parser("windows", help="dump reference window partitions")
    w.add_argument("--seedvr-src", required=True, help="SeedVR repo checkout")
    w.add_argument("--out", required=True)
    w.set_defaults(fn=cmd_windows)

    d = sub.add_parser("dit", help="dump a tiny NaDiT + its activations")
    d.add_argument("--seedvr-src", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_dit)

    v = sub.add_parser("vae", help="dump VAE encode/decode activations")
    v.add_argument("--seedvr-src", required=True)
    v.add_argument("--vae", required=True, help="ema_vae.pth")
    v.add_argument("--out", required=True)
    v.set_defaults(fn=cmd_vae)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
