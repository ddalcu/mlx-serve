"""vMLX reference fixtures for src/jangtq2.zig, from the real JANGH4 routed-expert banks of layers 0 (4-bit down) and
5 (6-bit down). vMLX's own kernels (vmlx_engine/jangh at v1.6.77, under the MLX 0.32.3 mlx-serve pins) run each case
and the case file stores its inputs, every intermediate of TQSwitchGLU.routed() and the output; the weights stay in
the bundle. The default set holds the NAX prefill references plus routed() under the other inputs `jangtq2.moe`
admits (f16/f32 activations, int32 indices, f32 scores, a swiglu_limit) and the fused gate/up kernel over an input
width with a partial last block; `--steel` writes the non-NAX prefill references instead (vMLX's
JANGTQ2_PREFILL=steel). Regenerate with
`PYTHONPATH=<vmlx checkout> python tests/dump_jangtq2_fixtures.py --bundle <JANGH4 dir> --out <dir> [--steel]` and
run the gated test with `zig build test-build -Dtest-filter=jangtq2 && JANGTQ2_FIXTURES=<dir> ./zig-out/tests/test`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ARGS = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ARGS.add_argument("--bundle", required=True, help="the JANGH4 bundle (JANGQ-AI/Qwen3.8-Flash-Next-JANGH4)")
ARGS.add_argument("--out", required=True, help="fixture directory to write")
ARGS.add_argument("--steel", action="store_true", help="non-NAX prefill references")
OPTS = ARGS.parse_args()
if OPTS.steel:
    os.environ["JANGTQ2_PREFILL"] = "steel"  # read when vmlx_engine is imported

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from vmlx_engine.jangh import kernels as K  # noqa: E402
from vmlx_engine.jangh.switch import TQSwitchGLU  # noqa: E402

D, I, E, TOPK = 2560, 640, 512, 10
DECODE_MAX_TOKENS = 96
LIMIT = 1.25
LAYERS = (0, 5)


def load_bank(bundle: str, layer: int) -> dict[str, mx.array]:
    weight_map = json.load(open(os.path.join(bundle, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for p in ("gate", "up", "down"):
        for leaf in ("tq2_packed", "tq2_scales"):
            name = f"model.layers.{layer}.mlp.switch_mlp.{p}_proj.{leaf}"
            out[f"{p}.{leaf}"] = mx.load(os.path.join(bundle, weight_map[name]))[name]
    return out


def build_module(bank: dict[str, mx.array], limit: float = 0.0) -> TQSwitchGLU:
    bits_gu = bank["gate.tq2_packed"].shape[2] * 32 // D
    bits_dn = bank["down.tq2_packed"].shape[2] * 32 // I
    mod = TQSwitchGLU(D, I, E, bits_gu, bits_dn, limit, "hadamard32", "hadamard32")
    for p, lin in (("gate", mod.gate_proj), ("up", mod.up_proj), ("down", mod.down_proj)):
        lin.tq2_packed = bank[f"{p}.tq2_packed"]
        lin.tq2_scales = bank[f"{p}.tq2_scales"]
    # The knobs vmlx_engine/models/qwen4_exp/loader.py sets for this family.
    mod.decode_max_tokens = DECODE_MAX_TOKENS
    mod.use_weighted_unsort = True
    for lin in (mod.gate_proj, mod.up_proj, mod.down_proj):
        lin.use_h32_rows = True
    return mod


def routing(rng: np.random.Generator, tokens: int) -> tuple[mx.array, mx.array]:
    inds = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(tokens)]).astype(np.uint32)
    logits = rng.standard_normal((tokens, TOPK)).astype(np.float32) * 2.0
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return mx.array(inds), mx.array(p).astype(mx.bfloat16)


def routed(mod: TQSwitchGLU, x: mx.array, inds: mx.array, scores: mx.array) -> mx.array:
    t = x.shape[0]
    return mod.routed(x.reshape(1, t, D), inds.reshape(1, t, TOPK), scores.reshape(1, t, TOPK)).reshape(t, D)


def decode_case(mod: TQSwitchGLU, x: mx.array, inds: mx.array, scores: mx.array) -> dict[str, mx.array]:
    g, u, d = mod.gate_proj, mod.up_proj, mod.down_proj
    xr = K.h32_rows(x, x.dtype)
    h = K.gather_qmv(xr, g.tq2_packed, g.tq2_scales, None, inds.reshape(-1), g.bits, x_per_dispatch=False,
                     packed_u=u.tq2_packed, scales_u=u.tq2_scales, limit=0.0, rotate=False)
    hr = K.h32_rows(h, h.dtype)
    y = K.gather_qmv_weighted_down(hr, d.tq2_packed, d.tq2_scales, None, inds, scores, d.bits, x.dtype, rotate=False)
    out = routed(mod, x, inds, scores)
    mx.eval(xr, h, hr, y, out)
    if not mx.array_equal(y, out).item():
        raise SystemExit("decode decomposition differs from TQSwitchGLU.routed")
    return {"x": x, "inds": inds, "scores": scores, "xr": xr, "h": h, "hr": hr, "out": out}


def prefill_case(mod: TQSwitchGLU, x: mx.array, inds: mx.array, scores: mx.array) -> dict[str, mx.array]:
    g, u, d = mod.gate_proj, mod.up_proj, mod.down_proj
    idx = inds.reshape(-1)
    order = mx.argsort(idx)
    inv = mx.argsort(order)
    idx_s = idx[order]
    xr = K.h32_rows(x, x.dtype)
    xs = xr[order // TOPK]
    h = K.gather_qmm_sorted(xs, g.tq2_packed, g.tq2_scales, None, idx_s, g.bits,
                            packed_u=u.tq2_packed, scales_u=u.tq2_scales, limit=0.0)
    hr = K.h32_rows(h, h.dtype)
    y = K.gather_qmm_sorted(hr, d.tq2_packed, d.tq2_scales, None, idx_s, d.bits)
    w = K.weighted_unsort(y, inv, scores)
    out = routed(mod, x, inds, scores)
    mx.eval(order, inv, idx_s, xr, xs, h, hr, y, w, out)
    if not mx.array_equal(w, out).item():
        raise SystemExit("prefill decomposition differs from TQSwitchGLU.routed")
    return {"x": x, "inds": inds, "scores": scores, "order": order.astype(mx.uint32), "inv": inv.astype(mx.uint32),
            "idx_sorted": idx_s, "xr": xr, "xs": xs, "h": h, "hr": hr, "y": y, "out": out}


class Writer:
    def __init__(self, out: str, bundle: str, steel: bool):
        self.out = out
        self.manifest = {"mlx": mx.__version__, "bundle": bundle, "cases": []}
        if steel:
            self.manifest["prefill"] = "steel"
        os.makedirs(out, exist_ok=True)

    def save(self, name: str, layer: int, kind: str, arrays: dict, **extra) -> None:
        mx.eval(*arrays.values())
        mx.save_safetensors(os.path.join(self.out, name + ".safetensors"), arrays)
        self.manifest["cases"].append({"name": name, "layer": layer, "kind": kind, **extra})
        print(name, flush=True)

    def close(self) -> None:
        with open(os.path.join(self.out, "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=2)


def routed_cases(w: Writer, layer: int, bank: dict, mod: TQSwitchGLU, rng: np.random.Generator) -> None:
    """routed() at 3 (gather) and 97 (sorted GEMM) tokens with f16/f32 activations, int32 indices and f32 scores,
    and with swiglu_limit. The fused f32 NAX tile does not fit threadgroup memory, so NAX has no f32 prefill case."""
    for tokens in (3, 97):
        for dt, label in ((mx.float16, "f16"), (mx.float32, "f32")):
            if dt == mx.float32 and tokens > DECODE_MAX_TOKENS:
                continue
            xt = mx.array(rng.standard_normal((tokens, D)).astype(np.float32)).astype(dt)
            it, st = routing(rng, tokens)
            it, st = it.astype(mx.int32), st.astype(mx.float32)
            w.save(f"L{layer}_routed_{label}_T{tokens}", layer, "routed",
                   {"x": xt, "inds": it, "scores": st, "out": routed(mod, xt, it, st)}, tokens=tokens, limit=0.0)
        xt = mx.array(rng.standard_normal((tokens, D)).astype(np.float32)).astype(mx.bfloat16)
        it, st = routing(rng, tokens)
        out = routed(build_module(bank, LIMIT), xt, it, st)
        if mx.array_equal(out, routed(mod, xt, it, st)).item():
            raise SystemExit("swiglu_limit did not change the output: pick a smaller LIMIT")
        w.save(f"L{layer}_routed_limit_T{tokens}", layer, "routed",
               {"x": xt, "inds": it, "scores": st, "out": out}, tokens=tokens, limit=LIMIT)


def tail_case(w: Writer, layer: int, mod: TQSwitchGLU, rng: np.random.Generator) -> None:
    """The fused gate/up decode kernel over the banks' first 2080 inputs: unguarded blocks, then a guarded tail."""
    g, u = mod.gate_proj, mod.up_proj
    x = mx.array(rng.standard_normal((3, D)).astype(np.float32)).astype(mx.bfloat16)
    inds, _ = routing(rng, 3)
    idx = inds.reshape(-1)
    xr = K.h32_rows(x, x.dtype)
    cols = 2080
    words = cols * g.bits // 32
    w.save(f"L{layer}_tl_fused", layer, "tl_fused", {
        "x": xr[:, :cols], "idx": idx,
        "out": K.gather_qmv(xr[:, :cols], g.tq2_packed[:, :, :words], g.tq2_scales, None, idx, g.bits,
                            x_per_dispatch=False, packed_u=u.tq2_packed[:, :, :words], scales_u=u.tq2_scales,
                            limit=0.0)}, cols=cols)


def main() -> None:
    if mx.__version__ != "0.32.3":
        raise SystemExit(f"expected MLX 0.32.3 (the mlx-serve pin), got {mx.__version__}")
    if K.nax_available() == OPTS.steel:
        raise SystemExit("--steel needs vMLX's steel arm, the default its NAX arm (an M5-class GPU)")
    w = Writer(OPTS.out, os.path.abspath(OPTS.bundle), OPTS.steel)
    rng = np.random.default_rng(20261009)
    extra_rng = np.random.default_rng(20261011 if OPTS.steel else 20261010)
    for layer in LAYERS:
        bank = load_bank(OPTS.bundle, layer)
        mod = build_module(bank)
        for tokens in (1, 3, 96, 97, 300):
            x = mx.array(rng.standard_normal((tokens, D)).astype(np.float32)).astype(mx.bfloat16)
            inds, scores = routing(rng, tokens)
            kind = "decode" if tokens <= DECODE_MAX_TOKENS else "prefill"
            w.save(f"L{layer}_T{tokens}_{kind}", layer, kind,
                   (decode_case if kind == "decode" else prefill_case)(mod, x, inds, scores), tokens=tokens)
        if OPTS.steel:
            for dt, label, limit in ((mx.float16, "f16", 0.0), (mx.float32, "f32", 0.0), (mx.bfloat16, "limit", LIMIT)):
                xt = mx.array(extra_rng.standard_normal((97, D)).astype(np.float32)).astype(dt)
                it, st = routing(extra_rng, 97)
                if label != "limit":
                    it, st = it.astype(mx.int32), st.astype(mx.float32)
                w.save(f"L{layer}_routed_{label}_T97", layer, "routed",
                       {"x": xt, "inds": it, "scores": st, "out": routed(build_module(bank, limit), xt, it, st)},
                       tokens=97, limit=limit)
        else:
            tail_case(w, layer, mod, extra_rng)
            routed_cases(w, layer, bank, mod, extra_rng)
        del bank, mod
        mx.clear_cache()
    w.close()


if __name__ == "__main__":
    sys.exit(main())
