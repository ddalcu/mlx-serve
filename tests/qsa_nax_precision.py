#!/usr/bin/env python3
"""Float64 oracle for the stock and NAX QSA gather kernels (mx.fast.metal_kernel + numpy).

Default: production bf16-store bar vs numpy float64, then a named f32-output
section. --sweep-terms / --bench exit non-zero if a comparison fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import mlx.core as mx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HQ, HK, HD, RATIO, NSG, BK = 24, 2, 256, 4, 2, 32
GQA = HQ // HK
SCALE = 1.0 / 16.0
INT_MAX = np.iinfo(np.int32).max
BF16_STORE_FLOOR = 2.0e-3

THREE_BLOCK = """      NAXTile<T,1,1> Shi, Slo, Stail;
      for(short j=0;j<8;++j) {
        Shi.frag_at(0,0)[j]=T(S.frag_at(0,ik)[j]);
        float residual=S.frag_at(0,ik)[j]-float(Shi.frag_at(0,0)[j]);
        Slo.frag_at(0,0)[j]=T(residual);
        Stail.frag_at(0,0)[j]=T(residual-float(Slo.frag_at(0,0)[j]));
      }
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Shi.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Slo.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Stail.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
"""

ONE_BLOCK = """      NAXTile<T,1,1> Shi;
      for(short j=0;j<8;++j) {
        Shi.frag_at(0,0)[j]=T(S.frag_at(0,ik)[j]);
      }
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Shi.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
"""

TWO_BLOCK = """      NAXTile<T,1,1> Shi, Slo;
      for(short j=0;j<8;++j) {
        Shi.frag_at(0,0)[j]=T(S.frag_at(0,ik)[j]);
        float residual=S.frag_at(0,ik)[j]-float(Shi.frag_at(0,0)[j]);
        Slo.frag_at(0,0)[j]=T(residual);
      }
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Shi.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Slo.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
"""


def zig_metal(name: str, path: str | None = None) -> str:
    text = open(path or os.path.join(ROOT, "src/transformer.zig")).read()
    start = text.find("const " + name + " =")
    if start < 0:
        raise RuntimeError("missing kernel " + name)
    end = text.find("\n;", start)
    out = []
    for line in text[start:end].splitlines():
        p = line.find("\\\\")
        if p >= 0:
            out.append(line[p + 2 :])
    return "\n".join(out) + "\n"


def nax_source(terms: int) -> str:
    src = open(os.path.join(ROOT, "src/kernels/qsa_nax.metal")).read()
    if THREE_BLOCK in src:
        block = THREE_BLOCK
    elif TWO_BLOCK in src:
        block = TWO_BLOCK
    elif ONE_BLOCK in src:
        block = ONE_BLOCK
    else:
        raise RuntimeError("PV term block not found in qsa_nax.metal")
    replacement = {1: ONE_BLOCK, 2: TWO_BLOCK, 3: THREE_BLOCK}[terms]
    if block == replacement:
        return src
    return src.replace(block, replacement)


def f32_out(src: str) -> str:
    return (
        src.replace("device T* Op=out", "device float* Op=out")
        .replace("device T* Optr = out", "device float* Optr = out")
        .replace("T(Ofrag[id].x * inv)", "(Ofrag[id].x * inv)")
        .replace("T(Ofrag[id].y * inv)", "(Ofrag[id].y * inv)")
    )


_KERNELS: dict = {}


def kernel(kind: str, terms: int = 3, out_f32: bool = True):
    key = (kind, terms, out_f32)
    if key in _KERNELS:
        return _KERNELS[key]
    if kind == "nax":
        src = nax_source(terms)
        header = open(os.path.join(ROOT, "src/kernels/qsa_nax_header.metal")).read()
        name = f"qsa_nax_t{terms}_{'f32' if out_f32 else 'bf16'}"
    else:
        src = zig_metal("ATTN_QSA256_KERNEL_SOURCE")
        header = zig_metal("ATTN256_KERNEL_HEADER") + zig_metal("ATTN_QSA256_KERNEL_HEADER")
        name = f"qsa_stock_{'f32' if out_f32 else 'bf16'}"
    if out_f32:
        src = f32_out(src)
    k = mx.fast.metal_kernel(
        name=name,
        input_names=["q", "k", "v", "scl", "blocks"],
        output_names=["out"],
        source=src,
        header=header,
        ensure_row_contiguous=False,
        atomic_outputs=False,
    )
    _KERNELS[key] = k
    return k


def run_kernel(k, q, kv_k, kv_v, blocks, out_f32, B, S, nax=False):
    scl = mx.array(np.array([SCALE], dtype=np.float32))
    dt = mx.float32 if out_f32 else mx.bfloat16
    tmpl = [("T", mx.bfloat16), ("NSG", NSG), ("BK", BK), ("RATIO", RATIO)]
    out = k(
        inputs=[q, kv_k, kv_v, scl, blocks],
        output_shapes=[(B, HQ, S, HD)],
        output_dtypes=[dt],
        grid=(S * 32, HK * NSG, B),
        threadgroup=(32, NSG, 1),
        template=tmpl,
    )[0]
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def make_blocks(B, S, KV, KB, rng):
    ids = np.full((B, S, KB), INT_MAX, dtype=np.int32)
    for b in range(B):
        for s in range(S):
            complete = (KV - S + s + 1) // RATIO
            choices = np.arange(complete, dtype=np.int32)
            rng.shuffle(choices)
            n = min(KB, complete)
            picked = np.sort(choices[:n])
            ids[b, s, :n] = picked
    return ids


def positions_for(s, KV, S, KB, ids_row):
    p = KV - S + s
    complete = (p + 1) // RATIO
    n = min(KB, complete)
    pos = []
    for j in range(n):
        for r in range(RATIO):
            pos.append(int(ids_row[j]) * RATIO + r)
    for r in range(complete * RATIO, p + 1):
        pos.append(r)
    return np.array(pos, dtype=np.int64)


def f64_rows(Q, K, V, ids, rows, heads):
    """Q/K/V are f32 [B,H,S/KV,D] contiguous. Returns (flat_index, value) lists."""
    B, _, S, _ = Q.shape
    KV = K.shape[2]
    KB = ids.shape[2]
    indices = []
    values = []
    for b in range(B):
        for s in rows:
            if s < 0:
                s = S + s
            for h in heads:
                pos = positions_for(s, KV, S, KB, ids[b, s])
                q = Q[b, h, s].astype(np.float64)
                k = K[b, h // GQA, pos].astype(np.float64)
                scores = (k @ q) * SCALE
                m = np.max(scores)
                w = np.exp(scores - m)
                w /= np.sum(w)
                v = V[b, h // GQA, pos].astype(np.float64)
                y = w @ v
                base = ((b * HQ + h) * S + s) * HD
                for d in range(HD):
                    indices.append(base + d)
                    values.append(y[d])
    return np.array(indices), np.array(values)


def make_qkv(B, S, KV, rng, kind="uniform"):
    if kind == "uniform":
        q = rng.uniform(-1, 1, (B, S, HQ, HD)).astype(np.float32)
        k = rng.uniform(-1, 1, (B, KV, HK, HD)).astype(np.float32)
        v = rng.uniform(-1, 1, (B, KV, HK, HD)).astype(np.float32)
    elif kind == "relu_zero":
        q = rng.normal(0, 0.15, (B, S, HQ, HD)).astype(np.float32)
        k = rng.normal(0, 0.15, (B, KV, HK, HD)).astype(np.float32)
        v = rng.normal(0, 0.15, (B, KV, HK, HD)).astype(np.float32)
        # ReLU-zero mass on the indexer side: most keys contribute nothing.
        drop = rng.random((B, KV, HK, 1)) < 0.7
        k = np.where(drop, 0.0, k)
    else:
        raise RuntimeError(kind)
    q_t = np.transpose(q, (0, 2, 1, 3))
    k_t = np.transpose(k, (0, 2, 1, 3))
    v_t = np.transpose(v, (0, 2, 1, 3))
    q_mx = mx.array(q_t).astype(mx.bfloat16)
    k_mx = mx.array(k_t).astype(mx.bfloat16)
    v_mx = mx.array(v_t).astype(mx.bfloat16)
    q_ref = np.array(q_mx.astype(mx.float32))
    k_ref = np.array(k_mx.astype(mx.float32))
    v_ref = np.array(v_mx.astype(mx.float32))
    return q_mx, k_mx, v_mx, q_ref, k_ref, v_ref


def errors_vs_f64(y, indices, reference):
    err = np.abs(y.reshape(-1)[indices] - reference)
    return float(np.max(err)), float(np.sqrt(np.mean(err * err))), err


def eval_case(B, S, KV, KB, seed, kind, terms, rows=None, heads=None):
    rng = np.random.default_rng(seed)
    ids = make_blocks(B, S, KV, KB, np.random.RandomState(seed))
    q, k, v, q_t, k_t, v_t = make_qkv(B, S, KV, rng, kind)
    blocks = mx.array(ids)
    mx.eval(q, k, v, blocks)
    if rows is None:
        rows = [0, 1, S - 2, S - 1]
    if heads is None:
        heads = [0, 7, 11, 12, 23]
    idx, ref = f64_rows(q_t, k_t, v_t, ids, rows, heads)
    stock = run_kernel(kernel("stock"), q, k, v, blocks, True, B, S, nax=False)
    nax = run_kernel(kernel("nax", terms), q, k, v, blocks, True, B, S, nax=True)
    stock_b = run_kernel(kernel("stock", out_f32=False), q, k, v, blocks, False, B, S, nax=False)
    nax_b = run_kernel(kernel("nax", terms, out_f32=False), q, k, v, blocks, False, B, S, nax=True)
    s_max, s_rmse, s_err = errors_vs_f64(stock, idx, ref)
    n_max, n_rmse, n_err = errors_vs_f64(nax, idx, ref)
    sb_max, sb_rmse, sb_err = errors_vs_f64(stock_b, idx, ref)
    nb_max, nb_rmse, nb_err = errors_vs_f64(nax_b, idx, ref)
    ratio = n_err / np.maximum(s_err, 1e-30)
    ratio_b = nb_err / np.maximum(sb_err, 1e-30)
    return {
        "B": B,
        "S": S,
        "KV": KV,
        "KB": KB,
        "kind": kind,
        "terms": terms,
        "checked": int(idx.size),
        "stock_max": s_max,
        "stock_rmse": s_rmse,
        "nax_max": n_max,
        "nax_rmse": n_rmse,
        "nax_le_1p5": bool(np.all(n_err <= np.maximum(s_err * 1.5, 2.5e-6))),
        "p99_ratio": float(np.quantile(ratio, 0.99)),
        "max_ratio": float(np.max(ratio)),
        "stock_bf16_max": sb_max,
        "nax_bf16_max": nb_max,
        "nax_bf16_finite": bool(np.all(np.isfinite(nax_b))),
        "nax_bf16_le_1p5": bool(np.all(nb_err <= np.maximum(sb_err * 1.5, BF16_STORE_FLOOR))),
        "max_ratio_bf16": float(np.max(ratio_b)),
        "n_err": n_err,
        "s_err": s_err,
        "nb_err": nb_err,
        "sb_err": sb_err,
    }


def med_ms(f, reps=9, warm=4):
    for _ in range(warm):
        mx.eval(f())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(f())
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e3


def chained(step, blocks, R):
    S = int(blocks.shape[1])

    def f():
        x = blocks
        for _ in range(R):
            out = step(x)
            glue = mx.astype(mx.reshape(out[0, 0, :, 0], (1, S, 1)), mx.int32) * 0
            x = blocks + glue
        return x

    return f


def marginal(step, blocks, R=16, R0=4):
    hi = med_ms(chained(step, blocks, R), reps=7, warm=3)
    lo = med_ms(chained(step, blocks, R0), reps=7, warm=2)
    return (hi - lo) / (R - R0)


def make_bench_inputs(S, nb, kb=512):
    KV = max(S, nb * RATIO)
    rng = np.random.default_rng(7)
    ids = make_blocks(1, S, KV, kb, np.random.RandomState(7))
    q, k, v, _, _, _ = make_qkv(1, S, KV, rng, "uniform")
    blocks = mx.array(ids)
    mx.eval(q, k, v, blocks)
    return q, k, v, blocks, KV


def step_for(kind, terms, q, k, v, B, S):
    kern = kernel(kind, terms, out_f32=False)
    scl = mx.array(np.array([SCALE], dtype=np.float32))
    nax = kind == "nax"
    tmpl = [("T", mx.bfloat16), ("NSG", NSG), ("BK", BK), ("RATIO", RATIO)]

    def step(blocks):
        return kern(
            inputs=[q, k, v, scl, blocks],
            output_shapes=[(B, HQ, S, HD)],
            output_dtypes=[mx.bfloat16],
            grid=(S * 32, HK * NSG, B),
            threadgroup=(32, NSG, 1),
            template=tmpl,
        )[0]

    return step


CASES = [
    (2, 17, 17, 512, 4289 + 17, "uniform"),
    (1, 40, 101, 6, 4289 + 40, "uniform"),
    (1, 65, 65599, 512, 4289 + 65, "uniform"),
    (1, 40, 101, 6, 99, "relu_zero"),
]


def cmd_parity(terms: int) -> int:
    failed = 0
    print(f"bf16 store vs f64 (floor={BF16_STORE_FLOOR:.3e}, one bf16 ulp at the observed stock output scale)")
    print(f"{'kind':>10} {'S':>4} {'KV':>6} {'KB':>4} {'t':>2} {'st_bf16':>11} {'nax_bf16':>11} {'finite':>6} {'bar':>5}")
    rows = []
    for B, S, KV, KB, seed, kind in CASES:
        r = eval_case(B, S, KV, KB, seed, kind, terms)
        rows.append(r)
        ok = r["nax_bf16_finite"] and r["nax_bf16_le_1p5"]
        if not ok:
            failed += 1
        print(
            f"{kind:>10} {S:4d} {KV:6d} {KB:4d} {terms:2d} {r['stock_bf16_max']:11.3e} {r['nax_bf16_max']:11.3e} "
            f"{str(r['nax_bf16_finite']):>6} {str(ok):>5}"
        )
    print("f32 output vs f64")
    print(f"{'kind':>10} {'S':>4} {'KV':>6} {'st_f32':>11} {'nax_f32':>11} {'bar':>5}")
    for r in rows:
        ok_f = bool(np.isfinite(r["nax_max"])) and r["nax_le_1p5"]
        if not ok_f:
            failed += 1
        print(
            f"{r['kind']:>10} {r['S']:4d} {r['KV']:6d} {r['stock_max']:11.3e} {r['nax_max']:11.3e} {str(ok_f):>5}"
        )
    if failed:
        print(f"FAIL: {failed} cases")
        return 1
    print("PASS: NAX bf16-store per-element error <= max(1.5 * stock, floor) vs f64")
    return 0


def cmd_sweep():
    print("term-count error vs f64 (f32 kernel output AND bf16 store, sampled sentinel rows)")
    print(
        f"{'terms':>5} {'kind':>10} {'S':>4} {'KV':>6} {'st_f32':>11} {'nax_f32':>11} "
        f"{'st_bf16':>11} {'nax_bf16':>11} {'f32*1.5':>8} {'bf16*1.5':>8}"
    )
    rows = []
    for terms in (1, 2, 3):
        for B, S, KV, KB, seed, kind in CASES:
            r = eval_case(B, S, KV, KB, seed, kind, terms)
            rows.append(r)
            print(
                f"{terms:5d} {kind:>10} {S:4d} {KV:6d} {r['stock_max']:11.3e} {r['nax_max']:11.3e} "
                f"{r['stock_bf16_max']:11.3e} {r['nax_bf16_max']:11.3e} "
                f"{str(r['nax_le_1p5']):>8} {str(r['nax_bf16_le_1p5']):>8}"
            )
    print("\nms per term at S=4096 KB=512 kv=65536 (chained marginal)")
    q, k, v, blocks, KV = make_bench_inputs(4096, 16384, 512)
    print(f"kv={KV}")
    print(f"{'arm':>10} {'ms':>10}")
    stock_ms = marginal(step_for("stock", 3, q, k, v, 1, 4096), blocks)
    print(f"{'stock':>10} {stock_ms:10.3f}")
    term_ms = {}
    for terms in (1, 2, 3):
        ms = marginal(step_for("nax", terms, q, k, v, 1, 4096), blocks)
        term_ms[terms] = ms
        print(f"{'nax-t'+str(terms):>10} {ms:10.3f}")
    failed = 0
    for r in rows:
        if r["terms"] == 2 and (not r["nax_bf16_finite"] or not r["nax_bf16_le_1p5"]):
            failed += 1
        if not np.isfinite(r["nax_max"]) or not np.isfinite(r["nax_bf16_max"]):
            failed += 1
    if failed:
        print(f"FAIL: {failed} production comparisons")
        return 1
    return 0


def cmd_bench(terms: int):
    print(f"chained marginal ms, S=4096, NAX terms={terms}, KB=512")
    print(f"{'nb':>8} {'kv':>8} {'stock':>10} {'nax':>10} {'speedup':>8}")
    table = []
    for nb in (1024, 16384, 65536):
        q, k, v, blocks, KV = make_bench_inputs(4096, nb, 512)
        sm = marginal(step_for("stock", 3, q, k, v, 1, 4096), blocks)
        nm = marginal(step_for("nax", terms, q, k, v, 1, 4096), blocks)
        table.append((nb, KV, sm, nm))
        print(f"{nb:8d} {KV:8d} {sm:10.3f} {nm:10.3f} {sm / nm:8.2f}x")
    bad = any(not np.isfinite(sm) or not np.isfinite(nm) or sm <= 0 or nm <= 0 for _, _, sm, nm in table)
    if bad:
        print("FAIL: non-finite or non-positive marginal times")
        return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-terms", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--terms", type=int, default=2, choices=(1, 2, 3))
    args = p.parse_args()
    if args.sweep_terms:
        return cmd_sweep()
    if args.bench:
        return cmd_bench(args.terms)
    return cmd_parity(args.terms)


if __name__ == "__main__":
    sys.exit(main())
