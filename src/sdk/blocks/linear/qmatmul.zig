//! sdk/blocks/linear/qmatmul.
//! Owns and 8 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const qmv_cuda = @import("../qmm/qmv_cuda.zig");
const rowqmv = @import("../qmm/rowqmv.zig");
const lane_qmm = @import("../qmm/lane_qmm.zig");
const simd_qmm = @import("../qmm/simd_qmm.zig");
const mlx_gguf = @import("../../quant/gguf/root.zig");
const sushi_pack = @import("../../quant/exl3/sushi_pack.zig");
const fp8_block128 = @import("../../quant/fp8_block128.zig");
const qmv_nax2 = @import("../qmm/qmv_nax2.zig");
const qmm_int8 = @import("../qmm/qmm_int8.zig");
const log = @import("../../util/log.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const mxfp8_qmv = @import("../qmm/mxfp8_qmv.zig");
const lastDim = @import("../../../mlx/array.zig").lastDim;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const BITS_CACHE_CAP = @import("../../quant/params.zig").BITS_CACHE_CAP;
const BitsCache = @import("../../quant/params.zig").BitsCache;
const QuantParams = @import("../../quant/params.zig").QuantParams;
const computeQuantParams = @import("../../quant/params.zig").computeQuantParams;
const quantParamsOrDense = @import("../../quant/params.zig").quantParamsOrDense;
const prefillDqGemm = @import("prefill_dq_gemm.zig").prefillDqGemm;
const VqmmTile = @import("verify_qmm.zig").VqmmTile;
const verifyQmm = @import("verify_qmm.zig").verifyQmm;
const verifyQmmTile = @import("verify_qmm_nax.zig").verifyQmmTile;

// ── Core ops ──

/// A row-exact 4/6/8-bit matmul: Nemotron-H's thin MoE trunk streams faster
/// through `rowqmv`'s per-row matvecs, the dense Qwen3.8 through `simd_qmm`,
/// whose multi-row MMA reads each weight once (`rowqmv_only` = the Nemotron-H trunk). Null outside both.
pub fn rowExactQmm(rowqmv_only: bool, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!rowqmv_only) {
        // The pick follows the weight and the GPU, never the row count: serial and verify rows share a kernel.
        if (naxAvailable()) if (try lane_qmm.qmm(x, w, sc, bi, bits, group_size, s)) |y| return y;
        if (try simd_qmm.qmm(x, w, sc, bi, bits, group_size, s)) |y| return y;
    }
    return rowqmv.qmv(x, w, sc, bi, bits, group_size, s);
}

/// The verify lanes cover 2..7 rows everywhere and take 8+ only with a NAX or shader tile,
/// so simd_qmm starts at 8 rows where there is no tile and stays off where there is.
pub fn simdQmmDecodeMinRows(tile: VqmmTile) c_int {
    return if (tile == .off) 8 else 0;
}

var simd_qmm_decode_rows: ?c_int = null;
/// Affine projections of N to simd_qmm.MAX_ROWS rows go through simd_qmm with no drafter
/// bound (batched decode widths). N comes from `simdQmmDecodeMinRows`.
pub fn simdQmmDecode(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    const min_rows = simd_qmm_decode_rows orelse blk: {
        const v = simdQmmDecodeMinRows(verifyQmmTile());
        simd_qmm_decode_rows = v;
        if (v > 0) log.info("[simd_qmm] batched projections of {d}..{d} rows\n", .{ v, simd_qmm.MAX_ROWS });
        break :blk v;
    };
    if (min_rows <= 0) return null;
    const xs = mlx.getShape(x);
    var rows: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= d;
    if (rows < min_rows) return null;
    return simd_qmm.qmm(x, w, sc, bi, bits, group_size, s);
}

/// MLX's `get_qmv_batch_limit` (quantized.cpp) on the gen-17 GPUs the joined path runs on:
/// below it a transposed qmm runs the per-row qmv kernel, so joined rows reduce like each slot's own.
pub fn qmvBatchLimit(dual_die: bool, k: c_int, n: c_int) c_int {
    const small = k <= 2048 and n <= 2048;
    const mid = k <= 4096 and n <= 4096;
    if (dual_die) return if (small) 32 else if (mid) 18 else 12;
    return if (small) 33 else if (mid) 25 else 13;
}

pub fn qmatmulBits(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, s: mlx.mlx_stream) !mlx.mlx_array {
    if (mlx_gguf.kernels.infoOf(w, sc)) |info| return mlx_gguf.kernels.linear(info, x, w, s);
    if (fp8_block128.isBlockScaled(w, sc)) return fp8_block128.linear(x, w, sc, s);
    if (sushi_pack.isRowScaledFp8(w, sc)) return sushi_pack.fp8Linear(x, w, sc, s);
    // A weight re-ordered in place is read in its tiled layout only.
    if (try lane_qmm.tiledQmm(x, w, s)) |y| return y;
    // Plain BF16 weight: scales array is unset. Used by mixed-precision Unsloth
    // Dynamic checkpoints that leave a subset of layers (e.g. linear_attn
    // projections in Qwen3.6 UD) unquantized. The weight is pre-transposed at
    // load to [in, out] so a single mlx_matmul does the contraction.
    if (sc.ctx == null) {
        var fp_result = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(fp_result);
        try mlx.check(mlx.mlx_matmul(&fp_result, x, w, s));
        return fp_result;
    }

    // Non-affine model-wide mode (nvfp4 / mxfp4 / mxfp8 from config.json):
    // bias-less by construction; bits/group_size come from the config like
    // affine. Checked BEFORE the legacy null-bias heuristic below, which
    // would misroute e.g. an nvfp4 weight to mxfp8.
    if (mode != .affine) {
        if (mode == .mxfp8) {
            if (try mxfp8_qmv.linear(x, w, sc, s)) |y| return y;
            if (try prefillDqGemm(x, w, sc, bi, bits, group_size, mode, s)) |py| return py;
        }
        var result = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_quantized_matmul(
            &result,
            x,
            w,
            sc,
            mlx.mlx_array{ .ctx = null },
            true,
            mlx.mlx_optional_int.some(@intCast(group_size)),
            mlx.mlx_optional_int.some(@intCast(bits)),
            mode.cstr(),
            s,
        ));
        return result;
    }

    if (bi.ctx != null) {
        if (try simdQmmDecode(x, w, sc, bi, bits, group_size, s)) |y| return y;
    }

    // Legacy heuristic for mixed checkpoints whose config declares affine but
    // whose individual tensors lack biases (NVIDIA mxfp8 layers): biases array
    // has null ctx (created with mlx_array_new()).
    const is_mxfp = bi.ctx == null;

    if (is_mxfp) {
        // mxfp8: bits is always 8; infer group_size from scales/weight shape ratio
        const mxfp_bits: u32 = 8;
        const s_shape = mlx.getShape(sc);
        const w_shape = mlx.getShape(w);
        const mxfp_gs: u32 = if (s_shape.len >= 2 and w_shape.len >= 2) blk: {
            const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
            const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
            if (s_cols > 0) break :blk (w_cols * 32) / (s_cols * mxfp_bits);
            break :blk 32;
        } else 32;

        const null_bi = mlx.mlx_array{ .ctx = null };
        var result = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_quantized_matmul(
            &result,
            x,
            w,
            sc,
            null_bi,
            true,
            mlx.mlx_optional_int.some(@intCast(mxfp_gs)),
            mlx.mlx_optional_int.some(@intCast(mxfp_bits)),
            "mxfp8",
            s,
        ));
        return result;
    }

    // Spec-verify fast path: M=2..6-row activations (verify forwards, small
    // decode batches) route through the split-K verify kernel when eligible
    // (see VERIFY_QMM_SOURCE) — stock qmm's qmv/steel dead zone.
    if (try verifyQmm(s, x, w, sc, bi, bits, group_size)) |vy| return vy;
    if (mode == .affine) if (try qmv_cuda.qmm(x, w, sc, bi, bits, group_size, s)) |y| return y;

    // Prefill-width fast path: M >= 2048 dequantizes + runs the steel bf16
    // GEMM (see prefillDqGemm) — stock qmm_t's 32x32x32 tile dead zone.
    if (try prefillDqGemm(x, w, sc, bi, bits, group_size, .affine, s)) |py| return py;

    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_quantized_matmul(
        &result,
        x,
        w,
        sc,
        bi,
        true,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));
    return result;
}
