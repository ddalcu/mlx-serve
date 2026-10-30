//! The QSA block score sheet: the fused NAX kernel (`msv_qsa_score`), its layouts and arms, and the composed chain.

const std = @import("std");
const env = @import("../../../../env.zig");
const kernelSource = @import("../../../../mlx/kernels.zig").source;
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../util/log.zig");
const select = @import("select.zig");
const QsaCfgCache = @import("../../../../mlx/kernel_cache.zig").QsaCfgCache;
const attn256MaxDiff = @import("../attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attn256.zig").attn256RandBf16;
const verifyQmmNaxAvailable = @import("../../linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const OneShotBits = @import("one_shot.zig").OneShotBits;
const qsaWidthBucket = @import("one_shot.zig").widthBucket;
const qsaProbeBitEqual = @import("probe.zig").qsaProbeBitEqual;
const qsaProbeCorruptFirst = @import("probe.zig").qsaProbeCorruptFirst;
const qsaProbeLcgBf16 = @import("probe.zig").qsaProbeLcgBf16;
const qsaArraysAllEqual = @import("../../../testing/parity.zig").qsaArraysAllEqual;
const qsaSelectComposedOps = @import("select.zig").qsaSelectComposedOps;
const qsaSelectKernelEnabled = @import("select.zig").qsaSelectKernelEnabled;
const qsaSelectTopBlocks = @import("select.zig").qsaSelectTopBlocks;
const tieBias = @import("select.zig").tieBias;
const verify_qmm_nax_vars = @import("../../linear/verify_qmm_nax.zig");
const Blocks = @import("params.zig").Blocks;
const ops = @import("../../../../mlx/ops.zig");
const Graph = ops.Graph;
const testing = std.testing;

/// The block geometry the score tests run at (Qwen3.8-Flash-Next's).
const blocks4: Blocks = .{ .size = 4, .top_k = 512 };

pub const QsaScoreLayout = enum(u8) { base = 0, h4 = 1 };

pub var qsa_score_probe_flip_override: bool = false;
pub var qsa_score_layout_override: ?QsaScoreLayout = null;
pub var qsa_score_tf32_override: ?bool = null;
var qsa_score_tf32_env: ?bool = null;
var qsa_score_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var qsa_score_kernel_builds: usize = 0;

/// Test seam: false keeps the composed chain.
pub var qsa_score_fused_override: ?bool = null;

pub fn qsaScoreFusedEnabled() bool {
    return qsa_score_fused_override orelse true;
}

pub fn qsaScoreFusedActive() bool {
    if (!qsaScoreFusedEnabled()) return false;
    if (!verifyQmmNaxAvailable()) return false;
    if (!qsaScoreTf32Enabled()) return false;
    return qsaScoreFusedProbeOk();
}

var qsa_score_probe_state: std.atomic.Value(u8) = .init(0);

fn qsaScoreFusedProbeOk() bool {
    return qsa_score_probe_state.load(.acquire) == 2;
}

pub fn qsaScoreFusedArm() void {
    if (qsa_score_probe_state.load(.acquire) != 0) return;
    if (!qsaScoreFusedEnabled() or !verifyQmmNaxAvailable() or !qsaScoreTf32Enabled()) return;
    const ok = qsaScoreFusedRunProbe();
    qsa_score_probe_state.store(if (ok) 2 else 1, .release);
}

pub fn qsaScoreFusedActiveFor(batch: c_int, n_idx: c_int, idx_hd: c_int) bool {
    return qsaScoreFusedActive() and batch == 1 and n_idx == 4 and idx_hd == 128;
}

fn qsaScoreTf32ValueOn(raw: []const u8) bool {
    var i: usize = 0;
    while (i < raw.len and std.ascii.isWhitespace(raw[i])) : (i += 1) {}
    if (i >= raw.len) return false;
    var sign: i32 = 1;
    if (raw[i] == '-') {
        sign = -1;
        i += 1;
    } else if (raw[i] == '+') {
        i += 1;
    }
    var val: i64 = 0;
    var any = false;
    while (i < raw.len) : (i += 1) {
        const c = raw[i];
        if (c < '0' or c > '9') break;
        any = true;
        val = val * 10 + @as(i64, c - '0');
    }
    if (!any) return false;
    return (if (sign < 0) -val else val) != 0;
}

fn qsaScoreTf32Enabled() bool {
    if (qsa_score_tf32_override) |v| return v;
    if (qsa_score_tf32_env) |v| return v;
    const v = blk: {
        const raw = env.get(.MLX_ENABLE_TF32) orelse break :blk true;
        break :blk qsaScoreTf32ValueOn(std.mem.sliceTo(raw, 0));
    };
    qsa_score_tf32_env = v;
    return v;
}

pub fn qsaScoreFusedEligible(batch: c_int, n_idx: c_int, idx_hd: c_int, q_dt: mlx.mlx_dtype, pooled_dt: mlx.mlx_dtype) bool {
    return qsaScoreFusedActiveFor(batch, n_idx, idx_hd) and
        q_dt == .bfloat16 and pooled_dt == .bfloat16;
}

pub fn qsaScoreFusedEligibleFrom(q: mlx.mlx_array, pooled: mlx.mlx_array) bool {
    if (q.ctx == null or pooled.ctx == null) return false;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(pooled) != 3) return false;
    const qs = mlx.getShape(q);
    const ps = mlx.getShape(pooled);
    if (qs[2] <= 0 or ps[1] <= 0) return false;
    if (qs[3] != ps[2]) return false;
    return qsaScoreFusedEligible(qs[0], qs[1], qs[3], mlx.mlx_array_dtype(q), mlx.mlx_array_dtype(pooled));
}

pub fn qsaPooledRowContiguous(pooled: mlx.mlx_array) bool {
    if (mlx.mlx_array_ndim(pooled) != 3) return false;
    const ps = mlx.getShape(pooled);
    if (ps[0] != 1 or ps[2] != 128) return false;
    const st = mlx.mlx_array_strides(pooled);
    return st[2] == 1 and st[1] == 128;
}

pub fn qsaScoreLayoutFor(rows: c_int) QsaScoreLayout {
    if (qsa_score_layout_override) |v| return v;
    return if (rows >= 128) .h4 else .base;
}

pub fn qsaScoreNsg(layout: QsaScoreLayout) c_int {
    return if (layout == .h4) 8 else 4;
}

pub fn qsaScoreRowsPerTg(layout: QsaScoreLayout, nsg: c_int) c_int {
    return nsg * (if (layout == .h4) @as(c_int, 16) else 4);
}

pub fn qsaScoreNsh(nslab: u32, ngrp: u32, layout: QsaScoreLayout) u32 {
    const slabs_per: u32 = if (layout == .h4) 32 else 1;
    var nsh: u32 = @max(@as(u32, 1), nslab / slabs_per);
    const min_nsh: u32 = @max(@as(u32, 1), (256 + ngrp - 1) / ngrp);
    if (nsh < min_nsh) nsh = min_nsh;
    if (nsh > nslab) nsh = nslab;
    return nsh;
}

pub const QSA_SCORE_KERNEL_HEADER = @import("../kv_quant_attn.zig").QSA_SCORE_KERNEL_HEADER;

// The lane -> (row, col) map (`fm`/`fn`) is the matmul2d cooperative-fragment layout,
// hand-derived from the SDK; every A, B and C fragment must use it. K stays 8 ascending
// 16-deep `op.run` calls: that is the stock tf32 GEMM's order and the bit-identity contract.
pub const QSA_SCORE_KERNEL_SOURCE = kernelSource("kernels/attention/qsa_score.metal");

const QsaScoreCfgKey = struct { nsg: c_int, rows: c_int, nb: c_int, layout: u8 };
var qsa_score_cfgs = QsaCfgCache(QsaScoreCfgKey, 1){};
var qsa_score_engaged_bits: OneShotBits = .{};

fn getQsaScoreKernel() !mlx.mlx_fast_metal_kernel {
    if (!verifyQmmNaxAvailable()) return error.NaxUnavailable;
    if (qsa_score_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "Q", "POOL" };
    const output_names = [_][*:0]const u8{"OUT"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_score",
        in_vec,
        out_vec,
        QSA_SCORE_KERNEL_SOURCE,
        QSA_SCORE_KERNEL_HEADER,
        // row-contiguous is load-bearing: q is an axis-2 slice whose parent head stride is
        // S*128; the copy is what makes Q_shape[2] the physical row stride the kernel indexes by
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_score_kernel_cached = kernel;
    qsa_score_kernel_builds += 1;
    return kernel;
}

pub fn qsaScoreSheetComposed(s: mlx.mlx_stream, q: mlx.mlx_array, k32t: mlx.mlx_array) !mlx.mlx_array {
    if (k32t.ctx == null) return error.QsaScoreBankMissing;
    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    try mlx.check(mlx.mlx_astype(&q32, q, .float32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_matmul(&sc, q32, k32t, s));
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var relu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(relu);
    try mlx.check(mlx.mlx_maximum(&relu, sc, zero, s));
    var scores = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum_axis(&scores, relu, 1, false, s));
    return scores;
}

pub fn qsaScoreK32tFromPooled(s: mlx.mlx_stream, pooled: mlx.mlx_array) !mlx.mlx_array {
    const ps = mlx.getShape(pooled);
    var k4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k4);
    try mlx.check(mlx.mlx_reshape(&k4, pooled, &[_]c_int{ ps[0], 1, ps[1], ps[2] }, 4, s));
    const perm = [_]c_int{ 0, 1, 3, 2 };
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_transpose_axes(&kt, k4, &perm, 4, s));
    var k32 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&k32, kt, .float32, s));
    return k32;
}

fn qsaScoreFusedFillConfig(config: mlx.mlx_fast_metal_kernel_config, rows: c_int, nb: c_int, nsg: c_int, ngrp: c_int, nsh: c_int, layout: QsaScoreLayout) !void {
    const o_shape = [_]c_int{ 1, rows, nb };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nsh * nsg * 32, ngrp, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, nsg * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSGVAL", nsg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "USEH4VAL", if (layout == .h4) 1 else 0));
}

fn qsaScoreFusedDispatch(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ps = mlx.getShape(pooled);
    const rows = qs[2];
    const nb = ps[1];
    const layout = qsaScoreLayoutFor(rows);
    const nsg = qsaScoreNsg(layout);
    const r_tg = qsaScoreRowsPerTg(layout, nsg);
    const ngrp: c_int = @divTrunc(rows + r_tg - 1, r_tg);
    const nslab: u32 = @intCast(@divTrunc(nb + 31, 32));
    const nsh: c_int = @intCast(qsaScoreNsh(nslab, @intCast(ngrp), layout));
    const kernel = try getQsaScoreKernel();
    const key = QsaScoreCfgKey{
        .nsg = nsg,
        .rows = rows,
        .nb = nb,
        .layout = @backingInt(layout),
    };
    const score_cfgs: [1]mlx.mlx_fast_metal_kernel_config = qsa_score_cfgs.get(key) orelse blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try qsaScoreFusedFillConfig(config, rows, nb, nsg, ngrp, nsh, layout);
        qsa_score_cfgs.put(key, .{config});
        break :blk .{config};
    };
    const inputs_arr = [_]mlx.mlx_array{ q, pooled };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, score_cfgs[0], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (qsa_score_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-score] engaged (S={d} nb={d} layout={s} nsg={d} nsh={d})\n",
            .{ rows, nb, if (layout == .h4) "h4" else "base", nsg, nsh },
        );
    }
    return out;
}

fn qsaScoreFusedProbeLayout(s: mlx.mlx_stream, rows: c_int, nb: c_int) bool {
    const q = qsaProbeLcgBf16(s, &[_]c_int{ 1, 4, rows, 128 }, 0x51C04E11) orelse return false;
    defer _ = mlx.mlx_array_free(q);
    const pooled = qsaProbeLcgBf16(s, &[_]c_int{ 1, nb, 128 }, 0xA5A55A5A) orelse return false;
    defer _ = mlx.mlx_array_free(pooled);
    const k32t = qsaScoreK32tFromPooled(s, pooled) catch return false;
    defer _ = mlx.mlx_array_free(k32t);
    const want = qsaScoreSheetComposed(s, q, k32t) catch return false;
    defer _ = mlx.mlx_array_free(want);
    var got = qsaScoreFusedDispatch(s, q, pooled) catch return false;
    defer _ = mlx.mlx_array_free(got);
    if (qsa_score_probe_flip_override) {
        const flipped = qsaProbeCorruptFirst(s, got) orelse return false;
        _ = mlx.mlx_array_free(got);
        got = flipped;
    }
    if (mlx.mlx_array_eval(got) != 0 or mlx.mlx_array_eval(want) != 0) return false;
    if (!qsaProbeBitEqual(s, got, want)) {
        log.info("[qsa-score] disabled: probe mismatch ({s})\n", .{if (rows >= 128) "h4" else "base"});
        return false;
    }
    return true;
}

fn qsaScoreFusedRunProbe() bool {
    if (mlx.noGpuBackend()) {
        log.info("[qsa-score] disabled: no gpu\n", .{});
        return false;
    }
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) {
        log.info("[qsa-score] disabled: not gpu stream\n", .{});
        return false;
    }
    _ = getQsaScoreKernel() catch |e| {
        log.info("[qsa-score] disabled: {s}\n", .{@errorName(e)});
        return false;
    };
    if (!qsaScoreFusedProbeLayout(s, 16, 700)) return false;
    if (!qsaScoreFusedProbeLayout(s, 128, 700)) return false;
    return true;
}

pub fn qsaScoreFused(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array) !?mlx.mlx_array {
    qsaScoreFusedArm();
    if (!qsaScoreFusedEligibleFrom(q, pooled)) return null;
    if (!mlx.streamIsGpu(s)) return null;
    if (!qsaPooledRowContiguous(pooled)) return null;
    return try qsaScoreFusedDispatch(s, q, pooled);
}

pub fn qsaScoreSheet(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array, k32t: mlx.mlx_array, fused: bool) !mlx.mlx_array {
    if (fused) {
        return (try qsaScoreFused(s, q, pooled)) orelse error.QsaScoreFusedDeclined;
    }
    if (k32t.ctx == null) return error.QsaScoreBankMissing;
    return qsaScoreSheetComposed(s, q, k32t);
}

/// `key_max` scores `[1, heads * rows, nb]` f32 (head-major) of index queries `q` `[1, heads, rows, d]` at cache
/// positions `row0..` over index keys `keys` `[1, 1, kv, d]`: per block (`nb` = kv / size, rounded up) the max of
/// q . k over its keys at or before the row, -inf where it has none.
pub fn keyMaxScores(a: std.mem.Allocator, s: mlx.mlx_stream, b: Blocks, q: mlx.mlx_array, keys: mlx.mlx_array, row0: c_int) !mlx.mlx_array {
    var g = Graph.init(a, s);
    defer g.deinit();
    const qs = ops.shape(q);
    const kv = ops.shape(keys)[2];
    const size: c_int = @intCast(b.size);
    const nb = @divTrunc(kv + size - 1, size);
    const dots = try g.matmul(try g.astype(q, .float32), try g.transpose(try g.astype(keys, .float32), &.{ 0, 1, 3, 2 }));
    const col = try g.reshape(try g.arange(0, @floatFromInt(kv), 1, .int32), &.{ 1, 1, 1, kv });
    const pos = try g.reshape(try g.arange(@floatFromInt(row0), @floatFromInt(row0 + qs[2]), 1, .int32), &.{ 1, 1, qs[2], 1 });
    const neg = try g.scalar(-std.math.inf(f32), .float32);
    const seen = try g.where(try g.call(mlx.mlx_less_equal, .{ col, pos }), dots, neg);
    const padded = try g.call(mlx.mlx_pad, .{ seen, &[_]c_int{3}, 1, &[_]c_int{0}, 1, &[_]c_int{nb * size - kv}, 1, neg, "constant" });
    const per_block = try g.max(try g.reshape(padded, &.{ 1, qs[1], qs[2], nb, size }), 4, false);
    return g.escape(try g.reshape(per_block, &.{ 1, qs[1] * qs[2], nb }));
}

/// Bound on the per-chunk [n_idx, rows, nb] f32 indexer score sheet: the
/// prefill selection runs in row chunks so 4096 rows x 64k blocks x 4 heads
/// (4.3 GB at 256k) never materializes at once.
///
/// At long context it decides the prefill's row chunk: at kv 383k (nb ~95.8k) 256 MB leaves room
/// for ~171 rows, so a 4096-token chunk is re-split into 24 indexer passes. The forward and the
/// memory bill both read it
/// through `qsaScoreRowsPerChunk`, so a wider sheet is billed, never just spent.
pub const QSA_SCORE_SHEET_BUDGET_MB_DEFAULT: u64 = 256;

/// Upper bound on the lever: a fat-fingered value must become a named 400, not a Metal OOM.
/// Test seam: the sheet budget in MB.
pub var qsa_score_sheet_mb_override: ?u64 = null;

/// The live sheet budget in bytes.
pub fn qsaScoreSheetBudget() u64 {
    return (qsa_score_sheet_mb_override orelse QSA_SCORE_SHEET_BUDGET_MB_DEFAULT) << 20;
}

pub fn qsaScoreRowsPerChunkFor(per_row: u64, rows: u64) u64 {
    if (per_row == 0) return rows;
    return @max(16, @min(rows, qsaScoreSheetBudget() / per_row));
}

pub fn qsaScoreRowsPerChunkFused(n_idx: u64, nb: u64, rows: u64, fused: bool) u64 {
    const per_row: u64 = if (fused) nb * 4 else n_idx * nb * 4;
    return qsaScoreRowsPerChunkFor(per_row, rows);
}

/// Prefill transient of the block-gathered QSA path for ONE live layer.
/// Composed: matmul out + relu per head + scores + argpartition temps + selection.
/// Fused: the [rows,nb] f32 sheet (×4 when the select kernel is off) + selection.
pub fn qsaPrefillTransientBytes(n_idx: u64, fwd: u64, kv: u64, ratio: u64, idx_hd: u64) u64 {
    if (ratio == 0) return 0;
    const nb = kv / ratio;
    const fused = qsaScoreFusedActiveFor(1, @intCast(n_idx), @intCast(idx_hd));
    const rows = qsaScoreRowsPerChunkFused(n_idx, nb, fwd, fused);
    if (fused) {
        const sheet = rows * nb * 4;
        const select_mult: u64 = if (qsaSelectKernelEnabled()) 1 else 4;
        return sheet * select_mult + fwd * @min(nb, 512) * 4;
    }
    return rows * nb * 4 * (n_idx * 2 + 4) + fwd * @min(nb, 512) * 4;
}

// ── tests ──

pub fn qsaScoreKernelBuilt() bool {
    return qsa_score_kernel_cached != null;
}

pub fn qsaScoreKernelBuilds() usize {
    return qsa_score_kernel_builds;
}

pub fn qsaScoreFusedProbeFailed() bool {
    return qsa_score_probe_state.load(.acquire) == 1;
}

pub fn qsaScoreFusedProbeReset() void {
    qsa_score_probe_state.store(0, .release);
}

pub fn qsaScoreRowsPerChunk(n_idx: u64, nb: u64, rows: u64, idx_hd: u64) u64 {
    return qsaScoreRowsPerChunkFused(n_idx, nb, rows, qsaScoreFusedActiveFor(1, @intCast(n_idx), @intCast(idx_hd)));
}

test "qsa score sheet budget: the row chunk it buys at nb 95k" {
    const t = std.testing;
    qsa_score_fused_override = false;
    defer qsa_score_fused_override = null;

    // Flash-Next at the 374k rung: nb 95801, n_idx 4, chunk 4096. At 256 MB the sheet fits
    // ~175 rows, so one chunk is 24 indexer passes; at 1024 MB it is 6.
    const n_idx: u64 = 4;
    const nb: u64 = 383205 / 4;
    const chunk: u64 = 4096;
    const prev = qsa_score_sheet_mb_override;
    defer qsa_score_sheet_mb_override = prev;

    qsa_score_sheet_mb_override = 256;
    const rows_256 = qsaScoreRowsPerChunk(n_idx, nb, chunk, 128);
    qsa_score_sheet_mb_override = 1024;
    const rows_1024 = qsaScoreRowsPerChunk(n_idx, nb, chunk, 128);

    try t.expectEqual(@as(u64, (256 << 20) / (4 * nb * 4)), rows_256);
    try t.expectEqual(rows_256 * 4, rows_1024);
    try t.expect(rows_256 < chunk and rows_1024 < chunk); // both still row-chunked
    const passes = struct {
        fn f(rows: u64, c: u64) u64 {
            return (c + rows - 1) / rows;
        }
    }.f;
    try t.expectEqual(@as(u64, 24), passes(rows_256, chunk));
    try t.expectEqual(@as(u64, 6), passes(rows_1024, chunk));

    // A wider sheet must be billed: the admission guard reads the same helper.
    qsa_score_sheet_mb_override = 256;
    const bill_256 = qsaPrefillTransientBytes(n_idx, chunk, 383205, 4, 128);
    qsa_score_sheet_mb_override = 1024;
    const bill_1024 = qsaPrefillTransientBytes(n_idx, chunk, 383205, 4, 128);
    try t.expect(bill_1024 > bill_256);
    try t.expectEqual((rows_1024 - rows_256) * nb * 4 * (n_idx * 2 + 4), bill_1024 - bill_256);

    qsa_score_sheet_mb_override = 4096;
    try t.expectEqual(@as(u64, 64), qsaScoreRowsPerChunk(n_idx, 1024, 64, 128));
    try t.expectEqual(@as(u64, 16), qsaScoreRowsPerChunk(n_idx, 1 << 30, 4096, 128));
}

test "qsa score kernel: fused score sheet is bit-identical to the composed chain" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    defer qsa_score_layout_override = null;
    var prng = std.Random.DefaultPrng.init(0x51C0_4E11);
    const rnd = prng.random();

    const nbs = [_]c_int{ 700, 2048, 15675, 65536 };
    const rowss = [_]c_int{ 1, 2, 3, 8, 15, 16, 17, 31, 64, 128, 256 };
    const layouts = [_]QsaScoreLayout{ .base, .h4 };
    for (nbs) |nb| {
        for (rowss) |rows| {
            const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
            defer _ = mlx.mlx_array_free(q);
            const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
            defer _ = mlx.mlx_array_free(pooled);
            try mlx.check(mlx.mlx_array_eval(pooled));
            const k32t = try qsaScoreK32tFromPooled(s, pooled);
            defer _ = mlx.mlx_array_free(k32t);
            const want = try qsaScoreSheetComposed(s, q, k32t);
            defer _ = mlx.mlx_array_free(want);
            for (layouts) |layout| {
                qsa_score_layout_override = layout;
                const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
                defer _ = mlx.mlx_array_free(got);
                try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
            }
        }
    }

    {
        const extras = [_]struct { nb: c_int, rows: c_int }{
            .{ .nb = 262144, .rows = 1 },
            .{ .nb = 262144, .rows = 16 },
            .{ .nb = 262144, .rows = 64 },
            .{ .nb = 2048, .rows = 4096 },
        };
        for (extras) |c| {
            const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, c.rows, 128 }, s);
            defer _ = mlx.mlx_array_free(q);
            const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, c.nb, 128 }, s);
            defer _ = mlx.mlx_array_free(pooled);
            try mlx.check(mlx.mlx_array_eval(pooled));
            const k32t = try qsaScoreK32tFromPooled(s, pooled);
            defer _ = mlx.mlx_array_free(k32t);
            const want = try qsaScoreSheetComposed(s, q, k32t);
            defer _ = mlx.mlx_array_free(want);
            for (layouts) |layout| {
                qsa_score_layout_override = layout;
                const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
                defer _ = mlx.mlx_array_free(got);
                try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
            }
        }
    }

    {
        const parent_rows: c_int = 96;
        const rows: c_int = 40;
        const row0: c_int = 24;
        const nb: c_int = 2048;
        const parent = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, parent_rows, 128 }, s);
        defer _ = mlx.mlx_array_free(parent);
        var q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q);
        try mlx.check(mlx.mlx_slice(&q, parent, &[_]c_int{ 0, 0, row0, 0 }, 4, &[_]c_int{ 1, 4, row0 + rows, 128 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
        defer _ = mlx.mlx_array_free(pooled);
        try mlx.check(mlx.mlx_array_eval(pooled));
        const k32t = try qsaScoreK32tFromPooled(s, pooled);
        defer _ = mlx.mlx_array_free(k32t);
        const want = try qsaScoreSheetComposed(s, q, k32t);
        defer _ = mlx.mlx_array_free(want);
        for (layouts) |layout| {
            qsa_score_layout_override = layout;
            const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
            defer _ = mlx.mlx_array_free(got);
            try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
        }
    }

    {
        const nb: c_int = 2048;
        const cap: c_int = 3072;
        const rows: c_int = 16;
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
        defer _ = mlx.mlx_array_free(q);
        const buf = try attn256RandBf16(rnd, &[_]c_int{ 1, cap, 128 }, s);
        defer _ = mlx.mlx_array_free(buf);
        var pooled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pooled);
        try mlx.check(mlx.mlx_slice(&pooled, buf, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, nb, 128 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_array_eval(pooled));
        const k32t = try qsaScoreK32tFromPooled(s, pooled);
        defer _ = mlx.mlx_array_free(k32t);
        const want = try qsaScoreSheetComposed(s, q, k32t);
        defer _ = mlx.mlx_array_free(want);
        for (layouts) |layout| {
            qsa_score_layout_override = layout;
            const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
            defer _ = mlx.mlx_array_free(got);
            try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
        }
    }

    {
        const nb: c_int = 2048;
        const cap: c_int = 3072;
        const rows: c_int = 16;
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
        defer _ = mlx.mlx_array_free(q);
        const buf = try attn256RandBf16(rnd, &[_]c_int{ 1, cap, 128 }, s);
        defer _ = mlx.mlx_array_free(buf);
        var pooled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pooled);
        try mlx.check(mlx.mlx_slice(&pooled, buf, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, nb, 128 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_array_eval(pooled));
        var k4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k4);
        try mlx.check(mlx.mlx_reshape(&k4, buf, &[_]c_int{ 1, 1, cap, 128 }, 4, s));
        var kt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kt);
        try mlx.check(mlx.mlx_transpose_axes(&kt, k4, &[_]c_int{ 0, 1, 3, 2 }, 4, s));
        var k32full = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k32full);
        try mlx.check(mlx.mlx_astype(&k32full, kt, .float32, s));
        var k32t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k32t);
        try mlx.check(mlx.mlx_slice(&k32t, k32full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, 1, 128, nb }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        const want = try qsaScoreSheetComposed(s, q, k32t);
        defer _ = mlx.mlx_array_free(want);
        for (layouts) |layout| {
            qsa_score_layout_override = layout;
            const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
            defer _ = mlx.mlx_array_free(got);
            try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
        }
    }
}

test "qsa score kernel: block ids through qsaSelectTopBlocks are identical on both arms" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    qsa_score_layout_override = .h4;
    defer qsa_score_layout_override = null;
    var prng = std.Random.DefaultPrng.init(0xB10C_1D5);
    const rnd = prng.random();
    const nb: c_int = 2048;
    const rows: c_int = 16;
    const ratio: c_int = 4;
    const k: c_int = 512;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    const k32t = try qsaScoreK32tFromPooled(s, pooled);
    defer _ = mlx.mlx_array_free(k32t);
    const composed = try qsaScoreSheetComposed(s, q, k32t);
    defer _ = mlx.mlx_array_free(composed);
    const fused = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
    defer _ = mlx.mlx_array_free(fused);

    const row0s = [_]c_int{ 0, 2044, nb * ratio };
    for (row0s) |row0| {
        const all_vis = blocks4.allSee(row0, nb);
        const bounds_host = try ta.alloc(i32, @intCast(rows));
        defer ta.free(bounds_host);
        blocks4.selectBounds(bounds_host, row0, nb, all_vis);
        const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
        defer _ = mlx.mlx_array_free(bounds);
        const ids_c = (try qsaSelectTopBlocks(s, composed, bounds, k)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(ids_c);
        const ids_f = (try qsaSelectTopBlocks(s, fused, bounds, k)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(ids_f);
        try testing.expect(try qsaArraysAllEqual(ids_c, ids_f, s));
    }
}

test "qsa score kernel: never built without NAX" {
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    verify_qmm_nax_vars.vqmm_nax_probe_override = false;
    defer verify_qmm_nax_vars.vqmm_nax_probe_override = null;
    const saved = qsa_score_kernel_cached;
    qsa_score_kernel_cached = null;
    defer qsa_score_kernel_cached = saved;
    try testing.expect(!qsaScoreFusedEligible(1, 4, 128, .bfloat16, .bfloat16));
    try testing.expect(!qsaScoreKernelBuilt());
    const builds = qsaScoreKernelBuilds();
    try testing.expectError(error.NaxUnavailable, getQsaScoreKernel());
    try testing.expectEqual(builds, qsaScoreKernelBuilds());
    try testing.expect(!qsaScoreKernelBuilt());
}

test "qsa score fused: a probe mismatch latches the composed arm" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaScoreTf32Enabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const saved_override = qsa_score_fused_override;
    const saved_flip = qsa_score_probe_flip_override;
    defer {
        qsa_score_fused_override = saved_override;
        qsa_score_probe_flip_override = saved_flip;
        qsaScoreFusedProbeReset();
    }
    qsa_score_fused_override = true;
    qsa_score_probe_flip_override = true;
    qsaScoreFusedProbeReset();
    qsaScoreFusedArm();
    try testing.expect(qsaScoreFusedProbeFailed());
    try testing.expect(!qsaScoreFusedActive());
    var prng = std.Random.DefaultPrng.init(0xF11);
    const q = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 4, 2, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 64, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    try testing.expect((try qsaScoreFused(s, q, pooled)) == null);
    const k32t = try qsaScoreK32tFromPooled(s, pooled);
    defer _ = mlx.mlx_array_free(k32t);
    const composed = try qsaScoreSheetComposed(s, q, k32t);
    defer _ = mlx.mlx_array_free(composed);
    try testing.expect(composed.ctx != null);
}

test "qsa score fused: probe passes on this machine" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaScoreTf32Enabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const saved_override = qsa_score_fused_override;
    const saved_flip = qsa_score_probe_flip_override;
    defer {
        qsa_score_fused_override = saved_override;
        qsa_score_probe_flip_override = saved_flip;
        qsaScoreFusedProbeReset();
    }
    qsa_score_fused_override = true;
    qsa_score_probe_flip_override = false;
    qsaScoreFusedProbeReset();
    qsaScoreFusedArm();
    try testing.expect(!qsaScoreFusedProbeFailed());
    try testing.expect(qsaScoreFusedActive());
}

test "qsa score fused: default on engages, the seam declines" {
    const saved_override = qsa_score_fused_override;
    defer qsa_score_fused_override = saved_override;
    qsa_score_fused_override = null;
    try testing.expect(qsaScoreFusedEnabled());
    qsa_score_fused_override = false;
    try testing.expect(!qsaScoreFusedEnabled());
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDEF0);
    const q = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 4, 2, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 64, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    try testing.expect((try qsaScoreFused(s, q, pooled)) == null);
    qsa_score_fused_override = true;
    if (verifyQmmNaxAvailable() and qsaScoreTf32Enabled()) {
        const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
        defer _ = mlx.mlx_array_free(got);
    }
}

test "qsa score kernel: TF32 off declines" {
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    qsa_score_tf32_override = false;
    defer qsa_score_tf32_override = null;
    try testing.expect(!qsaScoreFusedEligible(1, 4, 128, .bfloat16, .bfloat16));
}

test "qsa score fused: sizing policy" {
    try testing.expect(qsaScoreTf32ValueOn("1"));
    try testing.expect(qsaScoreTf32ValueOn(" 1"));
    try testing.expect(!qsaScoreTf32ValueOn("0"));
    try testing.expect(!qsaScoreTf32ValueOn("00"));
    try testing.expect(!qsaScoreTf32ValueOn("false"));
    try testing.expect(!qsaScoreTf32ValueOn(" 0"));
    try testing.expect(!qsaScoreTf32ValueOn("0x0"));
    try testing.expectEqual(QsaScoreLayout.base, qsaScoreLayoutFor(1));
    try testing.expectEqual(QsaScoreLayout.base, qsaScoreLayoutFor(127));
    try testing.expectEqual(QsaScoreLayout.h4, qsaScoreLayoutFor(128));
    try testing.expectEqual(@as(c_int, 4), qsaScoreNsg(.base));
    try testing.expectEqual(@as(c_int, 8), qsaScoreNsg(.h4));
    try testing.expectEqual(@as(c_int, 16), qsaScoreRowsPerTg(.base, 4));
    try testing.expectEqual(@as(c_int, 128), qsaScoreRowsPerTg(.h4, 8));
    try testing.expectEqual(@as(u32, 64), qsaScoreNsh(64, 1, .base));
    try testing.expectEqual(@as(u32, 32), qsaScoreNsh(1024, 8, .h4));
    const saved_fused = qsa_score_fused_override;
    const saved_nax = verify_qmm_nax_vars.vqmm_nax_probe_override;
    const saved_sheet = qsa_score_sheet_mb_override;
    defer {
        qsa_score_fused_override = saved_fused;
        verify_qmm_nax_vars.vqmm_nax_probe_override = saved_nax;
        qsa_score_sheet_mb_override = saved_sheet;
    }
    qsa_score_sheet_mb_override = QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    const composed = qsaScoreRowsPerChunkFor(4 * 262144 * 4, 4096);
    const fused = qsaScoreRowsPerChunkFor(262144 * 4, 4096);
    try testing.expectEqual(composed * 4, fused);
    try testing.expect(fused >= 128);
}

test "qsa score fused: the argpartition decline arm consumes the fused sheet" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    qsa_score_layout_override = .h4;
    defer qsa_score_layout_override = null;
    var prng = std.Random.DefaultPrng.init(0xA2B0);
    const rnd = prng.random();
    const nb: c_int = 2048;
    const rows: c_int = 16;
    const k: c_int = 512;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    const fused = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const bounds_host = try ta.alloc(i32, @intCast(rows));
    defer ta.free(bounds_host);
    blocks4.selectBounds(bounds_host, nb * 4, nb, true);
    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);
    const ids_k = (try qsaSelectTopBlocks(s, fused, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(ids_k);
    const saved_sel = select.qsa_select_kernel_override;
    defer select.qsa_select_kernel_override = saved_sel;
    select.qsa_select_kernel_override = false;
    try testing.expect((try qsaSelectTopBlocks(s, fused, bounds, k)) == null);
    const bias = try tieBias(s, nb);
    defer _ = mlx.mlx_array_free(bias);
    const vis = mlx.mlx_array{ .ctx = null };
    const ids_a = try qsaSelectComposedOps(s, fused, vis, bias, rows, nb, k, k, true);
    defer _ = mlx.mlx_array_free(ids_a);
    try testing.expect(try qsaArraysAllEqual(ids_k, ids_a, s));
}
