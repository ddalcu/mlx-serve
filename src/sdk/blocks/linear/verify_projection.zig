//! sdk/blocks/linear/verify_projection.
//! Owns SingleVerifyFeature, single_verify_test_features, mtp_verify_shared_rows_calls, mtp_verify_shared_gate_rows_calls, mtp_verify_gdn_rows_calls, mtp_verify_gdn_output_calls, and 36 more.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const MAX_BATCH_ROWS = @import("../../bases/run.zig").MAX_BATCH_ROWS;
const lastDim = @import("../../../mlx/array.zig").lastDim;
const concatAxis0 = @import("../../../mlx/array.zig").concatAxis0;
const gpuArchitecture = @import("../../../mlx/device.zig").gpuArchitecture;
const QsaCfgCache = @import("../../../mlx/kernel_cache.zig").QsaCfgCache;
const affineParamsFromGeometry = @import("../../quant/params.zig").affineParamsFromGeometry;
const qmvBatchLimit = @import("qmatmul.zig").qmvBatchLimit;
const cachedScalarInt = @import("verify_qmm.zig").cachedScalarInt;
const mixedNaxShapeEnabled = @import("verify_qmm.zig").mixedNaxShapeEnabled;
const mixedPlainEnabled = @import("verify_qmm.zig").mixedPlainEnabled;
const verifyQmmEnabled = @import("verify_qmm.zig").verifyQmmEnabled;
const naxMinMForBits = @import("verify_qmm_nax.zig").naxMinMForBits;
const verifyQmmNaxAvailable = @import("verify_qmm_nax.zig").verifyQmmNaxAvailable;

pub const SingleVerifyFeature = enum { routing, down_reduce };
pub var single_verify_test_features: [2]bool = @splat(true);

pub var mtp_verify_shared_rows_calls: u64 = 0;
pub var mtp_coarse_pair_dispatches: u64 = 0;
pub var mtp_verify_kernel_calls: [3]u64 = @splat(0);
pub var mtp_verify_shared_gate_rows_calls: u64 = 0;

pub var mtp_verify_gdn_rows_calls: u64 = 0;
pub var mtp_verify_gdn_output_calls: u64 = 0;

pub var mtp_verify_indexed_input_calls: u64 = 0;

pub var mtp_verify_expert_reduce_calls: u64 = 0;

pub var mtp_verify_expert_pairs_calls: u64 = 0;

pub var mtp_verify_attn_rows_calls: u64 = 0;
pub var mtp_verify_attn_output_calls: u64 = 0;

pub var mtp_verify_lmhead_rows_calls: u64 = 0;

pub const SharedProjKind = enum { qmatmul, mtp_rows, verify_rows };

pub fn sharedGateRowsEligible(widths: []const c_int) bool {
    if (widths.len < 2 or widths.len > 8) return false;
    var total: c_int = 0;
    for (widths) |width| {
        if (width < 2 or width > 7) return false;
        total += width;
    }
    return total <= 24;
}

pub const VerifyKernel = enum { dense_tiles, hc_rows, route_pack };

pub fn mtpHeadRowsFit(widths: []const c_int, hc: u32) bool {
    if (widths.len == 0 or widths.len > 8 or hc == 0) return false;
    if (@as(u64, hc) * widths.len > 16) return false;
    for (widths) |width| if (width != 1) return false;
    return true;
}

var verify_arch_buf: [128]u8 = undefined;
var verify_arch: ?[]const u8 = null;
var verify_arch_read = false;
/// Streams in the batched verify running now (`forwardRowAxisVerify`); 0 outside one.
pub var verify_group_rows: usize = 0;

// MLX affine qmv-fast reduction, with one 3-bit weight read serving two rows.
pub const MTP_COARSE_PAIRS_SOURCE = kernelSource("kernels/linear/mtp_coarse_pairs.metal");
var mtp_coarse_pairs_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mtp_coarse_pairs_cfgs = QsaCfgCache(c_int, 1){};

pub fn mtpCoarsePairs(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, gs: u32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or bits != 3 or gs != 64) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] < 2 or xs[0] > 8 or xs[1] != 1 or xs[2] != 2560) return null;
    if (!std.mem.eql(c_int, ws, &.{ 248320, 240 }) or !std.mem.eql(c_int, ss, &.{ 248320, 40 }) or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const kernel = blk: {
        if (mtp_coarse_pairs_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi", "K_size", "N_size" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("mtp_coarse_pairs", iv, ov, MTP_COARSE_PAIRS_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        mtp_coarse_pairs_kernel = value;
        break :blk value;
    };
    const cfg = blk: {
        if (mtp_coarse_pairs_cfgs.get(xs[0])) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ xs[0], 1, 248320 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, 248320 / 4, @divTrunc(xs[0] + 1, 2)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        mtp_coarse_pairs_cfgs.put(xs[0], .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ x, w, sc, bi, cachedScalarInt(2560), cachedScalarInt(248320) };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    mtp_coarse_pair_dispatches +%= 1;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[mtp] paired coarse head engaged (rows={d})\n", .{xs[0]});
    }
    return result;
}

/// The fused verify kernels need an M5 GPU. A dual-die one (`...d`, the Ultra) gains from them
/// only in a multi-stream group: they lift a batched verify and slow a solo one.
pub fn verifySharedFor(arch: []const u8, group_rows: usize) bool {
    if (!std.mem.startsWith(u8, arch, "applegpu_g17")) return false;
    return !dualDieArch(arch) or group_rows > 1;
}

/// A dual-die GPU (the Ultra) reports an architecture ending in `d`, e.g. `applegpu_g17d`.
pub fn dualDieArch(arch: []const u8) bool {
    return std.mem.endsWith(u8, arch, "d");
}

pub fn verifyArch() ?[]const u8 {
    if (!verify_arch_read) {
        verify_arch_read = true;
        verify_arch = gpuArchitecture(&verify_arch_buf);
    }
    return verify_arch;
}

/// Whether this GPU can run the fused verify kernels at all (load-time pack checks).
pub fn verifySharedCapable() bool {
    if (!verifyQmmNaxAvailable()) return false;
    return std.mem.startsWith(u8, verifyArch() orelse return false, "applegpu_g17");
}

/// Whether the verify running now takes the fused kernels.
pub fn verifySharedHardware() bool {
    if (!verifyQmmNaxAvailable()) return false;
    return verifySharedFor(verifyArch() orelse return false, verify_group_rows);
}

// Same subchunk and shuffle reduction as Apple's MLX qmv_wide_impl (MIT).
pub const VERIFY_WIDE_SOURCE = kernelSource("kernels/linear/verify_wide.metal");
var verify_wide_kernel: ?mlx.mlx_fast_metal_kernel = null;
const VerifyWideKey = struct { m: c_int, n: c_int, k: c_int, nv: c_int };
var verify_wide_configs = QsaCfgCache(VerifyWideKey, 1){};

pub fn verifyWideProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, nv: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or (nv != 4 and nv != 6 and nv != 8 and nv != 12)) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] != 1 or xs[1] < 6 or xs[1] > 12 or ws.len != 2 or ss.len != 2) return null;
    const m = xs[1];
    const k = xs[2];
    const n = ws[0];
    if ((k != 640 and k != 2560 and k != 6144) or n < 8 or @mod(n, 8) != 0 or ws[1] != @divExact(k, 4) or ss[0] != n or ss[1] != @divExact(k, 64) or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const kernel = blk: {
        if (verify_wide_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_dense_wide", iv, ov, VERIFY_WIDE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_wide_kernel = value;
        break :blk value;
    };
    const key = VerifyWideKey{ .m = m, .n = n, .k = k, .nv = nv };
    const config = blk: {
        if (verify_wide_configs.get(key)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ 1, m, n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, @intCast(@divTrunc(m + nv - 1, nv) * 32), @intCast(@divExact(n, 8) * 2), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        inline for (.{ "M", "N", "K", "NV" }, .{ m, n, k, nv }) |name, number| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, name, number));
        verify_wide_configs.put(key, .{value});
        break :blk value;
    };
    const iv = mlx.mlx_vector_array_new_data(&.{ x, w, sc, bi }, 4);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    return result;
}

const VerifyProjectionKind = enum { shared, gdn, lm_head, attention };

pub fn verifySharedProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .shared);
}

pub fn verifyGdnProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .gdn);
}

pub fn verifyAttnProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .attention);
}

pub fn verifyLmHeadProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .lm_head);
}

fn verifyJoinedProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int, comptime kind: VerifyProjectionKind) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or widths.len < 2 or widths.len > 8) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or
        mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    if (xs.len != 3 or xs[0] != 1 or ws.len != 2 or ss.len != 2 or bs.len != 2) return null;
    const k = xs[2];
    const n = ws[0];
    const wide = kind != .shared;
    const shape_ok = switch (kind) {
        .shared => (k == 2560 and n == 640) or (k == 640 and n == 2560),
        .gdn => (k == 2560 and (n == 10240 or n == 6144 or n == 48)) or (k == 6144 and n == 2560),
        .lm_head => k == 2560 and n == 248320,
        .attention => (k == 2560 and (n == 12288 or n == 512)) or (k == 6144 and n == 2560),
    };
    if (!shape_ok) return null;
    const qp = affineParamsFromGeometry(w, sc, @intCast(k)) orelse return null;
    if (ss[0] != n or !std.mem.eql(c_int, ss, bs)) return null;
    // Rows must match what each slot computes alone: a slot whose width a verify lane
    // serves does not take the reference qmv_wide K reduction.
    if (qp.bits == 4 or mixedPlainEnabled(qp.bits, qp.group_size, n)) return null;
    var total: c_int = 0;
    for (widths) |width| {
        if (width < 2 or width > (if (wide) @as(c_int, 6) else 7)) return null;
        if (wide and verifyQmmEnabled() and
            mixedNaxShapeEnabled(qp.bits, qp.group_size, n, width) and width >= naxMinMForBits(qp.bits) and @mod(k, 256) == 0) return null;
        total += width;
    }
    if (total != xs[1] or total > (if (wide) @as(c_int, 12) else 24)) return null;
    if (total == 12 and n >= 2560 and qp.bits == 8 and qp.group_size == 64) {
        if (try verifyWideProjection(s, x, w, sc, bi, 6)) |value| {
            mtp_verify_kernel_calls[0] +%= 1;
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[batched] wider dense verify tiles engaged\n", .{});
            }
            return value;
        }
    }
    const arch = verifyArch() orelse return null;
    if (total >= qmvBatchLimit(dualDieArch(arch), k, n)) return null;
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_quantized_matmul(&result, x, w, sc, bi, true, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), "affine", s));
    return result;
}

