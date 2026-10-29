//! sdk/blocks/moe/down_reduce.
//! Owns gatherQmvDownReduceSource, GQMV_DOWNRED_SOURCES, GQMV_DOWNRED_NAMES, gqmv_downred_kernels, getGatherQmvDownReduceKernel, downredLanesFor, and 20 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const gather_qmv = @import("gather_qmv.zig");
const deviceGeneration = @import("../../../mlx/device.zig").deviceGeneration;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const cachedScalarInt = @import("../linear/verify_qmm.zig").cachedScalarInt;
const GQMV_AFFINE_HEADER = @import("gather_qmv.zig").GQMV_AFFINE_HEADER;
const GQMV_BODY_AFFINE = @import("gather_qmv.zig").GQMV_BODY_AFFINE;
const GQMV_BODY_NVFP4 = @import("gather_qmv.zig").GQMV_BODY_NVFP4;
const GQMV_NVFP4_HEADER = @import("gather_qmv.zig").GQMV_NVFP4_HEADER;
const affineQmvPackUnits = @import("gather_qmv.zig").affineQmvPackUnits;
const affineQmvUnitOk = @import("gather_qmv.zig").affineQmvUnitOk;
const gatherQmv = @import("gather_qmv.zig").gatherQmv;
const moeRowsFusedEnabled = @import("gather_qmv_rows.zig").moeRowsFusedEnabled;

// ── Fused MoE down + router weighting + K-reduction (mlxfast down-path mega-fusion class) ──
//
// The decode expert tail ran down-gather -> score multiply -> sum over K as
// three dispatches, materializing a [K, hidden] intermediate between them.
// This kernel keeps the down projection's per-row arithmetic TEXTUALLY
// identical to `gatherQmv` (same GQMV_BODY_*, same lane-strided pack loop,
// same simd_sum tree, same T(acc) rounding — so each (slot, row) value is
// bit-identical to the unfused kernel's), parks the K per-row values in
// threadgroup memory, and finishes with the composed pair's exact semantics:
// per-slot product rounded to T (the elementwise `mlx_multiply`), then an
// ascending T accumulate over slots 0..K-1 (MLX's bf16 sum reduction
// accumulates in the INPUT dtype — the moeRouterTopK lesson). One dispatch
// replaces three, and the [K, hidden] round-trip never happens.
//
// Each simdgroup owns one top-K slot and computes ROWS consecutive output
// rows for it (mlxfast tuned 1-vs-4 rows per simd; 4 won). Bit-identity pinned by the
// "down+reduce" parity tests.
fn gatherQmvDownReduceSource(comptime nvfp4: bool) [:0]const u8 {
    // LPR lanes share one output row and ROWS = 32 / LPR rows ride one simdgroup
    // in parallel: K is short (the MoE intermediate), so one 32-lane reduction
    // per row cost more than the dot product it reduced, and K/8 packs over 32
    // lanes left a third of them idle. Every row's packs split exactly over LPR
    // lanes, and one log2(LPR)-step shuffle reduces all ROWS rows at once.
    return std.fmt.comptimePrint(kernelSource("kernels/moe/down_reduce.fmt.metal"), .{
        if (nvfp4) GQMV_BODY_NVFP4 else GQMV_BODY_AFFINE,
        if (nvfp4) " * 4194304.0f" else "",
    });
}

pub const GQMV_DOWNRED_SOURCES = [2][:0]const u8{ gatherQmvDownReduceSource(false), gatherQmvDownReduceSource(true) };
const GQMV_DOWNRED_NAMES = [2][*:0]const u8{ "mlxserve_moe_gather_qmv_downred", "mlxserve_moe_gather_qmv_downred_nvfp4" };
var gqmv_downred_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getGatherQmvDownReduceKernel(nvfp4: bool) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @intFromBool(nvfp4);
    if (gqmv_downred_kernels[mi]) |k| return k;
    const affine_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size" };
    const nvfp4_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "inds", "scores", "K_size", "N_size" };
    const input_names: []const [*:0]const u8 = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_DOWNRED_NAMES[mi],
        in_vec,
        out_vec,
        GQMV_DOWNRED_SOURCES[mi],
        if (nvfp4) GQMV_NVFP4_HEADER else GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_downred_kernels[mi] = kernel;
    return kernel;
}

/// Lanes per output row in the down+reduce kernels; 32/LPR rows per simdgroup. The solo and the rows
/// kernel share it so a row's sum keeps one accumulation order on both paths. 4 spills its hoisted packs.
/// A generation nobody measured keeps 8; 16 is measured on G17 (M5).
fn downredLanesFor(gen: u32, phone: bool) c_int {
    return if (gen == 17 and !phone) 16 else 8;
}

pub fn downredLpr() c_int {
    const g = deviceGeneration();
    return downredLanesFor(g.gen, g.phone);
}

const DownRedCfgKey = struct { topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype, lpr: c_int };
var downred_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var downred_cfg_key: DownRedCfgKey = std.mem.zeroes(DownRedCfgKey);
pub var downred_engaged: bool = false;


fn downReduceFusedEnabled() bool {
    return downred_override orelse true;
}

/// Fused down-projection + router weighting + K-reduction at decode width.
/// `x`: [TOPK, K] per-expert activations; `scores`: [TOPK] router weights in
/// the SAME dtype as x (the composed multiply is elementwise in that dtype —
/// a differing dtype would promote, so it declines). Returns the weighted
/// expert sum [N], or null when outside the kernel's envelope.
pub fn gatherQmvDownReduce(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    scores: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!downReduceFusedEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    const nvfp4 = switch (mode) {
        .affine => false,
        .nvfp4 => true,
        else => return null,
    };
    if (nvfp4) {
        if (bits != 4 or group_size != 16) return null;
        if (mlx.mlx_array_dtype(sc) != .uint8) return null;
        if (bi.ctx != null) return null;
    } else {
        if (!affineQmvUnitOk(bits, group_size)) return null;
        if (bi.ctx == null) return null;
    }
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(scores) != xd) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    // Eight lanes per row: from a 2048-wide expert the gathered down projection plus the
    // composed sum beats the lanes (GLM-5-Next and MiMo widths).
    if (K >= 2048) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    if (topk <= 0 or topk > 32) return null; // one simdgroup per slot, <= 1024 threads
    const ssh = mlx.getShape(scores);
    if (ssh.len != 1 or ssh[0] != topk) return null;
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != @as(i64, topk) * K) return null;
    const rows: c_int = @divExact(32, downredLpr());
    if (@rem(N, rows) != 0) return null;

    const key = DownRedCfgKey{ .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd, .lpr = downredLpr() };
    if (downred_cfg == null or !std.meta.eql(downred_cfg_key, key)) {
        if (downred_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{N};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 1, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(N, rows) * topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LPR", downredLpr()));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KP", affineQmvPackUnits(K, bits)));
        downred_cfg = cfg;
        downred_cfg_key = key;
    }

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const affine_inputs = [_]mlx.mlx_array{ x, w, sc, bi, inds, scores, K_arr, N_arr };
    const nvfp4_inputs = [_]mlx.mlx_array{ x, w, sc, inds, scores, K_arr, N_arr };
    const inputs_arr: []const mlx.mlx_array = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvDownReduceKernel(nvfp4);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, downred_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

fn gatherQmvDownReduceRowsSource(comptime nvfp4: bool) [:0]const u8 {
    // The solo kernel's layout with a batch row on grid.y; per-row output must
    // stay bit-identical to the solo call (the rows-vs-solo test pins it).
    return std.fmt.comptimePrint(kernelSource("kernels/moe/down_reduce_rows.fmt.metal"), .{
        if (nvfp4) GQMV_BODY_NVFP4 else GQMV_BODY_AFFINE,
        if (nvfp4) " * 4194304.0f" else "",
    });
}

pub const GQMV_DOWNRED_ROWS_SOURCES = [2][:0]const u8{ gatherQmvDownReduceRowsSource(false), gatherQmvDownReduceRowsSource(true) };
const GQMV_DOWNRED_ROWS_NAMES = [2][*:0]const u8{ "mlxserve_moe_gather_qmv_downred_rows", "mlxserve_moe_gather_qmv_downred_rows_nvfp4" };
var gqmv_downred_rows_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getGatherQmvDownReduceRowsKernel(nvfp4: bool) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @intFromBool(nvfp4);
    if (gqmv_downred_rows_kernels[mi]) |k| return k;
    const affine_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size" };
    const nvfp4_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "inds", "scores", "K_size", "N_size" };
    const input_names: []const [*:0]const u8 = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_DOWNRED_ROWS_NAMES[mi],
        in_vec,
        out_vec,
        GQMV_DOWNRED_ROWS_SOURCES[mi],
        if (nvfp4) GQMV_NVFP4_HEADER else GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_downred_rows_kernels[mi] = kernel;
    return kernel;
}

const DownRedRowsCfgKey = struct { nrows: c_int, topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype, lpr: c_int };
var downred_rows_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var downred_rows_cfg_key: DownRedRowsCfgKey = std.mem.zeroes(DownRedRowsCfgKey);
var downred_rows_engaged: bool = false;

/// Asked BEFORE the gate/up dispatch: a decline is static per geometry and
/// would otherwise cost a wasted gate/up launch on every MoE layer of every tick.
pub fn gatherQmvDownReduceRowsEligible(hidden_act: anytype, gate_qp: anytype, up_qp: anytype, down_qp: anytype, N: c_int) bool {
    if (!downReduceFusedEnabled()) return false;
    if (!moeRowsFusedEnabled()) return false;
    if (hidden_act != .silu) return false;
    if (gate_qp.bits != up_qp.bits or gate_qp.group_size != up_qp.group_size or gate_qp.mode != up_qp.mode) return false;
    switch (down_qp.mode) {
        .affine => {
            if (down_qp.bits != 2 and down_qp.bits != 4 and down_qp.bits != 8) return false;
            if (down_qp.group_size % (32 / down_qp.bits) != 0) return false;
        },
        .nvfp4 => {
            if (down_qp.bits != 4 or down_qp.group_size != 16) return false;
        },
        else => return false,
    }
    if (@rem(N, 4) != 0) return false;
    return true;
}

/// Row-axis twin of `gatherQmvDownReduce`: the K-expert threadgroup reduction
/// runs per row (grid.y = B) in the single-row order, never redistributed
/// across rows, so each row is bit-identical to its solo call.
pub fn gatherQmvDownReduceRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    scores: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!downReduceFusedEnabled()) return null;
    if (!moeRowsFusedEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    const nvfp4 = switch (mode) {
        .affine => false,
        .nvfp4 => true,
        else => return null,
    };
    if (nvfp4) {
        if (bits != 4 or group_size != 16) return null;
        if (mlx.mlx_array_dtype(sc) != .uint8) return null;
        if (bi.ctx != null) return null;
    } else {
        if (!affineQmvUnitOk(bits, group_size)) return null;
        if (bi.ctx == null) return null;
    }
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(scores) != xd) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 2) return null;
    const nrows: c_int = ish[0];
    const topk: c_int = ish[1];
    if (nrows < 1 or nrows > 8 or topk <= 0 or topk > 32) return null;
    const ssh = mlx.getShape(scores);
    if (ssh.len != 2 or ssh[0] != nrows or ssh[1] != topk) return null;
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != @as(i64, nrows) * topk * K) return null;
    const rows: c_int = @divExact(32, downredLpr());
    if (@rem(N, rows) != 0) return null;

    const key = DownRedRowsCfgKey{ .nrows = nrows, .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd, .lpr = downredLpr() };
    if (downred_rows_cfg == null or !std.meta.eql(downred_rows_cfg_key, key)) {
        if (downred_rows_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ nrows, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(N, rows) * topk * 32, nrows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LPR", downredLpr()));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KP", affineQmvPackUnits(K, bits)));
        downred_rows_cfg = cfg;
        downred_rows_cfg_key = key;
    }

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const affine_inputs = [_]mlx.mlx_array{ x, w, sc, bi, inds, scores, K_arr, N_arr };
    const nvfp4_inputs = [_]mlx.mlx_array{ x, w, sc, inds, scores, K_arr, N_arr };
    const inputs_arr: []const mlx.mlx_array = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvDownReduceRowsKernel(nvfp4);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, downred_rows_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!downred_rows_engaged) {
        downred_rows_engaged = true;
        log.info("[moe] fused rows down+reduce kernel engaged: B={d} E_topk={d} hidden={d} bits={d} gs={d}\n", .{ nrows, topk, N, bits, group_size });
    }
    return y;
}

pub var downred_override: ?bool = null;

// ── tests ──

test "down+reduce lanes per row: 16 on G17, 8 on every other generation and on phones" {
    try std.testing.expectEqual(@as(c_int, 16), downredLanesFor(17, false));
    try std.testing.expectEqual(@as(c_int, 8), downredLanesFor(17, true));
    for ([_]u32{ 0, 13, 14, 15, 16, 18, 19 }) |g| try std.testing.expectEqual(@as(c_int, 8), downredLanesFor(g, false));
}

test "moe down+reduce fused: no worse than gatherQmv + multiply + sum vs fp32 truth (nvfp4 + affine)" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xD09E);
    const rnd = prng.random();
    const t = std.testing;
    downred_override = true;
    defer downred_override = null;

    // Two geometries: a small one, and Laguna XS's real decode shape. The fused
    // kernel reduces each row over 8 lanes, the composed chain over 32, so the
    // bar is error against the f32 truth, never bit identity.
    const geoms = [_][4]c_int{
        .{ 16, 8, 64, 32 }, // E, TOPK, KIN, N
        .{ 64, 8, 512, 2048 },
    };
    for (geoms) |geom| {
        const E: usize = @intCast(geom[0]);
        const TOPK: c_int = geom[1];
        const KIN: c_int = geom[2];
        const N: c_int = geom[3];
        try moeDownReduceParityCase(s, rnd, t, E, TOPK, KIN, N, &parity_quants);
    }
}

// Bar: one process serving two expert widths at the same top-k and N (two models, or a trunk and its MTP head) gets
// each width's own kernel, never the first width's.
test "moe down+reduce fused: a second expert width at the same top-k and N runs its own width" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xD0A1);
    downred_override = true;
    defer downred_override = null;
    const affine4 = [_]ParityQuant{.{ .nvfp4 = false, .bits = 4, .gs = 32 }};
    try moeDownReduceParityCase(s, prng.random(), std.testing, 16, 8, 64, 32, &affine4);
    try moeDownReduceParityCase(s, prng.random(), std.testing, 16, 8, 128, 32, &affine4);
}

const ParityQuant = struct { nvfp4: bool, bits: u32, gs: u32 };
const parity_quants = [_]ParityQuant{
    .{ .nvfp4 = true, .bits = 4, .gs = 16 },
    .{ .nvfp4 = false, .bits = 4, .gs = 32 },
    .{ .nvfp4 = false, .bits = 3, .gs = 64 },
};

fn moeDownReduceParityCase(s: mlx.mlx_stream, rnd: std.Random, t: anytype, E: usize, TOPK: c_int, KIN: c_int, N: c_int, quants: []const ParityQuant) !void {
    for (quants) |cfg| {
        const nvfp4 = cfg.nvfp4;
        const bits = cfg.bits;
        const gs = cfg.gs;
        const words_per_row: usize = @intCast(@divExact(@as(u32, @intCast(KIN)) * bits, 32));
        const groups_per_row: usize = @intCast(@divExact(@as(u32, @intCast(KIN)), gs));

        // Packed weight bank: any bit pattern is a valid quantized row.
        const wn = E * @as(usize, @intCast(N)) * words_per_row;
        const wdata = try t.allocator.alloc(u32, wn);
        defer t.allocator.free(wdata);
        for (wdata) |*v| v.* = rnd.int(u32);
        const w_shape = [_]c_int{ @intCast(E), N, @intCast(words_per_row) };
        const w = mlx.mlx_array_new_data(wdata.ptr, &w_shape, 3, .uint32);
        defer _ = mlx.mlx_array_free(w);

        const sn = E * @as(usize, @intCast(N)) * groups_per_row;
        const s_shape = [_]c_int{ @intCast(E), N, @intCast(groups_per_row) };
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array{ .ctx = null };
        defer if (bi.ctx != null) {
            _ = mlx.mlx_array_free(bi);
        };
        if (nvfp4) {
            // uint8 e4m3 scales; keep clear of the NaN code (mag 0x7F).
            const sdata = try t.allocator.alloc(u8, sn);
            defer t.allocator.free(sdata);
            for (sdata) |*v| {
                var b = rnd.int(u8);
                if ((b & 0x7F) == 0x7F) b = 0x30;
                v.* = b;
            }
            sc = mlx.mlx_array_new_data(sdata.ptr, &s_shape, 3, .uint8);
        } else {
            const sf = try t.allocator.alloc(f32, sn);
            defer t.allocator.free(sf);
            for (sf) |*v| v.* = 0.01 + 0.05 * rnd.float(f32);
            const bf = try t.allocator.alloc(f32, sn);
            defer t.allocator.free(bf);
            for (bf) |*v| v.* = rnd.float(f32) - 0.5;
            const sc32 = mlx.mlx_array_new_data(sf.ptr, &s_shape, 3, .float32);
            defer _ = mlx.mlx_array_free(sc32);
            const bi32 = mlx.mlx_array_new_data(bf.ptr, &s_shape, 3, .float32);
            defer _ = mlx.mlx_array_free(bi32);
            try mlx.check(mlx.mlx_astype(&sc, sc32, .bfloat16, s));
            bi = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bi, bi32, .bfloat16, s));
        }

        const x_shape = [_]c_int{ TOPK, KIN };
        const x = try attn256RandBf16(rnd, &x_shape, s);
        defer _ = mlx.mlx_array_free(x);
        const sc_shape = [_]c_int{TOPK};
        const scores = try attn256RandBf16(rnd, &sc_shape, s);
        defer _ = mlx.mlx_array_free(scores);
        var idata: [8]u32 = undefined;
        for (&idata) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        const inds = mlx.mlx_array_new_data(&idata, &sc_shape, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);

        const mode: QuantMode = if (nvfp4) .nvfp4 else .affine;

        // Composed chain: gatherQmv down → [TOPK,N] → ·scores → sum over K.
        const down_2d = (try gatherQmv(s, x, w, sc, bi, inds, bits, gs, mode, true)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(down_2d);
        var down_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_4d);
        const d4 = [_]c_int{ 1, 1, TOPK, N };
        try mlx.check(mlx.mlx_reshape(&down_4d, down_2d, &d4, 4, s));
        var scores_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_exp);
        const se = [_]c_int{ 1, 1, TOPK, 1 };
        try mlx.check(mlx.mlx_reshape(&scores_exp, scores, &se, 4, s));
        var weighted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weighted);
        try mlx.check(mlx.mlx_multiply(&weighted, down_4d, scores_exp, s));
        var composed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(composed);
        try mlx.check(mlx.mlx_sum_axis(&composed, weighted, -2, false, s));

        // Fused: one dispatch.
        const fused_flat = (try gatherQmvDownReduce(s, x, w, sc, bi, inds, scores, bits, gs, mode)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(fused_flat);
        var fused = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(fused);
        const f3 = [_]c_int{ 1, 1, N };
        try mlx.check(mlx.mlx_reshape(&fused, fused_flat, &f3, 3, s));

        // f32 truth: dequantized bank, the TOPK experts, f32 matmul, f32 weighted sum.
        var deq32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(deq32);
        const null_bi = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_dequantize(&deq32, w, sc, if (nvfp4) null_bi else bi, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), if (nvfp4) "nvfp4" else "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        var sel = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sel);
        try mlx.check(mlx.mlx_take_axis(&sel, deq32, inds, 0, s));
        var x32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x32);
        try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        const x3s = [_]c_int{ TOPK, KIN, 1 };
        try mlx.check(mlx.mlx_reshape(&x3, x32, &x3s, 3, s));
        var prod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prod);
        try mlx.check(mlx.mlx_matmul(&prod, sel, x3, s));
        var sc32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc32);
        try mlx.check(mlx.mlx_astype(&sc32, scores, .float32, s));
        var sc3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc3);
        const sc3s = [_]c_int{ TOPK, 1, 1 };
        try mlx.check(mlx.mlx_reshape(&sc3, sc32, &sc3s, 3, s));
        var wprod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wprod);
        try mlx.check(mlx.mlx_multiply(&wprod, prod, sc3, s));
        var truth_n = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth_n);
        try mlx.check(mlx.mlx_sum_axis(&truth_n, wprod, 0, false, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_reshape(&truth, truth_n, &f3, 3, s));

        const cnt: usize = @intCast(N);
        const hf = try t.allocator.alloc(f32, cnt);
        defer t.allocator.free(hf);
        const hc = try t.allocator.alloc(f32, cnt);
        defer t.allocator.free(hc);
        const ht = try t.allocator.alloc(f32, cnt);
        defer t.allocator.free(ht);
        try testReadF32(fused, hf, s);
        try testReadF32(composed, hc, s);
        try testReadF32(truth, ht, s);
        var se_f: f64 = 0;
        var se_c: f64 = 0;
        for (hf, hc, ht) |a, b, c| {
            try t.expect(std.math.isFinite(a));
            se_f += (a - c) * (a - c);
            se_c += (b - c) * (b - c);
        }
        const rms_f = @sqrt(se_f / @as(f64, @floatFromInt(cnt)));
        const rms_c = @sqrt(se_c / @as(f64, @floatFromInt(cnt)));
        t.expect(rms_f <= rms_c * 1.15 + 1e-6) catch |e| {
            std.debug.print("\n[down-reduce] nvfp4={} bits={d} gs={d} K={d} N={d} fused_rms={e:.4} composed_rms={e:.4}\n", .{ nvfp4, bits, gs, KIN, N, rms_f, rms_c });
            return e;
        };
    }
}

test "moe down+reduce fused: declines outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xD09F);
    const rnd = prng.random();
    downred_override = true;
    defer downred_override = null;

    const TOPK: c_int = 8;
    const KIN: c_int = 64;
    const x_shape = [_]c_int{ TOPK, KIN };
    const x = try attn256RandBf16(rnd, &x_shape, s);
    defer _ = mlx.mlx_array_free(x);
    const sc_shape = [_]c_int{TOPK};
    const scores = try attn256RandBf16(rnd, &sc_shape, s);
    defer _ = mlx.mlx_array_free(scores);
    var idata: [8]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const inds = mlx.mlx_array_new_data(&idata, &sc_shape, 1, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const no_arr = mlx.mlx_array{ .ctx = null };

    // mxfp8 mode: wrong dequant arithmetic — must decline, not compute.
    try std.testing.expect((try gatherQmvDownReduce(s, x, no_arr, no_arr, no_arr, inds, scores, 8, 32, .mxfp8)) == null);
    // f32 scores against bf16 x: the composed multiply would promote — decline.
    var scores_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_f32);
    try mlx.check(mlx.mlx_astype(&scores_f32, scores, .float32, s));
    const wn: usize = 8 * 32 * 8;
    const wdata = try std.testing.allocator.alloc(u32, wn);
    defer std.testing.allocator.free(wdata);
    for (wdata) |*v| v.* = rnd.int(u32);
    const w_shape = [_]c_int{ 8, 32, 8 };
    const w = mlx.mlx_array_new_data(wdata.ptr, &w_shape, 3, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const sdata = try std.testing.allocator.alloc(u8, 8 * 32 * 4);
    defer std.testing.allocator.free(sdata);
    for (sdata) |*v| v.* = 0x30;
    const s_shape = [_]c_int{ 8, 32, 4 };
    const scq = mlx.mlx_array_new_data(sdata.ptr, &s_shape, 3, .uint8);
    defer _ = mlx.mlx_array_free(scq);
    try std.testing.expect((try gatherQmvDownReduce(s, x, w, scq, no_arr, inds, scores_f32, 4, 16, .nvfp4)) == null);
}
