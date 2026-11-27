//! sdk/blocks/moe/down_reduce.
//! Owns gatherQmvDownReduce(Rows), its eligibility, the lanes per row, and their tests.

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
const GQMV_BODY_AFFINE = @import("gather_qmv.zig").GQMV_BODY_AFFINE;
const GQMV_BODY_FP4 = @import("gather_qmv.zig").GQMV_BODY_FP4;
const Dequant = @import("gather_qmv.zig").Dequant;
const affineQmvPackUnits = @import("gather_qmv.zig").affineQmvPackUnits;
const bankDequant = @import("gather_qmv.zig").bankDequant;
const modeDequant = @import("gather_qmv.zig").modeDequant;
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
fn gatherQmvDownReduceSource(comptime d: Dequant) [:0]const u8 {
    // LPR lanes share one output row and ROWS = 32 / LPR rows ride one simdgroup
    // in parallel: K is short (the MoE intermediate), so one 32-lane reduction
    // per row cost more than the dot product it reduced, and K/8 packs over 32
    // lanes left a third of them idle. Every row's packs split exactly over LPR
    // lanes, and one log2(LPR)-step shuffle reduces all ROWS rows at once.
    // One source serves one row (ONE_ROW) and 2..8 rows (grid.y); at one row the two compile to the same code.
    return std.fmt.comptimePrint(kernelSource("kernels/moe/down_reduce.fmt.metal"), .{
        if (d == .fp4) GQMV_BODY_FP4 else GQMV_BODY_AFFINE,
    });
}

/// Indexed by `Dequant`.
pub const GQMV_DOWNRED_SOURCES = [2][:0]const u8{ gatherQmvDownReduceSource(.affine), gatherQmvDownReduceSource(.fp4) };
pub const GQMV_DOWNRED_NAMES = [2][:0]const u8{ "mlxserve_moe_gather_qmv_downred", "mlxserve_moe_gather_qmv_downred_fp4" };
var gqmv_downred_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getGatherQmvDownReduceKernel(d: Dequant) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @intFromEnum(d);
    if (gqmv_downred_kernels[mi]) |k| return k;
    const affine_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size" };
    const fp4_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "inds", "scores", "K_size", "N_size" };
    const input_names: []const [*:0]const u8 = if (d == .fp4) &fp4_inputs else &affine_inputs;
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_DOWNRED_NAMES[mi].ptr,
        in_vec,
        out_vec,
        GQMV_DOWNRED_SOURCES[mi],
        gather_qmv.header(d),
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

const DownRedCfgKey = struct { nrows: c_int, topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype, lpr: c_int, one_row: bool };
const DownRedCfgSlot = struct {
    cfg: ?mlx.mlx_fast_metal_kernel_config = null,
    key: DownRedCfgKey = std.mem.zeroes(DownRedCfgKey),
};
/// One slot per layout (single-row decode, rows tick).
var downred_cfg: [2]DownRedCfgSlot = .{ .{}, .{} };
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
    return downReduceLaunch(s, x, w, sc, bi, inds, scores, bits, group_size, mode, true);
}

/// Asked BEFORE the gate/up dispatch: a decline is static per geometry and
/// would otherwise cost a wasted gate/up launch on every MoE layer of every tick.
pub fn gatherQmvDownReduceRowsEligible(hidden_act: anytype, gate_qp: anytype, up_qp: anytype, down_qp: anytype, N: c_int) bool {
    if (!downReduceFusedEnabled()) return false;
    if (!moeRowsFusedEnabled()) return false;
    if (hidden_act != .silu) return false;
    if (gate_qp.bits != up_qp.bits or gate_qp.group_size != up_qp.group_size or gate_qp.mode != up_qp.mode) return false;
    const d = modeDequant(down_qp.mode, down_qp.bits, down_qp.group_size) orelse return false;
    // 3-bit down rows are unregistered in the kernel contract; those banks keep the sorted arm.
    if (d == .affine and down_qp.bits == 3) return false;
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
    const y = (try downReduceLaunch(s, x, w, sc, bi, inds, scores, bits, group_size, mode, false)) orelse return null;
    if (!downred_rows_engaged) {
        downred_rows_engaged = true;
        const ish = mlx.getShape(inds);
        log.info("[moe] fused rows down+reduce kernel engaged: B={d} E_topk={d} hidden={d} bits={d} gs={d}\n", .{ ish[0], ish[1], mlx.getShape(y)[1], bits, group_size });
    }
    return y;
}
var downred_rows_engaged: bool = false;

/// The down+reduce kernel. `one_row`: x [TOPK, K] (any shape of TOPK * K), inds and scores [TOPK], y [N], and an
/// expert width under 2048 (from 2048 the gathered down projection plus the composed sum beats the lanes: GLM-5-Next
/// and MiMo widths); else x [B, TOPK, K], inds and scores [B, TOPK] (B 1..8), y [B, N]. Refuses: a bank no dequant
/// reads (`bankDequant`), scores in another dtype than x, top-k past 32 (one simdgroup a slot), N off the row tile.
fn downReduceLaunch(
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
    one_row: bool,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const d = bankDequant(mode, bits, group_size, sc, bi) orelse return null;
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
    const ssh = mlx.getShape(scores);
    var nrows: c_int = 1;
    var topk: c_int = undefined;
    if (one_row) {
        if (K >= 2048) return null;
        if (ish.len != 1 or ssh.len != 1 or ssh[0] != ish[0]) return null;
        topk = ish[0];
    } else {
        if (ish.len != 2 or ssh.len != 2 or ssh[0] != ish[0] or ssh[1] != ish[1]) return null;
        nrows = ish[0];
        topk = ish[1];
        if (nrows < 1 or nrows > 8) return null;
    }
    if (topk <= 0 or topk > 32) return null;
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |v| xelems *= v;
    if (xelems != @as(i64, nrows) * topk * K) return null;
    const lpr = downredLpr();
    const rows: c_int = @divExact(32, lpr);
    if (@rem(N, rows) != 0) return null;

    const key = DownRedCfgKey{ .nrows = nrows, .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd, .lpr = lpr, .one_row = one_row };
    const slot = &downred_cfg[@intFromBool(one_row)];
    if (slot.cfg == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        if (one_row) {
            const y_shape = [_]c_int{N};
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 1, xd));
        } else {
            const y_shape = [_]c_int{ nrows, N };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(N, rows) * topk * 32, nrows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LPR", lpr));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KP", affineQmvPackUnits(K, bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ONE_ROW", one_row));
        slot.* = .{ .cfg = cfg, .key = key };
    }

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const affine_inputs = [_]mlx.mlx_array{ x, w, sc, bi, inds, scores, K_arr, N_arr };
    const fp4_inputs = [_]mlx.mlx_array{ x, w, sc, inds, scores, K_arr, N_arr };
    const inputs_arr: []const mlx.mlx_array = if (d == .fp4) &fp4_inputs else &affine_inputs;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvDownReduceKernel(d);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, slot.cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
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

test "moe rows down+reduce takes every registered width but 3-bit" {
    downred_override = true;
    defer downred_override = null;
    const Qp = struct { bits: u32, group_size: u32, mode: QuantMode = .affine };
    const gu: Qp = .{ .bits = 4, .group_size = 32 };
    const Act = enum { silu, gelu };
    for ([_]u32{ 2, 4, 5, 6, 8 }) |bits| try std.testing.expect(gatherQmvDownReduceRowsEligible(Act.silu, gu, gu, Qp{ .bits = bits, .group_size = 32 }, 2560));
    for ([_]u32{ 3, 7 }) |bits| try std.testing.expect(!gatherQmvDownReduceRowsEligible(Act.silu, gu, gu, Qp{ .bits = bits, .group_size = 32 }, 2560));
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
