// SPDX-License-Identifier: Apache-2.0
// Ported from oMLX (jundot/omlx) omlx/patches/m5_gather_qmm_nax.py @ d6b2b92.
const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const fp4 = @import("../../quant/fp4.zig");

/// The gathers' shared header, after the one fp4 scale decode its `Fp4Q` reads.
pub const NAX_HEADER = fp4.SCALE_HEADER ++ kernelSource("kernels/moe/gather_qmm_nax_header.metal");

pub const Plan = struct { sched: enum { seg, db }, bm: c_int, bk: c_int, gx: c_int, pad: c_int };
/// `mx`: an FP4 bank (uint8 scales, no biases: e8m0 per 32 for MXFP4, e4m3 per 16 for NVFP4) instead of affine.
const CanaryKey = struct { plan: Plan, bits: u32, group: u32, align_n: bool, align_k: bool, mx: bool };
const MmKey = struct { rows: c_int, n: c_int, k: c_int, max_tiles: c_int, plan: Plan, bits: u32, group: u32, mx: bool, mapped: bool = false, paired: bool = false, weighted: bool = false, limit: u32 = 0, esc: u32 = 0, routed: bool = false };
var scan_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mm_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mapped_kernel: ?mlx.mlx_fast_metal_kernel = null;
var paired_kernel: ?mlx.mlx_fast_metal_kernel = null;
var canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var mapped_canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var paired_canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var weighted_canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var engaged_logged = false;
var mapped_engaged_logged = false;
var paired_engaged_logged = false;

/// Tile configuration by mean rows per expert and K. The 96+-row rungs are ours: oMLX's
/// seg tiles were measured against mlx 0.32.2, whose sorted kernel masked whole row blocks;
/// 0.32.3 schedules row tiles itself and the plain db 96-row tile-on-x layout is the one
/// that still beats it. The paired gate/up kernel keeps seg 128.
pub fn plan(rows: c_int, experts: c_int, k: c_int, n: c_int, paired: bool) Plan {
    if (@rem(k, 64) != 0 or @rem(n, 64) != 0) return .{ .sched = .seg, .bm = 64, .bk = 64, .gx = 0, .pad = 0 };
    const per_expert = @divTrunc(rows, @max(experts, 1));
    // A prompt's few rows an expert (a replay's 128 rows over 384 experts): the weight reads bound it, so the
    // smallest tile wastes the least tensor work.
    if (per_expert < 4) return .{ .sched = .db, .bm = 32, .bk = 64, .gx = 0, .pad = 0 };
    if (per_expert < 36 or (k < 1024 and per_expert < 120)) return .{ .sched = .db, .bm = 64, .bk = 64, .gx = 0, .pad = 0 };
    if (k < 1024 and paired) return .{ .sched = .seg, .bm = 96, .bk = 128, .gx = 32, .pad = 0 };
    if (k >= 1024 and per_expert < 48) return .{ .sched = .db, .bm = 64, .bk = 64, .gx = 32, .pad = 0 };
    if (k < 1024 or per_expert < 96 or !paired) return .{ .sched = .db, .bm = 96, .bk = 64, .gx = 32, .pad = 0 };
    return .{ .sched = .seg, .bm = 128, .bk = 128, .gx = 32, .pad = 8192 };
}

fn supported(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, row_map: ?mlx.mlx_array, prompt: bool) bool {
    if (x.ctx == null or w.ctx == null or sc.ctx == null or idx.ctx == null) return false;
    if (mx) {
        if (bits != 4 or (group != 32 and group != 16) or bi.ctx != null or mlx.mlx_array_dtype(sc) != .uint8) return false;
    } else {
        if (bi.ctx == null) return false;
        if (bits != 2 and bits != 3 and bits != 4 and bits != 8) return false;
        if (group != 32 and group != 64 and group != 128) return false;
        if (mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16 or mlx.mlx_array_ndim(bi) != 3) return false;
        if (!std.mem.eql(c_int, mlx.getShape(sc), mlx.getShape(bi))) return false;
    }
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(idx) != .uint32) return false;
    if (mlx.mlx_array_ndim(x) != 3 or mlx.mlx_array_ndim(w) != 3 or mlx.mlx_array_ndim(sc) != 3 or mlx.mlx_array_ndim(idx) != 1) return false;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const ids = mlx.getShape(idx);
    if (xs[1] != 1 or xs[2] <= 0 or @rem(xs[2], 32) != 0 or @rem(xs[2], @as(c_int, @intCast(group))) != 0) return false;
    if (row_map) |map| {
        if (map.ctx == null or mlx.mlx_array_dtype(map) != .uint32 or mlx.mlx_array_ndim(map) != 1 or mlx.getShape(map)[0] != ids[0]) return false;
        if (@as(i64, xs[0]) * xs[2] >= std.math.maxInt(u32)) return false;
    } else if (xs[0] != ids[0]) return false;
    if (ws[0] <= 0 or ws[0] > 2048 or ws[1] <= 0 or @rem(ws[1], 32) != 0 or @as(i64, ws[2]) * 32 != @as(i64, xs[2]) * bits) return false;
    // Take only the calls MLX sends to its sorted row-block kernel (`B >= 16 && B / E >= 4`,
    // quantized.cpp GatherQMM::eval_gpu); below that it runs gather_qmv, which sums in another
    // order, and MTP verify rounds must keep its bits. A prompt (`prompt`) keeps no such bits: any 16 rows.
    if (ids[0] < 16 or (!prompt and @divTrunc(ids[0], ws[0]) < 4)) return false;
    return ss[0] == ws[0] and ss[1] == ws[1] and ss[2] == @divTrunc(xs[2], @as(c_int, @intCast(group)));
}

fn kernel(scan: bool) !mlx.mlx_fast_metal_kernel {
    if (scan) {
        if (scan_kernel) |v| return v;
    } else {
        if (mm_kernel) |v| return v;
    }
    const in_names_scan = [_][*:0]const u8{ "idx", "params" };
    const out_names_scan = [_][*:0]const u8{ "tiles", "tile_count" };
    const in_names_mm = [_][*:0]const u8{ "x", "w", "scales", "biases", "tiles", "tile_count", "params" };
    const out_names_mm = [_][*:0]const u8{"y"};
    const in_vec = if (scan) mlx.mlx_vector_string_new_data(&in_names_scan, in_names_scan.len) else mlx.mlx_vector_string_new_data(&in_names_mm, in_names_mm.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = if (scan) mlx.mlx_vector_string_new_data(&out_names_scan, out_names_scan.len) else mlx.mlx_vector_string_new_data(&out_names_mm, out_names_mm.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const value = if (scan)
        mlx.mlx_fast_metal_kernel_new("msv_gqmm_tile_scan", in_vec, out_vec, kernelSource("kernels/moe/gather_qmm_nax_scan.metal"), NAX_HEADER, true, false)
    else
        mlx.mlx_fast_metal_kernel_new("msv_gqmm_affine", in_vec, out_vec, kernelSource("kernels/moe/gather_qmm_nax.metal"), NAX_HEADER, true, false);
    if (value.ctx == null) return error.MetalKernelCompileFailed;
    if (scan) scan_kernel = value else mm_kernel = value;
    return value;
}

fn mapKernel(paired: bool) !mlx.mlx_fast_metal_kernel {
    const slot = if (paired) &paired_kernel else &mapped_kernel;
    if (slot.*) |v| return v;
    const names = [_][*:0]const u8{ "x", "w", "scales", "biases", "up_w", "up_scales", "up_biases", "tiles", "tile_count", "params", "sigtab", "rmap", "gsc", "usc", "rw", "row_wt" };
    const outs = [_][*:0]const u8{"y"};
    const ins_vec = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(ins_vec);
    const outs_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(outs_vec);
    const header = NAX_HEADER ++ kernelSource("kernels/moe/gather_qmm_nax_mapped_header.metal");
    const value = mlx.mlx_fast_metal_kernel_new(if (paired) "msv_gqmm_affine_pair" else "msv_gqmm_affine_map", ins_vec, outs_vec, kernelSource("kernels/moe/gather_qmm_nax_mapped.metal"), header, true, false);
    if (value.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = value;
    return value;
}

pub const Tiles = struct {
    tiles: mlx.mlx_array,
    count: mlx.mlx_array,
    max_tiles: c_int,

    pub fn deinit(self: Tiles) void {
        _ = mlx.mlx_array_free(self.tiles);
        _ = mlx.mlx_array_free(self.count);
    }
};

/// The expert-run tile pre-pass both matmul kernels read. Configs are built per call: their
/// output shapes carry the row count, so a cache keyed on it grew with every prompt length.
pub fn scanTiles(idx: mlx.mlx_array, m: c_int, e: c_int, bm: c_int, s: mlx.mlx_stream) !Tiles {
    const max_tiles = @divTrunc(m + bm - 1, bm) + @min(e, m);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const tile_shape = [_]c_int{max_tiles * 4};
    const one = [_]c_int{1};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &tile_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &one, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BM", bm));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MAXE", 2048));
    const params = mlx.mlx_array_new_data(&[_]c_int{ m, e, max_tiles }, &[_]c_int{3}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const inputs = [_]mlx.mlx_array{ idx, params };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try kernel(true), in_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outs) != 2) return error.MetalKernelBadOutputCount;
    const tiles = try outputAt(outs, 0);
    errdefer _ = mlx.mlx_array_free(tiles);
    return .{ .tiles = tiles, .count = try outputAt(outs, 1), .max_tiles = max_tiles };
}

/// Caller frees the config.
fn mmConfig(key: MmKey) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ key.rows, 1, key.n };
    const cols = @divTrunc((if (key.paired) key.n * 2 else key.n) + 63, 64);
    const grid_x = if (key.plan.gx > 0) key.plan.gx else cols;
    const grid_y = if (key.plan.gx > 0) @divTrunc(key.max_tiles + key.plan.gx - 1, key.plan.gx) * cols else key.max_tiles;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid_x * 32, grid_y * 2, @divTrunc(key.plan.bm, 32)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, @divTrunc(key.plan.bm, 32)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(key.group)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(key.bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MX", @intFromBool(key.mx)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SCHED", if (key.plan.sched == .db) 1 else 0));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ALIGN_N", @rem((if (key.paired) key.n * 2 else key.n), 64) == 0));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ALIGN_K", @rem(key.k, key.plan.bk) == 0));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BM", key.plan.bm));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BK", key.plan.bk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GX", key.plan.gx));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PAD", key.plan.pad));
    if (key.mapped) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPI", if (key.routed) -1 else if (key.paired) 1 + @as(c_int, @intCast(key.limit)) else 0));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ESC", @intCast(key.esc)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "WEIGHTED", key.weighted));
    }
    return cfg;
}

/// `limit`: a `swiglu_limit` the epilogue clamps at (whole numbers; 0 = none). `up_off`: the up rows' offset in an
/// expert of a bank fused along its rows (gate rows, then up rows; `w` is then the gate's bank too), else 0.
/// `scales`: the per-expert gate / up scales the epilogue applies (`ExpertScales`). `route`: the f32 routing weight of
/// each row, which selects the routed epilogue (`sortedGateUpRouted`, clamped under `route_limit`); `limit` is then unread.
const Pair = struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, sigtab: mlx.mlx_array, row_wt: mlx.mlx_array = .{ .ctx = null }, limit: u32 = 0, up_off: c_int = 0, scales: ExpertScales = .{}, route: ?mlx.mlx_array = null, route_limit: f32 = 0 };

/// Per-expert scales `[E]` in x's dtype (ModelOpt NVFP4's global `weight_scale_2`) on the paired kernel's two products:
/// `gate` multiplies the gate before the SiLU (and before a clamp), `up` the up before a clamp, each product rounded
/// to x's dtype first and the scaled value rounded again, as the split path's `scaleExperts` multiply and the fp4
/// decode kernel's GSCALE / USCALE do. Null = none (an MXFP4 bank, or ModelOpt's up scale riding the scores).
pub const ExpertScales = struct {
    gate: ?mlx.mlx_array = null,
    up: ?mlx.mlx_array = null,

    fn bits(self: ExpertScales) u32 {
        return @as(u32, @intFromBool(self.gate != null)) | (@as(u32, @intFromBool(self.up != null)) << 1);
    }

    /// Each scale an `[experts]` vector of `dt`, at least 8 long (MLX binds a shorter array in `constant`).
    fn fits(self: ExpertScales, experts: c_int, dt: mlx.mlx_dtype) bool {
        inline for (.{ self.gate, self.up }) |o| if (o) |a| {
            if (a.ctx == null or mlx.mlx_array_dtype(a) != dt or mlx.mlx_array_ndim(a) != 1 or mlx.getShape(a)[0] != experts or experts < 8) return false;
        };
        return true;
    }
};

/// An MXFP4 bank has no biases; the kernel signature still binds the slot. 16
/// elements: MLX binds an array under 8 in `constant`, not `device`.
var no_bias: mlx.mlx_array = .{ .ctx = null };
fn biasArg(bi: mlx.mlx_array) mlx.mlx_array {
    if (bi.ctx != null) return bi;
    if (no_bias.ctx == null) no_bias = mlx.mlx_array_new_data(&@as([16]u16, @splat(0)), &[_]c_int{16}, 1, .bfloat16);
    return no_bias;
}

/// The routing weights' slot without the routed epilogue: 16 f32 zeros (the slot stays a `device float*`).
var no_route: mlx.mlx_array = .{ .ctx = null };
fn routeArg(rw: ?mlx.mlx_array) mlx.mlx_array {
    if (rw) |v| return v;
    if (no_route.ctx == null) no_route = mlx.mlx_array_new_data(&@as([16]f32, @splat(0)), &[_]c_int{16}, 1, .float32);
    return no_route;
}

var no_wt: mlx.mlx_array = .{ .ctx = null };
fn wtArg(wt: mlx.mlx_array) mlx.mlx_array {
    if (wt.ctx != null) return wt;
    if (no_wt.ctx == null) no_wt = mlx.mlx_array_new_data(&@as([16]f32, @splat(0)), &[_]c_int{16}, 1, .float32);
    return no_wt;
}

fn launchMapped(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, pair: ?Pair, bits: u32, group: u32, mx: bool, p: Plan, s: mlx.mlx_stream) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const m = mlx.getShape(idx)[0];
    const e = ws[0];
    const up_off: c_int = if (pair) |v| v.up_off else 0;
    const n = ws[1] - up_off;
    const k = xs[2];
    const t = try scanTiles(idx, m, e, p.bm, s);
    defer t.deinit();
    const route: ?mlx.mlx_array = if (pair) |v| v.route else null;
    const lim: f32 = if (pair) |v| v.route_limit else 0;
    const params = mlx.mlx_array_new_data(&[_]c_int{ if (pair != null) n * 2 else n, k, up_off, @bitCast(lim) }, &[_]c_int{4}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const pw = if (pair) |v| v.w else w;
    const ps = if (pair) |v| v.sc else sc;
    const pb = if (pair) |v| v.bi else bi;
    const sigtab = if (pair) |v| v.sigtab else biasArg(.{ .ctx = null }); // unread without a pair
    const esc: ExpertScales = if (pair) |v| v.scales else .{};
    const row_wt: mlx.mlx_array = if (pair) |v| v.row_wt else .{ .ctx = null };
    // An absent scale binds the sigmoid table (a bf16 array past `constant`'s size); ESC keeps it unread.
    const inputs = [_]mlx.mlx_array{ x, w, sc, biasArg(bi), pw, ps, biasArg(pb), t.tiles, t.count, params, sigtab, row_map, esc.gate orelse sigtab, esc.up orelse sigtab, routeArg(route), wtArg(row_wt) };
    const cfg = try mmConfig(.{ .rows = m, .n = n, .k = k, .max_tiles = t.max_tiles, .plan = p, .bits = bits, .group = group, .mx = mx, .mapped = true, .paired = pair != null, .limit = if (pair) |v| v.limit else 0, .esc = esc.bits(), .routed = route != null, .weighted = row_wt.ctx != null });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    return applyOne(try mapKernel(pair != null), &inputs, cfg, s);
}

fn applyOne(k: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) !mlx.mlx_array {
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, in_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outs) != 1) return error.MetalKernelBadOutputCount;
    return outputAt(outs, 0);
}

fn outputAt(vec: mlx.mlx_vector_array, index: usize) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, vec, index));
    return out;
}

fn launch(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, p: Plan, s: mlx.mlx_stream) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const m = xs[0];
    const e = ws[0];
    const n = ws[1];
    const k = xs[2];
    const t = try scanTiles(idx, m, e, p.bm, s);
    defer t.deinit();
    const params = mlx.mlx_array_new_data(&[_]c_int{ n, k }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const inputs = [_]mlx.mlx_array{ x, w, sc, biasArg(bi), t.tiles, t.count, params };
    const cfg = try mmConfig(.{ .rows = m, .n = n, .k = k, .max_tiles = t.max_tiles, .plan = p, .bits = bits, .group = group, .mx = mx });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    return applyOne(try kernel(false), &inputs, cfg, s);
}

fn canary(key: CanaryKey, s: mlx.mlx_stream) !bool {
    const counts = [_]u32{ 70, 0, 5, 33, 64, 17, 140, 11 };
    const rows: c_int = 340;
    const n: c_int = if (key.align_n) 128 else 96;
    const k: c_int = if (key.align_k) 256 else if (key.plan.bk == 128) 320 else 160;
    const w_shape = [_]c_int{ 8, n, k };
    const x_shape = [_]c_int{ rows, 1, k };
    var random_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(random_key);
    try mlx.check(mlx.mlx_random_key(&random_key, 0x2267));
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_random_normal(&wf, &w_shape, 3, .bfloat16, 0, 0.05, random_key, s));
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(@intCast(key.group)), mlx.mlx_optional_int.some(@intCast(key.bits)), modeName(key.mx, key.group), .{}, s));
    const w = try outputAt(q, 0);
    defer _ = mlx.mlx_array_free(w);
    const sc = try outputAt(q, 1);
    defer _ = mlx.mlx_array_free(sc);
    const bi: mlx.mlx_array = if (key.mx) .{ .ctx = null } else try outputAt(q, 2);
    defer if (bi.ctx != null) {
        _ = mlx.mlx_array_free(bi);
    };
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &x_shape, 3, .bfloat16, 0, 0.5, random_key, s));
    var ids_data: [rows]u32 = undefined;
    var pos: usize = 0;
    for (counts, 0..) |count, expert| {
        for (0..count) |_| {
            ids_data[pos] = @intCast(expert);
            pos += 1;
        }
    }
    const ids_shape = [_]c_int{rows};
    const ids = mlx.mlx_array_new_data(&ids_data, &ids_shape, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const got = try launch(x, w, sc, bi, ids, key.bits, key.group, key.mx, key.plan, s);
    defer _ = mlx.mlx_array_free(got);
    // Stock NAX reads stale activations in a K tail: compare against a dequantized fp32 product there.
    if (@rem(k, 64) != 0) return floatReferenceClose(got, x, w, sc, bi, ids, key.bits, key.group, key.mx, s);
    const ref = try stockGather(x, w, sc, bi, ids, key.bits, key.group, key.mx, s);
    defer _ = mlx.mlx_array_free(ref);
    return arraysEqual(got, ref, s);
}

fn stockGather(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array, bits: u32, group: u32, mx: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    var ref = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_gather_qmm(&ref, x, w, sc, bi, .{}, ids, true, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), modeName(mx, group), true, s));
    return ref;
}

/// `got` against a dequantized fp32 gather-matmul, within 1/64 of the reference's peak.
fn floatReferenceClose(got: mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array, bits: u32, group: u32, mx: bool, s: mlx.mlx_stream) !bool {
    var wd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wd);
    try mlx.check(mlx.mlx_dequantize(&wd, w, sc, bi, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), modeName(mx, group), .{}, .{ .value = .float32, .has_value = true }, s));
    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_axis(&gathered, wd, ids, 0, s));
    var transposed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(transposed);
    const axes = [_]c_int{ 0, 2, 1 };
    try mlx.check(mlx.mlx_transpose_axes(&transposed, gathered, &axes, 3, s));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, xf, transposed, s));
    return withinPeak(got, ref, s);
}

/// max |got - ref| within 1/64 of ref's peak.
fn withinPeak(got: mlx.mlx_array, ref: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var gotf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gotf);
    try mlx.check(mlx.mlx_astype(&gotf, got, .float32, s));
    var delta = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(delta);
    try mlx.check(mlx.mlx_subtract(&delta, gotf, ref, s));
    var abs_delta = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(abs_delta);
    try mlx.check(mlx.mlx_abs(&abs_delta, delta, s));
    var max_delta = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(max_delta);
    try mlx.check(mlx.mlx_max(&max_delta, abs_delta, false, s));
    var abs_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(abs_ref);
    try mlx.check(mlx.mlx_abs(&abs_ref, ref, s));
    var max_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(max_ref);
    try mlx.check(mlx.mlx_max(&max_ref, abs_ref, false, s));
    var err: f32 = 0;
    var scale: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&err, max_delta));
    try mlx.check(mlx.mlx_array_item_float32(&scale, max_ref));
    return std.math.isFinite(err) and err <= scale / 64;
}

/// An FP4 bank is MXFP4 at group 32 (e8m0 scales) or NVFP4 at group 16 (e4m3 scales).
fn modeName(mx: bool, group: u32) [*:0]const u8 {
    return if (!mx) "affine" else if (group == 16) "nvfp4" else "mxfp4";
}

fn armed(key: CanaryKey, s: mlx.mlx_stream) bool {
    if (canaries.get(key)) |ok| return ok;
    // The canary runs inside a prefill forward: an error an earlier op latched belongs to
    // that forward and must reach its `checkError`, so drop only a latch the canary raised.
    const had_error = mlx.errorPending();
    const ok = canary(key, s) catch blk: {
        mlx.dropLatchedErrorUnless(had_error);
        break :blk false;
    };
    canaries.put(std.heap.c_allocator, key, ok) catch return false;
    if (!ok) log.warn("[gather-nax] disabled for {d}-bit group={d} {s} bm={d} bk={d} aligned-N={any} aligned-K={any}: canary mismatch\n", .{ key.bits, key.group, @tagName(key.plan.sched), key.plan.bm, key.plan.bk, key.align_n, key.align_k });
    return ok;
}

/// Sorted rhs indices must be one contiguous run per expert.
pub fn sortedGather(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGatherAt(x, w, sc, bi, idx, bits, group, mx, nax_available, false, s);
}

/// `sortedGather` for a prompt's rows, which keep no stock kernel's bits: it also takes calls of under four rows an
/// expert, which MLX would run as one gather_qmv a row (each row reading its experts' weights again).
pub fn sortedGatherPrompt(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGatherAt(x, w, sc, bi, idx, bits, group, mx, nax_available, true, s);
}

fn sortedGatherAt(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, prompt: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, mx, null, prompt)) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const p = plan(xs[0], ws[0], xs[2], ws[1], false);
    const key: CanaryKey = .{ .plan = p, .bits = bits, .group = group, .align_n = @rem(ws[1], 64) == 0, .align_k = @rem(xs[2], p.bk) == 0, .mx = mx };
    if (!armed(key, s)) return null;
    const out = try launch(x, w, sc, bi, idx, bits, group, mx, p, s);
    if (!engaged_logged) {
        engaged_logged = true;
        log.info("[gather-nax] engaged: {s} bm={d} bk={d} M={d} E={d} N={d} K={d} {s}\n", .{ @tagName(p.sched), p.bm, p.bk, xs[0], ws[0], ws[1], xs[2], modeName(mx, group) });
    }
    return out;
}

fn arraysEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}

fn expectBitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    try std.testing.expect(try arraysEqual(a, b, s));
}

fn canaryMapped(key: CanaryKey, paired: bool, weighted: bool, s: mlx.mlx_stream) !bool {
    const rows: c_int = 340;
    const tokens: c_int = 97;
    const experts: c_int = 8;
    const n: c_int = if (key.align_n) 128 else 96;
    const k: c_int = if (key.align_k) 256 else if (key.plan.bk == 128) 320 else 160;
    var random_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(random_key);
    try mlx.check(mlx.mlx_random_key(&random_key, 0x2267));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ tokens, 1, k }, 3, .bfloat16, 0, 0.5, random_key, s));
    const counts = [_]u32{ 70, 0, 5, 33, 64, 17, 140, 11 };
    var ids_data: [rows]u32 = undefined;
    var map_data: [rows]u32 = undefined;
    var wt_data: [rows]f32 = undefined;
    var pos: usize = 0;
    for (counts, 0..) |count, expert| {
        for (0..count) |_| {
            ids_data[pos] = @intCast(expert);
            map_data[pos] = @intCast((pos * 7 + 3) % tokens);
            wt_data[pos] = 0.15 + @as(f32, @floatFromInt((pos * 19 + 5) % 97)) * 0.02;
            pos += 1;
        }
    }
    const ids = mlx.mlx_array_new_data(&ids_data, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const map = mlx.mlx_array_new_data(&map_data, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(map);
    const row_wt: mlx.mlx_array = if (weighted) mlx.mlx_array_new_data(&wt_data, &[_]c_int{rows}, 1, .float32) else .{ .ctx = null };
    defer if (row_wt.ctx != null) {
        _ = mlx.mlx_array_free(row_wt);
    };
    var x_rep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_rep);
    try mlx.check(mlx.mlx_take_axis(&x_rep, x, map, 0, s));
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_random_normal(&wf, &[_]c_int{ experts, n, k }, 3, .bfloat16, 0, 0.05, random_key, s));
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    const group = mlx.mlx_optional_int.some(@intCast(key.group));
    const bits = mlx.mlx_optional_int.some(@intCast(key.bits));
    try mlx.check(mlx.mlx_quantize(&q, wf, group, bits, modeName(key.mx, key.group), .{}, s));
    const w = try outputAt(q, 0);
    defer _ = mlx.mlx_array_free(w);
    const sc = try outputAt(q, 1);
    defer _ = mlx.mlx_array_free(sc);
    const bi: mlx.mlx_array = if (key.mx) .{ .ctx = null } else try outputAt(q, 2);
    defer if (bi.ctx != null) {
        _ = mlx.mlx_array_free(bi);
    };
    const gate_ref = try launch(x_rep, w, sc, bi, ids, key.bits, key.group, key.mx, key.plan, s);
    defer _ = mlx.mlx_array_free(gate_ref);
    if (!paired) {
        const got = try launchMapped(x, map, w, sc, bi, ids, null, key.bits, key.group, key.mx, key.plan, s);
        defer _ = mlx.mlx_array_free(got);
        return arraysEqual(got, gate_ref, s);
    }
    const two = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(two);
    var two_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(two_bf16);
    try mlx.check(mlx.mlx_astype(&two_bf16, two, .bfloat16, s));
    var uf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uf);
    try mlx.check(mlx.mlx_multiply(&uf, wf, two_bf16, s));
    var uq = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(uq);
    try mlx.check(mlx.mlx_quantize(&uq, uf, group, bits, modeName(key.mx, key.group), .{}, s));
    const uw = try outputAt(uq, 0);
    defer _ = mlx.mlx_array_free(uw);
    const us = try outputAt(uq, 1);
    defer _ = mlx.mlx_array_free(us);
    const ub: mlx.mlx_array = if (key.mx) .{ .ctx = null } else try outputAt(uq, 2);
    defer if (ub.ctx != null) {
        _ = mlx.mlx_array_free(ub);
    };
    const up_ref = try launch(x_rep, uw, us, ub, ids, key.bits, key.group, key.mx, key.plan, s);
    defer _ = mlx.mlx_array_free(up_ref);
    const sigtab = try @import("../hc/hc_prefill.zig").sigmoidTable(s);
    const got = try launchMapped(x, map, w, sc, bi, ids, .{ .w = uw, .sc = us, .bi = ub, .sigtab = sigtab, .row_wt = row_wt }, key.bits, key.group, key.mx, key.plan, s);
    defer _ = mlx.mlx_array_free(got);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_sigmoid(&sig, gate_ref, s));
    try mlx.check(mlx.mlx_multiply(&act, gate_ref, sig, s));
    try mlx.check(mlx.mlx_multiply(&ref, act, up_ref, s));
    if (!weighted) return arraysEqual(got, ref, s);
    var wt3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt3);
    try mlx.check(mlx.mlx_reshape(&wt3, row_wt, &[_]c_int{ rows, 1, 1 }, 3, s));
    var ref_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_f32);
    try mlx.check(mlx.mlx_multiply(&ref_f32, ref, wt3, s));
    var ref_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_bf16);
    try mlx.check(mlx.mlx_astype(&ref_bf16, ref_f32, .bfloat16, s));
    return arraysEqual(got, ref_bf16, s);
}

fn armedMapped(key: CanaryKey, paired: bool, weighted: bool, s: mlx.mlx_stream) bool {
    const cache = if (!paired) &mapped_canaries else if (weighted) &weighted_canaries else &paired_canaries;
    if (cache.get(key)) |ok| return ok;
    const had_error = mlx.errorPending();
    const ok = canaryMapped(key, paired, weighted, s) catch blk: {
        mlx.dropLatchedErrorUnless(had_error);
        break :blk false;
    };
    cache.put(std.heap.c_allocator, key, ok) catch return false;
    if (!ok) log.warn("[gather-nax] {s} canary mismatch: {d}-bit group={d} {s} bm={d} bk={d}\n", .{ if (!paired) @as([]const u8, "mapped") else if (weighted) "weighted" else "paired", key.bits, key.group, @tagName(key.plan.sched), key.plan.bm, key.plan.bk });
    return ok;
}

fn mappedKey(x: mlx.mlx_array, w: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, paired: bool) CanaryKey {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const p = plan(mlx.getShape(idx)[0], ws[0], xs[2], ws[1], paired);
    return .{ .plan = p, .bits = bits, .group = group, .align_n = @rem(ws[1], 64) == 0, .align_k = @rem(xs[2], p.bk) == 0, .mx = mx };
}

pub fn sortedGatherMapped(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGatherMappedAt(x, row_map, w, sc, bi, idx, bits, group, mx, nax_available, false, s);
}

/// `sortedGatherMapped` for a prompt's rows (`sortedGatherPrompt`).
pub fn sortedGatherMappedPrompt(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGatherMappedAt(x, row_map, w, sc, bi, idx, bits, group, mx, nax_available, true, s);
}

fn sortedGatherMappedAt(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, prompt: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, mx, row_map, prompt)) return null;
    const key = mappedKey(x, w, idx, bits, group, mx, false);
    if (!armed(key, s) or !armedMapped(key, false, false, s)) return null;
    const out = try launchMapped(x, row_map, w, sc, bi, idx, null, bits, group, mx, key.plan, s);
    if (!mapped_engaged_logged) {
        mapped_engaged_logged = true;
        log.info("[gather-nax] row map engaged: M={d} E={d} N={d} K={d} {s}\n", .{ mlx.getShape(idx)[0], mlx.getShape(w)[0], mlx.getShape(w)[1], mlx.getShape(x)[2], modeName(mx, group) });
    }
    return out;
}

fn validWt(row_wt: ?mlx.mlx_array, idx: mlx.mlx_array) bool {
    const wt = row_wt orelse return true;
    return wt.ctx != null and mlx.mlx_array_dtype(wt) == .float32 and mlx.mlx_array_ndim(wt) == 1 and mlx.getShape(wt)[0] == mlx.getShape(idx)[0];
}

pub fn sortedGateUp(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, up_w: mlx.mlx_array, up_sc: mlx.mlx_array, up_bi: mlx.mlx_array, idx: mlx.mlx_array, sigtab: mlx.mlx_array, limit: u32, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGateUpAll(x, row_map, w, sc, bi, up_w, up_sc, up_bi, idx, sigtab, limit, .{}, null, bits, group, mx, nax_available, s);
}

var scaled_engaged_logged = false;

/// `sortedGateUp` with each expert's own gate / up scale in the epilogue (`ExpertScales`): the two gathers, the
/// `scaleExperts` multiplies and the SwiGLU (clamped under `limit`) of the split path in one dispatch, bit for bit.
pub fn sortedGateUpScaled(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, up_w: mlx.mlx_array, up_sc: mlx.mlx_array, up_bi: mlx.mlx_array, idx: mlx.mlx_array, sigtab: mlx.mlx_array, limit: u32, scales: ExpertScales, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGateUpAll(x, row_map, w, sc, bi, up_w, up_sc, up_bi, idx, sigtab, limit, scales, null, bits, group, mx, nax_available, s);
}

/// `sortedGateUp` with per-row routing weights `row_wt` (`[M]` f32) multiplied into the rounded activation.
pub fn sortedGateUpWeighted(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, up_w: mlx.mlx_array, up_sc: mlx.mlx_array, up_bi: mlx.mlx_array, idx: mlx.mlx_array, sigtab: mlx.mlx_array, row_wt: mlx.mlx_array, limit: u32, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGateUpAll(x, row_map, w, sc, bi, up_w, up_sc, up_bi, idx, sigtab, limit, .{}, row_wt, bits, group, mx, nax_available, s);
}

/// The paired gather over two banks with the per-expert scales `scales` and the per-row weights `row_wt` (null: none).
fn sortedGateUpAll(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, up_w: mlx.mlx_array, up_sc: mlx.mlx_array, up_bi: mlx.mlx_array, idx: mlx.mlx_array, sigtab: mlx.mlx_array, limit: u32, scales: ExpertScales, row_wt: ?mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, mx, row_map, false) or !supported(x, up_w, up_sc, up_bi, idx, bits, group, mx, row_map, false)) return null;
    if (!std.mem.eql(c_int, mlx.getShape(w), mlx.getShape(up_w)) or @rem(mlx.getShape(w)[1], 32) != 0) return null;
    if (sigtab.ctx == null or mlx.mlx_array_dtype(sigtab) != .bfloat16 or mlx.mlx_array_size(sigtab) != 65536) return null;
    if (!scales.fits(mlx.getShape(w)[0], mlx.mlx_array_dtype(x))) return null;
    if (!validWt(row_wt, idx)) return null;
    const weighted = row_wt != null;
    const key = mappedKey(x, w, idx, bits, group, mx, true);
    if (!armed(key, s) or !armedMapped(key, false, false, s) or !armedMapped(key, true, weighted, s)) return null;
    const out = try launchMapped(x, row_map, w, sc, bi, idx, .{ .w = up_w, .sc = up_sc, .bi = up_bi, .sigtab = sigtab, .row_wt = row_wt orelse .{ .ctx = null }, .limit = limit, .scales = scales }, bits, group, mx, key.plan, s);
    if (scales.bits() != 0) {
        if (!scaled_engaged_logged) {
            scaled_engaged_logged = true;
            log.info("[gather-nax] paired SwiGLU engaged with per-expert scales (gate={} up={} limit={d}): M={d} E={d} N={d} K={d} {s}\n", .{ scales.gate != null, scales.up != null, limit, mlx.getShape(idx)[0], mlx.getShape(w)[0], mlx.getShape(w)[1], mlx.getShape(x)[2], modeName(mx, group) });
        }
    } else if (!paired_engaged_logged) {
        paired_engaged_logged = true;
        log.info("[gather-nax] paired SwiGLU engaged: M={d} E={d} N={d} K={d}\n", .{ mlx.getShape(idx)[0], mlx.getShape(w)[0], mlx.getShape(w)[1], mlx.getShape(x)[2] });
    }
    return out;
}

/// `sortedGateUp` over one bank `[E, 2 n, K]` holding each expert's gate rows then its up rows, for a prompt's rows
/// (`sortedGatherPrompt`): `[M, 1, n]`, the SwiGLU of the two products.
pub fn sortedGateUpFused(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, sigtab: mlx.mlx_array, limit: u32, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGateUpFusedWithWt(x, row_map, w, sc, bi, idx, sigtab, null, limit, bits, group, mx, nax_available, s);
}

/// `sortedGateUpFused` with per-row routing weights `row_wt` (`[M]` f32) multiplied into the rounded activation.
pub fn sortedGateUpFusedWeighted(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, sigtab: mlx.mlx_array, row_wt: mlx.mlx_array, limit: u32, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGateUpFusedWithWt(x, row_map, w, sc, bi, idx, sigtab, row_wt, limit, bits, group, mx, nax_available, s);
}

fn sortedGateUpFusedWithWt(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, sigtab: mlx.mlx_array, row_wt: ?mlx.mlx_array, limit: u32, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, mx, row_map, true)) return null;
    if (sigtab.ctx == null or mlx.mlx_array_dtype(sigtab) != .bfloat16 or mlx.mlx_array_size(sigtab) != 65536) return null;
    if (!validWt(row_wt, idx)) return null;
    const ws = mlx.getShape(w);
    const xs = mlx.getShape(x);
    if (@rem(ws[1], 64) != 0) return null;
    const n = @divExact(ws[1], 2);
    const p = plan(mlx.getShape(idx)[0], ws[0], xs[2], n, true);
    const weighted = row_wt != null;
    const key: CanaryKey = .{ .plan = p, .bits = bits, .group = group, .align_n = @rem(n, 64) == 0, .align_k = @rem(xs[2], p.bk) == 0, .mx = mx };
    if (!armed(key, s) or !armedMapped(key, false, false, s) or !armedMapped(key, true, weighted, s)) return null;
    const out = try launchMapped(x, row_map, w, sc, bi, idx, .{ .w = w, .sc = sc, .bi = bi, .sigtab = sigtab, .row_wt = row_wt orelse .{ .ctx = null }, .limit = limit, .up_off = n }, bits, group, mx, p, s);
    if (!paired_engaged_logged) {
        paired_engaged_logged = true;
        log.info("[gather-nax] paired SwiGLU engaged (fused bank): M={d} E={d} N={d} K={d}\n", .{ mlx.getShape(idx)[0], ws[0], n, xs[2] });
    }
    return out;
}

var routed_engaged_logged = false;
/// Why the last `sortedGateUpRouted` returned null, for a caller's one-time decline line.
pub var routed_decline: []const u8 = "";

/// `sortedGateUpFused` with deepseek_v41's `moe_act` in the epilogue: each rounded product times its expert's scale
/// (`scales`, rounded again), then in f32 the SwiGLU clamped under `limit` (none at 0 or below) times the row's
/// routing weight (`route`, `[M]` f32), rounded once: `[M, 1, n]`, the math of the gather, the scale multiplies and
/// `moe_act` it replaces.
pub fn sortedGateUpRouted(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, route: mlx.mlx_array, limit: f32, scales: ExpertScales, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!routedFits(x, row_map, w, sc, bi, idx, route, scales, bits, group, mx, nax_available, s)) return null;
    const ws = mlx.getShape(w);
    const xs = mlx.getShape(x);
    const m = mlx.getShape(idx)[0];
    const n = @divExact(ws[1], 2);
    const p = plan(m, ws[0], xs[2], n, true);
    const key: CanaryKey = .{ .plan = p, .bits = bits, .group = group, .align_n = @rem(n, 64) == 0, .align_k = @rem(xs[2], p.bk) == 0, .mx = mx };
    if (!armed(key, s) or !armedMapped(key, false, false, s) or !armedMapped(key, true, false, s)) return declineRouted("canary: the paired kernel's canary did not arm for this plan");
    const out = try launchMapped(x, row_map, w, sc, bi, idx, .{ .w = w, .sc = sc, .bi = bi, .sigtab = biasArg(.{ .ctx = null }), .up_off = n, .scales = scales, .route = route, .route_limit = limit }, bits, group, mx, p, s);
    if (!routed_engaged_logged) {
        routed_engaged_logged = true;
        log.info("[gather-nax] paired SwiGLU engaged with routing weights (gate scale={} up scale={} limit={d}): M={d} E={d} N={d} K={d} {s}\n", .{ scales.gate != null, scales.up != null, limit, m, ws[0], n, xs[2], modeName(mx, group) });
    }
    return out;
}

/// The routed call's argument checks; false sets `routed_decline`.
fn routedFits(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, route: mlx.mlx_array, scales: ExpertScales, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) bool {
    if (!nax_available or !mlx.streamIsGpu(s)) return refuseRouted("no NAX GPU stream");
    if (!supported(x, w, sc, bi, idx, bits, group, mx, row_map, true)) return refuseRouted("shape: the bank or input is not a supported layout");
    const ws = mlx.getShape(w);
    if (@rem(ws[1], 64) != 0 or !scales.fits(ws[0], mlx.mlx_array_dtype(x))) return refuseRouted("shape: the gate/up width is not a multiple of 64 or the per-expert scales do not fit");
    if (route.ctx == null or mlx.mlx_array_dtype(route) != .float32 or mlx.mlx_array_ndim(route) != 1 or mlx.getShape(route)[0] != mlx.getShape(idx)[0]) return refuseRouted("shape: the routing weights are not f32 [pairs]");
    return true;
}

fn declineRouted(why: []const u8) ?mlx.mlx_array {
    routed_decline = why;
    return null;
}

fn refuseRouted(why: []const u8) bool {
    routed_decline = why;
    return false;
}

// ── the staged schedule (`gather_qmm_nax_staged_header.metal`) ──

pub const STAGED_HEADER = NAX_HEADER ++ kernelSource("kernels/moe/gather_qmm_nax_mapped_header.metal") ++ kernelSource("kernels/moe/gather_qmm_nax_staged_header.metal");
pub const STAGED_SOURCE = kernelSource("kernels/moe/gather_qmm_nax_staged.metal");
pub const STAGED_INPUTS = [_][*:0]const u8{ "x", "w", "scales", "biases", "up_w", "up_scales", "up_biases", "tiles", "tile_count", "params", "sigtab", "rmap", "gsc", "usc", "rw" };
var staged_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// A staged launch's tile: `bm` rows of one expert by `bn` columns over `wm` x `wn` simdgroups, `bk` deep a step, row
/// tiles `gx` apart on the grid (`tile_of`; 0: one grid row a tile); `tg_a`: its activations staged through
/// threadgroup memory (`gather_staged_tg`, 64 deep only), else each simdgroup's fragments a step ahead in registers.
pub const Staged = struct { bm: c_int, wm: c_int, wn: c_int, gx: c_int, tg_a: bool = false, bn: c_int = 64, bk: c_int = 64 };

/// The staged tile for `rows` routed rows over `experts`: 64 rows over four 16-row simdgroups at every prompt width.
pub fn stagedPlan(rows: c_int, experts: c_int) Staged {
    _ = .{ rows, experts };
    return .{ .bm = 64, .wm = 4, .wn = 1, .gx = 32 };
}

/// The down's tile under deepseek_v41's `MLX_SERVE_DSV41_PF_DOWN_BK32`: the planner's 64 x 64 tile, 32 deep a step,
/// where a call has 96 routed rows an expert and more; null below.
pub fn stagedDownWide(rows: c_int, experts: c_int) ?Staged {
    if (@divTrunc(rows, @max(experts, 1)) < 96) return null;
    return wide_down;
}
pub const wide_down: Staged = .{ .bm = 64, .wm = 4, .wn = 1, .gx = 32, .bk = 32 };

/// Whether the staged schedule serves a routed gate/up call (`rows` routed rows over `experts`, K `k`, `n` a product):
/// not where `plan` picks the seg 128 x 128 schedule (96 rows an expert and more), which runs the pair faster there.
pub fn stagedServesGateUp(rows: c_int, experts: c_int, k: c_int, n: c_int) bool {
    const p = plan(rows, experts, k, n, true);
    return !(p.sched == .seg and p.bm == 128);
}

test "staged NAX gate/up: DeepSeek-V4.1's one-pass and 8192-row calls take the seg 128 schedule, narrower calls the staged one" {
    for ([_]struct { tokens: c_int, staged: bool }{ .{ .tokens = 17638, .staged = false }, .{ .tokens = 8192, .staged = false }, .{ .tokens = 4096, .staged = true }, .{ .tokens = 1254, .staged = true } }) |c|
        try std.testing.expectEqual(c.staged, stagedServesGateUp(c.tokens * 6, 384, 5120, 2304));
}

/// Where the staged schedule serves: an fp4 bank (whole 64-deep steps, whole 64-column tiles; whole steps and tiles of
/// `st`).
fn stagedFits(k: c_int, width: c_int, mx: bool, st: Staged) bool {
    return mx and @rem(k, 64) == 0 and @rem(width, 64) == 0 and @rem(k, st.bk) == 0 and @rem(width, st.bn) == 0;
}

fn stagedKernel() !mlx.mlx_fast_metal_kernel {
    if (staged_kernel) |v| return v;
    const ins = mlx.mlx_vector_string_new_data(&STAGED_INPUTS, STAGED_INPUTS.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs_names = [_][*:0]const u8{"y"};
    const outs = mlx.mlx_vector_string_new_data(&outs_names, outs_names.len);
    defer _ = mlx.mlx_vector_string_free(outs);
    const value = mlx.mlx_fast_metal_kernel_new("msv_gqmm_staged", ins, outs, STAGED_SOURCE, STAGED_HEADER, true, false);
    if (value.ctx == null) return error.MetalKernelCompileFailed;
    staged_kernel = value;
    return value;
}

/// `sortedGatherPromptStagedTok`'s build: the same source with its rows scattered to their pairs' slots
/// (`GQMM_SCAT`, `k` picks a row: one build per k), `ridx` the flat picks after the staged inputs.
var staged_scat_kernel: [9]?mlx.mlx_fast_metal_kernel = @splat(null);
pub const STAGED_SCAT_INPUTS = STAGED_INPUTS ++ [_][*:0]const u8{"ridx"};

/// The scatter build's header at `k` picks a token.
pub fn stagedScatHeader(comptime k: c_int) [:0]const u8 {
    return std.fmt.comptimePrint("#define GQMM_SCAT 1\n#define GQMM_SCAT_K {d}\n", .{k}) ++ STAGED_HEADER;
}

fn stagedScatKernel(k: c_int) !mlx.mlx_fast_metal_kernel {
    if (k < 1 or k > 8) return error.ScatPicks;
    const i: usize = @intCast(k);
    if (staged_scat_kernel[i]) |v| return v;
    const ins = mlx.mlx_vector_string_new_data(&STAGED_SCAT_INPUTS, STAGED_SCAT_INPUTS.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs_names = [_][*:0]const u8{"y"};
    const outs = mlx.mlx_vector_string_new_data(&outs_names, outs_names.len);
    defer _ = mlx.mlx_vector_string_free(outs);
    const header: [*:0]const u8 = switch (k) {
        inline 1...8 => |kk| stagedScatHeader(kk),
        else => unreachable,
    };
    const value = mlx.mlx_fast_metal_kernel_new("msv_gqmm_staged_scat", ins, outs, STAGED_SOURCE, header, true, false);
    if (value.ctx == null) return error.MetalKernelCompileFailed;
    staged_scat_kernel[i] = value;
    return value;
}

/// The staged launch's configuration: `[rows, 1, out_n]` bf16 over the tiles of a `nw`-wide product (2 n for the pair).
pub const StagedKey = struct { rows: c_int, out_n: c_int, nw: c_int, k: c_int, max_tiles: c_int, st: Staged, bits: u32, group: u32, mx: bool, epi: c_int, esc: u32, map: bool };

/// Caller frees the config.
pub fn stagedConfig(key: StagedKey) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const st = key.st;
    const cols = @divExact(key.nw, st.bn);
    const grid_x = if (st.gx > 0) st.gx else cols;
    const grid_y = if (st.gx > 0) @divTrunc(key.max_tiles + st.gx - 1, st.gx) * cols else key.max_tiles;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.rows, 1, key.out_n }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid_x * 32, grid_y * st.wn, st.wm));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, st.wn, st.wm));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    inline for (.{ .{ "GS", @as(c_int, @intCast(key.group)) }, .{ "BITS", @as(c_int, @intCast(key.bits)) }, .{ "MX", @as(c_int, @intFromBool(key.mx)) }, .{ "BM", st.bm }, .{ "WM", st.wm }, .{ "WN", st.wn }, .{ "GX", st.gx }, .{ "NW", key.nw }, .{ "KD", key.k }, .{ "EPI", key.epi }, .{ "ESC", @as(c_int, @intCast(key.esc)) }, .{ "AS", @as(c_int, @intFromBool(st.tg_a)) }, .{ "BN", st.bn }, .{ "BK", st.bk } }) |t|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], t[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "MAP", key.map));
    return cfg;
}

/// `launchMapped` on the staged schedule; `row_map` null reads x's rows in order. `scan_bm`: the tile height the
/// scan cuts at (the plan's; a test plants another).
fn launchStaged(x: mlx.mlx_array, row_map: ?mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, pair: ?Pair, bits: u32, group: u32, mx: bool, st: Staged, scan_bm: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return launchStagedTo(x, row_map, w, sc, bi, idx, pair, bits, group, mx, st, scan_bm, null, s);
}

/// `sortedGatherPromptStagedTok`'s scatter: each sorted row's token `rows` `[m]` and the flat picks `ids` `[m]` (`k` a
/// token).
const Scat = struct { rows: mlx.mlx_array, ids: mlx.mlx_array, k: c_int };

fn launchStagedTo(x: mlx.mlx_array, row_map: ?mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, pair: ?Pair, bits: u32, group: u32, mx: bool, st: Staged, scan_bm: c_int, scat: ?Scat, s: mlx.mlx_stream) !mlx.mlx_array {
    const ws = mlx.getShape(w);
    const m = mlx.getShape(idx)[0];
    const up_off: c_int = if (pair) |v| v.up_off else 0;
    const n = ws[1] - up_off;
    const nw = if (pair != null) 2 * n else n;
    const k = mlx.getShape(x)[2];
    const t = try scanTiles(idx, m, ws[0], scan_bm, s);
    defer t.deinit();
    const route: ?mlx.mlx_array = if (pair) |v| v.route else null;
    const lim: f32 = if (pair) |v| v.route_limit else 0;
    const params = mlx.mlx_array_new_data(&[_]c_int{ nw, k, up_off, @bitCast(lim) }, &[_]c_int{4}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const sigtab = if (pair) |v| v.sigtab else biasArg(.{ .ctx = null });
    const esc: ExpertScales = if (pair) |v| v.scales else .{};
    const pw = if (pair) |v| v.w else w;
    const ps = if (pair) |v| v.sc else sc;
    const pb = if (pair) |v| v.bi else bi;
    // Unread slots bind an array already bound (MAP = false never reads `rmap`; ESC leaves an absent scale unread).
    const inputs = [_]mlx.mlx_array{ x, w, sc, biasArg(bi), pw, ps, biasArg(pb), t.tiles, t.count, params, sigtab, row_map orelse idx, esc.gate orelse sigtab, esc.up orelse sigtab, routeArg(route) };
    const epi: c_int = if (pair) |v| (if (v.route != null) -1 else 1 + @as(c_int, @intCast(v.limit))) else 0;
    const cfg = try stagedConfig(.{ .rows = m, .out_n = n, .nw = nw, .k = k, .max_tiles = t.max_tiles, .st = st, .bits = bits, .group = group, .mx = mx, .epi = epi, .esc = esc.bits(), .map = row_map != null });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    if (scat) |v| {
        // The plain tile only (MAP and EPI unread there): `rmap` carries each sorted row's token.
        if (pair != null or row_map != null) return error.ScatPlainOnly;
        var si = inputs ++ [_]mlx.mlx_array{v.ids};
        si[11] = v.rows;
        return applyOne(try stagedScatKernel(v.k), &si, cfg, s);
    }
    return applyOne(try stagedKernel(), &inputs, cfg, s);
}

/// `sortedGateUpRouted` on the staged schedule (an fp4 bank fused along its rows), the same words; null where it does
/// not serve (`routed_decline` says why).
pub fn sortedGateUpRoutedStaged(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, route: mlx.mlx_array, limit: f32, scales: ExpertScales, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGateUpRoutedTile(x, row_map, w, sc, bi, idx, route, limit, scales, bits, group, mx, nax_available, null, s);
}

/// `sortedGateUpRoutedStaged` at tile `tile` (null: `stagedPlan`'s).
pub fn sortedGateUpRoutedTile(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, route: mlx.mlx_array, limit: f32, scales: ExpertScales, bits: u32, group: u32, mx: bool, nax_available: bool, tile: ?Staged, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!routedFits(x, row_map, w, sc, bi, idx, route, scales, bits, group, mx, nax_available, s)) return null;
    const ws = mlx.getShape(w);
    const st = tile orelse stagedPlan(mlx.getShape(idx)[0], ws[0]);
    if (!stagedFits(mlx.getShape(x)[2], ws[1], mx, st)) return declineRouted("staged: the bank is not fp4 or K is not a multiple of 64");
    return try launchStaged(x, row_map, w, sc, bi, idx, .{ .w = w, .sc = sc, .bi = bi, .sigtab = biasArg(.{ .ctx = null }), .up_off = @divExact(ws[1], 2), .scales = scales, .route = route, .route_limit = limit }, bits, group, mx, st, st.bm, s);
}

/// `sortedGatherPrompt` on the staged schedule (an fp4 bank), the same words; null where it does not serve.
pub fn sortedGatherPromptStaged(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGatherPromptTile(x, w, sc, bi, idx, bits, group, mx, nax_available, null, s);
}

/// `sortedGatherPromptStaged` at tile `tile` (null: `stagedPlan`'s).
pub fn sortedGatherPromptTile(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, tile: ?Staged, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, mx, null, true)) return null;
    const ws = mlx.getShape(w);
    const st = tile orelse stagedPlan(mlx.getShape(idx)[0], ws[0]);
    if (!stagedFits(mlx.getShape(x)[2], ws[1], mx, st)) return null;
    return try launchStaged(x, null, w, sc, bi, idx, null, bits, group, mx, st, st.bm, s);
}

/// `sortedGatherPromptStaged` with each output row at its pair's slot: sorted row p (expert `idx[p]`, token `rows[p]`)
/// lands at row `rows[p] * k + j`, where `ids[rows[p] * k + j]` is that expert (`ids` `[m]` the flat picks, `k` a
/// token, each token's picks distinct). The output `[m, 1, n]` is in pair order, each row the same words as the
/// sorted call's row p; null where the staged schedule does not serve.
pub fn sortedGatherPromptStagedTok(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, rows: mlx.mlx_array, ids: mlx.mlx_array, k: c_int, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    return sortedGatherPromptTokTile(x, w, sc, bi, idx, rows, ids, k, bits, group, mx, nax_available, null, s);
}

/// `sortedGatherPromptStagedTok` at tile `tile` (null: `stagedPlan`'s).
pub fn sortedGatherPromptTokTile(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, rows: mlx.mlx_array, ids: mlx.mlx_array, k: c_int, bits: u32, group: u32, mx: bool, nax_available: bool, tile: ?Staged, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, mx, null, true)) return null;
    const ws = mlx.getShape(w);
    const m = mlx.getShape(idx)[0];
    const st = tile orelse stagedPlan(m, ws[0]);
    if (!stagedFits(mlx.getShape(x)[2], ws[1], mx, st)) return null;
    if (k < 1 or k > 8 or @rem(m, k) != 0) return null;
    for ([_]mlx.mlx_array{ rows, ids }) |a| if (mlx.mlx_array_dtype(a) != .uint32 or mlx.mlx_array_size(a) != @as(usize, @intCast(m))) return null;
    return try launchStagedTo(x, null, w, sc, bi, idx, null, bits, group, mx, st, st.bm, .{ .rows = rows, .ids = ids, .k = k }, s);
}

fn requireNax() !void {
    mlx.installErrorHandler();
    if (mlx.noGpuBackend() or !@import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
}

/// A kernel failure is its test's: name it and drop its latch, or the next test inherits it.
fn dropOwnLatch() void {
    var buf: [512]u8 = undefined;
    if (mlx.takeError(&buf)) |msg| std.debug.print("[gather-nax] mlx: {s}\n", .{msg});
}

test "segmented NAX sorted gather: a failed canary keeps an earlier op's latch and drops its own" {
    try requireNax();
    const s = mlx.gpuStream();
    const key: CanaryKey = .{ .plan = plan(81920, 512, 2560, 640, false), .bits = 4, .group = 64, .align_n = true, .align_k = true, .mx = false };
    // `armed` caches the failed verdict; forget it so later tests run the real canary.
    defer _ = canaries.remove(key);
    defer mlx.armLatchingFaultForTest(0);

    _ = canaries.remove(key);
    mlx.latchErrorForTest("earlier op in this forward");
    mlx.armLatchingFaultForTest(1);
    try std.testing.expect(!armed(key, s));
    var buf: [512]u8 = undefined;
    const msg = mlx.takeError(&buf) orelse return error.EarlierLatchLost;
    try std.testing.expect(std.mem.indexOf(u8, msg, "earlier op") != null);

    _ = canaries.remove(key);
    mlx.armLatchingFaultForTest(1);
    try std.testing.expect(!armed(key, s));
    try std.testing.expect(mlx.latchingFaultFiredForTest());
    try std.testing.expect(!mlx.errorPending());
}

test "segmented NAX sorted gather planner selects measured shape classes" {
    const cases = [_]struct { rows: c_int, experts: c_int, k: c_int, n: c_int, paired: bool = false, want: Plan }{
        .{ .rows = 81920, .experts = 512, .k = 2560, .n = 640, .want = .{ .sched = .db, .bm = 96, .bk = 64, .gx = 32, .pad = 0 } },
        .{ .rows = 81920, .experts = 512, .k = 2560, .n = 640, .paired = true, .want = .{ .sched = .seg, .bm = 128, .bk = 128, .gx = 32, .pad = 8192 } },
        .{ .rows = 65536, .experts = 256, .k = 2048, .n = 4096, .want = .{ .sched = .db, .bm = 96, .bk = 64, .gx = 32, .pad = 0 } },
        .{ .rows = 65536, .experts = 256, .k = 4096, .n = 2048, .paired = true, .want = .{ .sched = .seg, .bm = 128, .bk = 128, .gx = 32, .pad = 8192 } },
        .{ .rows = 81920, .experts = 512, .k = 640, .n = 2560, .want = .{ .sched = .db, .bm = 96, .bk = 64, .gx = 32, .pad = 0 } },
        .{ .rows = 33010, .experts = 512, .k = 2560, .n = 640, .want = .{ .sched = .db, .bm = 96, .bk = 64, .gx = 32, .pad = 0 } },
        .{ .rows = 16384, .experts = 512, .k = 2560, .n = 640, .want = .{ .sched = .db, .bm = 64, .bk = 64, .gx = 0, .pad = 0 } },
        .{ .rows = 129, .experts = 8, .k = 96, .n = 64, .want = .{ .sched = .seg, .bm = 64, .bk = 64, .gx = 0, .pad = 0 } },
    };
    for (cases) |case| try std.testing.expectEqualDeep(case.want, plan(case.rows, case.experts, case.k, case.n, case.paired));
}

test "segmented NAX sorted gather matches MLX on ragged expert runs" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const rows: c_int = 129;
    const experts: c_int = 8;
    const n: c_int = 64;
    const k: c_int = 128;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 0x2267));
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    const w_shape = [_]c_int{ experts, n, k };
    try mlx.check(mlx.mlx_random_normal(&wf, &w_shape, 3, .bfloat16, 0, 0.05, key, s));
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&w, q, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, q, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, q, 2));
    const x_shape = [_]c_int{ rows, 1, k };
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &x_shape, 3, .bfloat16, 0, 0.5, key, s));
    var idx_data: [rows]u32 = undefined;
    for (&idx_data, 0..) |*v, i| v.* = if (i < 70) 0 else if (i < 75) 2 else if (i < 108) 3 else if (i < 125) 5 else 7;
    const idx_shape = [_]c_int{rows};
    const idx = mlx.mlx_array_new_data(&idx_data, &idx_shape, 1, .uint32);
    defer _ = mlx.mlx_array_free(idx);
    const got = (try sortedGather(x, w, sc, bi, idx, 4, 64, false, true, s)) orelse return error.KernelDeclinedCanary;
    defer _ = mlx.mlx_array_free(got);
    const ref = try stockGather(x, w, sc, bi, idx, 4, 64, false, s);
    defer _ = mlx.mlx_array_free(ref);
    try expectBitEqual(got, ref, s);
}

/// A top-k routing's sorted expert runs: 1 in 16 experts empty, 1 in 64 eight
/// times the mean, the rest flat; the remainder lands on the last expert.
fn skewedRouting(ids: []u32, experts: usize) void {
    var weight_sum: usize = 0;
    for (0..experts) |e| weight_sum += if (e % 16 == 5) 0 else if (e % 64 == 0) 8 else 1;
    var pos: usize = 0;
    for (0..experts) |e| {
        const weight: usize = if (e % 16 == 5) 0 else if (e % 64 == 0) 8 else 1;
        for (0..@min(ids.len * weight / weight_sum, ids.len - pos)) |_| {
            ids[pos] = @intCast(e);
            pos += 1;
        }
    }
    while (pos < ids.len) : (pos += 1) ids[pos] = @intCast(experts - 1);
}

fn testCase(rows: c_int, experts: c_int, n: c_int, k: c_int, bits: u32, group: u32, mx: bool, s: mlx.mlx_stream) !void {
    errdefer dropOwnLatch();
    const alloc = std.testing.allocator;
    const ids_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(ids_data);
    if (rows == 81920) {
        var pos: usize = 0;
        for (0..@intCast(experts)) |expert| {
            const count: usize = if (expert < 16) 0 else if (expert < 32) 320 else if (expert < 272) 150 else 170;
            for (0..@min(count, ids_data.len - pos)) |_| {
                ids_data[pos] = @intCast(expert);
                pos += 1;
            }
        }
        while (pos < ids_data.len) : (pos += 1) ids_data[pos] = @intCast(experts - 1);
    } else skewedRouting(ids_data, @intCast(experts));
    const ids_shape = [_]c_int{rows};
    const ids = mlx.mlx_array_new_data(ids_data.ptr, &ids_shape, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const bank = try randomBank(experts, n, k, bits, group, mx, 0x2267, s);
    defer bank.deinit();
    const x_shape = [_]c_int{ rows, 1, k };
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &x_shape, 3, .bfloat16, 0, 0.5, bank.key, s));
    const got = (try sortedGather(x, bank.w, bank.sc, bank.bi, ids, bits, group, mx, true, s)) orelse return error.KernelDeclinedTestShape;
    defer _ = mlx.mlx_array_free(got);
    if (@rem(k, 64) != 0) {
        // Pinned stock NAX has the K-tail bug, so this arm uses fp32 dequantized matmul.
        try std.testing.expect(try floatReferenceClose(got, x, bank.w, bank.sc, bank.bi, ids, bits, group, mx, s));
        return;
    }
    // This pin includes MLX's sorted-row offset fix, so >32768 rows can use stock.
    const ref = try stockGather(x, bank.w, bank.sc, bank.bi, ids, bits, group, mx, s);
    defer _ = mlx.mlx_array_free(ref);
    try expectBitEqual(got, ref, s);
}

const Bank = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    key: mlx.mlx_array,

    fn deinit(self: Bank) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.sc);
        if (self.bi.ctx != null) _ = mlx.mlx_array_free(self.bi);
        _ = mlx.mlx_array_free(self.key);
    }
};

/// Quantized [experts, n, k] expert rows from a seeded normal draw; `key` is the draw's.
fn randomBank(experts: c_int, n: c_int, k: c_int, bits: u32, group: u32, mx: bool, seed: u64, s: mlx.mlx_stream) !Bank {
    var key = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_random_normal(&wf, &[_]c_int{ experts, n, k }, 3, .bfloat16, 0, 0.05, key, s));
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), modeName(mx, group), .{}, s));
    const w = try outputAt(q, 0);
    errdefer _ = mlx.mlx_array_free(w);
    const sc = try outputAt(q, 1);
    errdefer _ = mlx.mlx_array_free(sc);
    const bi: mlx.mlx_array = if (mx) .{ .ctx = null } else try outputAt(q, 2);
    for ([_]mlx.mlx_array{ w, sc, bi }) |a| if (a.ctx != null) try mlx.check(mlx.mlx_array_eval(a));
    return .{ .w = w, .sc = sc, .bi = bi, .key = key };
}

test "segmented NAX sorted gather matches stock across Flash Next and quantization shapes" {
    try requireNax();
    // Real Flash Next / MiMo bank shapes peak past 40 GB of Metal memory: `-Dslow-tests` only.
    if (!@import("build_options").slow_tests) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const cases = [_]struct { rows: c_int, experts: c_int, n: c_int, k: c_int, bits: u32, group: u32 }{
        .{ .rows = 81920, .experts = 512, .n = 640, .k = 2560, .bits = 4, .group = 64 },
        .{ .rows = 81920, .experts = 512, .n = 2560, .k = 640, .bits = 4, .group = 64 },
        .{ .rows = 33010, .experts = 512, .n = 64, .k = 128, .bits = 4, .group = 64 },
        .{ .rows = 2048, .experts = 32, .n = 64, .k = 128, .bits = 8, .group = 32 },
        .{ .rows = 2048, .experts = 32, .n = 96, .k = 256, .bits = 4, .group = 128 },
        .{ .rows = 4096, .experts = 32, .n = 96, .k = 96, .bits = 4, .group = 32 },
        // DeepSeek-V4.1-Flash iQ-MLX packs: 2/3-bit gate+up and 3-bit down banks at a 2048-row MoE call.
        .{ .rows = 12288, .experts = 384, .n = 4608, .k = 5120, .bits = 3, .group = 64 },
        .{ .rows = 12288, .experts = 384, .n = 4608, .k = 5120, .bits = 2, .group = 64 },
        .{ .rows = 12288, .experts = 384, .n = 5120, .k = 2304, .bits = 3, .group = 64 },
        .{ .rows = 2048, .experts = 32, .n = 96, .k = 160, .bits = 3, .group = 32 },
        .{ .rows = 2048, .experts = 32, .n = 96, .k = 160, .bits = 2, .group = 32 },
    };
    for (cases) |case| try testCase(case.rows, case.experts, case.n, case.k, case.bits, case.group, false, s);
}

test "segmented NAX sorted gather matches stock bit for bit on FP4 banks (MXFP4 MiMo shapes, NVFP4 Flash Next shapes)" {
    try requireNax();
    // Real Flash Next / MiMo bank shapes peak past 40 GB of Metal memory: `-Dslow-tests` only.
    if (!@import("build_options").slow_tests) return error.SkipZigTest;
    const s = mlx.gpuStream();
    // 65536 rows = an 8192-token top-8 chunk, past the 32768-row offset the stock kernel once overflowed.
    const cases = [_]struct { rows: c_int, experts: c_int, n: c_int, k: c_int, group: u32 }{
        .{ .rows = 65536, .experts = 256, .n = 2048, .k = 4096, .group = 32 },
        .{ .rows = 65536, .experts = 256, .n = 4096, .k = 2048, .group = 32 },
        .{ .rows = 2048, .experts = 32, .n = 64, .k = 128, .group = 32 },
        .{ .rows = 81920, .experts = 512, .n = 640, .k = 2560, .group = 16 },
        .{ .rows = 81920, .experts = 512, .n = 2560, .k = 640, .group = 16 },
        .{ .rows = 2048, .experts = 32, .n = 64, .k = 128, .group = 16 },
        .{ .rows = 4096, .experts = 32, .n = 96, .k = 96, .group = 16 },
    };
    for (cases) |c| try testCase(c.rows, c.experts, c.n, c.k, 4, c.group, true, s);
}

test "segmented NAX sorted gather row map and paired SwiGLU match composed projections" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const cases = [_]struct { tokens: c_int, bits: u32, group: u32, mx: bool = false }{
        .{ .tokens = 8192, .bits = 4, .group = 64 },
        .{ .tokens = 3301, .bits = 4, .group = 64 },
        .{ .tokens = 3301, .bits = 8, .group = 32 },
        .{ .tokens = 3301, .bits = 4, .group = 32, .mx = true },
        .{ .tokens = 3301, .bits = 4, .group = 16, .mx = true },
    };
    for (cases) |case| {
        const rows = case.tokens * 10;
        const experts: c_int = 512;
        const n: c_int = 640;
        const k: c_int = 2560;
        const alloc = std.testing.allocator;
        const ids_data = try alloc.alloc(u32, @intCast(rows));
        defer alloc.free(ids_data);
        const map_data = try alloc.alloc(u32, @intCast(rows));
        defer alloc.free(map_data);
        for (ids_data, map_data, 0..) |*id, *mapped, i| {
            const e = @min(@as(usize, @intCast(experts - 1)), (i * i / @as(usize, @intCast(rows))) * @as(usize, @intCast(experts)) / @as(usize, @intCast(rows)));
            id.* = @intCast(e);
            mapped.* = @intCast((i * 73 + 11) % @as(usize, @intCast(case.tokens)));
        }
        const ids = mlx.mlx_array_new_data(ids_data.ptr, &[_]c_int{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const row_map = mlx.mlx_array_new_data(map_data.ptr, &[_]c_int{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(row_map);
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, 0x2267));
        var up_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_key);
        try mlx.check(mlx.mlx_random_key(&up_key, 0x4521));
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ case.tokens, 1, k }, 3, .bfloat16, 0, 0.5, key, s));
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_take_axis(&x_rep, x, row_map, 0, s));
        var gate_f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_f);
        var up_f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_f);
        const wshape = [_]c_int{ experts, n, k };
        try mlx.check(mlx.mlx_random_normal(&gate_f, &wshape, 3, .bfloat16, 0, 0.05, key, s));
        try mlx.check(mlx.mlx_random_normal(&up_f, &wshape, 3, .bfloat16, 0, 0.05, up_key, s));
        var gate_q = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(gate_q);
        var up_q = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(up_q);
        const group = mlx.mlx_optional_int.some(@intCast(case.group));
        const bits = mlx.mlx_optional_int.some(@intCast(case.bits));
        try mlx.check(mlx.mlx_quantize(&gate_q, gate_f, group, bits, modeName(case.mx, case.group), .{}, s));
        try mlx.check(mlx.mlx_quantize(&up_q, up_f, group, bits, modeName(case.mx, case.group), .{}, s));
        const gw = try outputAt(gate_q, 0);
        defer _ = mlx.mlx_array_free(gw);
        const gs = try outputAt(gate_q, 1);
        defer _ = mlx.mlx_array_free(gs);
        const gb: mlx.mlx_array = if (case.mx) .{ .ctx = null } else try outputAt(gate_q, 2);
        defer if (gb.ctx != null) {
            _ = mlx.mlx_array_free(gb);
        };
        const uw = try outputAt(up_q, 0);
        defer _ = mlx.mlx_array_free(uw);
        const us = try outputAt(up_q, 1);
        defer _ = mlx.mlx_array_free(us);
        const ub: mlx.mlx_array = if (case.mx) .{ .ctx = null } else try outputAt(up_q, 2);
        defer if (ub.ctx != null) {
            _ = mlx.mlx_array_free(ub);
        };
        const plain_g = (try sortedGather(x_rep, gw, gs, gb, ids, case.bits, case.group, case.mx, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(plain_g);
        const mapped_g = (try sortedGatherMapped(x, row_map, gw, gs, gb, ids, case.bits, case.group, case.mx, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(mapped_g);
        try expectBitEqual(plain_g, mapped_g, s);
        const plain_u = (try sortedGather(x_rep, uw, us, ub, ids, case.bits, case.group, case.mx, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(plain_u);
        const sigtab = try @import("../hc/hc_prefill.zig").sigmoidTable(s);
        const fused = (try sortedGateUp(x, row_map, gw, gs, gb, uw, us, ub, ids, sigtab, 0, case.bits, case.group, case.mx, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(fused);
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_sigmoid(&sig, plain_g, s));
        try mlx.check(mlx.mlx_multiply(&act, plain_g, sig, s));
        try mlx.check(mlx.mlx_multiply(&ref, act, plain_u, s));
        try expectBitEqual(ref, fused, s);
        // A swiglu_limit of 1 clamps these magnitudes: the epilogue equals the clamp ops on the split gathers.
        const clamped = (try sortedGateUp(x, row_map, gw, gs, gb, uw, us, ub, ids, sigtab, 1, case.bits, case.group, case.mx, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(clamped);
        const clamp_ref = try @import("../mlp/activations.zig").clampedSwiGLU(s, plain_g, plain_u, 1.0);
        defer _ = mlx.mlx_array_free(clamp_ref);
        try expectBitEqual(clamp_ref, clamped, s);
    }
}

test "segmented NAX sorted gather: a gate/up bank fused along its rows gives the paired kernel's bits" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const alloc = std.testing.allocator;
    const tokens: c_int = 700;
    const rows = tokens * 6;
    const experts: c_int = 32;
    const n: c_int = 256;
    const k: c_int = 1024;
    const ids_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(ids_data);
    skewedRouting(ids_data, @intCast(experts));
    const map_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(map_data);
    for (map_data, 0..) |*v, i| v.* = @intCast((i * 73 + 11) % @as(usize, @intCast(tokens)));
    const ids = mlx.mlx_array_new_data(ids_data.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const row_map = mlx.mlx_array_new_data(map_data.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(row_map);
    for ([_]u32{ 2, 3 }) |bits| {
        const gate = try randomBank(experts, n, k, bits, 64, false, 0x2267, s);
        defer gate.deinit();
        const up = try randomBank(experts, n, k, bits, 64, false, 0x4521, s);
        defer up.deinit();
        var fused: [3]mlx.mlx_array = undefined;
        for ([_][2]mlx.mlx_array{ .{ gate.w, up.w }, .{ gate.sc, up.sc }, .{ gate.bi, up.bi } }, &fused) |pair, *o| {
            const v = mlx.mlx_vector_array_new_data(&pair, 2);
            defer _ = mlx.mlx_vector_array_free(v);
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(o, v, 1, s));
            try mlx.check(mlx.mlx_array_eval(o.*));
        }
        defer for (fused) |f| {
            _ = mlx.mlx_array_free(f);
        };
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ tokens, 1, k }, 3, .bfloat16, 0, 0.5, gate.key, s));
        const sigtab = try @import("../hc/hc_prefill.zig").sigmoidTable(s);
        for ([_]u32{ 0, 10 }) |limit| {
            const want = (try sortedGateUp(x, row_map, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, ids, sigtab, limit, bits, 64, false, true, s)) orelse return error.KernelDeclinedTestShape;
            defer _ = mlx.mlx_array_free(want);
            const got = (try sortedGateUpFused(x, row_map, fused[0], fused[1], fused[2], ids, sigtab, limit, bits, 64, false, true, s)) orelse return error.KernelDeclinedTestShape;
            defer _ = mlx.mlx_array_free(got);
            try expectBitEqual(want, got, s);
        }
    }
}

/// The paired kernel with per-expert scales against the split path it replaces (`sortedDown`'s arm without it): the two
/// mapped gathers, each times its expert's scale (`take` -> multiply, `scaleExperts`), then MLX's SwiGLU chain and the
/// served `fusedSwiGLU` (unclamped: the gate's scale alone, then both), or `clampedSwiGLU` under a limit (both scales).
fn scaledPairCase(tokens: c_int, experts: c_int, top_k: c_int, n: c_int, k: c_int, group: u32, mx: bool, s: mlx.mlx_stream) !void {
    errdefer dropOwnLatch();
    const alloc = std.testing.allocator;
    const rows = tokens * top_k;
    const ids_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(ids_data);
    skewedRouting(ids_data, @intCast(experts));
    const map_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(map_data);
    for (map_data, 0..) |*v, i| v.* = @intCast((i * 73 + 11) % @as(usize, @intCast(tokens)));
    const ids = mlx.mlx_array_new_data(ids_data.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const row_map = mlx.mlx_array_new_data(map_data.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(row_map);
    const gate = try randomBank(experts, n, k, 4, group, mx, 0x2267, s);
    defer gate.deinit();
    const up = try randomBank(experts, n, k, 4, group, mx, 0x4521, s);
    defer up.deinit();
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ tokens, 1, k }, 3, .bfloat16, 0, 0.5, gate.key, s));
    // Per-expert scales in x's dtype over two octaves either side of 1: the multiply's rounding and the clamp both act.
    var esc: [2]mlx.mlx_array = undefined;
    for (&esc, [_]u64{ 0x5CA1E, 0x5CA1F }) |*o, seed| {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        const lo = mlx.mlx_array_new_float(0.25);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_float(4.0);
        defer _ = mlx.mlx_array_free(hi);
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_uniform(o, lo, hi, &[_]c_int{experts}, 1, .bfloat16, key, s));
        try mlx.check(mlx.mlx_array_eval(o.*));
    }
    defer for (esc) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const bias = mlx.mlx_array{ .ctx = null };
    const mapped_g = (try sortedGatherMapped(x, row_map, gate.w, gate.sc, bias, ids, 4, group, mx, true, s)) orelse return error.KernelDeclinedTestShape;
    defer _ = mlx.mlx_array_free(mapped_g);
    const mapped_u = (try sortedGatherMapped(x, row_map, up.w, up.sc, bias, ids, 4, group, mx, true, s)) orelse return error.KernelDeclinedTestShape;
    defer _ = mlx.mlx_array_free(mapped_u);
    var scaled: [2]mlx.mlx_array = undefined;
    for (&scaled, [_]mlx.mlx_array{ mapped_g, mapped_u }, esc) |*o, y, sc| {
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, sc, ids, 0, s));
        var col = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col);
        try mlx.check(mlx.mlx_reshape(&col, picked, &[_]c_int{ rows, 1, 1 }, 3, s));
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(o, y, col, s));
    }
    defer for (scaled) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const sigtab = try @import("../hc/hc_prefill.zig").sigmoidTable(s);
    // The gate's scale alone (ModelOpt unclamped: up x down ride the scores), then both scales.
    for ([_]bool{ false, true }) |with_up| {
        const fused = (try sortedGateUpScaled(x, row_map, gate.w, gate.sc, bias, up.w, up.sc, bias, ids, sigtab, 0, .{ .gate = esc[0], .up = if (with_up) esc[1] else null }, 4, group, mx, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(fused);
        const u = if (with_up) scaled[1] else mapped_u;
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_sigmoid(&sig, scaled[0], s));
        try mlx.check(mlx.mlx_multiply(&act, scaled[0], sig, s));
        try mlx.check(mlx.mlx_multiply(&ref, act, u, s));
        try expectBitEqual(ref, fused, s);
        const served = (try @import("../mlp/fused_activations.zig").fusedSwiGLU(s, scaled[0], u)) orelse return error.FusedSwiGLUDeclined;
        defer _ = mlx.mlx_array_free(served);
        try expectBitEqual(served, fused, s);
    }
    // A swiglu_limit of 1 after both scales: the clamp ops on the scaled gathers.
    const clamped = (try sortedGateUpScaled(x, row_map, gate.w, gate.sc, bias, up.w, up.sc, bias, ids, sigtab, 1, .{ .gate = esc[0], .up = esc[1] }, 4, group, mx, true, s)) orelse return error.KernelDeclinedTestShape;
    defer _ = mlx.mlx_array_free(clamped);
    const clamp_ref = try @import("../mlp/activations.zig").clampedSwiGLU(s, scaled[0], scaled[1], 1.0);
    defer _ = mlx.mlx_array_free(clamp_ref);
    try expectBitEqual(clamp_ref, clamped, s);
    // Without scales the scaled entry is `sortedGateUp`.
    const plain = (try sortedGateUp(x, row_map, gate.w, gate.sc, bias, up.w, up.sc, bias, ids, sigtab, 0, 4, group, mx, true, s)) orelse return error.KernelDeclinedTestShape;
    defer _ = mlx.mlx_array_free(plain);
    const unscaled = (try sortedGateUpScaled(x, row_map, gate.w, gate.sc, bias, up.w, up.sc, bias, ids, sigtab, 0, .{}, 4, group, mx, true, s)) orelse return error.KernelDeclinedTestShape;
    defer _ = mlx.mlx_array_free(unscaled);
    try expectBitEqual(plain, unscaled, s);
}

test "segmented NAX paired SwiGLU with per-expert scales matches the split path (NVFP4 GLM-5.3 Pro and Flash Next shapes, MXFP4)" {
    try requireNax();
    const s = mlx.gpuStream();
    // GLM-5.3 Pro (glm_moe_dsa): 256 experts, top-8, gate/up 6144 -> 2048, NVFP4.
    try scaledPairCase(2048, 256, 8, 2048, 6144, 16, true, s);
    // Qwen3.8 Flash Next (ModelOpt NVFP4): 512 experts, top-10, 2560 -> 640 (the paired test's shape).
    try scaledPairCase(3301, 512, 10, 640, 2560, 16, true, s);
    // The same geometry on MXFP4 banks: the epilogue is format-blind.
    try scaledPairCase(3301, 512, 10, 640, 2560, 32, true, s);
}

/// `a` and `b` equal element for element, a NaN equal to a NaN.
fn expectBitEqualNan(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, true, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try std.testing.expect(same);
}

test "segmented NAX gathers on every fp4 code: all 16 e2m1 codes, every e4m3 / e8m0 scale code (NaNs, subnormals, 2^-127..2^127), NaN rows, the clamp on NaN" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const experts: c_int = 8;
    const n: c_int = 64;
    const k: c_int = 512;
    const tokens: c_int = 96;
    const rows = tokens * 4;
    const alloc = std.testing.allocator;
    const ids_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(ids_data);
    for (ids_data, 0..) |*v, i| v.* = @intCast(i * @as(usize, @intCast(experts)) / @as(usize, @intCast(rows)));
    const ids = mlx.mlx_array_new_data(ids_data.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const map_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(map_data);
    for (map_data, 0..) |*v, i| v.* = @intCast((i * 37 + 5) % @as(usize, @intCast(tokens)));
    const row_map = mlx.mlx_array_new_data(map_data.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(row_map);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 0xC0DE));
    var xn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xn);
    try mlx.check(mlx.mlx_random_normal(&xn, &[_]c_int{ tokens * k }, 1, .bfloat16, 0, 0.5, key, s));
    try mlx.check(mlx.mlx_array_eval(xn));
    // Token 3 is all NaN, token 7 has one NaN: NaN must reach every output it feeds.
    const xh = try alloc.alloc(u16, @intCast(tokens * k));
    defer alloc.free(xh);
    @memcpy(xh, @as([*]const u16, @ptrCast(mlx.mlx_array_data_bfloat16(xn) orelse return error.NoData))[0..xh.len]);
    for (xh[@intCast(3 * k)..][0..@intCast(k)]) |*v| v.* = 0x7FC0;
    xh[@intCast(7 * k + 100)] = 0x7FC0;
    const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ tokens, 1, k }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    var x_rep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_rep);
    try mlx.check(mlx.mlx_take_axis(&x_rep, x, row_map, 0, s));
    const sigtab = try @import("../hc/hc_prefill.zig").sigmoidTable(s);
    for ([_]u32{ 16, 32 }) |group| {
        // Every word cycles the 16 e2m1 codes (-0 included); the scale bytes cycle all 256 codes, the up bank's
        // shifted so gate and up meet different codes.
        const words = try alloc.alloc(u32, @intCast(experts * n * @divExact(k, 8)));
        defer alloc.free(words);
        for (words, 0..) |*w, i| w.* = @as(u32, 0x76543210) +% @as(u32, @intCast(i % 2)) *% 0x88888888;
        const nsc: usize = @intCast(experts * n * @divExact(k, @as(c_int, @intCast(group))));
        var banks: [2][2]mlx.mlx_array = undefined;
        for (&banks, 0..) |*b, which| {
            const codes = try alloc.alloc(u8, nsc);
            defer alloc.free(codes);
            for (codes, 0..) |*c, i| c.* = @intCast((i * 7 + which * 101) % 256);
            b.* = .{
                mlx.mlx_array_new_data(words.ptr, &[_]c_int{ experts, n, @divExact(k, 8) }, 3, .uint32),
                mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ experts, n, @divExact(k, @as(c_int, @intCast(group))) }, 3, .uint8),
            };
        }
        defer for (banks) |b| for (b) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const none = mlx.mlx_array{ .ctx = null };
        for (banks) |b| {
            const got = (try sortedGatherPrompt(x_rep, b[0], b[1], none, ids, 4, group, true, true, s)) orelse return error.KernelDeclinedTestShape;
            defer _ = mlx.mlx_array_free(got);
            const ref = try stockGather(x_rep, b[0], b[1], none, ids, 4, group, true, s);
            defer _ = mlx.mlx_array_free(ref);
            try expectBitEqualNan(ref, got, s);
        }
        // The paired kernel, clamped at 1 after per-expert scales (one NaN among them would be refused at load, so
        // finite ones here), against the split path on the same banks: a NaN gate or up stays NaN through the clamp.
        var esc: [2]mlx.mlx_array = undefined;
        for (&esc, [_]f32{ 0.75, 1.5 }) |*o, v| {
            var vals: [8]u16 = undefined;
            for (&vals, 0..) |*h, i| h.* = @truncate(@as(u32, @bitCast(v * @as(f32, @floatFromInt(i + 1)))) >> 16);
            o.* = mlx.mlx_array_new_data(&vals, &[_]c_int{experts}, 1, .bfloat16);
        }
        defer for (esc) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const mg = (try sortedGatherMapped(x, row_map, banks[0][0], banks[0][1], none, ids, 4, group, true, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(mg);
        const mu = (try sortedGatherMapped(x, row_map, banks[1][0], banks[1][1], none, ids, 4, group, true, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(mu);
        var scaled: [2]mlx.mlx_array = undefined;
        for (&scaled, [_]mlx.mlx_array{ mg, mu }, esc) |*o, y, sc| {
            var picked = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(picked);
            try mlx.check(mlx.mlx_take_axis(&picked, sc, ids, 0, s));
            var col = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(col);
            try mlx.check(mlx.mlx_reshape(&col, picked, &[_]c_int{ rows, 1, 1 }, 3, s));
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(o, y, col, s));
        }
        defer for (scaled) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for ([_]u32{ 0, 1 }) |limit| {
            const fused = (try sortedGateUpScaled(x, row_map, banks[0][0], banks[0][1], none, banks[1][0], banks[1][1], none, ids, sigtab, limit, .{ .gate = esc[0], .up = esc[1] }, 4, group, true, true, s)) orelse return error.KernelDeclinedTestShape;
            defer _ = mlx.mlx_array_free(fused);
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            if (limit > 0) {
                _ = mlx.mlx_array_free(ref);
                ref = try @import("../mlp/activations.zig").clampedSwiGLU(s, scaled[0], scaled[1], 1.0);
            } else {
                var sig = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(sig);
                var act = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(act);
                try mlx.check(mlx.mlx_sigmoid(&sig, scaled[0], s));
                try mlx.check(mlx.mlx_multiply(&act, scaled[0], sig, s));
                try mlx.check(mlx.mlx_multiply(&ref, act, scaled[1], s));
            }
            try expectBitEqualNan(ref, fused, s);
        }
    }
}

test "segmented NAX paired SwiGLU refuses scales it cannot bind" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const experts: c_int = 32;
    const tokens: c_int = 256;
    const rows = tokens * 4;
    const alloc = std.testing.allocator;
    const ids_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(ids_data);
    skewedRouting(ids_data, @intCast(experts));
    const ids = mlx.mlx_array_new_data(ids_data.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const map_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(map_data);
    for (map_data, 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(tokens)));
    const row_map = mlx.mlx_array_new_data(map_data.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(row_map);
    const gate = try randomBank(experts, 128, 512, 4, 16, true, 0x2267, s);
    defer gate.deinit();
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ tokens, 1, 512 }, 3, .bfloat16, 0, 0.5, gate.key, s));
    const sigtab = try @import("../hc/hc_prefill.zig").sigmoidTable(s);
    const bias = mlx.mlx_array{ .ctx = null };
    var f32_scale = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_scale);
    try mlx.check(mlx.mlx_ones(&f32_scale, &[_]c_int{experts}, 1, .float32, s));
    var short = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(short);
    try mlx.check(mlx.mlx_ones(&short, &[_]c_int{experts - 1}, 1, .bfloat16, s));
    // A scale not in x's dtype, or not one per expert, declines (the caller keeps the split path).
    for ([_]mlx.mlx_array{ f32_scale, short }) |bad| {
        try std.testing.expect((try sortedGateUpScaled(x, row_map, gate.w, gate.sc, bias, gate.w, gate.sc, bias, ids, sigtab, 0, .{ .gate = bad }, 4, 16, true, true, s)) == null);
        try std.testing.expect((try sortedGateUpScaled(x, row_map, gate.w, gate.sc, bias, gate.w, gate.sc, bias, ids, sigtab, 0, .{ .up = bad }, 4, 16, true, true, s)) == null);
    }
}

test "segmented NAX sorted gather: weighted paired SwiGLU matches unweighted times row_wt bit for bit" {
    try requireNax();
    if (mlx.virtualGpu()) return error.SkipZigTest;
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const alloc = std.testing.allocator;
    const sigtab = try @import("../hc/hc_prefill.zig").sigmoidTable(s);
    const Case = struct { tokens: c_int, experts: c_int, top_k: c_int, bits: u32, group: u32, mx: bool, limit: u32 };
    const cases = [_]Case{
        .{ .tokens = 700, .experts = 32, .top_k = 6, .bits = 2, .group = 64, .mx = false, .limit = 10 },
        .{ .tokens = 700, .experts = 32, .top_k = 6, .bits = 3, .group = 64, .mx = false, .limit = 10 },
        .{ .tokens = 320, .experts = 32, .top_k = 6, .bits = 4, .group = 64, .mx = false, .limit = 0 },
        .{ .tokens = 320, .experts = 32, .top_k = 6, .bits = 8, .group = 32, .mx = false, .limit = 0 },
        .{ .tokens = 320, .experts = 32, .top_k = 6, .bits = 4, .group = 32, .mx = true, .limit = 0 },
        .{ .tokens = 320, .experts = 32, .top_k = 6, .bits = 4, .group = 16, .mx = true, .limit = 0 },
    };
    for (cases) |c| {
        const rows = c.tokens * c.top_k;
        const n: c_int = 128;
        const k: c_int = 1024;
        const ids_data = try alloc.alloc(u32, @intCast(rows));
        defer alloc.free(ids_data);
        skewedRouting(ids_data, @intCast(c.experts));
        const map_data = try alloc.alloc(u32, @intCast(rows));
        defer alloc.free(map_data);
        const wt_data = try alloc.alloc(f32, @intCast(rows));
        defer alloc.free(wt_data);
        for (map_data, wt_data, 0..) |*m, *w, i| {
            m.* = @intCast((i * 73 + 11) % @as(usize, @intCast(c.tokens)));
            w.* = 0.05 + @as(f32, @floatFromInt((i * 37 + 7) % 101)) * 0.025;
        }
        const ids = mlx.mlx_array_new_data(ids_data.ptr, &[_]c_int{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const row_map = mlx.mlx_array_new_data(map_data.ptr, &[_]c_int{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(row_map);
        const row_wt = mlx.mlx_array_new_data(wt_data.ptr, &[_]c_int{rows}, 1, .float32);
        defer _ = mlx.mlx_array_free(row_wt);
        const gate = try randomBank(c.experts, n, k, c.bits, c.group, c.mx, 0x2267, s);
        defer gate.deinit();
        const up = try randomBank(c.experts, n, k, c.bits, c.group, c.mx, 0x4521, s);
        defer up.deinit();
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ c.tokens, 1, k }, 3, .bfloat16, 0, 0.5, gate.key, s));

        const base = (try sortedGateUp(x, row_map, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, ids, sigtab, c.limit, c.bits, c.group, c.mx, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(base);
        var wt3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt3);
        try mlx.check(mlx.mlx_reshape(&wt3, row_wt, &[_]c_int{ rows, 1, 1 }, 3, s));
        var ref_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_f32);
        try mlx.check(mlx.mlx_multiply(&ref_f32, base, wt3, s));
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_astype(&want, ref_f32, .bfloat16, s));

        const got_split = (try sortedGateUpWeighted(x, row_map, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, ids, sigtab, row_wt, c.limit, c.bits, c.group, c.mx, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(got_split);
        try expectBitEqual(want, got_split, s);

        var fused: [3]mlx.mlx_array = undefined;
        for ([_][2]mlx.mlx_array{ .{ gate.w, up.w }, .{ gate.sc, up.sc }, .{ gate.bi, up.bi } }, &fused, 0..) |pair, *o, fi| {
            if (pair[0].ctx == null) {
                o.* = .{ .ctx = null };
                continue;
            }
            _ = fi;
            const v = mlx.mlx_vector_array_new_data(&pair, 2);
            defer _ = mlx.mlx_vector_array_free(v);
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(o, v, 1, s));
            try mlx.check(mlx.mlx_array_eval(o.*));
        }
        defer for (fused) |f| if (f.ctx != null) {
            _ = mlx.mlx_array_free(f);
        };
        const got_fused = (try sortedGateUpFusedWeighted(x, row_map, fused[0], fused[1], fused[2], ids, sigtab, row_wt, c.limit, c.bits, c.group, c.mx, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(got_fused);
        try expectBitEqual(want, got_fused, s);
    }
}

/// `a` and `b` interleaved per rep after one warm run each: per-arm medians of the wall
/// time of the build plus the eval of what it returns.
fn abMedianMs(reps: comptime_int, ctx: BenchArm, a: BenchBuild, b: BenchBuild) ![2]f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var laps: [2][reps]u64 = undefined;
    for (0..reps + 1) |i| {
        for ([_]BenchBuild{ a, b }, 0..) |build, arm| {
            const mark = std.Io.Timestamp.now(io, .boot);
            const out = try build(ctx);
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
            if (i > 0) laps[arm][i - 1] = @intCast(mark.untilNow(io, .boot).nanoseconds);
        }
    }
    var out: [2]f64 = undefined;
    for (&laps, &out) |*l, *o| {
        std.mem.sort(u64, l, {}, std.sort.asc(u64));
        o.* = @as(f64, @floatFromInt(l[reps / 2])) / 1e6;
    }
    return out;
}

const BenchBuild = *const fn (BenchArm) anyerror!mlx.mlx_array;
const BenchArm = struct { x: mlx.mlx_array, bank: Bank, up: ?Bank = null, row_map: mlx.mlx_array = .{ .ctx = null }, sigtab: mlx.mlx_array = .{ .ctx = null }, ids: mlx.mlx_array, plan: Plan, bits: u32, group: u32, mx: bool, s: mlx.mlx_stream };

fn benchStock(c: BenchArm) anyerror!mlx.mlx_array {
    return stockGather(c.x, c.bank.w, c.bank.sc, c.bank.bi, c.ids, c.bits, c.group, c.mx, c.s);
}

fn benchPort(c: BenchArm) anyerror!mlx.mlx_array {
    return launch(c.x, c.bank.w, c.bank.sc, c.bank.bi, c.ids, c.bits, c.group, c.mx, c.plan, c.s);
}

/// The live gate/up call as stock runs it: gather the token rows, two projections, the SwiGLU.
fn benchStockGateUp(c: BenchArm) anyerror!mlx.mlx_array {
    var x_rep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_rep);
    try mlx.check(mlx.mlx_take_axis(&x_rep, c.x, c.row_map, 0, c.s));
    const g = try stockGather(x_rep, c.bank.w, c.bank.sc, c.bank.bi, c.ids, c.bits, c.group, c.mx, c.s);
    defer _ = mlx.mlx_array_free(g);
    const u = try stockGather(x_rep, c.up.?.w, c.up.?.sc, c.up.?.bi, c.ids, c.bits, c.group, c.mx, c.s);
    defer _ = mlx.mlx_array_free(u);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, c.s));
    try mlx.check(mlx.mlx_multiply(&act, g, sig, c.s));
    try mlx.check(mlx.mlx_multiply(&out, act, u, c.s));
    return out;
}

fn benchPortGateUp(c: BenchArm) anyerror!mlx.mlx_array {
    return launchMapped(c.x, c.row_map, c.bank.w, c.bank.sc, c.bank.bi, c.ids, .{ .w = c.up.?.w, .sc = c.up.?.sc, .bi = c.up.?.bi, .sigtab = c.sigtab }, c.bits, c.group, c.mx, c.plan, c.s);
}

/// `=sweep` times every tile configuration instead of the planner's.
fn benchPlans(buf: *[32]Plan, rows: c_int, experts: c_int, k: c_int, n: c_int, paired: bool) []const Plan {
    const raw = env.get(.GQMM_UBENCH) orelse return buf[0..0];
    if (!std.mem.eql(u8, std.mem.sliceTo(raw, 0), "sweep")) {
        buf[0] = plan(rows, experts, k, n, paired);
        return buf[0..1];
    }
    var count: usize = 0;
    for ([_]c_int{ 64, 96, 128 }) |bm| for ([_]c_int{ 0, 32 }) |gx| {
        buf[count] = .{ .sched = .db, .bm = bm, .bk = 64, .gx = gx, .pad = 0 };
        count += 1;
        for ([_]c_int{ 64, 128 }) |bk| for ([_]c_int{ 0, 8192 }) |pad| {
            if (pad != 0 and bm != 128) continue;
            buf[count] = .{ .sched = .seg, .bm = bm, .bk = bk, .gx = gx, .pad = pad };
            count += 1;
        };
    };
    return buf[0..count];
}

test "segmented NAX sorted gather µbench vs stock at 8192-token chunks (diag gqmm-ubench, gqmm-ubench=sweep)" {
    if (env.get(.GQMM_UBENCH) == null) return error.SkipZigTest;
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const tokens: c_int = 8192;
    const alloc = std.testing.allocator;
    const sigtab = try @import("../hc/hc_prefill.zig").sigmoidTable(s);
    // MiMo-V2.6 (MXFP4, top-8) and Qwen3.8-Flash-Next (4-bit g64 and NVIDIA's NVFP4, top-10) expert shapes.
    const Shape = struct { name: []const u8, experts: c_int, top_k: c_int, n: c_int, k: c_int, bits: u32, group: u32, mx: bool, paired: bool = false };
    const shapes = [_]Shape{
        .{ .name = "mimo gate 4096->2048", .experts = 256, .top_k = 8, .n = 2048, .k = 4096, .bits = 4, .group = 32, .mx = true },
        .{ .name = "mimo down 2048->4096", .experts = 256, .top_k = 8, .n = 4096, .k = 2048, .bits = 4, .group = 32, .mx = true },
        .{ .name = "mimo gate+up+SwiGLU", .experts = 256, .top_k = 8, .n = 2048, .k = 4096, .bits = 4, .group = 32, .mx = true, .paired = true },
        .{ .name = "flash-next gate 2560->640", .experts = 512, .top_k = 10, .n = 640, .k = 2560, .bits = 4, .group = 64, .mx = false },
        .{ .name = "flash-next gate+up+SwiGLU", .experts = 512, .top_k = 10, .n = 640, .k = 2560, .bits = 4, .group = 64, .mx = false, .paired = true },
        .{ .name = "flash-next down 640->2560", .experts = 512, .top_k = 10, .n = 2560, .k = 640, .bits = 4, .group = 64, .mx = false },
        .{ .name = "flash-next nvfp4 gate 2560->640", .experts = 512, .top_k = 10, .n = 640, .k = 2560, .bits = 4, .group = 16, .mx = true },
        .{ .name = "flash-next nvfp4 gate+up+SwiGLU", .experts = 512, .top_k = 10, .n = 640, .k = 2560, .bits = 4, .group = 16, .mx = true, .paired = true },
        .{ .name = "flash-next nvfp4 down 640->2560", .experts = 512, .top_k = 10, .n = 2560, .k = 640, .bits = 4, .group = 16, .mx = true },
        .{ .name = "dsv41 gate+up 5120->4608 2-bit", .experts = 384, .top_k = 6, .n = 4608, .k = 5120, .bits = 2, .group = 64, .mx = false },
        .{ .name = "dsv41 gate+up 5120->4608 3-bit", .experts = 384, .top_k = 6, .n = 4608, .k = 5120, .bits = 3, .group = 64, .mx = false },
        .{ .name = "dsv41 down 2304->5120 3-bit", .experts = 384, .top_k = 6, .n = 5120, .k = 2304, .bits = 3, .group = 64, .mx = false },
        .{ .name = "dsv41 gate+up+SwiGLU 2-bit", .experts = 384, .top_k = 6, .n = 2304, .k = 5120, .bits = 2, .group = 64, .mx = false, .paired = true },
    };
    for (shapes) |shape| {
        const rows = tokens * shape.top_k;
        const ids_data = try alloc.alloc(u32, @intCast(rows));
        defer alloc.free(ids_data);
        skewedRouting(ids_data, @intCast(shape.experts));
        const ids = mlx.mlx_array_new_data(ids_data.ptr, &[_]c_int{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const map_data = try alloc.alloc(u32, @intCast(rows));
        defer alloc.free(map_data);
        for (map_data, 0..) |*m, i| m.* = @intCast((i * 73 + 11) % @as(usize, @intCast(tokens)));
        const row_map = mlx.mlx_array_new_data(map_data.ptr, &[_]c_int{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(row_map);
        const bank = try randomBank(shape.experts, shape.n, shape.k, shape.bits, shape.group, shape.mx, 0x2267, s);
        defer bank.deinit();
        const up: ?Bank = if (shape.paired) try randomBank(shape.experts, shape.n, shape.k, shape.bits, shape.group, shape.mx, 0x4521, s) else null;
        defer if (up) |u| u.deinit();
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ if (shape.paired) tokens else rows, 1, shape.k }, 3, .bfloat16, 0, 0.5, bank.key, s));
        try mlx.check(mlx.mlx_array_eval(x));
        var plans: [32]Plan = undefined;
        for (benchPlans(&plans, rows, shape.experts, shape.k, shape.n, shape.paired)) |p| {
            if (p.sched == .db and (@rem(shape.k, 64) != 0 or @rem(shape.n, 64) != 0)) continue;
            const arm: BenchArm = .{ .x = x, .bank = bank, .up = up, .row_map = row_map, .sigtab = sigtab, .ids = ids, .plan = p, .bits = shape.bits, .group = shape.group, .mx = shape.mx, .s = s };
            const stock_build: BenchBuild = if (shape.paired) benchStockGateUp else benchStock;
            const port_build: BenchBuild = if (shape.paired) benchPortGateUp else benchPort;
            const ms = try abMedianMs(5, arm, stock_build, port_build);
            const ref = try stock_build(arm);
            defer _ = mlx.mlx_array_free(ref);
            const got = try port_build(arm);
            defer _ = mlx.mlx_array_free(got);
            std.debug.print("[gqmm-ubench] {s} M={d}: stock {d:.2} ms, port {d:.2} ms = {d:.3}x ({s} bm={d} bk={d} gx={d} pad={d}, {s})\n", .{ shape.name, rows, ms[0], ms[1], ms[0] / ms[1], @tagName(p.sched), p.bm, p.bk, p.gx, p.pad, if (try arraysEqual(got, ref, s)) @as([]const u8, "bit-equal") else "DIFFERS" });
        }
    }
}

// ── the staged schedule at DeepSeek-V4.1's routed experts ──

/// Every staged tile the planner or a sweep can pick; the equality test runs each.
const staged_variants = [_]Staged{
    .{ .bm = 32, .wm = 2, .wn = 1, .gx = 0 },
    .{ .bm = 64, .wm = 4, .wn = 1, .gx = 32 },
    .{ .bm = 128, .wm = 8, .wn = 1, .gx = 32 },
    .{ .bm = 256, .wm = 16, .wn = 1, .gx = 16 },
    .{ .bm = 64, .wm = 4, .wn = 1, .gx = 32, .tg_a = true },
    .{ .bm = 64, .wm = 2, .wn = 1, .gx = 32, .tg_a = true },
    .{ .bm = 64, .wm = 2, .wn = 2, .gx = 32, .tg_a = true },
    .{ .bm = 128, .wm = 8, .wn = 1, .gx = 32, .tg_a = true },
    .{ .bm = 128, .wm = 4, .wn = 1, .gx = 32, .tg_a = true },
    .{ .bm = 128, .wm = 4, .wn = 2, .gx = 32, .tg_a = true },
    .{ .bm = 16, .wm = 1, .wn = 2, .gx = 0 },
    .{ .bm = 16, .wm = 1, .wn = 2, .gx = 32 },
    .{ .bm = 32, .wm = 1, .wn = 2, .gx = 32 },
    .{ .bm = 32, .wm = 2, .wn = 1, .gx = 32 },
    .{ .bm = 32, .wm = 2, .wn = 2, .gx = 32 },
} ++ wide_tiles;

/// The wide-pass tiles the wide µbench times (and the equality test runs): 128 or 256 columns, 16 or 32 deep a step,
/// 16 x 64 or 32 x 32 a simdgroup.
const wide_tiles = [_]Staged{
    .{ .bm = 64, .wm = 4, .wn = 1, .gx = 32, .bk = 32 },
    .{ .bm = 64, .wm = 4, .wn = 2, .gx = 32, .bn = 128, .bk = 32 },
    .{ .bm = 128, .wm = 8, .wn = 2, .gx = 32, .bn = 128, .bk = 32 },
    .{ .bm = 128, .wm = 4, .wn = 2, .gx = 32, .bk = 32 },
    .{ .bm = 128, .wm = 4, .wn = 4, .gx = 32, .bn = 128, .bk = 32 },
    .{ .bm = 64, .wm = 2, .wn = 2, .gx = 32, .bk = 32 },
    .{ .bm = 64, .wm = 2, .wn = 4, .gx = 32, .bn = 128, .bk = 32 },
    .{ .bm = 32, .wm = 2, .wn = 4, .gx = 32, .bn = 128, .bk = 32 },
    .{ .bm = 64, .wm = 4, .wn = 4, .gx = 32, .bn = 256, .bk = 16 },
};

const ds_experts: c_int = 384;
const ds_top_k: usize = 6;
const ds_hidden: c_int = 5120;
const ds_inter: c_int = 2304;

/// An fp4 bank `[experts, n, k]` of random codes under random scale bytes about 1 (e4m3 0.5..1.9 per 16, or e8m0
/// 2^-2..2^1 per 32): every code is a value, so the draw needs no quantize pass over a bf16 bank.
fn randomFp4Bank(experts: c_int, n: c_int, k: c_int, group: u32, seed: u64, s: mlx.mlx_stream) !Bank {
    var key = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var w = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_random_bits(&w, &[_]c_int{ experts, n, @divExact(k, 8) }, 3, 4, key, s));
    var skey = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(skey);
    try mlx.check(mlx.mlx_random_key(&skey, seed ^ 0x5CA1E));
    var raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(raw);
    try mlx.check(mlx.mlx_random_bits(&raw, &[_]c_int{ experts, n, @divExact(k, @as(c_int, @intCast(group))) }, 3, 1, skey, s));
    const mask = mlx.mlx_array_new_data(&[_]u8{if (group == 16) 0x0f else 0x03}, &[_]c_int{1}, 1, .uint8);
    defer _ = mlx.mlx_array_free(mask);
    const base = mlx.mlx_array_new_data(&[_]u8{if (group == 16) 0x30 else 125}, &[_]c_int{1}, 1, .uint8);
    defer _ = mlx.mlx_array_free(base);
    var low = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(low);
    try mlx.check(mlx.mlx_bitwise_and(&low, raw, mask, s));
    var sc = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_add(&sc, low, base, s));
    try mlx.check(mlx.mlx_array_eval(w));
    try mlx.check(mlx.mlx_array_eval(sc));
    return .{ .w = w, .sc = sc, .bi = .{ .ctx = null }, .key = key };
}

/// A prompt call over DeepSeek-V4.1's routed experts: `tokens` rows, each on six distinct experts drawn with
/// `skewedRouting`'s skew (one in sixteen never picked, one in sixty-four eight times as often), the pairs sorted by
/// expert; the fused gate/up bank `[384, 4608, 5120]`, the down bank `[384, 5120, 2304]`, ModelOpt's per-expert scales,
/// the routing weights and a down input `[pairs, 1, 2304]`.
const DsCall = struct {
    x: mlx.mlx_array,
    map: mlx.mlx_array,
    ids: mlx.mlx_array,
    route: mlx.mlx_array,
    act: mlx.mlx_array,
    esc: [2]mlx.mlx_array,
    pairs: c_int,
    /// Experts with at least one row: the weight bytes a call must read.
    touched: usize,

    fn deinit(c: DsCall) void {
        for ([_]mlx.mlx_array{ c.x, c.map, c.ids, c.route, c.act, c.esc[0], c.esc[1] }) |a| _ = mlx.mlx_array_free(a);
    }
};

const DsBanks = struct {
    gu: Bank,
    down: Bank,
    group: u32,

    fn init(group: u32, s: mlx.mlx_stream) !DsBanks {
        const gu = try randomFp4Bank(ds_experts, 2 * ds_inter, ds_hidden, group, 0x6A7E, s);
        errdefer gu.deinit();
        return .{ .gu = gu, .down = try randomFp4Bank(ds_experts, ds_hidden, ds_inter, group, 0xD0E1, s), .group = group };
    }

    fn deinit(b: DsBanks) void {
        b.gu.deinit();
        b.down.deinit();
    }
};

fn dsCall(tokens: usize, seed: u64, s: mlx.mlx_stream) !DsCall {
    const alloc = std.testing.allocator;
    const ne: usize = @intCast(ds_experts);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var cum: [384]usize = undefined;
    var total: usize = 0;
    for (&cum, 0..) |*c, e| {
        total += if (e % 16 == 5) 0 else if (e % 64 == 0) 8 else 1;
        c.* = total;
    }
    const n = tokens * ds_top_k;
    const pick = try alloc.alloc(u32, n);
    defer alloc.free(pick);
    for (0..tokens) |t| {
        var j: usize = 0;
        while (j < ds_top_k) {
            const r = rng.uintLessThan(usize, total);
            var e: u32 = 0;
            while (cum[e] <= r) e += 1;
            if (std.mem.indexOfScalar(u32, pick[t * ds_top_k ..][0..j], e) == null) {
                pick[t * ds_top_k + j] = e;
                j += 1;
            }
        }
    }
    // Counting sort by expert, pairs in token order within an expert.
    var counts: [385]usize = @splat(0);
    for (pick) |e| counts[e + 1] += 1;
    var touched: usize = 0;
    for (1..ne + 1) |e| {
        if (counts[e] > 0) touched += 1;
        counts[e] += counts[e - 1];
    }
    const ids = try alloc.alloc(u32, n);
    defer alloc.free(ids);
    const map = try alloc.alloc(u32, n);
    defer alloc.free(map);
    const rw = try alloc.alloc(f32, n);
    defer alloc.free(rw);
    for (pick, 0..) |e, p| {
        const at = counts[e];
        counts[e] += 1;
        ids[at] = e;
        map[at] = @intCast(p / ds_top_k);
        rw[at] = 0.05 + rng.float(f32);
    }
    const pairs: c_int = @intCast(n);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ @intCast(tokens), 1, ds_hidden }, 3, .bfloat16, 0, 0.5, key, s));
    var akey = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(akey);
    try mlx.check(mlx.mlx_random_key(&akey, seed + 1));
    var act = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&act, &[_]c_int{ pairs, 1, ds_inter }, 3, .bfloat16, 0, 0.5, akey, s));
    var esc: [2]mlx.mlx_array = undefined;
    for (&esc, 0..) |*o, i| o.* = try randUniform(&[_]c_int{ds_experts}, 0.25, 4.0, seed + 2 + i, s);
    const c: DsCall = .{
        .x = x,
        .map = mlx.mlx_array_new_data(map.ptr, &[_]c_int{pairs}, 1, .uint32),
        .ids = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{pairs}, 1, .uint32),
        .route = mlx.mlx_array_new_data(rw.ptr, &[_]c_int{pairs}, 1, .float32),
        .act = act,
        .esc = esc,
        .pairs = pairs,
        .touched = touched,
    };
    for ([_]mlx.mlx_array{ c.x, c.act, c.esc[0], c.esc[1] }) |a| try mlx.check(mlx.mlx_array_eval(a));
    return c;
}

/// `[shape]` bf16 uniform over [lo, hi).
fn randUniform(shape: []const c_int, lo: f32, hi: f32, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_uniform(&out, lo_a, hi_a, shape.ptr, shape.len, .bfloat16, key, s));
    return out;
}

const ds_limit: f32 = 10;

/// Today's routed gate/up launch (the seg/db plan) and down launch on the call.
fn dsToday(c: DsCall, b: DsBanks, down: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    if (down) return (try sortedGatherPrompt(c.act, b.down.w, b.down.sc, b.down.bi, c.ids, 4, b.group, true, true, s)) orelse error.KernelDeclinedTestShape;
    return (try sortedGateUpRouted(c.x, c.map, b.gu.w, b.gu.sc, b.gu.bi, c.ids, c.route, ds_limit, .{ .gate = c.esc[0], .up = c.esc[1] }, 4, b.group, true, true, s)) orelse error.KernelDeclinedTestShape;
}

/// The staged launch of `st` on the call; `scan_bm` the scan's tile height (`st.bm`, or a planted other).
fn dsStaged(c: DsCall, b: DsBanks, down: bool, st: Staged, scan_bm: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (down) return launchStaged(c.act, null, b.down.w, b.down.sc, b.down.bi, c.ids, null, 4, b.group, true, st, scan_bm, s);
    return launchStaged(c.x, c.map, b.gu.w, b.gu.sc, b.gu.bi, c.ids, .{ .w = b.gu.w, .sc = b.gu.sc, .bi = b.gu.bi, .sigtab = biasArg(.{ .ctx = null }), .up_off = ds_inter, .scales = .{ .gate = c.esc[0], .up = c.esc[1] }, .route = c.route, .route_limit = ds_limit }, 4, b.group, true, st, scan_bm, s);
}

test "staged NAX sorted gather is the seg/db kernels' words at DeepSeek-V4.1's routed experts (NVFP4 and MXFP4, every staged tile, odd prompt rows)" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    for ([_]u32{ 16, 32 }) |group| {
        const b = try DsBanks.init(group, s);
        defer b.deinit();
        for ([_]usize{ 13, 777, 1254, 4097, 8192 }, 0..) |tokens, ti| {
            const c = try dsCall(tokens, 0x2000 + ti, s);
            defer c.deinit();
            for ([_]bool{ false, true }) |down| {
                const want = try dsToday(c, b, down, s);
                defer _ = mlx.mlx_array_free(want);
                for (staged_variants) |st| {
                    const got = try dsStaged(c, b, down, st, st.bm, s);
                    defer _ = mlx.mlx_array_free(got);
                    const same = try arraysEqual(want, got, s);
                    if (!same) std.debug.print("[gqmm-staged] {s} {s} tokens={d} bm={d} wm={d} wn={d} tg_a={}: DIFFERS\n", .{ modeName(true, group), if (down) @as([]const u8, "down") else "gate+up", tokens, st.bm, st.wm, st.wn, st.tg_a });
                    try std.testing.expect(same);
                }
                // The public entries take the planner's tile.
                const served = (if (down)
                    try sortedGatherPromptStaged(c.act, b.down.w, b.down.sc, b.down.bi, c.ids, 4, group, true, true, s)
                else
                    try sortedGateUpRoutedStaged(c.x, c.map, b.gu.w, b.gu.sc, b.gu.bi, c.ids, c.route, ds_limit, .{ .gate = c.esc[0], .up = c.esc[1] }, 4, group, true, true, s)) orelse return error.KernelDeclinedTestShape;
                defer _ = mlx.mlx_array_free(served);
                try expectBitEqual(want, served, s);
            }
        }
    }
}

test "staged NAX sorted gather: tiles cut taller than the kernel's tile (a planted plan mismatch) lose rows and differ" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const b = try DsBanks.init(16, s);
    defer b.deinit();
    const c = try dsCall(4096, 0x3001, s);
    defer c.deinit();
    const st: Staged = .{ .bm = 64, .wm = 4, .wn = 1, .gx = 32 };
    for ([_]bool{ false, true }) |down| {
        const want = try dsToday(c, b, down, s);
        defer _ = mlx.mlx_array_free(want);
        const bad = try dsStaged(c, b, down, st, 2 * st.bm, s);
        defer _ = mlx.mlx_array_free(bad);
        try std.testing.expect(!try arraysEqual(want, bad, s));
    }
    // Scales on the wrong experts move the gate/up words.
    const wrong: DsCall = blk: {
        var v = c;
        v.esc = .{ c.esc[1], c.esc[0] };
        break :blk v;
    };
    const want = try dsToday(c, b, false, s);
    defer _ = mlx.mlx_array_free(want);
    const bad = try dsStaged(wrong, b, false, st, st.bm, s);
    defer _ = mlx.mlx_array_free(bad);
    try std.testing.expect(!try arraysEqual(want, bad, s));
}

/// Milliseconds a call: four calls evaluated together, median of 7 after one warm round, today's and the staged
/// interleaved.
fn dsMedianMs(c: DsCall, b: DsBanks, down: bool, st: ?Staged, s: mlx.mlx_stream) !f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var laps: [7]u64 = undefined;
    for (0..laps.len + 1) |i| {
        const mark = std.Io.Timestamp.now(io, .boot);
        var outs: [4]mlx.mlx_array = undefined;
        for (&outs) |*o| o.* = if (st) |v| try dsStaged(c, b, down, v, v.bm, s) else try dsToday(c, b, down, s);
        defer for (outs) |o| {
            _ = mlx.mlx_array_free(o);
        };
        const vec = mlx.mlx_vector_array_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        if (i > 0) laps[i - 1] = @intCast(mark.untilNow(io, .boot).nanoseconds);
    }
    std.mem.sort(u64, &laps, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(laps[laps.len / 2])) / 4e6;
}

test "staged NAX sorted gather µbench at DeepSeek-V4.1's NVFP4 routed experts against today's seg/db (diag gqmm-ubench, =sweep: every staged tile)" {
    const raw = env.get(.GQMM_UBENCH) orelse return error.SkipZigTest;
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const sweep = std.mem.eql(u8, std.mem.sliceTo(raw, 0), "sweep");
    const b = try DsBanks.init(16, s);
    defer b.deinit();
    // Today's passes (4,096 + 4,096 + 8,192 + 1,254 rows of the p01 prompt) and one 17,639-row pass.
    for ([_]usize{ 1254, 4096, 8192, 17639 }, 0..) |tokens, ti| {
        const c = try dsCall(tokens, 0x4000 + ti, s);
        defer c.deinit();
        for ([_]bool{ false, true }) |down| {
            const flops = 2.0 * @as(f64, @floatFromInt(c.pairs)) * @as(f64, @floatFromInt(ds_hidden)) * @as(f64, @floatFromInt(if (down) ds_inter else 2 * ds_inter));
            // Codes plus one scale byte per 16, for every expert the call touches.
            const bytes = @as(f64, @floatFromInt(c.touched)) * @as(f64, @floatFromInt(ds_hidden)) * @as(f64, @floatFromInt(if (down) ds_inter else 2 * ds_inter)) * (0.5 + 1.0 / 16.0);
            const floor_ms = @max(flops / 120e9, bytes / 1.065e9);
            const today = try dsMedianMs(c, b, down, null, s);
            const p = plan(c.pairs, ds_experts, if (down) ds_inter else ds_hidden, if (down) ds_hidden else ds_inter, !down);
            std.debug.print("[gqmm-ubench] dsv41 nvfp4 {s} tokens={d} pairs={d} experts={d}: today {d:.3} ms {d:.1} TFLOPS ({s} bm={d}), floor {d:.3} ms ({s})\n", .{ if (down) @as([]const u8, "down 2304->5120") else "gate+up+act 5120->2x2304", tokens, c.pairs, c.touched, today, flops / today / 1e9, @tagName(p.sched), p.bm, floor_ms, if (flops / 120e9 >= bytes / 1.065e9) @as([]const u8, "compute") else "weights" });
            const pick = stagedPlan(c.pairs, ds_experts);
            for (staged_variants) |st| {
                if (!sweep and !std.meta.eql(st, pick)) continue;
                const ms = try dsMedianMs(c, b, down, st, s);
                std.debug.print("[gqmm-ubench]   staged bm={d} wm={d} wn={d} gx={d} {s}{s}: {d:.3} ms {d:.1} TFLOPS ({d:.0}% of 120), {d:.3}x today, {d:.0}% of floor\n", .{ st.bm, st.wm, st.wn, st.gx, if (st.tg_a) @as([]const u8, "tg-a") else "reg-a", if (std.meta.eql(st, pick)) @as([]const u8, " (plan)") else "", ms, flops / ms / 1e9, flops / ms / 1.2e9, today / ms, 100 * floor_ms / ms });
            }
        }
    }
}

/// A routing dump's calls (deepseek_v41's `DSV41_ROUTE_DUMP`): each record u32 rows, picks a row, experts and pairs,
/// then the pairs' experts (sorted) and their rows.
pub const RouteDump = struct {
    words: []align(1) const u32,
    at: usize = 0,

    pub const Call = struct { rows: usize, k: usize, experts: usize, ids: []align(1) const u32, rows_of: []align(1) const u32 };

    pub fn init(bytes: []const u8) RouteDump {
        return .{ .words = std.mem.bytesAsSlice(u32, bytes[0 .. bytes.len / 4 * 4]) };
    }

    pub fn next(self: *RouteDump) !?Call {
        if (self.at + 4 > self.words.len) return null;
        const w = self.words[self.at..];
        const n: usize = w[3];
        if (4 + 2 * n > w.len) return error.BadRouteDump;
        self.at += 4 + 2 * n;
        return .{ .rows = w[0], .k = w[1], .experts = w[2], .ids = w[4..][0..n], .rows_of = w[4 + n ..][0..n] };
    }
};

/// The flat picks of a dumped call (`flat[r * k + j]` token r's j-th expert, its picks in the order its sorted rows
/// come), as moePrefill's routing gives them to the down's scatter.
fn flatPicks(a: std.mem.Allocator, c: RouteDump.Call) ![]u32 {
    const flat = try a.alloc(u32, c.ids.len);
    errdefer a.free(flat);
    const fill = try a.alloc(u8, c.rows);
    defer a.free(fill);
    @memset(fill, 0);
    for (c.ids, c.rows_of) |e, r| {
        if (r >= c.rows or fill[r] >= c.k) return error.BadRouteDump;
        flat[r * c.k + fill[r]] = e;
        fill[r] += 1;
    }
    return flat;
}

/// A call over a served routing: the sorted pairs' experts and rows as `DSV41_ROUTE_DUMP` wrote them (`tokens` rows);
/// the activations, routing weights and scales drawn as `dsCall` draws them.
fn dsCallRouted(ids: []const u32, rows: []const u32, tokens: usize, seed: u64, s: mlx.mlx_stream) !DsCall {
    const pairs: c_int = @intCast(ids.len);
    var touched: usize = 0;
    for (ids, 0..) |e, i| {
        if (i == 0 or ids[i - 1] != e) touched += 1;
    }
    const rw = try std.testing.allocator.alloc(f32, ids.len);
    defer std.testing.allocator.free(rw);
    var prng = std.Random.DefaultPrng.init(seed);
    for (rw) |*v| v.* = 0.05 + prng.random().float(f32);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ @intCast(tokens), 1, ds_hidden }, 3, .bfloat16, 0, 0.5, key, s));
    var akey = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(akey);
    try mlx.check(mlx.mlx_random_key(&akey, seed + 1));
    var act = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&act, &[_]c_int{ pairs, 1, ds_inter }, 3, .bfloat16, 0, 0.5, akey, s));
    var esc: [2]mlx.mlx_array = undefined;
    for (&esc, 0..) |*o, i| o.* = try randUniform(&[_]c_int{ds_experts}, 0.25, 4.0, seed + 2 + i, s);
    const c: DsCall = .{
        .x = x,
        .map = mlx.mlx_array_new_data(rows.ptr, &[_]c_int{pairs}, 1, .uint32),
        .ids = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{pairs}, 1, .uint32),
        .route = mlx.mlx_array_new_data(rw.ptr, &[_]c_int{pairs}, 1, .float32),
        .act = act,
        .esc = esc,
        .pairs = pairs,
        .touched = touched,
    };
    for ([_]mlx.mlx_array{ c.x, c.act, c.esc[0], c.esc[1] }) |v| try mlx.check(mlx.mlx_array_eval(v));
    return c;
}

/// Milliseconds a call of the public staged entries (`st` null: today's plan) or of tile `st`, four evaluated
/// together, median of 7 after one warm round.
fn dsRoutedMs(c: DsCall, b: DsBanks, down: bool, st: ?Staged, s: mlx.mlx_stream) !f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var laps: [7]u64 = undefined;
    for (0..laps.len + 1) |i| {
        const mark = std.Io.Timestamp.now(io, .boot);
        var outs: [4]mlx.mlx_array = undefined;
        for (&outs) |*o| o.* = if (st) |v| try dsStaged(c, b, down, v, v.bm, s) else (if (down)
            try sortedGatherPromptStaged(c.act, b.down.w, b.down.sc, b.down.bi, c.ids, 4, b.group, true, true, s)
        else
            try sortedGateUpRoutedStaged(c.x, c.map, b.gu.w, b.gu.sc, b.gu.bi, c.ids, c.route, ds_limit, .{ .gate = c.esc[0], .up = c.esc[1] }, 4, b.group, true, true, s)) orelse return error.KernelDeclinedTestShape;
        defer for (outs) |o| {
            _ = mlx.mlx_array_free(o);
        };
        const vec = mlx.mlx_vector_array_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        if (i > 0) laps[i - 1] = @intCast(mark.untilNow(io, .boot).nanoseconds);
    }
    std.mem.sort(u64, &laps, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(laps[laps.len / 2])) / 4e6;
}

test "staged NAX sorted gather µbench on a served routing, today's plan against every staged tile (diag gqmm-ubench=route:<DSV41_ROUTE_DUMP file>)" {
    const raw = std.mem.sliceTo(env.get(.GQMM_UBENCH) orelse return error.SkipZigTest, 0);
    if (!std.mem.startsWith(u8, raw, "route:")) return error.SkipZigTest;
    try requireNax();
    errdefer dropOwnLatch();
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, raw["route:".len..], a, .limited(1 << 28));
    defer a.free(bytes);
    const b = try DsBanks.init(16, s);
    defer b.deinit();
    // The tiles run on the calls of the dump's last row count only (the CED replay's); every call runs today's plan.
    var sweep_rows: usize = 0;
    {
        var it = RouteDump.init(bytes);
        while (try it.next()) |c| sweep_rows = c.rows;
    }
    var tot: [2][1 + staged_variants.len]f64 = @splat(@splat(0));
    var floor_tot: [2]f64 = .{ 0, 0 };
    var calls: usize = 0;
    var it = RouteDump.init(bytes);
    var layer: usize = 0;
    while (try it.next()) |rc| : (layer += 1) {
        if (rc.experts != ds_experts or rc.k != ds_top_k) return error.BadRouteDump;
        const L = rc.rows;
        if (L != sweep_rows) continue;
        calls += 1;
        const c = try dsCallRouted(@alignCast(rc.ids), @alignCast(rc.rows_of), L, 0x6000 + layer, s);
        defer c.deinit();
        for ([_]bool{ false, true }, 0..) |down, di| {
            const wb = @as(f64, @floatFromInt(c.touched)) * @as(f64, @floatFromInt(ds_hidden)) * @as(f64, @floatFromInt(if (down) ds_inter else 2 * ds_inter)) * (0.5 + 1.0 / 16.0);
            floor_tot[di] += wb / 1.065e9;
            const today = try dsRoutedMs(c, b, down, null, s);
            tot[di][0] += today;
            for (staged_variants, 0..) |st, vi| tot[di][1 + vi] += try dsRoutedMs(c, b, down, st, s);
        }
    }
    const pick = stagedPlan(@intCast(6 * sweep_rows), ds_experts);
    for (0..2) |di| {
        std.debug.print("[gqmm-route] {d} calls of {d} rows {s}: today {d:.2} ms (plan bm={d} wm={d} wn={d} gx={d}), weight floor {d:.2} ms ({d:.0}%)\n", .{ calls, sweep_rows, if (di == 1) @as([]const u8, "down") else "gate+up+act", tot[di][0], pick.bm, pick.wm, pick.wn, pick.gx, floor_tot[di], 100 * floor_tot[di] / tot[di][0] });
        for (staged_variants, 0..) |st, vi| std.debug.print("[gqmm-route]   staged bm={d} wm={d} wn={d} gx={d} {s}: {d:.2} ms ({d:.0}% of floor), {d:.2} ms saved\n", .{ st.bm, st.wm, st.wn, st.gx, if (st.tg_a) @as([]const u8, "tg-a") else "reg-a", tot[di][1 + vi], 100 * floor_tot[di] / tot[di][1 + vi], tot[di][0] - tot[di][1 + vi] });
    }
}

/// Milliseconds a down call takes (four evaluated together, median of 7 after one warm round): today's staged entry, or
/// with `ridx` the scatter (`sortedGatherPromptStagedTok`, each row at its pair's slot).
fn dsDownMs(c: DsCall, b: DsBanks, ridx: ?mlx.mlx_array, s: mlx.mlx_stream) !f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var laps: [7]u64 = undefined;
    for (0..laps.len + 1) |i| {
        const mark = std.Io.Timestamp.now(io, .boot);
        var outs: [4]mlx.mlx_array = undefined;
        for (&outs) |*o| o.* = (if (ridx) |r|
            try sortedGatherPromptStagedTok(c.act, b.down.w, b.down.sc, b.down.bi, c.ids, c.map, r, @intCast(ds_top_k), 4, b.group, true, true, s)
        else
            try sortedGatherPromptStaged(c.act, b.down.w, b.down.sc, b.down.bi, c.ids, 4, b.group, true, true, s)) orelse return error.KernelDeclinedTestShape;
        defer for (outs) |o| {
            _ = mlx.mlx_array_free(o);
        };
        const vec = mlx.mlx_vector_array_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        if (i > 0) laps[i - 1] = @intCast(mark.untilNow(io, .boot).nanoseconds);
    }
    std.mem.sort(u64, &laps, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(laps[laps.len / 2])) / 4e6;
}

test "staged NAX down scatter µbench on a served routing: PF_DOWN_TOKORD's scatter against today's staged down at every call of the dump (diag gqmm-ubench=route:<DSV41_ROUTE_DUMP file>)" {
    const raw = std.mem.sliceTo(env.get(.GQMM_UBENCH) orelse return error.SkipZigTest, 0);
    if (!std.mem.startsWith(u8, raw, "route:")) return error.SkipZigTest;
    try requireNax();
    errdefer dropOwnLatch();
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, raw["route:".len..], a, .limited(1 << 30));
    defer a.free(bytes);
    const b = try DsBanks.init(16, s);
    defer b.deinit();
    // Totals by call width (rows), today's and the scatter's.
    const W = struct { rows: usize, calls: usize = 0, today: f64 = 0, tok: f64 = 0, floor: f64 = 0 };
    var ws: [16]W = undefined;
    var nw: usize = 0;
    var it = RouteDump.init(bytes);
    var layer: usize = 0;
    while (try it.next()) |rc| : (layer += 1) {
        if (rc.experts != ds_experts or rc.k != ds_top_k) return error.BadRouteDump;
        const L = rc.rows;
        const n = rc.ids.len;
        const flat = try flatPicks(a, rc);
        defer a.free(flat);
        const c = try dsCallRouted(@alignCast(rc.ids), @alignCast(rc.rows_of), L, 0x7000 + layer, s);
        defer c.deinit();
        const ridx = mlx.mlx_array_new_data(flat.ptr, &[_]c_int{@intCast(n)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(ridx);
        const wi = for (ws[0..nw], 0..) |w, i| {
            if (w.rows == L) break i;
        } else blk: {
            if (nw == ws.len) return error.TooManyWidths;
            ws[nw] = .{ .rows = L };
            nw += 1;
            break :blk nw - 1;
        };
        const today = try dsDownMs(c, b, null, s);
        const tok = try dsDownMs(c, b, ridx, s);
        ws[wi].calls += 1;
        ws[wi].today += today;
        ws[wi].tok += tok;
        ws[wi].floor += @as(f64, @floatFromInt(c.touched)) * @as(f64, @floatFromInt(ds_hidden)) * @as(f64, @floatFromInt(ds_inter)) * (0.5 + 1.0 / 16.0) / 1.065e9;
    }
    for (ws[0..nw]) |w| std.debug.print("[gqmm-route] down {d} calls of {d} rows: today {d:.3} ms, scatter {d:.3} ms ({d:.3} ms, {d:.2}%), weight floor {d:.3} ms\n", .{ w.calls, w.rows, w.today, w.tok, w.tok - w.today, 100 * (w.tok - w.today) / w.today, w.floor });
}

/// A launch the wide µbench times on a dumped call: the served pair (`sortedGateUpRouted`, the seg plan) or down
/// (`sortedGatherPromptStagedTok`, the planner's tile), a staged tile in place of either, or a seg/db plan.
const WideArm = union(enum) { pair, down, pair_tile: Staged, down_tile: Staged, down_sorted_tile: Staged, pair_plan: Plan, down_plan: Plan };

fn wideLabel(arm: WideArm, buf: []u8) []const u8 {
    return switch (arm) {
        .pair => "served pair",
        .down => "served down",
        .pair_tile, .down_tile, .down_sorted_tile => |st| std.fmt.bufPrint(buf, "staged bm={d} bn={d} bk={d} wm={d} wn={d} gx={d}{s}{s}", .{ st.bm, st.bn, st.bk, st.wm, st.wn, st.gx, if (st.tg_a) @as([]const u8, " tg-a") else "", if (arm == .down_sorted_tile) @as([]const u8, " (sorted)") else "" }) catch "?",
        .pair_plan, .down_plan => |p| std.fmt.bufPrint(buf, "{s} bm={d} bk={d} gx={d} pad={d}{s}", .{ @tagName(p.sched), p.bm, p.bk, p.gx, p.pad, if (arm == .down_plan) @as([]const u8, " (sorted)") else "" }) catch "?",
    };
}

fn wideRun(c: DsCall, b: DsBanks, ridx: mlx.mlx_array, arm: WideArm, s: mlx.mlx_stream) !mlx.mlx_array {
    const esc: ExpertScales = .{ .gate = c.esc[0], .up = c.esc[1] };
    const k: c_int = @intCast(ds_top_k);
    return switch (arm) {
        .pair => (try sortedGateUpRouted(c.x, c.map, b.gu.w, b.gu.sc, b.gu.bi, c.ids, c.route, ds_limit, esc, 4, b.group, true, true, s)) orelse error.KernelDeclinedTestShape,
        .down => (try sortedGatherPromptStagedTok(c.act, b.down.w, b.down.sc, b.down.bi, c.ids, c.map, ridx, k, 4, b.group, true, true, s)) orelse error.KernelDeclinedTestShape,
        .pair_tile => |st| (try sortedGateUpRoutedTile(c.x, c.map, b.gu.w, b.gu.sc, b.gu.bi, c.ids, c.route, ds_limit, esc, 4, b.group, true, true, st, s)) orelse error.KernelDeclinedTestShape,
        .down_tile => |st| (try sortedGatherPromptTokTile(c.act, b.down.w, b.down.sc, b.down.bi, c.ids, c.map, ridx, k, 4, b.group, true, true, st, s)) orelse error.KernelDeclinedTestShape,
        .down_sorted_tile => |st| (try sortedGatherPromptTile(c.act, b.down.w, b.down.sc, b.down.bi, c.ids, 4, b.group, true, true, st, s)) orelse error.KernelDeclinedTestShape,
        .pair_plan => |p| try launchMapped(c.x, c.map, b.gu.w, b.gu.sc, b.gu.bi, c.ids, .{ .w = b.gu.w, .sc = b.gu.sc, .bi = b.gu.bi, .sigtab = biasArg(.{ .ctx = null }), .up_off = ds_inter, .scales = esc, .route = c.route, .route_limit = ds_limit }, 4, b.group, true, p, s),
        // Sorted order (no scatter): the down's words in the served call's sorted rows.
        .down_plan => |p| try launch(c.act, b.down.w, b.down.sc, b.down.bi, c.ids, 4, b.group, true, p, s),
    };
}

/// Milliseconds one launch takes alone: each evaluated before the next starts (a served layer's routed call has the
/// GPU to itself but for the shared expert), median of 5 after one warm.
fn wideSoloMs(c: DsCall, b: DsBanks, ridx: mlx.mlx_array, arm: WideArm, s: mlx.mlx_stream) !f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var laps: [5]u64 = undefined;
    for (0..laps.len + 1) |i| {
        const mark = std.Io.Timestamp.now(io, .boot);
        const y = try wideRun(c, b, ridx, arm, s);
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
        if (i > 0) laps[i - 1] = @intCast(mark.untilNow(io, .boot).nanoseconds);
    }
    std.mem.sort(u64, &laps, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(laps[laps.len / 2])) / 1e6;
}

test "staged NAX wide µbench on a served routing: the served pair and down against staged tiles and seg/db plans at the dump's widest calls (diag gqmm-ubench=wide:<DSV41_ROUTE_DUMP file>)" {
    const raw = std.mem.sliceTo(env.get(.GQMM_UBENCH) orelse return error.SkipZigTest, 0);
    if (!std.mem.startsWith(u8, raw, "wide:")) return error.SkipZigTest;
    try requireNax();
    errdefer dropOwnLatch();
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, raw["wide:".len..], a, .limited(1 << 30));
    defer a.free(bytes);
    var widest: usize = 0;
    {
        var it = RouteDump.init(bytes);
        while (try it.next()) |rc| widest = @max(widest, rc.rows);
    }
    const tiles = [_]Staged{ comptime stagedPlan(0, 0), .{ .bm = 128, .wm = 8, .wn = 1, .gx = 32 }, .{ .bm = 256, .wm = 16, .wn = 1, .gx = 16 } } ++ wide_tiles;
    const pair_plans = [_]Plan{ plan(@intCast(widest * ds_top_k), ds_experts, ds_hidden, ds_inter, true), .{ .sched = .seg, .bm = 128, .bk = 128, .gx = 32, .pad = 0 }, .{ .sched = .seg, .bm = 128, .bk = 64, .gx = 32, .pad = 0 }, .{ .sched = .db, .bm = 128, .bk = 64, .gx = 32, .pad = 0 } };
    const down_plans = [_]Plan{ plan(@intCast(widest * ds_top_k), ds_experts, ds_inter, ds_hidden, false), .{ .sched = .seg, .bm = 128, .bk = 128, .gx = 32, .pad = 8192 }, .{ .sched = .seg, .bm = 128, .bk = 128, .gx = 32, .pad = 0 }, .{ .sched = .seg, .bm = 128, .bk = 64, .gx = 32, .pad = 0 } };
    const na = 1 + tiles.len + pair_plans.len;
    var arms: [2][na]WideArm = undefined;
    for (0..2) |di| {
        arms[di][0] = if (di == 0) .pair else .down;
        for (tiles, 0..) |t, i| arms[di][1 + i] = if (di == 0) .{ .pair_tile = t } else .{ .down_tile = t };
        for (pair_plans, down_plans, 0..) |pp, dp, i| arms[di][1 + tiles.len + i] = if (di == 0) .{ .pair_plan = pp } else .{ .down_plan = dp };
    }
    // A tile choice by expert: two launches, the experts of at least `split_rows[t]` rows on one tile and the rest on
    // another; every tile and plan on each part (the down in sorted rows).
    const split_rows = [_]u32{ 128, 256 };
    const ns = tiles.len + pair_plans.len;
    var sarms: [2][ns]WideArm = undefined;
    for (0..2) |di| {
        for (tiles, 0..) |t, i| sarms[di][i] = if (di == 0) .{ .pair_tile = t } else .{ .down_sorted_tile = t };
        for (pair_plans, down_plans, 0..) |pp, dp, i| sarms[di][tiles.len + i] = if (di == 0) .{ .pair_plan = pp } else .{ .down_plan = dp };
    }
    var split: [split_rows.len][2][2][ns]f64 = @splat(@splat(@splat(@splat(0))));
    var split_n: [split_rows.len][2][2]usize = @splat(@splat(.{ 0, 0 }));
    const b = try DsBanks.init(16, s);
    defer b.deinit();
    var tot: [2][na]f64 = @splat(@splat(0));
    var same: [2][na]bool = @splat(@splat(true));
    var flops: [2]f64 = .{ 0, 0 };
    var wfloor: [2]f64 = .{ 0, 0 };
    var calls: usize = 0;
    var it = RouteDump.init(bytes);
    var layer: usize = 0;
    while (try it.next()) |rc| : (layer += 1) {
        if (rc.experts != ds_experts or rc.k != ds_top_k) return error.BadRouteDump;
        if (rc.rows != widest) continue;
        const flat = try flatPicks(a, rc);
        defer a.free(flat);
        const c = try dsCallRouted(@alignCast(rc.ids), @alignCast(rc.rows_of), rc.rows, 0x8000 + layer, s);
        defer c.deinit();
        const ridx = mlx.mlx_array_new_data(flat.ptr, &[_]c_int{c.pairs}, 1, .uint32);
        defer _ = mlx.mlx_array_free(ridx);
        for (0..2) |di| {
            const width: f64 = @floatFromInt(if (di == 0) 2 * ds_inter else ds_hidden);
            const depth: f64 = @floatFromInt(if (di == 0) ds_hidden else ds_inter);
            flops[di] += 2 * @as(f64, @floatFromInt(c.pairs)) * width * depth;
            wfloor[di] += @as(f64, @floatFromInt(c.touched)) * width * depth * (0.5 + 1.0 / 16.0) / 1.065e9;
            // Words on the first call: each arm against the served call; a down plan (sorted rows) against the
            // planner's down plan.
            if (calls == 0) {
                const want = try wideRun(c, b, ridx, arms[di][0], s);
                defer _ = mlx.mlx_array_free(want);
                const sorted = try wideRun(c, b, ridx, .{ .down_plan = down_plans[0] }, s);
                defer _ = mlx.mlx_array_free(sorted);
                for (arms[di][1..], 1..) |arm, ai| {
                    const got = try wideRun(c, b, ridx, arm, s);
                    defer _ = mlx.mlx_array_free(got);
                    same[di][ai] = try arraysEqual(if (arm == .down_plan) sorted else want, got, s);
                }
            }
            for (arms[di], 0..) |arm, ai| tot[di][ai] += try wideSoloMs(c, b, ridx, arm, s);
        }
        var cnt: [384]u32 = @splat(0);
        for (rc.ids) |e| cnt[e] += 1;
        const ids_s = try a.alloc(u32, rc.ids.len);
        defer a.free(ids_s);
        const rows_s = try a.alloc(u32, rc.ids.len);
        defer a.free(rows_s);
        for (split_rows, 0..) |T, ti| for (0..2) |part| {
            var n: usize = 0;
            var ne: usize = 0;
            for (rc.ids, rc.rows_of) |e, r| if ((cnt[e] >= T) == (part == 0)) {
                if (n == 0 or ids_s[n - 1] != e) ne += 1;
                ids_s[n] = e;
                rows_s[n] = r;
                n += 1;
            };
            if (n == 0) continue;
            split_n[ti][part][0] += ne;
            split_n[ti][part][1] += n;
            const cs = try dsCallRouted(ids_s[0..n], rows_s[0..n], rc.rows, 0x9000 + 4 * layer + 2 * ti + part, s);
            defer cs.deinit();
            for (0..2) |di| for (sarms[di], 0..) |arm, ai| {
                split[ti][part][di][ai] += try wideSoloMs(cs, b, ridx, arm, s);
            };
        };
        calls += 1;
    }
    if (calls == 0) return error.NoWideCall;
    const nc: f64 = @floatFromInt(calls);
    for (0..2) |di| {
        const f = flops[di] / nc;
        const ctl = tot[di][0] / nc;
        std.debug.print("[gqmm-wide] {s} over {d} calls of {d} rows: served {d:.3} ms a call, {d:.1} TFLOPS; compute floor at 120 TFLOPS {d:.3} ms, weight floor {d:.3} ms\n", .{ if (di == 0) @as([]const u8, "gate+up+act 5120->2x2304") else "down 2304->5120 (scatter)", calls, widest, ctl, f / ctl / 1e9, f / 120e9, wfloor[di] / nc });
        var best_single: f64 = ctl;
        for (arms[di][1..], 1..) |arm, ai| {
            const ms = tot[di][ai] / nc;
            best_single = @min(best_single, ms);
            var name: [96]u8 = undefined;
            std.debug.print("[gqmm-wide]   {s}: {d:.3} ms a call, {d:.1} TFLOPS, {d:.3}x served, {d:.0}% of the 120 TFLOPS floor, {s}\n", .{ wideLabel(arm, &name), ms, f / ms / 1e9, ctl / ms, 100 * f / 120e9 / ms, if (same[di][ai]) @as([]const u8, "bit-equal") else "DIFFERS" });
        }
        for (split_rows, 0..) |T, ti| {
            var best: [2]f64 = .{ std.math.inf(f64), std.math.inf(f64) };
            var at: [2]usize = .{ 0, 0 };
            for (0..2) |part| {
                std.debug.print("[gqmm-wide]   split at {d} rows an expert, part {s} ({d:.0} experts, {d:.0} pairs a call):", .{ T, if (part == 0) @as([]const u8, "big") else "small", @as(f64, @floatFromInt(split_n[ti][part][0])) / nc, @as(f64, @floatFromInt(split_n[ti][part][1])) / nc });
                for (sarms[di], 0..) |_, ai| {
                    const ms = split[ti][part][di][ai] / nc;
                    std.debug.print(" {d:.2}", .{ms});
                    if (ms < best[part]) {
                        best[part] = ms;
                        at[part] = ai;
                    }
                }
                std.debug.print("\n", .{});
            }
            var n0: [96]u8 = undefined;
            var n1: [96]u8 = undefined;
            std.debug.print("[gqmm-wide]   split at {d}: best big {s} {d:.3} ms + best small {s} {d:.3} ms = {d:.3} ms a call ({d:.3}x served, {d:.3}x the best single launch {d:.3} ms)\n", .{ T, wideLabel(sarms[di][at[0]], &n0), best[0], wideLabel(sarms[di][at[1]], &n1), best[1], best[0] + best[1], ctl / (best[0] + best[1]), best_single / (best[0] + best[1]), best_single });
        }
    }
}
