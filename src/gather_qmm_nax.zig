// SPDX-License-Identifier: Apache-2.0
// Ported from oMLX (jundot/omlx) omlx/patches/m5_gather_qmm_nax.py @ d6b2b92.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const Plan = struct { sched: enum { seg, db }, bm: c_int, bk: c_int, gx: c_int, pad: c_int };
/// `mx`: an MXFP4 bank (uint8 e8m0 scales, no biases) instead of affine.
const CanaryKey = struct { plan: Plan, bits: u32, group: u32, align_n: bool, align_k: bool, mx: bool };
const MmKey = struct { rows: c_int, n: c_int, k: c_int, max_tiles: c_int, plan: Plan, bits: u32, group: u32, mx: bool, mapped: bool = false, paired: bool = false, limit: u32 = 0 };
var scan_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mm_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mapped_kernel: ?mlx.mlx_fast_metal_kernel = null;
var paired_kernel: ?mlx.mlx_fast_metal_kernel = null;
var canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var mapped_canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var paired_canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var engaged_logged = false;
var mapped_engaged_logged = false;
var paired_engaged_logged = false;

/// Tile configuration by mean rows per expert and K. The 96+-row rungs are ours: oMLX's
/// seg tiles were measured against mlx 0.32.2, whose sorted kernel masked whole row blocks;
/// 0.32.3 schedules row tiles itself and the plain db 96-row tile-on-x layout is the one
/// that still beats it. The paired gate/up kernel keeps seg 128.
fn plan(rows: c_int, experts: c_int, k: c_int, n: c_int, paired: bool) Plan {
    if (@rem(k, 64) != 0 or @rem(n, 64) != 0) return .{ .sched = .seg, .bm = 64, .bk = 64, .gx = 0, .pad = 0 };
    const per_expert = @divTrunc(rows, @max(experts, 1));
    if (per_expert < 36 or (k < 1024 and per_expert < 120)) return .{ .sched = .db, .bm = 64, .bk = 64, .gx = 0, .pad = 0 };
    if (k < 1024 and paired) return .{ .sched = .seg, .bm = 96, .bk = 128, .gx = 32, .pad = 0 };
    if (k >= 1024 and per_expert < 48) return .{ .sched = .db, .bm = 64, .bk = 64, .gx = 32, .pad = 0 };
    if (k < 1024 or per_expert < 96 or !paired) return .{ .sched = .db, .bm = 96, .bk = 64, .gx = 32, .pad = 0 };
    return .{ .sched = .seg, .bm = 128, .bk = 128, .gx = 32, .pad = 8192 };
}

fn supported(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, row_map: ?mlx.mlx_array) bool {
    if (x.ctx == null or w.ctx == null or sc.ctx == null or idx.ctx == null) return false;
    if (mx) {
        if (bits != 4 or group != 32 or bi.ctx != null or mlx.mlx_array_dtype(sc) != .uint8) return false;
    } else {
        if (bi.ctx == null) return false;
        if (bits != 4 and bits != 8) return false;
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
    // order, and MTP verify rounds must keep its bits.
    if (ids[0] < 16 or @divTrunc(ids[0], ws[0]) < 4) return false;
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
        mlx.mlx_fast_metal_kernel_new("msv_gqmm_tile_scan", in_vec, out_vec, @embedFile("kernels/gather_qmm_nax_scan.metal"), @embedFile("kernels/gather_qmm_nax_header.metal"), true, false)
    else
        mlx.mlx_fast_metal_kernel_new("msv_gqmm_affine", in_vec, out_vec, @embedFile("kernels/gather_qmm_nax.metal"), @embedFile("kernels/gather_qmm_nax_header.metal"), true, false);
    if (value.ctx == null) return error.MetalKernelCompileFailed;
    if (scan) scan_kernel = value else mm_kernel = value;
    return value;
}

fn mapKernel(paired: bool) !mlx.mlx_fast_metal_kernel {
    const slot = if (paired) &paired_kernel else &mapped_kernel;
    if (slot.*) |v| return v;
    const names = [_][*:0]const u8{ "x", "w", "scales", "biases", "up_w", "up_scales", "up_biases", "tiles", "tile_count", "params", "sigtab", "rmap" };
    const outs = [_][*:0]const u8{"y"};
    const ins_vec = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(ins_vec);
    const outs_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(outs_vec);
    const header = @embedFile("kernels/gather_qmm_nax_header.metal") ++ @embedFile("kernels/gather_qmm_nax_mapped_header.metal");
    const value = mlx.mlx_fast_metal_kernel_new(if (paired) "msv_gqmm_affine_pair" else "msv_gqmm_affine_map", ins_vec, outs_vec, @embedFile("kernels/gather_qmm_nax_mapped.metal"), header, true, false);
    if (value.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = value;
    return value;
}

const Tiles = struct {
    tiles: mlx.mlx_array,
    count: mlx.mlx_array,
    max_tiles: c_int,

    fn deinit(self: Tiles) void {
        _ = mlx.mlx_array_free(self.tiles);
        _ = mlx.mlx_array_free(self.count);
    }
};

/// The expert-run tile pre-pass both matmul kernels read. Configs are built per call: their
/// output shapes carry the row count, so a cache keyed on it grew with every prompt length.
fn scanTiles(idx: mlx.mlx_array, m: c_int, e: c_int, bm: c_int, s: mlx.mlx_stream) !Tiles {
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
    if (key.mapped) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPI", if (key.paired) 1 + @as(c_int, @intCast(key.limit)) else 0));
    return cfg;
}

/// `limit`: a `swiglu_limit` the epilogue clamps at (whole numbers; 0 = none).
const Pair = struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, sigtab: mlx.mlx_array, limit: u32 = 0 };

/// An MXFP4 bank has no biases; the kernel signature still binds the slot. 16
/// elements: MLX binds an array under 8 in `constant`, not `device`.
var no_bias: mlx.mlx_array = .{ .ctx = null };
fn biasArg(bi: mlx.mlx_array) mlx.mlx_array {
    if (bi.ctx != null) return bi;
    if (no_bias.ctx == null) no_bias = mlx.mlx_array_new_data(&@as([16]u16, @splat(0)), &[_]c_int{16}, 1, .bfloat16);
    return no_bias;
}

fn launchMapped(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, pair: ?Pair, bits: u32, group: u32, mx: bool, p: Plan, s: mlx.mlx_stream) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const m = mlx.getShape(idx)[0];
    const e = ws[0];
    const n = ws[1];
    const k = xs[2];
    const t = try scanTiles(idx, m, e, p.bm, s);
    defer t.deinit();
    const params = mlx.mlx_array_new_data(&[_]c_int{ if (pair != null) n * 2 else n, k }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const pw = if (pair) |v| v.w else w;
    const ps = if (pair) |v| v.sc else sc;
    const pb = if (pair) |v| v.bi else bi;
    const sigtab = if (pair) |v| v.sigtab else biasArg(.{ .ctx = null }); // unread without a pair
    const inputs = [_]mlx.mlx_array{ x, w, sc, biasArg(bi), pw, ps, biasArg(pb), t.tiles, t.count, params, sigtab, row_map };
    const cfg = try mmConfig(.{ .rows = m, .n = n, .k = k, .max_tiles = t.max_tiles, .plan = p, .bits = bits, .group = group, .mx = mx, .mapped = true, .paired = pair != null, .limit = if (pair) |v| v.limit else 0 });
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
    try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(@intCast(key.group)), mlx.mlx_optional_int.some(@intCast(key.bits)), modeName(key.mx), .{}, s));
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
    try mlx.check(mlx.mlx_gather_qmm(&ref, x, w, sc, bi, .{}, ids, true, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), modeName(mx), true, s));
    return ref;
}

/// `got` against a dequantized fp32 gather-matmul, within 1/64 of the reference's peak.
fn floatReferenceClose(got: mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array, bits: u32, group: u32, mx: bool, s: mlx.mlx_stream) !bool {
    var wd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wd);
    try mlx.check(mlx.mlx_dequantize(&wd, w, sc, bi, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), modeName(mx), .{}, .{ .value = .float32, .has_value = true }, s));
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

fn modeName(mx: bool) [*:0]const u8 {
    return if (mx) "mxfp4" else "affine";
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
    if (!nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, mx, null)) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const p = plan(xs[0], ws[0], xs[2], ws[1], false);
    const key: CanaryKey = .{ .plan = p, .bits = bits, .group = group, .align_n = @rem(ws[1], 64) == 0, .align_k = @rem(xs[2], p.bk) == 0, .mx = mx };
    if (!armed(key, s)) return null;
    const out = try launch(x, w, sc, bi, idx, bits, group, mx, p, s);
    if (!engaged_logged) {
        engaged_logged = true;
        log.info("[gather-nax] engaged: {s} bm={d} bk={d} M={d} E={d} N={d} K={d}\n", .{ @tagName(p.sched), p.bm, p.bk, xs[0], ws[0], ws[1], xs[2] });
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

fn canaryMapped(key: CanaryKey, paired: bool, s: mlx.mlx_stream) !bool {
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
    var pos: usize = 0;
    for (counts, 0..) |count, expert| {
        for (0..count) |_| {
            ids_data[pos] = @intCast(expert);
            map_data[pos] = @intCast((pos * 7 + 3) % tokens);
            pos += 1;
        }
    }
    const ids = mlx.mlx_array_new_data(&ids_data, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const map = mlx.mlx_array_new_data(&map_data, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(map);
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
    try mlx.check(mlx.mlx_quantize(&q, wf, group, bits, modeName(key.mx), .{}, s));
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
    try mlx.check(mlx.mlx_quantize(&uq, uf, group, bits, modeName(key.mx), .{}, s));
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
    const sigtab = try @import("hc_prefill.zig").sigmoidTable(s);
    const got = try launchMapped(x, map, w, sc, bi, ids, .{ .w = uw, .sc = us, .bi = ub, .sigtab = sigtab }, key.bits, key.group, key.mx, key.plan, s);
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
    return arraysEqual(got, ref, s);
}

fn armedMapped(key: CanaryKey, paired: bool, s: mlx.mlx_stream) bool {
    const cache = if (paired) &paired_canaries else &mapped_canaries;
    if (cache.get(key)) |ok| return ok;
    const had_error = mlx.errorPending();
    const ok = canaryMapped(key, paired, s) catch blk: {
        mlx.dropLatchedErrorUnless(had_error);
        break :blk false;
    };
    cache.put(std.heap.c_allocator, key, ok) catch return false;
    if (!ok) log.warn("[gather-nax] {s} canary mismatch: {d}-bit group={d} {s} bm={d} bk={d}\n", .{ if (paired) @as([]const u8, "paired") else "mapped", key.bits, key.group, @tagName(key.plan.sched), key.plan.bm, key.plan.bk });
    return ok;
}

fn mappedKey(x: mlx.mlx_array, w: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, paired: bool) CanaryKey {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const p = plan(mlx.getShape(idx)[0], ws[0], xs[2], ws[1], paired);
    return .{ .plan = p, .bits = bits, .group = group, .align_n = @rem(ws[1], 64) == 0, .align_k = @rem(xs[2], p.bk) == 0, .mx = mx };
}

pub fn sortedGatherMapped(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, mx, row_map)) return null;
    const key = mappedKey(x, w, idx, bits, group, mx, false);
    if (!armed(key, s) or !armedMapped(key, false, s)) return null;
    const out = try launchMapped(x, row_map, w, sc, bi, idx, null, bits, group, mx, key.plan, s);
    if (!mapped_engaged_logged) {
        mapped_engaged_logged = true;
        log.info("[gather-nax] row map engaged: M={d} E={d} N={d} K={d}\n", .{ mlx.getShape(idx)[0], mlx.getShape(w)[0], mlx.getShape(w)[1], mlx.getShape(x)[2] });
    }
    return out;
}

pub fn sortedGateUp(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, up_w: mlx.mlx_array, up_sc: mlx.mlx_array, up_bi: mlx.mlx_array, idx: mlx.mlx_array, sigtab: mlx.mlx_array, limit: u32, bits: u32, group: u32, mx: bool, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, mx, row_map) or !supported(x, up_w, up_sc, up_bi, idx, bits, group, mx, row_map)) return null;
    if (!std.mem.eql(c_int, mlx.getShape(w), mlx.getShape(up_w)) or @rem(mlx.getShape(w)[1], 32) != 0) return null;
    if (sigtab.ctx == null or mlx.mlx_array_dtype(sigtab) != .bfloat16 or mlx.mlx_array_size(sigtab) != 65536) return null;
    const key = mappedKey(x, w, idx, bits, group, mx, true);
    if (!armed(key, s) or !armedMapped(key, false, s) or !armedMapped(key, true, s)) return null;
    const out = try launchMapped(x, row_map, w, sc, bi, idx, .{ .w = up_w, .sc = up_sc, .bi = up_bi, .sigtab = sigtab, .limit = limit }, bits, group, mx, key.plan, s);
    if (!paired_engaged_logged) {
        paired_engaged_logged = true;
        log.info("[gather-nax] paired SwiGLU engaged: M={d} E={d} N={d} K={d}\n", .{ mlx.getShape(idx)[0], mlx.getShape(w)[0], mlx.getShape(w)[1], mlx.getShape(x)[2] });
    }
    return out;
}

fn requireNax() !void {
    mlx.installErrorHandler();
    if (mlx.noGpuBackend() or !@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
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
    try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), modeName(mx), .{}, s));
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
    const s = mlx.gpuStream();
    const cases = [_]struct { rows: c_int, experts: c_int, n: c_int, k: c_int, bits: u32, group: u32 }{
        .{ .rows = 81920, .experts = 512, .n = 640, .k = 2560, .bits = 4, .group = 64 },
        .{ .rows = 81920, .experts = 512, .n = 2560, .k = 640, .bits = 4, .group = 64 },
        .{ .rows = 33010, .experts = 512, .n = 64, .k = 128, .bits = 4, .group = 64 },
        .{ .rows = 2048, .experts = 32, .n = 64, .k = 128, .bits = 8, .group = 32 },
        .{ .rows = 2048, .experts = 32, .n = 96, .k = 256, .bits = 4, .group = 128 },
        .{ .rows = 4096, .experts = 32, .n = 96, .k = 96, .bits = 4, .group = 32 },
    };
    for (cases) |case| try testCase(case.rows, case.experts, case.n, case.k, case.bits, case.group, false, s);
}

test "segmented NAX sorted gather matches stock bit for bit on MXFP4 banks (MiMo shapes)" {
    try requireNax();
    const s = mlx.gpuStream();
    // 65536 rows = an 8192-token top-8 chunk, past the 32768-row offset the stock kernel once overflowed.
    const cases = [_]struct { rows: c_int, experts: c_int, n: c_int, k: c_int }{
        .{ .rows = 65536, .experts = 256, .n = 2048, .k = 4096 },
        .{ .rows = 65536, .experts = 256, .n = 4096, .k = 2048 },
        .{ .rows = 2048, .experts = 32, .n = 64, .k = 128 },
    };
    for (cases) |c| try testCase(c.rows, c.experts, c.n, c.k, 4, 32, true, s);
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
        try mlx.check(mlx.mlx_quantize(&gate_q, gate_f, group, bits, modeName(case.mx), .{}, s));
        try mlx.check(mlx.mlx_quantize(&up_q, up_f, group, bits, modeName(case.mx), .{}, s));
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
        const sigtab = try @import("hc_prefill.zig").sigmoidTable(s);
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
        const clamp_ref = try @import("transformer.zig").clampedSwiGLU(s, plain_g, plain_u, 1.0);
        defer _ = mlx.mlx_array_free(clamp_ref);
        try expectBitEqual(clamp_ref, clamped, s);
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
    const raw = std.c.getenv("MLX_SERVE_GQMM_UBENCH") orelse return buf[0..0];
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

test "segmented NAX sorted gather µbench vs stock at 8192-token chunks (MLX_SERVE_GQMM_UBENCH=1|sweep)" {
    if (std.c.getenv("MLX_SERVE_GQMM_UBENCH") == null) return error.SkipZigTest;
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const tokens: c_int = 8192;
    const alloc = std.testing.allocator;
    const sigtab = try @import("hc_prefill.zig").sigmoidTable(s);
    // MiMo-V2.6 (MXFP4, top-8) and Qwen3.8-Flash-Next (4-bit g64, top-10) expert shapes.
    const Shape = struct { name: []const u8, experts: c_int, top_k: c_int, n: c_int, k: c_int, bits: u32, group: u32, mx: bool, paired: bool = false };
    const shapes = [_]Shape{
        .{ .name = "mimo gate 4096->2048", .experts = 256, .top_k = 8, .n = 2048, .k = 4096, .bits = 4, .group = 32, .mx = true },
        .{ .name = "mimo down 2048->4096", .experts = 256, .top_k = 8, .n = 4096, .k = 2048, .bits = 4, .group = 32, .mx = true },
        .{ .name = "mimo gate+up+SwiGLU", .experts = 256, .top_k = 8, .n = 2048, .k = 4096, .bits = 4, .group = 32, .mx = true, .paired = true },
        .{ .name = "flash-next gate 2560->640", .experts = 512, .top_k = 10, .n = 640, .k = 2560, .bits = 4, .group = 64, .mx = false },
        .{ .name = "flash-next gate+up+SwiGLU", .experts = 512, .top_k = 10, .n = 640, .k = 2560, .bits = 4, .group = 64, .mx = false, .paired = true },
        .{ .name = "flash-next down 640->2560", .experts = 512, .top_k = 10, .n = 2560, .k = 640, .bits = 4, .group = 64, .mx = false },
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
