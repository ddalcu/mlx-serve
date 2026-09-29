// SPDX-License-Identifier: Apache-2.0
// Ported from oMLX (jundot/omlx) omlx/patches/m5_gather_qmm_nax.py @ d6b2b92.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const Plan = struct { sched: enum { seg, db }, bm: c_int, bk: c_int, gx: c_int, pad: c_int };
const CanaryKey = struct { plan: Plan, bits: u32, group: u32, align_n: bool, align_k: bool };
const ScanKey = struct { rows: c_int, experts: c_int, max_tiles: c_int, bm: c_int };
const MmKey = struct { rows: c_int, n: c_int, k: c_int, max_tiles: c_int, plan: Plan, bits: u32, group: u32, mapped: bool = false, paired: bool = false };
var scan_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mm_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mapped_kernel: ?mlx.mlx_fast_metal_kernel = null;
var paired_kernel: ?mlx.mlx_fast_metal_kernel = null;
var scan_cfg: std.AutoHashMapUnmanaged(ScanKey, mlx.mlx_fast_metal_kernel_config) = .{};
var mm_cfg: std.AutoHashMapUnmanaged(MmKey, mlx.mlx_fast_metal_kernel_config) = .{};
var canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var mapped_canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var paired_canaries: std.AutoHashMapUnmanaged(CanaryKey, bool) = .{};
var engaged_logged = false;
var mapped_engaged_logged = false;
var paired_engaged_logged = false;

fn plan(rows: c_int, experts: c_int, k: c_int, n: c_int) Plan {
    if (@rem(k, 64) != 0 or @rem(n, 64) != 0) return .{ .sched = .seg, .bm = 64, .bk = 64, .gx = 0, .pad = 0 };
    const per_expert = @divTrunc(rows, @max(experts, 1));
    if (per_expert < 36 or (k < 1024 and per_expert < 120)) return .{ .sched = .db, .bm = 64, .bk = 64, .gx = 0, .pad = 0 };
    if (k < 1024) return .{ .sched = .seg, .bm = 96, .bk = 128, .gx = 32, .pad = 0 };
    if (per_expert < 48) return .{ .sched = .db, .bm = 64, .bk = 64, .gx = 32, .pad = 0 };
    if (per_expert < 96) return .{ .sched = .db, .bm = 96, .bk = 64, .gx = 32, .pad = 0 };
    return .{ .sched = .seg, .bm = 128, .bk = 128, .gx = 32, .pad = 8192 };
}

pub fn enabled() bool {
    const raw = std.c.getenv("MLX_SERVE_GATHER_NAX") orelse return true;
    return !std.mem.eql(u8, std.mem.span(raw), "0");
}

fn supported(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, row_map: ?mlx.mlx_array) bool {
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null or idx.ctx == null) return false;
    if (bits != 4 and bits != 8) return false;
    if (group != 32 and group != 64 and group != 128) return false;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(idx) != .uint32) return false;
    if (mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    if (mlx.mlx_array_ndim(x) != 3 or mlx.mlx_array_ndim(w) != 3 or mlx.mlx_array_ndim(sc) != 3 or mlx.mlx_array_ndim(bi) != 3 or mlx.mlx_array_ndim(idx) != 1) return false;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
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
    return ss[0] == ws[0] and ss[1] == ws[1] and ss[2] == @divTrunc(xs[2], @as(c_int, @intCast(group))) and std.mem.eql(c_int, ss, bs);
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
        mlx.mlx_fast_metal_kernel_new("msv_gqmm_tile_scan", in_vec, out_vec, @embedFile("kernels/gather_qmm_nax_scan.metal"), @embedFile("kernels/gather_qmm_nax_header.metal"), false, false)
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

fn scanConfig(key: ScanKey) !mlx.mlx_fast_metal_kernel_config {
    if (scan_cfg.get(key)) |v| return v;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const tile_shape = [_]c_int{key.max_tiles * 4};
    const one = [_]c_int{1};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &tile_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &one, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BM", key.bm));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MAXE", 2048));
    try scan_cfg.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

fn mmConfig(key: MmKey) !mlx.mlx_fast_metal_kernel_config {
    if (mm_cfg.get(key)) |v| return v;
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
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SCHED", if (key.plan.sched == .db) 1 else 0));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ALIGN_N", @rem((if (key.paired) key.n * 2 else key.n), 64) == 0));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ALIGN_K", @rem(key.k, key.plan.bk) == 0));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BM", key.plan.bm));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BK", key.plan.bk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GX", key.plan.gx));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PAD", key.plan.pad));
    if (key.mapped) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPI", @intFromBool(key.paired)));
    try mm_cfg.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

const Pair = struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, sigtab: mlx.mlx_array };

fn launchMapped(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, pair: ?Pair, bits: u32, group: u32, p: Plan, s: mlx.mlx_stream) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const m = mlx.getShape(idx)[0];
    const e = ws[0];
    const n = ws[1];
    const k = xs[2];
    const max_tiles = @divTrunc(m + p.bm - 1, p.bm) + @min(e, m);
    const scan_params = mlx.mlx_array_new_data(&[_]c_int{ m, e, max_tiles }, &[_]c_int{3}, 1, .int32);
    defer _ = mlx.mlx_array_free(scan_params);
    const scan_inputs = [_]mlx.mlx_array{ idx, scan_params };
    const scan_in = mlx.mlx_vector_array_new_data(&scan_inputs, scan_inputs.len);
    defer _ = mlx.mlx_vector_array_free(scan_in);
    var scan_out = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(scan_out);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&scan_out, try kernel(true), scan_in, try scanConfig(.{ .rows = m, .experts = e, .max_tiles = max_tiles, .bm = p.bm }), s));
    const tiles = try outputAt(scan_out, 0);
    defer _ = mlx.mlx_array_free(tiles);
    const tile_count = try outputAt(scan_out, 1);
    defer _ = mlx.mlx_array_free(tile_count);
    const params = mlx.mlx_array_new_data(&[_]c_int{ if (pair != null) n * 2 else n, k }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const pw = if (pair) |v| v.w else w;
    const ps = if (pair) |v| v.sc else sc;
    const pb = if (pair) |v| v.bi else bi;
    const sigtab = if (pair) |v| v.sigtab else sc;
    const inputs = [_]mlx.mlx_array{ x, w, sc, bi, pw, ps, pb, tiles, tile_count, params, sigtab, row_map };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try mapKernel(pair != null), in_vec, try mmConfig(.{ .rows = m, .n = n, .k = k, .max_tiles = max_tiles, .plan = p, .bits = bits, .group = group, .mapped = true, .paired = pair != null }), s));
    return outputAt(outs, 0);
}

fn outputAt(vec: mlx.mlx_vector_array, index: usize) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, vec, index));
    return out;
}

fn launch(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, p: Plan, s: mlx.mlx_stream) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const m = xs[0];
    const e = ws[0];
    const n = ws[1];
    const k = xs[2];
    const max_tiles = @divTrunc(m + p.bm - 1, p.bm) + @min(e, m);
    const scan_params_data = [_]c_int{ m, e, max_tiles };
    const scan_params_shape = [_]c_int{3};
    const scan_params = mlx.mlx_array_new_data(&scan_params_data, &scan_params_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(scan_params);
    const scan_inputs = [_]mlx.mlx_array{ idx, scan_params };
    const scan_in = mlx.mlx_vector_array_new_data(&scan_inputs, scan_inputs.len);
    defer _ = mlx.mlx_vector_array_free(scan_in);
    var scan_out = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(scan_out);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&scan_out, try kernel(true), scan_in, try scanConfig(.{ .rows = m, .experts = e, .max_tiles = max_tiles, .bm = p.bm }), s));
    if (mlx.mlx_vector_array_size(scan_out) != 2) return error.MetalKernelBadOutputCount;
    const tiles = try outputAt(scan_out, 0);
    defer _ = mlx.mlx_array_free(tiles);
    const tile_count = try outputAt(scan_out, 1);
    defer _ = mlx.mlx_array_free(tile_count);
    const params_data = [_]c_int{ n, k };
    const params_shape = [_]c_int{2};
    const params = mlx.mlx_array_new_data(&params_data, &params_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const inputs = [_]mlx.mlx_array{ x, w, sc, bi, tiles, tile_count, params };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try kernel(false), in_vec, try mmConfig(.{ .rows = m, .n = n, .k = k, .max_tiles = max_tiles, .plan = p, .bits = bits, .group = group }), s));
    if (mlx.mlx_vector_array_size(outs) != 1) return error.MetalKernelBadOutputCount;
    return outputAt(outs, 0);
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
    try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(@intCast(key.group)), mlx.mlx_optional_int.some(@intCast(key.bits)), "affine", .{}, s));
    const w = try outputAt(q, 0);
    defer _ = mlx.mlx_array_free(w);
    const sc = try outputAt(q, 1);
    defer _ = mlx.mlx_array_free(sc);
    const bi = try outputAt(q, 2);
    defer _ = mlx.mlx_array_free(bi);
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
    const got = try launch(x, w, sc, bi, ids, key.bits, key.group, key.plan, s);
    defer _ = mlx.mlx_array_free(got);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    if (@rem(k, 64) == 0) {
        try mlx.check(mlx.mlx_gather_qmm(&ref, x, w, sc, bi, .{}, ids, true, mlx.mlx_optional_int.some(@intCast(key.group)), mlx.mlx_optional_int.some(@intCast(key.bits)), "affine", true, s));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, got, ref, false, s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        return same;
    }
    // Stock NAX has a K-tail read bug. Compare against a dequantized fp32 product.
    var wd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wd);
    try mlx.check(mlx.mlx_dequantize(&wd, w, sc, bi, mlx.mlx_optional_int.some(@intCast(key.group)), mlx.mlx_optional_int.some(@intCast(key.bits)), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
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
    try mlx.check(mlx.mlx_matmul(&ref, xf, transposed, s));
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

fn armed(key: CanaryKey, s: mlx.mlx_stream) bool {
    if (canaries.get(key)) |ok| return ok;
    const ok = canary(key, s) catch blk: {
        var msg: [256]u8 = undefined;
        _ = mlx.takeError(&msg);
        break :blk false;
    };
    canaries.put(std.heap.c_allocator, key, ok) catch return false;
    if (!ok) log.warn("[gather-nax] disabled for {d}-bit group={d} {s} bm={d} bk={d} aligned-N={any} aligned-K={any}: canary mismatch\n", .{ key.bits, key.group, @tagName(key.plan.sched), key.plan.bm, key.plan.bk, key.align_n, key.align_k });
    return ok;
}

/// Sorted rhs indices must be one contiguous run per expert.
pub fn sortedGather(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!enabled() or !nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, null)) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const p = plan(xs[0], ws[0], xs[2], ws[1]);
    const key: CanaryKey = .{ .plan = p, .bits = bits, .group = group, .align_n = @rem(ws[1], 64) == 0, .align_k = @rem(xs[2], p.bk) == 0 };
    if (!armed(key, s)) return null;
    const out = try launch(x, w, sc, bi, idx, bits, group, p, s);
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
    try mlx.check(mlx.mlx_quantize(&q, wf, group, bits, "affine", .{}, s));
    const w = try outputAt(q, 0);
    defer _ = mlx.mlx_array_free(w);
    const sc = try outputAt(q, 1);
    defer _ = mlx.mlx_array_free(sc);
    const bi = try outputAt(q, 2);
    defer _ = mlx.mlx_array_free(bi);
    const gate_ref = try launch(x_rep, w, sc, bi, ids, key.bits, key.group, key.plan, s);
    defer _ = mlx.mlx_array_free(gate_ref);
    if (!paired) {
        const got = try launchMapped(x, map, w, sc, bi, ids, null, key.bits, key.group, key.plan, s);
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
    try mlx.check(mlx.mlx_quantize(&uq, uf, group, bits, "affine", .{}, s));
    const uw = try outputAt(uq, 0);
    defer _ = mlx.mlx_array_free(uw);
    const us = try outputAt(uq, 1);
    defer _ = mlx.mlx_array_free(us);
    const ub = try outputAt(uq, 2);
    defer _ = mlx.mlx_array_free(ub);
    const up_ref = try launch(x_rep, uw, us, ub, ids, key.bits, key.group, key.plan, s);
    defer _ = mlx.mlx_array_free(up_ref);
    const sigtab = try @import("hc_prefill.zig").sigmoidTable(s);
    const got = try launchMapped(x, map, w, sc, bi, ids, .{ .w = uw, .sc = us, .bi = ub, .sigtab = sigtab }, key.bits, key.group, key.plan, s);
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
    const ok = canaryMapped(key, paired, s) catch blk: {
        var msg: [256]u8 = undefined;
        _ = mlx.takeError(&msg);
        break :blk false;
    };
    cache.put(std.heap.c_allocator, key, ok) catch return false;
    if (!ok) log.warn("[gather-nax] {s} canary mismatch: {d}-bit group={d} {s} bm={d} bk={d}\n", .{ if (paired) @as([]const u8, "paired") else "mapped", key.bits, key.group, @tagName(key.plan.sched), key.plan.bm, key.plan.bk });
    return ok;
}

fn mappedKey(x: mlx.mlx_array, w: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32) CanaryKey {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const p = plan(mlx.getShape(idx)[0], ws[0], xs[2], ws[1]);
    return .{ .plan = p, .bits = bits, .group = group, .align_n = @rem(ws[1], 64) == 0, .align_k = @rem(xs[2], p.bk) == 0 };
}

pub fn sortedGatherMapped(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, idx: mlx.mlx_array, bits: u32, group: u32, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!enabled() or !nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, row_map)) return null;
    const key = mappedKey(x, w, idx, bits, group);
    if (!armed(key, s) or !armedMapped(key, false, s)) return null;
    const out = try launchMapped(x, row_map, w, sc, bi, idx, null, bits, group, key.plan, s);
    if (!mapped_engaged_logged) {
        mapped_engaged_logged = true;
        log.info("[gather-nax] row map engaged: M={d} E={d} N={d} K={d}\n", .{ mlx.getShape(idx)[0], mlx.getShape(w)[0], mlx.getShape(w)[1], mlx.getShape(x)[2] });
    }
    return out;
}

pub fn sortedGateUp(x: mlx.mlx_array, row_map: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, up_w: mlx.mlx_array, up_sc: mlx.mlx_array, up_bi: mlx.mlx_array, idx: mlx.mlx_array, sigtab: mlx.mlx_array, bits: u32, group: u32, nax_available: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!enabled() or !nax_available or !mlx.streamIsGpu(s)) return null;
    if (!supported(x, w, sc, bi, idx, bits, group, row_map) or !supported(x, up_w, up_sc, up_bi, idx, bits, group, row_map)) return null;
    if (!std.mem.eql(c_int, mlx.getShape(w), mlx.getShape(up_w)) or @rem(mlx.getShape(w)[1], 32) != 0) return null;
    if (sigtab.ctx == null or mlx.mlx_array_dtype(sigtab) != .bfloat16 or mlx.mlx_array_size(sigtab) != 65536) return null;
    const key = mappedKey(x, w, idx, bits, group);
    if (!armed(key, s) or !armedMapped(key, false, s) or !armedMapped(key, true, s)) return null;
    const out = try launchMapped(x, row_map, w, sc, bi, idx, .{ .w = up_w, .sc = up_sc, .bi = up_bi, .sigtab = sigtab }, bits, group, key.plan, s);
    if (!paired_engaged_logged) {
        paired_engaged_logged = true;
        log.info("[gather-nax] paired SwiGLU engaged: M={d} E={d} N={d} K={d}\n", .{ mlx.getShape(idx)[0], mlx.getShape(w)[0], mlx.getShape(w)[1], mlx.getShape(x)[2] });
    }
    return out;
}

fn testGpuUsable() bool {
    mlx.installErrorHandler();
    if (mlx.noGpuBackend()) return false;
    const s = mlx.gpuStream();
    if (s.ctx == null) {
        var msg: [256]u8 = undefined;
        _ = mlx.takeError(&msg);
        return false;
    }
    defer _ = mlx.mlx_stream_free(s);
    const shape = [_]c_int{1};
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    if (mlx.mlx_zeros(&a, &shape, 1, .float32, s) != 0 or mlx.mlx_array_eval(a) != 0) {
        var msg: [256]u8 = undefined;
        _ = mlx.takeError(&msg);
        return false;
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) {
        var msg: [256]u8 = undefined;
        _ = mlx.takeError(&msg);
        return false;
    }
    var arch: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&arch, info, "architecture") != 0) {
        var msg: [256]u8 = undefined;
        _ = mlx.takeError(&msg);
        return false;
    }
    return std.mem.startsWith(u8, std.mem.span(arch), "applegpu_g17");
}

test "segmented NAX sorted gather planner selects measured shape classes" {
    const cases = [_]struct { rows: c_int, experts: c_int, k: c_int, n: c_int, want: Plan }{
        .{ .rows = 81920, .experts = 512, .k = 2560, .n = 640, .want = .{ .sched = .seg, .bm = 128, .bk = 128, .gx = 32, .pad = 8192 } },
        .{ .rows = 81920, .experts = 512, .k = 640, .n = 2560, .want = .{ .sched = .seg, .bm = 96, .bk = 128, .gx = 32, .pad = 0 } },
        .{ .rows = 33010, .experts = 512, .k = 2560, .n = 640, .want = .{ .sched = .db, .bm = 96, .bk = 64, .gx = 32, .pad = 0 } },
        .{ .rows = 129, .experts = 8, .k = 96, .n = 64, .want = .{ .sched = .seg, .bm = 64, .bk = 64, .gx = 0, .pad = 0 } },
    };
    for (cases) |case| try std.testing.expectEqualDeep(case.want, plan(case.rows, case.experts, case.k, case.n));
}

test "segmented NAX sorted gather matches MLX on ragged expert runs" {
    if (!testGpuUsable()) return error.SkipZigTest;
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
    const got = (try sortedGather(x, w, sc, bi, idx, 4, 64, true, s)) orelse return error.KernelDeclinedCanary;
    defer _ = mlx.mlx_array_free(got);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_gather_qmm(&ref, x, w, sc, bi, .{}, idx, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, got, ref, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try std.testing.expect(same);
}

fn testFloatReferenceClose(got: mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array, bits: u32, group: u32, s: mlx.mlx_stream) !void {
    var wd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wd);
    try mlx.check(mlx.mlx_dequantize(&wd, w, sc, bi, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
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
    try std.testing.expect(std.math.isFinite(err) and err <= scale / 64);
}

fn testCase(rows: c_int, experts: c_int, n: c_int, k: c_int, bits: u32, group: u32, s: mlx.mlx_stream) !void {
    const alloc = std.testing.allocator;
    const ids_data = try alloc.alloc(u32, @intCast(rows));
    defer alloc.free(ids_data);
    var pos: usize = 0;
    for (0..@intCast(experts)) |expert| {
        const count: usize = if (rows == 81920)
            (if (expert < 16) 0 else if (expert < 32) 320 else if (expert < 272) 150 else 170)
        else if (expert < 4) 0 else @intCast(@divTrunc(rows, experts - 4) + @as(c_int, if (expert - 4 < @as(usize, @intCast(@rem(rows, experts - 4)))) 1 else 0));
        for (0..@min(count, ids_data.len - pos)) |_| {
            ids_data[pos] = @intCast(expert);
            pos += 1;
        }
    }
    while (pos < ids_data.len) : (pos += 1) ids_data[pos] = @intCast(experts - 1);
    const ids_shape = [_]c_int{rows};
    const ids = mlx.mlx_array_new_data(ids_data.ptr, &ids_shape, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 0x2267));
    const w_shape = [_]c_int{ experts, n, k };
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_random_normal(&wf, &w_shape, 3, .bfloat16, 0, 0.05, key, s));
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
    const w = try outputAt(q, 0);
    defer _ = mlx.mlx_array_free(w);
    const sc = try outputAt(q, 1);
    defer _ = mlx.mlx_array_free(sc);
    const bi = try outputAt(q, 2);
    defer _ = mlx.mlx_array_free(bi);
    const x_shape = [_]c_int{ rows, 1, k };
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &x_shape, 3, .bfloat16, 0, 0.5, key, s));
    const got = (try sortedGather(x, w, sc, bi, ids, bits, group, true, s)) orelse return error.KernelDeclinedTestShape;
    defer _ = mlx.mlx_array_free(got);
    if (@rem(k, 64) != 0) {
        // Pinned stock NAX has the K-tail bug, so this arm uses fp32 dequantized matmul.
        try testFloatReferenceClose(got, x, w, sc, bi, ids, bits, group, s);
        return;
    }
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    // This pin includes MLX's sorted-row offset fix, so >32768 rows can use stock.
    try mlx.check(mlx.mlx_gather_qmm(&ref, x, w, sc, bi, .{}, ids, true, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", true, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, got, ref, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try std.testing.expect(same);
}

test "segmented NAX sorted gather matches stock across Flash Next and quantization shapes" {
    if (!testGpuUsable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const cases = [_]struct { rows: c_int, experts: c_int, n: c_int, k: c_int, bits: u32, group: u32 }{
        .{ .rows = 81920, .experts = 512, .n = 640, .k = 2560, .bits = 4, .group = 64 },
        .{ .rows = 81920, .experts = 512, .n = 2560, .k = 640, .bits = 4, .group = 64 },
        .{ .rows = 33010, .experts = 512, .n = 64, .k = 128, .bits = 4, .group = 64 },
        .{ .rows = 2048, .experts = 32, .n = 64, .k = 128, .bits = 8, .group = 32 },
        .{ .rows = 2048, .experts = 32, .n = 96, .k = 256, .bits = 4, .group = 128 },
        .{ .rows = 4096, .experts = 32, .n = 96, .k = 96, .bits = 4, .group = 32 },
    };
    for (cases) |case| try testCase(case.rows, case.experts, case.n, case.k, case.bits, case.group, s);
}

test "segmented NAX sorted gather row map and paired SwiGLU match composed projections" {
    if (!testGpuUsable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const cases = [_]struct { tokens: c_int, bits: u32, group: u32 }{
        .{ .tokens = 8192, .bits = 4, .group = 64 },
        .{ .tokens = 3301, .bits = 4, .group = 64 },
        .{ .tokens = 3301, .bits = 8, .group = 32 },
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
        try mlx.check(mlx.mlx_quantize(&gate_q, gate_f, group, bits, "affine", .{}, s));
        try mlx.check(mlx.mlx_quantize(&up_q, up_f, group, bits, "affine", .{}, s));
        const gw = try outputAt(gate_q, 0);
        defer _ = mlx.mlx_array_free(gw);
        const gs = try outputAt(gate_q, 1);
        defer _ = mlx.mlx_array_free(gs);
        const gb = try outputAt(gate_q, 2);
        defer _ = mlx.mlx_array_free(gb);
        const uw = try outputAt(up_q, 0);
        defer _ = mlx.mlx_array_free(uw);
        const us = try outputAt(up_q, 1);
        defer _ = mlx.mlx_array_free(us);
        const ub = try outputAt(up_q, 2);
        defer _ = mlx.mlx_array_free(ub);
        const plain_g = (try sortedGather(x_rep, gw, gs, gb, ids, case.bits, case.group, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(plain_g);
        const mapped_g = (try sortedGatherMapped(x, row_map, gw, gs, gb, ids, case.bits, case.group, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(mapped_g);
        try expectBitEqual(plain_g, mapped_g, s);
        const plain_u = (try sortedGather(x_rep, uw, us, ub, ids, case.bits, case.group, true, s)) orelse return error.KernelDeclinedTestShape;
        defer _ = mlx.mlx_array_free(plain_u);
        const sigtab = try @import("hc_prefill.zig").sigmoidTable(s);
        const fused = (try sortedGateUp(x, row_map, gw, gs, gb, uw, us, ub, ids, sigtab, case.bits, case.group, true, s)) orelse return error.KernelDeclinedTestShape;
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
    }
}
