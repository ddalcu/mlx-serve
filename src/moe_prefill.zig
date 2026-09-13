//! Exact prefill grouping and inverse/weighted reduction for Qwen4 E512/K10.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
var kernels: [4]?mlx.mlx_fast_metal_kernel = @splat(null);
var announced = false;
var captured_routes: u32 = 0;
var live_calls: u32 = 0;
var mpp_kernels: [3]?mlx.mlx_fast_metal_kernel = @splat(null);
var mpp_announced = false;

pub fn mppEnabled() bool {
    if (@import("prefill_experiment.zig").moeMpp()) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_MOE_PREFILL_MPP") orelse return false;
    return raw[0] == '1';
}

fn mppKernel(which: usize) !mlx.mlx_fast_metal_kernel {
    if (mpp_kernels[which]) |k| return k;
    const names: []const [*:0]const u8 = switch (which) {
        0 => &.{"sorted"},
        1 => &.{ "x", "order" },
        else => &.{ "x_rep", "w", "scales", "biases", "indices_input", "tiles", "up_w", "up_scales", "up_biases", "sigtab" },
    };
    const outs = mlx.mlx_vector_string_new_data(if (which == 0) &[_][*:0]const u8{"tiles"} else &[_][*:0]const u8{"out"}, 1);
    defer _ = mlx.mlx_vector_string_free(outs);
    const ins = mlx.mlx_vector_string_new_data(names.ptr, names.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const labels = [_][*:0]const u8{ "msv_moe_mpp_tiles", "msv_moe_mpp_gather", "msv_moe_mpp_gateup" };
    const sources = [_][*:0]const u8{ @embedFile("kernels/moe_prefill_mpp_tiles.metal"), @embedFile("kernels/moe_prefill_mpp_gather.metal"), @embedFile("kernels/moe_prefill_mpp_gateup.metal") };
    const k = mlx.mlx_fast_metal_kernel_new(labels[which], ins, outs, sources[which], if (which == 2) @embedFile("kernels/qsa_nax_header.metal") else "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    mpp_kernels[which] = k;
    return k;
}

fn mppApply(which: usize, s: mlx.mlx_stream, inputs: []const mlx.mlx_array, shape: []const c_int, dtype: mlx.mlx_dtype, grid: [3]c_int, tg: [3]c_int) !mlx.mlx_array {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg[0], tg[1], tg[2]));
    if (which == 2) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BM", 64));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WM", 2));
    }
    const ins = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try mppKernel(which), ins, cfg, s));
    return get(outs, 0);
}

fn makeTiles(s: mlx.mlx_stream, sorted: mlx.mlx_array) !mlx.mlx_array {
    const n: c_int = @intCast(mlx.mlx_array_size(sorted));
    return mppApply(0, s, &.{sorted}, &.{ @divTrunc(n, 64) + 512, 3 }, .int32, .{ 512, 1, 1 }, .{ 512, 1, 1 });
}

fn gatherInputs(s: mlx.mlx_stream, x: mlx.mlx_array, order: mlx.mlx_array) !mlx.mlx_array {
    const n: c_int = @intCast(mlx.mlx_array_size(order));
    return mppApply(1, s, &.{ x, order }, &.{ n + 64, 1, 2560 }, .bfloat16, .{ (n + 64) * 320, 1, 1 }, .{ 256, 1, 1 });
}

pub const Bank = struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array };
pub fn gateUp(s: mlx.mlx_stream, x: mlx.mlx_array, g: Group, gate: Bank, up: Bank) !?mlx.mlx_array {
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.getShape(x).len != 2 or mlx.getShape(x)[1] != 2560) return null;
    for ([_]Bank{ gate, up }) |bank| {
        if (!std.mem.eql(c_int, mlx.getShape(bank.w), &.{ 512, 640, 320 }) or !std.mem.eql(c_int, mlx.getShape(bank.sc), &.{ 512, 640, 40 }) or !std.mem.eql(c_int, mlx.getShape(bank.bi), &.{ 512, 640, 40 })) return null;
        if (mlx.mlx_array_dtype(bank.w) != .uint32 or mlx.mlx_array_dtype(bank.sc) != .bfloat16 or mlx.mlx_array_dtype(bank.bi) != .bfloat16) return null;
    }
    const n: c_int = @intCast(mlx.mlx_array_size(g.order));
    if (n != mlx.getShape(x)[0] * 10) return null;
    const tiles = try makeTiles(s, g.sorted);
    defer _ = mlx.mlx_array_free(tiles);
    const rep = try gatherInputs(s, x, g.order);
    defer _ = mlx.mlx_array_free(rep);
    const out = try mppApply(2, s, &.{ rep, gate.w, gate.sc, gate.bi, g.sorted, tiles, up.w, up.sc, up.bi, try @import("hc_prefill.zig").sigmoidTable(s) }, &.{ n, 640 }, .bfloat16, .{ 320, (@divTrunc(n, 64) + 512) * 4, 1 }, .{ 32, 4, 1 });
    if (!mpp_announced) {
        mpp_announced = true;
        log.info("[moe-mpp] engaged: tokens={d} E=512 K=10 H=2560 I=640 bits=4 BM=64 BN=128 gate_up_swiglu=true\n", .{@divTrunc(n, 10)});
    }
    return out;
}

pub fn liveProbe(tokens: c_int) ?u32 {
    if (tokens < 2048 or tokens > 8192 or live_calls >= 4) return null;
    const raw = std.c.getenv("QWEN4_MOE_LIVE_AB") orelse return null;
    if (raw[0] != '1') return null;
    const call = live_calls;
    live_calls += 1;
    return if (call == 0 or call == 3) call else null;
}

/// Diagnostic only: save four real wide-prefill routing fixtures, then stop.
/// Saving evaluates the router, so these runs must not be used for throughput.
pub fn captureRoutes(a: std.mem.Allocator, indices: mlx.mlx_array, tokens: c_int) !void {
    if (tokens < 2048 or captured_routes >= 4) return;
    const prefix = std.c.getenv("QWEN4_MOE_CAPTURE_PATH") orelse return;
    const path = try std.fmt.allocPrintSentinel(a, "{s}.{d}.safetensors", .{ std.mem.sliceTo(prefix, 0), captured_routes }, 0);
    defer a.free(path);
    const arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    const metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    try mlx.check(mlx.mlx_map_string_to_array_insert(arrays, "indices", indices));
    try mlx.check(mlx.mlx_save_safetensors(path.ptr, arrays, metadata));
    log.info("[moe-route-capture] fixture={d} tokens={d} synchronization_added=true\n", .{ captured_routes, tokens });
    captured_routes += 1;
}
pub fn enabled() bool {
    if (@import("prefill_experiment.zig").group()) |v| return v and !mlx.noGpuBackend();
    if (mppEnabled()) return !mlx.noGpuBackend();
    const raw = std.c.getenv("MLX_SERVE_MOE_PREFILL_GROUP") orelse return false;
    return raw[0] == '1' and !mlx.noGpuBackend();
}
pub const Group = struct {
    order: mlx.mlx_array,
    inverse: mlx.mlx_array,
    sorted: mlx.mlx_array,
    pub fn deinit(self: Group) void {
        _ = mlx.mlx_array_free(self.order);
        _ = mlx.mlx_array_free(self.inverse);
        _ = mlx.mlx_array_free(self.sorted);
    }
};
fn kernel(which: usize) !mlx.mlx_fast_metal_kernel {
    if (kernels[which]) |k| return k;
    const input_names: []const [*:0]const u8 = switch (which) {
        0 => &.{"indices"},
        1 => &.{"counts"},
        2 => &.{ "indices", "offsets" },
        else => &.{ "down", "scores", "inverse" },
    };
    const output_names: []const [*:0]const u8 = switch (which) {
        0 => &.{"counts"},
        1 => &.{"offsets"},
        2 => &.{ "order", "inverse", "sorted_ids" },
        else => &.{"out"},
    };
    const ins = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs = mlx.mlx_vector_string_new_data(output_names.ptr, output_names.len);
    defer _ = mlx.mlx_vector_string_free(outs);
    const names = [_][*:0]const u8{ "msv_moe_prefill_count", "msv_moe_prefill_prefix", "msv_moe_prefill_scatter", "msv_moe_prefill_reduce" };
    const sources = [_][*:0]const u8{ @embedFile("kernels/moe_prefill_count.metal"), @embedFile("kernels/moe_prefill_prefix.metal"), @embedFile("kernels/moe_prefill_scatter.metal"), @embedFile("kernels/moe_prefill_reduce.metal") };
    const k = mlx.mlx_fast_metal_kernel_new(names[which], ins, outs, sources[which], "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernels[which] = k;
    return k;
}
fn apply(which: usize, s: mlx.mlx_stream, inputs: []const mlx.mlx_array, shapes: []const []const c_int, dtype: mlx.mlx_dtype, grid: [3]c_int, tg: c_int) !mlx.mlx_vector_array {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (shapes) |shape| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    const ins = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(ins);
    var out = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(out);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out, try kernel(which), ins, cfg, s));
    return out;
}
fn get(v: mlx.mlx_vector_array, index: usize) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, v, index));
    return out;
}
pub fn group(s: mlx.mlx_stream, indices: mlx.mlx_array) !Group {
    const n: c_int = @intCast(mlx.mlx_array_size(indices));
    if (n <= 0 or n > 8192 * 10 or mlx.getShape(indices).len != 1) return error.UnsupportedMoeGroup;
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, indices, .uint32, s));
    const groups = @divTrunc(n + 1023, 1024);
    const shape = [_]c_int{ groups, 512 };
    const counts_vec = try apply(0, s, &.{ids}, &.{&shape}, .uint32, .{ groups * 512, 1, 1 }, 512);
    defer _ = mlx.mlx_vector_array_free(counts_vec);
    const counts = try get(counts_vec, 0);
    defer _ = mlx.mlx_array_free(counts);
    const prefix_vec = try apply(1, s, &.{counts}, &.{&shape}, .uint32, .{ 512, 1, 1 }, 512);
    defer _ = mlx.mlx_vector_array_free(prefix_vec);
    const offsets = try get(prefix_vec, 0);
    defer _ = mlx.mlx_array_free(offsets);
    const flat = [_]c_int{n};
    const scatter_vec = try apply(2, s, &.{ ids, offsets }, &.{ &flat, &flat, &flat }, .uint32, .{ groups * 512, 1, 1 }, 512);
    defer _ = mlx.mlx_vector_array_free(scatter_vec);
    const order = try get(scatter_vec, 0);
    errdefer _ = mlx.mlx_array_free(order);
    const inverse = try get(scatter_vec, 1);
    errdefer _ = mlx.mlx_array_free(inverse);
    const sorted = try get(scatter_vec, 2);
    return .{ .order = order, .inverse = inverse, .sorted = sorted };
}
pub fn reduce(s: mlx.mlx_stream, down: mlx.mlx_array, scores: mlx.mlx_array, inverse: mlx.mlx_array) !mlx.mlx_array {
    const count = mlx.mlx_array_size(inverse);
    if (count == 0 or count % 10 != 0 or mlx.mlx_array_size(down) != count * 2560 or mlx.mlx_array_size(scores) != count or mlx.mlx_array_dtype(down) != .bfloat16 or mlx.mlx_array_dtype(scores) != .bfloat16) return error.UnsupportedMoeReduce;
    const tokens: c_int = @intCast(count / 10);
    const shape = [_]c_int{ tokens, 2560 };
    const v = try apply(3, s, &.{ down, scores, inverse }, &.{&shape}, .bfloat16, .{ 2560, tokens, 1 }, 256);
    defer _ = mlx.mlx_vector_array_free(v);
    const out = try get(v, 0);
    if (!announced) {
        announced = true;
        log.info("[moe-prefill-group] engaged: tokens={d} E=512 K=10 H=2560 native_projections=true bf16_grouped_reduce=true\n", .{tokens});
    }
    return out;
}

test "MoE prefill grouping bijection and exact native BF16 weighted reduction" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const tokens = 257;
    const n = tokens * 10;
    var ids: [n]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = if (i < 1024) 17 else @intCast((i * 71 + 3) % 512);
    const ix = mlx.mlx_array_new_data(&ids, &[_]c_int{n}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ix);
    const grouped = try group(s, ix);
    defer grouped.deinit();
    try mlx.check(mlx.mlx_array_eval(grouped.order));
    try mlx.check(mlx.mlx_array_eval(grouped.inverse));
    try mlx.check(mlx.mlx_array_eval(grouped.sorted));
    const order = mlx.mlx_array_data_uint32(grouped.order).?;
    const inverse = mlx.mlx_array_data_uint32(grouped.inverse).?;
    const sorted = mlx.mlx_array_data_uint32(grouped.sorted).?;
    for (0..n) |i| {
        try std.testing.expect(order[i] < n);
        try std.testing.expectEqual(@as(u32, @intCast(i)), inverse[order[i]]);
        try std.testing.expectEqual(ids[order[i]], sorted[i]);
        if (i > 0) try std.testing.expect(sorted[i - 1] <= sorted[i]);
    }
    const plan = try makeTiles(s, grouped.sorted);
    defer _ = mlx.mlx_array_free(plan);
    try mlx.check(mlx.mlx_array_eval(plan));
    const tiles = mlx.mlx_array_data_int32(plan).?;
    var seen: [n]u8 = @splat(0);
    for (0..mlx.mlx_array_size(plan) / 3) |tile| {
        const count = tiles[tile * 3 + 1];
        if (count == 0) continue;
        const start: usize = @intCast(tiles[tile * 3]);
        const block: usize = @intCast(tiles[tile * 3 + 2]);
        const end = start + @as(usize, @intCast(count));
        try std.testing.expect(end <= n);
        for (start + block * 64..@min(end, start + (block + 1) * 64)) |i| {
            try std.testing.expectEqual(sorted[start], sorted[i]);
            seen[i] += 1;
        }
    }
    for (seen) |count| try std.testing.expectEqual(@as(u8, 1), count);
    const host = try a.alloc(u16, n * 2560);
    defer a.free(host);
    var score_bits: [n]u16 = undefined;
    const bf = struct {
        fn f(value: f32) u16 {
            const bits: u32 = @bitCast(value);
            return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
        }
    }.f;
    var rng: u32 = 8172931;
    for (host) |*v| {
        rng = rng *% 1664525 +% 1013904223;
        v.* = bf(@as(f32, @floatFromInt(@as(i32, @intCast(rng % 257)) - 128)) / 23.0);
    }
    for (&score_bits) |*v| {
        rng = rng *% 1664525 +% 1013904223;
        v.* = bf(@as(f32, @floatFromInt(rng % 1024)) / 5413.0);
    }
    const inputs = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ tokens, 2560 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(inputs);
    const gathered = try gatherInputs(s, inputs, grouped.order);
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_array_eval(gathered));
    const gx = mlx.mlx_array_data_bfloat16(gathered).?;
    for (0..n) |row| {
        const src = @as(usize, order[row] / 10) * 2560;
        try std.testing.expectEqualSlices(u16, host[src..][0..2560], gx[row * 2560 ..][0..2560]);
    }
    for (gx[n * 2560 ..][0 .. 64 * 2560]) |v| try std.testing.expectEqual(@as(u16, 0), v);
    const down = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ n, 2560 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(down);
    const scores = mlx.mlx_array_new_data(&score_bits, &[_]c_int{ tokens, 10, 1 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(scores);
    var sorted_down = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_down);
    try mlx.check(mlx.mlx_take_axis(&sorted_down, down, grouped.order, 0, s));
    const got = try reduce(s, sorted_down, scores, grouped.inverse);
    defer _ = mlx.mlx_array_free(got);
    var shaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shaped);
    try mlx.check(mlx.mlx_reshape(&shaped, down, &[_]c_int{ tokens, 10, 2560 }, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, shaped, scores, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_sum_axis(&ref, weighted, 1, false, s));
    try mlx.check(mlx.mlx_array_eval(got));
    try mlx.check(mlx.mlx_array_eval(ref));
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(ref).?[0 .. tokens * 2560], mlx.mlx_array_data_bfloat16(got).?[0 .. tokens * 2560]);
}
