//! Keyed Gumbel-max sampling on the GPU, ported from TensorFold's `gpu_sampling.py` (MIT, see NOTICE).
//! The token at absolute position p is argmax over the top-k/top-p candidates of v_i/T + g(seed, p, i),
//! g from a splitmix64 hash: a function of the row's own logits and its position only, so a verify row
//! draws the token the serial step draws, and a draft over the head's logits shares that noise.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");

/// Most candidates a row sorts; tokens within NEAR (in logits / T) of the max are gathered directly.
const CANDIDATES = 1024;
const NEAR: f32 = 20.0;

const HEADER = kernelSource("kernels/sampling/keyed_sample_header.metal");

const SOURCE = kernelSource("kernels/sampling/keyed_sample.metal");

var kernel: ?mlx.mlx_fast_metal_kernel = null;
const CfgKey = struct { rows: c_int, vocab: c_int };
var cfg_cache: std.AutoHashMapUnmanaged(CfgKey, mlx.mlx_fast_metal_kernel_config) = .{};

fn config(key: CfgKey) !mlx.mlx_fast_metal_kernel_config {
    if (cfg_cache.get(key)) |c| return c;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const out_shape = [_]c_int{key.rows};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 1024 * key.rows, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "V", key.vocab));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "C", CANDIDATES));
    try cfg_cache.put(std.heap.c_allocator, key, c);
    return c;
}

fn mix(v: u64) u64 {
    var x = v;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// The kernel's Gumbel noise for token `id` at `position`, on the host.
pub fn gumbel(seed: u64, position: u64, id: u32) f32 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (@as(u64, @as(u32, @truncate(position))) *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    const u = (@as(f32, @floatFromInt(@as(u32, @truncate(x >> 40)))) + 0.5) * (1.0 / 16777216.0);
    return -@log(-@log(u));
}

pub const Params = struct { seed: u64, temperature: f32, top_p: f32, top_k: u32, min_p: f32 = 0 };

/// Tokens `[R]` uint32 (lazy) for `logits` `[..., V]` (R rows), row r at absolute position `positions[r]`.
pub fn sample(logits: mlx.mlx_array, p: Params, positions: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(logits);
    const vocab = sh[sh.len - 1];
    const rows: c_int = @intCast(positions.len);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, logits, &[_]c_int{ rows, vocab }, 2, s));

    var seed_buf: [2 * 64]u32 = undefined;
    if (rows > 64) return error.TooManyRows;
    for (0..@intCast(rows)) |r| {
        seed_buf[2 * r] = @truncate(p.seed);
        seed_buf[2 * r + 1] = @truncate(p.seed >> 32);
    }
    const seeds = mlx.mlx_array_new_data(&seed_buf, &[_]c_int{2 * rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(seeds);
    const pos = mlx.mlx_array_new_data(positions.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(pos);
    const cfg_v = [_]f32{ 1.0 / @max(p.temperature, 1e-6), p.top_p, NEAR, p.min_p };
    const cfg = mlx.mlx_array_new_data(&cfg_v, &[_]c_int{4}, 1, .float32);
    defer _ = mlx.mlx_array_free(cfg);
    const kcap = mlx.mlx_array_new_data(&p.top_k, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(kcap);

    if (kernel == null) {
        const names_in = [_][*:0]const u8{ "L", "seeds", "positions", "cfg", "kcap" };
        const names_out = [_][*:0]const u8{"TOK"};
        const in_n = mlx.mlx_vector_string_new_data(&names_in, names_in.len);
        defer _ = mlx.mlx_vector_string_free(in_n);
        const out_n = mlx.mlx_vector_string_new_data(&names_out, names_out.len);
        defer _ = mlx.mlx_vector_string_free(out_n);
        const k = mlx.mlx_fast_metal_kernel_new("msv_keyed_sample", in_n, out_n, SOURCE, HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    const inputs = [_]mlx.mlx_array{ flat, seeds, pos, cfg, kcap };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kernel.?, in_vec, try config(.{ .rows = rows, .vocab = vocab }), s));
    var tok = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&tok, outs, 0));
    return tok;
}

const testing = std.testing;

test "keyed_sample: draws TensorFold's reference tokens, a row alone or among others" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const V = 3000;
    var v: [V]f32 = undefined;
    for (&v, 0..) |*x, i| {
        const f: f32 = @floatFromInt(i);
        x.* = 6.0 * @sin(f * 0.37) * @cos(f * 0.011) - @as(f32, @floatFromInt(i % 13)) * 0.25;
    }
    // Expected tokens from tensorfold.engine.gpu_sampling.reference on the same row.
    const Case = struct { p: Params, positions: []const u32, want: []const u32 };
    const cases = [_]Case{
        .{ .p = .{ .seed = 1234, .temperature = 1.0, .top_p = 0.95, .top_k = 20 }, .positions = &.{ 31, 32, 200 }, .want = &.{ 1158, 2263, 2314 } },
        .{ .p = .{ .seed = 99, .temperature = 1.0, .top_p = 0.95, .top_k = 20 }, .positions = &.{ 31, 32, 200 }, .want = &.{ 1703, 1159, 1405 } },
        .{ .p = .{ .seed = 99, .temperature = 0.7, .top_p = 0.9, .top_k = 0 }, .positions = &.{ 31, 32, 200 }, .want = &.{ 861, 2577, 2891 } },
        .{ .p = .{ .seed = 1234, .temperature = 2.0, .top_p = 1.0, .top_k = 0 }, .positions = &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, .want = &.{ 316, 1822, 4, 2718, 599, 616, 2576, 1756 } },
    };
    var rows_buf: [8 * V]f32 = undefined;
    for (cases) |c| {
        const n = c.positions.len;
        for (0..n) |r| @memcpy(rows_buf[r * V ..][0..V], &v);
        const logits = mlx.mlx_array_new_data(&rows_buf, &[_]c_int{ @intCast(n), V }, 2, .float32);
        defer _ = mlx.mlx_array_free(logits);
        const all = try sample(logits, c.p, c.positions, s);
        defer _ = mlx.mlx_array_free(all);
        try mlx.check(mlx.mlx_array_eval(all));
        try testing.expectEqualSlices(u32, c.want, mlx.mlx_array_data_uint32(all).?[0..n]);
        const one_row = mlx.mlx_array_new_data(&v, &[_]c_int{ 1, 1, V }, 3, .float32);
        defer _ = mlx.mlx_array_free(one_row);
        for (c.positions, c.want) |pos, want| {
            const one = try sample(one_row, c.p, &.{pos}, s);
            defer _ = mlx.mlx_array_free(one);
            try mlx.check(mlx.mlx_array_eval(one));
            try testing.expectEqual(want, mlx.mlx_array_data_uint32(one).?[0]);
        }
    }
}

test "keyed_sample.gumbel: the host noise picks the kernel's token" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const V = 3000;
    var v: [V]f32 = undefined;
    for (&v, 0..) |*x, i| x.* = 6.0 * @sin(@as(f32, @floatFromInt(i)) * 0.37);
    var ids: [V]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast(i);
    std.mem.sort(u32, &ids, &v, struct {
        fn lt(vals: *const [V]f32, a: u32, b: u32) bool {
            return vals[a] > vals[b] or (vals[a] == vals[b] and a < b);
        }
    }.lt);
    const logits = mlx.mlx_array_new_data(&v, &[_]c_int{ 1, V }, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    for (0..32) |pos| {
        const p: Params = .{ .seed = 1234, .temperature = 1.0, .top_p = 1.0, .top_k = 20 };
        const tok = try sample(logits, p, &.{@intCast(pos)}, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        var best: u32 = ids[0];
        var best_score = -std.math.inf(f32);
        for (ids[0..20]) |id| {
            const score = v[id] + gumbel(p.seed, pos, id);
            if (score > best_score) {
                best_score = score;
                best = id;
            }
        }
        try testing.expectEqual(best, mlx.mlx_array_data_uint32(tok).?[0]);
    }
}

test "keyed_sample min_p excludes low logits after temperature and top_k" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const raw = [_]f32{ 3, 2, 0, -5 };
    const logits = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const p: Params = .{ .seed = 9, .temperature = 1, .top_p = 1, .top_k = 2, .min_p = 0.5 };
    for (0..8) |draw| {
        const pos = [_]u32{@intCast(draw)};
        const tok = try sample(logits, p, &pos, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        try testing.expectEqual(@as(u32, 0), mlx.mlx_array_data_uint32(tok).?[0]);
    }
}

test "keyed_sample min_p retains the temperature-scaled nucleus" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const raw = [_]f32{ 3, 2.8, 0, -5 };
    const logits = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const p: Params = .{ .seed = 1234, .temperature = 2, .top_p = 0.9, .top_k = 3, .min_p = 0.5 };
    for (0..32) |draw| {
        const pos: u32 = @intCast(draw);
        const tok = try sample(logits, p, &.{pos}, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        const score0 = raw[0] / p.temperature + gumbel(p.seed, pos, 0);
        const score1 = raw[1] / p.temperature + gumbel(p.seed, pos, 1);
        try testing.expectEqual(@as(u32, if (score0 >= score1) 0 else 1), mlx.mlx_array_data_uint32(tok).?[0]);
    }
}
