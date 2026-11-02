//! `HostGpu.alias` on the GPU, behind the `ALIAS_TEST_GPU` fixture (skipped otherwise: the CPU suite makes no Metal
//! device here). An alias's bytes are its base's at its offset; a GPU kernel reads and writes them through it and an
//! overlapping alias sees the write; MLX's active bytes do not move for an alias; the base's buffer is held until the
//! last alias over it is freed; a misaligned offset, a range past the base and an unevaluated base are refused.

const std = @import("std");
const abi = @import("../abi/host.zig");
const env = @import("../env.zig");
const mlx = @import("../mlx/ffi.zig");
const impl = @import("plugin_host_impl.zig");

const testing = std.testing;
const has_shims = @import("build_options").mlx_stream;

fn gate() !void {
    if (!has_shims or !env.on(.ALIAS_TEST_GPU)) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
}

fn active() usize {
    var n: usize = 0;
    _ = mlx.mlx_get_active_memory(&n);
    return n;
}

fn uninit(shape: []const i32, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    const v = impl.gpu_table.alloc_uninit(null, shape.ptr, shape.len, @intFromEnum(dt), @ptrCast(&out));
    if (!v.isOk()) {
        _ = mlx.mlx_array_free(out);
        return error.AllocRefused;
    }
    return out;
}

fn alias(base: mlx.mlx_array, offset: u64, shape: []const i32, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    const v = impl.gpu_table.alias(null, .{ .ctx = base.ctx }, offset, shape.ptr, shape.len, @intFromEnum(dt), @ptrCast(&out));
    if (!v.isOk()) {
        _ = mlx.mlx_array_free(out);
        return error.AliasRefused;
    }
    return out;
}

fn refusedAlias(base: mlx.mlx_array, offset: u64, shape: []const i32) !void {
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    const v = impl.gpu_table.alias(null, .{ .ctx = base.ctx }, offset, shape.ptr, shape.len, @intFromEnum(mlx.mlx_dtype.uint32), @ptrCast(&out));
    try testing.expectEqual(abi.Verdict.Code.failed, v.code);
}

fn evalOne(x: mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{x}, 1);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

fn pattern(i: usize) u32 {
    return @truncate((i *% 2654435761) ^ 0x5bd1e995);
}

/// `x + 0` on the GPU: a kernel reads every word of `x` (MLX's own binary op), evaluated.
fn readBack(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const zero = mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(zero);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_add(&y, x, zero, s));
    try evalOne(y);
    return y;
}

const not_src = "uint i = thread_position_in_grid.x;\nif (i == 0) carrier[0] = 0;\n((device uint*)dst)[i] = ~dst[i];\n";

/// Every word of `dst` complemented in place by a kernel that writes it as an input (the write-into twins' form),
/// its one-word carrier evaluated; then the stream synchronized.
fn complementInPlace(dst: mlx.mlx_array, n: usize, s: mlx.mlx_stream) !void {
    const dst_names = [_][*:0]const u8{"dst"};
    const carrier_names = [_][*:0]const u8{"carrier"};
    const iv = mlx.mlx_vector_string_new_data(&dst_names, 1);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&carrier_names, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("alias_test_complement", iv, ov, not_src, "", true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{1}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(n), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const ins = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{dst}, 1);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_vector_array_get(&c, outs, 0));
    try evalOne(c);
    try mlx.check(mlx.mlx_synchronize(s));
}

test "builtin host gpu: an alias reads and writes its base's pages, holds the base, adds no active bytes (ALIAS_TEST_GPU)" {
    try gate();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const page = std.heap.pageSize();
    const words: usize = 1 << 22;
    const pool = try uninit(&.{@intCast(words)}, .uint32);
    var pool_live = true;
    defer if (pool_live) {
        _ = mlx.mlx_array_free(pool);
    };
    const p: [*]u32 = @constCast(mlx.mlx_array_data_uint32(pool) orelse return error.NoData);
    for (0..words) |i| p[i] = pattern(i);

    const with_pool = active();
    const off1: usize = 3 * page;
    const n1: usize = 1 << 20;
    const a1 = try alias(pool, off1, &.{@intCast(n1)}, .uint32);
    var a1_live = true;
    defer if (a1_live) {
        _ = mlx.mlx_array_free(a1);
    };
    const a2 = try alias(pool, off1 + page, &.{@intCast(n1)}, .uint32);
    var a2_live = true;
    defer if (a2_live) {
        _ = mlx.mlx_array_free(a2);
    };
    try testing.expectEqual(with_pool, active());
    try testing.expectEqual(@intFromPtr(p) + off1, @intFromPtr(mlx.mlx_array_data_uint32(a1).?));
    try testing.expectEqual(@intFromPtr(p) + off1 + page, @intFromPtr(mlx.mlx_array_data_uint32(a2).?));

    // A GPU read through the alias: the base's words at its offset.
    const w1 = off1 / 4;
    {
        const y = try readBack(a1, s);
        defer _ = mlx.mlx_array_free(y);
        try testing.expectEqualSlices(u32, p[w1..][0..n1], mlx.mlx_array_data_uint32(y).?[0..n1]);
    }
    // A GPU write through the alias lands in the base's pages, and the overlapping alias reads it.
    try complementInPlace(a1, n1, s);
    for (0..n1) |i| if (p[w1 + i] != ~pattern(w1 + i)) return error.AliasWriteMissed;
    {
        const y = try readBack(a2, s);
        defer _ = mlx.mlx_array_free(y);
        const q = mlx.mlx_array_data_uint32(y).?;
        const w2 = w1 + page / 4;
        for (0..n1) |i| {
            const want = if (w2 + i < w1 + n1) ~pattern(w2 + i) else pattern(w2 + i);
            if (q[i] != want) return error.AliasOverlapMissed;
        }
    }

    // The refusals: an offset off the page, a range past the base, an unevaluated base.
    try refusedAlias(pool, page / 2, &.{16});
    try refusedAlias(pool, words * 4 - page, &.{@intCast(page / 2)});
    {
        var lazy = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lazy);
        try mlx.check(mlx.mlx_add(&lazy, pool, pool, s));
        try refusedAlias(lazy, 0, &.{16});
    }

    // The base's buffer stays MLX's active bytes while an alias over it lives, and goes with the last one.
    try mlx.check(mlx.mlx_synchronize(s));
    const held = active();
    _ = mlx.mlx_array_free(pool);
    pool_live = false;
    try testing.expectEqual(held, active());
    _ = mlx.mlx_array_free(a1);
    a1_live = false;
    try testing.expectEqual(held, active());
    _ = mlx.mlx_array_free(a2);
    a2_live = false;
    try testing.expectEqual(held - words * 4, active());
    std.debug.print("[alias-test] words ok: read, write and overlap through aliases; active held={d} MiB, freed={d} MiB with the last alias\n", .{ held >> 20, (words * 4) >> 20 });
}
