//! Same format, independent decoders: sushi's own K3 mul1 fixture (src/sdk/quant/exl3/sushi/fixtures/exl3_k3_linear.safetensors
//! in the mlx-serve checkout, sushi's exllamav3 reference output) decoded through this plugin's EXL3 host decode
//! (`exl3_kernels.reconstruct`, the reference the kernel self-checks compare against). The fixture's `trellis` (U16
//! [8, 8, 48]: 16 x 16 tiles, 16 K words each at K = 3) must reconstruct to its `inner` (F16 [128, 128], the codebook
//! values before the Hadamard / suh / svh transforms) bit for bit. Skips when the host checkout has no sushi fixture
//! (run from the host root, as `zig build mlx-stream-test` does, or name the file in SUSHI_EXL3_K3_FIXTURE).

const std = @import("std");
const host_env = @import("mlx_host").env;
const kernels = @import("kernels.zig");
const nocache_io = @import("sdk").io_util;

const fixture_default = "src/sdk/quant/exl3/sushi/fixtures/exl3_k3_linear.safetensors";

const Tensor = struct { dtype: []const u8, shape: [4]u64, rank: usize, data_offsets: [2]u64 };

fn tensorOf(header: std.json.Value, name: []const u8) !Tensor {
    const t = header.object.get(name) orelse return error.FixtureTensorMissing;
    const shape = t.object.get("shape").?.array.items;
    var dims: [4]u64 = @splat(0);
    for (shape, 0..) |d, i| dims[i] = @intCast(d.integer);
    const offs = t.object.get("data_offsets").?.array.items;
    return .{
        .dtype = t.object.get("dtype").?.string,
        .shape = dims,
        .rank = shape.len,
        .data_offsets = .{ @intCast(offs[0].integer), @intCast(offs[1].integer) },
    };
}

test "exl3 sushi parity: sushi's K3 mul1 fixture trellis reconstructs to its inner weights bit for bit through this plugin's decode" {
    const a = std.testing.allocator;
    const env = host_env.get(.SUSHI_EXL3_K3_FIXTURE);
    const path = if (env) |p| std.mem.span(p) else fixture_default;
    const bytes = nocache_io.readAllNoCache(a, path, 1 << 20) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer a.free(bytes);

    const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{});
    defer parsed.deinit();
    const data = bytes[8 + n ..];

    const trellis = try tensorOf(parsed.value, "trellis");
    const inner = try tensorOf(parsed.value, "inner");
    try std.testing.expectEqualStrings("U16", trellis.dtype);
    try std.testing.expectEqualStrings("F16", inner.dtype);
    try std.testing.expect(trellis.rank == 3 and inner.rank == 2);
    const n_i: usize = @intCast(trellis.shape[0]);
    const n_j: usize = @intCast(trellis.shape[1]);
    const K: usize = @intCast(trellis.shape[2] / 16);
    try std.testing.expectEqual(@as(usize, 3), K);
    try std.testing.expectEqual(@as(u64, 16 * n_i), inner.shape[0]);
    try std.testing.expectEqual(@as(u64, 16 * n_j), inner.shape[1]);

    const code_bytes = data[trellis.data_offsets[0]..trellis.data_offsets[1]];
    const code = try a.alloc(i16, code_bytes.len / 2);
    defer a.free(code);
    for (code, 0..) |*c, i| c.* = std.mem.readInt(i16, code_bytes[2 * i ..][0..2], .little);
    const want_bytes = data[inner.data_offsets[0]..inner.data_offsets[1]];

    const table = try a.create([65536]u16);
    defer a.destroy(table);
    kernels.mul1Table(table);
    const got = try a.alloc(u16, 256 * n_i * n_j);
    defer a.free(got);
    kernels.reconstruct(code, n_i, n_j, K, table, got, null);

    var mismatches: usize = 0;
    var distinct = std.AutoHashMap(u16, void).init(a);
    defer distinct.deinit();
    for (got, 0..) |g, i| {
        if (g != std.mem.readInt(u16, want_bytes[2 * i ..][0..2], .little)) mismatches += 1;
        try distinct.put(g, {});
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
    // A real weight, not a constant fill (mul1 decodes to at most 1021 values): the comparison cannot pass vacuously.
    try std.testing.expect(distinct.count() > 100);
}
