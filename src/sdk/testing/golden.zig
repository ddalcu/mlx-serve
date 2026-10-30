//! Before/after proof for a port that must not change a bit: with `MLX_SERVE_GOLDEN_DIR` set, the first run saves
//! the named arrays as `<dir>/<name>.safetensors` and every later run compares them bit for bit (`expectBitEqual`).
//! Unset, the test skips. Run once on the old code, port, run again.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");

pub const Named = struct { name: [:0]const u8, array: mlx.mlx_array };

pub fn expect(comptime name: []const u8, arrays: []const Named) !void {
    const dir = std.mem.span(env.get(.MLX_SERVE_GOLDEN_DIR) orelse return error.SkipZigTest);
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/" ++ name ++ ".safetensors", .{dir}, 0);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    if (std.c.access(path, 0) != 0) {
        for (arrays) |x| try mlx.check(mlx.mlx_map_string_to_array_insert(map, x.name, x.array));
        try mlx.check(mlx.mlx_save_safetensors(path, map, meta));
        std.debug.print("golden: saved {s}\n", .{path});
        return;
    }
    try mlx.check(mlx.mlx_load_safetensors(&map, &meta, path, s));
    for (arrays) |x| {
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_map_string_to_array_get(&want, map, x.name));
        try mlx.check(mlx.mlx_array_eval(want));
        ops.expectBitEqual(x.array, want) catch |e| {
            std.debug.print("golden: {s} differs from {s}\n", .{ x.name, path });
            return e;
        };
    }
    std.debug.print("golden: {d} arrays bit-equal to {s}\n", .{ arrays.len, path });
}
