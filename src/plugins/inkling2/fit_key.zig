//! The plugin's own key for `MLX_SERVE_INKLING2_PIPELINES_AT_LOAD`'s record, from what it can read itself: the
//! server executable's bytes (an in-tree plugin builds into it, so this stands for the
//! plugin and the server build), the MLX version, the machine (chip and RAM), and the
//! server's `MLX_` settings (bar the diagnostics). Never the prompt: a record is machine tuning.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const offload = @import("../../sdk/ane/offload.zig");

/// `<exe sha256, 16 hex> mlx <version> <chip> <RAM GB>GB`, in `buf`.
pub fn own(buf: []u8) ![]const u8 {
    var exe: [16]u8 = @splat('0');
    if (exeDigest()) |d| exe = std.fmt.bytesToHex(d[0..8].*, .lower);
    var ver = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(ver);
    const version: []const u8 = if (mlx.mlx_version(&ver) == 0) std.mem.span(mlx.mlx_string_data(ver)) else "unknown";
    return std.fmt.bufPrint(buf, "{s} mlx {s} {s} {d}GB", .{ &exe, version, machine(offload.chipBrand()), (offload.totalMemBytes() + (1 << 29)) >> 30 });
}

fn machine(chip: []const u8) []const u8 {
    return if (chip.len > 0) chip else "unknown";
}

/// The server executable's sha256; null where it cannot be read.
fn exeDigest() ?[32]u8 {
    var path_buf: [4096]u8 = undefined;
    const path = exePath(&path_buf) orelse return null;
    const fh = std.c.fopen(path, "rb") orelse return null;
    defer _ = std.c.fclose(fh);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [1 << 16]u8 = undefined;
    while (true) {
        const n = std.c.fread(&buf, 1, buf.len, fh);
        if (n == 0) break;
        h.update(buf[0..n]);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    return d;
}

fn exePath(buf: []u8) ?[:0]const u8 {
    if (!builtin.os.tag.isDarwin()) return null;
    var n: u32 = @intCast(buf.len);
    if (std.c._NSGetExecutablePath(buf.ptr, &n) != 0) return null;
    return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf.ptr)), 0);
}

test "inkling2 draft: the plugin's own fit key names this build and machine and is stable" {
    var a: [512]u8 = undefined;
    var b: [512]u8 = undefined;
    const k1 = try own(&a);
    const k2 = try own(&b);
    try std.testing.expectEqualStrings(k1, k2);
    try std.testing.expect(std.mem.indexOf(u8, k1, " mlx ") != null);
    try std.testing.expect(std.mem.endsWith(u8, k1, "GB"));
    // The executable's digest, not a placeholder.
    try std.testing.expect(!std.mem.startsWith(u8, k1, "0000000000000000"));
}
