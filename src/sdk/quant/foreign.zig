//! Weights stored in a layout the generic blocks do not read themselves (GGUF ggml blocks, block-scaled and row-scaled
//! fp8). A format says which (weight, scales) pairs it stores and how it multiplies them; a block asks here and imports
//! no format, so the blocks build without the formats and a plugin's own format joins the same dispatch (`register`).
//! The server's formats are `foreign_builtin.zig`'s list, tried first and in its order.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const builtin_formats = @import("foreign_builtin.zig").formats;

pub const Format = struct {
    name: []const u8,
    /// True when this format stores the pair (cheap: dtype and shape reads).
    claims: *const fn (w: mlx.mlx_array, sc: mlx.mlx_array) bool,
    /// `x @ w^T` for a pair `claims` accepted.
    linear: *const fn (x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) anyerror!mlx.mlx_array,
    /// Gathered rows of a pair `claims` accepted (MoE experts), or null when the format has none.
    gather: ?*const fn (x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, lhs: mlx.mlx_array, rhs: mlx.mlx_array, s: mlx.mlx_stream) anyerror!mlx.mlx_array = null,
};

const max_registered = 8;
const Registry = struct { items: [max_registered]Format = undefined, n: usize = 0 };
/// Only an outside plugin registers (in its own dylib, its own copy of this file); a server unit never does, so the copies
/// a `-Dunits` build holds are all empty and need no sharing.
var registry: Registry = .{};

/// Adds a format after the server's, once, before the first forward (a plugin's `load`). Past the cap it is refused.
pub fn register(f: Format) error{TooManyFormats}!void {
    const r = &registry;
    if (r.n == max_registered) return error.TooManyFormats;
    r.items[r.n] = f;
    r.n += 1;
}

/// `x @ w^T` through the format that stores (w, sc), or null when none does.
pub fn linear(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    inline for (builtin_formats) |f| if (f.claims(w, sc)) return try f.linear(x, w, sc, s);
    const r = &registry;
    for (r.items[0..r.n]) |f| if (f.claims(w, sc)) return try f.linear(x, w, sc, s);
    return null;
}

/// Gathered rows through the format that stores (w, sc) and gathers, or null when none does.
pub fn gather(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, lhs: mlx.mlx_array, rhs: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    inline for (builtin_formats) |f| if (f.gather) |g| if (f.claims(w, sc)) return try g(x, w, sc, lhs, rhs, s);
    const r = &registry;
    for (r.items[0..r.n]) |f| if (f.gather) |g| if (f.claims(w, sc)) return try g(x, w, sc, lhs, rhs, s);
    return null;
}

test "foreign: a registered format claims its pair, and a pair no format stores falls through" {
    const T = struct {
        var calls: usize = 0;
        fn claims(w: mlx.mlx_array, _: mlx.mlx_array) bool {
            return w.ctx == @as(?*anyopaque, @ptrFromInt(0x10));
        }
        fn lin(_: mlx.mlx_array, w: mlx.mlx_array, _: mlx.mlx_array, _: mlx.mlx_stream) anyerror!mlx.mlx_array {
            calls += 1;
            return w;
        }
    };
    try register(.{ .name = "test", .claims = &T.claims, .linear = &T.lin });
    const mine: mlx.mlx_array = .{ .ctx = @ptrFromInt(0x10) };
    const none: mlx.mlx_array = .{ .ctx = null };
    const s: mlx.mlx_stream = .{ .ctx = null };
    try std.testing.expect((try linear(none, mine, none, s)).?.ctx == mine.ctx);
    try std.testing.expectEqual(@as(usize, 1), T.calls);
    try std.testing.expect(try gather(none, mine, none, none, none, s) == null);
    try std.testing.expect(try linear(none, .{ .ctx = @ptrFromInt(0x20) }, none, s) == null);
}
