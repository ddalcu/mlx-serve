//! copied_from: plugins/inkling/attention.zig (`inklingSconvCore`; conv in f32, sqrt(K+1) divisor, window returned)
//! Inkling 2's causal short convolutions over a previous f32 window: the static depthwise conv (`ShortConvolution`)
//! and the hidden-conditioned one of the global layers (`DynamicShortConvolution`).

const std = @import("std");
const ops = @import("../../mlx/ops.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;

const Array = ops.Array;
const Graph = ops.Graph;

/// `y`: x's dtype `[1, L, C]`; `tail`: f32 `[1, K-1, C]`; `window`: f32 `[1, K-1+L, C]`. All owned by `g`.
pub const ConvOut = struct { y: Array, tail: Array, window: Array };

/// `static` `[K, C]`; `proj`: hidden -> heads*K; `heads` = the config's kv heads.
pub const Dyn = struct { static: Array, proj: Linear, heads: u32 };

/// `x` `[1, L, C]` behind `prev` (null = zeros) through the depthwise conv `w` `[C, K, 1]`, with its residual.
pub fn shortConv(g: *Graph, x: Array, w: Array, prev: ?Array) !ConvOut {
    const L = ops.shape(x)[1];
    const C = ops.shape(x)[2];
    const K = ops.shape(w)[1];
    const x32 = try g.astype(x, .float32);
    const window = try windowOf(g, x32, prev, K);
    // The weight is cast up, not the window down: the reference keeps this conv in f32.
    const out = try g.conv1d(window, try g.astype(w, .float32), 1, 0, 1, C);
    const y = try g.div(try g.add(out, x32), try rootOf(g, K + 1));
    return .{ .y = try g.astype(y, ops.dtype(x)), .tail = try g.sliceAxis(window, 1, L, L + K - 1), .window = window };
}

/// `x` `[1, L, C]` behind `prev` through the static taps plus the taps `d.proj` reads from `dyn_in`, with its residual.
pub fn dynamicConv(g: *Graph, x: Array, dyn_in: Array, d: *const Dyn, prev: ?Array) !ConvOut {
    const L = ops.shape(x)[1];
    const C = ops.shape(x)[2];
    const K = ops.shape(d.static)[0];
    const H: c_int = @intCast(d.heads);
    const dynw = try g.reshape(try g.astype(try d.proj.forward(g, dyn_in), .float32), &.{ 1, L, H, 1, K });
    const stat = try g.reshape(try g.transpose(try g.astype(d.static, .float32), &.{ 1, 0 }), &.{ H, @divExact(C, H), K });
    const wts = try g.add(stat, dynw);
    const x32 = try g.astype(x, .float32);
    const window = try windowOf(g, x32, prev, K);
    // Tap `lag` reads x[t - lag]: the reverse of conv1d's tap order.
    const lags = try g.a.alloc(Array, @intCast(K));
    defer g.a.free(lags);
    for (lags, 0..) |*l, j| {
        const lo = K - 1 - @as(c_int, @intCast(j));
        l.* = try g.expandDims(try g.sliceAxis(window, 1, lo, lo + L), 3);
    }
    const wins = try g.reshape(try g.concat(lags, 3), &.{ 1, L, H, @divExact(C, H), K });
    const out = try g.reshape(try g.sum(try g.mul(wins, wts), 4, false), &.{ 1, L, C });
    const y = try g.div(try g.add(out, x32), try rootOf(g, 2 * K + 1));
    return .{ .y = try g.astype(y, ops.dtype(x)), .tail = try g.sliceAxis(window, 1, L, L + K - 1), .window = window };
}

/// `[prev ?? zeros, x32]` on the row axis: f32 `[1, K-1+L, C]`.
fn windowOf(g: *Graph, x32: Array, prev: ?Array, K: c_int) !Array {
    const left = prev orelse try g.zeros(&.{ 1, K - 1, ops.shape(x32)[2] }, .float32);
    return g.concat(&.{ left, x32 }, 1);
}

/// sqrt(n) as the reference's f64 constant rounded to f32.
fn rootOf(g: *Graph, n: c_int) !Array {
    return g.scalar(@floatCast(@sqrt(@as(f64, @floatFromInt(n)))), .float32);
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../mlx/ffi.zig");
const parity = @import("../../sdk/testing/parity.zig");

// Bar: against a host loop where `static[lag]` and dynamic tap `lag` multiply `x[t - lag]`, max rel err < 1e-5.
test "inkling2 conv: the dynamic conv's tap 0 multiplies the current token" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const L = 5;
    const C = 8;
    const Hd = 2;
    const K = 4;
    const D = 6;
    var prng = std.Random.DefaultPrng.init(0x1ce2);
    const r = prng.random();
    var x: [L * C]f32 = undefined;
    var dyn_in: [L * D]f32 = undefined;
    var static: [K * C]f32 = undefined;
    var proj: [D * Hd * K]f32 = undefined; // [in, out], as a dense Linear stores it
    var prev: [(K - 1) * C]f32 = undefined;
    inline for (.{ &x, &dyn_in, &static, &proj, &prev }) |buf| for (buf) |*v| {
        v.* = r.floatNorm(f32);
    };

    for ([_]bool{ false, true }) |with_prev| {
        var g = Graph.init(a, s);
        defer g.deinit();
        const d: Dyn = .{ .static = try g.fromSlice(f32, &static, &.{ K, C }), .proj = .{ .w = try g.fromSlice(f32, &proj, &.{ D, Hd * K }) }, .heads = Hd };
        const p: ?Array = if (with_prev) try g.fromSlice(f32, &prev, &.{ 1, K - 1, C }) else null;
        const out = try dynamicConv(&g, try g.fromSlice(f32, &x, &.{ 1, L, C }), try g.fromSlice(f32, &dyn_in, &.{ 1, L, D }), &d, p);
        const y = try parity.evalToF32(a, out.y, L * C, s);
        defer a.free(y);
        const tail = try parity.evalToF32(a, out.tail, (K - 1) * C, s);
        defer a.free(tail);

        // The window's rows: K-1 previous rows, then x.
        var win: [(K - 1 + L) * C]f32 = @splat(0);
        if (with_prev) @memcpy(win[0 .. (K - 1) * C], &prev);
        @memcpy(win[(K - 1) * C ..], &x);
        var worst: f32 = 0;
        var top: f32 = 0;
        for (0..L) |t| for (0..C) |c| {
            const h = c / (C / Hd);
            var acc: f64 = 0;
            for (0..K) |lag| {
                var dw: f64 = 0;
                for (0..D) |i| dw += @as(f64, dyn_in[t * D + i]) * proj[i * Hd * K + h * K + lag];
                acc += @as(f64, win[(t + K - 1 - lag) * C + c]) * (static[lag * C + c] + dw);
            }
            const want: f32 = @floatCast((acc + x[t * C + c]) / @sqrt(@as(f64, 2 * K + 1)));
            worst = @max(worst, @abs(y[t * C + c] - want));
            top = @max(top, @abs(want));
        };
        try testing.expect(worst / top < 1e-5);
        try testing.expectEqualSlices(f32, win[L * C ..], tail);
    }
}
