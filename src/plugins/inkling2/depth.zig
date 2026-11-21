//! Inkling 2 depth mixing: the block attention residual (a softmax over the block states and the partial block) and
//! the message gate that mixes a looped layer's input with its message.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernels = @import("kernels.zig");

const Array = ops.Array;
const Graph = ops.Graph;

/// `down` [r, 2D] in the pack's dtype; `forget`, `message` f32 [D, r].
pub const Gate = struct { down: Array, forget: Array, message: Array };

/// blocks: f32 [N,1,L,D] or null (N = 0); partial: f32 [1,L,D]; query: f32 [Q,D]. Returns f32 [1,L,Q,D].
pub fn attend(g: *Graph, blocks: ?Array, partial: Array, query: Array) !Array {
    if (try kernels.depthAttention(g, blocks, partial, query, scaleOf(partial))) |y| return y;
    return attendComposed(g, blocks, partial, query);
}

/// The op chain `attend` falls back to and the kernel is checked against.
pub fn attendComposed(g: *Graph, blocks: ?Array, partial: Array, query: Array) !Array {
    var t = g.scope();
    defer t.deinit();
    const p = try t.expandDims(partial, 0);
    const stacked = if (blocks) |b| try t.concat(&.{ b, p }, 0) else p;
    // moveaxis(0, -2): [N+1, ..., D] -> [..., N+1, D].
    const nd = ops.shape(stacked).len;
    var axes: [8]c_int = undefined;
    for (0..nd - 2) |i| axes[i] = @intCast(i + 1);
    axes[nd - 2] = 0;
    axes[nd - 1] = @intCast(nd - 1);
    const st = try t.transpose(stacked, axes[0..nd]);
    const lg = try t.mul(try t.matmul(st, try t.transpose(query, &.{ 1, 0 })), try t.scalar(scaleOf(partial), .float32));
    const w = try t.softmax(lg, -2, true);
    for (0..nd) |i| axes[i] = @intCast(i);
    axes[nd - 2] = @intCast(nd - 1);
    axes[nd - 1] = @intCast(nd - 2);
    return g.keep(try t.matmul(try t.transpose(w, axes[0..nd]), st));
}

/// h, msg: f32 [1,L,D]. Returns f32 [1,L,D].
pub fn gate(g: *Graph, w: *const Gate, h: Array, msg: Array, ones_h: Array) !Array {
    var t = g.scope();
    defer t.deinit();
    const st = try t.concat(&.{ try t.rmsNorm(h, ones_h, 1e-6), try t.rmsNorm(msg, ones_h, 1e-6) }, -1);
    const r = try t.astype(try t.matmul(try t.astype(st, ops.dtype(w.down)), try t.transpose(w.down, &.{ 1, 0 })), .float32);
    const f = try t.tanh(try t.matmul(r, try t.transpose(w.forget, &.{ 1, 0 })));
    const m = try t.tanh(try t.matmul(r, try t.transpose(w.message, &.{ 1, 0 })));
    return g.keep(try t.add(try t.mul(try t.sub(try t.scalar(1, .float32), f), h), try t.mul(m, msg)));
}

/// Row j of attend's output: f32 [1,L,D].
pub fn row(g: *Graph, x: Array, j: c_int) !Array {
    var t = g.scope();
    defer t.deinit();
    const sh = ops.shape(x);
    return g.keep(try t.reshape(try t.sliceAxis(x, 2, j, j + 1), &.{ sh[0], sh[1], sh[3] }));
}

/// mlx-lm's `dims ** -0.5`, rounded once to f32.
fn scaleOf(partial: Array) f32 {
    const sh = ops.shape(partial);
    return @floatCast(std.math.pow(f64, @floatFromInt(sh[sh.len - 1]), -0.5));
}

// ── tests ──

const testing = std.testing;
const testReadF32 = @import("../../sdk/testing/parity.zig").testReadF32;

fn randFill(rnd: std.Random, buf: []f32) void {
    for (buf) |*v| v.* = rnd.floatNorm(f32);
}

fn relErr(got: []const f32, want: []const f64) f64 {
    var diff: f64 = 0;
    var mag: f64 = 0;
    for (got, want) |a, b| {
        diff = @max(diff, @abs(@as(f64, a) - b));
        mag = @max(mag, @abs(b));
    }
    return diff / mag;
}

fn urms(x: []const f32, out: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + 1e-6);
    for (x, out) |v, *o| o.* = v * inv;
}

test "inkling2 depth: attendComposed and the message gate match host math" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const L = 3;
    const D = 32;
    const R = 8;
    const scale = std.math.pow(f64, D, -0.5);

    for ([_]usize{ 0, 3 }) |n| for ([_]usize{ 1, 3 }) |q| {
        const blocks = try a.alloc(f32, n * L * D);
        defer a.free(blocks);
        var partial: [L * D]f32 = undefined;
        const query = try a.alloc(f32, q * D);
        defer a.free(query);
        randFill(rnd, blocks);
        randFill(rnd, &partial);
        randFill(rnd, query);
        const bs: ?Array = if (n == 0) null else try g.fromSlice(f32, blocks, &.{ @intCast(n), 1, L, D });
        const y = try attendComposed(&g, bs, try g.fromSlice(f32, &partial, &.{ 1, L, D }), try g.fromSlice(f32, query, &.{ @intCast(q), D }));
        try testing.expectEqualSlices(c_int, &.{ 1, L, @intCast(q), D }, ops.shape(y));
        const got = try a.alloc(f32, L * q * D);
        defer a.free(got);
        try testReadF32(y, got, s);

        const want = try a.alloc(f64, L * q * D);
        defer a.free(want);
        @memset(want, 0);
        for (0..L) |l| for (0..q) |j| {
            var lg: [4]f64 = undefined;
            var mx: f64 = -std.math.inf(f64);
            for (0..n + 1) |m| {
                const st = if (m < n) blocks[(m * L + l) * D ..][0..D] else partial[l * D ..][0..D];
                var dot: f64 = 0;
                for (st, query[j * D ..][0..D]) |u, v| dot += @as(f64, u) * v;
                lg[m] = dot * scale;
                mx = @max(mx, lg[m]);
            }
            var z: f64 = 0;
            for (lg[0 .. n + 1]) |*v| {
                v.* = @exp(v.* - mx);
                z += v.*;
            }
            for (0..n + 1) |m| {
                const st = if (m < n) blocks[(m * L + l) * D ..][0..D] else partial[l * D ..][0..D];
                for (want[(l * q + j) * D ..][0..D], st) |*o, u| o.* += lg[m] / z * u;
            }
        };
        try testing.expect(relErr(got, want) < 1e-5);
    };

    // The message gate.
    var h: [L * D]f32 = undefined;
    var msg: [L * D]f32 = undefined;
    var down: [R * 2 * D]f32 = undefined;
    var forget: [D * R]f32 = undefined;
    var message: [D * R]f32 = undefined;
    for ([_][]f32{ &h, &msg, &down, &forget, &message }) |buf| randFill(rnd, buf);
    for (&down) |*v| v.* *= 0.2;
    const ones: [D]f32 = @splat(1);
    const w = Gate{
        .down = try g.fromSlice(f32, &down, &.{ R, 2 * D }),
        .forget = try g.fromSlice(f32, &forget, &.{ D, R }),
        .message = try g.fromSlice(f32, &message, &.{ D, R }),
    };
    const y = try gate(&g, &w, try g.fromSlice(f32, &h, &.{ 1, L, D }), try g.fromSlice(f32, &msg, &.{ 1, L, D }), try g.fromSlice(f32, &ones, &.{D}));
    try testing.expectEqualSlices(c_int, &.{ 1, L, D }, ops.shape(y));
    var got: [L * D]f32 = undefined;
    try testReadF32(y, &got, s);
    var want: [L * D]f64 = undefined;
    for (0..L) |l| {
        var st: [2 * D]f64 = undefined;
        urms(h[l * D ..][0..D], st[0..D]);
        urms(msg[l * D ..][0..D], st[D..]);
        var r: [R]f64 = undefined;
        for (&r, 0..) |*o, i| {
            o.* = 0;
            for (st, down[i * 2 * D ..][0 .. 2 * D]) |u, v| o.* += u * v;
        }
        for (0..D) |d| {
            var f: f64 = 0;
            var m: f64 = 0;
            for (r, forget[d * R ..][0..R], message[d * R ..][0..R]) |u, fv, mv| {
                f += u * fv;
                m += u * mv;
            }
            want[l * D + d] = (1 - std.math.tanh(f)) * h[l * D + d] + std.math.tanh(m) * msg[l * D + d];
        }
    }
    try testing.expect(relErr(&got, &want) < 1e-5);
}
