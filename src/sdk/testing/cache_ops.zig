//! A plain `CacheOps` for conformance (dense K/V grown by concatenation, states in slots) and `expectConforms`, the
//! behaviour every `CacheOps` owes: the engine's adapter and this one both pass it.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const abi = @import("../../abi/plugin.zig");

const Allocator = std.mem.Allocator;
const Array = mlx.mlx_array;

/// One slot's state: per layer the whole K/V history and the two recurrent states.
pub const TestCache = struct {
    a: Allocator,
    layers: []Layer,

    const Layer = struct { k: Array = .{}, v: Array = .{}, conv: Array = .{}, ssm: Array = .{}, conv_input: Array = .{}, state_seq: Array = .{}, rows: u64 = 0 };

    pub fn init(a: Allocator, n_layers: usize) !TestCache {
        const layers = try a.alloc(Layer, n_layers);
        @memset(layers, .{});
        return .{ .a = a, .layers = layers };
    }

    pub fn deinit(self: *TestCache) void {
        for (self.layers) |l| for ([_]Array{ l.k, l.v, l.conv, l.ssm, l.conv_input, l.state_seq }) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        };
        self.a.free(self.layers);
    }

    pub fn handle(self: *TestCache) abi.Handle {
        return @ptrCast(self);
    }
};

pub const ops: abi.CacheOps = .{
    .len = len,
    .append = append,
    .append_ring = appendRing,
    .view = view,
    .release = release,
    .append_rows = appendRows,
    .state = state,
    .set_state = setState,
    .state_rows = Rows.stateRows,
    .set_state_rows = Rows.setStateRows,
};

const Rows = @import("../util/state_rows.zig").Rows(state, setState);

fn of(h: abi.Handle) *TestCache {
    return @ptrCast(@alignCast(h.?));
}

fn mx(a: abi.Array) Array {
    return .{ .ctx = a.ctx };
}

fn ab(a: Array) abi.Array {
    return .{ .ctx = a.ctx };
}

fn failed(e: anyerror) abi.Verdict {
    return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
}

fn len(h: abi.Handle, layer: u32) callconv(.c) u64 {
    return of(h).layers[layer].rows;
}

fn grow(buf: *Array, x: Array, s: mlx.mlx_stream) !void {
    var out = mlx.mlx_array_new();
    if (buf.ctx == null) {
        _ = mlx.mlx_array_set(&out, x);
    } else {
        const v = mlx.mlx_vector_array_new_data(&[_]Array{ buf.*, x }, 2);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_concatenate_axis(&out, v, 2, s));
        _ = mlx.mlx_array_free(buf.*);
    }
    buf.* = out;
}

/// The newest `keep` rows of `buf` (all when 0), a new reference.
fn tail(buf: Array, keep: u32, s: mlx.mlx_stream) !Array {
    const sh = mlx.getShape(buf);
    const lo: c_int = if (keep > 0 and sh[2] > keep) sh[2] - @as(c_int, @intCast(keep)) else 0;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, buf, &[_]c_int{ 0, 0, lo, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return out;
}

fn viewOf(l: *const TestCache.Layer, keep: u32, s: mlx.mlx_stream, out: *abi.KvView) !void {
    if (l.k.ctx == null) {
        out.* = .{};
        return;
    }
    const k = try tail(l.k, keep, s);
    errdefer _ = mlx.mlx_array_free(k);
    out.* = .{ .k = ab(k), .v = ab(try tail(l.v, keep, s)), .owned = true };
}

fn appendInto(l: *TestCache.Layer, k: Array, v: Array, s: mlx.mlx_stream) !void {
    try grow(&l.k, k, s);
    try grow(&l.v, v, s);
    l.rows += @intCast(mlx.getShape(k)[2]);
}

fn append(h: abi.Handle, layer: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
    const l = &of(h).layers[layer];
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    appendInto(l, mx(k), mx(v), st) catch |e| return failed(e);
    viewOf(l, keep, st, out) catch |e| return failed(e);
    return .ok;
}

/// Keeps every row: a ring only bounds memory, and the view is the same tail.
fn appendRing(h: abi.Handle, layer: u32, k: abi.Array, v: abi.Array, keep: u32, _: u32, _: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
    return append(h, layer, k, v, keep, s, out);
}

fn view(h: abi.Handle, layer: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
    viewOf(&of(h).layers[layer], 0, .{ .ctx = s.ctx }, out) catch |e| return failed(e);
    return .ok;
}

fn release(v: *abi.KvView) callconv(.c) void {
    if (v.owned) {
        _ = mlx.mlx_array_free(mx(v.k));
        _ = mlx.mlx_array_free(mx(v.v));
    }
    v.* = .{};
}

fn appendRows(caches: [*]const abi.Handle, n: u32, layer: u32, k: abi.Array, v: abi.Array, _: u32, s: abi.Stream) callconv(.c) abi.Verdict {
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const sh = mlx.getShape(mx(k));
    for (caches[0..n], 0..) |h, i| {
        const r: c_int = @intCast(i);
        var rows: [2]Array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (rows) |x| {
            _ = mlx.mlx_array_free(x);
        };
        for ([_]abi.Array{ k, v }, 0..) |src, j| {
            const ssh = mlx.getShape(mx(src));
            mlx.check(mlx.mlx_slice(&rows[j], mx(src), &[_]c_int{ r, 0, 0, 0 }, 4, &[_]c_int{ r + 1, ssh[1], ssh[2], ssh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, st)) catch |e| return failed(e);
        }
        _ = sh;
        appendInto(&of(h).layers[layer], rows[0], rows[1], st) catch |e| return failed(e);
    }
    return .ok;
}

fn state(h: abi.Handle, layer: u32, which: abi.StateSlot) callconv(.c) abi.Array {
    const l = &of(h).layers[layer];
    return ab(switch (which) {
        .conv => l.conv,
        .ssm => l.ssm,
        .conv_input => l.conv_input,
        .state_seq => l.state_seq,
        .state_in, .prework_k, .prework_v, .prework_gb => Array{ .ctx = null },
    });
}

fn setState(h: abi.Handle, layer: u32, which: abi.StateSlot, value: abi.Array) callconv(.c) abi.Verdict {
    const l = &of(h).layers[layer];
    const dst = switch (which) {
        .conv => &l.conv,
        .ssm => &l.ssm,
        .conv_input => &l.conv_input,
        .state_seq => &l.state_seq,
        .state_in, .prework_k, .prework_v, .prework_gb => return abi.Verdict.refuse(.unsupported, "no draft-tree state"),
    };
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&r, mx(value));
    if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
    dst.* = r;
    return .ok;
}

// ── conformance ──

const testing = std.testing;

/// Layers the conformance caches must have.
pub const LAYERS = 2;

fn iota(s: mlx.mlx_stream, base: f32, sh: []const c_int) !Array {
    var n: c_int = 1;
    for (sh) |d| n *= d;
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_arange(&r, base, base + @as(f64, @floatFromInt(n)), 1, .float32, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, r, sh.ptr, sh.len, s));
    return out;
}

fn expectSame(s: mlx.mlx_stream, got: abi.Array, want: Array) !void {
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(mx(got)));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, mx(got), want, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var b = false;
    try mlx.check(mlx.mlx_array_item_bool(&b, eq));
    try testing.expect(b);
}

fn concat2(s: mlx.mlx_stream, a: Array, b: Array) !Array {
    const v = mlx.mlx_vector_array_new_data(&[_]Array{ a, b }, 2);
    defer _ = mlx.mlx_vector_array_free(v);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, v, 2, s));
    return out;
}

/// The behaviour every `CacheOps` owes, on two fresh dense caches of `LAYERS` layers with recurrent state.
pub fn expectConforms(o: *const abi.CacheOps, h1: abi.Handle, h2: abi.Handle) !void {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const as: abi.Stream = .{ .ctx = s.ctx };
    try testing.expectEqual(@as(u64, 0), o.len(h1, 0));
    try testing.expect(o.state(h1, 0, .conv).ctx == null and o.state(h1, 1, .ssm).ctx == null);

    // Two appends: the view is the whole history, `keep` its tail, and the other layer stays empty.
    const k1 = try iota(s, 0, &.{ 1, 2, 3, 4 });
    defer _ = mlx.mlx_array_free(k1);
    const k2 = try iota(s, 100, &.{ 1, 2, 2, 4 });
    defer _ = mlx.mlx_array_free(k2);
    var out: abi.KvView = .{};
    try testing.expect(o.append(h1, 0, ab(k1), ab(k1), 0, as, &out).isOk());
    try expectSame(s, out.k, k1);
    o.release(&out);
    try testing.expect(o.append(h1, 0, ab(k2), ab(k2), 0, as, &out).isOk());
    const both = try concat2(s, k1, k2);
    defer _ = mlx.mlx_array_free(both);
    try expectSame(s, out.k, both);
    try expectSame(s, out.v, both);
    o.release(&out);
    try testing.expectEqual(@as(u64, 5), o.len(h1, 0));
    try testing.expectEqual(@as(u64, 0), o.len(h1, 1));
    try testing.expect(o.view(h1, 0, as, &out).isOk());
    try expectSame(s, out.k, both);
    o.release(&out);

    // `keep` hands back the newest rows only, on a plain and on a ring append.
    const k3 = try iota(s, 200, &.{ 1, 2, 1, 4 });
    defer _ = mlx.mlx_array_free(k3);
    try testing.expect(o.append_ring(h2, 0, ab(both), ab(both), 3, 2, 4, as, &out).isOk());
    try testing.expectEqualSlices(c_int, &.{ 1, 2, 3, 4 }, mlx.getShape(mx(out.k)));
    o.release(&out);
    try testing.expect(o.append(h1, 0, ab(k3), ab(k3), 2, as, &out).isOk());
    try testing.expectEqualSlices(c_int, &.{ 1, 2, 2, 4 }, mlx.getShape(mx(out.k)));
    o.release(&out);
    try testing.expectEqual(@as(u64, 6), o.len(h1, 0));

    // Stacked rows: slot i gets row i.
    const st = try iota(s, 300, &.{ 2, 2, 1, 4 });
    defer _ = mlx.mlx_array_free(st);
    const hs = [_]abi.Handle{ h1, h2 };
    try testing.expect(o.append_rows(&hs, 2, 1, ab(st), ab(st), 0, as).isOk());
    for (hs, 0..) |h, i| {
        try testing.expectEqual(@as(u64, 1), o.len(h, 1));
        try testing.expect(o.view(h, 1, as, &out).isOk());
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        const r: c_int = @intCast(i);
        try mlx.check(mlx.mlx_slice(&row, st, &[_]c_int{ r, 0, 0, 0 }, 4, &[_]c_int{ r + 1, 2, 1, 4 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        try expectSame(s, out.k, row);
        o.release(&out);
    }

    // States round-trip per layer and slot, and a write keeps its own reference.
    {
        const c = try iota(s, 7, &.{ 1, 3, 4 });
        try testing.expect(o.set_state(h1, 1, .conv, ab(c)).isOk());
        _ = mlx.mlx_array_free(c);
    }
    const c_again = try iota(s, 7, &.{ 1, 3, 4 });
    defer _ = mlx.mlx_array_free(c_again);
    try expectSame(s, o.state(h1, 1, .conv), c_again);
    try testing.expect(o.state(h2, 1, .conv).ctx == null);
    try testing.expect(o.set_state(h1, 1, .ssm, ab(k1)).isOk());
    try expectSame(s, o.state(h1, 1, .ssm), k1);

    // State rows: null until every cache holds one, then stacked on the batch axis (axis 1 for a state sequence),
    // and a stacked write hands each cache its row.
    try testing.expect(o.state_rows(&hs, 2, 1, .conv, as).ctx == null);
    for ([_]abi.StateSlot{ .conv, .state_seq }) |slot| {
        const ax: usize = if (slot == .state_seq) 1 else 0;
        var sh = [_]c_int{ 3, 1, 4 };
        sh[ax] = 2;
        const stacked = try iota(s, 500, &sh);
        defer _ = mlx.mlx_array_free(stacked);
        try testing.expect(o.set_state_rows(&hs, 2, 0, slot, ab(stacked), as).isOk());
        for (hs, 0..) |h, i| {
            var start = [_]c_int{ 0, 0, 0 };
            var stop = sh;
            start[ax] = @intCast(i);
            stop[ax] = @intCast(i + 1);
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, stacked, &start, 3, &stop, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            try expectSame(s, o.state(h, 0, slot), row);
        }
        const back = o.state_rows(&hs, 2, 0, slot, as);
        defer _ = mlx.mlx_array_free(mx(back));
        try expectSame(s, back, stacked);
    }
}

// ── tests ──

test "bases cache ops: the plain test cache passes its own conformance checks" {
    var c1 = try TestCache.init(testing.allocator, LAYERS);
    defer c1.deinit();
    var c2 = try TestCache.init(testing.allocator, LAYERS);
    defer c2.deinit();
    try expectConforms(&ops, c1.handle(), c2.handle());
}
