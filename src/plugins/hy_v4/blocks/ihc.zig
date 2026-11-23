//! Hy4-flash's hyper-connections (iHC): the DeepSeek-V4 form WITHOUT the Sinkhorn `comb` — a sublayer's 4
//! streams are mixed by one f32 linear over the flattened lanes, gated per lane before the sublayer (`pre`) and
//! per lane after it (`post`), and a final head collapses the streams to one hidden row.
//!
//! Reference (transformers `HYV4FlashHyperConnection` / `HYV4FlashHyperHead`): the mix runs entirely in f32,
//! `mixed = (flat @ fn^T) * rsqrt(mean(flat^2) + rms_norm_eps)`, `pre = sigmoid(logits*scale[0] + base[i]) +
//! hc_eps`, `post = hc_magnitude*sigmoid(logits*scale[1] + base[N+i]) + hc_eps`; the sublayer's update is spread
//! over the lanes by `post` and added to each lane's own residual. The head gates the collapsed lanes by
//! `sigmoid(logits*hc_scale + hc_base) + hc_eps`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");

const Array = ops.Array;
const Graph = ops.Graph;

/// One sublayer's mix: the lanes' weights and the per-lane post gate.
pub const Pre = struct {
    /// `[slots, rows, lanes]` f32: multiplies the sublayer's update per lane.
    post: Array,
    /// `[slots, rows, hidden]` in the activation dtype: the sublayer's input.
    mixed: Array,

    pub fn deinit(self: *const Pre) void {
        _ = mlx.mlx_array_free(self.post);
        _ = mlx.mlx_array_free(self.mixed);
    }
};

/// The parameters one sublayer's mix reads, pre-transposed at load (`fn_t` is `[lanes*hidden, 2*lanes]`).
pub const Params = struct {
    /// `[lanes*hidden, 2*lanes]` f32 (`hc_fn^T`).
    fn_t: Array,
    /// `[2*lanes]` f32: pre gates then post gates.
    base: Array,
    /// `[2]` f32: pre scale, post scale.
    scale: Array,
    lanes: u32,
    hidden: u32,
    /// `hc_magnitude`: the post gate's ceiling coefficient.
    magnitude: f32,
    /// `hc_eps`: added after each sigmoid.
    eps: f32,
    /// `rms_norm_eps`: the mix's normalizer.
    rms_eps: f32,
};

/// The head's collapse parameters: `hc_fn_t` `[lanes*hidden, lanes]`, `hc_base` `[lanes]`, `hc_scale` `[1]`.
pub const HeadParams = struct {
    fn_t: Array,
    base: Array,
    scale: Array,
    lanes: u32,
    hidden: u32,
    eps: f32,
    rms_eps: f32,
};

/// The mix of one sublayer: `(post, mixed)` from the streams `[slots, rows, lanes, hidden]`.
pub fn pre(g: *Graph, streams: Array, p: *const Params) !Pre {
    const n = ops.shape(streams)[0];
    const s = ops.shape(streams)[1];
    const lanes: c_int = @intCast(p.lanes);
    const flat_width: c_int = @intCast(p.lanes * p.hidden);

    var t = g.scope();
    defer t.deinit();
    const streams_f = try t.astype(streams, .float32);
    const flat = try t.reshape(streams_f, &.{ n, s, flat_width });
    const mixes = try t.matmul(flat, p.fn_t);
    const inv = try t.rsqrt(try t.addScalar(try t.mean(try t.square(flat), -1, true), p.rms_eps));
    const scaled = try t.mul(mixes, inv);

    const pre_logits = try t.slice(scaled, &.{ 0, 0, 0 }, &.{ n, s, lanes });
    const post_logits = try t.slice(scaled, &.{ 0, 0, lanes }, &.{ n, s, 2 * lanes });
    const pre_b = try t.slice(p.base, &.{0}, &.{lanes});
    const post_b = try t.slice(p.base, &.{lanes}, &.{2 * lanes});
    const pre_scale = try t.slice(p.scale, &.{0}, &.{1});
    const post_scale = try t.slice(p.scale, &.{1}, &.{2});

    const pre_gates = try t.addScalar(try t.sigmoid(try t.add(try t.mul(pre_logits, pre_scale), pre_b)), p.eps);
    const post_gates = try t.addScalar(
        try t.mulScalar(try t.sigmoid(try t.add(try t.mul(post_logits, post_scale), post_b)), p.magnitude),
        p.eps,
    );
    // sum(pre[l] * streams[l]) over the lanes, f32, then the activation dtype.
    const mixed = try t.astype(
        try t.sum(try t.mul(try t.expandDims(pre_gates, -1), streams_f), 2, false),
        ops.dtype(streams),
    );
    return .{ .post = t.escape(post_gates), .mixed = t.escape(mixed) };
}

/// The sublayer's residual write: lane `l` of the new stream is `post[l] * update + residual[l]`.
pub fn combine(g: *Graph, streams: Array, update: Array, post: Array) !Array {
    var t = g.scope();
    defer t.deinit();
    const streams_f = try t.astype(streams, .float32);
    const update_f = try t.expandDims(try t.astype(update, .float32), -2);
    const post_f = try t.expandDims(post, -1);
    const out = try t.astype(try t.add(try t.mul(post_f, update_f), streams_f), ops.dtype(streams));
    return t.escape(out);
}

/// The final collapse: the lanes' gates, then the weighted sum, in f32, cast to the activation dtype.
pub fn head(g: *Graph, streams: Array, hp: *const HeadParams) !Array {
    const n = ops.shape(streams)[0];
    const s = ops.shape(streams)[1];
    const flat_width: c_int = @intCast(hp.lanes * hp.hidden);

    var t = g.scope();
    defer t.deinit();
    const streams_f = try t.astype(streams, .float32);
    const flat = try t.reshape(streams_f, &.{ n, s, flat_width });
    const mixes = try t.matmul(flat, hp.fn_t);
    const inv = try t.rsqrt(try t.addScalar(try t.mean(try t.square(flat), -1, true), hp.rms_eps));
    const logits = try t.mul(mixes, inv);
    const gates = try t.addScalar(
        try t.sigmoid(try t.add(try t.mul(logits, hp.scale), hp.base)),
        hp.eps,
    );
    const out = try t.astype(try t.sum(try t.mul(try t.expandDims(gates, -1), streams_f), 2, false), ops.dtype(streams));
    return t.escape(out);
}

// ── tests ──

const testing = std.testing;

fn readF32(a: Array) ![]const f32 {
    try ops.eval(a);
    const n = mlx.mlx_array_size(a);
    const data = mlx.mlx_array_data_float32(a) orelse return error.NoData;
    return data[0..n];
}

/// The reference's f64 form of one sublayer mix, per row: no MLX, just the formula.
fn refPre(flat: []const f64, streams: []const f64, fn_rows: []const f64, base: []const f64, scale: []const f64, lanes: usize, hidden: usize, magnitude: f64, eps: f64, rms_eps: f64, mixed: []f64, post: []f64) void {
    const width = lanes * hidden;
    var mean_sq: f64 = 0;
    for (flat[0..width]) |v| mean_sq += v * v;
    mean_sq /= @floatFromInt(width);
    const inv = 1.0 / @sqrt(mean_sq + rms_eps);
    for (0..lanes) |l| {
        const g_pre = 1.0 / (1.0 + @exp(-((dot(flat[0..width], fn_rows[l * width ..][0..width]) * inv) * scale[0] + base[l]))) + eps;
        const g_post = magnitude / (1.0 + @exp(-((dot(flat[0..width], fn_rows[(lanes + l) * width ..][0..width]) * inv) * scale[1] + base[lanes + l]))) + eps;
        post[l] = g_post;
        for (0..hidden) |d| mixed[d] += g_pre * streams[l * hidden + d];
    }
}

fn dot(a: []const f64, b: []const f64) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| s += x * y;
    return s;
}

fn randF64(rng: *std.Random.DefaultPrng, out: []f64, lo: f64, hi: f64) void {
    for (out) |*v| v.* = lo + rng.random().float(f64) * (hi - lo);
}

test "hy_v4 ihc: the sublayer mix and post gate match the reference f64 form" {
    const lanes: usize = 4;
    const hidden: usize = 5;
    const rows: usize = 2;
    const width = lanes * hidden;
    var rng = std.Random.DefaultPrng.init(11);
    var streams: [2 * lanes * hidden]f64 = undefined;
    randF64(&rng, &streams, -1, 1);
    var fn_data: [2 * lanes * width]f64 = undefined; // [2*lanes, width]
    randF64(&rng, &fn_data, -0.5, 0.5);
    var base: [2 * lanes]f64 = undefined;
    randF64(&rng, &base, -0.2, 0.2);
    const scale = [2]f64{ 0.7, 1.3 };

    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const s_arr = try g.fromSlice(f32, toF32(streams[0 .. rows * width], a), &.{ 1, @intCast(rows), @intCast(lanes), @intCast(hidden) });
    defer g.drop(s_arr);
    // fn_t [width, 2*lanes], transposed from the reference's [2*lanes, width].
    var fn_t: [2 * lanes * width]f32 = undefined;
    for (0..2 * lanes) |i| {
        for (0..width) |j| fn_t[j * 2 * lanes + i] = @floatCast(fn_data[i * width + j]);
    }
    const fn_arr = try g.fromSlice(f32, &fn_t, &.{ @intCast(width), @intCast(2 * lanes) });
    defer g.drop(fn_arr);
    const base_arr = try g.fromSlice(f32, toF32(base[0..], a), &.{2 * lanes});
    defer g.drop(base_arr);
    const scale_arr = try g.fromSlice(f32, &[_]f32{ @floatCast(scale[0]), @floatCast(scale[1]) }, &.{2});
    defer g.drop(scale_arr);

    const p: Params = .{ .fn_t = fn_arr, .base = base_arr, .scale = scale_arr, .lanes = lanes, .hidden = hidden, .magnitude = 2.0, .eps = 1e-6, .rms_eps = 1e-5 };
    const r = try pre(&g, s_arr, &p);
    defer r.deinit();
    const got_mixed = try readF32(r.mixed);
    const got_post = try readF32(r.post);

    var want_mixed: [rows * hidden]f64 = @splat(0);
    var want_post: [rows * lanes]f64 = @splat(0);
    for (0..rows) |row| refPre(streams[row * width ..][0..width], streams[row * width ..][0..width], fn_data[0..], base[0..], scale[0..], lanes, hidden, 2.0, 1e-6, 1e-5, want_mixed[row * hidden ..][0..hidden], want_post[row * lanes ..][0..lanes]);
    for (got_mixed, want_mixed) |v, w| try testing.expectApproxEqAbs(@as(f32, @floatCast(w)), v, 2e-5);
    for (got_post, want_post) |v, w| try testing.expectApproxEqAbs(@as(f32, @floatCast(w)), v, 2e-5);
}

test "hy_v4 ihc: the sublayer write spreads the update over lanes by its post gate" {
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();

    var streams: [2 * 3 * 2]f32 = undefined;
    var post: [2 * 3]f32 = undefined;
    var update: [2 * 2]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(5);
    for (&streams) |*v| v.* = rng.random().float(f32) * 2 - 1;
    for (&post) |*v| v.* = rng.random().float(f32) * 2 - 1;
    for (&update) |*v| v.* = rng.random().float(f32) * 2 - 1;

    const s_arr = try g.fromSlice(f32, &streams, &.{ 1, 2, 3, 2 });
    defer g.drop(s_arr);
    const p_arr = try g.fromSlice(f32, &post, &.{ 1, 2, 3 });
    defer g.drop(p_arr);
    const u_arr = try g.fromSlice(f32, &update, &.{ 1, 2, 2 });
    defer g.drop(u_arr);

    const out = try combine(&g, s_arr, u_arr, p_arr);
    defer _ = mlx.mlx_array_free(out);
    const got = try readF32(out);
    for (0..2) |row| {
        for (0..3) |lane| {
            for (0..2) |d| {
                const want = post[row * 3 + lane] * update[row * 2 + d] + streams[(row * 3 + lane) * 2 + d];
                try testing.expectApproxEqAbs(want, got[(row * 3 + lane) * 2 + d], 1e-6);
            }
        }
    }
}

test "hy_v4 ihc: the head collapses the lanes by its own gates" {
    const lanes: usize = 4;
    const hidden: usize = 3;
    const rows: usize = 2;
    const width = lanes * hidden;
    var rng = std.Random.DefaultPrng.init(21);
    var streams: [rows * width]f64 = undefined;
    randF64(&rng, &streams, -1, 1);
    var fn_data: [lanes * width]f64 = undefined;
    randF64(&rng, &fn_data, -0.5, 0.5);
    var base: [lanes]f64 = undefined;
    randF64(&rng, &base, -0.2, 0.2);
    const scale = [1]f64{0.9};

    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const s_arr = try g.fromSlice(f32, toF32(streams[0..], a), &.{ 1, @intCast(rows), @intCast(lanes), @intCast(hidden) });
    defer g.drop(s_arr);
    var fn_t: [width * lanes]f32 = undefined;
    for (0..lanes) |i| {
        for (0..width) |j| fn_t[j * lanes + i] = @floatCast(fn_data[i * width + j]);
    }
    const fn_arr = try g.fromSlice(f32, &fn_t, &.{ @intCast(width), @intCast(lanes) });
    defer g.drop(fn_arr);
    const base_arr = try g.fromSlice(f32, toF32(base[0..], a), &.{lanes});
    defer g.drop(base_arr);
    const scale_arr = try g.fromSlice(f32, &[_]f32{@floatCast(scale[0])}, &.{1});
    defer g.drop(scale_arr);

    const hp: HeadParams = .{ .fn_t = fn_arr, .base = base_arr, .scale = scale_arr, .lanes = lanes, .hidden = hidden, .eps = 1e-6, .rms_eps = 1e-5 };
    const out = try head(&g, s_arr, &hp);
    defer _ = mlx.mlx_array_free(out);
    const got = try readF32(out);

    for (0..rows) |row| {
        const flat = streams[row * width ..][0..width];
        var mean_sq: f64 = 0;
        for (flat) |v| mean_sq += v * v;
        mean_sq /= @floatFromInt(width);
        const inv = 1.0 / @sqrt(mean_sq + 1e-5);
        for (0..hidden) |d| {
            var want: f64 = 0;
            for (0..lanes) |l| {
                const gate = 1.0 / (1.0 + @exp(-((dot(flat, fn_data[l * width ..][0..width]) * inv) * scale[0] + base[l]))) + 1e-6;
                want += gate * flat[l * hidden + d];
            }
            try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got[row * hidden + d], 2e-5);
        }
    }
}

fn toF32(src: []const f64, a: std.mem.Allocator) []f32 {
    const out = a.alloc(f32, src.len) catch @panic("test");
    for (src, out) |v, *o| o.* = @floatCast(v);
    return out;
}
