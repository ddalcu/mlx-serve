//! Inkling 2's routed experts: the f32 sigmoid router with a selection bias, one gather over the interleaved
//! gate_up rows, the softcapped SwiGLU, the down gather and the f32 combine. An NVFP4 bank's per-expert global scale
//! is applied in f32: on gate_up's rows before the softcaps, folded into the routing weights for down.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const routing = @import("../../sdk/blocks/moe/routing.zig");
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const SdkBank = @import("../../sdk/bases/blocks/moe.zig").Bank;

const Array = ops.Array;
const Graph = ops.Graph;

pub const Weights = struct {
    /// f32 [E, D].
    router: Array,
    /// f32 [E], selection only.
    bias: Array,
    /// Out width 2I, gate and up rows alternating; `expert_scale` f32 [E] or null ctx.
    gate_up: SdkBank,
    down: SdkBank,
    top_k: u32,
    inter: u32,
    route_scale: f32,
    gate_cap: f32,
    up_cap: f32,
};

/// x [1,L,D] (dtype dt) -> [1,L,D] (dtype dt).
pub fn forward(w: *const Weights, g: *Graph, x: Array) !Array {
    var t = g.scope();
    defer t.deinit();
    const sh = ops.shape(x);
    const rows = sh[0] * sh[1];
    const d = sh[2];
    const k: c_int = @intCast(w.top_k);
    const n = rows * k;
    const xf = try t.reshape(x, &.{ rows, d });
    const logits = try t.matmul(try t.astype(xf, .float32), try t.transpose(w.router, &.{ 1, 0 }));
    const r = try routing.groupLimitedRouting(logits, w.bias, k, 1, 1, true, w.route_scale, t.s);
    const inds = t.adopt(r.inds) catch |e| {
        _ = mlx.mlx_array_free(r.norm_scores);
        return e;
    };
    var ws = try t.adopt(r.norm_scores);

    // As mlx-lm: rows are sorted by expert once there are 64 of them.
    const sorted = n >= 64;
    const flat = try t.reshape(inds, &.{n});
    const order = if (sorted) try t.argsort(flat, 0) else try t.arange(0, @floatFromInt(n), 1, .uint32);
    const rhs = if (sorted) try t.take(flat, order, 0) else flat;
    const tok = try t.call(mlx.mlx_floor_divide, .{ order, try t.scalar(@floatFromInt(k), .uint32) });
    const xr = try t.reshape(try t.take(xf, tok, 0), &.{ n, 1, d });

    var gu = try gather(&t, xr, &w.gate_up, rhs, sorted);
    const in_dt = ops.dtype(gu);
    if (w.gate_up.expert_scale.ctx != null) {
        // [n,1,1]: a [n,1] scale would broadcast against [n,1,2I] to [n,n,2I].
        const sc = try t.reshape(try t.take(w.gate_up.expert_scale, rhs, 0), &.{ n, 1, 1 });
        gu = try t.astype(try t.mul(try t.astype(gu, .float32), sc), in_dt);
    }
    const inter: c_int = @intCast(w.inter);
    const pairs = try t.reshape(gu, &.{ n, inter, 2 });
    const gt = try t.astype(try t.reshape(try t.sliceAxis(pairs, 2, 0, 1), &.{ n, 1, inter }), .float32);
    const up = try t.astype(try t.reshape(try t.sliceAxis(pairs, 2, 1, 2), &.{ n, 1, inter }), .float32);
    const gc = try t.scalar(w.gate_cap, .float32);
    const uc = try t.scalar(w.up_cap, .float32);
    const gate_capped = try t.mul(try t.tanh(try t.div(gt, gc)), gc);
    const up_capped = try t.mul(try t.tanh(try t.div(up, uc)), uc);
    const act = try t.astype(try t.mul(try t.mul(gate_capped, try t.sigmoid(gt)), up_capped), in_dt);

    var dn = try t.reshape(try gather(&t, act, &w.down, rhs, sorted), &.{ n, d });
    if (sorted) dn = try t.take(dn, try t.argsort(order, 0), 0);
    if (w.down.expert_scale.ctx != null) ws = try t.mul(ws, try t.take(w.down.expert_scale, inds, 0));
    const dn32 = try t.astype(try t.reshape(dn, &.{ rows, k, d }), .float32);
    const y = try t.sum(try t.mul(dn32, try t.expandDims(ws, -1)), 1, false);
    return g.keep(try t.reshape(try t.astype(y, ops.dtype(x)), sh));
}

fn gather(g: *Graph, x: Array, bank: *const SdkBank, rhs: Array, sorted: bool) !Array {
    var y = mlx.mlx_array_new();
    gatherExpertMm(&y, x, bank.w, bank.s, bank.b, .{ .ctx = null }, rhs, bank.qp.bits, bank.qp.group_size, bank.qp.mode, sorted, g.s) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return g.adopt(y);
}

// ── tests ──

const testing = std.testing;
const testReadF32 = @import("../../sdk/testing/parity.zig").testReadF32;
const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;

fn sigmoid(v: f64) f64 {
    return 1.0 / (1.0 + @exp(-v));
}

test "inkling2 moe: sorted and unsorted routing with expert scales match a host loop" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const E = 16;
    const D = 32;
    const I = 16;
    const K = 4;
    const route_scale: f32 = 2.5;
    const gate_cap: f32 = 2.0;
    const up_cap: f32 = 3.0;

    var router: [E * D]f32 = undefined;
    var bias: [E]f32 = undefined;
    // Dense banks are stored [E, in, out], as the loader pre-transposes them.
    var gu: [E * D * 2 * I]f32 = undefined;
    var dn: [E * I * D]f32 = undefined;
    var gu_scale: [E]f32 = undefined;
    var dn_scale: [E]f32 = undefined;
    for (&router) |*v| v.* = rnd.floatNorm(f32) * 0.5;
    for (&bias) |*v| v.* = rnd.floatNorm(f32) * 0.1;
    for (&gu) |*v| v.* = rnd.floatNorm(f32) * 0.3;
    for (&dn) |*v| v.* = rnd.floatNorm(f32) * 0.3;
    for (&gu_scale) |*v| v.* = 0.5 + rnd.float(f32);
    for (&dn_scale) |*v| v.* = 0.5 + rnd.float(f32);
    const qp: QuantParams = .{ .bits = 0, .group_size = 0 };
    const w = Weights{
        .router = try g.fromSlice(f32, &router, &.{ E, D }),
        .bias = try g.fromSlice(f32, &bias, &.{E}),
        .gate_up = .{ .w = try g.fromSlice(f32, &gu, &.{ E, D, 2 * I }), .qp = qp, .expert_scale = try g.fromSlice(f32, &gu_scale, &.{E}) },
        .down = .{ .w = try g.fromSlice(f32, &dn, &.{ E, I, D }), .qp = qp, .expert_scale = try g.fromSlice(f32, &dn_scale, &.{E}) },
        .top_k = K,
        .inter = I,
        .route_scale = route_scale,
        .gate_cap = gate_cap,
        .up_cap = up_cap,
    };

    // T*K = 4 runs unsorted, 160 sorted.
    for ([_]usize{ 1, 40 }) |t_rows| {
        const x = try a.alloc(f32, t_rows * D);
        defer a.free(x);
        for (x) |*v| v.* = rnd.floatNorm(f32);
        const y = try forward(&w, &g, try g.fromSlice(f32, x, &.{ 1, @intCast(t_rows), D }));
        try testing.expectEqual(mlx.mlx_dtype.float32, ops.dtype(y));
        try testing.expectEqualSlices(c_int, &.{ 1, @intCast(t_rows), D }, ops.shape(y));
        const got = try a.alloc(f32, t_rows * D);
        defer a.free(got);
        try testReadF32(y, got, s);

        var diff: f64 = 0;
        var mag: f64 = 0;
        for (0..t_rows) |t| {
            const xt = x[t * D ..][0..D];
            var score: [E]f64 = undefined;
            var biased: [E]f64 = undefined;
            for (0..E) |e| {
                var l: f64 = 0;
                for (xt, router[e * D ..][0..D]) |u, v| l += @as(f64, u) * v;
                score[e] = sigmoid(l);
                biased[e] = score[e] + bias[e];
            }
            var sel: [K]usize = undefined;
            var taken: [E]bool = @splat(false);
            var sum: f64 = 0;
            for (&sel) |*o| {
                var best: usize = 0;
                while (taken[best]) best += 1;
                for (0..E) |e| if (!taken[e] and biased[e] > biased[best]) {
                    best = e;
                };
                taken[best] = true;
                o.* = best;
                sum += score[best];
            }
            var out: [D]f64 = @splat(0);
            for (sel) |e| {
                const rw = score[e] / (sum + 1e-20) * route_scale;
                var act: [I]f64 = undefined;
                for (&act, 0..) |*o, j| {
                    var gt: f64 = 0;
                    var up: f64 = 0;
                    for (xt, 0..) |u, d| {
                        const rowp = gu[(e * D + d) * 2 * I ..];
                        gt += @as(f64, u) * rowp[2 * j];
                        up += @as(f64, u) * rowp[2 * j + 1];
                    }
                    gt *= gu_scale[e];
                    up *= gu_scale[e];
                    o.* = std.math.tanh(gt / gate_cap) * gate_cap * sigmoid(gt) * std.math.tanh(up / up_cap) * up_cap;
                }
                for (&out, 0..) |*o, d| {
                    var v: f64 = 0;
                    for (act, 0..) |u, j| v += u * dn[(e * I + j) * D + d];
                    o.* += rw * dn_scale[e] * v;
                }
            }
            for (out, got[t * D ..][0..D]) |want, have| {
                diff = @max(diff, @abs(have - want));
                mag = @max(mag, @abs(want));
            }
        }
        try testing.expect(diff / mag < 1e-5);
    }
}
