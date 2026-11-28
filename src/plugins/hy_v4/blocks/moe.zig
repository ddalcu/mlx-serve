//! Hy4-flash's FFN sublayers: the layer-0 dense MLP (SwiGLU + `post_fc2_norm` on the way out), and the latent
//! MoE (ℓ-MoE): the routed experts run in the LATENT width — `routed_expert_down_proj` maps hidden -> latent,
//! the 512 fused expert banks run SwiGLU (clamp 10) in the latent, `routed_expert_up_proj` maps back — while the
//! shared expert and the router see the full width, and `combine_output_norm` closes the block.
//!
//! Reference (`HYV4FlashMoE`): `routed = up(down(x) -> experts)`, `hidden = routed + shared(x)`, then
//! `combine_output_norm`. The shared expert is a plain MLP of width `moe_intermediate_size * n_shared_experts`
//! with NO post norm of its own.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const proj = @import("proj.zig");
const router = @import("router.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Proj = proj.Proj;

pub const Dense = struct {
    gate: Proj,
    up: Proj,
    down: Proj,
    /// `post_fc2_norm.weight`, the sandwich norm on the dense sublayer's output.
    post_fc2_norm: Array,
    rms_eps: f32,

    pub fn forward(d: *const Dense, g: *Graph, x: Array) !Array {
        const gated = try proj.swiGLU(g, try d.gate.forward(g, x), try d.up.forward(g, x));
        const y = try d.down.forward(g, gated);
        return g.rmsNorm(y, d.post_fc2_norm, d.rms_eps);
    }
};

pub const Moe = struct {
    router: router.Params,
    /// Packed `[hidden, latent]` (maps full width into the latent).
    routed_down: Proj,
    /// Packed `[latent, hidden]` (maps the routed latent sum back out).
    routed_up: Proj,
    /// `[experts, 2*moe_intermediate, latent/8]` fused gate+up.
    gate_up: proj.ExpertBank,
    /// `[experts, latent, moe_intermediate/8]` down.
    down: proj.ExpertBank,
    shared_gate: Proj,
    shared_up: Proj,
    shared_down: Proj,
    combine_output_norm: Array,
    rms_eps: f32,
    swiglu_limit: f32,

    pub fn forward(m: *const Moe, g: *Graph, x: Array) !Array {
        var t = g.scope();
        defer t.deinit();
        const sh = ops.shape(x);
        const n = sh[0];
        const s = sh[1];
        const k: c_int = @intCast(m.router.top_k);

        const choice = try router.route(&t, x, &m.router);

        // The latent MoE: hidden -> latent, the fused expert banks, latent -> hidden.
        const latent = try m.routed_down.forward(&t, x);
        const ids = choice.ids; // [n, s, k]
        const gu = try m.gate_up.gather(&t, latent, ids); // [n, s, k, 2*moe_intermediate]
        const mi = @divExact(ops.shape(gu)[3], 2);
        const gate = try t.slice(gu, &.{ 0, 0, 0, 0 }, &.{ n, s, k, mi });
        const up = try t.slice(gu, &.{ 0, 0, 0, mi }, &.{ n, s, k, 2 * mi });
        const act = try proj.clampedSwiGLU(&t, gate, up, m.swiglu_limit);
        const down = try m.down.gather(&t, act, ids); // [n, s, k, latent]
        const weighted = try t.mul(down, try t.expandDims(choice.weights, -1)); // [n, s, k, latent]
        const summed = try t.sum(weighted, 2, false); // [n, s, latent]
        const routed = try m.routed_up.forward(&t, summed); // [n, s, hidden]

        // The shared expert, full width, no norm of its own.
        const sg = try m.shared_gate.forward(&t, x);
        const su = try m.shared_up.forward(&t, x);
        const shared = try m.shared_down.forward(&t, try proj.swiGLU(&t, sg, su));

        const sum = try t.add(routed, shared);
        return g.adopt(t.escape(try t.rmsNorm(sum, m.combine_output_norm, m.rms_eps)));
    }
};

// ── tests ──

const testing = std.testing;

fn readF32(a: Array) ![]const f32 {
    try ops.eval(a);
    const n = mlx.mlx_array_size(a);
    const data = mlx.mlx_array_data_float32(a) orelse return error.NoData;
    return data[0..n];
}

/// One dense bf16 linear over the reference's f64 row form (`w` `[out, in]`), for the oracle below.
fn refLinear(x: []const f64, w: []const f64, out_dim: usize, in_dim: usize, out: []f64) void {
    for (0..out_dim) |o| {
        var acc: f64 = 0;
        for (0..in_dim) |i| acc += x[i] * w[o * in_dim + i];
        out[o] = acc;
    }
}

test "hy_v4 moe: the latent expert path, the shared expert and the combine norm match the reference f64 form" {
    const hidden: usize = 8;
    const latent: usize = 4;
    const mi: usize = 4;
    const experts: usize = 4;
    const top_k: usize = 2;
    const limit = 10.0;
    const scale = 3.998;

    var rng = std.Random.DefaultPrng.init(31);
    const arena = testing.allocator;
    const x = try arena.alloc(f64, hidden);
    defer arena.free(x);
    const gate_w = try arena.alloc(f64, experts * hidden);
    defer arena.free(gate_w);
    const bias = try arena.alloc(f64, experts);
    defer arena.free(bias);
    const rd_w = try arena.alloc(f64, latent * hidden); // [latent, hidden]
    defer arena.free(rd_w);
    const ru_w = try arena.alloc(f64, hidden * latent); // [hidden, latent]
    defer arena.free(ru_w);
    const gu_w = try arena.alloc(f64, experts * 2 * mi * latent);
    defer arena.free(gu_w);
    const dn_w = try arena.alloc(f64, experts * latent * mi);
    defer arena.free(dn_w);
    const sg_w = try arena.alloc(f64, mi * hidden);
    defer arena.free(sg_w);
    const su_w = try arena.alloc(f64, mi * hidden);
    defer arena.free(su_w);
    const sd_w = try arena.alloc(f64, hidden * mi);
    defer arena.free(sd_w);
    const norm = try arena.alloc(f64, hidden);
    defer arena.free(norm);
    for (x) |*v| v.* = rng.random().float(f64) * 2 - 1;
    for (gate_w) |*v| v.* = rng.random().float(f64) * 0.6 - 0.3;
    for (bias) |*v| v.* = rng.random().float(f64) * 0.4 - 0.2;
    for (rd_w) |*v| v.* = rng.random().float(f64) * 0.8 - 0.4;
    for (ru_w) |*v| v.* = rng.random().float(f64) * 0.8 - 0.4;
    for (gu_w) |*v| v.* = rng.random().float(f64) * 1.2 - 0.6;
    for (dn_w) |*v| v.* = rng.random().float(f64) * 1.2 - 0.6;
    for (sg_w) |*v| v.* = rng.random().float(f64) * 0.8 - 0.4;
    for (su_w) |*v| v.* = rng.random().float(f64) * 0.8 - 0.4;
    for (sd_w) |*v| v.* = rng.random().float(f64) * 0.8 - 0.4;
    for (norm) |*v| v.* = 1 + rng.random().float(f64) * 0.2 - 0.1;

    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    const aa = a.allocator();

    const x_arr = try g.fromSlice(f32, try toF32(x, aa), &.{ 1, 1, @intCast(hidden) });
    defer g.drop(x_arr);

    // Build the dense (unquantized) projections the pack would store packed: here bf16 pre-transposed.
    const gate_t = try preTranspose(gate_w, experts, hidden, aa); // [hidden, experts]
    const gate_arr = try g.fromSlice(f32, gate_t, &.{ @intCast(hidden), @intCast(experts) });
    defer g.drop(gate_arr);
    const bias_arr = try g.fromSlice(f32, try toF32(bias, aa), &.{@intCast(experts)});
    defer g.drop(bias_arr);
    // Expert banks as dense [experts, out, in] gathered with MLX's gather_mm (dense arm).
    const gu_arr = try g.fromSlice(f32, try toF32(gu_w, aa), &.{ @intCast(experts), 2 * mi, @intCast(latent) });
    defer g.drop(gu_arr);
    const dn_arr = try g.fromSlice(f32, try toF32(dn_w, aa), &.{ @intCast(experts), latent, @intCast(mi) });
    defer g.drop(dn_arr);
    const rd_arr = try g.fromSlice(f32, try preTranspose(rd_w, latent, hidden, aa), &.{ @intCast(hidden), @intCast(latent) });
    defer g.drop(rd_arr);
    const ru_arr = try g.fromSlice(f32, try preTranspose(ru_w, hidden, latent, aa), &.{ @intCast(latent), @intCast(hidden) });
    defer g.drop(ru_arr);
    const sg_arr = try g.fromSlice(f32, try preTranspose(sg_w, mi, hidden, aa), &.{ @intCast(hidden), @intCast(mi) });
    defer g.drop(sg_arr);
    const su_arr = try g.fromSlice(f32, try preTranspose(su_w, mi, hidden, aa), &.{ @intCast(hidden), @intCast(mi) });
    defer g.drop(su_arr);
    const sd_arr = try g.fromSlice(f32, try preTranspose(sd_w, hidden, mi, aa), &.{ @intCast(mi), @intCast(hidden) });
    defer g.drop(sd_arr);
    const norm_arr = try g.fromSlice(f32, try toF32(norm, aa), &.{@intCast(hidden)});
    defer g.drop(norm_arr);

    // The f64 oracle: router choice, latent path, swiglu clamp, shared expert, combine norm.
    var scores: [experts]f64 = undefined;
    for (0..experts) |e| {
        var logit: f64 = 0;
        for (0..hidden) |i| logit += x[i] * gate_w[e * hidden + i];
        scores[e] = 1.0 / (1.0 + @exp(-logit));
    }
    var choice: [experts]f64 = undefined;
    for (0..experts) |e| choice[e] = scores[e] + bias[e];
    var best: [top_k]usize = if (choice[0] >= choice[1]) .{ 0, 1 } else .{ 1, 0 };
    for (2..experts) |e| {
        if (choice[e] > choice[best[0]]) {
            best[1] = best[0];
            best[0] = e;
        } else if (choice[e] > choice[best[1]]) best[1] = e;
    }
    if (best[0] > best[1]) std.mem.swap(usize, &best[0], &best[1]);
    var wts: [top_k]f64 = undefined;
    for (best, 0..) |e, i| wts[i] = scores[e];
    const wsum = wts[0] + wts[1] + 1e-20;
    for (&wts) |*w| w.* = w.* / wsum * scale;

    var latent_x: [latent]f64 = undefined;
    refLinear(x, rd_w, latent, hidden, &latent_x);
    var routed: [latent]f64 = @splat(0);
    for (best, 0..) |e, slot| {
        var gu: [2 * mi]f64 = undefined;
        refLinear(&latent_x, gu_w[e * 2 * mi * latent ..][0 .. 2 * mi * latent], 2 * mi, latent, &gu);
        var act: [mi]f64 = undefined;
        for (0..mi) |i| {
            const gv = @min(gu[i], limit);
            const uv = @max(@min(gu[mi + i], limit), -limit);
            act[i] = (gv / (1.0 + @exp(-gv))) * uv;
        }
        var dv: [latent]f64 = undefined;
        refLinear(&act, dn_w[e * latent * mi ..][0 .. latent * mi], latent, mi, &dv);
        for (0..latent) |i| routed[i] += wts[slot] * dv[i];
    }
    var routed_hidden: [hidden]f64 = undefined;
    refLinear(&routed, ru_w, hidden, latent, &routed_hidden);
    var sgate: [mi]f64 = undefined;
    refLinear(x, sg_w, mi, hidden, &sgate);
    var sup: [mi]f64 = undefined;
    refLinear(x, su_w, mi, hidden, &sup);
    var sact: [mi]f64 = undefined;
    for (0..mi) |i| sact[i] = (sgate[i] / (1.0 + @exp(-sgate[i]))) * sup[i];
    var shared: [hidden]f64 = undefined;
    refLinear(&sact, sd_w, hidden, mi, &shared);
    var combined: [hidden]f64 = undefined;
    var mean_sq: f64 = 0;
    for (0..hidden) |i| {
        combined[i] = routed_hidden[i] + shared[i];
        mean_sq += combined[i] * combined[i];
    }
    mean_sq /= @floatFromInt(hidden);
    const inv = 1.0 / @sqrt(mean_sq + 1e-5);

    // A dense MoE (no quantized banks): the component's dense arm gathers with gather_mm.
    const moe = Moe{
        .router = .{ .gate_t = gate_arr, .bias = bias_arr, .experts = experts, .hidden = hidden, .top_k = top_k, .norm = true, .scale = scale, .n_group = 1, .topk_group = 1 },
        .routed_down = .{ .w = rd_arr },
        .routed_up = .{ .w = ru_arr },
        .gate_up = .{ .w = gu_arr },
        .down = .{ .w = dn_arr },
        .shared_gate = .{ .w = sg_arr },
        .shared_up = .{ .w = su_arr },
        .shared_down = .{ .w = sd_arr },
        .combine_output_norm = norm_arr,
        .rms_eps = 1e-5,
        .swiglu_limit = limit,
    };
    const out = try moe.forward(&g, x_arr);
    defer g.drop(out);
    const got = try readF32(out);
    for (0..hidden) |i| {
        const want = norm[i] * combined[i] * inv;
        try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got[i], 3e-4);
    }
}

fn toF32(src: []const f64, a: std.mem.Allocator) ![]f32 {
    const out = try a.alloc(f32, src.len);
    for (src, out) |v, *o| o.* = @floatCast(v);
    return out;
}

/// `w` `[out, in]` as `[in, out]` f32 (the dense arm's pre-transposed form).
fn preTranspose(w: []const f64, out_dim: usize, in_dim: usize, a: std.mem.Allocator) ![]f32 {
    const t = try a.alloc(f32, w.len);
    for (0..out_dim) |o| {
        for (0..in_dim) |i| t[i * out_dim + o] = @floatCast(w[o * in_dim + i]);
    }
    return t;
}
