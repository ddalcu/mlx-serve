//! A DeepSeek-V4.1 model at the release's shapes over random weights in the served formats, built through the plugin's
//! own `init` as a pair worker holding layers [lo, hi) (no embed, head or Engram), its state prefilled with `ctx`
//! random stream rows through those layers: what the decode kernels' tests and ubenches run against.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const resident = @import("../../../sdk/weights/resident.zig");
const sdk_fixture = @import("../../../sdk/testing/fixture.zig");
const config = @import("../config.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const device = @import("../../../mlx/device.zig");
const Shard = @import("../../../sdk/weights/shard.zig").Shard;

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const Opts = struct {
    lo: usize = 20,
    hi: usize = 25,
    /// Routed experts a layer (the release has 384; tests use few, a ubench the full bank).
    experts: u32 = 16,
    /// Positions prefilled before the decode step under test.
    ctx: usize = 1024,
    seed: u64 = 1,
    /// The activation dtype (f32: the reference a bf16 model's decode is measured against).
    act: mlx.mlx_dtype = .bfloat16,
    /// The model decodes on the fused kernels (`Dsv41Model.dec`); off, today's chain, what a kernel is measured against.
    fused: bool = false,
    /// The routed experts as the release ships them (mxfp4, group 32) instead of the iQ pack's affine 2/3-bit.
    experts_fp4: bool = false,
    /// This rank's slice of every expert and attention head, cut as the loader cuts it (`Shard.axis`).
    tp: ?Shard = null,
    /// Every routed projection affine at this width over groups of 128 (the expert-pruned packs) instead of the iQ
    /// pack's plan (gate/up 2 or 3 bits, down 3, groups of 64).
    experts_affine: ?u32 = null,
};

pub const Fixture = struct {
    gpa: std.mem.Allocator,
    m: *dsv41.Dsv41Model,
    w: resident.Weights,
    seed: u64,
    draws: u64 = 0,
    /// The routed experts' affine group.
    group: c_int = 64,

    pub fn init(gpa: std.mem.Allocator, opts: Opts) !*Fixture {
        const fx = try gpa.create(Fixture);
        errdefer gpa.destroy(fx);
        var cfg = try releaseConfig();
        cfg.num_experts = opts.experts;
        fx.* = .{ .gpa = gpa, .m = undefined, .w = resident.Weights.init(gpa), .seed = opts.seed, .group = if (opts.experts_affine != null) 128 else 64 };
        errdefer fx.w.deinit();
        for (opts.lo..opts.hi) |li| try fx.putLayer(&cfg, "layers.{d}.", li, opts.experts, opts.experts_fp4, opts.experts_affine);
        if (opts.tp) |t| try fx.sliceShard(t);
        fx.m = try dsv41.init(gpa, &cfg, &fx.w, mlx.gpuStream(), .{ .pair = .{ .lo = opts.lo, .hi = opts.hi, .worker = true }, .act = opts.act, .tp = opts.tp });
        errdefer fx.m.deinit();
        if (!opts.fused) fx.m.dec.deinit();
        var p0: usize = 0;
        while (p0 < opts.ctx) {
            const L = @min(opts.ctx - p0, 2048);
            const x = try fx.stream(L);
            defer _ = mlx.mlx_array_free(x);
            const pre = try fx.cpre(L);
            defer _ = mlx.mlx_array_free(pre);
            const out = try dsv41.workerStep(fx.m, gpa, x, pre, p0, false, false);
            for (out) |o| _ = mlx.mlx_array_free(o);
            p0 += L;
        }
        if (opts.ctx == 0) fx.m.dec_state = try dsv41.initState(fx.m, gpa);
        return fx;
    }

    /// Every tensor `Shard.axis` names cut to `t`'s part of its axis, in a buffer of its own.
    fn sliceShard(fx: *Fixture, t: Shard) !void {
        const s = mlx.gpuStream();
        var it = fx.w.map.iterator();
        while (it.next()) |kv| {
            const axis = Shard.axis(kv.key_ptr.*) orelse continue;
            const shape = mlx.getShape(kv.value_ptr.*);
            const ax = axis.of(shape.len);
            const part: c_int = @divExact(shape[ax], @as(c_int, @intCast(t.size)));
            var start: [4]c_int = @splat(0);
            var stop: [4]c_int = undefined;
            const strides: [4]c_int = @splat(1);
            @memcpy(stop[0..shape.len], shape);
            start[ax] = part * @as(c_int, @intCast(t.rank));
            stop[ax] = start[ax] + part;
            var view = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(view);
            try mlx.check(mlx.mlx_slice(&view, kv.value_ptr.*, &start, shape.len, &stop, shape.len, &strides, shape.len, s));
            var copy = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(copy);
            try mlx.check(mlx.mlx_contiguous(&copy, view, false, s));
            try mlx.check(mlx.mlx_array_eval(copy));
            _ = mlx.mlx_array_free(kv.value_ptr.*);
            kv.value_ptr.* = copy;
        }
    }

    pub fn deinit(fx: *Fixture) void {
        fx.m.deinit();
        fx.w.deinit();
        fx.gpa.destroy(fx);
    }

    /// The decode state (positions [0, ctx) filled).
    pub fn state(fx: *Fixture) *dsv41.State {
        return &fx.m.dec_state.?;
    }

    pub fn layer(fx: *Fixture, li: usize) *dsv41.Layer {
        return &fx.m.layers[li];
    }

    /// Random stream rows `[R, hc, dim]`, bf16 values in the model's activation dtype (caller frees).
    pub fn stream(fx: *Fixture, rows: usize) !A {
        const c = &fx.m.cfg;
        const x = try fx.normal(&.{ @intCast(rows), @intCast(c.dsv4_hc_mult), @intCast(c.hidden_size) }, .bfloat16, 0, 1);
        if (fx.m.act == .bfloat16) return x;
        defer _ = mlx.mlx_array_free(x);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&o, x, fx.m.act, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(o));
        return o;
    }

    /// A previous mix `[R, hc]` f32 as the hc split makes it (positive, summing near 1; caller frees).
    pub fn cpre(fx: *Fixture, rows: usize) !A {
        return fx.uniform(&.{ @intCast(rows), @intCast(fx.m.cfg.dsv4_hc_mult) }, 0.05, 0.45);
    }

    /// N(mean, std) of `shape` in `dt` (caller frees).
    pub fn normal(fx: *Fixture, shape: []const c_int, dt: mlx.mlx_dtype, mean: f32, std_: f32) !A {
        const k = try fx.nextKey();
        defer _ = mlx.mlx_array_free(k);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dt, mean, std_, k, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(o));
        return o;
    }

    pub fn uniform(fx: *Fixture, shape: []const c_int, lo: f32, hi: f32) !A {
        const k = try fx.nextKey();
        defer _ = mlx.mlx_array_free(k);
        const s = mlx.gpuStream();
        const l = mlx.mlx_array_new_float(lo);
        defer _ = mlx.mlx_array_free(l);
        const h = mlx.mlx_array_new_float(hi);
        defer _ = mlx.mlx_array_free(h);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_uniform(&o, l, h, shape.ptr, shape.len, .float32, k, s));
        try mlx.check(mlx.mlx_array_eval(o));
        return o;
    }

    fn nextKey(fx: *Fixture) !A {
        fx.draws += 1;
        var k = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_key(&k, fx.seed *% 1_000_003 +% fx.draws));
        return k;
    }

    fn put(fx: *Fixture, name: []const u8, x: A) !void {
        try fx.w.put(name, x);
    }

    fn putNamed(fx: *Fixture, comptime fmt: []const u8, args: anytype, x: A) !void {
        var buf: [160]u8 = undefined;
        try fx.put(try std.fmt.bufPrint(&buf, fmt, args), x);
    }

    /// `base`.weight (+ .scales, .biases) of an `[out, in]` linear (or `[E, out, in]` bank) in `fmt`.
    fn putLin(fx: *Fixture, comptime base: []const u8, args: anytype, shape: []const c_int, fmt: Fmt) !void {
        const fan_in: f32 = @floatFromInt(shape[shape.len - 1]);
        const w = try fx.normal(shape, .bfloat16, 0, 1 / @sqrt(fan_in));
        switch (fmt) {
            .bf16 => return fx.putNamed(base ++ ".weight", args, w),
            .fp8_exact => {
                // Values an mxfp8 code holds exactly, kept as bf16 (the release's fp8 wo_a).
                defer _ = mlx.mlx_array_free(w);
                const q = try quantize(w, 32, 8, "mxfp8");
                defer for (q) |a| if (a.ctx != null) {
                    _ = mlx.mlx_array_free(a);
                };
                var d = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_dequantize(&d, q[0], q[1], none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .bfloat16, .has_value = true }, mlx.gpuStream()));
                try mlx.check(mlx.mlx_array_eval(d));
                return fx.putNamed(base ++ ".weight", args, d);
            },
            .mxfp8, .mxfp4, .affine => {
                defer _ = mlx.mlx_array_free(w);
                const q = if (fmt == .mxfp8) try quantize(w, 32, 8, "mxfp8") else if (fmt == .mxfp4) try quantize(w, 32, 4, "mxfp4") else try quantize(w, fx.group, @intCast(fmt.affine), "affine");
                try fx.putNamed(base ++ ".weight", args, q[0]);
                try fx.putNamed(base ++ ".scales", args, q[1]);
                if (q[2].ctx != null) try fx.putNamed(base ++ ".biases", args, q[2]);
            },
        }
    }

    fn putVec(fx: *Fixture, comptime name: []const u8, args: anytype, shape: []const c_int, dt: mlx.mlx_dtype, mean: f32, std_: f32) !void {
        try fx.putNamed(name, args, try fx.normal(shape, dt, mean, std_));
    }

    /// One trunk layer's tensors under the names `loadLayer` reads, in the release's formats.
    /// Layer `li`'s tensors under `pfx` (a trunk layer's, or a DSpark stage's under "mtp.{d}.").
    pub fn putLayer(fx: *Fixture, c: *const config.Config, comptime pfx: []const u8, li: usize, E: u32, fp4: bool, affine: ?u32) !void {
        const d: c_int = @intCast(c.hidden_size);
        const hc: c_int = @intCast(c.dsv4_hc_mult);
        const mix = (2 + hc) * hc;
        const H: c_int = @intCast(c.num_attention_heads);
        const hd: c_int = @intCast(c.head_dim);
        const ql: c_int = @intCast(c.dsv4_q_lora_rank);
        const og: c_int = @intCast(c.dsv4_o_groups);
        const ol: c_int = @intCast(c.dsv4_o_lora_rank);
        const inter: c_int = @intCast(c.moe_intermediate_size);
        const ih: c_int = @intCast(c.dsv4_index_n_heads);
        const ihd: c_int = @intCast(c.dsv4_index_head_dim);
        const e: c_int = @intCast(E);
        const kv_src = (c.dsv41_kv_sources >> @intCast(li)) & 1 != 0;
        const ix_src = (c.dsv41_index_sources >> @intCast(li)) & 1 != 0;
        const ratio = c.dsv4_compress_ratios[li];
        // The release's expert plan: gate/up 3-bit at layers 1-9 and 14, 2-bit elsewhere; down 3-bit.
        const gu_bits: u32 = affine orelse if ((li >= 1 and li <= 9) or li == 14) 3 else 2;
        try fx.putVec(pfx ++ "attn_norm.weight", .{li}, &.{d}, .bfloat16, 1, 0.1);
        try fx.putVec(pfx ++ "ffn_norm.weight", .{li}, &.{d}, .bfloat16, 1, 0.1);
        try fx.putVec(pfx ++ "attn.q_norm.weight", .{li}, &.{ql}, .bfloat16, 1, 0.1);
        try fx.putVec(pfx ++ "attn.kv_norm.weight", .{li}, &.{hd}, .bfloat16, 1, 0.1);
        inline for (.{ "attn", "ffn" }) |which| {
            try fx.putVec(pfx ++ "hc_" ++ which ++ "_fn", .{li}, &.{ mix, hc * d }, .float32, 0, 0.01);
            try fx.putVec(pfx ++ "hc_" ++ which ++ "_scale", .{li}, &.{3}, .float32, 0.5, 0.1);
            try fx.putVec(pfx ++ "hc_" ++ which ++ "_base", .{li}, &.{mix}, .float32, 0, 0.5);
        }
        try fx.putLin(pfx ++ "attn.wq_a", .{li}, &.{ ql, d }, .mxfp8);
        try fx.putLin(pfx ++ "attn.wq_b", .{li}, &.{ H * hd, ql }, .mxfp8);
        try fx.putLin(pfx ++ "attn.wkv", .{li}, &.{ hd, d }, .mxfp8);
        try fx.putLin(pfx ++ "attn.wo_a", .{li}, &.{ og * ol, @divExact(H * hd, og) }, .fp8_exact);
        try fx.putLin(pfx ++ "attn.wo_b", .{li}, &.{ d, og * ol }, .mxfp8);
        try fx.putVec(pfx ++ "attn.attn_sink", .{li}, &.{H}, .float32, 0, 1);
        if (kv_src) {
            try fx.putLin(pfx ++ "attn.compressor.wkv", .{li}, &.{ hd, d }, .bf16);
            if (ratio > 1) try fx.putLin(pfx ++ "attn.compressor.wgate", .{li}, &.{ hd, d }, .bf16);
            try fx.putVec(pfx ++ "attn.compressor.norm.weight", .{li}, &.{hd}, .bfloat16, 1, 0.1);
        }
        if (ix_src) {
            try fx.putLin(pfx ++ "attn.indexer.wq_b", .{li}, &.{ ih * ihd, ql }, .mxfp8);
            try fx.putLin(pfx ++ "attn.indexer.weights_proj", .{li}, &.{ ih, d }, .bf16);
            if (kv_src) {
                try fx.putLin(pfx ++ "attn.indexer.wk", .{li}, &.{ ihd, hd }, .bf16);
                try fx.putVec(pfx ++ "attn.indexer.k_norm.weight", .{li}, &.{ihd}, .bfloat16, 1, 0.1);
            }
        }
        try fx.putLin(pfx ++ "ffn.gate", .{li}, &.{ e, d }, .bf16);
        try fx.putVec(pfx ++ "ffn.gate.bias", .{li}, &.{e}, .float32, 0, 0.05);
        try fx.putVec(pfx ++ "ffn.gate.bias_vl", .{li}, &.{e}, .float32, 0, 0.05);
        try fx.putLin(pfx ++ "ffn.shared_experts.w1", .{li}, &.{ inter, d }, .mxfp8);
        try fx.putLin(pfx ++ "ffn.shared_experts.w3", .{li}, &.{ inter, d }, .mxfp8);
        try fx.putLin(pfx ++ "ffn.shared_experts.w2", .{li}, &.{ d, inter }, .mxfp8);
        // One tensor an expert (the loader stacks them), so a full 384-expert bank never exists in bf16.
        for (0..E) |x| {
            try fx.putLin(pfx ++ "ffn.experts.{d}.w1", .{ li, x }, &.{ inter, d }, if (fp4) .mxfp4 else .{ .affine = gu_bits });
            try fx.putLin(pfx ++ "ffn.experts.{d}.w3", .{ li, x }, &.{ inter, d }, if (fp4) .mxfp4 else .{ .affine = gu_bits });
            try fx.putLin(pfx ++ "ffn.experts.{d}.w2", .{ li, x }, &.{ d, inter }, if (fp4) .mxfp4 else .{ .affine = affine orelse 3 });
        }
    }
};

const Fmt = union(enum) { bf16, fp8_exact, mxfp8, mxfp4, affine: u32 };

fn quantize(w: A, gs: c_int, bits: c_int, mode: [*:0]const u8) ![3]A {
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, w, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), mode, none, mlx.gpuStream()));
    var out: [3]A = .{ none, none, none };
    const n = mlx.mlx_vector_array_size(qv);
    for (out[0..n], 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, qv, i));
    }
    const vec = mlx.mlx_vector_array_new_data(&out, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    return out;
}

/// The release pack's config (`src/fixtures/model-configs`), parsed by the plugin.
pub fn releaseConfig() !config.Config {
    const json = @embedFile("../../../fixtures/model-configs/deepseek_v41-iq-mlx-3.0bpw.json");
    const mem: sdk_fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: @import("../../../abi/plugin.zig").Parsed = .{};
    const abi = @import("../../../abi/plugin.zig");
    if (!config.parse(&dir, abi.Str.of(json), .{}, null, false, &p).isOk()) return error.ConfigRefused;
    defer config.freeParsed(&p);
    return config.of(&p).*;
}

test "dsv41 dec fixture: layers 20..25 at the release's shapes build, prefill, and decode one token through today's path" {
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 300 });
    defer fx.deinit();
    try std.testing.expectEqual(@as(usize, 300), fx.state().n);
    const ly = fx.layer(20);
    try std.testing.expect(ly.q_kv != null and ly.wo_a.qp.mode == .mxfp8 and ly.experts.gu != null and ly.idx != null and ly.comp != null);
    try std.testing.expectEqual(@as(u32, 2), ly.experts.gu.?.qp.bits);
    const x = try fx.stream(1);
    defer _ = mlx.mlx_array_free(x);
    const pre = try fx.cpre(1);
    defer _ = mlx.mlx_array_free(pre);
    const out = try dsv41.workerStep(fx.m, gpa, x, pre, 300, false, false);
    defer for (out) |o| {
        _ = mlx.mlx_array_free(o);
    };
    const host = try dsv41.hostF32(out[0]);
    defer gpa.free(host);
    for (host) |v| try std.testing.expect(std.math.isFinite(v));
}

test "dsv41 tensor parallel: the two ranks' experts add up to the whole layer's MoE, decode and prompt rows" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const base: Opts = .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .act = .float32, .experts_fp4 = true };
    const whole = try Fixture.init(gpa, base);
    defer whole.deinit();
    var ranks: [2]*Fixture = undefined;
    for (&ranks, 0..) |*r, i| {
        var o = base;
        o.tp = .{ .rank = @intCast(i), .size = 2 };
        r.* = try Fixture.init(gpa, o);
    }
    defer for (ranks) |r| r.deinit();
    try std.testing.expectEqual(whole.m.cfg.moe_intermediate_size / 2, ranks[0].m.cfg.moe_intermediate_size);
    for ([_]usize{ 1, dsv41.region_rows + 9 }) |rows| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const x = try sc.keep(try whole.normal(&.{ @intCast(rows), @intCast(whole.m.cfg.hidden_size) }, .float32, 0, 1));
        const r = try dsv41.route(&sc, whole.m, whole.layer(20), x, none);
        const want = try dsv41.moeShare(whole.m, gpa, 20, x, r.idx, r.wts);
        defer _ = mlx.mlx_array_free(want);
        var got: [2]A = undefined;
        for (&got, ranks) |*g, rank| g.* = try dsv41.moeShare(rank.m, gpa, 20, x, r.idx, r.wts);
        defer for (got) |g| {
            _ = mlx.mlx_array_free(g);
        };
        const sum = try sc.add(got[0], got[1]);
        const a = try dsv41.hostF32(want);
        defer gpa.free(a);
        const b = try dsv41.hostF32(sum);
        defer gpa.free(b);
        var peak: f32 = 0;
        for (a) |v| peak = @max(peak, @abs(v));
        try std.testing.expect(peak > 0);
        for (a, b) |u, v| try std.testing.expect(@abs(u - v) <= 2e-4 * peak);
    }
}

test "dsv41 tensor parallel: two ranks on half the heads and experts, summing each attention and MoE, give the whole model's stream" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const base: Opts = .{ .lo = 20, .hi = 22, .experts = 8, .ctx = 0, .act = .float32, .experts_fp4 = true };
    const whole = try Fixture.init(gpa, base);
    defer whole.deinit();
    var half = base;
    half.tp = .{ .rank = 1, .size = 2 };
    const rank1 = try Fixture.init(gpa, half);
    defer rank1.deinit();
    try std.testing.expectEqual(whole.m.cfg.num_attention_heads / 2, rank1.m.cfg.num_attention_heads);
    // On a NAX GPU MLX runs these f32 GEMMs as TF32 (`MLX_ENABLE_TF32`, on by default): a 10-bit mantissa. A wrong
    // shard sum is off by the summand, far above either bound.
    const tol: f32 = if (device.naxAvailable()) 1e-2 else 1e-3;
    // Two spans and a tail: a wide chunk, whose spans each sum their attention.
    for ([_]usize{ 1, 6, dsv41.region_rows + 9, 2 * dsv41.SPAN + 8 }) |rows| {
        const in = try Input.of(whole, rows);
        defer in.deinit();
        const want = try streamOf(gpa, whole, in);
        defer gpa.free(want);
        const got = try tpStream(gpa, base, in);
        defer gpa.free(got);
        var peak: f32 = 0;
        for (want) |v| peak = @max(peak, @abs(v));
        try std.testing.expect(peak > 0);
        for (want, got) |u, v| try std.testing.expect(@abs(u - v) <= tol * peak);
    }
}

test "dsv41 tensor parallel: the ranks on the fused decode kernels decode no further from an f32 model than one model's chain" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    // A bf16 rounding can flip a near-tie route and move one seed's stream by percents, the chain's too: the bar is the
    // median over seeds.
    for ([_]usize{ 1, decode.max_rows }) |rows| {
        var tp: [5]f64 = undefined;
        var chain: [5]f64 = undefined;
        for (&tp, &chain, 1..) |*e_tp, *e_chain, seed| {
            const base: Opts = .{ .lo = 20, .hi = 22, .experts = 8, .ctx = 0, .experts_fp4 = true, .seed = seed };
            var o = base;
            o.act = .float32;
            const truth = try Fixture.init(gpa, o);
            defer truth.deinit();
            const plain = try Fixture.init(gpa, base);
            defer plain.deinit();
            const in = try Input.of(plain, rows);
            defer in.deinit();
            const t = try streamOf(gpa, truth, in);
            defer gpa.free(t);
            const c = try streamOf(gpa, plain, in);
            defer gpa.free(c);
            o = base;
            o.fused = true;
            const g = try tpStream(gpa, o, in);
            defer gpa.free(g);
            if (seed == 1) {
                // The kernels ran: the ranks on the chain alone land elsewhere.
                const g_chain = try tpStream(gpa, base, in);
                defer gpa.free(g_chain);
                try std.testing.expect(!std.mem.eql(f32, g, g_chain));
            }
            e_tp.* = decode.relRms(g, t);
            e_chain.* = decode.relRms(c, t);
        }
        std.mem.sort(f64, &tp, {}, std.sort.asc(f64));
        std.mem.sort(f64, &chain, {}, std.sort.asc(f64));
        std.debug.print("[dec] tensor parallel R={d}: median new-truth {e:.2}, chain-truth {e:.2} (rel rms)\n", .{ rows, tp[2], chain[2] });
        try std.testing.expect(tp[2] <= 1.25 * chain[2] + 1e-4);
    }
}

test "dsv41 tensor parallel: rank 0 holds a DSpark stage at the whole model's widths, and its MoE sums nothing" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const base: Opts = .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .act = .float32, .experts_fp4 = true };
    var tp = base;
    tp.tp = .{ .rank = 0, .size = 2 };
    var fxs: [2]*Fixture = undefined;
    var made: usize = 0;
    defer for (fxs[0..made]) |f| f.deinit();
    for ([_]Opts{ base, tp }, &fxs) |o, *f| {
        f.* = try Fixture.init(gpa, o);
        made += 1;
        // The same draws in both: the stage's tensors as the pack ships them, unsliced.
        var whole = try releaseConfig();
        whole.dsv41_dspark_experts = 8;
        f.*.m.cfg.dsv41_dspark_experts = 8;
        try f.*.putLayer(&whole, "mtp.{d}.", 0, 8, true, null);
        const m = f.*.m;
        const ly = try dsv41.loadLayer(m, &f.*.w, "mtp", 0, null);
        m.layers = try gpa.realloc(m.layers, m.n_layers + 1);
        m.layers[m.n_layers] = ly;
    }
    const whole_m = fxs[0].m;
    const rank0 = fxs[1].m;
    try std.testing.expectEqual(whole_m.cfg.num_attention_heads / 2, rank0.cfg.num_attention_heads);
    const st = &rank0.layers[rank0.n_layers];
    try std.testing.expectEqual(@as(usize, whole_m.cfg.num_attention_heads), @as(usize, @intCast(mlx.getShape(st.sink)[1])));
    try std.testing.expectEqual(@as(usize, whole_m.cfg.dsv4_o_groups), @as(usize, @intCast(mlx.getShape(st.wo_a.w)[0])));

    const Count = struct {
        var n: usize = 0;
        fn sum(x: A) anyerror!A {
            n += 1;
            var out = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out, x);
            return out;
        }
    };
    dsv41.clusterHooks().tp_sum = &Count.sum;
    defer dsv41.clusterHooks().tp_sum = null;
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const d: c_int = @intCast(whole_m.cfg.hidden_size);
    const x = try sc.keep(try fxs[0].normal(&.{ 3, d }, .float32, 0, 1));
    var got: [2][]f32 = undefined;
    var heads: [2][]f32 = undefined;
    for ([_]*dsv41.Dsv41Model{ whole_m, rank0 }, &got, &heads) |m, *g, *h| {
        const ly = &m.layers[m.n_layers];
        g.* = try dsv41.hostF32(try dsv41.moe(&sc, m, m.n_layers, x, none));
        // The stage's heads through its own wo_a groups and wo_b.
        const q = (try dsv41.qkvRows(&sc, m, ly, x)).q;
        h.* = try dsv41.hostF32(try dsv41.attnOutRoped(&sc, m, ly, q));
    }
    defer for (got, heads) |g, h| {
        gpa.free(g);
        gpa.free(h);
    };
    try std.testing.expectEqual(@as(usize, 0), Count.n);
    try std.testing.expectEqualSlices(f32, got[0], got[1]);
    try std.testing.expectEqualSlices(f32, heads[0], heads[1]);
}

test "dsv41 tensor parallel and pair workers: a verify chunk rolled back to where the next step starts leaves the state its kept rows build" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const o: Opts = .{ .lo = 20, .hi = 22, .experts = 8, .ctx = 66, .act = .float32, .experts_fp4 = true };
    const a = try Fixture.init(gpa, o);
    defer a.deinit();
    const b = try Fixture.init(gpa, o);
    defer b.deinit();
    const chunk = try Input.of(a, 5);
    defer chunk.deinit();
    const next = try Input.of(a, 1);
    defer next.deinit();
    const p0 = o.ctx;
    const keep = 3; // past a ratio-4 group's start: the open group is re-cut from the chunk's rows
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    for (try dsv41.workerStep(a.m, gpa, chunk.x, chunk.pre, p0, false, true)) |r| _ = mlx.mlx_array_free(r);
    for (try dsv41.workerStep(b.m, gpa, try sc.slice(chunk.x, 0, 0, keep), try sc.slice(chunk.pre, 0, 0, keep), p0, false, false)) |r| _ = mlx.mlx_array_free(r);
    var got: [2][]f32 = undefined;
    for ([_]*Fixture{ a, b }, &got) |f, *g| {
        const out = try dsv41.workerStep(f.m, gpa, next.x, next.pre, p0 + keep, false, false);
        defer for (out) |r| {
            _ = mlx.mlx_array_free(r);
        };
        g.* = try dsv41.hostF32(out[0]);
    }
    defer for (got) |g| gpa.free(g);
    try std.testing.expectEqual(a.state().n, b.state().n);
    var peak: f32 = 0;
    for (got[1]) |v| peak = @max(peak, @abs(v));
    for (got[0], got[1]) |u, v| try std.testing.expect(@abs(u - v) <= 1e-4 * peak);
}

test "dsv41 tensor parallel: a worker keeps rank 0's states by gen, the one it steps the module's, a free drops either" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .tp = .{ .rank = 1, .size = 2 } });
    defer fx.deinit();
    const m = fx.m;
    var parked: dsv41.Parked = .init(gpa);
    defer dsv41.dropParked(&parked);
    if (m.dec_state) |*st| st.deinit();
    m.dec_state = null;
    // Gen 5 new here: the module holds none until its step at 0 makes one.
    try dsv41.tpActivate(m, &parked, 5);
    try std.testing.expect(m.dec_state == null);
    m.dec_state = try dsv41.initState(m, gpa);
    m.dec_state.?.n = 50;
    try dsv41.tpActivate(m, &parked, 6);
    try std.testing.expect(m.dec_state == null and parked.get(5).?.n == 50);
    m.dec_state = try dsv41.initState(m, gpa);
    m.dec_state.?.n = 60;
    try dsv41.tpActivate(m, &parked, 5);
    try std.testing.expectEqual(@as(usize, 50), m.dec_state.?.n);
    try std.testing.expectEqual(@as(u64, 5), m.state_gen);
    try std.testing.expectEqual(@as(usize, 60), parked.get(6).?.n);
    dsv41.tpWorkerFree(m, &parked, 6);
    try std.testing.expectEqual(@as(usize, 0), parked.count());
    dsv41.tpWorkerFree(m, &parked, 5);
    try std.testing.expect(m.dec_state == null);
}

test "dsv41 tensor parallel: a worker keeps its copy of rank 0's snapshots by id: take, clone, restore, capture at the boundary and free" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .tp = .{ .rank = 1, .size = 2 } });
    defer fx.deinit();
    const m = fx.m;
    var parked: dsv41.Parked = .init(gpa);
    defer dsv41.dropParked(&parked);
    var snaps: dsv41.Snaps = .init(gpa);
    defer dsv41.dropSnaps(&snaps, gpa);
    if (m.dec_state) |*st| st.deinit();
    m.dec_state = null;
    try dsv41.tpActivate(m, &parked, 5);
    m.dec_state = try dsv41.initState(m, gpa);
    const st = &m.dec_state.?;
    st.n = 50;
    st.snap_at = 50;
    try dsv41.snapshotStateIfDue(st, 5);
    try dsv41.tpWorkerSnap(m, gpa, &parked, &snaps, .take, 5, 1);
    try std.testing.expect(st.snap == null and snaps.get(1).?.n == 50);
    try dsv41.tpWorkerSnap(m, gpa, &parked, &snaps, .clone, 1, 2);
    try dsv41.tpWorkerSnap(m, gpa, &parked, &snaps, .restore, 2, 7);
    try std.testing.expectEqual(@as(u64, 7), m.state_gen);
    try std.testing.expectEqual(@as(usize, 50), m.dec_state.?.n);
    try std.testing.expect(parked.contains(5));
    try dsv41.tpWorkerSnap(m, gpa, &parked, &snaps, .capture, 7, 0);
    try dsv41.tpWorkerSnap(m, gpa, &parked, &snaps, .take, 7, 3);
    try std.testing.expectEqual(@as(usize, 50), snaps.get(3).?.n);
    try dsv41.tpWorkerSnap(m, gpa, &parked, &snaps, .free, 3, 0);
    try dsv41.tpWorkerSnap(m, gpa, &parked, &snaps, .free, 1, 0);
    try dsv41.tpWorkerSnap(m, gpa, &parked, &snaps, .free, 2, 0);
    try std.testing.expectEqual(@as(usize, 0), snaps.count());
    try std.testing.expectError(error.DistPeerDiverged, dsv41.tpWorkerSnap(m, gpa, &parked, &snaps, .restore, 1, 8));
}

test "dsv41 tensor parallel: a chunk's image rows, sent alone, splice as the request's rows at their offset" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer fx.deinit();
    const m = fx.m;
    m.cfg.image_token_id = 7;
    const d = m.cfg.hidden_size;
    var sc = dsv41.Scope.init(gpa, m.s);
    defer sc.deinit();
    const vals = try gpa.alloc(f32, 10 * d);
    defer gpa.free(vals);
    for (vals, 0..) |*v, k| v.* = @floatFromInt(k / d + 1);
    const all = try sc.keep(mlx.mlx_array_new_data(vals.ptr, &[_]c_int{ 10, @intCast(d) }, 2, .float32));
    const ids = [_]u32{ 1, 7, 7, 2, 7, 3 };
    const vr: dsv41.VisionRows = .{ .rows = all, .offset = 4 };
    const sent = (try dsv41.chunkImageRows(&sc, m, &ids, vr)).?;
    const got = (try dsv41.chunkImageRows(&sc, m, &ids, .{ .rows = sent, .offset = 0 })).?;
    const id_arr = try sc.ints(@ptrCast(&ids), &.{ids.len});
    const emb = try sc.zeros(&.{ ids.len, @intCast(d) }, m.act);
    var masks: [2]A = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (masks) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const a = try dsv41.hostF32(try dsv41.spliceImages(&sc, m, emb, id_arr, &ids, sent, &masks[0]));
    defer gpa.free(a);
    const b = try dsv41.hostF32(try dsv41.spliceImages(&sc, m, emb, id_arr, &ids, got, &masks[1]));
    defer gpa.free(b);
    try std.testing.expectEqualSlices(f32, a, b);
    for ([_]f32{ 0, 5, 6, 0, 7, 0 }, 0..) |want, r| try std.testing.expectEqual(want, a[r * d + d / 2]);
    try std.testing.expectError(error.Dsv41VisionRowsShort, dsv41.chunkImageRows(&sc, m, &ids, .{ .rows = all, .offset = 8 }));
    try std.testing.expect(try dsv41.chunkImageRows(&sc, m, &[_]u32{ 1, 2 }, vr) == null);
}

test "dsv41 tensor parallel: a snapshot read from disk reaches the worker as its cache tensors, which rebuild it bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer fx.deinit();
    const m = fx.m;
    const n = 301;
    const p = try dsv41.tpLoadPlan(m, n);
    var sc = dsv41.Scope.init(gpa, m.s);
    defer sc.deinit();
    const disk = try gpa.alloc(A, p.len);
    defer gpa.free(disk);
    for (p.specs(), disk, 0..) |spec, *x, k| {
        const shape = [_]c_int{ @intCast(spec.rows), @intCast(spec.width) };
        const vals = try sc.astype(try sc.arange(k, k + spec.rows * spec.width), spec.dtype);
        x.* = try sc.reshape(vals, shape[0..spec.rank()]);
    }
    const ids = try gpa.alloc(u32, n);
    defer gpa.free(ids);
    @memset(ids, 3);
    var snaps: dsv41.Snaps = .init(gpa);
    defer dsv41.dropSnaps(&snaps, gpa);
    // Rank 0's import from disk, then what it sends the worker and what the worker rebuilds from that.
    try dsv41.tpWorkerLoad(m, gpa, &snaps, 1, ids, disk);
    const sent = try dsv41.tpLoadTensors(m, gpa, snaps.getPtr(1).?);
    defer dsv41.freeArrays(gpa, sent);
    try dsv41.tpWorkerLoad(m, gpa, &snaps, 2, ids, sent);
    const back = try dsv41.tpLoadTensors(m, gpa, snaps.getPtr(2).?);
    defer dsv41.freeArrays(gpa, back);
    for (disk, sent, back) |a, b, c| {
        try expectSameArray(gpa, a, b);
        try expectSameArray(gpa, a, c);
    }
    try std.testing.expectEqual(@as(usize, n), snaps.get(2).?.n);
    try std.testing.expectError(error.DistPeerDiverged, dsv41.tpWorkerLoad(m, gpa, &snaps, 3, ids, sent[1..]));
}

/// One step's input rows: a stream and its mix.
const Input = struct {
    x: A,
    pre: A,

    fn of(fx: *Fixture, rows: usize) !Input {
        const x = try fx.stream(rows);
        errdefer _ = mlx.mlx_array_free(x);
        return .{ .x = x, .pre = try fx.cpre(rows) };
    }

    fn deinit(in: Input) void {
        _ = mlx.mlx_array_free(in.x);
        _ = mlx.mlx_array_free(in.pre);
    }

    /// One step of `m` over these rows, the stream in `m`'s activation dtype (caller frees both).
    fn step(in: Input, gpa: std.mem.Allocator, m: *dsv41.Dsv41Model) ![2]A {
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, in.x, m.act, mlx.gpuStream()));
        return dsv41.workerStep(m, gpa, x, in.pre, 0, false, false);
    }
};

/// `fx`'s stream after its layers over `in`, from position 0 (caller frees).
fn streamOf(gpa: std.mem.Allocator, fx: *Fixture, in: Input) ![]f32 {
    const out = try in.step(gpa, fx.m);
    defer for (out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    return dsv41.hostF32(out[0]);
}

/// The stream two tensor ranks built from `base` reach over `in` (caller frees). One process cannot run
/// the ranks in lockstep: each pass runs one rank against the other's partials from its latest pass, and a sum depends
/// only on the sums before it, so passes fix them one by one; the last pass changes none.
fn tpStream(gpa: std.mem.Allocator, base: Opts, in: Input) ![]f32 {
    var ranks: [2]*Fixture = undefined;
    var made: usize = 0;
    defer for (ranks[0..made]) |r| r.deinit();
    for (&ranks, 0..) |*r, i| {
        var o = base;
        o.tp = .{ .rank = @intCast(i), .size = 2 };
        r.* = try Fixture.init(gpa, o);
        made += 1;
    }
    const Sums = struct {
        var rank: usize = 0;
        var parts: [2]std.ArrayList(A) = .{ .empty, .empty };
        var prev: [2]std.ArrayList(A) = .{ .empty, .empty };
        fn sum(x: A) anyerror!A {
            try mlx.check(mlx.mlx_array_eval(x));
            var kept = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&kept, x);
            const k = parts[rank].items.len;
            try parts[rank].append(std.testing.allocator, kept);
            const other = &parts[1 - rank];
            var out = mlx.mlx_array_new();
            if (k < other.items.len) try mlx.check(mlx.mlx_add(&out, x, other.items[k], mlx.gpuStream())) else _ = mlx.mlx_array_set(&out, x);
            return out;
        }
        fn clear(l: *std.ArrayList(A)) void {
            for (l.items) |a| _ = mlx.mlx_array_free(a);
            l.clearRetainingCapacity();
        }
    };
    dsv41.clusterHooks().tp_sum = &Sums.sum;
    defer {
        dsv41.clusterHooks().tp_sum = null;
        for (&Sums.parts, &Sums.prev) |*a, *b| {
            Sums.clear(a);
            Sums.clear(b);
            a.deinit(gpa);
            b.deinit(gpa);
            a.* = .empty;
            b.* = .empty;
        }
    }
    var got: [2][]f32 = .{ &.{}, &.{} };
    defer for (got) |g| gpa.free(g);
    // A pass fixes at least one more sum; the last pass changes none. A wide chunk sums each span's attention.
    const spans = (@as(usize, @intCast(mlx.getShape(in.x)[0])) + dsv41.SPAN - 1) / dsv41.SPAN;
    const sums = (spans + 1) * (base.hi - base.lo);
    for (0..sums + 2) |_| for (0..2) |r| {
        Sums.clear(&Sums.prev[r]);
        std.mem.swap(std.ArrayList(A), &Sums.prev[r], &Sums.parts[r]);
        Sums.rank = r;
        const out = try in.step(gpa, ranks[r].m);
        defer for (out) |a| {
            _ = mlx.mlx_array_free(a);
        };
        gpa.free(got[r]);
        got[r] = &.{};
        got[r] = try dsv41.hostF32(out[0]);
    };
    for (Sums.parts, Sums.prev) |now, before| {
        try std.testing.expectEqual(sums, now.items.len);
        for (now.items, before.items) |a, b| {
            const u = try dsv41.hostF32(a);
            defer gpa.free(u);
            const v = try dsv41.hostF32(b);
            defer gpa.free(v);
            try std.testing.expectEqualSlices(f32, u, v);
        }
    }
    try std.testing.expectEqualSlices(f32, got[0], got[1]);
    // The ranks' caches come off the same summed stream through whole KV weights: rank 0's snapshot is the worker's.
    for (ranks[0].m.dec_state.?.layers, ranks[1].m.dec_state.?.layers) |a, b| {
        try expectSameArray(gpa, a.win.buf, b.win.buf);
        if (a.comp) |r| try expectSameArray(gpa, r.buf, b.comp.?.buf);
        if (a.idx_k) |r| try expectSameArray(gpa, r.buf, b.idx_k.?.buf);
        if (a.pend_kv.ctx != null) try expectSameArray(gpa, a.pend_kv, b.pend_kv);
    }
    const out = got[0];
    got[0] = &.{};
    return out;
}

fn expectSameArray(gpa: std.mem.Allocator, a: A, b: A) !void {
    try std.testing.expectEqual(a.ctx == null, b.ctx == null);
    if (a.ctx == null) return;
    const u = try dsv41.hostF32(a);
    defer gpa.free(u);
    const v = try dsv41.hostF32(b);
    defer gpa.free(v);
    try std.testing.expectEqualSlices(f32, u, v);
}
