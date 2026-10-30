//! A DeepSeek-V4.1 model at the release's shapes over random weights in the served formats, built through the plugin's
//! own `init` as a pair worker holding layers [lo, hi) (no embed, head or Engram), its state prefilled with `ctx`
//! random stream rows through those layers: what the decode kernels' tests and ubenches run against.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const resident = @import("../../../sdk/weights/resident.zig");
const sdk_fixture = @import("../../../sdk/testing/fixture.zig");
const config = @import("../config.zig");
const dsv41 = @import("../deepseek_v41.zig");

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
};

pub const Fixture = struct {
    gpa: std.mem.Allocator,
    m: *dsv41.Dsv41Model,
    w: resident.Weights,
    seed: u64,
    draws: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, opts: Opts) !*Fixture {
        const fx = try gpa.create(Fixture);
        errdefer gpa.destroy(fx);
        var cfg = try releaseConfig();
        cfg.num_experts = opts.experts;
        fx.* = .{ .gpa = gpa, .m = undefined, .w = resident.Weights.init(gpa), .seed = opts.seed };
        errdefer fx.w.deinit();
        for (opts.lo..opts.hi) |li| try fx.putLayer(&cfg, li, opts.experts);
        fx.m = try dsv41.init(gpa, &cfg, &fx.w, mlx.gpuStream(), .{ .pair = .{ .lo = opts.lo, .hi = opts.hi, .worker = true }, .act = opts.act });
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
            .mxfp8, .affine => {
                defer _ = mlx.mlx_array_free(w);
                const q = if (fmt == .mxfp8) try quantize(w, 32, 8, "mxfp8") else try quantize(w, 64, @intCast(fmt.affine), "affine");
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
    fn putLayer(fx: *Fixture, c: *const config.Config, li: usize, E: u32) !void {
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
        const gu_bits: u32 = if ((li >= 1 and li <= 9) or li == 14) 3 else 2;
        const pfx = "layers.{d}.";
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
            try fx.putLin(pfx ++ "ffn.experts.{d}.w1", .{ li, x }, &.{ inter, d }, .{ .affine = gu_bits });
            try fx.putLin(pfx ++ "ffn.experts.{d}.w3", .{ li, x }, &.{ inter, d }, .{ .affine = gu_bits });
            try fx.putLin(pfx ++ "ffn.experts.{d}.w2", .{ li, x }, &.{ d, inter }, .{ .affine = 3 });
        }
    }
};

const Fmt = union(enum) { bf16, fp8_exact, mxfp8, affine: u32 };

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
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
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
