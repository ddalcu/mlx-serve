//! `MLX_SERVE_DSV41_WOA_PLAIN`: the decode attention's splits merged once (`attn_merge`: the sink, the inverse rope, f32
//! heads), then wo_a as a plain grouped mxfp8 GEMV over those heads (`dec_wo_a_plain.metal`): no per-tile merge, no
//! device atomics, no last-arriver sum. One launch more than `wo_a`'s fused merge, every tile's merge reads gone.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");

const A = mlx.mlx_array;
const Scope = dsv41.Scope;

pub const SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_wo_a_plain.metal");
pub const HEADER = decode.FP8_READ_HEADER;
const PLANT_HEADER = HEADER ++ "#define DSV41_WOA_PLAIN_PLANT 1\n";
pub const INS = [_][*:0]const u8{ "x", "w", "scales" };
pub const OUTS = [_][*:0]const u8{"out"};

const HD = 512;
const HPG = 8;

pub const Kernels = struct {
    merge: mlx.mlx_fast_metal_kernel,
    gemv: mlx.mlx_fast_metal_kernel,

    pub fn deinit(k: Kernels) void {
        _ = mlx.mlx_fast_metal_kernel_free(k.merge);
        _ = mlx.mlx_fast_metal_kernel_free(k.gemv);
    }
};

pub fn build() !Kernels {
    const km = try dsv41.buildKernel("dsv41_attn_merge", &dsv41.ATTN_MERGE_INS, &.{"out"}, dsv41.ATTN_MERGE_SOURCE, dsv41.UNROPE_HEADER);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(km);
    return .{ .merge = km, .gemv = try dsv41.buildKernel("dsv41_dec_wo_a_plain", &INS, &OUTS, SOURCE, HEADER) };
}

pub fn buildPlant() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_dec_wo_a_plain_plant", &INS, &OUTS, SOURCE, PLANT_HEADER);
}

/// `MLX_SERVE_DSV41_WOA_PLAIN` (read each call): =1 the decode attention's output takes `apply`.
pub fn on() bool {
    const v = env.get(.MLX_SERVE_DSV41_WOA_PLAIN) orelse return false;
    return v[0] == '1';
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// TR outputs a threadgroup, NB weight rows a lane has in flight.
pub const Plan = struct { tr: c_int = 64, nb: c_int = 8 };

/// The rows the plain GEMV serves: every decode row (the pair ubench: faster than wo_a's fused merge at 1..6).
pub const max_rows = decode.max_rows;

/// The pair ubench's fastest plan at every row count: 64 outputs a threadgroup, 2 weight rows in flight a lane (more
/// rows in flight cost registers past what they hide).
pub fn planFor(rows: usize) Plan {
    _ = rows;
    return .{ .tr = 64, .nb = 2 };
}

/// `oa` [R, groups * o_lora] in the model's activation dtype from the attention's split partials, for `rows` up to
/// `max_rows`: the splits merged once with the layer's sink and inverse rope `inv`, then the plain GEMV over wo_a's
/// mxfp8 weights. Null (wo_a's fused merge runs) where wo_a is not the release's 8-bit mxfp8 of group 32 or the rows
/// are past `max_rows`.
pub fn apply(sc: *Scope, k: Kernels, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, inv: A, meta: A, pacc: A, pm: A, pl: A) !?A {
    const wo = &ly.wo_a;
    const rows: usize = @intCast(mlx.getShape(pacc)[0]);
    if (!mlx.streamIsGpu(sc.s) or rows > max_rows or wo.ext != null or wo.s.ctx == null or wo.qp.mode != .mxfp8 or wo.qp.bits != 8 or wo.qp.group_size != 32) return null;
    if (mlx.mlx_array_size(inv) != m.cfg.dsv4_rope_head_dim / 2 or mlx.mlx_array_dtype(ly.sink) != .float32) return null;
    const x = try merge(sc, k.merge, m.cfg.dsv4_rope_head_dim, pacc, pm, pl, ly.sink, meta, inv);
    return gemv(sc, k.gemv, m.act, x, wo.w, wo.s, planFor(rows));
}

/// The heads `[R, H * 512]` f32 from the split partials (`pacc` [R, H, S, 512], `pm`, `pl` [R, H, S]) and the sink
/// [H]: `attn_merge`'s math, inverse-roped over the last `rd` channels at positions `meta[0]..`.
pub fn merge(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, rd: usize, pacc: A, pm: A, pl: A, sink: A, meta: A, periods: A) !A {
    const s = mlx.getShape(pacc);
    const R = s[0];
    const H = s[1];
    const mc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mc, &[_]c_int{ R, H * HD }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mc, HD, H, R));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mc, HD, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "D", HD));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "S", s[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mc, "OT", .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "RD", ci(rd)));
    const mv = mlx.mlx_vector_array_new_data(&[_]A{ pacc, pm, pl, sink, meta, periods }, 6);
    defer _ = mlx.mlx_vector_array_free(mv);
    var mo = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(mo);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&mo, kern, mv, mc, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, mo, 0), &o);
}

// ── the kernel contract: the grouped mxfp8 GEMV over merged heads in float64 ──

const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const contract = @import("../contract.zig");
const wo_a = @import("wo_a.zig");

const Gemv = struct {
    pub const Params = struct { groups: usize = 2, tr: usize = 64, nb: usize = 2 };

    pub fn geometry(p: Params, s: kit.Shape) ?kit.metal.Geometry {
        if (s.k != HPG * HD or s.rows < 1 or s.rows > decode.max_rows or s.n % p.tr != 0 or p.tr % p.nb != 0) return null;
        var g = kit.metal.Geometry{ .grid = .{ ci(s.n / p.tr * HPG * 32), ci(p.groups), 1 }, .tg = .{ HPG * 32, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(p.groups * s.n) }, .bfloat16);
        inline for (.{ .{ "R", s.rows }, .{ "H", p.groups * HPG }, .{ "D", HD }, .{ "HPG", HPG }, .{ "OL", s.n }, .{ "OW", p.groups * s.n }, .{ "TR", p.tr }, .{ "NB", p.nb } }) |t| g.int(t[0], t[1]);
        g.dtype("OT", .bfloat16);
        g.int("FOLD", 0);
        return g;
    }

    pub fn count(p: Params, s: kit.Shape) usize {
        return s.rows * p.groups * s.n;
    }

    pub fn fill(k: anytype, f: kit.Fill, rng: std.Random) !void {
        const R = k.shape.rows;
        const N = k.shape.n;
        const G = k.p.groups;
        const K = HPG * HD;
        const x = try k.add(.float32, &.{ R, G * K });
        const w = try k.add(.uint32, &.{ G, N, K / 4 });
        const scales = try k.add(.uint8, &.{ G, N, K / 32 });
        switch (f) {
            .random, .nonfinite => {
                kit.uniform(rng, x, -2, 2);
                for (w) |*word| {
                    var bytes: [4]u8 = undefined;
                    for (&bytes) |*b| {
                        b.* = rng.int(u8);
                        while (b.* & 0x7f == 0x7f) b.* = rng.int(u8);
                    }
                    word.* = @floatFromInt(std.mem.readInt(u32, &bytes, .little));
                }
                for (scales) |*v| v.* = @floatFromInt(118 + rng.uintLessThan(u8, 11));
                if (f == .nonfinite) kit.plantRows(x, R, G * K);
            },
            .stress => {
                // x the mirrored integers of a group's heads, the weights e4m3 integers at scale 1.
                const xs = try k.scratch().alloc(f64, K);
                defer k.scratch().free(xs);
                const wi = try k.scratch().alloc(f64, G * N * K);
                defer k.scratch().free(wi);
                try contract.stressDot(rng, xs, wi, G * N, K, 64, 15, k.scratch());
                for (0..R) |m| for (0..G) |g| @memcpy(x[(m * G + g) * K ..][0..K], xs);
                for (w, 0..) |*word, i| {
                    var bytes: [4]u8 = undefined;
                    for (&bytes, 0..) |*b, j| b.* = wo_a.e4m3Of(@intFromFloat(wi[i * 4 + j]));
                    word.* = @floatFromInt(std.mem.readInt(u32, &bytes, .little));
                }
                @memset(scales, 127);
            },
        }
    }

    /// float64: each output's dot of its row's dequantized weights with its group's heads; `mag` the terms' magnitude.
    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: kit.Sum) !void {
        const R = k.shape.rows;
        const N = k.shape.n;
        const G = k.p.groups;
        const K = HPG * HD;
        const a = k.scratch();
        const wrow = try a.alloc(f64, K);
        defer a.free(wrow);
        const t = try a.alloc(f64, K);
        defer a.free(t);
        for (0..G) |g| for (0..N) |n| {
            const row = g * N + n;
            for (k.in(1)[row * K / 4 ..][0 .. K / 4], 0..) |word, i| {
                var bytes: [4]u8 = undefined;
                std.mem.writeInt(u32, &bytes, @intFromFloat(word), .little);
                for (bytes, 0..) |b, j| {
                    const idx = i * 4 + j;
                    const e: i32 = @intFromFloat(k.in(2)[row * K / 32 + idx / 32]);
                    wrow[idx] = wo_a.e4m3(b) * std.math.ldexp(@as(f64, 1), e - 127);
                }
            }
            for (0..R) |m| {
                var mg: f64 = 0;
                for (t, wrow, k.in(0)[(m * G + g) * K ..][0..K]) |*term, wv, xv| {
                    term.* = xv * wv;
                    mg += @abs(term.*);
                }
                out[(m * G + g) * N + n] = sum.of(t);
                mag[(m * G + g) * N + n] = mg;
            }
        };
    }
};

fn depthGemv(s: kit.Shape) usize {
    return s.k + 16;
}

const gemv_decl: kit.metal.Launch = .{
    .name = "dsv41_dec_wo_a_plain",
    .inputs = &INS,
    .outputs = &OUTS,
    .source = SOURCE,
    .header = HEADER,
    .tg_bytes = 4 * HPG * decode.max_rows * 64,
    .m1 = 256,
    .m2 = 256,
};

/// A group's 256 of 1024 outputs and two groups (the float64 reference runs for seconds at the release's 8 x 1024); the
/// reduction (a group's 8 heads of 512) is the release's.
const served = [_]kit.Shape{
    .{ .rows = 1, .n = 256, .k = HPG * HD },
    .{ .rows = 2, .n = 256, .k = HPG * HD },
    .{ .rows = 3, .n = 256, .k = HPG * HD },
    .{ .rows = 5, .n = 256, .k = HPG * HD },
    .{ .rows = decode.max_rows, .n = 256, .k = HPG * HD },
};

pub fn register(set: *kc.Set) !void {
    try kit.add(set, Gemv, &gemv_decl, .{}, .{
        .name = "dsv41 dec wo_a plain (grouped mxfp8 GEMV over merged heads)",
        .accum = .f32,
        .depth = depthGemv,
        .ops = 4,
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = 64, .step = 64, .min = 64 }, .{} },
        .served = &served,
    });
}

/// `oa` [R, groups * o_lora] in `odt` from the merged heads `x` [R, H * 512] f32 and wo_a's mxfp8 words
/// [groups, o_lora, HPG * 512 / 4] and scales [groups, o_lora, HPG * 512 / 32]. Null where the shapes are not the release's.
pub fn gemv(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, odt: mlx.mlx_dtype, x: A, w: A, scales: A, p: Plan) !?A {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len != 2 or ws.len != 3 or mlx.mlx_array_dtype(x) != .float32 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(scales) != .uint8) return null;
    const R = xs[0];
    const G = ws[0];
    const OL = ws[1];
    if (R < 1 or R > decode.max_rows or ws[2] * 4 != HPG * HD or xs[1] != G * HPG * HD or @mod(OL, p.tr) != 0 or @mod(p.tr, p.nb) != 0) return null;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ R, G * OL }, 2, odt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(OL, p.tr) * HPG * 32, G, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, HPG * 32, 1, 1));
    inline for (.{ .{ "R", R }, .{ "H", G * HPG }, .{ "D", HD }, .{ "HPG", HPG }, .{ "OL", OL }, .{ "OW", G * OL }, .{ "TR", p.tr }, .{ "NB", p.nb } }) |t|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OT", odt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "FOLD", decode.fp8Fold()));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, w, scales }, 3);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, c, sc.s));
    var o = mlx.mlx_array_new();
    return try sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

// ── tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;
const device = @import("../../../mlx/device.zig");

test "dsv41 dec wo_a plain: the registered kernel passes the contract's gates 1-4" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var set = kc.Set.init(testing.allocator);
    defer set.deinit();
    try register(&set);
    var t: kit.gates.Tally = .{};
    try kit.gates.runAll(testing.allocator, set.kernels.items, 0xc0de, &t);
    try testing.expectEqual(@as(usize, 0), t.problems());
}

/// The release's wo_a (8 groups of 1024 outputs over 8 heads of 512) on random mxfp8 words with no NaN code, the
/// inverse rope's periods, and random split partials of 64 heads.
const Inputs = struct {
    w: A,
    s: A,
    periods: A,
    sink: A,
    pacc: A,
    pm: A,
    pl: A,
    meta: A,

    fn init(sc: *Scope, R: usize, S: usize, seed: u64) !Inputs {
        const gpa = testing.allocator;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const G = 8;
        const OL = 1024;
        const KW = HPG * HD;
        const words = try gpa.alloc(u32, G * OL * KW / 4);
        defer gpa.free(words);
        for (words) |*w| {
            var b: [4]u8 = undefined;
            for (&b) |*x| {
                x.* = rnd.int(u8);
                if (x.* & 0x7f == 0x7f) x.* -= 1;
            }
            w.* = std.mem.readInt(u32, &b, .little);
        }
        const sc8 = try gpa.alloc(u8, G * OL * KW / 32);
        defer gpa.free(sc8);
        for (sc8) |*e| e.* = rnd.intRangeAtMost(u8, 118, 124);
        var per: [32]f32 = undefined;
        for (&per, 0..) |*v, i| v.* = std.math.pow(f32, 10000, @as(f32, @floatFromInt(2 * i)) / 64.0);
        const H = G * HPG;
        const pacc = try gpa.alloc(f32, R * H * S * HD);
        defer gpa.free(pacc);
        const pm = try gpa.alloc(f32, R * H * S);
        defer gpa.free(pm);
        const pl = try gpa.alloc(f32, R * H * S);
        defer gpa.free(pl);
        var sink: [64]f32 = undefined;
        for (&sink) |*v| v.* = rnd.float(f32) * 6 - 3;
        for (pm, pl, 0..) |*m, *l, j| {
            const empty = j % 13 == 5;
            m.* = if (empty) -std.math.inf(f32) else rnd.float(f32) * 6 - 2;
            l.* = if (empty) 0 else 1 + rnd.float(f32) * 19;
            for (pacc[j * HD ..][0..HD]) |*v| v.* = if (empty) 0 else (rnd.float(f32) * 2 - 1) * 2 * l.*;
        }
        const ci_ = struct {
            fn f(v: usize) c_int {
                return @intCast(v);
            }
        }.f;
        return .{
            .w = try sc.keep(mlx.mlx_array_new_data(words.ptr, &[_]c_int{ G, OL, KW / 4 }, 3, .uint32)),
            .s = try sc.keep(mlx.mlx_array_new_data(sc8.ptr, &[_]c_int{ G, OL, KW / 32 }, 3, .uint8)),
            .periods = try sc.keep(mlx.mlx_array_new_data(&per, &[_]c_int{32}, 1, .float32)),
            .sink = try sc.keep(mlx.mlx_array_new_data(&sink, &[_]c_int{H}, 1, .float32)),
            .pacc = try sc.keep(mlx.mlx_array_new_data(pacc.ptr, &[_]c_int{ ci_(R), H, ci_(S), HD }, 4, .float32)),
            .pm = try sc.keep(mlx.mlx_array_new_data(pm.ptr, &[_]c_int{ ci_(R), H, ci_(S) }, 3, .float32)),
            .pl = try sc.keep(mlx.mlx_array_new_data(pl.ptr, &[_]c_int{ ci_(R), H, ci_(S) }, 3, .float32)),
            .meta = try sc.ints(&[_]i32{ 30001, @intCast(R), 512, 128, 144, 0 }, &.{6}),
        };
    }
};

test "dsv41 dec wo_a plain: the merged heads through the plain GEMV equal wo_a over the splits, R = 1..6, every plan; a dropped head fails" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const k = try build();
    defer k.deinit();
    const plant = try buildPlant();
    defer _ = mlx.mlx_fast_metal_kernel_free(plant);
    const kw = try wo_a.build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kw);
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const counter = try sc.keep(wo_a.newCounter());
    for (1..decode.max_rows + 1) |R| for ([_]usize{ 8, 2 }) |S| {
        const in = try Inputs.init(&sc, R, S, 77 + R * 10 + S);
        const want_a = (try wo_a.launch(&sc, kw, 64, .bfloat16, .{ in.pacc, in.pm, in.pl, in.sink, in.meta, in.periods, in.w, in.s }, counter, null)) orelse return error.Declined;
        const want = try dsv41.hostF32(want_a);
        defer gpa.free(want);
        const x = try merge(&sc, k.merge, 64, in.pacc, in.pm, in.pl, in.sink, in.meta, in.periods);
        // wo_a's matrix units (4 rows on) take the heads rounded to bf16; the plain GEMV keeps them f32.
        const bar: f64 = if (R >= 4 and device.naxAvailable()) 5e-3 else 1e-4;
        for ([_]Plan{ planFor(R), .{ .tr = 64, .nb = 4 }, .{ .tr = 32, .nb = 2 }, .{ .tr = 16, .nb = 8 } }) |p| {
            const got = try dsv41.hostF32((try gemv(&sc, k.gemv, .bfloat16, x, in.w, in.s, p)) orelse return error.Declined);
            defer gpa.free(got);
            for (got) |v| if (!std.math.isFinite(v)) return error.NonFinite;
            const e = decode.relRms(got, want);
            if (e > bar) {
                std.debug.print("[dec] wo_a plain R={d} S={d} TR={d} NB={d}: rel rms vs wo_a {e:.2} (bar {e:.0})\n", .{ R, S, p.tr, p.nb, e, bar });
                return error.TestUnexpectedResult;
            }
        }
        const bad = try dsv41.hostF32((try gemv(&sc, plant, .bfloat16, x, in.w, in.s, planFor(R))) orelse return error.Declined);
        defer gpa.free(bad);
        const eb = decode.relRms(bad, want);
        std.debug.print("[dec] wo_a plain R={d} S={d}: planted dropped head, rel rms vs wo_a {e:.2}\n", .{ R, S, eb });
        if (eb < 1e-2) return error.PlantedBugPassed;
    };
}

test "dsv41 dec wo_a plain: with the switch on, a fused decode step's layers take it (rows 1 and 4)" {
    if (!on()) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !device.naxAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fixture = @import("fixture.zig");
    for ([_]usize{ 1, 4 }) |rows| {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .ctx = 300, .fused = true });
        defer fx.deinit();
        try testing.expect(!fx.m.dec.woa_plain_logged);
        const x = try fx.stream(rows);
        defer _ = mlx.mlx_array_free(x);
        const pre = try fx.cpre(rows);
        defer _ = mlx.mlx_array_free(pre);
        const out = try dsv41.workerStep(fx.m, gpa, x, pre, 300, false, false);
        defer for (out) |o| {
            _ = mlx.mlx_array_free(o);
        };
        const host = try dsv41.hostF32(out[0]);
        defer gpa.free(host);
        for (host) |v| try testing.expect(std.math.isFinite(v));
        try testing.expect(fx.m.dec.woa_plain_logged);
    }
}
