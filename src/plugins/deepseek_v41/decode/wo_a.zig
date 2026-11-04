//! k6 `wo_a` (`~/plans/dsv41-decode-kernels-spec.md`): the attention's split partials merged (the sink in the
//! denominator, the inverse rope on each head's tail) and run through the grouped wo_a in one dispatch.
//!
//! A threadgroup owns a tile of one wo_a group's outputs and ONE of the group's heads: it merges that head's splits
//! into threadgroup memory, streams its rows' 512 mxfp8 weights against every row's head and publishes the partial
//! sums; the last of a tile's heads to arrive adds them in head order, so the bytes never depend on arrival order,
//! and writes `oa`. The merged heads stay f32 into the sums where today's chain rounds them to bf16 first.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mxfp8_qmv = @import("../../../sdk/blocks/qmm/mxfp8_qmv.zig");
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const contract = @import("../contract.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const fixture = @import("fixture.zig");

const A = mlx.mlx_array;
const Scope = dsv41.Scope;

pub const SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_wo_a.metal");
/// The e4m3 pair decode the SDK's mxfp8 GEMV uses, a word's four values, and the matrix units' tile op.
pub const HEADER = mxfp8_qmv.HEADER ++ kernelSource("plugins/deepseek_v41/kernels/dec_wo_a_header.metal") ++ kernelSource("plugins/deepseek_v41/kernels/dec_nax_header.metal");
pub const INS = [_][*:0]const u8{ "pacc", "pm", "pl", "sink", "meta", "periods", "w", "scales", "counter" };
pub const OUTS = [_][*:0]const u8{ "out", "parts" };

pub fn build() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_dec_wo_a", &INS, &OUTS, SOURCE, HEADER);
}

/// The head's width: 32 lanes of 16 channels.
const HD = 512;
/// Words of the arrival counter (a word per tile of every group); it counts modulo the heads of a group, so the
/// same zeroed array serves every dispatch.
pub const counter_words = 1024;

/// The arrival counter, zeroed once (by the model, beside its kernels).
pub fn newCounter() A {
    return mlx.mlx_array_new_data(&std.mem.zeroes([counter_words]u32), &[_]c_int{counter_words}, 1, .uint32);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// Simdgroups of a threadgroup and the weight rows each streams: a threadgroup owns NSG * NR outputs. The matrix units
/// take 32 rows a simdgroup, from `nax_rows` activation rows on (below, the vector units' one pass over the weights is
/// the cheaper); they exist on M5 and newer.
const NSG = 8;
const NR = 8;
const NAX_NSG = 4;
const NAX_NR = 32;
const nax_rows = 4;

fn useNax(rows: c_int) bool {
    return rows >= nax_rows and device.naxAvailable();
}

/// One dispatch from the shapes: the grid, the threadgroup, the outputs and the template ints; null where the kernel
/// declines (rows past a verify block, a head that is not 512 wide, tiles that do not cut the outputs).
pub const Plan = struct {
    grid: [3]c_int,
    tg: [3]c_int,
    out: [2]c_int,
    parts: [3]c_int,
    ints: [11]Int,

    pub const Int = struct { name: [*:0]const u8, v: c_int };
};

pub fn plan(rows: c_int, heads: c_int, splits: c_int, dim: c_int, groups: c_int, out_rank: c_int, rd: usize, nax: bool) ?Plan {
    const nsg: c_int = if (nax) NAX_NSG else NSG;
    const nr: c_int = if (nax) NAX_NR else NR;
    const nt = nsg * nr;
    if (rows < 1 or rows > decode.max_rows or splits < 1 or splits > 8 or dim != HD or groups < 1 or @mod(heads, groups) != 0) return null;
    if (out_rank < nt or @mod(out_rank, nt) != 0 or groups * @divExact(out_rank, nt) > counter_words) return null;
    const tg = nsg * 32;
    return .{
        .grid = .{ @divExact(out_rank, nt) * tg, heads, 1 },
        .tg = .{ tg, 1, 1 },
        .out = .{ rows, groups * out_rank },
        .parts = .{ @divExact(heads, groups), rows, groups * out_rank },
        .ints = .{
            .{ .name = "R", .v = rows },      .{ .name = "S", .v = splits },
            .{ .name = "H", .v = heads },     .{ .name = "D", .v = dim },
            .{ .name = "RD", .v = ci(rd) },   .{ .name = "HPG", .v = @divExact(heads, groups) },
            .{ .name = "OL", .v = out_rank }, .{ .name = "OW", .v = groups * out_rank },
            .{ .name = "NSG", .v = nsg },     .{ .name = "NR", .v = nr },
            .{ .name = "NAX", .v = @intFromBool(nax) },
        },
    };
}

/// `oa` [R, groups * o_lora] in `act` from the split partials (`pacc` [R, H, S, D] f32, `pm` and `pl` [R, H, S]),
/// the sink [H], `meta` (int32, [0] the first row's position), the inverse rope periods and wo_a's mxfp8 words
/// `w` [groups, o_lora, K / 4] and scales [groups, o_lora, K / 32]; `extra` is an input the kernel never reads (a
/// chain's dependency). Null where the shapes or formats are not the kernel's.
pub fn launch(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, rd: usize, odt: mlx.mlx_dtype, ins: [8]A, counter: A, extra: ?A) !?A {
    const ps = mlx.getShape(ins[0]);
    const ws = mlx.getShape(ins[6]);
    if (ps.len != 4 or ws.len != 3) return null;
    const dts = [_]mlx.mlx_dtype{ .float32, .float32, .float32, .float32, .int32, .float32, .uint32, .uint8 };
    for (ins, dts) |x, dt| if (mlx.mlx_array_dtype(x) != dt) return null;
    const p = plan(ps[0], ps[1], ps[2], ps[3], ws[0], ws[1], rd, useNax(ps[0])) orelse return null;
    if (mlx.mlx_array_size(ins[3]) != ps[1] or mlx.mlx_array_size(ins[5]) != rd / 2 or ws[2] * 4 != p.ints[5].v * ps[3]) return null;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &p.out, 2, odt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &p.parts, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, p.grid[0], p.grid[1], p.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, p.tg[0], p.tg[1], p.tg[2]));
    for (p.ints) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name, t.v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OT", odt));
    var all: [10]A = undefined;
    @memcpy(all[0..8], &ins);
    all[8] = counter;
    if (extra) |e| all[9] = e;
    const iv = mlx.mlx_vector_array_new_data(&all, if (extra != null) 10 else 9);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, c, sc.s));
    var o = mlx.mlx_array_new();
    return try sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// The layer's `attnOutRoped` input from the splits: `oa` [R, 8192] in the model's activation dtype, what wo_b reads.
/// Null (the old path runs) off the GPU, where wo_a is not the release's 8-bit mxfp8 of group 32, or past the rows,
/// splits or head width `plan` declines.
pub fn apply(sc: *Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, kern: mlx.mlx_fast_metal_kernel, counter: A, inv: A, meta: A, pacc: A, pm: A, pl: A) !?A {
    const wo = &ly.wo_a;
    if (!mlx.streamIsGpu(sc.s) or wo.s.ctx == null or wo.qp.mode != .mxfp8 or wo.qp.bits != 8 or wo.qp.group_size != 32) return null;
    return launch(sc, kern, m.cfg.dsv4_rope_head_dim, m.act, .{ pacc, pm, pl, ly.sink, meta, inv, wo.w, wo.s }, counter, null);
}

// ── the kernel contract ──

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;

const inf = kit.inf;
const nan = kit.nan;
const RD = 64;
const HPG = 8;
const OL = 1024;
const act: mlx.mlx_dtype = .bfloat16;

/// An e4m3 byte's value (sign, 4 exponent bits at bias 7, 3 mantissa bits; subnormal at exponent 0).
fn e4m3(b: u8) f64 {
    const s: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 15;
    const m: f64 = @floatFromInt(b & 7);
    return s * if (e == 0) m * 0x1p-9 else (1 + m / 8) * std.math.ldexp(@as(f64, 1), e - 7);
}

/// The e4m3 byte of the integer `c` (|c| <= 15, exact).
fn e4m3Of(c: i32) u8 {
    if (c == 0) return 0;
    const a: u32 = @abs(c);
    const e: u5 = @intCast(31 - @clz(a));
    const m = ((a << 3) >> e) & 7;
    return @intCast((if (c < 0) @as(u32, 0x80) else 0) | ((@as(u32, e) + 7) << 3) | m);
}

fn depthWoA(s: Shape) usize {
    return s.k + 16;
}

const WoA = struct {
    pub const Params = struct { s: usize = 4, groups: usize = 2, p0: usize = 3001, nax: bool = false };
    const slot = struct {
        const pacc = 0;
        const pm = 1;
        const pl = 2;
        const sink = 3;
        const periods = 5;
        const w = 6;
        const scales = 7;
    };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.k != HPG * HD) return null;
        const pl = plan(ci(s.rows), ci(p.groups * HPG), ci(p.s), HD, ci(p.groups), ci(s.n), RD, p.nax) orelse return null;
        var g = metal.Geometry{ .grid = pl.grid, .tg = pl.tg, .unread = 0b10 };
        g.out(&pl.out, act);
        g.out(&pl.parts, .float32);
        for (pl.ints) |t| g.int(t.name, t.v);
        g.dtype("OT", act);
        return g;
    }

    pub fn count(p: Params, s: Shape) usize {
        return s.rows * p.groups * s.n;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const R = k.shape.rows;
        const N = k.shape.n;
        const G = k.p.groups;
        const S = k.p.s;
        const H = G * HPG;
        const pacc = try k.add(.float32, &.{ R, H, S, HD });
        const pm = try k.add(.float32, &.{ R, H, S });
        const pl = try k.add(.float32, &.{ R, H, S });
        const sink = try k.add(.float32, &.{ 1, H, 1 });
        const meta = try k.add(.int32, &.{6});
        const periods = try k.add(.float32, &.{RD / 2});
        const w = try k.add(.uint32, &.{ G, N, HPG * HD / 4 });
        const scales = try k.add(.uint8, &.{ G, N, HPG * HD / 32 });
        _ = try k.add(.uint32, &.{counter_words});
        @memcpy(meta, &[_]f64{ @floatFromInt(k.p.p0), @floatFromInt(R), HD, 128, 144, 0 });
        switch (f) {
            .random, .nonfinite => {
                kit.uniform(rng, pm, -2, 4);
                kit.uniform(rng, pl, 1, 20);
                kit.uniform(rng, sink, -3, 3);
                for (periods, 0..) |*v, i| v.* = -std.math.pow(f64, 10000, @as(f64, @floatFromInt(i)) / 32);
                for (pm, pl, 0..) |*m, *l, j| {
                    // An empty split (no rows of its own) every so often, as the split kernel writes it.
                    const empty = j % 13 == 5;
                    if (empty) {
                        m.* = -inf;
                        l.* = 0;
                    }
                    for (pacc[j * HD ..][0..HD]) |*v| v.* = if (empty) 0 else (rng.float(f64) * 2 - 1) * 2 * l.*;
                }
                for (w) |*word| {
                    var bytes: [4]u8 = undefined;
                    for (&bytes) |*b| {
                        b.* = rng.int(u8);
                        while (b.* & 0x7f == 0x7f) b.* = rng.int(u8);
                    }
                    word.* = @floatFromInt(std.mem.readInt(u32, &bytes, .little));
                }
                for (scales) |*v| v.* = @floatFromInt(118 + rng.uintLessThan(u8, 11));
                if (f == .nonfinite) {
                    // A NaN, +Inf, -Inf and both in one head, in rows 0-3, at channels the rope leaves alone.
                    const at = [_][4]usize{ .{ 0, 0, 0, 170 }, .{ 1, 1, 0, 256 }, .{ 2, HPG + 1, 0, 300 }, .{ 3, 2, 0, 20 }, .{ 3, 2, 1, 400 } };
                    const put = [_]f64{ nan, inf, -inf, inf, -inf };
                    for (at, put) |q, v| if (q[0] < R) {
                        pacc[((q[0] * H + q[1]) * S + q[2]) * HD + q[3]] = v;
                    };
                }
            },
            .stress => {
                // Splits that merge to x exactly: the sink far below, every max 0 and sum 1, split 0 holding S * x,
                // the angles zero (periods 2^100); x the mirrored integers of every head, the weights e4m3 integers
                // at scale 1.
                @memset(sink, -1e30);
                @memset(pm, 0);
                @memset(pl, 1);
                @memset(periods, 0x1p100);
                const K = HPG * HD;
                const x = try k.scratch().alloc(f64, K);
                defer k.scratch().free(x);
                const wi = try k.scratch().alloc(f64, G * N * K);
                defer k.scratch().free(wi);
                try contract.stressDot(rng, x, wi, G * N, K, 64, 15, k.scratch());
                @memset(pacc, 0);
                for (0..R) |m| for (0..H) |h| {
                    const dst = pacc[((m * H + h) * S) * HD ..][0..HD];
                    for (dst, x[(h % HPG) * HD ..][0..HD]) |*d, xv| d.* = xv * @as(f64, @floatFromInt(S));
                };
                for (w, 0..) |*word, i| {
                    var bytes: [4]u8 = undefined;
                    for (&bytes, 0..) |*b, j| b.* = e4m3Of(@intFromFloat(wi[i * 4 + j]));
                    word.* = @floatFromInt(std.mem.readInt(u32, &bytes, .little));
                }
                @memset(scales, 127);
            },
        }
    }

    /// float64 over the inputs: each head merged with the sink, its tail unrotated, then every output's dot with its
    /// dequantized weight row; `mag` the terms' magnitude.
    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const R = k.shape.rows;
        const N = k.shape.n;
        const G = k.p.groups;
        const S = k.p.s;
        const H = G * HPG;
        const K = HPG * HD;
        const a = k.scratch();
        const x = try a.alloc(f64, R * K);
        defer a.free(x);
        const wrow = try a.alloc(f64, K);
        defer a.free(wrow);
        const t = try a.alloc(f64, K);
        defer a.free(t);
        const pacc = k.in(slot.pacc);
        for (0..G) |g| {
            for (0..R) |m| for (0..HPG) |hh| {
                const h = g * HPG + hh;
                const base = (m * H + h) * S;
                var mx = k.in(slot.sink)[h];
                for (0..S) |j| mx = kit.maxN(mx, k.in(slot.pm)[base + j]);
                var total = @exp(k.in(slot.sink)[h] - mx);
                var wj: [8]f64 = undefined;
                for (0..S) |j| {
                    const l = k.in(slot.pl)[base + j];
                    wj[j] = if (l > 0) @exp(k.in(slot.pm)[base + j] - mx) else 0;
                    total += l * wj[j];
                }
                const o = x[m * K + hh * HD ..][0..HD];
                for (o, 0..) |*v, d| {
                    var s: f64 = 0;
                    for (0..S) |j| s += pacc[(base + j) * HD + d] * wj[j];
                    v.* = s / total;
                }
                for (0..RD / 2) |q| {
                    const th = spec.roundTo(.f32, @as(f64, @floatFromInt(k.p.p0 + m)) * spec.roundTo(.f32, 1 / k.in(slot.periods)[q]));
                    const x1 = o[HD - RD + 2 * q];
                    const x2 = o[HD - RD + 2 * q + 1];
                    o[HD - RD + 2 * q] = x1 * @cos(th) - x2 * @sin(th);
                    o[HD - RD + 2 * q + 1] = x1 * @sin(th) + x2 * @cos(th);
                }
                // The matrix units take the head as the chain hands it to wo_a: in bf16.
                if (k.p.nax) for (o) |*v| {
                    v.* = spec.roundTo(.bf16, v.*);
                };
            };
            for (0..N) |n| {
                const row = g * N + n;
                const words = k.in(slot.w)[row * K / 4 ..][0 .. K / 4];
                for (words, 0..) |word, i| {
                    var bytes: [4]u8 = undefined;
                    std.mem.writeInt(u32, &bytes, @intFromFloat(word), .little);
                    for (bytes, 0..) |b, j| {
                        const idx = i * 4 + j;
                        const e: i32 = @intFromFloat(k.in(slot.scales)[row * K / 32 + idx / 32]);
                        wrow[idx] = e4m3(b) * std.math.ldexp(@as(f64, 1), e - 127);
                    }
                }
                for (0..R) |m| {
                    var mg: f64 = 0;
                    for (t, wrow, x[m * K ..][0..K]) |*term, wv, xv| {
                        term.* = xv * wv;
                        mg += @abs(term.*);
                    }
                    out[(m * G + g) * N + n] = sum.of(t);
                    mag[(m * G + g) * N + n] = mg;
                }
            }
        }
    }
};

/// Served at 256 of a group's 1024 outputs and two groups: the float64 reference would run for seconds otherwise; the
/// reduction (a group's 8 heads of 512) stays the release's.
const served = [_]Shape{
    .{ .rows = 1, .n = 256, .k = HPG * HD },
    .{ .rows = 2, .n = 256, .k = HPG * HD },
    .{ .rows = 3, .n = 256, .k = HPG * HD },
};
const served_nax = [_]Shape{
    .{ .rows = nax_rows, .n = 256, .k = HPG * HD },
    .{ .rows = decode.max_rows, .n = 256, .k = HPG * HD },
};

/// `tg_bytes`: the merged heads of six rows (a row is 512 f32), their split weights and the arrival flag. `m1`, `m2`: the
/// least M1/M2 pipeline thread limit over the instantiations (576 at one and two rows, 512 above), so 256 threads always fit.
const decl: metal.Launch = .{
    .name = "dsv41_dec_wo_a",
    .inputs = &INS,
    .outputs = &OUTS,
    .source = SOURCE,
    .header = HEADER,
    .tg_bytes = 4 * (4 * HD / 4 * decode.max_rows + decode.max_rows * 9 + 1),
    .m1 = 512,
    .m2 = 512,
};

/// The same text on the matrix units: the merged heads as bf16 rows padded 8 apart.
const decl_nax: metal.Launch = .{
    .name = "dsv41_dec_wo_a",
    .inputs = &INS,
    .outputs = &OUTS,
    .source = SOURCE,
    .header = HEADER,
    .tg_bytes = 4 * (decode.max_rows * (HD + 8) / 2 + decode.max_rows * 9 + 1),
    .offline = false,
    .min_gen = 17,
};

pub fn register(set: *kc.Set) !void {
    try kit.add(set, WoA, &decl, .{}, .{
        .name = "dsv41 dec wo_a (split merge, inverse rope, grouped mxfp8)",
        .accum = .f32,
        .depth = depthWoA,
        .ops = 8,
        .out = .bf16,
        .in_place = true,
        .tiles = .{ .{}, .{ .size = NR * NSG, .step = NR * NSG, .min = NR * NSG }, .{} },
        .served = &served,
    });
    try kit.add(set, WoA, &decl_nax, .{ .nax = true }, .{
        .name = "dsv41 dec wo_a (matrix units)",
        .accum = .f32,
        .depth = depthWoA,
        .ops = 8,
        .out = .bf16,
        .in_place = true,
        .tiles = .{ .{}, .{ .size = NAX_NR * NAX_NSG, .step = NAX_NR * NAX_NSG, .min = NAX_NR * NAX_NSG }, .{} },
        .served = &served_nax,
    });
}

// ── tests ──

const testing = std.testing;
const none: A = .{ .ctx = null };

/// Today's merge (`attn_merge`) of the partials into roped heads [R, H, D].
fn chainMerge(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, rd: usize, ins: [6]A, extra: ?A) !A {
    const ps = mlx.getShape(ins[0]);
    const mc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mc, &[_]c_int{ ps[0], ps[1], ps[3] }, 3, act));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mc, ps[3], ps[1], ps[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mc, ps[3], 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "D", ps[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "S", ps[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mc, "OT", act));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "RD", ci(rd)));
    var all: [7]A = undefined;
    @memcpy(all[0..6], &ins);
    if (extra) |e| all[6] = e;
    const mv = mlx.mlx_vector_array_new_data(&all, if (extra != null) 7 else 6);
    defer _ = mlx.mlx_vector_array_free(mv);
    var mo = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(mo);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&mo, kern, mv, mc, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, mo, 0), &o);
}

/// `attnOutRoped`'s grouped wo_a on roped heads `r` [R, H, D]: [R, groups * o_lora].
fn chainWoA(sc: *Scope, r: A, w: A, s: A) !A {
    const L = mlx.getShape(r)[0];
    const og = mlx.getShape(w)[0];
    const ob = try sc.transpose(try sc.reshape(r, &.{ L, og, -1 }), &.{ 1, 0, 2 });
    var ro = mlx.mlx_array_new();
    const oa = try sc.res(mlx.mlx_quantized_matmul(&ro, ob, w, s, none, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", sc.s), &ro);
    return sc.astype(try sc.reshape(try sc.transpose(oa, &.{ 1, 0, 2 }), &.{ L, -1 }), act);
}

const Parts = struct { pacc: []f32, pm: []f32, pl: []f32 };

/// Split partials with the structure the split kernel writes: each split's max `pm`, its weights' sum `pl` (>= 1),
/// its output `pacc` = the weighted rows (so about `pl` times a row); some splits empty (max -inf, sum 0).
fn makeParts(a: std.mem.Allocator, seed: u64, R: usize, H: usize, S: usize, D: usize) !Parts {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const pm = try a.alloc(f32, R * H * S);
    const pl = try a.alloc(f32, R * H * S);
    const pacc = try a.alloc(f32, R * H * S * D);
    for (pm, pl, 0..) |*m, *l, j| {
        m.* = -2 + 6 * rnd.float(f32);
        l.* = 1 + 19 * rnd.float(f32);
        const empty = j % 13 == 5;
        if (empty) {
            m.* = -std.math.inf(f32);
            l.* = 0;
        }
        for (pacc[j * D ..][0..D]) |*v| v.* = if (empty) 0 else rnd.floatNorm(f32) * l.*;
    }
    return .{ .pacc = pacc, .pm = pm, .pl = pl };
}

fn upload(data: []const f32, shape: []const c_int) A {
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

/// The float64 truth: merge of the partials with the sink, the inverse rope, then wo_a over the dequantized `wd`
/// [groups, o_lora, K] (f32 values, exact for the release's fp8), no rounding anywhere: [R, groups * o_lora].
fn truth(a: std.mem.Allocator, parts: Parts, sink: []const f32, periods: []const f32, wd: []const f32, p0: usize, R: usize, S: usize) ![]f32 {
    const H = 64;
    const G = 8;
    const K = H / G * HD;
    const x = try a.alloc(f64, R * H * HD);
    defer a.free(x);
    for (0..R) |m| for (0..H) |h| {
        const base = (m * H + h) * S;
        var mx: f64 = sink[h];
        for (0..S) |j| mx = @max(mx, parts.pm[base + j]);
        var total = @exp(@as(f64, sink[h]) - mx);
        var wj: [8]f64 = undefined;
        for (0..S) |j| {
            wj[j] = if (parts.pl[base + j] > 0) @exp(@as(f64, parts.pm[base + j]) - mx) else 0;
            total += @as(f64, parts.pl[base + j]) * wj[j];
        }
        const o = x[(m * H + h) * HD ..][0..HD];
        for (o, 0..) |*v, d| {
            var s: f64 = 0;
            for (0..S) |j| s += @as(f64, parts.pacc[(base + j) * HD + d]) * wj[j];
            v.* = s / total;
        }
        for (0..RD / 2) |p| {
            const th = @as(f64, @floatFromInt(p0 + m)) / @as(f64, periods[p]);
            const x1 = o[HD - RD + 2 * p];
            const x2 = o[HD - RD + 2 * p + 1];
            o[HD - RD + 2 * p] = x1 * @cos(th) - x2 * @sin(th);
            o[HD - RD + 2 * p + 1] = x1 * @sin(th) + x2 * @cos(th);
        }
    };
    const out = try a.alloc(f32, R * G * OL);
    for (0..R) |m| for (0..G) |g| for (0..OL) |n| {
        const wr = wd[(g * OL + n) * K ..][0..K];
        const xr = x[m * H * HD + g * K ..][0..K];
        var s: f64 = 0;
        for (wr, xr) |wv, xv| s += @as(f64, wv) * xv;
        out[m * G * OL + g * OL + n] = @floatCast(s);
    };
    return out;
}

test "dsv41 dec wo_a: 1 to 6 rows are no further from the float64 truth than today's merge + wo_a chain" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .ctx = 0 });
    defer fx.deinit();
    const li = 20;
    const ly = fx.layer(li);
    const rd = fx.m.cfg.dsv4_rope_head_dim;
    const inv = dsv41.ropeFor(fx.m, li).inv;
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const counter = try sc.keep(newCounter());

    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    try mlx.check(mlx.mlx_dequantize(&deq, ly.wo_a.w, ly.wo_a.s, none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .float32, .has_value = true }, mlx.gpuStream()));
    const wd = try dsv41.hostF32(deq);
    defer gpa.free(wd);
    const sink = try dsv41.hostF32(ly.sink);
    defer gpa.free(sink);
    const periods = try dsv41.hostF32(inv);
    defer gpa.free(periods);

    const p0: usize = 3001;
    for ([_][2]usize{ .{ 1, 4 }, .{ 2, 4 }, .{ 3, 2 }, .{ 4, 4 }, .{ 5, 8 }, .{ 6, 4 }, .{ 1, 8 }, .{ 6, 1 } }) |rs| {
        const R, const S = rs;
        const parts = try makeParts(gpa, 100 + R, R, 64, S, HD);
        defer gpa.free(parts.pacc);
        defer gpa.free(parts.pm);
        defer gpa.free(parts.pl);
        const pacc = try sc.keep(upload(parts.pacc, &.{ ci(R), 64, ci(S), HD }));
        const pm = try sc.keep(upload(parts.pm, &.{ ci(R), 64, ci(S) }));
        const pl = try sc.keep(upload(parts.pl, &.{ ci(R), 64, ci(S) }));
        const meta = try sc.ints(&.{ @intCast(p0), @intCast(R), 512, 128, 144, 0 }, &.{6});

        const heads = try chainMerge(&sc, fx.m.attn_merge_kernel, rd, .{ pacc, pm, pl, ly.sink, meta, inv }, null);
        const want = try chainWoA(&sc, heads, ly.wo_a.w, ly.wo_a.s);
        const t = try truth(gpa, parts, sink, periods, wd, p0, R, S);
        defer gpa.free(t);
        const w32 = try dsv41.hostF32(want);
        defer gpa.free(w32);
        const got = (try apply(&sc, fx.m, ly, kern, counter, inv, meta, pacc, pm, pl)) orelse return error.Declined;
        try testing.expectEqual(@as(c_int, ci(8 * OL)), mlx.getShape(got)[1]);
        const g32 = try dsv41.hostF32(got);
        defer gpa.free(g32);
        // The counter has moved on: the same inputs give the same bytes.
        const again = (try apply(&sc, fx.m, ly, kern, counter, inv, meta, pacc, pm, pl)) orelse return error.Declined;
        const a32 = try dsv41.hostF32(again);
        defer gpa.free(a32);
        try testing.expectEqualSlices(f32, g32, a32);
        std.debug.print("[dec] wo_a R={d} S={d}: new-chain {e:.2} (rel rms)\n", .{ R, S, decode.relRms(g32, w32) });
        try decode.expectNoWorse("wo_a", g32, w32, t);
    }
}

test "dsv41 dec wo_a: shapes the kernel was not built for are declined" {
    try testing.expect(plan(6, 64, 8, 512, 8, 1024, RD, false) != null);
    try testing.expect(plan(6, 64, 8, 512, 8, 1024, RD, true) != null);
    try testing.expect(plan(6, 64, 8, 512, 8, 1000 + 24, RD, true) != null);
    try testing.expect(plan(6, 64, 8, 512, 8, 96, RD, true) == null);
    try testing.expect(plan(7, 64, 4, 512, 8, 1024, RD, false) == null);
    try testing.expect(plan(0, 64, 4, 512, 8, 1024, RD, false) == null);
    try testing.expect(plan(1, 64, 9, 512, 8, 1024, RD, false) == null);
    try testing.expect(plan(1, 64, 4, 256, 8, 1024, RD, false) == null);
    try testing.expect(plan(1, 60, 4, 512, 8, 1024, RD, false) == null);
    try testing.expect(plan(1, 64, 4, 512, 8, 1000, RD, false) == null);
}

test "dsv41 dec wo_a: the declared launches are the wrapper's, byte for byte" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    var sc = Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    for ([_]bool{ false, true }) |nax| {
        if (nax and !device.naxAvailable()) continue;
        const c = try a.create(kit.Case(WoA));
        defer {
            c.deinit();
            a.destroy(c);
        }
        c.* = .init(a, if (nax) &decl_nax else &decl, .{ .nax = nax });
        try testing.expect(try c.prepare(.{ .rows = if (nax) 6 else 3, .n = 128, .k = HPG * HD }, .random, 7));
        try c.run();
        const mine = try a.dupe(u8, c.rig.outputBytes());
        defer a.free(mine);
        const got = (try launch(&sc, kern, RD, act, c.args[0..8].*, c.args[8], null)) orelse return error.Declined;
        try testing.expectEqualSlices(u8, mine, try c.rig.bytesOf(&.{got}));
    }
}

// ── ubench ──

const Copy = struct { w: A, s: A };

/// `n` distinct device copies of random wo_a weights (e4m3 words, e8m0 scales near 1).
fn randomCopies(gpa: std.mem.Allocator, n: usize, G: usize, KW: usize) ![]Copy {
    const wbuf = try gpa.alloc(u32, G * OL * KW);
    defer gpa.free(wbuf);
    const sbuf = try gpa.alloc(u8, G * OL * KW / 8);
    defer gpa.free(sbuf);
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (wbuf) |*v| v.* = rnd.int(u32);
    for (sbuf) |*v| v.* = 118 + rnd.uintLessThan(u8, 18);
    const out = try gpa.alloc(Copy, n);
    for (out) |*c| c.* = .{
        .w = mlx.mlx_array_new_data(wbuf.ptr, &[_]c_int{ ci(G), ci(OL), ci(KW) }, 3, .uint32),
        .s = mlx.mlx_array_new_data(sbuf.ptr, &[_]c_int{ ci(G), ci(OL), ci(KW / 8) }, 3, .uint8),
    };
    return out;
}

const Bench = struct {
    sc: *Scope,
    kern: mlx.mlx_fast_metal_kernel,
    merge: mlx.mlx_fast_metal_kernel,
    counter: A,
    old: bool,
    copies: []const Copy,
    common: [6]A,
    seed: A,
    prev: A = undefined,

    pub fn step(self: *Bench, i: usize) !A {
        const c = self.copies[i % self.copies.len];
        const dep = if (i == 0) self.seed else self.prev;
        self.prev = if (self.old) blk: {
            const heads = try chainMerge(self.sc, self.merge, RD, self.common, dep);
            break :blk try chainWoA(self.sc, heads, c.w, c.s);
        } else (try launch(self.sc, self.kern, RD, act, .{ self.common[0], self.common[1], self.common[2], self.common[3], self.common[4], self.common[5], c.w, c.s }, self.counter, dep)) orelse return error.Declined;
        return self.prev;
    }
};

/// Microseconds a step takes with its graph already built, median of 7: the GPU's time without the host's graph
/// building that `decode.chainUs` includes.
fn evalUs(ctx: anytype, n: usize) !f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    var times: [7]f64 = undefined;
    for (0..2 + times.len) |rep| {
        var last: A = .{ .ctx = null };
        for (0..n) |i| last = try ctx.step(i);
        const t0 = iou.nowUsMonotonic(io);
        try mlx.check(mlx.mlx_array_eval(last));
        const us = @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / @as(f64, @floatFromInt(n));
        if (rep >= 2) times[rep - 2] = us;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[times.len / 2];
}

test "dsv41 dec wo_a: ubench" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const G = 8;
    const H = 64;
    const KW = 1024;
    const S = 4;
    const bytes: f64 = @floatFromInt(G * OL * KW * 4 + G * OL * KW / 8);
    const copies = try randomCopies(gpa, 64, G, KW);
    defer {
        for (copies) |c| {
            _ = mlx.mlx_array_free(c.w);
            _ = mlx.mlx_array_free(c.s);
        }
        gpa.free(copies);
    }
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const kern = try dsv41.buildKernel("dsv41_dec_wo_a_dep", &(INS ++ [_][*:0]const u8{"dep"}), &OUTS, SOURCE, HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    const merge = try dsv41.buildKernel("dsv41_attn_merge_dep", &(dsv41.ATTN_MERGE_INS ++ [_][*:0]const u8{"dep"}), &.{"out"}, dsv41.ATTN_MERGE_SOURCE, dsv41.UNROPE_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(merge);
    const counter = try sc.keep(newCounter());

    var prng = std.Random.DefaultPrng.init(9);
    var sink: [H]f32 = undefined;
    for (&sink) |*v| v.* = prng.random().floatNorm(f32);
    var periods: [RD / 2]f32 = undefined;
    for (&periods, 0..) |*v, i| v.* = -std.math.pow(f32, 10000, @as(f32, @floatFromInt(i)) / 32);
    const sinka = try sc.keep(upload(&sink, &.{ 1, H, 1 }));
    const per = try sc.keep(upload(&periods, &.{RD / 2}));
    const seed = try sc.keep(upload(&sink, &.{H}));

    for ([_]usize{ 1, 2, 4, decode.max_rows }) |R| {
        const parts = try makeParts(gpa, 5, R, H, S, HD);
        defer gpa.free(parts.pacc);
        defer gpa.free(parts.pm);
        defer gpa.free(parts.pl);
        const pacc = try sc.keep(upload(parts.pacc, &.{ ci(R), H, S, HD }));
        const pm = try sc.keep(upload(parts.pm, &.{ ci(R), H, S }));
        const pl = try sc.keep(upload(parts.pl, &.{ ci(R), H, S }));
        const meta = try sc.ints(&.{ 3001, @intCast(R), 512, 128, 144, 0 }, &.{6});
        var b: Bench = .{ .sc = &sc, .kern = kern, .merge = merge, .counter = counter, .old = true, .copies = copies, .common = .{ pacc, pm, pl, sinka, meta, per }, .seed = seed };
        const old_us = try decode.chainUs(&b, copies.len);
        const old_eval = try evalUs(&b, copies.len);
        b.old = false;
        decode.report("wo_a", R, try decode.chainUs(&b, copies.len), bytes, old_us);
        std.debug.print("[dec-ubench] wo_a R={d}: eval-only {d:.1} us, replaced chain {d:.1} us\n", .{ R, try evalUs(&b, copies.len), old_eval });
    }
}
