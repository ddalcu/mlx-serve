//! The MoE's gate/up half for decode rows (R = 1..6) in one launch: routed and shared gate and up, clamped SwiGLU times the
//! routing weight, each distinct expert's weights read once.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const cv = @import("../contract.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const fixture = @import("fixture.zig");

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };
const testing = std.testing;

pub const SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_moe_up.metal");
pub const HEADER = kernelSource("plugins/deepseek_v41/kernels/dec_moe_up_header.metal");
pub const INS = [_][*:0]const u8{ "x", "w", "scales", "biases", "sw", "sscales", "inds", "wts", "lim" };

/// The pipeline and the swiglu clamp it reads (`dsv4_swiglu_limit`).
pub const Kernel = struct {
    k: mlx.mlx_fast_metal_kernel,
    lim: A,

    pub fn deinit(self: Kernel) void {
        _ = mlx.mlx_fast_metal_kernel_free(self.k);
        _ = mlx.mlx_array_free(self.lim);
    }
};

pub fn build(limit: f32) !Kernel {
    const k = try dsv41.buildKernel("dsv41_dec_moe_up", &INS, &.{"out"}, SOURCE, HEADER);
    return .{ .k = k, .lim = mlx.mlx_array_new_data(&[_]f32{limit}, &[_]c_int{1}, 1, .float32) };
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// The gate/up half of `h2`'s MoE (`h2` `[R, K]`, `idx` `[R, k]` u32, `wts` `[R, k]` f32; R <= `decode.max_rows`) as
/// `[R, k + 1, N]`: slot j < k the clamped SwiGLU of row r's j-th expert times its weight, slot k the shared expert's.
/// Null where the layer's formats are not the served ones (routed affine 2 or 3 bits, group 64; shared mxfp8) or the shape is not.
pub fn moeUp(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, kern: Kernel, ly: *const dsv41.Layer, h2: A, idx: A, wts: A) !?A {
    const gu = ly.experts.gu orelse return null;
    const sgu = ly.shared_gu orelse return null;
    if (!mlx.streamIsGpu(sc.s) or gu.s.ctx == null or gu.b.ctx == null or sgu.s.ctx == null) return null;
    const hs = mlx.getShape(h2);
    const is = mlx.getShape(idx);
    const gw = mlx.getShape(gu.w);
    if (hs.len != 2 or is.len != 2 or gw.len != 3 or is[0] != hs[0] or !std.mem.eql(c_int, is, mlx.getShape(wts))) return null;
    const rows: usize = @intCast(hs[0]);
    const topk: usize = @intCast(is[1]);
    const k: usize = @intCast(hs[1]);
    const bits = gu.qp.bits;
    if (rows < 1 or rows > decode.max_rows or topk < 1 or rows * topk > 64 or k % 1024 != 0) return null;
    if (gu.qp.mode != .affine or gu.qp.group_size != 64 or (bits != 2 and bits != 3) or gw[2] != ci(k * bits / 32) or @rem(gw[1], 2) != 0) return null;
    if (sgu.qp.mode != .mxfp8 or sgu.qp.bits != 8 or sgu.qp.group_size != 32) return null;
    const n: usize = @intCast(@divExact(gw[1], 2));
    const dt = mlx.mlx_array_dtype(h2);
    if (dt != m.act or mlx.mlx_array_dtype(gu.s) != dt or mlx.mlx_array_dtype(gu.b) != dt) return null;
    if (mlx.mlx_array_dtype(idx) != .uint32 or mlx.mlx_array_dtype(wts) != .float32) return null;
    const sw = mlx.getShape(sgu.w);
    const ss = mlx.getShape(sgu.s);
    if (sw.len != 2 or ss.len != 2 or sw[0] != ci(2 * n) or sw[1] != ci(k / 4) or ss[0] != ci(2 * n) or ss[1] != ci(k / 32)) return null;
    if (mlx.mlx_array_dtype(sgu.w) != .uint32 or mlx.mlx_array_dtype(sgu.s) != .uint8) return null;
    const vgu = if (ly.experts.views) |v| v[0] else gu;
    return try run(sc, kern, bits, h2, vgu.w, vgu.s, vgu.b, sgu.w, sgu.s, idx, wts);
}

/// The launch over arrays already checked: `w`, `scales` and `biases` the routed gate+up bank (or a view of its first
/// expert: experts are addressed from the buffer pointer), `sw` and `sscales` the shared mxfp8 gate+up.
pub fn run(sc: *dsv41.Scope, kern: Kernel, bits: usize, x: A, w: A, scales: A, biases: A, sw: A, sscales: A, idx: A, wts: A) !A {
    const rows: usize = @intCast(mlx.getShape(x)[0]);
    const k: usize = @intCast(mlx.getShape(x)[1]);
    const topk: usize = @intCast(mlx.getShape(idx)[1]);
    const n: usize = @intCast(@divExact(mlx.getShape(sw)[0], 2));
    const dt = mlx.mlx_array_dtype(x);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(topk + 1), ci(n) }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(32 * n * (topk * rows + 1)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg_threads, 1, 1));
    inline for (.{ .{ "R", rows }, .{ "K", k }, .{ "N", n }, .{ "TOPK", topk }, .{ "BITS", bits } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const ins = [_]A{ x, w, scales, biases, sw, sscales, idx, wts, kern.lim };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern.k, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

// ── the kernel contract ──

const TOPK = 6;
const act: mlx.mlx_dtype = .bfloat16;

/// Threads a threadgroup: two simdgroups share the decode table; M1 and M2 compile the one-row pipeline for 64.
const tg_threads = 64;

fn depth(s: Shape) usize {
    return s.k;
}

/// e4m3 (the OCP fn variant) code `b` as float64; the NaN code stays out of the fills.
fn e4m3(b: u8) f64 {
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 15;
    const m: f64 = @floatFromInt(b & 7);
    return sign * if (e == 0) m / 8 * 0x1p-6 else (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

/// Integers 0..8 as e4m3 codes (all exact).
const int_codes = [_]u8{ 0x00, 0x38, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54, 0x58 };

fn packBytes(words: []f64, bytes: []const u8) void {
    for (words, 0..) |*w, i| w.* = @floatFromInt(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
}

/// The shared bank's row `r`, input i, as float64 (e4m3 times its e8m0 scale).
fn sharedWeight(sw: []const f64, ss: []const f64, r: usize, i: usize, k: usize) f64 {
    const word: u32 = @intFromFloat(sw[r * (k / 4) + i / 4]);
    const byte: u8 = @truncate(word >> @intCast(8 * (i % 4)));
    return e4m3(byte) * std.math.pow(f64, 2, ss[r * (k / 32) + i / 32] - 127);
}

const MoeUp = struct {
    pub const Params = struct { bits: u32, experts: usize = 8, lim: f64 = 10 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.rows > decode.max_rows or s.n == 0 or s.k == 0 or s.k % 1024 != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(32 * s.n * (TOPK * s.rows + 1)), 1, 1 }, .tg = .{ tg_threads, 1, 1 } };
        g.out(&.{ ci(s.rows), TOPK + 1, ci(s.n) }, act);
        g.int("R", s.rows);
        g.int("K", s.k);
        g.int("N", s.n);
        g.int("TOPK", TOPK);
        g.int("BITS", p.bits);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const R = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const x = try k.add(act, &.{ R, K });
        const routed = try cv.Bank.add(k, &.{ p.experts, 2 * N }, p.bits, 64, K);
        const sw = try k.add(.uint32, &.{ 2 * N, K / 4 });
        const ss = try k.add(.uint8, &.{ 2 * N, K / 32 });
        const inds = try k.add(.uint32, &.{ R, TOPK });
        const wts = try k.add(.float32, &.{ R, TOPK });
        const lim = try k.add(.float32, &.{1});
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, wts, 0.05, 0.5);
        lim[0] = p.lim;
        for (0..R) |row| {
            var pool: [64]usize = undefined;
            for (pool[0..p.experts], 0..) |*e, i| e.* = i;
            rng.shuffle(usize, pool[0..p.experts]);
            for (0..TOPK) |j| inds[row * TOPK + j] = @floatFromInt(pool[j]);
        }
        const bytes = try k.scratch().alloc(u8, 2 * N * K);
        defer k.scratch().free(bytes);
        switch (f) {
            .random, .nonfinite => {
                try routed.random(rng, k.scratch());
                // e4m3 codes of exponent 5..9 (never the NaN code) under scales near 2^-6: weights near 0.02.
                for (bytes) |*b| b.* = @as(u8, @intCast(rng.uintLessThan(u32, 2))) << 7 | @as(u8, @intCast(5 + rng.uintLessThan(u32, 5))) << 3 | @as(u8, @intCast(rng.uintLessThan(u32, 8)));
                packBytes(sw, bytes);
                for (ss) |*e| e.* = @floatFromInt(120 + rng.uintLessThan(u32, 3));
            },
            .stress => {
                const perm = try k.scratch().alloc(usize, K / 2);
                defer k.scratch().free(perm);
                cv.mirroredX(rng, x[0..K], perm);
                for (1..R) |row| mirrorRow(rng, x[row * K ..][0..K], perm);
                try routed.mirrored(rng, perm, k.scratch());
                for (0..2 * N) |r| {
                    const h = K / 2;
                    for (0..h) |i| {
                        const c: u8 = if (i < 8) 1 else @intCast(rng.intRangeAtMost(u32, 1, 8));
                        bytes[r * K + i] = int_codes[c];
                        bytes[r * K + h + perm[i]] = int_codes[c] | 0x80;
                    }
                }
                packBytes(sw, bytes);
                @memset(ss, 127);
            },
        }
        if (f == .nonfinite) {
            // A NaN or infinite first-group scale in one gate or up row of row 0's slots 0-3 (gate, gate, up, both), the first group's x positive in every row (they share the experts).
            for (0..R) |row| kit.uniform(rng, x[row * K ..][0..64], 0.1, 2);
            const put = [_][3]f64{ .{ 0, kit.nan, 0 }, .{ 1, kit.inf, 0 }, .{ 2, 0, -kit.inf }, .{ 3, kit.inf, -kit.inf } };
            for (put) |q| {
                const e: usize = @intFromFloat(q[0]);
                const base = @as(usize, @intFromFloat(inds[e])) * 2 * N + e % N;
                if (q[1] != 0) try routed.plant(base, q[1], k.scratch());
                if (q[2] != 0) try routed.plant(base + N, q[2], k.scratch());
            }
        }
    }

    /// The mirrored x row for the same `perm` as the banks (x[h + perm[i]] = x[i] + 1 on the first 8, else x[i]).
    fn mirrorRow(rng: std.Random, x: []f64, perm: []const usize) void {
        const h = x.len / 2;
        for (0..h) |i| {
            x[i] = @floatFromInt(rng.intRangeAtMost(i32, 1, 64));
            x[h + perm[i]] = x[i] + @as(f64, if (i < 8) 1 else 0);
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * (TOPK + 1) * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const R = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const routed: cv.Bank = .{ .w = k.in(1), .s = k.in(2), .b = k.in(3), .bits = k.p.bits, .gs = 64, .k = K };
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        const lim = k.in(8)[0];
        for (0..R) |row| for (0..TOPK + 1) |j| for (0..N) |c| {
            const x = k.in(0)[row * K ..][0..K];
            var g: [2]f64 = undefined;
            var up: [2]f64 = undefined;
            var wt: f64 = 1;
            if (j < TOPK) {
                const e: usize = @intFromFloat(k.in(6)[row * TOPK + j]);
                g = routed.dot(e * 2 * N + c, x, sum, t);
                up = routed.dot(e * 2 * N + N + c, x, sum, t);
                wt = k.in(7)[row * TOPK + j];
            } else {
                g = sharedDot(k, c, x, sum, t);
                up = sharedDot(k, N + c, x, sum, t);
            }
            const gt = kit.spec.roundTo(.bf16, g[0]);
            const ut = kit.spec.roundTo(.bf16, up[0]);
            const gc = kit.minN(gt, lim);
            const uc = if (std.math.isNan(ut)) kit.nan else std.math.clamp(ut, -lim, lim);
            const sig = 1 / (1 + @exp(-gc));
            const o = (row * (TOPK + 1) + j) * N + c;
            out[o] = gc * sig * uc * wt;
            // Each sum's error (and its bf16 rounding) reaches y scaled by y's slope in it.
            const slope_g = @abs(sig * (1 + gc * (1 - sig)) * uc * wt);
            const u = 0x1p-24;
            mag[o] = (g[1] + @abs(g[0]) * 0x1p-8 / spec.gamma(@floatFromInt(K), u)) * slope_g * 2 + (up[1] + @abs(up[0]) * 0x1p-8 / spec.gamma(@floatFromInt(K), u)) * @abs(gc * sig * wt) * 2;
        };
    }

    /// Row `r` of the shared bank against `x`: the sum and its terms' magnitude.
    fn sharedDot(k: anytype, r: usize, x: []const f64, sum: Sum, t: []f64) [2]f64 {
        var mag: f64 = 0;
        for (0..k.shape.k) |i| {
            const w = sharedWeight(k.in(4), k.in(5), r, i, k.shape.k);
            t[i] = x[i] * w;
            mag += @abs(x[i] * w);
        }
        return .{ sum.of(t[0..k.shape.k]), mag };
    }
};

/// Threadgroup memory: the 2-bit decode table (256 float4); the 3-bit one is smaller.
const moe_up_l = cv.launch("dsv41_dec_moe_up", &INS, &.{"out"}, SOURCE, HEADER, 256 * 16, tg_threads, false);

/// Served shapes cut the output width to 64: each column runs the same code at its own index; the reduction length and
/// the rows (R = 1..6, over 8 experts so the rows share most of them) are the release's.
pub fn register(set: *kc.Set) !void {
    inline for (.{ 2, 3 }) |bits| {
        try kit.add(set, MoeUp, &moe_up_l, .{ .bits = bits }, .{
            .name = std.fmt.comptimePrint("dsv41 dec moe_up {d}-bit g64 + mxfp8 shared", .{bits}),
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{}, .{ .size = 1024, .step = 1024, .min = 1024 } },
            .served = &.{ .{ .rows = 1, .n = 64, .k = 5120 }, .{ .rows = 3, .n = 17, .k = 5120 }, .{ .rows = decode.max_rows, .n = 64, .k = 5120 } },
            .sums = .{ .f16 = 30, .bf16 = 30, .why = "a wrong sum's activation tends to 0, so a miss tops out near 1 / the bound" },
        });
    }
}

// ── tests ──

/// `res` of an op that wrote `o`.
fn own(sc: *dsv41.Scope, rc: c_int, o: A) !A {
    var x = o;
    return sc.res(rc, &x);
}

fn astype(sc: *dsv41.Scope, a: A, dt: mlx.mlx_dtype) !A {
    var o = mlx.mlx_array_new();
    return own(sc, mlx.mlx_astype(&o, a, dt, sc.s), o);
}

fn reshape(sc: *dsv41.Scope, a: A, shape: []const c_int) !A {
    var o = mlx.mlx_array_new();
    return own(sc, mlx.mlx_reshape(&o, a, shape.ptr, shape.len, sc.s), o);
}

fn mul(sc: *dsv41.Scope, a: A, b: A) !A {
    var o = mlx.mlx_array_new();
    return own(sc, mlx.mlx_multiply(&o, a, b, sc.s), o);
}

/// `a[..., lo:hi]`.
fn sliceLast(sc: *dsv41.Scope, a: A, lo: usize, hi: usize) !A {
    const sh = mlx.getShape(a);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    const step: [4]c_int = @splat(1);
    @memcpy(stop[0..sh.len], sh);
    start[sh.len - 1] = ci(lo);
    stop[sh.len - 1] = ci(hi);
    var o = mlx.mlx_array_new();
    return own(sc, mlx.mlx_slice(&o, a, &start, sh.len, &stop, sh.len, &step, sh.len, sc.s), o);
}

/// Row `row` of a 2-D array as `[1, W]`.
fn sliceRow(sc: *dsv41.Scope, a: A, row: usize) !A {
    const start: [2]c_int = .{ ci(row), 0 };
    const stop: [2]c_int = .{ ci(row + 1), mlx.getShape(a)[1] };
    const step: [2]c_int = .{ 1, 1 };
    var o = mlx.mlx_array_new();
    return own(sc, mlx.mlx_slice(&o, a, &start, 2, &stop, 2, &step, 2, sc.s), o);
}

fn concat(sc: *dsv41.Scope, xs: []const A, axis: c_int) !A {
    const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_array_new();
    return own(sc, mlx.mlx_concatenate_axis(&o, v, axis, sc.s), o);
}

/// Today's gate/up half for `h2`: `moe_gu` at one row, else `routed`'s gather, clamped SwiGLU and weight with the shared
/// expert's quantized matmul and SwiGLU: `[R, k + 1, N]`.
fn todays(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, h2: A, idx: A, wts: A) !A {
    const gu = ly.experts.gu.?;
    const sgu = ly.shared_gu.?;
    const rows: usize = @intCast(mlx.getShape(h2)[0]);
    const topk: usize = @intCast(mlx.getShape(idx)[1]);
    const k: usize = @intCast(mlx.getShape(h2)[1]);
    const n: usize = @intCast(@divExact(mlx.getShape(gu.w)[1], 2));
    if (rows == 1) {
        const vgu = ly.experts.views.?[0];
        const lim = try sc.keep(mlx.mlx_array_new_data(&[_]f32{m.cfg.dsv4_swiglu_limit}, &[_]c_int{1}, 1, .float32));
        const ins = [_]A{ h2, vgu.w, vgu.s, vgu.b, sgu.w, sgu.s, sgu.s, try reshape(sc, idx, &.{ci(topk)}), try reshape(sc, wts, &.{ci(topk)}), lim };
        const gu_act = try dsv41.moeKernel(sc, m.moe_gu_kernel, &ins, &.{ ci(topk + 1), ci(n) }, mlx.mlx_array_dtype(h2), gu.qp, sgu.qp, k, n, topk);
        return reshape(sc, gu_act, &.{ 1, ci(topk + 1), ci(n) });
    }
    const xe = try reshape(sc, h2, &.{ ci(rows), 1, 1, ci(k) });
    const gt, const up = try dsv41.gateUp(sc, &ly.experts, xe, none, idx, false); // [R, k, 1, N] each
    const routed = try astype(sc, try mul(sc, try dsv41.clampedSwiglu(sc, gt, up, m.cfg.dsv4_swiglu_limit), try reshape(sc, wts, &.{ ci(rows), ci(topk), 1, 1 })), m.act);
    var y = mlx.mlx_array_new();
    const sy = try own(sc, mlx.mlx_quantized_matmul(&y, h2, sgu.w, sgu.s, none, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", sc.s), y);
    const sh = try astype(sc, try dsv41.clampedSwiglu(sc, try sliceLast(sc, sy, 0, n), try sliceLast(sc, sy, n, 2 * n), m.cfg.dsv4_swiglu_limit), m.act);
    return concat(sc, &.{ try reshape(sc, routed, &.{ ci(rows), ci(topk), ci(n) }), try reshape(sc, sh, &.{ ci(rows), 1, ci(n) }) }, 1);
}

/// The same activations from the dequantized banks (each expert against every row in f32, then the picks and the
/// SwiGLU in float64): `[R, k + 1, N]`.
fn truth(gpa: std.mem.Allocator, sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, h2: A, ids: []const u32, wts: []const f32) ![]f32 {
    const gu = ly.experts.gu.?;
    const sgu = ly.shared_gu.?;
    const rows: usize = @intCast(mlx.getShape(h2)[0]);
    const topk = ids.len / rows;
    const n: usize = @intCast(@divExact(mlx.getShape(gu.w)[1], 2));
    const f32t: mlx.mlx_optional_dtype = .{ .value = .float32, .has_value = true };
    var xt = mlx.mlx_array_new();
    const x32 = try own(sc, mlx.mlx_transpose(&xt, try astype(sc, h2, .float32), sc.s), xt);
    var d = mlx.mlx_array_new();
    const bank = try own(sc, mlx.mlx_dequantize(&d, gu.w, gu.s, gu.b, mlx.mlx_optional_int.some(@intCast(gu.qp.group_size)), mlx.mlx_optional_int.some(@intCast(gu.qp.bits)), "affine", none, f32t, sc.s), d);
    var y = mlx.mlx_array_new();
    const yb = try dsv41.hostF32(try own(sc, mlx.mlx_matmul(&y, bank, x32, sc.s), y)); // [E, 2N, R]
    defer testing.allocator.free(yb);
    var ds = mlx.mlx_array_new();
    const sbank = try own(sc, mlx.mlx_dequantize(&ds, sgu.w, sgu.s, none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, f32t, sc.s), ds);
    var ys = mlx.mlx_array_new();
    const yh = try dsv41.hostF32(try own(sc, mlx.mlx_matmul(&ys, sbank, x32, sc.s), ys)); // [2N, R]
    defer testing.allocator.free(yh);
    const lim: f64 = m.cfg.dsv4_swiglu_limit;
    const out = try gpa.alloc(f32, rows * (topk + 1) * n);
    for (0..rows) |row| for (0..topk + 1) |j| for (0..n) |c| {
        const g: f64, const u: f64, const wt: f64 = if (j < topk) blk: {
            const e: usize = ids[row * topk + j];
            break :blk .{ yb[(e * 2 * n + c) * rows + row], yb[(e * 2 * n + n + c) * rows + row], wts[row * topk + j] };
        } else .{ yh[c * rows + row], yh[(n + c) * rows + row], 1 };
        const gc = @min(g, lim);
        const uc = std.math.clamp(u, -lim, lim);
        out[(row * (topk + 1) + j) * n + c] = @floatCast(gc / (1 + @exp(-gc)) * uc * wt);
    };
    return out;
}

const Routing = enum { random, shared_all, disjoint };

/// k distinct experts of `experts` a row, per `how`; weights in (0.05, 0.5).
fn route(rng: std.Random, how: Routing, rows: usize, topk: usize, experts: u32, ids: []u32, wts: []f32) void {
    for (0..rows) |row| for (0..topk) |j| {
        ids[row * topk + j] = switch (how) {
            .shared_all => @intCast(j * 2),
            .disjoint => @intCast((row * topk + j) % experts),
            .random => while (true) {
                const e = rng.uintLessThan(u32, experts);
                if (std.mem.indexOfScalar(u32, ids[row * topk ..][0..j], e) == null) break e;
            },
        };
        wts[row * topk + j] = 0.05 + 0.45 * rng.float(f32);
    };
}

fn bitsOf(a: A) ![]const u16 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_bfloat16(c) orelse return error.NoData;
    return testing.allocator.dupe(u16, p[0..mlx.mlx_array_size(c)]);
}

/// One layer of the fixture at R = 1, 2, 4, 6 and every routing: the kernel against today's path and f32 truth, and each
/// row of a launch equal to its own one-row launch.
fn equivalence(lo: usize, hi: usize, li: usize, bits: u32) !void {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = lo, .hi = hi, .experts = 16, .ctx = 0 });
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(li);
    try testing.expectEqual(bits, ly.experts.gu.?.qp.bits);
    const kern = try build(m.cfg.dsv4_swiglu_limit);
    defer kern.deinit();
    var prng = std.Random.DefaultPrng.init(0x4d0e + bits);
    const topk = ly.top_k;
    const k = m.cfg.hidden_size;
    for ([_]usize{ 1, 2, 4, 6 }) |rows| for ([_]Routing{ .random, .shared_all, .disjoint }) |how| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const h2 = try sc.keep(try fx.normal(&.{ ci(rows), ci(k) }, .bfloat16, 0, 1));
        var ids: [64]u32 = undefined;
        var ws: [64]f32 = undefined;
        route(prng.random(), how, rows, topk, 16, &ids, &ws);
        const idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ ci(rows), ci(topk) }, 2, .uint32));
        const wts = try sc.keep(mlx.mlx_array_new_data(&ws, &[_]c_int{ ci(rows), ci(topk) }, 2, .float32));
        const got = (try moeUp(&sc, m, kern, ly, h2, idx, wts)).?;
        const want = try todays(&sc, m, ly, h2, idx, wts);
        const tr = try truth(gpa, &sc, m, ly, h2, ids[0 .. rows * topk], ws[0 .. rows * topk]);
        defer gpa.free(tr);
        const g = try dsv41.hostF32(got);
        defer gpa.free(g);
        const w = try dsv41.hostF32(want);
        defer gpa.free(w);
        var label: [64]u8 = undefined;
        try decode.expectNoWorse(try std.fmt.bufPrint(&label, "moe_up {d}-bit R={d} {s}", .{ bits, rows, @tagName(how) }), g, w, tr);
        const gb = try bitsOf(got);
        defer testing.allocator.free(gb);
        const per = (topk + 1) * @as(usize, @intCast(mlx.getShape(got)[2]));
        for (0..rows) |row| {
            var rs = dsv41.Scope.init(gpa, mlx.gpuStream());
            defer rs.deinit();
            const one = (try moeUp(&rs, m, kern, ly, try sliceRow(&rs, h2, row), try sliceRow(&rs, idx, row), try sliceRow(&rs, wts, row))).?;
            const ob = try bitsOf(one);
            defer testing.allocator.free(ob);
            try testing.expectEqualSlices(u16, ob, gb[row * per ..][0..per]);
        }
    };
}

test "dsv41 dec moe_up: 2-bit routed experts, rows 1 2 4 6, against today's path and f32 truth" {
    try equivalence(20, 25, 20, 2);
}

test "dsv41 dec moe_up: 3-bit routed experts, rows 1 2 4 6, against today's path and f32 truth" {
    try equivalence(2, 4, 2, 3);
}

test "dsv41 dec moe_up: the declared launch is the wrapper's, byte for byte" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    inline for (.{ 2, 3 }) |bits| for ([_]Shape{ .{ .rows = 1, .n = 24, .k = 2048 }, .{ .rows = 4, .n = 9, .k = 1024 } }) |shape| {
        var c = kit.Case(MoeUp).init(a, &moe_up_l, .{ .bits = bits });
        defer c.deinit();
        try testing.expect(try c.prepare(shape, .random, 7));
        try c.run();
        const mine = try a.dupe(u8, c.rig.outputBytes());
        defer a.free(mine);
        var sc = dsv41.Scope.init(a, mlx.gpuStream());
        defer sc.deinit();
        const kern = try build(10);
        defer _ = mlx.mlx_fast_metal_kernel_free(kern.k);
        defer _ = mlx.mlx_array_free(kern.lim);
        const theirs = try run(&sc, .{ .k = kern.k, .lim = c.args[8] }, bits, c.args[0], c.args[1], c.args[2], c.args[3], c.args[4], c.args[5], c.args[6], c.args[7]);
        try testing.expectEqualSlices(u8, try c.rig.bytesOf(&.{theirs}), mine);
    };
}

test "dsv41 dec moe_up: formats and shapes it does not serve are declined" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 16, .ctx = 0 });
    defer fx.deinit();
    const m = fx.m;
    const kern = try build(m.cfg.dsv4_swiglu_limit);
    defer kern.deinit();
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const k = m.cfg.hidden_size;
    var ids: [42]u32 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(i % 6);
    const ws: [42]f32 = @splat(0.1);
    var served: ?A = null;
    for ([_]usize{ 2, 7 }) |rows| {
        const h2 = try sc.keep(try fx.normal(&.{ ci(rows), ci(k) }, .bfloat16, 0, 1));
        const idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ ci(rows), 6 }, 2, .uint32));
        const wts = try sc.keep(mlx.mlx_array_new_data(&ws, &[_]c_int{ ci(rows), 6 }, 2, .float32));
        var ly = fx.layer(20).*;
        const out = try moeUp(&sc, m, kern, &ly, h2, idx, wts);
        if (rows == 2) served = out else try testing.expect(out == null); // more rows than decode runs
        if (rows != 2) continue;
        ly.experts.gu.?.qp.mode = .mxfp4;
        try testing.expect(try moeUp(&sc, m, kern, &ly, h2, idx, wts) == null);
        ly = fx.layer(20).*;
        ly.experts.gu.?.qp.group_size = 32;
        try testing.expect(try moeUp(&sc, m, kern, &ly, h2, idx, wts) == null);
        ly = fx.layer(20).*;
        ly.shared_gu.?.qp.mode = .affine;
        try testing.expect(try moeUp(&sc, m, kern, &ly, h2, idx, wts) == null);
    }
    try testing.expect(served != null); // the unchanged layer is served: the declines above are the changes
}

test "dsv41 dec moe_up: the kernel contract's gates 1-4 over both widths" {
    // `mlx.virtualGpu()` reads the architecture string after freeing its owner, so it answers true on real GPUs here.
    var arch: [128]u8 = undefined;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !std.mem.startsWith(u8, device.gpuArchitecture(&arch) orelse "", "applegpu")) return error.SkipZigTest;
    var set = kc.Set.init(testing.allocator);
    defer set.deinit();
    try register(&set);
    var tally: kc.gates.Tally = .{};
    try kc.gates.runAll(testing.allocator, set.kernels.items, 0xc0de, &tally);
    try testing.expectEqual(@as(usize, 0), tally.problems());
}

// ── the ubench ──

/// Shared-expert copies the ubench rotates through, so no read comes from the system cache.
const bench_copies = 12;
const bench_steps = 60;

fn randomBits(shape: []const c_int, width: c_int, seed: u64) !A {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_bits(&o, shape.ptr, shape.len, width, key, mlx.gpuStream()));
    return o;
}

fn randomNormal(shape: []const c_int, std_: f32, seed: u64) !A {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&o, shape.ptr, shape.len, act, 0, std_, key, mlx.gpuStream()));
    return o;
}

/// Expert 0 of a `[E, rows, cols]` bank as a view: the kernels address the others from its buffer pointer.
fn firstExpert(a: A) !A {
    const sh = mlx.getShape(a);
    const start: [3]c_int = .{ 0, 0, 0 };
    const stop: [3]c_int = .{ 1, sh[1], sh[2] };
    const step: [3]c_int = .{ 1, 1, 1 };
    var v = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&v, a, &start, 3, &stop, 3, &step, 3, mlx.gpuStream()));
    return v;
}

fn evalAll(xs: []const A) !void {
    const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
}

/// The release's layer at full width over random bytes: a 384-expert gate+up bank at `bits` (2.8 GB at 2 bits), the
/// shared expert in rotating copies, and per step a routing of distinct experts in a row (rows share some with `overlap`).
const Bench = struct {
    gpa: std.mem.Allocator,
    bits: u32,
    rows: usize,
    m: dsv41.Dsv41Model = undefined,
    ly: [bench_copies]dsv41.Layer = undefined,
    kern: Kernel = undefined,
    chain: mlx.mlx_closure = .{ .ctx = null },
    x0: A = none,
    ids: [bench_steps]A = undefined,
    wts: [bench_steps]A = undefined,
    held: std.ArrayList(A) = .empty,
    distinct: f64 = 0,
    dim_k: usize = 5120,
    dim_n: usize = 2304,
    experts: usize = 384,

    fn init(gpa: std.mem.Allocator, bits: u32, rows: usize, overlap: bool) !*Bench {
        const b = try gpa.create(Bench);
        b.* = .{ .gpa = gpa, .bits = bits, .rows = rows };
        const K: c_int = @intCast(b.dim_k);
        const N: c_int = @intCast(b.dim_n);
        const E: c_int = @intCast(b.experts);
        b.m.cfg = .{};
        b.m.cfg.dsv4_swiglu_limit = 10;
        b.m.act = act;
        b.m.moe_gu_kernel = try dsv41.buildKernel("dsv41_moe_gu", &dsv41.MOE_GU_INS, &.{"out"}, dsv41.MOE_GU_SOURCE, dsv41.MOE_HEADER);
        b.kern = try build(10);
        const w = try randomBits(&.{ E, 2 * N, @intCast(b.dim_k * bits / 32) }, 4, 1);
        const sc = try randomNormal(&.{ E, 2 * N, @divExact(K, 64) }, 0.01, 2);
        const bi = try randomNormal(&.{ E, 2 * N, @divExact(K, 64) }, 0.01, 3);
        try evalAll(&.{ w, sc, bi });
        const bank: dsv41.Bank = .{ .w = w, .s = sc, .b = bi, .qp = .{ .bits = bits, .group_size = 64 } };
        const views: [2]dsv41.Bank = .{ .{ .w = try firstExpert(w), .s = try firstExpert(sc), .b = try firstExpert(bi), .qp = bank.qp }, .{ .w = none } };
        for (&b.ly, 0..) |*ly, i| {
            // e4m3 bytes without the NaN code (bit 6 clear), e8m0 scales near 1.
            const raw = try randomBits(&.{ 2 * N, @divExact(K, 4) }, 4, 100 + i);
            const mask = mlx.mlx_array_new_data(&[_]u32{0xBFBFBFBF}, &[_]c_int{1}, 1, .uint32);
            defer _ = mlx.mlx_array_free(mask);
            var sw = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_bitwise_and(&sw, raw, mask, mlx.gpuStream()));
            const eb = try randomBits(&.{ 2 * N, @divExact(K, 32) }, 1, 200 + i);
            const eight = mlx.mlx_array_new_data(&[_]u8{8}, &[_]c_int{1}, 1, .uint8);
            defer _ = mlx.mlx_array_free(eight);
            const base = mlx.mlx_array_new_data(&[_]u8{123}, &[_]c_int{1}, 1, .uint8);
            defer _ = mlx.mlx_array_free(base);
            var rem = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_remainder(&rem, eb, eight, mlx.gpuStream()));
            var ss = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&ss, rem, base, mlx.gpuStream()));
            try evalAll(&.{ sw, ss });
            for ([_]A{ raw, eb, rem }) |t| _ = mlx.mlx_array_free(t);
            try b.held.appendSlice(gpa, &.{ sw, ss });
            ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .down = .{ .w = none }, .gu = bank, .views = views };
            ly.shared_gu = .{ .w = sw, .s = ss, .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } };
        }
        try b.held.appendSlice(gpa, &.{ w, sc, bi, views[0].w, views[0].s, views[0].b });
        b.x0 = try randomNormal(&.{ @intCast(rows), K }, 1, 7);
        try mlx.check(mlx.mlx_array_eval(b.x0));
        var prng = std.Random.DefaultPrng.init(42 + rows);
        const rnd = prng.random();
        var distinct: f64 = 0;
        for (&b.ids, &b.wts) |*idx, *wt| {
            var ids: [64]u32 = undefined;
            var ws: [64]f32 = undefined;
            for (0..rows) |row| for (0..TOPK) |j| {
                // With `overlap`, a slot of a later row takes (one time in six) an expert an earlier row picked.
                ids[row * TOPK + j] = while (true) {
                    const e = if (overlap and row > 0 and rnd.uintLessThan(u32, 6) == 0) ids[rnd.uintLessThan(usize, row * TOPK)] else rnd.uintLessThan(u32, @intCast(b.experts));
                    if (std.mem.indexOfScalar(u32, ids[row * TOPK ..][0..j], e) == null) break e;
                };
                ws[row * TOPK + j] = 0.05 + 0.3 * rnd.float(f32);
            };
            var seen: [64]u32 = undefined;
            var ns: usize = 0;
            for (ids[0 .. rows * TOPK]) |e| if (std.mem.indexOfScalar(u32, seen[0..ns], e) == null) {
                seen[ns] = e;
                ns += 1;
            };
            distinct += @floatFromInt(ns);
            idx.* = mlx.mlx_array_new_data(&ids, &[_]c_int{ @intCast(rows), TOPK }, 2, .uint32);
            wt.* = mlx.mlx_array_new_data(&ws, &[_]c_int{ @intCast(rows), TOPK }, 2, .float32);
        }
        b.distinct = distinct / bench_steps;
        if (rows > 1) b.chain = try b.compileChain();
        return b;
    }

    fn deinit(b: *Bench) void {
        b.kern.deinit();
        _ = mlx.mlx_fast_metal_kernel_free(b.m.moe_gu_kernel);
        if (b.chain.ctx != null) _ = mlx.mlx_closure_free(b.chain);
        for (b.held.items) |a| _ = mlx.mlx_array_free(a);
        b.held.deinit(b.gpa);
        for (b.ids, b.wts) |i, w| {
            _ = mlx.mlx_array_free(i);
            _ = mlx.mlx_array_free(w);
        }
        _ = mlx.mlx_array_free(b.x0);
        b.gpa.destroy(b);
    }

    /// Bytes the weights of one call read: the picked experts' rows (gate+up, scales, biases) and the shared bank.
    fn bytes(b: *const Bench) f64 {
        const per_row: f64 = @floatFromInt(b.dim_k * b.bits / 8 + 2 * b.dim_k / 64 * 2);
        const expert = per_row * @as(f64, @floatFromInt(2 * b.dim_n));
        return b.distinct * expert + @as(f64, @floatFromInt(2 * b.dim_n * (b.dim_k + b.dim_k / 32)));
    }

    /// Today's rows > 1 path as the compiled FFN region runs it: `todays` over (x, idx, wts, shared w, shared scales).
    fn compileChain(b: *Bench) !mlx.mlx_closure {
        const raw = mlx.mlx_closure_new_func_payload(&chainBody, b, null);
        defer _ = mlx.mlx_closure_free(raw);
        var out = mlx.mlx_closure{ .ctx = null };
        try mlx.check(mlx.mlx_compile(&out, raw, false));
        return out;
    }

    fn chainBody(res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const b: *Bench = @ptrCast(@alignCast(payload.?));
        var a: [5]A = undefined;
        for (&a, 0..) |*x, i| {
            x.* = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(x, in, i) != 0) return -1;
        }
        defer for (a) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var sc = dsv41.Scope.init(std.heap.page_allocator, mlx.gpuStream());
        defer sc.deinit();
        var ly = b.ly[0];
        ly.shared_gu = .{ .w = a[3], .s = a[4], .qp = ly.shared_gu.?.qp };
        const out = todays(&sc, &b.m, &ly, a[0], a[1], a[2]) catch return -1;
        res.* = mlx.mlx_vector_array_new_data(&.{out}, 1);
        return 0;
    }

    /// Step i's input: the fixed x plus the previous output's first K values (one small elementwise dispatch, which
    /// every arm pays and `DepCtx` measures).
    fn depX(b: *Bench, sc: *dsv41.Scope, prev: A) !A {
        var o = mlx.mlx_array_new();
        const flat = try reshape(sc, prev, &.{ @intCast(b.rows), -1 });
        return own(sc, mlx.mlx_add(&o, b.x0, try sliceLast(sc, flat, 0, b.dim_k), sc.s), o);
    }
};

const NewCtx = struct {
    b: *Bench,
    sc: *dsv41.Scope,
    prev: A,

    fn step(c: *NewCtx, i: usize) !A {
        const b = c.b;
        const ly = &b.ly[i % bench_copies];
        const v = ly.experts.views.?[0];
        const sgu = ly.shared_gu.?;
        const x = try b.depX(c.sc, c.prev);
        c.prev = try run(c.sc, b.kern, b.bits, x, v.w, v.s, v.b, sgu.w, sgu.s, b.ids[i % bench_steps], b.wts[i % bench_steps]);
        return c.prev;
    }
};

const OldCtx = struct {
    b: *Bench,
    sc: *dsv41.Scope,
    prev: A,

    fn step(c: *OldCtx, i: usize) !A {
        const b = c.b;
        const ly = &b.ly[i % bench_copies];
        const x = try b.depX(c.sc, c.prev);
        if (b.rows == 1) {
            c.prev = try todays(c.sc, &b.m, ly, x, b.ids[i % bench_steps], b.wts[i % bench_steps]);
        } else {
            const sgu = ly.shared_gu.?;
            const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, b.ids[i % bench_steps], b.wts[i % bench_steps], sgu.w, sgu.s }, 5);
            defer _ = mlx.mlx_vector_array_free(iv);
            var ov = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ov);
            try mlx.check(mlx.mlx_closure_apply(&ov, b.chain, iv));
            var o = mlx.mlx_array_new();
            c.prev = try own(c.sc, mlx.mlx_vector_array_get(&o, ov, 0), o);
        }
        return c.prev;
    }
};

/// Only the dependency op: the cost every arm's chain carries.
const DepCtx = struct {
    b: *Bench,
    sc: *dsv41.Scope,
    prev: A,

    fn step(c: *DepCtx, i: usize) !A {
        _ = i;
        c.prev = try c.b.depX(c.sc, try reshape(c.sc, c.prev, &.{ @intCast(c.b.rows), -1 }));
        return c.prev;
    }
};

fn nowUs() i64 {
    return @import("../../../sdk/util/io_util.zig").nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

/// Median microseconds a step of an `n`-step chain takes on the device, the graph built before the clock starts.
fn deviceUs(ctx: anytype, n: usize) !f64 {
    var times: [7]f64 = undefined;
    for (0..2 + times.len) |rep| {
        var last: A = none;
        for (0..n) |i| last = try ctx.step(i);
        const t0 = nowUs();
        try mlx.check(mlx.mlx_array_eval(last));
        const us = @as(f64, @floatFromInt(nowUs() - t0)) / @as(f64, @floatFromInt(n));
        if (rep >= 2) times[rep - 2] = us;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[times.len / 2];
}

/// Milliseconds the first evaluation of a chain's step takes: MLX compiles the pipeline of its (R, bits) on it.
fn firstCallMs(ctx: anytype) !f64 {
    const last = try ctx.step(0);
    const t0 = nowUs();
    try mlx.check(mlx.mlx_array_eval(last));
    return @as(f64, @floatFromInt(nowUs() - t0)) / 1000;
}

test "dsv41 dec moe_up: ubench, R = 1 and 6 against today's chain (DSV41_DEC_UBENCH)" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    for ([_]u32{ 2, 3 }) |bits| for ([_][2]usize{ .{ 1, 0 }, .{ 6, 0 }, .{ 6, 1 } }) |case| {
        const rows = case[0];
        const b = try Bench.init(gpa, bits, rows, case[1] == 1);
        defer b.deinit();
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        var dep: DepCtx = .{ .b = b, .sc = &sc, .prev = b.x0 };
        var new: NewCtx = .{ .b = b, .sc = &sc, .prev = b.x0 };
        var old: OldCtx = .{ .b = b, .sc = &sc, .prev = b.x0 };
        const compile_ms = try firstCallMs(&new);
        const dep_us = try deviceUs(&dep, bench_steps);
        var new_us: f64 = std.math.inf(f64);
        var old_us: f64 = std.math.inf(f64);
        // Alternate the arms so a slow minute of the machine lands on both.
        for (0..3) |_| {
            new_us = @min(new_us, try deviceUs(&new, bench_steps));
            old_us = @min(old_us, try deviceUs(&old, bench_steps));
        }
        std.debug.print("[dec-ubench] moe_up {d}-bit g64, {s} routing (mean {d:.1} distinct experts, dependency op {d:.1} us taken off both, first call {d:.1} ms):\n", .{ bits, if (case[1] == 1) "overlapping" else "distinct", b.distinct, dep_us, compile_ms });
        decode.report("moe_up", rows, new_us - dep_us, b.bytes(), old_us - dep_us);
    };
}

