//! The MoE for decode rows (R = 1..6) over mxfp4 routed experts, in two launches: gate/up with the clamped SwiGLU and the
//! shared expert's, then the down projections, the combine and the hc expand. `moe_up.zig` and `moe_down.zig` serve the
//! affine 2- and 3-bit packs; this serves the pack that keeps the release's experts as shipped.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const moe_up = @import("moe_up.zig");
const params = @import("../../../sdk/quant/params.zig");

const A = mlx.mlx_array;

pub const UP_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_moe_fp4_up.metal");
pub const DOWN_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_moe_fp4_down.metal");
pub const HEADER = moe_up.HEADER ++ kernelSource("plugins/deepseek_v41/kernels/dec_moe_fp4_header.metal");
pub const UP_INS = [_][*:0]const u8{ "x", "w", "scales", "sw", "sscales", "inds", "wts", "lim" };
pub const DOWN_INS = [_][*:0]const u8{ "x", "w", "scales", "sw", "sscales", "inds", "res", "post", "comb" };

/// Threads a threadgroup of the gate/up launch: two simdgroups.
const up_threads = 64;

/// The pipelines and the swiglu clamp they read (`dsv4_swiglu_limit`).
pub const Kernels = struct {
    up: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    lim: A,

    pub fn deinit(self: Kernels) void {
        _ = mlx.mlx_fast_metal_kernel_free(self.up);
        _ = mlx.mlx_fast_metal_kernel_free(self.down);
        _ = mlx.mlx_array_free(self.lim);
    }
};

pub fn build(limit: f32) !Kernels {
    const up = try dsv41.buildKernel("dsv41_dec_moe_fp4_up", &UP_INS, &.{"out"}, UP_SOURCE, HEADER);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(up);
    const down = try dsv41.buildKernel("dsv41_dec_moe_fp4_down", &DOWN_INS, &.{"out"}, DOWN_SOURCE, HEADER);
    return .{ .up = up, .down = down, .lim = mlx.mlx_array_new_data(&[_]f32{limit}, &[_]c_int{1}, 1, .float32) };
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dims(a: A, axis: usize) usize {
    return @intCast(mlx.getShape(a)[axis]);
}

/// The gate/up half of `h2`'s MoE (`h2` `[R, K]`, `idx` `[R, k]` u32, `wts` `[R, k]` f32; R <= `decode.max_rows`) as
/// `[R, k + 1, N]`: slot j < k the clamped SwiGLU of row r's j-th expert times its weight, slot k the shared expert's.
/// Null where the layer's formats are not the served ones (routed mxfp4 group 32 with no biases, shared mxfp8, a bf16
/// stream) or the shape is not.
pub fn moeUp(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, kern: Kernels, ly: *const dsv41.Layer, h2: A, idx: A, wts: A) !?A {
    const gu = ly.experts.gu orelse return null;
    const sgu = ly.shared_gu orelse return null;
    if (!mlx.streamIsGpu(sc.s) or gu.s.ctx == null or gu.b.ctx != null or sgu.s.ctx == null) return null;
    if (mlx.getShape(h2).len != 2 or mlx.getShape(idx).len != 2 or mlx.getShape(gu.w).len != 3) return null;
    const rows = dims(h2, 0);
    const k = dims(h2, 1);
    const topk = dims(idx, 1);
    const n2 = dims(gu.w, 1);
    if (rows < 1 or rows > decode.max_rows or topk < 1 or topk > 16 or k == 0 or k % 1024 != 0 or n2 % 2 != 0) return null;
    if (dims(idx, 0) != rows or !std.mem.eql(c_int, mlx.getShape(idx), mlx.getShape(wts))) return null;
    if (gu.qp.mode != .mxfp4 or gu.qp.bits != 4 or gu.qp.group_size != 32 or dims(gu.w, 2) != k / 8 or dims(gu.s, 1) != n2 or dims(gu.s, 2) != k / 32) return null;
    if (sgu.qp.mode != .mxfp8 or sgu.qp.bits != 8 or sgu.qp.group_size != 32) return null;
    const n = n2 / 2;
    if (mlx.mlx_array_dtype(h2) != .bfloat16 or h2.ctx == null or m.act != .bfloat16) return null;
    if (mlx.mlx_array_dtype(gu.w) != .uint32 or mlx.mlx_array_dtype(gu.s) != .uint8) return null;
    if (mlx.mlx_array_dtype(idx) != .uint32 or mlx.mlx_array_dtype(wts) != .float32) return null;
    const sw = mlx.getShape(sgu.w);
    const ss = mlx.getShape(sgu.s);
    if (sw.len != 2 or ss.len != 2 or sw[0] != ci(n2) or sw[1] != ci(k / 4) or ss[0] != ci(n2) or ss[1] != ci(k / 32)) return null;
    if (mlx.mlx_array_dtype(sgu.w) != .uint32 or mlx.mlx_array_dtype(sgu.s) != .uint8) return null;
    const vgu = if (ly.experts.views) |v| v[0] else gu;
    return try runUp(sc, kern, rows, k, n, topk, h2, vgu.w, vgu.s, sgu.w, sgu.s, idx, wts);
}

/// The gate/up launch over arrays already checked.
pub fn runUp(sc: *dsv41.Scope, kern: Kernels, rows: usize, k: usize, n: usize, topk: usize, x: A, w: A, scales: A, sw: A, sscales: A, idx: A, wts: A) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(topk + 1), ci(n) }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(32 * n * (topk * rows + 1)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, up_threads, 1, 1));
    inline for (.{ .{ "R", rows }, .{ "K", k }, .{ "N", n }, .{ "TOPK", topk } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const ins = [_]A{ x, w, scales, sw, sscales, idx, wts, kern.lim };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern.up, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// Output columns a down threadgroup takes: the fewest whose lanes split a row's K / 32 units evenly (32 / NC lanes a
/// column), else 8; null where N or the combine's lanes (NC x hc) do not fit.
pub fn downCols(k: usize, n: usize, hc: usize) ?usize {
    const units = k / 32;
    const nc: usize = for ([_]usize{ 1, 2, 4, 8 }) |c| {
        if (units % (32 / c) == 0) break c;
    } else 8;
    if (n % nc != 0 or nc * hc > 32) return null;
    return nc;
}

/// x' `[R, hc, N]` = `hcExpand(x1, y, sp)` for y = the `idx` `[R, k]` experts' down rows plus the shared expert's over
/// `act` `[R, k + 1, K]` (what `moeUp` makes). Null where the kernel does not serve the call: the down bank is not
/// mxfp4 group 32, the shared is not mxfp8, the stream is not bf16, or K is not a multiple of 32.
pub fn moeDown(sc: *dsv41.Scope, kern: Kernels, ly: *const dsv41.Layer, act: A, idx: A, x1: A, sp: *const dsv41.Split) !?A {
    if (!mlx.streamIsGpu(sc.s)) return null;
    const down = if (ly.experts.views) |v| &v[1] else &ly.experts.down;
    const shared = &ly.shared[2];
    if (mlx.getShape(act).len != 3 or mlx.getShape(x1).len != 3 or mlx.getShape(idx).len != 2 or down.w.ctx == null or shared.w.ctx == null) return null;
    const R = dims(act, 0);
    const K = dims(act, 2);
    const N = dims(x1, 2);
    const hc = dims(x1, 1);
    const topk = dims(idx, 1);
    if (R < 1 or R > decode.max_rows or dims(act, 1) != topk + 1 or dims(x1, 0) != R or dims(idx, 0) != R or topk < 1 or topk > 16 or hc < 1 or hc > 32) return null;
    if (K == 0 or K % 32 != 0 or sp.post.ctx == null or sp.comb.ctx == null) return null;
    if (down.qp.mode != .mxfp4 or down.qp.bits != 4 or down.qp.group_size != 32 or down.s.ctx == null or down.b.ctx != null) return null;
    if (shared.qp.mode != .mxfp8 or shared.qp.bits != 8 or shared.qp.group_size != 32 or shared.s.ctx == null) return null;
    if (mlx.mlx_array_dtype(act) != .bfloat16 or mlx.mlx_array_dtype(x1) != .bfloat16 or mlx.mlx_array_dtype(idx) != .uint32) return null;
    if (mlx.mlx_array_dtype(down.w) != .uint32 or mlx.mlx_array_dtype(down.s) != .uint8 or mlx.mlx_array_dtype(shared.w) != .uint32 or mlx.mlx_array_dtype(shared.s) != .uint8) return null;
    if (mlx.mlx_array_dtype(sp.post) != .float32 or mlx.mlx_array_dtype(sp.comb) != .float32 or mlx.mlx_array_size(sp.post) != R * hc or mlx.mlx_array_size(sp.comb) != R * hc * hc) return null;
    if (mlx.getShape(down.w).len != 3 or mlx.getShape(down.s).len != 3 or dims(down.w, 1) != N or dims(down.w, 2) != K / 8 or dims(down.s, 1) != N or dims(down.s, 2) != K / 32) return null;
    if (mlx.getShape(shared.w).len != 2 or mlx.getShape(shared.s).len != 2 or dims(shared.w, 0) != N or dims(shared.w, 1) != K / 4 or dims(shared.s, 0) != N or dims(shared.s, 1) != K / 32) return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ ci(R), ci(hc), ci(N) };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, shape.len, .bfloat16));
    const nc = downCols(K, N, hc) orelse return null;
    const tg = 32 * (topk + 1);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(N / nc * tg), ci(R), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
    inline for (.{ .{ "K", K }, .{ "N", N }, .{ "TOPK", topk }, .{ "HC", hc }, .{ "NC", nc } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const ins = [_]A{ act, down.w, down.s, shared.w, shared.s, idx, x1, sp.post, sp.comb };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern.down, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return try sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

// ── the kernel contract ──

const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const cv = @import("../contract.zig");

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;

const adt: mlx.mlx_dtype = .bfloat16;
const TOPK = 6;
const HC = 4;
/// fp32's unit roundoff.
const u = 0x1p-24;

fn bf(v: f64) f64 {
    return spec.roundTo(.bf16, v);
}

fn depth(s: Shape) usize {
    return s.k;
}

const up_l = cv.launch("dsv41_dec_moe_fp4_up", &UP_INS, &.{"out"}, UP_SOURCE, HEADER, 0, up_threads, false);
const down_l = cv.launch("dsv41_dec_moe_fp4_down", &DOWN_INS, &.{"out"}, DOWN_SOURCE, HEADER, 4 * (TOPK + 1), 32 * (TOPK + 1), false);

/// e2m1 magnitudes by code bits 0..2; bit 3 is the sign.
const e2m1 = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

/// The code of the integer magnitude `c` (1, 2, 3, 4 or 6: the e2m1 values that are integers).
fn intCode(c: u32) u32 {
    return switch (c) {
        1 => 2,
        2 => 4,
        3 => 5,
        4 => 6,
        else => 7,
    };
}

/// An mxfp4 operand of rows `lead..` x K as the kernel binds it: words (eight 4-bit codes, the first in the low nibble)
/// and e8m0 scales, both held as integer-valued float64.
const Fp4 = struct {
    w: []f64,
    s: []f64,
    k: usize,

    fn add(c: anytype, lead: []const usize, K: usize) !Fp4 {
        var wd: [3]usize = undefined;
        @memcpy(wd[0..lead.len], lead);
        var sd = wd;
        wd[lead.len] = K / 8;
        sd[lead.len] = K / 32;
        return .{ .w = try c.add(.uint32, wd[0 .. lead.len + 1]), .s = try c.add(.uint8, sd[0 .. lead.len + 1]), .k = K };
    }

    fn rows(self: Fp4) usize {
        return self.s.len / (self.k / 32);
    }

    fn code(self: Fp4, row: usize, i: usize) u32 {
        const word: u32 = @intFromFloat(self.w[row * (self.k / 8) + i / 8]);
        return (word >> @intCast(4 * (i % 8))) & 15;
    }

    /// Row `row`'s codes written from `codes`.
    fn pack(self: Fp4, row: usize, codes: []const u32) void {
        for (0..self.k / 8) |j| {
            var word: u32 = 0;
            for (0..8) |b| word |= codes[8 * j + b] << @intCast(4 * b);
            self.w[row * (self.k / 8) + j] = @floatFromInt(word);
        }
    }

    fn weight(self: Fp4, row: usize, i: usize) f64 {
        const c = self.code(row, i);
        const v = e2m1[c & 7];
        return (if (c & 8 != 0) -v else v) * std.math.pow(f64, 2, self.s[row * (self.k / 32) + i / 32] - 127);
    }

    /// Random codes under scales near 2^-6.
    fn random(self: Fp4, rng: std.Random, scratch: std.mem.Allocator) !void {
        const codes = try scratch.alloc(u32, self.k);
        defer scratch.free(codes);
        for (0..self.rows()) |r| {
            for (codes) |*c| c.* = rng.uintLessThan(u32, 16);
            self.pack(r, codes);
        }
        for (self.s) |*e| e.* = @floatFromInt(119 + rng.uintLessThan(u32, 5));
    }

    /// Every row the same mirrored halves: integer magnitudes on the first half (1 on the first 8 entries, up to 6
    /// elsewhere), their negatives on the shuffled second (`perm`), unit scales.
    fn mirrored(self: Fp4, rng: std.Random, perm: []const usize, scratch: std.mem.Allocator) !void {
        const codes = try scratch.alloc(u32, self.k);
        defer scratch.free(codes);
        const h = self.k / 2;
        const mags = [_]u32{ 1, 2, 3, 4, 6 };
        for (0..self.rows()) |r| {
            for (0..h) |i| {
                const c = if (i < 8) intCode(1) else intCode(mags[rng.uintLessThan(usize, mags.len)]);
                codes[i] = c;
                codes[h + perm[i]] = c | 8;
            }
            self.pack(r, codes);
        }
        @memset(self.s, 127);
    }

    /// Row `r` against `x` [K]: the sum (`sum`'s accumulator) and its terms' magnitude.
    fn dot(self: Fp4, r: usize, x: []const f64, sum: Sum, t: []f64) [2]f64 {
        var mag: f64 = 0;
        for (0..self.k) |i| {
            const term = x[i] * self.weight(r, i);
            t[i] = term;
            mag += @abs(term);
        }
        return .{ sum.of(t[0..self.k]), mag };
    }
};

/// Integers 0..8 as e4m3 codes (all exact).
const int_codes = [_]u8{ 0x00, 0x38, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54, 0x58 };

fn packBytes(words: []f64, bytes: []const u8) void {
    for (words, 0..) |*w, i| w.* = @floatFromInt(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
}

/// Row `r` input i of an mxfp8 operand (`words` of e4m3 bytes, e8m0 `exps`) as float64.
fn sharedAt(words: []const f64, exps: []const f64, r: usize, i: usize, k: usize) f64 {
    const word: u32 = @intFromFloat(words[r * (k / 4) + i / 4]);
    const byte: u8 = @truncate(word >> @intCast(8 * (i % 4)));
    return kc.host.e4m3(byte) * std.math.pow(f64, 2, exps[r * (k / 32) + i / 32] - 127);
}

/// Fills an mxfp8 operand `[rows, k]` with random bytes (the NaN code excluded) or, for `.stress`, mirrored integers.
fn fillShared(rng: std.Random, words: []f64, exps: []f64, rows: usize, K: usize, f: Fill, perm: []const usize, scratch: std.mem.Allocator) !void {
    const bytes = try scratch.alloc(u8, rows * K);
    defer scratch.free(bytes);
    switch (f) {
        .random, .nonfinite => {
            // e4m3 codes of exponent 5..9 (never the NaN code) under scales near 2^-6: weights near 0.02.
            for (bytes) |*b| b.* = @as(u8, @intCast(rng.uintLessThan(u32, 2))) << 7 | @as(u8, @intCast(5 + rng.uintLessThan(u32, 5))) << 3 | @as(u8, @intCast(rng.uintLessThan(u32, 8)));
            for (exps) |*e| e.* = @floatFromInt(120 + rng.uintLessThan(u32, 3));
        },
        .stress => {
            for (0..rows) |r| {
                const h = K / 2;
                for (0..h) |i| {
                    const c: u8 = if (i < 8) 1 else @intCast(rng.intRangeAtMost(u32, 1, 8));
                    bytes[r * K + i] = int_codes[c];
                    bytes[r * K + h + perm[i]] = int_codes[c] | 0x80;
                }
            }
            @memset(exps, 127);
        },
    }
    packBytes(words, bytes);
}

/// The block's gate/up half: `rows` decode rows over `Params.experts` routed mxfp4 experts plus the shared mxfp8 expert.
const UpK = struct {
    pub const Params = struct { experts: usize = 8, lim: f64 = 10 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.rows > decode.max_rows or s.n == 0 or s.k == 0 or s.k % 1024 != 0 or p.experts < TOPK or p.experts > 64) return null;
        var g = metal.Geometry{ .grid = .{ ci(32 * s.n * (TOPK * s.rows + 1)), 1, 1 }, .tg = .{ up_threads, 1, 1 } };
        g.out(&.{ ci(s.rows), TOPK + 1, ci(s.n) }, adt);
        g.int("R", s.rows);
        g.int("K", s.k);
        g.int("N", s.n);
        g.int("TOPK", TOPK);
        g.dtype("T", adt);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const R = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const x = try k.add(adt, &.{ R, K });
        const routed = try Fp4.add(k, &.{ p.experts, 2 * N }, K);
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
        const perm = try k.scratch().alloc(usize, K / 2);
        defer k.scratch().free(perm);
        switch (f) {
            .random, .nonfinite => {
                try routed.random(rng, k.scratch());
                try fillShared(rng, sw, ss, 2 * N, K, f, perm, k.scratch());
                if (f == .nonfinite) kit.plantRows(x, R, K);
            },
            .stress => {
                cv.mirroredX(rng, x[0..K], perm);
                for (1..R) |row| mirrorRow(rng, x[row * K ..][0..K], perm);
                try routed.mirrored(rng, perm, k.scratch());
                try fillShared(rng, sw, ss, 2 * N, K, f, perm, k.scratch());
            },
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
        const routed: Fp4 = .{ .w = k.in(1), .s = k.in(2), .k = K };
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        const lim = k.in(7)[0];
        for (0..R) |row| for (0..TOPK + 1) |j| for (0..N) |c| {
            const x = k.in(0)[row * K ..][0..K];
            var g: [2]f64 = undefined;
            var up: [2]f64 = undefined;
            var wt: f64 = 1;
            if (j < TOPK) {
                const e: usize = @intFromFloat(k.in(5)[row * TOPK + j]);
                g = routed.dot(e * 2 * N + c, x, sum, t);
                up = routed.dot(e * 2 * N + N + c, x, sum, t);
                wt = k.in(6)[row * TOPK + j];
            } else {
                g = sharedDot(k, c, x, sum, t);
                up = sharedDot(k, N + c, x, sum, t);
            }
            const gt = bf(g[0]);
            const ut = bf(up[0]);
            const gc = kit.minN(gt, lim);
            const uc = if (std.math.isNan(ut)) kit.nan else std.math.clamp(ut, -lim, lim);
            const sig = 1 / (1 + @exp(-gc));
            const o = (row * (TOPK + 1) + j) * N + c;
            out[o] = gc * sig * uc * wt;
            // Each sum's error (and its bf16 rounding) reaches y scaled by y's slope in it.
            const slope_g = @abs(sig * (1 + gc * (1 - sig)) * uc * wt);
            mag[o] = (g[1] + @abs(g[0]) * 0x1p-8 / spec.gamma(@floatFromInt(K), u)) * slope_g * 2 + (up[1] + @abs(up[0]) * 0x1p-8 / spec.gamma(@floatFromInt(K), u)) * @abs(gc * sig * wt) * 2;
        };
    }

    /// Row `r` of the shared bank against `x`: the sum and its terms' magnitude.
    fn sharedDot(k: anytype, r: usize, x: []const f64, sum: Sum, t: []f64) [2]f64 {
        var mag: f64 = 0;
        for (0..k.shape.k) |i| {
            const w = sharedAt(k.in(3), k.in(4), r, i, k.shape.k);
            t[i] = x[i] * w;
            mag += @abs(x[i] * w);
        }
        return .{ sum.of(t[0..k.shape.k]), mag };
    }
};

/// The block's down half: `rows` decode rows over `Params.experts` routed mxfp4 experts plus the shared mxfp8 expert, y
/// rounded to bf16 once, then the hc expand.
const DownK = struct {
    pub const Params = struct { experts: usize = 8 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows < 1 or s.rows > decode.max_rows or s.n == 0 or s.k == 0 or s.k % 32 != 0 or p.experts < TOPK or p.experts > 64) return null;
        const nc = downCols(s.k, s.n, HC) orelse return null;
        const tg = 32 * (TOPK + 1);
        var g = metal.Geometry{ .grid = .{ ci(s.n / nc * tg), ci(s.rows), 1 }, .tg = .{ tg, 1, 1 } };
        g.out(&.{ ci(s.rows), HC, ci(s.n) }, adt);
        g.int("K", s.k);
        g.int("N", s.n);
        g.int("TOPK", TOPK);
        g.int("HC", HC);
        g.int("NC", nc);
        g.dtype("T", adt);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const R = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const x = try k.add(adt, &.{ R, TOPK + 1, K });
        const routed = try Fp4.add(k, &.{ p.experts, N }, K);
        const sw = try k.add(.uint32, &.{ N, K / 4 });
        const ss = try k.add(.uint8, &.{ N, K / 32 });
        const inds = try k.add(.uint32, &.{ R, TOPK });
        const res = try k.add(adt, &.{ R, HC, N });
        const post = try k.add(.float32, &.{ R, HC });
        const comb = try k.add(.float32, &.{ R, HC, HC });
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, res, -3, 3);
        kit.uniform(rng, post, 0, 2);
        kit.uniform(rng, comb, 0, 0.5);
        // Distinct experts in a row; rows may share them.
        for (0..R) |r| {
            var used: [64]bool = @splat(false);
            for (inds[r * TOPK ..][0..TOPK]) |*v| {
                var e = rng.uintLessThan(usize, p.experts);
                while (used[e]) e = (e + 1) % p.experts;
                used[e] = true;
                v.* = @floatFromInt(e);
            }
        }
        const perm = try k.scratch().alloc(usize, K / 2);
        defer k.scratch().free(perm);
        switch (f) {
            .random, .nonfinite => {
                try routed.random(rng, k.scratch());
                try fillShared(rng, sw, ss, N, K, f, perm, k.scratch());
                if (f == .nonfinite) kit.plantRows(x, R * (TOPK + 1), K);
            },
            .stress => {
                // Every slot's activations the same mirrored halves; post 1, comb 0: the output is the summed dots, exact.
                cv.mirroredX(rng, x[0..K], perm);
                for (1..R * (TOPK + 1)) |i| @memcpy(x[i * K ..][0..K], x[0..K]);
                try routed.mirrored(rng, perm, k.scratch());
                try fillShared(rng, sw, ss, N, K, f, perm, k.scratch());
                @memset(post, 1);
                @memset(comb, 0);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * HC * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const R = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const routed: Fp4 = .{ .w = k.in(1), .s = k.in(2), .k = K };
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        // y's one bf16 rounding, as a sum error: gamma(K) of the terms' magnitude covers an eighth-bit of the sum.
        const flip = 0x1p-8 / spec.gamma(@floatFromInt(K), u);
        var parts: [TOPK + 1]f64 = undefined;
        for (0..R) |r| for (0..N) |d| {
            var m: f64 = 0;
            for (0..TOPK) |s| {
                const e: usize = @intFromFloat(k.in(5)[r * TOPK + s]);
                const q = routed.dot(e * N + d, k.in(0)[(r * (TOPK + 1) + s) * K ..][0..K], sum, t);
                parts[s] = q[0];
                m += q[1];
            }
            const xs = k.in(0)[(r * (TOPK + 1) + TOPK) * K ..][0..K];
            var sm: f64 = 0;
            for (0..K) |i| {
                const term = xs[i] * sharedAt(k.in(3), k.in(4), d, i, K);
                t[i] = term;
                sm += @abs(term);
            }
            parts[TOPK] = sum.of(t[0..K]);
            m += sm;
            const tot = sum.of(&parts);
            const y = bf(tot);
            m += (@abs(tot) + @abs(y)) * flip;
            for (0..HC) |i| {
                const po = k.in(7)[r * HC + i];
                var acc = po * y;
                var cm: f64 = 0;
                for (0..HC) |j| {
                    const term = k.in(8)[(r * HC + j) * HC + i] * k.in(6)[(r * HC + j) * N + d];
                    acc += term;
                    cm += @abs(term);
                }
                out[(r * HC + i) * N + d] = acc;
                mag[(r * HC + i) * N + d] = @abs(po) * m + cm;
            }
        };
    }
};

/// Served shapes cut the output width to a few columns: each runs the same code at its own index; the reduction length
/// and the rows (R = 1..6, over 8 experts so the rows share most of them) are the release's.
pub fn register(set: *kc.Set) !void {
    try kit.add(set, UpK, &up_l, .{}, .{
        .name = "dsv41 dec moe_fp4 up (mxfp4 routed + mxfp8 shared)",
        .accum = .f32,
        .depth = depth,
        .ops = 8,
        .mids = 1,
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 1024, .step = 1024, .min = 1024 } },
        .served = &.{ .{ .rows = 1, .n = 64, .k = 5120 }, .{ .rows = 3, .n = 17, .k = 5120 }, .{ .rows = decode.max_rows, .n = 64, .k = 5120 } },
        .sums = .{ .f16 = 30, .bf16 = 30, .why = "a wrong sum's activation tends to 0, so a miss tops out near 1 / the bound" },
    });
    try kit.add(set, DownK, &down_l, .{}, .{
        .name = "dsv41 dec moe_fp4 down (mxfp4 routed + mxfp8 shared, hc expand)",
        .accum = .f32,
        .depth = depth,
        .ops = 4,
        .mids = 1,
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 32, .step = 32, .min = 32 } },
        .served = &.{ .{ .rows = 1, .n = 64, .k = 2304 }, .{ .rows = decode.max_rows, .n = 64, .k = 2304 }, .{ .rows = 2, .n = 32, .k = 32 } },
    });
}

// ── tests ──

const testing = std.testing;
const fixture = @import("fixture.zig");
const none: A = .{ .ctx = null };

/// A fixture layer whose routed experts are mxfp4, at the release's widths.
fn fp4Fixture(gpa: std.mem.Allocator) !*fixture.Fixture {
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 16, .ctx = 0, .experts_fp4 = true });
    const ly = fx.layer(20);
    try testing.expectEqual(params.QuantMode.mxfp4, ly.experts.gu.?.qp.mode);
    try testing.expectEqual(params.QuantMode.mxfp4, ly.experts.down.qp.mode);
    return fx;
}

/// Today's gate/up half: `routed`'s gather, clamped SwiGLU and weight, then the shared expert's quantized matmul and
/// SwiGLU: `[R, k + 1, N]`.
fn todaysUp(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, h2: A, idx: A, wts: A) !A {
    const sgu = ly.shared_gu.?;
    const rows = dims(h2, 0);
    const topk = dims(idx, 1);
    const k = dims(h2, 1);
    const n = dims(ly.experts.gu.?.w, 1) / 2;
    const xe = try moe_up.reshape(sc, h2, &.{ ci(rows), 1, 1, ci(k) });
    const gt, const up = try dsv41.gateUp(sc, &ly.experts, xe, none, idx, false);
    const routed = try moe_up.astype(sc, try moe_up.mul(sc, try dsv41.clampedSwiglu(sc, gt, up, m.cfg.dsv4_swiglu_limit), try moe_up.reshape(sc, wts, &.{ ci(rows), ci(topk), 1, 1 })), m.act);
    var y = mlx.mlx_array_new();
    const sy = try moe_up.own(sc, mlx.mlx_quantized_matmul(&y, h2, sgu.w, sgu.s, none, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", sc.s), y);
    const sh = try moe_up.astype(sc, try dsv41.clampedSwiglu(sc, try moe_up.sliceLast(sc, sy, 0, n), try moe_up.sliceLast(sc, sy, n, 2 * n), m.cfg.dsv4_swiglu_limit), m.act);
    return moe_up.concat(sc, &.{ try moe_up.reshape(sc, routed, &.{ ci(rows), ci(topk), ci(n) }), try moe_up.reshape(sc, sh, &.{ ci(rows), 1, ci(n) }) }, 1);
}

/// `bank` dequantized to f32 (`mode` with `bits`).
fn dequantF32(sc: *dsv41.Scope, w: A, s: A, bits: usize, mode: [*:0]const u8) !A {
    var d = mlx.mlx_array_new();
    return moe_up.own(sc, mlx.mlx_dequantize(&d, w, s, none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(@intCast(bits)), mode, none, .{ .value = .float32, .has_value = true }, sc.s), d);
}

/// The gate/up activations from the dequantized banks (each expert against every row in f32, the picks and the SwiGLU
/// after): `[R, k + 1, N]`.
fn truthUp(gpa: std.mem.Allocator, sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, h2: A, ids: []const u32, wts: []const f32) ![]f32 {
    const gu = ly.experts.gu.?;
    const sgu = ly.shared_gu.?;
    const rows = dims(h2, 0);
    const topk = ids.len / rows;
    const n = dims(gu.w, 1) / 2;
    var xt = mlx.mlx_array_new();
    const x32 = try moe_up.own(sc, mlx.mlx_transpose(&xt, try moe_up.astype(sc, h2, .float32), sc.s), xt);
    var y = mlx.mlx_array_new();
    const yb = try dsv41.hostF32(try moe_up.own(sc, mlx.mlx_matmul(&y, try dequantF32(sc, gu.w, gu.s, 4, "mxfp4"), x32, sc.s), y)); // [E, 2N, R]
    defer gpa.free(yb);
    var ys = mlx.mlx_array_new();
    const yh = try dsv41.hostF32(try moe_up.own(sc, mlx.mlx_matmul(&ys, try dequantF32(sc, sgu.w, sgu.s, 8, "mxfp8"), x32, sc.s), ys)); // [2N, R]
    defer gpa.free(yh);
    const lim: f64 = m.cfg.dsv4_swiglu_limit;
    const out = try gpa.alloc(f32, rows * (topk + 1) * n);
    for (0..rows) |row| for (0..topk + 1) |j| for (0..n) |c| {
        const g: f64, const up: f64, const wt: f64 = if (j < topk) blk: {
            const e: usize = ids[row * topk + j];
            break :blk .{ yb[(e * 2 * n + c) * rows + row], yb[(e * 2 * n + n + c) * rows + row], wts[row * topk + j] };
        } else .{ yh[c * rows + row], yh[(n + c) * rows + row], 1 };
        const gc = @min(g, lim);
        const uc = std.math.clamp(up, -lim, lim);
        out[(row * (topk + 1) + j) * n + c] = @floatCast(gc / (1 + @exp(-gc)) * uc * wt);
    };
    return out;
}

test "dsv41 dec moe_fp4: gate/up over mxfp4 experts, rows 1 2 4 6, against today's path and f32 truth" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fp4Fixture(gpa);
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(20);
    const kern = try build(m.cfg.dsv4_swiglu_limit);
    defer kern.deinit();
    var prng = std.Random.DefaultPrng.init(0xf04);
    const topk = ly.top_k;
    const k = m.cfg.hidden_size;
    for ([_]usize{ 1, 2, 4, 6 }) |rows| for ([_]moe_up.Routing{ .random, .shared_all, .disjoint }) |how| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const h2 = try sc.keep(try fx.normal(&.{ ci(rows), ci(k) }, .bfloat16, 0, 1));
        var ids: [64]u32 = undefined;
        var ws: [64]f32 = undefined;
        moe_up.route(prng.random(), how, rows, topk, 16, &ids, &ws);
        const idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ ci(rows), ci(topk) }, 2, .uint32));
        const wts = try sc.keep(mlx.mlx_array_new_data(&ws, &[_]c_int{ ci(rows), ci(topk) }, 2, .float32));
        const got = (try moeUp(&sc, m, kern, ly, h2, idx, wts)).?;
        const want = try todaysUp(&sc, m, ly, h2, idx, wts);
        const tr = try truthUp(gpa, &sc, m, ly, h2, ids[0 .. rows * topk], ws[0 .. rows * topk]);
        defer gpa.free(tr);
        const g = try dsv41.hostF32(got);
        defer gpa.free(g);
        const w = try dsv41.hostF32(want);
        defer gpa.free(w);
        var label: [64]u8 = undefined;
        try decode.expectNoWorse(try std.fmt.bufPrint(&label, "moe_fp4 up R={d} {s}", .{ rows, @tagName(how) }), g, w, tr);
        // A row of a launch equals its own one-row launch, bit for bit: a slot's greedy rounds never depend on its company.
        const gb = try moe_up.bitsOf(got);
        defer gpa.free(gb);
        const per = (topk + 1) * dims(got, 2);
        for (0..rows) |row| {
            var rs = dsv41.Scope.init(gpa, mlx.gpuStream());
            defer rs.deinit();
            const one = (try moeUp(&rs, m, kern, ly, try moe_up.sliceRow(&rs, h2, row), try moe_up.sliceRow(&rs, idx, row), try moe_up.sliceRow(&rs, wts, row))).?;
            const ob = try moe_up.bitsOf(one);
            defer gpa.free(ob);
            try testing.expectEqualSlices(u16, ob, gb[row * per ..][0..per]);
        }
    };
}

/// Row `row` of `a` (`[R, ...]`) as `[1, ...]`.
fn rowOf(sc: *dsv41.Scope, a: A, row: usize) !A {
    const sh = mlx.getShape(a);
    var flat: usize = 1;
    for (sh[1..]) |d| flat *= @intCast(d);
    var shape: [4]c_int = undefined;
    shape[0] = 1;
    @memcpy(shape[1..sh.len], sh[1..]);
    return moe_up.reshape(sc, try moe_up.sliceRow(sc, try moe_up.reshape(sc, a, &.{ sh[0], ci(flat) }), row), shape[0..sh.len]);
}

/// Today's down half: the routed gather over the weighted activations, the shared down, `addShared` and `hcExpand`.
fn todaysDown(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, act: A, idx: A, x1: A, sp: *const dsv41.Split) !A {
    const rows = dims(act, 0);
    const topk = dims(idx, 1);
    const k = dims(act, 2);
    const d = dims(x1, 2);
    var rt = mlx.mlx_array_new();
    const routed_act = try moe_up.reshape(sc, try moe_up.own(sc, mlx.mlx_slice(&rt, act, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ ci(rows), ci(topk), ci(k) }, 3, &[_]c_int{ 1, 1, 1 }, 3, sc.s), rt), &.{ ci(rows), ci(topk), 1, ci(k) });
    const dn = try dsv41.gatherMm(sc, routed_act, &ly.experts.down, none, try moe_up.astype(sc, idx, .uint32), false);
    const routed_sum = try sc.sum(try moe_up.astype(sc, try moe_up.reshape(sc, dn, &.{ ci(rows), ci(topk), ci(d) }), .float32), 1, false);
    var st = mlx.mlx_array_new();
    const shared_act = try moe_up.reshape(sc, try moe_up.own(sc, mlx.mlx_slice(&st, act, &[_]c_int{ 0, ci(topk), 0 }, 3, &[_]c_int{ ci(rows), ci(topk + 1), ci(k) }, 3, &[_]c_int{ 1, 1, 1 }, 3, sc.s), st), &.{ ci(rows), ci(k) });
    const y = try dsv41.addShared(sc, m, routed_sum, try ly.shared[2].apply(sc, shared_act));
    return dsv41.hcExpand(sc, x1, y, sp);
}

/// The stream the down half makes from the dequantized banks, summed in float64: `[R, hc, N]`.
fn truthDown(gpa: std.mem.Allocator, sc: *dsv41.Scope, ly: *const dsv41.Layer, act: A, ids: []const u32, x1: A, sp: *const dsv41.Split) ![]f32 {
    const rows = dims(act, 0);
    const topk = dims(act, 1) - 1;
    const k = dims(act, 2);
    const n = dims(x1, 2);
    const hc = dims(x1, 1);
    const a = try dsv41.hostF32(act);
    defer gpa.free(a);
    const xs = try dsv41.hostF32(x1);
    defer gpa.free(xs);
    const post = try dsv41.hostF32(sp.post);
    defer gpa.free(post);
    const comb = try dsv41.hostF32(sp.comb);
    defer gpa.free(comb);
    const bank = try dsv41.hostF32(try dequantF32(sc, ly.experts.down.w, ly.experts.down.s, 4, "mxfp4")); // [E, N, K]
    defer gpa.free(bank);
    const sbank = try dsv41.hostF32(try dequantF32(sc, ly.shared[2].w, ly.shared[2].s, 8, "mxfp8")); // [N, K]
    defer gpa.free(sbank);
    const out = try gpa.alloc(f32, rows * hc * n);
    const y = try gpa.alloc(f64, n);
    defer gpa.free(y);
    for (0..rows) |r| {
        for (y, 0..) |*v, c| {
            var s: f64 = 0;
            for (0..topk) |j| {
                const w = bank[(@as(usize, ids[r * topk + j]) * n + c) * k ..][0..k];
                const x = a[(r * (topk + 1) + j) * k ..][0..k];
                for (w, x) |wv, xv| s += @as(f64, wv) * xv;
            }
            const w = sbank[c * k ..][0..k];
            const x = a[(r * (topk + 1) + topk) * k ..][0..k];
            for (w, x) |wv, xv| s += @as(f64, wv) * xv;
            v.* = s;
        }
        for (0..hc) |h| for (0..n) |c| {
            var acc: f64 = 0;
            for (0..hc) |j| acc += @as(f64, comb[(r * hc + j) * hc + h]) * xs[(r * hc + j) * n + c];
            out[(r * hc + h) * n + c] = @floatCast(@as(f64, post[r * hc + h]) * y[c] + acc);
        };
    }
    return out;
}

test "dsv41 dec moe_fp4: down over mxfp4 experts, rows 1 2 4 6, against today's path and f32 truth" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fp4Fixture(gpa);
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(20);
    const kern = try build(m.cfg.dsv4_swiglu_limit);
    defer kern.deinit();
    var prng = std.Random.DefaultPrng.init(0xd04);
    const topk = ly.top_k;
    const inter = m.cfg.moe_intermediate_size;
    const hc = m.cfg.dsv4_hc_mult;
    for ([_]usize{ 1, 2, 4, 6 }) |rows| for ([_]moe_up.Routing{ .random, .shared_all, .disjoint }) |how| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const act = try sc.keep(try fx.normal(&.{ ci(rows), ci(topk + 1), ci(inter) }, .bfloat16, 0, 0.5));
        var ids: [64]u32 = undefined;
        var ws: [64]f32 = undefined;
        moe_up.route(prng.random(), how, rows, topk, 16, &ids, &ws);
        const idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ ci(rows), ci(topk) }, 2, .uint32));
        const x1 = try sc.keep(try fx.stream(rows));
        const sp: dsv41.Split = .{ .pre = none, .post = try sc.keep(try fx.uniform(&.{ ci(rows), ci(hc) }, 0.05, 1.5)), .comb = try sc.keep(try fx.uniform(&.{ ci(rows), ci(hc), ci(hc) }, -0.5, 0.5)) };
        const got = (try moeDown(&sc, kern, ly, act, idx, x1, &sp)).?;
        const want = try todaysDown(&sc, m, ly, act, idx, x1, &sp);
        const tr = try truthDown(gpa, &sc, ly, act, ids[0 .. rows * topk], x1, &sp);
        defer gpa.free(tr);
        const g = try dsv41.hostF32(got);
        defer gpa.free(g);
        const w = try dsv41.hostF32(want);
        defer gpa.free(w);
        var label: [64]u8 = undefined;
        try decode.expectNoWorse(try std.fmt.bufPrint(&label, "moe_fp4 down R={d} {s}", .{ rows, @tagName(how) }), g, w, tr);
        const gb = try moe_up.bitsOf(got);
        defer gpa.free(gb);
        const per = hc * dims(got, 2);
        for (0..rows) |row| {
            var rs = dsv41.Scope.init(gpa, mlx.gpuStream());
            defer rs.deinit();
            const sp1: dsv41.Split = .{ .pre = none, .post = try rowOf(&rs, sp.post, row), .comb = try rowOf(&rs, sp.comb, row) };
            const one = (try moeDown(&rs, kern, ly, try rowOf(&rs, act, row), try rowOf(&rs, idx, row), try rowOf(&rs, x1, row), &sp1)).?;
            const ob = try moe_up.bitsOf(one);
            defer gpa.free(ob);
            try testing.expectEqualSlices(u16, ob, gb[row * per ..][0..per]);
        }
    };
}

test "dsv41 dec moe_fp4: formats and shapes it does not serve are declined" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fp4Fixture(gpa);
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
        ly.experts.gu.?.qp.mode = .affine;
        try testing.expect(try moeUp(&sc, m, kern, &ly, h2, idx, wts) == null);
        ly = fx.layer(20).*;
        ly.experts.gu.?.qp.group_size = 64;
        try testing.expect(try moeUp(&sc, m, kern, &ly, h2, idx, wts) == null);
        ly = fx.layer(20).*;
        ly.shared_gu.?.qp.mode = .affine;
        try testing.expect(try moeUp(&sc, m, kern, &ly, h2, idx, wts) == null);
        ly = fx.layer(20).*;
        try testing.expect(try moeUp(&sc, m, kern, &ly, try moe_up.astype(&sc, h2, .float32), idx, wts) == null); // an f32 stream
    }
    try testing.expect(served != null); // the unchanged layer is served: the declines above are the changes
}

test "dsv41 dec moe_fp4: the declared launches are the wrappers', byte for byte" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const kern = try build(10);
    defer kern.deinit();
    for ([_]usize{ 1, 2, 6 }) |rows| {
        {
            const c = try a.create(kit.Case(UpK));
            defer {
                c.deinit();
                a.destroy(c);
            }
            c.* = .init(a, &up_l, .{});
            try testing.expect(try c.prepare(.{ .rows = rows, .n = 9, .k = 2048 }, .random, 7));
            try c.run();
            const mine = try a.dupe(u8, c.rig.outputBytes());
            defer a.free(mine);
            var sc = dsv41.Scope.init(a, c.rig.stream());
            defer sc.deinit();
            const theirs = try runUp(&sc, .{ .up = kern.up, .down = kern.down, .lim = c.args[7] }, rows, 2048, 9, TOPK, c.args[0], c.args[1], c.args[2], c.args[3], c.args[4], c.args[5], c.args[6]);
            try testing.expectEqualSlices(u8, try c.rig.bytesOf(&.{theirs}), mine);
        }
        {
            const c = try a.create(kit.Case(DownK));
            defer {
                c.deinit();
                a.destroy(c);
            }
            c.* = .init(a, &down_l, .{});
            try testing.expect(try c.prepare(.{ .rows = rows, .n = 64, .k = 256 }, .random, 7));
            try c.run();
            const mine = try a.dupe(u8, c.rig.outputBytes());
            defer a.free(mine);
            var sc = dsv41.Scope.init(a, c.rig.stream());
            defer sc.deinit();
            var ly: dsv41.Layer = undefined;
            ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .down = .{ .w = c.args[1], .s = c.args[2], .b = none, .qp = .{ .bits = 4, .group_size = 32, .mode = .mxfp4 } } };
            ly.shared[2] = .{ .w = c.args[3], .s = c.args[4], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } };
            const got = (try moeDown(&sc, kern, &ly, c.args[0], c.args[5], c.args[6], &.{ .pre = none, .post = c.args[7], .comb = c.args[8] })) orelse return error.Declined;
            try testing.expectEqualSlices(u8, try c.rig.bytesOf(&.{got}), mine);
        }
    }
}

test "dsv41 dec moe_fp4: the registered kernels pass the contract's gates 1-4" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var set = kc.Set.init(testing.allocator);
    defer set.deinit();
    try register(&set);
    var tally: kc.gates.Tally = .{};
    try kc.gates.runAll(testing.allocator, set.kernels.items, 0xc0de, &tally);
    try testing.expectEqual(@as(usize, 0), tally.problems());
}
