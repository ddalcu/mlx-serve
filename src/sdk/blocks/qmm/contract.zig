//! The qmm area's kernel contract: the row-exact affine matvecs (rowqmv: the plain and relu2 rows, the routed experts'
//! two passes, the dense router), the 2-bit Prism GEMVs (qmv2's one-row, multi-row and MMA kernels, NAX2's narrow
//! tile, the int8-activation prefill), the MXFP8 GEMV, simd_qmm's five kernels and lane_qmm's three. Shapes: rows are
//! input rows (routed experts: token rows, TOPK pairs each), n output columns, k the reduction. The weights are
//! host-built in MLX's packed layout (`host.quantize`); N is cut below the served models' widest projections where
//! the float64 reference would not fit a Debug test, K is always a served width.
//! References carry their own forward error (`host.Val`, `host.allowance`): f32 sums of any order bounded by
//! gamma(K + 16) over the terms' magnitudes, plus every rounding the kernel makes before them (MLX qmv's bf16 runs of
//! four inputs, a weight dequantized into T). Kernels that factor a group's scale and bias out of the dot
//! (s * sum(x q) + b * sum(x)) are referenced that way, so their NaN classes are the factored ones (0 * Inf).
//! rowqmv's contract is row exactness (a verify row equals the serial decode row), not stock bits: at whole passes it
//! also equals MLX's one-row quantized_matmul, and at Nemotron's K 2688 and 1856, where stock splits K differently,
//! the float64 bound is the contract.
//! The int8 route's quantizers write exact int8 codes beside f32 scales and code sums (`Spec.outs`).
//! Not registered: `qmv_cuda` (CUDA, no rig). Known issues: the quantizers drop a NaN from the group absmax (gate 1).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../testing/kernel_contract/root.zig");
const rowqmv = @import("rowqmv.zig");
const qmv2 = @import("qmv2.zig");
const mxfp8_qmv = @import("mxfp8_qmv.zig");
const simd_qmm = @import("simd_qmm.zig");
const lane_qmm = @import("lane_qmm.zig");
const device = @import("../../../mlx/device.zig");
const qmv_nax2 = @import("qmv_nax2.zig");
const qmm_int8 = @import("qmm_int8.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Val = host.Val;
const Shape = spec.Shape;
const U = host.U;

fn cInt(v: usize) c_int {
    return @intCast(v);
}

/// The f32 error of a sum of terms whose magnitudes add to `mag`, in any order, `k` terms deep.
fn sumErr(k: usize, mag: f64) f64 {
    return spec.gamma(@floatFromInt(k + 16), U) * mag;
}

/// Gate 4's weight codes for an [n, k] matrix, `lo` to `hi`: each row's second half is its first rotated by one, so
/// against `stressX` a dot is exactly 0 while its partial sums reach about A x hi x k / 4 (past fp16's range and
/// bf16's integers) and no summation tree pairs the halves' terms alike. A ternary pack draws codes 1 and 2 (weights
/// q - 1 of 0 and 1), so its sums climb too. `bump` adds 1 to two codes of the first half (drawn to `hi` - 1): the dot
/// is then 2 A, for a relu2 that would map 0 and any fp16 NaN alike.
fn stressCodes(rng: std.Random, codes: []u32, n: usize, k: usize, lo: u32, hi: u32, bump: bool) void {
    const half = k / 2;
    for (0..n) |r| {
        const row = codes[r * k ..][0..k];
        for (row[0..half]) |*c| c.* = rng.intRangeAtMost(u32, lo, if (bump) hi - 1 else hi);
        for (0..half) |i| row[half + i] = row[(i + 1) % half];
        if (bump) for (0..2) |j| {
            row[(r * 7 + j * (half / 2 + 1)) % half] += 1;
        };
        for (row[2 * half ..]) |*c| c.* = 0;
    }
}

/// Gate 4's inputs: row r is +A on the first half of k and -A on the second, times (-1)^r; A is the largest odd
/// integer (up to 251) that keeps every partial sum under 2^22 against codes up to `top` (odd: a multiple of a power
/// of two would keep fp16's and bf16's coarse sums exact).
fn stressA(k: usize, top: u32) f64 {
    const most = @as(f64, 0x1p22) / (@as(f64, @floatFromInt(top)) * @as(f64, @floatFromInt(k / 2 + 1)));
    const v = @max(1, @min(251, @floor(most)));
    return if (@mod(v, 2) == 0) v - 1 else v;
}

fn stressX(x: []f64, m: usize, k: usize, a: f64) void {
    for (0..m) |r| for (0..k) |c| {
        const sign: f64 = if (r % 2 == 0) 1 else -1;
        x[r * k + c] = if (c < k / 2) sign * a else -sign * a;
    };
}

/// One call's host operands: x, the packed weight (dequantized values in `q.w`) and its codes.
const Affine = struct { x: []f64, q: host.Quant, codes: []u32 };

/// The fill every family shares: x [m, k] in `dt`, the weight's codes (gate 4's pattern under `.stress`, with unit
/// scales and zero biases, or -1 biases for a ternary pack), NaN/Inf rows planted in x under `.nonfinite`.
fn fillAffine(a: std.mem.Allocator, rng: std.Random, fill: spec.Fill, m: usize, n: usize, k: usize, bits: u32, gs: usize, dt: mlx.mlx_dtype, ternary: bool, bump: bool) !Affine {
    var q = try host.quantize(a, rng, n, k, bits, gs, dt);
    const top: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
    const codes = try a.alloc(u32, n * k);
    if (fill == .stress) {
        stressCodes(rng, codes, n, k, @intFromBool(ternary), if (ternary) 2 else top, bump);
        @memset(q.scales, 1);
        @memset(q.biases, if (ternary) -1 else 0);
    } else {
        for (codes) |*c| c.* = rng.uintAtMost(u32, if (ternary) 2 else top);
        if (ternary) for (q.scales, q.biases) |sc, *bi| {
            bi.* = -sc;
        };
    }
    host.dequant(&q, n, k, bits, gs, codes);
    const x = try host.uniform(a, rng, m * k, -1, 1, dt);
    switch (fill) {
        .random => {},
        .nonfinite => host.plant(x, m, k),
        .stress => stressX(x, m, k, stressA(k, top)),
    }
    return .{ .x = x, .q = q, .codes = codes };
}

/// One output of a factored affine dot: per `chunk` of a group, s * sum(x q) + b * sum(x), as MLX's qmv family
/// computes it (q the code: scale and bias come out of the products, and an infinite scale meets each chunk's sign).
/// `xsum` gives a chunk's input sum as the kernel forms it (f32 adds when null).
fn factored(x: []const f64, w: Affine, row: usize, k: usize, gs: usize, chunk: usize, xsum: ?*const fn ([]const f64) Val) Val {
    const q = w.q;
    var v: f64 = 0;
    var mag: f64 = 0;
    var extra: f64 = 0;
    const g_n = k / gs;
    var c0: usize = 0;
    while (c0 < k) : (c0 += chunk) {
        const g = c0 / gs;
        const s = q.scales[row * g_n + g];
        const b = q.biases[row * g_n + g];
        var dot: f64 = 0;
        var am: f64 = 0;
        for (c0..c0 + chunk) |c| {
            const code: f64 = @floatFromInt(w.codes[row * k + c]);
            dot += x[c] * code;
            am += @abs(x[c] * code);
        }
        const xs = x[c0..][0..chunk];
        const sx = if (xsum) |f| f(xs) else Val{ .v = host.sum(xs) };
        v += s * dot + b * sx.v;
        mag += @abs(s) * am + @abs(b) * host.sumAbs(xs);
        extra += @abs(b) * sx.e;
    }
    return .{ .v = v, .e = sumErr(k, mag) + extra };
}

/// One output of a dot over dequantized weights `wr` (a ternary pack's s * (q - 1), or values rounded into T), each
/// weight within `we` of its value.
fn decoded(x: []const f64, wr: []const f64, we: ?[]const f64) Val {
    var v: f64 = 0;
    var mag: f64 = 0;
    var extra: f64 = 0;
    for (x, wr, 0..) |xv, wv, i| {
        v += xv * wv;
        mag += @abs(xv * wv);
        if (we) |e| extra += @abs(xv) * e[i];
    }
    return .{ .v = v, .e = sumErr(x.len, mag) + extra };
}

/// The outputs with the sums run in `acc`: each output's products x * w summed in `order`.
fn simDots(a: std.mem.Allocator, x: []const f64, wr: []const f64, m: usize, n: usize, k: usize, acc: spec.Accum, order: spec.Order, out: []f64) !void {
    const terms = try a.alloc(f64, k);
    defer a.free(terms);
    for (0..m) |r| for (0..n) |j| {
        for (terms, x[r * k ..][0..k], wr[j * k ..][0..k]) |*t, xv, wv| t.* = xv * wv;
        out[r * n + j] = spec.sumIn(acc, order, terms);
    };
}

// ── rowqmv: one simdgroup per input row ──

/// MLX qmv_fast's input sum at 4 and 6 bits: runs of four added in bf16 (a + b, + c, + d, each rounded), then in f32.
fn bf16Runs(x: []const f64) Val {
    var v: f64 = 0;
    var i: usize = 0;
    while (i < x.len) : (i += 4) {
        v += host.rt(.bfloat16, host.rt(.bfloat16, host.rt(.bfloat16, x[i] + x[i + 1]) + x[i + 2]) + x[i + 3]);
    }
    return .{ .v = v, .e = sumErr(x.len / 4, host.sumAbs(x)) };
}

const rq_inputs: []const [*:0]const u8 = &.{ "x", "w", "scales", "biases" };
const rq_launch = metal.Launch{ .name = "msv_rowqmv", .inputs = rq_inputs, .outputs = &.{"y"}, .source = rowqmv.QMV_SOURCE, .header = rowqmv.HEADER, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };
const rq_up_launch = metal.Launch{ .name = "msv_rowqmv_expert_up", .inputs = &.{ "x", "ids", "w", "scales", "biases" }, .outputs = &.{"y"}, .source = rowqmv.EXPERT_UP_SOURCE, .header = rowqmv.HEADER, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };
const rq_down_launch = metal.Launch{ .name = "msv_rowqmv_expert_down", .inputs = &.{ "x", "ids", "w", "scales", "biases" }, .outputs = &.{"y"}, .source = rowqmv.EXPERT_DOWN_SOURCE, .header = rowqmv.HEADER, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };
/// `part[MAX_ROWS][SG]` floats.
const rq_router_launch = metal.Launch{ .name = "msv_rowqmv_router", .inputs = &.{ "x", "w", "rows" }, .outputs = &.{"y"}, .source = rowqmv.ROUTER_SOURCE, .header = rowqmv.HEADER, .tg_bytes = rowqmv.MAX_ROWS * 8 * 4, .m1 = 1024, .m2 = 1024 };

const RqKind = enum { plain, relu2, up, down };
const RqCfg = struct { kind: RqKind, bits: u32, gs: usize };
const TOPK = 6;
const EXPERTS = 8;

/// rowqmv's matvec families. Plain / relu2: rows are input rows. Up / down: rows are token rows, TOPK (row, slot)
/// pairs each over a bank of EXPERTS experts (the kernel reads expert ids[p]; the bank's size is not in its code).
const Rq = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: RqCfg,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    w: Affine = undefined,
    ids: []f64 = &.{},
    args: [5]mlx.mlx_array = undefined,
    n_args: usize = 0,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: RqCfg) Rq {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .arena = .init(a) };
    }

    pub fn deinit(self: *Rq) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn routed(self: *const Rq) bool {
        return self.cfg.kind == .up or self.cfg.kind == .down;
    }

    /// Activation rows: the token rows, or (down) one per pair.
    fn xRows(self: *const Rq, shape: Shape) usize {
        return if (self.cfg.kind == .down) shape.rows * TOPK else shape.rows;
    }

    fn outRows(self: *const Rq, shape: Shape) usize {
        return if (self.routed()) shape.rows * TOPK else shape.rows;
    }

    pub fn geometry(self: *const Rq, shape: Shape) ?metal.Geometry {
        const c = self.cfg;
        if (shape.rows < 1 or shape.rows > rowqmv.MAX_ROWS or shape.k % 64 != 0 or shape.k % c.gs != 0 or shape.n % 8 != 0) return null;
        const n = cInt(shape.n);
        var g: metal.Geometry = undefined;
        if (self.routed()) {
            const pairs = cInt(shape.rows * TOPK);
            g = .{ .grid = .{ 64, @divExact(n, 8), pairs }, .tg = .{ 64, 1, 1 } };
            g.out(&.{ pairs, n }, .bfloat16);
        } else {
            const r = cInt(shape.rows);
            g = .{ .grid = .{ 32 * r, @divExact(n, 4), 1 }, .tg = .{ 32 * r, if (shape.rows <= 8) 2 else 1, 1 } };
            g.out(&.{ r, n }, .bfloat16);
        }
        g.int("K", shape.k);
        g.int("N", shape.n);
        g.int("GS", c.gs);
        g.int("RPS", 4);
        if (self.routed()) g.int("SG", 2);
        g.int("BITS", c.bits);
        switch (c.kind) {
            .plain, .relu2 => g.flag("RELU2", c.kind == .relu2),
            .up => g.int("TOPK", TOPK),
            .down => {},
        }
        return g;
    }

    pub fn prepare(self: *Rq, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const banks: usize = if (self.routed()) EXPERTS else 1;
        self.w = try fillAffine(a, rng, fill, self.xRows(shape), banks * shape.n, shape.k, c.bits, c.gs, .bfloat16, false, c.kind == .relu2 or c.kind == .up);
        const f = self.w;
        // relu2 maps NaN and -Inf to 0: row 4 at 1e30 makes the positive outputs' squares overflow f32 instead.
        if (fill == .nonfinite and (c.kind == .relu2 or c.kind == .up)) @memset(f.x[4 * shape.k ..][0..shape.k], host.rt(.bfloat16, 1e30));
        self.rig.reset();
        const kw = cInt(shape.k * c.bits / 32);
        const kg = cInt(shape.k / c.gs);
        const n = cInt(shape.n);
        const xr = cInt(self.xRows(shape));
        if (self.routed()) {
            self.ids = try a.alloc(f64, shape.rows * TOPK);
            for (self.ids) |*v| v.* = @floatFromInt(rng.uintLessThan(u32, EXPERTS));
            const e = cInt(EXPERTS);
            self.args = .{
                try self.rig.input(.bfloat16, &.{ xr, cInt(shape.k) }, f.x),
                try self.rig.input(.uint32, &.{cInt(self.ids.len)}, self.ids),
                try self.rig.input(.uint32, &.{ e, n, kw }, f.q.words),
                try self.rig.input(.bfloat16, &.{ e, n, kg }, f.q.scales),
                try self.rig.input(.bfloat16, &.{ e, n, kg }, f.q.biases),
            };
            self.n_args = 5;
        } else {
            self.args[0..4].* = .{
                try self.rig.input(.bfloat16, &.{ xr, cInt(shape.k) }, f.x),
                try self.rig.input(.uint32, &.{ n, kw }, f.q.words),
                try self.rig.input(.bfloat16, &.{ n, kg }, f.q.scales),
                try self.rig.input(.bfloat16, &.{ n, kg }, f.q.biases),
            };
            self.n_args = 4;
        }
        return true;
    }

    pub fn run(self: *Rq) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Rq) usize {
        return self.outRows(self.shape) * self.shape.n;
    }

    /// Output (p, j): pair p reads x row p / TOPK (up) or p (down) against expert ids[p]'s row j.
    fn dot(self: *const Rq, p: usize, j: usize, acc: ?spec.Accum, order: spec.Order, terms: []f64) Val {
        const sh = self.shape;
        const c = self.cfg;
        const xi = if (self.cfg.kind == .up) p / TOPK else p;
        const wrow = if (self.routed()) @as(usize, @intFromFloat(self.ids[p])) * sh.n + j else j;
        const x = self.w.x[xi * sh.k ..][0..sh.k];
        if (acc) |ac| {
            for (terms, x, self.w.q.w[wrow * sh.k ..][0..sh.k]) |*t, xv, wv| t.* = xv * wv;
            return .{ .v = spec.sumIn(ac, order, terms) };
        }
        return factored(x, self.w, wrow, sh.k, c.gs, if (c.bits == 4) 16 else 8, if (c.bits == 8) null else &bf16Runs);
    }

    fn outputs(self: *const Rq, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const terms = try self.rig.a.alloc(f64, self.shape.k);
        defer self.rig.a.free(terms);
        const n = self.shape.n;
        for (0..self.outRows(self.shape)) |p| for (0..n) |j| {
            var y = self.dot(p, j, acc, order, terms);
            if (self.cfg.kind == .relu2 or self.cfg.kind == .up) {
                // mlx-lm's relu2 on the bf16 output: bf16(max(bf16(y), 0)^2).
                const h = host.round(.bfloat16, y);
                const r = Val{ .v = @max(h.v, 0), .e = if (h.v + h.e > 0) h.e else 0 };
                // The square is an f32 product: past f32's range it is +Inf.
                const sq = r.v * r.v;
                y = .{ .v = if (sq > std.math.floatMax(f32)) std.math.inf(f64) else sq, .e = 2 * r.v * r.e + r.e * r.e };
            }
            want[p * n + j] = y.v;
            if (mag) |m| m[p * n + j] = host.allowance(y.e);
        };
    }

    pub fn reference(self: *const Rq, want: []f64, mag: []f64) !void {
        try self.outputs(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Rq, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputs(out, null, acc, order);
    }

    /// MLX's quantized_matmul, one call per input row (plain registrations).
    pub fn replaced(self: *Rq) ![]const u8 {
        const s = self.rig.stream();
        const sh = self.shape;
        const rows = try self.rig.a.alloc(mlx.mlx_array, sh.rows);
        defer self.rig.a.free(rows);
        var made: usize = 0;
        defer for (rows[0..made]) |r| {
            _ = mlx.mlx_array_free(r);
        };
        for (0..sh.rows) |r| {
            var xr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xr);
            try mlx.check(mlx.mlx_slice(&xr, self.args[0], &.{ cInt(r), 0 }, 2, &.{ cInt(r + 1), cInt(sh.k) }, 2, &.{ 1, 1 }, 2, s));
            rows[r] = mlx.mlx_array_new();
            made += 1;
            try mlx.check(mlx.mlx_quantized_matmul(&rows[r], xr, self.args[1], self.args[2], self.args[3], true, mlx.mlx_optional_int.some(cInt(self.cfg.gs)), mlx.mlx_optional_int.some(cInt(self.cfg.bits)), "affine", s));
        }
        const vec = mlx.mlx_vector_array_new_data(rows.ptr, rows.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(all);
        try mlx.check(mlx.mlx_concatenate_axis(&all, vec, 0, s));
        return self.rig.bytesOf(&.{all});
    }

    fn wrapper(self: *Rq) ![]const u8 {
        const s = self.rig.stream();
        const c = self.cfg;
        const got = switch (c.kind) {
            .plain, .relu2 => (try rowqmv.qmvAct(self.args[0], self.args[1], self.args[2], self.args[3], c.bits, @intCast(c.gs), c.kind == .relu2, s)) orelse return error.Declined,
            .up, .down => try rowqmv.expertLaunch(c.kind == .up, self.args[0], self.args[1], self.args[2], self.args[3], self.args[4], cInt(self.shape.rows * TOPK), TOPK, .{ c.bits, @intCast(c.gs) }, s),
        };
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};

/// rowqmv's dense router: rows are input rows (a runtime count), n experts, k the hidden width.
const Router = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    w: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) Router {
        return .{ .rig = metal.Rig.init(a), .launch = &rq_router_launch, .arena = .init(a) };
    }

    pub fn deinit(self: *Router) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Router, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.rows > rowqmv.MAX_ROWS or shape.k % 32 != 0 or shape.n == 0) return null;
        var g = metal.Geometry{ .grid = .{ 256, cInt(shape.n), 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ cInt(shape.rows), cInt(shape.n) }, .bfloat16);
        g.int("D", shape.k);
        g.int("E", shape.n);
        g.int("SG", 8);
        g.int("MAX_ROWS", rowqmv.MAX_ROWS);
        return g;
    }

    pub fn prepare(self: *Router, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try host.uniform(a, rng, shape.rows * shape.k, -1, 1, .bfloat16);
        self.w = try host.uniform(a, rng, shape.n * shape.k, -0.05, 0.05, .bfloat16);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.x, shape.rows, shape.k),
            .stress => {
                const codes = try a.alloc(u32, shape.n * shape.k);
                stressCodes(rng, codes, shape.n, shape.k, 0, 15, false);
                for (self.w, codes) |*w, c| w.* = @floatFromInt(c);
                stressX(self.x, shape.rows, shape.k, stressA(shape.k, 15));
            },
        }
        self.rig.reset();
        self.args = .{
            try self.rig.input(.bfloat16, &.{ cInt(shape.rows), cInt(shape.k) }, self.x),
            try self.rig.input(.bfloat16, &.{ cInt(shape.n), cInt(shape.k) }, self.w),
            try host.intScalar(&self.rig, cInt(shape.rows)),
        };
        return true;
    }

    pub fn run(self: *Router) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Router) usize {
        return self.shape.rows * self.shape.n;
    }

    fn outputs(self: *const Router, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const k = self.shape.k;
        const terms = try self.rig.a.alloc(f64, k);
        defer self.rig.a.free(terms);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |e| {
            for (terms, self.x[r * k ..][0..k], self.w[e * k ..][0..k]) |*t, x, w| t.* = x * w;
            const i = r * self.shape.n + e;
            want[i] = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms);
            if (mag) |m| m[i] = host.allowance(sumErr(k, host.sumAbs(terms)));
        };
    }

    pub fn reference(self: *const Router, want: []f64, mag: []f64) !void {
        try self.outputs(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Router, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputs(out, null, acc, order);
    }

    fn wrapper(self: *Router) ![]const u8 {
        const got = (try rowqmv.router(self.args[0], self.args[1], self.rig.stream())) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};

// ── qmv2: 2-bit, group-128 GEMVs for Prism packs ──

/// M1/M2 limits: the least the offline builds give over every registration's shapes (the rows kernel's moves with
/// M, R and G, 576 at the least).
const q2_h2_launch = metal.Launch{ .name = "msv_qmv2_h2", .inputs = &.{ "x", "w", "scales", "biases" }, .outputs = &.{"y"}, .source = qmv2.SOURCE, .header = qmv2.DECODE, .tg_bytes = 0, .m1 = 704, .m2 = 704 };
const q2_rows_launch = metal.Launch{ .name = "msv_qmv2_rows", .inputs = &.{ "x", "w", "scales" }, .outputs = &.{"y"}, .source = qmv2.ROWS_SOURCE, .header = qmv2.DECODE, .tg_bytes = 0, .m1 = 576, .m2 = 576 };
const q2_mma_launch = metal.Launch{ .name = "msv_qmv2_mma", .inputs = &.{ "x", "w", "scales" }, .outputs = &.{"y"}, .source = qmv2.MMA_SOURCE, .header = "#include <metal_simdgroup_matrix>\n", .tg_bytes = 0, .m1 = 928, .m2 = 928 };

const Q2Kind = enum { h2, rows, mma };
const Q2Cfg = struct { kind: Q2Kind, dt: mlx.mlx_dtype = .float16, rows: qmv2.Geom = .{ .r = 4, .g = 2 }, mma: qmv2.MmaGeom = .{ .rb = 1, .sgs = 4, .split = 1 } };

/// qmv2's kernels. `h2` is one activation row (any bias): its shape's rows count 8-column output blocks
/// (N = rows x 8), and its non-finite fill plants the weight's scales by output row (one x row would make every
/// output non-finite). `rows` and `mma` read a ternary pack (biases == -scales, codes decode to q - 1): rows are
/// input rows.
const Q2 = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: Q2Cfg,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    w: Affine = undefined,
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, cfg: Q2Cfg) Q2 {
        return .{ .rig = metal.Rig.init(a), .launch = switch (cfg.kind) {
            .h2 => &q2_h2_launch,
            .rows => &q2_rows_launch,
            .mma => &q2_mma_launch,
        }, .cfg = cfg, .arena = .init(a) };
    }

    pub fn deinit(self: *Q2) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn mRows(self: *const Q2, shape: Shape) usize {
        return if (self.cfg.kind == .h2) 1 else shape.rows;
    }

    fn nCols(self: *const Q2, shape: Shape) usize {
        return if (self.cfg.kind == .h2) shape.rows * 8 else shape.n;
    }

    pub fn geometry(self: *const Q2, shape: Shape) ?metal.Geometry {
        const c = self.cfg;
        const m = self.mRows(shape);
        const n = self.nCols(shape);
        if (m < 1 or n == 0 or shape.k == 0) return null;
        var g: metal.Geometry = undefined;
        switch (c.kind) {
            .h2 => {
                if (shape.k % 512 != 0) return null;
                g = .{ .grid = .{ 64, cInt(n / 8), 1 }, .tg = .{ 64, 1, 1 } };
                g.out(&.{cInt(n)}, c.dt);
                g.dtype("T", c.dt);
            },
            .rows => {
                const per: usize = @intCast(c.rows.r * c.rows.g);
                if (m > qmv2.MAX_ROWS or shape.k % 512 != 0 or n % per != 0) return null;
                g = .{ .grid = .{ 32 * c.rows.g, cInt(n / per), 1 }, .tg = .{ 32 * c.rows.g, 1, 1 } };
                g.out(&.{ cInt(m), cInt(n) }, c.dt);
                g.dtype("T", c.dt);
                g.int("M", m);
                g.int("R", c.rows.r);
                g.int("G", c.rows.g);
            },
            .mma => {
                const mg = c.mma;
                const per: usize = @intCast(8 * mg.rb * mg.sgs);
                if (m > qmv2.MMA_MAX_ROWS or n % per != 0 or shape.k % (128 * @as(usize, @intCast(mg.split))) != 0) return null;
                const ot: mlx.mlx_dtype = if (mg.split > 1) .float32 else c.dt;
                g = .{ .grid = .{ cInt(n / per) * 32 * mg.sgs, mg.split, 1 }, .tg = .{ 32 * mg.sgs, 1, 1 } };
                g.out(&.{ mg.split * cInt(m), cInt(n) }, ot);
                g.dtype("T", c.dt);
                g.dtype("OT", ot);
                g.int("M", m);
                g.int("K", shape.k);
                g.int("N", n);
                g.int("RB", mg.rb);
                g.int("SGS", mg.sgs);
                g.int("SPLIT", mg.split);
                g.int("NB", (m + 7) / 8);
            },
        }
        return g;
    }

    pub fn prepare(self: *Q2, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const m = self.mRows(shape);
        const n = self.nCols(shape);
        const h2 = c.kind == .h2;
        self.w = try fillAffine(a, rng, if (h2 and fill == .nonfinite) .random else fill, m, n, shape.k, 2, 128, c.dt, !h2, false);
        if (h2 and fill == .nonfinite) host.plant(self.w.q.scales, n, shape.k / 128);
        self.rig.reset();
        const kg = cInt(shape.k / 128);
        self.args[0..3].* = .{
            try self.rig.input(c.dt, &.{ cInt(m), cInt(shape.k) }, self.w.x),
            try self.rig.input(.uint32, &.{ cInt(n), cInt(shape.k / 16) }, self.w.q.words),
            try self.rig.input(c.dt, &.{ cInt(n), kg }, self.w.q.scales),
        };
        if (h2) self.args[3] = try self.rig.input(c.dt, &.{ cInt(n), kg }, self.w.q.biases);
        return true;
    }

    pub fn run(self: *Q2) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..if (self.cfg.kind == .h2) 4 else 3]);
    }

    pub fn count(self: *const Q2) usize {
        return self.mRows(self.shape) * self.nCols(self.shape);
    }

    pub fn reference(self: *const Q2, want: []f64, mag: []f64) !void {
        const k = self.shape.k;
        const n = self.nCols(self.shape);
        for (0..self.mRows(self.shape)) |r| for (0..n) |j| {
            const x = self.w.x[r * k ..][0..k];
            // h2 adds s * dot + b * sum per lane's 16 inputs; rows and mma decode q - 1 and scale per lane / group.
            const y = if (self.cfg.kind == .h2) factored(x, self.w, j, k, 128, 16, null) else decoded(x, self.w.q.w[j * k ..][0..k], null);
            want[r * n + j] = y.v;
            mag[r * n + j] = host.allowance(y.e);
        };
    }

    pub fn simulate(self: *const Q2, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try simDots(self.rig.a, self.w.x, self.w.q.w, self.mRows(self.shape), self.nCols(self.shape), self.shape.k, acc, order, out);
    }

    fn wrapper(self: *Q2) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args;
        // The multi-row kernels never read biases; the wrapper only checks they are there.
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, x[2], s));
        const got = switch (self.cfg.kind) {
            .h2 => (try qmv2.qmv(x[0], x[1], x[2], x[3], 2, 128, s)) orelse return error.Declined,
            .rows => (try qmv2.qmvRowsAt(x[0], x[1], x[2], neg, 2, 128, true, self.cfg.rows, s)) orelse return error.Declined,
            .mma => (try qmv2.qmmMmaAt(x[0], x[1], x[2], neg, 2, 128, true, self.cfg.mma, s)) orelse return error.Declined,
        };
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};

// ── mxfp8_qmv: e4m3 weights, an e8m0 exponent per 32 ──

const fp8_launch = metal.Launch{ .name = "mlxserve_mxfp8_qmv", .inputs = &.{ "x", "w_q", "scales" }, .outputs = &.{"y"}, .source = kernelSource("kernels/qmm/mxfp8_qmv.metal"), .header = mxfp8_qmv.HEADER, .tg_bytes = 0, .m1 = 832, .m2 = 832, .min_gen = 16 };

/// An e4m3 byte's value (the OCP fn variant: no infinities; the kernel reads S.1111.111, MLX's NaN, as 480, so the
/// fills never draw it).
fn e4m3(b: u8) f64 {
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 15;
    const m: f64 = @floatFromInt(b & 7);
    return sign * if (e == 0) m / 8 * 0x1p-6 else (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

/// The MXFP8 GEMV at 1..MAX_ROWS rows: rows are input rows.
const Fp8 = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    /// Each weight's value times its block's scale, as float64.
    w: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, dt: mlx.mlx_dtype) Fp8 {
        return .{ .rig = metal.Rig.init(a), .launch = &fp8_launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Fp8) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Fp8, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.rows > mxfp8_qmv.MAX_ROWS or shape.k % 32 != 0 or shape.n == 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, cInt(shape.n), 1 }, .tg = .{ 32, 4, 1 } };
        g.out(&.{ cInt(shape.rows), cInt(shape.n) }, self.dt);
        g.dtype("T", self.dt);
        g.int("K", shape.k);
        g.int("N", shape.n);
        g.int("M", shape.rows);
        return g;
    }

    pub fn prepare(self: *Fp8, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.n;
        const k = shape.k;
        self.x = try host.uniform(a, rng, shape.rows * k, -1, 1, self.dt);
        const bytes = try a.alloc(u8, n * k);
        const exps = try a.alloc(f64, n * k / 32);
        if (fill == .stress) {
            // Integer weights 0..15 (exact in e4m3) under unit scales.
            const codes = try a.alloc(u32, n * k);
            stressCodes(rng, codes, n, k, 0, 15, false);
            for (bytes, codes) |*b, c| {
                b.* = 0;
                while (e4m3(b.*) != @as(f64, @floatFromInt(c))) b.* += 1;
            }
            @memset(exps, 127);
            stressX(self.x, shape.rows, k, stressA(k, 15));
        } else {
            for (bytes) |*b| {
                b.* = rng.int(u8);
                if (b.* & 0x7f == 0x7f) b.* -= 1;
            }
            for (exps) |*e| e.* = @floatFromInt(rng.intRangeAtMost(u8, 120, 130));
            if (fill == .nonfinite) host.plant(self.x, shape.rows, k);
        }
        self.w = try a.alloc(f64, n * k);
        for (self.w, bytes, 0..) |*w, b, i| w.* = e4m3(b) * std.math.pow(f64, 2, exps[i / 32] - 127);
        const words = try a.alloc(f64, n * k / 4);
        for (words, 0..) |*wd, i| wd.* = @floatFromInt(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
        self.rig.reset();
        self.args = .{
            try self.rig.input(self.dt, &.{ cInt(shape.rows), cInt(k) }, self.x),
            try self.rig.input(.uint32, &.{ cInt(n), cInt(k / 4) }, words),
            try self.rig.input(.uint8, &.{ cInt(n), cInt(k / 32) }, exps),
        };
        return true;
    }

    pub fn run(self: *Fp8) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Fp8) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *const Fp8, want: []f64, mag: []f64) !void {
        const k = self.shape.k;
        const n = self.shape.n;
        for (0..self.shape.rows) |r| for (0..n) |j| {
            const y = decoded(self.x[r * k ..][0..k], self.w[j * k ..][0..k], null);
            want[r * n + j] = y.v;
            mag[r * n + j] = host.allowance(y.e);
        };
    }

    pub fn simulate(self: *const Fp8, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try simDots(self.rig.a, self.x, self.w, self.shape.rows, self.shape.n, self.shape.k, acc, order, out);
    }

    fn wrapper(self: *Fp8) ![]const u8 {
        const xfm = @import("../../../mlx/device.zig");
        xfm.device_gen_override = .{ .gen = 17, .phone = false };
        defer xfm.device_gen_override = null;
        const got = (try mxfp8_qmv.linear(self.args[0], self.args[1], self.args[2], self.rig.stream())) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};

// ── simd_qmm: row-exact 4/6/8-bit, group-64 GEMMs on simdgroup matrices (any Apple GPU) ──

const sq_undef = "  #undef LOAD8\n";
/// The wrapper probes each shape's one-row build once (`prepare`) and declines the shape where it fails: `adapts`.
const sq_scalar_launch = metal.Launch{ .name = "msv_simd_qmm_scalar", .inputs = &.{ "X", "W", "SC", "BI", "ONE" }, .outputs = &.{"OUT"}, .source = simd_qmm.LOAD8 ++ simd_qmm.SCALAR ++ sq_undef, .header = simd_qmm.HEADER, .tg_bytes = 32 * 76 * 4, .m1 = 1024, .m2 = 1024, .adapts = true };
/// `xs[RR][XB * 76]` floats at three rows (XB 16).
const sq_rows_launch = metal.Launch{ .name = "msv_simd_qmm_rows", .inputs = &.{ "X", "W", "SC", "BI", "ONE" }, .outputs = &.{"OUT"}, .source = simd_qmm.LOAD8 ++ simd_qmm.SCALAR_ROWS ++ sq_undef, .header = simd_qmm.HEADER, .tg_bytes = 3 * 16 * 76 * 4, .m1 = 1024, .m2 = 1024 };
/// `red[S * RT * NT * 64]` floats, at most 16 KB (`tiles`). The wrapper halves the simdgroups where a GPU caps the
/// threadgroup (M1/M2) on every new plan: `adapts`.
const sq_mma_launch = metal.Launch{ .name = "msv_simd_qmm_mma", .inputs = &.{ "X", "W", "SC", "BI", "ONE" }, .outputs = &.{"OUT"}, .source = simd_qmm.LOAD8 ++ simd_qmm.MMA ++ sq_undef, .header = simd_qmm.HEADER, .tg_bytes = 16384, .m1 = 1024, .m2 = 1024, .adapts = true };
const sq_frag_launch = metal.Launch{ .name = "msv_simd_qmm_frag", .inputs = &.{ "XF", "XS", "W", "SC", "BI", "ONE" }, .outputs = &.{"OUT"}, .source = simd_qmm.LOAD8 ++ simd_qmm.MMA_FRAG ++ sq_undef, .header = simd_qmm.HEADER, .tg_bytes = 16384, .m1 = 1024, .m2 = 1024, .adapts = true };
const sq_prep_launch = metal.Launch{ .name = "msv_simd_qmm_prep", .inputs = &.{ "X", "ONE" }, .outputs = &.{ "XF", "XS" }, .source = simd_qmm.LOAD8 ++ simd_qmm.PREP ++ sq_undef, .header = simd_qmm.HEADER, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// M1/M2 compiled thread limits (`offline.zig`), the least over a registration's shapes: the build's register use
/// moves with K, N and the bit width.
fn sqLimit(kind: SqKind, bits: u32, m: usize) u32 {
    return switch (kind) {
        .scalar => if (bits == 6) 640 else 192,
        .rows => if (m == 2) 832 else 224,
        .mma => if (bits == 6) 224 else 512,
        .frag => if (bits == 4) 704 else 640,
        .prep => 1024,
    };
}

fn sqLaunch(comptime kind: SqKind, comptime bits: u32, comptime m: usize) *const metal.Launch {
    return &struct {
        const v = blk: {
            var l = switch (kind) {
                .scalar => sq_scalar_launch,
                .rows => sq_rows_launch,
                .mma => sq_mma_launch,
                .frag => sq_frag_launch,
                .prep => sq_prep_launch,
            };
            l.m1 = sqLimit(kind, bits, m);
            l.m2 = l.m1;
            break :blk l;
        };
    }.v;
}

const SqKind = enum { scalar, rows, mma, frag, prep };

/// The wrapper's shape-only choices (simd_qmm.zig): K chunks, row tiles, 8-column tiles per simdgroup.
fn sqSplits(n: usize) usize {
    return if (n <= 2048) 16 else 8;
}

fn sqRowTiles(rows: usize) usize {
    return @min(2, (rows + 7) / 8);
}

fn sqTiles(n: usize, rows: usize, s: usize) usize {
    var nt: usize = if (n % 32 == 0) 4 else if (n % 16 == 0) 2 else 1;
    while (nt > 1 and s * ((rows + 7) / 8) * nt * 64 * 4 > 16384) nt /= 2;
    return nt;
}

/// The input pre-scale cancelling field `p`'s bit offset (`wpre`).
fn wpre(bits: u32, p: usize) f64 {
    if (bits == 6) return 1;
    const e: i32 = if (bits == 4) 4 * @as(i32, @intCast(p)) else 8 * @as(i32, @intCast(p % 4));
    return std.math.pow(f64, 2, @floatFromInt(-e));
}

/// One simd_qmm kernel at 4, 6 or 8 bits: rows are input rows, except where the kernel takes a fixed count `m`
/// (scalar one, rows two or three): there a shape's rows count 8-column output blocks (N = rows x 8) and the
/// non-finite fill plants the weight's scales by output row. `frag` reads the pre-scaled input fragments and group
/// sums `prep` writes (built here from x: the fragments exactly, the sums rounded to f32); `prep`'s outputs are those.
const Sq = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: SqKind,
    bits: u32,
    m: usize,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    w: Affine = undefined,
    args: [6]mlx.mlx_array = undefined,
    n_args: usize = 0,

    fn init(a: std.mem.Allocator, comptime kind: SqKind, comptime bits: u32, comptime m: usize) Sq {
        return .{ .rig = metal.Rig.init(a), .launch = sqLaunch(kind, bits, m), .kind = kind, .bits = bits, .m = m, .arena = .init(a) };
    }

    fn mRows(self: *const Sq, shape: Shape) usize {
        return if (self.m > 0) self.m else shape.rows;
    }

    fn nCols(self: *const Sq, shape: Shape) usize {
        return if (self.m > 0) shape.rows * 8 else shape.n;
    }

    pub fn deinit(self: *Sq) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn units(shape: Shape) usize {
        return (shape.rows + 7) / 8 * (shape.k / 64);
    }

    pub fn geometry(self: *const Sq, shape: Shape) ?metal.Geometry {
        const rows = self.mRows(shape);
        const n = self.nCols(shape);
        if (rows < 1 or rows > simd_qmm.MAX_ROWS or shape.k % 64 != 0 or n % 8 != 0 or n == 0) return null;
        const s = sqSplits(n);
        var g: metal.Geometry = undefined;
        switch (self.kind) {
            .prep => {
                const u = units(shape);
                g = .{ .grid = .{ cInt((u + 3) / 4 * 128), 1, 1 }, .tg = .{ 128, 1, 1 } };
                g.out(&.{cInt(u * 512)}, .float32);
                g.out(&.{ cInt(rows), cInt(shape.k / 64) }, .float32);
                g.int("K", shape.k);
                g.int("BITS", self.bits);
                return g;
            },
            .scalar, .rows => {
                if (self.kind == .scalar and rows != 1) return null;
                if (self.kind == .rows and (rows < 2 or rows > simd_qmm.SCALAR_ROWS_MAX or self.bits != 4 or rows * s * 76 * 4 > 32 * 1024)) return null;
                const nr: usize = if (n > 2048) 2 else 1;
                const sgs: usize = if (n > 2048) 2 else 8;
                const per = sgs * (32 / s) * nr;
                const rr: usize = if (self.kind == .rows) rows else 1;
                const xb: usize = if (self.kind == .rows) @max(s, (32 / rr) & ~(s - 1)) else 32;
                g = .{ .grid = .{ cInt((n + per - 1) / per * sgs * 32), 1, 1 }, .tg = .{ cInt(sgs * 32), 1, 1 } };
                g.out(&.{ cInt(rows), cInt(n) }, .bfloat16);
                g.int("K", shape.k);
                g.int("N", n);
                g.int("BITS", self.bits);
                g.int("S", s);
                g.int("SGS", sgs);
                g.int("NR", nr);
                g.int("XB", xb);
                g.int("RR", rr);
            },
            .mma, .frag => {
                if (self.kind == .frag and (rows < 2 or rows > simd_qmm.FRAGMENT_ROWS)) return null;
                const rt = sqRowTiles(rows);
                const nt = sqTiles(n, rt * 8, s);
                g = .{ .grid = .{ cInt((n + 8 * nt - 1) / (8 * nt) * s * 32), cInt((rows + 8 * rt - 1) / (8 * rt)), 1 }, .tg = .{ cInt(s * 32), 1, 1 } };
                g.out(&.{ cInt(rows), cInt(n) }, .bfloat16);
                g.int("K", shape.k);
                g.int("N", n);
                g.int("BITS", self.bits);
                g.int("S", s);
                g.int("NT", nt);
                g.int("RT", rt);
                g.int("SG", s);
            },
        }
        return g;
    }

    pub fn prepare(self: *Sq, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const m = self.mRows(shape);
        const n = self.nCols(shape);
        const plant_scales = fill == .nonfinite and self.m > 0;
        self.w = try fillAffine(a, rng, if (plant_scales) .random else fill, m, n, shape.k, self.bits, 64, .bfloat16, false, false);
        if (plant_scales) host.plant(self.w.q.scales, n, shape.k / 64);
        // prep's sums are of x alone: varied integers, so no order keeps fp16's or bf16's partial sums exact.
        if (self.kind == .prep and fill == .stress) self.w.x = try host.ints(a, rng, m * shape.k, -255, 255);
        const f = self.w;
        self.rig.reset();
        const rows = cInt(m);
        const k = cInt(shape.k);
        const one = try self.rig.input(.float32, &.{1}, &.{1});
        const weights = [3]mlx.mlx_array{
            try self.rig.input(.uint32, &.{ cInt(n), cInt(shape.k * self.bits / 32) }, f.q.words),
            try self.rig.input(.bfloat16, &.{ cInt(n), cInt(shape.k / 64) }, f.q.scales),
            try self.rig.input(.bfloat16, &.{ cInt(n), cInt(shape.k / 64) }, f.q.biases),
        };
        switch (self.kind) {
            .prep => {
                self.args[0..2].* = .{ try self.rig.input(.bfloat16, &.{ rows, k }, f.x), one };
                self.n_args = 2;
            },
            .frag => {
                const frags = try self.fragments(a);
                self.args = .{
                    try self.rig.input(.float32, &.{cInt(frags.xf.len)}, frags.xf),
                    try self.rig.input(.float32, &.{ rows, cInt(shape.k / 64) }, frags.xs),
                    weights[0],
                    weights[1],
                    weights[2],
                    one,
                };
                self.n_args = 6;
            },
            else => {
                self.args[0..5].* = .{ try self.rig.input(.bfloat16, &.{ rows, k }, f.x), weights[0], weights[1], weights[2], one };
                self.n_args = 5;
            },
        }
        return true;
    }

    /// prep's outputs from x: unit (tile t, group g)'s lane l, pass p holds rows 8t + fn and + 1 (clamped to the
    /// last) at input 64 g + 8 fm + p, pre-scaled; and each row's group sums, rounded to f32.
    fn fragments(self: *const Sq, a: std.mem.Allocator) !struct { xf: []f64, xs: []f64 } {
        const sh = self.shape;
        const g_n = sh.k / 64;
        const xf = try a.alloc(f64, units(sh) * 512);
        for (0..units(sh)) |u| {
            const t = u / g_n;
            const g = u % g_n;
            for (0..32) |lane| {
                const qid = lane / 4;
                const fm = (qid & 4) + (lane / 2) % 4;
                const fn_ = (qid & 2) * 2 + (lane % 2) * 2;
                for (0..8) |p| for (0..2) |e| {
                    const r = @min(8 * t + fn_ + e, sh.rows - 1);
                    xf[(u * 256 + 32 * p + lane) * 2 + e] = self.w.x[r * sh.k + 64 * g + 8 * fm + p] * wpre(self.bits, p);
                };
            }
        }
        const xs = try a.alloc(f64, sh.rows * g_n);
        for (0..sh.rows) |r| for (0..g_n) |g| {
            xs[r * g_n + g] = host.rt(.float32, host.sum(self.w.x[r * sh.k + 64 * g ..][0..64]));
        };
        return .{ .xf = xf, .xs = xs };
    }

    pub fn run(self: *Sq) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Sq) usize {
        const sh = self.shape;
        return if (self.kind == .prep) units(sh) * 512 + sh.rows * (sh.k / 64) else self.mRows(sh) * self.nCols(sh);
    }

    fn prepOutputs(self: *const Sq, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const a = self.rig.a;
        const frags = try self.fragments(a);
        defer a.free(frags.xf);
        defer a.free(frags.xs);
        @memcpy(want[0..frags.xf.len], frags.xf);
        if (mag) |m| @memset(m[0..frags.xf.len], 0);
        const sh = self.shape;
        const g_n = sh.k / 64;
        for (0..sh.rows) |r| for (0..g_n) |g| {
            const x = self.w.x[r * sh.k + 64 * g ..][0..64];
            const i = frags.xf.len + r * g_n + g;
            want[i] = if (acc) |ac| spec.sumIn(ac, order, x) else host.sum(x);
            if (mag) |m| m[i] = host.allowance(sumErr(64, host.sumAbs(x)));
        };
    }

    pub fn reference(self: *const Sq, want: []f64, mag: []f64) !void {
        if (self.kind == .prep) return self.prepOutputs(want, mag, null, .sequential);
        const sh = self.shape;
        const n = self.nCols(sh);
        for (0..self.mRows(sh)) |r| for (0..n) |j| {
            const y = factored(self.w.x[r * sh.k ..][0..sh.k], self.w, j, sh.k, 64, 64, null);
            want[r * n + j] = y.v;
            mag[r * n + j] = host.allowance(y.e);
        };
    }

    pub fn simulate(self: *const Sq, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        if (self.kind == .prep) return self.prepOutputs(out, null, acc, order);
        try simDots(self.rig.a, self.w.x, self.w.q.w, self.mRows(self.shape), self.nCols(self.shape), self.shape.k, acc, order, out);
    }

    /// prep then frag on the declared launches, for the drift guard against the wrapper's two dispatches.
    fn chained(self: *Sq, prep: *Sq) ![]u8 {
        try prep.run();
        const xf = prep.rig.output(0);
        const xs = prep.rig.output(1);
        var args = self.args;
        args[0] = xf;
        args[1] = xs;
        try self.rig.launch(self.launch, self.geometry(self.shape).?, args[0..6]);
        return self.rig.a.dupe(u8, self.rig.outputBytes());
    }

    fn wrapper(self: *Sq, x: mlx.mlx_array) ![]const u8 {
        const kind: simd_qmm.Kind = switch (self.kind) {
            .scalar => .scalar,
            .rows => .rows,
            .mma => .mma,
            .frag => .frag,
            .prep => unreachable,
        };
        const got = (try simd_qmm.qmmKind(kind, x, self.args[self.n_args - 4], self.args[self.n_args - 3], self.args[self.n_args - 2], self.bits, 64, self.rig.stream())) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};

// ── lane_qmm: row-exact matmul on the M5 tensor units ──

fn laneSource(comptime body: []const u8, comptime bits: u32, comptime tiled: bool) [:0]const u8 {
    return std.fmt.comptimePrint("#define BITS {d}\n#define TILED {d}\n", .{ bits, @intFromBool(tiled) }) ++ body;
}

/// Threadgroup bytes at SK 8: the narrow tile's slice partials (`part`, 7 x 16 x 32 floats) and, at 5 and 6 bits,
/// the widened codes (`stage_all`, 8 x 32 x 16 words); the 64-column tile's partials (7 x 16 x 64 floats); prefill's
/// dequantized block (`Ws`, 64 x 72 bf16).
fn laneLaunch(comptime kind: LaneKind, comptime bits: u32, comptime tiled: bool) metal.Launch {
    const plain: []const [*:0]const u8 = &.{ "X", "W", "SBt", "mdims" };
    const split: []const [*:0]const u8 = &.{ "X", "W", "ST", "BT", "mdims" };
    return .{
        .name = std.fmt.comptimePrint("msv_lane_qmm_{t}{s}_b{d}", .{ kind, if (tiled) "_tiled" else "", bits }),
        .inputs = if (tiled) split else plain,
        .outputs = &.{"Y"},
        .source = laneSource(switch (kind) {
            .narrow => lane_qmm.NARROW,
            .coop => lane_qmm.COOP,
            .prefill => lane_qmm.PREFILL,
        }, bits, tiled),
        .header = lane_qmm.HEADER,
        .tg_bytes = switch (kind) {
            .narrow => 7 * 16 * 32 * 4 + if (bits == 5 or bits == 6) 8 * 32 * 16 * 4 else 0,
            .coop => 7 * 16 * 64 * 4,
            .prefill => 64 * 72 * 2,
        },
        .m1 = 1024,
        .m2 = 1024,
        .offline = false,
        .min_gen = 17,
    };
}

const LaneKind = enum { narrow, coop, prefill };
/// A tiled weight's view: its first row in the whole buffer and the rows after it (0, 0: the whole weight).
const LaneCfg = struct { kind: LaneKind, bits: u32, gs: usize = 64, tiled: bool = false, r0: usize = 0, tail: usize = 0 };

/// lane_qmm's K slices for an (n, k) weight.
fn laneSplitK(n: usize, k: usize) usize {
    const tiles = (n + 31) / 32;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 1024 and (k / 64) / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// One lane_qmm kernel: rows are input rows, n the view's output columns. Tiled registrations hold the whole buffer
/// in `tileInPlace`'s order (column tiles of `nt`, each group one block; (scale, bias) pairs group-major, the first
/// half of the groups in the scales' buffer); `prefill` (MLX's NAX qmm with tiled addressing, past 128 rows)
/// replaces MLX's quantized_matmul bit for bit where MLX does not split K.
const Lane = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: LaneCfg,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    w: Affine = undefined,
    args: [5]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: LaneCfg) Lane {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .arena = .init(a) };
    }

    pub fn deinit(self: *Lane) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn ntot(self: *const Lane, shape: Shape) usize {
        return self.cfg.r0 + shape.n + self.cfg.tail;
    }

    /// The tiled column tile `tileInPlace` picks for the whole weight.
    fn nt(self: *const Lane, shape: Shape) usize {
        return if (self.cfg.bits == 4 and self.ntot(shape) % 64 == 0) 64 else 32;
    }

    pub fn geometry(self: *const Lane, shape: Shape) ?metal.Geometry {
        const c = self.cfg;
        const m = shape.rows;
        const n = shape.n;
        const k = shape.k;
        if (m < 1 or n == 0 or k % 64 != 0 or k % c.gs != 0 or n % 4 != 0) return null;
        if (c.tiled and (self.ntot(shape) % 32 != 0 or (k / 64) % 2 != 0)) return null;
        // The kind the wrapper picks for this view.
        const lane_rows = m <= lane_qmm.MAX_ROWS and (!c.tiled or c.r0 % self.nt(shape) == 0);
        const coop = if (c.tiled) self.nt(shape) == 64 and n % 64 == 0 else c.bits == 4 and n % 64 == 0;
        const kind: LaneKind = if (!lane_rows) .prefill else if (coop) .coop else .narrow;
        // prefill also runs below 128 rows (gates 1 and 4 take a few rows); the lane kinds only where picked.
        if (if (c.kind == .prefill) !c.tiled else kind != c.kind) return null;
        var g: metal.Geometry = undefined;
        if (c.kind == .prefill) {
            const bm: usize = if (m <= 32) 32 else 64;
            g = .{ .grid = .{ cInt((n + 63) / 64 * 128), cInt((m + bm - 1) / bm), 1 }, .tg = .{ 128, 1, 1 } };
            g.out(&.{ cInt(m), cInt(n) }, .bfloat16);
            g.int("TMR", bm);
            g.int("BM", bm);
            g.int("K", k);
            g.int("N", n);
            g.int("GS", 64);
            g.int("SK", 1);
            g.int("NTOT", self.ntot(shape));
            g.int("R0", c.r0);
            g.int("NT", self.nt(shape));
        } else {
            const sk = laneSplitK(n, k);
            const width: usize = if (c.kind == .coop) 64 else 32;
            const tmr: usize = if (m <= 16) 1 else 2;
            g = .{ .grid = .{ cInt((n + width - 1) / width * width * sk), cInt((m + 16 * tmr - 1) / (16 * tmr)), 1 }, .tg = .{ cInt(width * sk), 1, 1 } };
            g.out(&.{ cInt(m), cInt(n) }, .bfloat16);
            g.int("TMR", tmr);
            g.int("BM", tmr);
            g.int("K", k);
            g.int("N", n);
            g.int("GS", c.gs);
            g.int("SK", sk);
            g.int("NTOT", if (c.tiled) self.ntot(shape) else 0);
            g.int("R0", c.r0);
            g.int("NT", 0);
        }
        g.flag("ALIGNED_N", n % 64 == 0);
        return g;
    }

    pub fn prepare(self: *Lane, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const all = self.ntot(shape);
        self.w = try fillAffine(a, rng, fill, shape.rows, all, shape.k, c.bits, c.gs, .bfloat16, false, false);
        const f = self.w;
        const kg = shape.k / c.gs;
        const kw = shape.k * c.bits / 32;
        // (scale, bias) pairs group-major over the whole weight.
        const sbt = try a.alloc(f64, kg * all * 2);
        for (0..kg) |g| for (0..all) |j| {
            sbt[(g * all + j) * 2] = f.q.scales[j * kg + g];
            sbt[(g * all + j) * 2 + 1] = f.q.biases[j * kg + g];
        };
        self.rig.reset();
        const x = try self.rig.input(.bfloat16, &.{ cInt(shape.rows), cInt(shape.k) }, f.x);
        const mdims = try self.rig.input(.int32, &.{1}, &.{@floatFromInt(shape.rows)});
        if (!c.tiled) {
            self.args[0..4].* = .{
                x,
                try self.rig.input(.uint32, &.{ cInt(all), cInt(kw) }, f.q.words),
                try self.rig.input(.bfloat16, &.{ cInt(kg), cInt(all), 2 }, sbt),
                mdims,
            };
            return true;
        }
        const t = self.nt(shape);
        const wpg = 2 * c.bits;
        const tiled = try a.alloc(f64, all * kw);
        for (0..all) |j| for (0..kg) |g| for (0..wpg) |i| {
            tiled[(((j / t) * kg + g) * t + j % t) * wpg + i] = f.q.words[j * kw + g * wpg + i];
        };
        const half = kg / 2 * all * 2;
        self.args = .{
            x,
            try self.rig.input(.uint32, &.{ cInt(all), cInt(kw) }, tiled),
            try self.rig.input(.bfloat16, &.{ cInt(all), cInt(kg) }, sbt[0..half]),
            try self.rig.input(.bfloat16, &.{ cInt(all), cInt(kg) }, sbt[half..]),
            mdims,
        };
        return true;
    }

    pub fn run(self: *Lane) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..if (self.cfg.tiled) 5 else 4]);
    }

    pub fn count(self: *const Lane) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *const Lane, want: []f64, mag: []f64) !void {
        const sh = self.shape;
        const c = self.cfg;
        const wt = try self.rig.a.alloc(f64, sh.k);
        defer self.rig.a.free(wt);
        const we = try self.rig.a.alloc(f64, sh.k);
        defer self.rig.a.free(we);
        for (0..sh.n) |j| {
            const row = c.r0 + j;
            // prefill dequantizes each weight to bf16 (code * s + b in f32) before the tensor-unit product.
            if (c.kind == .prefill) for (wt, we, self.w.q.w[row * sh.k ..][0..sh.k]) |*t, *e, v| {
                const r = host.round(.bfloat16, .{ .v = v, .e = U * @abs(v) });
                t.* = r.v;
                e.* = r.e;
            };
            for (0..sh.rows) |r| {
                const x = self.w.x[r * sh.k ..][0..sh.k];
                const y = if (c.kind == .prefill) decoded(x, wt, we) else factored(x, self.w, row, sh.k, c.gs, c.gs, null);
                want[r * sh.n + j] = y.v;
                mag[r * sh.n + j] = host.allowance(y.e);
            }
        }
    }

    pub fn simulate(self: *const Lane, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const sh = self.shape;
        try simDots(self.rig.a, self.w.x, self.w.q.w[self.cfg.r0 * sh.k ..], sh.rows, sh.n, sh.k, acc, order, out);
    }

    /// The untiled view's weight, scales and biases as MLX packs them.
    fn untiled(self: *Lane, whole: bool) ![3]mlx.mlx_array {
        const sh = self.shape;
        const c = self.cfg;
        const r0 = if (whole) 0 else c.r0;
        const rows = if (whole) self.ntot(sh) else sh.n;
        const kg = sh.k / c.gs;
        const kw = sh.k * c.bits / 32;
        const q = self.w.q;
        return .{
            try self.rig.input(.uint32, &.{ cInt(rows), cInt(kw) }, q.words[r0 * kw ..][0 .. rows * kw]),
            try self.rig.input(.bfloat16, &.{ cInt(rows), cInt(kg) }, q.scales[r0 * kg ..][0 .. rows * kg]),
            try self.rig.input(.bfloat16, &.{ cInt(rows), cInt(kg) }, q.biases[r0 * kg ..][0 .. rows * kg]),
        };
    }

    /// MLX's quantized_matmul on the untiled view (prefill).
    pub fn replaced(self: *Lane) ![]const u8 {
        const u = try self.untiled(false);
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_quantized_matmul(&y, self.args[0], u[0], u[1], u[2], true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(cInt(self.cfg.bits)), "affine", self.rig.stream()));
        return self.rig.bytesOf(&.{y});
    }

    /// The shipped path: `qmm` on the untiled weight, or `tileInPlace` on a fresh untiled buffer, then `tiledQmm` on
    /// the view.
    fn wrapper(self: *Lane) ![]const u8 {
        const s = self.rig.stream();
        const c = self.cfg;
        const sh = self.shape;
        if (!c.tiled) {
            defer lane_qmm.release(0);
            const u = try self.untiled(true);
            const got = (try lane_qmm.qmm(self.args[0], u[0], u[1], u[2], c.bits, @intCast(c.gs), s)) orelse return error.Declined;
            defer _ = mlx.mlx_array_free(got);
            return self.rig.bytesOf(&.{got});
        }
        const owner: usize = 0x1a7e;
        defer lane_qmm.release(owner);
        const u = try self.untiled(true);
        const widths = [_]c_int{ cInt(c.r0), cInt(sh.n), cInt(c.tail) };
        if (try lane_qmm.tileInPlace(owner, u[0], u[1], u[2], c.bits, 64, if (c.r0 + c.tail > 0) &widths else &.{}, s) == 0) return error.Declined;
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        const kw = cInt(sh.k * c.bits / 32);
        try mlx.check(mlx.mlx_slice(&view, u[0], &.{ cInt(c.r0), 0 }, 2, &.{ cInt(c.r0 + sh.n), kw }, 2, &.{ 1, 1 }, 2, s));
        const got = (try lane_qmm.tiledQmm(self.args[0], view, s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};

// ── NAX2 and the int8-activation prefill: 2-bit Prism packs on the M5 tensor units ──

/// `red_all[2 * 16 * 32]` floats.
const nax2_launch = metal.Launch{ .name = "nax2_qmm_m16", .inputs = &.{ "x", "w", "scales", "biases" }, .outputs = &.{"y"}, .source = qmv_nax2.NARROW_SOURCE, .header = qmv_nax2.HEADER, .tg_bytes = 2 * 16 * 32 * 4, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 };
const int8_launch = metal.Launch{ .name = "msv_qmm_int8_prefill", .inputs = &.{ "xq", "w", "scales", "biases", "ascale", "rsum" }, .outputs = &.{"y"}, .source = qmm_int8.SOURCE, .header = qmm_int8.HEADER, .tg_bytes = 0, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 };

/// NAX2's narrow tile, 6..16 rows of a 2-bit group-128 weight (any bias): each weight dequantized into T (f16 in
/// half arithmetic, one rounding; bf16 through f32) before the tensor unit's products. Rows are input rows.
const Nax2 = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    w: Affine = undefined,
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, dt: mlx.mlx_dtype) Nax2 {
        return .{ .rig = metal.Rig.init(a), .launch = &nax2_launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Nax2) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Nax2, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.rows > 16 or shape.k % 128 != 0 or shape.n % 32 != 0 or shape.n == 0) return null;
        var g = metal.Geometry{ .grid = .{ 128, cInt((shape.n + 63) / 64), cInt((shape.rows + 15) / 16) }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ cInt(shape.rows), cInt(shape.n) }, self.dt);
        g.int("HDQ", @intFromBool(self.dt == .float16 and qmv_nax2.half_dq));
        g.dtype("T", self.dt);
        return g;
    }

    pub fn prepare(self: *Nax2, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        var prng = std.Random.DefaultPrng.init(seed);
        self.w = try fillAffine(self.arena.allocator(), prng.random(), fill, shape.rows, shape.n, shape.k, 2, 128, self.dt, false, false);
        const f = self.w;
        self.rig.reset();
        const kg = cInt(shape.k / 128);
        self.args = .{
            try self.rig.input(self.dt, &.{ cInt(shape.rows), cInt(shape.k) }, f.x),
            try self.rig.input(.uint32, &.{ cInt(shape.n), cInt(shape.k / 16) }, f.q.words),
            try self.rig.input(self.dt, &.{ cInt(shape.n), kg }, f.q.scales),
            try self.rig.input(self.dt, &.{ cInt(shape.n), kg }, f.q.biases),
        };
        return true;
    }

    pub fn run(self: *Nax2) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Nax2) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *const Nax2, want: []f64, mag: []f64) !void {
        const sh = self.shape;
        const a = self.rig.a;
        const wt = try a.alloc(f64, sh.k);
        defer a.free(wt);
        const we = try a.alloc(f64, sh.k);
        defer a.free(we);
        for (0..sh.n) |j| {
            for (wt, we, self.w.q.w[j * sh.k ..][0..sh.k]) |*t, *e, v| {
                // f16: one half fma (code * s + b, exact inputs); bf16: f32 code * s + b, then bf16.
                const r = host.round(self.dt, .{ .v = v, .e = if (self.dt == .float16) 0 else U * @abs(v) });
                t.* = r.v;
                e.* = r.e;
            }
            for (0..sh.rows) |r| {
                const y = decoded(self.w.x[r * sh.k ..][0..sh.k], wt, we);
                want[r * sh.n + j] = y.v;
                mag[r * sh.n + j] = host.allowance(y.e);
            }
        }
    }

    pub fn simulate(self: *const Nax2, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try simDots(self.rig.a, self.w.x, self.w.q.w, self.shape.rows, self.shape.n, self.shape.k, acc, order, out);
    }

    fn wrapper(self: *Nax2) ![]const u8 {
        const x = self.args;
        const got = (try qmv_nax2.qmm(x[0], x[1], x[2], x[3], 2, 128, self.rig.stream())) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};

/// The int8-activation prefill (opt-in, LOSSY by its own design: the loss is the quantizer's, upstream of this
/// kernel): int8 activation codes against the 2-bit codes on the tensor unit (exact int32 group products C), then
/// per group as * (s C + b rs) in f32 (`neg`, a ternary pack: s (as C - as rs)). A shape's rows count 32-row tiles
/// (MPAD = rows x 32); the codes, their per-group scales `as` and sums `rs` are inputs (the quantizer's outputs), the
/// codes stored in the kernel's K order (position 4b + kq of a 16-block at 4kq + b). Non-finite values go in `as`.
const Int8 = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    neg: bool,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    w: Affine = undefined,
    xq: []f64 = &.{},
    as: []f64 = &.{},
    rs: []f64 = &.{},
    args: [6]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, dt: mlx.mlx_dtype, neg: bool) Int8 {
        return .{ .rig = metal.Rig.init(a), .launch = &int8_launch, .dt = dt, .neg = neg, .arena = .init(a) };
    }

    pub fn deinit(self: *Int8) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn mpad(shape: Shape) usize {
        return shape.rows * 32;
    }

    pub fn geometry(self: *const Int8, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.k % 128 != 0 or shape.n == 0) return null;
        const m = mpad(shape);
        var g = metal.Geometry{ .grid = .{ cInt(128 * ((shape.n + 127) / 128)), cInt(m / 32), 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ cInt(m), cInt(shape.n) }, self.dt);
        g.dtype("T", self.dt);
        g.int("MPAD", m);
        g.flag("NEG", self.neg);
        g.dtype("ST", self.dt);
        g.flag("FULL", shape.n % 128 == 0);
        return g;
    }

    pub fn prepare(self: *Int8, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const m = mpad(shape);
        const k = shape.k;
        const kg = k / 128;
        self.w = try fillAffine(a, rng, if (fill == .nonfinite) .random else fill, m, shape.n, k, 2, 128, self.dt, self.neg, false);
        const f = self.w;
        self.xq = try host.ints(a, rng, m * k, -127, 127);
        self.as = try host.uniform(a, rng, m * kg, 1e-3, 1e-2, .float32);
        if (fill == .stress) {
            stressX(self.xq, m, k, 127);
            @memset(self.as, 1);
        }
        if (fill == .nonfinite) host.plant(self.as, m, kg);
        self.rs = try a.alloc(f64, m * kg);
        for (0..m) |r| for (0..kg) |g| {
            self.rs[r * kg + g] = host.sum(self.xq[r * k + g * 128 ..][0..128]);
        };
        // The kernel's K order inside each 16-block.
        const stored = try a.alloc(f64, m * k);
        for (0..m) |r| for (0..k / 16) |b| for (0..16) |p| {
            stored[r * k + b * 16 + (p & 3) * 4 + ((p >> 2) & 3)] = self.xq[r * k + b * 16 + p];
        };
        // Scales and biases group-major ([KG, N]); a ternary pack passes its scales twice.
        const st = try a.alloc(f64, kg * shape.n);
        const bt = try a.alloc(f64, kg * shape.n);
        for (0..shape.n) |j| for (0..kg) |g| {
            st[g * shape.n + j] = f.q.scales[j * kg + g];
            bt[g * shape.n + j] = if (self.neg) f.q.scales[j * kg + g] else f.q.biases[j * kg + g];
        };
        self.rig.reset();
        self.args = .{
            try self.rig.input(.int8, &.{ cInt(m), cInt(k) }, stored),
            try self.rig.input(.uint32, &.{ cInt(shape.n), cInt(k / 16) }, f.q.words),
            try self.rig.input(self.dt, &.{ cInt(kg), cInt(shape.n) }, st),
            try self.rig.input(self.dt, &.{ cInt(kg), cInt(shape.n) }, bt),
            try self.rig.input(.float32, &.{ cInt(m), cInt(kg) }, self.as),
            try self.rig.input(.float32, &.{ cInt(m), cInt(kg) }, self.rs),
        };
        return true;
    }

    pub fn run(self: *Int8) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Int8) usize {
        return mpad(self.shape) * self.shape.n;
    }

    /// Output (r, j): its per-group terms (the f32 accumulation's), or float64 with the allowance.
    fn outputs(self: *const Int8, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const sh = self.shape;
        const k = sh.k;
        const kg = k / 128;
        const terms = try self.rig.a.alloc(f64, kg);
        defer self.rig.a.free(terms);
        for (0..mpad(sh)) |r| for (0..sh.n) |j| {
            var size: f64 = 0;
            for (terms, 0..) |*t, g| {
                var c: f64 = 0;
                for (g * 128..(g + 1) * 128) |i| c += self.xq[r * k + i] * @as(f64, @floatFromInt(self.w.codes[j * k + i]));
                const s = self.w.q.scales[j * kg + g];
                const b = self.w.q.biases[j * kg + g];
                const as = self.as[r * kg + g];
                const rs = self.rs[r * kg + g];
                t.* = if (self.neg) s * (as * c - as * rs) else as * (s * c + b * rs);
                size += if (self.neg) @abs(s) * (@abs(as * c) + @abs(as * rs)) else @abs(as) * (@abs(s * c) + 2 * @abs(b * rs));
            }
            const i = r * sh.n + j;
            want[i] = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms);
            if (mag) |m| m[i] = host.allowance(sumErr(kg, size));
        };
    }

    pub fn reference(self: *const Int8, want: []f64, mag: []f64) !void {
        try self.outputs(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Int8, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputs(out, null, acc, order);
    }

    /// The shipped route from a float activation: its quantizer's codes, scales and sums into the declared launch,
    /// against `qmm_int8.qmm` on the same activation.
    fn chainedAndShipped(self: *Int8) !struct { ours: []u8, theirs: []const u8 } {
        const s = self.rig.stream();
        const sh = self.shape;
        const m = cInt(mpad(sh));
        const k = cInt(sh.k);
        const kg = sh.k / 128;
        var prng = std.Random.DefaultPrng.init(11);
        const xv = try host.uniform(self.rig.a, prng.random(), mpad(sh) * sh.k, -2, 2, self.dt);
        defer self.rig.a.free(xv);
        const x = try self.rig.input(self.dt, &.{ m, k }, xv);
        var q = try qmm_int8.quantizeRows(x, m, k, s);
        defer q.deinit();
        var args = self.args;
        args[0] = q.xq;
        args[4] = q.ascale;
        args[5] = q.rsum;
        try self.rig.launch(self.launch, self.geometry(sh).?, &args);
        const ours = try self.rig.a.dupe(u8, self.rig.outputBytes());
        errdefer self.rig.a.free(ours);
        const sc = try self.rig.input(self.dt, &.{ cInt(sh.n), cInt(kg) }, self.w.q.scales);
        const bi = try self.rig.input(self.dt, &.{ cInt(sh.n), cInt(kg) }, self.w.q.biases);
        var cache: qmm_int8.Cache = .{};
        defer cache.deinit();
        const got = (try qmm_int8.qmm(&cache, x, self.args[1], sc, bi, 2, 128, self.neg, s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return .{ .ours = ours, .theirs = try self.rig.bytesOf(&.{got}) };
    }
};

// ── the int8 route's quantizers: activation codes, group scales and code sums ──

const iq_outputs: []const [*:0]const u8 = &.{ "xq", "ascale", "rsum" };
// Only the int8 route dispatches them, and it runs on NAX GPUs (M5 up).
const iq_launch = metal.Launch{ .name = "msv_int8_quant", .inputs = &.{"x"}, .outputs = iq_outputs, .source = qmm_int8.QUANT_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024, .min_gen = 17 };
const rotq_launch = metal.Launch{ .name = "msv_int8_rotate_quant", .inputs = &.{ "x", "signs" }, .outputs = iq_outputs, .source = qmm_int8.ROTQ_SOURCE, .tg_bytes = 1024 * 4, .m1 = 1024, .m2 = 1024, .min_gen = 17 };

/// The quantized rows: the input's, padded to the prefill kernel's 32-row tiles (`qmm_int8.qmm`'s MPAD).
fn iqPad(rows: usize) usize {
    return (rows + 31) / 32 * 32;
}

fn iqCodes(s: Shape) usize {
    return iqPad(s.rows) * s.k;
}

fn iqGroups(s: Shape) usize {
    return iqPad(s.rows) * (s.k / 128);
}

/// Metal's `rint`: to nearest, a tie to even.
fn rintEven(t: f32) f32 {
    const r = @round(t);
    return if (@abs(t - @trunc(t)) == 0.5) 2 * @round(t / 2) else r;
}

/// `msv_int8_quant` (`rotate`: `msv_int8_rotate_quant`, H(signs x) per 1024 block first). A shape's rows are the
/// input's; the outputs cover them padded to 32 (zeros). f16 only: the route runs on Hadamard packs.
const Iq = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    rotate: bool,
    shape: Shape = undefined,
    x: []f64 = &.{},
    signs: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, rotate: bool) Iq {
        return .{ .rig = metal.Rig.init(a), .launch = if (rotate) &rotq_launch else &iq_launch, .rotate = rotate };
    }

    pub fn deinit(self: *Iq) void {
        const a = self.rig.a;
        self.rig.deinit();
        a.free(self.x);
        a.free(self.signs);
    }

    pub fn geometry(self: *const Iq, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.k == 0 or shape.k % @as(usize, if (self.rotate) 1024 else 128) != 0) return null;
        const m = cInt(iqPad(shape.rows));
        const kg = cInt(shape.k / 128);
        var g: metal.Geometry = if (self.rotate)
            .{ .grid = .{ cInt(shape.k / 1024 * 128), m, 1 }, .tg = .{ 128, 1, 1 } }
        else
            .{ .grid = .{ 32 * kg, m, 1 }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ m, cInt(shape.k) }, .int8);
        g.out(&.{ m, kg }, .float32);
        g.out(&.{ m, kg }, .float32);
        g.dtype("T", .float16);
        return g;
    }

    pub fn prepare(self: *Iq, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        a.free(self.x);
        a.free(self.signs);
        self.x = &.{};
        self.signs = &.{};
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.rotate) {
            // Eighths up to 16: every f32 butterfly sum is exact, so a rotated value's one rounding is to T.
            self.x = try host.ints(a, rng, shape.rows * shape.k, -128, 128);
            for (self.x) |*v| v.* /= 8;
            self.signs = try host.ints(a, rng, shape.k, 0, 1);
            for (self.signs) |*v| v.* = 2 * v.* - 1;
        } else self.x = try host.uniform(a, rng, shape.rows * shape.k, -4, 4, .float16);
        if (fill == .nonfinite) host.plant(self.x, shape.rows, shape.k);
        self.rig.reset();
        self.args[0] = try self.rig.input(.float16, &.{ cInt(shape.rows), cInt(shape.k) }, self.x);
        if (self.rotate) self.args[1] = try self.rig.input(.float16, &.{cInt(shape.k)}, self.signs);
        return true;
    }

    pub fn run(self: *Iq) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..if (self.rotate) 2 else 1]);
    }

    pub fn count(self: *const Iq) usize {
        return iqCodes(self.shape) + 2 * iqGroups(self.shape);
    }

    /// Per 128-group of a row: v as the kernel reads it (rotated: H(signs x) / 32 rounded to f16), the scale
    /// max(max |v| / 127, 1e-20) and each code rint(v / scale) clamped to +-127, both in f32 as the kernel computes
    /// them (MLX builds in safe math: IEEE division), the codes at the kernel's K order, their sum beside the scale.
    /// A NaN in a group makes its scale, codes and sum NaN; an infinite value's code is NaN (Inf / Inf).
    pub fn reference(self: *const Iq, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const sh = self.shape;
        const k = sh.k;
        const kg = k / 128;
        const codes = want[0..iqCodes(sh)];
        const scales = want[codes.len..][0..iqGroups(sh)];
        const sums = want[codes.len + scales.len ..][0..scales.len];
        const row = try self.rig.a.alloc(f64, k);
        defer self.rig.a.free(row);
        const nan = std.math.nan(f64);
        for (0..iqPad(sh.rows)) |r| {
            if (r < sh.rows) @memcpy(row, self.x[r * k ..][0..k]) else @memset(row, 0);
            if (self.rotate) {
                for (row, self.signs) |*v, s| v.* *= s;
                var b: usize = 0;
                while (b < k) : (b += 1024) host.fwht(row[b..][0..1024]);
                for (row) |*v| v.* = spec.roundTo(.f16, v.* / 32);
            }
            for (0..kg) |g| {
                const v = row[g * 128 ..][0..128];
                var amax: f64 = 0;
                var has_nan = false;
                for (v) |x| {
                    if (std.math.isNan(x)) has_nan = true else amax = @max(amax, @abs(x));
                }
                const sc = @max(@as(f32, @floatCast(amax)) / 127.0, 1.0e-20);
                var rs: f64 = 0;
                for (v, 0..) |x, p| {
                    const t = @as(f32, @floatCast(x)) / sc;
                    const q: f64 = if (has_nan or std.math.isNan(t)) nan else std.math.clamp(rintEven(t), -127, 127);
                    codes[r * k + g * 128 + (p & ~@as(usize, 15)) + (p & 3) * 4 + ((p >> 2) & 3)] = q;
                    rs += q;
                }
                scales[r * kg + g] = if (has_nan) nan else sc;
                sums[r * kg + g] = rs;
            }
        }
    }

    fn wrapper(self: *Iq) ![]const u8 {
        const s = self.rig.stream();
        const m = cInt(iqPad(self.shape.rows));
        const k = cInt(self.shape.k);
        var q = if (self.rotate) try qmm_int8.rotateQuantizeRows(self.args[0], self.args[1], m, k, s) else try qmm_int8.quantizeRows(self.args[0], m, k, s);
        defer q.deinit();
        return self.rig.bytesOf(&.{ q.xq, q.ascale, q.rsum });
    }
};

fn shp(rows: usize, n: usize, k: usize) Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

/// The affine matvecs' spec: f32 sums whose allowance the reference carries; `block` the k a pass covers.
fn affineSpec(comptime name: []const u8, out: spec.Elem, served: []const Shape, tiles: [3]spec.Tile, replaces: ?[]const u8) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = out,
        .tiles = tiles,
        .served = served,
        .replaces = replaces,
    };
}

/// rows 1..16 around the 8 where rowqmv's threadgroup halves; n in 8s; k around one pass of 32 lanes.
fn rqTiles(bits: u32, gs: usize) [3]spec.Tile {
    const step = @max(gs, 64);
    return .{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = if (bits == 4) 512 else 256, .step = step, .min = step } };
}

const one_row_qmm = "MLX quantized_matmul, one call per input row";

/// Nemotron-H's rows (hidden 2688, q 4096, shared-expert 3712, routed 1856 wide, 128 experts top-6) and Qwen3.8-27B's
/// (5120, 17408; dense qwen3_5 under DFlash) at the quantizations the row-exact decode reads with rowqmv: group 128
/// anywhere, group 32 and 64 where the lane and simd kernels decline. Where K is a whole number of 32-lane passes
/// (512 inputs at 4 bits, 256 at 6 and 8) each row is MLX's own one-row quantized_matmul, bit for bit; Nemotron-H's
/// 2688 and 1856 take the tail pass, which is not stock's bits (seen at 4, 6 and 8 bits), so those are held to
/// float64.
const rq_registrations = .{
    .{ "msv_rowqmv bf16 (4-bit, group 64, whole passes)", RqCfg{ .kind = .plain, .bits = 4, .gs = 64 }, &[_]Shape{ shp(1, 512, 5120), shp(16, 264, 4096), shp(3, 136, 17408) }, true },
    .{ "msv_rowqmv bf16 (4-bit, group 64)", RqCfg{ .kind = .plain, .bits = 4, .gs = 64 }, &[_]Shape{ shp(1, 512, 2688), shp(16, 264, 4096), shp(7, 136, 1856) }, false },
    .{ "msv_rowqmv bf16 (4-bit, group 128, whole passes)", RqCfg{ .kind = .plain, .bits = 4, .gs = 128 }, &[_]Shape{ shp(1, 512, 5120), shp(16, 264, 4096) }, true },
    .{ "msv_rowqmv bf16 (4-bit, group 128)", RqCfg{ .kind = .plain, .bits = 4, .gs = 128 }, &[_]Shape{ shp(1, 512, 2688), shp(16, 264, 4096) }, false },
    .{ "msv_rowqmv bf16 (6-bit, group 32, whole passes)", RqCfg{ .kind = .plain, .bits = 6, .gs = 32 }, &[_]Shape{ shp(1, 512, 5120), shp(9, 136, 4096) }, true },
    .{ "msv_rowqmv bf16 (6-bit, group 32)", RqCfg{ .kind = .plain, .bits = 6, .gs = 32 }, &[_]Shape{ shp(1, 512, 2688), shp(9, 136, 1856) }, false },
    .{ "msv_rowqmv bf16 (8-bit, group 64, whole passes)", RqCfg{ .kind = .plain, .bits = 8, .gs = 64 }, &[_]Shape{ shp(1, 512, 5120), shp(16, 264, 4096) }, true },
    .{ "msv_rowqmv bf16 (8-bit, group 64)", RqCfg{ .kind = .plain, .bits = 8, .gs = 64 }, &[_]Shape{ shp(1, 512, 2688), shp(16, 264, 4096) }, false },
    .{ "msv_rowqmv bf16 (4-bit, group 64, relu2)", RqCfg{ .kind = .relu2, .bits = 4, .gs = 64 }, &[_]Shape{ shp(1, 512, 2688), shp(5, 512, 2688) }, false },
};

const rq_expert_registrations = .{
    .{ "msv_rowqmv_expert_up bf16 (4-bit, group 64, relu2)", RqCfg{ .kind = .up, .bits = 4, .gs = 64 }, &rq_up_launch, &[_]Shape{ shp(1, 256, 2688), shp(2, 256, 2688) } },
    .{ "msv_rowqmv_expert_up bf16 (8-bit, group 64, relu2)", RqCfg{ .kind = .up, .bits = 8, .gs = 64 }, &rq_up_launch, &[_]Shape{ shp(1, 256, 2688), shp(2, 256, 2688) } },
    .{ "msv_rowqmv_expert_down bf16 (4-bit, group 64)", RqCfg{ .kind = .down, .bits = 4, .gs = 64 }, &rq_down_launch, &[_]Shape{ shp(1, 256, 1856), shp(2, 256, 1856) } },
    .{ "msv_rowqmv_expert_down bf16 (8-bit, group 64)", RqCfg{ .kind = .down, .bits = 8, .gs = 64 }, &rq_down_launch, &[_]Shape{ shp(1, 256, 1856), shp(2, 256, 1856) } },
};

/// Prism Bonsai 2 (hidden 2048 at 1.7B, 5120 and MLP 17408 at 27B) from N 2048, where `planFor` stops leaving them to
/// stock. h2: one row on unmeasured GPUs (legacy) and the Hadamard fallback; rows: each measured generation's
/// geometries (R4 G2 on M1 and odd widths on M4, R2 G4 one row on M4, R2 G8 / R4 G8 on M5 and legacy 2..3 rows);
/// mma: M4 from 5 rows.
const q2_registrations = .{
    .{ "msv_qmv2_h2 f16 (one row)", Q2Cfg{ .kind = .h2 }, &[_]Shape{ shp(256, 1, 5120), shp(256, 1, 17408) } },
    .{ "msv_qmv2_h2 bf16 (one row)", Q2Cfg{ .kind = .h2, .dt = .bfloat16 }, &[_]Shape{ shp(256, 1, 5120), shp(256, 1, 17408) } },
    .{ "msv_qmv2_rows f16 (R4 G2)", Q2Cfg{ .kind = .rows }, &[_]Shape{ shp(1, 2048, 5120), shp(3, 2048, 2048), shp(8, 2048, 2048) } },
    .{ "msv_qmv2_rows bf16 (R4 G2)", Q2Cfg{ .kind = .rows, .dt = .bfloat16 }, &[_]Shape{ shp(1, 2048, 5120), shp(3, 2048, 2048) } },
    .{ "msv_qmv2_rows f16 (R2 G4)", Q2Cfg{ .kind = .rows, .rows = .{ .r = 2, .g = 4 } }, &[_]Shape{shp(1, 2048, 5120)} },
    .{ "msv_qmv2_rows f16 (R2 G8)", Q2Cfg{ .kind = .rows, .rows = .{ .r = 2, .g = 8 } }, &[_]Shape{ shp(2, 2048, 5120), shp(8, 2048, 2048) } },
    .{ "msv_qmv2_rows f16 (R4 G8)", Q2Cfg{ .kind = .rows, .rows = .{ .r = 4, .g = 8 } }, &[_]Shape{ shp(3, 2048, 2048), shp(2, 2048, 17408) } },
    .{ "msv_qmv2_mma f16 (RB 1, 4 simdgroups)", Q2Cfg{ .kind = .mma }, &[_]Shape{ shp(5, 2048, 2048), shp(8, 2048, 5120), shp(16, 2048, 2048) } },
    .{ "msv_qmv2_mma bf16 (RB 1, 4 simdgroups)", Q2Cfg{ .kind = .mma, .dt = .bfloat16 }, &[_]Shape{ shp(5, 2048, 2048), shp(16, 2048, 2048) } },
};

fn q2Tiles(kind: Q2Kind) [3]spec.Tile {
    return switch (kind) {
        .h2 => .{ .{}, .{}, .{ .size = 512, .step = 512, .min = 512 } },
        .rows => .{ .{ .size = 4, .step = 1, .min = 1 }, .{}, .{ .size = 512, .step = 512, .min = 512 } },
        .mma => .{ .{ .size = 8, .step = 1, .min = 5 }, .{}, .{ .size = 128, .step = 128, .min = 128 } },
    };
}

/// DeepSeek-V4.1's and MiMo's MXFP8 projections on M4 and M5 (hidden 4096, a tensor-parallel rank's 320-wide expert
/// half, K not a whole 512 step), 1..8 rows.
const fp8_served = [_]Shape{ shp(1, 640, 4096), shp(8, 640, 2560), shp(3, 512, 320) };

/// The row-exact decode's projections on M1..M4 (and wherever lane_qmm declines): Nemotron-H (2688, 1856) and dense
/// qwen3_5 widths (4096, 1024), both K-chunk counts (16 to N 2048, 8 past it). scalar: one row; rows: 2..3 at 4 bits;
/// frag (with prep): 2..8; mma: past 8, and one row where scalar's bits are not mma's.
const sq_registrations = .{
    .{ SqKind.scalar, 1, &[_]Shape{ shp(256, 1, 4096), shp(512, 1, 2688) }, [3]spec.Tile{ .{ .size = 2, .step = 1, .min = 1 }, .{}, .{ .size = 1024, .step = 64, .min = 64 } } },
    .{ SqKind.rows, 2, &[_]Shape{ shp(256, 1, 4096), shp(512, 1, 2688) }, [3]spec.Tile{ .{ .size = 2, .step = 1, .min = 1 }, .{}, .{} } },
    .{ SqKind.rows, 3, &[_]Shape{ shp(256, 1, 4096), shp(512, 1, 2688) }, [3]spec.Tile{ .{ .size = 2, .step = 1, .min = 1 }, .{}, .{} } },
    .{ SqKind.mma, 0, &[_]Shape{ shp(16, 1024, 2688), shp(12, 4096, 1024), shp(9, 1024, 1856) }, [3]spec.Tile{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 32, .step = 8, .min = 8 }, .{} } },
    .{ SqKind.frag, 0, &[_]Shape{ shp(2, 2048, 4096), shp(8, 1024, 2688) }, [3]spec.Tile{ .{ .size = 8, .step = 1, .min = 2 }, .{}, .{} } },
    .{ SqKind.prep, 0, &[_]Shape{ shp(2, 8, 4096), shp(8, 8, 2688) }, [3]spec.Tile{ .{ .size = 8, .step = 1, .min = 1 }, .{}, .{} } },
};

/// The row-exact decode on M5 (DFlash verify windows up to 16 rows, any width to 128): Nemotron-H and dense
/// qwen3_5 widths, the four bit widths, groups of 64 and 32, in MLX's layout and tiled in place (the whole weight and
/// a view of a joined one); prefill past 128 rows and for views off the tile grid, at shapes where MLX's own qmm does
/// not split K (its bits are prefill's there).
const lane_registrations = .{
    .{ LaneCfg{ .kind = .narrow, .bits = 4 }, &[_]Shape{ shp(128, 72, 1024), shp(1, 136, 4096), shp(16, 264, 2688) } },
    .{ LaneCfg{ .kind = .narrow, .bits = 4, .gs = 32 }, &[_]Shape{ shp(128, 72, 1024), shp(1, 136, 4096), shp(16, 264, 2688) } },
    .{ LaneCfg{ .kind = .narrow, .bits = 5 }, &[_]Shape{ shp(128, 128, 1024), shp(1, 512, 4096), shp(16, 264, 2688) } },
    .{ LaneCfg{ .kind = .narrow, .bits = 6 }, &[_]Shape{ shp(128, 128, 1024), shp(1, 512, 4096), shp(16, 264, 2688) } },
    .{ LaneCfg{ .kind = .narrow, .bits = 8 }, &[_]Shape{ shp(128, 128, 1024), shp(1, 512, 4096), shp(16, 264, 2688) } },
    .{ LaneCfg{ .kind = .coop, .bits = 4 }, &[_]Shape{ shp(128, 128, 1024), shp(1, 512, 4096), shp(16, 256, 2688) } },
    .{ LaneCfg{ .kind = .coop, .bits = 4, .gs = 32 }, &[_]Shape{ shp(128, 128, 1024), shp(1, 512, 4096), shp(16, 256, 2688) } },
    .{ LaneCfg{ .kind = .coop, .bits = 4, .tiled = true }, &[_]Shape{ shp(128, 128, 1024), shp(1, 512, 4096), shp(16, 256, 2688) } },
    .{ LaneCfg{ .kind = .coop, .bits = 4, .tiled = true, .r0 = 64, .tail = 64 }, &[_]Shape{ shp(16, 256, 1024), shp(1, 256, 4096) } },
    .{ LaneCfg{ .kind = .narrow, .bits = 4, .tiled = true }, &[_]Shape{ shp(128, 96, 1024), shp(1, 160, 4096), shp(16, 96, 2688) } },
    .{ LaneCfg{ .kind = .narrow, .bits = 5, .tiled = true }, &[_]Shape{ shp(128, 128, 1024), shp(1, 512, 4096), shp(16, 256, 2688) } },
    .{ LaneCfg{ .kind = .narrow, .bits = 6, .tiled = true }, &[_]Shape{ shp(128, 128, 1024), shp(1, 512, 4096), shp(16, 256, 2688) } },
    .{ LaneCfg{ .kind = .narrow, .bits = 8, .tiled = true }, &[_]Shape{ shp(128, 128, 1024), shp(1, 512, 4096), shp(16, 256, 2688) } },
    .{ LaneCfg{ .kind = .narrow, .bits = 8, .tiled = true, .r0 = 32, .tail = 32 }, &[_]Shape{ shp(16, 256, 1024), shp(1, 256, 4096) } },
    .{ LaneCfg{ .kind = .prefill, .bits = 4, .tiled = true }, &[_]Shape{ shp(129, 4096, 1024), shp(300, 2048, 1024) } },
    .{ LaneCfg{ .kind = .prefill, .bits = 5, .tiled = true }, &[_]Shape{ shp(129, 4096, 1024), shp(300, 2048, 1024) } },
    .{ LaneCfg{ .kind = .prefill, .bits = 6, .tiled = true }, &[_]Shape{ shp(129, 4096, 1024), shp(300, 2048, 1024) } },
    .{ LaneCfg{ .kind = .prefill, .bits = 8, .tiled = true }, &[_]Shape{ shp(129, 4096, 1024), shp(300, 2048, 1024) } },
    .{ LaneCfg{ .kind = .prefill, .bits = 4, .tiled = true, .r0 = 48, .tail = 464 }, &[_]Shape{shp(1100, 512, 1024)} },
};

/// Each registration's launch: the bits and tiling are preprocessor switches in the source. prefill's header (MLX's
/// steel NAX headers) is built at run time.
var lane_launches = blk: {
    var out: [lane_registrations.len]metal.Launch = undefined;
    for (lane_registrations, 0..) |r, i| out[i] = laneLaunch(r[0].kind, r[0].bits, r[0].tiled);
    break :blk out;
};

fn laneName(comptime c: LaneCfg) []const u8 {
    const view = if (c.r0 > 0) std.fmt.comptimePrint(", view at row {d}", .{c.r0}) else "";
    return std.fmt.comptimePrint("msv_lane_qmm_{t} ({d}-bit, group {d}{s}{s})", .{ c.kind, c.bits, c.gs, if (c.tiled) ", tiled" else "", view });
}

/// Bonsai 2's projections at verify width (NAX2: 6..16 rows) and prompt width (int8: 64 rows up, in 32-row tiles),
/// widths a whole 128 and not.
const nax2_served = [_]Shape{ shp(8, 2048, 2048), shp(16, 1024, 5120), shp(6, 512, 17408) };
const int8_served = [_]Shape{ shp(2, 192, 2048), shp(3, 256, 5120) };

/// Prompt widths of Bonsai 2's projections (64 rows up, one padded to 128).
const iq_served = [_]Shape{ shp(100, 1, 2048), shp(64, 1, 5120) };

const iq_drops_nan = spec.Known{ .gate = .nonfinite, .finding = "the group absmax runs through metal::max, which returns the other operand for a NaN: a group holding a NaN gets a finite scale and finite codes and sum where float64 gives NaN; an infinite value's code (Inf / Inf, NaN) clamps to a finite code" };

/// The quantizers' spec: exact codes (i8) then the f32 scales and code sums, `block` the K a threadgroup covers.
fn iqSpec(comptime name: []const u8, block: usize) spec.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = host.depthOne,
        .out = .f32,
        .outs = &.{ .{ .elem = .i8, .count = iqCodes }, .{ .elem = .f32, .count = iqGroups }, .{ .elem = .f32, .count = iqGroups } },
        // Rows in the prefill kernel's 32-row tiles (padding quantizes as zeros).
        .tiles = .{ .{ .size = 32, .step = 1, .min = 1 }, .{}, .{ .size = block, .step = block, .min = block } },
        .served = &iq_served,
        .known = iq_drops_nan,
    };
}

/// The three-row kernel is the one simd_qmm build no probe meets: the wrapper probes a shape's one-row and mma builds
/// (`prepare`) and halves mma's simdgroups on a refusal, but launches `rows` as planned.
const sq_rows3_known = spec.Known{ .gate = .limits, .finding = "msv_simd_qmm rows (three rows, 4-bit) at N <= 2048 (8 simdgroups, 256 threads): M1/M2 build it for 224 threads at K 4096, and no probe covers it. At that shape the one-row probe fails first (its build takes 224 too) and declines the shape to rowqmv; a shape where the one-row build fits and the three-row one does not would fail a three-row call on M1/M2" };

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    inline for (rq_registrations) |r| {
        // Bit equality holds at whole passes only: those registrations keep k at its served values.
        var tiles = rqTiles(r[1].bits, r[1].gs);
        if (r[3]) tiles[2] = .{};
        try set.add(Rq, .init(a, &rq_launch, r[1]), affineSpec(r[0], .bf16, r[2], tiles, if (r[3]) one_row_qmm else null), null);
    }
    inline for (rq_expert_registrations) |r| try set.add(Rq, .init(a, r[2], r[1]), affineSpec(r[0], .bf16, r[3], .{ .{}, .{}, .{} }, null), null);
    inline for (q2_registrations) |r| try set.add(Q2, .init(a, r[1]), affineSpec(r[0], host.elemOf(r[1].dt), r[2], q2Tiles(r[1].kind), null), null);
    inline for (.{ .{ "mlxserve_mxfp8_qmv bf16", .bfloat16 }, .{ "mlxserve_mxfp8_qmv f16", .float16 } }) |r| try set.add(Fp8, .init(a, r[1]), affineSpec(r[0], host.elemOf(r[1]), &fp8_served, .{ .{ .size = 4, .step = 1, .min = 1 }, .{}, .{ .size = 512, .step = 32, .min = 32 } }, null), null);
    inline for (sq_registrations) |r| inline for (.{ 4, 6, 8 }) |bits| {
        const rows = if (r[1] > 0) std.fmt.comptimePrint(", {d} row{s}", .{ r[1], if (r[1] > 1) "s" else "" }) else "";
        var sp = affineSpec(std.fmt.comptimePrint("msv_simd_qmm_{t} ({d}-bit{s})", .{ r[0], bits, rows }), if (r[0] == .prep) .f32 else .bf16, r[2], r[3], null);
        if (r[0] == .rows and r[1] == 3) sp.known = sq_rows3_known;
        if (r[0] != .rows or bits == 4) try set.add(Sq, .init(a, r[0], bits, r[1]), sp, null);
    };
    inline for (lane_registrations, &lane_launches) |r, *l| {
        if (r[0].kind == .prefill) l.header = try lane_qmm.prefillHeader();
        const tiles: [3]spec.Tile = if (r[0].kind == .prefill) .{ .{}, .{}, .{} } else .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{} };
        var sp = affineSpec(laneName(r[0]), .bf16, r[1], tiles, if (r[0].kind == .prefill) "MLX quantized_matmul (its NAX qmm, K unsplit)" else null);
        sp.nax = true; // the engine tiles lane weights and reads them only where naxAvailable()
        try set.add(Lane, .init(a, l, r[0]), sp, null);
    }
    inline for (.{ .{ "nax2_qmm_m16 f16 (2-bit, group 128)", .float16 }, .{ "nax2_qmm_m16 bf16 (2-bit, group 128)", .bfloat16 } }) |r| {
        var sp = affineSpec(r[0], host.elemOf(r[1]), &nax2_served, .{ .{ .size = 16, .step = 1, .min = 6 }, .{ .size = 64, .step = 32, .min = 32 }, .{} }, null);
        sp.nax = true;
        try set.add(Nax2, .init(a, r[1]), sp, null);
    }
    // f16 only: the route runs on Hadamard packs, whose activations are f16 (`ModelConfig.actDtype`).
    inline for (.{ .{ "msv_qmm_int8_prefill f16", .float16, false }, .{ "msv_qmm_int8_prefill f16 (ternary)", .float16, true } }) |r| {
        var sp = affineSpec(r[0], host.elemOf(r[1]), &int8_served, .{ .{}, .{ .size = 128, .step = 32, .min = 32 }, .{} }, null);
        sp.nax = true;
        try set.add(Int8, .init(a, r[1], r[2]), sp, null);
    }
    try set.add(Router, .init(a), affineSpec("msv_rowqmv_router bf16", .bf16, comptime &.{ shp(1, 128, 2688), shp(16, 128, 2688) }, .{ .{ .size = 8, .step = 1, .min = 1 }, .{}, .{} }, null), null);
    try set.add(Iq, .init(a, false), iqSpec("msv_int8_quant f16", 128), null);
    try set.add(Iq, .init(a, true), iqSpec("msv_int8_rotate_quant f16", 1024), null);
}

// ── tests ──

test "kernel contract qmm: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var at: usize = 0;
    inline for (.{ .{ Rq, rq_registrations.len + rq_expert_registrations.len }, .{ Q2, q2_registrations.len }, .{ Fp8, 2 }, .{ Sq, sq_registrations.len * 3 - 4 }, .{ Lane, lane_registrations.len }, .{ Nax2, 2 }, .{ Int8, 2 }, .{ Router, 1 }, .{ Iq, 2 } }) |fam| {
        for (set.kernels.items[at..][0..fam[1]]) |k| {
            const t: *fam[0] = @ptrCast(@alignCast(k.ctx));
            // simd_qmm's prep rides the frag chain; the tensor-unit kernels need NAX.
            if (fam[0] == Sq and t.kind == .prep) continue;
            if ((fam[0] == Lane or fam[0] == Nax2 or fam[0] == Int8) and !device.naxAvailable()) continue;
            for (k.spec.served) |shape| {
                try std.testing.expect(try t.prepare(shape, .random, 7));
                var prep: Sq = if (fam[0] == Sq) switch (t.bits) {
                    inline 4, 6, 8 => |b| .init(a, .prep, b, 0),
                    else => unreachable,
                } else undefined;
                defer if (fam[0] == Sq) prep.deinit();
                if (fam[0] == Int8) {
                    const both = try t.chainedAndShipped();
                    defer a.free(both.ours);
                    if (!std.mem.eql(u8, both.ours, both.theirs)) std.debug.print("{s} at {d}x{d}x{d}: the declared launch differs from the wrapper's\n", .{ k.spec.name, shape.rows, shape.n, shape.k });
                    try std.testing.expectEqualSlices(u8, both.theirs, both.ours);
                    continue;
                }
                const ours = if (fam[0] == Sq and t.kind == .frag) blk: {
                    try std.testing.expect(try prep.prepare(shape, .random, 7));
                    break :blk try t.chained(&prep);
                } else blk: {
                    try t.run();
                    break :blk try a.dupe(u8, t.rig.outputBytes());
                };
                defer a.free(ours);
                const theirs = if (fam[0] == Sq) try t.wrapper(if (t.kind == .frag) prep.args[0] else t.args[0]) else try t.wrapper();
                if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s} at {d}x{d}x{d}: the declared launch differs from the wrapper's\n", .{ k.spec.name, shape.rows, shape.n, shape.k });
                try std.testing.expectEqualSlices(u8, theirs, ours);
            }
        }
        at += fam[1];
    }
}
