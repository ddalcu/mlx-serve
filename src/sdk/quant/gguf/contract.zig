//! The GGUF format's kernel contract: every block type's dequantize, the decode matvec of the two types whose
//! blocks can carry integer weights (Q8_0, Q4_0), and the expert gathers at those two and every K-quant, against
//! ggml's reference dequant (the CPU port in `quants.zig`, checked against ggml's own). Listed in
//! `kernel_contract/registry.zig`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const device = @import("../../../mlx/device.zig");
const kernels = @import("kernels.zig");
const quants = @import("quants.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;
const GgmlType = quants.GgmlType;

/// `kernelFor`'s kernel of `kind` and `ty` at output `dt`; its source declares no threadgroup arrays. Its M1/M2
/// limit is the least `offline.zig` compiled over the registration's shapes (0: every shape INCONCLUSIVE).
fn Launch(comptime kind: kernels.Kind, comptime ty: GgmlType, comptime dt: mlx.mlx_dtype) type {
    const m = limit(kind, ty, dt);
    return struct {
        const launch: metal.Launch = .{
            .name = "gguf_" ++ @tagName(kind) ++ "_" ++ @tagName(ty),
            .inputs = if (kind == .dequant) &.{"w"} else &.{ "x", "w" },
            .outputs = &.{"out"},
            .source = kernels.source(kind, ty),
            .header = kernels.headerFor(ty),
            .tg_bytes = 0,
            .m1 = m,
            .m2 = m,
        };
    };
}

/// Recorded where M1 and M2 build an instantiation for fewer than 1024 threads; each launch is at most 64.
fn limit(comptime kind: kernels.Kind, comptime ty: GgmlType, comptime dt: mlx.mlx_dtype) u32 {
    const f32_out = dt == .float32;
    if (kind == .matvec or kind == .matvec_cols) return switch (ty) {
        .q4_0 => if (f32_out) 512 else 1024,
        else => if (f32_out) 768 else 1024,
    };
    return switch (ty) {
        .q3_k => if (f32_out) 0 else 1024,
        .q4_1 => if (f32_out) 1024 else 832,
        .iq2_xxs, .q5_k => if (f32_out) 1024 else 928,
        .iq3_xxs, .q4_k => if (f32_out) 928 else 1024,
        .iq3_s => 928,
        else => 1024,
    };
}

/// A block's decoded weights stay under this, so the random fill's products are ordinary numbers.
pub const MAX_WEIGHT = 64.0;

/// One registration: a kind at a type and dtype. Dequant shapes are (weight rows, 1, K); matvec shapes are
/// (activation rows, weight rows, K).
const Gguf = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: kernels.Kind,
    ty: GgmlType,
    /// The output's dtype, and the matvec's activations'.
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// The weights as the reference decodes them, [weight rows, K].
    w: []f64 = &.{},
    /// The matvec's activations, [rows, K], already rounded to `dt`.
    x: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, kind: kernels.Kind, ty: GgmlType, dt: mlx.mlx_dtype) Gguf {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .kind = kind, .ty = ty, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Gguf) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn elem(self: *const Gguf) spec.Elem {
        return switch (self.dt) {
            .float32 => .f32,
            .float16 => .f16,
            else => .bf16,
        };
    }

    fn weightRows(self: *const Gguf, shape: Shape) usize {
        return if (self.kind == .dequant) shape.rows else shape.n;
    }

    /// The wrappers' dispatch (`dequantize`, `matvec`): grid, group, template args in their order, output.
    pub fn geometry(self: *const Gguf, shape: Shape) ?metal.Geometry {
        const be = self.ty.blockElems();
        if (shape.k == 0 or shape.k % be != 0) return null;
        const k: c_int = @intCast(shape.k);
        const rb: c_int = @intCast(kernels.rowBytes(self.ty, shape.k));
        const split = kernels.isSplit(self.ty, shape.k);
        if (self.kind == .dequant) {
            const rows: c_int = @intCast(shape.rows);
            const units = rows * @divExact(k, @as(c_int, @intCast(self.ty.unitElems())));
            var g = metal.Geometry{ .grid = .{ units, 1, 1 }, .tg = .{ @min(units, 64), 1, 1 } };
            g.out(&.{ rows, k }, self.dt);
            g.int("K", k);
            g.int("ROWB", rb);
            g.int("SPLIT", @intFromBool(split));
            g.dtype("T", self.dt);
            return g;
        }
        const m: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.n);
        const per = kernels.rowsPerInstance(m);
        const rpt = kernels.matvecRpt(self.kind == .matvec_cols, per, n);
        const groups = @divFloor(n + rpt * kernels.SIMDGROUPS - 1, rpt * kernels.SIMDGROUPS);
        var g = metal.Geometry{ .grid = .{ kernels.SIMD_WIDTH, kernels.SIMDGROUPS * groups, @divExact(m, per) }, .tg = .{ kernels.SIMD_WIDTH, kernels.SIMDGROUPS, 1 } };
        g.out(&.{ m, n }, self.dt);
        g.int("K", k);
        g.int("ROWB", rb);
        g.int("SPLIT", @intFromBool(split));
        g.int("N", n);
        g.int("M", per);
        g.int("RPT", rpt);
        g.dtype("T", self.dt);
        return g;
    }

    pub fn prepare(self: *Gguf, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const wrows = self.weightRows(shape);
        const bank = try bankOf(a, self.ty, wrows, shape.k, fill, self.kind == .dequant, rng);
        self.w = bank.w;
        self.rig.reset();
        const w = try self.rig.input(.uint8, &.{ @intCast(wrows), @intCast(kernels.rowBytes(self.ty, shape.k)) }, bank.bytes);
        if (self.kind == .dequant) {
            self.args[0] = w;
            return true;
        }
        self.x = try activations(a, self.elem(), shape.rows, shape.k, fill, rng);
        self.args[0] = try self.rig.input(self.dt, &.{ @intCast(shape.rows), @intCast(shape.k) }, self.x);
        self.args[1] = w;
        return true;
    }

    pub fn run(self: *Gguf) !void {
        const n: usize = if (self.kind == .dequant) 1 else 2;
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..n]);
    }

    pub fn count(self: *const Gguf) usize {
        return if (self.kind == .dequant) self.shape.rows * self.shape.k else self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *const Gguf, want: []f64, mag: []f64) !void {
        if (self.kind == .dequant) {
            @memcpy(want, self.w);
            for (mag, self.w) |*m, v| m.* = @abs(v);
            return;
        }
        const k = self.shape.k;
        for (0..self.shape.rows) |r| for (0..self.shape.n) |o| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (self.x[r * k ..][0..k], self.w[o * k ..][0..k]) |xv, wv| {
                sum += xv * wv;
                abs += @abs(xv * wv);
            }
            want[r * self.shape.n + o] = sum;
            mag[r * self.shape.n + o] = abs;
        };
    }

    pub fn simulate(self: *const Gguf, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const k = self.shape.k;
        const terms = try self.arena.child_allocator.alloc(f64, k);
        defer self.arena.child_allocator.free(terms);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |o| {
            for (terms, self.x[r * k ..][0..k], self.w[o * k ..][0..k]) |*t, xv, wv| t.* = xv * wv;
            out[r * self.shape.n + o] = spec.sumIn(acc, order, terms);
        };
    }

    /// The shipped wrapper on the same inputs: its output's bytes (the declared launch must be its launch).
    fn wrapper(self: *Gguf) ![]const u8 {
        const s = self.rig.stream();
        const y = if (self.kind == .dequant)
            try kernels.dequantize(self.ty, self.args[0], self.dt, s)
        else
            try kernels.matvec(self.ty, self.args[0], self.args[1], s);
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

/// `wrows` rows of `k` weights of `ty`: the values the reference decodes and the stored bytes (as the rig binds a
/// uint8 input). `.stress`: integer blocks, each row's second half repeating its first (against `activations`'
/// mirror the sums cancel); `.nonfinite` with `planted`: a NaN or infinite scale in three known blocks.
fn bankOf(a: std.mem.Allocator, ty: GgmlType, wrows: usize, k: usize, fill: spec.Fill, planted: bool, rng: std.Random) !struct { w: []f64, bytes: []f64 } {
    const be = ty.blockElems();
    const bb = ty.blockBytes();
    const nb = k / be;
    const raw = try a.alloc(u8, wrows * nb * bb);
    const w = try a.alloc(f64, wrows * k);
    var dec: [256]f32 = undefined;
    for (0..wrows * nb) |b| {
        const blk = raw[b * bb ..][0..bb];
        const bad = fill == .nonfinite and planted and (b == 0 or b == nb + 1 or b == 3 * nb + nb / 2);
        if (fill == .stress and b % nb >= nb / 2) {
            @memcpy(blk, raw[(b - nb / 2) * bb ..][0..bb]);
        } else if (fill == .stress) integerBlock(ty, blk, rng) else try randomBlock(ty, blk, dec[0..be], rng, bad);
        try quants.dequantize(ty, blk, dec[0..be]);
        for (w[b * be ..][0..be], dec[0..be]) |*o, v| o.* = v;
    }
    const stored = try a.alloc(u8, wrows * kernels.rowBytes(ty, k));
    if (kernels.isSplit(ty, k)) kernels.packRows(ty, k, stored, raw) else @memcpy(stored, raw);
    const bytes = try a.alloc(f64, stored.len);
    for (bytes, stored) |*o, v| o.* = @floatFromInt(v);
    return .{ .w = w, .bytes = bytes };
}

/// `rows` activation rows of `k`, rounded to `e`. `.stress`: positive integers whose partial sums outgrow fp16 and
/// bf16 yet stay exact in fp32 in any order, the second half mirroring the first so eight terms of a sum remain;
/// `.nonfinite`: NaN, +Inf and -Inf in rows 0, 1 and 2.
fn activations(a: std.mem.Allocator, e: spec.Elem, rows: usize, k: usize, fill: spec.Fill, rng: std.Random) ![]f64 {
    const x = try a.alloc(f64, rows * k);
    for (x) |*v| v.* = spec.roundTo(e, rng.float(f64) * 2 - 1);
    if (fill == .stress) for (0..rows) |r| {
        const row = x[r * k ..][0..k];
        const half = k / 2;
        for (row[0..half], row[half..], 0..) |*lo, *hi, i| {
            lo.* = @floatFromInt(1 + rng.uintLessThan(u32, 16));
            hi.* = -lo.* + @as(f64, if (i < 8) 1 else 0);
        }
    };
    if (fill == .nonfinite) {
        const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 } };
        const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
        for (at, put) |q, v| if (q[0] < rows) {
            x[q[0] * k + q[1]] = v;
        };
    }
    return x;
}

/// Random bytes until the block decodes to finite weights under `MAX_WEIGHT`, or (`bad`) to some non-finite ones.
pub fn randomBlock(ty: GgmlType, blk: []u8, dec: []f32, rng: std.Random, bad: bool) !void {
    for (0..1 << 16) |_| {
        rng.bytes(blk);
        try quants.dequantize(ty, blk, dec);
        var finite = true;
        var small = true;
        for (dec) |v| {
            if (!std.math.isFinite(v)) finite = false else if (@abs(v) > MAX_WEIGHT) small = false;
        }
        if (bad and !finite) return;
        if (!bad and finite and small) return;
    }
    return error.NoBlockFound;
}

/// A block of integer weights: Q8_0 or Q4_0 at scale 1, quants 1..127 (Q8_0) or 1..7 (Q4_0, nibble - 8); a K-quant
/// at superblock scale 1 and min 0 with random quants under small integer sub-scales (Q2_K and Q3_K 1..15 x 0..3,
/// Q4_K 0..15, Q5_K 0..31, Q6_K -32..31).
pub fn integerBlock(ty: GgmlType, blk: []u8, rng: std.Random) void {
    const one: u16 = @bitCast(@as(f16, 1.0));
    switch (ty) {
        .q2_k => {
            rng.bytes(blk);
            for (blk[0..16]) |*sc| sc.* = @intCast(1 + rng.uintLessThan(u32, 15));
            std.mem.writeInt(u16, blk[80..82], one, .little);
            std.mem.writeInt(u16, blk[82..84], 0, .little);
        },
        .q3_k => {
            rng.bytes(blk);
            // Every high-mask bit set (no -4 offset); each 6-bit scale 33..47 (low nibble 1..15, high bits 2), 1..15 past
            // the -32 bias: a scale of one power of two would keep fp16 sums exact.
            @memset(blk[0..32], 0xFF);
            for (blk[96..104]) |*sc| sc.* = @intCast((1 + rng.uintLessThan(u32, 15)) | ((1 + rng.uintLessThan(u32, 15)) << 4));
            @memset(blk[104..108], 0xAA);
            std.mem.writeInt(u16, blk[108..110], one, .little);
        },
        .q4_k, .q5_k => {
            rng.bytes(blk);
            std.mem.writeInt(u16, blk[0..2], one, .little);
            std.mem.writeInt(u16, blk[2..4], 0, .little);
            // Scales 1, mins 0 (gg_scale_min_k4's 12-byte layout).
            @memcpy(blk[4..16], &[_]u8{ 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 });
        },
        .q6_k => {
            rng.bytes(blk);
            @memset(blk[192..208], 1);
            std.mem.writeInt(u16, blk[208..210], one, .little);
        },
        else => {
            std.mem.writeInt(u16, blk[0..2], one, .little);
            for (blk[2..]) |*q| q.* = switch (ty) {
                .q8_0 => @intCast(1 + rng.uintLessThan(u32, 127)),
                else => @intCast((9 + rng.uintLessThan(u32, 7)) | ((9 + rng.uintLessThan(u32, 7)) << 4)),
            };
        },
    }
}

/// Byte offset of a block's fp16 superblock scale `d`.
pub fn scaleAt(ty: GgmlType) usize {
    return switch (ty) {
        .q2_k => 80,
        .q3_k => 108,
        .q6_k => 208,
        else => 0,
    };
}

/// An `integerBlock` with every weight negated: its `d`'s sign flipped (the K-quants' mins are 0).
pub fn negate(ty: GgmlType, blk: []u8) void {
    blk[scaleAt(ty) + 1] ^= 0x80;
}

/// An `integerBlock`'s weights 0..7 set to 1.
pub fn onesFirst8(ty: GgmlType, blk: []u8) void {
    switch (ty) {
        .q8_0 => @memset(blk[2..10], 1),
        .q4_0 => for (blk[2..10]) |*q| {
            q.* = (q.* & 0xF0) | 9;
        },
        .q2_k => {
            blk[0] = 1;
            for (blk[16..24]) |*q| q.* = (q.* & 0xFC) | 1;
        },
        .q3_k => {
            blk[96] = (blk[96] & 0xF0) | 1; // the first group's scale back to 1
            for (blk[32..40]) |*q| q.* = (q.* & 0xFC) | 1;
            for (blk[0..8]) |*h| h.* |= 1;
        },
        .q4_k => for (blk[16..24]) |*q| {
            q.* = (q.* & 0xF0) | 1;
        },
        .q5_k => {
            for (blk[48..56]) |*q| q.* = (q.* & 0xF0) | 1;
            for (blk[16..24]) |*h| h.* &= 0xFE;
        },
        .q6_k => {
            for (blk[0..8]) |*q| q.* = (q.* & 0xF0) | 1;
            for (blk[128..136]) |*h| h.* = (h.* & 0xFC) | 2;
        },
        else => unreachable,
    }
}

/// The gathers' types: the two that carry positive integer weights and every K-quant (the NAX ones a 32-weight slice at
/// a time).
const gather_types = .{ GgmlType.q8_0, GgmlType.q4_0, GgmlType.q2_k, GgmlType.q3_k, GgmlType.q4_k, GgmlType.q5_k, GgmlType.q6_k };

/// The gathers' bank: `GATHER_EXPERTS` experts of a shape's n rows. The decode gather's pairs come `TOPK` a token.
const GATHER_EXPERTS = 8;
const TOPK = 6;

/// The expert gathers at shapes (rows, rows an expert, K), bf16 activations: the decode gather (`.gather`, rows
/// tokens of TOPK pairs sharing the token's row, experts unsorted), the matrix units' gather (`.gather_nax`, a row a
/// pair, sorted by expert in ragged runs, one expert empty) and its gate+up (`.gather_nax_gu`, two banks, the clipped
/// SwiGLU times each pair's weight). The matrix units take each decoded weight rounded to bf16 and at least 64 pairs:
/// rows r is 61 r + 3 of them (64 at one row, a ragged total past it).
const Gather = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: kernels.Kind,
    ty: GgmlType,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// Each bank as the reference reads it, [experts * rows, K].
    w: [2][]f64 = .{ &.{}, &.{} },
    x: []f64 = &.{},
    ids: []usize = &.{},
    wts: []f64 = &.{},
    args: [7]mlx.mlx_array = undefined,
    n_args: usize = 0,

    const dt: mlx.mlx_dtype = .bfloat16;
    const lim = 10.0;

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, kind: kernels.Kind, ty: GgmlType) Gather {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .kind = kind, .ty = ty, .arena = .init(a) };
    }

    pub fn deinit(self: *Gather) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn nax(self: *const Gather) bool {
        return self.kind != .gather;
    }

    fn pairs(self: *const Gather, shape: Shape) usize {
        return if (self.nax()) 61 * shape.rows + 3 else TOPK * shape.rows;
    }

    /// The wrappers' dispatch (`gatherLinear`'s decode gather and `gatherNax`, `gatherNaxSwiGLU`).
    pub fn geometry(self: *const Gather, shape: Shape) ?metal.Geometry {
        if (shape.k == 0 or shape.k % self.ty.blockElems() != 0 or shape.n == 0 or shape.rows == 0) return null;
        const p: c_int = @intCast(self.pairs(shape));
        const n: c_int = @intCast(shape.n);
        var g: metal.Geometry = undefined;
        if (self.kind == .gather) {
            const groups = @divFloor(n + kernels.GATHER_RPT * kernels.GATHER_SIMDGROUPS - 1, kernels.GATHER_RPT * kernels.GATHER_SIMDGROUPS);
            g = .{ .grid = .{ kernels.SIMD_WIDTH, kernels.GATHER_SIMDGROUPS * groups, p }, .tg = .{ kernels.SIMD_WIDTH, kernels.GATHER_SIMDGROUPS, 1 } };
        } else {
            if (shape.n % 128 != 0 or shape.k % 64 != 0) return null;
            const groups = @divFloor(p + 63, 64) + GATHER_EXPERTS;
            g = .{ .grid = .{ if (self.kind == .gather_nax_gu) 2 * n else n, groups, 1 }, .tg = .{ 128, 1, 1 } };
        }
        g.out(&.{ p, n }, dt);
        for (kernels.layoutConsts(self.ty, @intCast(shape.k), @intCast(kernels.rowBytes(self.ty, shape.k)))) |c| g.int(std.mem.span(c[0]), c[1]);
        g.int("N", n);
        if (self.kind == .gather) {
            g.int("M", 1);
            g.int("RPT", kernels.GATHER_RPT);
            g.int("XDIV", TOPK);
        } else g.int("E", GATHER_EXPERTS);
        g.dtype("T", dt);
        return g;
    }

    pub fn prepare(self: *Gather, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const P = self.pairs(shape);
        self.ids = try a.alloc(usize, P);
        if (self.kind == .gather) {
            for (0..P / TOPK) |t| {
                var pool: [GATHER_EXPERTS]usize = undefined;
                for (&pool, 0..) |*e, i| e.* = i;
                rng.shuffle(usize, &pool);
                @memcpy(self.ids[t * TOPK ..][0..TOPK], pool[0..TOPK]);
            }
        } else {
            // Ragged sorted runs, expert 1's empty.
            for (self.ids) |*e| {
                e.* = rng.uintLessThan(usize, GATHER_EXPERTS - 1);
                if (e.* >= 1) e.* += 1;
            }
            std.mem.sort(usize, self.ids, {}, std.sort.asc(usize));
        }
        self.rig.reset();
        const x_rows = if (self.kind == .gather) P / TOPK else P;
        self.x = try activations(a, .bf16, x_rows, shape.k, fill, rng);
        // Gate sums of an RMS-normed input's size: f32's exp overflows past |g| 88, where float64's does not.
        if (self.kind == .gather_nax_gu and fill != .stress) for (self.x) |*v| {
            if (std.math.isFinite(v.*)) v.* = spec.roundTo(.bf16, v.* / 64);
        };
        self.args[0] = try self.rig.input(dt, &.{ @intCast(x_rows), @intCast(shape.k) }, self.x);
        const banks: usize = if (self.kind == .gather_nax_gu) 2 else 1;
        const rb: c_int = @intCast(kernels.rowBytes(self.ty, shape.k));
        for (0..banks) |b| {
            const bank = try bankOf(a, self.ty, GATHER_EXPERTS * shape.n, shape.k, fill, false, rng);
            if (self.nax()) for (bank.w) |*v| {
                v.* = spec.roundTo(.bf16, v.*);
            };
            self.w[b] = bank.w;
            self.args[1 + b] = try self.rig.input(.uint8, &.{ GATHER_EXPERTS, @intCast(shape.n), rb }, bank.bytes);
        }
        var at = 1 + banks;
        if (self.kind == .gather) {
            const v = try a.alloc(f64, P);
            for (v, self.ids) |*o, e| o.* = @floatFromInt(e);
            self.args[at] = try self.rig.input(.uint32, &.{@intCast(P)}, v);
            at += 1;
        } else {
            // Each expert's first row and first 64-row tile (`runStarts`).
            const offs = try a.alloc(f64, GATHER_EXPERTS + 1);
            const tiles = try a.alloc(f64, GATHER_EXPERTS + 1);
            offs[0] = 0;
            tiles[0] = 0;
            for (0..GATHER_EXPERTS) |e| {
                var c: usize = 0;
                for (self.ids) |i| c += @intFromBool(i == e);
                offs[e + 1] = offs[e] + @as(f64, @floatFromInt(c));
                tiles[e + 1] = tiles[e] + @as(f64, @floatFromInt((c + 63) / 64));
            }
            self.args[at] = try self.rig.input(.int32, &.{GATHER_EXPERTS + 1}, offs);
            self.args[at + 1] = try self.rig.input(.int32, &.{GATHER_EXPERTS + 1}, tiles);
            at += 2;
        }
        if (self.kind == .gather_nax_gu) {
            self.wts = try a.alloc(f64, P);
            for (self.wts) |*v| v.* = spec.roundTo(.f32, 0.05 + rng.float(f64));
            self.args[at] = try self.rig.input(.float32, &.{@intCast(P)}, self.wts);
            self.args[at + 1] = try self.rig.input(.float32, &.{1}, &.{lim});
            at += 2;
        }
        self.n_args = at;
        return true;
    }

    pub fn run(self: *Gather) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Gather) usize {
        return self.pairs(self.shape) * self.shape.n;
    }

    /// Pair `p`'s row `o` of bank `b`: the sum (in `acc`, or float64) and its terms' magnitude.
    fn dot(self: *const Gather, b: usize, p: usize, o: usize, acc: ?spec.Accum, order: spec.Order, t: []f64) [2]f64 {
        const k = self.shape.k;
        const xr = if (self.kind == .gather) p / TOPK else p;
        const w = self.w[b][(self.ids[p] * self.shape.n + o) * k ..][0..k];
        var mag: f64 = 0;
        for (t[0..k], self.x[xr * k ..][0..k], w) |*v, xv, wv| {
            v.* = xv * wv;
            mag += @abs(xv * wv);
        }
        if (acc) |c| return .{ spec.sumIn(c, order, t[0..k]), mag };
        var sum: f64 = 0;
        for (t[0..k]) |v| sum += v;
        return .{ sum, mag };
    }

    fn compute(self: *const Gather, out: []f64, mag: []f64, acc: ?spec.Accum, order: spec.Order) !void {
        const t = try self.arena.child_allocator.alloc(f64, self.shape.k);
        defer self.arena.child_allocator.free(t);
        const N = self.shape.n;
        for (0..self.pairs(self.shape)) |p| for (0..N) |o| {
            const g = self.dot(0, p, o, acc, order, t);
            const i = p * N + o;
            if (self.kind != .gather_nax_gu) {
                out[i] = g[0];
                mag[i] = g[1];
                continue;
            }
            // The epilogue: each sum rounded to bf16, the clip, g * sigmoid(g) * u * the pair's weight.
            const v = self.dot(1, p, o, acc, order, t);
            const gt = spec.roundTo(.bf16, g[0]);
            const ut = spec.roundTo(.bf16, v[0]);
            const gc = if (std.math.isNan(gt)) gt else @min(gt, lim);
            const uc = if (std.math.isNan(ut)) ut else std.math.clamp(ut, -lim, lim);
            const sig = 1 / (1 + @exp(-gc));
            const wt = self.wts[p];
            out[i] = gc * sig * uc * wt;
            // Each sum's error (and its bf16 rounding) reaches y scaled by y's slope in it.
            const slope_g = @abs(sig * (1 + gc * (1 - sig)) * uc * wt);
            const flip = 0x1p-8 / spec.gamma(@floatFromInt(self.shape.k), 0x1p-24);
            mag[i] = (g[1] + @abs(g[0]) * flip) * slope_g * 2 + (v[1] + @abs(v[0]) * flip) * @abs(gc * sig * wt) * 2;
        };
    }

    pub fn reference(self: *const Gather, want: []f64, mag: []f64) !void {
        try self.compute(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Gather, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const mag = try self.arena.child_allocator.alloc(f64, out.len);
        defer self.arena.child_allocator.free(mag);
        try self.compute(out, mag, acc, order);
    }

    /// The shipped wrapper on the same inputs (the declared launch must be its launch).
    fn wrapper(self: *Gather) ![]const u8 {
        const s = self.rig.stream();
        const P: c_int = @intCast(self.pairs(self.shape));
        const K: c_int = @intCast(self.shape.k);
        const x_rows: c_int = if (self.kind == .gather) @divExact(P, TOPK) else P;
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_reshape(&x3, self.args[0], &[_]c_int{ x_rows, 1, K }, 3, s));
        const ids = try self.arena.allocator().alloc(u32, self.ids.len);
        for (ids, self.ids) |*o, e| o.* = @intCast(e);
        const idv = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ x_rows, @divExact(P, x_rows) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(idv);
        const y = switch (self.kind) {
            .gather, .gather_nax => try kernels.gatherLinear(.{ .ty = self.ty }, x3, self.args[1], .{ .ctx = null }, idv, self.kind == .gather_nax, s),
            else => (try kernels.gatherNaxSwiGLU(.{ .ty = self.ty }, .{ .ty = self.ty }, x3, self.args[1], self.args[2], idv, self.args[5], self.args[6], s)) orelse return error.NotServed,
        };
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

/// The NAX gathers' threadgroup tile (`buf`: 128 rows of 2 units plus padding, as floats over halves).
const nax_tg_bytes = 4 * (128 * (2 * 32 + 8) / 2);

fn GatherLaunch(comptime kind: kernels.Kind, comptime ty: GgmlType) type {
    return struct {
        const launch: metal.Launch = .{
            .name = "gguf_" ++ @tagName(kind) ++ "_" ++ @tagName(ty),
            .inputs = kernels.inputNames(kind),
            .outputs = &.{"out"},
            .source = kernels.source(kind, ty),
            .header = kernels.kernelHeader(kind, ty),
            .tg_bytes = if (kind == .gather) 0 else nax_tg_bytes,
            .m1 = if (kind == .gather) 1024 else 0,
            .m2 = if (kind == .gather) 1024 else 0,
            .offline = kind == .gather,
            .min_gen = if (kind == .gather) 13 else 17,
        };
    };
}

/// Served: the decode gather at one token and a 3-token verify (whole 32-unit strides of K and not), the matrix units
/// at 125 and 308 pairs over the 8 experts; a 4096-deep row each, where Q4_0's small integers outgrow fp16 sums.
const gather_shapes: []const Shape = &.{ .{ .rows = 1, .n = 256, .k = 2048 }, .{ .rows = 3, .n = 96, .k = 1280 }, .{ .rows = 1, .n = 32, .k = 4096 } };
const nax_shapes: []const Shape = &.{ .{ .rows = 2, .n = 128, .k = 512 }, .{ .rows = 5, .n = 256, .k = 1024 }, .{ .rows = 1, .n = 128, .k = 4096 } };

fn gatherSpec(comptime kind: kernels.Kind, comptime ty: GgmlType) spec.Spec {
    const gu = kind == .gather_nax_gu;
    return .{
        .name = "gguf " ++ @tagName(kind) ++ " " ++ @tagName(ty) ++ " bf16",
        .accum = .f32,
        .depth = depthK,
        .ops = if (gu) 8 else 0,
        .mids = if (gu) 1 else 0,
        .out = .bf16,
        .tiles = if (kind == .gather)
            .{ .{}, .{ .size = 32 }, .{ .size = 1024, .step = @max(32, ty.blockElems()), .min = @max(32, ty.blockElems()) } }
        else
            .{ .{}, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 64, .step = @max(64, ty.blockElems()), .min = @max(64, ty.blockElems()) } },
        .served = if (kind == .gather) gather_shapes else nax_shapes,
        .sums = if (gu) .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "each sum rounds to bf16 before the activation; the plain gather_nax arm proves the sums" } else .{},
        .known = if (gu) swiglu_nan else null,
        .nax = kind != .gather,
    };
}

const swiglu_nan: spec.Known = .{ .gate = .nonfinite, .finding = "a NaN gate or up sum becomes the clip limit: metal::min and clamp drop NaN where MLX's minimum (the composed chain) propagates it" };

fn depthK(shape: Shape) usize {
    return shape.k;
}

fn none(_: Shape) usize {
    return 0;
}

const dequant_shapes: []const Shape = &.{ .{ .rows = 3, .n = 1, .k = 2048 }, .{ .rows = 16, .n = 1, .k = 4096 } };
/// The striped matvec serves what the column-major one declines (rows under 1024 weights, or 3+ activation rows).
const matvec_shapes: []const Shape = &.{ .{ .rows = 1, .n = 2048, .k = 512 }, .{ .rows = 3, .n = 512, .k = 4096 }, .{ .rows = 8, .n = 1024, .k = 1024 } };
const matvec_cols_shapes: []const Shape = &.{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 1, .n = 2048, .k = 2560 }, .{ .rows = 2, .n = 512, .k = 4096 } };

const mxfp4_tiny_scale: spec.Known = .{ .gate = .shapes, .finding = "an E8M0 scale byte of 0 or 1 is a subnormal fp32 scale, which the GPU flushes to zero where ggml's reference keeps it: such blocks dequantize to 0 instead of values below 1e-37" };
const q3k_inconclusive: spec.Known = .{ .gate = .limits, .finding = "every offline M1/M2 build of the f32 Q3_K dequant changes with the thread request (INCONCLUSIVE), so no limit is recorded; it launches 64 threads" };

fn dequantSpec(comptime ty: GgmlType, comptime out: spec.Elem) spec.Spec {
    return .{
        .known = if (ty == .q3_k and out == .f32) q3k_inconclusive else if (ty == .mxfp4) mxfp4_tiny_scale else null,
        .name = "gguf dequant " ++ @tagName(ty) ++ (if (out == .f32) " f32" else " bf16"),
        .accum = .none,
        .depth = none,
        // The scale products: up to a superblock scale, a sub-scale, a quant and a min, each an fp32 rounding.
        .ops = 4,
        .out = out,
        .tiles = .{ .{}, .{}, .{ .size = ty.blockElems(), .step = ty.blockElems(), .min = ty.blockElems() } },
        .served = dequant_shapes,
    };
}

fn matvecSpec(comptime kind: kernels.Kind, comptime ty: GgmlType, comptime out: spec.Elem) spec.Spec {
    return .{
        .name = "gguf " ++ @tagName(kind) ++ " " ++ @tagName(ty) ++ (if (out == .f32) " f32" else " bf16"),
        .accum = .f32,
        .depth = depthK,
        .out = out,
        .tiles = .{ .{ .size = 8 }, .{ .size = 16 }, .{ .size = 256, .step = 32, .min = 32 } },
        .served = if (kind == .matvec_cols) matvec_cols_shapes else matvec_shapes,
    };
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    inline for (kernels.all_types) |ty| {
        try set.add(Gguf, .init(a, &Launch(.dequant, ty, .float32).launch, .dequant, ty, .float32), dequantSpec(ty, .f32), null);
        try set.add(Gguf, .init(a, &Launch(.dequant, ty, .bfloat16).launch, .dequant, ty, .bfloat16), dequantSpec(ty, .bf16), null);
    }
    inline for (.{ GgmlType.q8_0, GgmlType.q4_0 }) |ty| inline for (.{ kernels.Kind.matvec, kernels.Kind.matvec_cols }) |kind| {
        try set.add(Gguf, .init(a, &Launch(kind, ty, .float32).launch, kind, ty, .float32), matvecSpec(kind, ty, .f32), null);
        try set.add(Gguf, .init(a, &Launch(kind, ty, .bfloat16).launch, kind, ty, .bfloat16), matvecSpec(kind, ty, .bf16), null);
    };
    inline for (gather_types) |ty| inline for (.{ kernels.Kind.gather, kernels.Kind.gather_nax, kernels.Kind.gather_nax_gu }) |kind| {
        try set.add(Gather, .init(a, &GatherLaunch(kind, ty).launch, kind, ty), gatherSpec(kind, ty), null);
    };
}

// ── tests ──

test "kernel contract gguf: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        if (std.mem.startsWith(u8, k.spec.name, "gguf gather")) continue;
        const g: *Gguf = @ptrCast(@alignCast(k.ctx));
        try std.testing.expect(try g.prepare(k.spec.served[0], .random, 7));
        try g.run();
        const ours = try a.dupe(u8, g.rig.outputBytes());
        defer a.free(ours);
        const theirs = try g.wrapper();
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}

test "kernel contract gguf gathers: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        if (!std.mem.startsWith(u8, k.spec.name, "gguf gather")) continue;
        if (k.spec.nax and !device.naxAvailable()) continue;
        const g: *Gather = @ptrCast(@alignCast(k.ctx));
        try std.testing.expect(try g.prepare(k.spec.served[0], .random, 7));
        try g.run();
        const ours = try a.dupe(u8, g.rig.outputBytes());
        defer a.free(ours);
        const theirs = try g.wrapper();
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}
