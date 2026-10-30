//! The GGUF format's kernel contract: every block type's dequantize, and the decode matvec of the two types whose
//! blocks can carry integer weights (Q8_0, Q4_0), against ggml's reference dequant (the CPU port in `quants.zig`,
//! checked against ggml's own). Listed in `kernel_contract/registry.zig`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
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
    if (kind == .matvec) return switch (ty) {
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
const MAX_WEIGHT = 64.0;

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
        var rpt = kernels.matvecRows(per);
        while (rpt > 1 and @divFloor(n, rpt) < kernels.MIN_SIMDGROUPS) rpt = @divExact(rpt, 2);
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
        const be = self.ty.blockElems();
        const bb = self.ty.blockBytes();
        const wrows = self.weightRows(shape);
        const nb = shape.k / be;
        const raw = try a.alloc(u8, wrows * nb * bb);
        self.w = try a.alloc(f64, wrows * shape.k);
        var dec: [256]f32 = undefined;
        for (0..wrows * nb) |b| {
            const blk = raw[b * bb ..][0..bb];
            // Non-finite blocks at known places for the dequant (a NaN or infinite scale); the matvec plants its own.
            const bad = fill == .nonfinite and self.kind == .dequant and (b == 0 or b == nb + 1 or b == 3 * nb + nb / 2);
            // Gate 4: a row's second half repeats its first, against mirrored activations, so the sums cancel.
            if (fill == .stress and b % nb >= nb / 2) {
                @memcpy(blk, raw[(b - nb / 2) * bb ..][0..bb]);
            } else if (fill == .stress) integerBlock(self.ty, blk, rng) else try randomBlock(self.ty, blk, dec[0..be], rng, bad);
            try quants.dequantize(self.ty, blk, dec[0..be]);
            for (self.w[b * be ..][0..be], dec[0..be]) |*o, v| o.* = v;
        }
        const rb = kernels.rowBytes(self.ty, shape.k);
        const stored = try a.alloc(u8, wrows * rb);
        if (kernels.isSplit(self.ty, shape.k)) kernels.packRows(self.ty, shape.k, stored, raw) else @memcpy(stored, raw);
        const bytes = try a.alloc(f64, stored.len);
        for (bytes, stored) |*o, v| o.* = @floatFromInt(v);

        self.rig.reset();
        const w = try self.rig.input(.uint8, &.{ @intCast(wrows), @intCast(rb) }, bytes);
        if (self.kind == .dequant) {
            self.args[0] = w;
            return true;
        }
        self.x = try a.alloc(f64, shape.rows * shape.k);
        for (self.x) |*v| v.* = spec.roundTo(self.elem(), rng.float(f64) * 2 - 1);
        if (fill == .stress) for (0..shape.rows) |r| {
            // Positive integers whose partial sums outgrow fp16 and bf16 yet stay exact in fp32 in any order; the
            // mirror leaves eight terms of each row's sum.
            const row = self.x[r * shape.k ..][0..shape.k];
            const half = shape.k / 2;
            for (row[0..half], row[half..], 0..) |*lo, *hi, i| {
                lo.* = @floatFromInt(1 + rng.uintLessThan(u32, 16));
                hi.* = -lo.* + @as(f64, if (i < 8) 1 else 0);
            }
        };
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, shape.k / 3 }, .{ 1, shape.k / 2 }, .{ 2, shape.k - 1 } };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.x[p[0] * shape.k + p[1]] = v;
            };
        }
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

/// Random bytes until the block decodes to finite weights under `MAX_WEIGHT`, or (`bad`) to some non-finite ones.
fn randomBlock(ty: GgmlType, blk: []u8, dec: []f32, rng: std.Random, bad: bool) !void {
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

/// A Q8_0 or Q4_0 block of positive integer weights: scale 1, quants 1..127 (Q8_0) or 1..7 (Q4_0, nibble - 8).
fn integerBlock(ty: GgmlType, blk: []u8, rng: std.Random) void {
    std.mem.writeInt(u16, blk[0..2], @bitCast(@as(f16, 1.0)), .little);
    for (blk[2..]) |*q| q.* = switch (ty) {
        .q8_0 => @intCast(1 + rng.uintLessThan(u32, 127)),
        else => @intCast((9 + rng.uintLessThan(u32, 7)) | ((9 + rng.uintLessThan(u32, 7)) << 4)),
    };
}

fn depthK(shape: Shape) usize {
    return shape.k;
}

fn none(_: Shape) usize {
    return 0;
}

const dequant_shapes: []const Shape = &.{ .{ .rows = 3, .n = 1, .k = 2048 }, .{ .rows = 16, .n = 1, .k = 4096 } };
const matvec_shapes: []const Shape = &.{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 3, .n = 512, .k = 4096 }, .{ .rows = 8, .n = 1024, .k = 1024 } };

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

fn matvecSpec(comptime ty: GgmlType, comptime out: spec.Elem) spec.Spec {
    return .{
        .name = "gguf matvec " ++ @tagName(ty) ++ (if (out == .f32) " f32" else " bf16"),
        .accum = .f32,
        .depth = depthK,
        .out = out,
        .tiles = .{ .{ .size = 5 }, .{ .size = 16 }, .{ .size = 256, .step = 32, .min = 32 } },
        .served = matvec_shapes,
    };
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    inline for (kernels.all_types) |ty| {
        try set.add(Gguf, .init(a, &Launch(.dequant, ty, .float32).launch, .dequant, ty, .float32), dequantSpec(ty, .f32), null);
        try set.add(Gguf, .init(a, &Launch(.dequant, ty, .bfloat16).launch, .dequant, ty, .bfloat16), dequantSpec(ty, .bf16), null);
    }
    inline for (.{ GgmlType.q8_0, GgmlType.q4_0 }) |ty| {
        try set.add(Gguf, .init(a, &Launch(.matvec, ty, .float32).launch, .matvec, ty, .float32), matvecSpec(ty, .f32), null);
        try set.add(Gguf, .init(a, &Launch(.matvec, ty, .bfloat16).launch, .matvec, ty, .bfloat16), matvecSpec(ty, .bf16), null);
    }
}

// ── tests ──

test "kernel contract gguf: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
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
