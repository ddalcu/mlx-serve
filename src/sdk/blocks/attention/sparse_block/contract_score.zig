//! The QSA indexer's score sheet (`msv_qsa_score`, NAX tensor ops) in the kernel contract: for each query row and
//! pooled block, the sum over the 4 indexer heads of relu(q . k). It replaces the composed chain (astype f32,
//! matmul, maximum 0, sum over heads) bit for bit at every served width: decode, verify and prefill pick the same
//! blocks. Rows = query rows (the h4 layout from 128), n = pooled blocks, k = the 128 indexer dims.

const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const kc = @import("../../../testing/kernel_contract/root.zig");
const host = @import("../../../testing/kernel_contract/host.zig");
const score = @import("score.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

const D = 128;
const HEADS = 4;

const launch: metal.Launch = .{
    .name = "msv_qsa_score",
    .inputs = &.{ "Q", "POOL" },
    .outputs = &.{"OUT"},
    .source = score.QSA_SCORE_KERNEL_SOURCE,
    .header = score.QSA_SCORE_KERNEL_HEADER,
    // The bf16 pooled-key tile, 128 x 32.
    .tg_bytes = 128 * 32 * 2,
    .offline = false,
    .min_gen = 17,
};

pub const Score = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &launch,
    shape: Shape = undefined,
    q: []f64 = &.{},
    pool: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) Score {
        return .{ .rig = metal.Rig.init(a) };
    }

    pub fn deinit(self: *Score) void {
        self.rig.a.free(self.q);
        self.rig.a.free(self.pool);
        self.rig.deinit();
    }

    /// The wrapper's dispatch: the base layout below 128 rows (4 simdgroups, 4 rows each over the heads), h4 from
    /// 128 (8 simdgroups of 16 rows); slab-strided groups along the blocks.
    pub fn geometry(_: *const Score, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.n < 1 or shape.k != D) return null;
        const rows: c_int = @intCast(shape.rows);
        const nb: c_int = @intCast(shape.n);
        const layout = score.qsaScoreLayoutFor(rows);
        const nsg = score.qsaScoreNsg(layout);
        const r_tg = score.qsaScoreRowsPerTg(layout, nsg);
        const ngrp: c_int = @divTrunc(rows + r_tg - 1, r_tg);
        const nslab: u32 = @intCast(@divTrunc(nb + 31, 32));
        const nsh: c_int = @intCast(score.qsaScoreNsh(nslab, @intCast(ngrp), layout));
        var g = metal.Geometry{ .grid = .{ nsh * nsg * 32, ngrp, 1 }, .tg = .{ nsg * 32, 1, 1 } };
        g.out(&.{ 1, rows, nb }, .float32);
        g.int("NSGVAL", nsg);
        g.int("USEH4VAL", @intFromBool(layout == .h4));
        return g;
    }

    /// Random bf16 operands. Non-finite: a NaN in head 0's row 0, a +Inf in head 1's row 1, a -Inf in pooled block
    /// 2. Stress: mirrored integer q rows (`host.mirroredWide`) against all-ones blocks: every dot a small positive
    /// integer, its partial sums past fp16 and off bf16's grid, exact in fp32.
    pub fn prepare(self: *Score, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        a.free(self.q);
        a.free(self.pool);
        self.q = try a.alloc(f64, HEADS * shape.rows * D);
        self.pool = try a.alloc(f64, shape.n * D);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            host.mirroredWide(rng, self.q, D);
            @memset(self.pool, 1);
        } else {
            for (self.q) |*x| x.* = host.rt(.bfloat16, rng.floatNorm(f64));
            for (self.pool) |*x| x.* = host.rt(.bfloat16, rng.floatNorm(f64));
        }
        if (fill == .nonfinite) {
            self.q[(0 * shape.rows + 0) * D + 3] = std.math.nan(f64);
            if (shape.rows > 1) self.q[(1 * shape.rows + 1) * D + 9] = std.math.inf(f64);
            if (shape.n > 2) self.pool[2 * D + 5] = -std.math.inf(f64);
        }
        self.rig.reset();
        self.args = .{
            try self.rig.input(.bfloat16, &.{ 1, HEADS, @intCast(shape.rows), D }, self.q),
            try self.rig.input(.bfloat16, &.{ 1, @intCast(shape.n), D }, self.pool),
        };
        return true;
    }

    pub fn run(self: *Score) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Score) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Each output with its dots run in `acc` (null: float64) and the heads summed in order; relu keeps a NaN, as
    /// the chain's `maximum` does.
    fn outputs(self: *const Score, acc: ?spec.Accum, order: spec.Order, want: []f64, mag: []f64) void {
        var t: [D]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..self.shape.n) |n| {
            var v: f64 = 0;
            var m: f64 = 0;
            for (0..HEADS) |h| {
                const qr = self.q[(h * self.shape.rows + r) * D ..][0..D];
                for (&t, qr, self.pool[n * D ..][0..D]) |*x, a, b| x.* = a * b;
                const d = if (acc) |ac| spec.sumIn(ac, order, &t) else host.sum(&t);
                const relu = if (std.math.isNan(d)) d else @max(d, 0);
                v = if (acc) |ac| spec.sumIn(ac, .sequential, &.{ v, relu }) else v + relu;
                m += host.sumAbs(&t);
            }
            want[r * self.shape.n + n] = v;
            mag[r * self.shape.n + n] = m;
        };
    }

    pub fn reference(self: *Score, want: []f64, mag: []f64) !void {
        self.outputs(null, .sequential, want, mag);
    }

    pub fn simulate(self: *Score, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const scratch = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(scratch);
        self.outputs(acc, order, out, scratch);
    }

    /// The chain it replaces on the same operands: `qsaScoreSheetComposed` over `qsaScoreK32tFromPooled`.
    pub fn replaced(self: *Score) ![]const u8 {
        const s = self.rig.stream();
        const k32t = try score.qsaScoreK32tFromPooled(s, self.args[1]);
        defer _ = mlx.mlx_array_free(k32t);
        const out = try score.qsaScoreSheetComposed(s, self.args[0], k32t);
        defer _ = mlx.mlx_array_free(out);
        return self.rig.bytesOf(&.{out});
    }

    /// The shipped wrapper on the same operands (null where its probe or eligibility declines).
    fn wrapper(self: *Score) !?[]const u8 {
        const out = (try score.qsaScoreFused(self.rig.stream(), self.args[0], self.args[1])) orelse return null;
        defer _ = mlx.mlx_array_free(out);
        return try self.rig.bytesOf(&.{out});
    }
};

fn depth(_: Shape) usize {
    // A dot's 128 terms, then the four heads.
    return D + HEADS;
}

const relu_drops_nan: spec.Known = .{ .gate = .nonfinite, .finding = "relu runs metal::max(dot, 0), which returns 0 for a NaN dot, where MLX's maximum (the chain it replaces) keeps the NaN: a NaN in q or a pooled key drops out of the score instead of reaching the selection" };

pub fn register(set: *kc.Set) !void {
    try set.add(Score, .init(set.a), .{
        .name = "qsa_score f32 (4 heads, relu, summed; NAX)",
        .nax = true,
        .accum = .f32,
        .depth = depth,
        .out = .f32,
        .replaces = "astype f32 -> matmul -> maximum 0 -> sum over heads (qsaScoreSheetComposed)",
        // rows: the base layout's 16-row group; n: a slab boundary (544 = 17 slabs) inside the indexer's domain,
        // which starts past the 2048-key budget (513 blocks).
        .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{ .size = 544, .step = 1, .min = 513 }, .{} },
        // Decode past the budget, a verify step, either side of the h4 layout's 128 rows, a prefill chunk.
        .served = &.{ .{ .rows = 1, .n = 513, .k = D }, .{ .rows = 8, .n = 2048, .k = D }, .{ .rows = 127, .n = 700, .k = D }, .{ .rows = 128, .n = 700, .k = D }, .{ .rows = 2048, .n = 2048, .k = D } },
        .known = relu_drops_nan,
    }, null);
}

// ── tests ──

test "kernel contract sparse_block: the declared qsa_score launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var k = Score.init(a);
    defer k.deinit();
    for ([_]Shape{ .{ .rows = 1, .n = 513, .k = D }, .{ .rows = 128, .n = 700, .k = D } }) |shape| {
        try std.testing.expect(try k.prepare(shape, .random, 3));
        try k.run();
        const ours = try a.dupe(u8, k.rig.outputBytes());
        defer a.free(ours);
        const theirs = (try k.wrapper()) orelse return error.SkipZigTest;
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}
