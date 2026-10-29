//! The spec area's kernel contract: DFlash2's grouped dynamic conv (bit-equal to its bf16 op chain), DFlash's per-row
//! top-k, and the MTP rerank's exact top-32 shortlist (partial and final). The selections check themselves
//! (`check`): an exact replay of their stated order. Served on every Apple GPU (M1 and up).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const dflash = @import("dflash/dflash.zig");
const mtp = @import("../../../sdk/spec/mtp/mtp.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const dyn_conv_launch: metal.Launch = .{
    .name = "msv_dflash_dyn_conv",
    .inputs = &.{ "x", "dyn", "base" },
    .outputs = &.{"y"},
    .source = kernelSource("kernels/spec/dflash_dyn_conv.metal"),
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

/// Threadgroup bytes: `sv`, `si` [8 * K] at K 32.
const topk_launch: metal.Launch = .{
    .name = "msv_dflash_topk_rows",
    .inputs = &.{"logits"},
    .outputs = &.{ "idx", "val" },
    .source = kernelSource("kernels/spec/dflash_topk.metal"),
    .tg_bytes = 2 * 8 * 32 * 4,
    .m1 = 704,
    .m2 = 704,
};

/// Threadgroup bytes: `sc_ord`, `sc_idx` [8 * 32].
const top32_partial_launch: metal.Launch = .{
    .name = "msv_mtp_top32_partial",
    .inputs = &.{"logits"},
    .outputs = &.{ "cand_ord", "cand_idx" },
    .source = kernelSource("kernels/spec/mtp_top32_partial.metal"),
    .header = kernelSource("kernels/spec/mtp_top32_ordinal_header.metal"),
    .tg_bytes = 2 * 8 * 32 * 4,
    .m1 = 1024,
    .m2 = 1024,
};

const top32_final_launch: metal.Launch = .{
    .name = "msv_mtp_top32_finalize",
    .inputs = &.{ "cand_ord", "cand_idx" },
    .outputs = &.{"token_ids"},
    .source = kernelSource("kernels/spec/mtp_top32_final.metal"),
    .tg_bytes = 2 * 8 * 32 * 4,
    .m1 = 1024,
    .m2 = 1024,
};

// ── grouped dynamic conv ──

/// A shape's rows are the block rows L, `n` the hidden width, `k` the taps; channel groups of GS share a dynamic
/// coefficient.
const DynConv = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    gs: usize,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    dyn: []f64 = &.{},
    base: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, gs: usize) DynConv {
        return .{ .rig = metal.Rig.init(a), .launch = &dyn_conv_launch, .gs = gs, .arena = .init(a) };
    }

    pub fn deinit(self: *DynConv) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const DynConv, shape: Shape) ?metal.Geometry {
        if (shape.n % self.gs != 0) return null;
        const l: c_int = @intCast(shape.rows);
        const h: c_int = @intCast(shape.n);
        var g = metal.Geometry{ .grid = .{ l * h, 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ 1, l, h }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("L", l);
        g.int("H", h);
        g.int("KS", shape.k);
        g.int("GS", self.gs);
        return g;
    }

    pub fn prepare(self: *DynConv, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const l = shape.rows;
        const h = shape.n;
        const groups = h / self.gs;
        self.x = try host.uniform(a, rng, l * h, -3, 3, .bfloat16);
        self.dyn = try host.uniform(a, rng, l * shape.k * groups, -1, 1, .bfloat16);
        self.base = try host.uniform(a, rng, shape.k * h, -1, 1, .bfloat16);
        if (fill == .nonfinite) host.plant(self.x, l, h);
        const rig = &self.rig;
        rig.reset();
        self.args = .{
            try rig.input(.bfloat16, &.{ 1, @intCast(l), @intCast(h) }, self.x),
            try rig.input(.bfloat16, &.{ 1, @intCast(l), @intCast(shape.k), @intCast(groups) }, self.dyn),
            try rig.input(.bfloat16, &.{ @intCast(shape.k), @intCast(h) }, self.base),
        };
        return true;
    }

    pub fn run(self: *DynConv) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const DynConv) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Each tap adds base * x then dyn * x, every product and sum rounded to bf16 (the chain's own roundings).
    pub fn reference(self: *const DynConv, want: []f64, mag: []f64) !void {
        const h = self.shape.n;
        const taps = self.shape.k;
        const groups = h / self.gs;
        for (0..self.shape.rows) |t| for (0..h) |c| {
            var acc: f64 = 0;
            var m: f64 = 0;
            for (0..taps) |tap| {
                const v = if (t >= tap) self.x[(t - tap) * h + c] else 0;
                const b = host.rt(.bfloat16, self.base[tap * h + c] * v);
                acc = host.rt(.bfloat16, acc + b);
                const d = host.rt(.bfloat16, self.dyn[(t * taps + tap) * groups + c / self.gs] * v);
                acc = acc + d;
                if (tap + 1 < taps) acc = host.rt(.bfloat16, acc);
                m += @abs(b) + @abs(d);
            }
            want[t * h + c] = acc;
            mag[t * h + c] = m;
        };
    }

    pub fn replaced(self: *DynConv) ![]const u8 {
        const y = try dflash.groupedDynConvOps(self.args[0], self.args[1], self.args[2], @intCast(self.gs), self.rig.stream());
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }

    fn wrapper(self: *DynConv) ![]const u8 {
        const y = try dflash.groupedDynConv(self.args[0], self.args[1], self.args[2], @intCast(self.gs), self.rig.stream());
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

// ── DFlash per-row top-k ──

/// A shape's rows are the block rows M, `k` the vocabulary; K is the selector's top-k.
const TopK = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kk: usize,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    logits: []f64 = &.{},
    args: [1]mlx.mlx_array = undefined,
    msg: [160]u8 = undefined,

    fn init(a: std.mem.Allocator, kk: usize, dt: mlx.mlx_dtype) TopK {
        return .{ .rig = metal.Rig.init(a), .launch = &topk_launch, .kk = kk, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *TopK) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const TopK, shape: Shape) ?metal.Geometry {
        if (shape.k < 256 * self.kk) return null;
        const m: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ 256, m, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ 1, m, @intCast(self.kk) }, .int32);
        g.out(&.{ 1, m, @intCast(self.kk) }, .float32);
        g.dtype("T", self.dt);
        g.int("V", shape.k);
        g.int("K", self.kk);
        return g;
    }

    pub fn prepare(self: *TopK, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.logits = try host.uniform(a, rng, shape.rows * shape.k, -10, 10, self.dt);
        if (fill == .nonfinite) host.plant(self.logits, shape.rows, shape.k);
        self.rig.reset();
        self.args = .{try self.rig.input(self.dt, &.{ 1, @intCast(shape.rows), @intCast(shape.k) }, self.logits)};
        return true;
    }

    pub fn run(self: *TopK) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const TopK) usize {
        return 2 * self.shape.rows * self.kk;
    }

    pub fn reference(_: *const TopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn fail(self: *TopK, comptime fmt: []const u8, args: anytype) ?[]const u8 {
        return std.fmt.bufPrint(&self.msg, fmt, args) catch "check failed";
    }

    /// The stated rule: each row's K largest values in descending order (NaN and -Inf never selected), each id
    /// distinct and holding its value; equal values may come in either id order.
    pub fn check(self: *TopK, _: spec.Gate) !?[]const u8 {
        const out = self.rig.outputBytes();
        const v = self.shape.k;
        const kk = self.kk;
        const n = self.shape.rows * kk;
        const a = self.rig.a;
        const sorted = try a.alloc(f64, v);
        defer a.free(sorted);
        for (0..self.shape.rows) |r| {
            const row = self.logits[r * v ..][0..v];
            var len: usize = 0;
            for (row) |x| if (!std.math.isNan(x) and x != -std.math.inf(f64)) {
                sorted[len] = x;
                len += 1;
            };
            std.mem.sort(f64, sorted[0..len], {}, std.sort.desc(f64));
            for (0..kk) |j| {
                const id: usize = @intCast(std.mem.readInt(i32, out[4 * (r * kk + j) ..][0..4], .little));
                const val = spec.read(.f32, out[4 * n ..], r * kk + j);
                if (id >= v) return self.fail("row {d} slot {d}: id {d} past the vocabulary", .{ r, j, id });
                if (val != sorted[j]) return self.fail("row {d} slot {d}: value {d} where the {d}th largest is {d}", .{ r, j, val, j, sorted[j] });
                if (row[id] != val) return self.fail("row {d} slot {d}: id {d} holds {d}, not {d}", .{ r, j, id, row[id], val });
                for (0..j) |i| if (std.mem.readInt(i32, out[4 * (r * kk + i) ..][0..4], .little) == @as(i32, @intCast(id)))
                    return self.fail("row {d}: id {d} selected twice", .{ r, id });
            }
        }
        return null;
    }

    fn wrapper(self: *TopK) ![]const u8 {
        const got = (try dflash.topKRows(self.args[0], self.kk, self.rig.stream())) orelse return error.Declined;
        defer for (got) |o| {
            _ = mlx.mlx_array_free(o);
        };
        return self.rig.bytesOf(&got);
    }
};

// ── MTP top-32 shortlist ──

const TOPK32 = 32;
const TILES = 64;
const CANDS = TILES * TOPK32;

/// The kernels' monotone map: value order, NaN above every number, -0 == +0.
fn ordinal(v: f64) u32 {
    if (std.math.isNan(v)) return 0xFFFFFFFF;
    const f: f32 = @floatCast(v);
    if (f == 0) return 0x80000000;
    const u: u32 = @bitCast(f);
    return if (u & 0x80000000 != 0) ~u else u | 0x80000000;
}

const Pair = struct {
    ord: u32,
    idx: u32,

    fn before(_: void, a: Pair, b: Pair) bool {
        return a.ord > b.ord or (a.ord == b.ord and a.idx > b.idx);
    }
};

/// One vocabulary row (`k` its width; a shape's rows are ignored). `final` tests the second stage on candidate
/// lists the first would write.
const Top32 = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    final: bool,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    logits: []f64 = &.{},
    cands: []Pair = &.{},
    args: [2]mlx.mlx_array = undefined,
    msg: [160]u8 = undefined,

    fn init(a: std.mem.Allocator, final: bool, dt: mlx.mlx_dtype) Top32 {
        return .{ .rig = metal.Rig.init(a), .launch = if (final) &top32_final_launch else &top32_partial_launch, .final = final, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Top32) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Top32, shape: Shape) ?metal.Geometry {
        if (self.final) {
            var g = metal.Geometry{ .grid = .{ 256, 1, 1 }, .tg = .{ 256, 1, 1 } };
            g.out(&.{TOPK32}, .uint32);
            return g;
        }
        if (shape.k < TILES * 256 or (shape.k + TILES * 256 - 1) / (TILES * 256) > 32) return null;
        var g = metal.Geometry{ .grid = .{ TILES * 256, 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{CANDS}, .uint32);
        g.out(&.{CANDS}, .uint32);
        g.int("RC", shape.k);
        return g;
    }

    pub fn prepare(self: *Top32, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // Coarse logits: ties are common at bf16, which the (value, index) order breaks.
        self.logits = try host.uniform(a, rng, shape.k, -12, 12, self.dt);
        if (fill == .nonfinite) {
            host.plant(self.logits, 1, shape.k);
            self.logits[shape.k / 5] = std.math.inf(f64);
            self.logits[shape.k / 7] = -std.math.inf(f64);
        }
        self.rig.reset();
        if (!self.final) {
            self.args[0] = try self.rig.input(self.dt, &.{@intCast(shape.k)}, self.logits);
            return true;
        }
        // The first stage's output: each tile's top 32, best first.
        self.cands = try a.alloc(Pair, CANDS);
        const tile = try a.alloc(Pair, (shape.k + TILES * 256 - 1) / TILES + 256);
        for (0..TILES) |t| {
            var n: usize = 0;
            var i = t * 256;
            while (i < shape.k) : (i += TILES * 256) for (i..@min(i + 256, shape.k)) |j| {
                tile[n] = .{ .ord = ordinal(self.logits[j]), .idx = @intCast(j) };
                n += 1;
            };
            std.mem.sort(Pair, tile[0..n], {}, Pair.before);
            @memcpy(self.cands[t * TOPK32 ..][0..TOPK32], tile[0..TOPK32]);
        }
        const ords = try a.alloc(f64, CANDS);
        const idxs = try a.alloc(f64, CANDS);
        for (self.cands, ords, idxs) |c, *o, *x| {
            o.* = @floatFromInt(c.ord);
            x.* = @floatFromInt(c.idx);
        }
        self.args = .{ try self.rig.input(.uint32, &.{CANDS}, ords), try self.rig.input(.uint32, &.{CANDS}, idxs) };
        return true;
    }

    pub fn run(self: *Top32) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..if (self.final) 2 else 1]);
    }

    pub fn count(self: *const Top32) usize {
        return if (self.final) TOPK32 else 2 * CANDS;
    }

    pub fn reference(_: *const Top32, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn fail(self: *Top32, comptime fmt: []const u8, args: anytype) ?[]const u8 {
        return std.fmt.bufPrint(&self.msg, fmt, args) catch "check failed";
    }

    fn word(self: *Top32, i: usize) u32 {
        return std.mem.readInt(u32, self.rig.outputBytes()[4 * i ..][0..4], .little);
    }

    /// The stated order, exactly: (ordinal, index) descending per tile (stage 1, best first), the 32 best of all
    /// candidates ascending (stage 2).
    pub fn check(self: *Top32, _: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        if (self.final) {
            const all = try a.dupe(Pair, self.cands);
            defer a.free(all);
            std.mem.sort(Pair, all, {}, Pair.before);
            for (0..TOPK32) |r| if (self.word(TOPK32 - 1 - r) != all[r].idx)
                return self.fail("slot {d}: id {d} where the order gives {d}", .{ TOPK32 - 1 - r, self.word(TOPK32 - 1 - r), all[r].idx });
            return null;
        }
        const k = self.shape.k;
        const tile = try a.alloc(Pair, (k + TILES * 256 - 1) / TILES + 256);
        defer a.free(tile);
        for (0..TILES) |t| {
            var n: usize = 0;
            var i = t * 256;
            while (i < k) : (i += TILES * 256) for (i..@min(i + 256, k)) |j| {
                tile[n] = .{ .ord = ordinal(self.logits[j]), .idx = @intCast(j) };
                n += 1;
            };
            std.mem.sort(Pair, tile[0..n], {}, Pair.before);
            for (0..TOPK32) |r| {
                const at = t * TOPK32 + r;
                if (self.word(at) != tile[r].ord or self.word(CANDS + at) != tile[r].idx)
                    return self.fail("tile {d} slot {d}: ({x}, {d}) where the order gives ({x}, {d})", .{ t, r, self.word(at), self.word(CANDS + at), tile[r].ord, tile[r].idx });
            }
        }
        return null;
    }
};

fn depthNone(_: Shape) usize {
    return 0;
}

fn convDepth(shape: Shape) usize {
    return 2 * shape.k;
}

fn selection(comptime name: []const u8, rule: []const u8, served: []const Shape) spec.Spec {
    return .{ .name = name, .accum = .none, .depth = depthNone, .out = .f32, .nan = .{ .stated = rule }, .served = served };
}

const topk_rule = "NaN and -Inf are never selected; +Inf ranks above every number; equal values in either id order";
const top32_rule = "NaN ranks above every number, then +Inf; -0 equals +0; ties go to the higher index";

const block_rows: []const Shape = &.{ .{ .rows = 8, .n = 2560, .k = 2 }, .{ .rows = 9, .n = 2560, .k = 2 } };
const draft_vocab: []const Shape = &.{ .{ .rows = 8, .n = 1, .k = 248320 }, .{ .rows = 1, .n = 1, .k = 248320 } };
const vocab_row: []const Shape = &.{.{ .rows = 1, .n = 1, .k = 248320 }};
const qwen3_row: []const Shape = &.{.{ .rows = 1, .n = 1, .k = 151936 }};

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(DynConv, .init(a, 16), .{
        .name = "dflash_dyn_conv bf16 (2 taps, groups of 16)",
        // Every product and sum is rounded to bf16, as the op chain it replaces does.
        .accum = .bf16,
        .depth = convDepth,
        .out = .bf16,
        .replaces = "per tap: multiply -> add, multiply -> add (bf16)",
        .served = block_rows,
    }, null);
    try set.add(TopK, .init(a, 16, .bfloat16), selection("dflash_topk_rows bf16 (K 16)", topk_rule, draft_vocab), null);
    try set.add(TopK, .init(a, 32, .float32), selection("dflash_topk_rows f32 (K 32)", topk_rule, draft_vocab[1..]), null);
    try set.add(Top32, .init(a, false, .bfloat16), selection("mtp_top32_partial bf16", top32_rule, vocab_row), null);
    try set.add(Top32, .init(a, false, .float32), selection("mtp_top32_partial f32", top32_rule, qwen3_row), null);
    try set.add(Top32, .init(a, true, .bfloat16), selection("mtp_top32_final", top32_rule, vocab_row), null);
}

// ── tests ──

test "kernel contract spec: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    {
        var t = DynConv.init(a, 16);
        defer t.deinit();
        try std.testing.expect(try t.prepare(block_rows[0], .random, 5));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try t.wrapper(), ours);
    }
    {
        var t = TopK.init(a, 16, .bfloat16);
        defer t.deinit();
        try std.testing.expect(try t.prepare(draft_vocab[0], .random, 6));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try t.wrapper(), ours);
    }
    // The two shortlist stages chained equal `draftTop32`.
    var p = Top32.init(a, false, .bfloat16);
    defer p.deinit();
    try std.testing.expect(try p.prepare(vocab_row[0], .random, 8));
    try p.run();
    var f = Top32.init(a, true, .bfloat16);
    defer f.deinit();
    try std.testing.expect(try f.prepare(vocab_row[0], .random, 8));
    try f.rig.launch(f.launch, f.geometry(f.shape).?, &.{ p.rig.output(0), p.rig.output(1) });
    const ours = try a.dupe(u8, f.rig.outputBytes());
    defer a.free(ours);
    const ids = try mtp.draftTop32(p.rig.stream(), p.args[0], 248320);
    defer _ = mlx.mlx_array_free(ids);
    try std.testing.expectEqualSlices(u8, try f.rig.bytesOf(&.{ids}), ours);
}
