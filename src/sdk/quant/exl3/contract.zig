//! The EXL3 format's kernel contract: the one-row decode GEMV (`sushi_glm_down_one`) under every codebook, the STQ
//! tables included, against the host trellis decode; and the routed decode's finish reduce (`sushi_exl3_down_reduce`,
//! which every codebook shares): each slot's down output through the 128-point Hadamard, scaled by its expert's
//! `svh`, summed over the row's top-k by the router weights. Listed in `kernel_contract/registry.zig`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const exl3 = @import("format.zig");
const sushi = @import("sushi_exl3");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;
const kernels = exl3.kernels;

/// Experts in the `svh` bank the slots pick from.
const EXPERTS = 16;
/// The kernel's 1/sqrt(128), as its source spells it.
const SCV: f64 = @as(f32, 0.08838834764831845);

/// `downFinishReduce`'s kernel: `TOPK` simdgroups a threadgroup, `vals[TOPK * 128]` floats of threadgroup memory
/// (the largest served top-k is 10).
fn Launch(comptime m: u32) type {
    return struct {
        const launch: metal.Launch = .{
            .name = "sushi_exl3_down_reduce",
            .inputs = &.{ "inner", "svh", "slots", "sc" },
            .outputs = &.{"y"},
            .source = kernels.REDUCE_SOURCE,
            .tg_bytes = 10 * 128 * 4,
            .m1 = m,
            .m2 = m,
        };
    };
}

/// Shapes are (rows, ODIM, top-k).
const Reduce = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    /// The output's dtype, and the router weights'.
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    inner: []f64 = &.{},
    svh: []f64 = &.{},
    slots: []f64 = &.{},
    sc: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) Reduce {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Reduce) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn elem(self: *const Reduce) spec.Elem {
        return if (self.dt == .float32) .f32 else .bf16;
    }

    /// `downFinishReduceWithShared`'s config: grid, group, output, template args in its order.
    pub fn geometry(self: *const Reduce, shape: Shape) ?metal.Geometry {
        if (shape.n == 0 or shape.n % 128 != 0 or shape.k < 1 or shape.k > kernels.REDUCE_MAX_TOPK) return null;
        const rows: c_int = @intCast(shape.rows);
        const od: c_int = @intCast(shape.n);
        const topk: c_int = @intCast(shape.k);
        var g = metal.Geometry{ .grid = .{ 32 * topk * @divExact(od, 128), rows, 1 }, .tg = .{ 32 * topk, 1, 1 } };
        if (rows == 1) g.out(&.{od}, self.dt) else g.out(&.{ rows, od }, self.dt);
        g.dtype("T", self.dt);
        g.int("ODIM", od);
        g.int("TOPK", topk);
        return g;
    }

    pub fn prepare(self: *Reduce, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const slots_n = shape.rows * shape.k;
        self.inner = try a.alloc(f64, slots_n * shape.n);
        self.svh = try a.alloc(f64, EXPERTS * shape.n);
        self.slots = try a.alloc(f64, slots_n);
        self.sc = try a.alloc(f64, slots_n);
        for (self.inner) |*v| v.* = if (fill == .stress)
            @floatFromInt(1 + rng.uintLessThan(u32, 1024))
        else
            spec.roundTo(.f16, rng.float(f64) * 4 - 2);
        for (self.svh) |*v| v.* = if (fill == .stress) 1 else spec.roundTo(.f16, (0.5 + rng.float(f64)) * @as(f64, if (rng.boolean()) 1 else -1));
        for (0..shape.rows) |r| {
            // Distinct experts a row, as the router picks them.
            var used: u32 = 0;
            for (self.slots[r * shape.k ..][0..shape.k]) |*s| {
                var e = rng.uintLessThan(u32, EXPERTS);
                while (used & (@as(u32, 1) << @intCast(e)) != 0) e = (e + 1) % EXPERTS;
                used |= @as(u32, 1) << @intCast(e);
                s.* = @floatFromInt(e);
            }
        }
        for (self.sc) |*v| v.* = spec.roundTo(self.elem(), if (fill == .stress) 1 else 0.05 + rng.float(f64));
        if (fill == .nonfinite) {
            const at = [_][3]usize{ .{ 0, 0, 3 }, .{ 1, 1, 70 }, .{ 2, 0, 127 }, .{ 3, 0, 200 } };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64) };
            for (at, put) |p, v| if (p[0] < shape.rows and p[1] < shape.k and p[2] < shape.n) {
                self.inner[(p[0] * shape.k + p[1]) * shape.n + p[2]] = v;
            };
        }
        const rig = &self.rig;
        rig.reset();
        const od: c_int = @intCast(shape.n);
        self.args = .{
            try rig.input(.float16, &.{ @intCast(slots_n), od }, self.inner),
            try rig.input(.float16, &.{ EXPERTS, od }, self.svh),
            try rig.input(.uint32, &.{@intCast(slots_n)}, self.slots),
            try rig.input(self.dt, &.{@intCast(slots_n)}, self.sc),
        };
        return true;
    }

    pub fn run(self: *Reduce) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Reduce) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Element `p` of slot `s`'s 128-block `b` after the Hadamard, unscaled; its terms' magnitude in `mag`.
    fn hadamard(self: *const Reduce, s: usize, b: usize, p: usize, terms: ?[]f64, mag: *f64) f64 {
        const row = self.inner[s * self.shape.n + b * 128 ..][0..128];
        var sum: f64 = 0;
        mag.* = 0;
        for (row, 0..) |v, c| {
            const t = if (@popCount(p & c) % 2 == 0) v else -v;
            if (terms) |out| out[c] = t;
            sum += t;
            mag.* += @abs(v);
        }
        return sum;
    }

    fn outputsIn(self: *const Reduce, want: []f64, mags: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const n = self.shape.n;
        const k = self.shape.k;
        const al = self.arena.child_allocator;
        const terms = try al.alloc(f64, 128);
        defer al.free(terms);
        const slot_terms = try al.alloc(f64, k);
        defer al.free(slot_terms);
        for (0..self.shape.rows) |r| for (0..n) |o| {
            const b = o / 128;
            const p = o % 128;
            var total_mag: f64 = 0;
            for (0..k) |j| {
                const s = r * k + j;
                const e: usize = @intFromFloat(self.slots[s]);
                var mag: f64 = 0;
                var h = self.hadamard(s, b, p, terms, &mag);
                if (acc) |ac| h = spec.sumIn(ac, order, terms);
                const scale = SCV * self.svh[e * n + o];
                slot_terms[j] = h * scale * self.sc[s];
                total_mag += mag * @abs(scale * self.sc[s]);
            }
            want[r * n + o] = if (acc) |ac| spec.sumIn(ac, order, slot_terms) else sumF64(slot_terms);
            if (mags) |m| m[r * n + o] = total_mag;
        };
    }

    pub fn reference(self: *const Reduce, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Reduce, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    /// The shipped wrapper on the same inputs: its output's bytes (the declared launch must be its launch).
    fn wrapper(self: *Reduce) ![]const u8 {
        const x = self.args;
        const y = try kernels.downFinishReduce(self.rig.stream(), x[0], x[1], x[2], x[3], @intCast(self.shape.n), @intCast(self.shape.rows), @intCast(self.shape.k), self.dt);
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

/// The one-row cooperative down GEMV of `dec`: x [slots, IDIM] half (prepared order, `weightRow`) through each slot's
/// expert's trellis bank [E, IDIM/16, ODIM/16, N] into y [slots, ODIM] half. Shapes are (slots, ODIM, IDIM).
const OneRow = struct {
    rig: metal.Rig,
    /// Built at registration (a STQ table's rows lead its header); owned.
    launch: *const metal.Launch,
    dec: exl3.Decode,
    rate: u32,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    /// Every expert's decoded weights, [E, IDIM, ODIM].
    w: []f64 = &.{},
    slots: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, comptime cb: exl3.Codebook, comptime win: exl3.Window, rate: u32, m: u32) !OneRow {
        const dec: exl3.Decode = .tabled(cb, win);
        kernels.setDecodeParams(dec);
        const l = try a.create(metal.Launch);
        l.* = .{
            .name = "sushi_glm_down_one",
            .inputs = &.{ "x", "trellis", "slots" },
            .outputs = &.{"y"},
            .source = kernels.ONE_ROW_COOP_SOURCE,
            .header = try kernels.withTableRows(cb, comptime kernels.codebookHelpers(cb, win)),
            .tg_bytes = 2 * 4 * 256 * 4,
            .m1 = m,
            .m2 = m,
        };
        return .{ .rig = metal.Rig.init(a), .launch = l, .dec = dec, .rate = rate, .arena = .init(a) };
    }

    pub fn deinit(self: *OneRow) void {
        self.rig.deinit();
        self.arena.deinit();
        self.rig.a.destroy(self.launch);
    }

    /// `downLaneCoopWith(.., one_row = true)`'s config; it declines unless the input is whole 128-blocks and the
    /// output an even count of tiles.
    pub fn geometry(self: *const OneRow, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.k == 0 or shape.n == 0 or shape.k % 128 != 0 or shape.n % 32 != 0) return null;
        const id: c_int = @intCast(shape.k);
        const od: c_int = @intCast(shape.n);
        const topk: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ @divExact(od, 16) * 64, topk, 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ topk, od }, .float16);
        g.int("IDIM", id);
        g.int("ODIM", od);
        g.int("NHW", self.rate);
        return g;
    }

    pub fn prepare(self: *OneRow, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const it = shape.k / 16;
        const ot = shape.n / 16;
        const tile_hw = self.rate;
        const words = try a.alloc(u16, EXPERTS * it * ot * tile_hw);
        for (words) |*v| v.* = rng.int(u16);
        // Gate 4: a bank's second half of input tiles repeats its first, against mirrored inputs, so sums cancel.
        if (fill == .stress) for (0..EXPERTS) |e| for (it / 2..it) |tk| {
            const at = (e * it + tk) * ot * tile_hw;
            @memcpy(words[at..][0 .. ot * tile_hw], words[at - it / 2 * ot * tile_hw ..][0 .. ot * tile_hw]);
        };
        self.w = try a.alloc(f64, EXPERTS * shape.k * shape.n);
        const rate: sushi.format.Rate = .{ .n = self.rate };
        var tile: [sushi.format.TILE_VALUES]u16 = undefined;
        for (0..EXPERTS) |e| for (0..it) |tk| for (0..ot) |o| {
            sushi.format.decodeTile(words[((e * it + tk) * ot + o) * tile_hw ..][0..tile_hw], rate, self.dec, &tile);
            for (tile, 0..) |h, pos| self.w[(e * shape.k + tk * 16 + pos / 16) * shape.n + o * 16 + pos % 16] = @as(f16, @bitCast(h));
        };
        self.slots = try a.alloc(f64, shape.rows);
        for (self.slots) |*v| v.* = @floatFromInt(rng.uintLessThan(u32, EXPERTS));
        self.x = try a.alloc(f64, shape.rows * shape.k);
        for (self.x) |*v| v.* = spec.roundTo(.f16, rng.float(f64) * 2 - 1);
        if (fill == .stress) for (0..shape.rows) |r| {
            const row = self.x[r * shape.k ..][0..shape.k];
            const half = shape.k / 2;
            for (row[0..half], row[half..], 0..) |*lo, *hi, i| {
                lo.* = @floatFromInt(1 + rng.uintLessThan(u32, 1024));
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
        const wv = try a.alloc(f64, words.len);
        for (wv, words) |*o, v| o.* = @floatFromInt(v);
        const rig = &self.rig;
        rig.reset();
        self.args = .{
            try rig.input(.float16, &.{ @intCast(shape.rows), @intCast(shape.k) }, self.x),
            try rig.input(.uint16, &.{ EXPERTS, @intCast(it), @intCast(ot), @intCast(tile_hw) }, wv),
            try rig.input(.uint32, &.{@intCast(shape.rows)}, self.slots),
        };
        return true;
    }

    pub fn run(self: *OneRow) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const OneRow) usize {
        return self.shape.rows * self.shape.n;
    }

    fn outputsIn(self: *const OneRow, want: []f64, mags: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const n = self.shape.n;
        const k = self.shape.k;
        const al = self.arena.child_allocator;
        const terms = try al.alloc(f64, k);
        defer al.free(terms);
        for (0..self.shape.rows) |r| {
            const e: usize = @intFromFloat(self.slots[r]);
            for (0..n) |o| {
                var mag: f64 = 0;
                for (terms, self.x[r * k ..][0..k], 0..) |*t, xv, i| {
                    t.* = xv * self.w[(e * k + weightRow(i)) * n + o];
                    mag += @abs(t.*);
                }
                want[r * n + o] = if (acc) |ac| spec.sumIn(ac, order, terms) else sumF64(terms);
                if (mags) |m| m[r * n + o] = mag;
            }
        }
    }

    pub fn reference(self: *const OneRow, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const OneRow, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    /// The shipped wrapper on the same inputs, under the same decode: its output's bytes.
    fn wrapper(self: *OneRow) ![]const u8 {
        kernels.setDecodeParams(self.dec);
        const y = try kernels.downLaneCoopWith(self.rig.stream(), self.args[0], self.args[1], self.args[2], true);
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

/// The weight row input `i` multiplies: the kernel reads x in its prepared order, lane quad q's four inputs being
/// tile rows 2q, 2q + 1, 2q + 8 and 2q + 9 (the tensor-core fragment rows of the lane's weights).
fn weightRow(i: usize) usize {
    const m = i % 16;
    const quad = [4]usize{ 0, 1, 8, 9 };
    return i - m + 2 * (m / 4) + quad[m % 4];
}

fn sumF64(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += v;
    return s;
}

/// The Hadamard's seven butterfly levels, the scale products, then the top-k sum.
fn depth(shape: Shape) usize {
    return 9 + shape.k;
}

/// Each codebook at a rate it ships (STD1 and SPEC1 under their tables' w15, PAIR1 under its own w16).
/// `m` is the least thread limit M1/M2 compile any served shape's instantiation for (`offline.zig`).
const OneRowCase = struct { cb: exl3.Codebook, win: exl3.Window, n: u32, m: u32 };
const one_row_cases = [_]OneRowCase{
    .{ .cb = .mcg, .win = .w14, .n = 40, .m = 928 },
    .{ .cb = .mul1, .win = .w16, .n = 48, .m = 1024 },
    .{ .cb = .spec1, .win = .w15, .n = 32, .m = 928 },
    .{ .cb = .std1, .win = .w15, .n = 48, .m = 1024 },
    .{ .cb = .pair1, .win = .w16, .n = 48, .m = 224 },
};

/// Each thread's chain over its input tiles, then the threadgroup's sum of 16 rows by 4 simdgroups.
fn oneRowDepth(shape: Shape) usize {
    return shape.k / 128 + 64;
}

const one_row_shapes: []const Shape = &.{ .{ .rows = 8, .n = 1024, .k = 512 }, .{ .rows = 5, .n = 2048, .k = 1536 } };

fn oneRowSpec(comptime c: OneRowCase) spec.Spec {
    return .{
        .name = std.fmt.comptimePrint("exl3 down_one {t} {t} n{d}", .{ c.cb, c.win, c.n }),
        .accum = .f32,
        .depth = oneRowDepth,
        // The fold scale (STD1, SPEC1, PAIR1) is a power of two: exact.
        .out = .f16,
        .tiles = .{ .{}, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 128, .step = 128, .min = 128 } },
        .served = one_row_shapes,
        // The decoded weights are the trellis's pseudo-random signed halves, which no input can line up: half-precision
        // sums of the stress inputs miss its bound by 15-30x, not 100x.
        .sums = .{ .f16 = 10, .bf16 = 10, .exact = false, .why = "trellis weights are pseudo-random signed halves the stress cannot choose; half-precision sums miss by 10x+, not 100x" },
    };
}

const served: []const Shape = &.{ .{ .rows = 1, .n = 4096, .k = 8 }, .{ .rows = 3, .n = 2048, .k = 10 }, .{ .rows = 2, .n = 6144, .k = 8 } };

fn reduceSpec(comptime out: spec.Elem) spec.Spec {
    return .{
        .name = "exl3 down_reduce " ++ (if (out == .f32) "f32" else "bf16"),
        .accum = .f32,
        .depth = depth,
        .out = out,
        .tiles = .{ .{}, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 8, .min = 1 } },
        .served = served,
        // The 1/sqrt(128) scale is not exact in fp32, so gate 4 holds the sums to the ordinary bound.
        .sums = if (out == .f32) .{ .exact = false } else .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the bf16 output's rounding is coarser than either half-precision sum's error" },
    };
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    inline for (one_row_cases) |c| try set.add(OneRow, try .init(a, c.cb, c.win, c.n, c.m), oneRowSpec(c), null);
    try set.add(Reduce, .init(a, &Launch(928).launch, .float32), reduceSpec(.f32), null);
    try set.add(Reduce, .init(a, &Launch(928).launch, .bfloat16), reduceSpec(.bf16), null);
}

// ── tests ──

/// The declared launch at `shape` on random inputs: its output's bytes, owned by the caller.
fn driftRun(comptime T: type, a: std.mem.Allocator, k: kc.gates.Kernel, shape: Shape) ![]u8 {
    const t: *T = @ptrCast(@alignCast(k.ctx));
    try std.testing.expect(try t.prepare(shape, .random, 7));
    if (T == OneRow) kernels.setDecodeParams(t.dec);
    try t.run();
    return a.dupe(u8, t.rig.outputBytes());
}

test "kernel contract exl3: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const one_row = std.mem.startsWith(u8, k.spec.name, "exl3 down_one");
        for (k.spec.served) |shape| {
            const ours = if (one_row) try driftRun(OneRow, a, k, shape) else try driftRun(Reduce, a, k, shape);
            defer a.free(ours);
            const theirs = if (one_row) try @as(*OneRow, @ptrCast(@alignCast(k.ctx))).wrapper() else try @as(*Reduce, @ptrCast(@alignCast(k.ctx))).wrapper();
            if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
            try std.testing.expectEqualSlices(u8, theirs, ours);
        }
    }
}
