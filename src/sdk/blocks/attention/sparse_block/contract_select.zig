//! The QSA block select in the kernel contract: per query row, the top-K visible pooled blocks by (score
//! descending, index ascending), emitted ascending with INT_MAX past the row's count (`qsaChunkSelect`'s
//! torch.topk semantics, exact where the composed chain's index bias only approximates the tie rule). Three
//! launches over one radix body: the single-threadgroup select (`msv_qsa_select`), and the split pair the decode
//! widths take past 24k blocks (`msv_qsa_select_split_local`: each of 16 slices' own top-K; then
//! `msv_qsa_select_split_merge` over those candidates), chained against the wrapper below. Rows = query rows,
//! n = pooled blocks, k = K.

const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const kc = @import("../../../testing/kernel_contract/root.zig");
const select = @import("select.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

const SENTINEL: i32 = std.math.maxInt(i32);
const G: usize = @intCast(select.QSA_SELECT_SPLIT_G);
/// Flash Next's block budget: 2048 keys at ratio 4.
const KB = 512;
/// The slice kernel's threadgroup (the wrapper's unforced width).
const SLICE_TG = 512;

const Mode = enum { single, local, merge };

/// The radix histogram at 11-bit digits (2048 atomic bins), two per-simdgroup sums at 32 simdgroups, four words.
const TG_BYTES = 2048 * 4 + 2 * 32 * 4 + 4 * 4;

const launches = [_]metal.Launch{
    .{ .name = "msv_qsa_select", .inputs = &.{ "scores", "bounds" }, .outputs = &.{"ids"}, .source = select.QSA_SELECT_KERNEL_SOURCE, .header = select.QSA_SELECT_KERNEL_HEADER, .tg_bytes = TG_BYTES, .m1 = 1024, .m2 = 1024 },
    .{ .name = "msv_qsa_select_split_local", .inputs = &.{ "scores", "bounds" }, .outputs = &.{"ids"}, .source = select.QSA_SELECT_SPLIT_LOCAL_SOURCE, .header = select.QSA_SELECT_KERNEL_HEADER, .tg_bytes = TG_BYTES, .m1 = 1024, .m2 = 1024 },
    .{ .name = "msv_qsa_select_split_merge", .inputs = &.{ "scores", "local_ids" }, .outputs = &.{"ids"}, .source = select.QSA_SELECT_SPLIT_MERGE_SOURCE, .header = select.QSA_SELECT_KERNEL_HEADER, .tg_bytes = TG_BYTES, .m1 = 1024, .m2 = 1024 },
};

/// The kernels' order key: ascending with the score, NaN above every number, -0.0 and +0.0 one key.
fn ord(v: f32) u32 {
    if (std.math.isNan(v)) return 0xffff_ffff;
    if (v == 0) return 0x8000_0000;
    const u: u32 = @bitCast(v);
    return if (u & 0x8000_0000 != 0) ~u else u | 0x8000_0000;
}

const Select = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    mode: Mode,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    scores: []f32 = &.{},
    bounds: []i32 = &.{},
    /// The merge's candidates: each slice's exact top-K (what the local launch writes).
    cands: []i32 = &.{},
    args: [2]mlx.mlx_array = undefined,
    why: [160]u8 = undefined,

    fn init(a: std.mem.Allocator, mode: Mode) Select {
        return .{ .rig = metal.Rig.init(a), .launch = &launches[@backingInt(mode)], .mode = mode, .arena = .init(a) };
    }

    pub fn deinit(self: *Select) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    /// The wrapper's dispatches: one threadgroup a row (its width by rows, `qsaSelectTgFor`), or G slice groups a
    /// row at 512 threads, then one 1024-thread group a row for the merge; 8-bit digits to 15 rows, else 11.
    pub fn geometry(self: *const Select, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.k < 1 or shape.k > shape.n) return null;
        const rows: c_int = @intCast(shape.rows);
        const k: c_int = @intCast(shape.k);
        const tg: c_int = switch (self.mode) {
            .single => select.qsaSelectTgFor(rows),
            .local => SLICE_TG,
            .merge => select.QSA_SELECT_TG_DEFAULT,
        };
        const groups: c_int = if (self.mode == .local) @intCast(G) else 1;
        var g = metal.Geometry{ .grid = .{ tg * groups, rows, 1 }, .tg = .{ tg, 1, 1 } };
        g.out(&.{ 1, rows, if (self.mode == .local) k * groups else k }, .int32);
        g.int("TGS", tg);
        g.int("K", k);
        g.int("DIGITS", select.qsaSelectDigitBits(rows));
        return g;
    }

    /// Scores on a 1/16 grid in [0, 0.5] with zeros of both signs: ties everywhere, so the index rule decides.
    /// Bounds: row 0 sees every block, row 1 exactly K, row 2 K + 1, row 3 a negative bound (none), the rest
    /// random. Non-finite: NaN, +Inf, -Inf and -0.0 in every row's visible span, and a NaN past row 1's bound.
    pub fn prepare(self: *Select, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.scores = try a.alloc(f32, shape.rows * shape.n);
        for (self.scores) |*s| {
            const r = rng.uintLessThan(u32, 12);
            s.* = if (r == 0) 0 else if (r == 1) -0.0 else @as(f32, @floatFromInt(rng.uintLessThan(u32, 9))) / 16;
        }
        self.bounds = try a.alloc(i32, shape.rows);
        for (self.bounds, 0..) |*b, r| {
            const n: i32 = @intCast(shape.n);
            const k: i32 = @intCast(shape.k);
            b.* = switch (r) {
                0 => n,
                1 => k,
                2 => @min(k + 1, n),
                3 => -3,
                else => @intCast(rng.uintAtMost(u32, @intCast(n))),
            };
        }
        if (fill == .nonfinite) for (0..shape.rows) |r| {
            const row = self.scores[r * shape.n ..][0..shape.n];
            const plants = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), -0.0 };
            for (plants, 0..) |p, i| {
                const at = 3 + 4 * i;
                if (at < shape.n) row[at] = p;
            }
            if (r == 1 and shape.k + 1 < shape.n) row[shape.n - 1] = std.math.nan(f32);
        };
        self.rig.reset();
        const vals = try a.alloc(f64, self.scores.len);
        for (vals, self.scores) |*v, s| v.* = s;
        self.args[0] = try self.rig.input(.float32, &.{ 1, @intCast(shape.rows), @intCast(shape.n) }, vals);
        if (self.mode == .merge) {
            self.cands = try self.expected(.local);
            self.args[1] = try self.rig.inputRaw(.int32, &.{ 1, @intCast(shape.rows), @intCast(G * shape.k) }, std.mem.sliceAsBytes(self.cands));
        } else {
            self.args[1] = try self.rig.inputRaw(.int32, &.{@intCast(shape.rows)}, std.mem.sliceAsBytes(self.bounds));
        }
        return true;
    }

    pub fn run(self: *Select) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Select) usize {
        return self.shape.rows * self.shape.k * @as(usize, if (self.mode == .local) G else 1);
    }

    /// The top `k` of `ids` by (key descending, index ascending), written ascending into `out` with INT_MAX after.
    fn topInto(self: *const Select, row: []const f32, ids: []i32, out: []i32) void {
        const Ctx = struct {
            row: []const f32,
            fn lt(c: @This(), x: i32, y: i32) bool {
                const ox = ord(c.row[@intCast(x)]);
                const oy = ord(c.row[@intCast(y)]);
                return if (ox != oy) ox > oy else x < y;
            }
        };
        _ = self;
        const take = @min(ids.len, out.len);
        if (ids.len > out.len) std.sort.pdq(i32, ids, Ctx{ .row = row }, Ctx.lt);
        @memcpy(out[0..take], ids[0..take]);
        std.sort.pdq(i32, out[0..take], {}, std.sort.asc(i32));
        @memset(out[take..], SENTINEL);
    }

    /// The host select of `mode` on the prepared scores (local: each slice's own list).
    fn expected(self: *Select, mode: Mode) ![]i32 {
        const a = self.arena.allocator();
        const n = self.shape.n;
        const k = self.shape.k;
        const per: usize = if (mode == .local) G * k else k;
        const out = try a.alloc(i32, self.shape.rows * per);
        const ids = try a.alloc(i32, @max(n, G * k));
        for (0..self.shape.rows) |r| {
            const row = self.scores[r * n ..][0..n];
            const vb: usize = @intCast(@max(self.bounds[r], 0));
            switch (mode) {
                .single => {
                    for (ids[0..vb], 0..) |*x, i| x.* = @intCast(i);
                    self.topInto(row, ids[0..vb], out[r * k ..][0..k]);
                },
                .local => for (0..G) |g| {
                    const lo = g * vb / G;
                    const hi = (g + 1) * vb / G;
                    for (ids[0 .. hi - lo], lo..) |*x, i| x.* = @intCast(i);
                    self.topInto(row, ids[0 .. hi - lo], out[(r * G + g) * k ..][0..k]);
                },
                .merge => {
                    var m: usize = 0;
                    for (self.cands[r * G * k ..][0 .. G * k]) |c| if (c != SENTINEL) {
                        ids[m] = c;
                        m += 1;
                    };
                    self.topInto(row, ids[0..m], out[r * k ..][0..k]);
                },
            }
        }
        return out;
    }

    pub fn reference(self: *Select, want: []f64, mag: []f64) !void {
        for (want, try self.expected(self.mode)) |*w, e| w.* = @floatFromInt(e);
        @memset(mag, 0);
    }

    /// Ids, not values: every slot equals the host select's.
    pub fn check(self: *Select, _: spec.Gate) !?[]const u8 {
        const want = try self.expected(self.mode);
        const got = self.rig.outputBytes();
        for (want, 0..) |w, i| {
            const g = std.mem.readInt(i32, got[4 * i ..][0..4], .little);
            if (g != w) return std.fmt.bufPrint(&self.why, "row {d} slot {d} holds {d} where the host select gives {d}", .{ i / (self.count() / self.shape.rows), i % (self.count() / self.shape.rows), g, w }) catch "a slot differs from the host select";
        }
        return null;
    }
};

const rule = "torch.topk's order on the f32 key: NaN above every number, both infinities in place, -0.0 equal to +0.0, ties to the lower index; a NaN past a row's bound is never read";

fn none(_: Shape) usize {
    return 0;
}

pub fn register(set: *kc.Set) !void {
    try set.add(Select, .init(set.a, .single), .{
        .name = "qsa_select (top-K visible blocks, one threadgroup a row)",
        .accum = .none,
        .depth = none,
        .out = .u32,
        .nan = .{ .stated = rule },
        // rows: the 8/11-bit digit switch at 16; n: the 1024-thread walk; k: up to the budget.
        .tiles = .{ .{ .size = 16 }, .{ .size = 1024, .min = KB + 1 }, .{ .size = KB } },
        // Decode just past the budget and below the split floor, a verify step, the 512- and 256-thread prefill.
        .served = &.{ .{ .rows = 1, .n = 700, .k = KB }, .{ .rows = 1, .n = 20000, .k = KB }, .{ .rows = 8, .n = 4096, .k = KB }, .{ .rows = 64, .n = 2048, .k = KB }, .{ .rows = 512, .n = 2048, .k = KB } },
    }, null);
    // The split pair at decode widths past `QSA_SELECT_SPLIT_MIN_NB`.
    const split: []const Shape = &.{ .{ .rows = 1, .n = 24576, .k = KB }, .{ .rows = 15, .n = 65536, .k = KB } };
    const split_tiles: [3]spec.Tile = .{ .{ .size = 8 }, .{}, .{ .size = KB } };
    try set.add(Select, .init(set.a, .local), .{ .name = "qsa_select split local (16 slices' own top-K)", .accum = .none, .depth = none, .out = .u32, .nan = .{ .stated = rule }, .tiles = split_tiles, .served = split }, null);
    try set.add(Select, .init(set.a, .merge), .{ .name = "qsa_select split merge (top-K of the slices' candidates)", .accum = .none, .depth = none, .out = .u32, .nan = .{ .stated = rule }, .tiles = split_tiles, .served = split }, null);
}

// ── tests ──

/// The shipped wrapper on `k`'s scores and bounds, the split arm on or off.
fn wrapper(k: *Select, split: bool) ![]const u8 {
    const prev = select.qsa_select_split_override;
    defer select.qsa_select_split_override = prev;
    select.qsa_select_split_override = split;
    const out = (try select.qsaSelectTopBlocks(k.rig.stream(), k.args[0], k.args[1], @intCast(k.shape.k))) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(out);
    try std.testing.expectEqual(split, select.qsa_select_used_split);
    return k.rig.bytesOf(&.{out});
}

test "kernel contract sparse_block: qsa_select's declared launches, chained, are the shipped wrapper's bytes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var single = Select.init(a, .single);
    defer single.deinit();
    for ([_]Shape{ .{ .rows = 1, .n = 700, .k = KB }, .{ .rows = 64, .n = 2048, .k = KB } }) |shape| {
        try std.testing.expect(try single.prepare(shape, .random, 5));
        try single.run();
        const ours = try a.dupe(u8, single.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try wrapper(&single, false), ours);
    }
    var local = Select.init(a, .local);
    defer local.deinit();
    var merge = Select.init(a, .merge);
    defer merge.deinit();
    for ([_]Shape{ .{ .rows = 1, .n = 24576, .k = KB }, .{ .rows = 15, .n = 65536, .k = KB } }) |shape| {
        try std.testing.expect(try local.prepare(shape, .random, 9));
        try local.run();
        try std.testing.expect(try merge.prepare(shape, .random, 9));
        // The merge's prepared candidates are the local launch's own bytes.
        try std.testing.expectEqualSlices(u8, local.rig.outputBytes(), std.mem.sliceAsBytes(merge.cands));
        try merge.run();
        const ours = try a.dupe(u8, merge.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try wrapper(&local, true), ours);
    }
}
