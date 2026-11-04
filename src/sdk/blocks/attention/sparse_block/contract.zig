//! The sparse-block kernels in the kernel contract: the one-row key mask (`qsa_row_mask`: a token is visible when its
//! block is in the sorted pick list or it sits in the incomplete tail; rows = the pick list's length, n = kv, k = 1)
//! here, the score sheet, the exact select and the decode attention in `contract_*.zig`.

const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const kc = @import("../../../testing/kernel_contract/root.zig");
const qsa_row_mask = @import("row_mask.zig");
const qsaMaskFromBlocks = @import("mask.zig").qsaMaskFromBlocks;

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

/// The row-mask block size the contract runs at (Qwen3.8-Flash-Next's).
const RATIO = 4;
const SENTINEL: f64 = std.math.maxInt(i32);

const row_mask_launch: metal.Launch = .{
    .name = "mlxserve_qsa_row_mask",
    .inputs = &.{"blocks"},
    .outputs = &.{"mask"},
    .source = qsa_row_mask.SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

const RowMask = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &row_mask_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    picks: []f64 = &.{},
    args: [1]mlx.mlx_array = undefined,
    /// A planted control: one thread short, so the last key (in the tail: visible) is never written.
    short: bool = false,

    fn init(a: std.mem.Allocator) RowMask {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *RowMask) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    /// The wrapper's launch: one thread a key, the step's kv wide.
    pub fn geometry(self: *const RowMask, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.n < @as(usize, if (self.short) 2 else 1) or shape.k != 1) return null;
        const kv: c_int = @intCast(shape.n);
        const threads = if (self.short) kv - 1 else kv;
        var g = metal.Geometry{ .grid = .{ threads, 1, 1 }, .tg = .{ @min(threads, 256), 1, 1 } };
        g.out(&.{ 1, 1, 1, kv }, .bool_);
        g.int("KB", shape.rows);
        g.int("RATIO", RATIO);
        return g;
    }

    /// Sorted picks with INT_MAX sentinels after them. `.nonfinite` (integer ids have no NaN) plants the edges
    /// instead: block 0, the last whole block, an id at the tail's own block and one past every key.
    pub fn prepare(self: *RowMask, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const nb = shape.n / RATIO;
        self.picks = try a.alloc(f64, shape.rows);
        @memset(self.picks, SENTINEL);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (fill) {
            .random, .stress => if (nb > 0) {
                const pool = try a.alloc(usize, nb);
                for (pool, 0..) |*p, i| p.* = i;
                rng.shuffle(usize, pool);
                const n = @min(shape.rows, 1 + rng.uintLessThan(usize, nb));
                for (self.picks[0..n], pool[0..n]) |*p, b| p.* = @floatFromInt(b);
            },
            .nonfinite => {
                const edges = [_]f64{ 0, @floatFromInt(nb -| 1), @floatFromInt(nb), @floatFromInt(nb + 7) };
                const n = @min(shape.rows, edges.len);
                for (self.picks[0..n], edges[0..n]) |*p, e| p.* = e;
            },
        }
        std.mem.sort(f64, self.picks, {}, std.sort.asc(f64));
        self.rig.reset();
        self.args[0] = try self.rig.input(.int32, &.{ 1, 1, @intCast(shape.rows) }, self.picks);
        return true;
    }

    pub fn run(self: *RowMask) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const RowMask) usize {
        return self.shape.n;
    }

    pub fn reference(self: *const RowMask, want: []f64, mag: []f64) !void {
        const tail = self.shape.n / RATIO * RATIO;
        for (want, mag, 0..) |*w, *m, t| {
            const blk: f64 = @floatFromInt(t / RATIO);
            const picked = for (self.picks) |p| (if (p == blk) break true) else false;
            w.* = if (t >= tail or picked) 1 else 0;
            m.* = 0;
        }
    }

    /// Integer ids carry no NaN: the planted edge ids give exactly the reference's mask.
    pub fn check(self: *RowMask, _: spec.Gate) !?[]const u8 {
        const want = try self.arena.allocator().alloc(f64, 2 * self.count());
        try self.reference(want[0..self.count()], want[self.count()..]);
        const out = self.rig.outputBytes();
        for (want[0..self.count()], 0..) |w, i| if (spec.read(.bool, out, i) != w) return "a key's visibility differs from the pick list's";
        return null;
    }

    /// The op chain on the same picks: compare, select, scatter, slice, then the tail and causal compares.
    pub fn replaced(self: *RowMask) ![]const u8 {
        qsa_row_mask.override = false;
        defer qsa_row_mask.override = null;
        const out = try qsaMaskFromBlocks(self.rig.stream(), self.args[0], @intCast(self.shape.n), RATIO);
        defer _ = mlx.mlx_array_free(out);
        return self.rig.bytesOf(&.{out});
    }

    /// The shipped wrapper on the same picks.
    fn wrapper(self: *RowMask) ![]const u8 {
        qsa_row_mask.override = true;
        defer qsa_row_mask.override = null;
        const out = (try qsa_row_mask.rowMask(self.rig.stream(), self.args[0], @intCast(self.shape.n), RATIO)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(out);
        return self.rig.bytesOf(&.{out});
    }
};

fn none(_: Shape) usize {
    return 0;
}

pub fn register(set: *kc.Set) !void {
    try set.add(RowMask, .init(set.a), .{
        .name = "qsa_row_mask (one decode row's key visibility)",
        .accum = .none,
        .depth = none,
        .out = .bool,
        .replaces = "less -> where -> put_along_axis -> slice -> tail and causal compares (qsaMaskFromBlocks)",
        .nan = .{ .stated = "integer block ids: the edge ids (block 0, the last whole block, the tail's block, past kv) give the reference mask" },
        // kv a 256-thread group's edges; the pick list up to Flash Next's 512-block budget.
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        // Just past the 2048-key budget, a long decode with a three-key tail, a tiny cache.
        .served = &.{ .{ .rows = 512, .n = 2053, .k = 1 }, .{ .rows = 512, .n = 8079, .k = 1 }, .{ .rows = 16, .n = 65, .k = 1 } },
    }, null);
    var short = RowMask.init(set.a);
    short.short = true;
    try set.add(RowMask, short, .{
        .name = "control qsa_row_mask one thread short (its last key, visible, never written)",
        .accum = .none,
        .depth = none,
        .out = .bool,
        .served = &.{.{ .rows = 16, .n = 65, .k = 1 }},
    }, .repeat);
    try @import("contract_decode.zig").register(set);
    try @import("contract_score.zig").register(set);
    try @import("contract_select.zig").register(set);
}

// ── tests ──

test "kernel contract sparse_block: the declared row-mask launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    const k = set.kernels.items[0];
    const p: *RowMask = @ptrCast(@alignCast(k.ctx));
    for (k.spec.served) |shape| {
        try std.testing.expect(try p.prepare(shape, .random, 7));
        try p.run();
        const ours = try a.dupe(u8, p.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
    }
}

test {
    _ = @import("contract_decode.zig");
    _ = @import("contract_score.zig");
    _ = @import("contract_select.zig");
}
