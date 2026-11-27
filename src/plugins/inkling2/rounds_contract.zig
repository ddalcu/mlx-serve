//! Inkling 2's speculative-round kernels in the kernel contract (`contract.zig` registers them): the DFlash2 draft's
//! grouped dynamic convolution (`dflash_conv`), its selector's sharded top 16 (`dflash_top16_shard`, `_merge`), edge
//! lattice (`dflash_edges`) and tree builder (`dflash_tree`); the verify's sampling (`sample_tree_mass`, `_search`,
//! `_draw`), its tree walk (`accept_tree`) and the commit of the accepted path's conv rows (`commit_conv`).
//! Indices, tokens and copied values are exact; probabilities and masses carry their f32 and fast-math forward error
//! (`host.Val` style, in `mag`). A kernel whose choices compare computed values (a draw against its uniform, a cut
//! against a mass, a heap of path scores) is given inputs whose every such choice clears the device's error by `TOL`:
//! `prepare` picks the laws, seeds or lattice until it does, so the float64 reference's choice is the device's.
//! Each kernel has one planted control (a wrong tie, an off-by-one, a dropped tap, shard or rank) that gate 3 must
//! catch. No offline M1/M2 compile has been run for these launches yet (`kernel contract offline`): each records its
//! own launch's width as `m1`/`m2`, the least its pipeline must take.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const source = @import("../../mlx/kernels.zig").source;
const project = @import("../../sdk/bases/blocks/head.zig").project;
const kernels = @import("kernels.zig");
const dflash2 = @import("dflash2.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;
const U = host.U;
const inf = std.math.inf(f64);
const nan = std.math.nan(f64);

// ── Inkling 2 Mini's sizes ──

/// The head's real rows (the logits' vocab) and its padded rows (a DFlash2 draft's logits stride).
const VOCAB = 200058;
const HEAD_ROWS = 201024;
/// The selector's candidates a row and the sharded top-k's shards (`dflash2.topCandidates`, `sampleAndAcceptTree`).
const KTOP = 16;
const SHARDS = 16;
/// The DFlash2 selector's rank, the stream width, the draft's block, its conv taps and group.
const RANK = 256;
const HIDDEN = 2304;
const BLOCK = 8;
const CONV_KT = 2;
const CONV_GS = 16;
const SOFTCAP: f64 = 30;
/// The trunk's short conv taps and the open window's width (keys, values, the output projection and the stream).
const SCONV_KT = 4;
const SW = 2 * 1024 + 2 * HIDDEN;
/// A tree lane's copy chain, the stop ids a round carries, the children a node draws (`plugin.stepGroup`'s 4).
const MAX_COPY = dflash2.Drafter.max_copy;
const MAX_STOPS = 16;
const MAX_CH = 4;
/// The codebook rows an edge test binds (the kernel reads only the rows its tokens name).
const TABLE = 1024;
/// The spikes a lane's logits carry above their bulk.
const SPIKES = 24;
/// The least relative gap a computed choice must clear: far past the f32 and fast-math error of every compared value.
const TOL = 0x1p-12;
/// f32 unit roundoff of a precise tanh / exp (a few ulps).
const PRECISE = 4 * 0x1p-24;

/// A lane's tree rows below its root: the block's rows after the anchor, at most the width less one.
fn rowsOf(w: usize) usize {
    return @min(BLOCK - 1, @max(1, w - 1));
}

fn vocabOf(shape: Shape) usize {
    return @min(shape.k, VOCAB);
}

fn cint(x: usize) c_int {
    return @intCast(x);
}

fn f32of(v: f64) f64 {
    return spec.roundTo(.f32, v);
}

fn bfOf(v: f64) f64 {
    return spec.roundTo(.bf16, v);
}

/// The f32 whose bits are `id` (`as_type<float>(int)`), as float64: exact, a subnormal for every id here.
fn idBits(id: u32) f64 {
    return @as(f32, @bitCast(id));
}

fn bitsAt(bytes: []const u8, i: usize) u32 {
    return std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
}

fn tokU(x: f64) u32 {
    return @intFromFloat(x);
}

/// Metal's fast `log(x)` absolute error: 2^-21 on [0.5, 2], 3 ulps past it.
fn fastLogErr(x: f64) f64 {
    return 0x1p-21 + 3 * 0x1p-23 * @abs(@log(x));
}

fn none(_: Shape) usize {
    return 0;
}

/// Sums in an accumulator (gate 4) or float64.
const Sim = ?struct { acc: spec.Accum, order: spec.Order };

fn inAcc(acc: spec.Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => spec.roundTo(.f32, v),
        .f16 => spec.roundTo(.f16, v),
        .bf16 => spec.roundTo(.bf16, v),
        .none => v,
    };
}

fn total(sim: Sim, t: []const f64) f64 {
    return if (sim) |x| spec.sumIn(x.acc, x.order, t) else host.sum(t);
}

/// A running sum's next value (the kernels' cumulative walks run in one chain).
fn step(sim: Sim, s: f64, t: f64) f64 {
    return if (sim) |x| inAcc(x.acc, s + inAcc(x.acc, t)) else s + t;
}

/// `a` and `b` differ by more than the device's error on either: a choice between them is the device's too.
fn clear(a: f64, b: f64) bool {
    return @abs(a - b) > TOL * @max(@abs(a), @abs(b)) + 0x1p-40;
}

/// `got` is `want` within `e` and its f32 rounding (classes exact).
fn close(got: f64, want: f64, e: f64) bool {
    if (std.math.isNan(want)) return std.math.isNan(got);
    if (got == want) return true;
    if (!std.math.isFinite(want) or !std.math.isFinite(got)) return false;
    // A GPU that flushes f32 subnormals (M1-M4) reads a value under the normal range as zero.
    if (got == 0 and @abs(want) < 4 * std.math.floatMin(f32)) return true;
    return @abs(got - want) <= e + U * (@abs(want) + e) + 0x1p-149;
}

fn mix64(v0: u64) u64 {
    var v = v0;
    v = (v ^ (v >> 30)) *% 0xbf58476d1ce4e5b9;
    v = (v ^ (v >> 27)) *% 0x94d049bb133111eb;
    return v ^ (v >> 31);
}

/// sample_tree_draw's uniform for `row` and `tag`: exact (an integer hash, 23 bits and a half).
fn uniform01(seed: u64, row: usize, tag: u32) f64 {
    const t: u32 = @as(u32, @intCast(row)) *% 8 +% tag +% 1;
    const h = mix64(seed ^ (0x9e3779b97f4a7c15 *% @as(u64, t)));
    return (@as(f64, @floatFromInt(h >> 41)) + 0.5) * 0x1p-23;
}

fn fail(note: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(note, fmt, args) catch "an output breaks the rule";
}

// ── launches (names, inputs and sources the shipped wrappers' `kernels.Kernel`s carry) ──

const commit_launch = metal.Launch{ .name = "msv_inkling2_commit_conv", .inputs = &.{ "WIN", "IDX" }, .outputs = &.{"NST"}, .source = source("plugins/inkling2/kernels/commit_conv.metal"), .tg_bytes = 0, .m1 = 64, .m2 = 64 };
const accept_launch = metal.Launch{ .name = "msv_inkling2_accept_tree", .inputs = &.{ "TOKENS", "PARENT", "SELECTED", "ACCEPT_CAP", "STOPS", "N_STOPS" }, .outputs = &.{"RESULT"}, .source = source("plugins/inkling2/kernels/accept_tree.metal"), .tg_bytes = 0, .m1 = 1, .m2 = 1 };
// tg_mx, tg_id, tg_sum, tg_adm: eight each.
const mass_launch = metal.Launch{ .name = "msv_inkling2_sample_mass", .inputs = &.{ "LOGITS", "SUPPRESS", "LPARAMS" }, .outputs = &.{"PMASS"}, .source = source("plugins/inkling2/kernels/sample_tree_mass.metal"), .tg_bytes = 4 * 4 * 8, .m1 = 256, .m2 = 256 };
// The merge's four, ps[8], the 512 bins' counts and masses, the cut's five, the boundary bin's 512 values and ids,
// its count, sg_above[8].
const search_launch = metal.Launch{ .name = "msv_inkling2_sample_search", .inputs = &.{ "LOGITS", "SUPPRESS", "LPARAMS", "PMASS" }, .outputs = &.{"VROW"}, .source = source("plugins/inkling2/kernels/sample_tree_search.metal"), .tg_bytes = 4 * (4 + 8 + 2 * 512 + 5 + 2 * 512 + 1 + 8), .m1 = 256, .m2 = 256 };
// sh_done, sh_target_mass, sh_has_cands, sh_cands[K], sg_sums[8], tg_last[256], chosen.
const draw_launch = metal.Launch{ .name = "msv_inkling2_sample_draw", .inputs = &.{ "LOGITS", "SUPPRESS", "LPARAMS", "VROW", "SEEDS", "TMETA", "CANDS", "NEXT_Q", "DRAWS" }, .outputs = &.{"SELECTED"}, .source = source("plugins/inkling2/kernels/sample_tree_draw.metal"), .tg_bytes = 4 * (3 + KTOP + 8 + 256 + 1), .m1 = 256, .m2 = 256 };
const conv_launch = metal.Launch{ .name = "msv_inkling2_dflash_conv", .inputs = &.{ "X", "DYN", "BASE", "RES" }, .outputs = &.{"OUT"}, .source = source("plugins/inkling2/kernels/dflash_conv.metal"), .tg_bytes = 0, .m1 = 64, .m2 = 64 };
// sg_v, sg_i, sg_t: eight each, and win_tid.
const shard_launch = metal.Launch{ .name = "msv_inkling2_dflash_top16_shard", .inputs = &.{"LOGITS"}, .outputs = &.{ "PIDS", "PVALS" }, .source = source("plugins/inkling2/kernels/dflash_top16_shard.metal"), .tg_bytes = 4 * (3 * 8 + 1), .m1 = 256, .m2 = 256 };
const merge_launch = metal.Launch{ .name = "msv_inkling2_dflash_top16_merge", .inputs = &.{ "PIDS", "PVALS" }, .outputs = &.{ "IDS", "VALS" }, .source = source("plugins/inkling2/kernels/dflash_top16_merge.metal"), .tg_bytes = 0, .m1 = 32, .m2 = 32 };
const edges_launch = metal.Launch{ .name = "msv_inkling2_dflash_edges", .inputs = &.{ "T1", "CANDS", "UNARY", "PROJ", "PRED", "SUCC", "CAP" }, .outputs = &.{"PROBS"}, .source = source("plugins/inkling2/kernels/dflash_edges.metal"), .tg_bytes = 0, .m1 = 32, .m2 = 32 };
const tree_launch = metal.Launch{ .name = "msv_inkling2_dflash_tree", .inputs = &.{ "T1", "CANDS", "PROBS", "CALIB", "TEMPS", "SEEDS", "COPY_IDS", "COPY_LEN", "COPY_RATE", "STOPS", "PARAMS" }, .outputs = &.{ "TOKENS", "PARENT", "TMETA", "NEXT_Q", "DRAWS", "SCORES" }, .source = source("plugins/inkling2/kernels/dflash_tree.metal"), .tg_bytes = 0, .m1 = 1, .m2 = 1 };

// Each dispatch as its wrapper builds it (`kernels.commitPath`, `kernels.sampleAndAcceptTree`, `dflash2.Drafter`'s
// `prepare`, `finishAdd`, `topCandidates`, `buildTreeFromLogits`), template args in the wrapper's order.

fn commitGeometry(sw: usize, slot: usize) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ cint((SCONV_KT - 1) * sw), 1, 1 }, .tg = .{ 64, 1, 1 } };
    g.out(&.{ 1, SCONV_KT - 1, cint(sw) }, .float32);
    g.int("SLOT", slot);
    return g;
}

fn acceptGeometry(b: usize, w: usize) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ cint(b), 1, 1 }, .tg = .{ 1, 1, 1 } };
    g.out(&.{ cint(b), cint(2 + 2 * w) }, .int32);
    g.int("W", w);
    return g;
}

fn massGeometry(rows: usize, w: usize, v: usize, stride: usize, masked: bool) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ SHARDS * 256, cint(rows), 1 }, .tg = .{ 256, 1, 1 } };
    g.out(&.{ cint(rows), SHARDS, 4 }, .float32);
    g.dtype("T", .bfloat16);
    g.int("V", v);
    g.int("STRIDE", stride);
    g.int("SHARDS", SHARDS);
    g.int("W", w);
    g.int("MASK", @intFromBool(masked));
    return g;
}

fn searchGeometry(rows: usize, w: usize, v: usize, stride: usize, masked: bool) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ 256, cint(rows), 1 }, .tg = .{ 256, 1, 1 } };
    g.out(&.{ cint(rows), 4 }, .float32);
    g.dtype("T", .bfloat16);
    g.int("V", v);
    g.int("STRIDE", stride);
    g.int("SHARDS", SHARDS);
    g.int("W", w);
    g.int("MASK", @intFromBool(masked));
    return g;
}

fn drawGeometry(rows: usize, w: usize, v: usize, stride: usize, masked: bool, r: usize) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ 256, cint(rows), 1 }, .tg = .{ 256, 1, 1 } };
    g.out(&.{cint(rows)}, .int32);
    g.dtype("T", .bfloat16);
    g.int("V", v);
    g.int("STRIDE", stride);
    g.int("W", w);
    g.int("MASK", @intFromBool(masked));
    g.int("TREE", 1);
    g.int("R", r);
    g.int("K", KTOP);
    return g;
}

fn convGeometry(b: usize, l: usize, stage: usize, kt: usize) metal.Geometry {
    const total_n = b * l * HIDDEN;
    var g = metal.Geometry{ .grid = .{ cint((total_n + 63) / 64 * 64), 1, 1 }, .tg = .{ 64, 1, 1 } };
    g.out(&.{ cint(b), cint(l), HIDDEN }, .bfloat16);
    g.dtype("T", .bfloat16);
    g.dtype("BT", .bfloat16);
    g.int("H", HIDDEN);
    g.int("GS", CONV_GS);
    g.int("KT", kt);
    g.int("STAGE", stage);
    return g;
}

fn shardGeometry(rt: usize, stride: usize, v: usize) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ SHARDS * 256, cint(rt), 1 }, .tg = .{ 256, 1, 1 } };
    g.out(&.{ cint(rt), SHARDS, KTOP }, .int32);
    g.out(&.{ cint(rt), SHARDS, KTOP }, .float32);
    g.dtype("T", .bfloat16);
    g.int("V", v);
    g.int("STRIDE", stride);
    g.int("SHARDS", SHARDS);
    g.int("K", KTOP);
    return g;
}

fn mergeGeometry(rt: usize, shards: usize) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ 32, cint(rt), 1 }, .tg = .{ 32, 1, 1 } };
    g.out(&.{ cint(rt), KTOP }, .int32);
    g.out(&.{ cint(rt), KTOP }, .float32);
    g.int("SHARDS", shards);
    g.int("K", KTOP);
    return g;
}

fn edgesGeometry(b: usize, r: usize, rank: usize) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ KTOP * 32, cint(r), cint(b) }, .tg = .{ 32, 1, 1 } };
    g.out(&.{ cint(b), cint(r), KTOP, KTOP }, .float32);
    g.dtype("PT", .bfloat16);
    g.dtype("CT", .bfloat16);
    g.int("R", r);
    g.int("K", KTOP);
    g.int("RANK", rank);
    return g;
}

fn treeGeometry(b: usize, w: usize, r: usize) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ cint(b), 1, 1 }, .tg = .{ 1, 1, 1 } };
    g.out(&.{ cint(b), cint(w) }, .int32);
    g.out(&.{ cint(b), cint(w) }, .int32);
    g.out(&.{ cint(b * w), 2 }, .int32);
    g.out(&.{ cint(b), cint(w), KTOP }, .float32);
    g.out(&.{ cint(b), cint(w), 4 }, .uint8);
    g.out(&.{ cint(b), cint(w) }, .float32);
    g.int("W", w);
    g.int("R", r);
    g.int("K", KTOP);
    g.int("MAX_COPY", MAX_COPY);
    return g;
}

// ── commit_conv ──

/// The accepted path's conv rows of one slot of a verify: rows = slots, n = rows a slot (the tree's width), k = the
/// window's width. WIN [slots, KT - 1 + n, k] f32, IDX [KT - 1] the path's rows as `commitPath` builds them; the last
/// slot's rows are kept. The planted control reads the slot before it.
const CommitConv = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &commit_launch,
    arena: std.heap.ArenaAllocator,
    planted: bool = false,
    shape: Shape = undefined,
    win: []f64 = &.{},
    idx: [SCONV_KT - 1]f64 = undefined,
    args: [2]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator, planted: bool) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a), .planted = planted };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0 or shape.n > 16 or shape.k == 0) return null;
        if (self.planted and shape.rows < 2) return null;
        return commitGeometry(shape.k, if (self.planted) shape.rows - 2 else shape.rows - 1);
    }

    fn depth(shape: Shape) usize {
        return SCONV_KT - 1 + shape.n;
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const d = depth(shape);
        self.win = try a.alloc(f64, shape.rows * d * shape.k);
        for (self.win) |*v| v.* = f32of((rng.float(f64) * 2 - 1) * std.math.pow(f64, 2, @floatFromInt(rng.intRangeAtMost(i32, -8, 8))));
        // A path through the tree: the root, then rows below it in order.
        var path: [16]usize = undefined;
        var len: usize = 1;
        path[0] = 0;
        for (1..shape.n) |r| if (rng.boolean()) {
            path[len] = r;
            len += 1;
        };
        for (&self.idx, 0..) |*x, j| {
            const pos = len + j;
            x.* = @floatFromInt(if (pos < SCONV_KT - 1) pos else (SCONV_KT - 1) + path[pos - (SCONV_KT - 1)]);
        }
        if (fill == .nonfinite) {
            // NaN and both infinities in rows the path keeps, and a NaN in a row it does not.
            const s = shape.rows - 1;
            const put = [_]f64{ nan, inf, -inf };
            const cols = [_]usize{ shape.k / 3, shape.k / 2, shape.k - 1 };
            for (self.idx, put, cols) |r, v, c| self.win[(s * d + tokU(r)) * shape.k + c] = v;
            if (shape.rows > 1) self.win[shape.k / 4] = nan;
        }
        self.rig.reset();
        self.args = .{
            try self.rig.input(.float32, &.{ cint(shape.rows), cint(d), cint(shape.k) }, self.win),
            try self.rig.input(.int32, &.{SCONV_KT - 1}, &self.idx),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return (SCONV_KT - 1) * self.shape.k;
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        const k = self.shape.k;
        const s = self.shape.rows - 1;
        for (want, mag, 0..) |*w, *m, e| {
            w.* = self.win[(s * depth(self.shape) + tokU(self.idx[e / k])) * k + e % k];
            m.* = @abs(w.*);
        }
    }

    /// `commitPath`'s CPU arm on the same window: the slot, then its rows along the path.
    pub fn replaced(self: *Self) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const s = cint(self.shape.rows - 1);
        const window = if (self.shape.rows == 1) self.args[0] else try g.sliceAxis(self.args[0], 0, s, s + 1);
        return self.rig.bytesOf(&.{try g.take(window, self.args[1], 1)});
    }

    /// The shipped kernel (`kernels.commit_conv_kernel`) launched as `commitPath` launches it.
    fn wrapper(self: *Self) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const k = self.shape.k;
        const st = try kernels.launch(&g, &kernels.commit_conv_kernel, &.{ self.args[0], self.args[1] }, 1, .{.{ .shape = &.{ 1, SCONV_KT - 1, cint(k) }, .dtype = .float32 }}, cint((SCONV_KT - 1) * k), 64, &.{.{ "SLOT", .{ .int = cint(self.shape.rows - 1) } }});
        return self.rig.bytesOf(&.{st[0]});
    }
};

// ── accept_tree ──

/// A verify's tree walk: rows = lanes, n = rows a lane (the tree's width). Small token alphabets, so siblings share
/// tokens and selections match children often; lane 0's root has two children carrying its selection (the walk takes
/// the lower row; the planted control the higher).
const AcceptTree = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &accept_launch,
    arena: std.heap.ArenaAllocator,
    planted: bool = false,
    shape: Shape = undefined,
    tokens: []f64 = &.{},
    parent: []f64 = &.{},
    selected: []f64 = &.{},
    caps: []f64 = &.{},
    stops: [MAX_STOPS]f64 = @splat(-1),
    n_stops: usize = 0,
    note: [200]u8 = undefined,
    args: [6]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator, planted: bool) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a), .planted = planted };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n < 3 or shape.n > 16) return null;
        return acceptGeometry(shape.rows, shape.n);
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const b = shape.rows;
        const w = shape.n;
        self.tokens = try a.alloc(f64, b * w);
        self.parent = try a.alloc(f64, b * w);
        self.selected = try a.alloc(f64, b * w);
        self.caps = try a.alloc(f64, b);
        for (0..b) |l| {
            const base = l * w;
            self.tokens[base] = @floatFromInt(1000 + rng.uintLessThan(u32, 50));
            self.parent[base] = -1;
            for (1..w) |r| {
                self.parent[base + r] = @floatFromInt(rng.uintLessThan(usize, r));
                self.tokens[base + r] = @floatFromInt(100 + rng.uintLessThan(u32, 6));
            }
            for (0..w) |r| self.selected[base + r] = @floatFromInt(100 + rng.uintLessThan(u32, 7));
            self.caps[l] = @floatFromInt(if (rng.uintLessThan(u32, 4) == 0) 15 else rng.uintLessThan(usize, w));
        }
        // 105 is an alphabet token; 999 no row carries.
        self.stops = @splat(-1);
        self.stops[0] = 105;
        self.stops[1] = 999;
        self.n_stops = 2;
        self.selected[0] = 102;
        for (1..3) |r| {
            self.tokens[r] = 102;
            self.parent[r] = 0;
        }
        self.caps[0] = @max(self.caps[0], 2);
        // Gate 1: integers carry no NaN; it plants stop ids in the selections, which end a walk before its child.
        if (fill == .nonfinite) for (0..b) |l| {
            if (l % 2 == 1) self.selected[l * w] = 105;
            self.selected[l * w + w - 1] = 105;
        };
        self.rig.reset();
        self.args = .{
            try self.rig.input(.int32, &.{ cint(b), cint(w) }, self.tokens),
            try self.rig.input(.int32, &.{ cint(b), cint(w) }, self.parent),
            try self.rig.input(.int32, &.{cint(b * w)}, self.selected),
            try self.rig.input(.uint32, &.{cint(b)}, self.caps),
            try self.rig.input(.int32, &.{MAX_STOPS}, &self.stops),
            try self.rig.input(.uint32, &.{1}, &.{@as(f64, @floatFromInt(self.n_stops))}),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return self.shape.rows * (2 + 2 * self.shape.n);
    }

    fn isStop(self: *const Self, tok: f64) bool {
        for (self.stops[0..self.n_stops]) |s| if (s == tok) return true;
        return false;
    }

    /// Lane `l`'s walk: { accepted, next token, path rows [W], emitted tokens [W] }.
    fn walk(self: *const Self, l: usize, out: []f64) void {
        const w = self.shape.n;
        const base = l * w;
        const cap = @min(tokU(self.caps[l]), w - 1);
        @memset(out, 0);
        const path = out[2..][0..w];
        const emitted = out[2 + w ..][0..w];
        emitted[0] = self.tokens[base];
        var cur: usize = 0;
        var accepted: usize = 0;
        var next = self.selected[base];
        while (true) {
            const tok = self.selected[base + cur];
            next = tok;
            if (accepted >= cap or self.isStop(tok)) break;
            var matched: usize = 0;
            for (cur + 1..w) |child| {
                if (self.parent[base + child] == @as(f64, @floatFromInt(cur)) and self.tokens[base + child] == tok) {
                    matched = child;
                    if (!self.planted) break;
                }
            }
            if (matched == 0) break;
            accepted += 1;
            path[accepted] = @floatFromInt(matched);
            emitted[accepted] = tok;
            cur = matched;
        }
        out[0] = @floatFromInt(accepted);
        out[1] = next;
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        const stride = 2 + 2 * self.shape.n;
        for (0..self.shape.rows) |l| self.walk(l, want[l * stride ..][0..stride]);
        @memset(mag, 0);
    }

    /// The walk is integers: every output equals the reference's.
    pub fn check(self: *Self, _: spec.Gate) !?[]const u8 {
        const want = try self.arena.allocator().alloc(f64, 2 * self.count());
        try self.reference(want[0..self.count()], want[self.count()..]);
        const bytes = self.rig.outputBytes();
        const stride = 2 + 2 * self.shape.n;
        for (want[0..self.count()], 0..) |w, i| {
            const got: f64 = @floatFromInt(@as(i32, @bitCast(bitsAt(bytes, i))));
            if (got != w) return fail(&self.note, "lane {d} slot {d}: {d} where the walk gives {d}", .{ i / stride, i % stride, got, w });
        }
        return null;
    }
};

// ── the verify's sampling: shared rows and laws ──

/// A lane's law as `LPARAMS` and `SEEDS` carry it (f32 values).
const Law = struct { temp: f64 = 0, top_k: f64 = 0, top_p: f64 = 1, min_p: f64 = 0, seed: u64 = 0 };

/// A verify's logits: rows = lanes, n = rows a lane, k = the logits' stride (its first `vocabOf` columns real, the
/// rest padding the kernels must not read). A lane's rows share 24 spike columns (14 to 20) above a bulk of
/// N(0, 1.5^2), as a language model's head peaks; bf16 values.
const Rows = struct {
    b: usize,
    w: usize,
    v: usize,
    stride: usize,
    logits: []f64,
    /// [v]: 1 suppresses the column; empty: no mask (`MASK` 0).
    mask: []u8,
    laws: []Law,
    spikes: []usize,

    fn n(self: *const Rows) usize {
        return self.b * self.w;
    }
};

fn makeRows(a: std.mem.Allocator, rng: std.Random, shape: Shape, masked: bool, fill: spec.Fill) !Rows {
    const v = vocabOf(shape);
    var rows: Rows = .{
        .b = shape.rows,
        .w = shape.n,
        .v = v,
        .stride = shape.k,
        .logits = try a.alloc(f64, shape.rows * shape.n * shape.k),
        .mask = if (masked) try a.alloc(u8, v) else &.{},
        .laws = try a.alloc(Law, shape.rows),
        .spikes = try a.alloc(usize, shape.rows * SPIKES),
    };
    for (rows.mask, 0..) |*m, c| m.* = @intFromBool(c % 97 == 13);
    for (rows.spikes) |*s| s.* = rng.uintLessThan(usize, v);
    for (0..rows.n()) |r| {
        const row = rows.logits[r * rows.stride ..][0..rows.stride];
        for (row[0..v]) |*x| x.* = bfOf(1.5 * rng.floatNorm(f64));
        for (row[v..]) |*x| x.* = 900;
        for (rows.spikes[(r / rows.w) * SPIKES ..][0..SPIKES]) |c| row[c] = bfOf(14 + 6 * rng.float(f64));
    }
    // A suppressed spike.
    if (masked) rows.mask[rows.spikes[0]] = 1;
    // Row 0's shard 0 holds two equal maxima in two threads' columns.
    for ([_]usize{ 3, 700 }) |c| {
        rows.logits[c] = 21;
        if (masked) rows.mask[c] = 0;
    }
    // Gate 1: NaN in row 0, +Inf in row 1, -Inf in row 2, both in row 3 (`host.plant`; past `v` the padding).
    if (fill == .nonfinite) host.plant(rows.logits, rows.n(), rows.stride);
    for (rows.laws) |*law| law.* = .{ .seed = rng.int(u64) };
    return rows;
}

/// Row `r`'s scaled logits v = f32(raw * f32(1 / temp)) where admitted, NaN where suppressed, NaN or -Inf.
fn scaled(rows: *const Rows, r: usize, vs: []f64) void {
    const law = rows.laws[r / rows.w];
    const inv = if (law.temp > 0) f32of(1 / law.temp) else 1;
    const src = rows.logits[r * rows.stride ..][0..rows.v];
    for (vs[0..rows.v], src, 0..) |*o, raw, c| {
        const off = (rows.mask.len > 0 and rows.mask[c] != 0) or std.math.isNan(raw) or raw == -inf;
        o.* = if (off) nan else f32of(raw * inv);
    }
}

const Tok = struct { v: f64, c: u32 };

fn rankDesc(_: void, x: Tok, y: Tok) bool {
    return x.v > y.v or (x.v == y.v and x.c < y.c);
}

/// The admitted columns at or above `cut` and `floor`, ranked (value descending, id ascending).
fn collect(vs: []const f64, cut: f64, floor: f64, out: []Tok) []Tok {
    var n: usize = 0;
    for (vs, 0..) |x, c| if (!std.math.isNan(x) and x >= cut and x >= floor) {
        out[n] = .{ .v = x, .c = @intCast(c) };
        n += 1;
    };
    std.mem.sort(Tok, out[0..n], {}, rankDesc);
    return out[0..n];
}

/// A row's admitted max (lowest id at it), count and float64 mass exp(v - max).
const Stats = struct { gmx: f64, adm: usize, rsum: f64 };

fn statsOf(vs: []const f64) Stats {
    var st: Stats = .{ .gmx = -inf, .adm = 0, .rsum = 0 };
    for (vs) |x| if (!std.math.isNan(x)) {
        st.adm += 1;
        st.gmx = @max(st.gmx, x);
    };
    if (std.math.isFinite(st.gmx)) for (vs) |x| if (!std.math.isNan(x)) {
        st.rsum += @exp(x - st.gmx);
    };
    return st;
}

/// A min-p whose cut, on every row of lane `l`, lies clear of every admitted value (0: none found).
fn pickMinP(rows: *Rows, l: usize, rng: std.Random, vs: []f64, toks: []Tok) f64 {
    scaled(rows, l * rows.w, vs);
    const st0 = statsOf(vs[0..rows.v]);
    if (!std.math.isFinite(st0.gmx)) return 0;
    const top = collect(vs[0..rows.v], -inf, st0.gmx - 12, toks);
    if (top.len < 3) return 0;
    var tries: usize = 0;
    while (tries < 16) : (tries += 1) {
        const j = 1 + rng.uintLessThan(usize, @min(8, top.len - 1));
        if (top[j - 1].v - top[j].v < 0.05) continue;
        const mp = f32of(@exp((top[j - 1].v + top[j].v) / 2 - st0.gmx));
        if (!(mp > 0 and mp < 1)) continue;
        var ok = true;
        for (0..rows.w) |i| {
            scaled(rows, l * rows.w + i, vs);
            const st = statsOf(vs[0..rows.v]);
            const cut = st.gmx + @log(mp);
            for (vs[0..rows.v]) |x| if (!std.math.isNan(x) and !clear(x, cut)) {
                ok = false;
            };
        }
        if (ok) return mp;
    }
    return 0;
}

/// Every row of lane `l` under `law` stops its ranked walk within the window, its every mass comparison clear.
fn pValid(rows: *Rows, l: usize, law: Law, vs: []f64, toks: []Tok) bool {
    for (0..rows.w) |i| {
        scaled(rows, l * rows.w + i, vs);
        const st = statsOf(vs[0..rows.v]);
        const cut = if (law.min_p > 0) st.gmx + @log(law.min_p) else -inf;
        const floor = st.gmx - 12;
        const ranked = collect(vs[0..rows.v], cut, floor, toks);
        const lim = law.top_p * st.rsum;
        const top_k = tokU(law.top_k);
        const k_lim = if (top_k > 0 and top_k < st.adm) top_k else st.adm;
        var cum: f64 = 0;
        const stopped = for (ranked, 1..) |t, cnt| {
            cum += @exp(t.v - st.gmx);
            if (!clear(cum, lim)) return false;
            if (cnt >= k_lim or cum >= lim) break t.v >= floor + 0.1;
        } else false;
        if (!stopped) return false;
    }
    return true;
}

/// A top-p whose crossing, on every row of lane `l`, falls mid-token (1: none found).
fn pickTopP(rows: *Rows, l: usize, rng: std.Random, vs: []f64, toks: []Tok) f64 {
    const law = rows.laws[l];
    scaled(rows, l * rows.w, vs);
    const st0 = statsOf(vs[0..rows.v]);
    if (!std.math.isFinite(st0.gmx)) return 1;
    const cut = if (law.min_p > 0) st0.gmx + @log(law.min_p) else -inf;
    const top = collect(vs[0..rows.v], cut, st0.gmx - 12, toks);
    if (top.len < 3) return 1;
    var cum: [16]f64 = undefined;
    var s: f64 = 0;
    for (top[0..@min(16, top.len)], 0..) |t, i| {
        s += @exp(t.v - st0.gmx);
        cum[i] = s;
    }
    var tries: usize = 0;
    while (tries < 16) : (tries += 1) {
        const at = 1 + rng.uintLessThan(usize, @min(15, top.len) - 1);
        const p = f32of((cum[at - 1] + cum[at]) / 2 / st0.rsum);
        if (!(p > 0 and p < 1)) continue;
        var with = law;
        with.top_p = p;
        if (pValid(rows, l, with, vs, toks)) return p;
    }
    return 1;
}

/// Each lane's law: greedy; sampled with no filter; top-k; top-p; min-p; all three; top-k 1 (lane kinds rotated by
/// the seed). Temperatures are powers of two, as any f32 reciprocal is exact on both sides. Gate 1's lane 0 samples
/// with no filter (its rows hold the planted values).
fn chooseLaws(rows: *Rows, rng: std.Random, fill: spec.Fill, vs: []f64, toks: []Tok) void {
    const off = rng.uintLessThan(usize, 7);
    for (0..rows.b) |l| {
        const law = &rows.laws[l];
        const kind = if (fill == .nonfinite and l == 0) 1 else (l + off) % 7;
        switch (kind) {
            0 => {},
            1 => law.temp = 1,
            2 => {
                law.temp = 0.5;
                law.top_k = @floatFromInt(2 + rng.uintLessThan(u32, 40));
            },
            3 => {
                law.temp = 1;
                law.top_p = pickTopP(rows, l, rng, vs, toks);
            },
            4 => {
                law.temp = 1;
                law.min_p = pickMinP(rows, l, rng, vs, toks);
            },
            5 => {
                law.temp = 0.5;
                law.top_k = 40;
                law.min_p = pickMinP(rows, l, rng, vs, toks);
                law.top_p = pickTopP(rows, l, rng, vs, toks);
            },
            else => {
                law.temp = 1;
                law.top_k = 1;
            },
        }
    }
}

/// One shard's PMASS entry: its admitted max, sum of exp(v - max) (sampled lanes), count and argmax id.
const Part = struct { mx: f64, sum: f64, e: f64 = 0, adm: usize, id: u32 };

/// Shard `s` of a row as sample_tree_mass computes it; `planted` breaks a tie toward the higher id.
fn massOf(v: usize, vs: []const f64, s: usize, sampled: bool, sim: Sim, planted: bool, terms: []f64) Part {
    const chunk = (v + SHARDS - 1) / SHARDS;
    const lo = @min(v, s * chunk);
    const hi = @min(v, lo + chunk);
    var p: Part = .{ .mx = -inf, .sum = 0, .adm = 0, .id = @intCast(v) };
    for (lo..hi) |c| {
        const x = vs[c];
        if (std.math.isNan(x)) continue;
        p.adm += 1;
        if (x > p.mx or (planted and x == p.mx)) {
            p.mx = x;
            p.id = @intCast(c);
        }
    }
    if (!sampled or p.mx == -inf) return p;
    if (p.mx == inf) {
        // exp(+Inf - +Inf): the sum is not a number.
        p.sum = nan;
        return p;
    }
    // Each term passes an exp at its insertion and at most a thread's columns of rescales, then the SIMD group's and
    // the threadgroup's: their arguments add to v - max, so their fast-exp errors add to (3 a step + 2 |v - max|) ulps.
    const per = (chunk + 255) / 256;
    var n: usize = 0;
    var e: f64 = 0;
    for (lo..hi) |c| {
        const x = vs[c];
        if (std.math.isNan(x)) continue;
        const t = @exp(x - p.mx);
        terms[n] = t;
        n += 1;
        e += t * (3 * @as(f64, @floatFromInt(per + 3)) + 2 * @abs(x - p.mx)) * 0x1p-23;
    }
    p.sum = total(sim, terms[0..n]);
    // Any order of the shard's adds (gate 4 sums them in each), the kernel's included.
    p.e = e + spec.gamma(@floatFromInt(n + 13), U) * p.sum;
    return p;
}

/// A row's VROW: { max, cut logit, cut id, kept mass } and the errors of the cut and the mass.
const Vrow = struct { rmax: f64, cut: f64, cut_e: f64 = 0, id: u32, km: f64, km_e: f64 = 0 };

fn binOf(rmax: f64, v: f64) usize {
    const diff = f32of(rmax - v);
    if (diff <= 0) return 0;
    if (diff >= 20) return 511;
    return @min(510, 1 + @as(usize, @intFromFloat(@floor(f32of(diff * 25.5)))));
}

/// sample_tree_search on a row: the shards merged, then the exact top-k / top-p / min-p cut in rank order (value
/// descending, id ascending), as `ranksDescending`; past 512 tokens in the cut's histogram bin, the kernel's stated
/// fallback (keep down to the bin's low edge). `planted` keeps one token past the top-k.
fn vrowOf(law: Law, v: usize, vs: []const f64, parts: []const Part, sim: Sim, planted: bool, toks: []Tok, terms: []f64) Vrow {
    const sampled = law.temp > 0;
    var gmx = -inf;
    var gid: u32 = 0;
    var gadm: usize = 0;
    for (parts) |p| {
        gadm += p.adm;
        if (p.mx > gmx or (p.mx == gmx and p.id < gid)) {
            gmx = p.mx;
            gid = p.id;
        }
    }
    var rsum: f64 = 0;
    var rsum_e: f64 = 0;
    if (sampled and gmx > -inf) {
        var n: usize = 0;
        var m: f64 = 0;
        for (parts) |p| if (p.mx > -inf) {
            const t = p.sum * @exp(p.mx - gmx);
            terms[n] = t;
            n += 1;
            rsum_e += @abs(t) * (host.fastExp(p.mx - gmx) + U);
            m += @abs(t);
        };
        rsum = total(sim, terms[0..n]);
        rsum_e += spec.gamma(SHARDS, U) * m;
    }
    const top_k = tokU(law.top_k);
    if (!sampled or top_k == 1 or gadm <= 1 or !(rsum > 0)) return .{ .rmax = gmx, .cut = gmx, .id = gid, .km = 1 };
    const cut = if (law.min_p > 0) gmx + @log(law.min_p) else -inf;
    const cut_e = if (law.min_p > 0) fastLogErr(law.min_p) + U * @abs(cut) else 0;
    const has_k = top_k > 0 and top_k < gadm;
    const has_p = law.top_p > 0 and law.top_p < 1;
    const vid: u32 = @intCast(v);
    if (!has_k and !has_p) {
        if (!(law.min_p > 0)) return .{ .rmax = gmx, .cut = -inf, .id = vid, .km = rsum, .km_e = rsum_e };
        var n: usize = 0;
        var e: f64 = 0;
        for (vs[0..v]) |x| if (!std.math.isNan(x) and x >= cut) {
            const t = @exp(x - gmx);
            terms[n] = t;
            n += 1;
            e += t * (host.fastExp(x - gmx) + U);
        };
        const km = total(sim, terms[0..n]);
        return .{ .rmax = gmx, .cut = cut, .cut_e = cut_e, .id = vid, .km = if (km > 0) km else 1, .km_e = e + spec.gamma(@floatFromInt(n + 13), U) * km };
    }
    const k_lim = (if (has_k) top_k else gadm) + @intFromBool(planted);
    const p_lim = if (has_p) law.top_p * rsum else 2 * rsum;
    var floor = gmx - 12;
    while (true) {
        const ranked = collect(vs[0..v], cut, floor, toks);
        var cum: f64 = 0;
        var e: f64 = 0;
        var stop: ?usize = null;
        for (ranked, 0..) |t, i| {
            const x = t.v - gmx;
            const wt = @exp(x);
            cum = step(sim, cum, wt);
            e += wt * (host.fastExp(x) + U);
            if (i + 1 >= k_lim or cum >= p_lim) {
                stop = i;
                break;
            }
        }
        if (stop == null and floor != -inf) {
            floor = -inf;
            continue;
        }
        if (stop) |i| if (ranked[i].v < floor + 0.1 and floor != -inf) {
            floor = -inf;
            continue;
        };
        // The cut's bin: the crossing token's, or the last bin when the limits are never reached.
        const cb = if (stop) |i| binOf(gmx, ranked[i].v) else 511;
        var in_bin: usize = 0;
        var above: f64 = 0;
        var above_n: usize = 0;
        for (ranked) |t| {
            const bn = binOf(gmx, t.v);
            if (bn == cb) in_bin += 1;
            if (bn < cb) {
                above += @exp(t.v - gmx);
                above_n += 1;
            }
        }
        if (in_bin > 0 and in_bin <= 512) {
            const at = stop orelse ranked.len - 1;
            return .{ .rmax = gmx, .cut = ranked[at].v, .id = ranked[at].c, .km = if (cum > 0) cum else 1, .km_e = e + spec.gamma(@floatFromInt(at + 14), U) * cum };
        }
        // The fallback: an empty or overfull bin keeps down to its low edge, or its high edge when the bins above
        // already reach a limit.
        const hi_lo: [2]f64 = if (cb == 0) .{ gmx, gmx } else if (cb == 511) .{ f32of(gmx - 20), -inf } else .{ f32of(gmx - f32of(@as(f64, @floatFromInt(cb - 1)) / 25.5)), f32of(gmx - f32of(@as(f64, @floatFromInt(cb)) / 25.5)) };
        const reached = above > 0 and (above_n >= k_lim or above >= p_lim);
        const cl = if (reached) hi_lo[0] else @max(hi_lo[1], cut);
        return .{ .rmax = gmx, .cut = cl, .cut_e = if (!reached and cl == cut) cut_e else 0, .id = vid, .km = if (above > 0) above else rsum, .km_e = if (above > 0) e + spec.gamma(@floatFromInt(above_n + 13), U) * above else rsum_e };
    }
}

/// A row's draft-tree inputs for sample_tree_draw: each lane's candidates a depth (16 of its spike columns), each
/// row's depth, its drawn candidates (none in a quarter of rows) and the draft's normalized q over the candidates.
const Draws = struct { r: usize, tmeta: []f64, cands: []f64, next_q: []f64, draws: []f64 };

fn makeDraws(a: std.mem.Allocator, rng: std.Random, rows: *const Rows) !Draws {
    const r = rowsOf(rows.w);
    const d: Draws = .{
        .r = r,
        .tmeta = try a.alloc(f64, 2 * rows.n()),
        .cands = try a.alloc(f64, rows.b * r * KTOP),
        .next_q = try a.alloc(f64, rows.n() * KTOP),
        .draws = try a.alloc(f64, rows.n() * 4),
    };
    for (0..rows.b) |l| for (0..r) |pos| {
        var perm: [SPIKES]usize = undefined;
        for (&perm, 0..) |*x, i| x.* = i;
        rng.shuffle(usize, &perm);
        for (0..KTOP) |i| d.cands[(l * r + pos) * KTOP + i] = @floatFromInt(rows.spikes[l * SPIKES + perm[i]]);
    };
    for (0..rows.n()) |row| {
        d.tmeta[2 * row] = @floatFromInt(rng.uintLessThan(usize, r + 1));
        d.tmeta[2 * row + 1] = 1;
        var q: [KTOP]f64 = undefined;
        var s: f64 = 0;
        for (&q) |*x| {
            const u = rng.float(f64);
            x.* = u * u + 0.02;
            s += x.*;
        }
        for (q, 0..) |x, i| d.next_q[row * KTOP + i] = f32of(x / s);
        var perm: [KTOP]usize = undefined;
        for (&perm, 0..) |*x, i| x.* = i;
        rng.shuffle(usize, &perm);
        const nd = if (rng.uintLessThan(u32, 4) == 0) 0 else 1 + rng.uintLessThan(usize, 4);
        for (0..4) |ch| d.draws[row * 4 + ch] = if (ch < nd) @floatFromInt(perm[ch]) else 255;
    }
    return d;
}

const Stage = enum { mass, search, draw };

/// One of the verify's three sampling kernels over `Rows`: mass (PMASS [rows, 16, 4]), search (VROW [rows, 4] from
/// the mass's PMASS) or draw (SELECTED [rows], from the search's VROW and the draft tree). The planted controls: the
/// mass breaks a tie toward the higher id, the search keeps one token past the top-k, the draw takes the token after
/// the one its uniform lands on.
const Sample = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    arena: std.heap.ArenaAllocator,
    stage: Stage,
    masked: bool,
    planted: bool = false,
    shape: Shape = undefined,
    rows: Rows = undefined,
    /// Per row: the PMASS the search reads, the VROW the draw reads.
    parts: [][SHARDS]Part = &.{},
    vrows: []Vrow = &.{},
    tree: Draws = undefined,
    vs: []f64 = &.{},
    toks: []Tok = &.{},
    terms: []f64 = &.{},
    note: [240]u8 = undefined,
    args: [9]mlx.mlx_array = undefined,
    n_args: usize = 0,

    const Self = @This();

    fn init(a: std.mem.Allocator, stage: Stage, masked: bool, planted: bool) Self {
        const l: *const metal.Launch = switch (stage) {
            .mass => &mass_launch,
            .search => &search_launch,
            .draw => &draw_launch,
        };
        return .{ .rig = metal.Rig.init(a), .launch = l, .arena = .init(a), .stage = stage, .masked = masked, .planted = planted };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.rows > 16 or shape.n < 2 or shape.n > 16 or shape.k < 4096) return null;
        const rows = shape.rows * shape.n;
        const v = vocabOf(shape);
        return switch (self.stage) {
            .mass => massGeometry(rows, shape.n, v, shape.k, self.masked),
            .search => searchGeometry(rows, shape.n, v, shape.k, self.masked),
            .draw => drawGeometry(rows, shape.n, v, shape.k, self.masked, rowsOf(shape.n)),
        };
    }

    fn sampled(self: *const Self, r: usize) bool {
        return self.rows.laws[r / self.rows.w].temp > 0;
    }

    fn partsOf(self: *Self, r: usize, sim: Sim, planted: bool) [SHARDS]Part {
        scaled(&self.rows, r, self.vs);
        var out: [SHARDS]Part = undefined;
        for (&out, 0..) |*p, s| p.* = massOf(self.rows.v, self.vs[0..self.rows.v], s, self.sampled(r), sim, planted, self.terms);
        return out;
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const v = vocabOf(shape);
        self.vs = try a.alloc(f64, v);
        self.toks = try a.alloc(Tok, v);
        self.terms = try a.alloc(f64, v);
        self.rows = try makeRows(a, rng, shape, self.masked, fill);
        chooseLaws(&self.rows, rng, fill, self.vs, self.toks);
        const n = self.rows.n();
        if (self.stage != .mass) {
            // The PMASS a device's mass kernel would hand on: its sums rounded to f32.
            self.parts = try a.alloc([SHARDS]Part, n);
            for (self.parts, 0..) |*p, r| {
                p.* = self.partsOf(r, null, false);
                for (p) |*x| x.sum = f32of(x.sum);
            }
        }
        if (self.stage == .draw) {
            self.vrows = try a.alloc(Vrow, n);
            for (self.vrows, 0..) |*x, r| {
                scaled(&self.rows, r, self.vs);
                x.* = vrowOf(self.rows.laws[r / self.rows.w], v, self.vs[0..v], &self.parts[r], null, false, self.toks, self.terms);
                x.cut = f32of(x.cut);
                x.km = f32of(x.km);
            }
            self.tree = try makeDraws(a, rng, &self.rows);
            // Gate 1's planted rows draw no candidates; a row whose candidates hold nearly all its kept mass draws
            // none either (the outside mass would sit inside the device's error of zero).
            for (0..n) |r| {
                if (fill == .nonfinite and r < 4) self.tree.tmeta[2 * r] = @floatFromInt(self.tree.r);
                if (!self.hasDraw(r)) continue;
                scaled(&self.rows, r, self.vs);
                var psum: f64 = 0;
                for (self.candsOf(r)) |c| psum += self.admittedProb(r, self.vs, c);
                if (!clear(1, psum)) self.tree.draws[4 * r] = 255;
            }
            // Each sampled lane's seed: redrawn until every choice of its rows clears the device's error.
            for (self.rows.laws, 0..) |*law, l| {
                if (!(law.temp > 0) or law.top_k == 1) continue;
                var tries: usize = 0;
                while (!self.laneClear(l)) : (tries += 1) {
                    if (tries == 256) return error.NoClearSeed;
                    law.seed = rng.int(u64);
                }
            }
        }
        try self.bind();
        return true;
    }

    fn laneClear(self: *Self, l: usize) bool {
        for (l * self.rows.w..(l + 1) * self.rows.w) |r| {
            scaled(&self.rows, r, self.vs);
            if (!self.drawRow(r, self.vs, null, false).ok) return false;
        }
        return true;
    }

    fn bind(self: *Self) !void {
        const rows = &self.rows;
        const n = rows.n();
        const a = self.arena.allocator();
        self.rig.reset();
        self.args[0] = try self.rig.input(.bfloat16, &.{ cint(n), cint(rows.stride) }, rows.logits);
        self.args[1] = if (self.masked) try self.rig.inputRaw(.bool_, &.{cint(rows.v)}, rows.mask) else try self.rig.inputRaw(.bool_, &.{1}, &.{0});
        const lp = try a.alloc(f64, 4 * rows.b);
        for (rows.laws, 0..) |law, l| lp[4 * l ..][0..4].* = .{ law.temp, law.top_k, law.top_p, law.min_p };
        self.args[2] = try self.rig.input(.float32, &.{ cint(rows.b), 4 }, lp);
        self.n_args = 3;
        switch (self.stage) {
            .mass => {},
            .search => {
                const pm = try a.alloc(f64, n * SHARDS * 4);
                for (self.parts, 0..) |ps, r| for (ps, 0..) |p, s| {
                    pm[(r * SHARDS + s) * 4 ..][0..4].* = .{ p.mx, p.sum, @floatFromInt(p.adm), idBits(p.id) };
                };
                self.args[3] = try self.rig.input(.float32, &.{ cint(n), SHARDS, 4 }, pm);
                self.n_args = 4;
            },
            .draw => {
                const vr = try a.alloc(f64, n * 4);
                for (self.vrows, 0..) |x, r| vr[4 * r ..][0..4].* = .{ x.rmax, x.cut, idBits(x.id), x.km };
                const seeds = try a.alloc(f64, 2 * rows.b);
                for (rows.laws, 0..) |law, l| seeds[2 * l ..][0..2].* = .{ @floatFromInt(@as(u32, @truncate(law.seed))), @floatFromInt(@as(u32, @truncate(law.seed >> 32))) };
                const t = &self.tree;
                self.args[3] = try self.rig.input(.float32, &.{ cint(n), 4 }, vr);
                self.args[4] = try self.rig.input(.uint32, &.{ cint(rows.b), 2 }, seeds);
                self.args[5] = try self.rig.input(.int32, &.{ cint(n), 2 }, t.tmeta);
                self.args[6] = try self.rig.input(.int32, &.{ cint(rows.b), cint(t.r), KTOP }, t.cands);
                self.args[7] = try self.rig.input(.float32, &.{ cint(rows.b), cint(rows.w), KTOP }, t.next_q);
                self.args[8] = try self.rig.input(.uint8, &.{ cint(rows.b), cint(rows.w), 4 }, t.draws);
                self.n_args = 9;
            },
        }
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Self) usize {
        const n = self.shape.rows * self.shape.n;
        return switch (self.stage) {
            .mass => n * SHARDS * 4,
            .search => n * 4,
            .draw => n,
        };
    }

    // The draw's pieces, as sample_tree_draw's.

    fn hasDraw(self: *const Self, r: usize) bool {
        return tokU(self.tree.tmeta[2 * r]) < self.tree.r and self.tree.draws[4 * r] < KTOP;
    }

    fn candsOf(self: *const Self, r: usize) []const f64 {
        const depth = tokU(self.tree.tmeta[2 * r]);
        return self.tree.cands[((r / self.rows.w) * self.tree.r + depth) * KTOP ..][0..KTOP];
    }

    fn admitted(self: *const Self, r: usize, vs: []const f64, c: usize) bool {
        const x = vs[c];
        const vr = self.vrows[r];
        return !std.math.isNan(x) and (x > vr.cut or (x == vr.cut and c <= vr.id));
    }

    fn admittedProb(self: *const Self, r: usize, vs: []const f64, cid: f64) f64 {
        if (cid < 0 or cid >= @as(f64, @floatFromInt(self.rows.v))) return 0;
        const c = tokU(cid);
        if (!self.admitted(r, vs, c)) return 0;
        return @exp(vs[c] - self.vrows[r].rmax) / self.vrows[r].km;
    }

    const Pick = struct { tok: u32, ok: bool };

    /// The full-vocab walk: the first admitted column (not a candidate, with `excl`) whose running mass reaches
    /// `target`; past the end the last admitted one; none admitted, the cut's id.
    fn scan(self: *const Self, r: usize, vs: []const f64, excl: ?[]const f64, target: f64, sim: Sim, planted: bool) Pick {
        const vr = self.vrows[r];
        var cum: f64 = 0;
        var ok = true;
        var last: ?usize = null;
        var hit: ?usize = null;
        for (0..self.rows.v) |c| {
            if (!self.admitted(r, vs, c)) continue;
            if (excl) |e| if (std.mem.indexOfScalar(f64, e, @floatFromInt(c)) != null) continue;
            if (hit != null) return .{ .tok = @intCast(c), .ok = ok };
            const prev = cum;
            cum = step(sim, cum, @exp(vs[c] - vr.rmax));
            last = c;
            if (cum >= target) {
                if (!clear(cum, target) or !clear(prev, target)) ok = false;
                hit = c;
                if (!planted) break;
            }
        }
        if (hit) |h| return .{ .tok = @intCast(h), .ok = ok };
        if (last) |l| {
            if (!std.math.isNan(cum) and !clear(cum, target)) ok = false;
            return .{ .tok = @intCast(l), .ok = ok };
        }
        return .{ .tok = vr.id, .ok = ok };
    }

    /// Row `r`'s selection, and whether every choice it made clears the device's error.
    fn drawRow(self: *const Self, r: usize, vs: []const f64, sim: Sim, planted: bool) Pick {
        const law = self.rows.laws[r / self.rows.w];
        const vr = self.vrows[r];
        if (!(law.temp > 0) or law.top_k == 1) return .{ .tok = vr.id, .ok = true };
        if (!self.hasDraw(r)) return self.scan(r, vs, null, f32of(uniform01(law.seed, r, 5) * vr.km), sim, planted);
        const cands = self.candsOf(r);
        const q_in = self.tree.next_q[r * KTOP ..][0..KTOP];
        var p_rem: [KTOP]f64 = undefined;
        var q_rem: [KTOP]f64 = undefined;
        for (&p_rem, &q_rem, cands, q_in) |*p, *q, c, qi| {
            p.* = self.admittedProb(r, vs, c);
            q.* = qi;
        }
        var ok = true;
        const outside = @max(0, 1 - total(sim, &p_rem));
        var oscale: f64 = 1;
        var qmass: f64 = 1;
        for (0..4) |ch| {
            const di = tokU(self.tree.draws[4 * r + ch]);
            if (di >= KTOP) break;
            const qc = if (qmass > 0) q_rem[di] / qmass else 0;
            const ap = if (qc > 0) @min(1, p_rem[di] / qc) else 0;
            const u = uniform01(law.seed, r, @intCast(ch));
            if (ap > 0 and ap < 1 and !clear(u, ap)) ok = false;
            if (u < ap) return .{ .tok = tokU(cands[if (planted) (di + 1) % KTOP else di]), .ok = ok };
            var next: [KTOP]f64 = undefined;
            var z = outside * oscale;
            for (&next, p_rem, q_rem) |*x, p, q| {
                x.* = @max(0, p - (if (qmass > 0) q / qmass else 0));
                z += x.*;
            }
            if (!(z > 1e-9)) ok = false;
            if (z <= 1e-12) break;
            for (&p_rem, next) |*p, x| p.* = x / z;
            oscale /= z;
            qmass = @max(0, qmass - q_rem[di]);
            q_rem[di] = 0;
        }
        const cand_res = total(sim, &p_rem);
        const u_res = uniform01(law.seed, r, 5);
        if (!clear(u_res, cand_res)) ok = false;
        if (u_res < cand_res or !(outside * oscale > 1e-12)) {
            const target = if (u_res < cand_res) u_res else u_res * cand_res;
            var acc: f64 = 0;
            for (p_rem, 0..) |p, i| {
                const prev = acc;
                acc = step(sim, acc, p);
                if (acc >= target) {
                    if (!clear(acc, target) or !clear(prev, target)) ok = false;
                    return .{ .tok = tokU(cands[if (planted) (i + 1) % KTOP else i]), .ok = ok };
                }
            }
            return .{ .tok = tokU(cands[0]), .ok = ok };
        }
        const pick = self.scan(r, vs, cands, f32of((u_res - cand_res) / oscale * vr.km), sim, planted);
        return .{ .tok = pick.tok, .ok = ok and pick.ok };
    }

    /// Outputs and their errors (as `mag` allowances) by stage; `sim` runs the sums in an accumulator.
    fn emit(self: *Self, want: []f64, mag: []f64, sim: Sim) void {
        const v = self.rows.v;
        for (0..self.rows.n()) |r| switch (self.stage) {
            .mass => {
                const ps = self.partsOf(r, sim, self.planted);
                for (ps, 0..) |p, s| {
                    const o = (r * SHARDS + s) * 4;
                    want[o..][0..4].* = .{ p.mx, p.sum, @floatFromInt(p.adm), idBits(p.id) };
                    mag[o..][0..4].* = .{ 0, host.allowance(p.e), 0, 0 };
                }
            },
            .search => {
                scaled(&self.rows, r, self.vs);
                const x = vrowOf(self.rows.laws[r / self.rows.w], v, self.vs[0..v], &self.parts[r], sim, self.planted, self.toks, self.terms);
                want[4 * r ..][0..4].* = .{ x.rmax, x.cut, idBits(x.id), x.km };
                mag[4 * r ..][0..4].* = .{ 0, host.allowance(x.cut_e), 0, host.allowance(x.km_e) };
            },
            .draw => {
                scaled(&self.rows, r, self.vs);
                want[r] = @floatFromInt(self.drawRow(r, self.vs, sim, self.planted).tok);
                mag[r] = 0;
            },
        };
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        @constCast(self).emit(want, mag, null);
    }

    pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const mag = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(mag);
        @constCast(self).emit(out, mag, .{ .acc = acc, .order = order });
    }

    /// The stated rule: indices and counts exact, an id slot's bits exact, a max or cut a token's value exactly
    /// (min-p's cut within its log's error), masses within their f32 and fast-exp error; NaN where a +Inf logit's
    /// sampled mass is exp(Inf - Inf).
    pub fn check(self: *Self, _: spec.Gate) !?[]const u8 {
        const n = self.count();
        const buf = try self.arena.allocator().alloc(f64, 2 * n);
        const want = buf[0..n];
        const mag = buf[n..];
        try self.reference(want, mag);
        const bytes = self.rig.outputBytes();
        for (want, mag, 0..) |w, m, i| {
            const id_slot = switch (self.stage) {
                .mass => i % 4 == 3,
                .search => i % 4 == 2,
                .draw => false,
            };
            if (self.stage == .draw or id_slot) {
                const got: f64 = if (self.stage == .draw) @floatFromInt(@as(i32, @bitCast(bitsAt(bytes, i)))) else @floatFromInt(bitsAt(bytes, i));
                const ref: f64 = if (self.stage == .draw) w else @floatFromInt(@as(u32, @bitCast(@as(f32, @floatCast(w)))));
                if (got != ref) return fail(&self.note, "output {d}: {d} where the reference gives {d}", .{ i, got, ref });
                continue;
            }
            const got = spec.read(.f32, bytes, i);
            // `mag` is an allowance: the error it stands for.
            const e = m * U / (1 - U);
            if (!close(got, w, e)) return fail(&self.note, "output {d}: {e} where float64 gives {e} (error {e})", .{ i, got, w, e });
        }
        return null;
    }

    fn wrapperLanes(self: *const Self, out: []kernels.SampleLane, caps: []const f64) void {
        for (self.rows.laws, out, caps) |law, *o, c| o.* = .{ .temperature = @floatCast(law.temp), .top_k = tokU(law.top_k), .top_p = @floatCast(law.top_p), .min_p = @floatCast(law.min_p), .seed = law.seed, .accept_cap = tokU(c) };
    }
};

// ── dflash_conv ──

/// The draft's grouped dynamic convolution, both stages: rows = lanes, n = the block, k = the stream width. STAGE 0
/// (`prepare`) convolves its input; STAGE 1 (`finishAdd`) rounds the convolution to bf16 and adds the residual. bf16
/// throughout, as the draft runs. The planted control drops a tap (KT 1).
fn Conv(comptime stage: usize) type {
    return struct {
        rig: metal.Rig,
        launch: *const metal.Launch = &conv_launch,
        arena: std.heap.ArenaAllocator,
        planted: bool = false,
        shape: Shape = undefined,
        x: []f64 = &.{},
        dyn: []f64 = &.{},
        base: []f64 = &.{},
        res: []f64 = &.{},
        args: [4]mlx.mlx_array = undefined,

        const Self = @This();
        const G = HIDDEN / CONV_GS;

        fn init(a: std.mem.Allocator, planted: bool) Self {
            return .{ .rig = metal.Rig.init(a), .arena = .init(a), .planted = planted };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn geometry(self: *const Self, shape: Shape) ?metal.Geometry {
            if (shape.rows == 0 or shape.n == 0 or shape.k != HIDDEN) return null;
            return convGeometry(shape.rows, shape.n, stage, if (self.planted) CONV_KT - 1 else CONV_KT);
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const bl = shape.rows * shape.n;
            self.x = try host.uniform(a, rng, bl * HIDDEN, -2, 2, .bfloat16);
            self.dyn = try host.uniform(a, rng, bl * 2 * CONV_KT * G, -0.3, 0.3, .bfloat16);
            self.base = try host.uniform(a, rng, 2 * CONV_KT * HIDDEN, -0.5, 0.5, .bfloat16);
            self.res = try host.uniform(a, rng, bl * HIDDEN, -2, 2, .bfloat16);
            if (fill == .nonfinite) host.plant(self.x, bl, HIDDEN);
            self.rig.reset();
            const b = cint(shape.rows);
            const l = cint(shape.n);
            self.args[0] = try self.rig.input(.bfloat16, &.{ b, l, HIDDEN }, self.x);
            self.args[1] = try self.rig.input(.bfloat16, &.{ b, l, 2, CONV_KT, G }, self.dyn);
            self.args[2] = try self.rig.input(.bfloat16, &.{ 2, CONV_KT, HIDDEN }, self.base);
            // `prepare` hands its input as the unread residual.
            self.args[3] = if (stage == 1) try self.rig.input(.bfloat16, &.{ b, l, HIDDEN }, self.res) else self.args[0];
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            return self.shape.rows * self.shape.n * HIDDEN;
        }

        /// Output `i`'s four products (two a tap: the base's and the dynamic kernel's), zero before the first row.
        fn terms(self: *const Self, i: usize, t: *[2 * CONV_KT]f64) void {
            const len = self.shape.n;
            const c = i % HIDDEN;
            const l = (i / HIDDEN) % len;
            const b = i / (HIDDEN * len);
            for (0..CONV_KT) |tap| {
                if (l < tap) {
                    t[2 * tap] = 0;
                    t[2 * tap + 1] = 0;
                    continue;
                }
                const v = self.x[(b * len + l - tap) * HIDDEN + c];
                const bk = self.base[(stage * CONV_KT + tap) * HIDDEN + c];
                const dk = self.dyn[(((b * len + l) * 2 + stage) * CONV_KT + tap) * G + c / CONV_GS];
                t[2 * tap] = bk * v;
                t[2 * tap + 1] = dk * v;
            }
        }

        fn emit(self: *const Self, want: []f64, mag: ?[]f64, sim: Sim) void {
            for (want, 0..) |*w, i| {
                var t: [2 * CONV_KT]f64 = undefined;
                self.terms(i, &t);
                // A tap's two products, then the taps, as the kernel adds them (NaN and infinities included).
                var acc: f64 = 0;
                for (0..CONV_KT) |tap| acc += t[2 * tap] + t[2 * tap + 1];
                if (sim) |x| acc = spec.sumIn(x.acc, x.order, &t);
                const e = spec.gamma(2 * CONV_KT, U) * host.sumAbs(&t);
                var out: host.Val = .{ .v = acc, .e = e };
                if (stage == 1) {
                    const r1 = host.round(.bfloat16, out);
                    const s = r1.v + self.res[i];
                    out = .{ .v = s, .e = r1.e + U * @abs(s) };
                }
                w.* = out.v;
                if (mag) |m| m[i] = if (std.math.isFinite(out.e)) host.allowance(out.e) else 0;
            }
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            self.emit(want, mag, null);
        }

        pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
            self.emit(out, null, .{ .acc = acc, .order = order });
        }
    };
}

/// A drafter carrying only what the conv and selector wrappers read.
fn testDrafter(s: mlx.mlx_stream, hidden: usize) dflash2.Drafter {
    return .{
        .cfg = .{ .layers = 1, .heads = 1, .kv_heads = 1, .head_dim = 16, .intermediate = 32, .window = 16, .block = BLOCK, .mask_id = 0, .tap_count = 2, .conv_kernel = CONV_KT, .conv_group = CONV_GS, .rank = RANK, .top_k = KTOP, .softcap = @floatCast(SOFTCAP) },
        .hidden = @intCast(hidden),
        .s = s,
    };
}

// ── dflash_top16_shard, dflash_top16_merge ──

fn shardOuts(shape: Shape) usize {
    return shape.rows * SHARDS * KTOP;
}

fn mergeOuts(shape: Shape) usize {
    return shape.rows * KTOP;
}

/// Each row's top 16 a shard: rows = the block rows of every lane, n = the logits' stride, k = the vocab. The
/// padding past the vocab holds 1000, which a read past V would rank first. Row 0's shard 0 holds two equal maxima
/// (the lower id ranks first; the planted control ranks the higher).
const Shard = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &shard_launch,
    arena: std.heap.ArenaAllocator,
    planted: bool = false,
    shape: Shape = undefined,
    logits: []f64 = &.{},
    args: [1]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator, planted: bool) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a), .planted = planted };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.k < SHARDS * KTOP or shape.n < shape.k) return null;
        return shardGeometry(shape.rows, shape.n, shape.k);
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.logits = try a.alloc(f64, shape.rows * shape.n);
        for (0..shape.rows) |r| {
            const row = self.logits[r * shape.n ..][0..shape.n];
            for (row[0..shape.k]) |*x| x.* = bfOf(2 * rng.floatNorm(f64));
            for (row[shape.k..]) |*x| x.* = 1000;
            for (0..SPIKES) |_| row[rng.uintLessThan(usize, shape.k)] = bfOf(10 + 8 * rng.float(f64));
        }
        self.logits[5] = 50;
        self.logits[9] = 50;
        if (fill == .nonfinite) host.plant(self.logits, shape.rows, shape.n);
        self.rig.reset();
        self.args[0] = try self.rig.input(.bfloat16, &.{ cint(shape.rows), cint(shape.n) }, self.logits);
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return 2 * shardOuts(self.shape);
    }

    /// Rank order; planted: a tie's higher id first.
    fn tieLess(planted: bool, x: Tok, y: Tok) bool {
        return x.v > y.v or (x.v == y.v and (if (planted) x.c > y.c else x.c < y.c));
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        const v = self.shape.k;
        const chunk = (v + SHARDS - 1) / SHARDS;
        const half = shardOuts(self.shape);
        const buf = try self.rig.a.alloc(Tok, chunk);
        defer self.rig.a.free(buf);
        for (0..self.shape.rows) |r| for (0..SHARDS) |s| {
            const lo = @min(v, s * chunk);
            const hi = @min(v, lo + chunk);
            var n: usize = 0;
            for (lo..hi) |c| {
                const x = self.logits[r * self.shape.n + c];
                if (std.math.isNan(x)) continue;
                buf[n] = .{ .v = x, .c = @intCast(c) };
                n += 1;
            }
            std.mem.sort(Tok, buf[0..n], self.planted, tieLess);
            const o = (r * SHARDS + s) * KTOP;
            for (0..KTOP) |m| {
                const t: Tok = if (m < n) buf[m] else .{ .v = -inf, .c = @intCast(v) };
                want[o + m] = @floatFromInt(t.c);
                want[half + o + m] = t.v;
            }
        };
        for (want, mag) |w, *m| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }
};

/// The shards' lists merged: rows = block rows, n = shards, k = 16. Each shard's list sorted (value descending, id
/// ascending) over its own ids, values on a coarse grid so lists tie within and across shards, some shards padded
/// with (-Inf, V) as a shard short of 16 numbers hands on. Row 0's shard 15 heads the row (the planted control merges
/// 15 shards).
const Merge = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &merge_launch,
    arena: std.heap.ArenaAllocator,
    planted: bool = false,
    shape: Shape = undefined,
    ids: []f64 = &.{},
    vals: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator, planted: bool) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a), .planted = planted };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n != SHARDS or shape.k != KTOP) return null;
        return mergeGeometry(shape.rows, if (self.planted) SHARDS - 1 else SHARDS);
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * SHARDS * KTOP;
        self.ids = try a.alloc(f64, n);
        self.vals = try a.alloc(f64, n);
        const chunk = (VOCAB + SHARDS - 1) / SHARDS;
        for (0..shape.rows) |r| for (0..SHARDS) |s| {
            var list: [KTOP]Tok = undefined;
            for (&list) |*t| t.* = .{ .v = 10 + 0.25 * @as(f64, @floatFromInt(rng.uintLessThan(u32, 40))), .c = @intCast(s * chunk + rng.uintLessThan(usize, chunk)) };
            if (r == 0 and s == SHARDS - 1) list[0].v = 30;
            if (fill == .nonfinite) {
                if (r == 1 and s == 3) list[0].v = inf;
                if (r == 3 and s == 7) list[0].v = inf;
                if (r == 2 and s == 0) for (list[10..]) |*t| {
                    t.v = -inf;
                };
            }
            std.mem.sort(Tok, &list, {}, rankDesc);
            // A shard short of 16 numbers: (-Inf, V) past them.
            const real: usize = if (r % 2 == 1 and s == SHARDS - 1) KTOP - 4 else KTOP;
            for (list[real..]) |*t| t.* = .{ .v = -inf, .c = VOCAB };
            const o = (r * SHARDS + s) * KTOP;
            for (list, 0..) |t, m| {
                self.ids[o + m] = @floatFromInt(t.c);
                self.vals[o + m] = t.v;
            }
        };
        self.rig.reset();
        self.args = .{
            try self.rig.input(.int32, &.{ cint(shape.rows), SHARDS, KTOP }, self.ids),
            try self.rig.input(.float32, &.{ cint(shape.rows), SHARDS, KTOP }, self.vals),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return 2 * mergeOuts(self.shape);
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        const half = mergeOuts(self.shape);
        for (0..self.shape.rows) |r| {
            var all: [SHARDS * KTOP]Tok = undefined;
            for (&all, 0..) |*t, j| {
                const at = r * SHARDS * KTOP + j;
                t.* = .{ .v = self.vals[at], .c = tokU(self.ids[at]) };
            }
            std.mem.sort(Tok, &all, {}, rankDesc);
            for (all[0..KTOP], 0..) |t, m| {
                want[r * KTOP + m] = @floatFromInt(t.c);
                want[half + r * KTOP + m] = t.v;
            }
        }
        for (want, mag) |w, *m| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }
};

// ── dflash_edges ──

/// The selector's R x K x K lattice: rows = lanes, n = tree rows R, k = the rank. PRED and SUCC are the codebooks'
/// first TABLE rows (the kernel reads the rows its tokens name). Products of bf16 values are exact in f32: only the
/// sums, the soft cap's tanh and the softmax round. The planted control sums 255 of the 256 rank terms.
const Edges = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &edges_launch,
    arena: std.heap.ArenaAllocator,
    planted: bool = false,
    shape: Shape = undefined,
    t1: []f64 = &.{},
    cands: []f64 = &.{},
    unary: []f64 = &.{},
    proj: []f64 = &.{},
    pred: []f64 = &.{},
    succ: []f64 = &.{},
    errs: []f64 = &.{},
    note: [200]u8 = undefined,
    args: [7]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator, planted: bool) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a), .planted = planted };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0 or shape.n > BLOCK - 1 or shape.k != RANK) return null;
        return edgesGeometry(shape.rows, shape.n, if (self.planted) RANK - 1 else RANK);
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const b = shape.rows;
        const r = shape.n;
        self.t1 = try a.alloc(f64, b);
        for (self.t1) |*x| x.* = @floatFromInt(rng.uintLessThan(usize, TABLE));
        self.cands = try a.alloc(f64, b * r * KTOP);
        for (self.cands) |*x| x.* = @floatFromInt(rng.uintLessThan(usize, TABLE));
        // A row's candidates' logits, descending as the top 16 hands them on.
        self.unary = try a.alloc(f64, b * r * KTOP);
        for (self.unary, 0..) |*x, i| x.* = f32of(18 - 0.3 * @as(f64, @floatFromInt(i % KTOP)) + 0.2 * rng.float(f64));
        self.proj = try host.uniform(a, rng, b * r * RANK, -1, 1, .bfloat16);
        self.pred = try host.uniform(a, rng, TABLE * RANK, -0.5, 0.5, .bfloat16);
        self.succ = try host.uniform(a, rng, TABLE * RANK, -0.5, 0.5, .bfloat16);
        if (fill == .nonfinite) {
            // A NaN candidate logit; a +Inf one (capped to the soft cap); a position with every one -Inf; one -Inf.
            self.unary[3] = nan;
            if (r > 1) self.unary[KTOP + 5] = inf;
            if (r > 2) @memset(self.unary[2 * KTOP ..][0..KTOP], -inf);
            if (b > 1) self.unary[r * KTOP] = -inf;
        }
        self.errs = try a.alloc(f64, b * r * KTOP * KTOP);
        self.rig.reset();
        self.args = .{
            try self.rig.input(.int32, &.{cint(b)}, self.t1),
            try self.rig.input(.int32, &.{ cint(b), cint(r), KTOP }, self.cands),
            try self.rig.input(.float32, &.{ cint(b), cint(r), KTOP }, self.unary),
            try self.rig.input(.bfloat16, &.{ cint(b), cint(r), RANK }, self.proj),
            try self.rig.input(.bfloat16, &.{ TABLE, RANK }, self.pred),
            try self.rig.input(.bfloat16, &.{ TABLE, RANK }, self.succ),
            try self.rig.input(.float32, &.{1}, &.{SOFTCAP}),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return self.shape.rows * self.shape.n * KTOP * KTOP;
    }

    /// Each probability and its error. A NaN score, or a position whose scores are all -Inf, makes the mass not a
    /// number: the kernel then writes the uniform 1/K (its stated rule).
    fn emit(self: *const Self, want: []f64, errs: []f64, sim: Sim) void {
        const b_n = self.shape.rows;
        const r = self.shape.n;
        for (0..b_n) |b| for (0..r) |pos| for (0..KTOP) |pi| {
            const o = ((b * r + pos) * KTOP + pi) * KTOP;
            const out = want[o..][0..KTOP];
            const err = errs[o..][0..KTOP];
            @memset(err, 0);
            if (pos == 0 and pi > 0) {
                @memset(out, 0);
                continue;
            }
            const prev = tokU(if (pos == 0) self.t1[b] else self.cands[(b * r + pos - 1) * KTOP + pi]);
            const row = (b * r + pos) * RANK;
            var sc: [KTOP]f64 = undefined;
            var se: [KTOP]f64 = undefined;
            var bad = false;
            var mx = -inf;
            var mx_e: f64 = 0;
            for (0..KTOP) |s| {
                const sid = tokU(self.cands[(b * r + pos) * KTOP + s]);
                var t: [RANK]f64 = undefined;
                for (&t, 0..) |*x, c| x.* = self.pred[prev * RANK + c] * self.proj[row + c] * self.succ[sid * RANK + c];
                const edge = if (sim) |x| spec.sumIn(x.acc, x.order, &t) else host.sum(&t);
                const raw = self.unary[(b * r + pos) * KTOP + s] + edge;
                const e_raw = spec.gamma(RANK + 5, U) * host.sumAbs(&t) + U * @abs(raw);
                if (raw > -inf and std.math.isFinite(raw)) {
                    const th = std.math.tanh(raw / SOFTCAP);
                    sc[s] = SOFTCAP * th;
                    se[s] = (1 - th * th) * (e_raw + U * @abs(raw)) + (PRECISE + 2 * U) * @abs(sc[s]);
                } else {
                    // +Inf caps to the soft cap exactly; -Inf and NaN pass as they are.
                    sc[s] = if (raw == inf) SOFTCAP else raw;
                    se[s] = 0;
                }
                if (std.math.isNan(sc[s])) bad = true else if (sc[s] > mx) {
                    mx = sc[s];
                    mx_e = se[s];
                }
            }
            if (bad or mx == -inf) {
                @memset(out, 1.0 / @as(f64, KTOP));
                continue;
            }
            var w: [KTOP]f64 = undefined;
            var rel: [KTOP]f64 = undefined;
            var m: f64 = 0;
            for (&w, &rel, sc, se) |*x, *re, s, e| {
                x.* = @exp(s - mx);
                re.* = e + mx_e + PRECISE;
                m += x.* * re.*;
            }
            const mass = total(sim, &w);
            const mass_rel = (m + spec.gamma(KTOP, U) * mass) / mass;
            for (out, err, w, rel) |*p, *e, x, re| {
                p.* = x / mass;
                e.* = p.* * (re + mass_rel + U);
            }
        };
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        self.emit(want, self.errs, null);
        for (mag, self.errs) |*m, e| m.* = host.allowance(e);
    }

    pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const errs = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(errs);
        self.emit(out, errs, .{ .acc = acc, .order = order });
    }

    /// The stated rule (the uniform 1/K where the mass is not a number), and every probability within its error.
    pub fn check(self: *Self, _: spec.Gate) !?[]const u8 {
        const want = try self.arena.allocator().alloc(f64, self.count());
        self.emit(want, self.errs, null);
        const bytes = self.rig.outputBytes();
        for (want, self.errs, 0..) |w, e, i| {
            const got = spec.read(.f32, bytes, i);
            if (!close(got, w, e)) return fail(&self.note, "probability {d}: {e} where float64 gives {e} (error {e})", .{ i, got, w, e });
        }
        return null;
    }
};

// ── dflash_tree ──

fn treeTok(shape: Shape) usize {
    return shape.rows * shape.n;
}

fn treeMeta(shape: Shape) usize {
    return 2 * shape.rows * shape.n;
}

fn treeQ(shape: Shape) usize {
    return shape.rows * shape.n * KTOP;
}

fn treeDraws(shape: Shape) usize {
    return 4 * shape.rows * shape.n;
}

const tree_outs: []const spec.Part = &.{ .{ .elem = .u32, .count = treeTok }, .{ .elem = .u32, .count = treeTok }, .{ .elem = .u32, .count = treeMeta }, .{ .elem = .f32, .count = treeQ }, .{ .elem = .i8, .count = treeDraws }, .{ .elem = .f32, .count = treeTok } };
/// The tree's outputs as `spec.readOut` reads them.
const tree_read: spec.Spec = .{ .name = "inkling2 dflash_tree", .accum = .none, .depth = none, .out = .u32, .outs = tree_outs, .served = &.{} };

const HCAP = 96;

/// One lane's best-first build, as dflash_tree runs it, in float64; `ok` falls when a comparison of computed values
/// lies inside the device's error.
const Build = struct {
    t: *const Tree,
    l: usize,
    sim: Sim,
    ok: bool = true,
    sampled: bool,
    temp: f64,
    copy_n: usize,
    copy_r: f64,
    tok: [16]f64 = undefined,
    par: [16]f64 = undefined,
    dep: [16]usize = undefined,
    anc: [16]u32 = undefined,
    on_copy: [16]bool = undefined,
    q: [16]f64 = undefined,
    qe: [16]f64 = undefined,
    key: [16]u64 = undefined,
    draw: [16][4]u8 = undefined,
    placed: [16]usize = undefined,
    next: [16][KTOP]f64 = undefined,
    next_e: [16][KTOP]f64 = undefined,
    h_score: [HCAP]f64 = undefined,
    h_par: [HCAP]usize = undefined,
    h_pos: [HCAP]usize = undefined,
    h_idx: [HCAP]usize = undefined,
    h_ord: [HCAP]usize = undefined,
    h_size: usize = 0,

    fn gt(s: *Build, a: f64, b: f64) bool {
        if (a != b and std.math.isFinite(a) and std.math.isFinite(b) and !clear(a, b)) s.ok = false;
        return a > b;
    }

    fn ge(s: *Build, a: f64, b: f64) bool {
        if (a != b and std.math.isFinite(a) and std.math.isFinite(b) and !clear(a, b)) s.ok = false;
        return a >= b;
    }

    fn isStop(s: *const Build, tok: f64) bool {
        for (s.t.stops[0..s.t.n_stops]) |x| if (x == tok) return true;
        return false;
    }

    fn cand(s: *const Build, pos: usize, i: usize) f64 {
        return s.t.cands[(s.l * s.t.r + pos) * KTOP + i];
    }

    fn copyAt(s: *const Build, pos: usize) f64 {
        return s.t.copy_ids[s.l * MAX_COPY + pos];
    }

    fn estProb(s: *const Build, q: f64, pos: usize) f64 {
        const qc = std.math.clamp(q, f32of(1e-6), f32of(1 - f32of(1e-6)));
        const lq = @log(qc / (1 - qc));
        const calib = s.t.calib[s.l * (1 + s.t.r) ..];
        const z = calib[0] * lq + calib[1 + pos];
        return 1 / (1 + @exp(-z));
    }

    fn gumbel(key: u64, tok: u32) f64 {
        const h = mix64(key ^ (0xd1b54a32d192ed03 *% (@as(u64, tok) + 1)));
        const u = (@as(f64, @floatFromInt(h >> 41)) + 0.5) * 0x1p-23;
        return -@log(-@log(u));
    }

    fn push(s: *Build, sc: f64, par: usize, pos: usize, idx: usize, ord: usize) void {
        if (s.h_size >= HCAP) return;
        var cur = s.h_size;
        s.h_size += 1;
        while (cur > 0) {
            const p = (cur - 1) >> 1;
            if (s.ge(s.h_score[p], sc)) break;
            s.h_score[cur] = s.h_score[p];
            s.h_par[cur] = s.h_par[p];
            s.h_pos[cur] = s.h_pos[p];
            s.h_idx[cur] = s.h_idx[p];
            s.h_ord[cur] = s.h_ord[p];
            cur = p;
        }
        s.h_score[cur] = sc;
        s.h_par[cur] = par;
        s.h_pos[cur] = pos;
        s.h_idx[cur] = idx;
        s.h_ord[cur] = ord;
    }

    const Entry = struct { sc: f64, par: usize, pos: usize, idx: usize, ord: usize };

    fn pop(s: *Build) Entry {
        const top: Entry = .{ .sc = s.h_score[0], .par = s.h_par[0], .pos = s.h_pos[0], .idx = s.h_idx[0], .ord = s.h_ord[0] };
        s.h_size -= 1;
        if (s.h_size > 0) {
            const n = s.h_size;
            const last: Entry = .{ .sc = s.h_score[n], .par = s.h_par[n], .pos = s.h_pos[n], .idx = s.h_idx[n], .ord = s.h_ord[n] };
            var cur: usize = 0;
            while (true) {
                const l = cur * 2 + 1;
                if (l >= n) break;
                const r = l + 1;
                const best = if (r < n and s.gt(s.h_score[r], s.h_score[l])) r else l;
                if (s.ge(last.sc, s.h_score[best])) break;
                s.h_score[cur] = s.h_score[best];
                s.h_par[cur] = s.h_par[best];
                s.h_pos[cur] = s.h_pos[best];
                s.h_idx[cur] = s.h_idx[best];
                s.h_ord[cur] = s.h_ord[best];
                cur = best;
            }
            s.h_score[cur] = last.sc;
            s.h_par[cur] = last.par;
            s.h_pos[cur] = last.pos;
            s.h_idx[cur] = last.idx;
            s.h_ord[cur] = last.ord;
        }
        return top;
    }

    fn expand(s: *Build, row: usize, pos: usize, pred_i: usize, path: f64) void {
        const t = s.t;
        if (pos >= t.r) return;
        const q = t.probs[((s.l * t.r + pos) * KTOP + pred_i) * KTOP ..][0..KTOP];
        var top: f64 = 0;
        for (q) |x| if (x > top) {
            top = x;
        };
        var wt: [KTOP]f64 = undefined;
        var rel: [KTOP]f64 = undefined;
        for (&wt, &rel, q) |*w, *re, x| {
            if (!s.sampled or !(x > 0)) {
                w.* = x;
                re.* = 0;
                continue;
            }
            const d = @log(x) - @log(top);
            const z = d / s.temp;
            w.* = @exp(z);
            re.* = (fastLogErr(x) + fastLogErr(top) + U * @abs(d)) / s.temp + U * @abs(z) + host.fastExp(z) + U;
        }
        const mass = total(s.sim, &wt);
        if (mass > 0) {
            var m: f64 = 0;
            for (wt, rel) |w, re| m += w * re;
            const mass_rel = (m + spec.gamma(KTOP, U) * mass) / mass;
            for (&s.next[row], &s.next_e[row], wt, rel) |*n, *e, w, re| {
                n.* = w / mass;
                e.* = n.* * (re + mass_rel + U);
            }
        } else for (&s.next[row], &s.next_e[row], wt) |*n, *e, w| {
            n.* = w;
            e.* = 0;
        }
        var gkey: [KTOP]f64 = undefined;
        if (s.sampled) for (&gkey, s.next[row], 0..) |*g, n, i| {
            g.* = if (n > 0) @log(n) + gumbel(s.key[row], tokU(s.cand(pos, i))) else -inf;
        };
        const has_copy = s.on_copy[row] and pos < s.copy_n;
        const copy_tok: f64 = if (has_copy) s.copyAt(pos) else -1;
        const c_rate = if (has_copy) s.copy_r else 0;
        var taken: u32 = 0;
        var bound = path;
        var left = 1 - c_rate;
        var reserved = c_rate;
        var joined = false;
        for (0..@min(t.max_ch, 4)) |ch| {
            var picked: usize = KTOP;
            for (0..KTOP) |i| {
                if (taken & (@as(u32, 1) << @intCast(i)) != 0) continue;
                if (!s.sampled) {
                    if (!(q[i] > 0)) continue;
                    if (picked == KTOP or q[i] > q[picked] or (t.planted and q[i] == q[picked])) picked = i;
                } else {
                    if (!(gkey[i] > -inf)) continue;
                    if (picked == KTOP or s.gt(gkey[i], gkey[picked])) picked = i;
                }
            }
            if (picked == KTOP) break;
            taken |= @as(u32, 1) << @intCast(picked);
            const tok = s.cand(pos, picked);
            if (!s.sampled and s.isStop(tok)) continue;
            const joins = has_copy and tok == copy_tok;
            if (joins) joined = true;
            const cap = if (joins) left + reserved else left;
            const own = @min(s.estProb(s.next[row][picked], pos), @max(cap, 0));
            const eff = if (joins) @max(own, reserved) else own;
            left = if (joins) @max(0, cap - eff) else @max(0, left - own);
            if (joins) reserved = 0;
            bound = @min(bound, path * eff);
            s.draw[row][ch] = @intCast(picked);
            s.push(bound, row, pos, picked, ch);
        }
        if (has_copy and !joined and !s.sampled and !s.isStop(copy_tok)) s.push(path * c_rate, row, pos, 255, 255);
    }

    fn run(s: *Build) void {
        const t = s.t;
        const t1 = t.t1[s.l];
        const seed0 = t.seeds[s.l];
        for (0..16) |i| {
            s.tok[i] = t1;
            s.par[i] = -1;
            s.dep[i] = 0;
            s.anc[i] = 1;
            s.on_copy[i] = false;
            s.q[i] = -1;
            s.qe[i] = 0;
            s.key[i] = seed0;
            s.placed[i] = 0;
            s.draw[i] = @splat(255);
            s.next[i] = @splat(0);
            s.next_e[i] = @splat(0);
        }
        s.on_copy[0] = s.copy_n > 0;
        s.q[0] = 1;
        const want_w = @min(t.w, t.want_w);
        var made: usize = 1;
        if (want_w > 1) s.expand(0, 0, 0, 1.0);
        while (s.h_size > 0 and made < want_w) {
            const e = s.pop();
            if (s.sampled and e.ord != 255 and e.ord != s.placed[e.par]) continue;
            const row = made;
            made += 1;
            const is_copy = e.idx == 255;
            const tok = if (is_copy) s.copyAt(e.pos) else s.cand(e.pos, e.idx);
            s.tok[row] = tok;
            s.par[row] = @floatFromInt(e.par);
            s.dep[row] = e.pos + 1;
            s.anc[row] = s.anc[e.par] | (@as(u32, 1) << @intCast(row));
            s.on_copy[row] = s.on_copy[e.par] and e.pos < s.copy_n and tok == s.copyAt(e.pos);
            s.q[row] = if (is_copy) -1 else s.next[e.par][e.idx];
            s.qe[row] = if (is_copy) 0 else s.next_e[e.par][e.idx];
            s.key[row] = mix64(s.key[e.par] +% 0x9e3779b97f4a7c15 *% (@as(u64, tokU(tok)) + 1));
            if (e.ord != 255) s.placed[e.par] = e.ord + 1;
            if (!s.isStop(tok)) {
                if (!is_copy and e.pos + 1 < t.r) {
                    s.expand(row, e.pos + 1, e.idx, e.sc);
                } else if (s.on_copy[row] and e.pos + 1 < s.copy_n and !s.sampled) {
                    if (!s.isStop(s.copyAt(e.pos + 1))) s.push(e.sc * s.copy_r, row, e.pos + 1, 255, 255);
                }
            }
        }
        for (0..t.w) |row| for (s.placed[row]..4) |ch| {
            s.draw[row][ch] = 255;
        };
    }
};

/// The draft tree builder: rows = lanes, n = the tree's width (verify rows), k = 16. Even lanes greedy, odd lanes
/// sampled at 0.8; lattices from softmaxes of random scores; copy chains on some lanes, joining a candidate at some
/// depths; one stop id among the candidates. Inputs are redrawn until every comparison of computed path scores and
/// Gumbel keys clears the device's error. Lane 0's root holds two equal best probabilities (the lower index is drawn
/// first; the planted control draws the higher).
const Tree = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &tree_launch,
    arena: std.heap.ArenaAllocator,
    planted: bool = false,
    shape: Shape = undefined,
    w: usize = 0,
    r: usize = 0,
    t1: []f64 = &.{},
    cands: []f64 = &.{},
    probs: []f64 = &.{},
    calib: []f64 = &.{},
    temps: []f64 = &.{},
    seeds: []u64 = &.{},
    copy_ids: []f64 = &.{},
    copy_len: []f64 = &.{},
    copy_rate: []f64 = &.{},
    stops: [MAX_STOPS]f64 = @splat(-1),
    n_stops: usize = 0,
    max_ch: usize = MAX_CH,
    want_w: usize = 0,
    errs: []f64 = &.{},
    note: [240]u8 = undefined,
    args: [11]mlx.mlx_array = undefined,

    const Self = @This();
    const intercepts = [8]f64{ 0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45, -0.45 };

    fn init(a: std.mem.Allocator, planted: bool) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a), .planted = planted };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n < 2 or shape.n > 16 or shape.k != KTOP) return null;
        return treeGeometry(shape.rows, shape.n, rowsOf(shape.n));
    }

    fn fillInputs(self: *Self, rng: std.Random, fill: spec.Fill) void {
        const b = self.shape.rows;
        const r = self.r;
        for (0..b) |l| {
            self.t1[l] = @floatFromInt(5000 + l);
            self.temps[l] = if (l % 2 == 1) f32of(0.8) else 0;
            self.seeds[l] = rng.int(u64);
            self.calib[l * (1 + r)] = f32of(0.7);
            for (0..r) |p| self.calib[l * (1 + r) + 1 + p] = f32of(intercepts[@min(p, 7)]);
            for (0..r) |pos| for (0..KTOP) |i| {
                self.cands[(l * r + pos) * KTOP + i] = @floatFromInt(100 + l * 1000 + pos * KTOP + i);
            };
            // Copy chains on every third lane, joining the candidates at some depths.
            const cn: usize = if (l % 3 == 2) 1 + rng.uintLessThan(usize, MAX_COPY) else 0;
            self.copy_len[l] = @floatFromInt(cn);
            self.copy_rate[l] = f32of(0.9);
            for (0..MAX_COPY) |p| self.copy_ids[l * MAX_COPY + p] = if (p >= cn) 0 else if (p < r and rng.boolean()) self.cands[(l * r + p) * KTOP + rng.uintLessThan(usize, 4)] else @floatFromInt(90000 + p);
            for (0..r) |pos| for (0..KTOP) |pi| {
                const o = ((l * r + pos) * KTOP + pi) * KTOP;
                if (pos == 0 and pi > 0) {
                    @memset(self.probs[o..][0..KTOP], 0);
                    continue;
                }
                var sc: [KTOP]f64 = undefined;
                var s: f64 = 0;
                for (&sc) |*x| {
                    x.* = @exp(1.5 * rng.floatNorm(f64));
                    s += x.*;
                }
                for (sc, 0..) |x, i| self.probs[o + i] = f32of(x / s);
            };
        }
        // Lane 0's root: candidates 2 and 5 tie at its best probability.
        const root = self.probs[0..KTOP];
        var best: f64 = 0;
        for (root) |x| best = @max(best, x);
        root[2] = best;
        root[5] = best;
        self.copy_len[0] = 0;
        // A stop id among lane 0's depth-1 candidates (greedy skips it), and one no candidate carries.
        self.stops = @splat(-1);
        self.stops[0] = if (r > 1) self.cands[KTOP + 1] else 99999;
        self.stops[1] = 99998;
        self.n_stops = 2;
        self.want_w = self.w;
        // Gate 1: NaN and -Inf probabilities in greedy lanes' rows the build expands (their rows' weights then stay
        // unnormalized, and they are never drawn).
        if (fill == .nonfinite) {
            self.probs[3] = nan;
            self.probs[7] = -inf;
            if (b > 2 and r > 1) for (0..KTOP) |pi| {
                self.probs[((2 * r + 1) * KTOP + pi) * KTOP + 4] = nan;
            };
        }
    }

    fn lanes(self: *const Self, out_ok: *bool, want: []f64, errs: []f64, sim: Sim) void {
        const shape = self.shape;
        const b_n = shape.rows;
        const w = self.w;
        const nt = treeTok(shape);
        const tok_at = want[0..nt];
        const par_at = want[nt..][0..nt];
        const meta_at = want[2 * nt ..][0..treeMeta(shape)];
        const q_at = want[4 * nt ..][0..treeQ(shape)];
        const d_at = want[4 * nt + treeQ(shape) ..][0..treeDraws(shape)];
        const s_at = want[8 * nt + treeQ(shape) ..][0..nt];
        @memset(errs, 0);
        const qe_at = errs[4 * nt ..][0..treeQ(shape)];
        const se_at = errs[8 * nt + treeQ(shape) ..][0..nt];
        for (0..b_n) |l| {
            var s: Build = .{
                .t = self,
                .l = l,
                .sim = sim,
                .sampled = self.temps[l] > 0,
                .temp = self.temps[l],
                .copy_n = tokU(@max(0, self.copy_len[l])),
                .copy_r = self.copy_rate[l],
            };
            s.run();
            if (!s.ok) out_ok.* = false;
            for (0..w) |row| {
                const at = l * w + row;
                tok_at[at] = s.tok[row];
                // -1 read as the u32 it is stored as.
                par_at[at] = if (s.par[row] < 0) 4294967295 else s.par[row];
                meta_at[2 * at] = @floatFromInt(s.dep[row]);
                meta_at[2 * at + 1] = @floatFromInt(s.anc[row]);
                for (0..KTOP) |k| {
                    q_at[at * KTOP + k] = s.next[row][k];
                    qe_at[at * KTOP + k] = s.next_e[row][k];
                }
                for (0..4) |ch| d_at[at * 4 + ch] = if (s.draw[row][ch] == 255) -1 else @floatFromInt(s.draw[row][ch]);
                s_at[at] = s.q[row];
                se_at[at] = s.qe[row];
            }
        }
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        self.w = shape.n;
        self.r = rowsOf(shape.n);
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const b = shape.rows;
        const r = self.r;
        self.t1 = try a.alloc(f64, b);
        self.cands = try a.alloc(f64, b * r * KTOP);
        self.probs = try a.alloc(f64, b * r * KTOP * KTOP);
        self.calib = try a.alloc(f64, b * (1 + r));
        self.temps = try a.alloc(f64, b);
        self.seeds = try a.alloc(u64, b);
        self.copy_ids = try a.alloc(f64, b * MAX_COPY);
        self.copy_len = try a.alloc(f64, b);
        self.copy_rate = try a.alloc(f64, b);
        self.errs = try a.alloc(f64, self.count());
        const want = try a.alloc(f64, self.count());
        var tries: usize = 0;
        while (true) : (tries += 1) {
            if (tries == 256) return error.NoClearTree;
            self.fillInputs(rng, fill);
            var ok = true;
            self.lanes(&ok, want, self.errs, null);
            if (ok) break;
        }
        const seeds = try a.alloc(f64, 2 * b);
        for (self.seeds, 0..) |x, l| seeds[2 * l ..][0..2].* = .{ @floatFromInt(@as(u32, @truncate(x))), @floatFromInt(@as(u32, @truncate(x >> 32))) };
        self.rig.reset();
        self.args = .{
            try self.rig.input(.int32, &.{cint(b)}, self.t1),
            try self.rig.input(.int32, &.{ cint(b), cint(r), KTOP }, self.cands),
            try self.rig.input(.float32, &.{ cint(b), cint(r), KTOP, KTOP }, self.probs),
            try self.rig.input(.float32, &.{ cint(b), cint(1 + r) }, self.calib),
            try self.rig.input(.float32, &.{cint(b)}, self.temps),
            try self.rig.input(.uint32, &.{ cint(b), 2 }, seeds),
            try self.rig.input(.int32, &.{ cint(b), MAX_COPY }, self.copy_ids),
            try self.rig.input(.int32, &.{cint(b)}, self.copy_len),
            try self.rig.input(.float32, &.{cint(b)}, self.copy_rate),
            try self.rig.input(.int32, &.{MAX_STOPS}, &self.stops),
            try self.rig.input(.int32, &.{3}, &.{ @as(f64, @floatFromInt(self.n_stops)), @as(f64, @floatFromInt(self.max_ch)), @as(f64, @floatFromInt(self.want_w)) }),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return 4 * treeTok(self.shape) + treeQ(self.shape) + treeDraws(self.shape) + treeTok(self.shape);
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        var ok = true;
        self.lanes(&ok, want, self.errs, null);
        for (mag, self.errs) |*m, e| m.* = host.allowance(e);
    }

    pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const errs = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(errs);
        var ok = true;
        self.lanes(&ok, out, errs, .{ .acc = acc, .order = order });
    }

    /// The stated rule: tokens, parents, depths, ancestor masks and draws exact; each node's next-position q and
    /// score within its f32 and fast-math error, NaN or -Inf where a greedy row's planted probability leaves its
    /// weights unnormalized.
    pub fn check(self: *Self, _: spec.Gate) !?[]const u8 {
        const n = self.count();
        const want = try self.arena.allocator().alloc(f64, n);
        var ok = true;
        self.lanes(&ok, want, self.errs, null);
        const bytes = self.rig.outputBytes();
        for (want, self.errs, 0..) |w, e, i| {
            const o = spec.readOut(tree_read, self.shape, bytes, i);
            const good = if (o.elem == .f32) close(o.v, w, e) else o.v == w;
            if (!good) return fail(&self.note, "output {d}: {d} where the build gives {d}", .{ i, o.v, w });
        }
        return null;
    }
};

// ── registrations ──

const served_commit: []const Shape = &.{ .{ .rows = 1, .n = 8, .k = SW }, .{ .rows = 2, .n = 8, .k = SW }, .{ .rows = 4, .n = 5, .k = SW }, .{ .rows = 8, .n = 6, .k = SW }, .{ .rows = 16, .n = 4, .k = SW } };
const served_accept: []const Shape = &.{ .{ .rows = 1, .n = 8, .k = 1 }, .{ .rows = 2, .n = 8, .k = 1 }, .{ .rows = 4, .n = 5, .k = 1 }, .{ .rows = 8, .n = 6, .k = 1 }, .{ .rows = 16, .n = 4, .k = 1 } };
// One lane, four lanes on the padded head's stride, sixteen sampled lanes, sixteen greedy lanes (`plugin.treeWidth`).
const served_sample: []const Shape = &.{ .{ .rows = 1, .n = 8, .k = VOCAB }, .{ .rows = 4, .n = 5, .k = HEAD_ROWS }, .{ .rows = 16, .n = 4, .k = VOCAB }, .{ .rows = 16, .n = 6, .k = VOCAB } };
const served_conv: []const Shape = &.{ .{ .rows = 1, .n = BLOCK, .k = HIDDEN }, .{ .rows = 4, .n = BLOCK, .k = HIDDEN }, .{ .rows = 16, .n = BLOCK, .k = HIDDEN } };
// A draft's block rows past the anchor: one lane at width 8, four at 5, sixteen at 6 (`plugin.treeWidth`).
const served_shard: []const Shape = &.{ .{ .rows = 7, .n = HEAD_ROWS, .k = VOCAB }, .{ .rows = 4 * 4, .n = HEAD_ROWS, .k = VOCAB }, .{ .rows = 16 * 5, .n = HEAD_ROWS, .k = VOCAB } };
const served_merge: []const Shape = &.{ .{ .rows = 7, .n = SHARDS, .k = KTOP }, .{ .rows = 4 * 4, .n = SHARDS, .k = KTOP }, .{ .rows = 16 * 5, .n = SHARDS, .k = KTOP } };
const served_edges: []const Shape = &.{ .{ .rows = 1, .n = 7, .k = RANK }, .{ .rows = 2, .n = 7, .k = RANK }, .{ .rows = 4, .n = 4, .k = RANK }, .{ .rows = 16, .n = 5, .k = RANK }, .{ .rows = 16, .n = 3, .k = RANK } };
const served_tree: []const Shape = &.{ .{ .rows = 1, .n = 8, .k = KTOP }, .{ .rows = 2, .n = 8, .k = KTOP }, .{ .rows = 4, .n = 5, .k = KTOP }, .{ .rows = 16, .n = 6, .k = KTOP }, .{ .rows = 16, .n = 4, .k = KTOP } };

const sampled_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "sums of exps: the stress inputs check the fp32 sums within the float64 bound" };

fn sampleSpec(comptime what: []const u8, comptime masked: bool, stage: Stage) spec.Spec {
    return .{
        .name = "inkling2 " ++ what ++ " bf16" ++ (if (masked) " (suppress mask)" else ""),
        // The draw's sums only pick a token: no output value for gate 4 to bound.
        .accum = if (stage == .draw) .none else .f32,
        .depth = host.depthOne,
        .out = if (stage == .draw) .u32 else .f32,
        .nan = .{ .stated = switch (stage) {
            .mass => "a NaN or -Inf logit is never admitted; a sampled shard's +Inf makes its sum exp(Inf - Inf), NaN",
            .search => "a row whose mass is not a number keeps its max alone (its cut the max, its mass 1)",
            .draw => "a NaN or -Inf logit is never drawn; a +Inf one, kept alone, is drawn",
        } },
        .served = served_sample,
        .sums = sampled_sums,
    };
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(CommitConv, .init(a, false), .{
        .name = "inkling2 commit_conv f32",
        .accum = .none,
        .depth = none,
        .out = .f32,
        .replaces = "slice the slot -> take the path's rows (commitPath's CPU arm)",
        .served = served_commit,
    }, null);
    try set.add(AcceptTree, .init(a, false), .{
        .name = "inkling2 accept_tree",
        .accum = .none,
        .depth = none,
        .out = .u32,
        .nan = .{ .stated = "integer tokens carry no NaN: gate 1 plants stop ids in the selections, which end a walk" },
        .served = served_accept,
    }, null);
    inline for (.{ false, true }) |masked| try set.add(Sample, .init(a, .mass, masked, false), sampleSpec("sample_tree_mass", masked, .mass), null);
    inline for (.{ false, true }) |masked| try set.add(Sample, .init(a, .search, masked, false), sampleSpec("sample_tree_search", masked, .search), null);
    inline for (.{ false, true }) |masked| try set.add(Sample, .init(a, .draw, masked, false), sampleSpec("sample_tree_draw", masked, .draw), null);
    const conv_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "two taps of two products: a half sum's miss stays near the bf16 output's rounding" };
    try set.add(Conv(0), .init(a, false), .{ .name = "inkling2 dflash_conv bf16 (prepare)", .accum = .f32, .depth = host.depthOne, .out = .bf16, .served = served_conv, .sums = conv_sums }, null);
    try set.add(Conv(1), .init(a, false), .{ .name = "inkling2 dflash_conv bf16 (finish + residual)", .accum = .f32, .depth = host.depthOne, .out = .bf16, .served = served_conv, .sums = conv_sums }, null);
    const shard_outs: []const spec.Part = &.{ .{ .elem = .u32, .count = shardOuts }, .{ .elem = .f32, .count = shardOuts } };
    const merge_outs: []const spec.Part = &.{ .{ .elem = .u32, .count = mergeOuts }, .{ .elem = .f32, .count = mergeOuts } };
    try set.add(Shard, .init(a, false), .{ .name = "inkling2 dflash_top16_shard bf16", .accum = .none, .depth = none, .out = .f32, .outs = shard_outs, .served = served_shard }, null);
    try set.add(Merge, .init(a, false), .{ .name = "inkling2 dflash_top16_merge", .accum = .none, .depth = none, .out = .f32, .outs = merge_outs, .served = served_merge }, null);
    const edges_spec: spec.Spec = .{
        .name = "inkling2 dflash_edges bf16",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .nan = .{ .stated = "a NaN score, or a position whose scores are all -Inf, gives the uniform 1/K; +Inf caps to the soft cap" },
        .served = served_edges,
        .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the edge's sum passes a tanh and an exp: the stress inputs check the fp32 sums within the float64 bound" },
    };
    try set.add(Edges, .init(a, false), edges_spec, null);
    const tree_spec: spec.Spec = .{
        .name = "inkling2 dflash_tree",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .u32,
        .outs = tree_outs,
        .nan = .{ .stated = "a NaN or -Inf probability is never drawn; its row's weights stay unnormalized" },
        .served = served_tree,
        .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "16 weights a row, then a divide: the stress inputs check the fp32 sums within the float64 bound" },
    };
    try set.add(Tree, .init(a, false), tree_spec, null);

    // Planted controls: each must go red at gate 3.
    try set.add(CommitConv, .init(a, true), .{ .name = "control inkling2 commit_conv reading the slot before its own", .accum = .none, .depth = none, .out = .f32, .replaces = "slice the slot -> take the path's rows (commitPath's CPU arm)", .served = &.{.{ .rows = 2, .n = 6, .k = SW }} }, .shapes);
    try set.add(AcceptTree, .init(a, true), .{ .name = "control inkling2 accept_tree matching the later of two equal children", .accum = .none, .depth = none, .out = .u32, .nan = .{ .stated = "integers" }, .served = &.{.{ .rows = 4, .n = 5, .k = 1 }} }, .shapes);
    var mass_c = sampleSpec("sample_tree_mass", true, .mass);
    mass_c.name = "control inkling2 sample_tree_mass breaking a max's tie toward the higher id";
    mass_c.served = served_sample[0..1];
    try set.add(Sample, .init(a, .mass, true, true), mass_c, .shapes);
    var search_c = sampleSpec("sample_tree_search", false, .search);
    search_c.name = "control inkling2 sample_tree_search keeping one token past its top-k";
    search_c.served = &.{.{ .rows = 8, .n = 5, .k = VOCAB }};
    try set.add(Sample, .init(a, .search, false, true), search_c, .shapes);
    var draw_c = sampleSpec("sample_tree_draw", false, .draw);
    draw_c.name = "control inkling2 sample_tree_draw taking the token after its uniform's";
    draw_c.served = &.{.{ .rows = 8, .n = 5, .k = VOCAB }};
    try set.add(Sample, .init(a, .draw, false, true), draw_c, .shapes);
    try set.add(Conv(1), .init(a, true), .{ .name = "control inkling2 dflash_conv dropping its second tap", .accum = .f32, .depth = host.depthOne, .out = .bf16, .served = served_conv[1..2], .sums = conv_sums }, .shapes);
    try set.add(Shard, .init(a, true), .{ .name = "control inkling2 dflash_top16_shard ranking a tie's higher id first", .accum = .none, .depth = none, .out = .f32, .outs = shard_outs, .served = served_shard[0..1] }, .shapes);
    try set.add(Merge, .init(a, true), .{ .name = "control inkling2 dflash_top16_merge over 15 of its 16 shards", .accum = .none, .depth = none, .out = .f32, .outs = merge_outs, .served = served_merge[0..1] }, .shapes);
    var edges_c = edges_spec;
    edges_c.name = "control inkling2 dflash_edges summing 255 of its 256 rank terms";
    edges_c.served = served_edges[2..3];
    try set.add(Edges, .init(a, true), edges_c, .shapes);
    var tree_c = tree_spec;
    tree_c.name = "control inkling2 dflash_tree drawing a tie's higher index first";
    tree_c.served = served_tree[2..3];
    try set.add(Tree, .init(a, true), tree_c, .shapes);
}

// ── the drift guard: each declared launch is the shipped wrapper's ──

fn ptr(comptime T: type, k: kc.gates.Kernel) *T {
    return @ptrCast(@alignCast(k.ctx));
}

/// Checks this file's declared launches against the shipped wrappers, byte for byte, at served shapes: `items` are
/// `register`'s registrations, in its order. `commit_conv` against `kernels.commit_conv_kernel` launched as
/// `commitPath` launches it (its wrapper is inline there); `dflash_conv`'s stages against `Drafter.prepare` and
/// `finishAdd`; the top 16's shard and merge chained against `topCandidates`; shard, merge, edges and tree chained
/// against `buildTreeFromLogits`; mass, search, draw and accept chained against `sampleAndAcceptTree`.
pub fn checkWrappers(a: std.mem.Allocator, items: []const kc.gates.Kernel) !void {
    const commit = ptr(CommitConv, items[0]);
    for (items[0].spec.served) |shape| {
        try std.testing.expect(try commit.prepare(shape, .random, 7));
        try commit.run();
        const ours = try a.dupe(u8, commit.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try commit.wrapper(), ours);
    }
    try checkConv(a, ptr(Conv(0), items[8]), ptr(Conv(1), items[9]));
    try checkTop(a, ptr(Shard, items[10]), ptr(Merge, items[11]));
    try checkTree(a, ptr(Shard, items[10]), ptr(Merge, items[11]), ptr(Edges, items[12]), ptr(Tree, items[13]));
    inline for (.{ false, true }) |masked| {
        const m: usize = @intFromBool(masked);
        try checkSample(a, ptr(Sample, items[2 + m]), ptr(Sample, items[4 + m]), ptr(Sample, items[6 + m]), ptr(AcceptTree, items[1]));
    }
}

fn checkConv(a: std.mem.Allocator, prep: *Conv(0), fin: *Conv(1)) !void {
    for (served_conv) |shape| {
        // finishAdd: its inputs are the kernel's.
        try std.testing.expect(try fin.prepare(shape, .random, 7));
        try fin.run();
        const ours = try a.dupe(u8, fin.rig.outputBytes());
        defer a.free(ours);
        var g = ops.Graph.init(a, fin.rig.stream());
        defer g.deinit();
        const d = testDrafter(fin.rig.stream(), HIDDEN);
        const conv: dflash2.Conv = .{ .base = fin.args[2], .proj = .{} };
        try std.testing.expectEqualSlices(u8, try fin.rig.bytesOf(&.{try d.finishAdd(&g, &conv, fin.args[0], fin.args[1], fin.args[3])}), ours);

        // prepare: the dynamic kernels are its own projection of the input; the declared launch runs on them.
        try std.testing.expect(try prep.prepare(shape, .random, 7));
        var pg = ops.Graph.init(a, prep.rig.stream());
        defer pg.deinit();
        const G = HIDDEN / CONV_GS;
        var prng = std.Random.DefaultPrng.init(11);
        const w = try host.uniform(a, prng.random(), 2 * CONV_KT * G * HIDDEN, -0.05, 0.05, .float32);
        defer a.free(w);
        const w32 = try a.alloc(f32, w.len);
        defer a.free(w32);
        for (w32, w) |*o, x| o.* = @floatCast(x);
        const pconv: dflash2.Conv = .{ .base = prep.args[2], .proj = .{ .w = try pg.astype(try pg.fromSlice(f32, w32, &.{ 2 * CONV_KT * G, HIDDEN }), .bfloat16) } };
        const pd = testDrafter(prep.rig.stream(), HIDDEN);
        const got = try pd.prepare(&pg, &pconv, prep.args[0]);
        const theirs = try a.dupe(u8, try prep.rig.bytesOf(&.{got.out}));
        defer a.free(theirs);
        try prep.rig.launch(prep.launch, prep.geometry(shape).?, &.{ prep.args[0], got.kernel, prep.args[2], prep.args[0] });
        try std.testing.expectEqualSlices(u8, theirs, prep.rig.outputBytes());
    }
}

fn checkTop(a: std.mem.Allocator, shard: *Shard, merge: *Merge) !void {
    for (served_shard) |shape| {
        try std.testing.expect(try shard.prepare(shape, .random, 7));
        try shard.run();
        try merge.rig.launch(merge.launch, mergeGeometry(shape.rows, SHARDS), &.{ shard.rig.output(0), shard.rig.output(1) });
        const ours = try a.dupe(u8, merge.rig.outputBytes());
        defer a.free(ours);
        var g = ops.Graph.init(a, shard.rig.stream());
        defer g.deinit();
        const d = testDrafter(shard.rig.stream(), 64);
        const top = try d.topCandidates(&g, try g.reshape(shard.args[0], &.{ 1, cint(shape.rows), cint(shape.n) }), cint(shape.k));
        try std.testing.expectEqualSlices(u8, try shard.rig.bytesOf(&top), ours);
    }
}

fn checkTree(a: std.mem.Allocator, shard: *Shard, merge: *Merge, edges: *Edges, tree: *Tree) !void {
    const b = 4;
    const w = 5;
    const r = comptime rowsOf(w);
    const h = 64;
    const v = TABLE;
    const stride = TABLE + 76;
    var prng = std.Random.DefaultPrng.init(0x7ee);
    const rng = prng.random();
    const s = shard.rig.stream();
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const randoms = struct {
        fn make(gr: *ops.Graph, al: std.mem.Allocator, rn: std.Random, n: usize, scale: f32, sh: []const c_int) !mlx.mlx_array {
            const buf = try al.alloc(f32, n);
            defer al.free(buf);
            for (buf) |*x| x.* = rn.floatNorm(f32) * scale;
            return gr.astype(try gr.fromSlice(f32, buf, sh), .bfloat16);
        }
    };
    var d = testDrafter(s, h);
    d.pred = try randoms.make(&g, a, rng, v * RANK, 0.3, &.{ v, RANK });
    d.succ = try randoms.make(&g, a, rng, v * RANK, 0.3, &.{ v, RANK });
    d.sel = .{ .w = try randoms.make(&g, a, rng, RANK * h, 0.2, &.{ RANK, h }) };
    const rows = try randoms.make(&g, a, rng, b * r * h, 0.5, &.{ b, r, h });
    const logits = try randoms.make(&g, a, rng, b * r * stride, 2.0, &.{ b, r, stride });
    const copy = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    const lanes = [b]dflash2.Drafter.TreeLane{
        .{ .t1 = 5, .anchor = .{}, .taps = .{}, .position = 10, .cache = null, .temperature = 0, .seed = 1, .copy = &copy, .copy_rate = 0.9 },
        .{ .t1 = 9, .anchor = .{}, .taps = .{}, .position = 10, .cache = null, .temperature = 0.8, .seed = 42 },
        .{ .t1 = 33, .anchor = .{}, .taps = .{}, .position = 12, .cache = null, .temperature = 0, .seed = 3 },
        .{ .t1 = 70, .anchor = .{}, .taps = .{}, .position = 14, .cache = null, .temperature = 1.0, .seed = 77 },
    };
    const stops = [_]u32{2};
    const tb = try d.buildTreeFromLogits(&g, &lanes, rows, logits, v, w, MAX_CH, &stops);
    const theirs = try a.dupe(u8, try tree.rig.bytesOf(&.{ tb.tokens, tb.parent, tb.tmeta, tb.next_q, tb.draws, tb.scores }));
    defer a.free(theirs);

    // The same chain through the declared launches, the host arrays built as `buildTreeFromLogits` builds them.
    try shard.rig.launch(shard.launch, shardGeometry(b * r, stride, v), &.{try g.reshape(logits, &.{ b * r, stride })});
    try merge.rig.launch(merge.launch, mergeGeometry(b * r, SHARDS), &.{ shard.rig.output(0), shard.rig.output(1) });
    const cands = try g.reshape(merge.rig.output(0), &.{ b, r, KTOP });
    const unary = try g.reshape(merge.rig.output(1), &.{ b, r, KTOP });
    var t1: [b]i32 = undefined;
    var calib: [b * (1 + r)]f32 = undefined;
    var temps: [b]f32 = undefined;
    var seeds: [2 * b]u32 = undefined;
    var copy_ids: [b * MAX_COPY]i32 = @splat(0);
    var copy_len: [b]i32 = @splat(0);
    var copy_rate: [b]f32 = @splat(0);
    for (lanes, 0..) |lane, i| {
        t1[i] = @intCast(lane.t1);
        calib[i * (1 + r)] = lane.calib_slope;
        for (0..r) |p| calib[i * (1 + r) + 1 + p] = lane.calib_intercepts[@min(p, lane.calib_intercepts.len - 1)];
        temps[i] = lane.temperature;
        seeds[2 * i] = @truncate(lane.seed);
        seeds[2 * i + 1] = @truncate(lane.seed >> 32);
        const cn = @min(lane.copy.len, MAX_COPY);
        copy_len[i] = @intCast(cn);
        copy_rate[i] = lane.copy_rate;
        for (0..cn) |j| copy_ids[i * MAX_COPY + j] = @intCast(lane.copy[j]);
    }
    var stop_buf: [MAX_STOPS]i32 = @splat(-1);
    stop_buf[0] = @intCast(stops[0]);
    const t1_arr = try g.fromSlice(i32, &t1, &.{b});
    const proj = try project(&g, &d.sel, rows);
    try edges.rig.launch(edges.launch, edgesGeometry(b, r, RANK), &.{ t1_arr, cands, unary, proj, d.pred, d.succ, try g.fromSlice(f32, &[_]f32{@floatCast(SOFTCAP)}, &.{1}) });
    try tree.rig.launch(tree.launch, treeGeometry(b, w, r), &.{
        t1_arr,
        cands,
        edges.rig.output(0),
        try g.fromSlice(f32, &calib, &.{ b, 1 + r }),
        try g.fromSlice(f32, &temps, &.{b}),
        try g.fromSlice(u32, &seeds, &.{ b, 2 }),
        try g.fromSlice(i32, &copy_ids, &.{ b, MAX_COPY }),
        try g.fromSlice(i32, &copy_len, &.{b}),
        try g.fromSlice(f32, &copy_rate, &.{b}),
        try g.fromSlice(i32, &stop_buf, &.{MAX_STOPS}),
        try g.fromSlice(i32, &.{ 1, MAX_CH, w }, &.{3}),
    });
    try std.testing.expectEqualSlices(u8, theirs, tree.rig.outputBytes());
}

fn checkSample(a: std.mem.Allocator, mass: *Sample, search: *Sample, draw: *Sample, accept: *AcceptTree) !void {
    // The draw's inputs: its rows, laws, seeds and draft tree; the accept's: a tree over the same lanes.
    const shape = served_sample[1];
    try std.testing.expect(try draw.prepare(shape, .random, 7));
    try std.testing.expect(try accept.prepare(.{ .rows = shape.rows, .n = shape.n, .k = 1 }, .random, 7));
    const rows = &draw.rows;
    const n = rows.n();
    const s = draw.rig.stream();
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const logits = draw.args[0];
    const mask = draw.args[1];
    var lanes: [16]kernels.SampleLane = undefined;
    draw.wrapperLanes(lanes[0..rows.b], accept.caps);
    var stop_u: [2]u32 = undefined;
    for (&stop_u, accept.stops[0..2]) |*o, x| o.* = tokU(x);
    const r = draw.tree.r;
    const tree_in: kernels.SampleTreeIn = .{ .tmeta = draw.args[5], .cands = draw.args[6], .next_q = draw.args[7], .draws = draw.args[8], .r = cint(r), .k = KTOP };
    const result = try kernels.sampleAndAcceptTree(&g, logits, cint(rows.v), if (draw.masked) mask else .{}, accept.args[0], accept.args[1], lanes[0..rows.b], cint(rows.w), &stop_u, tree_in);
    const theirs = try a.dupe(u8, try draw.rig.bytesOf(&.{result}));
    defer a.free(theirs);

    // The declared launches chained, LPARAMS and SEEDS built as `sampleAndAcceptTree` builds them.
    var lp: [16 * 4]f32 = undefined;
    var seeds: [16 * 2]u32 = undefined;
    var caps: [16]u32 = undefined;
    for (lanes[0..rows.b], 0..) |l, i| {
        lp[i * 4 ..][0..4].* = .{ l.temperature, @floatFromInt(l.top_k), l.top_p, l.min_p };
        seeds[2 * i] = @truncate(l.seed);
        seeds[2 * i + 1] = @truncate(l.seed >> 32);
        caps[i] = l.accept_cap;
    }
    const lparams = try g.fromSlice(f32, lp[0 .. 4 * rows.b], &.{ cint(rows.b), 4 });
    const supp = if (draw.masked) mask else try g.zeros(&.{1}, .bool_);
    try mass.rig.launch(mass.launch, massGeometry(n, rows.w, rows.v, rows.stride, draw.masked), &.{ logits, supp, lparams });
    try search.rig.launch(search.launch, searchGeometry(n, rows.w, rows.v, rows.stride, draw.masked), &.{ logits, supp, lparams, mass.rig.output(0) });
    try draw.rig.launch(draw.launch, drawGeometry(n, rows.w, rows.v, rows.stride, draw.masked, r), &.{ logits, supp, lparams, search.rig.output(0), try g.fromSlice(u32, seeds[0 .. 2 * rows.b], &.{ cint(rows.b), 2 }), draw.args[5], draw.args[6], draw.args[7], draw.args[8] });
    var stop_buf: [MAX_STOPS]i32 = @splat(-1);
    for (stop_buf[0..2], stop_u) |*o, x| o.* = @intCast(x);
    try accept.rig.launch(accept.launch, acceptGeometry(rows.b, rows.w), &.{
        accept.args[0],
        accept.args[1],
        draw.rig.output(0),
        try g.fromSlice(u32, caps[0..rows.b], &.{cint(rows.b)}),
        try g.fromSlice(i32, &stop_buf, &.{MAX_STOPS}),
        try g.fromSlice(u32, &.{2}, &.{1}),
    });
    try std.testing.expectEqualSlices(u8, theirs, accept.rig.outputBytes());
}
