//! The fused QSA sparse attention at verify widths (2..15 tokens) in the kernel contract: a split-K pass over each
//! token's own selected blocks and tail (`msv_attn_qsa256_dsplit` on a bf16 cache, `msv_attn_qsa256_qsplit` on
//! the 8- and 4-bit affine cache, unpacked in the kernel), then `msv_attn_qsa256_qmerge`. The split pass is held to
//! float64 per (token, head, split) on the kernel's own running max: the max within the scores' bound, l and the
//! unnormalized output within each weight's score, exp2 and rescale errors and the sums. The quantized cache's
//! values are the kernel's own bf16 dequant (an exact product, one f32 add, one bf16 store), reproduced on the host.
//! The merge is held to float64 on its inputs. Chained, the launches are `qsaSparseAttn`'s bytes. Split: rows =
//! tokens, n = kv, k = the selection width; merge: rows = tokens, n = 256, k = splits. Flash Next's heads.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const host = @import("../../sdk/testing/kernel_contract/host.zig");
const sparse = @import("qsa/sparse_attn.zig");
const gather = @import("qsa/gather.zig");
const DenseKVView = @import("../../sdk/blocks/attention/kv_view.zig").DenseKVView;

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

const BD = 256;
const HQ = 24;
const HK = 2;
const GQA = HQ / HK;
const NSG = (GQA + 7) / 8;
const RATIO = 4;
const GS = 64;
const SCALE: f32 = 1.0 / 16.0;
const LOG2E: f64 = 1.44269504088896340736;
const SENTINEL: f64 = std.math.maxInt(i32);
const BK: usize = sparse.QSA_ATTN_BK;
const NS: usize = sparse.QSA_ATTN_NSPLIT_DEFAULT;
const U: f64 = 0x1p-24;
/// exp2's relative error, priced at 16 f32 ulps.
const EXP2_ERR: f64 = 0x1p-20;
/// A score's chain: the 256 products' f32 sums in any order, then the f32 scale (two roundings and the constant's).
const SCORE_DEPTH = BD + 3;
/// Rows of the cache past kv: the views the kernels read are strided.
const SPARE = 64;

const V8 = @Vector(8, f64);

fn lanes(x: []const f64, i: usize) V8 {
    return x[8 * i ..][0..8].*;
}

/// The cache: bf16, or the affine scheme at 8 or 4 bits (group 64).
const Kv = enum { dense, q8, q4 };

fn bitsOf(kv: Kv) u32 {
    return switch (kv) {
        .dense => 16,
        .q8 => 8,
        .q4 => 4,
    };
}

const split_tg = (BK + 8) * BD * 2;

const split_launches = [_]metal.Launch{
    .{ .name = "msv_attn_qsa256_dsplit", .inputs = &.{ "q", "scl", "blocks", "kq", "vq" }, .outputs = &.{ "pacc", "pml" }, .source = sparse.QSA_ATTN_D_SOURCE, .header = gather.ATTN_QSA256_KERNEL_HEADER, .contiguous = false, .tg_bytes = split_tg, .m1 = 1024, .m2 = 1024 },
    .{ .name = "msv_attn_qsa256_qsplit", .inputs = &.{ "q", "scl", "blocks", "kq", "ksc", "kbi", "vq", "vsc", "vbi" }, .outputs = &.{ "pacc", "pml" }, .source = sparse.QSA_ATTN_Q_SOURCE, .header = gather.ATTN_QSA256_KERNEL_HEADER, .contiguous = false, .tg_bytes = split_tg, .m1 = 1024, .m2 = 1024 },
};

const slots_launch: metal.Launch = .{
    .name = "msv_attn_qsa256_qsplit_slots",
    .inputs = &(.{ "q", "scl", "blocks" } ++ slotNames() ++ .{"meta"}),
    .outputs = &.{ "pacc", "pml" },
    .source = sparse.QSA_ATTN_SLOTS_SOURCE,
    .header = gather.ATTN_QSA256_KERNEL_HEADER,
    .contiguous = false,
    .tg_bytes = split_tg,
    .m1 = 1024,
    .m2 = 1024,
};

/// Each slot's six cache arrays, slot-major.
fn slotNames() [24][*:0]const u8 {
    var nm: [24][*:0]const u8 = undefined;
    for (0..24) |i| nm[i] = std.fmt.comptimePrint("{s}{d}", .{ ([_][]const u8{ "kq", "ksc", "kbi", "vq", "vsc", "vbi" })[i % 6], i / 6 });
    return nm;
}

const merge_launch: metal.Launch = .{ .name = "msv_attn_qsa256_qmerge", .inputs = &.{ "pacc", "pml" }, .outputs = &.{"out"}, .source = sparse.QSA_ATTN_MERGE_SOURCE, .header = gather.ATTN_QSA256_KERNEL_HEADER, .contiguous = false, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// One K or V cache as the kernels bind it: the values the attention reads (bf16, or the dequantized codes), and
/// for the affine scheme its codes, scales and biases.
const Cache = struct {
    vals: []f64 = &.{},
    codes: []u8 = &.{},
    scales: []f64 = &.{},
    biases: []f64 = &.{},
};

const Split = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kv: Kv,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    q: []f64 = &.{},
    k: Cache = .{},
    v: Cache = .{},
    blocks: []f64 = &.{},
    /// Token s's visible cache rows are `rows[off[s]..off[s + 1]]`; head h's scores of them (log2 units) and their
    /// bounds sit at `h * off[tokens] + off[s]` in `t` and `dt`.
    off: []usize = &.{},
    rows: []usize = &.{},
    t: []f64 = &.{},
    dt: []f64 = &.{},
    args: [9]mlx.mlx_array = undefined,
    n_args: usize = 0,
    views: [6]mlx.mlx_array = @splat(.{ .ctx = null }),
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator, kv: Kv) Split {
        return .{ .rig = metal.Rig.init(a), .launch = &split_launches[@intFromBool(kv != .dense)], .kv = kv, .arena = .init(a) };
    }

    pub fn deinit(self: *Split) void {
        self.dropViews();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn dropViews(self: *Split) void {
        for (&self.views) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    fn cap(self: *const Split) usize {
        return self.shape.n + SPARE;
    }

    /// The wrapper's split pass: one threadgroup a (token, kv head, split), NSG simdgroups of 8 heads.
    pub fn geometry(self: *const Split, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.rows > 15 or shape.n < shape.rows or shape.k < 1 or shape.rows * shape.k < 8) return null;
        const s: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ s * 32, HK * NSG, NS }, .tg = .{ 32, NSG, 1 } };
        g.out(&.{ HQ, s, NS, BD }, .float32);
        g.out(&.{ HQ, s, NS, 2 }, .float32);
        g.dtype("T", .bfloat16);
        g.int("NSG", NSG);
        g.int("BK", @as(c_int, BK));
        g.int("RATIO", RATIO);
        if (self.kv != .dense) {
            g.int("BITS", bitsOf(self.kv));
            g.int("GS", GS);
        }
        g.int("NSPLIT", @as(c_int, NS));
        return g;
    }

    fn visibleRows(self: *const Split, s: usize) []const usize {
        return self.rows[self.off[s]..self.off[s + 1]];
    }

    /// A cache of random bf16 values, or of random codes under random bf16 scales and biases (each group's values
    /// near [-2, 2]); its values are what the kernel reads.
    fn fillCache(self: *Split, a: std.mem.Allocator, rng: std.Random, c: *Cache) !void {
        const n = HK * self.cap() * BD;
        c.vals = try a.alloc(f64, n);
        if (self.kv == .dense) {
            for (c.vals) |*x| x.* = host.rt(.bfloat16, rng.floatNorm(f64));
            return;
        }
        const top: f64 = @floatFromInt((@as(u32, 1) << @intCast(bitsOf(self.kv))) - 1);
        c.codes = try a.alloc(u8, n);
        c.scales = try a.alloc(f64, n / GS);
        c.biases = try a.alloc(f64, n / GS);
        for (c.codes) |*x| x.* = @intCast(rng.uintAtMost(u32, @intFromFloat(top)));
        for (c.scales, c.biases) |*sc, *b| {
            sc.* = host.rt(.bfloat16, (4 / top) * (0.5 + rng.float(f64)));
            b.* = host.rt(.bfloat16, -2 + rng.float(f64));
        }
    }

    /// The kernel's dequant: code x scale (exact in f32) plus the bias in f32, stored as bf16.
    fn dequant(c: *Cache) void {
        if (c.codes.len == 0) return;
        for (c.vals, c.codes, 0..) |*x, q, i| {
            const f: f32 = @floatCast(@as(f64, @floatFromInt(q)) * c.scales[i / GS] + c.biases[i / GS]);
            x.* = host.rt(.bfloat16, f);
        }
    }

    /// Random q and caches, each token a sorted random selection of its complete blocks (all of them below the
    /// width), INT_MAX after. Stress: q 0 (every score 0, every weight 1), every token the first blocks, and each
    /// 16-key tile of V's first 2048 rows mirrored down every column (8 big values, then their negatives with a
    /// small offset): any split, whole tiles, sums to a small integer past fp16 and off bf16's grid, fp32 exact in
    /// any order; the rows after hold 3. Non-finite: a NaN in head 0's q at token 0, a +Inf in kv head 1's first key
    /// token 0 sees, a -Inf in kv head 0's value at token 0's own key (the bf16 cache; the affine one gets a NaN
    /// scale on K and a +Inf bias on V instead).
    pub fn prepare(self: *Split, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        try self.fillHost(shape, fill, seed);
        self.rig.reset();
        self.dropViews();
        self.args[0] = try self.rig.input(.bfloat16, &.{ 1, HQ, @intCast(shape.rows), BD }, self.q);
        self.args[1] = try self.rig.input(.float32, &.{1}, &.{SCALE});
        self.args[2] = try self.rig.input(.int32, &.{ 1, @intCast(shape.rows), @intCast(shape.k) }, self.blocks);
        self.n_args = 3 + try self.bindCaches(&self.rig, &self.views, self.args[3..]);
        return true;
    }

    /// The host operands at `shape` (the scores and their bounds included), nothing bound.
    fn fillHost(self: *Split, shape: Shape, fill: spec.Fill, seed: u64) !void {
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.q = try a.alloc(f64, HQ * shape.rows * BD);
        for (self.q) |*x| x.* = host.rt(.bfloat16, rng.floatNorm(f64));
        try self.fillCache(a, rng, &self.k);
        try self.fillCache(a, rng, &self.v);
        self.blocks = try a.alloc(f64, shape.rows * shape.k);
        @memset(self.blocks, SENTINEL);
        const pool = try a.alloc(usize, shape.n / RATIO + 1);
        self.off = try a.alloc(usize, shape.rows + 1);
        var total: usize = 0;
        for (0..shape.rows) |s| {
            const p = shape.n - shape.rows + s;
            const complete = (p + 1) / RATIO;
            for (pool[0..complete], 0..) |*b, i| b.* = i;
            if (fill != .stress) rng.shuffle(usize, pool[0..complete]);
            const picks = @min(complete, shape.k);
            const row = self.blocks[s * shape.k ..][0..picks];
            for (row, pool[0..picks]) |*x, b| x.* = @floatFromInt(b);
            std.mem.sort(f64, row, {}, std.sort.asc(f64));
            self.off[s] = total;
            total += picks * RATIO + p + 1 - complete * RATIO;
        }
        self.off[shape.rows] = total;
        self.rows = try a.alloc(usize, total);
        for (0..shape.rows) |s| {
            const p = shape.n - shape.rows + s;
            const complete = (p + 1) / RATIO;
            const sel = @min(complete, shape.k) * RATIO;
            for (self.rows[self.off[s]..self.off[s + 1]], 0..) |*r, vi| {
                r.* = if (vi < sel) @as(usize, @intFromFloat(self.blocks[s * shape.k + vi / RATIO])) * RATIO + vi % RATIO else complete * RATIO + vi - sel;
            }
        }
        if (fill == .stress) {
            @memset(self.q, 0);
            try self.stressV(rng);
        }
        if (fill == .nonfinite) {
            const own = self.visibleRows(0);
            self.q[3] = std.math.nan(f64);
            const kat = (1 * self.cap() + own[0]) * BD + 7;
            const vat = (0 * self.cap() + own[own.len - 1]) * BD + 11;
            if (self.kv == .dense) {
                self.k.vals[kat] = std.math.inf(f64);
                self.v.vals[vat] = -std.math.inf(f64);
            } else {
                self.k.scales[kat / GS] = std.math.nan(f64);
                self.v.biases[vat / GS] = std.math.inf(f64);
            }
        }
        dequant(&self.k);
        dequant(&self.v);
        self.t = try a.alloc(f64, HQ * total);
        self.dt = try a.alloc(f64, HQ * total);
        const c = @as(f64, SCALE) * LOG2E;
        for (0..HQ) |h| for (0..shape.rows) |s| {
            const qr = self.q[(h * shape.rows + s) * BD ..][0..BD];
            for (self.visibleRows(s), h * total + self.off[s]..) |r, i| {
                const kr = self.k.vals[((h / GQA) * self.cap() + r) * BD ..][0..BD];
                var d: V8 = @splat(0);
                var m: V8 = @splat(0);
                for (0..BD / 8) |j| {
                    const x = lanes(qr, j) * lanes(kr, j);
                    d += x;
                    m += @abs(x);
                }
                self.t[i] = @reduce(.Add, d) * c;
                self.dt[i] = spec.gamma(SCORE_DEPTH, U) * @reduce(.Add, m) * c;
            }
        };
    }

    /// V's stress columns: the bf16 values directly, or codes the scheme's scale and bias turn into them.
    fn stressV(self: *Split, rng: std.Random) !void {
        const tiles = @min(2048, self.cap()) / BK;
        const half = BK / 2;
        const bits = bitsOf(self.kv);
        // A big value is 64 (128 + c) (8-bit codes c < 128) or 512 (16 + c) (4-bit): eight of them overflow fp16.
        const step: f64 = if (bits == 4) 512 else 64;
        const base: f64 = 8192;
        for (0..HK) |g| {
            const row0 = g * self.cap();
            for (0..self.cap()) |r| {
                const off: f64 = if (r < tiles * BK and r % BK >= half and (r % BK - half) % 5 == 0) 64 else 0;
                const big = r < tiles * BK and r % BK < half;
                const neg = r < tiles * BK and !big;
                if (self.kv != .dense) for (0..BD / GS) |grp| {
                    const gi = (row0 + r) * (BD / GS) + grp;
                    self.v.scales[gi] = if (big) step else if (neg) -step else 0;
                    self.v.biases[gi] = if (big) base else if (neg) -base + off else 3;
                };
                for (0..BD) |d| {
                    const i = (row0 + r) * BD + d;
                    if (neg) {
                        // The partner row's code: the value's negative.
                        if (self.kv == .dense) self.v.vals[i] = -self.v.vals[i - half * BD] + off else self.v.codes[i] = self.v.codes[i - half * BD];
                        continue;
                    }
                    const code: u8 = @intCast(rng.uintLessThan(u32, if (bits == 4) 16 else 128));
                    if (self.kv == .dense) {
                        self.v.vals[i] = if (big) base + step * @as(f64, @floatFromInt(code)) else 3;
                    } else self.v.codes[i] = code;
                }
            }
        }
    }

    /// K and V (bf16, or each one's codes, scales and biases) through `rig`, as kv-row views of their capacity;
    /// `views` owns the views, `out` gets them in binding order. Returns how many.
    fn bindCaches(self: *Split, rig: *metal.Rig, views: []mlx.mlx_array, out: []mlx.mlx_array) !usize {
        const st = rig.stream();
        const capc: c_int = @intCast(self.cap());
        const kvn: c_int = @intCast(self.shape.n);
        var full: [6]mlx.mlx_array = undefined;
        var width: [6]c_int = undefined;
        var n: usize = 0;
        for ([_]*Cache{ &self.k, &self.v }) |c| {
            if (self.kv == .dense) {
                full[n] = try rig.input(.bfloat16, &.{ 1, HK, capc, BD }, c.vals);
                width[n] = BD;
                n += 1;
                continue;
            }
            const bits = bitsOf(self.kv);
            const wpr: c_int = @intCast(BD * bits / 32);
            const words = try host.packAffine(self.arena.allocator(), c.codes, HK * self.cap(), BD, bits);
            full[n] = try rig.input(.uint32, &.{ 1, HK, capc, wpr }, words);
            full[n + 1] = try rig.input(.bfloat16, &.{ 1, HK, capc, BD / GS }, c.scales);
            full[n + 2] = try rig.input(.bfloat16, &.{ 1, HK, capc, BD / GS }, c.biases);
            width[n..][0..3].* = .{ wpr, BD / GS, BD / GS };
            n += 3;
        }
        for (full[0..n], width[0..n], views[0..n], out[0..n]) |f, w, *view, *arg| {
            view.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(view, f, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, HK, kvn, w }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, st));
            arg.* = view.*;
        }
        return n;
    }

    pub fn run(self: *Split) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Split) usize {
        return HQ * self.shape.rows * NS * (BD + 2);
    }

    /// Split j of token s: its virtual keys [lo, hi) (empty when lo >= hi), whole BK tiles but the last.
    fn range(self: *const Split, s: usize, j: usize) [2]usize {
        const L = self.off[s + 1] - self.off[s];
        const per = ((L + BK - 1) / BK + NS - 1) / NS * BK;
        return .{ j * per, @min(L, j * per + per) };
    }

    fn vrow(self: *const Split, h: usize, r: usize) []const f64 {
        return self.v.vals[((h / GQA) * self.cap() + r) * BD ..][0..BD];
    }

    /// Exact outputs, every split's max taken exactly and its sums in `acc` (null: float64): gate 4.
    fn exact(self: *Split, acc: ?spec.Accum, order: spec.Order, out: []f64) !void {
        const a = self.rig.a;
        const total = self.off[self.shape.rows];
        const n_acc = HQ * self.shape.rows * NS * BD;
        const buf = try a.alloc(f64, 2 * BK * NS);
        defer a.free(buf);
        for (0..HQ) |h| for (0..self.shape.rows) |s| for (0..NS) |j| {
            const row = (h * self.shape.rows + s) * NS + j;
            const r = self.range(s, j);
            const o = out[row * BD ..][0..BD];
            if (r[0] >= r[1]) {
                @memset(o, 0);
                out[n_acc + 2 * row] = -std.math.inf(f64);
                out[n_acc + 2 * row + 1] = 0;
                continue;
            }
            const t = self.t[h * total + self.off[s] ..][r[0]..r[1]];
            const rows = self.visibleRows(s)[r[0]..r[1]];
            const p = buf[0..t.len];
            const terms = buf[t.len..][0..t.len];
            var m: f64 = -std.math.inf(f64);
            for (t) |x| m = if (std.math.isNan(x) or std.math.isNan(m)) std.math.nan(f64) else @max(m, x);
            for (p, t) |*w, x| w.* = std.math.exp2(x - m);
            for (o, 0..) |*y, d| {
                for (terms, p, rows) |*x, w, kr| x.* = w * self.vrow(h, kr)[d];
                y.* = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms);
            }
            out[n_acc + 2 * row] = m;
            out[n_acc + 2 * row + 1] = if (acc) |ac| spec.sumIn(ac, order, p) else host.sum(p);
        };
    }

    pub fn reference(self: *Split, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        try self.exact(null, .sequential, want);
    }

    pub fn simulate(self: *Split, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.exact(acc, order, out);
    }

    /// Per (token, head, split): m within the scores' bound of their max (NaN with a NaN score); then l and the
    /// output on the kernel's own m, each weight 2^(t - m) off by its score's bound, the subtractions against the
    /// running max (which climbs by at most the split's score range), each exp2 and rescale, and the sums.
    pub fn check(self: *Split, at: spec.Gate) !?[]const u8 {
        return self.checkBytes(self.rig.outputBytes(), at);
    }

    /// `check` on `out`, this split pass's outputs (acc, then the m and l pairs).
    fn checkBytes(self: *Split, out: []const u8, at: spec.Gate) !?[]const u8 {
        const total = self.off[self.shape.rows];
        const n_acc = HQ * self.shape.rows * NS * BD;
        var got: [BD + 2]f64 = undefined;
        var want: [BD + 2]f64 = undefined;
        var allow: [BD + 2]f64 = undefined;
        var planted: usize = 0;
        for (0..HQ) |h| for (0..self.shape.rows) |s| for (0..NS) |j| {
            const row = (h * self.shape.rows + s) * NS + j;
            for (got[0..BD], 0..) |*g, d| g.* = spec.read(.f32, out, row * BD + d);
            got[BD] = spec.read(.f32, out, n_acc + 2 * row);
            got[BD + 1] = spec.read(.f32, out, n_acc + 2 * row + 1);
            const r = self.range(s, j);
            if (r[0] >= r[1]) {
                // An empty split: the merge's identity.
                @memset(&want, 0);
                want[BD] = -std.math.inf(f64);
                @memset(&allow, 0);
            } else {
                const base = h * total + self.off[s];
                const t = self.t[base..][r[0]..r[1]];
                const dt = self.dt[base..][r[0]..r[1]];
                const rows = self.visibleRows(s)[r[0]..r[1]];
                var hi: f64 = -std.math.inf(f64);
                var lo: f64 = std.math.inf(f64);
                var dmax: f64 = 0;
                var nan = false;
                for (t, dt) |x, e| {
                    if (std.math.isNan(x)) nan = true;
                    hi = @max(hi, x);
                    lo = @min(lo, x);
                    dmax = @max(dmax, e);
                }
                want[BD] = if (nan) std.math.nan(f64) else hi;
                allow[BD] = dmax;
                const mk = got[BD];
                const tiles: f64 = @floatFromInt((t.len + BK - 1) / BK);
                const eta = std.math.exp2(2 * U * (hi - lo + 2 * dmax)) * std.math.pow(f64, 1 + EXP2_ERR, tiles + 1) * std.math.pow(f64, 1 + U, tiles) - 1;
                const g = spec.gamma(@as(f64, @floatFromInt(t.len)) + 2 * tiles + 2, U);
                var acc: [BD / 8]V8 = @splat(@splat(0));
                var err: [BD / 8]V8 = @splat(@splat(0));
                var mag: [BD / 8]V8 = @splat(@splat(0));
                var vmx: [BD / 8]V8 = @splat(@splat(0));
                var l: f64 = 0;
                var le: f64 = 0;
                for (t, dt, rows) |x, e, kr| {
                    const p = std.math.exp2(x - mk);
                    const delta = std.math.exp2(e) * (1 + eta) - 1;
                    l += p;
                    le += p * delta;
                    const v = self.vrow(h, kr);
                    const pv: V8 = @splat(p);
                    const pd: V8 = @splat(p * delta);
                    for (&acc, &err, &mag, &vmx, 0..) |*y, *ye, *ym, *ymx, i| {
                        const x8 = lanes(v, i);
                        y.* += pv * x8;
                        ye.* += pd * @abs(x8);
                        ym.* += pv * @abs(x8);
                        ymx.* = @max(ymx.*, @abs(x8));
                    }
                }
                // A weight exp2 flushes to zero is off by up to 2^-126.
                const flushed = 0x1p-126 * @as(f64, @floatFromInt(t.len));
                for (acc, err, mag, vmx, 0..) |y, ye, ym, ymx, i| {
                    want[8 * i ..][0..8].* = y;
                    allow[8 * i ..][0..8].* = ye + @as(V8, @splat(g)) * (ym + ye) + @as(V8, @splat(flushed)) * ymx + @as(V8, @splat(0x1p-149));
                }
                want[BD + 1] = l;
                allow[BD + 1] = le + g * (l + le) + flushed + 0x1p-149;
            }
            for (want) |x| {
                if (spec.classOf(x) != .finite) planted += 1;
            }
            var why: [200]u8 = undefined;
            // An empty split's -Inf max is its flag, not a planted value.
            if (r[0] >= r[1]) planted -= 1;
            if (host.within(&why, if (at == .nonfinite) .repeat else at, &got, &want, &allow)) |w| {
                return std.fmt.bufPrint(&self.msg, "token {d} head {d} split {d}: {s}", .{ s, h, j, w }) catch &self.msg;
            }
        };
        if (at == .nonfinite and planted == 0) return "the planted values leave every output finite: no proof";
        return null;
    }

    /// The split and merge launches chained, and `qsaSparseAttn` on the same operands: ours and its bytes.
    fn chained(self: *Split) ![2][]const u8 {
        const a = self.rig.a;
        try self.run();
        var pa = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pa);
        var pm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pm);
        try mlx.check(mlx.mlx_array_set(&pa, self.rig.output(0)));
        try mlx.check(mlx.mlx_array_set(&pm, self.rig.output(1)));
        try self.rig.launch(&merge_launch, Merge.geometryAt(.{ .rows = self.shape.rows, .n = BD, .k = NS }).?, &.{ pa, pm });
        const ours = try a.dupe(u8, self.rig.outputBytes());
        errdefer a.free(ours);
        var view: DenseKVView = .{ .k = self.args[3], .v = self.args[4], .owned = false };
        if (self.kv != .dense) view = .{
            .k = .{ .ctx = null },
            .v = .{ .ctx = null },
            .owned = false,
            .k_triple_q = self.args[3],
            .k_triple_scales = self.args[4],
            .k_triple_biases = self.args[5],
            .v_triple_q = self.args[6],
            .v_triple_scales = self.args[7],
            .v_triple_biases = self.args[8],
            .has_quant_triple = true,
            .bits = @intCast(bitsOf(self.kv)),
            .group_size = GS,
            .kv_cap = @intCast(self.cap()),
        };
        const y = (try sparse.qsaSparseAttn(self.rig.stream(), self.args[0], &view, self.args[2], RATIO, SCALE)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return .{ ours, try self.rig.bytesOf(&.{y}) };
    }
};

/// The batched decode step's split pass: 2..4 slots of one token, each over its own affine cache (rows = slots,
/// n = slot 0's kv, each later slot 37 keys longer). Each slot is held to `Split`'s check on its share of the
/// outputs.
const Slots = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &slots_launch,
    kv: Kv,
    shape: Shape = undefined,
    slot: [4]Split,
    args: [28]mlx.mlx_array = undefined,
    views: [24]mlx.mlx_array = @splat(.{ .ctx = null }),
    bytes: std.ArrayList(u8) = .empty,
    msg: [300]u8 = undefined,

    fn init(a: std.mem.Allocator, kv: Kv) Slots {
        var s: Slots = .{ .rig = metal.Rig.init(a), .kv = kv, .slot = undefined };
        for (&s.slot) |*x| x.* = Split.init(a, kv);
        return s;
    }

    pub fn deinit(self: *Slots) void {
        for (&self.views) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
        for (&self.slot) |*x| x.deinit();
        self.bytes.deinit(self.rig.a);
        self.rig.deinit();
    }

    fn slotShape(shape: Shape, i: usize) Shape {
        return .{ .rows = 1, .n = shape.n + 37 * i, .k = shape.rows };
    }

    pub fn geometry(self: *const Slots, shape: Shape) ?metal.Geometry {
        // The block lists bind as one array: under 8 elements MLX would pass it by value.
        if (self.kv == .dense or shape.k < 2 or shape.k > 4 or shape.rows < 1 or shape.rows * shape.k < 8 or shape.n < 1) return null;
        const n: c_int = @intCast(shape.k);
        var g = metal.Geometry{ .grid = .{ 32, HK * NSG, n * @as(c_int, NS) }, .tg = .{ 32, NSG, 1 } };
        g.out(&.{ n * HQ, 1, @as(c_int, NS), BD }, .float32);
        g.out(&.{ n * HQ, 1, @as(c_int, NS), 2 }, .float32);
        g.dtype("T", .bfloat16);
        g.int("NSG", NSG);
        g.int("BK", @as(c_int, BK));
        g.int("RATIO", RATIO);
        g.int("BITS", bitsOf(self.kv));
        g.int("GS", GS);
        g.int("NSPLIT", @as(c_int, NS));
        return g;
    }

    /// Each slot filled as `Split` fills one token; q and the block lists bound stacked, each slot's six arrays
    /// after them (an absent slot repeats slot 0's), then `meta`: Hq, Hk, the width, each slot's kv and capacity.
    pub fn prepare(self: *Slots, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        const a = self.rig.a;
        const n = shape.k;
        for (self.slot[0..n], 0..) |*x, i| try x.fillHost(slotShape(shape, i), fill, seed +% i);
        self.rig.reset();
        for (&self.views) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
        const q = try a.alloc(f64, n * HQ * BD);
        defer a.free(q);
        const blocks = try a.alloc(f64, n * shape.rows);
        defer a.free(blocks);
        var meta: [11]f64 = @splat(0);
        meta[0..3].* = .{ HQ, HK, @floatFromInt(shape.rows) };
        for (self.slot[0..n], 0..) |*x, i| {
            @memcpy(q[i * HQ * BD ..][0 .. HQ * BD], x.q);
            @memcpy(blocks[i * shape.rows ..][0..shape.rows], x.blocks);
            _ = try x.bindCaches(&self.rig, self.views[6 * i ..][0..6], self.args[3 + 6 * i ..][0..6]);
            meta[3 + 2 * i ..][0..2].* = .{ @floatFromInt(x.shape.n), @floatFromInt(x.cap()) };
        }
        for (n..4) |i| self.args[3 + 6 * i ..][0..6].* = self.args[3..9].*;
        self.args[0] = try self.rig.input(.bfloat16, &.{ @intCast(n), HQ, 1, BD }, q);
        self.args[1] = try self.rig.input(.float32, &.{1}, &.{SCALE});
        self.args[2] = try self.rig.input(.int32, &.{ @intCast(n), 1, @intCast(shape.rows) }, blocks);
        self.args[27] = try self.rig.input(.int32, &.{meta.len}, &meta);
        return true;
    }

    pub fn run(self: *Slots) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    fn per(_: *const Slots) [2]usize {
        return .{ HQ * NS * BD, HQ * NS * 2 };
    }

    pub fn count(self: *const Slots) usize {
        const p = self.per();
        return self.shape.k * (p[0] + p[1]);
    }

    /// Each slot's outputs from `exact`, scattered to the stacked layout (every slot's acc, then every slot's pairs).
    fn outputs(self: *Slots, acc: ?spec.Accum, order: spec.Order, out: []f64) !void {
        const p = self.per();
        const n = self.shape.k;
        const buf = try self.rig.a.alloc(f64, p[0] + p[1]);
        defer self.rig.a.free(buf);
        for (self.slot[0..n], 0..) |*x, i| {
            try x.exact(acc, order, buf);
            @memcpy(out[i * p[0] ..][0..p[0]], buf[0..p[0]]);
            @memcpy(out[n * p[0] + i * p[1] ..][0..p[1]], buf[p[0]..]);
        }
    }

    pub fn reference(self: *Slots, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        try self.outputs(null, .sequential, want);
    }

    pub fn simulate(self: *Slots, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputs(acc, order, out);
    }

    pub fn check(self: *Slots, at: spec.Gate) !?[]const u8 {
        const p = self.per();
        const n = self.shape.k;
        const out = self.rig.outputBytes();
        for (self.slot[0..n], 0..) |*x, i| {
            self.bytes.clearRetainingCapacity();
            try self.bytes.appendSlice(self.rig.a, out[4 * i * p[0] ..][0 .. 4 * p[0]]);
            try self.bytes.appendSlice(self.rig.a, out[4 * (n * p[0] + i * p[1]) ..][0 .. 4 * p[1]]);
            if (try x.checkBytes(self.bytes.items, at)) |why| return std.fmt.bufPrint(&self.msg, "slot {d}: {s}", .{ i, why }) catch why;
        }
        return null;
    }

    /// The slot launch and the merge chained, and `qsaSparseAttnFloor` over the same slots: ours and its bytes.
    fn chained(self: *Slots) ![2][]const u8 {
        const a = self.rig.a;
        try self.run();
        var pa = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pa);
        var pm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pm);
        try mlx.check(mlx.mlx_array_set(&pa, self.rig.output(0)));
        try mlx.check(mlx.mlx_array_set(&pm, self.rig.output(1)));
        const n: c_int = @intCast(self.shape.k);
        var g = metal.Geometry{ .grid = .{ BD, 1, n * HQ }, .tg = .{ BD, 1, 1 } };
        g.out(&.{ n, HQ, 1, BD }, .bfloat16);
        g.dtype("T", .bfloat16);
        try self.rig.launch(&merge_launch, g, &.{ pa, pm });
        const ours = try a.dupe(u8, self.rig.outputBytes());
        errdefer a.free(ours);
        var views: [4]DenseKVView = undefined;
        var blocks: [4]mlx.mlx_array = undefined;
        const st = self.rig.stream();
        defer for (blocks[0..self.shape.k]) |b| {
            _ = mlx.mlx_array_free(b);
        };
        for (self.slot[0..self.shape.k], 0..) |*x, i| {
            const t = self.args[3 + 6 * i ..][0..6];
            views[i] = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .k_triple_q = t[0], .k_triple_scales = t[1], .k_triple_biases = t[2], .v_triple_q = t[3], .v_triple_scales = t[4], .v_triple_biases = t[5], .has_quant_triple = true, .bits = @intCast(bitsOf(self.kv)), .group_size = GS, .kv_cap = @intCast(x.cap()) };
            blocks[i] = mlx.mlx_array_new();
            const k: c_int = @intCast(self.shape.rows);
            const ii: c_int = @intCast(i);
            try mlx.check(mlx.mlx_slice(&blocks[i], self.args[2], &[_]c_int{ ii, 0, 0 }, 3, &[_]c_int{ ii + 1, 1, k }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
        }
        const y = (try sparse.qsaSparseAttnFloor(st, self.args[0], views[0..self.shape.k], blocks[0..self.shape.k], RATIO, SCALE, 1)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return .{ ours, try self.rig.bytesOf(&.{y}) };
    }
};

const Merge = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &merge_launch,
    shape: Shape = undefined,
    acc: []f64 = &.{},
    ml: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator) Merge {
        return .{ .rig = metal.Rig.init(a) };
    }

    pub fn deinit(self: *Merge) void {
        self.rig.a.free(self.acc);
        self.rig.a.free(self.ml);
        self.rig.deinit();
    }

    /// One 256-thread group a (q head, token).
    fn geometryAt(shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.n != BD or shape.k < 1 or shape.k > 64) return null;
        const s: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ BD, s, HQ }, .tg = .{ BD, 1, 1 } };
        g.out(&.{ 1, HQ, s, BD }, .bfloat16);
        g.dtype("T", .bfloat16);
        return g;
    }

    pub fn geometry(_: *const Merge, shape: Shape) ?metal.Geometry {
        return geometryAt(shape);
    }

    fn rowsOf(self: *const Merge) usize {
        return HQ * self.shape.rows;
    }

    /// Splits with a max in [-4, 24] and a denominator in [1, 64], their outputs a denominator's worth of values in
    /// [-1, 1); each row's last split empty (m -Inf, l 0). Stress: equal maxes and unit denominators over integer
    /// outputs that telescope. Non-finite: a NaN, +Inf and -Inf output in rows 0..2 and a NaN denominator in row 3.
    pub fn prepare(self: *Merge, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        const ns = shape.k;
        const nr = self.rowsOf();
        a.free(self.acc);
        a.free(self.ml);
        self.acc = try a.alloc(f64, nr * ns * BD);
        self.ml = try a.alloc(f64, nr * ns * 2);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..nr) |h| for (0..ns) |s| {
            const r = h * ns + s;
            const empty = s + 1 == ns and ns > 1;
            const l: f64 = if (fill == .stress) 1 else host.rt(.float32, 1 + 63 * rng.float(f64));
            self.ml[2 * r] = if (empty) -std.math.inf(f64) else if (fill == .stress) 0 else host.rt(.float32, -4 + 28 * rng.float(f64));
            self.ml[2 * r + 1] = if (empty) 0 else l;
            for (self.acc[r * BD ..][0..BD]) |*x| x.* = if (empty) 0 else host.rt(.float32, l * (2 * rng.float(f64) - 1));
        };
        if (fill == .stress) {
            // Down each column's splits: B_i + r_i, then -B_(i+1); the column telescopes to the small sum of the r,
            // every B overflows fp16 alone and every partial sum stays under 2^24.
            const used = ns - @intFromBool(ns > 1);
            const half = used / 2;
            var bs: [64]f64 = undefined;
            for (0..nr) |h| for (0..BD) |d| {
                for (bs[0..half]) |*b| b.* = @floatFromInt(65536 + rng.uintLessThan(u32, 65536));
                for (0..used) |s| self.acc[(h * ns + s) * BD + d] = if (s < half) bs[s] + @as(f64, @floatFromInt(1 + rng.uintLessThan(u32, 15))) else if (s < 2 * half) -bs[(s - half + 1) % half] else 3;
            };
        }
        if (fill == .nonfinite) {
            self.acc[(0 * ns) * BD + 5] = std.math.nan(f64);
            self.acc[(1 * ns) * BD + 6] = std.math.inf(f64);
            self.acc[(2 * ns) * BD + 7] = -std.math.inf(f64);
            if (nr > 3) self.ml[2 * (3 * ns) + 1] = std.math.nan(f64);
        }
        self.rig.reset();
        const s: c_int = @intCast(shape.rows);
        const cns: c_int = @intCast(ns);
        self.args = .{ try self.rig.input(.float32, &.{ HQ, s, cns, BD }, self.acc), try self.rig.input(.float32, &.{ HQ, s, cns, 2 }, self.ml) };
        return true;
    }

    pub fn run(self: *Merge) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Merge) usize {
        return self.rowsOf() * BD;
    }

    /// Row h's weights 2^(m - M) over its non-empty splits (a NaN l is a split like any other), and Z.
    fn weights(self: *const Merge, h: usize, w: []f64) f64 {
        const ns = self.shape.k;
        var M: f64 = -std.math.inf(f64);
        for (0..ns) |s| {
            if (!(self.ml[2 * (h * ns + s) + 1] == 0)) M = @max(M, self.ml[2 * (h * ns + s)]);
        }
        var z: f64 = 0;
        for (w, 0..) |*x, s| {
            const l = self.ml[2 * (h * ns + s) + 1];
            x.* = if (l == 0) 0 else std.math.exp2(self.ml[2 * (h * ns + s)] - M);
            z += x.* * l;
        }
        return z;
    }

    /// The float64 output on sums run in `acc` (null: exact), with each output's allowance.
    fn outputs(self: *const Merge, acc: ?spec.Accum, order: spec.Order, want: []f64, allow: []f64) !void {
        const a = self.rig.a;
        const ns = self.shape.k;
        const w = try a.alloc(f64, ns);
        defer a.free(w);
        const t = try a.alloc(f64, ns);
        defer a.free(t);
        for (0..self.rowsOf()) |h| {
            const z = self.weights(h, w);
            for (0..BD) |d| {
                var mag: f64 = 0;
                for (t, w, 0..) |*x, wj, s| {
                    x.* = wj * self.acc[(h * ns + s) * BD + d];
                    mag += @abs(x.*);
                }
                const y = (if (acc) |ac| spec.sumIn(ac, order, t) else host.sum(t)) / z;
                want[h * BD + d] = y;
                // exp2 on each weight, the column's and Z's sequential sums, the division and the bf16 output.
                const rel = 2 * EXP2_ERR + spec.gamma(@floatFromInt(2 * ns + 4), U);
                const work = rel * (mag / @abs(z) + @abs(y));
                allow[h * BD + d] = work + 0x1p-8 * (@abs(y) + work) + 0x1p-133;
            }
        }
    }

    pub fn reference(self: *Merge, want: []f64, mag: []f64) !void {
        try self.outputs(null, .sequential, want, mag);
    }

    pub fn simulate(self: *Merge, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const scratch = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(scratch);
        try self.outputs(acc, order, out, scratch);
    }

    pub fn check(self: *Merge, at: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const n = self.count();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        try self.outputs(null, .sequential, buf[0..n], buf[n .. 2 * n]);
        const out = self.rig.outputBytes();
        for (buf[2 * n ..], 0..) |*g, i| g.* = spec.read(.bf16, out, i);
        return host.within(&self.msg, at, buf[2 * n ..], buf[0..n], buf[n .. 2 * n]);
    }
};

fn noDepth(_: Shape) usize {
    return 0;
}

fn depthMerge(shape: Shape) usize {
    return 2 * shape.k + 4;
}

const nan_split: spec.Known = .{ .gate = .nonfinite, .finding = "a split's running max goes through metal::max, which drops a NaN score: a split holding one reports a finite m (-3e38 when every score is NaN) where float64 gives NaN (l and acc carry the NaN)" };

/// Verify steps of 8 and 2 tokens just past the budget and deep in a long cache (full 512-block selections), 15 on
/// a short cache (every complete block selected); each token's tail 0..3 keys.
const served: []const Shape = &.{ .{ .rows = 8, .n = 2100, .k = 512 }, .{ .rows = 2, .n = 9000, .k = 512 }, .{ .rows = 15, .n = 700, .k = 512 } };

pub fn register(set: *kc.Set) !void {
    inline for (.{ Kv.dense, Kv.q8, Kv.q4 }) |kv| {
        try set.add(Split, .init(set.a, kv), .{
            .name = "qsa_attn split " ++ switch (kv) {
                .dense => "bf16 cache",
                .q8 => "8-bit cache (g64)",
                .q4 => "4-bit cache (g64)",
            } ++ " (verify widths, 16 splits)",
            .accum = .f32,
            .depth = noDepth,
            .out = .f32,
            .served = served,
            .known = nan_split,
        }, null);
    }
    inline for (.{ Kv.q8, Kv.q4 }) |kv| {
        try set.add(Slots, .init(set.a, kv), .{
            .name = "qsa_attn split slots " ++ (if (kv == .q8) "8-bit" else "4-bit") ++ " cache (g64) (batched decode, 16 splits)",
            .accum = .f32,
            .depth = noDepth,
            .out = .f32,
            // Two, three and four slots of one token, just past the budget and deep in a long cache.
            .served = &.{ .{ .rows = 512, .n = 2100, .k = 2 }, .{ .rows = 512, .n = 9000, .k = 3 }, .{ .rows = 512, .n = 2052, .k = 4 } },
            .known = nan_split,
        }, null);
    }
    try set.add(Merge, .init(set.a), .{
        .name = "qsa_attn merge (16 splits)",
        .accum = .f32,
        .depth = depthMerge,
        .out = .bf16,
        .served = &.{ .{ .rows = 8, .n = BD, .k = NS }, .{ .rows = 15, .n = BD, .k = NS } },
        .tiles = .{ .{}, .{}, .{ .size = 16, .step = 1, .min = 1 } },
    }, null);
}

// ── tests ──

test "kernel contract qwen4_exp: the qsa_attn launches, chained over strided cache views, are qsaSparseAttn's bytes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    inline for (.{ Kv.dense, Kv.q8, Kv.q4 }) |kv| {
        var p = Split.init(a, kv);
        defer p.deinit();
        for (served) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, shape.n));
            const got = try p.chained();
            defer a.free(got[0]);
            try std.testing.expectEqualSlices(u8, got[1], got[0]);
        }
    }
    inline for (.{ Kv.q8, Kv.q4 }) |kv| {
        var p = Slots.init(a, kv);
        defer p.deinit();
        for ([_]usize{ 2, 4 }) |n| {
            try std.testing.expect(try p.prepare(.{ .rows = 512, .n = 2100, .k = n }, .random, n));
            const got = try p.chained();
            defer a.free(got[0]);
            try std.testing.expectEqualSlices(u8, got[1], got[0]);
        }
    }
}
