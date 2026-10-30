//! qsa_decode's two launches in the kernel contract. The split pass gives each (head, split of 32 visible keys) its
//! running max m (log2 units), denominator l and unnormalized output acc; the merge pass rescales the splits and
//! divides. Parts (rows = q heads per kv head, n = kv, k = the 256 head dims) is held to float64 per split on the
//! kernel's own m: |m - max score| within the scores' bound, l and acc within each term's exp2 and sum errors. Merge
//! (rows = q heads, n = 256, k = splits) is held to float64 on its inputs. Chained, the launches are
//! `qsa_decode.attend`'s bytes, over a strided cache view as the decode step binds it.

const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const kc = @import("../../../testing/kernel_contract/root.zig");
const host = @import("../../../testing/kernel_contract/host.zig");
const qsa_decode = @import("decode.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

const BD = 256;
const CH = 32;
const HK = 2;
const KB = 512;
const RATIO = 4;
const SCALE: f32 = 1.0 / 16.0;
const LOG2E: f64 = 1.44269504088896340736;
/// A score's chain: q times the f32 scale, the lane's two fma chains of four and their sum, the five-stage butterfly.
const SCORE_DEPTH = 12;
/// exp2's relative error, priced at 16 f32 ulps.
const EXP2_ERR: f64 = 0x1p-20;
/// An empty split's max as the kernel writes it (f32).
const EMPTY_M: f64 = @as(f32, -3.0e38);
/// Rows of the cache past kv: the view the kernels read is strided.
const SPARE = 64;

const parts_launch: metal.Launch = .{
    .name = "msv_qsa_dec_parts",
    .inputs = &.{ "q", "scl", "blocks", "k", "v" },
    .outputs = &.{ "pacc", "pml" },
    .source = qsa_decode.PARTS_SOURCE,
    .header = qsa_decode.HEADER,
    .contiguous = false,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

const merge_launch: metal.Launch = .{
    .name = "msv_qsa_dec_merge",
    .inputs = &.{ "pacc", "pml" },
    .outputs = &.{"out"},
    .source = qsa_decode.MERGE_SOURCE,
    .header = qsa_decode.HEADER,
    .contiguous = false,
    // NS floats and the denominator, at the longest served cache.
    .tg_bytes = 4 * 512 + 4,
    .m1 = 1024,
    .m2 = 1024,
};

/// Splits for `kb` picks of RATIO keys and the incomplete tail, as the wrapper counts them.
fn splits(kb: usize) usize {
    return (kb * RATIO + RATIO - 1 + CH - 1) / CH;
}

/// The decode step's operands: q, a strided K/V cache view and the indexer's sorted picks.
const Operands = struct {
    hq: usize = 0,
    kv: usize = 0,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    picks: []f64 = &.{},
    views: [2]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null } },

    fn deinit(o: *Operands, a: std.mem.Allocator) void {
        for ([_][]f64{ o.q, o.k, o.v, o.picks }) |x| a.free(x);
        o.dropViews();
        o.* = .{};
    }

    fn dropViews(o: *Operands) void {
        for (&o.views) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    fn cap(o: *const Operands) usize {
        return o.kv + SPARE;
    }

    /// Random bf16 operands; under `.nonfinite` head 0's q holds a NaN, one visible key of kv head 1 a +Inf and one
    /// value of kv head 0 a -Inf. `.stress`: q 1 and K 0 (every score exactly 0, every weight 1), V mirrored integers
    /// down each split's 32 keys (big values a split's partial sums carry past fp16 and off bf16's grid).
    fn fill(o: *Operands, a: std.mem.Allocator, hq: usize, kv: usize, fill_kind: spec.Fill, seed: u64) !void {
        o.deinit(a);
        o.hq = hq;
        o.kv = kv;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        o.q = try a.alloc(f64, hq * BD);
        o.k = try a.alloc(f64, HK * o.cap() * BD);
        o.v = try a.alloc(f64, HK * o.cap() * BD);
        for (o.q) |*x| x.* = host.rt(.bfloat16, rng.floatNorm(f64));
        for (o.k) |*x| x.* = host.rt(.bfloat16, rng.floatNorm(f64));
        for (o.v) |*x| x.* = host.rt(.bfloat16, rng.floatNorm(f64));
        const nb = kv / RATIO;
        const pool = try a.alloc(usize, nb);
        defer a.free(pool);
        for (pool, 0..) |*p, i| p.* = i;
        rng.shuffle(usize, pool);
        o.picks = try a.alloc(f64, KB);
        for (o.picks, pool[0..KB]) |*p, b| p.* = @floatFromInt(b);
        std.mem.sort(f64, o.picks, {}, std.sort.asc(f64));
        if (fill_kind == .stress) {
            @memset(o.q, 1);
            @memset(o.k, 0);
            const L = o.visible();
            for (0..HK) |g| for (0..BD) |d| {
                var j: usize = 0;
                while (j < L) : (j += CH) {
                    const n: usize = if (L - j < CH) L - j else CH;
                    for (0..n / 2) |i| {
                        // Sixteen of these overflow fp16 together; they and an offset of 32 are on bf16's grid.
                        const big: f64 = @floatFromInt(32 * (128 + rng.uintLessThan(u32, 128)));
                        o.v[(g * o.cap() + o.pos(j + i)) * BD + d] = big;
                        o.v[(g * o.cap() + o.pos(j + n / 2 + i)) * BD + d] = -big + @as(f64, if (i % 5 == 0) 32 else 0);
                    }
                    if (n % 2 == 1) o.v[(g * o.cap() + o.pos(j + n - 1)) * BD + d] = 3;
                }
            };
        }
        if (fill_kind == .nonfinite) {
            o.q[3] = std.math.nan(f64);
            o.k[(1 * o.cap() + o.pos(40)) * BD + 7] = std.math.inf(f64);
            o.v[(0 * o.cap() + o.pos(900)) * BD + 11] = -std.math.inf(f64);
        }
    }

    fn selLen(o: *const Operands) usize {
        const nb: usize = o.kv / RATIO;
        return (if (nb < KB) nb else KB) * RATIO;
    }

    /// Visible keys: the picks' blocks, then the incomplete tail.
    fn visible(o: *const Operands) usize {
        return o.selLen() + (o.kv - o.kv / RATIO * RATIO);
    }

    /// The cache row of visible key `j`.
    fn pos(o: *const Operands, j: usize) usize {
        const sl = o.selLen();
        if (j < sl) return @as(usize, @intFromFloat(o.picks[j / RATIO])) * RATIO + j % RATIO;
        return o.kv / RATIO * RATIO + (j - sl);
    }

    /// q, scl, blocks, K and V as `attend` binds them, the caches sliced to kv rows of their capacity.
    fn bind(o: *Operands, rig: *metal.Rig) ![5]mlx.mlx_array {
        o.dropViews();
        const s = rig.stream();
        const q = try rig.input(.bfloat16, &.{ 1, @intCast(o.hq), 1, BD }, o.q);
        const scl = try rig.input(.float32, &.{1}, &.{SCALE});
        const blocks = try rig.input(.int32, &.{ 1, 1, KB }, o.picks);
        const full = [2]mlx.mlx_array{ try rig.input(.bfloat16, &.{ 1, HK, @intCast(o.cap()), BD }, o.k), try rig.input(.bfloat16, &.{ 1, HK, @intCast(o.cap()), BD }, o.v) };
        for (&o.views, full) |*view, f| {
            view.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(view, f, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, HK, @intCast(o.kv), BD }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        }
        return .{ q, scl, blocks, o.views[0], o.views[1] };
    }

    /// Head `h`'s score of visible key `j` (log2 units) and its bound.
    fn score(o: *const Operands, h: usize, j: usize) [2]f64 {
        const g = h / (o.hq / HK);
        const row = (g * o.cap() + o.pos(j)) * BD;
        const d = host.dot(o.q[h * BD ..][0..BD], o.k[row..][0..BD]);
        const c = @as(f64, SCALE) * LOG2E;
        return .{ d[0] * c, spec.gamma(SCORE_DEPTH, 0x1p-24) * d[1] * c };
    }
};

pub const Parts = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &parts_launch,
    ops: Operands = .{},
    shape: Shape = undefined,
    args: [5]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator) Parts {
        return .{ .rig = metal.Rig.init(a) };
    }

    pub fn deinit(self: *Parts) void {
        self.ops.deinit(self.rig.a);
        self.rig.deinit();
    }

    pub fn geometry(_: *const Parts, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.rows > 32 or shape.k != BD or shape.n / RATIO < KB) return null;
        const gqa: c_int = @intCast(shape.rows);
        const ns: c_int = @intCast(splits(KB));
        const hq = gqa * HK;
        var g = metal.Geometry{ .grid = .{ ns * gqa * 32, HK, 1 }, .tg = .{ gqa * 32, 1, 1 } };
        g.out(&.{ hq, ns, BD }, .float32);
        g.out(&.{ hq, ns, 2 }, .float32);
        g.dtype("T", .bfloat16);
        g.int("GQA", gqa);
        g.int("KB", KB);
        g.int("RATIO", RATIO);
        g.int("CH", CH);
        g.int("NS", ns);
        return g;
    }

    pub fn prepare(self: *Parts, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        self.rig.reset();
        try self.ops.fill(self.rig.a, shape.rows * HK, shape.n, fill, seed);
        self.args = try self.ops.bind(&self.rig);
        return true;
    }

    pub fn run(self: *Parts) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Parts) usize {
        return self.ops.hq * splits(KB) * (BD + 2);
    }

    /// The outputs with each split's max taken exactly and its sums run in `acc` (null: float64): gate 4, whose
    /// stress scores are all 0.
    fn exact(self: *Parts, acc: ?spec.Accum, order: spec.Order, out: []f64) !void {
        const a = self.rig.a;
        const o = &self.ops;
        const ns = splits(KB);
        const n_acc = o.hq * ns * BD;
        const L = o.visible();
        const t = try a.alloc(f64, CH);
        defer a.free(t);
        const p = try a.alloc(f64, CH);
        defer a.free(p);
        for (0..o.hq) |h| for (0..ns) |s| {
            const row = h * ns + s;
            const lo = s * CH;
            const hi = @min(lo + CH, L);
            if (lo >= hi) {
                @memset(out[row * BD ..][0..BD], 0);
                out[n_acc + 2 * row] = EMPTY_M;
                out[n_acc + 2 * row + 1] = 0;
                continue;
            }
            const terms = hi - lo;
            var m: f64 = -std.math.inf(f64);
            for (lo..hi) |j| m = @max(m, o.score(h, j)[0]);
            for (p[0..terms], lo..hi) |*x, j| x.* = std.math.exp2(o.score(h, j)[0] - m);
            const g = (h / (o.hq / HK)) * o.cap();
            for (0..BD) |d| {
                for (t[0..terms], p[0..terms], lo..hi) |*x, w, j| x.* = w * o.v[(g + o.pos(j)) * BD + d];
                out[row * BD + d] = if (acc) |ac| spec.sumIn(ac, order, t[0..terms]) else host.sum(t[0..terms]);
            }
            out[n_acc + 2 * row] = m;
            out[n_acc + 2 * row + 1] = if (acc) |ac| spec.sumIn(ac, order, p[0..terms]) else host.sum(p[0..terms]);
        };
    }

    pub fn reference(self: *Parts, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        try self.exact(null, .sequential, want);
    }

    pub fn simulate(self: *Parts, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.exact(acc, order, out);
    }

    /// Per (head, split): m within the scores' bound of their max (a NaN score makes it NaN); then l and acc on the
    /// kernel's own m, each term 2^(score - m) priced with its score's bound, exp2 and the rescales, and the sums.
    pub fn check(self: *Parts, at: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const o = &self.ops;
        const ns = splits(KB);
        const out = self.rig.outputBytes();
        const n_acc = o.hq * ns * BD;
        const L = o.visible();
        const got = try a.alloc(f64, BD + 2);
        defer a.free(got);
        const want = try a.alloc(f64, 3 * (BD + 2));
        defer a.free(want);
        const sc = try a.alloc(f64, 2 * CH);
        defer a.free(sc);
        var planted: usize = 0;
        for (0..o.hq) |h| for (0..ns) |s| {
            const row = h * ns + s;
            for (0..BD) |d| got[d] = spec.read(.f32, out, row * BD + d);
            got[BD] = spec.read(.f32, out, n_acc + 2 * row);
            got[BD + 1] = spec.read(.f32, out, n_acc + 2 * row + 1);
            const lo = s * CH;
            const hi = @min(lo + CH, L);
            const w = want[0 .. BD + 2];
            const allow = want[BD + 2 .. 2 * (BD + 2)];
            if (lo >= hi) {
                // An empty split: no key, the skip flag.
                @memset(w, 0);
                w[BD] = EMPTY_M;
                @memset(allow, 0);
            } else {
                var m: f64 = -std.math.inf(f64);
                var dm: f64 = 0;
                var nan = false;
                for (lo..hi, 0..) |j, i| {
                    const x = o.score(h, j);
                    sc[2 * i] = x[0];
                    sc[2 * i + 1] = x[1];
                    if (std.math.isNan(x[0])) nan = true else m = @max(m, x[0]);
                    dm = @max(dm, x[1]);
                }
                w[BD] = if (nan) std.math.nan(f64) else m;
                allow[BD] = dm;
                const mk = got[BD];
                const g = (h / (o.hq / HK)) * o.cap();
                const terms = hi - lo;
                var l: f64 = 0;
                var lm: f64 = 0;
                for (0..terms) |i| {
                    const p = std.math.exp2(sc[2 * i] - mk);
                    l += p;
                    lm += p * (std.math.exp2(sc[2 * i + 1]) - 1 + 6 * EXP2_ERR);
                }
                w[BD + 1] = l;
                allow[BD + 1] = lm + spec.gamma(@floatFromInt(terms + 8), 0x1p-24) * @abs(l) + 0x1p-149;
                for (0..BD) |d| {
                    var acc: f64 = 0;
                    var am: f64 = 0;
                    var mag: f64 = 0;
                    for (0..terms) |i| {
                        const p = std.math.exp2(sc[2 * i] - mk);
                        const v = o.v[(g + o.pos(lo + i)) * BD + d];
                        acc += p * v;
                        am += @abs(p * v) * (std.math.exp2(sc[2 * i + 1]) - 1 + 6 * EXP2_ERR);
                        mag += @abs(p * v);
                    }
                    w[d] = acc;
                    allow[d] = am + spec.gamma(@floatFromInt(terms + 8), 0x1p-24) * mag + 0x1p-149;
                }
            }
            for (w) |x| {
                if (spec.classOf(x) != .finite) planted += 1;
            }
            var why_buf: [200]u8 = undefined;
            if (host.within(&why_buf, if (at == .nonfinite) .repeat else at, got, w, allow)) |why| {
                return std.fmt.bufPrint(&self.msg, "head {d} split {d}: {s}", .{ h, s, why }) catch &self.msg;
            }
        };
        if (at == .nonfinite and planted == 0) return "the planted values leave every output finite: no proof";
        return null;
    }

    /// Both launches through the declared ones and `qsa_decode.attend` on the same operands: ours and its bytes.
    fn chained(self: *Parts) ![2][]const u8 {
        const a = self.rig.a;
        try self.run();
        var pa = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pa);
        var pm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pm);
        try mlx.check(mlx.mlx_array_set(&pa, self.rig.output(0)));
        try mlx.check(mlx.mlx_array_set(&pm, self.rig.output(1)));
        try self.rig.launch(&merge_launch, Merge.geometryAt(.{ .rows = self.ops.hq, .n = BD, .k = splits(KB) }).?, &.{ pa, pm });
        const ours = try a.dupe(u8, self.rig.outputBytes());
        errdefer a.free(ours);
        const y = (try qsa_decode.attend(self.rig.stream(), self.args[0], self.args[3], self.args[4], self.args[2], RATIO, SCALE)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return .{ ours, try self.rig.bytesOf(&.{y}) };
    }
};

pub const Merge = struct {
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

    fn geometryAt(shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.n != BD or shape.k < 1 or shape.k > 512) return null;
        const hq: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ BD * hq, 1, 1 }, .tg = .{ BD, 1, 1 } };
        g.out(&.{ 1, hq, 1, BD }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("NS", shape.k);
        return g;
    }

    pub fn geometry(_: *const Merge, shape: Shape) ?metal.Geometry {
        return geometryAt(shape);
    }

    /// Splits with a max in [-4, 24] and a denominator in [1, 64], their outputs a denominator's worth of values in
    /// [-1, 1); every head's last split empty (m -3e38, l 0). Stress: equal maxes and unit denominators over integer
    /// outputs that telescope. Non-finite: a NaN, +Inf and -Inf output in heads 0..2 and a NaN denominator in head 3.
    pub fn prepare(self: *Merge, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        const ns = shape.k;
        a.free(self.acc);
        a.free(self.ml);
        self.acc = try a.alloc(f64, shape.rows * ns * BD);
        self.ml = try a.alloc(f64, shape.rows * ns * 2);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.rows) |h| for (0..ns) |s| {
            const r = h * ns + s;
            const empty = s + 1 == ns and ns > 1;
            const l: f64 = if (fill == .stress) 1 else host.rt(.float32, 1 + 63 * rng.float(f64));
            self.ml[2 * r] = if (empty) -3.0e38 else if (fill == .stress) 0 else host.rt(.float32, -4 + 28 * rng.float(f64));
            self.ml[2 * r + 1] = if (empty) 0 else l;
            for (self.acc[r * BD ..][0..BD]) |*x| x.* = if (empty) 0 else host.rt(.float32, l * (2 * rng.float(f64) - 1));
        };
        if (fill == .stress) {
            // Down each column's splits: B_i + r_i, then -B_(i+1). No two terms in one strided lane or one half
            // cancel, the column telescopes to the small sum of the r, every B overflows fp16 alone and every
            // partial sum stays under 2^24 (fp32 exact in any order).
            const used = ns - @intFromBool(ns > 1);
            const half = used / 2;
            var bs: [256]f64 = undefined;
            for (0..shape.rows) |h| for (0..BD) |d| {
                for (bs[0..half]) |*b| b.* = @floatFromInt(65536 + rng.uintLessThan(u32, 65536));
                for (0..used) |s| self.acc[(h * ns + s) * BD + d] = if (s < half) bs[s] + @as(f64, @floatFromInt(1 + rng.uintLessThan(u32, 15))) else if (s < 2 * half) -bs[(s - half + 1) % half] else 3;
            };
        }
        if (fill == .nonfinite) {
            self.acc[(0 * ns) * BD + 5] = std.math.nan(f64);
            self.acc[(1 * ns) * BD + 6] = std.math.inf(f64);
            self.acc[(2 * ns) * BD + 7] = -std.math.inf(f64);
            if (shape.rows > 3) self.ml[2 * (3 * ns) + 1] = std.math.nan(f64);
        }
        self.rig.reset();
        const hq: c_int = @intCast(shape.rows);
        const cns: c_int = @intCast(ns);
        self.args = .{ try self.rig.input(.float32, &.{ hq, cns, BD }, self.acc), try self.rig.input(.float32, &.{ hq, cns, 2 }, self.ml) };
        return true;
    }

    pub fn run(self: *Merge) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Merge) usize {
        return self.shape.rows * BD;
    }

    /// Head h's weights 2^(m - M) over the splits with l > 0 (a NaN l is a split like any other), and Z.
    fn weights(self: *const Merge, h: usize, w: []f64) f64 {
        const ns = self.shape.k;
        var M: f64 = -std.math.inf(f64);
        for (0..ns) |s| {
            const l = self.ml[2 * (h * ns + s) + 1];
            if (!(l == 0)) M = @max(M, self.ml[2 * (h * ns + s)]);
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
        for (0..self.shape.rows) |h| {
            const z = self.weights(h, w);
            for (0..BD) |d| {
                var mag: f64 = 0;
                for (t, w, 0..) |*x, wj, s| {
                    x.* = wj * self.acc[(h * ns + s) * BD + d];
                    mag += @abs(x.*);
                }
                const num = if (acc) |ac| spec.sumIn(ac, order, t) else host.sum(t);
                const y = num / z;
                want[h * BD + d] = y;
                // exp2 on each weight, the column's and Z's sums, the division and the bf16 output.
                const rel = 2 * EXP2_ERR + spec.gamma(@floatFromInt(ns / 4 + 4), 0x1p-24);
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

fn depthMerge(shape: Shape) usize {
    return shape.k / 4 + 4;
}

fn noDepth(_: Shape) usize {
    return 0;
}

const nan_split: spec.Known = .{ .gate = .nonfinite, .finding = "a split's running max goes through metal::max, which drops a NaN score: a split whose scores are all NaN reports m = -3e38 where float64 gives NaN (l and acc carry the NaN)" };

pub fn register(set: *kc.Set) !void {
    // Flash Next: 24 q heads over 2 kv heads; the cache just past the 2048-key budget, mid and long.
    try set.add(Parts, .init(set.a), .{
        .name = "qsa_decode parts (one token, 512 picks of 4 keys)",
        .accum = .f32,
        .depth = noDepth,
        .out = .f32,
        .nan = .propagate,
        .served = &.{ .{ .rows = 12, .n = 2052, .k = BD }, .{ .rows = 12, .n = 8079, .k = BD }, .{ .rows = 12, .n = 11000, .k = BD } },
        .known = nan_split,
    }, null);
    try set.add(Merge, .init(set.a), .{
        .name = "qsa_decode merge (65 splits, one empty)",
        .accum = .f32,
        .depth = depthMerge,
        .out = .bf16,
        .served = &.{.{ .rows = 24, .n = BD, .k = comptime splits(KB) }},
        .tiles = .{ .{}, .{}, .{ .size = 32, .step = 1, .min = 1 } },
    }, null);
}

// ── tests ──

test "kernel contract sparse_block: the qsa_decode launches, chained over a strided cache view, are attend's bytes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var p = Parts.init(a);
    defer p.deinit();
    for ([_]usize{ 2052, 8079 }) |kv| {
        try std.testing.expect(try p.prepare(.{ .rows = 12, .n = kv, .k = BD }, .random, kv));
        const got = try p.chained();
        defer a.free(got[0]);
        try std.testing.expectEqualSlices(u8, got[1], got[0]);
    }
}
