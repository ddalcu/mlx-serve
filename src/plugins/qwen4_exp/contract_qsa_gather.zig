//! QSA's block-gathered prefill attention in the kernel contract: each query token attends to the keys of its sorted
//! selected blocks, then its own incomplete tail, with an online softmax over BK-key tiles. Two kernels compute it:
//! the SIMD-matrix kernel (`msv_attn_qsa256`) and the NAX one Flash Next's geometry takes on M5
//! (`msv_qsa_nax_precise`, P split in two halves for the tensor unit). Both are held to float64 per (token, head):
//! each weight's score error and the online rescales priced as one relative error per term (the common factor of
//! the running max cancels in the ratio), the sums' gamma, the divide and the bf16 store. Rows = query tokens,
//! n = kv, k = the selection width. Flash Next's heads: 24 q heads over 2 kv heads, hd 256, scale 1/16.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const host = @import("../../sdk/testing/kernel_contract/host.zig");
const gather = @import("qsa/gather.zig");
const device = @import("../../mlx/device.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

const BD = 256;
const HQ = 24;
const HK = 2;
const GQA = HQ / HK;
const RATIO = 4;
const SCALE: f32 = 1.0 / 16.0;
const LOG2E: f64 = 1.44269504088896340736;
const SENTINEL: f64 = std.math.maxInt(i32);
/// A score's chain: the 256 products' f32 sums in any order, then the f32 scale (two roundings and the constant's).
const SCORE_DEPTH = BD + 3;
const U: f64 = 0x1p-24;
/// exp2's relative error, priced at 16 f32 ulps.
const EXP2_ERR: f64 = 0x1p-20;
/// Rows of the cache past kv: the view the kernels read is strided.
const SPARE = 64;
const NSG = (GQA + 7) / 8;
/// Keys a tile (the stock kernel's default BK; NAX's eight blocks a step).
const BK: usize = gather.QSA_GATHER_BK_DEFAULT;

/// The host loops run eight lanes wide: this file's float64 work is a few hundred million terms a shape.
const V8 = @Vector(8, f64);

fn lanes(x: []const f64, i: usize) V8 {
    return x[8 * i ..][0..8].*;
}

/// q . k and sum |q k| over the 256 dims.
fn dot(q: []const f64, k: []const f64) [2]f64 {
    var d: V8 = @splat(0);
    var m: V8 = @splat(0);
    for (0..BD / 8) |i| {
        const x = lanes(q, i) * lanes(k, i);
        d += x;
        m += @abs(x);
    }
    return .{ @reduce(.Add, d), @reduce(.Add, m) };
}

const launches = [_]metal.Launch{
    .{
        .name = "msv_attn_qsa256",
        .inputs = &.{ "q", "k", "v", "scl", "blocks" },
        .outputs = &.{"out"},
        .source = gather.ATTN_QSA256_KERNEL_SOURCE,
        .header = gather.ATTN_QSA256_KERNEL_HEADER,
        .contiguous = false,
        // The K/V tile: BK + 8 rows of 256 bf16.
        .tg_bytes = (BK + 8) * BD * 2,
        .m1 = 1024,
        .m2 = 1024,
    },
    .{
        .name = "msv_qsa_nax_precise",
        .inputs = &.{ "q", "k", "v", "scl", "blocks" },
        .outputs = &.{"out"},
        .source = gather.QSA_NAX_SOURCE,
        .header = gather.QSA_NAX_HEADER,
        .contiguous = false,
        // The two halves' partial score exchange.
        .tg_bytes = 2 * 8 * 32 * 4,
        .offline = false,
        .min_gen = 17,
    },
};

const Gather = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    nax: bool,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    blocks: []f64 = &.{},
    /// Token s's visible cache rows are `rows[off[s]..off[s + 1]]`; head h's scores of them (log2 units) and their
    /// bounds sit at `h * off[tokens] + off[s]` in `t` and `dt`.
    off: []usize = &.{},
    rows: []usize = &.{},
    t: []f64 = &.{},
    dt: []f64 = &.{},
    args: [5]mlx.mlx_array = undefined,
    views: [2]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null } },
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator, nax: bool) Gather {
        return .{ .rig = metal.Rig.init(a), .launch = &launches[@intFromBool(nax)], .nax = nax, .arena = .init(a) };
    }

    pub fn deinit(self: *Gather) void {
        self.dropViews();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn dropViews(self: *Gather) void {
        for (&self.views) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    fn cap(self: *const Gather) usize {
        return self.shape.n + SPARE;
    }

    /// The wrapper's dispatch: one threadgroup a (token, kv head), NSG simdgroups of 8 heads (NAX: the two
    /// 128-dim halves); NAX serves only Flash Next's 512-block selection, from 16 tokens.
    pub fn geometry(self: *const Gather, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.n < shape.rows or shape.k < 1) return null;
        if (self.nax and shape.k != 512) return null;
        const s: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ s * 32, HK * NSG, 1 }, .tg = .{ 32, NSG, 1 } };
        g.out(&.{ 1, HQ, s, BD }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("NSG", NSG);
        g.int("BK", @as(c_int, BK));
        g.int("RATIO", RATIO);
        return g;
    }

    /// Token s (at key position p, bottom-right aligned) sees its `min(complete, k)` selected blocks, then the
    /// keys of its incomplete block up to p.
    fn listRows(self: *Gather, a: std.mem.Allocator) !void {
        const S = self.shape.rows;
        self.off = try a.alloc(usize, S + 1);
        var total: usize = 0;
        for (0..S) |s| {
            const p = self.shape.n - S + s;
            const complete = (p + 1) / RATIO;
            self.off[s] = total;
            total += @min(complete, self.shape.k) * RATIO + p + 1 - complete * RATIO;
        }
        self.off[S] = total;
        self.rows = try a.alloc(usize, total);
        for (0..S) |s| {
            const p = self.shape.n - S + s;
            const complete = (p + 1) / RATIO;
            const sel = @min(complete, self.shape.k) * RATIO;
            for (self.rows[self.off[s]..self.off[s + 1]], 0..) |*r, vi| {
                r.* = if (vi < sel) @as(usize, @intFromFloat(self.blocks[s * self.shape.k + vi / RATIO])) * RATIO + vi % RATIO else complete * RATIO + vi - sel;
            }
        }
    }

    fn visibleRows(self: *const Gather, s: usize) []const usize {
        return self.rows[self.off[s]..self.off[s + 1]];
    }

    /// Random bf16 operands and, per token, a sorted random selection of its complete blocks (all of them below
    /// the width), INT_MAX after. Stress: q 0 (every score 0, every weight 1), every token the first blocks, and V's
    /// first 2048 rows mirrored integers down each column (big values whose partial sums overflow fp16 and leave
    /// bf16's grid, summing to a small integer; fp32 exact in any order). Non-finite: a NaN in head 0's q at
    /// token 0, a +Inf in kv head 1's first key token 0 sees, a -Inf in kv head 0's value at token 0's own key.
    pub fn prepare(self: *Gather, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.q = try a.alloc(f64, HQ * shape.rows * BD);
        self.k = try a.alloc(f64, HK * self.cap() * BD);
        self.v = try a.alloc(f64, HK * self.cap() * BD);
        for ([_][]f64{ self.q, self.k, self.v }) |x| for (x) |*e| {
            e.* = host.rt(.bfloat16, rng.floatNorm(f64));
        };
        self.blocks = try a.alloc(f64, shape.rows * shape.k);
        @memset(self.blocks, SENTINEL);
        const pool = try a.alloc(usize, shape.n / RATIO + 1);
        for (0..shape.rows) |s| {
            const complete = (shape.n - shape.rows + s + 1) / RATIO;
            for (pool[0..complete], 0..) |*b, i| b.* = i;
            if (fill != .stress) rng.shuffle(usize, pool[0..complete]);
            const picks = @min(complete, shape.k);
            const row = self.blocks[s * shape.k ..][0..picks];
            for (row, pool[0..picks]) |*x, b| x.* = @floatFromInt(b);
            std.mem.sort(f64, row, {}, std.sort.asc(f64));
        }
        try self.listRows(a);
        if (fill == .stress) {
            @memset(self.q, 0);
            const half = @min(2048, self.cap()) / 2;
            for (0..HK) |g| for (0..BD) |d| {
                const col = self.v[g * self.cap() * BD + d ..];
                for (0..self.cap()) |r| col[r * BD] = 3;
                for (0..half) |i| {
                    // Sixteen of these overflow fp16 together; they and an offset of 32 are on bf16's grid.
                    const big: f64 = @floatFromInt(32 * (128 + rng.uintLessThan(u32, 128)));
                    col[i * BD] = big;
                    col[(half + i) * BD] = -big + @as(f64, if (i % 5 == 0) 32 else 0);
                }
            };
        }
        if (fill == .nonfinite) {
            const own = self.visibleRows(0);
            self.q[3] = std.math.nan(f64);
            self.k[(1 * self.cap() + own[0]) * BD + 7] = std.math.inf(f64);
            self.v[(0 * self.cap() + own[own.len - 1]) * BD + 11] = -std.math.inf(f64);
        }
        const total = self.off[shape.rows];
        self.t = try a.alloc(f64, HQ * total);
        self.dt = try a.alloc(f64, HQ * total);
        const c = @as(f64, SCALE) * LOG2E;
        for (0..HQ) |h| for (0..shape.rows) |s| {
            const qr = self.q[(h * shape.rows + s) * BD ..][0..BD];
            for (self.visibleRows(s), h * total + self.off[s]..) |r, i| {
                const d = dot(qr, self.k[((h / GQA) * self.cap() + r) * BD ..][0..BD]);
                self.t[i] = d[0] * c;
                self.dt[i] = spec.gamma(SCORE_DEPTH, U) * d[1] * c;
            }
        };
        self.rig.reset();
        self.dropViews();
        const st = self.rig.stream();
        const full = [2]mlx.mlx_array{
            try self.rig.input(.bfloat16, &.{ 1, HK, @intCast(self.cap()), BD }, self.k),
            try self.rig.input(.bfloat16, &.{ 1, HK, @intCast(self.cap()), BD }, self.v),
        };
        for (&self.views, full) |*view, f| {
            view.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(view, f, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, HK, @intCast(shape.n), BD }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, st));
        }
        self.args = .{
            try self.rig.input(.bfloat16, &.{ 1, HQ, @intCast(shape.rows), BD }, self.q),
            self.views[0],
            self.views[1],
            try self.rig.input(.float32, &.{1}, &.{SCALE}),
            try self.rig.input(.int32, &.{ 1, @intCast(shape.rows), @intCast(shape.k) }, self.blocks),
        };
        return true;
    }

    pub fn run(self: *Gather) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Gather) usize {
        return HQ * self.shape.rows * BD;
    }

    fn vrow(self: *const Gather, h: usize, r: usize) []const f64 {
        return self.v[((h / GQA) * self.cap() + r) * BD ..][0..BD];
    }

    /// Head h's scores and their bounds at token s.
    fn scores(self: *const Gather, h: usize, s: usize) [2][]const f64 {
        const i = h * self.off[self.shape.rows] + self.off[s];
        const L = self.off[s + 1] - self.off[s];
        return .{ self.t[i..][0..L], self.dt[i..][0..L] };
    }

    /// Head h's exact weights 2^(t - max t) at token s into `p`, and its output with both sums in `acc` (null:
    /// float64) into `out`.
    fn exact(self: *const Gather, h: usize, s: usize, acc: ?spec.Accum, order: spec.Order, p: []f64, terms: []f64, out: []f64) void {
        const t = self.scores(h, s)[0];
        const rows = self.visibleRows(s);
        var m: f64 = -std.math.inf(f64);
        for (t) |x| m = if (std.math.isNan(x) or std.math.isNan(m)) std.math.nan(f64) else @max(m, x);
        for (p, t) |*w, x| w.* = std.math.exp2(x - m);
        const ac = acc orelse {
            var o: [BD / 8]V8 = @splat(@splat(0));
            for (p, rows) |w, r| {
                const v = self.vrow(h, r);
                for (&o, 0..) |*x, i| x.* += @as(V8, @splat(w)) * lanes(v, i);
            }
            const l: V8 = @splat(host.sum(p));
            for (o, 0..) |x, i| out[8 * i ..][0..8].* = x / l;
            return;
        };
        const l = spec.sumIn(ac, order, p);
        for (out, 0..) |*o, d| {
            for (terms, p, rows) |*x, w, r| x.* = w * self.vrow(h, r)[d];
            o.* = spec.sumIn(ac, order, terms) / l;
        }
    }

    fn outputs(self: *Gather, acc: ?spec.Accum, order: spec.Order, out: []f64) !void {
        const n = self.shape.k * RATIO + RATIO;
        const buf = try self.rig.a.alloc(f64, 2 * n);
        defer self.rig.a.free(buf);
        for (0..HQ) |h| for (0..self.shape.rows) |s| {
            const L = self.off[s + 1] - self.off[s];
            self.exact(h, s, acc, order, buf[0..L], buf[n..][0..L], out[(h * self.shape.rows + s) * BD ..][0..BD]);
        };
    }

    pub fn reference(self: *Gather, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        try self.outputs(null, .sequential, want);
    }

    pub fn simulate(self: *Gather, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputs(acc, order, out);
    }

    /// Per (token, head): o = sum(p v) / sum(p) over the exact weights p = 2^(t - max t). The kernel's weight is
    /// p (1 + delta) up to a factor common to every key: delta prices the score's error, the subtractions against
    /// the running max (which climbs by at most the scores' range), each exp2 and each rescale. NAX's two-half P
    /// adds 2^-22 p + 2^-25 to the values' weights; a flushed exp2 adds 2^-126.
    pub fn check(self: *Gather, at: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const out = self.rig.outputBytes();
        const n = self.shape.k * RATIO + RATIO;
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        var got: [BD]f64 = undefined;
        var want: [BD]f64 = undefined;
        var allow: [BD]f64 = undefined;
        var num: [BD]f64 = undefined;
        var mag: [BD]f64 = undefined;
        var planted: usize = 0;
        for (0..HQ) |h| for (0..self.shape.rows) |s| {
            const base = (h * self.shape.rows + s) * BD;
            for (&got, 0..) |*g, d| g.* = spec.read(.bf16, out, base + d);
            const rows = self.visibleRows(s);
            const L = rows.len;
            const sc = self.scores(h, s);
            const p = buf[0..L];
            const delta = buf[n..][0..L];
            self.exact(h, s, null, .sequential, p, buf[2 * n ..][0..L], &want);
            var hi: f64 = -std.math.inf(f64);
            var lo: f64 = std.math.inf(f64);
            var dmax: f64 = 0;
            for (sc[0], sc[1]) |x, e| {
                hi = @max(hi, x);
                lo = @min(lo, x);
                dmax = @max(dmax, e);
            }
            const tiles: f64 = @floatFromInt((L + BK - 1) / BK);
            const eta = std.math.exp2(2 * U * (hi - lo + 2 * dmax)) * std.math.pow(f64, 1 + EXP2_ERR, tiles + 1) * std.math.pow(f64, 1 + U, tiles) - 1;
            const g = spec.gamma(@as(f64, @floatFromInt(L)) + 2 * tiles + 2, U);
            var l_lo: f64 = 0;
            var l_hi: f64 = 0;
            for (delta, p, sc[1]) |*dl, w, e| {
                dl.* = std.math.exp2(e) * (1 + eta) - 1;
                l_lo += w * (1 - dl.*);
                l_hi += w * (1 + dl.*);
            }
            var nv: [BD / 8]V8 = @splat(@splat(0));
            var mv: [BD / 8]V8 = @splat(@splat(0));
            for (p, delta, rows) |w, dl, r| {
                const piece: V8 = @splat(0x1p-126 + (if (self.nax) 0x1p-22 * w + 0x1p-25 else 0));
                const wd: V8 = @splat(w * dl);
                const wm: V8 = @splat(w * (1 + dl));
                const v = self.vrow(h, r);
                for (&nv, &mv, 0..) |*nu, *mg, i| {
                    const x = lanes(v, i);
                    nu.* += wd * @abs(x - lanes(&want, i)) + piece * @abs(x);
                    mg.* += wm * @abs(x);
                }
            }
            for (nv, mv, 0..) |x, y, i| {
                num[8 * i ..][0..8].* = x;
                mag[8 * i ..][0..8].* = y;
            }
            const flushed = 0x1p-126 * @as(f64, @floatFromInt(L));
            for (&allow, want, num, mag) |*al, o, nu, mg| {
                const e1 = (nu + flushed * @abs(o) + g * (mg + @abs(o) * l_hi)) / (l_lo * (1 - g));
                const e2 = e1 + 2 * U * (@abs(o) + e1);
                al.* = e2 + 0x1p-8 * (@abs(o) + e2) + 0x1p-134;
            }
            for (want) |x| {
                if (spec.classOf(x) != .finite) planted += 1;
            }
            var why: [200]u8 = undefined;
            if (host.within(&why, if (at == .nonfinite) .repeat else at, &got, &want, &allow)) |w| {
                return std.fmt.bufPrint(&self.msg, "token {d} head {d}: {s}", .{ s, h, w }) catch &self.msg;
            }
        };
        if (at == .nonfinite and planted == 0) return "the planted values leave every output finite: no proof";
        return null;
    }

    /// The shipped wrapper on the same operands, its NAX arm forced on or off.
    fn wrapper(self: *Gather) ![]const u8 {
        const prev = gather.qsa_nax_override;
        defer gather.qsa_nax_override = prev;
        gather.qsa_nax_override = self.nax;
        const out = (try gather.gatherQsa256(self.rig.stream(), self.args[0], self.args[1], self.args[2], SCALE, self.args[4], RATIO)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expectEqual(self.nax, gather.qsa_gather_used_nax);
        return self.rig.bytesOf(&.{out});
    }
};

fn depth(shape: Shape) usize {
    // The longest sum: every visible key, plus a rescale a tile.
    const L = shape.k * RATIO + RATIO;
    return L + 2 * (L / BK + 1);
}

/// Flash Next prefill: a chunk just past the budget and one deep in a long prompt (full 512-block selections), a
/// short prompt below it (every complete block selected), each token's tail 0..3 keys. Tokens are independent
/// threadgroups, so a few stand in for a 2048-token chunk; NAX takes chunks from 16 tokens.
const served: []const Shape = &.{ .{ .rows = 8, .n = 2100, .k = 512 }, .{ .rows = 4, .n = 8192, .k = 512 }, .{ .rows = 16, .n = 700, .k = 512 } };
const nax_served: []const Shape = &.{ .{ .rows = 16, .n = 2100, .k = 512 }, .{ .rows = 16, .n = 700, .k = 512 } };

const inf_nan: spec.Known = .{ .gate = .nonfinite, .finding = "P reaches the tensor unit as two half pieces summing to it; a weight exact in half leaves a zero second piece, and zero times an infinite value is NaN: an infinite value comes out NaN where float64 gives the infinity" };

pub fn register(set: *kc.Set) !void {
    inline for (.{ false, true }) |nax| {
        try set.add(Gather, .init(set.a, nax), .{
            .name = if (nax) "qsa_nax_precise bf16 (gathered QSA prefill attention, NAX)" else "attn_qsa256 bf16 (gathered QSA prefill attention)",
            .accum = .f32,
            .depth = depth,
            // The reciprocal of the denominator and the product with it.
            .ops = 2,
            .out = .bf16,
            .served = if (nax) nax_served else served,
            .known = if (nax) inf_nan else null,
            .nax = nax,
        }, null);
    }
}

// ── tests ──

test "kernel contract qwen4_exp: the declared gathered-attention launches are the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    inline for (.{ false, true }) |nax| {
        if (nax and !(gather.qsaNaxEnabled() and device.naxAvailable())) return error.SkipZigTest;
        var k = Gather.init(a, nax);
        defer k.deinit();
        for (if (nax) nax_served else served) |shape| {
            try std.testing.expect(try k.prepare(shape, .random, 4));
            try k.run();
            const ours = try a.dupe(u8, k.rig.outputBytes());
            defer a.free(ours);
            try std.testing.expectEqualSlices(u8, try k.wrapper(), ours);
        }
    }
}
