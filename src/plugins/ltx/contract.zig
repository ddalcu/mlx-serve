//! LTX's kernel contract: the DiffVAE's fused 3D neighborhood attention, one launch per window it ships. Listed in
//! `kernel_contract/registry.zig`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const na = @import("diffvae_kernel.zig");
const geom = @import("diffvae.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

fn launchFor(comptime name: [:0]const u8) metal.Launch {
    // The launch is 32 threads, under any pipeline's limit; the offline M1/M2 builds come back inconclusive here.
    return .{ .name = name, .inputs = &.{ "q", "k", "v" }, .outputs = &.{"out"}, .source = na.SOURCE, .header = na.HEADER, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };
}

const k777 = launchFor("ltx_na3d_k3_7_7");
const k355 = launchFor("ltx_na3d_k3_5_5");

/// q, k, v [1, T, H, W, heads, hd] -> [1, T, H, W, heads * hd], each query over its NATTEN window (shifted inward
/// at an edge). `rows` is W, `n` the heads, `k` the head width; T and H are the registration's.
const Na3d = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kern: geom.Kernel,
    t: usize,
    h: usize,
    dt: mlx.mlx_dtype = .bfloat16,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, kern: geom.Kernel, t: usize, h: usize) Na3d {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .kern = kern, .t = t, .h = h, .arena = .init(a) };
    }

    pub fn deinit(self: *Na3d) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Na3d, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return null;
        const w: c_int = @intCast(shape.rows);
        const heads: c_int = @intCast(shape.n);
        var g = metal.Geometry{ .grid = .{ w, @intCast(self.h), @as(c_int, @intCast(self.t)) * heads }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ 1, @intCast(self.t), @intCast(self.h), w * heads * @as(c_int, @intCast(shape.k)) }, self.dt);
        g.dtype("T", self.dt);
        g.int("KT", self.kern[0]);
        g.int("KH", self.kern[1]);
        g.int("KW", self.kern[2]);
        g.int("HD", shape.k);
        return g;
    }

    fn count3(self: *const Na3d) usize {
        return self.t * self.h * self.shape.rows * self.shape.n * self.shape.k;
    }

    pub fn prepare(self: *Na3d, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = self.count3();
        const scale = 1 / @sqrt(@as(f64, @floatFromInt(shape.k)));
        for ([_]*[]f64{ &self.q, &self.k, &self.v }, 0..) |x, i| {
            x.* = try a.alloc(f64, n);
            // Q arrives pre-scaled, as the forward hands it over.
            for (x.*) |*e| e.* = spec.roundTo(.bf16, (rng.float(f64) * 4 - 2) * (if (i == 0) scale else 1));
        }
        switch (fill) {
            .random => {},
            .nonfinite => {
                const hd = shape.k;
                self.q[3 * hd + 1] = std.math.nan(f64);
                self.k[(n / hd / 2) * hd] = std.math.inf(f64);
                self.v[(n / hd - 1) * hd + 2] = std.math.inf(f64);
                self.v[(n / hd - 1) * hd + 3] = -std.math.inf(f64);
            },
            .stress => {
                // Small integers: every score sum exact in fp32; the exp keeps the outputs inexact.
                for ([_][]f64{ self.q, self.k, self.v }) |x| for (x) |*e| {
                    e.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 5))) - 2);
                };
            },
        }
        const rig = &self.rig;
        rig.reset();
        const shp = [_]c_int{ 1, @intCast(self.t), @intCast(self.h), @intCast(shape.rows), @intCast(shape.n), @intCast(shape.k) };
        for (&self.args, [_][]f64{ self.q, self.k, self.v }) |*arg, x| arg.* = try rig.input(self.dt, &shp, x);
        return true;
    }

    pub fn run(self: *Na3d) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Na3d) usize {
        return self.count3();
    }

    fn keys(self: *const Na3d) usize {
        return @min(self.kern[0], self.t) * @min(self.kern[1], self.h) * @min(self.kern[2], self.shape.rows);
    }

    /// The definition (`naNaiveF32` in float64). `mag` carries the first-order error as a sum magnitude at depth
    /// `keys` (the bound multiplies it by gamma(keys)): the P*V sums, the running rescales and the normalizer, and
    /// the score sums' own error.
    fn outputsIn(self: *const Na3d, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const a = self.arena.child_allocator;
        const nk = self.keys();
        const hd = self.shape.k;
        const heads = self.shape.n;
        const nw = self.shape.rows;
        const kr = try a.alloc(usize, nk);
        defer a.free(kr);
        const sc = try a.alloc(f64, nk);
        defer a.free(sc);
        const sa = try a.alloc(f64, nk);
        defer a.free(sa);
        const terms = try a.alloc(f64, @max(nk, hd));
        defer a.free(terms);
        const g_keys = spec.gamma(@floatFromInt(nk), spec.unit(.f32));
        const g_d = spec.gamma(@floatFromInt(hd), spec.unit(.f32));
        const dims = [3]usize{ self.t, self.h, nw };
        for (0..self.t) |t| for (0..self.h) |h| for (0..nw) |w| for (0..heads) |hh| {
            const pos = [3]usize{ t, h, w };
            var start: [3]usize = undefined;
            var len: [3]usize = undefined;
            for (&start, &len, dims, pos, 0..) |*s0, *l, d, p, ax| {
                s0.* = geom.naWindowStart(@intCast(d), self.kern[ax], @intCast(p));
                l.* = @min(self.kern[ax], d);
            }
            var j: usize = 0;
            for (0..len[0]) |x| for (0..len[1]) |y| for (0..len[2]) |z| {
                kr[j] = ((((start[0] + x) * self.h + start[1] + y) * nw + start[2] + z) * heads + hh) * hd;
                j += 1;
            };
            const qo = (((t * self.h + h) * nw + w) * heads + hh) * hd;
            const q = self.q[qo..][0..hd];
            var mx = -std.math.inf(f64);
            for (sc, sa, kr) |*s, *s_abs, r| {
                var abs_sum: f64 = 0;
                for (terms[0..hd], q, self.k[r..][0..hd]) |*tm, qd, kd| {
                    tm.* = qd * kd;
                    abs_sum += @abs(tm.*);
                }
                s.* = if (acc) |ac| spec.sumIn(ac, order, terms[0..hd]) else sumF64(terms[0..hd]);
                s_abs.* = abs_sum;
                mx = @max(mx, s.*);
            }
            var lsum: f64 = 0;
            for (sc) |*s| {
                s.* = @exp(s.* - mx);
                lsum += s.*;
            }
            for (0..hd) |d| {
                var pv_abs: f64 = 0;
                for (terms[0..nk], sc, kr) |*tm, p, r| {
                    tm.* = p * self.v[r + d];
                    pv_abs += @abs(tm.*);
                }
                const o = (if (acc) |ac| spec.sumIn(ac, order, terms[0..nk]) else sumF64(terms[0..nk])) / lsum;
                want[qo + d] = o;
                if (mag) |m| {
                    var spread: f64 = 0;
                    for (sc, sa, kr) |p, s_abs, r| spread += p * @abs(self.v[r + d] - o) * (g_d * s_abs + 8 * spec.unit(.f32));
                    m[qo + d] = (3 * g_keys * (pv_abs / lsum + @abs(o)) + spread / lsum) / g_keys;
                }
            }
        };
    }

    pub fn reference(self: *const Na3d, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Na3d, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }
};

fn sumF64(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += v;
    return s;
}

fn depth377(shape: Shape) usize {
    return @as(usize, 3 * 7) * @min(7, shape.rows);
}

fn depth355(shape: Shape) usize {
    return @as(usize, 3 * 5) * @min(5, shape.rows);
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    const known: spec.Known = .{ .gate = .nonfinite, .finding = "a +Inf score (an Inf in K) returns that key's value: the new-maximum branch sets the running max to +Inf and never computes exp(Inf - Inf), where float64 softmax (and the reference's torch softmax) give NaN" };
    const sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the softmax's exp leaves no sum exact: the stress inputs check the fp32 sums within the float64 bound" };
    // W spans a threadgroup edge (37) and a window wider than the axis (5); T and H sit inside and on the edges.
    try set.add(Na3d, .init(a, &k777, .{ 3, 7, 7 }, 4, 9), .{
        .name = "ltx diffvae na3d 3x7x7 bf16",
        .accum = .f32,
        .depth = depth377,
        .out = .bf16,
        .served = &.{ .{ .rows = 37, .n = 2, .k = 64 }, .{ .rows = 5, .n = 1, .k = 64 } },
        .sums = sums,
        .known = known,
    }, null);
    try set.add(Na3d, .init(a, &k355, .{ 3, 5, 5 }, 5, 6), .{
        .name = "ltx diffvae na3d 3x5x5 bf16",
        .accum = .f32,
        .depth = depth355,
        .out = .bf16,
        .served = &.{ .{ .rows = 33, .n = 2, .k = 64 }, .{ .rows = 3, .n = 1, .k = 64 } },
        .sums = sums,
        .known = known,
    }, null);
}

// ── tests ──

const testing = std.testing;

test "kernel contract ltx: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    inline for (.{ .{ &k777, geom.Kernel{ 3, 7, 7 } }, .{ &k355, geom.Kernel{ 3, 5, 5 } } }) |c| {
        var t = Na3d.init(a, c[0], c[1], 4, 6);
        defer t.deinit();
        try testing.expect(try t.prepare(.{ .rows = 35, .n = 2, .k = 64 }, .random, 5));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        const theirs = try na.na3d(t.args[0], t.args[1], t.args[2], c[1], t.rig.stream());
        defer _ = mlx.mlx_array_free(theirs);
        try testing.expectEqualSlices(u8, try t.rig.bytesOf(&.{theirs}), ours);
    }
}
