//! moe_affine4's two decode dispatches in the kernel contract: gate+up+SwiGLU over a token's top-k affine-4 experts
//! (plus the ungated shared expert's slices) and the score-weighted down sum (plus the gated or ungated shared
//! expert). Each output is held to float64 on the kernels' own formula, a group's `s * dot(q, x) + b * sum(x)`, with
//! its chain of roundings priced; chained, the two launches are `moe_affine4.decode`'s bytes. Rows are the top-k (one
//! token); gate+up: n = the expert width, k = hidden; down: n = hidden, k = the expert width.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const host = @import("../../sdk/testing/kernel_contract/host.zig");
const fa = @import("../../sdk/blocks/mlp/fused_activations.zig");
const moe_affine4 = @import("../../sdk/blocks/moe/moe_affine4.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

const DT: mlx.mlx_dtype = .bfloat16;
const GS: usize = 64;
const UT: f64 = 0x1p-8;
const FLOOR: f64 = 0x1p-133;

const gateup_in = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "sg_q", "sg_scales", "sg_biases", "su_q", "su_scales", "su_biases" };
const downred_in = [_][*:0]const u8{ "act", "wd_q", "d_scales", "d_biases", "inds", "scores", "sdown", "glog", "sd_q", "sd_scales", "sd_biases" };
const launches = [2]metal.Launch{
    .{ .name = "mlxserve_moe_a4_gateup", .inputs = &gateup_in, .outputs = &.{"y"}, .source = moe_affine4.GATEUP_SOURCE, .header = moe_affine4.DOT16, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "mlxserve_moe_a4_downred", .inputs = &downred_in, .outputs = &.{"y"}, .source = moe_affine4.DOWNRED_SOURCE, .header = moe_affine4.DOT16, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 },
};

pub const Part = enum { gateup, downred };
/// The layer's shared expert: none, gated (a down row and a logit the down kernel adds) or ungated (one more slice of
/// the expert width through both kernels).
pub const Shared = enum { none, gated, ungated };

pub const Affine4 = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    part: Part,
    shared: Shared,
    /// Fill both dispatches' inputs (the drift guard chains them).
    both: bool = false,
    sp: spec.Spec = undefined,
    /// gate, up, down; then the ungated shared expert's gate, up, down.
    banks: [6]host.Affine = @splat(.{ .bits = 4, .gs = GS, .n = 0, .k = 0 }),
    shape: Shape = undefined,
    x: []f64 = &.{},
    act: []f64 = &.{},
    inds: []f64 = &.{},
    scores: []f64 = &.{},
    sdown: []f64 = &.{},
    glog: f64 = 0,
    table: []f64 = &.{},
    dv: []f64 = &.{},
    dm: []f64 = &.{},
    gu: [15]mlx.mlx_array = undefined,
    dn: [11]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    pub fn init(a: std.mem.Allocator, part: Part, shared: Shared) Affine4 {
        return .{ .rig = metal.Rig.init(a), .launch = &launches[@backingInt(part)], .part = part, .shared = shared };
    }

    pub fn deinit(self: *Affine4) void {
        const a = self.rig.a;
        self.rig.deinit();
        for (&self.banks) |*b| b.deinit(a);
        self.freeVecs();
        a.free(self.table);
    }

    fn freeVecs(self: *Affine4) void {
        const a = self.rig.a;
        for ([_]*[]f64{ &self.x, &self.act, &self.inds, &self.scores, &self.sdown, &self.dv, &self.dm }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
    }

    /// Ungated shared slices: one expert width (SE 1).
    fn se(self: *const Affine4) usize {
        return @intFromBool(self.shared == .ungated);
    }

    fn hiddenOf(part: Part, shape: Shape) usize {
        return if (part == .gateup) shape.k else shape.n;
    }

    fn interOf(part: Part, shape: Shape) usize {
        return if (part == .gateup) shape.n else shape.k;
    }

    /// The wrapper's dispatch: a simdgroup a row, two a group; gate+up's z the slots, down's one.
    fn geometryOf(self: *const Affine4, part: Part, shape: Shape) ?metal.Geometry {
        const h = hiddenOf(part, shape);
        const i = interOf(part, shape);
        if (shape.rows == 0 or h % 16 != 0 or i % 16 != 0 or h % GS != 0 or i % GS != 0 or h % 2 != 0 or i % 2 != 0) return null;
        const topk: c_int = @intCast(shape.rows);
        const ch: c_int = @intCast(h);
        const ci: c_int = @intCast(i);
        const s: c_int = @intCast(self.se());
        var g: metal.Geometry = undefined;
        if (part == .gateup) {
            g = .{ .grid = .{ 32, ci, topk + s }, .tg = .{ 32, 2, 1 } };
            g.out(&.{ topk + s, ci }, DT);
        } else {
            g = .{ .grid = .{ 32, ch, 1 }, .tg = .{ 32, 2, 1 } };
            g.out(&.{ch}, DT);
        }
        g.dtype("T", DT);
        g.int("ROWS", 1);
        if (part == .gateup) {
            g.int("K", ch);
            g.int("N", ci);
            g.int("GS", GS);
            g.int("TOPK", topk);
        } else {
            g.int("I", ci);
            g.int("H", ch);
            g.int("GS", GS);
            g.int("TOPK", topk);
            g.int("SHARED", @intFromBool(self.shared == .gated));
            g.int("SI", self.se() * i);
        }
        return g;
    }

    pub fn geometry(self: *const Affine4, shape: Shape) ?metal.Geometry {
        return self.geometryOf(self.part, shape);
    }

    fn slots(self: *const Affine4) usize {
        return self.shape.rows + self.se();
    }

    pub fn prepare(self: *Affine4, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        self.freeVecs();
        if (self.table.len == 0) self.table = try host.read(a, try fa.swigluSigTable(self.rig.stream(), DT, std.heap.c_allocator), self.rig.stream());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const topk = shape.rows;
        const h = hiddenOf(self.part, shape);
        const i = interOf(self.part, shape);
        const stress = fill == .stress;
        const up = self.part == .gateup or self.both;
        const down = self.part == .downred or self.both;
        if (up) for (self.banks[0..2]) |*b| try b.fill(a, rng, topk * i, h, DT, stress);
        if (down) try self.banks[2].fill(a, rng, topk * h, i, DT, stress);
        if (self.shared == .ungated) {
            if (up) for (self.banks[3..5]) |*b| try b.fill(a, rng, i, h, DT, stress);
            if (down) try self.banks[5].fill(a, rng, h, i, DT, stress);
        }
        self.x = try a.alloc(f64, h);
        self.act = try a.alloc(f64, self.slots() * i);
        if (stress) {
            stressX(rng, self.x, h);
            stressX(rng, self.act, i);
        } else {
            // Positive under the non-finite fill: an infinite weight then gives one signed infinity however its
            // group's x is split over the lanes (mixed-sign partial sums would make it NaN on some splits only).
            const lo: f64 = if (fill == .nonfinite) 0.25 else -2;
            host.uniformInto(rng, self.x, lo, 2, DT);
            host.uniformInto(rng, self.act, lo, 2, DT);
        }
        self.inds = try a.alloc(f64, topk);
        for (self.inds, 0..) |*v, j| v.* = @floatFromInt(j);
        rng.shuffle(f64, self.inds);
        self.scores = try a.alloc(f64, topk);
        for (self.scores) |*v| v.* = if (stress) 1 else host.rt(DT, 0.05 + 0.25 * rng.float(f64));
        self.sdown = try a.alloc(f64, h);
        for (self.sdown) |*v| v.* = if (stress) 0 else host.rt(DT, -1 + 2 * rng.float(f64));
        self.glog = host.rt(DT, -3 + 6 * rng.float(f64));
        if (fill == .nonfinite) {
            // NaN in a scale, +Inf and -Inf in a bias, each in its own slot's own output row's first group:
            // one output each goes non-finite (gate+up: that slot's row; down: that row of the sum).
            const bk = if (self.part == .gateup) [3]usize{ 0, 1, 0 } else [3]usize{ 2, 2, 2 };
            const n = shape.n;
            const k = shape.k;
            for (0..3) |e| {
                const r = self.expert(e) * n + e;
                const g = r * (k / GS);
                switch (e) {
                    0 => self.banks[bk[e]].scales[g] = std.math.nan(f64),
                    1 => self.banks[bk[e]].biases[g] = std.math.inf(f64),
                    else => self.banks[bk[e]].biases[g] = -std.math.inf(f64),
                }
            }
        }
        try self.bind();
        return true;
    }

    fn bank3(rig: *metal.Rig, b: *const host.Affine, lead: ?usize, n: usize, k: usize) ![3]mlx.mlx_array {
        const ci: c_int = @intCast(n);
        var w: [3]mlx.mlx_array = undefined;
        if (lead) |e| {
            const ce: c_int = @intCast(e);
            w[0] = try rig.input(.uint32, &.{ ce, ci, @intCast(k / 8) }, b.words);
            w[1] = try rig.input(DT, &.{ ce, ci, @intCast(k / GS) }, b.scales);
            w[2] = try rig.input(DT, &.{ ce, ci, @intCast(k / GS) }, b.biases);
        } else {
            w[0] = try rig.input(.uint32, &.{ ci, @intCast(k / 8) }, b.words);
            w[1] = try rig.input(DT, &.{ ci, @intCast(k / GS) }, b.scales);
            w[2] = try rig.input(DT, &.{ ci, @intCast(k / GS) }, b.biases);
        }
        return w;
    }

    /// The inputs in each kernel's order, bound as the wrapper binds them: absent shared operands are the routed
    /// banks again (gate+up's, down's) and the scores (the gated expert's row and logit), which the kernels never read.
    fn bind(self: *Affine4) !void {
        const rig = &self.rig;
        rig.reset();
        const topk = self.shape.rows;
        const h = hiddenOf(self.part, self.shape);
        const i = interOf(self.part, self.shape);
        const ci: c_int = @intCast(i);
        const ind = try rig.input(.uint32, &.{@intCast(topk)}, self.inds);
        if (self.part == .gateup or self.both) {
            const g = try bank3(rig, &self.banks[0], topk, i, h);
            const u = try bank3(rig, &self.banks[1], topk, i, h);
            const sg = if (self.shared == .ungated) try bank3(rig, &self.banks[3], null, i, h) else g;
            const su = if (self.shared == .ungated) try bank3(rig, &self.banks[4], null, i, h) else u;
            self.gu = .{ try rig.input(DT, &.{@intCast(h)}, self.x), g[0], g[1], g[2], u[0], u[1], u[2], ind, try rig.input(DT, &.{65536}, self.table), sg[0], sg[1], sg[2], su[0], su[1], su[2] };
        }
        if (self.part == .downred or self.both) {
            const d = try bank3(rig, &self.banks[2], topk, h, i);
            const sd = if (self.shared == .ungated) try bank3(rig, &self.banks[5], null, h, i) else d;
            const scores = try rig.input(DT, &.{@intCast(topk)}, self.scores);
            const gated = self.shared == .gated;
            const sdown = if (gated) try rig.input(DT, &.{@intCast(h)}, self.sdown) else scores;
            const glog = if (gated) try rig.input(DT, &.{1}, &.{self.glog}) else scores;
            self.dn = .{ try rig.input(DT, &.{ @intCast(self.slots()), ci }, self.act), d[0], d[1], d[2], ind, scores, sdown, glog, sd[0], sd[1], sd[2] };
        }
    }

    pub fn run(self: *Affine4) !void {
        const g = self.geometry(self.shape).?;
        if (self.part == .gateup) try self.rig.launch(self.launch, g, &self.gu) else try self.rig.launch(self.launch, g, &self.dn);
    }

    pub fn count(self: *const Affine4) usize {
        return if (self.part == .gateup) self.slots() * self.shape.n else self.shape.n;
    }

    fn expert(self: *const Affine4, e: usize) usize {
        return @intFromFloat(self.inds[e]);
    }

    /// Row `r` of `b` against `x` on the kernels' formula: per group `s * sum(q x) + b * sum(x)`, and its magnitude.
    fn rowDot(b: *const host.Affine, r: usize, x: []const f64) [2]f64 {
        var v: f64 = 0;
        var m: f64 = 0;
        for (0..b.k / GS) |gi| {
            var qx: f64 = 0;
            var qm: f64 = 0;
            var xs: f64 = 0;
            var xm: f64 = 0;
            for (gi * GS..(gi + 1) * GS) |i| {
                const q: f64 = @floatFromInt(b.codes[r * b.k + i]);
                qx += q * x[i];
                qm += @abs(q * x[i]);
                xs += x[i];
                xm += @abs(x[i]);
            }
            const s = b.scales[r * (b.k / GS) + gi];
            const bb = b.biases[r * (b.k / GS) + gi];
            v += s * qx + bb * xs;
            m += @abs(s) * qm + @abs(bb) * xm;
        }
        return .{ v, m };
    }

    /// Every slot's dot products (gate and up, or down per slot plus the ungated shared row), once a prepare.
    fn dots(self: *Affine4) !void {
        if (self.dv.len > 0) return;
        const a = self.rig.a;
        const n = self.shape.n;
        const k = self.shape.k;
        const topk = self.shape.rows;
        const nb: usize = if (self.part == .gateup) 2 else 1;
        self.dv = try a.alloc(f64, nb * self.slots() * n);
        self.dm = try a.alloc(f64, nb * self.slots() * n);
        for (0..nb) |bi| for (0..self.slots()) |s| for (0..n) |j| {
            const sh = s >= topk;
            const at = (bi * self.slots() + s) * n + j;
            const d = if (self.part == .gateup)
                rowDot(&self.banks[bi + @as(usize, if (sh) 3 else 0)], if (sh) j else self.expert(s) * n + j, self.x)
            else
                rowDot(&self.banks[if (sh) 5 else 2], if (sh) j else self.expert(s) * n + j, self.act[s * k ..][0..k]);
            self.dv[at] = d[0];
            self.dm[at] = d[1];
        };
    }

    fn silu(v: f64) f64 {
        return v / (1 + @exp(-v));
    }

    /// The outputs on dot products `dv`: with `exact`, the float64 function and each output's allowance (gate 3);
    /// else the kernel's own rounding chain after its sums (gates 1 and 4).
    fn epilogue(self: *const Affine4, dv: []const f64, dm: []const f64, exact: bool, want: []f64, allow: []f64) void {
        const n = self.shape.n;
        const dep = self.sp.depth(self.shape);
        if (self.part == .gateup) {
            const sn = self.slots() * n;
            for (want, allow, 0..) |*y, *l, at| {
                if (!exact) {
                    const gt = host.rt(DT, dv[at]);
                    y.* = host.rt(DT, gt * self.table[spec.toBf16(@floatCast(gt))]) * host.rt(DT, dv[sn + at]);
                    l.* = 0;
                    continue;
                }
                const g = dv[at];
                const u = dv[sn + at];
                const sg = silu(g);
                y.* = sg * u;
                // Each sum's error and its rounding to T; SiLU's slope is at most 1.1; the table's rounding and the
                // product's; the final multiply's.
                const eg = gamma32(dep) * dm[at];
                const eu = gamma32(dep) * dm[sn + at];
                const dg = eg + UT * (@abs(g) + eg);
                const du = eu + UT * (@abs(u) + eu);
                const ds = 1.1 * dg + 2 * UT * (@abs(sg) + 1.1 * dg);
                const work = @abs(u) * ds + (@abs(sg) + ds) * du;
                l.* = work + UT * (@abs(y.*) + work) + FLOOR;
            }
            return;
        }
        const topk = self.shape.rows;
        for (want, allow, 0..) |*y, *l, j| {
            var v: f64 = 0;
            var mag: f64 = 0;
            for (0..topk) |e| {
                v += self.scores[e] * dv[e * n + j];
                mag += @abs(self.scores[e]) * dm[e * n + j];
            }
            if (self.shared == .ungated) {
                v += dv[topk * n + j];
                mag += dm[topk * n + j];
            }
            const ev = gamma32(dep) * mag;
            if (self.shared != .gated) {
                y.* = v;
                l.* = ev + UT * (@abs(v) + ev) + FLOOR;
                continue;
            }
            const sig = 1 / (1 + @exp(-self.glog));
            const sd = self.sdown[j];
            if (!exact) {
                y.* = host.rt(DT, v) + host.rt(DT, host.rt(DT, sig) * sd);
                l.* = 0;
                continue;
            }
            y.* = v + sig * sd;
            // T(v), the sigmoid (precise exp, then T), its product with the shared row, the f32 add, the output.
            const dsig = UT * sig + 0x1p-20;
            const work = ev + UT * (@abs(v) + ev) + @abs(sd) * dsig + UT * @abs(sig * sd) + 0x1p-24 * (@abs(v) + @abs(sig * sd));
            l.* = work + UT * (@abs(y.*) + work) + FLOOR;
        }
    }

    /// The kernel's rounding chain on exact sums (gate 4).
    pub fn reference(self: *Affine4, want: []f64, mag: []f64) !void {
        try self.dots();
        self.epilogue(self.dv, self.dm, false, want, mag);
    }

    /// fp32 sums of the integer stress terms are exact in any order; fp16 and bf16 sums are run for the first 32
    /// outputs of every slot: a lower bound on their miss.
    pub fn simulate(self: *Affine4, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.dots();
        const a = self.rig.a;
        const scratch = try a.alloc(f64, out.len);
        defer a.free(scratch);
        if (acc == .f32) {
            for (self.dm) |m| if (m >= 0x1p24) return error.StressPastExact;
            return self.epilogue(self.dv, self.dm, false, out, scratch);
        }
        const dv = try a.dupe(f64, self.dv);
        defer a.free(dv);
        const n = self.shape.n;
        const k = self.shape.k;
        const topk = self.shape.rows;
        const t = try a.alloc(f64, k);
        defer a.free(t);
        const nb: usize = if (self.part == .gateup) 2 else 1;
        for (0..nb) |bi| for (0..self.slots()) |s| for (0..n) |j| {
            const sh = s >= topk;
            const b = &self.banks[if (self.part == .gateup) bi + @as(usize, if (sh) 3 else 0) else if (sh) 5 else 2];
            const r = if (sh) j else self.expert(s) * n + j;
            const xr = if (self.part == .gateup) self.x else self.act[s * k ..][0..k];
            for (t, xr, 0..) |*o, v, i| {
                const q: f64 = @floatFromInt(b.codes[r * k + i]);
                o.* = v * (q * b.scales[(r * k + i) / GS] + b.biases[(r * k + i) / GS]);
            }
            dv[(bi * self.slots() + s) * n + j] = spec.sumIn(acc, order, t);
        };
        self.epilogue(dv, self.dm, false, out, scratch);
    }

    /// Gate 1 on the kernel's chain, gate 3 against the float64 function within each output's allowance.
    pub fn check(self: *Affine4, at: spec.Gate) !?[]const u8 {
        try self.dots();
        const a = self.rig.a;
        const n = self.count();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        self.epilogue(self.dv, self.dm, at == .shapes, buf[0..n], buf[n .. 2 * n]);
        const out = self.rig.outputBytes();
        for (buf[2 * n ..], 0..) |*g, i| g.* = spec.read(.bf16, out, i);
        return host.within(&self.msg, at, buf[2 * n ..], buf[0..n], buf[n .. 2 * n]);
    }

    /// Both dispatches through the declared launches, then `moe_affine4.decode` on the same inputs: the bytes of
    /// the weighted down sum, ours and the wrapper's.
    pub fn chained(self: *Affine4) ![2][]const u8 {
        const a = self.rig.a;
        const gshape: Shape = .{ .rows = self.shape.rows, .n = interOf(self.part, self.shape), .k = hiddenOf(self.part, self.shape) };
        const dshape: Shape = .{ .rows = self.shape.rows, .n = gshape.k, .k = gshape.n };
        try self.rig.launch(&launches[0], self.geometryOf(.gateup, gshape).?, &self.gu);
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        try mlx.check(mlx.mlx_array_set(&act, self.rig.output(0)));
        var dn = self.dn;
        dn[0] = act;
        try self.rig.launch(&launches[1], self.geometryOf(.downred, dshape).?, &dn);
        const ours = try a.dupe(u8, self.rig.outputBytes());
        errdefer a.free(ours);
        const gu = self.gu;
        const B = moe_affine4.Bank;
        const shared: ?moe_affine4.Shared = if (self.shared == .gated) .{ .down = dn[6], .logit = dn[7] } else null;
        const ungated: ?moe_affine4.Ungated = if (self.shared == .ungated) .{ .gate = B{ .w = gu[9], .s = gu[10], .b = gu[11] }, .up = B{ .w = gu[12], .s = gu[13], .b = gu[14] }, .down = B{ .w = dn[8], .s = dn[9], .b = dn[10] } } else null;
        const y = (try moe_affine4.decode(self.rig.stream(), gu[0], B{ .w = gu[1], .s = gu[2], .b = gu[3] }, B{ .w = gu[4], .s = gu[5], .b = gu[6] }, B{ .w = dn[1], .s = dn[2], .b = dn[3] }, null, shared, ungated, gu[7], dn[5], gu[8], GS)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return .{ ours, try self.rig.bytesOf(&.{y}) };
    }
};

fn gamma32(d: usize) f64 {
    return spec.gamma(@floatFromInt(d), 0x1p-24);
}

/// Mirrored integer rows (`host.mirroredWide`) with the second half reversed: lane-strided partial sums leave fp16's
/// grid while the whole row stays a small integer.
fn stressX(rng: std.Random, v: []f64, k: usize) void {
    host.mirroredWide(rng, v, k);
    const h = k / 2;
    for (0..v.len / k) |r| std.mem.reverse(f64, v[r * k + h ..][0 .. k - h]);
}

/// A row's sums: its K terms behind the dequant and the product, the top-k slots and the simdgroup's sum.
fn depth(shape: Shape) usize {
    return shape.k + shape.rows + 8;
}

fn specOf(comptime name: []const u8, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        // n: whole groups; k: one pass of the 32 lanes over 16 values, in whole groups.
        .tiles = .{ .{}, .{ .size = GS, .step = GS, .min = GS }, .{ .size = 512, .step = GS, .min = GS } },
        .served = served,
    };
}

fn add(set: *kc.Set, comptime name: []const u8, part: Part, shared: Shared, served: []const Shape) !void {
    var k = Affine4.init(set.a, part, shared);
    k.sp = specOf(name, served);
    try set.add(Affine4, k, k.sp, null);
}

/// Flash Next: top-10 of 640-wide experts over hidden 2560, a gated shared expert; the ungated arm (Flint) on a
/// smaller layer.
pub fn register(set: *kc.Set) !void {
    try add(set, "moe_affine4 gate+up bf16 (top-10, 640 x 2560)", .gateup, .none, &.{.{ .rows = 10, .n = 640, .k = 2560 }});
    try add(set, "moe_affine4 gate+up bf16 (top-6 + an ungated shared slice)", .gateup, .ungated, &.{.{ .rows = 6, .n = 256, .k = 1024 }});
    try add(set, "moe_affine4 down+reduce bf16 (top-10, a gated shared expert)", .downred, .gated, &.{.{ .rows = 10, .n = 2560, .k = 640 }});
    try add(set, "moe_affine4 down+reduce bf16 (top-10, no shared expert)", .downred, .none, &.{.{ .rows = 10, .n = 2560, .k = 640 }});
    try add(set, "moe_affine4 down+reduce bf16 (top-6 + an ungated shared slice)", .downred, .ungated, &.{.{ .rows = 6, .n = 1024, .k = 256 }});
}

// ── tests ──

test "kernel contract qwen4_exp: the affine-4 launches, chained, are moe_affine4.decode's bytes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    for ([_]Shared{ .none, .gated, .ungated }) |sh| {
        var k = Affine4.init(a, .gateup, sh);
        defer k.deinit();
        k.both = true;
        k.sp = specOf("t", &.{.{ .rows = 6, .n = 256, .k = 1024 }});
        try std.testing.expect(try k.prepare(.{ .rows = 6, .n = 256, .k = 1024 }, .random, 11));
        const got = try k.chained();
        defer a.free(got[0]);
        try std.testing.expectEqualSlices(u8, got[1], got[0]);
    }
}
