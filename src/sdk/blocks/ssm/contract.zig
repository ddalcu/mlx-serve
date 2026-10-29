//! The ssm area's kernel contract: the fused Mamba2 step (conv window, depthwise conv + SiLU, dt softplus, the SSM
//! recurrence, D skip, SiLU(z) gate) and its grouped RMS norm, at Nemotron-H's served geometry (64 heads x 64, 8
//! groups, state 128, conv 4). Neither replaces its op chain bit for bit: both meet the float64 bound. The step
//! writes two dtypes (y and the conv state in T, the SSM state in f32), so it registers twice, once per dtype; its
//! reference carries a forward-error allowance (`host.Val`): the device rounds conv, SiLU, A and y to bf16 after
//! fast-math exp, so a rounding float64 cannot reproduce may move one ulp. Served on every Apple GPU (M1 and up).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const m2 = @import("mamba2_decode.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Val = host.Val;
const Shape = spec.Shape;
const U = host.U;

/// The model geometry the template takes.
const Geom = struct {
    h: usize,
    dh: usize,
    ng: usize,
    ds: usize,
    kc: usize,

    fn xd(g: Geom) usize {
        return g.h * g.dh;
    }
    fn cd(g: Geom) usize {
        return g.xd() + 2 * g.ng * g.ds;
    }
    fn proj(g: Geom) usize {
        return 2 * g.xd() + 2 * g.ng * g.ds + g.h;
    }
};

/// Nemotron-3.5 Lightning / Nemotron-3 Nano 30B (`src/fixtures/model-configs/nemotron-h.json`).
const nano = Geom{ .h = 64, .dh = 64, .ng = 8, .ds = 128, .kc = 4 };

/// Threadgroup bytes: `bc[MAXR * 2 * DS]` f32 at the 16-row bucket.
const step_launch: metal.Launch = .{
    .name = "msv_mamba2_step",
    .inputs = &.{ "P", "CS_IN", "S_IN", "CW", "CB", "A_LOG", "DT_BIAS", "DSKIP", "dt_lo", "dt_hi", "rows" },
    .outputs = &.{ "Y", "S_OUT", "CS_OUT" },
    .source = m2.STEP_SOURCE,
    .tg_bytes = 16 * 2 * 128 * 4,
    .m1 = 1024,
    .m2 = 1024,
};

/// Threadgroup bytes: `partial[TN / 32]` f32 at GS 1024.
const gnorm_launch: metal.Launch = .{
    .name = "msv_mamba2_gnorm",
    .inputs = &.{ "X", "W", "eps" },
    .outputs = &.{"OUT"},
    .source = m2.GNORM_SOURCE,
    .tg_bytes = 32,
    .m1 = 1024,
    .m2 = 1024,
};

const DT_LO: f32 = 0.001;
const DT_HI: f32 = 0.1;
const EPS: f32 = 1e-5;
const TGY = 8;

/// Which outputs a registration reads: y and the conv state (T), or the SSM state (f32).
const Part = enum { y_conv, state };

/// The step at one instantiation; a shape's rows are the window's rows (one batch row).
const Step = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    g: Geom,
    part: Part,
    all_rows: bool,
    dt: mlx.mlx_dtype = .bfloat16,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    stress: bool = false,
    p: []f64 = &.{},
    cs: []f64 = &.{},
    s0: []f64 = &.{},
    cw: []f64 = &.{},
    cb: []f64 = &.{},
    alog: []f64 = &.{},
    dtb: []f64 = &.{},
    dskip: []f64 = &.{},
    lo: f64 = DT_LO,
    hi: f64 = DT_HI,
    args: [11]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, g: Geom, part: Part, all_rows: bool) Step {
        return .{ .rig = metal.Rig.init(a), .launch = &step_launch, .g = g, .part = part, .all_rows = all_rows, .arena = .init(a) };
    }

    pub fn deinit(self: *Step) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn bucket(rows: usize) usize {
        return if (rows <= 1) 1 else if (rows <= 4) 4 else 16;
    }

    pub fn geometry(self: *const Step, shape: Shape) ?metal.Geometry {
        const g = self.g;
        const r: c_int = @intCast(shape.rows);
        if (shape.rows < 1 or bucket(shape.rows) * 2 * g.ds * 4 > 16 * 1024) return null;
        var geo = metal.Geometry{ .grid = .{ 32, @intCast(g.dh), @intCast(g.h) }, .tg = .{ 32, TGY, 1 } };
        geo.out(&.{ 1, r, @intCast(g.xd()) }, self.dt);
        // The wrapper's [srows, B, H, Dh, Ds] with B = 1 folded (the rig's outputs take four dims; same bytes).
        geo.out(&.{ if (self.all_rows) r else 1, @intCast(g.h), @intCast(g.dh), @intCast(g.ds) }, .float32);
        geo.out(&.{ 1, @intCast(g.kc - 1), @intCast(g.cd()) }, self.dt);
        geo.unread = if (self.part == .y_conv) 0b010 else 0b101;
        geo.dtype("T", self.dt);
        geo.int("XD", g.xd());
        geo.int("DH", g.dh);
        geo.int("DS", g.ds);
        geo.int("H", g.h);
        geo.int("NG", g.ng);
        geo.int("KC", g.kc);
        geo.int("PROJ", g.proj());
        geo.int("B", 1);
        geo.int("MAXR", bucket(shape.rows));
        geo.int("TGY", TGY);
        geo.flag("ALLROWS", self.all_rows);
        return geo;
    }

    pub fn prepare(self: *Step, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        self.stress = fill == .stress;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const g = self.g;
        const dt = self.dt;
        const r = shape.rows;
        self.p = try host.uniform(a, rng, r * g.proj(), -2, 2, dt);
        self.cs = try host.uniform(a, rng, (g.kc - 1) * g.cd(), -2, 2, dt);
        self.s0 = try host.uniform(a, rng, g.h * g.dh * g.ds, -1, 1, .float32);
        self.cw = try host.uniform(a, rng, g.cd() * g.kc, -0.6, 0.6, dt);
        self.cb = try host.uniform(a, rng, g.cd(), -0.2, 0.2, dt);
        self.alog = try host.uniform(a, rng, g.h, -1, 1.5, dt);
        self.dtb = try host.uniform(a, rng, g.h, -3, 0, dt);
        self.dskip = try host.uniform(a, rng, g.h, 0.5, 1.5, dt);
        self.lo = DT_LO;
        self.hi = DT_HI;
        switch (fill) {
            .random => {},
            .nonfinite => {
                // NaN and infinities in x's channels (rows 0..3 of the projection) and in the carried state.
                host.plant(self.p, r, g.proj());
                self.s0[g.ds * 5 + 7] = std.math.nan(f64);
            },
            .stress => {
                // Gate 4: every stage exact but the sums. Each conv reads only its current row (weight 1 on the last
                // tap, bias 0) of integers past 17, where SiLU is the identity in f32; dt clamps to 1, A_log -100
                // makes A 0 (dA 1), D 1, z past 17 (SiLU(z) = z): the state sums x * B over the rows, y sums s * C.
                // The state's registration takes products past fp16's integers; y's keeps its sum exact in fp32.
                const top: i32 = if (self.part == .state) 64 else 24;
                for (self.p) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, 17, top));
                for (self.s0) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -8, 8));
                @memset(self.cb, 0);
                for (0..g.cd()) |c| for (0..g.kc) |k| {
                    self.cw[c * g.kc + k] = if (k == g.kc - 1) 1 else 0;
                };
                @memset(self.alog, -100);
                @memset(self.dskip, 1);
                self.lo = 1;
                self.hi = 1;
            },
        }
        try self.bind();
        return true;
    }

    fn bind(self: *Step) !void {
        const rig = &self.rig;
        rig.reset();
        const g = self.g;
        const dt = self.dt;
        const r: c_int = @intCast(self.shape.rows);
        const h: c_int = @intCast(g.h);
        self.args = .{
            try rig.input(dt, &.{ 1, r, @intCast(g.proj()) }, self.p),
            try rig.input(dt, &.{ 1, @intCast(g.kc - 1), @intCast(g.cd()) }, self.cs),
            try rig.input(.float32, &.{ 1, h, @intCast(g.dh), @intCast(g.ds) }, self.s0),
            try rig.input(dt, &.{ @intCast(g.cd()), @intCast(g.kc), 1 }, self.cw),
            try rig.input(dt, &.{@intCast(g.cd())}, self.cb),
            try rig.input(dt, &.{h}, self.alog),
            try rig.input(dt, &.{h}, self.dtb),
            try rig.input(dt, &.{h}, self.dskip),
            try rig.scalar(@floatCast(self.lo)),
            try rig.scalar(@floatCast(self.hi)),
            try host.intScalar(rig, r),
        };
    }

    pub fn run(self: *Step) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    fn stateRows(self: *const Step) usize {
        return if (self.all_rows) self.shape.rows else 1;
    }

    pub fn count(self: *const Step) usize {
        const g = self.g;
        return switch (self.part) {
            .y_conv => self.shape.rows * g.xd() + (g.kc - 1) * g.cd(),
            .state => self.stateRows() * g.h * g.dh * g.ds,
        };
    }

    /// Input `ch` of the conv at window position `pos` (negative: the carried conv state).
    fn tap(self: *const Step, ch: usize, pos: isize) f64 {
        const g = self.g;
        const kc1: isize = @intCast(g.kc - 1);
        if (pos < 0) return self.cs[@as(usize, @intCast(pos + kc1)) * g.cd() + ch];
        return self.p[@as(usize, @intCast(pos)) * g.proj() + g.xd() + ch];
    }

    /// SiLU(T(conv)) at channel `ch`, row `rr`, rounded to T as the device stores it.
    fn convSilu(self: *const Step, ch: usize, rr: usize) Val {
        const g = self.g;
        var a = self.cb[ch];
        var m = @abs(a);
        for (0..g.kc) |k| {
            const t = self.cw[ch * g.kc + k] * self.tap(ch, @as(isize, @intCast(rr + k)) - @as(isize, @intCast(g.kc - 1)));
            a += t;
            m += @abs(t);
        }
        const cv = host.round(self.dt, .{ .v = a, .e = spec.gamma(@floatFromInt(g.kc), U) * m });
        return host.round(self.dt, host.silu(cv));
    }

    /// Every output in float64 with its allowance; the SSM state's and y's sums in `acc` (null: float64) in `order`.
    fn outputsIn(self: *const Step, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const g = self.g;
        const dt = self.dt;
        const a = self.rig.a;
        const r = self.shape.rows;
        const ns = g.ds / 32;
        const st = try a.alloc(Val, g.h * g.dh * g.ds);
        defer a.free(st);
        for (st, self.s0) |*o, v| o.* = .exact(v);
        const bc = try a.alloc(Val, g.ng * 2 * g.ds);
        defer a.free(bc);
        const terms = try a.alloc(f64, g.ds);
        defer a.free(terms);
        const y_len = r * g.xd();
        for (0..r) |rr| {
            for (0..g.ng) |gi| for (0..2 * g.ds) |c| {
                const ch = if (c < g.ds) g.xd() + gi * g.ds + c else g.xd() + g.ng * g.ds + gi * g.ds + (c - g.ds);
                bc[gi * 2 * g.ds + c] = self.convSilu(ch, rr);
            };
            for (0..g.h) |h| {
                const gi = h / (g.h / g.ng);
                const B = bc[gi * 2 * g.ds ..][0..g.ds];
                const C = bc[gi * 2 * g.ds + g.ds ..][0..g.ds];
                const ex = @exp(self.alog[h]);
                const A = host.round(dt, .{ .v = ex, .e = ex * host.fastExp(self.alog[h]) });
                const dtv = self.softplusDt(self.p[rr * g.proj() + g.xd() + g.cd() + h] + self.dtb[h]);
                const arg = Val{ .v = -A.v * dtv.v, .e = @abs(dtv.v) * A.e + @abs(A.v) * dtv.e + U * @abs(A.v * dtv.v) };
                const da_v = @exp(arg.v);
                const dA = Val{ .v = da_v, .e = da_v * (arg.e + host.fastExp(arg.v)) };
                for (0..g.dh) |d| {
                    const cx = h * g.dh + d;
                    const x = self.convSilu(cx, rr);
                    const xdt = Val{ .v = x.v * dtv.v, .e = @abs(dtv.v) * x.e + @abs(x.v) * dtv.e + U * @abs(x.v * dtv.v) };
                    const row = st[(h * g.dh + d) * g.ds ..][0..g.ds];
                    var acc_e: f64 = 0;
                    var acc_m: f64 = 0;
                    for (row, B, C, terms) |*s, b, cc, *t| {
                        const keep = dA.v * s.v;
                        const add = xdt.v * b.v;
                        const nv = if (acc) |ac| spec.sumIn(ac, .sequential, &.{ keep, add }) else keep + add;
                        const old = s.*;
                        s.* = .{ .v = nv, .e = @abs(dA.v) * old.e + @abs(old.v) * dA.e + @abs(b.v) * xdt.e + @abs(xdt.v) * b.e + spec.gamma(2, U) * (@abs(keep) + @abs(add)) };
                        t.* = s.v * cc.v;
                        acc_e += @abs(cc.v) * s.e + @abs(s.v) * cc.e;
                        acc_m += @abs(t.*);
                    }
                    const sum = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms);
                    acc_e += spec.gamma(@floatFromInt(ns + 5), U) * acc_m;
                    const dx = host.round(dt, .{ .v = x.v * self.dskip[h], .e = @abs(self.dskip[h]) * x.e });
                    const y = host.round(dt, .{ .v = sum + dx.v, .e = acc_e + dx.e + U * @abs(sum + dx.v) });
                    const z = self.p[rr * g.proj() + cx];
                    const sz = host.round(dt, host.silu(.exact(z)));
                    if (self.part == .y_conv) {
                        want[rr * g.xd() + cx] = sz.v * y.v;
                        if (mag) |mm| mm[rr * g.xd() + cx] = host.allowance(@abs(sz.v) * y.e + @abs(y.v) * sz.e);
                    }
                }
            }
            if (self.part == .state and (self.all_rows or rr == r - 1)) {
                const at = (if (self.all_rows) rr else 0) * st.len;
                for (st, 0..) |s, i| {
                    want[at + i] = s.v;
                    if (mag) |mm| mm[at + i] = host.allowance(s.e);
                }
            }
        }
        if (self.part == .y_conv) {
            // The conv state after the last row: its last KC-1 inputs, copied.
            for (0..g.kc - 1) |k| for (0..g.cd()) |ch| {
                const i = y_len + k * g.cd() + ch;
                want[i] = self.tap(ch, @as(isize, @intCast(r + k)) - @as(isize, @intCast(g.kc - 1)));
                if (mag) |mm| mm[i] = 0;
            };
        }
    }

    /// softplus(dt + bias) clamped to [lo, hi], with fast exp and log's error.
    fn softplusDt(self: *const Step, raw_in: f64) Val {
        const raw = Val{ .v = raw_in, .e = U * @abs(raw_in) };
        const e = @exp(-@abs(raw.v));
        const sp = @max(raw.v, 0) + std.math.log1p(e);
        // log(1 + e) on (1, 2]: fast log's absolute error 2^-21; exp's relative error carried through log1p.
        const err = raw.e + e * host.fastExp(raw.v) + 0x1p-21 + U * (1 + @abs(sp));
        return .{ .v = std.math.clamp(sp, self.lo, self.hi), .e = err };
    }

    pub fn reference(self: *const Step, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Step, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    /// The shipped wrapper on the same inputs: all three outputs' bytes.
    fn wrapper(self: *Step) ![]const u8 {
        const g = self.g;
        const x = self.args;
        const geo: m2.Geometry = .{ .batch = 1, .rows = @intCast(self.shape.rows), .heads = @intCast(g.h), .groups = @intCast(g.ng), .head_dim = @intCast(g.dh), .state = @intCast(g.ds), .conv_kernel = @intCast(g.kc), .proj = @intCast(g.proj()) };
        const out = (try m2.step(geo, .{ .proj = x[0], .conv_state = x[1], .ssm_state = x[2], .conv_w = x[3], .conv_b = x[4], .A_log = x[5], .dt_bias = x[6], .D = x[7], .dt_lo = x[8], .dt_hi = x[9], .rows = x[10], .all_rows = self.all_rows }, self.rig.stream())) orelse return error.Declined;
        defer for ([_]mlx.mlx_array{ out.y, out.ssm_state, out.conv_state }) |o| {
            _ = mlx.mlx_array_free(o);
        };
        return self.rig.bytesOf(&.{ out.y, out.ssm_state, out.conv_state });
    }
};

/// The grouped RMS norm: a shape's rows are the rows, `k` the group width.
const Gnorm = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    groups: usize,
    dt: mlx.mlx_dtype = .bfloat16,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    w: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, groups: usize) Gnorm {
        return .{ .rig = metal.Rig.init(a), .launch = &gnorm_launch, .groups = groups, .arena = .init(a) };
    }

    pub fn deinit(self: *Gnorm) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn xd(self: *const Gnorm, shape: Shape) usize {
        return self.groups * shape.k;
    }

    pub fn geometry(self: *const Gnorm, shape: Shape) ?metal.Geometry {
        const gs = shape.k;
        if (gs % 128 != 0 or gs > 4096) return null;
        const tn: c_int = @intCast(gs / 4);
        var g = metal.Geometry{ .grid = .{ tn, @intCast(self.groups), @intCast(shape.rows) }, .tg = .{ tn, 1, 1 } };
        g.out(&.{ 1, @intCast(shape.rows), @intCast(self.xd(shape)) }, self.dt);
        g.dtype("T", self.dt);
        g.int("XD", self.xd(shape));
        g.int("GS", gs);
        return g;
    }

    pub fn prepare(self: *Gnorm, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * self.xd(shape);
        self.x = try host.uniform(a, rng, n, -3, 3, self.dt);
        self.w = try host.uniform(a, rng, self.xd(shape), 0.5, 1.5, self.dt);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.x, n / shape.k, shape.k),
            .stress => {
                for (self.x) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                @memset(self.w, 1);
            },
        }
        self.rig.reset();
        const r: c_int = @intCast(shape.rows);
        self.args = .{
            try self.rig.input(self.dt, &.{ 1, r, @intCast(self.xd(shape)) }, self.x),
            try self.rig.input(self.dt, &.{@intCast(self.xd(shape))}, self.w),
            try self.rig.scalar(EPS),
        };
        return true;
    }

    pub fn run(self: *Gnorm) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Gnorm) usize {
        return self.shape.rows * self.xd(self.shape);
    }

    fn outputsIn(self: *const Gnorm, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const gs = self.shape.k;
        const sq = try self.rig.a.alloc(f64, gs);
        defer self.rig.a.free(sq);
        var at: usize = 0;
        while (at < want.len) : (at += gs) {
            const x = self.x[at..][0..gs];
            for (sq, x) |*q, v| q.* = v * v;
            const ss = if (acc) |ac| spec.sumIn(ac, order, sq) else host.sum(sq);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(gs)) + EPS);
            const c0 = at % self.xd(self.shape);
            for (want[at..][0..gs], x, 0..) |*o, v, i| o.* = self.w[c0 + i] * v * inv;
            if (mag) |m| for (m[at..][0..gs], want[at..][0..gs]) |*o, y| {
                o.* = @abs(y) / 2;
            };
        }
    }

    pub fn reference(self: *const Gnorm, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Gnorm, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    fn wrapper(self: *Gnorm) ![]const u8 {
        const out = (try m2.groupNorm(self.args[0], self.args[1], self.args[2], @intCast(self.groups), self.rig.stream())) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(out);
        return self.rig.bytesOf(&.{out});
    }
};

/// The squares' sum: four per thread, the simdgroup tree, then the partials in order.
fn gnormDepth(shape: Shape) usize {
    return 4 + 5 + shape.k / 128;
}

const step_rows: []const Shape = &.{ .{ .rows = 1, .n = 1, .k = 128 }, .{ .rows = 4, .n = 1, .k = 128 }, .{ .rows = 16, .n = 1, .k = 128 }, .{ .rows = 33, .n = 1, .k = 128 } };
const capture_rows: []const Shape = &.{ .{ .rows = 4, .n = 1, .k = 128 }, .{ .rows = 16, .n = 1, .k = 128 } };

fn stepSpec(comptime name: []const u8, part: Part, served: []const Shape, tiled: bool) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        // The reference carries every rounding as its allowance (`host.allowance`).
        .depth = host.depthOne,
        .out = if (part == .state) .f32 else .bf16,
        .tiles = .{ if (tiled) .{ .size = 16 } else .{}, .{}, .{} },
        .served = served,
        // y's sum over the state's columns: integers past fp16's range. A bf16 accumulator misses by less than y's
        // own two bf16 roundings; the state sums one product a row.
        .sums = if (part == .state) .{ .f16 = 100, .bf16 = 100 } else .{ .f16 = 100, .bf16 = 0, .why = "y is rounded to bf16 twice after its sum" },
    };
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(Step, .init(a, nano, .y_conv, false), stepSpec("mamba2_step bf16 (y, conv state)", .y_conv, step_rows, true), null);
    try set.add(Step, .init(a, nano, .state, false), stepSpec("mamba2_step bf16 (SSM state, last row)", .state, step_rows, true), null);
    try set.add(Step, .init(a, nano, .state, true), stepSpec("mamba2_step bf16 (SSM state, every row)", .state, capture_rows, false), null);
    inline for (.{ .{ "mamba2_gnorm bf16 (GS 512)", 512 }, .{ "mamba2_gnorm bf16 (GS 1024)", 1024 } }) |r| {
        try set.add(Gnorm, .init(a, 8), .{
            .name = r[0],
            .accum = .f32,
            .depth = gnormDepth,
            // The mean, + eps, fast rsqrt (2 ulps), the product with the row, the weight product.
            .ops = 6,
            .mids = 1,
            .out = .bf16,
            .served = comptime &.{ .{ .rows = 1, .n = 1, .k = r[1] }, .{ .rows = 16, .n = 1, .k = r[1] }, .{ .rows = 256, .n = 1, .k = r[1] } },
            .sums = .{ .f16 = 32, .bf16 = 0, .why = "positive squares: a bf16 sum stays near the output's rounding" },
        }, null);
    }
}

// ── tests ──

test "kernel contract ssm: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const shape = k.spec.served[k.spec.served.len - 1];
        var ours: []u8 = undefined;
        var theirs: []const u8 = undefined;
        if (std.mem.startsWith(u8, k.spec.name, "mamba2_step")) {
            const t: *Step = @ptrCast(@alignCast(k.ctx));
            try std.testing.expect(try t.prepare(shape, .random, 7));
            var geo = t.geometry(shape).?;
            geo.unread = 0;
            try t.rig.launch(t.launch, geo, &t.args);
            ours = try a.dupe(u8, t.rig.outputBytes());
            theirs = try t.wrapper();
        } else {
            const t: *Gnorm = @ptrCast(@alignCast(k.ctx));
            try std.testing.expect(try t.prepare(shape, .random, 7));
            try t.run();
            ours = try a.dupe(u8, t.rig.outputBytes());
            theirs = try t.wrapper();
        }
        defer a.free(ours);
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}
