//! Idol's kernel contract: the denoiser block's fused q/k RoPE + per-head RMS norm + head-major layout (bit-equal to
//! the chain it replaces) and the fc1 bias add + tanh GELU. Listed in `kernel_contract/registry.zig`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const L = @import("layers.zig");
const den = @import("denoiser.zig");
const glue = @import("glue.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const HD = 64;
const PAIRS = HD / 2;
const EPS: f32 = L.HEAD_NORM_EPS;

// ── q/k RoPE + RMS norm + head-major layout ──

const qkv_f16: metal.Launch = .{ .name = "msv_idol_qkv_prep", .inputs = &.{ "Y", "B", "COS", "SIN", "QG", "KG", "eps" }, .outputs = &.{ "OQ", "OK", "OV" }, .source = glue.QKV_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// y [rows, 3*H*64] + bias -> q, k, v [1, H, rows, 64]: `rows` is the tokens, `n` the heads, `k` the head width (the
/// norm's sum).
const QkvPrep = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    y: []f64 = &.{},
    b: []f64 = &.{},
    cos: []f64 = &.{},
    sin: []f64 = &.{},
    qg: []f64 = &.{},
    kg: []f64 = &.{},
    args: [7]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) QkvPrep {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *QkvPrep) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const QkvPrep, shape: Shape) ?metal.Geometry {
        if (shape.k != HD or shape.n == 0) return null;
        const h: c_int = @intCast(shape.n);
        const seq: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ 256 * @divTrunc(3 * h + 7, 8), seq, 1 }, .tg = .{ 256, 1, 1 } };
        for (0..3) |_| g.out(&.{ 1, h, seq, HD }, self.dt);
        g.dtype("T", self.dt);
        g.int("H", h);
        return g;
    }

    pub fn prepare(self: *QkvPrep, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const width = 3 * shape.n * HD;
        const table = shape.rows * shape.n * PAIRS;
        self.y = try host.uniform(a, rng, shape.rows * width, -3, 3, self.dt);
        self.b = try host.uniform(a, rng, width, -1, 1, self.dt);
        self.cos = try a.alloc(f64, table);
        self.sin = try a.alloc(f64, table);
        for (self.cos, self.sin) |*c, *s| {
            const ang = 6.3 * rng.float(f64);
            c.* = spec.roundTo(.f32, @cos(ang));
            s.* = spec.roundTo(.f32, @sin(ang));
        }
        self.qg = try host.uniform(a, rng, shape.n * HD, 0.5, 1.5, .float32);
        self.kg = try host.uniform(a, rng, shape.n * HD, 0.5, 1.5, .float32);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.y, shape.rows, width),
            .stress => {
                // Gate 4: integer rows, sums of squares exact in fp32; unit gammas and the identity rotation, so the
                // norm is the only rounding left.
                for (self.y) |*v| v.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32);
                @memset(self.b, 0);
                @memset(self.qg, 1);
                @memset(self.kg, 1);
                @memset(self.cos, 1);
                @memset(self.sin, 0);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const seq: c_int = @intCast(shape.rows);
        const h: c_int = @intCast(shape.n);
        self.args = .{
            try rig.input(self.dt, &.{ seq, @intCast(width) }, self.y),
            try rig.input(self.dt, &.{@intCast(width)}, self.b),
            try rig.input(.float32, &.{ seq, h, PAIRS }, self.cos),
            try rig.input(.float32, &.{ seq, h, PAIRS }, self.sin),
            try rig.input(.float32, &.{ h, HD }, self.qg),
            try rig.input(.float32, &.{ h, HD }, self.kg),
            try rig.scalar(EPS),
        };
        return true;
    }

    pub fn run(self: *QkvPrep) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const QkvPrep) usize {
        return 3 * self.shape.rows * self.shape.n * HD;
    }

    /// Every output, the norm's sum of squares run in `acc` and `order` (null: float64).
    fn outputsIn(self: *const QkvPrep, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) void {
        const e = host.elemOf(self.dt);
        const seq = self.shape.rows;
        const heads = self.shape.n;
        var rot: [HD]f64 = undefined;
        var sq: [HD]f64 = undefined;
        var src: [HD]f64 = undefined;
        for (0..seq) |r| for (0..3) |which| for (0..heads) |h| {
            const col = (which * heads + h) * HD;
            for (&src, self.y[r * 3 * heads * HD + col ..][0..HD], self.b[col..][0..HD]) |*o, y, b| o.* = spec.roundTo(e, y + b);
            const at = ((which * heads + h) * seq + r) * HD;
            if (which == 2) {
                @memcpy(want[at..][0..HD], &src);
                if (mag) |m| for (m[at..][0..HD], src) |*mo, x| {
                    mo.* = @abs(x);
                };
                continue;
            }
            const tab = (r * heads + h) * PAIRS;
            for (0..PAIRS) |i| {
                const c = self.cos[tab + i];
                const s = self.sin[tab + i];
                rot[2 * i] = spec.roundTo(e, src[2 * i] * c - src[2 * i + 1] * s);
                rot[2 * i + 1] = spec.roundTo(e, src[2 * i] * s + src[2 * i + 1] * c);
            }
            for (&sq, rot) |*q, x| q.* = x * x;
            const ss = if (acc) |ac| spec.sumIn(ac, order, &sq) else blk: {
                var t: f64 = 0;
                for (sq) |q| t += q;
                break :blk t;
            };
            const inv = 1 / @sqrt(ss / HD + EPS);
            const g = (if (which == 0) self.qg else self.kg)[h * HD ..][0..HD];
            for (want[at..][0..HD], rot, g) |*o, x, gi| o.* = x * inv * gi;
            if (mag) |m| for (m[at..][0..HD], want[at..][0..HD]) |*mo, o| {
                mo.* = @abs(o);
            };
        };
    }

    pub fn reference(self: *const QkvPrep, want: []f64, mag: []f64) !void {
        self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const QkvPrep, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        self.outputsIn(out, null, acc, order);
    }

    /// The denoiser's chain: the bias add, split, `applyRope`, `headNorm`, head-major transpose.
    pub fn replaced(self: *QkvPrep) ![]const u8 {
        var sc = L.Scope.init(self.rig.a, self.rig.stream());
        defer sc.deinit();
        const seq: c_int = @intCast(self.shape.rows);
        const h: c_int = @intCast(self.shape.n);
        const x = self.args;
        const y = try sc.reshape(try sc.add(x[0], x[1]), &.{ 1, seq, 3 * h * HD });
        const cos = try sc.reshape(x[2], &.{ 1, seq, h, PAIRS });
        const sin = try sc.reshape(x[3], &.{ 1, seq, h, PAIRS });
        const unit: [HD]f32 = @splat(1);
        const ones = try sc.fromF32(&unit, &.{HD});
        var outs: [3]mlx.mlx_array = undefined;
        for (&outs, 0..) |*o, i| {
            const part = try L.part(&sc, y, 3, @intCast(i), h, HD);
            const ready = if (i == 2) part else try L.headNorm(&sc, try den.applyRope(&sc, part, cos, sin), x[4 + i], ones);
            o.* = try sc.transpose(ready, &.{ 0, 2, 1, 3 });
        }
        return self.rig.bytesOf(&outs);
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *QkvPrep) ![]const u8 {
        const x = self.args;
        const got = (try glue.qkvPrep(x[0], x[1], x[4], x[5], x[2], x[3], EPS, @intCast(self.shape.n), self.rig.stream())) orelse return error.Declined;
        defer for ([_]mlx.mlx_array{ got.q, got.k, got.v }) |o| {
            _ = mlx.mlx_array_free(o);
        };
        return self.rig.bytesOf(&.{ got.q, got.k, got.v });
    }
};

// ── bias add + tanh GELU ──

const gelu_f16: metal.Launch = .{ .name = "msv_idol_bias_gelu", .inputs = &.{ "Y", "B" }, .outputs = &.{"O"}, .source = glue.GELU_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// y [rows, N] + b [N] -> gelu_tanh: `k` is N.
const BiasGelu = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    y: []f64 = &.{},
    b: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) BiasGelu {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *BiasGelu) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const BiasGelu, shape: Shape) ?metal.Geometry {
        const rows: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.k);
        if (n == 0 or @rem(n, 4) != 0) return null;
        var g = metal.Geometry{ .grid = .{ @divExact(rows * n, 4), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ rows, n }, self.dt);
        g.dtype("T", self.dt);
        return g;
    }

    pub fn prepare(self: *BiasGelu, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.y = try host.uniform(a, rng, shape.rows * shape.k, -6, 6, self.dt);
        self.b = try host.uniform(a, rng, shape.k, -2, 2, self.dt);
        if (fill == .nonfinite) host.plant(self.y, shape.rows, shape.k);
        self.rig.reset();
        self.args = .{
            try self.rig.input(self.dt, &.{ @intCast(shape.rows), @intCast(shape.k) }, self.y),
            try self.rig.input(self.dt, &.{@intCast(shape.k)}, self.b),
        };
        return true;
    }

    pub fn run(self: *BiasGelu) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const BiasGelu) usize {
        return self.shape.rows * self.shape.k;
    }

    /// The sum rounded to T, then `0.5 x (1 + tanh a)` = `x / (1 + e^(-2a))` with a = sqrt(2/pi) (x + 0.044715 x^3).
    pub fn reference(self: *const BiasGelu, want: []f64, mag: []f64) !void {
        const e = host.elemOf(self.dt);
        const n = self.shape.k;
        for (0..self.shape.rows) |r| for (0..n) |i| {
            const x = spec.roundTo(e, self.y[r * n + i] + self.b[i]);
            const a = 0.7978845608028654 * (x + 0.044715 * x * x * x);
            const out = x / (1 + @exp(-2 * a));
            want[r * n + i] = out;
            mag[r * n + i] = @abs(out);
        };
    }
};

// ── bias, gate and residual add ──

const gate_f16: metal.Launch = .{ .name = "msv_idol_gate_residual", .inputs = &.{ "X", "Y", "B", "G" }, .outputs = &.{"O"}, .source = glue.GATE_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// x + (y + b) * g, x and y [rows, N], b and g [N]: `k` is N.
const GateResidual = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    x: []f64 = &.{},
    y: []f64 = &.{},
    b: []f64 = &.{},
    g: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) GateResidual {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *GateResidual) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const GateResidual, shape: Shape) ?metal.Geometry {
        const rows: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.k);
        if (n == 0 or @rem(n, 4) != 0) return null;
        var g = metal.Geometry{ .grid = .{ @divExact(rows * n, 4), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ rows, n }, self.dt);
        g.dtype("T", self.dt);
        return g;
    }

    pub fn prepare(self: *GateResidual, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try host.uniform(a, rng, shape.rows * shape.k, -4, 4, self.dt);
        self.y = try host.uniform(a, rng, shape.rows * shape.k, -4, 4, self.dt);
        self.b = try host.uniform(a, rng, shape.k, -1, 1, self.dt);
        self.g = try host.uniform(a, rng, shape.k, -2, 2, self.dt);
        if (fill == .nonfinite) host.plant(self.y, shape.rows, shape.k);
        self.rig.reset();
        const rows: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.k);
        self.args = .{
            try self.rig.input(self.dt, &.{ rows, n }, self.x),
            try self.rig.input(self.dt, &.{ rows, n }, self.y),
            try self.rig.input(self.dt, &.{n}, self.b),
            try self.rig.input(self.dt, &.{n}, self.g),
        };
        return true;
    }

    pub fn run(self: *GateResidual) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const GateResidual) usize {
        return self.shape.rows * self.shape.k;
    }

    /// The bias sum and the gated product rounded to T, the residual sum left exact.
    pub fn reference(self: *const GateResidual, want: []f64, mag: []f64) !void {
        const e = host.elemOf(self.dt);
        const n = self.shape.k;
        for (0..self.shape.rows) |r| for (0..n) |i| {
            const yg = spec.roundTo(e, spec.roundTo(e, self.y[r * n + i] + self.b[i]) * self.g[i]);
            want[r * n + i] = self.x[r * n + i] + yg;
            mag[r * n + i] = @max(@abs(self.x[r * n + i]), @abs(yg));
        };
    }

    /// The denoiser's chain: add, multiply, add.
    pub fn replaced(self: *GateResidual) ![]const u8 {
        var sc = L.Scope.init(self.rig.a, self.rig.stream());
        defer sc.deinit();
        const x = self.args;
        return self.rig.bytesOf(&.{try sc.add(x[0], try sc.mul(try sc.add(x[1], x[2]), x[3]))});
    }
};

// ── LayerNorm + adaLN modulation ──

const norm_mod_f16: metal.Launch = .{ .name = "msv_idol_norm_mod", .inputs = &.{ "X", "SC", "SH", "eps" }, .outputs = &.{"O"}, .source = glue.NORM_MOD_SOURCE, .tg_bytes = 128, .m1 = 1024, .m2 = 1024 };

/// LayerNorm(x) * (1 + scale) + shift, x [rows, N], scale and shift [N]: `k` is N.
const NormMod = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    x: []f64 = &.{},
    sc: []f64 = &.{},
    sh: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) NormMod {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *NormMod) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const NormMod, shape: Shape) ?metal.Geometry {
        const rows: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.k);
        if (n == 0 or @rem(n, 128) != 0 or n > 4096) return null;
        var g = metal.Geometry{ .grid = .{ rows * @divExact(n, 4), 1, 1 }, .tg = .{ @divExact(n, 4), 1, 1 } };
        g.out(&.{ rows, n }, self.dt);
        g.dtype("T", self.dt);
        return g;
    }

    pub fn prepare(self: *NormMod, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.k;
        self.x = try host.uniform(a, rng, shape.rows * n, -3, 5, self.dt);
        self.sc = try host.uniform(a, rng, n, -0.5, 0.5, self.dt);
        self.sh = try host.uniform(a, rng, n, -0.5, 0.5, self.dt);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.x, shape.rows, n),
            .stress => {
                // Gate 4: integer rows of mean zero (each value beside its negation), so both sums are exact in fp32.
                for (0..shape.rows) |r| for (0..n / 2) |i| {
                    const v: f64 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32);
                    self.x[r * n + 2 * i] = v;
                    self.x[r * n + 2 * i + 1] = -v;
                };
                @memset(self.sc, 0);
                @memset(self.sh, 0);
            },
        }
        self.rig.reset();
        self.args = .{
            try self.rig.input(self.dt, &.{ @intCast(shape.rows), @intCast(n) }, self.x),
            try self.rig.input(self.dt, &.{@intCast(n)}, self.sc),
            try self.rig.input(self.dt, &.{@intCast(n)}, self.sh),
            try self.rig.scalar(L.LN_EPS),
        };
        return true;
    }

    pub fn run(self: *NormMod) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const NormMod) usize {
        return self.shape.rows * self.shape.k;
    }

    /// Every output: the mean and the sum of squares in float64 (`acc` null) or in `acc` and `order` (gate 4).
    fn outputsIn(self: *const NormMod, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const e = host.elemOf(self.dt);
        const n = self.shape.k;
        const nf: f64 = @floatFromInt(n);
        const d = try self.arena.child_allocator.alloc(f64, n);
        defer self.arena.child_allocator.free(d);
        for (0..self.shape.rows) |r| {
            const row = self.x[r * n ..][0..n];
            const mean = (if (acc) |ac| spec.sumIn(ac, order, row) else sumF64(row)) / nf;
            for (d, row) |*o, v| o.* = (v - mean) * (v - mean);
            const ss = if (acc) |ac| spec.sumIn(ac, order, d) else sumF64(d);
            const inv = 1 / @sqrt(ss / nf + L.LN_EPS);
            for (0..n) |i| {
                const s1 = spec.roundTo(e, self.sc[i] + 1);
                const out = (row[i] - mean) * inv * s1 + self.sh[i];
                want[r * n + i] = out;
                if (mag) |m| m[r * n + i] = @max(@abs(out), @abs((row[i] - mean) * inv * s1));
            }
        }
    }

    pub fn reference(self: *const NormMod, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const NormMod, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }
};

// ── RePo LayerNorm ──

const repo_norm_f16: metal.Launch = .{ .name = "msv_idol_repo_norm", .inputs = &.{ "X", "W", "BI", "eps" }, .outputs = &.{"O"}, .source = glue.REPO_NORM_SOURCE, .tg_bytes = 128, .m1 = 1024, .m2 = 1024 };

/// LayerNorm(x) * w + b in f32, x [rows, N] f16, w and b [N] f32: `k` is N. The unsplit arm; the split one is its
/// output's TF32 halves (`idol: the split RePo norm is its output's TF32 halves`).
const RepoNorm = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    x: []f64 = &.{},
    w: []f64 = &.{},
    b: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) RepoNorm {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *RepoNorm) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const RepoNorm, shape: Shape) ?metal.Geometry {
        _ = self;
        const rows: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.k);
        if (n == 0 or @rem(n, 128) != 0 or n > 4096) return null;
        var g = metal.Geometry{ .grid = .{ rows * @divExact(n, 4), 1, 1 }, .tg = .{ @divExact(n, 4), 1, 1 } };
        g.out(&.{ rows, n }, .float32);
        g.dtype("T", .float16);
        g.flag("SPLIT", false);
        return g;
    }

    pub fn prepare(self: *RepoNorm, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.k;
        self.x = try host.uniform(a, rng, shape.rows * n, -3, 5, self.dt);
        self.w = try host.uniform(a, rng, n, 0.5, 1.5, .float32);
        self.b = try host.uniform(a, rng, n, -0.5, 0.5, .float32);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.x, shape.rows, n),
            .stress => {
                // Gate 4: integer rows of mean zero (each value beside its negation), so both sums are exact in fp32.
                for (0..shape.rows) |r| for (0..n / 2) |i| {
                    const v: f64 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32);
                    self.x[r * n + 2 * i] = v;
                    self.x[r * n + 2 * i + 1] = -v;
                };
                @memset(self.w, 1);
                @memset(self.b, 0);
            },
        }
        self.rig.reset();
        self.args = .{
            try self.rig.input(self.dt, &.{ @intCast(shape.rows), @intCast(n) }, self.x),
            try self.rig.input(.float32, &.{@intCast(n)}, self.w),
            try self.rig.input(.float32, &.{@intCast(n)}, self.b),
            try self.rig.scalar(1e-5),
        };
        return true;
    }

    pub fn run(self: *RepoNorm) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const RepoNorm) usize {
        return self.shape.rows * self.shape.k;
    }

    fn outputsIn(self: *const RepoNorm, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const n = self.shape.k;
        const nf: f64 = @floatFromInt(n);
        const d = try self.arena.child_allocator.alloc(f64, n);
        defer self.arena.child_allocator.free(d);
        for (0..self.shape.rows) |r| {
            const row = self.x[r * n ..][0..n];
            const mean = (if (acc) |ac| spec.sumIn(ac, order, row) else sumF64(row)) / nf;
            for (d, row) |*o, v| o.* = (v - mean) * (v - mean);
            const ss = if (acc) |ac| spec.sumIn(ac, order, d) else sumF64(d);
            const inv = 1 / @sqrt(ss / nf + 1e-5);
            for (0..n) |i| {
                const scaled = (row[i] - mean) * inv * self.w[i];
                want[r * n + i] = scaled + self.b[i];
                if (mag) |m| m[r * n + i] = @max(@abs(want[r * n + i]), @abs(scaled));
            }
        }
    }

    pub fn reference(self: *const RepoNorm, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const RepoNorm, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }
};

// ── RePo angles ──

const repo_angles_f32: metal.Launch = .{ .name = "msv_idol_repo_angles", .inputs = &.{ "GC", "WF", "FQ", "AXIS" }, .outputs = &.{ "OC", "OS" }, .source = glue.REPO_ANGLES_SOURCE, .tg_bytes = 4864, .m1 = 1024, .m2 = 1024 };

/// The pairs' axes as a head of 64 splits them: 10, 10, 12.
const AXIS = blk: {
    var a: [32]i32 = undefined;
    for (&a, 0..) |*v, i| v.* = if (i < 10) 0 else if (i < 20) 1 else 2;
    break :blk a;
};

/// gate | content [rows, 2K], final map [K, 3H] -> cos, sin [rows, H, 32]: `n` is H, `k` is K (the sum).
const RepoAngles = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    g: []f64 = &.{},
    c: []f64 = &.{},
    wf: []f64 = &.{},
    fq: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch) RepoAngles {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .arena = .init(a) };
    }

    pub fn deinit(self: *RepoAngles) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const RepoAngles, shape: Shape) ?metal.Geometry {
        _ = self;
        if (shape.k == 0 or shape.k > 1024 or shape.n == 0 or shape.n > 64) return null;
        const rows: c_int = @intCast(shape.rows);
        const h: c_int = @intCast(shape.n);
        var g = metal.Geometry{ .grid = .{ rows * 128, 1, 1 }, .tg = .{ 128, 1, 1 } };
        for (0..2) |_| g.out(&.{ rows, h, 32 }, .float32);
        g.int("H", h);
        return g;
    }

    pub fn prepare(self: *RepoAngles, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = shape.k;
        const d = 3 * shape.n;
        self.g = try host.uniform(a, rng, shape.rows * k, -4, 4, .float32);
        self.c = try host.uniform(a, rng, shape.rows * k, -2, 2, .float32);
        self.wf = try host.uniform(a, rng, k * d, -0.3, 0.3, .float32);
        self.fq = try host.uniform(a, rng, 32, -1, 1, .float32);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.g, shape.rows, k),
            .stress => {
                // Gate 4: silu(33) is 33 in f32, so integer content and map make every product and sum an integer
                // (33, not 32: multiples of a power of two would stay exact in fp16 too).
                @memset(self.g, 33);
                for (self.c) |*v| v.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 9))) - 4);
                for (self.wf) |*v| v.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 9))) - 4);
                @memset(self.fq, 0x1p-12);
            },
        }
        self.rig.reset();
        const rows: c_int = @intCast(shape.rows);
        const ki: c_int = @intCast(k);
        var axis: [32]f64 = undefined;
        for (&axis, AXIS) |*o, v| o.* = @floatFromInt(v);
        const gc = try a.alloc(f64, 2 * shape.rows * k);
        for (0..shape.rows) |r| {
            @memcpy(gc[2 * r * k ..][0..k], self.g[r * k ..][0..k]);
            @memcpy(gc[(2 * r + 1) * k ..][0..k], self.c[r * k ..][0..k]);
        }
        self.args = .{
            try self.rig.input(.float32, &.{ rows, 2 * ki }, gc),
            try self.rig.input(.float32, &.{ ki, @intCast(d) }, self.wf),
            try self.rig.input(.float32, &.{32}, self.fq),
            try self.rig.input(.int32, &.{32}, &axis),
        };
        return true;
    }

    pub fn run(self: *RepoAngles) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const RepoAngles) usize {
        return 2 * self.shape.rows * self.shape.n * 32;
    }

    /// cos then sin of every pair's angle; the map's sum in `acc` and `order` (null: float64). `mag` is the angle's
    /// own scale, |freq| pi sum |u w|: its error moves both outputs by as much.
    fn outputsIn(self: *const RepoAngles, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const k = self.shape.k;
        const heads = self.shape.n;
        const d = 3 * heads;
        const half = self.shape.rows * heads * 32;
        const al = self.arena.child_allocator;
        const u = try al.alloc(f64, k);
        defer al.free(u);
        const terms = try al.alloc(f64, k);
        defer al.free(terms);
        var delta: [192]f64 = undefined;
        var scale: [192]f64 = undefined;
        for (0..self.shape.rows) |r| {
            for (u, 0..) |*o, i| {
                const g = self.g[r * k + i];
                o.* = g / (1 + @exp(-g)) * self.c[r * k + i];
            }
            for (0..d) |j| {
                var abs: f64 = 0;
                for (terms, 0..) |*t, i| {
                    t.* = u[i] * self.wf[i * d + j];
                    abs += @abs(t.*);
                }
                delta[j] = if (acc) |ac| spec.sumIn(ac, order, terms) else sumF64(terms);
                scale[j] = abs;
            }
            for (0..heads) |h| for (0..32) |p| {
                const j = h * 3 + @as(usize, @intCast(AXIS[p]));
                const ang = delta[j] * self.fq[p] * std.math.pi;
                const at = (r * heads + h) * 32 + p;
                want[at] = @cos(ang);
                want[half + at] = @sin(ang);
                if (mag) |m| {
                    const s = @max(scale[j], @abs(delta[j])) * @abs(self.fq[p]) * std.math.pi;
                    m[at] = s;
                    m[half + at] = s;
                }
            };
        }
    }

    pub fn reference(self: *const RepoAngles, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const RepoAngles, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }
};

// ── BiRefNet's deformable columns ──

const deform_f16: metal.Launch = .{ .name = "msv_idol_deform_cols", .inputs = &.{ "X", "OFF", "MSK" }, .outputs = &.{"O"}, .source = glue.DEFORM_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// x [H*H, C], offsets [H*H, 2K], modulators [H*H, K] -> columns [H*H, K*C]: `rows` is the plane's side H, `n` the
/// kernel size k (K = k*k taps), `k` the channels C.
const DeformCols = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    x: []f64 = &.{},
    off: []f64 = &.{},
    msk: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) DeformCols {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *DeformCols) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const DeformCols, shape: Shape) ?metal.Geometry {
        const side: c_int = @intCast(shape.rows);
        const ks: c_int = @intCast(shape.n);
        const c: c_int = @intCast(shape.k);
        if (side == 0 or ks == 0 or @rem(ks, 2) != 1 or c == 0 or @rem(c, 4) != 0) return null;
        const cx = @min(@divExact(c, 4), 32);
        var g = metal.Geometry{ .grid = .{ @divExact(c, 4), ks * ks, side * side }, .tg = .{ cx, 1, @divTrunc(256, cx) } };
        g.out(&.{ side * side, ks * ks * c }, self.dt);
        g.dtype("T", self.dt);
        g.int("KS", ks);
        g.int("H", side);
        g.int("W", side);
        return g;
    }

    pub fn prepare(self: *DeformCols, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const hw = shape.rows * shape.rows;
        const taps = shape.n * shape.n;
        self.x = try host.uniform(a, rng, hw * shape.k, -3, 3, self.dt);
        // Offsets up to a third of the plane: taps land inside, on the border and outside.
        const reach: f64 = @floatFromInt(@max(shape.rows / 3, 2));
        self.off = try host.uniform(a, rng, hw * 2 * taps, -reach, reach, .float32);
        self.msk = try host.uniform(a, rng, hw * taps, 0, 2, self.dt);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.x, hw, shape.k),
            .stress => {
                // Gate 4: integer samples and half-pixel offsets, so every corner weight is a power of two.
                for (self.x) |*v| v.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 17))) - 8);
                for (self.off) |*v| v.* = 0.5 * @as(f64, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 5))) - 2));
                @memset(self.msk, 1);
            },
        }
        self.rig.reset();
        self.args = .{
            try self.rig.input(self.dt, &.{ @intCast(hw), @intCast(shape.k) }, self.x),
            try self.rig.input(.float32, &.{ @intCast(hw), @intCast(2 * taps) }, self.off),
            try self.rig.input(self.dt, &.{ @intCast(hw), @intCast(taps) }, self.msk),
        };
        return true;
    }

    pub fn run(self: *DeformCols) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const DeformCols) usize {
        return self.shape.rows * self.shape.rows * self.shape.n * self.shape.n * self.shape.k;
    }

    /// Each column: the valid corners' weighted samples summed (in `acc`, null: float64) times the modulator. `mag` is
    /// the corners' magnitude times the modulator: the positions' rounding moves the weights by the plane's ulps.
    fn outputsIn(self: *const DeformCols, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) void {
        const side = self.shape.rows;
        const ks = self.shape.n;
        const taps = ks * ks;
        const c = self.shape.k;
        const pad: f64 = @floatFromInt(ks / 2);
        for (0..side * side) |pos| for (0..taps) |p| {
            const y: f64 = @floatFromInt(pos / side);
            const x: f64 = @floatFromInt(pos % side);
            const py = y + @as(f64, @floatFromInt(p / ks)) - pad + self.off[pos * 2 * taps + 2 * p];
            const px = x + @as(f64, @floatFromInt(p % ks)) - pad + self.off[pos * 2 * taps + 2 * p + 1];
            const fy = @floor(py);
            const fx = @floor(px);
            const m = self.msk[pos * taps + p];
            for (0..c) |ch| {
                var terms: [4]f64 = @splat(0);
                var abs: f64 = 0;
                for (0..2) |dy| for (0..2) |dx| {
                    const yy = fy + @as(f64, @floatFromInt(dy));
                    const xx = fx + @as(f64, @floatFromInt(dx));
                    if (yy < 0 or yy >= @as(f64, @floatFromInt(side)) or xx < 0 or xx >= @as(f64, @floatFromInt(side))) continue;
                    const wy = if (dy == 0) 1 - (py - fy) else py - fy;
                    const wx = if (dx == 0) 1 - (px - fx) else px - fx;
                    const v = self.x[(@as(usize, @intFromFloat(yy)) * side + @as(usize, @intFromFloat(xx))) * c + ch];
                    terms[dy * 2 + dx] = wy * wx * v;
                    abs += @abs(v);
                };
                const s = if (acc) |ac| spec.sumIn(ac, order, &terms) else terms[0] + terms[1] + terms[2] + terms[3];
                const at = pos * taps * c + p * c + ch;
                want[at] = s * m;
                if (mag) |mg| mg[at] = abs * m;
            }
        };
    }

    pub fn reference(self: *const DeformCols, want: []f64, mag: []f64) !void {
        self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const DeformCols, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        self.outputsIn(out, null, acc, order);
    }
};

/// The deformable columns' rounding depth: four corners, and positions as large as the plane.
fn planeDepth(shape: Shape) usize {
    return 2 * shape.rows + 4;
}

fn sumF64(x: []const f64) f64 {
    var t: f64 = 0;
    for (x) |v| t += v;
    return t;
}

// ── registrations ──

fn headWidth(shape: Shape) usize {
    return shape.k;
}

fn noDepth(_: Shape) usize {
    return 0;
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(QkvPrep, .init(a, &qkv_f16, .float16), .{
        .name = "idol qkv_prep f16 (rope, then norm, head-major)",
        .accum = .f32,
        .depth = headWidth,
        // f32 roundings: the rotation's two products and add, then the norm's mean, + eps, rsqrt and two products.
        .ops = 7,
        // The bias sum's rounding to T, the rotation's rounding to T before the norm reads it, and that read's.
        .mids = 3,
        .out = .f16,
        .replaces = "add bias -> split -> rope (slice, multiply, subtract, add, concat) -> rms_norm -> multiply by gamma -> transpose",
        .served = &.{ .{ .rows = 64, .n = 16, .k = HD }, .{ .rows = 9, .n = 2, .k = HD } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "positive squares: a pairwise sum stays near the output's rounding" },
    }, null);
    try set.add(BiasGelu, .init(a, &gelu_f16, .float16), .{
        .name = "idol bias_gelu f16",
        .accum = .none,
        .depth = noDepth,
        // The exponent's f32 chain (cube, product, sum, scale), the exp, + 1 and the quotient.
        .ops = 10,
        // The bias sum's rounding to T.
        .mids = 1,
        .out = .f16,
        .tiles = .{ .{}, .{}, .{ .size = 4, .step = 4, .min = 4 } },
        .served = &.{ .{ .rows = 64, .n = 1, .k = 4096 }, .{ .rows = 7, .n = 1, .k = 512 } },
    }, null);
    try set.add(GateResidual, .init(a, &gate_f16, .float16), .{
        .name = "idol gate_residual f16 (bias, gate, residual add)",
        .accum = .none,
        .depth = noDepth,
        .ops = 3,
        // The bias sum's and the gated product's roundings to T.
        .mids = 2,
        .out = .f16,
        .replaces = "add bias -> multiply by gate -> add residual",
        .tiles = .{ .{}, .{}, .{ .size = 4, .step = 4, .min = 4 } },
        .served = &.{ .{ .rows = 64, .n = 1, .k = 1024 }, .{ .rows = 7, .n = 1, .k = 128 } },
    }, null);
    try set.add(NormMod, .init(a, &norm_mod_f16, .float16), .{
        .name = "idol norm_mod f16 (layer norm, then x * (1 + scale) + shift)",
        .accum = .f32,
        .depth = headWidth,
        // f32 roundings after the sums: the mean, the difference, the variance, + eps, rsqrt (2), the three products
        // and the shift's sum.
        .ops = 10,
        .out = .f16,
        .tiles = .{ .{}, .{}, .{ .size = 128, .step = 128, .min = 128 } },
        .served = &.{ .{ .rows = 64, .n = 1, .k = 1024 }, .{ .rows = 7, .n = 1, .k = 128 } },
        .sums = .{ .f16 = 1, .bf16 = 1 },
    }, null);
    try set.add(RepoNorm, .init(a, &repo_norm_f16, .float16), .{
        .name = "idol repo_norm f16 -> f32 (layer norm with its weight and bias)",
        .accum = .f32,
        .depth = headWidth,
        // f32 roundings after the sums: the mean, the difference, the variance, + eps, rsqrt (2), the two products
        // and the bias's sum.
        .ops = 9,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = 128, .step = 128, .min = 128 } },
        .served = &.{ .{ .rows = 64, .n = 1, .k = 1024 }, .{ .rows = 7, .n = 1, .k = 128 } },
        .sums = .{ .f16 = 1, .bf16 = 1 },
    }, null);
    try set.add(RepoAngles, .init(a, &repo_angles_f32), .{
        .name = "idol repo_angles f32 (silu gate x content, final map, cos and sin)",
        .accum = .f32,
        .depth = headWidth,
        // The two products after the sum and the cos or sin.
        .ops = 3,
        .out = .f32,
        .served = &.{ .{ .rows = 64, .n = 16, .k = 128 }, .{ .rows = 7, .n = 2, .k = 16 } },
        // An angle's rounding moves cos and sin by its own scale, not theirs: gate 4 holds the sums to that scale.
        .sums = .{ .f16 = 1, .bf16 = 1, .exact = false },
    }, null);
    try set.add(DeformCols, .init(a, &deform_f16, .float16), .{
        .name = "idol deform_cols f16 (deform_conv2d columns: bilinear taps times their modulator)",
        .accum = .f32,
        .depth = planeDepth,
        // The modulator's product.
        .ops = 1,
        .out = .f16,
        .served = &.{ .{ .rows = 32, .n = 7, .k = 64 }, .{ .rows = 16, .n = 3, .k = 64 }, .{ .rows = 8, .n = 1, .k = 64 } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "four corners of samples within the output's own range: an fp16 sum stays near its rounding" },
    }, null);
}

// ── tests ──

test "kernel contract idol: the fused kernels pass gates 1-4" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var t: kc.gates.Tally = .{};
    try kc.gates.runAll(a, set.kernels.items, 0xc0de, &t);
    try std.testing.expectEqual(@as(usize, 0), t.problems());
    try std.testing.expectEqual(@as(usize, 7), t.passed);
}

test "kernel contract idol: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var q = QkvPrep.init(a, &qkv_f16, .float16);
    defer q.deinit();
    try std.testing.expect(try q.prepare(.{ .rows = 9, .n = 2, .k = HD }, .random, 3));
    try q.run();
    const ours = try a.dupe(u8, q.rig.outputBytes());
    defer a.free(ours);
    try std.testing.expectEqualSlices(u8, try q.wrapper(), ours);

    var g = BiasGelu.init(a, &gelu_f16, .float16);
    defer g.deinit();
    try std.testing.expect(try g.prepare(.{ .rows = 7, .n = 1, .k = 512 }, .random, 4));
    try g.run();
    const mine = try a.dupe(u8, g.rig.outputBytes());
    defer a.free(mine);
    const theirs = (try glue.biasGelu(g.args[0], g.args[1], g.rig.stream())) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(theirs);
    try std.testing.expectEqualSlices(u8, try g.rig.bytesOf(&.{theirs}), mine);

    var r = GateResidual.init(a, &gate_f16, .float16);
    defer r.deinit();
    try std.testing.expect(try r.prepare(.{ .rows = 7, .n = 1, .k = 128 }, .random, 5));
    try r.run();
    const ran = try a.dupe(u8, r.rig.outputBytes());
    defer a.free(ran);
    const gated = (try glue.gateResidual(r.args[0], r.args[1], r.args[2], r.args[3], r.rig.stream())) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(gated);
    try std.testing.expectEqualSlices(u8, try r.rig.bytesOf(&.{gated}), ran);

    var m = NormMod.init(a, &norm_mod_f16, .float16);
    defer m.deinit();
    try std.testing.expect(try m.prepare(.{ .rows = 7, .n = 1, .k = 128 }, .random, 6));
    try m.run();
    const normed = try a.dupe(u8, m.rig.outputBytes());
    defer a.free(normed);
    const modded = (try glue.normMod(m.args[0], m.args[1], m.args[2], L.LN_EPS, m.rig.stream())) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(modded);
    try std.testing.expectEqualSlices(u8, try m.rig.bytesOf(&.{modded}), normed);

    var rn = RepoNorm.init(a, &repo_norm_f16, .float16);
    defer rn.deinit();
    try std.testing.expect(try rn.prepare(.{ .rows = 7, .n = 1, .k = 128 }, .random, 7));
    try rn.run();
    const rn_bytes = try a.dupe(u8, rn.rig.outputBytes());
    defer a.free(rn_bytes);
    const rn_out = (try glue.repoNorm(rn.args[0], rn.args[1], rn.args[2], 1e-5, false, rn.rig.stream())) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(rn_out);
    try std.testing.expectEqualSlices(u8, try rn.rig.bytesOf(&.{rn_out}), rn_bytes);

    var ra = RepoAngles.init(a, &repo_angles_f32);
    defer ra.deinit();
    try std.testing.expect(try ra.prepare(.{ .rows = 7, .n = 2, .k = 16 }, .random, 8));
    try ra.run();
    const ra_bytes = try a.dupe(u8, ra.rig.outputBytes());
    defer a.free(ra_bytes);
    const t = (try glue.repoAngles(ra.args[0], ra.args[1], ra.args[2], ra.args[3], 2, ra.rig.stream())) orelse return error.Declined;
    defer for ([_]mlx.mlx_array{ t.cos, t.sin }) |o| {
        _ = mlx.mlx_array_free(o);
    };
    try std.testing.expectEqualSlices(u8, try ra.rig.bytesOf(&.{ t.cos, t.sin }), ra_bytes);

    var dc = DeformCols.init(a, &deform_f16, .float16);
    defer dc.deinit();
    try std.testing.expect(try dc.prepare(.{ .rows = 16, .n = 3, .k = 64 }, .random, 9));
    try dc.run();
    const dc_bytes = try a.dupe(u8, dc.rig.outputBytes());
    defer a.free(dc_bytes);
    const cols = (try glue.deformCols(dc.args[0], dc.args[1], dc.args[2], 3, 16, 16, dc.rig.stream())) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(cols);
    try std.testing.expectEqualSlices(u8, try dc.rig.bytesOf(&.{cols}), dc_bytes);
}

test "idol: the split RePo norm is its output's TF32 halves: rows [hi | hi | lo], hi + lo the norm" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var rn = RepoNorm.init(a, &repo_norm_f16, .float16);
    defer rn.deinit();
    const n = 256;
    try std.testing.expect(try rn.prepare(.{ .rows = 7, .n = 1, .k = n }, .random, 11));
    try rn.run();
    const whole = try a.dupe(u8, rn.rig.outputBytes());
    defer a.free(whole);
    const f = (try glue.repoNorm(rn.args[0], rn.args[1], rn.args[2], 1e-5, true, rn.rig.stream())) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(f);
    const three = try rn.rig.bytesOf(&.{f});
    const word = struct {
        fn at(b: []const u8, i: usize) u32 {
            return std.mem.readInt(u32, b[4 * i ..][0..4], .little);
        }
    }.at;
    for (0..7) |r| for (0..n) |i| {
        const o: f32 = @bitCast(word(whole, r * n + i));
        const hi = word(three, r * 3 * n + i);
        try std.testing.expectEqual(@as(u32, @bitCast(o)) & 0xFFFFE000, hi);
        try std.testing.expectEqual(hi, word(three, r * 3 * n + n + i));
        try std.testing.expectEqual(o, @as(f32, @bitCast(hi)) + @as(f32, @bitCast(word(three, r * 3 * n + 2 * n + i))));
    };
}
