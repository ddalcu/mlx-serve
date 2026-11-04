//! MiniMax-H3's kernel contract: the DiT glue (q/k RMSNorm + partial RoPE + head-major layout, SwiGLU), each
//! bit-equal to the op chain it replaces, and VSA's pooling, assembly and block-sparse attention kernels.
//! Listed in `kernel_contract/registry.zig`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const glue = @import("glue.zig");
const h3 = @import("h3.zig");
const vsa = @import("vsa.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

// ── SwiGLU ──

const swiglu_bf16: metal.Launch = .{ .name = "msv_h3_swiglu", .inputs = &.{"Y"}, .outputs = &.{"OUT"}, .source = glue.SWIGLU_SOURCE, .header = glue.SWIGLU_HEADER, .tg_bytes = 0, .m1 = 928, .m2 = 928 };

/// y [rows, 2F] = gate | up -> silu(gate) * up [rows, F]; `k` is F.
const Swiglu = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    y: []f64 = &.{},
    args: [1]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) Swiglu {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt };
    }

    pub fn deinit(self: *Swiglu) void {
        self.rig.a.free(self.y);
        self.rig.deinit();
    }

    pub fn geometry(self: *const Swiglu, shape: Shape) ?metal.Geometry {
        const rows: c_int = @intCast(shape.rows);
        const f: c_int = @intCast(shape.k);
        if (f == 0 or @rem(f, 4) != 0) return null;
        var g = metal.Geometry{ .grid = .{ @divExact(rows * f, 4), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ rows, f }, self.dt);
        g.dtype("T", self.dt);
        g.int("F", f);
        return g;
    }

    pub fn prepare(self: *Swiglu, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        const a = self.rig.a;
        var prng = std.Random.DefaultPrng.init(seed);
        a.free(self.y);
        self.y = &.{};
        self.y = try host.uniform(a, prng.random(), shape.rows * 2 * shape.k, -6, 6, self.dt);
        if (fill == .nonfinite) host.plant(self.y, shape.rows, 2 * shape.k);
        self.rig.reset();
        self.args[0] = try self.rig.input(self.dt, &.{ @intCast(shape.rows), @intCast(2 * shape.k) }, self.y);
        return true;
    }

    pub fn run(self: *Swiglu) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Swiglu) usize {
        return self.shape.rows * self.shape.k;
    }

    /// `msv_sigmoid` (1 / (1 + e^|x|), mirrored), the product with the gate and the up value, each rounded to T but
    /// the last.
    pub fn reference(self: *const Swiglu, want: []f64, mag: []f64) !void {
        const e = host.elemOf(self.dt);
        const f = self.shape.k;
        for (0..self.shape.rows) |r| for (0..f) |i| {
            const g = self.y[r * 2 * f + i];
            const u = self.y[r * 2 * f + f + i];
            const y = 1 / (1 + @exp(@abs(g)));
            const sg = spec.roundTo(e, if (g < 0) y else 1 - y);
            const out = spec.roundTo(e, g * sg) * u;
            want[r * f + i] = out;
            mag[r * f + i] = @abs(out);
        };
    }

    /// The chain `swigluAct` runs without the kernel: split, sigmoid, two multiplies.
    pub fn replaced(self: *Swiglu) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const f: c_int = @intCast(self.shape.k);
        const y = self.args[0];
        const gate = try g.sliceAxis(y, 1, 0, f);
        const up = try g.sliceAxis(y, 1, f, 2 * f);
        return self.rig.bytesOf(&.{try g.mul(try g.mul(gate, try g.sigmoid(gate)), up)});
    }
};

// ── q/k RMSNorm + partial RoPE + head-major layout ──

const qkv_bf16: metal.Launch = .{ .name = "msv_h3_qkv_prep", .inputs = &.{ "QKV", "QW", "KW", "COS", "SIN", "eps" }, .outputs = &.{ "OQ", "OK", "OV" }, .source = glue.QKV_SOURCE, .tg_bytes = 0, .m1 = 928, .m2 = 928 };

const HD = 128;
const QK_EPS: f32 = 1e-5;

/// qkv [S, 3*H*128] -> q, k, v [1, H, S, 128]: `rows` is S, `n` the heads, `k` the head width (the norm's sum).
const QkvPrep = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    half: usize,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    qkv: []f64 = &.{},
    qw: []f64 = &.{},
    kw: []f64 = &.{},
    cos: []f64 = &.{},
    sin: []f64 = &.{},
    args: [6]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype, half: usize) QkvPrep {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .half = half, .arena = .init(a) };
    }

    pub fn deinit(self: *QkvPrep) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const QkvPrep, shape: Shape) ?metal.Geometry {
        if (shape.k != HD or shape.n == 0 or self.half % 4 != 0 or 2 * self.half > HD) return null;
        const h: c_int = @intCast(shape.n);
        const seq: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ 32 * 3 * h, seq, 1 }, .tg = .{ 32, 1, 1 } };
        for (0..3) |_| g.out(&.{ 1, h, seq, HD }, self.dt);
        g.dtype("T", self.dt);
        g.int("H", h);
        g.int("HALF", self.half);
        return g;
    }

    pub fn prepare(self: *QkvPrep, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const e = host.elemOf(self.dt);
        const width = 3 * shape.n * HD;
        self.qkv = try host.uniform(a, rng, shape.rows * width, -3, 3, self.dt);
        self.qw = try host.uniform(a, rng, HD, 0.5, 1.5, self.dt);
        self.kw = try host.uniform(a, rng, HD, 0.5, 1.5, self.dt);
        self.cos = try a.alloc(f64, shape.rows * self.half);
        self.sin = try a.alloc(f64, shape.rows * self.half);
        for (self.cos, self.sin) |*c, *s| {
            const ang = 6.3 * rng.float(f64);
            c.* = spec.roundTo(e, @cos(ang));
            s.* = spec.roundTo(e, @sin(ang));
        }
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.qkv, shape.rows, width),
            .stress => {
                // Gate 4: integer rows, sums of squares exact in fp32 past fp16's range; unit weights and the
                // identity rotation, so the norm is the only rounding left.
                for (self.qkv) |*v| v.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32);
                @memset(self.qw, 1);
                @memset(self.kw, 1);
                @memset(self.cos, 1);
                @memset(self.sin, 0);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const seq: c_int = @intCast(shape.rows);
        const hf: c_int = @intCast(self.half);
        self.args = .{
            try rig.input(self.dt, &.{ seq, @intCast(width) }, self.qkv),
            try rig.input(self.dt, &.{HD}, self.qw),
            try rig.input(self.dt, &.{HD}, self.kw),
            try rig.input(self.dt, &.{ seq, hf }, self.cos),
            try rig.input(self.dt, &.{ seq, hf }, self.sin),
            try rig.scalar(QK_EPS),
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
        const half = self.half;
        var sq: [HD]f64 = undefined;
        var nrm: [HD]f64 = undefined;
        for (0..seq) |r| for (0..3) |which| for (0..heads) |h| {
            const src = self.qkv[r * 3 * heads * HD + (which * heads + h) * HD ..][0..HD];
            const out = want[((which * heads + h) * seq + r) * HD ..][0..HD];
            if (which == 2) {
                @memcpy(out, src);
                if (mag) |m| @memcpy(m[((which * heads + h) * seq + r) * HD ..][0..HD], src);
                continue;
            }
            for (&sq, src) |*q, x| q.* = x * x;
            const ss = if (acc) |ac| spec.sumIn(ac, order, &sq) else blk: {
                var t: f64 = 0;
                for (sq) |q| t += q;
                break :blk t;
            };
            const inv = 1 / @sqrt(ss / HD + QK_EPS);
            const w = if (which == 0) self.qw else self.kw;
            for (&nrm, src, w) |*n, x, wi| n.* = spec.roundTo(e, wi * spec.roundTo(e, x * inv));
            const tab = r * half;
            for (out, 0..) |*o, j| {
                o.* = if (j < half)
                    spec.roundTo(e, nrm[j] * self.cos[tab + j]) - spec.roundTo(e, nrm[j + half] * self.sin[tab + j])
                else if (j < 2 * half)
                    spec.roundTo(e, nrm[j - half] * self.sin[tab + j - half]) + spec.roundTo(e, nrm[j] * self.cos[tab + j - half])
                else
                    nrm[j];
            }
            if (mag) |m| for (m[((which * heads + h) * seq + r) * HD ..][0..HD], 0..) |*mo, j| {
                mo.* = if (j < 2 * half) @abs(nrm[j % half]) + @abs(nrm[j % half + half]) else @abs(nrm[j]);
            };
        };
    }

    pub fn reference(self: *const QkvPrep, want: []f64, mag: []f64) !void {
        self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const QkvPrep, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        self.outputsIn(out, null, acc, order);
    }

    /// `qkvHeadMajor`'s chain: split, per-head rms_norm, `applyRopePub`, head-major transpose.
    pub fn replaced(self: *QkvPrep) ![]const u8 {
        const s = self.rig.stream();
        var g = ops.Graph.init(self.rig.a, s);
        defer g.deinit();
        const seq: c_int = @intCast(self.shape.rows);
        const h: c_int = @intCast(self.shape.n);
        const hf: c_int = @intCast(self.half);
        const x = self.args;
        const cos = try g.reshape(x[3], &.{ 1, seq, 1, hf });
        const sin = try g.reshape(x[4], &.{ 1, seq, 1, hf });
        var outs: [3]mlx.mlx_array = undefined;
        for (&outs, 0..) |*o, i| {
            const part = try g.reshape(try g.sliceAxis(x[0], 1, @intCast(i * self.shape.n * HD), @intCast((i + 1) * self.shape.n * HD)), &.{ 1, seq, h, HD });
            const ready = if (i == 2) part else try g.adopt(try h3.applyRopePub(try g.rmsNorm(part, x[1 + i], QK_EPS), .{ .cos = cos, .sin = sin }, hf, HD, s));
            o.* = try g.contiguous(try g.transpose(ready, &.{ 0, 2, 1, 3 }));
        }
        return self.rig.bytesOf(&outs);
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *QkvPrep) ![]const u8 {
        const x = self.args;
        const got = (try glue.qkvPrep(x[0], x[1], x[2], x[3], x[4], QK_EPS, @intCast(self.shape.n), @intCast(self.half), self.rig.stream())) orelse return error.Declined;
        defer for ([_]mlx.mlx_array{ got.q, got.k, got.v }) |o| {
            _ = mlx.mlx_array_free(o);
        };
        return self.rig.bytesOf(&.{ got.q, got.k, got.v });
    }
};

// ── VSA ──

const TILE = vsa.TILE;
const D = 128;

/// A VSA layout for the kernels: `n_prefix` dense tiles (ragged, keys only), then `n_video` full video tiles, `heads`
/// heads; head-major tiled rows [1, H, n_tiles * TILE, 128].
const Layout = struct {
    n_prefix: usize = 2,
    n_video: usize,
    heads: usize,
    k_sel: usize,

    fn tiles(l: Layout) usize {
        return l.n_prefix + l.n_video;
    }

    /// Valid rows per tile: the dense tiles 37 and 128 (a text strip and a full one), every video tile full.
    fn validRows(l: Layout, a: std.mem.Allocator) ![]f64 {
        const out = try a.alloc(f64, l.tiles());
        for (out, 0..) |*v, t| v.* = if (t == 0) 37 else TILE;
        return out;
    }

    /// `[n_prefix, n_video, k_sel, n_tiles]`, as `Plan.meta`.
    fn meta(l: Layout) [4]f64 {
        return .{ @floatFromInt(l.n_prefix), @floatFromInt(l.n_video), @floatFromInt(l.k_sel), @floatFromInt(l.tiles()) };
    }

    /// Tiled rows: values where a tile's row is valid, `pad` past it.
    fn rows(l: Layout, a: std.mem.Allocator, rng: std.Random, vbs: []const f64, e: spec.Elem, pad: f64) ![]f64 {
        const out = try a.alloc(f64, l.heads * l.tiles() * TILE * D);
        for (0..l.heads) |h| for (0..l.tiles()) |t| for (0..TILE) |r| {
            const row = out[((h * l.tiles() + t) * TILE + r) * D ..][0..D];
            for (row) |*v| v.* = if (@as(f64, @floatFromInt(r)) < vbs[t]) spec.roundTo(e, rng.float(f64) * 4 - 2) else pad;
        };
        return out;
    }
};

/// The pool kernel's valid-row means: `rows` the tiles, `n` the heads, `k` the tile height (the sums' length).
const pool_bf16: metal.Launch = .{ .name = "msv_h3_vsa_pool", .inputs = &.{ "x", "vbs", "meta" }, .outputs = &.{"out"}, .source = vsa.POOL_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

const VsaPool = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    vbs: []f64 = &.{},
    x: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) VsaPool {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *VsaPool) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn layout(shape: Shape) Layout {
        return .{ .n_video = shape.rows - 2, .heads = shape.n, .k_sel = 1 };
    }

    pub fn geometry(self: *const VsaPool, shape: Shape) ?metal.Geometry {
        if (shape.k != TILE or shape.rows < 3) return null;
        const nt: c_int = @intCast(shape.rows);
        const h: c_int = @intCast(shape.n);
        var g = metal.Geometry{ .grid = .{ D, nt, h }, .tg = .{ D, 1, 1 } };
        g.out(&.{ h, nt, D }, .float32);
        g.dtype("T", self.dt);
        g.int("TILE", TILE);
        return g;
    }

    pub fn prepare(self: *VsaPool, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const l = layout(shape);
        self.vbs = try l.validRows(a);
        // Rows past a tile's valid count are NaN: the pool never reads them.
        self.x = try l.rows(a, rng, self.vbs, host.elemOf(self.dt), std.math.nan(f64));
        switch (fill) {
            .random => {},
            .nonfinite => {
                // NaN, +Inf, -Inf in valid rows of tiles 1-3 (column 5); both infinities in tile 4's column 9.
                const put = [_]struct { t: usize, r: usize, c: usize, v: f64 }{
                    .{ .t = 1, .r = 3, .c = 5, .v = std.math.nan(f64) },     .{ .t = 2, .r = 0, .c = 5, .v = std.math.inf(f64) },
                    .{ .t = 3, .r = 127, .c = 5, .v = -std.math.inf(f64) }, .{ .t = 4, .r = 1, .c = 9, .v = std.math.inf(f64) },
                    .{ .t = 4, .r = 2, .c = 9, .v = -std.math.inf(f64) },
                };
                for (put) |p| if (p.t < l.tiles()) {
                    self.x[(p.t * TILE + p.r) * D + p.c] = p.v;
                };
            },
            .stress => {
                // Positive integers exact in bf16: every partial sum of a full tile passes fp16's exact integers
                // (2048) and bf16's (256) in any order, the total stays exact in fp32.
                for (0..l.heads * l.tiles()) |ht| {
                    const n: usize = @intFromFloat(self.vbs[ht % l.tiles()]);
                    for (0..D) |c| for (0..n) |r| {
                        self.x[(ht * TILE + r) * D + c] = @floatFromInt(65 + rng.uintLessThan(u32, 63));
                    };
                }
            },
        }
        const m = l.meta();
        const rig = &self.rig;
        rig.reset();
        self.args = .{
            try rig.input(self.dt, &.{ 1, @intCast(l.heads), @intCast(l.tiles() * TILE), D }, self.x),
            try rig.input(.int32, &.{@intCast(l.tiles())}, self.vbs),
            try rig.input(.int32, &.{4}, &m),
        };
        return true;
    }

    pub fn run(self: *VsaPool) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const VsaPool) usize {
        return self.shape.n * self.shape.rows * D;
    }

    fn outputsIn(self: *const VsaPool, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const a = self.arena.child_allocator;
        const t = try a.alloc(f64, TILE);
        defer a.free(t);
        const nt = self.shape.rows;
        for (0..self.shape.n) |h| for (0..nt) |tile| for (0..D) |c| {
            const n: usize = @intFromFloat(self.vbs[tile]);
            var m: f64 = 0;
            for (t[0..n], 0..) |*o, r| {
                o.* = self.x[((h * nt + tile) * TILE + r) * D + c];
                m += @abs(o.*);
            }
            const sum = if (acc) |ac| spec.sumIn(ac, order, t[0..n]) else sumF64(t[0..n]);
            const i = (h * nt + tile) * D + c;
            want[i] = sum / @as(f64, @floatFromInt(n));
            if (mag) |mg| mg[i] = m / @as(f64, @floatFromInt(n));
        };
    }

    pub fn reference(self: *const VsaPool, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const VsaPool, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }
};

/// The assembly: each row's dense or video attention row plus T(comp * gate); `rows` the sequence, `n` the heads.
const assemble_bf16: metal.Launch = .{ .name = "msv_h3_vsa_assemble", .inputs = &.{ "dense", "video", "comp", "gate", "out_src", "row_tile", "shape" }, .outputs = &.{"out"}, .source = vsa.ASSEMBLE_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

const VsaAssemble = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    v: struct { dense: []f64 = &.{}, video: []f64 = &.{}, comp: []f64 = &.{}, gate: []f64 = &.{}, src: []f64 = &.{}, tile: []f64 = &.{} } = .{},
    args: [7]mlx.mlx_array = undefined,

    /// comp has a row per tile.
    const N_TILES = 5;

    /// The first half of the attention rows are dense ones, the rest video rows.
    fn denseRows(shape: Shape) usize {
        return shape.rows / 2;
    }

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) VsaAssemble {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *VsaAssemble) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn videoRows(shape: Shape) usize {
        return shape.rows - denseRows(shape);
    }

    pub fn geometry(self: *const VsaAssemble, shape: Shape) ?metal.Geometry {
        if (shape.k != D or shape.rows < 2 or shape.n == 0) return null;
        const h: c_int = @intCast(shape.n);
        const seq: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ h * 32, seq, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ seq, h * D }, self.dt);
        g.dtype("T", self.dt);
        g.int("TILE", TILE);
        return g;
    }

    pub fn prepare(self: *VsaAssemble, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const h = shape.n;
        const seq = shape.rows;
        const nd = denseRows(shape);
        const nvr = videoRows(shape);
        self.v.dense = try host.uniform(a, rng, h * nd * D, -3, 3, self.dt);
        self.v.video = try host.uniform(a, rng, h * nvr * D, -3, 3, self.dt);
        self.v.comp = try host.uniform(a, rng, h * N_TILES * D, -3, 3, .float32);
        self.v.gate = try host.uniform(a, rng, seq * h * D, -2, 2, self.dt);
        // A permutation of every attention row, and each row's tile.
        self.v.src = try a.alloc(f64, seq);
        for (self.v.src, 0..) |*x, i| x.* = @floatFromInt(i);
        rng.shuffle(f64, self.v.src);
        self.v.tile = try a.alloc(f64, seq);
        for (self.v.tile) |*x| x.* = @floatFromInt(rng.uintLessThan(u32, N_TILES));
        if (fill == .nonfinite) {
            host.plant(self.v.gate, seq, h * D);
            self.v.comp[D + 7] = std.math.nan(f64);
        }
        const rig = &self.rig;
        rig.reset();
        const hc: c_int = @intCast(h);
        const shp = [_]f64{ @floatFromInt(nd), N_TILES, @floatFromInt(nvr), @floatFromInt(h * D) };
        self.args = .{
            try rig.input(self.dt, &.{ 1, hc, @intCast(nd), D }, self.v.dense),
            try rig.input(self.dt, &.{ 1, hc, @intCast(nvr), D }, self.v.video),
            try rig.input(.float32, &.{ hc, N_TILES, D }, self.v.comp),
            try rig.input(self.dt, &.{ @intCast(seq), hc * D }, self.v.gate),
            try rig.input(.int32, &.{@intCast(seq)}, self.v.src),
            try rig.input(.int32, &.{@intCast(seq)}, self.v.tile),
            try rig.input(.int32, &.{4}, &shp),
        };
        return true;
    }

    pub fn run(self: *VsaAssemble) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const VsaAssemble) usize {
        return self.shape.rows * self.shape.n * D;
    }

    /// `out + T(comp * gate)`: the product rounded to T, the sum the output's last rounding.
    pub fn reference(self: *const VsaAssemble, want: []f64, mag: []f64) !void {
        const e = host.elemOf(self.dt);
        const h = self.shape.n;
        const nd = denseRows(self.shape);
        const nvr = videoRows(self.shape);
        for (0..self.shape.rows) |r| for (0..h) |hh| for (0..D) |d| {
            const src: usize = @intFromFloat(self.v.src[r]);
            const att = if (src < nd) self.v.dense[(hh * nd + src) * D + d] else self.v.video[(hh * nvr + src - nd) * D + d];
            const tile: usize = @intFromFloat(self.v.tile[r]);
            const gated = spec.roundTo(e, self.v.comp[(hh * N_TILES + tile) * D + d] * self.v.gate[r * h * D + hh * D + d]);
            const i = r * h * D + hh * D + d;
            want[i] = att + gated;
            mag[i] = @abs(want[i]);
        };
    }
};

/// Block-sparse attention: each video query tile over its selected key tiles. `rows` the video tiles, `n` the
/// heads, `k` the selected tiles per query tile (the dense prefix first).
const simd_launch: metal.Launch = .{
    .name = "msv_h3_vsa",
    .inputs = &.{ "q", "k", "v", "sel", "vbs", "meta", "scale" },
    .outputs = &.{"out"},
    .source = vsa.KERNEL_SOURCE,
    .header = vsa.KERNEL_HEADER,
    // ksm + vsm: two 16 x 136 tiles of T (bf16).
    .tg_bytes = 2 * 16 * 136 * 2,
    .m1 = 1024,
    .m2 = 1024,
};

const nax_launch: metal.Launch = .{
    .name = "msv_h3_vsa_nax",
    .inputs = &.{ "q", "k", "v", "sel", "vbs", "meta", "scale" },
    .outputs = &.{"out"},
    .source = vsa.NAX_SOURCE,
    .header = vsa.NAX_HEADER,
    .tg_bytes = 0,
    .offline = false,
    .min_gen = 17,
};

const VsaAttn = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    nax: bool,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    vbs: []f64 = &.{},
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    sel: []f64 = &.{},
    args: [7]mlx.mlx_array = undefined,

    const SCALE: f32 = 0.08838834764831845; // 1 / sqrt(128)

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, nax: bool, dt: mlx.mlx_dtype) VsaAttn {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .nax = nax, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *VsaAttn) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn layout(shape: Shape) Layout {
        return .{ .n_video = shape.rows, .heads = shape.n, .k_sel = shape.k };
    }

    pub fn geometry(self: *const VsaAttn, shape: Shape) ?metal.Geometry {
        const l = layout(shape);
        if (l.n_video == 0 or l.heads == 0 or l.k_sel <= l.n_prefix or l.k_sel > l.tiles()) return null;
        const nv: c_int = @intCast(l.n_video);
        const h: c_int = @intCast(l.heads);
        var g = if (self.nax)
            metal.Geometry{ .grid = .{ 128 * nv * (TILE / 64), h, 1 }, .tg = .{ 128, 1, 1 } }
        else
            metal.Geometry{ .grid = .{ 128 * (TILE / 32), nv, h }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ 1, h, nv * TILE, D }, self.dt);
        g.dtype("T", self.dt);
        g.int("TILE", TILE);
        return g;
    }

    pub fn prepare(self: *VsaAttn, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const l = layout(shape);
        const e = host.elemOf(self.dt);
        self.vbs = try l.validRows(a);
        // Padding rows are zero, as `attend` tiles them: the kernel loads key chunks past a tile's valid count and
        // masks their scores, so a P of zero meets those value rows.
        self.q = try l.rows(a, rng, self.vbs, e, 0);
        self.k = try l.rows(a, rng, self.vbs, e, 0);
        self.v = try l.rows(a, rng, self.vbs, e, 0);
        // Every dense tile, then distinct video tiles in a random order.
        self.sel = try a.alloc(f64, l.heads * l.n_video * l.k_sel);
        const pool = try a.alloc(usize, l.n_video);
        for (pool, 0..) |*p, i| p.* = l.n_prefix + i;
        for (0..l.heads * l.n_video) |hq| {
            rng.shuffle(usize, pool);
            const row = self.sel[hq * l.k_sel ..][0..l.k_sel];
            for (row, 0..) |*x, i| x.* = @floatFromInt(if (i < l.n_prefix) i else pool[i - l.n_prefix]);
        }
        switch (fill) {
            .random => {},
            .nonfinite => {
                // In video tile 0 of head 0: a NaN query element, an infinite key element (scores of +-Inf) and
                // both infinities in one value row; every query row of tile 1 reads that tile only if selected.
                const nt = l.tiles();
                self.q[(l.n_prefix * TILE + 2) * D + 7] = std.math.nan(f64);
                self.k[(l.n_prefix * TILE + 5) * D + 3] = std.math.inf(f64);
                self.v[(TILE + 9) * D + 1] = std.math.inf(f64);
                self.v[(TILE + 9) * D + 2] = -std.math.inf(f64);
                _ = nt;
            },
            .stress => {
                // Small integers make the score and P*V sums exact in fp32; the bound (not exact sums) still
                // carries the softmax's exp roundings.
                for ([_][]f64{ self.q, self.k, self.v }) |t| for (t) |*x| if (x.* != 0) {
                    x.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 5))) - 2);
                };
            },
        }
        const m = l.meta();
        const rig = &self.rig;
        rig.reset();
        const tiled = [_]c_int{ 1, @intCast(l.heads), @intCast(l.tiles() * TILE), D };
        self.args = .{
            try rig.input(self.dt, &tiled, self.q),
            try rig.input(self.dt, &tiled, self.k),
            try rig.input(self.dt, &tiled, self.v),
            try rig.input(.int32, &.{ @intCast(l.heads), @intCast(l.n_video), @intCast(l.k_sel) }, self.sel),
            try rig.input(.int32, &.{@intCast(l.tiles())}, self.vbs),
            try rig.input(.int32, &.{4}, &m),
            try rig.scalar(SCALE),
        };
        return true;
    }

    pub fn run(self: *VsaAttn) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const VsaAttn) usize {
        return self.shape.n * self.shape.rows * TILE * D;
    }

    fn keys(self: *const VsaAttn) usize {
        var n: usize = 0;
        for (self.sel[0..self.shape.k]) |t| n += @intFromFloat(self.vbs[@intFromFloat(t)]);
        return n;
    }

    /// Exact softmax attention over each query row's selected keys. `mag` carries the attention's first-order
    /// error as a sum magnitude at depth `keys` (spec.bound multiplies it by gamma(keys)): the P*V sums, the
    /// running rescales and the normalizer, P's rounding to T before its MMA, and the score sums' own error.
    fn outputsIn(self: *const VsaAttn, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const l = layout(self.shape);
        const a = self.arena.child_allocator;
        const nk = self.keys();
        const kr = try a.alloc(usize, nk); // key row index (into the tiled planes)
        defer a.free(kr);
        const sc = try a.alloc(f64, nk);
        defer a.free(sc);
        const sa = try a.alloc(f64, nk); // sum of |q k| per score
        defer a.free(sa);
        const terms = try a.alloc(f64, @max(nk, D));
        defer a.free(terms);
        const nt = l.tiles();
        const g_keys = spec.gamma(@floatFromInt(nk), spec.unit(.f32));
        const g_d = spec.gamma(D, spec.unit(.f32));
        for (0..l.heads) |h| for (0..l.n_video) |qt| {
            const picks = self.sel[(h * l.n_video + qt) * l.k_sel ..][0..l.k_sel];
            var j: usize = 0;
            for (picks) |t| {
                const tile: usize = @intFromFloat(t);
                for (0..@as(usize, @intFromFloat(self.vbs[tile]))) |r| {
                    kr[j] = (h * nt + tile) * TILE + r;
                    j += 1;
                }
            }
            for (0..TILE) |row| {
                const q = self.q[((h * nt + l.n_prefix + qt) * TILE + row) * D ..][0..D];
                var mx = -std.math.inf(f64);
                for (sc, sa, kr) |*s, *s_abs, r| {
                    const k = self.k[r * D ..][0..D];
                    var abs_sum: f64 = 0;
                    for (terms[0..D], q, k) |*tm, qd, kd| {
                        tm.* = qd * kd;
                        abs_sum += @abs(tm.*);
                    }
                    s.* = SCALE * (if (acc) |ac| spec.sumIn(ac, order, terms[0..D]) else sumF64(terms[0..D]));
                    s_abs.* = SCALE * abs_sum;
                    mx = @max(mx, s.*);
                }
                var lsum: f64 = 0;
                for (sc) |*s| {
                    s.* = @exp(s.* - mx);
                    lsum += s.*;
                }
                const out_base = ((h * l.n_video + qt) * TILE + row) * D;
                for (0..D) |d| {
                    var pv_abs: f64 = 0;
                    for (terms[0..nk], sc, kr) |*tm, p, r| {
                        tm.* = p * self.v[r * D + d];
                        pv_abs += @abs(tm.*);
                    }
                    const o = (if (acc) |ac| spec.sumIn(ac, order, terms[0..nk]) else sumF64(terms[0..nk])) / lsum;
                    want[out_base + d] = o;
                    if (mag) |m| {
                        var spread: f64 = 0;
                        for (sc, sa, kr) |p, s_abs, r| spread += p * @abs(self.v[r * D + d] - o) * (g_d * s_abs + 8 * spec.unit(.f32));
                        const a_term = pv_abs / lsum;
                        // P*V sums, rescales and normalizer (3 gamma(keys)), P in T, the scores.
                        const allow = 3 * g_keys * (a_term + @abs(o)) + spec.unit(if (self.dt == .float16) .f16 else .bf16) * a_term + spread / lsum;
                        m[out_base + d] = allow / g_keys;
                    }
                }
            }
        };
    }

    pub fn reference(self: *const VsaAttn, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const VsaAttn, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }
};

fn keysDepth(shape: Shape) usize {
    // The dense prefix (37 + 128 rows) and every selected video tile at full height.
    return 37 + TILE + (shape.k - 2) * TILE;
}

fn tileHeight(shape: Shape) usize {
    return shape.k;
}

fn naxAvailable() bool {
    return @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable();
}

fn headWidth(shape: Shape) usize {
    return shape.k;
}

fn noDepth(_: Shape) usize {
    return 0;
}

fn sumF64(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += v;
    return s;
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(Swiglu, .init(a, &swiglu_bf16, .bfloat16), .{
        .name = "h3 swiglu bf16",
        .accum = .none,
        .depth = noDepth,
        // The sigmoid's f32 exp, add and divide, then its rounding to T and the gate product's.
        .ops = 3,
        .mids = 2,
        .out = .bf16,
        .replaces = "slice -> sigmoid -> multiply -> multiply",
        .tiles = .{ .{}, .{}, .{ .size = 4, .step = 4, .min = 4 } },
        .served = &.{ .{ .rows = 64, .n = 1, .k = 14336 }, .{ .rows = 7, .n = 1, .k = 14336 } },
    }, null);
    try set.add(QkvPrep, .init(a, &qkv_bf16, .bfloat16, 48), .{
        .name = "h3 qkv_prep bf16 (rotary 96 of 128)",
        .accum = .f32,
        .depth = headWidth,
        // f32 roundings after the sum: the mean, + eps, rsqrt, the product with the row.
        .ops = 4,
        // T(x * inv), w * that, the rotation's two products.
        .mids = 4,
        .out = .bf16,
        .replaces = "split -> rms_norm -> rope (slice, multiply, subtract, add, concat) -> transpose -> contiguous",
        .served = &.{ .{ .rows = 64, .n = 56, .k = HD }, .{ .rows = 9, .n = 4, .k = HD } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "positive squares: a pairwise sum stays near the output's rounding" },
    }, null);
    try set.add(VsaPool, .init(a, &pool_bf16, .bfloat16), .{
        .name = "h3 vsa_pool bf16",
        .accum = .f32,
        .depth = tileHeight,
        // The division by the valid count.
        .ops = 1,
        .out = .f32,
        .served = &.{ .{ .rows = 9, .n = 4, .k = TILE }, .{ .rows = 3, .n = 1, .k = TILE } },
    }, null);
    try set.add(VsaAssemble, .init(a, &assemble_bf16, .bfloat16), .{
        .name = "h3 vsa_assemble bf16",
        .accum = .none,
        .depth = noDepth,
        // T(comp * gate) before the sum's rounding.
        .mids = 1,
        .out = .bf16,
        .served = &.{ .{ .rows = 300, .n = 4, .k = D }, .{ .rows = 41, .n = 1, .k = D } },
    }, null);
    const attn_shapes: []const Shape = &.{ .{ .rows = 2, .n = 1, .k = 4 }, .{ .rows = 2, .n = 4, .k = 3 } };
    const attn_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the softmax's exp leaves no sum exact: the stress inputs check the fp32 sums within the float64 bound" };
    try set.add(VsaAttn, .init(a, &simd_launch, false, .bfloat16), .{
        .name = "h3 vsa simdgroup bf16",
        .accum = .f32,
        .depth = keysDepth,
        .out = .bf16,
        .served = attn_shapes,
        .sums = attn_sums,
    }, null);
    if (naxAvailable()) try set.add(VsaAttn, .init(a, &nax_launch, true, .bfloat16), .{
        .name = "h3 vsa nax bf16",
        .accum = .f32,
        .depth = keysDepth,
        .out = .bf16,
        .served = attn_shapes,
        .sums = attn_sums,
    }, null);
}

// ── tests ──

test "kernel contract h3: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var q = QkvPrep.init(a, &qkv_bf16, .bfloat16, 48);
    defer q.deinit();
    try std.testing.expect(try q.prepare(.{ .rows = 9, .n = 4, .k = HD }, .random, 3));
    try q.run();
    const ours = try a.dupe(u8, q.rig.outputBytes());
    defer a.free(ours);
    try std.testing.expectEqualSlices(u8, try q.wrapper(), ours);

    var sw = Swiglu.init(a, &swiglu_bf16, .bfloat16);
    defer sw.deinit();
    try std.testing.expect(try sw.prepare(.{ .rows = 7, .n = 1, .k = 64 }, .random, 4));
    try sw.run();
    const mine = try a.dupe(u8, sw.rig.outputBytes());
    defer a.free(mine);
    const theirs = (try glue.swiglu(sw.args[0], sw.rig.stream())) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(theirs);
    try std.testing.expectEqualSlices(u8, try sw.rig.bytesOf(&.{theirs}), mine);
}

test "kernel contract h3: the VSA launches are the shipped wrappers', byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const Scope = @import("../../mlx/scope.zig").Scope;
    var rig = metal.Rig.init(a);
    defer rig.deinit();
    const s = rig.stream();
    var plan = try vsa.Plan.init(a, &.{ .{ .dense = 37 }, .{ .dense = 20 }, .{ .grid = .{ 4, 4, 12 } }, .{ .grid = .{ 6, 6, 10 } } }, .{ .sparsity = 0.75, .ref_keep_rate = 0.5 }, s);
    defer plan.deinit();
    const g = &plan.geo;
    const h = 2;
    const nt = g.n_tiles;
    const nv = nt - g.n_prefix;
    const ks = g.keyTiles();
    var sc = Scope.init(a, s);
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();
    const vals = try a.alloc(f64, h * (g.seq + 1) * D);
    defer a.free(vals);
    var tiled: [3]mlx.mlx_array = undefined;
    for (&tiled) |*t| {
        for (vals) |*v| v.* = spec.roundTo(.bf16, rng.float(f64) * 4 - 2);
        // Row `seq` of each head is the zero row padding reads, as `attend` builds it.
        for (0..h) |hh| @memset(vals[(hh * (g.seq + 1) + g.seq) * D ..][0..D], 0);
        const x = try rig.input(.bfloat16, &.{ 1, h, @intCast(g.seq + 1), D }, vals);
        t.* = try sc.takeAxis(x, plan.tile_rows, 2);
    }

    // Pool.
    var pool = VsaPool.init(a, &pool_bf16, .bfloat16);
    defer pool.deinit();
    try rig.launch(&pool_bf16, pool.geometry(.{ .rows = nt, .n = h, .k = TILE }).?, &.{ tiled[0], plan.vbs, plan.meta });
    const ours = try a.dupe(u8, rig.outputBytes());
    defer a.free(ours);
    try std.testing.expectEqualSlices(u8, try rig.bytesOf(&.{try vsa.poolKernel(&sc, &plan, tiled[0])}), ours);

    // Attention, on the simdgroup arm and (where the device has it) the NAX arm.
    const sel_host = try a.alloc(i32, h * nv * ks);
    defer a.free(sel_host);
    for (sel_host, 0..) |*x, i| x.* = @intCast(i % ks);
    const sel = mlx.mlx_array_new_data(sel_host.ptr, &[_]c_int{ h, @intCast(nv), @intCast(ks) }, 3, .int32);
    defer _ = mlx.mlx_array_free(sel);
    const scale = mlx.mlx_array_new_float(VsaAttn.SCALE);
    defer _ = mlx.mlx_array_free(scale);
    for ([_]bool{ false, true }) |nax| {
        if (nax and !naxAvailable()) continue;
        plan.arm = if (nax) .nax else .simd;
        var t = VsaAttn.init(a, if (nax) &nax_launch else &simd_launch, nax, .bfloat16);
        defer t.deinit();
        try rig.launch(t.launch, t.geometry(.{ .rows = nv, .n = h, .k = ks }).?, &.{ tiled[0], tiled[1], tiled[2], sel, plan.vbs, plan.meta, scale });
        const mine = try a.dupe(u8, rig.outputBytes());
        defer a.free(mine);
        const theirs = try rig.bytesOf(&.{try vsa.kernelAttend(&sc, &plan, tiled, sel, VsaAttn.SCALE)});
        // Rows past a query tile's valid count are never read: compare the written ones.
        const row_bytes = D * 2;
        for (0..h) |hh| for (0..nv) |qt| {
            const at = (hh * nv + qt) * TILE * row_bytes;
            const n: usize = @intCast(g.vbs[g.n_prefix + qt]);
            try std.testing.expectEqualSlices(u8, theirs[at..][0 .. n * row_bytes], mine[at..][0 .. n * row_bytes]);
        };
    }

    // Assembly.
    const nd: c_int = @intCast(g.dense_rows.len);
    const dense = try sc.takeAxis(tiled[0], try sc.arange(0, nd, 1), 2);
    const video = try sc.slice(tiled[1], 2, @intCast(g.n_prefix * TILE), @intCast(nt * TILE));
    const comp = try sc.astype(try sc.reshape(try sc.slice(tiled[2], 2, 0, @intCast(nt)), &.{ h, @intCast(nt), D }), .float32);
    const gate = try sc.reshape(try sc.transpose(try sc.slice(tiled[1], 2, 0, @intCast(g.seq)), &.{ 0, 2, 1, 3 }), &.{ @intCast(g.seq), h * D });
    var as = VsaAssemble.init(a, &assemble_bf16, .bfloat16);
    defer as.deinit();
    const shape_host = [_]i32{ nd, @intCast(nt), @intCast(nv * TILE), h * D };
    const shp = mlx.mlx_array_new_data(&shape_host, &[_]c_int{4}, 1, .int32);
    defer _ = mlx.mlx_array_free(shp);
    try rig.launch(&assemble_bf16, as.geometry(.{ .rows = g.seq, .n = h, .k = D }).?, &.{ dense, video, comp, gate, plan.out_src, plan.row_tile, shp });
    const asm_ours = try a.dupe(u8, rig.outputBytes());
    defer a.free(asm_ours);
    try std.testing.expectEqualSlices(u8, try rig.bytesOf(&.{try vsa.assembleKernel(&sc, &plan, dense, video, comp, gate)}), asm_ours);
}
