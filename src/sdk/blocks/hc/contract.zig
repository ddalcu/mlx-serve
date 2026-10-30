//! The hc area's kernel contract (Qwen3.8-Flash-Next's hyper-connections, hc 4 x 2560, low rank 320):
//! the prefill norm (bit-equal to x + out * inj -> rms_norm -> weight) and mix (bit-equal to sigmoid -> multiply ->
//! mean, summed in bf16 as that mean is), and the two-launch decode read: the norm + down launch (`nd`) and the
//! up + mix launch (`u2` per row, `u2v` vectorized for 8-bit). The decode launches meet the float64 bound with a
//! forward-error allowance (`host.Val`): they round the norm, SiLU, sigmoid and mix to bf16 after fast-math rsqrt and
//! exp. `nd` writes bf16 and f32 outputs, so it registers once per dtype. Served on every Apple GPU (M1 and up).
//! CUDA twins (`hc_cuda*.cu`, `hc_prefill_cuda*.cu`): no rig yet, not registered.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../testing/kernel_contract/root.zig");
const hp = @import("hc_prefill.zig");
const hd2 = @import("hc_decode2.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Val = host.Val;
const Shape = spec.Shape;
const U = host.U;

/// Flash Next's hyper-connection geometry.
const HC = 4;
const HID = 2560;
const K = HC * HID;
const R = 320;
const EPS: f32 = 1e-6;

/// Threadgroup bytes: `partial[32]` + `inv` (f32).
const norm_launch: metal.Launch = .{
    .name = "msv_hc_prefill_norm",
    .inputs = &.{ "x", "w", "eps", "wo", "wi" },
    .outputs = &.{ "normed", "stream" },
    .source = kernelSource("kernels/hc/hc_prefill_norm.metal"),
    .tg_bytes = 132,
    .m1 = 1024,
    .m2 = 1024,
};

const mix_launch: metal.Launch = .{
    .name = "msv_hc_prefill_mix",
    .inputs = &.{ "up", "normed", "sigtab", "rows" },
    .outputs = &.{"out"},
    .source = kernelSource("kernels/hc/hc_prefill_mix.metal"),
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

/// Threadgroup bytes: `xs[H]` (bf16), `npart[8]`, `tgi[8 * HC]` (f32).
fn ndLaunch(comptime m: u32) metal.Launch {
    return .{
        .name = "mlxserve_hc_read_nd",
        .inputs = &.{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q", "dw_s", "dw_b" },
        .outputs = &.{ "xn_out", "parts_out", "xs_out" },
        .source = hd2.ND_SOURCE,
        .header = hd2.PACK8_HEADER,
        .tg_bytes = 2 * HID + 4 * 8 + 4 * 8 * HC,
        .m1 = m,
        .m2 = m,
    };
}

/// Threadgroup bytes: `acts[R]` (bf16).
fn u2Launch(comptime m: u32) metal.Launch {
    return .{
        .name = "mlxserve_hc_read_u2",
        .inputs = &.{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" },
        .outputs = &.{ "mixed_out", "inj_out" },
        .source = hd2.U2_SOURCE,
        .header = hd2.PACK8_HEADER,
        .tg_bytes = 2 * R,
        .m1 = m,
        .m2 = m,
    };
}

/// Threadgroup bytes: `act_tg[ROWS * R]` (f32) at 16 rows.
fn u2vLaunch(comptime m: u32) metal.Launch {
    return .{
        .name = "mlxserve_hc_read_u2v",
        .inputs = &.{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" },
        .outputs = &.{ "mixed_out", "inj_out" },
        .source = hd2.U2V_SOURCE,
        .header = hd2.PACK8_HEADER,
        .tg_bytes = 4 * 16 * R,
        .m1 = m,
        .m2 = m,
    };
}

// ── prefill norm ──

/// A shape's rows are the streams, `n` the tokens (one batch row), `k` the hidden width: the gates' small shapes
/// vary the streams (the kernels take 1..8), since the wrappers take more than 16 tokens.
const PrefillNorm = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    wr: bool,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    w: []f64 = &.{},
    wo: []f64 = &.{},
    wi: []f64 = &.{},
    args: [5]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, wr: bool) PrefillNorm {
        return .{ .rig = metal.Rig.init(a), .launch = &norm_launch, .wr = wr, .arena = .init(a) };
    }

    pub fn deinit(self: *PrefillNorm) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const PrefillNorm, shape: Shape) ?metal.Geometry {
        const s: c_int = @intCast(shape.n);
        const hc: c_int = @intCast(shape.rows);
        const h: c_int = @intCast(shape.k);
        if (shape.n <= 16 or shape.k % 128 != 0 or shape.k > 4096) return null;
        var g = metal.Geometry{ .grid = .{ s * hc * @divTrunc(h, 4), 1, 1 }, .tg = .{ @divTrunc(h, 4), 1, 1 } };
        g.out(&.{ 1, s, hc, h }, .bfloat16);
        if (self.wr) g.out(&.{ 1, s, hc * h }, .bfloat16) else g.out(&.{ 1, 1, 1 }, .bfloat16);
        if (!self.wr) g.unread = 0b10;
        g.dtype("T", .bfloat16);
        g.int("H", h);
        g.int("HC", hc);
        g.int("WR", @intFromBool(self.wr));
        return g;
    }

    pub fn prepare(self: *PrefillNorm, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const s = shape.n;
        const width = shape.rows * shape.k;
        self.x = try host.uniform(a, rng, s * width, -3, 3, .bfloat16);
        self.w = try host.uniform(a, rng, width, 0.5, 1.5, .bfloat16);
        self.wo = try host.uniform(a, rng, s * shape.k, -2, 2, .bfloat16);
        self.wi = try host.uniform(a, rng, s * shape.rows, 0, 2, .bfloat16);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.x, s * shape.rows, shape.k),
            .stress => {
                for (self.x) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                @memset(self.w, 1);
                @memset(self.wo, 0);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const S: c_int = @intCast(s);
        const x = try rig.input(.bfloat16, &.{ 1, S, @intCast(width) }, self.x);
        self.args = .{
            x,
            try rig.input(.bfloat16, &.{ @intCast(shape.rows), @intCast(shape.k) }, self.w),
            try rig.scalar(EPS),
            if (self.wr) try rig.input(.bfloat16, &.{ 1, S, 1, @intCast(shape.k) }, self.wo) else x,
            if (self.wr) try rig.input(.bfloat16, &.{ 1, S, @intCast(shape.rows), 1 }, self.wi) else x,
        };
        return true;
    }

    pub fn run(self: *PrefillNorm) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const PrefillNorm) usize {
        const n = self.shape.n * self.shape.rows * self.shape.k;
        return if (self.wr) 2 * n else n;
    }

    /// The stream after the pending write (exact chain values), row `row` (token * streams + stream).
    fn streamAt(self: *const PrefillNorm, row: usize, i: usize) f64 {
        const h = self.shape.k;
        const x = self.x[row * h + i];
        if (!self.wr) return x;
        const token = row / self.shape.rows;
        const delta = host.rt(.bfloat16, self.wo[token * h + i] * self.wi[row]);
        return host.rt(.bfloat16, x + delta);
    }

    fn outputsIn(self: *const PrefillNorm, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const h = self.shape.k;
        const rows = self.shape.n * self.shape.rows;
        const v = try self.rig.a.alloc(f64, h);
        defer self.rig.a.free(v);
        const sq = try self.rig.a.alloc(f64, h);
        defer self.rig.a.free(sq);
        for (0..rows) |row| {
            for (v, sq, 0..) |*o, *q, i| {
                o.* = self.streamAt(row, i);
                q.* = o.* * o.*;
            }
            const ss = if (acc) |ac| spec.sumIn(ac, order, sq) else host.sum(sq);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(h)) + EPS);
            const stream_idx = row % self.shape.rows;
            for (want[row * h ..][0..h], v, 0..) |*o, x, i| o.* = x * inv * self.w[stream_idx * h + i];
            if (self.wr) @memcpy(want[rows * h + row * h ..][0..h], v);
        }
        if (mag) |m| for (m, want) |*o, y| {
            o.* = @abs(y) / 2;
        };
    }

    pub fn reference(self: *const PrefillNorm, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const PrefillNorm, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    /// The chain: the pending write (bf16 multiply, add), rms_norm with unit weight, the weight multiply.
    pub fn replaced(self: *PrefillNorm) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const S: c_int = @intCast(self.shape.n);
        const hc: c_int = @intCast(self.shape.rows);
        const h: c_int = @intCast(self.shape.k);
        const x4 = try g.reshape(self.args[0], &.{ 1, S, hc, h });
        const stream = if (self.wr) try g.add(x4, try g.mul(self.args[3], self.args[4])) else x4;
        const normed = try g.mul(try g.rmsNorm(stream, try g.ones(&.{h}, .bfloat16), EPS), self.args[1]);
        return if (self.wr) self.rig.bytesOf(&.{ normed, stream }) else self.rig.bytesOf(&.{normed});
    }

    fn wrapper(self: *PrefillNorm) ![]const u8 {
        const x = self.args;
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        // The inject weight feeds the matmul after the kernel, not the kernel; any [width, hc] stands in.
        const iw = try g.zeros(&.{ @intCast(self.shape.rows * self.shape.k), @intCast(self.shape.rows) }, .bfloat16);
        const n = (try hp.norm(self.rig.stream(), x[0], x[1], iw, x[2], 1, @intCast(self.shape.n), if (self.wr) .{ .out = x[3], .inj = x[4] } else null)) orelse return error.Declined;
        defer n.deinit();
        return if (self.wr) self.rig.bytesOf(&.{ n.normalized, n.stream }) else self.rig.bytesOf(&.{n.normalized});
    }
};

// ── prefill mix ──

const PrefillMix = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    up: []f64 = &.{},
    normed: []f64 = &.{},
    tab: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) PrefillMix {
        return .{ .rig = metal.Rig.init(a), .launch = &mix_launch, .arena = .init(a) };
    }

    pub fn deinit(self: *PrefillMix) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const PrefillMix, shape: Shape) ?metal.Geometry {
        if (shape.n <= 16 or shape.k % 128 != 0 or shape.k > 4096) return null;
        var g = metal.Geometry{ .grid = .{ @intCast(shape.n * shape.k), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ 1, @intCast(shape.n), @intCast(shape.k) }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("HC", shape.rows);
        g.int("H", shape.k);
        return g;
    }

    pub fn prepare(self: *PrefillMix, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.n * shape.rows * shape.k;
        self.up = try host.uniform(a, rng, n, -6, 6, .bfloat16);
        self.normed = try host.uniform(a, rng, n, -3, 3, .bfloat16);
        if (fill == .nonfinite) {
            host.plant(self.normed, shape.n * shape.rows, shape.k);
            host.plant(self.up, 4, shape.k);
        }
        const rig = &self.rig;
        rig.reset();
        // The table is the wrapper's: MLX's sigmoid at every bf16 bit pattern.
        self.tab = try host.read(a, try hp.sigmoidTable(rig.stream()), rig.stream());
        const S: c_int = @intCast(shape.n);
        const sh: []const c_int = &.{ 1, S, @intCast(shape.rows), @intCast(shape.k) };
        self.args = .{
            try rig.input(.bfloat16, sh, self.up),
            try rig.input(.bfloat16, sh, self.normed),
            try rig.input(.bfloat16, &.{65536}, self.tab),
            try host.intScalar(rig, S),
        };
        return true;
    }

    pub fn run(self: *PrefillMix) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const PrefillMix) usize {
        return self.shape.n * self.shape.k;
    }

    /// The streams' sigmoid-gated values summed in bf16 in stream order, times bf16(1 / HC): the chain's own
    /// roundings, so float64 holds the device's bits until the last.
    pub fn reference(self: *const PrefillMix, want: []f64, mag: []f64) !void {
        const hc = self.shape.rows;
        const h = self.shape.k;
        for (0..self.shape.n) |row| for (0..h) |col| {
            var value: f64 = 0;
            var m: f64 = 0;
            for (0..hc) |s| {
                const off = (row * hc + s) * h + col;
                const sig = self.tab[spec.toBf16(@floatCast(self.up[off]))];
                const product = host.rt(.bfloat16, sig * self.normed[off]);
                value = if (s == 0) product else host.rt(.bfloat16, product + value);
                m += @abs(product);
            }
            want[row * h + col] = value * host.rt(.bfloat16, 1.0 / @as(f64, @floatFromInt(hc)));
            mag[row * h + col] = m;
        };
    }

    pub fn replaced(self: *PrefillMix) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const prod = try g.mul(try g.sigmoid(self.args[0]), self.args[1]);
        return self.rig.bytesOf(&.{try g.mean(prod, 2, false)});
    }

    fn wrapper(self: *PrefillMix) ![]const u8 {
        const out = (try hp.mix(self.rig.stream(), self.args[0], self.args[1], 1, @intCast(self.shape.n))) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(out);
        return self.rig.bytesOf(&.{out});
    }
};

// ── decode: norm + down ──

/// Which outputs an `nd` registration reads: the normed and written streams (bf16), or the down/inject partials (f32).
const NdPart = enum { streams, partials };

const Cfg = struct { bits: u32, gs: usize = 64, inj: bool = true, wr: bool = true };

/// A shape's rows are the decode rows; the geometry is Flash Next's.
const Nd = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: Cfg,
    part: NdPart,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    nw: []f64 = &.{},
    iw: []f64 = &.{},
    wo: []f64 = &.{},
    wi: []f64 = &.{},
    dw: host.Quant = undefined,
    args: [9]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: Cfg, part: NdPart) Nd {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .part = part, .arena = .init(a) };
    }

    pub fn deinit(self: *Nd) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Nd, shape: Shape) ?metal.Geometry {
        const c = self.cfg;
        const rows: c_int = @intCast(shape.rows);
        if (shape.rows < 1 or shape.rows > 16) return null;
        const inj: c_int = @intFromBool(c.inj);
        const groups = @divExact(R, 8) + inj;
        var g = metal.Geometry{ .grid = .{ 256, groups * HC, rows }, .tg = .{ 256, 1, 1 } };
        g.out(&.{rows * K}, .bfloat16);
        g.out(&.{rows * HC * (R + HC)}, .float32);
        g.out(&.{if (c.wr) rows * K else 1}, .bfloat16);
        g.unread = if (self.part == .partials) 0b101 else if (c.wr) 0b010 else 0b110;
        g.dtype("T", .bfloat16);
        g.int("HC", HC);
        g.int("H", HID);
        g.int("R", R);
        g.int("GS", c.gs);
        g.int("BITS", c.bits);
        g.int("INJ", inj);
        g.int("WR", @intFromBool(c.wr));
        g.int("RPS", 1);
        g.int("ROWS_TG", 8);
        return g;
    }

    pub fn prepare(self: *Nd, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const rows = shape.rows;
        self.x = try host.uniform(a, rng, rows * K, -4, 4, .bfloat16);
        self.nw = try host.uniform(a, rng, K, 0.5, 1.5, .bfloat16);
        self.iw = try host.uniform(a, rng, K * HC, -0.05, 0.05, .bfloat16);
        self.wo = try host.uniform(a, rng, rows * HID, -1, 1, .bfloat16);
        self.wi = try host.uniform(a, rng, rows * HC, 0.5, 1.5, .bfloat16);
        self.dw = try host.quantize(a, rng, R, K, c.bits, c.gs, .bfloat16);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.x, rows * HC, HID),
            .stress => {
                // Gate 4: rows of +-8 norm to +-1 exactly (fast rsqrt's ulps cannot move a bf16 rounding there), unit
                // weights times small integers; the partials sum integer products past fp16's integers.
                for (self.x) |*v| v.* = if (rng.boolean()) 8 else -8;
                for (self.nw) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, 1, 4));
                for (self.iw) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -4, 4));
                @memset(self.wo, 0);
                try integerWeights(a, rng, &self.dw, R, K, c.bits, c.gs, false);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const rws: c_int = @intCast(rows);
        const words: c_int = @intCast(K * c.bits / 32);
        const dense = c.bits == 16;
        const nw = try rig.input(.bfloat16, &.{ HC, HID }, self.nw);
        self.args = .{
            try rig.input(.bfloat16, &.{ 1, rws, K }, self.x),
            nw,
            if (c.inj) try rig.input(.bfloat16, &.{ K, HC }, self.iw) else nw,
            try rig.input(.float32, &.{1}, &[_]f64{EPS}),
            if (c.wr) try rig.input(.bfloat16, &.{ 1, rws, HID }, self.wo) else nw,
            if (c.wr) try rig.input(.bfloat16, &.{ 1, rws, HC, 1 }, self.wi) else nw,
            try rig.input(.uint32, &.{ R, words }, self.dw.words),
            if (dense) nw else try rig.input(.bfloat16, &.{ R, @intCast(K / c.gs) }, self.dw.scales),
            if (dense) nw else try rig.input(.bfloat16, &.{ R, @intCast(K / c.gs) }, self.dw.biases),
        };
        return true;
    }

    pub fn run(self: *Nd) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Nd) usize {
        const rows = self.shape.rows;
        return switch (self.part) {
            .streams => if (self.cfg.wr) 2 * rows * K else rows * K,
            .partials => rows * HC * (R + HC),
        };
    }

    fn outputsIn(self: *const Nd, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const c = self.cfg;
        const a = self.rig.a;
        const xv = try a.alloc(f64, HID);
        defer a.free(xv);
        const sq = try a.alloc(f64, HID);
        defer a.free(sq);
        const xn = try a.alloc(Val, HID);
        defer a.free(xn);
        const terms = try a.alloc(f64, HID);
        defer a.free(terms);
        const per = HID / 256;
        const ss_depth: f64 = @floatFromInt(per + 5 + 8);
        for (0..self.shape.rows) |row| for (0..HC) |s| {
            const base = s * HID;
            for (xv, sq, 0..) |*o, *q, i| {
                const x = self.x[row * K + base + i];
                o.* = if (c.wr) host.rt(.bfloat16, x + host.rt(.bfloat16, self.wo[row * HID + i] * self.wi[row * HC + s])) else x;
                q.* = o.* * o.*;
            }
            const ss = if (acc) |ac| spec.sumIn(ac, order, sq) else host.sum(sq);
            const rsh = 1 / @sqrt(ss / HID + EPS);
            // The sum's rounding (halved by the square root), the mean, + eps, fast rsqrt (2 ulps), the product.
            const rel = spec.gamma(ss_depth, U) / 2 + 3 * U + 2 * 0x1p-23;
            for (xn, xv, 0..) |*o, v, i| {
                const inner = host.round(.bfloat16, .{ .v = v * rsh, .e = @abs(v * rsh) * rel });
                const w = self.nw[base + i];
                o.* = host.round(.bfloat16, .{ .v = inner.v * w, .e = @abs(w) * inner.e });
                if (self.part == .streams) {
                    const at = row * K + base + i;
                    want[at] = inner.v * w;
                    if (mag) |m| m[at] = host.allowance(@abs(w) * inner.e);
                    if (c.wr) {
                        want[self.shape.rows * K + at] = v;
                        if (mag) |m| m[self.shape.rows * K + at] = 0;
                    }
                }
            }
            if (self.part == .streams) continue;
            const pp = (row * HC + s) * (R + HC);
            // A down row's slice: 16-value dots, the scale/bias combine, the lane's slots, the simdgroup sum.
            const down_depth: f64 = @floatFromInt(16 + 2 + (HID / 16 + 31) / 32 + 5);
            for (0..R) |n| {
                var e: f64 = 0;
                var m: f64 = 0;
                for (terms, xn, 0..) |*t, x, i| {
                    const w = self.dw.w[n * K + base + i];
                    t.* = x.v * w;
                    e += @abs(w) * x.e;
                    m += @abs(t.*) + @abs(x.v * self.biasAt(n, base + i));
                }
                want[pp + n] = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms);
                if (mag) |mm| mm[pp + n] = host.allowance(e + spec.gamma(down_depth, U) * m);
            }
            for (0..HC) |col| {
                var e: f64 = 0;
                var m: f64 = 0;
                for (terms, xn, 0..) |*t, x, i| {
                    const w = if (c.inj) self.iw[(base + i) * HC + col] else 0;
                    t.* = x.v * w;
                    e += @abs(w) * x.e;
                    m += @abs(t.*);
                }
                want[pp + R + col] = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms);
                if (mag) |mm| mm[pp + R + col] = host.allowance(e + spec.gamma(ss_depth, U) * m);
            }
        };
    }

    fn biasAt(self: *const Nd, n: usize, k: usize) f64 {
        return if (self.cfg.bits == 16) 0 else self.dw.biases[n * (K / self.cfg.gs) + k / self.cfg.gs];
    }

    pub fn reference(self: *const Nd, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Nd, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }
};

/// Integer weights for gate 4: scale 1, bias minus half the code range, so a value is code - mid; with `mirror`, the
/// first half of each row is positive (its partial sums grow past fp16's integers), the second negates it, and the
/// first value is 1.
fn integerWeights(a: std.mem.Allocator, rng: std.Random, q: *host.Quant, rows: usize, cols: usize, bits: u32, gs: usize, mirror: bool) !void {
    if (bits == 16) {
        const per_row = cols / 2;
        @memset(q.words, 0);
        for (0..rows) |r| for (0..cols) |c| {
            const half = cols / 2;
            const v: f64 = if (mirror and c == 0) 1 else if (mirror and c >= half) -q.w[r * cols + c - half] else @floatFromInt(rng.intRangeAtMost(i32, if (mirror) 1 else -64, 64));
            q.w[r * cols + c] = v;
            host.setCode(q.words[r * per_row ..][0..per_row], c, 16, spec.toBf16(@floatCast(v)));
        };
        return;
    }
    const mid: u32 = @as(u32, 1) << @intCast(bits - 1);
    @memset(q.scales, 1);
    @memset(q.biases, -@as(f64, @floatFromInt(mid)));
    const codes = try a.alloc(u32, rows * cols);
    defer a.free(codes);
    for (0..rows) |r| for (0..cols) |c| {
        const half = cols / 2;
        codes[r * cols + c] = if (mirror and c == 0) mid + 1 else if (mirror and c >= half) 2 * mid - codes[r * cols + c - half] else rng.intRangeAtMost(u32, if (mirror) mid + 1 else 1, 2 * mid - 1);
    };
    host.dequant(q, rows, cols, bits, gs, codes);
}

// ── decode: up + mix ──

const Up = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: Cfg,
    vec: bool,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    xn: []f64 = &.{},
    parts: []f64 = &.{},
    uw: host.Quant = undefined,
    args: [5]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: Cfg, vec: bool) Up {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .vec = vec, .arena = .init(a) };
    }

    pub fn deinit(self: *Up) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn sgs(rows: usize) c_int {
        return if (rows == 1) 16 else 8;
    }

    pub fn geometry(self: *const Up, shape: Shape) ?metal.Geometry {
        const c = self.cfg;
        const rows: c_int = @intCast(shape.rows);
        if (shape.rows < 1 or shape.rows > 16) return null;
        var g: metal.Geometry = if (self.vec)
            .{ .grid = .{ 32 * sgs(shape.rows) * @divExact(HID, 2 * sgs(shape.rows)), 1, 1 }, .tg = .{ 32 * sgs(shape.rows), 1, 1 } }
        else
            .{ .grid = .{ 32 * 16, HID / 16, rows }, .tg = .{ 32 * 16, 1, 1 } };
        g.out(&.{rows * HID}, .bfloat16);
        g.out(&.{rows * HC}, .bfloat16);
        if (!c.inj) g.unread = 0b10;
        g.dtype("T", .bfloat16);
        g.int("HC", HC);
        g.int("H", HID);
        g.int("R", R);
        g.int("GS", c.gs);
        g.int("BITS", c.bits);
        g.int("INJ", @intFromBool(c.inj));
        if (self.vec) {
            g.int("ROWS", rows);
            g.int("SGS", sgs(shape.rows));
        } else g.int("COLS", 16);
        return g;
    }

    pub fn prepare(self: *Up, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const rows = shape.rows;
        const pr = R + HC;
        self.xn = try host.uniform(a, rng, rows * K, -2, 2, .bfloat16);
        self.parts = try host.uniform(a, rng, rows * HC * pr, -1.5, 1.5, .float32);
        self.uw = try host.quantize(a, rng, K, R, c.bits, c.gs, .bfloat16);
        switch (fill) {
            .random => {},
            .nonfinite => {
                host.plant(self.parts, rows * HC, pr);
                host.plant(self.xn, rows * HC, HID);
            },
            .stress => {
                // Gate 4: activations are integers in 100..200 (SiLU is the identity there), the same in both halves of a
                // row but the first, which is one more in the second half, against mirrored integer weights whose
                // first is 1: each up dot is exactly -1 in fp32 (sigmoid 0.27, far from a bf16 rounding point) after
                // partial sums past fp16's and bf16's integers; the mix of integer streams is then exact.
                @memset(self.parts, 0);
                for (0..rows) |r| for (0..R / 2) |k| {
                    const v: f64 = @floatFromInt(rng.intRangeAtMost(i32, 100, 200));
                    self.parts[r * HC * pr + k] = v;
                    self.parts[r * HC * pr + k + R / 2] = v + @as(f64, if (k == 0) 1 else 0);
                };
                for (self.xn) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -8, 8));
                try integerWeights(a, rng, &self.uw, K, R, c.bits, c.gs, true);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const dense = c.bits == 16;
        const xn = try rig.input(.bfloat16, &.{@intCast(rows * K)}, self.xn);
        self.args = .{
            xn,
            try rig.input(.float32, &.{@intCast(rows * HC * pr)}, self.parts),
            try rig.input(.uint32, &.{ K, @intCast(R * c.bits / 32) }, self.uw.words),
            if (dense) xn else try rig.input(.bfloat16, &.{ K, @intCast(R / c.gs) }, self.uw.scales),
            if (dense) xn else try rig.input(.bfloat16, &.{ K, @intCast(R / c.gs) }, self.uw.biases),
        };
        return true;
    }

    pub fn run(self: *Up) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Up) usize {
        return self.shape.rows * (HID + if (self.cfg.inj) @as(usize, HC) else 0);
    }

    fn outputsIn(self: *const Up, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const a = self.rig.a;
        const pr = R + HC;
        const act = try a.alloc(Val, R);
        defer a.free(act);
        const terms = try a.alloc(f64, R);
        defer a.free(terms);
        var four: [HC]f64 = undefined;
        // u2: four interleaved chains per lane, their pairwise sum, the simdgroup tree; u2v: a lane's quarter of the
        // row in order, then two shuffles.
        const up_depth: f64 = if (self.vec) @floatFromInt(R / 4 + 2) else @floatFromInt((R / (32 / self.cfg.bits) + 31) / 32 * 8 + 2 + 5);
        const inv_hc = host.rt(.bfloat16, 1.0 / @as(f64, HC));
        const rows = self.shape.rows;
        for (0..rows) |row| {
            const p = self.parts[row * HC * pr ..][0 .. HC * pr];
            for (act, 0..) |*o, k| {
                for (&four, 0..) |*t, s| t.* = p[s * pr + k];
                o.* = activation(four, acc, order, true);
            }
            if (self.cfg.inj) for (0..HC) |cc| {
                for (&four, 0..) |*t, s| t.* = p[s * pr + R + cc];
                const sig = activation(four, acc, order, false);
                const at = rows * HID + row * HC + cc;
                want[at] = 2 * sig.v;
                if (mag) |m| m[at] = host.allowance(2 * sig.e);
            };
            for (0..HID) |j| {
                var sum = Val.exact(0);
                var parts_m: [HC]f64 = undefined;
                for (0..HC) |h| {
                    const wrow = h * HID + j;
                    var e: f64 = 0;
                    var m: f64 = 0;
                    for (terms, act, 0..) |*t, x, k| {
                        const w = self.uw.w[wrow * R + k];
                        t.* = x.v * w;
                        e += @abs(w) * x.e;
                        m += @abs(t.*);
                    }
                    const dot = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms);
                    const u = host.round(.bfloat16, .{ .v = dot, .e = e + spec.gamma(up_depth, U) * m });
                    const sgv = host.round(.bfloat16, host.sigmoid(u));
                    const xn = self.xn[row * K + wrow];
                    const mix = host.round(.bfloat16, .{ .v = sgv.v * xn, .e = @abs(xn) * sgv.e });
                    parts_m[h] = mix.v;
                    sum.e += mix.e + spec.gamma(HC, U) * @abs(mix.v);
                }
                sum.v = if (acc) |ac| spec.sumIn(ac, order, &parts_m) else host.sum(&parts_m);
                const y = host.round(.bfloat16, sum);
                want[row * HID + j] = y.v * inv_hc;
                if (mag) |m| m[row * HID + j] = host.allowance(inv_hc * y.e);
            }
        }
    }

    pub fn reference(self: *const Up, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Up, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }
};

/// v = T(sum of the streams' partials), then silu (T(v * T(sigmoid(v)))) or the sigmoid alone.
fn activation(p: [HC]f64, acc: ?spec.Accum, order: spec.Order, silu: bool) Val {
    var m: f64 = 0;
    for (p) |x| m += @abs(x);
    const sum = if (acc) |ac| spec.sumIn(ac, order, &p) else host.sum(&p);
    const v = host.round(.bfloat16, .{ .v = sum, .e = spec.gamma(HC, U) * m });
    const sig = host.round(.bfloat16, host.sigmoid(v));
    if (!silu) return sig;
    return host.round(.bfloat16, .{ .v = v.v * sig.v, .e = @abs(sig.v) * v.e + @abs(v.v) * sig.e });
}

/// The streams' partials: four per thread sequentially, the simdgroup tree, the partials' tree.
fn prefillDepth(_: Shape) usize {
    return 4 + 5 + 5;
}

fn mixDepth(shape: Shape) usize {
    return shape.rows;
}

const prefill_shapes: []const Shape = &.{ .{ .rows = HC, .n = 17, .k = HID }, .{ .rows = HC, .n = 64, .k = HID } };
const decode_rows: []const Shape = &.{ .{ .rows = 1, .n = R, .k = K }, .{ .rows = 4, .n = R, .k = K }, .{ .rows = 16, .n = R, .k = K } };
const decode_one: []const Shape = &.{.{ .rows = 1, .n = R, .k = K }};

fn prefillNormSpec(comptime name: []const u8, wr: bool) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = prefillDepth,
        // The mean, + eps, precise rsqrt, the product with the row, the weight product.
        .ops = 5,
        .mids = 1,
        .out = .bf16,
        .replaces = if (wr) "multiply -> add -> rms_norm -> multiply" else "rms_norm -> multiply",
        .served = prefill_shapes,
        .sums = .{ .f16 = 32, .bf16 = 0, .why = "positive squares: a bf16 sum stays near the output's rounding" },
    };
}

fn decodeSpec(comptime name: []const u8, out: spec.Elem, served: []const Shape, sums: spec.Sums) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        // The reference carries every rounding as its allowance (`host.allowance`).
        .depth = host.depthOne,
        .out = out,
        .served = served,
        .sums = sums,
    };
}

/// The norm's output: its sum of squares sits behind the rsqrt, where bf16 accumulation stays near the rounding.
const norm_sums: spec.Sums = .{ .f16 = 100, .bf16 = 0, .why = "the squares' sum reaches the output through rsqrt" };

// M1/M2 records (`offline.zig`): nd 512 where conclusive (the 8-bit write + inject build is not); u2v 576 at 16
// rows, the other row counts inconclusive; u2 inconclusive at every shape, so its record stays 1024.
const nd8 = ndLaunch(512);
const nd4 = ndLaunch(512);
const nd16 = ndLaunch(512);
const up4 = u2Launch(1024);
const up16 = u2Launch(1024);
const upv8 = u2vLaunch(576);

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(PrefillNorm, .init(a, false), prefillNormSpec("hc_prefill_norm bf16", false), null);
    try set.add(PrefillNorm, .init(a, true), prefillNormSpec("hc_prefill_norm bf16 (pending write)", true), null);
    try set.add(PrefillMix, .init(a), .{
        .name = "hc_prefill_mix bf16",
        // MLX's mean over the bf16 streams accumulates in bf16; the kernel reproduces it.
        .accum = .bf16,
        .depth = mixDepth,
        .out = .bf16,
        .replaces = "sigmoid -> multiply -> mean",
        .served = prefill_shapes,
    }, null);
    const exact: spec.Sums = .{};
    try set.add(Nd, .init(a, &nd8, .{ .bits = 8 }, .streams), decodeSpec("hc_read_nd 8-bit (normed + written streams)", .bf16, decode_rows, norm_sums), null);
    try set.add(Nd, .init(a, &nd8, .{ .bits = 8 }, .partials), decodeSpec("hc_read_nd 8-bit (down + inject partials)", .f32, decode_rows, exact), null);
    try set.add(Nd, .init(a, &nd8, .{ .bits = 8, .wr = false }, .streams), decodeSpec("hc_read_nd 8-bit (no pending write)", .bf16, decode_one, norm_sums), null);
    try set.add(Nd, .init(a, &nd8, .{ .bits = 8, .inj = false }, .streams), decodeSpec("hc_read_nd 8-bit (no inject)", .bf16, decode_one, norm_sums), null);
    try set.add(Nd, .init(a, &nd4, .{ .bits = 4 }, .partials), decodeSpec("hc_read_nd 4-bit (down + inject partials)", .f32, decode_rows, exact), null);
    try set.add(Nd, .init(a, &nd16, .{ .bits = 16 }, .partials), decodeSpec("hc_read_nd bf16 weights (down + inject partials)", .f32, decode_one, exact), null);
    const up_sums: spec.Sums = .{ .f16 = 100, .bf16 = 100 };
    try set.add(Up, .init(a, &upv8, .{ .bits = 8 }, true), decodeSpec("hc_read_u2v 8-bit", .bf16, decode_rows, up_sums), null);
    try set.add(Up, .init(a, &up4, .{ .bits = 4 }, false), decodeSpec("hc_read_u2 4-bit", .bf16, decode_rows, up_sums), null);
    try set.add(Up, .init(a, &up16, .{ .bits = 16, .inj = false }, false), decodeSpec("hc_read_u2 bf16 weights (no inject)", .bf16, decode_one, up_sums), null);
}

// ── tests ──

/// The decode registrations' two launches chained (norm + down, then up + mix) equal the shipped read's outputs.
fn decodeDrift(a: std.mem.Allocator, nd: *Nd, up: *Up, rows: usize) !void {
    const c = nd.cfg;
    try std.testing.expect(try nd.prepare(.{ .rows = rows, .n = R, .k = K }, .random, 11));
    var geo = nd.geometry(nd.shape).?;
    geo.unread = 0;
    try nd.rig.launch(nd.launch, geo, &nd.args);
    try std.testing.expect(try up.prepare(.{ .rows = rows, .n = R, .k = K }, .random, 12));
    // The up launch reads the declared nd launch's outputs and the nd test's up weights.
    up.args[0] = nd.rig.output(0);
    up.args[1] = nd.rig.output(1);
    try up.run();
    const ours = try a.dupe(u8, up.rig.outputBytes());
    defer a.free(ours);
    const x = nd.args;
    const u = up.args;
    const dense = c.bits == 16;
    const none = mlx.mlx_array{ .ctx = null };
    const out = (try hd2.read(nd.rig.stream(), x[0], x[1], x[6], if (dense) none else x[7], if (dense) none else x[8], u[2], if (dense) none else u[3], if (dense) none else u[4], if (c.inj) x[2] else none, x[3], @intCast(rows), HC, HID, c.bits, @intCast(c.gs), if (c.wr) .{ .out = x[4], .inj = x[5] } else null)) orelse return error.Declined;
    defer for ([_]mlx.mlx_array{ out.mixed, out.inj, out.stream }) |o| if (o.ctx != null) {
        _ = mlx.mlx_array_free(o);
    };
    const theirs = if (c.inj) try up.rig.bytesOf(&.{ out.mixed, out.inj }) else try up.rig.bytesOf(&.{out.mixed});
    try std.testing.expectEqualSlices(u8, theirs, ours);
}

test "kernel contract hc: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |wr| {
        var n = PrefillNorm.init(a, wr);
        defer n.deinit();
        try std.testing.expect(try n.prepare(prefill_shapes[0], .random, 3));
        try n.run();
        const ours = try a.dupe(u8, n.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try n.wrapper(), ours);
    }
    {
        var m = PrefillMix.init(a);
        defer m.deinit();
        try std.testing.expect(try m.prepare(prefill_shapes[0], .random, 4));
        try m.run();
        const ours = try a.dupe(u8, m.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try m.wrapper(), ours);
    }
    // The down weights the wrapper reads are the nd test's; the up launch binds them through `u`'s slots.
    inline for (.{ .{ Cfg{ .bits = 8 }, &nd8, &upv8, true, 4 }, .{ Cfg{ .bits = 4 }, &nd4, &up4, false, 3 }, .{ Cfg{ .bits = 16, .inj = false }, &nd16, &up16, false, 1 } }) |case| {
        var nd = Nd.init(a, case[1], case[0], .streams);
        defer nd.deinit();
        var up = Up.init(a, case[2], case[0], case[3]);
        defer up.deinit();
        try decodeDrift(a, &nd, &up, case[4]);
    }
}
