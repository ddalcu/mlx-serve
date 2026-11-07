//! Ming TTS's kernel contract: the fused decode kernels of `fused.zig`, DiT (A-D, E, F) and LM (LA-LE), one test type
//! per area of arithmetic they share. Listed in `kernel_contract/registry.zig`.
//! The kernels round to bf16 at the op chain's points but fold rmsnorm into the GEMV and sum in their own order, so no
//! chain is bit-equal: each output is held to float64 with its forward error carried value by value (`host.Val`).
//! Shapes: DiT `rows` R and `n` B (L = R / B); `lm_attn` `rows` Lc; `lm_qkv` `n` POS; the rest are fixed geometry and
//! ignore `rows`. A's ATT is checked against attention over the OUT the same dispatch wrote; OUT has its own check.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const fused = @import("fused.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;
const Val = host.Val;
const U = host.U;

const D = 1024;
const FF = 4096;
const WL = 896;
const MLP = 4864;
const CFG: f32 = 2;
const STEP = 3;
const LOG2_THETA: f32 = @log2(@as(f32, 10000));

const Kind = enum { dit_qkv, dit_out, dit_ff1, dit_ff2, embed, final, lm_qkv, lm_attn, lm_o, lm_up, lm_down };

/// GEMV threadgroup bytes: `red[SG * NO * R]` and `ssr[SG * R]` floats.
fn gemvBytes(no: u32, r: u32) u32 {
    return 4 * (4 * no * r + 4 * r);
}

/// `limit`: the M1/M2 thread limit `offline.zig` compiles, the least over the launch's shapes.
fn launchOf(comptime name: [:0]const u8, comptime inputs: []const [*:0]const u8, comptime outputs: []const [*:0]const u8, comptime source: [:0]const u8, comptime tg_bytes: u32, comptime limit: u32) metal.Launch {
    return .{ .name = name, .inputs = inputs, .outputs = outputs, .source = source, .header = fused.HEADER, .tg_bytes = tg_bytes, .m1 = limit, .m2 = limit };
}

// A adds `hq[3 * R * 32]` uints and a bool at R 6; LB `q[64]`, `sc[512]`, `rd[256]` floats.
const qkv_launch = launchOf("ming_dit_qkv", &.{ "X", "NW", "WQ", "WK", "WV", "BQ", "BK", "BV", "COS", "SIN", "CNT" }, &.{ "OUT", "ATT" }, fused.SRC_QKV, gemvBytes(2, 6) + 2304 + 4, 640);
const attn_launch = launchOf("ming_dit_attn_out", &.{ "ATT", "X", "WO", "BO" }, &.{"OUT"}, fused.SRC_ATTN, gemvBytes(2, 6), 1024);
const ff1_launch = launchOf("ming_dit_ff1", &.{ "X", "NW", "W1", "B1" }, &.{"OUT"}, fused.SRC_FF1, gemvBytes(2, 6), 640);
const ff2_launch = launchOf("ming_dit_ff2", &.{ "HH", "X", "W2", "B2" }, &.{"OUT"}, fused.SRC_FF2, gemvBytes(2, 6), 1024);
const embed_launch = launchOf("ming_cfm_embed", &.{ "Y", "WX", "BX", "TE", "CE", "XH" }, &.{"OUT"}, fused.SRC_EMBED, 0, 1024);
const final_launch = launchOf("ming_cfm_final", &.{ "X", "NW", "WF", "BF", "Y", "DTT", "SQ", "SDE" }, &.{"OUT"}, fused.SRC_FINAL, 0, 1024);
const lm_qkv_launch = launchOf("ming_lm_qkv", &.{ "X", "NW", "WQ", "WK", "WV", "BQ", "BK", "BV", "POS" }, &.{"OUT"}, fused.SRC_LM_QKV, gemvBytes(2, 1), 1024);
const lm_attn_launch = launchOf("ming_lm_attn", &.{ "QKV", "KC", "VC" }, &.{ "OUT", "KN", "VN" }, fused.SRC_LM_ATTN, 4 * (64 + 512 + 256), 1024);
const lm_o_launch = launchOf("ming_lm_o", &.{ "O", "X", "WO" }, &.{"OUT"}, fused.SRC_LM_O, gemvBytes(2, 1), 1024);
const lm_up_launch = launchOf("ming_lm_up", &.{ "X", "NW", "WG", "WU" }, &.{"OUT"}, fused.SRC_LM_UP, gemvBytes(4, 1), 1024);
const lm_down_launch = launchOf("ming_lm_down", &.{ "HH", "X", "WD" }, &.{"OUT"}, fused.SRC_LM_DOWN, gemvBytes(2, 1), 1024);

/// What an input holds, which decides how `.random`, `.nonfinite` and `.stress` fill it. `act`: activations a
/// non-finite row may hit (DiT); `fixed`: the same but kept finite (one row, or a row every output reads); `res`: the
/// residual x.
const Role = enum { act, fixed, res, norm, w, bias, cos, sin, count, y, dtt, sq, sde, qkv, kv_k, kv_v, pos };

const In = struct {
    role: Role,
    dims: [4]c_int = @splat(0),
    nd: usize = 0,

    fn len(i: In) usize {
        var n: usize = 1;
        for (i.dims[0..i.nd]) |d| n *= @intCast(d);
        return n;
    }

    /// A row's width: the last dimension.
    fn width(i: In) usize {
        return if (i.nd == 0) 1 else @intCast(i.dims[i.nd - 1]);
    }

    fn dtype(i: In) mlx.mlx_dtype {
        return switch (i.role) {
            .cos, .sin => .float32,
            .count, .pos => .int32,
            else => .bfloat16,
        };
    }
};

const Layout = struct {
    items: [11]In = undefined,
    n: usize = 0,

    fn add(l: *Layout, role: Role, dims: []const c_int) void {
        var i = In{ .role = role, .nd = dims.len };
        @memcpy(i.dims[0..dims.len], dims);
        l.items[l.n] = i;
        l.n += 1;
    }
};

/// The inputs the wrapper binds at `shape`, in its order and at its shapes.
fn layout(kind: Kind, shape: Shape) Layout {
    var l = Layout{};
    const r: c_int = @intCast(shape.rows);
    const b: c_int = @intCast(@max(shape.n, 1));
    const seq = @divTrunc(r, b);
    switch (kind) {
        .dit_qkv => {
            l.add(.act, &.{ b, seq, D });
            l.add(.norm, &.{D});
            for (0..3) |_| l.add(.w, &.{ D, D });
            for (0..3) |_| l.add(.bias, &.{D});
            l.add(.cos, &.{ 1, 1, seq, 32 });
            l.add(.sin, &.{ 1, 1, seq, 32 });
            l.add(.count, &.{D});
        },
        .dit_out => {
            l.add(.act, &.{ r, D });
            l.add(.res, &.{ b, seq, D });
            l.add(.w, &.{ D, D });
            l.add(.bias, &.{D});
        },
        .dit_ff1 => {
            l.add(.act, &.{ b, seq, D });
            l.add(.norm, &.{D});
            l.add(.w, &.{ FF, D });
            l.add(.bias, &.{FF});
        },
        .dit_ff2 => {
            l.add(.act, &.{ r, FF });
            l.add(.res, &.{ b, seq, D });
            l.add(.w, &.{ D, FF });
            l.add(.bias, &.{D});
        },
        .embed => {
            l.add(.fixed, &.{ 1, 1, 64 });
            l.add(.w, &.{ D, 64 });
            l.add(.bias, &.{D});
            l.add(.res, &.{ 2, 1, D });
            l.add(.res, &.{ 2, 1, D });
            l.add(.res, &.{ 1, 1, D });
        },
        .final => {
            l.add(.fixed, &.{ 2, seq, D });
            l.add(.norm, &.{D});
            l.add(.w, &.{ 64, D });
            l.add(.bias, &.{64});
            l.add(.y, &.{ 1, 1, 64 });
            l.add(.dtt, &.{10});
            l.add(.sq, &.{10});
            l.add(.sde, &.{ 10, 1, 1, 64 });
        },
        .lm_qkv => {
            l.add(.fixed, &.{ 1, 1, WL });
            l.add(.norm, &.{WL});
            l.add(.w, &.{ WL, WL });
            l.add(.w, &.{ 128, WL });
            l.add(.w, &.{ 128, WL });
            l.add(.bias, &.{WL});
            l.add(.bias, &.{128});
            l.add(.bias, &.{128});
            l.add(.pos, &.{});
        },
        .lm_attn => {
            l.add(.qkv, &.{1152});
            l.add(.kv_k, &.{ 1, 2, r, 64 });
            l.add(.kv_v, &.{ 1, 2, r, 64 });
        },
        .lm_o => {
            l.add(.fixed, &.{ 1, 1, WL });
            l.add(.res, &.{ 1, 1, WL });
            l.add(.w, &.{ WL, WL });
        },
        .lm_up => {
            l.add(.fixed, &.{ 1, 1, WL });
            l.add(.norm, &.{WL});
            l.add(.w, &.{ MLP, WL });
            l.add(.w, &.{ MLP, WL });
        },
        .lm_down => {
            l.add(.fixed, &.{MLP});
            l.add(.res, &.{ 1, 1, WL });
            l.add(.w, &.{ WL, MLP });
        },
    }
    return l;
}

// ── f32 arithmetic with its forward error ──

fn addV(a: Val, b: Val) Val {
    const v = a.v + b.v;
    return .{ .v = v, .e = a.e + b.e + U * @abs(v) };
}

fn subV(a: Val, b: Val) Val {
    return addV(a, .{ .v = -b.v, .e = b.e });
}

fn mulV(a: Val, b: Val) Val {
    const m = host.mul(a, b);
    return .{ .v = m.v, .e = m.e + U * @abs(m.v) };
}

fn rb(x: Val) Val {
    return host.round(.bfloat16, x);
}

/// A kernel's `rb(constant)`.
fn cst(x: f32) Val {
    return .{ .v = spec.roundTo(.bf16, x) };
}

/// `sum * rinv + bias` as the DiT and LM GEMVs fold them.
fn normed(sum: Val, inv: Val, bias: f64) Val {
    return addV(mulV(sum, inv), .exact(bias));
}

/// The interleaved rotary: rounded once by the caller's store.
fn rotate(a: Val, b: Val, c: Val, s: Val) [2]Val {
    return .{ subV(mulV(a, c), mulV(b, s)), addV(mulV(a, s), mulV(b, c)) };
}

/// `dit_ff1`'s gelu_approx with the op chain's bf16 roundings; x*x*x is exact in f32 up to x's own error.
fn gelu(x: Val) Val {
    const x3 = rb(mulV(mulV(x, x), x));
    const inner = rb(addV(x, rb(mulV(cst(0.044715), x3))));
    const a = rb(mulV(cst(0.7978845608), inner));
    const t = std.math.tanh(a.v);
    const th = rb(.{ .v = t, .e = (1 - t * t) * a.e + 4 * 2 * U * @abs(t) });
    return mulV(rb(mulV(.exact(0.5), x)), rb(addV(.exact(1), th)));
}

const Sim = ?struct { acc: spec.Accum, order: spec.Order };

fn put(want: []f64, mag: ?[]f64, i: usize, x: Val) void {
    want[i] = x.v;
    if (mag) |m| m[i] = host.allowance(x.e);
}

const Ming = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: Kind,
    /// `ming_cfm_final`'s sigma * sqrt(temperature); 0 leaves its SDE term out.
    shift: f32 = 0,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    fill: spec.Fill = .random,
    v: [11][]f64 = undefined,
    args: [11]mlx.mlx_array = undefined,
    n_args: usize = 0,
    /// x * norm weight (or x) per row: what every GEMV term multiplies a weight row by.
    xn: []f64 = &.{},
    terms: []f64 = &.{},
    keys: []f64 = &.{},
    vals: []f64 = &.{},
    aux: [3][]f64 = undefined,
    /// The float64 outputs and allowances of the prepared call, computed once.
    ref: []f64 = &.{},

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, kind: Kind, shift: f32) Ming {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .kind = kind, .shift = shift, .arena = .init(a) };
    }

    pub fn deinit(self: *Ming) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn gemv(g: *metal.Geometry, cols: c_int, no: c_int) void {
        const tg = 32 * 4;
        g.grid = .{ @divExact(cols, no) * tg, 1, 1 };
        g.tg = .{ tg, 1, 1 };
        g.int("NO", no);
        g.int("SG", 4);
    }

    pub fn geometry(self: *const Ming, shape: Shape) ?metal.Geometry {
        var g = metal.Geometry{ .grid = undefined, .tg = undefined };
        const r: c_int = @intCast(shape.rows);
        const b: c_int = @intCast(@max(shape.n, 1));
        const seq = @divTrunc(r, b);
        const dit = switch (self.kind) {
            .dit_qkv, .dit_out, .dit_ff1, .dit_ff2, .final => true,
            else => false,
        };
        if (dit and (r == 0 or r > 6 or @rem(r, b) != 0)) return null;
        switch (self.kind) {
            .dit_qkv => {
                gemv(&g, 3 * D, 2);
                g.out(&.{ 3, r, D }, .bfloat16);
                g.out(&.{ r, D }, .bfloat16);
                g.int("R", r);
                g.int("L", seq);
                g.int("NTG", 3 * 64 / 2);
            },
            .dit_out, .dit_ff2 => {
                gemv(&g, D, 2);
                g.out(&.{ b, seq, D }, .bfloat16);
                g.int("R", r);
                g.int("L", seq);
            },
            .dit_ff1 => {
                gemv(&g, FF, 2);
                g.out(&.{ r, FF }, .bfloat16);
                g.int("R", r);
                g.int("L", seq);
            },
            .embed => {
                g.grid = .{ D, 1, 1 };
                g.tg = .{ 128, 1, 1 };
                g.out(&.{ 2, 3, D }, .bfloat16);
            },
            .final => {
                g.grid = .{ 16 * 128, 1, 1 };
                g.tg = .{ 128, 1, 1 };
                g.out(&.{ 1, 1, 64 }, .bfloat16);
                g.int("L", seq);
                g.int("S", STEP);
                g.int("CFG", @as(c_int, @bitCast(CFG)));
                g.int("SHIFT", @as(c_int, @bitCast(self.shift)));
            },
            .lm_qkv => {
                gemv(&g, 1152, 2);
                g.out(&.{1152}, .bfloat16);
                g.int("R", 1);
                g.int("LOG2BASE", @as(c_int, @bitCast(LOG2_THETA)));
            },
            .lm_attn => {
                if (r == 0 or r > 511) return null;
                g.grid = .{ 14 * 256, 1, 1 };
                g.tg = .{ 256, 1, 1 };
                g.out(&.{ 1, 1, WL }, .bfloat16);
                g.out(&.{ 1, 2, r + 1, 64 }, .bfloat16);
                g.out(&.{ 1, 2, r + 1, 64 }, .bfloat16);
            },
            .lm_o, .lm_down => {
                gemv(&g, WL, 2);
                g.out(&.{ 1, 1, WL }, .bfloat16);
                g.int("R", 1);
            },
            .lm_up => {
                gemv(&g, 2 * MLP, 4);
                g.out(&.{MLP}, .bfloat16);
                g.int("R", 1);
            },
        }
        return g;
    }

    /// One row a half at +`a` and a half at -`a`, signed per row.
    fn signedRows(x: []f64, width: usize, a: f64, rng: std.Random) void {
        var r: usize = 0;
        while (r < x.len) : (r += width) {
            const s: f64 = if (rng.boolean()) a else -a;
            for (x[r..][0..width], 0..) |*e, i| e.* = if (i < width / 2) s else -s;
        }
    }

    /// Gate 4's weights: integers exact in bf16. A row's second half is its first rotated by one, so against
    /// `signedRows` the dot is a small integer, no summation tree pairs the halves alike and the partial sums leave
    /// fp16's range and bf16's precision (odd small values among multiples of 16), while staying exact in fp32.
    fn stressWeights(w: []f64, k: usize, rng: std.Random) void {
        const h = k / 2;
        var r: usize = 0;
        while (r < w.len) : (r += k) {
            const row = w[r..][0..k];
            for (row[0..h], 0..) |*e, i| e.* = @floatFromInt(if (i % 16 == 0) 2 * rng.uintLessThan(u32, 64) + 1 else 16 * (128 + rng.uintLessThan(u32, 64)));
            for (row[h..], 0..) |*e, i| e.* = row[(i + 1) % h] + @as(f64, if (i == 15 or i == 31) @floatFromInt(rng.uintLessThan(u32, 2)) else 0);
        }
    }

    fn fillRandom(self: *Ming, i: usize, in: In, rng: std.Random) void {
        const x = self.v[i];
        const r = struct { lo: f64, hi: f64 };
        const span: r = switch (in.role) {
            .act, .fixed, .res, .qkv, .kv_k, .kv_v => .{ .lo = -2, .hi = 2 },
            .norm => .{ .lo = 0.5, .hi = 1.5 },
            .w => .{ .lo = -0.05, .hi = 0.05 },
            .bias => .{ .lo = -0.1, .hi = 0.1 },
            .y, .sde => .{ .lo = -1, .hi = 1 },
            .dtt => .{ .lo = 0.05, .hi = 0.2 },
            .sq => .{ .lo = 0.2, .hi = 0.5 },
            .cos, .sin => {
                if (in.role == .cos) for (x, self.v[i + 1]) |*c, *s| {
                    const th = rng.float(f64) * 2 * std.math.pi;
                    c.* = spec.roundTo(.f32, @cos(th));
                    s.* = spec.roundTo(.f32, @sin(th));
                };
                return;
            },
            .count => {
                @memset(x, 0);
                return;
            },
            .pos => return,
        };
        host.uniformInto(rng, x, span.lo, span.hi, .bfloat16);
    }

    fn fillStress(self: *Ming, i: usize, in: In, rng: std.Random) void {
        const x = self.v[i];
        switch (in.role) {
            .act, .fixed => signedRows(x, in.width(), if (self.kind == .embed) 64 else 1, rng),
            .res, .y, .sde => host.intsInto(rng, x, -8, 8),
            .norm => @memset(x, 1),
            .w => stressWeights(x, in.width(), rng),
            .bias => @memset(x, 0),
            .dtt => @memset(x, 0.25),
            .sq => @memset(x, 0.5),
            .cos => for (x, self.v[i + 1]) |*c, *s| {
                const t = [4][2]f64{ .{ 1, 0 }, .{ 0, 1 }, .{ -1, 0 }, .{ 0, -1 } };
                const q = t[rng.uintLessThan(usize, 4)];
                c.* = q[0];
                s.* = q[1];
            },
            // Keys rotated across their halves against q at +-251: dots a few levels apart, partial sums past fp16.
            .qkv => {
                signedRows(x[0 .. 14 * 64], 64, 8 * 251, rng);
                keyRows(x[14 * 64 .. 16 * 64], rng);
                host.intsInto(rng, x[16 * 64 ..], -8, 8);
            },
            .kv_k => keyRows(x, rng),
            .kv_v => host.intsInto(rng, x, -8, 8),
            .count => @memset(x, 0),
            .sin, .pos => {},
        }
    }

    fn keyRows(k: []f64, rng: std.Random) void {
        var r: usize = 0;
        while (r < k.len) : (r += 64) {
            const row = k[r..][0..64];
            for (row[0..32]) |*e| e.* = @floatFromInt(rng.uintLessThan(u32, 128));
            for (row[32..], 0..) |*e, i| e.* = row[(i + 1) % 32] + @as(f64, if (i < 2) @floatFromInt(rng.uintLessThan(u32, 2)) else 0);
        }
    }

    /// Weights hold every class by column; DiT rows and residuals hold them by row. Heads differ per weight so q, k
    /// and v plant in different heads of A.
    fn plant(self: *Ming, i: usize, in: In) void {
        const x = self.v[i];
        const width = in.width();
        const rows = x.len / width;
        switch (in.role) {
            .w => {
                const first = if (rows > 256) (i % 3) * 64 else 0;
                host.plant(x[first * width ..], @min(rows, 4), width);
            },
            .act => host.plant(x, 3, width),
            .res => host.plant(x, @min(rows, 3), width),
            .y => host.plant(x, 1, width),
            .qkv => host.plant(x[7 * 64 ..], 4, 64),
            .kv_v => host.plant(x, @min(rows, 3), width),
            else => {},
        }
    }

    pub fn prepare(self: *Ming, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        self.fill = fill;
        self.ref = &.{};
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const lay = layout(self.kind, shape);
        const ins = lay.items[0..lay.n];
        for (ins, 0..) |in, i| self.v[i] = try a.alloc(f64, in.len());
        for (ins, 0..) |in, i| if (fill == .stress) self.fillStress(i, in, rng) else self.fillRandom(i, in, rng);
        if (fill == .nonfinite) for (ins, 0..) |in, i| self.plant(i, in);
        const rig = &self.rig;
        rig.reset();
        self.n_args = lay.n;
        for (ins, 0..) |in, i| self.args[i] = if (in.role == .pos)
            try host.intScalar(rig, @intCast(shape.n))
        else
            try rig.input(in.dtype(), in.dims[0..in.nd], self.v[i]);
        const width = ins[0].width();
        self.xn = try a.dupe(f64, self.v[0]);
        if (ins[1].role == .norm) {
            var r: usize = 0;
            while (r < self.xn.len) : (r += width) for (self.xn[r..][0..width], self.v[1]) |*x, w| {
                x.* *= w;
            };
        }
        self.terms = try a.alloc(f64, @max(MLP, 512));
        if (self.kind == .dit_qkv or self.kind == .lm_attn) {
            self.keys = try a.alloc(f64, 512 * 64);
            self.vals = try a.alloc(f64, 512 * 64);
            for (&self.aux) |*x| x.* = try a.alloc(f64, 512);
        }
        return true;
    }

    pub fn run(self: *Ming) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Ming) usize {
        const r = self.shape.rows;
        return switch (self.kind) {
            .dit_qkv => 4 * r * D,
            .dit_out, .dit_ff2 => r * D,
            .dit_ff1 => r * FF,
            .embed => 6 * D,
            .final => 64,
            .lm_qkv => 1152,
            .lm_attn => WL + 4 * (r + 1) * 64,
            .lm_o, .lm_down => WL,
            .lm_up => MLP,
        };
    }

    /// A row's dot with a weight row. Gate 4's integer terms sum exactly in fp32 (`simulate(.f32)` shows it).
    fn sumOf(self: *const Ming, xn: []const f64, w: []const f64, sim: Sim) Val {
        if (sim) |s| {
            const t = self.terms[0..xn.len];
            for (t, xn, w) |*o, x, y| o.* = x * y;
            return .exact(spec.sumIn(s.acc, s.order, t));
        }
        const d = host.dot(xn, w);
        return .{ .v = d[0], .e = if (self.fill == .stress) 0 else spec.gamma(@floatFromInt(xn.len), U) * d[1] };
    }

    /// `rsqrt(ss / K + 1e-6)`, its squares summed in `sim`'s accumulator; the mean, eps and rsqrt add a few roundings.
    fn invOf(self: *const Ming, x: []const f64, sim: Sim) Val {
        var ss: f64 = 0;
        var rel: f64 = 6 * U;
        if (sim) |s| {
            const t = self.terms[0..x.len];
            for (t, x) |*o, y| o.* = y * y;
            ss = spec.sumIn(s.acc, s.order, t);
        } else {
            ss = host.dot(x, x)[0];
            if (self.fill != .stress) rel += spec.gamma(@floatFromInt(x.len), U) / 2;
        }
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + 1e-6);
        return .{ .v = inv, .e = inv * rel };
    }

    /// Softmax over `n` keys (`self.keys`, `self.vals`, rows of 64) for a scaled q: the two-pass max + exp, then
    /// sum(p v) / sum(p). Exp is fast (`host.fastExp`); the division a fast divide.
    fn attend(self: *const Ming, q: []const f64, n: usize, out: []Val, sim: Sim) void {
        const sc = self.aux[0][0..n];
        const rel = self.aux[1][0..n];
        const p = self.aux[2][0..n];
        const g64 = spec.gamma(64, U);
        const gn = spec.gamma(@floatFromInt(n), U);
        const exact = self.fill == .stress;
        var m = -std.math.inf(f64);
        var em: f64 = 0;
        for (sc, rel, 0..) |*s, *r, j| {
            const row = self.keys[j * 64 ..][0..64];
            const d = host.dot(q, row);
            r.* = if (exact) 0 else g64 * d[1];
            if (sim) |x| {
                const t = self.terms[0..64];
                for (t, q, row) |*o, a, b| o.* = a * b;
                s.* = spec.sumIn(x.acc, x.order, t);
            } else s.* = d[0];
            m = @max(m, s.*);
            em = @max(em, r.*);
        }
        var den: f64 = 0;
        var err_den: f64 = 0;
        for (sc, rel, p) |s, *r, *o| {
            const d = s - m;
            o.* = @exp(d);
            r.* += em + U * @abs(d) + host.fastExp(d);
            den += o.*;
            err_den += o.* * r.*;
        }
        if (sim) |x| den = spec.sumIn(x.acc, x.order, p);
        err_den += gn * den;
        for (out, 0..) |*y, d| {
            const t = self.terms[0..n];
            var abs: f64 = 0;
            var err_num: f64 = 0;
            for (t, p, rel, 0..) |*o, pj, r, j| {
                o.* = pj * self.vals[j * 64 + d];
                abs += @abs(o.*);
                err_num += @abs(o.*) * (r + U);
            }
            const num = if (sim) |x| spec.sumIn(x.acc, x.order, t) else host.sum(t);
            const v = num / den;
            y.* = .{ .v = v, .e = (err_num + gn * abs) / den + @abs(v) * err_den / den + 5 * U * @abs(v) };
        }
    }

    /// Every output into `want` / `mag`. Under `sim` the outputs stay as they are but the first eight columns of each
    /// GEMV, whose sums run in `sim`'s accumulator (a lower bound on the miss).
    fn emit(self: *const Ming, want: []f64, mag: ?[]f64, sim: Sim) void {
        const v = &self.v;
        switch (self.kind) {
            .dit_qkv => self.emitQkv(want, mag, sim),
            .dit_out, .dit_ff2, .lm_o, .lm_down => {
                const r = if (self.kind == .lm_o or self.kind == .lm_down) 1 else self.shape.rows;
                const k = self.v[2].len / self.outWidth();
                const cols = self.outWidth();
                const has_bias = self.kind == .dit_out or self.kind == .dit_ff2;
                for (0..r) |ri| for (0..lim(cols, sim)) |n| {
                    var y = self.sumOf(self.xn[ri * k ..][0..k], v[2][n * k ..][0..k], sim);
                    if (has_bias) y = addV(y, .exact(v[3][n]));
                    put(want, mag, ri * cols + n, addV(.exact(v[1][ri * cols + n]), rb(y)));
                };
            },
            .dit_ff1 => for (0..self.shape.rows) |ri| {
                const x = self.xn[ri * D ..][0..D];
                const inv = self.invOf(v[0][ri * D ..][0..D], sim);
                for (0..lim(FF, sim)) |n| put(want, mag, ri * FF + n, gelu(rb(normed(self.sumOf(x, v[2][n * D ..][0..D], sim), inv, v[3][n]))));
            },
            .embed => for (0..lim(D, sim)) |o| {
                const xe = addV(self.sumOf(v[0], v[1][o * 64 ..][0..64], sim), .exact(v[2][o]));
                for (0..2) |b| {
                    put(want, mag, (b * 3) * D + o, addV(.exact(v[3][b * D + o]), .exact(v[4][b * D + o])));
                    put(want, mag, (b * 3 + 1) * D + o, .exact(v[5][o]));
                    put(want, mag, (b * 3 + 2) * D + o, xe);
                }
            },
            .final => self.emitFinal(want, mag, sim),
            .lm_qkv => self.emitLmQkv(want, mag, sim),
            .lm_attn => self.emitLmAttn(want, mag, sim),
            .lm_up => {
                const inv = self.invOf(v[0], sim);
                for (0..lim(MLP, sim)) |c| {
                    const g = rb(mulV(self.sumOf(self.xn, v[2][c * WL ..][0..WL], sim), inv));
                    const u = rb(mulV(self.sumOf(self.xn, v[3][c * WL ..][0..WL], sim), inv));
                    put(want, mag, c, mulV(rb(mulV(g, rb(host.sigmoid(g)))), u));
                }
            },
        }
    }

    fn lim(n: usize, sim: Sim) usize {
        return if (sim != null) @min(8, n) else n;
    }

    fn outWidth(self: *const Ming) usize {
        return if (self.kind == .lm_o or self.kind == .lm_down) WL else D;
    }

    fn emitQkv(self: *const Ming, want: []f64, mag: ?[]f64, sim: Sim) void {
        const v = &self.v;
        const r = self.shape.rows;
        const seq = r / self.shape.n;
        var inv: [6]Val = undefined;
        for (inv[0..r], 0..) |*o, ri| o.* = self.invOf(v[0][ri * D ..][0..D], sim);
        for (0..3) |wh| {
            const w = v[2 + wh];
            const b = v[5 + wh];
            var c: usize = 0;
            while (c < lim(D, sim)) : (c += 2) for (0..r) |ri| {
                const x = self.xn[ri * D ..][0..D];
                const e = normed(self.sumOf(x, w[c * D ..][0..D], sim), inv[ri], b[c]);
                const od = normed(self.sumOf(x, w[(c + 1) * D ..][0..D], sim), inv[ri], b[c + 1]);
                const at = (wh * r + ri) * D + c;
                if (wh == 2) {
                    put(want, mag, at, e);
                    put(want, mag, at + 1, od);
                    continue;
                }
                const tab = (ri % seq) * 32 + (c % 64) / 2;
                const t = rotate(rb(e), rb(od), .exact(v[8][tab]), .exact(v[9][tab]));
                put(want, mag, at, t[0]);
                put(want, mag, at + 1, t[1]);
            };
        }
        if (sim != null) return;
        const bytes = self.rig.out_bytes.items;
        var q: [64]f64 = undefined;
        var o: [64]Val = undefined;
        for (0..16) |h| for (0..r) |ri| {
            const b = ri / seq;
            for (&q, 0..) |*e, d| e.* = 0.125 * spec.read(.bf16, bytes, ri * D + h * 64 + d);
            for (0..seq) |j| for (0..64) |d| {
                self.keys[j * 64 + d] = spec.read(.bf16, bytes, (r + b * seq + j) * D + h * 64 + d);
                self.vals[j * 64 + d] = spec.read(.bf16, bytes, (2 * r + b * seq + j) * D + h * 64 + d);
            };
            self.attend(&q, seq, &o, null);
            for (o, 0..) |y, d| put(want, mag, 3 * r * D + ri * D + h * 64 + d, y);
        };
    }

    fn emitFinal(self: *const Ming, want: []f64, mag: ?[]f64, sim: Sim) void {
        const v = &self.v;
        const seq = self.shape.rows / self.shape.n;
        const r0 = (seq - 1) * D;
        const r1 = (2 * seq - 1) * D;
        const inv0 = self.invOf(v[0][r0..][0..D], sim);
        const inv1 = self.invOf(v[0][r1..][0..D], sim);
        for (0..lim(64, sim)) |o| {
            const w = v[2][o * D ..][0..D];
            const pc = rb(normed(self.sumOf(self.xn[r0..][0..D], w, sim), inv0, v[3][o]));
            const pu = rb(normed(self.sumOf(self.xn[r1..][0..D], w, sim), inv1, v[3][o]));
            const f0 = rb(addV(pc, rb(mulV(rb(subV(pc, pu)), cst(CFG)))));
            var y = addV(.exact(v[4][o]), rb(mulV(.exact(v[5][STEP]), f0)));
            if (self.shift != 0) y = addV(rb(y), rb(mulV(rb(mulV(cst(self.shift), .exact(v[6][STEP]))), .exact(v[7][STEP * 64 + o]))));
            put(want, mag, o, y);
        }
    }

    fn emitLmQkv(self: *const Ming, want: []f64, mag: ?[]f64, sim: Sim) void {
        const v = &self.v;
        const inv = self.invOf(v[0], sim);
        const pos: f64 = @floatFromInt(self.shape.n);
        const log2b: f64 = LOG2_THETA;
        for ([_]struct { usize, usize }{ .{ 448, 0 }, .{ 64, 896 } }, 0..) |part, wh| {
            for (0..lim(part[0], sim)) |p| {
                const c1 = (p / 32) * 64 + p % 32;
                const c2 = c1 + 32;
                const w = v[2 + wh];
                const b = v[5 + wh];
                const x1 = rb(normed(self.sumOf(self.xn, w[c1 * WL ..][0..WL], sim), inv, b[c1]));
                const x2 = rb(normed(self.sumOf(self.xn, w[c2 * WL ..][0..WL], sim), inv, b[c2]));
                // exp2 is fast; fast cos and sin hold a few f32 ulps of their value's scale.
                const arg = -(@as(f64, @floatFromInt(c1 % 64)) / 32) * log2b;
                const th = pos * @exp2(arg);
                const eth = @abs(th) * (host.fastExp(arg) + 2 * U);
                const co = @cos(th);
                const si = @sin(th);
                const t = rotate(x1, x2, .{ .v = co, .e = @abs(si) * eth + 16 * U }, .{ .v = si, .e = @abs(co) * eth + 16 * U });
                put(want, mag, part[1] + c1, t[0]);
                put(want, mag, part[1] + c2, t[1]);
            }
        }
        for (0..lim(128, sim)) |c| put(want, mag, 1024 + c, normed(self.sumOf(self.xn, v[4][c * WL ..][0..WL], sim), inv, v[7][c]));
    }

    fn emitLmAttn(self: *const Ming, want: []f64, mag: ?[]f64, sim: Sim) void {
        const qkv = self.v[0];
        const lc = self.shape.rows;
        const n = lc + 1;
        var q: [64]f64 = undefined;
        var o: [64]Val = undefined;
        for (0..14) |h| {
            const kh = h / 7;
            for (&q, 0..) |*e, d| e.* = 0.125 * qkv[h * 64 + d];
            @memcpy(self.keys[0 .. lc * 64], self.v[1][kh * lc * 64 ..][0 .. lc * 64]);
            @memcpy(self.vals[0 .. lc * 64], self.v[2][kh * lc * 64 ..][0 .. lc * 64]);
            @memcpy(self.keys[lc * 64 ..][0..64], qkv[896 + kh * 64 ..][0..64]);
            @memcpy(self.vals[lc * 64 ..][0..64], qkv[1024 + kh * 64 ..][0..64]);
            self.attend(&q, n, &o, sim);
            for (o, 0..) |y, d| put(want, mag, h * 64 + d, y);
        }
        if (sim != null) return;
        for (0..2) |kh| for (0..n) |j| for (0..64) |d| {
            const i = (kh * n + j) * 64 + d;
            const k = if (j < lc) self.v[1][(kh * lc + j) * 64 + d] else qkv[896 + kh * 64 + d];
            const u = if (j < lc) self.v[2][(kh * lc + j) * 64 + d] else qkv[1024 + kh * 64 + d];
            put(want, mag, WL + i, .exact(k));
            put(want, mag, WL + 2 * n * 64 + i, .exact(u));
        };
    }

    fn outputs(self: *Ming) ![]f64 {
        if (self.ref.len == 0) {
            const n = self.count();
            self.ref = try self.arena.allocator().alloc(f64, 2 * n);
            self.emit(self.ref[0..n], self.ref[n..], null);
        }
        return self.ref;
    }

    pub fn reference(self: *Ming, want: []f64, mag: []f64) !void {
        const r = try self.outputs();
        @memcpy(want, r[0..want.len]);
        @memcpy(mag, r[want.len..]);
    }

    pub fn simulate(self: *Ming, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const r = try self.outputs();
        @memcpy(out, r[0..out.len]);
        self.emit(out, null, .{ .acc = acc, .order = order });
    }
};

fn mingSpec(comptime name: []const u8, served: []const Shape) spec.Spec {
    return .{ .name = name, .accum = .f32, .depth = host.depthOne, .out = .bf16, .served = served, .sums = .{ .exact = false } };
}

const dit_served: []const Shape = &.{ .{ .rows = 6, .n = 2, .k = D }, .{ .rows = 2, .n = 1, .k = D } };

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    var qkv = mingSpec("ming_dit_qkv", dit_served);
    // CNT is the election counter every threadgroup bumps: the one input a run changes by design.
    qkv.in_place = true;
    try set.add(Ming, .init(a, &qkv_launch, .dit_qkv, 0), qkv, null);
    try set.add(Ming, .init(a, &attn_launch, .dit_out, 0), mingSpec("ming_dit_attn_out", dit_served), null);
    try set.add(Ming, .init(a, &ff1_launch, .dit_ff1, 0), mingSpec("ming_dit_ff1", dit_served), null);
    try set.add(Ming, .init(a, &ff2_launch, .dit_ff2, 0), mingSpec("ming_dit_ff2", dit_served), null);
    try set.add(Ming, .init(a, &embed_launch, .embed, 0), mingSpec("ming_cfm_embed", &.{.{ .rows = 6, .n = 2, .k = 64 }}), null);
    const cfm: []const Shape = &.{.{ .rows = 6, .n = 2, .k = D }};
    try set.add(Ming, .init(a, &final_launch, .final, 0.3), mingSpec("ming_cfm_final", cfm), null);
    try set.add(Ming, .init(a, &final_launch, .final, 0), mingSpec("ming_cfm_final (no SDE term)", cfm), null);
    try set.add(Ming, .init(a, &lm_qkv_launch, .lm_qkv, 0), mingSpec("ming_lm_qkv", &.{ .{ .rows = 1, .n = 1, .k = WL }, .{ .rows = 1, .n = 511, .k = WL } }), null);
    var attn = mingSpec("ming_lm_attn", &.{ .{ .rows = 6, .n = 14, .k = 64 }, .{ .rows = 511, .n = 14, .k = 64 } });
    // N = Lc + 1 keys over the 256 threads: the tile turns at N 256.
    attn.tiles = .{ .{ .size = 255 }, .{}, .{} };
    try set.add(Ming, .init(a, &lm_attn_launch, .lm_attn, 0), attn, null);
    const lm: []const Shape = &.{.{ .rows = 1, .n = 1, .k = WL }};
    try set.add(Ming, .init(a, &lm_o_launch, .lm_o, 0), mingSpec("ming_lm_o", lm), null);
    try set.add(Ming, .init(a, &lm_up_launch, .lm_up, 0), mingSpec("ming_lm_up", lm), null);
    try set.add(Ming, .init(a, &lm_down_launch, .lm_down, 0), mingSpec("ming_lm_down", lm), null);
}

// ── tests ──

const testing = std.testing;

fn expectLaunch(t: *Ming, theirs: []const mlx.mlx_array) !void {
    try t.run();
    const ours = try testing.allocator.dupe(u8, t.rig.outputBytes());
    defer testing.allocator.free(ours);
    try testing.expectEqualSlices(u8, try t.rig.bytesOf(theirs), ours);
}

test "kernel contract ming_tts: the DiT launches are the shipped block's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const shape = Shape{ .rows = 6, .n = 2, .k = D };
    var ta = Ming.init(a, &qkv_launch, .dit_qkv, 0);
    defer ta.deinit();
    var tb = Ming.init(a, &attn_launch, .dit_out, 0);
    defer tb.deinit();
    var tc = Ming.init(a, &ff1_launch, .dit_ff1, 0);
    defer tc.deinit();
    var td = Ming.init(a, &ff2_launch, .dit_ff2, 0);
    defer td.deinit();
    for ([_]*Ming{ &ta, &tb, &tc, &td }, 1..) |t, seed| try testing.expect(try t.prepare(shape, .random, seed));
    const s = ta.rig.stream();
    var cnt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cnt);
    try mlx.check(mlx.mlx_zeros(&cnt, &[_]c_int{D}, 1, .int32, s));
    const x = ta.args;
    const b = fused.Block{ .norm1 = x[1], .wq = x[2], .wk = x[3], .wv = x[4], .bq = x[5], .bk = x[6], .bv = x[7], .norm2 = tc.args[1], .wo = tb.args[2], .bo = tb.args[3], .w1 = tc.args[2], .b1 = tc.args[3], .w2 = td.args[2], .b2 = td.args[3], .cnt = cnt };
    const o = try fused.block(&b, x[0], x[8], x[9], s);
    defer for ([_]mlx.mlx_array{ o.qkv, o.att, o.x1, o.hh, o.x2 }) |y| {
        _ = mlx.mlx_array_free(y);
    };
    try expectLaunch(&ta, &.{ o.qkv, o.att });
    tb.args[0] = o.att;
    tb.args[1] = x[0];
    try expectLaunch(&tb, &.{o.x1});
    tc.args[0] = o.x1;
    try expectLaunch(&tc, &.{o.hh});
    td.args[0] = o.hh;
    td.args[1] = o.x1;
    try expectLaunch(&td, &.{o.x2});
}

test "kernel contract ming_tts: the CFM launches are the shipped wrappers', byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    var te = Ming.init(a, &embed_launch, .embed, 0);
    defer te.deinit();
    try testing.expect(try te.prepare(.{ .rows = 6, .n = 2, .k = 64 }, .random, 11));
    const s = te.rig.stream();
    const e = te.args;
    const h = try fused.embed(e[0], e[1], e[2], e[3], e[4], e[5], s);
    defer _ = mlx.mlx_array_free(h);
    try expectLaunch(&te, &.{h});
    var tf = Ming.init(a, &final_launch, .final, 0.3);
    defer tf.deinit();
    try testing.expect(try tf.prepare(.{ .rows = 6, .n = 2, .k = D }, .random, 12));
    const f = tf.args;
    const y = try fused.final(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], STEP, CFG, 0.3, s);
    defer _ = mlx.mlx_array_free(y);
    try expectLaunch(&tf, &.{y});
}

test "kernel contract ming_tts: the LM launches are the shipped layer's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    var la = Ming.init(a, &lm_qkv_launch, .lm_qkv, 0);
    defer la.deinit();
    var lb = Ming.init(a, &lm_attn_launch, .lm_attn, 0);
    defer lb.deinit();
    var lo = Ming.init(a, &lm_o_launch, .lm_o, 0);
    defer lo.deinit();
    var lu = Ming.init(a, &lm_up_launch, .lm_up, 0);
    defer lu.deinit();
    var ld = Ming.init(a, &lm_down_launch, .lm_down, 0);
    defer ld.deinit();
    const lc = 6;
    try testing.expect(try la.prepare(.{ .rows = 1, .n = lc, .k = WL }, .random, 1));
    try testing.expect(try lb.prepare(.{ .rows = lc, .n = 14, .k = 64 }, .random, 2));
    for ([_]*Ming{ &lo, &lu, &ld }, 3..) |t, seed| try testing.expect(try t.prepare(.{ .rows = 1, .n = 1, .k = WL }, .random, seed));
    const s = la.rig.stream();
    const x = la.args;
    const l = fused.LmLayer{ .ln1 = x[1], .wq = x[2], .wk = x[3], .wv = x[4], .bq = x[5], .bk = x[6], .bv = x[7], .ln2 = lu.args[1], .wo = lo.args[2], .wg = lu.args[2], .wu = lu.args[3], .wd = ld.args[2] };
    const o = try fused.lmLayer(&l, x[0], x[8], lb.args[1], lb.args[2], LOG2_THETA, s);
    defer for ([_]mlx.mlx_array{ o.qkv, o.o, o.x1, o.hh, o.x2, o.k, o.v }) |y| {
        _ = mlx.mlx_array_free(y);
    };
    try expectLaunch(&la, &.{o.qkv});
    lb.args[0] = o.qkv;
    try expectLaunch(&lb, &.{ o.o, o.k, o.v });
    lo.args[0] = o.o;
    lo.args[1] = x[0];
    try expectLaunch(&lo, &.{o.x1});
    lu.args[0] = o.x1;
    try expectLaunch(&lu, &.{o.hh});
    ld.args[0] = o.hh;
    ld.args[1] = o.x1;
    try expectLaunch(&ld, &.{o.x2});
}
