//! The kda area's kernel contract, at GLM-5.3-Flash's KDA layers (64 heads of 128, bf16 activations, f32 state,
//! lower bound -5): oMLX's per-core prefill recurrence (the per-channel bounded-sigmoid gate evaluated while staging)
//! and its decode step (short conv + SiLU, the low-rank gates' quantized second stage, q/k l2 norms, the gate, the
//! delta rule, the gated RMSNorm; 4- and 8-bit gates up to 8 rows, 5-bit at one row, with or without the per-row
//! state capture). Both meet the float64 bound with the gdn area's forward-error allowance (`host.Val`, `gdn.step`).
//! Each launch's thread count is probed by its wrapper (`adapts`); the per-core plan follows the GPU's core count
//! (80 cores: 80 threadgroups of 104 rows, 64 cores: 64 of 128).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../testing/kernel_contract/root.zig");
const gdn = @import("../gdn/contract.zig");
const kr = @import("recurrence.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Val = host.Val;
const Shape = spec.Shape;
const U = host.U;

const H = 64;
const D = 128;
const QKV = H * D;
const PLANE = H * D * D;
const LB = -5.0;

/// What a registration reads: the bf16 outputs (y, the conv state) or the f32 ones (the states).
const Part = enum { act, state };

/// The gate exp(lb * sigmoid(exp(A_log) * (a + dt_bias))) for one channel: f32 adds and products, MLX's sigmoid
/// form in f32 (fast or precise exp), precise exps.
fn gateOf(a: Val, dtb: f64, alog: f64) Val {
    const ea = @exp(alog);
    const s1v = a.v + dtb;
    const s1 = Val{ .v = s1v, .e = a.e + U * @abs(s1v) };
    var s2 = host.mul(.{ .v = ea, .e = 0x1p-21 * ea }, s1);
    s2.e += U * @abs(s2.v);
    const s3 = host.mlxSig(.float32, s2);
    const p = Val{ .v = LB * s3.v, .e = @abs(LB) * s3.e + U * @abs(LB * s3.v) };
    const g = @exp(p.v);
    if (!std.math.isFinite(p.v)) return .{ .v = g };
    return .{ .v = g, .e = g * (std.math.expm1(p.e) + 0x1p-21) };
}

// ── the per-core prefill recurrence ──

const Plan = struct { ntg: c_int, mr: c_int };

fn coreLaunch(comptime plan: Plan) metal.Launch {
    return .{
        .name = "msv_kda_percore",
        .inputs = &.{ "q", "k", "v", "a", "beta", "A_log", "dt_bias", "lower_bound", "state_in", "T" },
        .outputs = &.{ "y", "state_out" },
        .source = kernelSource("kernels/kda/kda_percore.metal"),
        // k_s, q_s [2][12][136] and v_s [12][MR] bf16, g_s [2][12][132] and b_s [2][12] f32.
        .tg_bytes = 2 * 2 * 12 * 136 * 2 + 12 * @as(u32, @intCast(plan.mr)) * 2 + 2 * 12 * 132 * 4 + 2 * 12 * 4,
        .m1 = 1024,
        .m2 = 1024,
        .adapts = true,
    };
}

/// A shape's rows are the tokens T.
const Core = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    plan: Plan,
    part: Part,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    a_in: []f64 = &.{},
    beta: []f64 = &.{},
    alog: []f64 = &.{},
    dtb: []f64 = &.{},
    s0: []f64 = &.{},
    args: [10]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, plan: Plan, part: Part) Core {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .plan = plan, .part = part, .arena = .init(a) };
    }

    pub fn deinit(self: *Core) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Core, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1) return null;
        const p = self.plan;
        var g = metal.Geometry{ .grid = .{ p.mr * 8 * p.ntg, 1, 1 }, .tg = .{ p.mr * 8, 1, 1 } };
        g.out(&.{ 1, @intCast(shape.rows), H, D }, .bfloat16);
        g.out(&.{ 1, H, D, D }, .float32);
        g.unread = if (self.part == .act) 0b10 else 0b1;
        g.dtype("InT", .bfloat16);
        inline for (.{ .{ "Dk", D }, .{ "Dv", D }, .{ "H", H } }) |kv| g.int(kv[0], kv[1]);
        g.int("NTG", p.ntg);
        g.int("MR", p.mr);
        return g;
    }

    pub fn prepare(self: *Core, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const T = shape.rows;
        // q and k at about unit norm, as the prework leaves them.
        self.q = try host.uniform(a, rng, T * QKV, -0.15, 0.15, .bfloat16);
        self.k = try host.uniform(a, rng, T * QKV, -0.15, 0.15, .bfloat16);
        self.v = try host.uniform(a, rng, T * QKV, -2, 2, .bfloat16);
        self.a_in = try host.uniform(a, rng, T * QKV, -2, 2, .bfloat16);
        self.beta = try host.uniform(a, rng, T * H, 0.05, 0.95, .bfloat16);
        self.alog = try a.alloc(f64, H);
        for (self.alog) |*x| x.* = host.rt(.float32, @log(1 + 15 * rng.float(f64)));
        self.dtb = try host.uniform(a, rng, QKV, -0.2, 0.2, .float32);
        self.s0 = try host.uniform(a, rng, PLANE, -0.3, 0.3, .float32);
        switch (fill) {
            .random => {},
            .nonfinite => {
                const vals = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
                const sites = [_][2]usize{ .{ 0, 5 }, .{ 1, 7 }, .{ 2, 9 }, .{ 3, 11 }, .{ 3, 12 } };
                for (sites, vals) |s, x| self.v[s[0] * D + s[1]] = x;
                self.a_in[5 * D + 3] = std.math.nan(f64);
                self.s0[(9 * D + 3) * D + 17] = std.math.nan(f64);
            },
            .stress => {
                // Gate 4: exact q, k; a = -100 makes every gate 1; unit betas; a positive state of 8-bit multiples of
                // 256 drives k.S past fp16's range.
                @memset(self.q, 1.0 / 16.0);
                @memset(self.k, 1.0 / 8.0);
                for (self.v) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                @memset(self.a_in, -100);
                @memset(self.beta, 1);
                @memset(self.alog, 0);
                @memset(self.dtb, 0);
                for (self.s0) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, 1, 255) * 256);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const Ti: c_int = @intCast(T);
        self.args = .{
            try rig.input(.bfloat16, &.{ 1, Ti, H, D }, self.q),
            try rig.input(.bfloat16, &.{ 1, Ti, H, D }, self.k),
            try rig.input(.bfloat16, &.{ 1, Ti, H, D }, self.v),
            try rig.input(.bfloat16, &.{ 1, Ti, QKV }, self.a_in),
            try rig.input(.bfloat16, &.{ 1, Ti, H }, self.beta),
            try rig.input(.float32, &.{H}, self.alog),
            try rig.input(.float32, &.{QKV}, self.dtb),
            try rig.input(.float32, &.{1}, &.{LB}),
            try rig.input(.float32, &.{ 1, H, D, D }, self.s0),
            try host.intScalar(rig, Ti),
        };
        return true;
    }

    pub fn run(self: *Core) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Core) usize {
        return if (self.part == .act) self.shape.rows * QKV else PLANE;
    }

    fn emit(self: *const Core, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const a = self.rig.a;
        const T = self.shape.rows;
        const g = try a.alloc(Val, T * QKV);
        defer a.free(g);
        for (0..T) |t| for (0..QKV) |c| {
            g[t * QKV + c] = gateOf(.exact(self.a_in[t * QKV + c]), self.dtb[c], self.alog[c / D]);
        };
        const st = try a.alloc(Val, D);
        defer a.free(st);
        const kv = try a.alloc(Val, D);
        defer a.free(kv);
        const qv = try a.alloc(Val, D);
        defer a.free(qv);
        const terms = try a.alloc(f64, D);
        defer a.free(terms);
        for (0..H) |h| for (0..D) |r| {
            const row = (h * D + r) * D;
            for (st, self.s0[row..][0..D]) |*o, x| o.* = .exact(x);
            for (0..T) |t| {
                const at = (t * H + h) * D;
                for (kv, qv, self.k[at..][0..D], self.q[at..][0..D]) |*ko, *qo, x, y| {
                    ko.* = .exact(x);
                    qo.* = .exact(y);
                }
                const y = gdn.step(st, kv, qv, .exact(self.v[at + r]), g[t * QKV + h * D ..][0..D], .exact(self.beta[t * H + h]), 9, acc, order, terms);
                if (self.part == .act) {
                    const o = host.round(.bfloat16, y);
                    want[at + r] = o.v;
                    if (mag) |m| m[at + r] = host.allowance(o.e);
                }
            }
            if (self.part == .state) for (st, 0..) |s, i| {
                want[row + i] = s.v;
                if (mag) |m| m[row + i] = host.allowance(s.e);
            };
        };
    }

    pub fn reference(self: *const Core, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Core, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, acc, order);
    }
};

// ── the decode step ──

const Bits = enum(u32) { b4 = 4, b5 = 5, b8 = 8 };
const GS = 64;
const OFF_GA = 3 * QKV;
const OFF_FA = OFF_GA + D;
const OFF_B = OFF_FA + D;
const PROJ_W = OFF_B + H;
const NORM_EPS = 1e-6;

fn stepLaunch(comptime bits: Bits) metal.Launch {
    const body = kernelSource("kernels/kda/kda_decode.metal");
    return .{
        .name = if (bits == .b5) "msv_kda_decode_q5" else "msv_kda_decode",
        .inputs = &.{ "proj", "conv_w", "a_log", "dt_bias", "norm_w", "consts", "conv_state", "state_in", "fb_w", "fb_s", "fb_b", "gb_w", "gb_s", "gb_b" },
        .outputs = &.{ "y", "conv_state_out", "state_out", "state_seq" },
        .source = (if (bits == .b5) "#define GATE5 1\n" else "#define GATE5 0\n") ++ body,
        .header = kernelSource("plugins/glm5_next/kernels/qmv_header.metal"),
        // qs, ks, vs, as_, gates, ys [8][128] bf16, gs [8][128] f32, betas [8] bf16.
        .tg_bytes = 6 * 8 * D * 2 + 8 * D * 4 + 8 * 2,
        // M1/M2 build the 4-bit launch for 96..448 threads past one row, the 8-bit one for 640..832 (captured or not
        // alike); the wrapper probes each width and declines there. One 4-bit row: inconclusive.
        .m1 = if (bits == .b8) 640 else 1024,
        .m2 = if (bits == .b8) 640 else 1024,
        .caps = switch (bits) {
            .b4 => &.{ .{ .rows = 2, .m1 = 96, .m2 = 96 }, .{ .rows = 3, .m1 = 128, .m2 = 128 }, .{ .rows = 4, .m1 = 224, .m2 = 224 }, .{ .rows = 5, .m1 = 448, .m2 = 448 }, .{ .rows = 6, .m1 = 448, .m2 = 448 }, .{ .rows = 7, .m1 = 448, .m2 = 448 }, .{ .rows = 8, .m1 = 448, .m2 = 448 } },
            .b8 => &.{.{ .rows = 1, .m1 = 832, .m2 = 832 }},
            .b5 => &.{},
        },
        .adapts = true,
    };
}

/// A shape's rows are the tokens (the 5-bit gate launch runs one whatever the rows).
const Step = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    bits: Bits,
    cap: bool,
    part: Part,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    proj: []f64 = &.{},
    conv_w: []f64 = &.{},
    alog: []f64 = &.{},
    dtb: []f64 = &.{},
    norm_w: []f64 = &.{},
    cs: []f64 = &.{},
    s0: []f64 = &.{},
    fb: host.Quant = undefined,
    gb: host.Quant = undefined,
    args: [14]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, bits: Bits, cap: bool, part: Part) Step {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .bits = bits, .cap = cap, .part = part, .arena = .init(a) };
    }

    pub fn deinit(self: *Step) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn toks(self: *const Step, shape: Shape) usize {
        return if (self.bits == .b5) 1 else shape.rows;
    }

    pub fn geometry(self: *const Step, shape: Shape) ?metal.Geometry {
        const T = self.toks(shape);
        if (T < 1 or T > 8) return null;
        const t: c_int = @intCast(T);
        var g = metal.Geometry{ .grid = .{ 1024 * H, 1, 1 }, .tg = .{ 1024, 1, 1 } };
        g.out(&.{ 1, t, QKV }, .bfloat16);
        g.out(&.{ 1, 3, 3 * QKV }, .bfloat16);
        g.out(&.{ 1, H, D, D }, .float32);
        if (self.cap) g.out(&.{ t, H, D, D }, .float32) else g.out(&.{ 1, 1, 1, 1 }, .float32);
        g.unread = if (self.part == .act) 0b1100 else if (self.cap) 0b11 else 0b1011;
        g.dtype("T", .bfloat16);
        g.int("TOK", t);
        inline for (.{ .{ "DK", D }, .{ "QKV", QKV }, .{ "PROJ_W", PROJ_W }, .{ "OFF_FA", OFF_FA }, .{ "OFF_GA", OFF_GA }, .{ "OFF_B", OFF_B } }) |kv| g.int(kv[0], kv[1]);
        g.int("BITS", @backingInt(self.bits));
        g.int("GS", GS);
        g.int("CAP", @intFromBool(self.cap));
        return g;
    }

    fn consts() [5]f64 {
        return .{ host.rt(.float32, 1 / @sqrt(@as(f64, 128))), host.rt(.float32, 1e-6), host.rt(.float32, NORM_EPS), LB, 1.0 / 128.0 };
    }

    pub fn prepare(self: *Step, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const T = self.toks(shape);
        const bits = @backingInt(self.bits);
        self.proj = try host.uniform(a, rng, T * PROJ_W, -2, 2, .bfloat16);
        self.conv_w = try host.uniform(a, rng, 3 * QKV * 4, -0.6, 0.6, .bfloat16);
        self.alog = try a.alloc(f64, H);
        for (self.alog) |*x| x.* = host.rt(.float32, @log(1 + 15 * rng.float(f64)));
        self.dtb = try host.uniform(a, rng, QKV, -0.2, 0.2, .float32);
        self.norm_w = try host.uniform(a, rng, D, 0.5, 1.5, .bfloat16);
        self.cs = try host.uniform(a, rng, 3 * 3 * QKV, -2, 2, .bfloat16);
        self.s0 = try host.uniform(a, rng, PLANE, -0.3, 0.3, .float32);
        self.fb = try host.quantize(a, rng, QKV, D, bits, GS, .bfloat16);
        self.gb = try host.quantize(a, rng, QKV, D, bits, GS, .bfloat16);
        switch (fill) {
            .random => {},
            .nonfinite => {
                const vals = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
                const sites = [_][2]usize{ .{ 0, 5 }, .{ 1, 7 }, .{ 2, 9 }, .{ 3, 11 }, .{ 3, 12 } };
                for (sites, vals) |s, x| self.proj[2 * QKV + s[0] * D + s[1]] = x;
                self.proj[OFF_B + 5] = std.math.nan(f64);
                self.s0[(9 * D + 3) * D + 17] = std.math.nan(f64);
            },
            .stress => {
                // Gate 4: conv weights [0, 0, 0, 1] and inputs 16 make SiLU, q and k exact; zero gate inputs leave
                // a 0 pre-gate (dt_bias -100: every gate 1) and an output gate of exactly 1/2; b = 16 a unit beta; a
                // positive state of 8-bit multiples of 256 drives k.S past fp16's range.
                for (0..3 * QKV) |c| self.conv_w[c * 4 ..][0..4].* = .{ 0, 0, 0, 1 };
                for (0..T) |t| {
                    const row = self.proj[t * PROJ_W ..][0..PROJ_W];
                    @memset(row[0 .. 3 * QKV], 16);
                    @memset(row[OFF_GA..OFF_B], 0);
                    @memset(row[OFF_B..], 16);
                }
                @memset(self.dtb, -100);
                @memset(self.alog, 0);
                @memset(self.norm_w, 1);
                for (self.s0) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, 1, 255) * 256);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const t: c_int = @intCast(T);
        const wcols: c_int = @intCast(D * bits / 32);
        const groups: c_int = D / GS;
        const c5 = consts();
        self.args = .{
            try rig.input(.bfloat16, &.{ 1, t, PROJ_W }, self.proj),
            try rig.input(.bfloat16, &.{ 3 * QKV, 4 }, self.conv_w),
            try rig.input(.float32, &.{H}, self.alog),
            try rig.input(.float32, &.{QKV}, self.dtb),
            try rig.input(.bfloat16, &.{D}, self.norm_w),
            try rig.input(.float32, &.{5}, &c5),
            try rig.input(.bfloat16, &.{ 1, 3, 3 * QKV }, self.cs),
            try rig.input(.float32, &.{ 1, H, D, D }, self.s0),
            try rig.input(.uint32, &.{ QKV, wcols }, self.fb.words),
            try rig.input(.bfloat16, &.{ QKV, groups }, self.fb.scales),
            try rig.input(.bfloat16, &.{ QKV, groups }, self.fb.biases),
            try rig.input(.uint32, &.{ QKV, wcols }, self.gb.words),
            try rig.input(.bfloat16, &.{ QKV, groups }, self.gb.scales),
            try rig.input(.bfloat16, &.{ QKV, groups }, self.gb.biases),
        };
        return true;
    }

    pub fn run(self: *Step) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Step) usize {
        const T = self.toks(self.shape);
        return if (self.part == .act) T * QKV + 3 * 3 * QKV else PLANE * (if (self.cap) T + 1 else 1);
    }

    /// Row `row` of a quantized gate's second stage against x (128 wide): MLX's qmv (codes against x pre-scaled by
    /// powers of two, then scale * dot + bias * sum per group, the lanes' partials summed), up to 40 adds deep. Its
    /// load_vector adds 4 (4-bit) or 8 (5-bit) x values in bf16 before the f32 sum: the allowance takes the bias times
    /// what those roundings move the group's sum (they may also stay in f32).
    fn qmv(q: host.Quant, row: usize, x: []const f64, bits: u32) Val {
        var v: f64 = 0;
        var m: f64 = 0;
        var e: f64 = 0;
        const top: f64 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
        const chunk: usize = switch (bits) {
            4 => 4,
            5 => 8,
            else => 1,
        };
        for (0..D / GS) |g| {
            const s = q.scales[row * (D / GS) + g];
            const b = q.biases[row * (D / GS) + g];
            var sx: f64 = 0;
            var moved: f64 = 0;
            for (g * GS..(g + 1) * GS) |c| {
                v += x[c] * q.w[row * D + c];
                sx += @abs(x[c]);
            }
            var c0 = g * GS;
            while (c0 < (g + 1) * GS) : (c0 += chunk) {
                var exact: f64 = 0;
                var bf: f64 = x[c0];
                for (x[c0..][0..chunk], 0..) |xv, j| {
                    exact += xv;
                    if (j > 0) bf = host.rt(.bfloat16, bf + xv);
                }
                moved += bf - exact;
            }
            m += (@abs(s) * top + @abs(b)) * sx;
            e += @abs(b * moved);
        }
        return .{ .v = v, .e = spec.gamma(40, U) * m + e };
    }

    fn emit(self: *const Step, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        var arena = std.heap.ArenaAllocator.init(self.rig.a);
        defer arena.deinit();
        const a = arena.allocator();
        const T = self.toks(self.shape);
        const bits = @backingInt(self.bits);
        const c5 = consts();
        const bf: mlx.mlx_dtype = .bfloat16;
        // Rows of the conv window: the conv state's three, then the tokens'.
        const win = struct {
            fn at(s: *const Step, r: usize, col: usize) f64 {
                return if (r < 3) s.cs[r * 3 * QKV + col] else s.proj[(r - 3) * PROJ_W + col];
            }
        };
        const act = try a.alloc(Val, T * 3 * QKV);
        for (0..T) |t| for (0..3 * QKV) |col| {
            var sum: f64 = 0;
            var m: f64 = 0;
            for (0..4) |j| {
                const p = win.at(self, t + j, col) * self.conv_w[col * 4 + j];
                sum += p;
                m += @abs(p);
            }
            const conv = host.round(bf, .{ .v = sum, .e = spec.gamma(3, U) * m });
            act[t * 3 * QKV + col] = host.round(bf, host.mul(conv, host.mlxSig(bf, conv)));
        };
        const ys = try a.alloc(Val, T * QKV);
        const states = try a.alloc(Val, (T + 1) * PLANE);
        const terms = try a.alloc(f64, D);
        const qn = try a.alloc(Val, T * D);
        const kn = try a.alloc(Val, T * D);
        const g = try a.alloc(Val, T * D);
        const beta = try a.alloc(Val, T);
        const gates = try a.alloc(Val, T * QKV);
        const st = try a.alloc(Val, D);
        for (0..H) |h| {
            for (0..T) |t| {
                // l2 norms: q scaled by 1/sqrt(128), each product an f32 rounding before the bf16 one.
                for ([_]usize{ 0, 1 }) |part| {
                    const xs = act[t * 3 * QKV + part * QKV + h * D ..][0..D];
                    const inv = gdn.rmsInv(xs, 1, c5[1], null, order, terms);
                    for (xs, 0..) |x, i| {
                        var xn = host.mul(x, inv);
                        xn.e += U * @abs(xn.v);
                        if (part == 0) {
                            const v = xn.v * c5[0];
                            qn[t * D + i] = host.round(bf, .{ .v = v, .e = xn.e * c5[0] + U * @abs(v) });
                        } else kn[t * D + i] = host.round(bf, xn);
                    }
                }
                const xf = self.proj[t * PROJ_W + OFF_FA ..][0..D];
                const xg = self.proj[t * PROJ_W + OFF_GA ..][0..D];
                for (0..D) |i| {
                    const as_ = host.round(bf, qmv(self.fb, h * D + i, xf, bits));
                    gates[t * QKV + h * D + i] = host.round(bf, qmv(self.gb, h * D + i, xg, bits));
                    g[t * D + i] = gateOf(as_, self.dtb[h * D + i], self.alog[h]);
                }
                beta[t] = host.mlxSig(bf, .exact(self.proj[t * PROJ_W + OFF_B + h]));
            }
            for (0..D) |r| {
                const row = (h * D + r) * D;
                for (st, self.s0[row..][0..D]) |*o, x| o.* = .exact(x);
                for (0..T) |t| {
                    const v = act[t * 3 * QKV + 2 * QKV + h * D + r];
                    const y = gdn.step(st, kn[t * D ..][0..D], qn[t * D ..][0..D], v, g[t * D ..][0..D], beta[t], 9, acc, order, terms);
                    ys[t * QKV + h * D + r] = host.round(bf, y);
                    @memcpy(states[(1 + t) * PLANE + row ..][0..D], st);
                }
                @memcpy(states[row..][0..D], st);
            }
        }
        var at: usize = 0;
        const put = struct {
            fn one(w: []f64, mg: ?[]f64, i: *usize, x: Val) void {
                w[i.*] = x.v;
                if (mg) |m| m[i.*] = host.allowance(x.e);
                i.* += 1;
            }
        };
        if (self.part == .act) {
            // The gated RMSNorm: rsqrt(mean(y^2) + eps), x * r, * norm_w, * sigmoid(gate) (precise exp), each in f32.
            for (0..T) |t| for (0..H) |h| {
                const xs = ys[t * QKV + h * D ..][0..D];
                const inv = gdn.rmsInv(xs, c5[4], c5[2], acc, order, terms);
                for (xs, 0..) |x, c| {
                    var xn = host.mul(x, inv);
                    xn.e += U * @abs(xn.v);
                    const w = self.norm_w[c];
                    const wx = Val{ .v = w * xn.v, .e = @abs(w) * xn.e + U * @abs(w * xn.v) };
                    var o = host.mul(wx, host.mlxSig(.float32, gates[t * QKV + h * D + c]));
                    o.e += U * @abs(o.v);
                    put.one(want, mag, &at, host.round(bf, o));
                }
            };
            for (0..3) |r| for (0..3 * QKV) |col| put.one(want, mag, &at, .exact(win.at(self, T + r, col)));
        } else {
            for (states[0..PLANE]) |x| put.one(want, mag, &at, x);
            if (self.cap) for (states[PLANE..]) |x| put.one(want, mag, &at, x);
        }
        std.debug.assert(at == want.len);
    }

    pub fn reference(self: *const Step, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Step, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, acc, order);
    }
};

fn tokens(comptime list: []const usize) []const Shape {
    comptime var out: [list.len]Shape = undefined;
    inline for (list, 0..) |r, i| out[i] = .{ .rows = r, .n = H, .k = D };
    const final = out;
    return &final;
}

fn kdaSpec(comptime name: []const u8, part: Part, served: []const Shape, tiled: usize) spec.Spec {
    const y_only = part == .act and tiled > 0;
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = if (part == .act) .bf16 else .f32,
        .tiles = .{ .{ .size = tiled }, .{}, .{} },
        .served = served,
        .sums = if (y_only) .{ .bf16 = 0, .exact = false, .why = "y leaves in bf16, whose rounding is the size of a bf16 sum's error at these q" } else .{ .exact = false },
    };
}

const plan80 = Plan{ .ntg = 80, .mr = 104 };
const plan64 = Plan{ .ntg = 64, .mr = 128 };
const core80 = coreLaunch(plan80);
const core64 = coreLaunch(plan64);
const step4 = stepLaunch(.b4);
const step5 = stepLaunch(.b5);
const step8 = stepLaunch(.b8);

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    const prefill = tokens(&.{ 1, 64 });
    inline for (.{ .{ plan80, &core80, "80 cores" }, .{ plan64, &core64, "64 cores" } }) |p| {
        try set.add(Core, .init(a, p[1], p[0], .act), kdaSpec("kda_percore bf16, " ++ p[2] ++ " (y)", .act, prefill, 12), null);
        try set.add(Core, .init(a, p[1], p[0], .state), kdaSpec("kda_percore bf16, " ++ p[2] ++ " (state)", .state, prefill, 12), null);
    }
    inline for (.{
        .{ &step4, Bits.b4, false, tokens(&.{ 1, 4, 8 }), "4-bit gates" },
        .{ &step4, Bits.b4, true, tokens(&.{ 2, 8 }), "4-bit gates, capture" },
        .{ &step8, Bits.b8, false, tokens(&.{1}), "8-bit gates" },
        .{ &step5, Bits.b5, false, tokens(&.{1}), "5-bit gates" },
    }) |c| {
        try set.add(Step, .init(a, c[0], c[1], c[2], .act), kdaSpec("kda_decode bf16, " ++ c[4] ++ " (y, conv)", .act, c[3], 0), null);
        try set.add(Step, .init(a, c[0], c[1], c[2], .state), kdaSpec("kda_decode bf16, " ++ c[4] ++ " (states)", .state, c[3], 0), null);
    }
}

// ── tests ──

fn hostBytes(a: std.mem.Allocator, x: mlx.mlx_array, s: mlx.mlx_stream) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    return a.dupe(u8, p[0 .. mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c)]);
}

fn expectSame(name: []const u8, what: usize, ours: mlx.mlx_array, theirs: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const a = std.testing.allocator;
    const x = try hostBytes(a, ours, s);
    defer a.free(x);
    const y = try hostBytes(a, theirs, s);
    defer a.free(y);
    if (!std.mem.eql(u8, x, y)) {
        std.debug.print("{s}: output {d} of the declared launch differs from the wrapper's\n", .{ name, what });
        return error.TestUnexpectedResult;
    }
}

test "kernel contract kda: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    const cores: c_int = @intCast(@import("../../util/system.zig").gpuCoreCount());
    for (set.kernels.items) |k| {
        const shape = k.spec.served[k.spec.served.len - 1];
        if (std.mem.startsWith(u8, k.spec.name, "kda_percore")) {
            const t: *Core = @ptrCast(@alignCast(k.ctx));
            const plan = kr.planFor(H, D, cores) orelse continue;
            if (plan.ntg != t.plan.ntg or plan.rows != t.plan.mr) continue; // this GPU's wrapper takes another plan
            if (!kr.serves(t.rig.stream(), H, H, D, D, .bfloat16, .float32)) continue;
            try std.testing.expect(try t.prepare(shape, .random, 7));
            var geo = t.geometry(shape).?;
            geo.unread = 0;
            // A GPU that caps this pipeline below the launch (M1/M2) never runs it: the wrapper declines there too.
            t.rig.launch(t.launch, geo, &t.args) catch |e| if (e == error.ThreadgroupCap) continue else return e;
            const g = t.args;
            const r = try kr.recur(t.rig.stream(), g[0], g[1], g[2], g[3], g[4], g[5], g[6], LB, g[8]);
            defer {
                _ = mlx.mlx_array_free(r.y);
                _ = mlx.mlx_array_free(r.state);
            }
            try expectSame(k.spec.name, 0, t.rig.output(0), r.y, t.rig.stream());
            try expectSame(k.spec.name, 1, t.rig.output(1), r.state, t.rig.stream());
        } else {
            const t: *Step = @ptrCast(@alignCast(k.ctx));
            try std.testing.expect(try t.prepare(shape, .random, 7));
            var geo = t.geometry(shape).?;
            geo.unread = 0;
            // A GPU that caps this pipeline below the launch (M1/M2) never runs it: the wrapper declines there too.
            t.rig.launch(t.launch, geo, &t.args) catch |e| if (e == error.ThreadgroupCap) continue else return e;
            const g = t.args;
            const bits = @backingInt(t.bits);
            const fb = kr.Low{ .w = g[8], .s = g[9], .b = g[10], .bits = bits, .gs = GS };
            const gb = kr.Low{ .w = g[11], .s = g[12], .b = g[13], .bits = bits, .gs = GS };
            const r = (try kr.decodeStep(t.rig.stream(), g[0], OFF_GA, OFF_FA, OFF_B, H, g[6], g[1], g[2], g[3], g[7], g[4], fb, gb, LB, NORM_EPS, t.cap)) orelse return error.WrapperDeclined;
            defer for ([_]mlx.mlx_array{ r.y, r.conv_state, r.state, r.state_seq }) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(x);
            };
            for ([_]mlx.mlx_array{ r.y, r.conv_state, r.state, r.state_seq }, 0..) |x, i| if (x.ctx != null) try expectSame(k.spec.name, i, t.rig.output(i), x, t.rig.stream());
        }
    }
}
