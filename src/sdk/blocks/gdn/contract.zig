//! The gdn area's kernel contract. The delta-rule recurrence kernels (stock per-token with its scalar or per-channel
//! gate and per-step state capture, blocked-seq and pipelined prefill) run at Qwen3.5-2B's 16 key and 16 value heads
//! of 128; the decode-width kernels (packed prework, norm-gate, K1, the Hadamard K2, K1S and its fold, the tree's
//! K1P / K1TR, the replay K1R) at Qwen3.5-4B's 16 key and 32 value heads. Each meets the float64 bound with a
//! forward-error allowance (`host.Val`): the reference rounds the conv, MLX's sigmoid, the norms, the gate and the
//! state where the kernels round, and carries the fast-math exp / log and fp32 errors. A kernel writing two dtypes
//! registers once per dtype. M1 and up; K1S's 1024-thread build runs on NAX only, the fold probes its limit.
//! CUDA twins (`gated_delta*.cu`): no rig yet, not registered.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../testing/kernel_contract/root.zig");
const gk = @import("kernels.zig");
const dec = @import("decode.zig");
const df = @import("decode_fused.zig");
const fa = @import("../mlp/fused_activations.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Val = host.Val;
const Shape = spec.Shape;
const U = host.U;

/// Qwen3.5-2B's GatedDeltaNet geometry.
const HK = 16;
const HV = 16;
const DK = 128;
const DV = 128;

// ── the recurrence, in float64 with its allowance ──

/// One step of the delta rule on a state row: decay, kv = state . k, delta = (v - kv) * beta, state += k * delta,
/// out = state . q. Sums in `acc` (null: float64) in `order`; `depth` the device's longest summation chain.
pub fn step(st: []Val, k: []const Val, q: []const Val, v: Val, g: []const Val, beta: Val, depth: f64, acc: ?spec.Accum, order: spec.Order, terms: []f64) Val {
    var e: f64 = 0;
    var m: f64 = 0;
    for (st, k, terms, 0..) |*s, kk, *t, i| {
        const gg = g[if (g.len == 1) 0 else i];
        const d = s.v * gg.v;
        const old = s.*;
        s.* = .{ .v = d, .e = @abs(gg.v) * old.e + @abs(old.v) * gg.e + U * @abs(d) };
        t.* = s.v * kk.v;
        e += @abs(kk.v) * s.e + @abs(s.v) * kk.e;
        m += @abs(t.*);
    }
    const kv = Val{ .v = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms), .e = e + spec.gamma(depth, U) * m };
    const diff = v.v - kv.v;
    const delta = Val{ .v = diff * beta.v, .e = (v.e + kv.e + U * (@abs(v.v) + @abs(kv.v))) * @abs(beta.v) + @abs(diff) * beta.e + U * @abs(diff * beta.v) };
    e = 0;
    m = 0;
    for (st, k, q, terms) |*s, kk, qq, *t| {
        const add = kk.v * delta.v;
        const nv = s.v + add;
        const old = s.*;
        s.* = .{ .v = nv, .e = old.e + @abs(kk.v) * delta.e + @abs(delta.v) * kk.e + spec.gamma(2, U) * (@abs(old.v) + @abs(add)) };
        t.* = s.v * qq.v;
        e += @abs(qq.v) * s.e + @abs(s.v) * qq.e;
        m += @abs(t.*);
    }
    return .{ .v = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms), .e = e + spec.gamma(depth, U) * m };
}

// ── the recurrence kernels: stock, blocked, pipelined ──

const Route = enum { stock, stock_vec, blocked, pipelined };
/// What a registration reads: y (OutT), the states (StT), or both when they share a dtype.
const Part = enum { y, states, both };

fn recurLaunch(comptime route: Route, comptime capture: bool, comptime m: u32) metal.Launch {
    const vec = route == .stock_vec;
    return .{
        .name = switch (route) {
            .stock, .stock_vec => "gated_delta_step" ++ (if (capture) "_seq" else "") ++ (if (vec) "_vec" else ""),
            .blocked => "gated_delta_blocked_tb32",
            .pipelined => "gated_delta_pipelined_tb12",
        },
        .inputs = if (capture) &.{ "q", "k", "v", "g", "beta", "state_in", "T", "seq_stride" } else &.{ "q", "k", "v", "g", "beta", "state_in", "T" },
        .outputs = if (capture) &.{ "y", "state_seq", "state_out" } else &.{ "y", "state_out" },
        .source = switch (route) {
            .stock, .stock_vec => gk.gdnKernelSource(vec, capture),
            .blocked => "constexpr int TB = 32;\n" ++ kernelSource("kernels/gdn/gated_delta_blocked_body.metal"),
            .pipelined => kernelSource("kernels/gdn/gdn_pipelined.metal"),
        },
        // blocked: k_s, q_s [32][136] and v_s [32][40] bf16, g_s, b_s [32] f32; pipelined: k_s, q_s [12][128] bf16,
        // v_s [12][16] f32, g_s, b_s [12] f32, y_s [12][16] bf16.
        .tg_bytes = switch (route) {
            .stock, .stock_vec => 0,
            .blocked => 32 * (2 * 136 * 2 + 40 * 2 + 8),
            .pipelined => 2 * 12 * 128 * 2 + 12 * 16 * 4 + 2 * 12 * 4 + 12 * 16 * 2,
        },
        .m1 = m,
        .m2 = m,
    };
}

/// A shape's rows are the tokens T (one batch row); the head geometry is fixed.
const Recur = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    route: Route,
    capture: bool,
    st_dt: mlx.mlx_dtype,
    part: Part,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    g: []f64 = &.{},
    beta: []f64 = &.{},
    s0: []f64 = &.{},
    args: [8]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, route: Route, capture: bool, st_dt: mlx.mlx_dtype, part: Part) Recur {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .route = route, .capture = capture, .st_dt = st_dt, .part = part, .arena = .init(a) };
    }

    pub fn deinit(self: *Recur) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn vec(self: *const Recur) bool {
        return self.route == .stock_vec;
    }

    pub fn geometry(self: *const Recur, shape: Shape) ?metal.Geometry {
        const t: c_int = @intCast(shape.rows);
        if (shape.rows < 1) return null;
        var g: metal.Geometry = switch (self.route) {
            .stock, .stock_vec => .{ .grid = .{ 32, DV, HV }, .tg = .{ 32, 4, 1 } },
            .blocked => .{ .grid = .{ 256 * DV / 32, HV, 1 }, .tg = .{ 256, 1, 1 } },
            .pipelined => .{ .grid = .{ 128 * DV / 16, HV, 1 }, .tg = .{ 128, 1, 1 } },
        };
        g.out(&.{ 1, t, HV, DV }, .bfloat16);
        if (self.capture) g.out(&.{ t, HV, DV, DK }, self.st_dt);
        g.out(&.{ 1, HV, DV, DK }, self.st_dt);
        g.unread = switch (self.part) {
            .y => if (self.capture) 0b110 else 0b10,
            .states => 0b1,
            .both => 0,
        };
        // The capture's last row is never written by design (the final state is its own output): it starts, and
        // stays, at 0, which the reference expects there; every other element still has to be written.
        if (self.capture) g.poison = 0;
        g.dtype("InT", .bfloat16);
        g.dtype("StT", self.st_dt);
        g.dtype("OutT", .bfloat16);
        g.int("Dk", DK);
        g.int("Dv", DV);
        g.int("Hk", HK);
        g.int("Hv", HV);
        return g;
    }

    pub fn prepare(self: *Recur, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const t = shape.rows;
        const ng: usize = if (self.vec()) DK else 1;
        // q and k as the prework leaves them (unit rows times 1/dk and 1/sqrt(dk)), v and the state of order one.
        self.q = try host.uniform(a, rng, t * HK * DK, -0.02, 0.02, .bfloat16);
        self.k = try host.uniform(a, rng, t * HK * DK, -0.15, 0.15, .bfloat16);
        self.v = try host.uniform(a, rng, t * HV * DV, -2, 2, .bfloat16);
        self.g = try host.uniform(a, rng, t * HV * ng, 0.8, 1, .float32);
        self.beta = try host.uniform(a, rng, t * HV, 0.1, 0.9, .bfloat16);
        self.s0 = try host.uniform(a, rng, HV * DV * DK, -1, 1, self.st_dt);
        switch (fill) {
            .random => {},
            .nonfinite => {
                host.plant(self.v, t * HV, DV);
                self.s0[DK * 3 + 9] = std.math.nan(f64);
            },
            .stress => {
                // Gate 4: integers and unit betas; the state row's dots are the sums, exact in fp32 by construction.
                // States: kv sums 128 products past fp16's integers at the first token, after which the gate is 0
                // (the state restarts from k * v, exactly). y: unit gates, two nonzero k channels per token (the
                // state stays small), q and the state past fp16's range in their products.
                const states = self.part == .states;
                for (self.q) |*x| x.* = if (states) 0 else @floatFromInt(rng.intRangeAtMost(i32, -255, 255));
                @memset(self.k, 0);
                for (0..t * HK) |r| {
                    if (states) {
                        for (self.k[r * DK ..][0..DK]) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                    } else for (0..2) |_| {
                        self.k[r * DK + rng.uintLessThan(usize, DK)] = if (rng.boolean()) 1 else -1;
                    }
                }
                for (self.v) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                @memset(self.g, 1);
                if (states) @memset(self.g[HV * ng ..], 0);
                @memset(self.beta, 1);
                for (self.s0) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, if (states) -8 else -64, if (states) 8 else 64));
            },
        }
        const rig = &self.rig;
        rig.reset();
        const tt: c_int = @intCast(t);
        self.args[0] = try rig.input(.bfloat16, &.{ 1, tt, HK, DK }, self.q);
        self.args[1] = try rig.input(.bfloat16, &.{ 1, tt, HK, DK }, self.k);
        self.args[2] = try rig.input(.bfloat16, &.{ 1, tt, HV, DV }, self.v);
        self.args[3] = if (self.vec()) try rig.input(.float32, &.{ 1, tt, HV, DK }, self.g) else try rig.input(.float32, &.{ 1, tt, HV }, self.g);
        self.args[4] = try rig.input(.bfloat16, &.{ 1, tt, HV }, self.beta);
        self.args[5] = try rig.input(self.st_dt, &.{ 1, HV, DV, DK }, self.s0);
        self.args[6] = try host.intScalar(rig, tt);
        self.args[7] = try host.intScalar(rig, HV * DV * DK);
        return true;
    }

    pub fn run(self: *Recur) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..if (self.capture) 8 else 7]);
    }

    pub fn count(self: *const Recur) usize {
        const t = self.shape.rows;
        const y = t * HV * DV;
        const states = (if (self.capture) t + 1 else 1) * HV * DV * DK;
        return switch (self.part) {
            .y => y,
            .states => states,
            .both => y + states,
        };
    }

    /// The device's longest dot: four (stock) or sixteen (blocked, pipelined) products a lane, then the lane tree.
    fn depth(self: *const Recur) f64 {
        return if (self.route == .stock or self.route == .stock_vec) 4 + 5 else 16 + 3;
    }

    fn outputsIn(self: *const Recur, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const a = self.rig.a;
        const t_len = self.shape.rows;
        const st = try a.alloc(Val, DK);
        defer a.free(st);
        const terms = try a.alloc(f64, DK);
        defer a.free(terms);
        const kv = try a.alloc(Val, DK);
        defer a.free(kv);
        const qv = try a.alloc(Val, DK);
        defer a.free(qv);
        const gv = try a.alloc(Val, if (self.vec()) DK else 1);
        defer a.free(gv);
        const y_len = t_len * HV * DV;
        const with_y = self.part != .states;
        const st_base: usize = if (with_y) y_len else 0;
        const plane = HV * DV * DK;
        for (0..HV) |hv| for (0..DV) |dv| {
            const hk = hv / (HV / HK);
            const row = (hv * DV + dv) * DK;
            for (st, self.s0[row..][0..DK]) |*o, x| o.* = .exact(x);
            for (0..t_len) |t| {
                for (kv, qv, self.k[(t * HK + hk) * DK ..][0..DK], self.q[(t * HK + hk) * DK ..][0..DK]) |*ko, *qo, kx, qx| {
                    ko.* = .exact(kx);
                    qo.* = .exact(qx);
                }
                const gs = self.g[(t * HV + hv) * gv.len ..][0..gv.len];
                for (gv, gs) |*o, x| o.* = .exact(x);
                const y = step(st, kv, qv, .exact(self.v[(t * HV + hv) * DV + dv]), gv, .exact(self.beta[t * HV + hv]), self.depth(), acc, order, terms);
                if (with_y) {
                    const at = (t * HV + hv) * DV + dv;
                    want[at] = y.v;
                    if (mag) |m| m[at] = host.allowance(y.e);
                }
                if (self.part != .y and self.capture and t + 1 < t_len) self.putState(want, mag, st_base + t * plane + row, st);
            }
            if (self.part != .y) self.putState(want, mag, st_base + (if (self.capture) t_len * plane else 0) + row, st);
        };
        // The capture's last row is never written (the final state is its own output): the poison, 0.
        if (self.part != .y and self.capture) for (want[st_base + (t_len - 1) * plane ..][0..plane], 0..) |*o, i| {
            o.* = 0;
            if (mag) |m| m[st_base + (t_len - 1) * plane + i] = 0;
        };
    }

    fn putState(_: *const Recur, want: []f64, mag: ?[]f64, at: usize, st: []const Val) void {
        for (st, 0..) |s, i| {
            want[at + i] = s.v;
            if (mag) |m| m[at + i] = host.allowance(s.e);
        }
    }

    pub fn reference(self: *const Recur, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Recur, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }
};

fn recurSpec(comptime name: []const u8, part: Part, st_dt: mlx.mlx_dtype, served: []const Shape, tiled: usize) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = if (part == .states) host.elemOf(st_dt) else .bf16,
        .tiles = .{ .{ .size = tiled }, .{}, .{} },
        .served = served,
        .sums = .{},
    };
}

const decode_t: []const Shape = &.{ .{ .rows = 1, .n = HV, .k = DK }, .{ .rows = 4, .n = HV, .k = DK } };
const verify_t: []const Shape = &.{ .{ .rows = 4, .n = HV, .k = DK }, .{ .rows = 16, .n = HV, .k = DK } };
const prefill_t: []const Shape = &.{ .{ .rows = 1, .n = HV, .k = DK }, .{ .rows = 64, .n = HV, .k = DK } };

const stock = recurLaunch(.stock, false, 1024);
const stock_seq = recurLaunch(.stock, true, 1024);
const stock_vec = recurLaunch(.stock_vec, false, 1024);
const blocked = recurLaunch(.blocked, false, 640);
const pipelined = recurLaunch(.pipelined, false, 1024);

// ── the decode-width kernels: prework, norm-gate, K1, K2, K1S, its fold, K1P / K1TR, K1R ──

/// The decode kernels' geometry: Qwen3.5-4B / Bonsai-4B's 16 key and 32 value heads (two value heads a key head).
const NK = 16;
const NV = 32;
const CH = 2 * NK * DK + NV * DV;
const VD = NV * DV;
const PLANE = NV * DV * DK;

const Kind = enum { prework, normgate, k1, k2, k1s, fold, k1p, k1tr, k1r };
/// Which outputs a registration reads: those in the activation dtype, the wider ones (state, an f32 gate), or all.
const Reads = enum { act, wide, all };

const Cfg = struct {
    kind: Kind,
    dt: mlx.mlx_dtype = .bfloat16,
    st: mlx.mlx_dtype = .bfloat16,
    reads: Reads = .all,
    tab: bool = false,
    gate: bool = true,
    swish: bool = true,
    block: c_int = 512,
    nt: c_int = 256,

    fn gt(c: Cfg) mlx.mlx_dtype {
        return dec.gateDtype(c.dt, c.st);
    }

    /// The dtype of each output, in the launch's order.
    fn outDtypes(c: Cfg, buf: *[6]mlx.mlx_dtype) []const mlx.mlx_dtype {
        const d = c.dt;
        const list: []const mlx.mlx_dtype = switch (c.kind) {
            .prework => &.{ d, d, d, d, c.gt(), d },
            .normgate, .k2, .k1tr => &.{d},
            .k1 => &.{ d, d, c.st },
            .k1s => &.{ d, d, c.st, c.st },
            .fold => &.{ d, d, c.st, c.st, d },
            .k1p => &.{ .float32, .float32, .float32, .float32, d },
            .k1r => &.{ c.st, d },
        };
        @memcpy(buf[0..list.len], list);
        return buf[0..list.len];
    }

    fn reads_(c: Cfg, i: usize, dt: mlx.mlx_dtype) bool {
        if (c.kind == .prework and i == 4 and !c.gate) return false;
        return switch (c.reads) {
            .all => true,
            .act => dt == c.dt,
            .wide => dt != c.dt,
        };
    }

    fn outElem(c: Cfg) spec.Elem {
        if (c.reads != .wide) return host.elemOf(c.dt);
        var buf: [6]mlx.mlx_dtype = undefined;
        for (c.outDtypes(&buf)) |d| if (d != c.dt) return host.elemOf(d);
        unreachable;
    }
};

const log1p_header = kernelSource("kernels/gdn/log1p.metal");

fn decLaunch(comptime c: Cfg) metal.Launch {
    return switch (c.kind) {
        .prework => .{
            .name = "mlxserve_gdn_prework2",
            .inputs = &.{ "qkv", "conv_state", "conv_w", "q_scale", "k_scale", "b_in", "a_in", "A_log", "dt_bias", "sigtab", "seq", "qk_eps" },
            .outputs = &.{ "q_out", "k_out", "v_out", "conv_out", "g_out", "beta_out" },
            .source = kernelSource("kernels/gdn/gdn_prework.metal"),
            .header = log1p_header,
            .tg_bytes = 0,
            .m1 = 1024,
            .m2 = 1024,
        },
        .normgate => .{
            .name = "mlxserve_gdn_normgate",
            .inputs = &.{ "y", "z", "norm_w", "eps", "sigtab" },
            .outputs = &.{"out"},
            .source = kernelSource("kernels/gdn/gdn_normgate.metal"),
            .tg_bytes = 0,
            .m1 = if (c.dt == .bfloat16 and !c.tab) 928 else 1024,
            .m2 = if (c.dt == .bfloat16 and !c.tab) 928 else 1024,
        },
        .k1 => .{
            .name = "msv_gdn_decode_recur",
            .inputs = &.{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale" },
            .outputs = &.{ "y", "conv_out", "state_out" },
            .source = dec.K1_SOURCE,
            .header = dec.HEADER,
            .tg_bytes = (DK + DK + DV) * 4 + 8,
            .m1 = 1024,
            .m2 = 1024,
        },
        .k2 => .{
            .name = "msv_gdn_decode_normgate_rot",
            .inputs = &.{ "y", "z", "norm_w", "eps", "signs" },
            .outputs = &.{"rot"},
            .source = dec.K2_SOURCE,
            .tg_bytes = @intCast(c.block * 4),
            .m1 = 1024,
            .m2 = 1024,
        },
        .k1s => .{
            .name = "msv_gdn_decode_recur_seq",
            .inputs = &.{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale" },
            .outputs = &.{ "y", "conv_out", "state_out", "state_seq" },
            .source = dec.K1S_SOURCE,
            .header = dec.HEADER,
            .tg_bytes = dec.MAX_SEQ * ((DK + DK + DV) * 4 + 8),
            .m1 = if (c.nt > 256) 1024 else if (c.st == .bfloat16) 448 else 832,
            .m2 = if (c.nt > 256) 1024 else if (c.st == .bfloat16) 448 else 832,
            .min_gen = if (c.nt > 256) 17 else 13,
        },
        .fold => .{
            .name = "msv_gdn_decode_recur_seq_fold",
            .inputs = &.{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale", "z", "norm_w", "eps" },
            .outputs = &.{ "gated", "conv_out", "state_out", "state_seq", "conv_in" },
            .source = dec.K1S_FOLD_SOURCE,
            .header = dec.HEADER,
            .tg_bytes = dec.FOLD_MAX_SEQ * (DV * 4 + (DK + DK + DV) * 4 + 8),
            .m1 = 1024,
            .m2 = 1024,
            .adapts = true,
        },
        .k1p => .{
            .name = "msv_gdn_decode_tree_pre",
            .inputs = &.{ "qkv", "a_in", "b_in", "conv_state", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale", "parents" },
            .outputs = &.{ "pq", "pk", "pv", "pg", "conv_in" },
            .source = dec.K1P_SOURCE,
            .header = dec.HEADER,
            .tg_bytes = 0,
            .m1 = 1024,
            .m2 = 1024,
        },
        .k1tr => .{
            .name = "msv_gdn_decode_tree",
            .inputs = &.{ "pq", "pk", "pv", "pg", "state_in", "parents" },
            .outputs = &.{"y"},
            .source = dec.K1TR_SOURCE,
            .tg_bytes = 0,
            .m1 = 1024,
            .m2 = 1024,
        },
        .k1r => .{
            .name = "msv_gdn_decode_replay",
            .inputs = &.{ "state_in", "pk", "pv", "pg", "path", "n_path", "conv_in", "conv_rows" },
            .outputs = &.{ "state_out", "conv_out" },
            .source = dec.K1R_SOURCE,
            .header = dec.HEADER,
            .tg_bytes = 0,
            .m1 = 1024,
            .m2 = 1024,
        },
    };
}

/// rsqrt(sum(x^2) * inv_n + eps) as the kernels take it (inv_n 1 / n: an RMS norm, 1: an l2 norm): 4 squares a lane,
/// then the simd tree (depth 9); a power-of-two inv_n is exact; precise rsqrt.
pub fn rmsInv(xs: []const Val, inv_n: f64, eps: f64, acc: ?spec.Accum, order: spec.Order, terms: []f64) Val {
    var e: f64 = 0;
    var m: f64 = 0;
    for (xs, terms[0..xs.len]) |x, *t| {
        t.* = x.v * x.v;
        e += 2 * @abs(x.v) * x.e + x.e * x.e;
        m += t.*;
    }
    const ss = if (acc) |ac| spec.sumIn(ac, order, terms[0..xs.len]) else host.sum(terms[0..xs.len]);
    const ms = Val{ .v = ss * inv_n + eps, .e = (e + spec.gamma(9, U) * m) * inv_n + U * @abs(ss * inv_n + eps) };
    const inv = 1 / @sqrt(ms.v);
    return .{ .v = inv, .e = inv * (0.51 * ms.e / ms.v + 4 * U) };
}

/// The per-token prework: normed and scaled q / k, activated v, the gate and beta, rounded where the kernels round.
const Pw = struct { q: []Val, k: []Val, v: []Val, g: []Val, beta: []Val };

/// A draft tree: each row's parent (row 0 the root), mostly a chain with a few branches.
fn treeParents(rng: std.Random, out: []i32) void {
    for (out, 0..) |*p, t| p.* = if (t == 0) -1 else if (rng.uintLessThan(u32, 3) > 0) @as(i32, @intCast(t)) - 1 else @intCast(rng.uintLessThan(usize, t));
}

/// One registration's fixture; a shape's rows are the tokens (window rows for the tree kernels).
const Dec = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: Cfg,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    qkv: []f64 = &.{},
    cs: []f64 = &.{},
    cw: []f64 = &.{},
    a_in: []f64 = &.{},
    b_in: []f64 = &.{},
    alog: []f64 = &.{},
    dtb: []f64 = &.{},
    z: []f64 = &.{},
    nw: []f64 = &.{},
    s0: []f64 = &.{},
    signs: []f64 = &.{},
    y_in: []f64 = &.{},
    table: []f64 = &.{},
    path: []f64 = &.{},
    conv_in: []f64 = &.{},
    rows3: [3]f64 = .{ 0, 0, 0 },
    pq: []f64 = &.{},
    pk: []f64 = &.{},
    pv: []f64 = &.{},
    pg: []f64 = &.{},
    qsc: f64 = 0,
    ksc: f64 = 0,
    eps: f64 = 0,
    args: [13]mlx.mlx_array = undefined,
    n_args: usize = 0,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: Cfg) Dec {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .arena = .init(a) };
    }

    pub fn deinit(self: *Dec) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    /// Tokens a launch runs: K1 and K2 are one token whatever the shape's rows (gates 1 and 2 ask for several).
    fn toks(self: *const Dec, shape: Shape) usize {
        return switch (self.cfg.kind) {
            .k1, .k2 => 1,
            else => shape.rows,
        };
    }

    fn rowsOk(self: *const Dec, s: usize) bool {
        return switch (self.cfg.kind) {
            .k1, .k2 => s >= 1,
            .k1s, .k1p, .k1tr, .k1r => s >= 1 and s <= dec.MAX_SEQ,
            .fold => s >= 1 and s <= dec.FOLD_MAX_SEQ,
            .prework, .normgate => s >= 1 and s <= 64,
        };
    }

    pub fn geometry(self: *const Dec, shape: Shape) ?metal.Geometry {
        if (!self.rowsOk(shape.rows)) return null;
        const c = self.cfg;
        const S: c_int = @intCast(self.toks(shape));
        const d = c.dt;
        var g: metal.Geometry = undefined;
        switch (c.kind) {
            .prework => {
                g = .{ .grid = .{ 32, @intCast(S), 2 * NK + NV }, .tg = .{ 32, 1, 1 } };
                g.out(&.{ 1, S, NK, DK }, d);
                g.out(&.{ 1, S, NK, DK }, d);
                g.out(&.{ 1, S, NV, DV }, d);
                g.out(&.{ 1, 3, CH }, d);
                g.out(&.{ 1, S, NV }, c.gt());
                g.out(&.{ 1, S, NV }, d);
                g.dtype("T", d);
                g.dtype("GT", c.gt());
                inline for (.{ .{ "HK", NK }, .{ "HV", NV }, .{ "DK", DK }, .{ "DV", DV }, .{ "NKEEP", 3 }, .{ "C", CH } }) |kv| g.int(kv[0], kv[1]);
                g.int("TAB", @intFromBool(c.tab));
                inline for (.{ .{ "QSTRIDE", CH }, .{ "QOFF", 0 }, .{ "BSTRIDE", NV }, .{ "BOFF", 0 }, .{ "ASTRIDE", NV }, .{ "AOFF", 0 } }) |kv| g.int(kv[0], kv[1]);
                g.int("GATE", @intFromBool(c.gate));
            },
            .normgate => {
                g = .{ .grid = .{ 32, @intCast(S), NV }, .tg = .{ 32, 1, 1 } };
                g.out(&.{ 1, S, VD }, d);
                g.dtype("T", d);
                inline for (.{ .{ "HV", NV }, .{ "DV", DV }, .{ "ZSTRIDE", VD }, .{ "ZOFF", 0 } }) |kv| g.int(kv[0], kv[1]);
                g.int("TAB", @intFromBool(c.tab));
                g.int("SWISH", @intFromBool(c.swish));
            },
            .k2 => {
                g = .{ .grid = .{ @intCast(@divExact(VD, c.block) * 128), 1, 1 }, .tg = .{ 128, 1, 1 } };
                g.out(&.{ 1, 1, VD }, d);
                g.dtype("T", d);
                g.int("DV", DV);
                g.int("J", @divExact(c.block, 32));
            },
            .k1, .k1s, .fold, .k1p, .k1tr => {
                switch (c.kind) {
                    .k1 => {
                        g = .{ .grid = .{ NV * 4 * 256, 1, 1 }, .tg = .{ 256, 1, 1 } };
                        g.out(&.{ 1, 1, NV, DV }, d);
                        g.out(&.{ 1, 3, CH }, d);
                        g.out(&.{ 1, NV, DV, DK }, c.st);
                    },
                    .k1s => {
                        g = .{ .grid = .{ @intCast(NV * 4 * c.nt), 1, 1 }, .tg = .{ @intCast(c.nt), 1, 1 } };
                        g.out(&.{ 1, S, NV, DV }, d);
                        g.out(&.{ 1, 3, CH }, d);
                        g.out(&.{ 1, NV, DV, DK }, c.st);
                        g.out(&.{ S, NV, DV, DK }, c.st);
                        g.poison = 0;
                    },
                    .fold => {
                        g = .{ .grid = .{ NV * 1024, 1, 1 }, .tg = .{ 1024, 1, 1 } };
                        g.out(&.{ 1, S, VD }, d);
                        g.out(&.{ 1, 3, CH }, d);
                        g.out(&.{ 1, NV, DV, DK }, c.st);
                        g.out(&.{ S, NV, DV, DK }, c.st);
                        g.out(&.{ 1, 3 + S, CH }, d);
                        g.poison = 0;
                    },
                    .k1p => {
                        g = .{ .grid = .{ 32, 2 * NK + NV, @intCast(S) }, .tg = .{ 32, 1, 1 } };
                        g.out(&.{ S, NK, DK }, .float32);
                        g.out(&.{ S, NK, DK }, .float32);
                        g.out(&.{ S, NV, DV }, .float32);
                        g.out(&.{ S, NV, 2 }, .float32);
                        g.out(&.{ 1, 3 + S, CH }, d);
                        inline for (.{ .{ "QS", CH }, .{ "ABS", NV }, .{ "AOFF", 0 }, .{ "BOFF", 0 } }) |kv| g.int(kv[0], kv[1]);
                    },
                    .k1tr => {
                        g = .{ .grid = .{ 32, DV, NV }, .tg = .{ 32, 4, 1 } };
                        g.out(&.{ 1, S, NV, DV }, d);
                    },
                    else => unreachable,
                }
                g.dtype("T", d);
                g.dtype("StT", c.st);
                g.dtype("GT", c.gt());
                inline for (.{ .{ "HK", NK }, .{ "HV", NV }, .{ "DK", DK }, .{ "DV", DV }, .{ "C", CH } }) |kv| g.int(kv[0], kv[1]);
                switch (c.kind) {
                    .k1 => {
                        g.int("NT", 256);
                        g.int("SPLIT", 4);
                    },
                    .k1s => {
                        g.int("NT", c.nt);
                        g.int("SPLIT", 4);
                        g.int("TL", S);
                    },
                    .fold => {
                        g.int("NT", 1024);
                        g.int("SPLIT", 1);
                        g.int("TL", S);
                        g.int("SWISH", @intFromBool(c.swish));
                    },
                    else => g.int("TL", S),
                }
            },
            .k1r => {
                g = .{ .grid = .{ 32, DV, NV }, .tg = .{ 32, 4, 1 } };
                g.out(&.{ 1, NV, DV, DK }, c.st);
                g.out(&.{ 1, 3, CH }, d);
                g.dtype("StT", c.st);
                inline for (.{ .{ "HK", NK }, .{ "HV", NV }, .{ "DK", DK }, .{ "DV", DV }, .{ "C", CH } }) |kv| g.int(kv[0], kv[1]);
            },
        }
        var buf: [6]mlx.mlx_dtype = undefined;
        g.unread = 0;
        for (c.outDtypes(&buf), 0..) |dt, i| if (!c.reads_(i, dt)) {
            g.unread |= @as(u8, 1) << @intCast(i);
        };
        return g;
    }

    pub fn prepare(self: *Dec, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const S = self.toks(shape);
        const dt = c.dt;
        // The gate's A_log rides the activation dtype in K1's family, f32 beside the prework (as the models load it).
        const adt: mlx.mlx_dtype = if (c.kind == .prework) .float32 else dt;
        self.qkv = try host.uniform(a, rng, S * CH, -2, 2, dt);
        self.cs = try host.uniform(a, rng, 3 * CH, -2, 2, dt);
        self.cw = try host.uniform(a, rng, CH * 4, -0.6, 0.6, dt);
        self.a_in = try host.uniform(a, rng, S * NV, -3, 3, dt);
        self.b_in = try host.uniform(a, rng, S * NV, -3, 3, dt);
        self.alog = try host.uniform(a, rng, NV, -2, 1.5, adt);
        self.dtb = try host.uniform(a, rng, NV, -1, 1, dt);
        self.z = try host.uniform(a, rng, S * VD, -3, 3, dt);
        self.nw = try host.uniform(a, rng, DV, 0.5, 1.5, dt);
        self.s0 = try host.uniform(a, rng, PLANE, -1, 1, c.st);
        self.signs = try a.alloc(f64, VD);
        for (self.signs) |*x| x.* = if (rng.boolean()) 1 else -1;
        self.y_in = try host.uniform(a, rng, S * VD, -3, 3, dt);
        self.pq = try host.uniform(a, rng, S * NK * DK, -0.02, 0.02, .float32);
        self.pk = try host.uniform(a, rng, S * NK * DK, -0.15, 0.15, .float32);
        self.pv = try host.uniform(a, rng, S * VD, -2, 2, .float32);
        self.pg = try a.alloc(f64, S * NV * 2);
        for (0..S * NV) |i| {
            self.pg[2 * i] = host.rt(.float32, 0.8 + 0.2 * rng.float(f64));
            self.pg[2 * i + 1] = host.rt(.float32, 0.1 + 0.8 * rng.float(f64));
        }
        self.conv_in = try host.uniform(a, rng, (3 + S) * CH, -2, 2, dt);
        const parents = try a.alloc(i32, S);
        treeParents(rng, parents);
        const tab = try a.alloc(i32, 6 * S);
        dec.treeTable(parents, tab);
        self.table = try a.alloc(f64, 6 * S);
        for (self.table, tab) |*o, x| o.* = @floatFromInt(x);
        // The replay's path: root to the last row; the conv rows it keeps, any three.
        var n_path: usize = 0;
        var r: i32 = @intCast(S - 1);
        while (r >= 0) : (r = parents[@intCast(r)]) n_path += 1;
        self.path = try a.alloc(f64, n_path);
        r = @intCast(S - 1);
        var at = n_path;
        while (r >= 0) : (r = parents[@intCast(r)]) {
            at -= 1;
            self.path[at] = @floatFromInt(r);
        }
        for (&self.rows3) |*x| x.* = @floatFromInt(rng.uintLessThan(usize, 3 + S));
        self.qsc = host.rt(dt, 1.0 / 128.0);
        self.ksc = host.rt(dt, 1 / @sqrt(@as(f64, 128)));
        self.eps = host.rt(.float32, 1e-6);

        switch (fill) {
            .random => {},
            .nonfinite => {
                // NaN, +Inf, -Inf and both into four value channels of different heads, NaN into a q channel.
                const vb = 2 * NK * DK;
                const qt: usize = @min(1, S - 1);
                const sites = [_][2]usize{ .{ 0, 5 }, .{ 1, 7 }, .{ 2, 9 }, .{ 3, 11 }, .{ 3, 12 } };
                const vals = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
                for (sites, vals) |s, x| {
                    self.qkv[vb + s[0] * DV + s[1]] = x;
                    self.y_in[s[0] * DV + s[1]] = x;
                    self.pv[@as(usize, @intFromFloat(self.path[0])) * VD + s[0] * DV + s[1]] = x;
                    self.pv[s[0] * DV + s[1]] = x;
                }
                self.qkv[qt * CH + 4 * DK + 3] = std.math.nan(f64);
                // The gate and beta: NaN and +Inf into a (msv_log1p gives NaN at +Inf), NaN into b.
                self.a_in[1] = std.math.nan(f64);
                self.a_in[2] = std.math.inf(f64);
                self.b_in[3] = std.math.nan(f64);
                for ([_]usize{ 5, 6, 7 }, vals[0..3]) |h, x| self.z[h * DV + h] = x;
                self.s0[(9 * DV + 3) * DK + 17] = std.math.nan(f64);
                self.conv_in[@as(usize, @intFromFloat(self.rows3[0])) * CH + 3] = std.math.nan(f64);
            },
            .stress => {
                // Gate 4. Conv weights [0, 0, 0, 1] and inputs >= 16 make SiLU exact (its sigmoid rounds to 1), the
                // beta exact (b = 16) and the gate 1 (A_log = -100). The recurrence kernels get every channel at 16:
                // q and k are then exact (the scales), and a state of 8-bit multiples of 256 drives kv past fp16's
                // range (all positive, so its sum passes 65504). The prework alone gets distinct integers (its squares pass fp16's range).
                for (0..CH) |ch| self.cw[ch * 4 ..][0..4].* = .{ 0, 0, 0, 1 };
                const flat = c.kind != .prework and c.kind != .k1p;
                for (self.qkv) |*x| x.* = if (flat) 16 else @floatFromInt(rng.intRangeAtMost(i32, 16, 64));
                @memset(self.b_in, 16);
                @memset(self.a_in, 0);
                @memset(self.dtb, 0);
                @memset(self.alog, -100);
                @memset(self.z, 16);
                @memset(self.nw, 1);
                for (self.s0) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, 1, 255) * 256);
                for (self.y_in) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                @memset(self.pq, 1.0 / 16.0);
                @memset(self.pk, 1.0 / 8.0);
                for (self.pv) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                @memset(self.pg, 1);
            },
        }

        const rig = &self.rig;
        rig.reset();
        const Si: c_int = @intCast(S);
        const qkv = try rig.input(dt, &.{ 1, Si, CH }, self.qkv);
        const cs = try rig.input(dt, &.{ 1, 3, CH }, self.cs);
        const cw = try rig.input(dt, &.{ CH, 4, 1 }, self.cw);
        const qs = try rig.input(dt, &.{}, &.{self.qsc});
        const ks = try rig.input(dt, &.{}, &.{self.ksc});
        var n: usize = 0;
        const args = &self.args;
        switch (c.kind) {
            .prework => {
                const tabv: mlx.mlx_array = if (c.tab) try rig.input(dt, &.{65536}, try host.read(a, try fa.sigmoidTableFor(rig.stream(), dt), rig.stream())) else qkv;
                args[0..12].* = .{ qkv, cs, cw, qs, ks, try rig.input(dt, &.{ 1, Si, NV }, self.b_in), try rig.input(dt, &.{ 1, Si, NV }, self.a_in), try rig.input(adt, &.{NV}, self.alog), try rig.input(dt, &.{NV}, self.dtb), tabv, try host.intScalar(rig, Si), try rig.scalar(@floatCast(self.eps)) };
                n = 12;
            },
            .normgate => {
                const y = try rig.input(dt, &.{ 1, Si, NV, DV }, self.y_in);
                const tabv: mlx.mlx_array = if (c.tab) try rig.input(dt, &.{65536}, try host.read(a, try fa.sigmoidTableFor(rig.stream(), dt), rig.stream())) else y;
                args[0..5].* = .{ y, try rig.input(dt, &.{ 1, Si, VD }, self.z), try rig.input(dt, &.{DV}, self.nw), try rig.scalar(@floatCast(self.eps)), tabv };
                n = 5;
            },
            .k2 => {
                args[0..5].* = .{ try rig.input(dt, &.{ 1, 1, NV, DV }, self.y_in), try rig.input(dt, &.{ 1, 1, VD }, self.z), try rig.input(dt, &.{DV}, self.nw), try rig.scalar(@floatCast(self.eps)), try rig.input(.float32, &.{VD}, self.signs) };
                n = 5;
            },
            .k1, .k1s, .fold => {
                args[0..10].* = .{ qkv, try rig.input(dt, &.{ 1, Si, NV }, self.a_in), try rig.input(dt, &.{ 1, Si, NV }, self.b_in), cs, try rig.input(c.st, &.{ 1, NV, DV, DK }, self.s0), cw, try rig.input(dt, &.{NV}, self.alog), try rig.input(dt, &.{NV}, self.dtb), qs, ks };
                n = 10;
                if (c.kind == .fold) {
                    args[10..13].* = .{ try rig.input(dt, &.{ 1, Si, VD }, self.z), try rig.input(dt, &.{DV}, self.nw), try rig.scalar(@floatCast(self.eps)) };
                    n = 13;
                }
            },
            .k1p => {
                args[0..10].* = .{ qkv, try rig.input(dt, &.{ 1, Si, NV }, self.a_in), try rig.input(dt, &.{ 1, Si, NV }, self.b_in), cs, cw, try rig.input(dt, &.{NV}, self.alog), try rig.input(dt, &.{NV}, self.dtb), qs, ks, try rig.input(.int32, &.{6 * Si}, self.table) };
                n = 10;
            },
            .k1tr => {
                args[0..6].* = .{ try rig.input(.float32, &.{ Si, NK, DK }, self.pq), try rig.input(.float32, &.{ Si, NK, DK }, self.pk), try rig.input(.float32, &.{ Si, NV, DV }, self.pv), try rig.input(.float32, &.{ Si, NV, 2 }, self.pg), try rig.input(c.st, &.{ 1, NV, DV, DK }, self.s0), try rig.input(.int32, &.{6 * Si}, self.table) };
                n = 6;
            },
            .k1r => {
                args[0..8].* = .{ try rig.input(c.st, &.{ 1, NV, DV, DK }, self.s0), try rig.input(.float32, &.{ Si, NK, DK }, self.pk), try rig.input(.float32, &.{ Si, NV, DV }, self.pv), try rig.input(.float32, &.{ Si, NV, 2 }, self.pg), try rig.input(.int32, &.{@intCast(self.path.len)}, self.path), try rig.input(.int32, &.{1}, &.{@floatFromInt(self.path.len)}), try rig.input(dt, &.{ 1, 3 + Si, CH }, self.conv_in), try rig.input(.int32, &.{3}, &self.rows3) };
                n = 8;
            },
        }
        self.n_args = n;
        return true;
    }

    pub fn run(self: *Dec) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Dec) usize {
        var buf: [6]mlx.mlx_dtype = undefined;
        var lens: [6]usize = undefined;
        const outs = self.cfg.outDtypes(&buf);
        self.lengths(&lens);
        var n: usize = 0;
        for (outs, 0..) |dt, i| if (self.cfg.reads_(i, dt)) {
            n += lens[i];
        };
        return n;
    }

    fn lengths(self: *const Dec, lens: *[6]usize) void {
        const S = self.toks(self.shape);
        const l: []const usize = switch (self.cfg.kind) {
            .prework => &.{ S * NK * DK, S * NK * DK, S * VD, 3 * CH, S * NV, S * NV },
            .normgate, .k1tr => &.{S * VD},
            .k2 => &.{VD},
            .k1 => &.{ VD, 3 * CH, PLANE },
            .k1s => &.{ S * VD, 3 * CH, PLANE, S * PLANE },
            .fold => &.{ S * VD, 3 * CH, PLANE, S * PLANE, (3 + S) * CH },
            .k1p => &.{ S * NK * DK, S * NK * DK, S * VD, S * NV * 2, (3 + S) * CH },
            .k1r => &.{ PLANE, 3 * CH },
        };
        @memcpy(lens[0..l.len], l);
    }

    /// Row w of concat(conv_state, qkv).
    fn convRow(self: *const Dec, w: usize) []const f64 {
        return if (w < 3) self.cs[w * CH ..][0..CH] else self.qkv[(w - 3) * CH ..][0..CH];
    }

    /// The prework of every token over its conv window (`win`: four rows of concat(conv_state, qkv) a token).
    fn prework(self: *const Dec, a: std.mem.Allocator, win: []const usize, acc: ?spec.Accum, order: spec.Order) !Pw {
        const S = self.toks(self.shape);
        const dt = self.cfg.dt;
        const pw = Pw{
            .q = try a.alloc(Val, S * NK * DK),
            .k = try a.alloc(Val, S * NK * DK),
            .v = try a.alloc(Val, S * VD),
            .g = try a.alloc(Val, S * NV),
            .beta = try a.alloc(Val, S * NV),
        };
        const act = try a.alloc(Val, CH);
        const terms = try a.alloc(f64, DK);
        for (0..S) |t| {
            for (0..CH) |ch| {
                var sum: f64 = 0;
                var m: f64 = 0;
                for (0..4) |tap| {
                    const p = self.convRow(win[t * 4 + tap])[ch] * self.cw[ch * 4 + tap];
                    sum += p;
                    m += @abs(p);
                }
                const conv = host.round(dt, .{ .v = sum, .e = spec.gamma(3, U) * m });
                act[ch] = host.round(dt, host.mul(conv, host.mlxSig(dt, conv)));
            }
            for (0..2 * NK) |h| {
                const xs = act[h * DK ..][0..DK];
                const inv = rmsInv(xs, 1.0 / @as(f64, DK), 1e-6, acc, order, terms);
                const scale = if (h < NK) self.qsc else self.ksc;
                const dst = if (h < NK) pw.q[(t * NK + h) * DK ..][0..DK] else pw.k[(t * NK + h - NK) * DK ..][0..DK];
                for (dst, xs) |*o, x| {
                    const r = host.round(dt, host.mul(x, inv));
                    o.* = host.round(dt, .{ .v = scale * r.v, .e = @abs(scale) * r.e });
                }
            }
            @memcpy(pw.v[t * VD ..][0..VD], act[2 * NK * DK ..][0..VD]);
            for (0..NV) |h| {
                pw.beta[t * NV + h] = host.mlxSig(dt, .exact(self.b_in[t * NV + h]));
                pw.g[t * NV + h] = self.gateOf(self.a_in[t * NV + h], self.dtb[h], self.alog[h]);
            }
        }
        return pw;
    }

    /// g = exp(-exp(A_log) * softplus(a + dt_bias)) in GT: f32 add, precise exp, the fast-log log1p, precise exp.
    fn gateOf(self: *const Dec, a_in: f64, dtb: f64, alog: f64) Val {
        const gt = self.cfg.gt();
        const apd = host.rt(gt, host.rt(.float32, a_in + dtb));
        if (!std.math.isFinite(apd)) {
            // msv_log1p's own formula: x * (log(1 + x) / (1 + x - 1)), NaN at +Inf.
            const x = @exp(apd);
            const xp1 = 1 + x;
            const l = if (xp1 == 1) x else x * (@log(xp1) / (xp1 - 1));
            return .{ .v = host.rt(gt, @exp(-(@exp(alog) * l))) };
        }
        const ex = @exp(apd);
        const sp = std.math.log1p(ex);
        // The log1p's fast log: 2^-21 absolute in [0.5, 2] (times ex / (1 + ex - 1) <= 2), 3 ulps past it.
        const log_err = if (1 + ex <= 2) 0x1p-20 else 6 * U * @log(1 + ex) * 2;
        const sp_e = 0x1p-21 * ex / (1 + ex) + 4 * U * sp + log_err + (if (ex < 0x1p-22) ex else 0);
        const ea = @exp(alog);
        // A subnormal exp(A_log) may flush to 0.
        const ea_e = 0x1p-21 * ea + (if (ea < 0x1p-126) ea else 0);
        const p = ea * sp;
        const p_e = ea_e * sp + ea * sp_e + U * p;
        const gx = @exp(-p);
        return host.round(gt, .{ .v = gx, .e = gx * std.math.expm1(p_e) + 0x1p-21 * gx });
    }

    /// The norm-gate of `y` rows ([S, NV, DV]) against z: rms_norm(y) * norm_w, then silu(z) * it or it * sigmoid(z).
    fn normGate(self: *const Dec, a: std.mem.Allocator, y: []const Val, acc: ?spec.Accum, order: spec.Order) ![]Val {
        const dt = self.cfg.dt;
        const out = try a.alloc(Val, y.len);
        const terms = try a.alloc(f64, DV);
        for (0..y.len / DV) |row| {
            const xs = y[row * DV ..][0..DV];
            const inv = rmsInv(xs, 1.0 / @as(f64, DV), self.eps, acc, order, terms);
            for (out[row * DV ..][0..DV], xs, 0..) |*o, x, i| {
                const r = host.round(dt, host.mul(x, inv));
                const w = self.nw[i];
                const normed = host.round(dt, .{ .v = w * r.v, .e = @abs(w) * r.e });
                const zv = Val.exact(self.z[row * DV + i]);
                const sig = host.mlxSig(dt, zv);
                o.* = if (self.cfg.swish) host.round(dt, host.mul(host.round(dt, host.mul(zv, sig)), normed)) else host.round(dt, host.mul(normed, sig));
            }
        }
        return out;
    }

    /// The recurrence along each row's token order, the state rounded to StT after every token (K1 holds one).
    /// `ys` [S, NV, DV] (null: not kept); `seq` the states after rows 0..S-2; `final` the last state.
    fn recur(self: *const Dec, a: std.mem.Allocator, pw: Pw, acc: ?spec.Accum, order: spec.Order, ys: ?[]Val, seq: ?[]Val, final: []Val) !void {
        const S = self.toks(self.shape);
        const c = self.cfg;
        const st = try a.alloc(Val, DK);
        const kept = try a.alloc(Val, S * DK);
        const terms = try a.alloc(f64, DK);
        const parents = self.table[0..S];
        for (0..NV) |hv| for (0..DV) |dv| {
            const hk = hv / (NV / NK);
            const row = (hv * DV + dv) * DK;
            for (st, self.s0[row..][0..DK]) |*o, x| o.* = .exact(x);
            const n_steps = if (c.kind == .k1r) self.path.len else S;
            for (0..n_steps) |i| {
                const t: usize = if (c.kind == .k1r) @intFromFloat(self.path[i]) else i;
                if (c.kind == .k1tr and t > 0) {
                    const p: usize = @intFromFloat(parents[t]);
                    if (p + 1 != t) @memcpy(st, kept[p * DK ..][0..DK]);
                }
                const y = step(st, pw.k[(t * NK + hk) * DK ..][0..DK], pw.q[(t * NK + hk) * DK ..][0..DK], pw.v[t * VD + hv * DV + dv], pw.g[t * NV + hv ..][0..1], pw.beta[t * NV + hv], 9, acc, order, terms);
                if (ys) |o| o[t * VD + hv * DV + dv] = host.round(c.dt, y);
                for (st) |*s| s.* = host.round(c.st, s.*);
                @memcpy(kept[t * DK ..][0..DK], st);
                if (seq) |o| if (t + 1 < S) @memcpy(o[t * PLANE + row ..][0..DK], st);
            }
            @memcpy(final[row..][0..DK], st);
        };
        // The capture's last row is never written: the poison, 0.
        if (seq) |o| @memset(o[(S - 1) * PLANE ..][0..PLANE], Val.exact(0));
    }

    fn exactRows(a: std.mem.Allocator, x: []const f64) ![]Val {
        const out = try a.alloc(Val, x.len);
        for (out, x) |*o, v| o.* = .exact(v);
        return out;
    }

    /// Rows `rows` of concat(conv_state, qkv), exact copies.
    fn copyRows(self: *const Dec, a: std.mem.Allocator, rows: []const usize) ![]Val {
        const out = try a.alloc(Val, rows.len * CH);
        for (rows, 0..) |w, j| for (out[j * CH ..][0..CH], self.convRow(w)) |*o, x| {
            o.* = .exact(x);
        };
        return out;
    }

    /// Every output, in the launch's order (unread ones too), as the device should leave them.
    fn outputs(self: *const Dec, a: std.mem.Allocator, acc: ?spec.Accum, order: spec.Order) ![6][]Val {
        const S = self.toks(self.shape);
        const c = self.cfg;
        var o: [6][]Val = @splat(&.{});
        const pre_acc = if (c.kind == .prework or c.kind == .k1p) acc else null;
        const win = try a.alloc(usize, S * 4);
        for (0..S) |t| for (0..4) |tap| {
            win[t * 4 + tap] = if (c.kind == .k1p) @intFromFloat(self.table[S + t * 4 + tap]) else t + tap;
        };
        const next = try a.alloc(usize, 3);
        for (next, 0..) |*w, j| w.* = S + j;
        const all = try a.alloc(usize, 3 + S);
        for (all, 0..) |*w, j| w.* = j;
        switch (c.kind) {
            .prework => {
                const pw = try self.prework(a, win, pre_acc, order);
                o[0..6].* = .{ pw.q, pw.k, pw.v, try self.copyRows(a, next), pw.g, pw.beta };
            },
            .normgate => o[0] = try self.normGate(a, try exactRows(a, self.y_in), acc, order),
            .k2 => {
                const gated = try self.normGate(a, try exactRows(a, self.y_in), acc, order);
                o[0] = try self.rotate(a, gated, acc);
            },
            .k1, .k1s, .fold => {
                const pw = try self.prework(a, win, null, order);
                const ys = try a.alloc(Val, S * VD);
                const final = try a.alloc(Val, PLANE);
                const seq: ?[]Val = if (c.kind == .k1) null else try a.alloc(Val, S * PLANE);
                try self.recur(a, pw, acc, order, ys, seq, final);
                o[0] = if (c.kind == .fold) try self.normGate(a, ys, acc, order) else ys;
                o[1] = try self.copyRows(a, next);
                o[2] = final;
                if (seq) |x| o[3] = x;
                if (c.kind == .fold) o[4] = try self.copyRows(a, all);
            },
            .k1p => {
                const pw = try self.prework(a, win, pre_acc, order);
                const gb = try a.alloc(Val, S * NV * 2);
                for (0..S * NV) |i| {
                    gb[2 * i] = pw.g[i];
                    gb[2 * i + 1] = pw.beta[i];
                }
                o[0..5].* = .{ pw.q, pw.k, pw.v, gb, try self.copyRows(a, all) };
            },
            .k1tr, .k1r => {
                const pw = Pw{ .q = try exactRows(a, self.pq), .k = try exactRows(a, self.pk), .v = try exactRows(a, self.pv), .g = try a.alloc(Val, S * NV), .beta = try a.alloc(Val, S * NV) };
                for (0..S * NV) |i| {
                    pw.g[i] = .exact(self.pg[2 * i]);
                    pw.beta[i] = .exact(self.pg[2 * i + 1]);
                }
                const final = try a.alloc(Val, PLANE);
                if (c.kind == .k1tr) {
                    o[0] = try a.alloc(Val, S * VD);
                    try self.recur(a, pw, acc, order, o[0], null, final);
                } else {
                    try self.recur(a, pw, acc, order, null, null, final);
                    o[0] = final;
                    o[1] = try a.alloc(Val, 3 * CH);
                    for (self.rows3, 0..) |w, j| for (o[1][j * CH ..][0..CH], self.conv_in[@as(usize, @intFromFloat(w)) * CH ..][0..CH]) |*x, v| {
                        x.* = .exact(v);
                    };
                }
            },
        }
        return o;
    }

    /// K2's out_proj rotation: per Hadamard block, H (x * signs) / sqrt(block) in Sylvester order, rounded to T.
    fn rotate(self: *const Dec, a: std.mem.Allocator, x: []const Val, acc: ?spec.Accum) ![]Val {
        const B: usize = @intCast(self.cfg.block);
        const out = try a.alloc(Val, x.len);
        const buf = try a.alloc(f64, B);
        const levels: f64 = @floatFromInt(std.math.log2_int(usize, B));
        const scale = 1 / @sqrt(@as(f64, @floatFromInt(B)));
        for (0..x.len / B) |blk| {
            const xs = x[blk * B ..][0..B];
            var e_in: f64 = 0;
            var m: f64 = 0;
            for (buf, xs, self.signs[blk * B ..][0..B]) |*o, v, s| {
                o.* = v.v * s;
                e_in += v.e;
                m += @abs(v.v);
            }
            var h: usize = 1;
            while (h < B) : (h *= 2) {
                var i: usize = 0;
                while (i < B) : (i += 2 * h) for (i..i + h) |j| {
                    const u = buf[j];
                    const w = buf[j + h];
                    buf[j] = if (acc) |ac| spec.sumIn(ac, .sequential, &.{ u, w }) else u + w;
                    buf[j + h] = if (acc) |ac| spec.sumIn(ac, .sequential, &.{ u, -w }) else u - w;
                };
            }
            for (out[blk * B ..][0..B], buf) |*o, v| {
                const r = v * scale;
                // The fast rsqrt of the block size: 2^-20 of the value.
                o.* = host.round(self.cfg.dt, .{ .v = r, .e = (e_in + spec.gamma(levels, U) * m) * scale + @abs(r) * (0x1p-20 + U) });
            }
        }
        return out;
    }

    fn emit(self: *const Dec, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        var arena = std.heap.ArenaAllocator.init(self.rig.a);
        defer arena.deinit();
        const outs = try self.outputs(arena.allocator(), acc, order);
        var buf: [6]mlx.mlx_dtype = undefined;
        var at: usize = 0;
        for (self.cfg.outDtypes(&buf), 0..) |dt, i| if (self.cfg.reads_(i, dt)) {
            for (outs[i]) |v| {
                want[at] = v.v;
                if (mag) |m| m[at] = host.allowance(v.e);
                at += 1;
            }
        };
        std.debug.assert(at == want.len);
    }

    pub fn reference(self: *const Dec, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Dec, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, acc, order);
    }
};

fn launchOf(comptime c: Cfg) *const metal.Launch {
    return &struct {
        const l = decLaunch(c);
    }.l;
}

fn tokens(comptime list: []const usize) []const Shape {
    comptime var out: [list.len]Shape = undefined;
    inline for (list, 0..) |r, i| out[i] = .{ .rows = r, .n = NV, .k = DK };
    const final = out;
    return &final;
}

fn decSpec(comptime name: []const u8, comptime c: Cfg, comptime served: []const Shape) spec.Spec {
    // These reads only copy rows.
    const moves = c.reads == .act and (c.kind == .k1p or c.kind == .k1r);
    const gate_only = c.reads == .wide and c.kind == .prework;
    return .{
        .name = name,
        .accum = if (moves) .none else .f32,
        .depth = host.depthOne,
        .out = c.outElem(),
        .served = served,
        .sums = if (gate_only)
            .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "g sums nothing: its allowance is the exps' and the fast log's" }
        else if (c.kind == .prework or c.kind == .normgate or c.kind == .k1tr)
            .{ .bf16 = 0, .exact = false, .why = "the outputs are bf16 or f16, whose own rounding is the size of a bf16 sum's error here" }
        else
            .{ .exact = false },
    };
}

fn addDec(set: *kc.Set, comptime name: []const u8, comptime c: Cfg, comptime served: []const usize) !void {
    try set.add(Dec, .init(set.a, launchOf(c), c), decSpec(name, c, tokens(served)), null);
}

fn registerDecode(set: *kc.Set) !void {
    const f32s = mlx.mlx_dtype.float32;
    const f16s = mlx.mlx_dtype.float16;
    try addDec(set, "gdn_prework2 bf16, bf16 gate", .{ .kind = .prework }, &.{ 1, 4, 9 });
    try addDec(set, "gdn_prework2 bf16, f32 gate (bf16 outputs)", .{ .kind = .prework, .st = f32s, .reads = .act }, &.{ 1, 4, 9 });
    try addDec(set, "gdn_prework2 bf16, f32 gate (g)", .{ .kind = .prework, .st = f32s, .reads = .wide }, &.{ 1, 4, 9 });
    try addDec(set, "gdn_prework2 bf16, sigmoid table (prefill)", .{ .kind = .prework, .tab = true }, &.{ 17, 64 });
    try addDec(set, "gdn_prework2 bf16, no gate (KDA)", .{ .kind = .prework, .gate = false }, &.{ 1, 4 });
    try addDec(set, "gdn_prework2 f16, f16 gate", .{ .kind = .prework, .dt = f16s, .st = f32s }, &.{ 1, 4 });
    try addDec(set, "gdn_normgate bf16 swish", .{ .kind = .normgate }, &.{ 1, 4, 16 });
    try addDec(set, "gdn_normgate bf16 sigmoid", .{ .kind = .normgate, .swish = false }, &.{ 1, 4, 16 });
    try addDec(set, "gdn_normgate bf16 swish, sigmoid table (prefill)", .{ .kind = .normgate, .tab = true }, &.{ 17, 64 });
    try addDec(set, "gdn_normgate f16 swish", .{ .kind = .normgate, .dt = f16s }, &.{ 1, 4 });
    try addDec(set, "gdn_decode_recur bf16, bf16 state", .{ .kind = .k1 }, &.{1});
    try addDec(set, "gdn_decode_recur bf16, f32 state (y, conv)", .{ .kind = .k1, .st = f32s, .reads = .act }, &.{1});
    try addDec(set, "gdn_decode_recur bf16, f32 state (state)", .{ .kind = .k1, .st = f32s, .reads = .wide }, &.{1});
    try addDec(set, "gdn_decode_recur f16, f32 state (y, conv)", .{ .kind = .k1, .dt = f16s, .st = f32s, .reads = .act }, &.{1});
    try addDec(set, "gdn_decode_recur f16, f32 state (state)", .{ .kind = .k1, .dt = f16s, .st = f32s, .reads = .wide }, &.{1});
    try addDec(set, "gdn_decode_normgate_rot f16, block 512", .{ .kind = .k2, .dt = f16s, .st = f32s }, &.{1});
    try addDec(set, "gdn_decode_normgate_rot f16, block 1024", .{ .kind = .k2, .dt = f16s, .st = f32s, .block = 1024 }, &.{1});
    try addDec(set, "gdn_decode_recur_seq bf16, bf16 state, 256 threads", .{ .kind = .k1s }, &.{ 2, 8, 16 });
    try addDec(set, "gdn_decode_recur_seq bf16, f32 state, 256 threads (y, conv)", .{ .kind = .k1s, .st = f32s, .reads = .act }, &.{ 2, 8 });
    try addDec(set, "gdn_decode_recur_seq bf16, f32 state, 256 threads (states)", .{ .kind = .k1s, .st = f32s, .reads = .wide }, &.{ 2, 8 });
    try addDec(set, "gdn_decode_recur_seq bf16, bf16 state, 1024 threads (NAX)", .{ .kind = .k1s, .nt = 1024 }, &.{ 2, 8, 16 });
    try addDec(set, "gdn_decode_recur_seq bf16, f32 state, 1024 threads (NAX) (y, conv)", .{ .kind = .k1s, .st = f32s, .reads = .act, .nt = 1024 }, &.{ 2, 8 });
    try addDec(set, "gdn_decode_recur_seq bf16, f32 state, 1024 threads (NAX) (states)", .{ .kind = .k1s, .st = f32s, .reads = .wide, .nt = 1024 }, &.{ 2, 8 });
    try addDec(set, "gdn_decode_recur_seq_fold bf16 swish, bf16 state", .{ .kind = .fold }, &.{ 1, 4, 8 });
    try addDec(set, "gdn_decode_recur_seq_fold bf16 sigmoid, f32 state (gated, conv)", .{ .kind = .fold, .st = f32s, .swish = false, .reads = .act }, &.{ 2, 8 });
    try addDec(set, "gdn_decode_recur_seq_fold bf16 sigmoid, f32 state (states)", .{ .kind = .fold, .st = f32s, .swish = false, .reads = .wide }, &.{ 2, 8 });
    try addDec(set, "gdn_decode_tree_pre bf16, bf16 gate (q, k, v, g)", .{ .kind = .k1p, .reads = .wide }, &.{ 2, 8, 16 });
    try addDec(set, "gdn_decode_tree_pre bf16 (conv input)", .{ .kind = .k1p, .reads = .act }, &.{ 2, 8, 16 });
    try addDec(set, "gdn_decode_tree_pre bf16, f32 gate (q, k, v, g)", .{ .kind = .k1p, .st = f32s, .reads = .wide }, &.{ 2, 8 });
    try addDec(set, "gdn_decode_tree bf16, bf16 state", .{ .kind = .k1tr }, &.{ 2, 8, 16 });
    try addDec(set, "gdn_decode_tree bf16, f32 state", .{ .kind = .k1tr, .st = f32s }, &.{ 2, 8 });
    try addDec(set, "gdn_decode_replay bf16 state", .{ .kind = .k1r }, &.{ 2, 8, 16 });
    try addDec(set, "gdn_decode_replay f32 state (conv)", .{ .kind = .k1r, .st = f32s, .reads = .act }, &.{ 2, 8 });
    try addDec(set, "gdn_decode_replay f32 state (state)", .{ .kind = .k1r, .st = f32s, .reads = .wide }, &.{ 2, 8 });
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    const f32s = mlx.mlx_dtype.float32;
    try set.add(Recur, .init(a, &stock, .stock, false, f32s, .y), recurSpec("gated_delta_step bf16, f32 state (y)", .y, f32s, decode_t, 0), null);
    try set.add(Recur, .init(a, &stock, .stock, false, f32s, .states), recurSpec("gated_delta_step bf16, f32 state (state)", .states, f32s, decode_t, 0), null);
    try set.add(Recur, .init(a, &stock, .stock, false, .bfloat16, .both), recurSpec("gated_delta_step bf16, bf16 state", .both, .bfloat16, decode_t, 0), null);
    try set.add(Recur, .init(a, &stock_seq, .stock, true, f32s, .y), recurSpec("gated_delta_step_seq bf16, f32 state (y)", .y, f32s, verify_t, 0), null);
    try set.add(Recur, .init(a, &stock_seq, .stock, true, f32s, .states), recurSpec("gated_delta_step_seq bf16, f32 state (states)", .states, f32s, verify_t, 0), null);
    try set.add(Recur, .init(a, &stock_vec, .stock_vec, false, f32s, .y), recurSpec("gated_delta_step_vec bf16, f32 state (y)", .y, f32s, decode_t, 0), null);
    try set.add(Recur, .init(a, &stock_vec, .stock_vec, false, f32s, .states), recurSpec("gated_delta_step_vec bf16, f32 state (state)", .states, f32s, decode_t, 0), null);
    try set.add(Recur, .init(a, &blocked, .blocked, false, f32s, .y), recurSpec("gated_delta_blocked_tb32 bf16, f32 state (y)", .y, f32s, prefill_t, 32), null);
    try set.add(Recur, .init(a, &blocked, .blocked, false, f32s, .states), recurSpec("gated_delta_blocked_tb32 bf16, f32 state (state)", .states, f32s, prefill_t, 32), null);
    try set.add(Recur, .init(a, &pipelined, .pipelined, false, f32s, .y), recurSpec("gated_delta_pipelined_tb12 bf16, f32 state (y)", .y, f32s, prefill_t, 12), null);
    try set.add(Recur, .init(a, &pipelined, .pipelined, false, f32s, .states), recurSpec("gated_delta_pipelined_tb12 bf16, f32 state (state)", .states, f32s, prefill_t, 12), null);
    try registerDecode(set);
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

/// Ours and theirs hold the same bytes, but for `skip_tail` trailing bytes neither writes.
fn expectSame(name: []const u8, what: usize, ours: mlx.mlx_array, theirs: mlx.mlx_array, skip_tail: usize, s: mlx.mlx_stream) !void {
    const a = std.testing.allocator;
    const x = try hostBytes(a, ours, s);
    defer a.free(x);
    const y = try hostBytes(a, theirs, s);
    defer a.free(y);
    if (x.len != y.len or !std.mem.eql(u8, x[0 .. x.len - skip_tail], y[0 .. y.len - skip_tail])) {
        std.debug.print("{s}: output {d} of the declared launch differs from the wrapper's\n", .{ name, what });
        return error.TestUnexpectedResult;
    }
}

fn freeAll(arrs: []const mlx.mlx_array) void {
    for (arrs) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
}

const geo_dec = dec.Geometry{ .hk = NK, .hv = NV, .dk = DK, .dv = DV };

/// K1's family's inputs as the wrappers take them, from a fixture's arguments (`state`: the tree's state).
fn decInputs(t: *const Dec, state: mlx.mlx_array) dec.Inputs {
    const g = t.args;
    const none = mlx.mlx_array{ .ctx = null };
    if (t.cfg.kind == .k1p) return .{ .qkv = g[0], .a = g[1], .b = g[2], .conv_state = g[3], .ssm_state = state, .conv_w = g[4], .A_log = g[5], .dt_bias = g[6], .q_scale = g[7], .k_scale = g[8], .z = none, .norm_w = none, .eps = none, .signs = none };
    const fold = t.cfg.kind == .fold;
    return .{ .qkv = g[0], .a = g[1], .b = g[2], .conv_state = g[3], .ssm_state = g[4], .conv_w = g[5], .A_log = g[6], .dt_bias = g[7], .q_scale = g[8], .k_scale = g[9], .z = if (fold) g[10] else none, .norm_w = if (fold) g[11] else none, .eps = if (fold) g[12] else none, .signs = none };
}

/// Runs a fixture's declared launch with every output kept.
fn runAll(t: *Dec, shape: Shape) !void {
    var g = t.geometry(shape).?;
    g.unread = 0;
    try t.rig.launch(t.launch, g, t.args[0..t.n_args]);
}

fn decDrift(t: *Dec, name: []const u8, shape: Shape) !void {
    const a = std.testing.allocator;
    const c = t.cfg;
    try std.testing.expect(try t.prepare(shape, .random, 7));
    const s = t.rig.stream();
    const S: c_int = @intCast(t.toks(shape));
    switch (c.kind) {
        .prework => {
            try runAll(t, shape);
            const g = t.args;
            const r = (try df.gdnPreworkFused(s, .{ .qkv = g[0], .qkv_off = 0, .qkv_stride = CH, .b = g[5], .b_off = 0, .b_stride = NV, .a = g[6], .a_off = 0, .a_stride = NV, .A_log = g[7], .dt_bias = g[8], .conv_state = g[1], .conv_w = g[2], .q_scale = g[3], .k_scale = g[4], .hk = NK, .hv = NV, .dk = DK, .dv = DV, .seq = S, .gate = c.gate, .g_dtype = c.gt() })) orelse return error.WrapperDeclined;
            const theirs = [_]mlx.mlx_array{ r.q, r.k, r.v, r.conv_state, r.g, r.beta };
            defer freeAll(&theirs);
            for (theirs, 0..) |x, i| if (i != 4 or c.gate) try expectSame(name, i, t.rig.output(i), x, 0, s);
        },
        .normgate => {
            try runAll(t, shape);
            const g = t.args;
            const r = (try df.gdnNormGateFused(s, g[0], g[1], 0, VD, g[2], g[3], c.swish, NV, DV, 1, S)) orelse return error.WrapperDeclined;
            defer _ = mlx.mlx_array_free(r);
            try expectSame(name, 0, t.rig.output(0), r, 0, s);
        },
        .k1 => {
            try runAll(t, shape);
            const r = (try dec.recur(geo_dec, decInputs(t, .{ .ctx = null }), s)) orelse return error.WrapperDeclined;
            const theirs = [_]mlx.mlx_array{ r.y, r.conv_state, r.ssm_state };
            defer freeAll(&theirs);
            for (theirs, 0..) |x, i| try expectSame(name, i, t.rig.output(i), x, 0, s);
        },
        .k2 => {
            // K2 reads K1's y: the wrapper chains both, ours runs on K1's y.
            const kc1 = comptime Cfg{ .kind = .k1, .dt = .float16, .st = .float32 };
            var k1 = Dec.init(a, launchOf(kc1), kc1);
            defer k1.deinit();
            try std.testing.expect(try k1.prepare(shape, .random, 7));
            var in = decInputs(&k1, .{ .ctx = null });
            const y = (try dec.recur(geo_dec, in, s)) orelse return error.WrapperDeclined;
            defer freeAll(&.{ y.y, y.conv_state, y.ssm_state });
            in.z = t.args[1];
            in.norm_w = t.args[2];
            in.eps = t.args[3];
            in.signs = t.args[4];
            const r = (try dec.step(geo_dec, c.block, in, s)) orelse return error.WrapperDeclined;
            defer freeAll(&.{ r.rot, r.conv_state, r.ssm_state });
            t.args[0] = y.y;
            try runAll(t, shape);
            try expectSame(name, 0, t.rig.output(0), r.rot, 0, s);
        },
        .k1s => {
            const nt: c_int = if (@import("../../../mlx/device.zig").naxAvailable()) 1024 else 256;
            if (c.nt != nt) return; // this GPU's wrapper launches the other width
            try runAll(t, shape);
            const r = (try dec.recurSeq(geo_dec, S, decInputs(t, .{ .ctx = null }), s)) orelse return error.WrapperDeclined;
            const theirs = [_]mlx.mlx_array{ r.y, r.conv_state, r.ssm_state, r.state_seq };
            defer freeAll(&theirs);
            for (theirs, 0..) |x, i| try expectSame(name, i, t.rig.output(i), x, if (i == 3) PLANE * mlx.mlx_array_itemsize(x) else 0, s);
        },
        .fold => {
            try runAll(t, shape);
            const r = (try dec.recurSeqFold(geo_dec, S, decInputs(t, .{ .ctx = null }), c.swish, s)) orelse return error.WrapperDeclined;
            const theirs = [_]mlx.mlx_array{ r.gated, r.conv_state, r.ssm_state, r.state_seq, r.conv_input };
            defer freeAll(&theirs);
            for (theirs, 0..) |x, i| try expectSame(name, i, t.rig.output(i), x, if (i == 3) PLANE * mlx.mlx_array_itemsize(x) else 0, s);
        },
        .k1p, .k1tr => {
            // The wrapper runs K1P then K1TR: ours run the same pair from a K1P fixture of the same seed.
            const pc = comptime Cfg{ .kind = .k1p, .reads = .act };
            var pre = Dec.init(a, launchOf(pc), .{ .kind = .k1p, .st = c.st });
            defer pre.deinit();
            try std.testing.expect(try pre.prepare(shape, .random, 7));
            try runAll(&pre, shape);
            const state = try pre.rig.input(c.st, &.{ 1, NV, DV, DK }, pre.s0);
            const r = (try dec.recurTree(geo_dec, S, decInputs(&pre, state), null, pre.args[9], s)) orelse return error.WrapperDeclined;
            defer freeAll(&.{ r.y, r.conv_input, r.prework.k, r.prework.v, r.prework.gb });
            if (c.kind == .k1p) {
                for ([_]mlx.mlx_array{ r.prework.k, r.prework.v, r.prework.gb, r.conv_input }, 1..) |x, i| try expectSame(name, i, pre.rig.output(i), x, 0, s);
            } else {
                for (0..4) |i| t.args[i] = pre.rig.output(i);
                try runAll(t, shape);
                try expectSame(name, 0, t.rig.output(0), r.y, 0, s);
            }
        },
        .k1r => {
            try runAll(t, shape);
            const g = t.args;
            var path: [dec.MAX_SEQ]i32 = undefined;
            for (t.path, 0..) |x, i| path[i] = @intFromFloat(x);
            var rows: [3]i32 = undefined;
            for (t.rows3, &rows) |x, *o| o.* = @intFromFloat(x);
            const r = (try dec.replay(geo_dec, g[0], .{ .k = g[1], .v = g[2], .gb = g[3] }, path[0..t.path.len], g[6], rows, s)) orelse return error.WrapperDeclined;
            const theirs = [_]mlx.mlx_array{ r.ssm_state, r.conv_state };
            defer freeAll(&theirs);
            for (theirs, 0..) |x, i| try expectSame(name, i, t.rig.output(i), x, 0, s);
        },
    }
}

fn recurDrift(t: *Recur, name: []const u8, shape: Shape) !void {
    try std.testing.expect(try t.prepare(shape, .random, 7));
    try runAllRecur(t, shape);
    const s = t.rig.stream();
    const T: c_int = @intCast(shape.rows);
    const g = t.args;
    if (!t.capture and t.route != .stock_vec) {
        const route: gk.GdnRoute = switch (t.route) {
            .stock => .stock,
            .blocked => .blocked,
            .pipelined => .pipelined,
            .stock_vec => unreachable,
        };
        const r = try gk.gdnRunYState(route, 32, g[0], g[1], g[2], g[3], g[4], g[5], 1, T, HK, HV, DK, DV, s);
        defer freeAll(&.{ r.y, r.state });
        try expectSame(name, 0, t.rig.output(0), r.y, 0, s);
        try expectSame(name, 1, t.rig.output(1), r.state, 0, s);
        return;
    }
    // The capture and per-channel kernels have no launch helper outside the forward: the shipped kernel objects
    // with the forward's config.
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, T, HV, DV }, 4, .bfloat16));
    if (t.capture) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ T, 1, HV, DV, DK }, 5, t.st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, HV, DV, DK }, 4, t.st_dt));
    try gk.gdnSetGrid(cfg, .stock, DV, HV, 1);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", t.st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", .bfloat16));
    inline for (.{ .{ "Dk", DK }, .{ "Dv", DV }, .{ "Hk", HK }, .{ "Hv", HV } }) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    const kern = if (t.capture) try gk.getGdnKernelSeq(false) else try gk.getGdnKernel(true);
    const v = mlx.mlx_vector_array_new_data(&g, if (t.capture) 8 else 7);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kern, v, cfg, s));
    for (0..mlx.mlx_vector_array_size(o)) |i| {
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_vector_array_get(&x, o, i));
        const skip = if (t.capture and i == 1) HV * DV * DK * mlx.mlx_array_itemsize(x) else 0;
        try expectSame(name, i, t.rig.output(i), x, skip, s);
    }
}

fn runAllRecur(t: *Recur, shape: Shape) !void {
    var g = t.geometry(shape).?;
    g.unread = 0;
    try t.rig.launch(t.launch, g, t.args[0..if (t.capture) 8 else 7]);
}

test "kernel contract gdn: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const shape = k.spec.served[k.spec.served.len - 1];
        if (std.mem.startsWith(u8, k.spec.name, "gated_delta")) {
            try recurDrift(@ptrCast(@alignCast(k.ctx)), k.spec.name, shape);
        } else try decDrift(@ptrCast(@alignCast(k.ctx)), k.spec.name, shape);
    }
}
