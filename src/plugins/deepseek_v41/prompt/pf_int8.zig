//! `MLX_SERVE_DSV41_PF_INT8` (default 0), NOT LOSSLESS: a resident wide prompt pass's dense trunk projections on int8
//! tensor ops, w8a8's GEMM (`sdk/blocks/dit/w8a8.zig`). The input and the weight are rotated along K by the same signed
//! 256-point Hadamard transform, one block at a time (exact in real arithmetic; it spreads outliers). Each side gets one
//! f32 scale a row over int8 values, and the int32 sums are scaled once.
//!   Level 3: `wq_b` and `wo_b`.
//!   Level 4: level 3, and the shared expert's gate/up and down and the fused wq_a+wkv.
//! A single-Mac (M5 Ultra 512 GB) feature.
//!
//! One owner decides which linears run int8: `Plan.of`, a pure function of the level, the wide layers and the load's
//! facts. The bill (`residentBytes`, `transientBytes`) and the builder (`trunkLoad`) both call it. The builder makes
//! the plan's weights once at load into the model's `Table` ((layer, slot) entries, freed with the model). A forward
//! call takes int8 only by finding its entry there; a missing entry is the exact path. Nothing makes a weight after
//! load. The plan's layers are the CED encoder's (`cedSplit`): the decoder layers run exact even on a wide call
//! (`CED=0`, every row's logits, a perplexity window). A pair, a tensor split, streamed weights, no NAX or a CPU stream
//! leave the plan inert, with a load line naming the reason. The mxfp8 weights stay for decode and the narrow calls.

const std = @import("std");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const dsv41 = @import("../deepseek_v41.zig");
const w8a8 = @import("../../../sdk/blocks/dit/w8a8.zig");
const Config = @import("../config.zig").Config;

const A = mlx.mlx_array;

/// The switch's value, read by the bill and once by the load (`Table.plan`); call sites read the table.
pub fn level() u32 {
    return env.int(.MLX_SERVE_DSV41_PF_INT8, u32, 0);
}

/// The linears a level can take, one table column each.
pub const Slot = enum(u3) { wq_b, wo_b, q_kv, shared_gu, shared_down };
const N_SLOTS = std.enums.values(Slot).len;

/// What a load knows about where it runs. The bill sees the config and the device only (`factsOf`), so a pair or a
/// tensor split bills the plan's bytes and builds none: an overbill, never an underbill.
pub const Facts = struct {
    streamed: bool = false,
    pair: bool = false,
    tp: bool = false,
    nax: bool = true,
    gpu: bool = true,
};

/// Which linears run int8: the level's slots in layers `[0, layers)`; `layers` 0 (with `inert` naming why, or the
/// level off) builds and bills nothing.
pub const Plan = struct {
    level: u32 = 0,
    layers: usize = 0,
    inert: ?[]const u8 = null,

    /// The one decision. `wide`: the layers whose prompt calls are wide by design, the CED encoder's (`cedSplit`), or
    /// every layer where the config has no split.
    pub fn of(lv: u32, wide: usize, f: Facts) Plan {
        if (lv != 3 and lv != 4) return .{};
        const why: ?[]const u8 = if (f.pair) "single-Mac only (a two-Mac pair runs the exact path)" else if (f.tp) "a tensor split runs the exact path" else if (f.streamed) "streamed weights run the exact path" else if (!f.nax) "it needs NAX (the M5 matrix units)" else if (!f.gpu) "it needs a GPU stream" else null;
        return .{ .level = lv, .layers = if (why == null) wide else 0, .inert = why };
    }

    pub fn on(p: Plan) bool {
        return p.layers > 0;
    }

    pub fn takes(p: Plan, s: Slot) bool {
        if (!p.on()) return false;
        return switch (s) {
            .wq_b, .wo_b => true,
            .q_kv, .shared_gu, .shared_down => p.level == 4,
        };
    }

    /// The plan's int8 weights at the config's shapes: a byte a weight and an f32 scale a row.
    pub fn residentBytes(p: Plan, c: anytype) u64 {
        if (!p.on()) return 0;
        var b: u64 = 0;
        inline for (comptime std.enums.values(Slot)) |s| {
            if (p.takes(s)) {
                const nk = shape(c, s);
                b += nk[0] * (nk[1] + 4);
            }
        }
        return p.layers * b;
    }

    /// The plan's per-row working set in a wide pass of `rows` rows (0 for a call no wider than `span`): each linear it
    /// takes adds its int8 input (K bytes a row and an f32 scale) and its bf16 output (2 N bytes a row) beside the exact
    /// path's arrays, which the pass's reserve does not hold (measured at 1M: +7.57 GB of MLX peak against the 2.57 GB of
    /// resident weights, at 38,656-row passes).
    pub fn transientBytes(p: Plan, c: anytype, rows: u64, span: u64) u64 {
        if (!p.on() or rows <= span) return 0;
        var per_row: u64 = 0;
        inline for (comptime std.enums.values(Slot)) |s| {
            if (p.takes(s)) {
                const nk = shape(c, s);
                per_row += nk[1] + 4 + 2 * nk[0];
            }
        }
        return rows * per_row;
    }
};

/// (N, K) of slot `s` at the config's shapes.
fn shape(c: anytype, s: Slot) [2]u64 {
    const h: u64 = c.hidden_size;
    const ql: u64 = c.dsv4_q_lora_rank;
    const qn: u64 = @as(u64, c.num_attention_heads) * c.head_dim;
    const on: u64 = @as(u64, c.dsv4_o_groups) * c.dsv4_o_lora_rank;
    const si: u64 = c.moe_intermediate_size;
    return switch (s) {
        .wq_b => .{ qn, ql },
        .wo_b => .{ h, on },
        .q_kv => .{ ql + c.head_dim, h },
        .shared_gu => .{ 2 * si, h },
        .shared_down => .{ h, si },
    };
}

/// The facts the bill can see: the config's streamed bank and the device. A pair or a tensor split is not in them.
pub fn factsOf(c: anytype) Facts {
    const T = @TypeOf(c.*);
    var f: Facts = .{ .nax = device.naxAvailable() };
    if (comptime @hasField(T, "dsv41_streamed")) f.streamed = c.dsv41_streamed;
    if (comptime @hasField(T, "plugin")) if (c.plugin) |b| {
        f.streamed = f.streamed or b.parsed.caps.streamed_weights;
    };
    return f;
}

/// The layers the plan covers for plugin config `c`: the CED encoder's, or every layer without a split.
pub fn wideLayers(c: *const Config) usize {
    return dsv41.cedSplit(c) orelse c.num_hidden_layers;
}

/// The resident bill of the level's int8 weights (`plugin.residentBytes`), from the same plan the load builds.
pub fn residentBytes(c: *const Config) u64 {
    return Plan.of(level(), wideLayers(c), factsOf(c)).residentBytes(c);
}

/// The level's per-row working set in a wide pass (`dsv41.prefillBytes`), from the same plan; 0 where it is inert.
pub fn transientBytes(c: anytype, rows: u64, span: u64) u64 {
    return Plan.of(level(), 1, factsOf(c)).transientBytes(c, rows, span);
}

/// The fewest rows an int8 call takes: a wide pass's (the replay's 128 rows and decode stay on the mxfp8 weight).
pub const TRUNK_ROWS: usize = 1024;

/// Int8 weights alive in this process (every table's; a test's proof that nothing is built after load or left after
/// a model goes).
pub var live: std.atomic.Value(u32) = .init(0);
var trunk_logged: std.atomic.Value(bool) = .init(false);
/// Int8 calls served (a test's proof it engaged).
pub var trunk_calls: std.atomic.Value(u32) = .init(0);
/// The GEMM test's planted bug: the input's rows a row late.
pub var trunk_plant = false;
/// The bill test's planted bug: the builder takes one layer past the plan.
pub var plan_plant = false;

/// A model's int8 weights, `[layer][slot]`, made once at load from its plan (`trunkLoad`); freed with the model.
pub const Table = struct {
    plan: Plan = .{},
    w: []?w8a8.Weight = &.{},

    pub fn get(t: *const Table, li: usize, s: Slot) ?*const w8a8.Weight {
        const i = li * N_SLOTS + @intFromEnum(s);
        if (i >= t.w.len) return null;
        return if (t.w[i]) |*w| w else null;
    }

    pub fn count(t: *const Table) usize {
        var n: usize = 0;
        for (t.w) |w| n += @intFromBool(w != null);
        return n;
    }

    pub fn bytes(t: *const Table) u64 {
        var b: u64 = 0;
        for (t.w) |w| if (w) |x| {
            b += @as(u64, @intCast(x.n)) * (@as(u64, @intCast(x.k)) + 4);
        };
        return b;
    }

    pub fn deinit(t: *Table, gpa: std.mem.Allocator) void {
        for (t.w) |*w| if (w.*) |*x| {
            x.deinit();
            _ = live.fetchSub(1, .monotonic);
        };
        if (t.w.len > 0) gpa.free(t.w);
        t.* = .{};
    }
};

/// The load's facts for model `m`.
fn loadFacts(m: *const dsv41.Dsv41Model) Facts {
    return .{
        .streamed = m.cfg.dsv41_streamed or m.streamed_experts or m.streamed != null,
        .pair = m.pair != null,
        .tp = m.tp != null,
        .nax = device.naxAvailable(),
        .gpu = mlx.streamIsGpu(m.s),
    };
}

/// The linear of slot `s` in layer `ly`, where the layer has it.
fn slotLin(ly: *const dsv41.Layer, s: Slot) ?*const dsv41.Lin {
    return switch (s) {
        .wq_b => &ly.wq_b,
        .wo_b => &ly.wo_b,
        .q_kv => if (ly.q_kv) |*v| v else null,
        .shared_gu => if (ly.shared_gu) |*v| v else null,
        .shared_down => &ly.shared[2],
    };
}

/// Whether linear `l` can run int8: an mxfp8 weight with no biases, resident.
fn serves(l: *const dsv41.Lin) bool {
    return l.ext == null and l.s.ctx != null and l.b.ctx == null and l.qp.mode == .mxfp8;
}

/// The int8 weight of linear `l` from its exact bf16 values (null where it cannot run int8). The caller owns it.
pub fn weightOf(sc: *dsv41.Scope, l: *const dsv41.Lin) !?w8a8.Weight {
    if (!serves(l)) return null;
    var wt = try w8a8.Weight.fromDense(try l.dense(sc, .bfloat16), sc.s);
    errdefer wt.deinit();
    try dsv41.evalAll(&.{ wt.wq, wt.ws });
    return wt;
}

/// At load: the plan's int8 weights into `m.trunk8`, so no prompt makes one. Fails where it built more than the bill
/// holds (`residentBytes`), which only a planted builder can.
pub fn trunkLoad(gpa: std.mem.Allocator, m: *dsv41.Dsv41Model) !void {
    const plan = Plan.of(level(), wideLayers(&m.cfg), loadFacts(m));
    if (plan.level == 0) return;
    if (plan.inert) |why| {
        m.trunk8 = .{ .plan = plan };
        return log.info("[dsv41] MLX_SERVE_DSV41_PF_INT8={d} inert: {s}\n", .{ plan.level, why });
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    const t0 = std.Io.Timestamp.now(io, .boot);
    const layers = @min(plan.layers + @intFromBool(plan_plant), m.n_layers);
    var t: Table = .{ .plan = plan, .w = try gpa.alloc(?w8a8.Weight, layers * N_SLOTS) };
    @memset(t.w, null);
    errdefer t.deinit(gpa);
    for (m.layers[0..layers], 0..) |*ly, li| inline for (comptime std.enums.values(Slot)) |s| {
        if (plan.takes(s)) if (slotLin(ly, s)) |l| {
            var sc = dsv41.Scope.init(gpa, m.s);
            defer sc.deinit();
            if (try weightOf(&sc, l)) |w| {
                t.w[li * N_SLOTS + @intFromEnum(s)] = w;
                _ = live.fetchAdd(1, .monotonic);
            }
        };
    };
    const billed = residentBytes(&m.cfg);
    if (t.bytes() > billed) {
        log.err("[dsv41] MLX_SERVE_DSV41_PF_INT8={d}: built {d} bytes of int8 weights, past the {d} the bill holds\n", .{ plan.level, t.bytes(), billed });
        return error.Dsv41Int8OverBill;
    }
    const ms = @as(f64, @floatFromInt(t0.untilNow(io, .boot).nanoseconds)) / 1e6;
    log.info("[dsv41] MLX_SERVE_DSV41_PF_INT8={d}: {d} int8 trunk weights made at load ({d} wide layers, {d:.0} ms){s}\n", .{ plan.level, t.count(), plan.layers, ms, if (dsv41.cedEnabled()) "" else "; CED off: the decoder layers' wide calls run exact (not measured)" });
    m.trunk8 = t;
}

/// `ly`'s index in `m.layers` when it is a trunk layer (the DSpark stage layers and a test's loose layer are not).
fn layerIndex(m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer) ?usize {
    const base = @intFromPtr(m.layers.ptr);
    const p = @intFromPtr(ly);
    if (p < base or (p - base) % @sizeOf(dsv41.Layer) != 0) return null;
    const li = (p - base) / @sizeOf(dsv41.Layer);
    return if (li < m.n_layers) li else null;
}

/// Slot `s` of layer `ly` applied to `x` `[L, K]` on its int8 weight where the model's table holds one and the call
/// is a wide pass's; null (the caller's exact path) elsewhere. `name` for the engagement line.
pub fn trunk(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, s: Slot, x: A, name: []const u8) !?A {
    if (m.trunk8.w.len == 0) return null;
    const li = layerIndex(m, ly) orelse return null;
    const w = m.trunk8.get(li, s) orelse return null;
    return apply(sc, m.trunk8.plan.level, w, x, m.act, name);
}

/// Weight `w` applied to `x` `[L, K]` on w8a8's int8 GEMM where the call is wide (`TRUNK_ROWS` rows or more, a bf16
/// input whose K the rotation tiles and the weight's); null elsewhere. The tests' entry.
pub fn apply(sc: *dsv41.Scope, lv: u32, w: *const w8a8.Weight, x: A, out_dt: mlx.mlx_dtype, name: []const u8) !?A {
    const sh = mlx.getShape(x);
    if (sh.len != 2 or @as(usize, @intCast(sh[0])) < TRUNK_ROWS or @rem(sh[1], 256) != 0 or mlx.mlx_array_dtype(x) != .bfloat16 or w.k != sh[1]) return null;
    const L: usize = @intCast(sh[0]);
    const xin = if (!trunk_plant) x else try sc.concat(&.{ try sc.slice(x, 0, 1, L), try sc.slice(x, 0, 0, 1) }, 0);
    var x8 = try w8a8.quantize(xin, sc.s);
    defer x8.deinit();
    const y = try sc.keep(try w8a8.gemm(x8, w, out_dt, false, sc.s));
    if (!trunk_logged.swap(true, .monotonic)) log.info("[dsv41] MLX_SERVE_DSV41_PF_INT8={d} engaged (not lossless): trunk {s} on int8 (w8a8: rotated, a scale a row), first {s} over {d} rows\n", .{ lv, if (lv == 4) @as([]const u8, "wq_b, wo_b, the shared expert and wq_a+wkv") else "wq_b and wo_b", name, L });
    _ = trunk_calls.fetchAdd(1, .monotonic);
    return y;
}

// ── tests (pure: no GPU) ──

test "dsv41 pf int8 plan: one decision for the bill and the builder; levels 3 and 4 take their slots in the wide layers; a pair (single-Mac only), a tensor split, streamed weights, no NAX and a CPU stream are inert and bill nothing" {
    const t = std.testing;
    const C = struct {
        hidden_size: u32 = 5120,
        dsv4_q_lora_rank: u32 = 1280,
        num_attention_heads: u32 = 64,
        head_dim: u32 = 512,
        dsv4_o_groups: u32 = 8,
        dsv4_o_lora_rank: u32 = 1024,
        moe_intermediate_size: u32 = 2304,
    };
    const c = C{};
    try t.expect(!Plan.of(0, 20, .{}).on());
    try t.expect(!Plan.of(2, 20, .{}).on());
    try t.expect(!Plan.of(5, 20, .{}).on());
    const p3 = Plan.of(3, 20, .{});
    const p4 = Plan.of(4, 20, .{});
    try t.expect(p3.takes(.wq_b) and p3.takes(.wo_b) and !p3.takes(.q_kv) and !p3.takes(.shared_gu) and !p3.takes(.shared_down));
    try t.expect(p4.takes(.q_kv) and p4.takes(.shared_gu) and p4.takes(.shared_down));
    // The release's 20 encoder layers: 1.68 GB and 2.57 GB; 85,256 and 120,852 bytes a row.
    try t.expectEqual(@as(u64, 1_680_752_640), p3.residentBytes(&c));
    try t.expectEqual(@as(u64, 2_572_963_840), p4.residentBytes(&c));
    try t.expectEqual(@as(u64, 38_656 * 85_256), p3.transientBytes(&c, 38_656, 256));
    try t.expectEqual(@as(u64, 38_656 * 120_852), p4.transientBytes(&c, 38_656, 256));
    try t.expectEqual(@as(u64, 0), p4.transientBytes(&c, 256, 256));
    for ([_]Facts{ .{ .pair = true }, .{ .tp = true }, .{ .streamed = true }, .{ .nax = false }, .{ .gpu = false } }) |f| {
        const p = Plan.of(4, 20, f);
        try t.expect(!p.on() and p.inert != null);
        try t.expectEqual(@as(u64, 0), p.residentBytes(&c));
        try t.expectEqual(@as(u64, 0), p.transientBytes(&c, 38_656, 256));
    }
    try t.expect(std.mem.indexOf(u8, Plan.of(4, 20, .{ .pair = true }).inert.?, "single-Mac only") != null);
}
