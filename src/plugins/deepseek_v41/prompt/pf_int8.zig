//! `MLX_SERVE_DSV41_PF_INT8` (default 0), NOT LOSSLESS: a resident wide prompt pass's dense trunk projections on int8
//! tensor ops, w8a8's GEMM (`sdk/blocks/dit/w8a8.zig`). The input and the weight are rotated by the same 256-point
//! Hadamard transform (exact in real arithmetic; it spreads outliers), each side gets one int8 scale a row, and the
//! int32 sums are scaled once.
//!   Level 3: `wq_b` and `wo_b`.
//!   Level 4: level 3, and the shared expert's gate/up and down and the fused wq_a+wkv.
//! The int8 weights are made from the mxfp8 weights' exact bf16 values at load, for the wide layers only, and billed
//! (`residentBytes`); the mxfp8 weights stay for decode and the narrow calls (the CED replay's 128 rows). Each call's
//! input goes through w8a8's rotate-and-quantize pass. The outputs, and so the served text, are not the exact
//! path's; the router, attention, the routed experts and every other stage are unchanged.

const std = @import("std");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const dsv41 = @import("../deepseek_v41.zig");
const w8a8 = @import("../../../sdk/blocks/dit/w8a8.zig");

const A = mlx.mlx_array;

pub fn level() u32 {
    return env.int(.MLX_SERVE_DSV41_PF_INT8, u32, 0);
}

/// The bytes the level adds (billed at load when it is on): the wide layers' (the CED encoder's, or every layer's)
/// int8 weights of the linears it takes, a byte a weight and an f32 scale a row.
pub fn residentBytes(c: anytype) u64 {
    const lv = level();
    if (lv != 3 and lv != 4) return 0;
    const wide: u64 = dsv41.cedSplit(c) orelse c.num_hidden_layers;
    const qn: u64 = @as(u64, c.num_attention_heads) * c.head_dim;
    const on: u64 = @as(u64, c.dsv4_o_groups) * c.dsv4_o_lora_rank;
    const h: u64 = c.hidden_size;
    var b: u64 = qn * c.dsv4_q_lora_rank + 4 * qn + h * on + 4 * h;
    if (lv == 4) {
        // the shared gate/up and down, the fused wq_a+wkv
        const si: u64 = c.moe_intermediate_size;
        const qk: u64 = @as(u64, c.dsv4_q_lora_rank) + c.head_dim;
        b += 2 * si * (h + 4) + h * (si + 4) + qk * (h + 4);
    }
    return wide * b;
}

/// The fewest rows a level-3 call takes: a wide pass's (the replay's 128 rows and decode stay on the mxfp8 weight).
pub const TRUNK_ROWS: usize = 1024;

/// Whether level 3 or 4 serves this model: resident, one GPU with NAX.
pub fn trunkOn(m: *const dsv41.Dsv41Model) bool {
    return (level() == 3 or level() == 4) and m.streamed == null and m.tp == null and device.naxAvailable() and mlx.streamIsGpu(m.s);
}

/// The int8 weights made so far, keyed by the mxfp8 weight's buffer.
var trunk_weights: std.AutoHashMapUnmanaged(usize, w8a8.Weight) = .{};
var trunk_logged: std.atomic.Value(bool) = .init(false);
/// Level-3 calls served (a test's proof it engaged).
pub var trunk_calls: std.atomic.Value(u32) = .init(0);
/// Level 3's planted bug (its test): the input's rows a row late.
pub var trunk_plant = false;

/// Drops the int8 weights made so far (a test's, between linears that may reuse a freed buffer's address).
pub fn trunkForget() void {
    var it = trunk_weights.valueIterator();
    while (it.next()) |w| w.deinit();
    trunk_weights.clearRetainingCapacity();
}

fn trunkWeight(sc: *dsv41.Scope, l: *const dsv41.Lin) !*const w8a8.Weight {
    const key = @intFromPtr(l.w.ctx);
    if (trunk_weights.getPtr(key)) |w| return w;
    var wt = try w8a8.Weight.fromDense(try l.dense(sc, .bfloat16), sc.s);
    errdefer wt.deinit();
    try dsv41.evalAll(&.{ wt.wq, wt.ws });
    try trunk_weights.put(std.heap.c_allocator, key, wt);
    return trunk_weights.getPtr(key).?;
}

/// At load: the int8 weights of the wide layers' linears the level takes (`wq_b` and `wo_b`; level 4 also the shared
/// expert and wq_a+wkv) for the wide layers `[0, wide)` (the CED encoder's, or every layer's), so no prompt makes them.
pub fn trunkLoad(gpa: std.mem.Allocator, m: *const dsv41.Dsv41Model, wide: usize) !void {
    if (!trunkOn(m)) return;
    const io = std.Io.Threaded.global_single_threaded.io();
    const t0 = std.Io.Timestamp.now(io, .boot);
    var n: usize = 0;
    const four = level() == 4;
    for (m.layers[0..@min(wide, m.n_layers)]) |*ly| for ([_]?*const dsv41.Lin{ &ly.wq_b, &ly.wo_b, if (four) optPtr(&ly.shared_gu) else null, if (four) &ly.shared[2] else null, if (four) optPtr(&ly.q_kv) else null }) |lo| {
        const l = lo orelse continue;
        if (!trunkServes(l)) continue;
        var sc = dsv41.Scope.init(gpa, m.s);
        defer sc.deinit();
        _ = try trunkWeight(&sc, l);
        n += 1;
    };
    const ms = @as(f64, @floatFromInt(t0.untilNow(io, .boot).nanoseconds)) / 1e6;
    log.info("[dsv41] MLX_SERVE_DSV41_PF_INT8={d}: {d} int8 trunk weights made at load ({d} wide layers, {d:.0} ms)\n", .{ level(), n, wide, ms });
}

fn optPtr(l: *const ?dsv41.Lin) ?*const dsv41.Lin {
    return if (l.*) |*v| v else null;
}

/// Whether level 3 takes linear `l` (an mxfp8 weight with no biases, resident).
fn trunkServes(l: *const dsv41.Lin) bool {
    return l.ext == null and l.s.ctx != null and l.b.ctx == null and l.qp.mode == .mxfp8;
}

/// `l` applied to `x` `[L, K]` on w8a8's int8 GEMM where level 3 serves it (an mxfp8 linear, a wide pass's rows, a
/// bf16 input whose K the rotation tiles); null (the caller's exact path) elsewhere. `name` for the engagement line.
pub fn trunk(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, l: *const dsv41.Lin, x: A, name: []const u8) !?A {
    if (!trunkOn(m)) return null;
    return trunkMatmul(sc, l, x, m.act, name);
}

/// `trunk` for level 4's linears (the shared expert, wq_a+wkv): null below level 4.
pub fn trunk4(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, l: *const dsv41.Lin, x: A, name: []const u8) !?A {
    if (level() != 4 or !trunkOn(m)) return null;
    return trunkMatmul(sc, l, x, m.act, name);
}

/// `trunk` past the model's checks (its test's entry).
pub fn trunkMatmul(sc: *dsv41.Scope, l: *const dsv41.Lin, x: A, out_dt: mlx.mlx_dtype, name: []const u8) !?A {
    if (!trunkServes(l)) return null;
    const sh = mlx.getShape(x);
    if (sh.len != 2 or @as(usize, @intCast(sh[0])) < TRUNK_ROWS or @rem(sh[1], 256) != 0 or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const w = try trunkWeight(sc, l);
    if (w.k != sh[1]) return null;
    const L: usize = @intCast(sh[0]);
    const xin = if (!trunk_plant) x else try sc.concat(&.{ try sc.slice(x, 0, 1, L), try sc.slice(x, 0, 0, 1) }, 0);
    var x8 = try w8a8.quantize(xin, sc.s);
    defer x8.deinit();
    const y = try sc.keep(try w8a8.gemm(x8, w, out_dt, false, sc.s));
    if (!trunk_logged.swap(true, .monotonic)) log.info("[dsv41] MLX_SERVE_DSV41_PF_INT8={d} engaged (not lossless): trunk {s} on int8 (w8a8: rotated, a scale a row), first {s} over {d} rows\n", .{ level(), if (level() == 4) @as([]const u8, "wq_b, wo_b, the shared expert and wq_a+wkv") else "wq_b and wo_b", name, L });
    _ = trunk_calls.fetchAdd(1, .monotonic);
    return y;
}
