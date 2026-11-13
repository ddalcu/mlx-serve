//! A GLM KDA layer's prompt recurrence and its bounded gate on oMLX's per-core kernel at up to two threadgroups a core
//! (MLX_SERVE_KDA_PROMPT_PERCORE, off by default). The stock route (`sdk/blocks/kda/recurrence.zig`) plans one
//! threadgroup a core of at most 128 value rows, so it declines GLM-5.3-Flash's 64 x 128 value rows below 64 cores;
//! this plan takes two a core there (40 cores: 80 threadgroups of 104 rows), through the stock route's kernel text and
//! launch (`recurAt`). The gate is built in the kernel (its sigmoid on the fast exp) and the delta rule sums in another
//! order than the stock prompt kernel: rounding-class against the stock prompt path, not its bits.
//! copied_from: turboexl3_glm53f src/kda_prompt.zig (`percorePlan`, `percoreServes`, `percoreRecur`: G2's
//! TURBOEXL3_GLM53F_KDA_PERCORE).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const status = @import("../../sdk/util/system.zig");
const kr = @import("../../sdk/blocks/kda/recurrence.zig");

const Array = mlx.mlx_array;

/// Value rows a threadgroup carries at most (eight lanes a row: 1,024 threads), and threadgroups a core at most.
const max_rows: c_int = 128;
const max_per_core: c_int = 2;

/// The fewest threadgroups a core, at most two, whose even split of `h * dv` value rows keeps every threadgroup at
/// `max_rows` or fewer; null past two a core. One a core is the stock route's own plan (`kr.planFor`).
pub fn planFor(h: c_int, dv: c_int, cores: c_int) ?kr.Plan {
    const total = h * dv;
    if (cores < 1 or total < 2) return null;
    const per_core = @divTrunc(total + cores * max_rows - 1, cores * max_rows);
    if (per_core > max_per_core) return null;
    const ntg = @min(cores * per_core, @divTrunc(total, 2));
    const rows = 2 * @divTrunc(@divTrunc(total, 2) + ntg - 1, ntg);
    return if (rows <= max_rows) .{ .ntg = ntg, .rows = rows } else null;
}

/// Whether this GPU launches the kernel at this plan (probed once).
var launchable: ?bool = null;
var engaged_said = false;
/// Launches since the load (the tests read that the route ran).
pub var runs: u64 = 0;

/// Whether the route takes a prompt at `hk` x `dk` key and `hv` x `dv` value heads: heads alike and 128 wide, 16-bit
/// activations, an f32 state, a plan for this GPU's cores and a launch at it this GPU runs.
pub fn serves(s: mlx.mlx_stream, hk: c_int, hv: c_int, dk: c_int, dv: c_int, act: mlx.mlx_dtype, state: mlx.mlx_dtype) bool {
    if (!mlx.streamIsGpu(s) or hk != hv or dk != 128 or dv != 128) return false;
    if ((act != .bfloat16 and act != .float16) or state != .float32) return false;
    const plan = planFor(hv, dv, @intCast(status.gpuCoreCount())) orelse return false;
    if (launchable == null) launchable = kr.probe(s, hv, act, plan) catch false;
    return launchable.?;
}

/// `kr.recur`'s recurrence at this GPU's two-a-core plan: q, k [B, T, H, 128] normed and scaled, v [B, T, H, 128], the
/// gate input `a` [B, T, H * 128], beta [B, T, H], the per-head A_log, the f32 dt_bias [H * 128] and the f32 state
/// [B, H, 128, 128]. Returns y [B, T, H, 128] in q's dtype and the new state. Call only where `serves`.
pub fn recur(s: mlx.mlx_stream, q: Array, k: Array, v: Array, a: Array, beta: Array, a_log: Array, dt_bias: Array, lower_bound: f32, state: Array) !kr.Out {
    const qs = mlx.getShape(q);
    const rows = qs[1];
    const heads = qs[2];
    const plan = planFor(heads, 128, @intCast(status.gpuCoreCount())).?;
    const out = try kr.recurAt(s, q, k, v, a, beta, a_log, dt_bias, lower_bound, state, plan);
    runs += 1;
    if (!engaged_said) {
        engaged_said = true;
        log.info("[kda] prompt percore engaged: S={d} Hv={d} on {d} threadgroups x {d} rows\n", .{ rows, heads, plan.ntg, plan.rows });
    }
    return out;
}

// ── tests ──

const testing = std.testing;

// Bar: GLM-5.3-Flash's 8,192 value rows take two threadgroups a core where one a core passes 128 rows, the stock plan
// where one a core fits, and none past two a core; every plan covers the rows in even ranges of at most 128.
test "glm5_next kda percore: the plan puts 64 heads of 128 on 40 cores as 80 threadgroups of 104 rows" {
    try testing.expectEqual(@as(?kr.Plan, .{ .ntg = 80, .rows = 104 }), planFor(64, 128, 40));
    try testing.expectEqual(@as(?kr.Plan, .{ .ntg = 76, .rows = 108 }), planFor(64, 128, 38));
    try testing.expectEqual(@as(?kr.Plan, .{ .ntg = 64, .rows = 128 }), planFor(64, 128, 32));
    try testing.expectEqual(@as(?kr.Plan, .{ .ntg = 64, .rows = 128 }), planFor(64, 128, 64));
    try testing.expectEqual(@as(?kr.Plan, .{ .ntg = 80, .rows = 104 }), planFor(64, 128, 80));
    try testing.expectEqual(@as(?kr.Plan, null), planFor(64, 128, 31));
    try testing.expectEqual(@as(?kr.Plan, null), planFor(64, 128, 0));
    var cores: c_int = 32;
    while (cores <= 128) : (cores += 1) {
        const p = planFor(64, 128, cores).?;
        try testing.expect(p.ntg * p.rows >= 64 * 128 and @rem(p.rows, 2) == 0 and p.rows <= max_rows);
        try testing.expect(p.ntg <= max_per_core * cores);
        if (kr.planFor(64, 128, cores)) |stock| try testing.expectEqual(stock, p);
    }
}
