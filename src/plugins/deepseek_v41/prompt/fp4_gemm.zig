//! A prompt call's routed gate/up (with `moe_act`) and down over an fp4 bank on the SDK's staged NAX gather
//! (`MLX_SERVE_DSV41_PF_FP4_GEMM`, `gather_qmm_nax.sortedGateUpRoutedStaged` / `sortedGatherPromptStaged`): each
//! expert tile reads its next activations and weight words a 64-deep step ahead, with one barrier a step and K and N
//! compiled in. The k16 blocks sum in the seg/db launches' order, so the outputs are their words.

const std = @import("std");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const mlx = @import("../../../mlx/ffi.zig");
const gather_nax = @import("../../../sdk/blocks/moe/gather_qmm_nax.zig");
const verify_qmm_nax = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig");
const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

pub fn on() bool {
    return env.onUnlessZero(.MLX_SERVE_DSV41_PF_FP4_GEMM);
}

/// `MLX_SERVE_DSV41_PF_DOWN_BK32`: the down's tile for a call of `rows` routed rows over `experts`
/// (`gather_nax.stagedDownWide`), or null (the planner's).
fn downTile(rows: c_int, experts: c_int) ?gather_nax.Staged {
    if (!env.onUnlessZero(.MLX_SERVE_DSV41_PF_DOWN_BK32)) return null;
    const st = gather_nax.stagedDownWide(rows, experts) orelse return null;
    if (!bk32_logged.swap(true, .monotonic)) log.info("[dsv41] MLX_SERVE_DSV41_PF_DOWN_BK32 engaged (pairs {d}; down tile bm={d} bn={d} bk={d})\n", .{ rows, st.bm, st.bn, st.bk });
    _ = bk32_launches.fetchAdd(1, .monotonic);
    return st;
}

/// The planted bug of `MLX_SERVE_DSV41_PF_DOWN_BK32`'s test: where its tile serves, each sorted row handed the next
/// token, as `tok_plant`.
pub var bk32_plant = false;
/// Launches `MLX_SERVE_DSV41_PF_DOWN_BK32` served (a test's proof it engaged).
pub var bk32_launches: std.atomic.Value(u32) = .init(0);
var bk32_logged: std.atomic.Value(bool) = .init(false);

/// `gather_nax.sortedGateUpRouted`'s output on the staged schedule, for the caller to keep; null where it declines, or
/// under `MLX_SERVE_DSV41_PF_GU_SEG` where the seg 128 schedule serves the call (`gather_nax.stagedServesGateUp`): the
/// caller runs that paired launch.
pub fn gateUp(gu: *const dsv41.Bank, x3: A, rows: A, sidx: A, w_sorted: A, limit: f32, scales: gather_nax.ExpertScales, s: mlx.mlx_stream) !?A {
    const ws = mlx.getShape(gu.w);
    if (env.onUnlessZero(.MLX_SERVE_DSV41_PF_GU_SEG) and !gather_nax.stagedServesGateUp(mlx.getShape(sidx)[0], ws[0], mlx.getShape(x3)[2], @divTrunc(ws[1], 2))) {
        if (!gu_seg_logged.swap(true, .monotonic)) log.info("[dsv41] MLX_SERVE_DSV41_PF_GU_SEG engaged (pairs {d})\n", .{mlx.getShape(sidx)[0]});
        return null;
    }
    const y = (try gather_nax.sortedGateUpRoutedStaged(x3, rows, gu.w, gu.s, gu.b, sidx, w_sorted, limit, scales, gu.qp.bits, gu.qp.group_size, gu.qp.mode.isFp4(), verify_qmm_nax.verifyQmmNaxAvailable(), s)) orelse
        return decline("gate+up", mlx.getShape(sidx)[0], gather_nax.routed_decline);
    gate_up_engaged.store(true, .monotonic);
    _ = launches.fetchAdd(1, .monotonic);
    return y;
}

/// The down rows `[n, 1, d]` of the sorted pairs' activations on the staged schedule; null where it declines.
pub fn down(sc: *dsv41.Scope, b: *const dsv41.Bank, act: A, sidx: A) !?A {
    const P = mlx.getShape(sidx)[0];
    if (b.s.ctx == null) return decline("down", P, "the bank has no scales");
    const y = (if (downTile(P, mlx.getShape(b.w)[0])) |st| try gather_nax.sortedGatherPromptTile(act, b.w, b.s, b.b, sidx, b.qp.bits, b.qp.group_size, b.qp.mode.isFp4(), verify_qmm_nax.verifyQmmNaxAvailable(), st, sc.s) else try gather_nax.sortedGatherPromptStaged(act, b.w, b.s, b.b, sidx, b.qp.bits, b.qp.group_size, b.qp.mode.isFp4(), verify_qmm_nax.verifyQmmNaxAvailable(), sc.s)) orelse
        return decline("down", P, "the bank is not fp4, or K or N is not a multiple of 64");
    if (!engaged_logged.swap(true, .monotonic)) {
        const st = gather_nax.stagedPlan(P, mlx.getShape(b.w)[0]);
        log.info("[dsv41] MLX_SERVE_DSV41_PF_FP4_GEMM engaged (pairs {d}, gate+up {s}, down; tile bm={d} wm={d} wn={d})\n", .{ P, if (gate_up_engaged.load(.monotonic)) @as([]const u8, "staged") else "seg 128", st.bm, st.wm, st.wn });
    }
    _ = launches.fetchAdd(1, .monotonic);
    return try sc.keep(y);
}

/// `down` with each row at its pair's slot (`MLX_SERVE_DSV41_PF_DOWN_TOKORD`): `[n, 1, d]` in pair (token) order, sorted
/// row p at `rows[p] * k + j` where `flat[rows[p] * k + j]` is its expert; the same words as `down`'s row p. Null where
/// it declines (the caller then runs `down`).
pub fn downTok(sc: *dsv41.Scope, b: *const dsv41.Bank, act: A, sidx: A, rows: A, flat: A, k: usize) !?A {
    const P = mlx.getShape(sidx)[0];
    if (b.s.ctx == null) return null;
    var r = try sc.astype(rows, .uint32);
    const st = downTile(P, mlx.getShape(b.w)[0]);
    // The planted control: each sorted row handed the next token, so its value lands in another token's slots.
    if (tok_plant or (bk32_plant and st != null)) {
        const tokens: u32 = @intCast(@divExact(mlx.getShape(flat)[0], @as(c_int, @intCast(k))));
        r = try sc.op2(mlx.mlx_remainder, try sc.add(r, try sc.u(1)), try sc.u(tokens));
    }
    const y = (try gather_nax.sortedGatherPromptTokTile(act, b.w, b.s, b.b, sidx, r, try sc.astype(flat, .uint32), @intCast(k), b.qp.bits, b.qp.group_size, b.qp.mode.isFp4(), verify_qmm_nax.verifyQmmNaxAvailable(), st, sc.s)) orelse return null;
    if (!tok_logged.swap(true, .monotonic)) log.info("[dsv41] MLX_SERVE_DSV41_PF_DOWN_TOKORD engaged (pairs {d}, {d} a row)\n", .{ P, k });
    _ = launches.fetchAdd(1, .monotonic);
    _ = tok_launches.fetchAdd(1, .monotonic);
    return try sc.keep(y);
}

/// `downTok`'s planted bug (its test): each sorted row handed the next token.
pub var tok_plant = false;

/// `downTok` launches served (a test's proof `MLX_SERVE_DSV41_PF_DOWN_TOKORD` engaged).
pub var tok_launches: std.atomic.Value(u32) = .init(0);
var tok_logged: std.atomic.Value(bool) = .init(false);

/// Staged launches served, gate+up and down each one (a test's proof the switch engaged).
pub var launches: std.atomic.Value(u32) = .init(0);

// One line a process on the first engagement and one on the first decline; no per-call output.
var engaged_logged: std.atomic.Value(bool) = .init(false);
var gate_up_engaged: std.atomic.Value(bool) = .init(false);
var declined_logged: std.atomic.Value(bool) = .init(false);
var gu_seg_logged: std.atomic.Value(bool) = .init(false);

fn decline(what: []const u8, P: c_int, why: []const u8) ?A {
    if (!declined_logged.swap(true, .monotonic)) log.info("[dsv41] MLX_SERVE_DSV41_PF_FP4_GEMM declined for {s} (pairs {d}): {s}\n", .{ what, P, why });
    return null;
}
