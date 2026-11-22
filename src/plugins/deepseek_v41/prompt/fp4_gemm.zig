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

/// `gather_nax.sortedGateUpRouted`'s output on the staged schedule, for the caller to keep; null where it declines (the
/// caller runs today's).
pub fn gateUp(gu: *const dsv41.Bank, x3: A, rows: A, sidx: A, w_sorted: A, limit: f32, scales: gather_nax.ExpertScales, s: mlx.mlx_stream) !?A {
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
    const y = (try gather_nax.sortedGatherPromptStaged(act, b.w, b.s, b.b, sidx, b.qp.bits, b.qp.group_size, b.qp.mode.isFp4(), verify_qmm_nax.verifyQmmNaxAvailable(), sc.s)) orelse
        return decline("down", P, "the bank is not fp4, or K or N is not a multiple of 64");
    if (!engaged_logged.swap(true, .monotonic)) {
        const st = gather_nax.stagedPlan(P, mlx.getShape(b.w)[0]);
        log.info("[dsv41] MLX_SERVE_DSV41_PF_FP4_GEMM engaged (pairs {d}, gate+up {s}, down; tile bm={d} wm={d} wn={d})\n", .{ P, if (gate_up_engaged.load(.monotonic)) @as([]const u8, "staged") else "today's", st.bm, st.wm, st.wn });
    }
    _ = launches.fetchAdd(1, .monotonic);
    return try sc.keep(y);
}

/// Staged launches served, gate+up and down each one (a test's proof the switch engaged).
pub var launches: std.atomic.Value(u32) = .init(0);

// One line a process on the first engagement and one on the first decline; no per-call output.
var engaged_logged: std.atomic.Value(bool) = .init(false);
var gate_up_engaged: std.atomic.Value(bool) = .init(false);
var declined_logged: std.atomic.Value(bool) = .init(false);

fn decline(what: []const u8, P: c_int, why: []const u8) ?A {
    if (!declined_logged.swap(true, .monotonic)) log.info("[dsv41] MLX_SERVE_DSV41_PF_FP4_GEMM declined for {s} (pairs {d}): {s}\n", .{ what, P, why });
    return null;
}
