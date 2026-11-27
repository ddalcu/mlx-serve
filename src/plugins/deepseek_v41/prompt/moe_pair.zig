//! The one owner of `moePrefill`'s paired gate/up: a prompt call's routed gate and up over a fused bank as one NAX
//! launch, so the `[n, 2 inter]` products never go to memory. Two arms by the bank's mode:
//! - affine: the clamped SwiGLU, then the routing weight on the rounded activation (`sortedGateUpFusedWeighted`,
//!   main's routing-weight epilogue); no switch.
//! - fp4, either format (`MLX_SERVE_DSV41_MOE_PAIR_FP4`, on): the fused-bank gather and `moe_act` in the epilogue
//!   (`sortedGateUpRouted`). NVFP4 brings ModelOpt's per-expert scales, multiplied in the epilogue; MXFP4 (or the MXFP4
//!   view of NVFP4) has none, so the same kernel runs with that step compiled out (the scales' presence is part of its
//!   key). A reorder: the split path's roundings with the sums in another order, so about 0.2% of the words differ on
//!   NVFP4 (0.0015% on MXFP4), each within one bf16 step of its row's scale (a word the split path gives as zero keeps
//!   a tiny value here, many ulps away), and the float64 l2 is the split path's.

const std = @import("std");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const log = @import("../../../sdk/util/log.zig");
const mlx = @import("../../../mlx/ffi.zig");
const gather_nax = @import("../../../sdk/blocks/moe/gather_qmm_nax.zig");
const verify_qmm_nax = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig");
const dsv41 = @import("../deepseek_v41.zig");
const fp4_gemm = @import("fp4_gemm.zig");

const A = mlx.mlx_array;

/// `moePrefill`'s activation `[n, 1, inter]` of the sorted pairs (`x3` `[L, 1, d]` read through `rows`, experts
/// `sidx`, f32 routing weights `w_sorted`) on a fused gate/up bank; null where it does not serve: no fused bank,
/// `MOE_PAIR_FP4` off, or the kernel declines.
pub fn act(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.Experts, x3: A, rows: A, sidx: A, w_sorted: A) !?A {
    if (e.gu) |*gu| if (gu.qp.mode == .affine) return affine(sc, m, gu, x3, rows, sidx, w_sorted);
    if (!pairOn(lev.get(.MOE_PAIR_FP4))) return null;
    const name: []const u8 = "MOE_PAIR_FP4";
    const L = mlx.getShape(x3)[0];
    const P = mlx.getShape(sidx)[0];
    // Not an fp4 bank (affine split banks, GGUF): not this arm's, so no decline line.
    const gu = e.gu orelse return if (e.gate.qp.mode.isFp4()) decline(name, L, P, "the bank has no fused gate/up") else null;
    if (!gu.qp.mode.isFp4()) return null;
    const scales: gather_nax.ExpertScales = .{ .gate = e.scale[0], .up = e.scale[1] };
    const staged = if (fp4_gemm.on()) try fp4_gemm.gateUp(&gu, x3, rows, sidx, w_sorted, m.cfg.dsv4_swiglu_limit, scales, sc.s) else null;
    const y = staged orelse (try gather_nax.sortedGateUpRouted(x3, rows, gu.w, gu.s, gu.b, sidx, w_sorted, m.cfg.dsv4_swiglu_limit, scales, gu.qp.bits, gu.qp.group_size, true, verify_qmm_nax.verifyQmmNaxAvailable(), sc.s)) orelse return decline(name, L, P, gather_nax.routed_decline);
    if (!engaged_logged.swap(true, .monotonic)) log.info(lev.line(.MOE_PAIR_FP4) ++ " ({s}, rows {d}, pairs {d})\n", .{ @tagName(gu.qp.mode), L, P });
    return try sc.keep(y);
}

/// The affine arm: the fused bank's weighted products with the clamped SwiGLU in one NAX kernel, or null where it does
/// not serve (no scales, a limit that is not a whole number, no NAX).
fn affine(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, gu: *const dsv41.Bank, x3: A, rows: A, sidx: A, w_sorted: A) !?A {
    const lim = m.cfg.dsv4_swiglu_limit;
    if (gu.s.ctx == null or lim < 0 or lim != @floor(lim) or !verify_qmm_nax.verifyQmmNaxAvailable()) return null;
    const sigtab = try @import("../../../sdk/blocks/hc/hc_prefill.zig").sigmoidTable(sc.s);
    const y = (try gather_nax.sortedGateUpFusedWeighted(x3, rows, gu.w, gu.s, gu.b, sidx, sigtab, w_sorted, @intFromFloat(lim), gu.qp.bits, gu.qp.group_size, false, true, sc.s)) orelse return null;
    return try sc.keep(y);
}

// One line a process on the switch's first use and one on its first decline; no per-call output.
var engaged_logged: std.atomic.Value(bool) = .init(false);
var declined_logged: std.atomic.Value(bool) = .init(false);

fn decline(name: []const u8, L: c_int, P: c_int, why: []const u8) ?A {
    if (!declined_logged.swap(true, .monotonic)) log.info("[dsv41] {s} declined (rows {d}, pairs {d}): {s}\n", .{ name, L, P, why });
    return null;
}

/// `MLX_SERVE_DSV41_MOE_PAIR_FP4`: on unless set to `0` (an fp4 bank of either format pairs by default).
fn pairOn(raw: ?[*:0]const u8) bool {
    const v = raw orelse return true;
    return v[0] != '0';
}

test "dsv41 moe pair: MOE_PAIR_FP4 is on unset and off only at 0" {
    try std.testing.expect(pairOn(null));
    try std.testing.expect(pairOn("1"));
    try std.testing.expect(!pairOn("0"));
}
