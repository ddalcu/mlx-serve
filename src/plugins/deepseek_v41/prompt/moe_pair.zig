//! A prompt call's routed gate and up over an fp4 bank (MXFP4; NVFP4 with ModelOpt's per-expert scales) as one paired
//! NAX launch with `moe_act` in its epilogue (`gather_qmm_nax.sortedGateUpRouted`): `moePrefill`'s fused-bank gather,
//! its scale multiplies and their join, and `moe_act` in one dispatch (the epilogue in f32, one rounding fewer), so the
//! `[n, 2 inter]` products never go to memory. `MLX_SERVE_DSV41_MOE_PAIR` serves banks without per-expert scales,
//! `MLX_SERVE_DSV41_MOE_PAIR_SCALED` banks with them.

const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const gather_nax = @import("../../../sdk/blocks/moe/gather_qmm_nax.zig");
const verify_qmm_nax = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig");
const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

/// `moePrefill`'s activation `[n, 1, inter]` of the sorted pairs (`x3` `[L, 1, d]` read through `rows`, experts
/// `sidx`, f32 routing weights `w_sorted`) on an fp4 gate/up bank; null where the bank's switch is off or the kernel
/// declines.
pub fn act(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.Experts, x3: A, rows: A, sidx: A, w_sorted: A) !?A {
    const gu = e.gu orelse return null;
    if (!gu.qp.mode.isFp4()) return null;
    const scaled = e.scale[0] != null or e.scale[1] != null;
    if (!(if (scaled) env.on(.MLX_SERVE_DSV41_MOE_PAIR_SCALED) else env.on(.MLX_SERVE_DSV41_MOE_PAIR))) return null;
    const scales: gather_nax.ExpertScales = .{ .gate = e.scale[0], .up = e.scale[1] };
    const y = (try gather_nax.sortedGateUpRouted(x3, rows, gu.w, gu.s, gu.b, sidx, w_sorted, m.cfg.dsv4_swiglu_limit, scales, gu.qp.bits, gu.qp.group_size, true, verify_qmm_nax.verifyQmmNaxAvailable(), sc.s)) orelse return null;
    return try sc.keep(y);
}
