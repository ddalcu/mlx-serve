//! The row-exact lanes: while a DFlash drafter is bound (`Forward.row_exact`) and the model's weights fit them, every
//! projection, the MoE router and experts, and attention over a short window read kernels whose bits for a row do not
//! depend on how many rows share the call, so a verify row reproduces serial decode (`Caps.exact_drafted`).
//! copied_from: sdk/blocks/linear/qmatmul.zig (`rowExactQmm`), plugins/nemotron_h/moe.zig (the row-exact arms),
//! sdk/blocks/attention/hybrid_attn.zig (the row-by-row window), sdk/blocks/attention/gated_full.zig (the
//! `lane_attn`/`row_attn` arms), sdk/blocks/mlp/dense.zig (`rowGroupServes`).

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const Linear = @import("linear.zig").Linear;
const rowqmv = @import("../../blocks/qmm/rowqmv.zig");
const lane_qmm = @import("../../blocks/qmm/lane_qmm.zig");
const simd_qmm = @import("../../blocks/qmm/simd_qmm.zig");
const sdpa = @import("../../blocks/attention/sdpa.zig");
const row_attn = @import("../../blocks/attention/row_attn.zig");
const lane_attn = @import("../../blocks/attention/lane_attn.zig");
const FUSED_ROWS_MAX_M = @import("../../blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;

const Array = ops.Array;
const Graph = ops.Graph;

/// `off`, or the kernel family a model's projections take: `rows` = `rowqmv`'s per-row matvecs (Nemotron-H's thin
/// MoE trunk), `mma` = `lane_qmm` (NAX) then `simd_qmm`, whose multi-row MMA reads each weight once (dense Qwen3.8).
pub const Lane = enum(u8) { off, rows, mma };

/// The widest window the row kernels and the row-by-row attention take.
pub const MAX_ROWS = rowqmv.MAX_ROWS;

/// A projection the row kernels cover: dense, or affine in `rowqmv`'s geometry (the legacy `firstRowExactMiss`).
pub fn fits(l: *const Linear) bool {
    if (l.s.ctx == null) return true;
    return l.mode == .affine and rowqmv.fits(l.w, l.s, l.b, l.bits, l.group_size);
}

/// A joined projection group serves `rows`: one-row steps always, every width the lane's matmul takes while it runs
/// (serial steps and verify rows then share the group's shapes).
pub fn groupServes(lane: Lane, rows: c_int) bool {
    const exact_max: c_int = if (naxAvailable()) lane_qmm.MAX_ROWS else simd_qmm.MAX_ROWS;
    return rows <= FUSED_ROWS_MAX_M or (lane != .off and rows <= exact_max);
}

/// `x @ l` on the lane's kernel where it serves, else the projection as always.
pub fn linear(lane: Lane, g: *Graph, l: *const Linear, x: Array) !Array {
    if (lane != .off and l.mode == .affine and l.bias.ctx == null) {
        if (try qmm(lane, x, l, g.s)) |y| return g.adopt(y);
    }
    return l.forward(g, x);
}

fn qmm(lane: Lane, x: Array, l: *const Linear, s: ops.Stream) !?mlx.mlx_array {
    if (lane == .mma) {
        // The pick follows the weight and the GPU, never the row count: serial and verify rows share a kernel.
        if (naxAvailable()) if (try lane_qmm.qmm(x, l.w, l.s, l.b, l.bits, l.group_size, s)) |y| return y;
        if (try simd_qmm.qmm(x, l.w, l.s, l.b, l.bits, l.group_size, s)) |y| return y;
    }
    return rowqmv.qmv(x, l.w, l.s, l.b, l.bits, l.group_size, s);
}

/// `relu(x @ up)^2` in one row-kernel pass (the bf16 output squared, as mlx-lm); null = project and activate apart.
pub fn reluSqUp(lane: Lane, g: *Graph, up: *const Linear, x: Array) !?Array {
    if (lane == .off) return null;
    return if (try rowqmv.qmvAct(x, up.w, up.s, up.b, up.bits, up.group_size, true, g.s)) |y| try g.adopt(y) else null;
}

/// A dense router's logits from its stored `[experts, hidden]` rows; null = the router's projection.
pub fn router(lane: Lane, g: *Graph, rows: Array, x: Array) !?Array {
    if (lane == .off or rows.ctx == null) return null;
    return if (try rowqmv.router(x, rows, g.s)) |y| try g.adopt(y) else null;
}

/// ReLU² experts `[T*K, D]` over `uint32` expert ids, one row per (token, expert) pair; null = the in-place kernels.
pub fn experts(lane: Lane, g: *Graph, x_flat: Array, inds_u32: Array, k: c_int, fc1: *const Linear, fc2: *const Linear) !?Array {
    if (lane == .off or fc1.mode != .affine or fc1.s.ctx == null or fc2.s.ctx == null) return null;
    const y = (try rowqmv.experts(x_flat, inds_u32, k, .{ fc1.w, fc1.s, fc1.b }, .{ fc2.w, fc2.s, fc2.b }, .{ fc1.bits, fc1.group_size }, .{ fc2.bits, fc2.group_size }, g.s)) orelse return null;
    return try g.adopt(y);
}

/// A window of 2..MAX_ROWS rows attends row by row, each row the one-query call serial decode makes over its own
/// keys (the multi-query kernel sums in another order); null = attend as usual.
pub fn attend(lane: Lane, g: *Graph, q: Array, k: Array, v: Array, scale: f32, rows: c_int) !?Array {
    if (lane == .off or rows <= 1 or rows > MAX_ROWS) return null;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try sdpa.rowByRowSdpa(&out, q, k, v, scale, rows, g.s);
    return try g.adopt(out);
}

/// A solo window of up to `row_attn.MAX_ROWS` rows, a serial step included, through one fixed arithmetic whatever the
/// width (`lane_attn` on NAX, else `row_attn`); a draft tree's rows over their own ancestors' keys. Null = off, or
/// both decline.
pub fn attendFixed(lane: Lane, g: *Graph, q: Array, k: Array, v: Array, scale: f32, rows: c_int, tree: ?row_attn.Tree) !?Array {
    if ((lane == .off and tree == null) or rows > row_attn.MAX_ROWS) return null;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if ((naxAvailable() and try lane_attn.sdpa(&out, q, k, v, scale, tree, g.s)) or try row_attn.sdpa(&out, q, k, v, scale, tree, g.s))
        return try g.adopt(out);
    _ = mlx.mlx_array_free(out);
    return null;
}
