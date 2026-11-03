//! The legacy `Transformer`'s methods over `sdk/blocks/mlp/dense.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const lane_qmm = @import("../../../sdk/blocks/qmm/lane_qmm.zig");
const simd_qmm = @import("../../../sdk/blocks/qmm/simd_qmm.zig");
const Transformer = @import("../transformer.zig").Transformer;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const FUSED_ROWS_MAX_M = @import("../../../sdk/blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const fusedResidualNorm = @import("../../../sdk/blocks/norms/residual_norm.zig").fusedResidualNorm;
const swigluJoined = @import("../../../sdk/blocks/mlp/fused_activations.zig").swigluJoined;
const DenseMlpWeights = @import("../../../sdk/blocks/mlp/dense.zig").DenseMlpWeights;

/// A joined row group serves one-row steps, and every width while the
/// exact path runs: serial steps and verify rows then share its shapes.
pub fn rowGroupServes(self: *const Transformer, rows: c_int) bool {
    const exact_max: c_int = if (naxAvailable()) lane_qmm.MAX_ROWS else simd_qmm.MAX_ROWS;
    return rows <= FUSED_ROWS_MAX_M or (self.config.rowExactDecode() and rows <= exact_max);
}

pub fn denseMLP(self: *Transformer, x: mlx.mlx_array, dw: *const DenseMlpWeights) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    if (dw.gu.w.ctx != null and self.rowGroupServes(rows)) {
        const both = try self.qmatmul(x, dw.gu.w, dw.gu.s, dw.gu.b);
        defer _ = mlx.mlx_array_free(both);
        if (self.config.hidden_act == .silu and dw.gu.widths[0] == dw.gu.widths[1]) {
            if (try swigluJoined(self.s, both)) |act| {
                defer _ = mlx.mlx_array_free(act);
                return self.qmatmul(act, dw.down_w, dw.down_s, dw.down_b);
            }
        }
        try dw.gu.part(&gate, both, 0, self.s);
        try dw.gu.part(&up, both, 1, self.s);
    } else {
        gate = try self.qmatmul(x, dw.gate_w, dw.gate_s, dw.gate_b);
        up = try self.qmatmul(x, dw.up_w, dw.up_s, dw.up_b);
    }
    const activated = try self.computeGeglu(gate, up);
    defer _ = mlx.mlx_array_free(activated);
    return self.qmatmul(activated, dw.down_w, dw.down_s, dw.down_b);
}
