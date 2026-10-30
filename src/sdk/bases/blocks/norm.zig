//! copied_from: sdk/blocks/norms/rms.zig (`rmsNorm` without grouped norms, `layerNorm`). The norms a base puts
//! between blocks, at the model's epsilon (`Shared.eps`).

const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");

const Array = ops.Array;
const Ctx = block.Ctx;

/// RMS norm over the last axis. `fusable`: a seam may fold it into its residual add (one dispatch, same bits).
pub const RmsNorm = struct {
    pub const kind: block.Kind = .norm;
    pub const supports_batched_rows = true;
    pub const fusable = true;
    pub const Weights = Array;

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        return c.g.rmsNorm(x, w.*, c.shared.eps);
    }
};

/// Layer norm with a bias (BERT).
pub const LayerNorm = struct {
    pub const kind: block.Kind = .norm;
    pub const supports_batched_rows = true;
    pub const fusable = false;
    pub const Weights = struct { w: Array, b: Array };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        return c.g.layerNorm(x, w.w, w.b, c.shared.eps);
    }
};
