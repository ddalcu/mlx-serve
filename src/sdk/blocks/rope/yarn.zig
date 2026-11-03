//! sdk/blocks/rope/yarn.
//! Owns computeYarnFreqs.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../vision/mrope.zig");
const testing = std.testing;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const constTableAs = @import("../../../mlx/dtype.zig").constTableAs;
const jsonF64 = @import("../../testing/parity.zig").jsonF64;

/// HF `_compute_yarn_parameters` → the per-dim RoPE DENOMINATOR array (`freqs`)
/// that mlx_fast_rope consumes: mlx computes angle = position / freqs[i], so we
/// return 1/inv_freq where inv_freq is the YaRN-corrected inverse frequency.
/// `out.len` must equal the rotary half-dim (= int(head_dim * partial) / 2).
/// Pure f64 math (no MLX) so it unit-tests directly against reference values.
/// The spectrum itself lives in `mrope.Yarn` — shared with the host cos/sin
/// tables, which need inv_freq and not the reciprocal — so the device and host
/// rope paths can never drift apart. `truncate` is the HF flag (laguna's
/// rope_parameters omits it → true).
pub fn computeYarnFreqs(
    out: []f64,
    head_dim: u32,
    partial: f32,
    base: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    orig_max: u32,
    truncate: bool,
) void {
    const spec = mrope.Yarn{
        .theta = base,
        .factor = factor,
        .orig_max = orig_max,
        .rotary_dim = @intFromFloat(@floor(@as(f64, @floatFromInt(head_dim)) * @as(f64, partial))),
        .beta_fast = beta_fast,
        .beta_slow = beta_slow,
        .truncate = truncate,
    };
    spec.invFreq(out);
    for (out) |*v| v.* = 1.0 / v.*;
}
