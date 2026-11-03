//! The device tables of a `rope_scaling` block (`sdk/config/rope_scaling.zig` reads it): the `mlx_fast_rope`
//! denominators of a reshaped spectrum and YaRN's per-dim factor.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const rope_scaling = @import("../../config/rope_scaling.zig");

/// The denominators `[dims / 2]` f32 of a scaling that reshapes the spectrum (llama3, YaRN); null for one that keeps
/// the base's (caller owns the array).
pub fn freqs(s: rope_scaling.Scaling, dims: u32, base: f64) !?mlx.mlx_array {
    var f64s: [512]f64 = undefined;
    var f32s: [512]f32 = undefined;
    const half = dims / 2;
    if (half > f64s.len) return error.RopeDimsOutOfRange;
    if (!rope_scaling.freqs(s, f64s[0..half], dims, base)) return null;
    for (f64s[0..half], f32s[0..half]) |v, *x| x.* = @floatCast(v);
    return mlx.mlx_array_new_data(&f32s, &[_]c_int{@intCast(half)}, 1, .float32);
}

/// YaRN's factor on the first `dims` of each `head_dim` row, 1 on the rest, in `dt` (caller owns the array).
pub fn mscaleTable(factor: f32, dims: u32, head_dim: u32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var f32s: [1024]f32 = undefined;
    if (head_dim > f32s.len) return error.RopeDimsOutOfRange;
    for (f32s[0..head_dim], 0..) |*m, i| m.* = if (i < dims) factor else 1.0;
    const f = mlx.mlx_array_new_data(&f32s, &[_]c_int{@intCast(head_dim)}, 1, .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The load's one line for a scaling: what the rope serves.
pub fn logScaling(model: []const u8, s: rope_scaling.Scaling) void {
    switch (s) {
        .none, .llama3 => {},
        .unmodeled => log.info("[rope] {s}: a rope scaling this engine does not build; served unscaled\n", .{model}),
        .linear => |f| log.info("[rope] {s}: linear scaling, positions / {d}\n", .{ model, f }),
        .dynamic => |f| log.info("[rope] {s}: dynamic NTK scaling, factor {d}\n", .{ model, f }),
        .yarn => |y| log.info("[rope] {s}: YaRN, factor {d} over {d} positions, mscale {d:.6}\n", .{ model, y.factor, y.original_max_position_embeddings, rope_scaling.mscale(s) }),
    }
}
