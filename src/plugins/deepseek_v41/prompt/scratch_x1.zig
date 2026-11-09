//! A wide prompt layer's streams after its attention (`x1`, [rows, 4, 5120] f32, 1.34 GB at 16K rows) written into one
//! per-prompt region (scratch.zig), not a fresh MLX allocation each layer.
//!
//! The writer is glm5_next's hc expand writing its words into the region (kernels/hc_expand_into.metal: the same loads,
//! fmas and stores) behind the region's last reader (the layer before's MoE output, bound and unread); the norms and the
//! MoE tail read the region's published alias (`Region.publish`). Let go at the decode phase change and with the model.
const kernelSource = @import("../../../mlx/kernels.zig").source;

const mlx = @import("../../../mlx/ffi.zig");
const scratch = @import("scratch.zig");

const A = mlx.mlx_array;

/// The region.
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    x1: scratch.Region = .{},
};
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_scratch_x1", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

/// The decode phase change (or the model's end): the region's buffer let go.
pub fn releaseAll() void {
    ps().x1.free();
}

/// The writer (kernels/hc_expand_into.metal): glm5_next's expand of the streams `res` [1, L, HC, D] (T) by `x`
/// [1, L, D] (T: glm5's caller casts it first) with each row's `post` [L, HC] and `comb` [L, HC, HC] (f32), its words
/// written into `dst` (a region at [L, HC, D] of T) instead of a fresh array, behind `after` (`Region.after`, bound
/// and unread); returns the carrier (a new handle).
pub const expand_into_source = kernelSource("plugins/deepseek_v41/kernels/hc_expand_into.metal");
var expand_into_kernel: ?mlx.mlx_fast_metal_kernel = null;
pub fn expandInto(s: mlx.mlx_stream, x: A, res: A, post: A, comb: A, dst: A, after_: A) !A {
    const kern = expand_into_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "x", "res", "post", "comb", "dst", "after" };
        const outs = [_][*:0]const u8{"carrier"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("dsv41_hc_expand_into", iv, ov, expand_into_source, "", true, false);
        if (k.ctx == null) return error.ScratchKernel;
        expand_into_kernel = k;
        break :blk k;
    };
    const sh = mlx.getShape(res);
    if (sh.len != 4) return error.ScratchShape;
    const dt = mlx.mlx_array_dtype(res);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{1}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(sh[3], 4), sh[0] * sh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(256, @divTrunc(sh[3], 4)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", sh[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    const inputs = [_]A{ x, res, post, comb, dst, after_ };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

