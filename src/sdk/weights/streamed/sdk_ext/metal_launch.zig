//! The plugin's one seam for MLX custom Metal kernels (every arch: DeepSeek V4.1's EXL3 texts, MiMo's decode and prompt
//! texts, GLM-5.3's and GLM-5.3-Flash's): mlx-c's `mlx_fast_metal_kernel_new` / `_free` / `_apply`, same signatures and
//! contracts, through the plugin's shim (csrc/mlx_launch_cache.inc). A kernel made here keeps a launch cache: each
//! variant (template arguments, input dtypes and address classes, output dtypes) runs MLX's own apply until MLX has
//! built its library, then launches MLX's cached pipeline directly, without regenerating and rehashing the kernel's
//! source on every call (the pinned MLX's per-launch cost). Same pipeline, same arguments: bit-identical results.
//! Grid, threadgroup and output shapes are read from the config on every launch, as MLX's; a config with an init value
//! always runs MLX's apply.

const mlx = @import("sdk").mlx;

extern "c" fn mlxlc_kernel_new(name: [*c]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*c]const u8, header: [*c]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx.mlx_fast_metal_kernel;
extern "c" fn mlxlc_kernel_free(k: mlx.mlx_fast_metal_kernel) void;
extern "c" fn mlxlc_apply(outputs: *mlx.mlx_vector_array, k: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, config: mlx.mlx_fast_metal_kernel_config, stream: mlx.mlx_stream) c_int;

pub fn mlx_fast_metal_kernel_new(name: [*c]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*c]const u8, header: [*c]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx.mlx_fast_metal_kernel {
    return mlxlc_kernel_new(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
}

/// The cache's eviction, then mlx-c's free.
pub fn mlx_fast_metal_kernel_free(k: mlx.mlx_fast_metal_kernel) void {
    mlxlc_kernel_free(k);
}

pub fn mlx_fast_metal_kernel_apply(outputs: *mlx.mlx_vector_array, k: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, config: mlx.mlx_fast_metal_kernel_config, stream: mlx.mlx_stream) c_int {
    return mlxlc_apply(outputs, k, inputs, config, stream);
}

// ---- CPU tests (no device: graphs are built on an explicit GPU stream and never evaluated) ----

const std = @import("std");
const testing = std.testing;

extern "c" fn mlxlc_test_config(config: mlx.mlx_fast_metal_kernel_config, out: *[9]c_int) void;
extern "c" fn mlxlc_test_bench(k: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, n_in: c_int, ndims: [*]const c_int, shapes: [*]const c_int, dtypes: [*]const mlx.mlx_dtype, iters: c_int, out: *[3]i64) c_int;
extern "c" fn dsv41_alloc_last_error() [*:0]const u8;

test "metal_launch: the shim reads mlx-c's config as mlx-c wrote it (grid, threadgroup, outputs, template args, no init value)" {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ 3, 5 }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{7}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 11, 13, 17));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 19, 23, 29));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "N", 4));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "B", true));
    var got: [9]c_int = undefined;
    mlxlc_test_config(c, &got);
    // No init value (the host's binding has no setter: no plugin config sets one).
    try testing.expectEqual([9]c_int{ 11, 13, 17, 19, 23, 29, 2, 3, 0 }, got);
}

const Case = struct {
    name: [*:0]const u8,
    ins: []const [*:0]const u8,
    outs: []const [*:0]const u8,
    source: [:0]const u8,
    header: [:0]const u8,
    shapes: []const []const c_int,
    dtypes: []const mlx.mlx_dtype,
    out_shapes: []const []const c_int,
    out_dtypes: []const mlx.mlx_dtype,
    grid: [3]c_int,
    tg: [3]c_int,
    tmpl_int: []const struct { [*:0]const u8, c_int } = &.{},
    tmpl_bf16: []const [*:0]const u8 = &.{},
};

fn bench(cs: Case) ![3]i64 {
    const iv = mlx.mlx_vector_string_new_data(cs.ins.ptr, cs.ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(cs.outs.ptr, cs.outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx_fast_metal_kernel_new(cs.name, iv, ov, cs.source.ptr, cs.header.ptr, true, false);
    defer mlx_fast_metal_kernel_free(k);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (cs.out_shapes, cs.out_dtypes) |sh, dt| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, cs.grid[0], cs.grid[1], cs.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, cs.tg[0], cs.tg[1], cs.tg[2]));
    for (cs.tmpl_bf16) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, t, .bfloat16));
    for (cs.tmpl_int) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    var ndims: [8]c_int = undefined;
    var shapes: [32]c_int = undefined;
    var at: usize = 0;
    for (cs.shapes, 0..) |sh, i| {
        ndims[i] = @intCast(sh.len);
        @memcpy(shapes[at .. at + sh.len], sh);
        at += sh.len;
    }
    var out: [3]i64 = undefined;
    if (mlxlc_test_bench(k, c, @intCast(cs.shapes.len), &ndims, &shapes, cs.dtypes.ptr, 2000, &out) != 0) {
        std.debug.print("mlxlc_test_bench: {s}\n", .{dsv41_alloc_last_error()});
        return error.MlxError;
    }
    return out;
}

