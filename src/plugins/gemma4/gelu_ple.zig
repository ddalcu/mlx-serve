//! Gemma 4's fused PLE tail at decode: `per_layer_projection(gelu(gate) * ple)` in one dispatch over the GELU table
//! and the 4-bit group-64 projection, bit-identical to the composed ops (the `fused gelu-ple projection` test).

const std = @import("std");
const kernelSource = @import("../../mlx/kernels.zig").source;
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const swigluFusedEnabled = @import("../../sdk/blocks/mlp/fused_activations.zig").swigluFusedEnabled;
const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;

/// Gemma PLE tail at decode: `y = W_proj · (gelu(g) * ple)` in one dispatch.
/// The activation is the table lookup + one multiply; the matmul is MLX's own
/// `qmv` (4-bit, group 64, K = 256 = one block, the "remaining" arm) written
/// out op for op — the bf16 partial sums of `load_vector`, the masked-nibble
/// products of `qdot`, `scale * accum + sum * bias`, `simd_sum`, one rounding
/// — so it is bit-identical to `quantized_matmul` (pinned by the
/// `fused gelu-ple projection` test). Threadgroup = MLX's (32, 2): 8 rows.
pub const GELU_PLE_PROJ_SOURCE = kernelSource("plugins/gemma4/kernels/gelu_ple_proj.metal");

var gelu_ple_kernel: ?mlx.mlx_fast_metal_kernel = null;
const GeluPleCfgKey = struct { hid: c_int, pd: c_int, dtype: mlx.mlx_dtype };
var gelu_ple_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gelu_ple_cfg_key: GeluPleCfgKey = std.mem.zeroes(GeluPleCfgKey);
var gelu_ple_engaged: bool = false;

/// Null unless the projection is 4-bit affine group 64 over exactly 256
/// inputs at decode width (the served Gemma 4 geometry); anything else keeps
/// the composed ops.
pub fn fusedGeluPleProj(s: mlx.mlx_stream, g: mlx.mlx_array, ple: mlx.mlx_array, table: mlx.mlx_array, pw: mlx.mlx_array, ps: mlx.mlx_array, pb: mlx.mlx_array, qp: QuantParams) !?mlx.mlx_array {
    if (!swigluFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    if (qp.mode != .affine or qp.bits != 4 or qp.group_size != 64) return null;
    if (pw.ctx == null or ps.ctx == null or pb.ctx == null) return null;
    const dt = mlx.mlx_array_dtype(g);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(ple) != dt or mlx.mlx_array_dtype(ps) != dt or mlx.mlx_array_dtype(pb) != dt or mlx.mlx_array_dtype(table) != dt) return null;
    const gsh = mlx.getShape(g);
    const psh = mlx.getShape(ple);
    const wsh = mlx.getShape(pw);
    const ssh = mlx.getShape(ps);
    if (gsh.len != 3 or psh.len != 3 or gsh[0] != 1 or gsh[1] != 1 or psh[0] != 1 or psh[1] != 1) return null;
    const pd = gsh[2];
    if (pd != 256 or psh[2] != pd) return null;
    if (wsh.len != 2 or ssh.len != 2 or wsh[1] * 8 != pd or ssh[1] * 64 != pd or ssh[0] != wsh[0]) return null;
    const hid = wsh[0];
    if (hid < 8 or @rem(hid, 8) != 0) return null;

    const key = GeluPleCfgKey{ .hid = hid, .pd = pd, .dtype = dt };
    if (gelu_ple_cfg == null or !std.meta.eql(gelu_ple_cfg_key, key)) {
        if (gelu_ple_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const ysh = [_]c_int{ 1, 1, hid };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ysh, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 64 * @divTrunc(hid, 8), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HID", hid));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "PD", pd));
        gelu_ple_cfg = config;
        gelu_ple_cfg_key = key;
    }
    if (gelu_ple_kernel == null) {
        const input_names = [_][*:0]const u8{ "g", "ple", "gtab", "pw", "ps", "pb" };
        const output_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kh = mlx.mlx_fast_metal_kernel_new("mlxserve_gelu_ple_proj", in_vec, out_vec, GELU_PLE_PROJ_SOURCE, "", true, false);
        if (kh.ctx == null) return error.MetalKernelCompileFailed;
        gelu_ple_kernel = kh;
    }
    const inputs_arr = [_]mlx.mlx_array{ g, ple, table, pw, ps, pb };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gelu_ple_kernel.?, inputs_vec, gelu_ple_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gelu_ple_engaged) {
        gelu_ple_engaged = true;
        log.info("[ple] fused gelu*ple + projection engaged: hid={d} pd={d}\n", .{ hid, pd });
    }
    return y;
}
