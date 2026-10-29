//! sdk/blocks/linear/qmv_rows.
//! Owns MSV_QMV_ROWS_SOURCE, msv_qmv_rows_kernel, MsvQmvRowsCfgKey, msv_qmv_rows_cfgs, msv_qmv_rows_logged, getMsvQmvRowsKernel, and 1 more.

const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const KernelCfgCache = @import("../../../mlx/kernel_cache.zig").KernelCfgCache;
const cachedScalarInt = @import("verify_qmm.zig").cachedScalarInt;

pub const MSV_QMV_ROWS_SOURCE = kernelSource("kernels/linear/qmv_rows.metal");

pub var msv_qmv_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;
const MsvQmvRowsCfgKey = struct { nrows: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
// One slot per head projection shape and row count a batched round cycles through.
pub var msv_qmv_rows_cfgs = KernelCfgCache(MsvQmvRowsCfgKey, 1, 16){};
/// One engagement line per width.
var msv_qmv_rows_logged: [9]bool = @splat(false);

fn getMsvQmvRowsKernel() !mlx.mlx_fast_metal_kernel {
    if (msv_qmv_rows_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qmv_rows",
        in_vec,
        out_vec,
        MSV_QMV_ROWS_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    msv_qmv_rows_kernel = kernel;
    return kernel;
}

pub fn msvQmvRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (bits < 3 or bits > 8 or bits == 7) return null;
    if (sc.ctx == null or bi.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(sc) != xd or mlx.mlx_array_dtype(bi) != xd) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return null;
    const K = xsh[xsh.len - 1];
    var nrows: c_int = 1;
    var i: usize = 0;
    while (i + 1 < xsh.len) : (i += 1) nrows *= xsh[i];
    if (nrows < 2 or nrows > 16) return null;
    // MLX qmv_fast's packs: two per thread, 8 values per 3/5 bytes at 3/5 bits, 4 per 3 at 6.
    const vpt: c_int = switch (bits) {
        3, 5 => 16,
        6 => 8,
        else => @intCast((32 / bits) * 2),
    };
    const block: c_int = vpt * 32;
    if (@rem(K, block) != 0) return null;
    // A thread's values sit inside one group.
    if (group_size < vpt or @rem(@as(c_int, @intCast(group_size)), vpt) != 0 or @rem(block, @as(c_int, @intCast(group_size))) != 0) return null;
    const wsh = mlx.getShape(w);
    if (wsh.len != 2) return null;
    const Nout = wsh[0];
    if (@rem(Nout, 8) != 0) return null;
    if (@as(i64, wsh[1]) * 32 != @as(i64, K) * bits) return null;
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;

    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    if (xsh.len == 2 and xsh[0] == nrows) {
        try mlx.check(mlx.mlx_array_set(&x2, x));
    } else {
        try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ nrows, K }, 2, s));
    }

    const key = MsvQmvRowsCfgKey{ .nrows = nrows, .n = Nout, .k = K, .bits = bits, .gs = group_size, .dtype = xd };
    const config = blk: {
        if (msv_qmv_rows_cfgs.get(key)) |hit| break :blk hit[0];
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const y_shape = [_]c_int{ nrows, Nout };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @divExact(Nout, 4), nrows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        msv_qmv_rows_cfgs.put(key, .{cfg});
        break :blk cfg;
    };

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(Nout);
    const inputs = [_]mlx.mlx_array{ x2, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getMsvQmvRowsKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    mtp_head_row_dispatches += 1;
    if (!msv_qmv_rows_logged[bits]) {
        msv_qmv_rows_logged[bits] = true;
        log.info("[mtp] row-axis projection engaged: rows={d} K={d} N={d} bits={d} gs={d}\n", .{ nrows, K, Nout, bits, group_size });
    }
    var y2 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));
    if (xsh.len == 2) return y2;
    var ysh_buf: [8]c_int = undefined;
    for (xsh[0 .. xsh.len - 1], 0..) |d, di| ysh_buf[di] = d;
    ysh_buf[xsh.len - 1] = Nout;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &ysh_buf, @intCast(xsh.len), s));
    _ = mlx.mlx_array_free(y2);
    return y;
}

pub var mtp_head_row_dispatches: usize = 0;
