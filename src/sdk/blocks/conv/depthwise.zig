//! sdk/blocks/conv/depthwise.
//! Owns depthwiseConvStep, DWCONV_CUDA_SOURCE, dwconv_cuda_kernel, depthwiseConvCuda.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const hostF32 = @import("../../testing/parity.zig").hostF32;
const rmsVsTruth = @import("../../testing/parity.zig").rmsVsTruth;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;

/// Depthwise conv1d over exactly `kernel` positions (one output row) as
/// sum_k window[k] * w[k]. MLX's CUDA grouped conv costs ~10 ms of encode per
/// call at decode width. window [B,K,C], w [C,K,1] -> [B,1,C].
pub fn depthwiseConvStep(window: mlx.mlx_array, w: mlx.mlx_array, cdim: c_int, kernel: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var w_ck = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_ck);
    try mlx.check(mlx.mlx_reshape(&w_ck, w, &[_]c_int{ cdim, kernel }, 2, s));
    var w_kc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_kc);
    try mlx.check(mlx.mlx_transpose(&w_kc, w_ck, s));
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, window, w_kc, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum_axis(&out, prod, 1, true, s));
    return out;
}

const DWCONV_CUDA_SOURCE = kernelSource("kernels/conv/dwconv.cu");

var dwconv_cuda_kernel: ?mlx.mlx_fast_cuda_kernel = null;

/// Depthwise "valid" conv1d over [B, L+K-1, C] on CUDA, one thread per output: MLX's grouped
/// conv runs cuDNN's direct kernel there, milliseconds per prefill chunk. The length is read
/// from the input's shape, so one compiled kernel serves every chunk width.
pub fn depthwiseConvCuda(x: mlx.mlx_array, w: mlx.mlx_array, cdim: c_int, kernel: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (dwconv_cuda_kernel == null) {
        const in_names = [_][*:0]const u8{ "x", "w" };
        const out_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_cuda_kernel_new("msv_depthwise_conv", in_vec, out_vec, DWCONV_CUDA_SOURCE, "", true, 0);
        if (k.ctx == null) return error.CudaKernelCreateFailed;
        dwconv_cuda_kernel = k;
    }
    const xs = mlx.getShape(x);
    const out_len = xs[1] - kernel + 1;
    const out_dtype = mlx.mlx_array_dtype(x);
    const config = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(config, &[_]c_int{ xs[0], out_len, cdim }, 3, out_dtype));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(config, xs[0] * out_len * cdim, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(config, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(config, "OutT", out_dtype));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "C", cdim));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "K", kernel));
    const ins = [_]mlx.mlx_array{ x, w };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&outs, dwconv_cuda_kernel.?, in_vec, config, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return y;
}

/// Test seams (inference thread only): the copy/eval counters let the retention and cadence tests see what the loop did.
pub var compact_conv_state_override: ?bool = null;
pub var compact_conv_state_copy_count: u64 = 0;

pub fn compactConvStateEnabled() bool {
    return compact_conv_state_override orelse true;
}

// ── tests ──

test "depthwiseConvStep equals a grouped conv1d over one window" {
    const s = mlx.gpuStream();
    const B: c_int = 2;
    const K: c_int = 3;
    const C: c_int = 8;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 7));
    var window = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(window);
    try mlx.check(mlx.mlx_random_normal(&window, &[_]c_int{ B, K, C }, 3, .float32, 0, 1, key, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_random_normal(&w, &[_]c_int{ C, K, 1 }, 3, .float32, 0, 1, key, s));
    const got = try depthwiseConvStep(window, w, C, K, s);
    defer _ = mlx.mlx_array_free(got);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_conv1d(&want, window, w, 1, 0, 1, C, s));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, got, want, s));
    var adiff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(adiff);
    try mlx.check(mlx.mlx_abs(&adiff, diff, s));
    var worst = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(worst);
    try mlx.check(mlx.mlx_max(&worst, adiff, false, s));
    var v: f32 = 1;
    try mlx.check(mlx.mlx_array_item_float32(&v, worst));
    try std.testing.expect(v < 1e-5);
}

test "depthwiseConvCuda equals a grouped conv1d over a prefill-length window, no worse against f64 truth" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const B: c_int = 2;
    const K: c_int = 4;
    const C: c_int = 96;
    const L: c_int = 37;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 11));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ B, L + K - 1, C }, 3, .float32, 0, 1, key, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    var wkey = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wkey);
    try mlx.check(mlx.mlx_random_key(&wkey, 12));
    try mlx.check(mlx.mlx_random_normal(&w, &[_]c_int{ C, K, 1 }, 3, .float32, 0, 1, wkey, s));
    const got = try depthwiseConvCuda(x, w, C, K, s);
    defer _ = mlx.mlx_array_free(got);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_conv1d(&want, x, w, 1, 0, 1, C, s));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    try std.testing.expect(try attn256MaxDiff(got, want, s) < 1e-4);

    // No worse than the grouped conv against an f64 truth.
    const xs = try hostF32(x);
    const ws = try hostF32(w);
    const truth = try std.testing.allocator.alloc(f64, @intCast(B * L * C));
    defer std.testing.allocator.free(truth);
    const Lw: usize = @intCast(L + K - 1);
    for (0..@intCast(B)) |b| for (0..@intCast(L)) |l| for (0..@intCast(C)) |c| {
        var acc: f64 = 0;
        for (0..@intCast(K)) |i| acc += @as(f64, xs[(b * Lw + l + i) * @as(usize, @intCast(C)) + c]) * @as(f64, ws[c * @as(usize, @intCast(K)) + i]);
        truth[(b * @as(usize, @intCast(L)) + l) * @as(usize, @intCast(C)) + c] = acc;
    };
    try std.testing.expect(try rmsVsTruth(got, truth) <= try rmsVsTruth(want, truth));
}
