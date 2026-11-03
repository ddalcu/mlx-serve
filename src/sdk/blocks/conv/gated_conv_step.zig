//! LFM2's gated short-conv decode step as one Metal dispatch (`fusedGatedConvStep`), bit-identical to the composed
//! multiply -> concatenate -> conv1d -> multiply and the state shift.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const testing = std.testing;

/// LFM2 gated short-conv, one decode token: for channel c of row b,
///   g   = B[c] * x[c]                       (mlx_multiply, rounds to T)
///   acc = sum_i float(window_i[c]) * w[c,i]  (depthwise_conv_1d's own loop:
///         window = [state rows..., g], float accumulate in tap order)
///   y   = C[c] * T(acc)                      (mlx_multiply)
///   state' = [state rows 1.., g]
/// `proj` is [B, 1, 3H] laid out B | C | x; `state` [B, K-1, H]; `w` [H, K, 1].
const GATED_CONV_STEP_SOURCE = kernelSource("kernels/conv/gated_conv_step.metal");

var gated_conv_kernel: ?mlx.mlx_fast_metal_kernel = null;
const GatedConvCfgKey = struct { batch: c_int, hidden: c_int, kernel: c_int, dtype: mlx.mlx_dtype };
var gated_conv_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gated_conv_cfg_key: GatedConvCfgKey = std.mem.zeroes(GatedConvCfgKey);
var gated_conv_engaged: bool = false;

pub const GatedConvStep = struct { gated: mlx.mlx_array, state: mlx.mlx_array };

/// Bit-identical to multiply -> concatenate -> conv1d(groups=H) -> multiply
/// plus the state slice (pinned by the `fused gated conv step` test). Null →
/// caller keeps the composed ops.
pub fn fusedGatedConvStep(s: mlx.mlx_stream, proj: mlx.mlx_array, state: mlx.mlx_array, w: mlx.mlx_array, hidden: c_int, kernel: c_int) !?GatedConvStep {
    if (kernel < 2 or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(proj);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(state) != dt or mlx.mlx_array_dtype(w) != dt) return null;
    const psh = mlx.getShape(proj);
    const ssh = mlx.getShape(state);
    const wsh = mlx.getShape(w);
    if (psh.len != 3 or psh[1] != 1 or psh[2] != 3 * hidden) return null;
    if (ssh.len != 3 or ssh[0] != psh[0] or ssh[1] != kernel - 1 or ssh[2] != hidden) return null;
    if (wsh.len != 3 or wsh[0] != hidden or wsh[1] != kernel or wsh[2] != 1) return null;
    const batch = psh[0];

    const key = GatedConvCfgKey{ .batch = batch, .hidden = hidden, .kernel = kernel, .dtype = dt };
    if (gated_conv_cfg == null or !std.meta.eql(gated_conv_cfg_key, key)) {
        if (gated_conv_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const ysh = [_]c_int{ batch, 1, hidden };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ysh, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, ssh.ptr, ssh.len, dt));
        const tg: c_int = 256;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(hidden + tg - 1, tg) * tg, batch, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "H", hidden));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KS", kernel));
        gated_conv_cfg = config;
        gated_conv_cfg_key = key;
    }
    if (gated_conv_kernel == null) {
        const input_names = [_][*:0]const u8{ "proj", "state", "w" };
        const output_names = [_][*:0]const u8{ "y", "nstate" };
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kernel_h = mlx.mlx_fast_metal_kernel_new("mlxserve_gated_conv_step", in_vec, out_vec, GATED_CONV_STEP_SOURCE, "", true, false);
        if (kernel_h.ctx == null) return error.MetalKernelCompileFailed;
        gated_conv_kernel = kernel_h;
    }
    const inputs_arr = [_]mlx.mlx_array{ proj, state, w };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gated_conv_kernel.?, inputs_vec, gated_conv_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    var ns = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ns);
    try mlx.check(mlx.mlx_vector_array_get(&ns, outputs_vec, 1));
    if (!gated_conv_engaged) {
        gated_conv_engaged = true;
        log.info("[conv] fused gated conv step engaged: hidden={d} kernel={d}\n", .{ hidden, kernel });
    }
    return .{ .gated = y, .state = ns };
}

// ── tests ──

test "fused gated conv step is bit-identical to multiply -> concat -> conv1d -> multiply" {
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0AF);
    const rnd = prng.random();
    const H: c_int = 2048;
    const K: c_int = 3;
    const mk = struct {
        fn f(a: std.mem.Allocator, r: std.Random, st: mlx.mlx_stream, shape: []const c_int, scale: f32) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try a.alloc(f32, n);
            defer a.free(buf);
            for (buf) |*v| v.* = (r.float(f32) - 0.5) * scale;
            const a32 = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(a32);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, a32, .bfloat16, st));
            return out;
        }
    }.f;
    for ([_]f32{ 2.0, 40.0 }) |scale| {
        const proj = try mk(allocator, rnd, s, &.{ 1, 1, 3 * H }, scale);
        defer _ = mlx.mlx_array_free(proj);
        const state = try mk(allocator, rnd, s, &.{ 1, K - 1, H }, scale);
        defer _ = mlx.mlx_array_free(state);
        const w = try mk(allocator, rnd, s, &.{ H, K, 1 }, 1.0);
        defer _ = mlx.mlx_array_free(w);
        // Reference: the composed ops of gatedConv + conv1dWithCache.
        const st3 = [_]c_int{ 1, 1, 1 };
        var bg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bg);
        try mlx.check(mlx.mlx_slice(&bg, proj, &.{ 0, 0, 0 }, 3, &.{ 1, 1, H }, 3, &st3, 3, s));
        var cg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cg);
        try mlx.check(mlx.mlx_slice(&cg, proj, &.{ 0, 0, H }, 3, &.{ 1, 1, 2 * H }, 3, &st3, 3, s));
        var xc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xc);
        try mlx.check(mlx.mlx_slice(&xc, proj, &.{ 0, 0, 2 * H }, 3, &.{ 1, 1, 3 * H }, 3, &st3, 3, s));
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_multiply(&g, bg, xc, s));
        const parts = [_]mlx.mlx_array{ state, g };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var window = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(window);
        try mlx.check(mlx.mlx_concatenate_axis(&window, vec, 1, s));
        var conv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv);
        try mlx.check(mlx.mlx_conv1d(&conv, window, w, 1, 0, 1, H, s));
        var ref_y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_y);
        try mlx.check(mlx.mlx_multiply(&ref_y, cg, conv, s));
        var ref_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_state);
        try mlx.check(mlx.mlx_slice(&ref_state, window, &.{ 0, 1, 0 }, 3, &.{ 1, K, H }, 3, &st3, 3, s));

        const got = (try fusedGatedConvStep(s, proj, state, w, H, K)) orelse return error.FusedGatedConvDeclined;
        defer _ = mlx.mlx_array_free(got.gated);
        defer _ = mlx.mlx_array_free(got.state);
        for ([_][2]mlx.mlx_array{ .{ ref_y, got.gated }, .{ ref_state, got.state } }) |pr| {
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, pr[0], pr[1], true, s));
            try mlx.check(mlx.mlx_array_eval(eq));
            var eq_v: bool = false;
            try mlx.check(mlx.mlx_array_item_bool(&eq_v, eq));
            try testing.expect(eq_v);
        }
    }
}
