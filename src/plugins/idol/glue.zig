//! Idol's fused elementwise glue for the denoiser block, each kernel one pass in place of an op chain that makes
//! many over [12k, 1024..4096] tensors: q/k RoPE then per-head RMS norm then the head-major layout SDPA reads, and
//! the fc1 bias add with the tanh GELU. Both decline (null) off the GPU and off f16/bf16; the caller keeps the chain.

const std = @import("std");
const kernelSource = @import("../../mlx/kernels.zig").source;
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");

pub const QKV_SOURCE = kernelSource("plugins/idol/kernels/qkv_prep.metal");
pub const GELU_SOURCE = kernelSource("plugins/idol/kernels/bias_gelu.metal");
pub const GATE_SOURCE = kernelSource("plugins/idol/kernels/gate_residual.metal");
pub const NORM_MOD_SOURCE = kernelSource("plugins/idol/kernels/norm_mod.metal");
pub const REPO_NORM_SOURCE = kernelSource("plugins/idol/kernels/repo_norm.metal");
pub const REPO_ANGLES_SOURCE = kernelSource("plugins/idol/kernels/repo_angles.metal");
pub const DEFORM_SOURCE = kernelSource("plugins/idol/kernels/deform_cols.metal");

const HD: c_int = 64;

/// Lets a test run the chain these kernels replace.
pub var disabled_for_test = false;

fn eligible(dt: mlx.mlx_dtype) bool {
    return dt == .float16 or dt == .bfloat16;
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// Runs `kernel`; `outs` receives one handle per declared output.
fn apply(kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, arrs: []const mlx.mlx_array, outs: []mlx.mlx_array, s: mlx.mlx_stream) !void {
    const v = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, cfg, s));
    if (mlx.mlx_vector_array_size(o) != outs.len) return error.MetalKernelBadOutputCount;
    var got: usize = 0;
    errdefer for (outs[0..got]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (outs, 0..) |*slot, i| {
        slot.* = mlx.mlx_array_new();
        got = i + 1;
        try mlx.check(mlx.mlx_vector_array_get(slot, o, i));
    }
}

var qkv_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gelu_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var norm_mod_kernel: ?mlx.mlx_fast_metal_kernel = null;
var repo_norm_kernel: ?mlx.mlx_fast_metal_kernel = null;
var repo_angles_kernel: ?mlx.mlx_fast_metal_kernel = null;
var deform_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// torchvision `deform_conv2d`'s columns for BiRefNet's decoder: `x` `[h*w, C]` (f16 or bf16, C a multiple of 4),
/// `off` f32 `[h*w, 2K]` (each tap's row then column offset), `msk` `[h*w, K]` of x's dtype, K = k*k taps. Out
/// `[h*w, K*C]`, tap-major: each tap's bilinear sample (zero outside the plane) times its modulator.
pub fn deformCols(x: mlx.mlx_array, off: mlx.mlx_array, msk: mlx.mlx_array, k: c_int, h: c_int, w: c_int, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (disabled_for_test or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (!eligible(dt) or mlx.mlx_array_dtype(msk) != dt or mlx.mlx_array_dtype(off) != .float32) return null;
    const hw: usize = @intCast(h * w);
    const taps = k * k;
    const c: c_int = @intCast(mlx.mlx_array_size(x) / @max(hw, 1));
    if (hw == 0 or @rem(c, 4) != 0 or mlx.mlx_array_size(x) != hw * @as(usize, @intCast(c))) return null;
    if (mlx.mlx_array_size(off) != hw * @as(usize, @intCast(2 * taps)) or mlx.mlx_array_size(msk) != hw * @as(usize, @intCast(taps))) return null;
    const x2 = try rowsOf(x, c, s);
    defer _ = mlx.mlx_array_free(x2);
    const off2 = try rowsOf(off, 2 * taps, s);
    defer _ = mlx.mlx_array_free(off2);
    const msk2 = try rowsOf(msk, taps, s);
    defer _ = mlx.mlx_array_free(msk2);
    if (deform_kernel == null) deform_kernel = try makeKernel("msv_idol_deform_cols", &.{ "X", "OFF", "MSK" }, &.{"O"}, DEFORM_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ h * w, taps * c }, 2, dt));
    const cx = @min(@divExact(c, 4), 32);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(c, 4), taps, h * w));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, cx, 1, @divTrunc(256, cx)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KS", k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", h));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "W", w));
    var out: [1]mlx.mlx_array = undefined;
    try apply(deform_kernel.?, cfg, &.{ x2, off2, msk2 }, &out, s);
    return out[0];
}

/// `x` as `[rows, n]` (its last axis `n`): the row-major view a kernel reads.
fn rowsOf(x: mlx.mlx_array, n: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const rows: c_int = @intCast(mlx.mlx_array_size(x) / @as(usize, @intCast(n)));
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_reshape(&o, x, &[_]c_int{ rows, n }, 2, s));
    return o;
}

/// One output shaped like `like`.
fn shapedLike(out: mlx.mlx_array, like: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    defer _ = mlx.mlx_array_free(out);
    const sh = mlx.getShape(like);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_reshape(&o, out, sh.ptr, sh.len, s));
    return o;
}

/// `x + (y + bias) * gate` for `x`, `y` `[.., N]` and `bias`, `gate` N values, all one dtype: each op rounds to that
/// dtype as the chain's add, multiply and add do (bit-equal).
pub fn gateResidual(x: mlx.mlx_array, y: mlx.mlx_array, bias: mlx.mlx_array, gate: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (disabled_for_test or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(y);
    if (!eligible(dt)) return null;
    for ([_]mlx.mlx_array{ x, bias, gate }) |a| if (mlx.mlx_array_dtype(a) != dt) return null;
    const sh = mlx.getShape(y);
    if (sh.len == 0 or mlx.mlx_array_size(x) != mlx.mlx_array_size(y)) return null;
    const n: c_int = sh[sh.len - 1];
    if (@rem(n, 4) != 0 or mlx.mlx_array_size(bias) != @as(usize, @intCast(n)) or mlx.mlx_array_size(gate) != @as(usize, @intCast(n))) return null;
    const x2 = try rowsOf(x, n, s);
    defer _ = mlx.mlx_array_free(x2);
    const y2 = try rowsOf(y, n, s);
    defer _ = mlx.mlx_array_free(y2);
    if (gate_kernel == null) gate_kernel = try makeKernel("msv_idol_gate_residual", &.{ "X", "Y", "B", "G" }, &.{"O"}, GATE_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const rows = mlx.getShape(y2)[0];
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, n }, 2, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(rows * n, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    var out: [1]mlx.mlx_array = undefined;
    try apply(gate_kernel.?, cfg, &.{ x2, y2, bias, gate }, &out, s);
    return try shapedLike(out[0], y, s);
}

/// LayerNorm of `x` `[.., N]` (no affine, statistics in f32) then `* (1 + scale) + shift`, `scale` and `shift` N
/// values of x's dtype; each step rounds to that dtype as the chain's do. N a multiple of 128, at most 4096.
pub fn normMod(x: mlx.mlx_array, scale: mlx.mlx_array, shift: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (disabled_for_test or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (!eligible(dt) or mlx.mlx_array_dtype(scale) != dt or mlx.mlx_array_dtype(shift) != dt) return null;
    const sh = mlx.getShape(x);
    if (sh.len == 0) return null;
    const n: c_int = sh[sh.len - 1];
    if (@rem(n, 128) != 0 or n > 4096 or mlx.mlx_array_size(scale) != @as(usize, @intCast(n)) or mlx.mlx_array_size(shift) != @as(usize, @intCast(n))) return null;
    const x2 = try rowsOf(x, n, s);
    defer _ = mlx.mlx_array_free(x2);
    if (norm_mod_kernel == null) norm_mod_kernel = try makeKernel("msv_idol_norm_mod", &.{ "X", "SC", "SH", "eps" }, &.{"O"}, NORM_MOD_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const rows = mlx.getShape(x2)[0];
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, n }, 2, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * @divExact(n, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @divExact(n, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const eps_a = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_a);
    var out: [1]mlx.mlx_array = undefined;
    try apply(norm_mod_kernel.?, cfg, &.{ x2, scale, shift, eps_a }, &out, s);
    return try shapedLike(out[0], x, s);
}

pub const Qkv = struct { q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array };

/// `y` the fused qkv projection `[.., 3*H*64]` (q | k | v, head-major inside each) before its `bias`; `cos` / `sin`
/// f32 `[.., H, 32]` per token and head; `q_gamma` / `k_gamma` f32 `[H, 64]`. The bias is added (rounded to y's dtype),
/// q and k are rotated (adjacent pairs, f32 math, rounded to y's dtype) and then RMS-normalized per head; all three
/// come out heads-major `[1, H, L, 64]`.
pub fn qkvPrep(y: mlx.mlx_array, bias: mlx.mlx_array, q_gamma: mlx.mlx_array, k_gamma: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, eps: f32, heads: c_int, s: mlx.mlx_stream) !?Qkv {
    if (disabled_for_test or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(y);
    if (!eligible(dt) or mlx.mlx_array_dtype(bias) != dt) return null;
    const width: c_int = 3 * heads * HD;
    const sh = mlx.getShape(y);
    if (sh.len == 0 or sh[sh.len - 1] != width or mlx.mlx_array_size(bias) != @as(usize, @intCast(width))) return null;
    const rows: c_int = @intCast(mlx.mlx_array_size(y) / @as(usize, @intCast(width)));
    const table: usize = @as(usize, @intCast(rows)) * @as(usize, @intCast(heads)) * @as(usize, @intCast(HD / 2));
    for ([_]mlx.mlx_array{ cos, sin }) |t| if (mlx.mlx_array_dtype(t) != .float32 or mlx.mlx_array_size(t) != table) return null;
    for ([_]mlx.mlx_array{ q_gamma, k_gamma }) |g| if (mlx.mlx_array_dtype(g) != .float32 or mlx.mlx_array_size(g) != @as(usize, @intCast(heads * HD))) return null;

    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_reshape(&y2, y, &[_]c_int{ rows, width }, 2, s));
    if (qkv_kernel == null) qkv_kernel = try makeKernel("msv_idol_qkv_prep", &.{ "Y", "B", "COS", "SIN", "QG", "KG", "eps" }, &.{ "OQ", "OK", "OV" }, QKV_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, heads, rows, HD }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 256 * @divTrunc(3 * heads + 7, 8), rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", heads));
    // A 0-dim f32 binds `constant float&`; a [1] array would bind a pointer.
    const eps_a = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_a);
    var outs: [3]mlx.mlx_array = undefined;
    try apply(qkv_kernel.?, cfg, &.{ y2, bias, cos, sin, q_gamma, k_gamma, eps_a }, &outs, s);
    return .{ .q = outs[0], .k = outs[1], .v = outs[2] };
}

/// The RePo layer's LayerNorm of `x` `[.., N]` (f16 or bf16) with f32 `w` and `b`: statistics and output in f32. N a
/// multiple of 128, at most 4096. `split` writes each row as `[hi | hi | lo]` `[.., 3N]` (hi exact in TF32, lo the
/// rest): times `[w_hi; w_lo; w_hi]`, one TF32 GEMM gives the 3xTF32 product.
pub fn repoNorm(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, eps: f32, split: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (disabled_for_test or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (!eligible(dt) or mlx.mlx_array_dtype(w) != .float32 or mlx.mlx_array_dtype(b) != .float32) return null;
    const sh = mlx.getShape(x);
    if (sh.len == 0 or sh.len > 8) return null;
    const n: c_int = sh[sh.len - 1];
    if (@rem(n, 128) != 0 or n > 4096 or mlx.mlx_array_size(w) != @as(usize, @intCast(n)) or mlx.mlx_array_size(b) != @as(usize, @intCast(n))) return null;
    const x2 = try rowsOf(x, n, s);
    defer _ = mlx.mlx_array_free(x2);
    if (repo_norm_kernel == null) repo_norm_kernel = try makeKernel("msv_idol_repo_norm", &.{ "X", "W", "BI", "eps" }, &.{"O"}, REPO_NORM_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const rows = mlx.getShape(x2)[0];
    const width = if (split) 3 * n else n;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, width }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * @divExact(n, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @divExact(n, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "SPLIT", split));
    const eps_a = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_a);
    var out: [1]mlx.mlx_array = undefined;
    try apply(repo_norm_kernel.?, cfg, &.{ x2, w, b, eps_a }, &out, s);
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..sh.len], sh);
    shape[sh.len - 1] = width;
    defer _ = mlx.mlx_array_free(out[0]);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_reshape(&o, out[0], &shape, sh.len, s));
    return o;
}

pub const Tables = struct { cos: mlx.mlx_array, sin: mlx.mlx_array };

/// The rotary tables of a RePo layer from its gate and content projections side by side, `gc` `[.., 2K]` f32: `wf` its
/// final map dense f32 `[K, 3H]` (transposed), `freqs` the 32 per-pair frequencies (f32) and `axis` the axis each reads
/// (i32). cos and sin of `delta * freq * pi` out, f32 `[.., H, 32]`. K at most 1024, H at most 64.
pub fn repoAngles(gc: mlx.mlx_array, wf: mlx.mlx_array, freqs: mlx.mlx_array, axis: mlx.mlx_array, heads: c_int, s: mlx.mlx_stream) !?Tables {
    if (disabled_for_test or !mlx.streamIsGpu(s)) return null;
    for ([_]mlx.mlx_array{ gc, wf, freqs }) |a| if (mlx.mlx_array_dtype(a) != .float32) return null;
    if (mlx.mlx_array_dtype(axis) != .int32 or heads > 64 or heads < 1) return null;
    const sh = mlx.getShape(gc);
    if (sh.len == 0 or @rem(sh[sh.len - 1], 2) != 0) return null;
    const k: c_int = @divExact(sh[sh.len - 1], 2);
    if (k > 1024 or mlx.mlx_array_size(wf) != @as(usize, @intCast(k * 3 * heads)) or mlx.mlx_array_size(freqs) != 32 or mlx.mlx_array_size(axis) != 32) return null;
    const gc2 = try rowsOf(gc, 2 * k, s);
    defer _ = mlx.mlx_array_free(gc2);
    if (repo_angles_kernel == null) repo_angles_kernel = try makeKernel("msv_idol_repo_angles", &.{ "GC", "WF", "FQ", "AXIS" }, &.{ "OC", "OS" }, REPO_ANGLES_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const rows = mlx.getShape(gc2)[0];
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0 .. sh.len - 1], sh[0 .. sh.len - 1]);
    out_shape[sh.len - 1] = heads;
    out_shape[sh.len] = 32;
    for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, sh.len + 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", heads));
    var out: [2]mlx.mlx_array = undefined;
    try apply(repo_angles_kernel.?, cfg, &.{ gc2, wf, freqs, axis }, &out, s);
    return .{ .cos = out[0], .sin = out[1] };
}

/// `gelu_tanh(y + bias)` for `y` `[.., N]` and `bias` `[N]` of one dtype: the sum rounds to that dtype as an add
/// would, the GELU runs in f32 and rounds once (as the reference's fused one does; the chain rounded each of its
/// eight ops).
pub fn biasGelu(y: mlx.mlx_array, bias: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (disabled_for_test or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(y);
    if (!eligible(dt) or mlx.mlx_array_dtype(bias) != dt) return null;
    const sh = mlx.getShape(y);
    if (sh.len == 0) return null;
    const n: c_int = sh[sh.len - 1];
    if (@rem(n, 4) != 0 or mlx.mlx_array_size(bias) != @as(usize, @intCast(n))) return null;
    const rows: c_int = @intCast(mlx.mlx_array_size(y) / @as(usize, @intCast(n)));

    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_reshape(&y2, y, &[_]c_int{ rows, n }, 2, s));
    if (gelu_kernel == null) gelu_kernel = try makeKernel("msv_idol_bias_gelu", &.{ "Y", "B" }, &.{"O"}, GELU_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, n }, 2, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(rows * n, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    var out: [1]mlx.mlx_array = undefined;
    try apply(gelu_kernel.?, cfg, &.{ y2, bias }, &out, s);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_reshape(&o, out[0], sh.ptr, sh.len, s));
    _ = mlx.mlx_array_free(out[0]);
    return o;
}
