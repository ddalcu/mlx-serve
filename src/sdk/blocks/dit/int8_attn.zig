//! Int8 attention for media DiTs on NAX GPUs (default on, `--no-w8a8`, LOSSY), SageAttention's
//! recipe: Q int8 per token, K smoothed (minus its per-channel mean, which no
//! softmax row can see) and int8 per 32-key block, P int8 against each block's
//! row max, V int8 per channel. Both products run on the int8 tensor unit with
//! the online softmax in registers; ~1.3-1.5x MLX's bf16 sdpa at 16k tokens.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");

pub const HEADER = kernelSource("kernels/common/mpp_tensor_ops.metal");

/// Keys per block: the K scale granularity and the attention kernel's step.
const BK: c_int = 32;
/// Query rows per simdgroup and simdgroups per threadgroup.
const SM: c_int = 16;
const NSG: c_int = 4;

/// One simdgroup per ROWS-row block of a [B, H, L, D] view, SG blocks per threadgroup: int8 codes,
/// row-contiguous, and one fp32 scale (absmax / 127, at least `eps` / 127) per block, less `mean`
/// [B, H, D] when given; DIVIDE rounds x / scale, else x * (1 / scale). Each lane owns D / 32
/// adjacent columns as float4s and stores them as char4. CONTIG (MLX made x row-contiguous, < 2^32
/// elements) loads them as vectors with 32-bit offsets, which per-row Q prep needs to stay at the
/// bandwidth bound; otherwise scalar reads through x's strides.
pub const QUANT_SOURCE = kernelSource("kernels/dit/int8_attn_quant.metal");

/// One simdgroup per (kv head, row) of v [1, KV, L, 128], over its per-channel |max| `vm`.
pub const VPREP_SOURCE = kernelSource("kernels/dit/int8_attn_vprep.metal");

/// One simdgroup per 16 query rows of one head. Element i of an SM x N tensor-op
/// destination sits at column cb + (i&3) + 16*((i>>3)&(N/16-1)) and row
/// rb + 8*slot, slot = ((i>>2)&1) + 2*(i >> (3 + log2(N/16))); S's destination
/// and P's left operand share that layout, so P never leaves registers.
pub const ATTN_SOURCE = kernelSource("kernels/dit/int8_attn.metal");

/// Blocks (one simdgroup each) per threadgroup of `QUANT_SOURCE` when CONTIG; the strided arm
/// keeps one, its long-K prep ran slower in wide threadgroups.
pub const QUANT_SG: c_int = 8;
var quant_kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
var vprep_kernel: ?mlx.mlx_fast_metal_kernel = null;
var attn_kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged_logged = false;

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

const Out = struct { shape: []const c_int, dtype: mlx.mlx_dtype };

/// Launch `kernel` over `threads` x `rows` (threadgroup `tg` x 1) with template `T` = `dt`.
fn launch(kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, outs: []const Out, threads: c_int, rows: c_int, tg: c_int, dt: mlx.mlx_dtype, results: []mlx.mlx_array, s: mlx.mlx_stream) !void {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, threads, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    for (results, 0..) |*r, i| {
        r.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(r, ov, i));
    }
}

/// Grid width for one simdgroup per item, 8 simdgroups per threadgroup.
fn simdGrid(items: c_int) c_int {
    return 256 * @divTrunc(items + 7, 8);
}

/// Int8 codes of x [B, H, L, D] and its scales [B * H * blocks] per `QUANT_SOURCE`; `contig` lets
/// MLX copy a strided x into rows first, else the kernel reads x's strides. The caller frees both.
pub fn quantRows(x: mlx.mlx_array, rows: c_int, mean: ?mlx.mlx_array, eps: f32, divide: bool, contig_ok: bool, s: mlx.mlx_stream) ![2]mlx.mlx_array {
    const contig = contig_ok and mlx.mlx_array_size(x) <= std.math.maxInt(u32);
    const slot = &quant_kernels[@intFromBool(contig)];
    if (slot.* == null) {
        const iv = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{ "x", "mean", "eps" }, 3);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{ "q", "scales" }, 2);
        defer _ = mlx.mlx_vector_string_free(ov);
        const name = if (contig) "msv_int8_quant_rows_contig" else "msv_int8_quant_rows";
        const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, QUANT_SOURCE, "", contig, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        slot.* = k;
    }
    const sh = mlx.getShape(x);
    const blocks = sh[0] * sh[1] * @divTrunc(sh[2] + rows - 1, rows);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, 4, .int8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{blocks}, 1, .float32));
    const sg: c_int = if (contig) QUANT_SG else 1;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divTrunc(blocks + sg - 1, sg) * 32 * sg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * sg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "SG", sg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", mlx.mlx_array_dtype(x)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "HAS_MEAN", mean != null));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "DIVIDE", divide));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "CONTIG", contig));
    const e = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(e);
    const iv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ x, mean orelse x, e }, 3);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, slot.*.?, iv, c, s));
    var out: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&out, 0..) |*r, i| try mlx.check(mlx.mlx_vector_array_get(r, ov, i));
    return out;
}

/// q [1, H, L, 128]: int8 codes and one scale per row.
pub fn prepQ(q: mlx.mlx_array, s: mlx.mlx_stream) ![2]mlx.mlx_array {
    return quantRows(q, 1, null, 1e-20, true, true, s);
}

/// k [1, KV, L, 128] less its per-channel mean: int8 codes and one scale per 32-key block.
pub fn prepK(k: mlx.mlx_array, s: mlx.mlx_stream) ![2]mlx.mlx_array {
    var mu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mu);
    try mlx.check(mlx.mlx_mean_axis(&mu, k, 2, false, s));
    return quantRows(k, BK, mu, 1e-20, true, true, s);
}

/// Softmax(q k^T * scale) v for heads-major q [1, H, Lq, 128] and k, v
/// [1, KV, L, 128] (H a multiple of KV), unmasked, returned token-major
/// [Lq, H*128] in q's dtype; null when the shape is outside the kernel.
pub fn attention(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!@import("w8a8.zig").nax()) return null; // the kernels are written for the NAX matrix unit
    const qs_ = mlx.getShape(q);
    const ks_ = mlx.getShape(k);
    if (qs_.len != 4 or qs_[0] != 1 or qs_[3] != 128 or ks_[3] != 128) return null;
    const h = qs_[1];
    const kv = ks_[1];
    const lq = qs_[2];
    const l = ks_[2];
    if (@rem(h, kv) != 0 or l < BK) return null;
    const dt = mlx.mlx_array_dtype(q);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (vprep_kernel == null) {
        vprep_kernel = try newKernel("msv_i8attn_vprep", &.{ "v", "vm" }, &.{"vi"}, VPREP_SOURCE, "");
        attn_kernel = try newKernel("msv_i8attn", &.{ "qi", "qs", "ki", "ks", "vi", "vm", "scale" }, &.{"o"}, ATTN_SOURCE, HEADER);
    }
    const qp = try prepQ(q, s);
    defer for (qp) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const kp = try prepK(k, s);
    defer for (kp) |a| {
        _ = mlx.mlx_array_free(a);
    };

    var av = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(av);
    try mlx.check(mlx.mlx_abs(&av, v, s));
    var vmx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vmx);
    try mlx.check(mlx.mlx_max_axis(&vmx, av, 2, false, s));
    var vm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vm);
    try mlx.check(mlx.mlx_astype(&vm, vmx, .float32, s));
    var vp: [1]mlx.mlx_array = undefined;
    try launch(vprep_kernel.?, &.{ v, vm }, &.{.{ .shape = &.{ kv, l, 128 }, .dtype = .int8 }}, simdGrid(kv * l), 1, 256, dt, &vp, s);
    defer _ = mlx.mlx_array_free(vp[0]);

    const sc = mlx.mlx_array_new_float(scale * std.math.log2e);
    defer _ = mlx.mlx_array_free(sc);
    var o: [1]mlx.mlx_array = undefined;
    try launch(attn_kernel.?, &.{ qp[0], qp[1], kp[0], kp[1], vp[0], vm, sc }, &.{.{ .shape = &.{ lq, h * 128 }, .dtype = dt }}, @divTrunc(lq + SM * NSG - 1, SM * NSG) * 32 * NSG, h, 32 * NSG, dt, &o, s);
    if (!engaged_logged) {
        engaged_logged = true;
        log.info("[int8-attn] engaged: NAX int8 QK and PV (first H={d} KV={d} Lq={d} L={d})\n", .{ h, kv, lq, l });
    }
    return o[0];
}

/// `attention` for callers that hold sdpa's heads-first layout, any batch and any float dtype: q, k, v
/// are cast to bf16 for the kernel (one batch row at a time) and the result comes back heads-first
/// [B, H, Lq, 128] in q's dtype, like sdpa. Null when the shape is outside the kernel.
pub fn attentionHeads(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: mlx.mlx_stream) !?mlx.mlx_array {
    const sh = mlx.getShape(q);
    if (!@import("w8a8.zig").nax() or sh.len != 4 or sh[3] != 128 or sh[0] < 1 or sh[0] > 8) return null;
    var in: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (in) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&in, [_]mlx.mlx_array{ q, k, v }) |*d, src| try mlx.check(mlx.mlx_astype(d, src, .bfloat16, s));
    const nb: usize = @intCast(sh[0]);
    var rows: [8]mlx.mlx_array = undefined;
    var done: usize = 0;
    defer for (rows[0..done]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    while (done < nb) : (done += 1) {
        const b: c_int = @intCast(done);
        var one: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (one) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (&one, in) |*d, src| {
            const ish = mlx.getShape(src);
            try mlx.check(mlx.mlx_slice(d, src, &[_]c_int{ b, 0, 0, 0 }, 4, &[_]c_int{ b + 1, ish[1], ish[2], ish[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        }
        const o = (try attention(one[0], one[1], one[2], scale, s)) orelse return null;
        defer _ = mlx.mlx_array_free(o);
        rows[done] = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(rows[done]);
        // [Lq, H*128] token-major → [1, H, Lq, 128]
        var tok = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_reshape(&tok, o, &[_]c_int{ 1, sh[2], sh[1], 128 }, 4, s));
        try mlx.check(mlx.mlx_transpose_axes(&rows[done], tok, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    }
    var heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(heads);
    if (nb == 1) {
        try mlx.check(mlx.mlx_array_set(&heads, rows[0]));
    } else {
        const vec = mlx.mlx_vector_array_new_data(&rows, nb);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&heads, vec, 0, s));
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, heads, mlx.mlx_array_dtype(q), s));
    return out;
}

const testing = std.testing;

fn randn(shape: []const c_int, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&x, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
    return x;
}

fn bf16(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&y, x, .bfloat16, s));
    return y;
}

/// Error of `attention` against MLX's sdpa, relative to the reference's RMS.
fn attnError(h: c_int, kv: c_int, lq: c_int, l: c_int, seed: u64, s: mlx.mlx_stream) !f32 {
    const qf = try randn(&.{ 1, h, lq, 128 }, seed, s);
    defer _ = mlx.mlx_array_free(qf);
    const kf = try randn(&.{ 1, kv, l, 128 }, seed + 1, s);
    defer _ = mlx.mlx_array_free(kf);
    // A per-channel offset on every key, the bias smoothing removes.
    const bias = try randn(&.{ 1, kv, 1, 128 }, seed + 2, s);
    defer _ = mlx.mlx_array_free(bias);
    const two = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(two);
    var b2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b2);
    try mlx.check(mlx.mlx_multiply(&b2, bias, two, s));
    var kb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kb);
    try mlx.check(mlx.mlx_add(&kb, kf, b2, s));
    const vf = try randn(&.{ 1, kv, l, 128 }, seed + 3, s);
    defer _ = mlx.mlx_array_free(vf);
    const q = try bf16(qf, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try bf16(kb, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try bf16(vf, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(128.0);
    const y = (try attention(q, k, v, scale, s)).?;
    defer _ = mlx.mlx_array_free(y);
    try testing.expectEqual(lq, mlx.getShape(y)[0]);
    try testing.expectEqual(h * 128, mlx.getShape(y)[1]);
    const none = mlx.mlx_array{ .ctx = null };
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&r, q, k, v, scale, "", none, none, false, s));
    var rt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rt);
    try mlx.check(mlx.mlx_transpose_axes(&rt, r, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_reshape(&ref, rt, &[_]c_int{ lq, h * 128 }, 2, s));
    return @import("w8a8.zig").relRms(y, ref, s);
}

test "int8 attention equals sdpa within quantization error (GQA, ragged length)" {
    if (!@import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try testing.expect(try attnError(8, 2, 333, 333, 21, s) < 0.03);
}

test "int8 attention equals sdpa within quantization error (MHA, whole blocks)" {
    if (!@import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try testing.expect(try attnError(4, 4, 512, 512, 31, s) < 0.03);
}

test "int8 attention equals sdpa within quantization error (fewer queries than keys)" {
    if (!@import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try testing.expect(try attnError(4, 4, 200, 517, 41, s) < 0.03);
}

test "int8 attention heads: a batch of f16 rows equals sdpa row for row, in sdpa's layout and dtype" {
    if (!@import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const parts = [_]mlx.mlx_array{ try randn(&.{ 2, 4, 300, 128 }, 51, s), try randn(&.{ 2, 4, 257, 128 }, 52, s), try randn(&.{ 2, 4, 257, 128 }, 53, s) };
    var h: [3]mlx.mlx_array = undefined;
    for (&h, parts) |*d, p| {
        d.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(d, p, .float16, s));
        _ = mlx.mlx_array_free(p);
    }
    defer for (h) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const scale: f32 = 1.0 / @sqrt(128.0);
    const y = (try attentionHeads(h[0], h[1], h[2], scale, s)).?;
    defer _ = mlx.mlx_array_free(y);
    try testing.expectEqualSlices(c_int, &.{ 2, 4, 300, 128 }, mlx.getShape(y));
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(y));
    const none = mlx.mlx_array{ .ctx = null };
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, h[0], h[1], h[2], scale, "", none, none, false, s));
    try testing.expect(try @import("w8a8.zig").relRms(y, ref, s) < 0.03);
    // outside the kernel (head_dim 64) the caller keeps sdpa
    const q64 = try randn(&.{ 1, 4, 64, 64 }, 54, s);
    defer _ = mlx.mlx_array_free(q64);
    try testing.expect((try attentionHeads(q64, q64, q64, scale, s)) == null);
}

test "int8 attention declines where the matrix unit is absent, so a default-on machine without NAX keeps sdpa" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const w8a8 = @import("w8a8.zig");
    w8a8.nax_override = false;
    defer w8a8.nax_override = null;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const q = try bf16(try randn(&.{ 1, 4, 64, 128 }, 61, s), s);
    defer _ = mlx.mlx_array_free(q);
    const scale: f32 = 1.0 / @sqrt(128.0);
    try testing.expect((try attention(q, q, q, scale, s)) == null);
    try testing.expect((try attentionHeads(q, q, q, scale, s)) == null);
}
