//! The qwen4 hyper-connection read at prefill widths on MLX's CUDA backend: the chain's elementwise
//! ops in two launches. `norm`: the deferred stream write, each stream's RMS norm (`rms_norm_small`'s
//! per-thread order and block reduce) and its weight. `mix`: sigmoid(up) times the normed streams,
//! summed over the streams in order in bf16 (`col_reduce_small`) and scaled by bf16(1/hc). Every op
//! is MLX's functor in the order of the compiled closures it replaces, so the outputs are bit-identical.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

/// `rms_norm_small`'s launch for a row: 64 threads, 8 values per load.
const THREADS = 64;
const READS = 8;

const HEADER =
    \\#include "mlx/backend/cuda/device/binary_ops.cuh"
    \\#include "mlx/backend/cuda/device/unary_ops.cuh"
    \\#include <cooperative_groups.h>
    \\#include <cooperative_groups/reduce.h>
    \\template <typename T> struct alignas(16) msv_vec8 { T v[8]; };
    \\
;

// Block per (token, stream) row of H: the write `stream + out * inj`, then rms_norm_small's sum of
// squares (chunk i, read j per thread; warp reduce, then across the two warps) and its scaling.
const NORM_SOURCE =
    \\using namespace mlx::core::cu;
    \\namespace cg = cooperative_groups;
    \\auto block = cg::this_thread_block();
    \\__shared__ float temp[2];
    \\const int row = blockIdx.x;
    \\const int token = row / HC, h = row % HC;
    \\const int tid = threadIdx.x;
    \\using V = msv_vec8<T>;
    \\V xv[NCH];
    \\float normalizer = 0;
    \\#pragma unroll
    \\for (int i = 0; i < NCH; ++i) {
    \\  const long off = (long)row * H + (64 * i + tid) * 8;
    \\  xv[i] = *reinterpret_cast<const V*>(x + off);
    \\  if (WR) {
    \\    const T g = wi[token * HC + h];
    \\    const V ov = *reinterpret_cast<const V*>(wo + (long)token * H + (64 * i + tid) * 8);
    \\    #pragma unroll
    \\    for (int j = 0; j < 8; ++j) {
    \\      xv[i].v[j] = Add{}(xv[i].v[j], Multiply{}(ov.v[j], g));
    \\    }
    \\    *reinterpret_cast<V*>(stream + off) = xv[i];
    \\  }
    \\  #pragma unroll
    \\  for (int j = 0; j < 8; ++j) {
    \\    float t = static_cast<float>(xv[i].v[j]);
    \\    normalizer += t * t;
    \\  }
    \\}
    \\auto warp = cg::tiled_partition<32>(block);
    \\float r = cg::reduce(warp, normalizer, cg::plus<float>{});
    \\if (warp.thread_rank() == 0) temp[warp.meta_group_rank()] = r;
    \\block.sync();
    \\r = warp.thread_rank() < warp.meta_group_size() ? temp[warp.thread_rank()] : float{};
    \\normalizer = cg::reduce(warp, r, cg::plus<float>{});
    \\const uint32_t axis_size = H;
    \\normalizer = rsqrt(normalizer / axis_size + eps[0]);
    \\#pragma unroll
    \\for (int i = 0; i < NCH; ++i) {
    \\  const long off = (long)row * H + (64 * i + tid) * 8;
    \\  const V wv = *reinterpret_cast<const V*>(w + h * H + (64 * i + tid) * 8);
    \\  V nv;
    \\  #pragma unroll
    \\  for (int j = 0; j < 8; ++j) {
    \\    float y = static_cast<float>(xv[i].v[j]) * normalizer;
    \\    nv.v[j] = Multiply{}(static_cast<T>(y), wv.v[j]);
    \\  }
    \\  *reinterpret_cast<V*>(normed + off) = nv;
    \\}
;

// Thread per (token, column): the chain's sigmoid-gated product per stream, col_reduce_small's
// sequential bf16 sum from 0, then mean's multiply by bf16(1/HC).
const MIX_SOURCE =
    \\using namespace mlx::core::cu;
    \\const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    \\if (idx >= (long)up_shape[0] * up_shape[1] * H) return;
    \\const long token = idx / H, col = idx % H;
    \\float acc = 0.f;
    \\#pragma unroll
    \\for (int h = 0; h < HC; ++h) {
    \\  const long off = (token * HC + h) * H + col;
    \\  const T p = Multiply{}(Sigmoid{}(up[off]), normed[off]);
    \\  acc = static_cast<float>(static_cast<T>(acc + static_cast<float>(p)));
    \\}
    \\out[idx] = Multiply{}(static_cast<T>(acc), static_cast<T>(1.0f / HC));
;

var norm_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var mix_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var engaged = false;

/// `MLX_SERVE_HC_PREFILL_CUDA=0` keeps the composed chain.
pub fn enabled() bool {
    const Env = struct {
        var on: ?bool = null;
    };
    if (Env.on == null) {
        const raw = std.c.getenv("MLX_SERVE_HC_PREFILL_CUDA");
        Env.on = raw == null or raw.?[0] != '0';
    }
    return Env.on.?;
}

fn served(hc: c_int, hidden: c_int) bool {
    return mlx.cudaAvailable() and enabled() and hc >= 1 and hc <= 8 and @rem(hidden, THREADS * READS) == 0 and hidden <= THREADS * READS * 8;
}

fn kernel(slot: *?mlx.mlx_fast_cuda_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_cuda_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, source, HEADER, true, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    slot.* = k;
    return k;
}

pub const Pending = struct { out: mlx.mlx_array, inj: mlx.mlx_array };
pub const Norm = struct {
    normalized: mlx.mlx_array,
    stream: mlx.mlx_array,

    pub fn deinit(self: Norm) void {
        _ = mlx.mlx_array_free(self.normalized);
        _ = mlx.mlx_array_free(self.stream);
    }
};

/// The grouped norm of `x` [B, S, hc*hidden] by `w` [hc, hidden] (eps an f32 [1]), after writing
/// `pending` (`x + out * inj`, out [B, S, hidden], inj [B, S, hc, 1]) into the stream. Returns
/// normalized [B, S, hc, hidden] and the stream, or null outside the kernel's envelope.
pub fn norm(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, batch: c_int, seq: c_int, pending: ?Pending) !?Norm {
    const ws = mlx.getShape(w);
    if (ws.len != 2 or !served(ws[0], ws[1])) return null;
    const hc = ws[0];
    const hidden = ws[1];
    const rows = batch * seq;
    if (rows < 1 or mlx.mlx_array_size(x) != @as(usize, @intCast(rows * hc * hidden))) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .bfloat16 or mlx.mlx_array_dtype(eps) != .float32) return null;
    if (pending) |p| {
        if (mlx.mlx_array_dtype(p.out) != .bfloat16 or mlx.mlx_array_dtype(p.inj) != .bfloat16) return null;
        if (mlx.mlx_array_size(p.out) != @as(usize, @intCast(rows * hidden)) or mlx.mlx_array_size(p.inj) != @as(usize, @intCast(rows * hc))) return null;
    }
    const k = try kernel(&norm_kernel, "msv_hc_prefill_norm", &.{ "x", "w", "eps", "wo", "wi" }, &.{ "normed", "stream" }, NORM_SOURCE);
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{ batch, seq, hc, hidden }, 4, .bfloat16));
    const ssh = if (pending != null) [_]c_int{ batch, seq, hc * hidden } else [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &ssh, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, THREADS * rows * hc, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, THREADS, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    inline for (.{ .{ "H", hidden }, .{ "HC", hc }, .{ "NCH", @divExact(hidden, THREADS * READS) }, .{ "WR", @as(c_int, @intFromBool(pending != null)) } }) |a|
        try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, a[0], a[1]));
    const ins = [_]mlx.mlx_array{ x, w, eps, if (pending) |p| p.out else x, if (pending) |p| p.inj else x };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, k, v, c, s));
    var normalized = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_vector_array_get(&normalized, o, 0));
    var stream = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(stream);
    if (pending != null) try mlx.check(mlx.mlx_vector_array_get(&stream, o, 1)) else try mlx.check(mlx.mlx_array_set(&stream, x));
    if (!engaged) {
        engaged = true;
        log.info("[hc-prefill] CUDA hyper-connection read engaged: hc={d} hidden={d} (MLX_SERVE_HC_PREFILL_CUDA=0 restores the chain)\n", .{ hc, hidden });
    }
    return .{ .normalized = normalized, .stream = stream };
}

/// mean over the streams of sigmoid(up) * normalized, both [B, S, hc, hidden] bf16, as [B, S,
/// hidden]; null outside the kernel's envelope.
pub fn mix(s: mlx.mlx_stream, up: mlx.mlx_array, normalized: mlx.mlx_array) !?mlx.mlx_array {
    const sh = mlx.getShape(normalized);
    if (sh.len != 4 or !served(sh[2], sh[3])) return null;
    if (mlx.mlx_array_dtype(up) != .bfloat16 or mlx.mlx_array_dtype(normalized) != .bfloat16) return null;
    if (!std.mem.eql(c_int, mlx.getShape(up), sh)) return null;
    const k = try kernel(&mix_kernel, "msv_hc_prefill_mix", &.{ "up", "normed" }, &.{"out"}, MIX_SOURCE);
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{ sh[0], sh[1], sh[3] }, 3, .bfloat16));
    const n = sh[0] * sh[1] * sh[3];
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, @divTrunc(n + 255, 256) * 256, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "HC", sh[2]));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "H", sh[3]));
    const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ up, normalized }, 2);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, k, v, c, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    return out;
}
