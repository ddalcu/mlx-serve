//! Inkling 2's Metal kernels: relative-logit prefill attention and the depth (block) attention read, ports of
//! mlx-lm's `relative_attention` and `depth_attention` (`kernels/*.metal`). Every wrapper returns null when it cannot
//! serve the call; the caller then runs the composed path (`attention.referenceAttention`, `depth.attendComposed`),
//! which stays the reference. `MLX_SERVE_INKLING2_KERNELS=0` turns both off. Contract: `contract.zig`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernels = @import("../../mlx/kernels.zig");
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");

const Array = ops.Array;
const Graph = ops.Graph;

pub const RELATIVE_ATTENTION_SOURCE = kernels.source("plugins/inkling2/kernels/relative_attention.metal");
pub const DEPTH_ATTENTION_SOURCE = kernels.source("plugins/inkling2/kernels/depth_attention.metal");

/// relative_attention: queries a threadgroup (8 for each of its 4 simdgroups), its threads, and the key block.
pub const BQ = 32;
pub const TG = 128;
pub const BK = 16;

/// Threadgroup memory every Apple GPU family gives a pipeline.
const TG_MEMORY = 32768;

/// relative_attention's threadgroup bytes at head dim `hd`: the half K^T tile `Ks[hd][BK + 8]` and V tile `Vs[BK][hd + 8]`.
pub fn relTgBytes(hd: usize) usize {
    return 2 * (hd * (BK + 8) + BK * (hd + 8));
}

/// depth_attention's threads a row (mlx-lm's `row_threads`).
pub fn rowThreads(d: usize) usize {
    return @min(1024, @max(32, d / 8 / 32 * 32));
}

/// depth_attention's threadgroup bytes: `buf[(K + 1) * Q][TG / 32]` floats.
pub fn depthTgBytes(k: usize, q: usize, tg: usize) usize {
    return 4 * (k + 1) * q * (tg / 32);
}

/// Default on: only `MLX_SERVE_INKLING2_KERNELS=0` runs the composed path.
fn enabled() bool {
    return !env.present(.MLX_SERVE_INKLING2_KERNELS) or env.on(.MLX_SERVE_INKLING2_KERNELS);
}

var rel_kernel: ?mlx.mlx_fast_metal_kernel = null;
var rel_engaged = false;
var depth_kernel: ?mlx.mlx_fast_metal_kernel = null;
var depth_engaged = false;

fn build(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, src: [*:0]const u8, contiguous: bool) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const outputs = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outputs, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, src, "", contiguous, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// One dispatch of `k` on `inputs` with `cfg`; its single output, owned by `g`.
fn apply(g: *Graph, k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const Array) !Array {
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, k, in_vec, cfg, g.s));
    if (mlx.mlx_vector_array_size(out_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, out_vec, 0));
    return g.adopt(y);
}

/// q bf16 [1,H,L,hd]; k,v bf16 [1,Hk,S,hd]; pb bf16 [1,L,H,R]; sinks [1,L,H] (cast to f32 inside);
/// key_start = S - total (<= 0). Returns bf16 [1,L,H,hd] owned by g, or null = use referenceAttention.
pub fn relativeAttention(g: *Graph, q: Array, k: Array, v: Array, pb: Array, sinks: Array, scale: f32, window: u32, key_start: i32) !?Array {
    if (!enabled() or !mlx.streamIsGpu(g.s)) return null;
    for ([_]Array{ q, k, v }) |x| if (mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const ps = mlx.getShape(pb);
    const ss = mlx.getShape(sinks);
    if (qs.len != 4 or ks.len != 4 or ps.len != 4 or ss.len != 3 or !std.mem.eql(c_int, ks, mlx.getShape(v))) return null;
    const hq = qs[1];
    const l = qs[2];
    const hd = qs[3];
    const hkv = ks[1];
    const s = ks[2];
    if (qs[0] != 1 or ks[0] != 1 or l < 2 or hd < 8 or @rem(hd, 8) != 0 or ks[3] != hd or s < l) return null;
    if (hkv < 1 or @rem(hq, hkv) != 0 or relTgBytes(@intCast(hd)) > TG_MEMORY) return null;
    if (ps[0] != 1 or ps[1] != l or ps[2] != hq or ps[3] < 1 or ss[0] != 1 or ss[1] != l or ss[2] != hq) return null;

    // The kernel reads the bias as q's dtype and the sinks as f32.
    const rel = try g.contiguous(if (mlx.mlx_array_dtype(pb) == .bfloat16) pb else try g.astype(pb, .bfloat16));
    const sk = try g.contiguous(if (mlx.mlx_array_dtype(sinks) == .float32) sinks else try g.astype(sinks, .float32));
    const start = try g.adopt(mlx.mlx_array_new_data(&[_]i32{key_start}, &[_]c_int{1}, 1, .int32));
    const sizes = try g.adopt(mlx.mlx_array_new_data(&[_]i32{ l, s }, &[_]c_int{2}, 1, .int32));
    const sc = try g.adopt(mlx.mlx_array_new_float(scale));

    const kh = try build(&rel_kernel, "inkling2_relative_attention", &.{ "q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale" }, RELATIVE_ATTENTION_SOURCE, false);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{ 1, l, hq, hd };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, out_shape.len, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(l + BQ - 1, BQ) * TG, hq, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", hd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HQ", hq));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HKV", hkv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", ps[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WINDOW", @intCast(window)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BQ", BQ));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", TG));
    const y = try apply(g, kh, cfg, &.{ q, k, v, rel, sk, start, sizes, sc });
    if (!rel_engaged) {
        rel_engaged = true;
        log.info("[inkling2] relative_attention engaged\n", .{});
    }
    return y;
}

/// blocks f32 [N,...,D] or null; partial f32 [...,D]; queries f32 [Q,D]. Returns f32 [...,Q,D] or null = attendComposed.
pub fn depthAttention(g: *Graph, blocks: ?Array, partial: Array, queries: Array, scale: f32) !?Array {
    if (!enabled() or !mlx.streamIsGpu(g.s)) return null;
    const b = blocks orelse return null;
    for ([_]Array{ b, partial, queries }) |x| if (mlx.mlx_array_dtype(x) != .float32) return null;
    const ps = mlx.getShape(partial);
    const bs = mlx.getShape(b);
    const qs = mlx.getShape(queries);
    if (ps.len == 0 or ps.len > 6 or bs.len != ps.len + 1 or qs.len != 2 or !std.mem.eql(c_int, bs[1..], ps)) return null;
    const d = ps[ps.len - 1];
    const nk = bs[0];
    const nq = qs[0];
    if (d < 1 or qs[1] != d or nk < 1 or nq < 1) return null;
    const tg = rowThreads(@intCast(d));
    if (depthTgBytes(@intCast(nk), @intCast(nq), tg) > TG_MEMORY) return null;
    const rows: c_int = @intCast(mlx.mlx_array_size(partial) / @as(usize, @intCast(d)));

    const sc = try g.adopt(mlx.mlx_array_new_float(scale));
    const kh = try build(&depth_kernel, "inkling2_depth_attention", &.{ "blocks", "partial", "queries", "scale" }, DEPTH_ATTENTION_SOURCE, true);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var out_shape: [7]c_int = undefined;
    @memcpy(out_shape[0 .. ps.len - 1], ps[0 .. ps.len - 1]);
    out_shape[ps.len - 1] = nq;
    out_shape[ps.len] = d;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, ps.len + 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * @as(c_int, @intCast(tg)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(tg), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", nk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Q", nq));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", @intCast(tg)));
    const y = try apply(g, kh, cfg, &.{ b, partial, queries, sc });
    if (!depth_engaged) {
        depth_engaged = true;
        log.info("[inkling2] depth_attention engaged\n", .{});
    }
    return y;
}
