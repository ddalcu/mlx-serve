//! Fused Metal kernels for Inkling 2's vision tower (`vision.zig`): QKV bias + 2-D rotary + head split, Metal 4 NAX
//! bidirectional flash attention, bias + layer-scaled residual + next LayerNorm, and bias + exact erf-GELU.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const verifyQmmNaxAvailable = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const config = @import("config.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Dtype = ops.Dtype;

const Kernel = struct {
    name: [*:0]const u8,
    ins: []const [*:0]const u8,
    outs: []const [*:0]const u8,
    source: [:0]const u8,
    header: [:0]const u8 = "",
    contiguous: bool = true,
    built: ?mlx.mlx_fast_metal_kernel = null,
};

pub const QKV_ROPE_SOURCE = kernelSource("plugins/inkling2/kernels/vision_qkv_rope.metal");
pub const ATTN_NAX_SOURCE = kernelSource("plugins/inkling2/kernels/vision_attn_nax.metal");
pub const ATTN_NAX_HEADER = kernelSource("kernels/common/mpp_tensor_ops.metal");
pub const RES_NORM_SOURCE = kernelSource("plugins/inkling2/kernels/vision_res_norm.metal");
pub const BIAS_GELU_SOURCE = kernelSource("plugins/inkling2/kernels/vision_bias_gelu.metal");

pub var qkv_rope_kernel: Kernel = .{
    .name = "msv_inkling2_vis_qkv_rope",
    .ins = &.{ "QKV", "B", "COS", "SIN" },
    .outs = &.{ "OQ", "OK", "OV" },
    .source = QKV_ROPE_SOURCE,
};

pub var attn_nax_kernel: Kernel = .{
    .name = "msv_inkling2_vis_attn_nax",
    .ins = &.{ "Q", "K", "V", "scale" },
    .outs = &.{"O"},
    .source = ATTN_NAX_SOURCE,
    .header = ATTN_NAX_HEADER,
};

pub var res_norm_kernel: Kernel = .{
    .name = "msv_inkling2_vis_res_norm",
    .ins = &.{ "X", "Y", "B", "SC", "NW", "NB", "eps" },
    .outs = &.{ "OUT", "NORM" },
    .source = RES_NORM_SOURCE,
};

pub var bias_gelu_kernel: Kernel = .{
    .name = "msv_inkling2_vis_bias_gelu",
    .ins = &.{ "Y", "B" },
    .outs = &.{"O"},
    .source = BIAS_GELU_SOURCE,
};

const TArg = union(enum) { dtype: Dtype, int: c_int };
const Out = struct { shape: []const c_int, dtype: Dtype };
const TArgs = []const struct { [*:0]const u8, TArg };

fn launchGrid(g: *Graph, k: *Kernel, inputs: []const Array, comptime n: usize, outs: [n]Out, grid: [3]c_int, tg: [3]c_int, targs: TArgs) ![n]Array {
    const kernel = k.built orelse blk: {
        const iv = mlx.mlx_vector_string_new_data(k.ins.ptr, k.ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(k.outs.ptr, k.outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new(k.name, iv, ov, k.source, k.header, k.contiguous, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        k.built = made;
        break :blk made;
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg[0], tg[1], tg[2]));
    for (targs) |t| switch (t[1]) {
        .dtype => |d| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t[0], d)),
        .int => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], v)),
    };
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, g.s));
    var res: [n]Array = undefined;
    for (&res, 0..) |*r, i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, ov, i));
        r.* = try g.adopt(o);
    }
    return res;
}

/// Whether the fused vision kernels serve `s` and `dt` (honors `MLX_SERVE_INKLING2_FUSED=0`).
pub fn runs(s: mlx.mlx_stream, dt: Dtype) bool {
    return mlx.streamIsGpu(s) and config.fusedOn() and (dt == .bfloat16 or dt == .float16 or dt == .float32);
}

/// Whether the NAX bidirectional attention kernel serves `s`, `dt` and head width `hd`.
pub fn attnNaxOk(s: mlx.mlx_stream, dt: Dtype, hd: c_int) bool {
    return runs(s, dt) and verifyQmmNaxAvailable() and (dt == .bfloat16 or dt == .float16) and hd == 64;
}

/// Threadgroup size for `vision_res_norm.metal`: multiple of 32, capped at 256.
pub fn resNormTg(d: c_int) c_int {
    const groups = @max(@as(c_int, 1), @min(@as(c_int, 8), @divTrunc(d + 127, 128)));
    return groups * 32;
}

/// `vision_qkv_rope.metal`: `qkv` `[L, 3*H*D]` + `bias` `[3*H*D]` + `cos`/`sin` `[L, D]` f32 -> `q, k, v` `[1, H, L, D]`.
pub fn qkvRope(g: *Graph, qkv: Array, bias: Array, cos: Array, sin: Array, heads: c_int, hd: c_int) ![3]Array {
    const dt = ops.dtype(qkv);
    const l = ops.shape(qkv)[0];
    const total = 3 * heads * @divExact(hd, 2);
    const tg = @min(total, @as(c_int, 256));
    const gx = @divTrunc(total + tg - 1, tg) * tg;
    const sh = [_]c_int{ 1, heads, l, hd };
    return launchGrid(g, &qkv_rope_kernel, &.{ qkv, bias, cos, sin }, 3, .{
        .{ .shape = &sh, .dtype = dt },
        .{ .shape = &sh, .dtype = dt },
        .{ .shape = &sh, .dtype = dt },
    }, .{ gx, l, 1 }, .{ tg, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },
        .{ "H", .{ .int = heads } },
        .{ "D", .{ .int = hd } },
    });
}

/// `vision_attn_nax.metal`: bidirectional NAX attention on `q, k, v` `[1, H, L, 64]` -> `[L, H*64]` token-major.
pub fn attnNax(g: *Graph, q: Array, k: Array, v: Array, scale: Array) !Array {
    var sg = g.scope();
    defer sg.deinit();
    const dt = ops.dtype(q);
    const heads = ops.shape(q)[1];
    const l = ops.shape(q)[2];
    const out = try launchGrid(&sg, &attn_nax_kernel, &.{ q, k, v, scale }, 1, .{
        .{ .shape = &.{ l, heads * 64 }, .dtype = dt },
    }, .{ @divTrunc(l + 63, 64) * 128, heads, 1 }, .{ 128, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },
        .{ "H", .{ .int = heads } },
        .{ "BK", .{ .int = 32 } },
    });
    return g.keep(out[0]);
}

/// `vision_res_norm.metal`: `out = x + sc * (y + bias)` and (when `do_norm`) `norm = LayerNorm(out, nw, nb, eps)`.
pub fn resNorm(g: *Graph, x: Array, y: Array, bias: Array, sc: Array, nw: Array, nb: Array, eps: Array, do_norm: bool) ![2]Array {
    const dt = ops.dtype(x);
    const sh = ops.shape(x);
    const l = sh[0];
    const d = sh[1];
    const tg = resNormTg(d);
    const out_sh = [_]c_int{ l, d };
    const dummy = [_]c_int{1};
    return launchGrid(g, &res_norm_kernel, &.{ x, y, bias, sc, nw, nb, eps }, 2, .{
        .{ .shape = &out_sh, .dtype = dt },
        .{ .shape = if (do_norm) &out_sh else &dummy, .dtype = dt },
    }, .{ tg, l, 1 }, .{ tg, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },
        .{ "D", .{ .int = d } },
        .{ "TG", .{ .int = tg } },
        .{ "DO_NORM", .{ .int = @intFromBool(do_norm) } },
    });
}

/// `vision_bias_gelu.metal`: `gelu_erf(y + bias)` (or `gelu_erf(y)` when `bias` is null) on `[L, M]`.
pub fn biasGelu(g: *Graph, y: Array, bias: Array) !Array {
    var sg = g.scope();
    defer sg.deinit();
    const dt = ops.dtype(y);
    const sh = ops.shape(y);
    const l = sh[0];
    const m = sh[1];
    const has_bias = bias.ctx != null;
    const b_arg = if (has_bias) bias else try sg.zeros(&.{1}, dt);
    const quads = @divTrunc(m + 3, 4);
    const tg = @min(quads, @as(c_int, 256));
    const gx = @divTrunc(quads + tg - 1, tg) * tg;
    const out = try launchGrid(&sg, &bias_gelu_kernel, &.{ y, b_arg }, 1, .{
        .{ .shape = &.{ l, m }, .dtype = dt },
    }, .{ gx, l, 1 }, .{ tg, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },
        .{ "HAS_BIAS", .{ .int = @intFromBool(has_bias) } },
    });
    return g.keep(out[0]);
}
