//! Fused Mamba2 mixer step for any window (decode, spec verify, prefill): the
//! conv window, depthwise conv + bias + SiLU, dt softplus/clamp, the SSM
//! recurrence, the D skip and the SiLU(z) gate run as ONE dispatch over R
//! consecutive rows; the grouped RMS norm is a second. Same structure as
//! TensorFold's `mamba_step`/`group_norm` (MIT), arithmetic kept to the op
//! chain's: f32 math, bf16 where the chain stores bf16. The SSM state stays
//! f32; `y` comes back in x's dtype.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");

/// Rows of B/C the step kernel stages per threadgroup pass; a wider window
/// loops over passes. Also the decode/verify width `add_norm` keys on.
pub const MAX_ROWS: c_int = 16;
const TGY: c_int = 8;
/// Threadgroup memory is an occupancy decision: the step kernel keeps B/C for
/// a row BUCKET, never the 16-row pass on a one-row decode step.
const TG_BYTES_MAX: usize = 16 * 1024;

fn rowBucket(rows: c_int) c_int {
    return if (rows <= 1) 1 else if (rows <= 4) 4 else MAX_ROWS;
}

fn stepThreadgroupBytes(rows: c_int, state: c_int) usize {
    return @as(usize, @intCast(rowBucket(rows) * 2 * state)) * @sizeOf(f32);
}

fn stepFits(rows: c_int, state: c_int) bool {
    return stepThreadgroupBytes(rows, state) <= TG_BYTES_MAX;
}

// grid (32, Dh, B*H), threadgroup (32, TGY, 1): one simdgroup per (b, h, d),
// each lane owning Ds/32 state columns. B/C of the head's group are conv'd
// once per threadgroup into shared memory.
pub const STEP_SOURCE = kernelSource("kernels/ssm/mamba2_decode_step.metal");

// grid (GS/4, NG, rows): one threadgroup per (row, group), 4 elements a thread.
pub const GNORM_SOURCE = kernelSource("kernels/ssm/mamba2_decode_gnorm.metal");

pub const Geometry = struct { batch: c_int, rows: c_int, heads: c_int, groups: c_int, head_dim: c_int, state: c_int, conv_kernel: c_int, proj: c_int };

pub const Inputs = struct {
    proj: mlx.mlx_array, // [B,R,PROJ] in_proj output: z | x B C | dt
    conv_state: mlx.mlx_array, // [B,KC-1,CD]
    ssm_state: mlx.mlx_array, // [B,H,Dh,Ds] f32
    conv_w: mlx.mlx_array, // [CD,KC,1]
    conv_b: mlx.mlx_array, // [CD]
    A_log: mlx.mlx_array, // [H]
    dt_bias: mlx.mlx_array, // [H]
    D: mlx.mlx_array, // [H]
    dt_lo: mlx.mlx_array, // 0-dim f32 (MLX binds a scalar input by value)
    dt_hi: mlx.mlx_array, // 0-dim f32
    rows: mlx.mlx_array, // 0-dim int32
    all_rows: bool, // state after EVERY row (spec capture) or only the last
};

pub const Outputs = struct {
    y: mlx.mlx_array, // [B,R,H*Dh] gated, x dtype
    ssm_state: mlx.mlx_array, // [R,B,H,Dh,Ds] f32 when all_rows, else [1,B,H,Dh,Ds]
    conv_state: mlx.mlx_array, // [B,KC-1,CD]
};

var step_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gnorm_kernel: ?mlx.mlx_fast_metal_kernel = null;
const StepKey = struct { g: Geometry, dt: mlx.mlx_dtype, all_rows: bool };
var step_key: ?StepKey = null;
var step_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
const GnormKey = struct { rows: c_int, xd: c_int, groups: c_int, dt: mlx.mlx_dtype };
var gnorm_key: ?GnormKey = null;
var gnorm_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn tmplInt(c: mlx.mlx_fast_metal_kernel_config, name: [*:0]const u8, v: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
}

fn buildStepConfig(g: Geometry, dt: mlx.mlx_dtype, all_rows: bool) !void {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const xd = g.heads * g.head_dim;
    const cd = xd + 2 * g.groups * g.state;
    const srows: c_int = if (all_rows) g.rows else 1;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ g.batch, g.rows, xd }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ srows, g.batch, g.heads, g.head_dim, g.state }, 5, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ g.batch, g.conv_kernel - 1, cd }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32, g.head_dim, g.batch * g.heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, TGY, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    try tmplInt(c, "XD", xd);
    try tmplInt(c, "DH", g.head_dim);
    try tmplInt(c, "DS", g.state);
    try tmplInt(c, "H", g.heads);
    try tmplInt(c, "NG", g.groups);
    try tmplInt(c, "KC", g.conv_kernel);
    try tmplInt(c, "PROJ", g.proj);
    try tmplInt(c, "B", g.batch);
    try tmplInt(c, "MAXR", rowBucket(g.rows));
    try tmplInt(c, "TGY", TGY);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "ALLROWS", all_rows));
    if (step_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(step_cfg);
    step_cfg = c;
    step_key = .{ .g = g, .dt = dt, .all_rows = all_rows };
}

/// Null when the geometry, dtypes or stream are outside the kernel (the
/// caller keeps the op chain).
pub fn step(g: Geometry, in: Inputs, s: mlx.mlx_stream) !?Outputs {
    if (!mlx.streamIsGpu(s)) return null;
    if (g.rows < 1) return null;
    if (!stepFits(g.rows, g.state)) return null;
    if (@rem(g.state, 32) != 0 or @rem(g.head_dim, TGY) != 0 or @rem(g.heads, g.groups) != 0 or g.conv_kernel < 2) return null;
    const xd = g.heads * g.head_dim;
    if (g.proj != 2 * xd + 2 * g.groups * g.state + g.heads) return null;
    const dt = mlx.mlx_array_dtype(in.proj);
    if (dt != .bfloat16 and dt != .float16) return null;
    for ([_]mlx.mlx_array{ in.conv_state, in.conv_w, in.conv_b, in.A_log, in.dt_bias, in.D }) |arr|
        if (arr.ctx == null or mlx.mlx_array_dtype(arr) != dt) return null;
    if (mlx.mlx_array_dtype(in.ssm_state) != .float32) return null;
    if (step_kernel == null) {
        const ins = [_][*:0]const u8{ "P", "CS_IN", "S_IN", "CW", "CB", "A_LOG", "DT_BIAS", "DSKIP", "dt_lo", "dt_hi", "rows" };
        const outs = [_][*:0]const u8{ "Y", "S_OUT", "CS_OUT" };
        step_kernel = try makeKernel("msv_mamba2_step", &ins, &outs, STEP_SOURCE);
    }
    const key = StepKey{ .g = g, .dt = dt, .all_rows = in.all_rows };
    if (step_key == null or !std.meta.eql(step_key.?, key)) try buildStepConfig(g, dt, in.all_rows);

    const arrs = [_]mlx.mlx_array{ in.proj, in.conv_state, in.ssm_state, in.conv_w, in.conv_b, in.A_log, in.dt_bias, in.D, in.dt_lo, in.dt_hi, in.rows };
    const v = mlx.mlx_vector_array_new_data(&arrs, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, step_kernel.?, v, step_cfg, s));
    var out: Outputs = .{ .y = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new() };
    errdefer {
        _ = mlx.mlx_array_free(out.y);
        _ = mlx.mlx_array_free(out.ssm_state);
        _ = mlx.mlx_array_free(out.conv_state);
    }
    try mlx.check(mlx.mlx_vector_array_get(&out.y, o, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.ssm_state, o, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.conv_state, o, 2));
    return out;
}

/// Grouped RMS norm times the norm weight: x [B,R,XD] over XD/groups-wide
/// groups. Null outside the kernel's set.
pub fn groupNorm(x: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, groups: c_int, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(x);
    if (sh.len != 3) return null;
    const rows = sh[0] * sh[1];
    const xd = sh[2];
    if (@rem(xd, groups) != 0) return null;
    const gs = @divExact(xd, groups);
    if (@rem(gs, 128) != 0 or gs > 4096) return null; // GS/4 threads, whole simdgroups
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(w) != dt) return null;
    if (gnorm_kernel == null) {
        const ins = [_][*:0]const u8{ "X", "W", "eps" };
        const outs = [_][*:0]const u8{"OUT"};
        gnorm_kernel = try makeKernel("msv_mamba2_gnorm", &ins, &outs, GNORM_SOURCE);
    }
    const key = GnormKey{ .rows = rows, .xd = xd, .groups = groups, .dt = dt };
    if (gnorm_key == null or !std.meta.eql(gnorm_key.?, key)) {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, sh.len, dt));
        const tn = @divExact(gs, 4);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, tn, groups, rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tn, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
        try tmplInt(c, "XD", xd);
        try tmplInt(c, "GS", gs);
        if (gnorm_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(gnorm_cfg);
        gnorm_cfg = c;
        gnorm_key = key;
    }
    const arrs = [_]mlx.mlx_array{ x, w, eps };
    const v = mlx.mlx_vector_array_new_data(&arrs, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, gnorm_kernel.?, v, gnorm_cfg, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    return y;
}

// Bar: a one-row decode step reserves one row of B/C, and a state too wide
// for the threadgroup budget declines instead of failing at dispatch.
test "mamba2 step: threadgroup memory follows the row bucket and stays in budget" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 1024), stepThreadgroupBytes(1, 128));
    try t.expectEqual(@as(usize, 4096), stepThreadgroupBytes(3, 128));
    try t.expectEqual(@as(usize, 16384), stepThreadgroupBytes(16, 128));
    try t.expect(stepFits(16, 128));
    try t.expect(stepFits(1, 512));
    try t.expect(!stepFits(16, 256));
    try t.expect(!stepFits(4, 1024));
}
