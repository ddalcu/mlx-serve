const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

var kernels: ?[2]mlx.mlx_fast_metal_kernel = null;
var announced = false;
var fallback_announced = false;
pub var enabled_override: ?bool = null;
pub const MAX_KV: c_int = 1_048_576;

// Approximate reduction: per-element error vs float64 <= max(1.5 * stock,
// 2.5e-6) before BF16, <= max(1.5 * stock, 2e-3) after BF16; not byte identity.
pub fn enabled() bool {
    if (enabled_override) |value| return value;
    const raw = std.c.getenv("MLX_SERVE_QSA_PAIR") orelse return false;
    return std.mem.eql(u8, std.mem.sliceTo(raw, 0), "1");
}

pub fn supports(batch: c_int, seq: c_int, kv: c_int, ratio: c_int, kb: c_int) bool {
    return ratio == 4 and kb > 0 and kb <= 512 and seq >= 16 and seq <= 8192 and
        batch >= 1 and batch <= 2 and kv >= seq and kv <= MAX_KV;
}

pub fn tileCount(kb: c_int) c_int {
    return 2 * @divTrunc((kb + 1) * 4 + 31, 32) + 3;
}

/// One base position per four-key block, plus one query-membership mask per tile.
pub fn plannerBytes(batch: u64, seq: u64, kb: u64) u64 {
    if (batch == 0 or seq == 0 or kb == 0) return 0;
    return batch * ((seq + 1) / 2) * @as(u64, @intCast(tileCount(@intCast(@min(kb, 512))))) * (8 + 1) * 4;
}

fn getKernels() ![2]mlx.mlx_fast_metal_kernel {
    if (kernels) |ks| return ks;
    const plan_in = [_][*:0]const u8{ "blocks", "kvlen" };
    const plan_out = [_][*:0]const u8{ "tilepos", "tilemask" };
    const gather_in = [_][*:0]const u8{ "q", "k", "v", "scl", "blocks", "tilepos", "tilemask" };
    const gather_out = [_][*:0]const u8{"out"};
    const pi = mlx.mlx_vector_string_new_data(&plan_in, plan_in.len);
    defer _ = mlx.mlx_vector_string_free(pi);
    const po = mlx.mlx_vector_string_new_data(&plan_out, plan_out.len);
    defer _ = mlx.mlx_vector_string_free(po);
    const gi = mlx.mlx_vector_string_new_data(&gather_in, gather_in.len);
    defer _ = mlx.mlx_vector_string_free(gi);
    const go = mlx.mlx_vector_string_new_data(&gather_out, gather_out.len);
    defer _ = mlx.mlx_vector_string_free(go);
    const plan = mlx.mlx_fast_metal_kernel_new("msv_qsa_pair_plan", pi, po, @embedFile("kernels/qsa_pair_plan.metal"), "", false, false);
    if (plan.ctx == null) return error.MetalKernelCompileFailed;
    errdefer _ = mlx.mlx_fast_metal_kernel_free(plan);
    const gather = mlx.mlx_fast_metal_kernel_new("msv_qsa_pair", gi, go, @embedFile("kernels/qsa_pair.metal"), @embedFile("kernels/qsa_nax_header.metal"), false, false);
    if (gather.ctx == null) return error.MetalKernelCompileFailed;
    kernels = .{ plan, gather };
    return kernels.?;
}

/// Called only after gatherQsa256 has validated the tensors and NAX eligibility.
pub fn apply(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, blocks: mlx.mlx_array, ratio: c_int) !?mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    if (!supports(qs[0], qs[2], ks[2], ratio, bs[2])) {
        if (!fallback_announced) {
            fallback_announced = true;
            log.info("[qsa-pair] fallback: unsupported S={d} kv={d} KB={d} B={d} ratio={d} (max_kv={d})\n", .{ qs[2], ks[2], bs[2], qs[0], ratio, MAX_KV });
        }
        return null;
    }
    const ng = @divTrunc(qs[2] + 1, 2);
    const nt = tileCount(bs[2]);
    const pair = try getKernels();
    const pc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(pc);
    const pos_shape = [_]c_int{ qs[0] * ng, nt, 8 };
    const mask_shape = [_]c_int{ qs[0] * ng, nt };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(pc, &pos_shape, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(pc, &mask_shape, 2, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(pc, ng * 32, 1, qs[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(pc, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(pc, "KB", bs[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(pc, "NT", nt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(pc, "RATIO", ratio));
    const kvlen = mlx.mlx_array_new_int(ks[2]);
    defer _ = mlx.mlx_array_free(kvlen);
    const pin = mlx.mlx_vector_array_new_data(&.{ blocks, kvlen }, 2);
    defer _ = mlx.mlx_vector_array_free(pin);
    var pout = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pout);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&pout, pair[0], pin, pc, s));
    if (mlx.mlx_vector_array_size(pout) != 2) return error.MetalKernelBadOutputCount;
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_vector_array_get(&pos, pout, 0));
    try mlx.check(mlx.mlx_vector_array_get(&mask, pout, 1));
    const gc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(gc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(gc, qs.ptr, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(gc, ng * 32, ks[1] * 4, qs[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(gc, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(gc, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(gc, "NSG", 4));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(gc, "BK", 32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(gc, "G", 2));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(gc, "RATIO", ratio));
    const scl = mlx.mlx_array_new_data(&[_]f32{scale}, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const gin = mlx.mlx_vector_array_new_data(&.{ q, k, v, scl, blocks, pos, mask }, 7);
    defer _ = mlx.mlx_vector_array_free(gin);
    var gout = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(gout);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&gout, pair[1], gin, gc, s));
    if (mlx.mlx_vector_array_size(gout) != 1) return error.MetalKernelBadOutputCount;
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, gout, 0));
    if (!announced) {
        announced = true;
        log.info("[qsa-pair] engaged: S={d} kv={d} KB={d} B={d}\n", .{ qs[2], ks[2], bs[2], qs[0] });
    }
    return result;
}
