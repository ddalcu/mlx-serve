const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

var kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
var sigmoid_table: ?mlx.mlx_array = null;
var announced = false;

pub fn enabled() bool {
    const raw = std.c.getenv("MLX_SERVE_HC_PREFILL") orelse return true;
    return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
}

pub fn eligible(batch: c_int, seq: c_int, hc: u32, hidden: u32) bool {
    return enabled() and geometry(batch, seq) and hc >= 1 and hc <= 8 and hidden >= 128 and hidden <= 4096 and hidden % 128 == 0;
}

fn geometry(batch: c_int, seq: c_int) bool {
    return batch >= 1 and batch <= 2 and seq > 16 and seq <= @import("qwen4_prefill_limits.zig").max_seq;
}

pub const Pending = struct { out: mlx.mlx_array, inj: mlx.mlx_array };
pub const Norm = struct {
    normalized: mlx.mlx_array,
    raw_inject: mlx.mlx_array,
    stream: mlx.mlx_array,

    pub fn deinit(self: Norm) void {
        _ = mlx.mlx_array_free(self.normalized);
        _ = mlx.mlx_array_free(self.raw_inject);
        _ = mlx.mlx_array_free(self.stream);
    }
};

fn getKernel(which: usize) !mlx.mlx_fast_metal_kernel {
    if (kernels[which]) |k| return k;
    const ni = [_][*:0]const u8{ "x", "w", "eps", "wo", "wi" };
    const no = [_][*:0]const u8{ "normed", "stream" };
    const mi = [_][*:0]const u8{ "up", "normed", "sigtab" };
    const mo = [_][*:0]const u8{"out"};
    const ins = if (which == 0) mlx.mlx_vector_string_new_data(&ni, ni.len) else mlx.mlx_vector_string_new_data(&mi, mi.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs = if (which == 0) mlx.mlx_vector_string_new_data(&no, no.len) else mlx.mlx_vector_string_new_data(&mo, mo.len);
    defer _ = mlx.mlx_vector_string_free(outs);
    const k = mlx.mlx_fast_metal_kernel_new(if (which == 0) "msv_hc_prefill_norm" else "msv_hc_prefill_mix", ins, outs, if (which == 0) @embedFile("kernels/hc_prefill_norm.metal") else @embedFile("kernels/hc_prefill_mix.metal"), "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernels[which] = k;
    return k;
}

pub fn norm(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, iw: mlx.mlx_array, eps: mlx.mlx_array, batch: c_int, seq: c_int, pending: ?Pending) !?Norm {
    if (!geometry(batch, seq) or iw.ctx == null) return null;
    const ws = mlx.getShape(w);
    if (ws.len != 2) return null;
    const hc = ws[0];
    const hidden = ws[1];
    if (hc < 1 or hc > 8 or hidden < 128 or hidden > 4096 or @mod(hidden, 128) != 0) return null;
    const width = hc * hidden;
    if (mlx.mlx_array_size(x) != @as(usize, @intCast(batch * seq * width)) or mlx.mlx_array_size(iw) != @as(usize, @intCast(width * hc))) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .bfloat16 or mlx.mlx_array_dtype(iw) != .bfloat16) return null;
    if (pending) |p| if (mlx.mlx_array_dtype(p.out) != .bfloat16 or mlx.mlx_array_dtype(p.inj) != .bfloat16) return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ batch, seq, hc, hidden };
    const xsh = if (pending != null) [_]c_int{ batch, seq, width } else [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &xsh, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, batch * seq * hc * @divTrunc(hidden, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @divTrunc(hidden, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", hidden));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", hc));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WR", @intFromBool(pending != null)));
    const arrays = [_]mlx.mlx_array{ x, w, eps, if (pending) |p| p.out else x, if (pending) |p| p.inj else x };
    const ins = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try getKernel(0), ins, cfg, s));
    var normalized = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(normalized);
    var stream = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(stream);
    try mlx.check(mlx.mlx_vector_array_get(&normalized, outs, 0));
    if (pending != null) try mlx.check(mlx.mlx_vector_array_get(&stream, outs, 1)) else try mlx.check(mlx.mlx_array_set(&stream, x));
    var raw = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(raw);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, normalized, &[_]c_int{ batch, seq, width }, 3, s));
    // Preserve the native inject reduction order for near-boundary BF16 gates.
    try mlx.check(mlx.mlx_matmul(&raw, flat, iw, s));
    // Keep the prefill engagement visible after a short server readiness probe.
    if (!announced and seq >= 512) {
        announced = true;
        log.info("[hc-prefill] engaged: S={d} B={d} pending={} native_inject=true\n", .{ seq, batch, pending != null });
    }
    return .{ .normalized = normalized, .raw_inject = raw, .stream = stream };
}

pub fn sigmoidTable(s: mlx.mlx_stream) !mlx.mlx_array {
    if (sigmoid_table) |table| return table;
    var bits: [65536]u16 = undefined;
    for (&bits, 0..) |*v, i| v.* = @intCast(i);
    const x = mlx.mlx_array_new_data(&bits, &[_]c_int{65536}, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    var table = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(table);
    try mlx.check(mlx.mlx_sigmoid(&table, x, s));
    sigmoid_table = table;
    return table;
}

pub fn mix(s: mlx.mlx_stream, up: mlx.mlx_array, normalized: mlx.mlx_array, batch: c_int, seq: c_int) !?mlx.mlx_array {
    if (!geometry(batch, seq) or mlx.mlx_array_dtype(up) != .bfloat16 or mlx.mlx_array_dtype(normalized) != .bfloat16) return null;
    const shape4 = mlx.getShape(normalized);
    if (shape4.len != 4) return null;
    const hc = shape4[2];
    const hidden = shape4[3];
    if (hc < 1 or hc > 8 or hidden < 128 or hidden > 4096 or @mod(hidden, 128) != 0) return null;
    const count = @as(usize, @intCast(batch * seq * hc * hidden));
    if (mlx.mlx_array_size(up) != count or mlx.mlx_array_size(normalized) != count) return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ batch, seq, hidden };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, batch * seq * hidden, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "M", batch * seq));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", hc));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", hidden));
    const ins = mlx.mlx_vector_array_new_data(&.{ up, normalized, try sigmoidTable(s) }, 3);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try getKernel(1), ins, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}
