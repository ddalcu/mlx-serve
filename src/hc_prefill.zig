const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

var kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
var sigmoid_table: ?mlx.mlx_array = null;
var announced = false;
var upmix_kernel: ?mlx.mlx_fast_metal_kernel = null;
var upmix_announced = false;
var upmix_verified: u32 = 0;

pub fn upmixEnabled() bool {
    if (@import("prefill_experiment.zig").hcUpmix()) |on| return on;
    const raw = std.c.getenv("MLX_SERVE_HC_UPMIX") orelse return false;
    return raw[0] == '1';
}

pub fn verifyUpmix() bool {
    const raw = std.c.getenv("QWEN4_HC_UPMIX_VERIFY") orelse return false;
    if (raw[0] != '1' or upmix_verified >= 2) return false;
    upmix_verified += 1;
    return true;
}

pub fn upMix(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, normalized: mlx.mlx_array) !?mlx.mlx_array {
    const ns = mlx.getShape(normalized);
    if (ns.len != 4 or ns[2] != 4 or ns[3] != 2560 or !geometry(ns[0], ns[1])) return null;
    const rows = ns[0] * ns[1];
    if (rows < 2048 or @mod(rows, 64) != 0 or mlx.mlx_array_size(x) != @as(usize, @intCast(rows * 320))) return null;
    if (!std.mem.eql(c_int, mlx.getShape(w), &.{ 10240, 80 }) or !std.mem.eql(c_int, mlx.getShape(sc), &.{ 10240, 5 }) or !std.mem.eql(c_int, mlx.getShape(bi), &.{ 10240, 5 })) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(normalized) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    var dq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq);
    try mlx.check(mlx.mlx_dequantize(&dq, w, sc, bi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, s));
    var shaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shaped);
    try mlx.check(mlx.mlx_reshape(&shaped, dq, &.{ 4, 2560, 320 }, 3, s));
    var transposed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(transposed);
    try mlx.check(mlx.mlx_transpose_axes(&transposed, shaped, &.{ 1, 0, 2 }, 3, s));
    var flat_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_w);
    try mlx.check(mlx.mlx_reshape(&flat_w, transposed, &.{ 10240, 320 }, 2, s));
    var flat_x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_x);
    try mlx.check(mlx.mlx_reshape(&flat_x, x, &.{ rows, 320 }, 2, s));
    var flat_n = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_n);
    try mlx.check(mlx.mlx_reshape(&flat_n, normalized, &.{ rows, 4, 2560 }, 3, s));
    if (upmix_kernel == null) {
        const names = [_][*:0]const u8{ "x", "w", "normed", "sigtab" };
        const ins = mlx.mlx_vector_string_new_data(&names, names.len);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{"out"}, 1);
        defer _ = mlx.mlx_vector_string_free(outs);
        const k = mlx.mlx_fast_metal_kernel_new("msv_hc_upmix", ins, outs, @embedFile("kernels/hc_prefill_upmix.metal"), @embedFile("kernels/qsa_nax_header.metal"), true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        upmix_kernel = k;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ ns[0], ns[1], 2560 }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 160 * 32, @divTrunc(rows, 64) * 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const inputs = mlx.mlx_vector_array_new_data(&.{ flat_x, flat_w, flat_n, try sigmoidTable(s) }, 4);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, upmix_kernel.?, inputs, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
    if (!upmix_announced) {
        upmix_announced = true;
        log.info("[hc-upmix] engaged: rows={d} HC=4 H=2560 K=320 bits=8 transient_up_eliminated=true weight_cache=false\n", .{rows});
    }
    return out;
}

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
