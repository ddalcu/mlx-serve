//! sdk/diag/standin.
//! Owns Standin, standin_cached, qwen4_standin_override, qwen4Standin, standin_ones, standinOnes, and 1 more.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");

// Self-contained monotonic lap timer (Zig 0.17 has no std.time.Timer;
// the repo times via std.Io). `lap()` returns ns since the previous lap.
/// QWEN4_STANDIN=gdn,attn,mlp,gdn_recur,gdn_proj,attn_qsa,attn_sdpa,hc,moe_shared,moe_router,moe_gateup,moe_down — replace a block
/// with a free stand-in (a +1 ref of its input / a cached ones array) so the
/// in-situ fwd-ubench reports what that block costs. Diagnostic only.
pub const Standin = packed struct(u16) { gdn: bool = false, attn: bool = false, mlp: bool = false, gdn_recur: bool = false, gdn_proj: bool = false, attn_qsa: bool = false, attn_sdpa: bool = false, hc: bool = false, moe_shared: bool = false, moe_router: bool = false, moe_gateup: bool = false, moe_down: bool = false, _pad: u4 = 0 };
var standin_cached: ?Standin = null;
pub var qwen4_standin_override: ?Standin = null;
pub fn qwen4Standin() Standin {
    if (qwen4_standin_override) |v| return v;
    if (standin_cached) |v| return v;
    var v = Standin{};
    if (env.get(.QWEN4_STANDIN)) |raw| {
        var it = std.mem.splitScalar(u8, std.mem.sliceTo(raw, 0), ',');
        while (it.next()) |tok| {
            if (std.mem.eql(u8, tok, "gdn")) v.gdn = true;
            if (std.mem.eql(u8, tok, "attn")) v.attn = true;
            if (std.mem.eql(u8, tok, "mlp")) v.mlp = true;
            if (std.mem.eql(u8, tok, "gdn_recur")) v.gdn_recur = true;
            if (std.mem.eql(u8, tok, "gdn_proj")) v.gdn_proj = true;
            if (std.mem.eql(u8, tok, "attn_qsa")) v.attn_qsa = true;
            if (std.mem.eql(u8, tok, "attn_sdpa")) v.attn_sdpa = true;
            if (std.mem.eql(u8, tok, "hc")) v.hc = true;
            if (std.mem.eql(u8, tok, "moe_shared")) v.moe_shared = true;
            if (std.mem.eql(u8, tok, "moe_router")) v.moe_router = true;
            if (std.mem.eql(u8, tok, "moe_gateup")) v.moe_gateup = true;
            if (std.mem.eql(u8, tok, "moe_down")) v.moe_down = true;
        }
    }
    standin_cached = v;
    return v;
}
var standin_ones: ?mlx.mlx_array = null;
pub fn standinOnes(shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (standin_ones) |a| {
        if (std.mem.eql(c_int, mlx.getShape(a), shape)) {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&r, a));
            return r;
        }
        _ = mlx.mlx_array_free(a);
    }
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&a, shape.ptr, @intCast(shape.len), .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(a));
    standin_ones = a;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, a));
    return r;
}
pub fn standinRef(x: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, x));
    return r;
}
