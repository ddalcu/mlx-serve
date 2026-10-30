//! Loads one hyper-connection block (`loadHcWeights`, its 1/hc fold into the down and inject projections) and the
//! PLE layer's weights (`loadPleWeights`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const Weights = @import("../../sdk/config/model.zig").Weights;
const getLayerWeight = @import("../../engines/mlx/load/weights.zig").getLayerWeight;
const groupedNormView = @import("../../engines/mlx/load/weights.zig").groupedNormView;
const scaleTriple = @import("../../engines/mlx/load/weights.zig").scaleTriple;
const transposeBf16Weight = @import("../../engines/mlx/load/weights.zig").transposeBf16Weight;
const weightTriple = @import("../../engines/mlx/load/weights.zig").weightTriple;
const HcWeights = @import("weights.zig").HcWeights;
const PleWeights = @import("weights.zig").PleWeights;

pub fn loadHcWeights(weights: *const Weights, base: []const u8, with_inject: bool, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !HcWeights {
    var nb: [160]u8 = undefined;
    const down = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.input_mix_weight_down", .{base}) catch unreachable, null, owned_bf16, allocator, s);
    const up = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.input_mix_weight_up", .{base}) catch unreachable, null, owned_bf16, allocator, s);
    var out: HcWeights = .{
        .norm_w = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.hc_norm.weight", .{base}) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .down_w = down.w,
        .down_s = down.sc,
        .down_b = down.bi,
        .up_w = up.w,
        .up_s = up.sc,
        .up_b = up.bi,
    };
    // The reference divides the down and inject projections by hc_count;
    // fold that into the (de)quantization affine or the dense weight.
    const inv_hc = mlx.mlx_array_new_float(1.0 / @as(f32, @floatFromInt(hc)));
    defer _ = mlx.mlx_array_free(inv_hc);
    if (out.down_s.ctx == null and out.up_s.ctx == null) {
        const rows = struct {
            fn view(a: mlx.mlx_array, o: *std.ArrayList(mlx.mlx_array), al: std.mem.Allocator, st: mlx.mlx_stream) !mlx.mlx_array {
                var v = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_view(&v, a, .uint32, st));
                try mlx.check(mlx.mlx_array_eval(v));
                try o.append(al, v);
                return v;
            }
        };
        const down_src = weights.get(std.fmt.bufPrint(&nb, "{s}.input_mix_weight_down.weight", .{base}) catch unreachable).?;
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        try mlx.check(mlx.mlx_astype(&k, inv_hc, mlx.mlx_array_dtype(down_src), s));
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, down_src, k, s));
        try mlx.check(mlx.mlx_array_eval(scaled));
        try owned_bf16.append(allocator, scaled);
        out.down_w = try transposeBf16Weight(scaled, s);
        try owned_bf16.append(allocator, out.down_w);
        const up_src = weights.get(std.fmt.bufPrint(&nb, "{s}.input_mix_weight_up.weight", .{base}) catch unreachable).?;
        if (mlx.mlx_array_dtype(down_src) == .bfloat16 and mlx.mlx_array_dtype(up_src) == .bfloat16) {
            out.down_rows = try rows.view(scaled, owned_bf16, allocator, s);
            out.up_rows = try rows.view(up_src, owned_bf16, allocator, s);
        }
    } else try scaleTriple(&out.down_w, &out.down_s, &out.down_b, inv_hc, owned_bf16, allocator, s);
    if (with_inject) {
        const inj = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.block_inject_weight", .{base}) catch unreachable, hc * hidden, owned_bf16, allocator, s);
        out.inject_w = inj.w;
        out.inject_s = inj.sc;
        out.inject_b = inj.bi;
        try scaleTriple(&out.inject_w, &out.inject_s, &out.inject_b, inv_hc, owned_bf16, allocator, s);
        if (out.inject_s.ctx == null) {
            var flat = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_contiguous(&flat, out.inject_w, false, s));
            try mlx.check(mlx.mlx_array_eval(flat));
            try owned_bf16.append(allocator, flat);
            out.inject_flat = flat;
        }
    }
    return out;
}

pub fn loadPleWeights(weights: *const Weights, name_buf: *[256]u8, prefix: []const u8, li: u32, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !PleWeights {
    var nb: [160]u8 = undefined;
    const key = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.key_proj", .{ prefix, li }) catch unreachable, null, owned_bf16, allocator, s);
    const value = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.value_proj", .{ prefix, li }) catch unreachable, null, owned_bf16, allocator, s);
    return .{
        .key_w = key.w,
        .key_s = key.sc,
        .key_b = key.bi,
        .value_w = value.w,
        .value_s = value.sc,
        .value_b = value.bi,
        .norm_key = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_key.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .norm_query = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_query.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .norm_conv = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_conv.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .conv_w = try getLayerWeight(weights, name_buf, prefix, li, "ple.conv1d.weight"),
    };
}
