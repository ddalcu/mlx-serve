//! sdk/quant/hadamard.
//! Owns registerRhtSigns.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const rht = @import("hadamard/rht.zig");
const log = @import("../util/log.zig");
const Weights = @import("../weights/resident.zig").Weights;
const biasNegScalePeak = @import("ternary.zig").biasNegScalePeak;

/// Binds every `<base>.signs` to `<base>.weight`; ONE eval for every sign
/// hash and bias probe (a sync per weight was ~800 syncs at load). A sign
/// vector without its weight fails the load by name.
pub fn registerRhtSigns(reg: *rht.Registry, weights: *const Weights, s: mlx.mlx_stream) !void {
    const Pending = struct { w: mlx.mlx_array, signs: mlx.mlx_array, f32: mlx.mlx_array, peak: ?mlx.mlx_array };
    var pending = std.ArrayList(Pending).empty;
    defer {
        for (pending.items) |p| {
            _ = mlx.mlx_array_free(p.f32);
            if (p.peak) |pk| _ = mlx.mlx_array_free(pk);
        }
        pending.deinit(reg.allocator);
    }
    var it = weights.map.iterator();
    var nb: [256]u8 = undefined;
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        if (!std.mem.endsWith(u8, key, rht.SIGNS_SUFFIX)) continue;
        const base = key[0 .. key.len - rht.SIGNS_SUFFIX.len];
        const w = weights.get(std.fmt.bufPrint(&nb, "{s}.weight", .{base}) catch return error.MissingWeight) orelse {
            log.err("MISSING WEIGHT: {s}.weight (has Hadamard signs)\n", .{base});
            return error.MissingWeight;
        };
        const f = try rht.signsF32(kv.value_ptr.*, s);
        errdefer _ = mlx.mlx_array_free(f);
        const peak = try biasNegScalePeak(weights, base, s);
        try pending.append(reg.allocator, .{ .w = w, .signs = kv.value_ptr.*, .f32 = f, .peak = peak });
    }
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (pending.items) |p| {
            try mlx.check(mlx.mlx_vector_array_append_value(vec, p.f32));
            if (p.peak) |pk| try mlx.check(mlx.mlx_vector_array_append_value(vec, pk));
        }
        try mlx.check(mlx.mlx_eval(vec));
    }
    // Prism's codec writes affine biases as exactly -scales (ternary codes
    // 0..2 -> -1..1); the verify-width kernel's fast path depends on it.
    for (pending.items) |p| {
        try reg.register(p.w, p.signs, p.f32);
        if (reg.bias_is_neg_scale) {
            var v: f32 = 1;
            if (p.peak) |pk| try mlx.check(mlx.mlx_array_item_float32(&v, pk));
            reg.bias_is_neg_scale = v == 0;
        }
    }
    log.info("[rht] {d} weights carry block-{d} Hadamard rotations (bias == -scale: {}, conflicting widths: {d})\n", .{ reg.count(), reg.block, reg.bias_is_neg_scale, reg.conflictingWidths() });
}
