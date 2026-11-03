//! Hunyuan 3's expert container name for the engine's layered loader: `mlp.experts` (ox-ox) or `mlp.switch_mlp`
//! (mlx-lm), whichever the checkpoint holds.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const Weights = @import("../../sdk/weights/load.zig").Weights;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const getWeightAtOpt = @import("../../engines/mlx/load/weights.zig").getWeightAtOpt;
const layerBase = @import("../../engines/mlx/load/weights.zig").layerBase;
const moeExpertSuffix = @import("../../engines/mlx/load/weights.zig").moeExpertSuffix;

/// Resolve the hy_v3 stacked-expert container name for a layer. ox-ox-style MLX
/// conversions stack the experts under `mlp.experts.*`; mlx-lm's converter
/// (mlx-community Hy3-oQ2*, pipenetwork Hy3-REAP*) names the SAME [E, out, in]
/// tensors `mlp.switch_mlp.*`. Prefers `experts` when present, so an ox-ox
/// checkpoint binds byte-identically to before this fallback existed.
pub fn hy3ExpertContainer(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32) []const u8 {
    var base_buf: [96]u8 = undefined;
    return expertContainerAt(weights, buf, layerBase(&base_buf, prefix, layer));
}

pub fn expertContainerAt(weights: *const Weights, buf: *[256]u8, base: []const u8) []const u8 {
    return if (getWeightAtOpt(weights, buf, base, "mlp.experts.gate_proj.weight") != null)
        "mlp.experts"
    else
        "mlp.switch_mlp";
}

// ── tests ──

test "hy3ExpertContainer resolves both mlx-lm switch_mlp and ox-ox experts naming" {
    // hy_v3 MoE experts stack under `mlp.experts.*` in ox-ox-style MLX builds
    // but `mlp.switch_mlp.*` in mlx-lm builds (mlx-community Hy3-oQ2e,
    // pipenetwork Hy3-REAP*). Same [E, out, in] tensor; the loader must accept
    // either name or it crashes with MISSING WEIGHT at load.
    const allocator = testing.allocator;
    var buf: [256]u8 = undefined;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // mlx-lm build: only switch_mlp present → resolver picks it (was the crash).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.layers.1.mlp.switch_mlp.gate_proj.weight");
        try testing.expectEqualStrings("mlp.switch_mlp", hy3ExpertContainer(&w, &buf, "model", 1));
    }
    // ox-ox build: experts present → preferred, so it binds byte-identically.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.layers.1.mlp.experts.gate_proj.weight");
        try testing.expectEqualStrings("mlp.experts", hy3ExpertContainer(&w, &buf, "model", 1));
    }
    // The suffix builder composes the resolved container with each leaf.
    var sbuf: [64]u8 = undefined;
    try testing.expectEqualStrings(
        "mlp.switch_mlp.down_proj.scales",
        moeExpertSuffix(&sbuf, "mlp.switch_mlp", "down_proj.scales"),
    );
}
