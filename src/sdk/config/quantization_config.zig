//! mlx-lm's `save_config` writes the `quantization` block twice, as `quantization` and `quantization_config`; the
//! engine reads the quant from `quantization`. A `quantization_config` naming a `quant_method` is another library's
//! checkpoint layout (transformers, ModelOpt), which a plugin serves only by name.

const std = @import("std");

/// Whether the root's `quantization_config` names a `quant_method` (JSON null is absent).
pub fn foreign(root: std.json.ObjectMap) bool {
    const qc = root.get("quantization_config") orelse return false;
    if (qc != .object) return false;
    const m = qc.object.get("quant_method") orelse return false;
    return m != .null;
}

test "quantization_config: mlx-lm's copy of the quant block is the pack's own; a quant_method is foreign" {
    const cases = [_]struct { []const u8, bool }{
        .{ "{}", false },
        .{ "{\"quantization\":{\"bits\":4},\"quantization_config\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\"}}", false },
        .{ "{\"quantization_config\":{\"quant_method\":null,\"bits\":4}}", false },
        .{ "{\"quantization_config\":{\"quant_method\":\"fp8\"}}", true },
        .{ "{\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"NVFP4\"}}", true },
    };
    for (cases) |c| {
        const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c[0], .{});
        defer v.deinit();
        try std.testing.expectEqual(c[1], foreign(v.value.object));
    }
}
