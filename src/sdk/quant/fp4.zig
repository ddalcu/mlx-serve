//! The fp4 formats every fp4 kernel takes: MXFP4 (e8m0 scale per 32) and NVFP4 (e4m3 scale per 16, ModelOpt's
//! optional per-expert global scales beside it), and the one Metal header that decodes either's group scale
//! (`kernels/common/fp4_scale.metal`). A kernel takes the format as its group size (`GS`), never as a second kernel.
const kernelSource = @import("../../mlx/kernels.zig").source;
const QuantMode = @import("params.zig").QuantMode;

pub const Format = enum {
    nvfp4,
    mxfp4,

    /// The scale group the format fixes.
    pub fn group(f: Format) u32 {
        return switch (f) {
            .nvfp4 => 16,
            .mxfp4 => 32,
        };
    }

    pub fn mode(f: Format) QuantMode {
        return switch (f) {
            .nvfp4 => .nvfp4,
            .mxfp4 => .mxfp4,
        };
    }
};

/// The fp4 format of a bank at `mode` and `group_size`; null for any other pair (affine, mxfp8, an fp4 mode at another
/// group size).
pub fn of(mode: QuantMode, group_size: u32) ?Format {
    return switch (mode) {
        .nvfp4 => if (group_size == 16) .nvfp4 else null,
        .mxfp4 => if (group_size == 32) .mxfp4 else null,
        else => null,
    };
}

/// `mlxserve_fp4_scale_x2m8<GS>` / `mlxserve_fp4_scale_of<GS>`: an e8m0 or e4m3 group scale as MLX reads it.
pub const SCALE_HEADER = kernelSource("kernels/common/fp4_scale.metal");

const std = @import("std");

test "fp4 formats: the (mode, group) pairs and nothing else" {
    try std.testing.expectEqual(Format.nvfp4, of(.nvfp4, 16).?);
    try std.testing.expectEqual(Format.mxfp4, of(.mxfp4, 32).?);
    try std.testing.expect(of(.nvfp4, 32) == null);
    try std.testing.expect(of(.mxfp4, 16) == null);
    try std.testing.expect(of(.affine, 32) == null);
    try std.testing.expect(of(.mxfp8, 32) == null);
    for ([_]Format{ .nvfp4, .mxfp4 }) |f| try std.testing.expectEqual(f, of(f.mode(), f.group()).?);
}
