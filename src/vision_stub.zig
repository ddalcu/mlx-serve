//! vision surface for builds without MLX.

const std = @import("std");
const mlx = @import("mlx_stub.zig");

pub const Error = error{MlxUnavailable};

pub const VisionEncoder = struct {
    s: mlx.mlx_stream = .{},

    pub fn init(_: std.mem.Allocator, _: anytype, _: anytype) anyerror!VisionEncoder {
        return Error.MlxUnavailable;
    }

    pub fn deinit(_: *VisionEncoder) void {}

    pub fn supportsAudio(_: *const VisionEncoder) bool {
        return false;
    }

    pub fn forward(_: *VisionEncoder, _: anytype) anyerror!mlx.mlx_array {
        return Error.MlxUnavailable;
    }

    pub fn forwardPatches(_: *VisionEncoder, _: anytype, _: u32, _: u32) anyerror!mlx.mlx_array {
        return Error.MlxUnavailable;
    }

    pub fn forwardVideoPatches(_: *VisionEncoder, _: anytype, _: u32, _: u32, _: u32) anyerror!mlx.mlx_array {
        return Error.MlxUnavailable;
    }

    pub fn forwardAudio(_: *VisionEncoder, _: anytype) anyerror!mlx.mlx_array {
        return Error.MlxUnavailable;
    }
};
