//! The Linux and iOS builds, which have no mlx-stream: the EXL3 repack is refused by name at load.

const std = @import("std");
const mlx = @import("../mlx.zig");
const ModelConfig = @import("../model.zig").ModelConfig;

pub const built = false;

pub const Model = struct {
    pub const DraftRound = struct {
        tokens: []u32,
        accepted: u32,
        next_token: u32,
    };
};

fn refuse() error{MlxStreamNotBuilt} {
    @import("../log.zig").err("deepseek_v41: this EXL3 repack runs on mlx-stream, which only the macOS build includes\n", .{});
    return error.MlxStreamNotBuilt;
}

pub fn loadBytes(_: std.mem.Allocator, _: std.Io, _: *const ModelConfig) !u64 {
    return refuse();
}

pub fn contextLength(_: *const ModelConfig) u32 {
    return 0;
}

pub fn open(_: std.mem.Allocator, _: std.Io, _: mlx.mlx_stream, _: *const ModelConfig) !*Model {
    return refuse();
}

pub fn close(_: *Model) void {}
pub fn begin(_: *Model, _: []const u32, _: u32, _: u64) !u64 {
    return 0;
}
pub fn forward(_: *Model, _: []const u32, _: mlx.mlx_stream) !mlx.mlx_array {
    return error.MlxStreamNotBuilt;
}
pub fn position(_: *const Model) u64 {
    return 0;
}
pub fn blockSize(_: *const Model) u32 {
    return 0;
}
pub const SamplingParams = struct { temperature: f32 = 1.0, top_p: f32 = 1.0, top_k: u32 = 0, min_p: ?f32 = null, seed: u64 = 0 };
pub fn arm(_: *Model, _: ?SamplingParams) bool {
    return false;
}
pub fn round(_: *Model, _: std.mem.Allocator, _: u32, _: u32) !Model.DraftRound {
    return error.MlxStreamNotBuilt;
}
