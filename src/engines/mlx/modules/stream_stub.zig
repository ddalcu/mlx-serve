//! The Linux and iOS builds, which have no mlx-stream: a pack it serves is refused by name at load.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;

pub const built = false;

pub fn setForeignBytes(_: u64) void {}

pub const Model = struct {
    pub const DraftRound = struct {
        tokens: []u32,
        accepted: u32,
        next_token: u32,
    };
};

fn refuse(config: *const ModelConfig) error{MlxStreamNotBuilt} {
    @import("../../../sdk/util/log.zig").err("{s}: this pack streams its experts on mlx-stream, which only the macOS build includes\n", .{config.model_type});
    return error.MlxStreamNotBuilt;
}

pub fn loadBytes(_: std.mem.Allocator, _: std.Io, config: *const ModelConfig) !u64 {
    return refuse(config);
}

pub fn contextLength(_: *const ModelConfig) u32 {
    return 0;
}

pub fn noteMaxOutput(_: []const u8, _: u64) void {}

pub fn open(_: std.mem.Allocator, _: std.Io, _: mlx.mlx_stream, config: *const ModelConfig) !*Model {
    return refuse(config);
}

pub fn close(_: *Model) void {}
pub fn begin(_: *Model, _: []const u32, _: u32, _: u64) !u64 {
    return 0;
}
pub fn end(_: *Model) void {}
pub fn forward(_: *Model, _: []const u32, _: mlx.mlx_stream) !mlx.mlx_array {
    return error.MlxStreamNotBuilt;
}
pub fn position(_: *const Model) u64 {
    return 0;
}
pub fn blockSize(_: *const Model) u32 {
    return 0;
}
pub fn laneName(_: *const Model) []const u8 {
    return "serial";
}
pub const SamplingParams = struct { temperature: f32 = 1.0, top_p: f32 = 1.0, top_k: u32 = 0, min_p: ?f32 = null, seed: u64 = 0 };
pub fn arm(_: *Model, _: ?SamplingParams) bool {
    return false;
}
pub fn round(_: *Model, _: std.mem.Allocator, _: u32, _: u32) !Model.DraftRound {
    return error.MlxStreamNotBuilt;
}
pub fn laneReceipt(_: *Model, _: []u8) ?[]const u8 {
    return null;
}
