//! Decline the MLX GGUF path so embedded llama.cpp owns GGUF models.
const std = @import("std");
const mlx = @import("mlx_stub.zig");

pub var enabled: bool = false;
pub const Sidecar = enum { config, tokenizer, tokenizer_config, generation_config };

pub fn servablePath(_: std.Io, _: std.mem.Allocator, _: []const u8) ?[]u8 {
    return null;
}

pub fn sidecar(_: std.Io, _: std.mem.Allocator, _: []const u8, _: Sidecar) !?[]u8 {
    return null;
}

pub fn weightBytes(_: std.Io, _: []const u8) ?u64 {
    return null;
}

pub fn loadWeights(_: std.Io, _: std.mem.Allocator, _: []const u8, _: *std.StringHashMap(mlx.mlx_array)) !bool {
    return false;
}
