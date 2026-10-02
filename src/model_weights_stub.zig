//! model weights surface for builds without MLX.

const std = @import("std");

pub const Error = error{
    MlxUnavailable,
};

pub const Weights = struct {
    map: std.StringHashMap(@import("mlx_stub.zig").mlx_array),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(@import("mlx_stub.zig").mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        self.map.deinit();
        self.* = undefined;
    }

    pub fn count(_: *const Weights) u32 {
        return 0;
    }

    pub fn get(_: *const Weights, _: []const u8) ?@import("mlx_stub.zig").mlx_array {
        return null;
    }
};

pub fn resolveWeightPrefix(_: anytype, _: *const Weights) void {}

pub fn loadWeights(_: std.Io, _: std.mem.Allocator, _: []const u8) anyerror!Weights {
    return Error.MlxUnavailable;
}

pub fn loadWeightsSingleFile(_: std.mem.Allocator, _: []const u8) anyerror!Weights {
    return Error.MlxUnavailable;
}

pub fn loadWeightsWithVision(_: std.Io, _: std.mem.Allocator, _: []const u8) anyerror!Weights {
    return Error.MlxUnavailable;
}

pub fn reportF16Narrowing() void {}

pub fn loadSafetensorsFile(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!void {
    return Error.MlxUnavailable;
}

pub fn narrowsLoadedF16(_: []const u8, _: usize, _: anytype) bool {
    return false;
}

const testing = std.testing;

test "every safetensors loader refuses by name rather than returning an empty map" {
    var buf: [256]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const alloc = fba.allocator();
    try testing.expectError(Error.MlxUnavailable, loadWeightsSingleFile(alloc, "/x.safetensors"));
}

pub fn loadModelWeights(_: std.Io, _: std.mem.Allocator, _: []const u8, _: *const @import("model.zig").ModelConfig, _: bool) Error!Weights {
    return error.MlxUnavailable;
}
pub fn narrowHadamardPackTables(_: anytype, _: *Weights, _: anytype) Error!void {
    return error.MlxUnavailable;
}

pub const Spec = struct {};
pub fn parseExpertQuant(_: anytype) error{MlxUnavailable}!?Spec {
    return error.MlxUnavailable;
}
pub fn admitTopK(_: anytype) error{MlxUnavailable}!void {
    return error.MlxUnavailable;
}
