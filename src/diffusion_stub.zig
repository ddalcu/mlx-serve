//! diffusion surface for builds without MLX.

const std = @import("std");

pub const Error = error{MlxUnavailable};

pub const CanvasResult = struct {
    tokens: []u32 = &.{},
    steps: u32 = 0,
};

pub const Runner = struct {
    cancel_flag: ?*const std.atomic.Value(bool) = null,

    pub fn init(_: std.mem.Allocator, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!Runner {
        return Error.MlxUnavailable;
    }
    pub fn deinit(_: *Runner) void {}

    pub fn prefill(_: *Runner, _: []const u32) anyerror!void {
        return Error.MlxUnavailable;
    }

    pub fn nextCanvas(_: *Runner, _: std.mem.Allocator) anyerror!?CanvasResult {
        return Error.MlxUnavailable;
    }
};
