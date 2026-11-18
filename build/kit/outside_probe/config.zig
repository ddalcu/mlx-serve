//! The outside probe's claim and parse: its own model type, an empty shell. MLX-free.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;

pub const model_types = [_][]const u8{"_outside_probe"};

pub const claims = kit.sdk.claimsByType(&model_types);

pub fn parse(_: *const abi.PackDir, _: abi.Str, _: abi.Str, _: ?*const abi.LayerSpan, _: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    out.* = .{};
    out.shell.model_type = abi.Str.of(model_types[0]);
    return .ok;
}

pub fn freeParsed(_: *abi.Parsed) callconv(.c) void {}
