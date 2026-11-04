//! Laya's claims, caps and bills, MLX-free: a checkpoint with no root config.json, told by the two configs every Laya
//! export carries; the decision route; the weights resident.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "laya",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"aac6fef/laya-multilingual-mlx"},
};

pub const model_types = [_][]const u8{"laya"};

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    _ = model_type;
    return if (dir.has("rl_agent_config.json") and dir.has("encoder/config.json")) .by_marker else .no;
}

/// No completeness marker: the pack counts as discovered (legacy had none).
pub const required_file = "";

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{};
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    return bill.safetensorsIn(dir, "");
}

pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}
