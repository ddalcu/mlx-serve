//! Clef's claims, caps and bills, MLX-free: a qwen3_5 pack told from a chat model by `joint_head_config.json` (a
//! marker claim outranks qwen3_5's type claim), the decision route, the weights resident.

const abi = @import("../../abi/media.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "clef",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{},
};

pub const model_types = [_][]const u8{"clef"};

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    _ = model_type;
    return if (dir.has("joint_head_config.json")) .by_marker else .no;
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
