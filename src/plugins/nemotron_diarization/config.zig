//! Nemotron 3 Diarization's claims, caps and bills, MLX-free: a `nemotron_diarization` config.json, no route of its
//! own (the `diarized_json` transcription loads it), every weight resident.

const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "nemotron_diarization",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/Nemotron-3-Diarization-8bit"},
};

pub const model_types = [_][]const u8{"nemotron_diarization"};

pub const claims = sdk.mediaClaimsByType(&model_types);

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

/// A clip's chunks run one at a time over a bounded FIFO and speaker cache: nothing measured beyond the load.
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}
