//! spec surface for builds without MLX.

const std = @import("std");

pub const Error = error{MlxUnavailable};
pub fn mtpCtxWithinLimit(max: u32, ctx_tokens: usize) bool {
    return max == 0 or ctx_tokens <= max;
}

pub const SidecarConfig = struct {
    block_size: u32 = 0,
    mask_token_id: u32 = 0,
    target_layer_ids: []const u32 = &.{},
};

pub const Sidecar = struct {
    config: SidecarConfig = .{},

    pub fn deinit(_: *Sidecar) void {}

    pub fn bind(_: *Sidecar, _: anytype) anyerror!void {
        return Error.MlxUnavailable;
    }

    pub fn m5NaxCostProfile(_: *const Sidecar, _: anytype) MtpCostProfile {
        return .generic;
    }
};

pub const Drafter = Sidecar;
pub const DFlash = Sidecar;
pub const MtpModel = Sidecar;

pub fn load(_: std.Io, _: std.mem.Allocator, _: []const u8, _: anytype) anyerror!*Sidecar {
    return Error.MlxUnavailable;
}

pub fn resolveInDirDrafter(_: std.Io, _: std.mem.Allocator, _: []const u8) ?[]u8 {
    return null;
}

pub fn resolveMtpSource(_: std.Io, _: std.mem.Allocator, _: []const u8) ?[]u8 {
    return null;
}

pub const BlockCap = struct { cap: u32 = 0, label: []const u8 = "no-mlx" };

pub fn blockCapForMachine(_: anytype) BlockCap {
    return .{};
}

pub fn adaptiveDepthCapForMachine() u32 {
    return 0;
}

pub const DrafterModel = Sidecar;
pub const DflashModel = Sidecar;
pub const DflashCtx = struct {
    cache: struct { step: usize = 0 } = .{},
    base_pos: usize = 0,

    pub fn init(_: std.mem.Allocator, _: anytype, _: usize) anyerror!DflashCtx {
        return Error.MlxUnavailable;
    }

    pub fn deinit(_: *DflashCtx) void {}

    pub fn absLen(_: *const DflashCtx) usize {
        return 0;
    }
};

pub const DEFAULT_BLOCK_SIZE: u32 = 4;
pub const DEFAULT_DEPTH: u32 = 3;
pub const MAX_DEPTH: u32 = 8;

pub const MtpCostProfile = enum { generic };

pub fn loadDrafter(_: std.Io, _: std.mem.Allocator, _: anytype, _: []const u8) anyerror!Sidecar {
    return Error.MlxUnavailable;
}

pub fn loadDflash(_: std.Io, _: std.mem.Allocator, _: anytype, _: []const u8) anyerror!Sidecar {
    return Error.MlxUnavailable;
}

pub fn loadMtp(_: std.Io, _: std.mem.Allocator, _: anytype, _: []const u8) anyerror!MtpModel {
    return Error.MlxUnavailable;
}

pub fn hasMtpHead(_: std.Io, _: std.mem.Allocator, _: []const u8) bool {
    return false;
}

pub fn probeIsDflash(_: std.Io, _: std.mem.Allocator, _: []const u8) bool {
    return false;
}

pub fn recommendedBlockSize(_: anytype) u32 {
    return 0;
}

pub fn wideVerifyLaneAvailable() bool {
    return false;
}

pub fn resolveBlockSize(_: u32, _: u32, _: bool, _: bool, _: u32) u32 {
    return 0;
}

pub var enabled: bool = false;
