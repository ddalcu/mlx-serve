//! ane surface for builds without MLX.

const std = @import("std");

pub const GATE_BASELINE_BYTES: u64 = 0;
pub const MAX_UNITS: usize = 2;
pub const MIN_CONTEXT_TOKENS: usize = 0;

pub var live_int8_bytes: std.atomic.Value(u64) = .init(0);
pub var live_layers: std.atomic.Value(u64) = .init(0);

pub fn anePrefillAllowed(_: anytype, _: anytype) bool {
    return false;
}

pub fn chipBrand() []const u8 {
    return "";
}

pub fn splitShare() f32 {
    return 0;
}

pub const AneUnit = struct {
    instance: u32 = 0,
    evals_ok: std.atomic.Value(u64) = .init(0),
    evals_failed: std.atomic.Value(u64) = .init(0),
};

pub const AneEngine = struct {
    units: []AneUnit = &.{},
    mode: enum { channel, row } = .channel,
    rows: u32 = 0,
    chunk_rows: u32 = 0,
    share: f32 = 0,
    int8_bytes: u64 = 0,
    evals: u64 = 0,
    eval_failures: u64 = 0,

    pub fn coveredLayers(_: *const AneEngine) usize {
        return 0;
    }

    pub fn coveredGdnLayers(_: *const AneEngine) usize {
        return 0;
    }
};

pub const MediaOffload = struct { image: bool = false, video: bool = false, audio: bool = false, share: ?f32 = null };
pub var media_offload: MediaOffload = .{};
pub fn explicitShareEnv() ?f32 {
    return null;
}
