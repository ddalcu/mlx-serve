//! mlx cache surface for builds without MLX.

const std = @import("std");

pub const Error = error{MlxUnavailable};

pub const MIN_CANCELLED_COMMIT_TOKENS: usize = 256;

pub const DflashCommit = struct {
    cache: *anyopaque = undefined,
    base_pos: usize = 0,
};

pub const LookupResult = struct {
    matched: usize = 0,
    full_match: bool = false,
    dflash_base: ?usize = null,
    mtp_base: ?usize = null,
};

pub const HotPrefixCache = struct {
    pub fn spillIdleEntries(_: *HotPrefixCache, _: anytype) void {}
    pub fn dropQsaGapEntry(_: *HotPrefixCache) bool {
        return false;
    }
    pub fn resetSsmEntries(_: anytype) void {}
    pub fn releaseCheckout(_: *HotPrefixCache, _: usize, _: []const u8) void {}
    pub fn reclaimableBytes(_: *const HotPrefixCache) u64 {
        return 0;
    }
    pub const EntryDigest = struct { fingerprint: u64, len: u32, kv_bytes: u64 };
    pub fn digestsAlloc(_: *const HotPrefixCache, _: std.mem.Allocator) ![]EntryDigest {
        return &.{};
    }
    pub fn residentBytes(_: *const HotPrefixCache) u64 {
        return 0;
    }
    pub fn prefixFingerprint(_: []const u32) ?u64 {
        return null;
    }
    pub fn reclaimableFromDigests(_: []const EntryDigest, _: u64, _: ?u64) u64 {
        return 0;
    }

    disk: ?DiskTier = null,

    pub fn deinit(_: *HotPrefixCache) void {}

    pub fn shouldUse(_: anytype, _: bool) bool {
        return false;
    }

    pub fn initWithMem(_: std.mem.Allocator, _: anytype, _: anytype) ?HotPrefixCache {
        return null;
    }

    pub fn lookupAndRestore(_: *HotPrefixCache, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!LookupResult {
        return LookupResult{};
    }

    pub fn commit(_: *HotPrefixCache, _: anytype, _: anytype, _: anytype) anyerror!void {
        return Error.MlxUnavailable;
    }

    pub fn commitWithState(_: *HotPrefixCache, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!void {
        return Error.MlxUnavailable;
    }

    pub fn flushPendingDisk(_: *HotPrefixCache, _: anytype) void {}
};

pub const DEFAULT_CHUNK_TOKENS: usize = 256;

pub const DiskTier = struct {
    pub fn deinit(_: *DiskTier) void {}

    pub fn init(_: std.mem.Allocator, _: std.Io, _: []const u8, _: anytype, _: u64, _: usize) anyerror!DiskTier {
        return Error.MlxUnavailable;
    }
};

pub fn defaultBaseDir(_: std.mem.Allocator) anyerror![]u8 {
    return Error.MlxUnavailable;
}

pub fn modelFingerprint(_: std.mem.Allocator, _: std.Io, _: []const u8) anyerror![]u8 {
    return Error.MlxUnavailable;
}

pub fn warmEnvCaches() void {}

pub const MediaSpan = struct { start: u32, key: u64 };
pub fn ssdFirstActive(_: anytype, _: bool) bool {
    return false;
}
