//! A plugin's one declaration and its negotiation with the host (`negotiate`, which the conformance suite runs
//! against `sdk.host`); each kind's `of` checks the kind's interface.

const std = @import("std");

pub const Version = struct { major: u16, minor: u16 };

/// The `plugin` declaration of a plugin's root file. Comptime only: `provides` names the namespace that implements
/// each kind, and the registry builds the kinds' tables from them (`sdk.Arch.of` and its siblings).
pub const Plugin = struct {
    name: []const u8,
    /// The SDK the plugin was built against.
    api: Version,
    /// The MLX the plugin was tested on: `v` + its version (`v0.32.3`).
    mlx: []const u8,
    /// Registers nothing on graphs without the macOS-only sources (Linux, iOS).
    macos_only: bool = false,
    provides: Provides,
};

/// Any subset of the three kinds. A quant or an expert source is its arch's internal (bound at comptime), not a kind.
pub const Provides = struct {
    source: ?type = null,
    arch: ?type = null,
    engine: ?type = null,
};

/// What the host checks a plugin against: its SDK version and its MLX.
pub const Host = struct { api: Version, mlx: []const u8 };

pub const NegotiationError = error{
    /// A breaking SDK change between the plugin and the host. A newer minor on either side is compatible.
    ApiMajorMismatch,
    /// One MLX per process: bumping MLX is one change that bumps every plugin's pin.
    MlxPinMismatch,
};

/// The host's check of one plugin, as a pure function. The kinds' interfaces are checked by their `of`.
pub fn negotiate(comptime p: Plugin, host: Host) NegotiationError!void {
    if (p.api.major != host.api.major) return error.ApiMajorMismatch;
    if (!std.mem.eql(u8, p.mlx, host.mlx)) return error.MlxPinMismatch;
}

const testing = std.testing;

test "sdk negotiation: a major mismatch and another MLX are refused by name; minors are compatible both ways" {
    const host: Host = .{ .api = .{ .major = 1, .minor = 3 }, .mlx = "v0.32.3" };
    const ok: Plugin = .{ .name = "ok", .api = .{ .major = 1, .minor = 0 }, .mlx = "v0.32.3", .provides = .{} };
    try negotiate(ok, host);
    try negotiate(.{ .name = "newer-minor", .api = .{ .major = 1, .minor = 9 }, .mlx = "v0.32.3", .provides = .{} }, host);
    try testing.expectError(error.ApiMajorMismatch, negotiate(.{ .name = "old", .api = .{ .major = 0, .minor = 7 }, .mlx = "v0.32.3", .provides = .{} }, host));
    try testing.expectError(error.ApiMajorMismatch, negotiate(.{ .name = "new", .api = .{ .major = 2, .minor = 0 }, .mlx = "v0.32.3", .provides = .{} }, host));
    try testing.expectError(error.MlxPinMismatch, negotiate(.{ .name = "mlx", .api = .{ .major = 1, .minor = 0 }, .mlx = "v0.31.2", .provides = .{} }, host));
}

test "sdk negotiation: the pin is compared whole (no prefix, no suffix, no case folding); the major is checked first" {
    const host: Host = .{ .api = .{ .major = 1, .minor = 0 }, .mlx = "v0.32.3" };
    const pins = [_][]const u8{ "v0.32", "v0.32.3-rc1", "V0.32.3", "0.32.3", "", "v0.32.3 " };
    inline for (pins) |pin| try testing.expectError(error.MlxPinMismatch, negotiate(.{ .name = "p", .api = .{ .major = 1, .minor = 0 }, .mlx = pin, .provides = .{} }, host));
    // a plugin wrong on both counts is refused for the SDK, the breaking change
    try testing.expectError(error.ApiMajorMismatch, negotiate(.{ .name = "both", .api = .{ .major = 3, .minor = 0 }, .mlx = "v0.1.0", .provides = .{} }, host));
    // the minor's whole range is compatible
    try negotiate(.{ .name = "max-minor", .api = .{ .major = 1, .minor = std.math.maxInt(u16) }, .mlx = "v0.32.3", .provides = .{} }, host);
    try negotiate(.{ .name = "old-minor", .api = .{ .major = 1, .minor = 0 }, .mlx = "v0.32.3", .provides = .{} }, .{ .api = .{ .major = 1, .minor = std.math.maxInt(u16) }, .mlx = "v0.32.3" });
}
