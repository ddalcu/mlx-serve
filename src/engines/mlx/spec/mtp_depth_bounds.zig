//! `--mtp-min-depth` / `--mtp-max-depth`: the range every MTP draft-depth decision stays inside.

const std = @import("std");

pub const Bounds = struct {
    min: u32 = 1,
    /// 0 = no flag: the planner's own cap (cost profile, per-silicon row) applies.
    max: u32 = 0,

    /// The one depth every round drafts when both flags name it.
    pub fn pinned(self: Bounds) ?u32 {
        return if (self.max != 0 and self.min == self.max) self.min else null;
    }
};

/// Process-wide, set once from the launch flags before a model loads.
pub var active: Bounds = .{};

pub const deprecated_flag_message = "--mtp-depth is now --mtp-max-depth; the old spelling is still accepted with the same meaning";

pub fn parseDepth(text: []const u8, max_depth: u32) error{NotADepth}!u32 {
    const n = std.fmt.parseInt(u32, text, 10) catch return error.NotADepth;
    if (n < 1 or n > max_depth) return error.NotADepth;
    return n;
}

/// The deprecated `--mtp-depth`: any integer, clamped into 1..max_depth as it always was.
pub fn parseClamped(text: []const u8, max_depth: u32) error{NotADepth}!u32 {
    const n = std.fmt.parseInt(u32, text, 10) catch return error.NotADepth;
    return std.math.clamp(n, 1, max_depth);
}

pub fn validate(b: Bounds) error{MinAboveMax}!void {
    if (b.max != 0 and b.min > b.max) return error.MinAboveMax;
}

/// A cap the planner chose on its own, lifted to the explicit floor.
pub fn liftCap(cap: u32, b: Bounds, max_depth: u32) u32 {
    return @min(if (b.max != 0) @min(b.max, max_depth) else max_depth, @max(cap, b.min));
}

/// The floor a round can honour: a head count or verify budget below `--mtp-min-depth` wins.
pub fn floorFor(b: Bounds, cap: u32) u32 {
    return @min(b.min, @max(cap, 1));
}

pub fn clampWidth(width: u32, lo: u32, hi: u32) u32 {
    return @min(@max(width, @min(lo, hi)), hi);
}

/// The width the grouped planner gives a row of cap `cap`: a pinned depth for every row, else the
/// planner's own choice held to the range. Width 0 is a plain tick, not a depth, and stays.
pub fn plannerWidth(width: u32, cap: u32, b: Bounds) u32 {
    if (b.pinned()) |d| return @min(d, cap);
    if (width == 0) return 0;
    const hi = lookupCap(cap, b);
    return clampWidth(width, floorFor(b, hi), hi);
}

/// The widest prompt-lookup draft: its own cap, held to an explicit `--mtp-max-depth`.
pub fn lookupCap(base: u32, b: Bounds) u32 {
    return if (b.max != 0) @min(base, b.max) else base;
}

test "mtp depth bounds: parseDepth takes 1..max_depth and nothing else" {
    try std.testing.expectEqual(@as(u32, 1), try parseDepth("1", 8));
    try std.testing.expectEqual(@as(u32, 8), try parseDepth("8", 8));
    try std.testing.expectError(error.NotADepth, parseDepth("0", 8));
    try std.testing.expectError(error.NotADepth, parseDepth("9", 8));
    try std.testing.expectError(error.NotADepth, parseDepth("-1", 8));
    try std.testing.expectError(error.NotADepth, parseDepth("three", 8));
}

test "mtp depth bounds: the deprecated --mtp-depth clamps any integer into range" {
    try std.testing.expectEqual(@as(u32, 1), try parseClamped("0", 8));
    try std.testing.expectEqual(@as(u32, 5), try parseClamped("5", 8));
    try std.testing.expectEqual(@as(u32, 8), try parseClamped("12", 8));
    try std.testing.expectError(error.NotADepth, parseClamped("three", 8));
    try std.testing.expectError(error.NotADepth, parseClamped("-1", 8));
}

test "mtp depth bounds: min above an explicit max is refused, min alone is not" {
    try validate(.{ .min = 2, .max = 7 });
    try validate(.{ .min = 4, .max = 4 });
    try validate(.{ .min = 7, .max = 0 });
    try std.testing.expectError(error.MinAboveMax, validate(.{ .min = 5, .max = 4 }));
}

test "mtp depth bounds: only equal explicit flags pin a depth" {
    try std.testing.expectEqual(@as(?u32, 3), (Bounds{ .min = 3, .max = 3 }).pinned());
    try std.testing.expectEqual(@as(?u32, null), (Bounds{ .min = 2, .max = 7 }).pinned());
    try std.testing.expectEqual(@as(?u32, null), (Bounds{ .min = 3, .max = 0 }).pinned());
    try std.testing.expectEqual(@as(?u32, null), (Bounds{}).pinned());
}

test "mtp depth bounds: a floor above the planner's own cap lifts it, the default changes nothing" {
    try std.testing.expectEqual(@as(u32, 6), liftCap(6, .{}, 8));
    try std.testing.expectEqual(@as(u32, 7), liftCap(6, .{ .min = 7 }, 8));
    try std.testing.expectEqual(@as(u32, 6), liftCap(6, .{ .min = 2 }, 8));
    try std.testing.expectEqual(@as(u32, 4), liftCap(6, .{ .max = 4 }, 8));
    try std.testing.expectEqual(@as(u32, 8), liftCap(6, .{ .min = 9 }, 8));
}

test "mtp depth bounds: a head count below the floor wins over the floor" {
    try std.testing.expectEqual(@as(u32, 3), floorFor(.{ .min = 5 }, 3));
    try std.testing.expectEqual(@as(u32, 2), floorFor(.{ .min = 2 }, 6));
    try std.testing.expectEqual(@as(u32, 1), floorFor(.{}, 6));
}

test "mtp depth bounds: clampWidth holds a width inside lo..hi" {
    try std.testing.expectEqual(@as(u32, 2), clampWidth(1, 2, 7));
    try std.testing.expectEqual(@as(u32, 7), clampWidth(8, 2, 7));
    try std.testing.expectEqual(@as(u32, 5), clampWidth(5, 2, 7));
    try std.testing.expectEqual(@as(u32, 3), clampWidth(6, 5, 3));
}

test "mtp depth bounds: the grouped planner's widths stay in range, a plain tick stays plain, a pin reaches every row" {
    const b = Bounds{ .min = 2, .max = 7 };
    try std.testing.expectEqual(@as(u32, 2), plannerWidth(1, 7, b));
    try std.testing.expectEqual(@as(u32, 4), plannerWidth(4, 7, b));
    try std.testing.expectEqual(@as(u32, 7), plannerWidth(8, 8, b));
    try std.testing.expectEqual(@as(u32, 5), plannerWidth(9, 5, b));
    try std.testing.expectEqual(@as(u32, 0), plannerWidth(0, 7, b));
    try std.testing.expectEqual(@as(u32, 1), plannerWidth(1, 7, .{}));
    const pin = Bounds{ .min = 4, .max = 4 };
    try std.testing.expectEqual(@as(u32, 4), plannerWidth(0, 6, pin));
    try std.testing.expectEqual(@as(u32, 4), plannerWidth(2, 6, pin));
    try std.testing.expectEqual(@as(u32, 3), plannerWidth(2, 3, pin));
    try std.testing.expectEqual(@as(u32, 0), plannerWidth(0, 0, pin));
}

test "mtp depth bounds: a lookup draft is held to an explicit max only" {
    try std.testing.expectEqual(@as(u32, 8), lookupCap(8, .{}));
    try std.testing.expectEqual(@as(u32, 8), lookupCap(8, .{ .min = 3 }));
    try std.testing.expectEqual(@as(u32, 4), lookupCap(8, .{ .max = 4 }));
    try std.testing.expectEqual(@as(u32, 7), lookupCap(7, .{ .max = 8 }));
}
