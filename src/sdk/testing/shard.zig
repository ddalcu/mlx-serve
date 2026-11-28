//! This test process's shard: a `sharded:` test runs in every one (build/test_runner.zig) and takes every n-th of
//! its own work from k.

const std = @import("std");
const env = @import("../../env.zig");

pub const Shard = struct {
    k: usize = 0,
    n: usize = 1,

    pub fn owns(self: Shard, i: usize) bool {
        return i % self.n == self.k;
    }
};

/// `MLX_SERVE_TEST_SHARD=k/n`; unset or malformed is the whole run.
pub fn current() Shard {
    return parse(env.span(.MLX_SERVE_TEST_SHARD) orelse return .{});
}

fn parse(raw: []const u8) Shard {
    const slash = std.mem.indexOfScalar(u8, raw, '/') orelse return .{};
    const k = std.fmt.parseUnsigned(usize, raw[0..slash], 10) catch return .{};
    const n = std.fmt.parseUnsigned(usize, raw[slash + 1 ..], 10) catch return .{};
    return if (n == 0 or k >= n) .{} else .{ .k = k, .n = n };
}

test "test shard: the n shards own every item exactly once, and a malformed value is the whole run" {
    for ([_][]const u8{ "", "1", "a/4", "0/0", "4/4" }) |raw| try std.testing.expectEqual(Shard{}, parse(raw));
    for (0..13) |i| {
        var owners: usize = 0;
        for ([_][]const u8{ "0/4", "1/4", "2/4", "3/4" }) |raw| owners += @intFromBool(parse(raw).owns(i));
        try std.testing.expectEqual(@as(usize, 1), owners);
    }
}
