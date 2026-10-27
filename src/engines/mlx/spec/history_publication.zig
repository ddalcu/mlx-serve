//! Engine-emitted history, not speculative commit counts or transport delivery.
const std = @import("std");

pub const Outcome = enum { completed, cancelled, failed };

/// Borrows the generator's committed stream. Only the early prefill token can
/// precede that backing; retain that scalar, never a second output allocation.
pub const Ledger = struct {
    count: usize = 0,
    first: [1]u32 = .{0},
    valid: bool = true,
    finalized: bool = false,

    pub fn emitted(self: *Ledger, token: u32, generated: []const u32) void {
        if (self.finalized or !self.valid) return;
        if (self.count == 0) self.first[0] = token;
        if (self.count < generated.len) {
            if (generated[self.count] != token or generated[0] != self.first[0]) self.valid = false;
        } else if (self.count != 0 or generated.len != 0) {
            self.valid = false;
        }
        self.count += 1;
    }

    /// A failed output append must not permit a later token to fill its hole.
    pub fn failedEmission(self: *Ledger) void {
        self.valid = false;
    }

    /// One terminal attempt, including an append failure. Failed generations
    /// and retry teardown are not completed training documents.
    pub fn finish(self: *Ledger, generated: []const u32, outcome: Outcome) !?[]const u32 {
        if (self.finalized) return null;
        self.finalized = true;
        if (outcome == .failed or self.count == 0) return null;
        if (!self.valid) return error.InvalidPublishedPrefix;
        if (generated.len == 0 and self.count == 1) return &self.first;
        if (generated.len < self.count or generated[0] != self.first[0]) return error.InvalidPublishedPrefix;
        return generated[0..self.count];
    }
};

test "publication excludes verified tail and records stop callback token" {
    var ledger: Ledger = .{};
    const committed = [_]u32{ 31, 32, 33, 34 };
    ledger.emitted(31, &committed);
    // The sink processes this event before returning stop.
    ledger.emitted(32, &committed);
    try std.testing.expectEqualSlices(u32, &.{ 31, 32 }, (try ledger.finish(&committed, .cancelled)).?);
    try std.testing.expect((try ledger.finish(&committed, .completed)) == null);
}

test "early token cancellation and later committed backing" {
    var early: Ledger = .{};
    early.emitted(41, &.{});
    try std.testing.expectEqualSlices(u32, &.{41}, (try early.finish(&.{}, .cancelled)).?);
    var backed: Ledger = .{};
    backed.emitted(41, &.{});
    // The duplicate early token is swallowed by Slot, not emitted twice.
    backed.emitted(42, &.{ 41, 42, 43 });
    try std.testing.expectEqualSlices(u32, &.{ 41, 42 }, (try backed.finish(&.{ 41, 42, 43 }, .completed)).?);
}

test "publication rejects holes and mismatched committed backing" {
    var hole: Ledger = .{};
    hole.emitted(1, &.{ 1, 2, 3 });
    hole.failedEmission();
    hole.emitted(3, &.{ 1, 2, 3 });
    try std.testing.expectError(error.InvalidPublishedPrefix, hole.finish(&.{ 1, 2, 3 }, .completed));
    var mismatch: Ledger = .{};
    mismatch.emitted(1, &.{});
    try std.testing.expectError(error.InvalidPublishedPrefix, mismatch.finish(&.{2}, .completed));
}

test "failed generation never publishes its partial output" {
    var ledger: Ledger = .{};
    ledger.emitted(1, &.{1});
    try std.testing.expect((try ledger.finish(&.{1}, .failed)) == null);
    try std.testing.expect((try ledger.finish(&.{1}, .cancelled)) == null);
}
