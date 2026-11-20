//! The decode's expert lookahead: at each predicting layer call the stream reads ahead the first `base_budget`
//! experts of the next layer's predicted ranking (each verify row's top 8 on the next layer's gate over this layer's
//! input, ordered by rank over the rows, then score) that the next layer does not hold, and, while a credit lasts, up
//! to `max_extra` more of the same ranking. The stream is sized for the lookahead's widest budget (4: eight staging
//! slots, 51 MiB more in the bill); the credit grows `per_round` / (layers - 1) at each predicting call (about
//! `per_round` a round) and spends one per record read past the budget, at most `max_extra` held. The extra records go
//! out behind the budget's in the same step, so a staging-slot shortfall skips an extra record, never a budget one.
//! Exact: the lookahead only reads ahead; the served bytes are the same.
//!
//! The extra records cost nothing measurable and are hardly read (M5 Max, a code-ctx16384 card then greedy: card +0.17%,
//! greedy +0.04%, the texts the same bytes): the speculative class's two threads are busy with the budget's records,
//! so the extras wait queued behind them.

const std = @import("std");

/// The records read ahead a layer call (the streamer's own constant).
pub const base_budget: u32 = 2;
/// Records a round read ahead past the per-call budget.
pub const per_round: u32 = 10;
/// The most records past the budget a layer call reads (the selector's `max_budget` 4 less `base_budget`).
pub const max_extra: u32 = 2;

/// A layer call's credit, held by the stream: `grow` at each predicting call, `take` the records past the budget the
/// call may read, `spend` the ones it read.
pub const Credit = struct {
    /// Credit grown a predicting call (N / (layers - 1)).
    rate: f32 = 0,
    held: f32 = 0,

    pub fn of(n: u32, layers: usize) Credit {
        return .{ .rate = if (n == 0 or layers < 2) 0 else @as(f32, @floatFromInt(n)) / @as(f32, @floatFromInt(layers - 1)) };
    }
    /// The call's records past the budget, at most `room` (the selector's room past the budget).
    pub fn take(c: *Credit, room: u32) u32 {
        if (c.rate == 0) return 0;
        c.held = @min(c.held + c.rate, @as(f32, @floatFromInt(max_extra)));
        return @min(room, @as(u32, @intFromFloat(@floor(c.held))));
    }
    /// The call read `n` records past the budget.
    pub fn spend(c: *Credit, n: u32) void {
        c.held = @max(c.held - @as(f32, @floatFromInt(n)), 0);
    }
};

const testing = std.testing;

test "lookahead wide credit: about N a round over the predicting calls, at most 2 a call, spent by the reads" {
    // 10 a round over 43 layers (42 predicting calls): 10 records past the budget a round when every one is read.
    var c = Credit.of(10, 43);
    var taken: u32 = 0;
    for (0..42) |_| {
        const n = c.take(2);
        c.spend(n);
        taken += n;
    }
    try testing.expect(taken >= 9 and taken <= 10);
    // Unread credit is held up to 2, no more.
    var h = Credit.of(10, 43);
    for (0..100) |_| _ = h.take(2);
    try testing.expectEqual(@as(u32, 2), h.take(2));
    try testing.expectEqual(@as(u32, 1), h.take(1));
    // Off: nothing past the budget.
    var z = Credit.of(0, 43);
    try testing.expectEqual(@as(u32, 0), z.take(2));
}
