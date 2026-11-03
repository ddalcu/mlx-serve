//! Forced decode tails: the reasoning-budget bound (ThinkBound) and tool_choice enforcement (ToolForce).
const std = @import("std");
const testing = std.testing;


/// A thinking budget enforced while decoding: once `budget` tokens have been
/// generated inside the think block, `forced` (an early-stop sentence, the
/// pack's atomic closer, a separator) is committed through the model and the
/// answer follows. Markers are single token ids, so only packs whose think
/// tags are atomic tokens can be bound; `observe` walks the generated ids
/// incrementally and `due` asks whether the tail is an open block at budget.
pub const ThinkBound = struct {
    budget: u32,
    opener_id: ?u32,
    closer_id: u32,
    forced: []const u32,
    in_think: bool,
    count: u32 = 0,
    cursor: usize = 0,
    fired: bool = false,

    pub fn observe(self: *ThinkBound, ids: []const u32) void {
        while (self.cursor < ids.len) : (self.cursor += 1) {
            const id = ids[self.cursor];
            if (id == self.closer_id) {
                self.in_think = false;
            } else if (self.opener_id != null and id == self.opener_id.?) {
                self.in_think = true;
                self.count = 0;
            } else if (self.in_think) {
                self.count += 1;
            }
        }
    }

    pub fn due(self: *const ThinkBound) bool {
        return !self.fired and self.in_think and self.count >= self.budget;
    }
};

/// `tool_choice` that obliges a call: the family's tool-call opener is
/// committed once the think block closes, or at once when none is open. A
/// thought still open with a quarter of `max_tokens` left is closed for it, so
/// the call fits. `forced` = closer, separator, opener; `opener_at` indexes the opener.
pub const ToolForce = struct {
    closer_id: ?u32,
    forced: []const u32,
    opener_at: usize,
    in_think: bool,
    cursor: usize = 0,
    fired: bool = false,

    pub const Due = struct { tokens: []const u32, closes_thought: bool };

    /// The tokens to commit now, or null.
    pub fn due(self: *ToolForce, ids: []const u32, completion_tokens: u32, max_tokens: u32) ?Due {
        if (self.fired) return null;
        while (self.in_think and self.cursor < ids.len) : (self.cursor += 1) {
            if (ids[self.cursor] == self.closer_id) self.in_think = false;
        }
        if (!self.in_think) return .{ .tokens = self.forced[self.opener_at..], .closes_thought = false };
        if (completion_tokens +| max_tokens / 4 >= max_tokens) return .{ .tokens = self.forced, .closes_thought = true };
        return null;
    }
};

// ── tests ──


test "ToolForce: the opener after the thought closes, the closer too at the deadline, never twice" {
    const CLOSE: u32 = 11;
    const forced = [_]u32{ CLOSE, 20, 40, 41 }; // closer, separator, opener
    var tf = ToolForce{ .closer_id = CLOSE, .forced = &forced, .opener_at = 2, .in_think = true };
    try testing.expect(tf.due(&[_]u32{ 6, 7 }, 2, 100) == null);
    const after = tf.due(&[_]u32{ 6, 7, CLOSE }, 3, 100).?;
    try testing.expectEqualSlices(u32, &[_]u32{ 40, 41 }, after.tokens);
    try testing.expect(!after.closes_thought);
    tf.fired = true;
    try testing.expect(tf.due(&[_]u32{ 6, 7, CLOSE, 40 }, 4, 100) == null);

    // Still thinking with a quarter of max_tokens left: close it and call.
    var late = ToolForce{ .closer_id = CLOSE, .forced = &forced, .opener_at = 2, .in_think = true };
    try testing.expect(late.due(&[_]u32{6}, 74, 100) == null);
    const closing = late.due(&[_]u32{6}, 75, 100).?;
    try testing.expectEqualSlices(u32, &forced, closing.tokens);
    try testing.expect(closing.closes_thought);

    var open = ToolForce{ .closer_id = null, .forced = forced[2..], .opener_at = 0, .in_think = false };
    try testing.expectEqualSlices(u32, &[_]u32{ 40, 41 }, open.due(&[_]u32{}, 0, 100).?.tokens);
}

test "ThinkBound: counts only tokens inside the think block and fires at the budget" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, 31, CLOSE, 32 };
    var tb = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = false };

    // Content before any opener never counts.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5 });
    try testing.expect(!tb.due());

    // Opened by the model: the budget counts from the opener.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7 });
    try testing.expect(!tb.due());
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7, 8 });
    try testing.expect(tb.due());

    // A model that closed on its own is never forced.
    var closed = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    closed.observe(&[_]u32{ 6, 7, CLOSE, 8, 9, 10, 11 });
    try testing.expect(!closed.due());

    // Prompt-opened: the count starts at token 0 and the forced closer ends it.
    var po = ThinkBound{ .budget = 2, .opener_id = null, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    po.observe(&[_]u32{ 6, 7 });
    try testing.expect(po.due());
    po.fired = true;
    po.observe(&[_]u32{ 6, 7, 30, 31, CLOSE, 32, 40 });
    try testing.expect(!po.in_think);
    try testing.expect(!po.due());
}
