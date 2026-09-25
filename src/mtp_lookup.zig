//! Prompt-lookup drafts for the MTP round. When the last committed tokens plus
//! the round's first token occurred earlier in the prompt or output, the tokens
//! that followed make a draft with no head forward; `gate` decides when that
//! beats the MTP chain. Only committed tokens are indexed.

const std = @import("std");

pub const NGRAM: usize = 4;
/// Shorter context matches land too few drafts to pay for their verify.
pub const MIN_SUFFIX: u32 = 8;
/// Keeps an ordinary lookup's verify at S <= 8, inside the fused paths.
pub const MAX_DRAFT: u32 = 7;
/// S=16 verifies disproportionately slower than S=15.
pub const MAX_DRAFT_STRONG: u32 = 14;
pub const STRONG_SUFFIX: u32 = 32;
const SUFFIX_CAP: u32 = 64;
/// Per-draft MTP head cost, in `verifyCost` units; lookup drafts are free.
const MTP_DRAFT_COST: f32 = 0.75;

/// Relative verify cost over `rows` positions: linear, with a step where verify
/// leaves the fused MoE rows and GDN recurSeq paths. Fitted on an M5 Ultra; the
/// gate uses it only where `Costs` has no measurement yet.
fn verifyCost(rows: u32) f32 {
    const r: f32 = @floatFromInt(rows);
    if (rows <= 8) return 11.4 + 2.6 * (r - 1);
    return 37.0 + 1.8 * (r - 9);
}

/// Fitted MTP round cost at `width` drafts, in `verifyCost` units.
fn mtpFitCost(width: u32) f32 {
    return verifyCost(width + 1) + MTP_DRAFT_COST * @as(f32, @floatFromInt(width));
}

/// Round ms measured on this machine and model (`round_cost.Table`), null until measured.
pub const Costs = struct {
    /// An MTP round at the chain's width.
    mtp_ms: ?f32 = null,
    /// A lookup round, by draft count.
    lookup_ms: [MAX_DRAFT_STRONG + 1]?f32 = @splat(null),
};

/// Accepted drafts per round, smoothed.
pub fn emaStep(ema: f32, accepted: u32) f32 {
    return 0.7 * ema + 0.3 * @as(f32, @floatFromInt(accepted));
}

/// Pulls the lookup EMA back toward its prior on MTP rounds, so lookups get retried.
pub fn driftStep(ema: f32) f32 {
    return 0.98 * ema + 0.02 * @as(f32, @floatFromInt(MAX_DRAFT));
}

pub const Match = struct {
    /// Borrowed from the index.
    draft: []const u32,
    /// Tokens the two sites agree on going back, n-gram included, capped.
    suffix: u32,
};

pub const Index = struct {
    allocator: std.mem.Allocator,
    toks: std.ArrayList(u32) = .empty,
    /// n-gram -> position of the token after its latest occurrence.
    next: std.AutoHashMapUnmanaged(u128, u32) = .empty,

    pub fn init(allocator: std.mem.Allocator) Index {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Index) void {
        self.toks.deinit(self.allocator);
        self.next.deinit(self.allocator);
    }

    pub fn extend(self: *Index, ids: []const u32) !void {
        try self.toks.ensureUnusedCapacity(self.allocator, ids.len);
        for (ids) |id| {
            self.toks.appendAssumeCapacity(id);
            const n = self.toks.items.len;
            if (n <= NGRAM) continue;
            const g = self.toks.items[n - 1 - NGRAM .. n - 1];
            try self.next.put(self.allocator, key(g[0], g[1], g[2], g[3]), @intCast(n - 1));
        }
    }

    /// Continuation after the latest earlier occurrence of the last NGRAM-1
    /// committed tokens plus `t1` (not yet committed).
    pub fn match(self: *const Index, t1: u32, max_draft: u32) ?Match {
        const toks = self.toks.items;
        if (toks.len < NGRAM - 1 or max_draft == 0) return null;
        const t = toks[toks.len - (NGRAM - 1) ..];
        const p: usize = self.next.get(key(t[0], t[1], t[2], t1)) orelse return null;
        var suffix: u32 = 0;
        var a: usize = p;
        var b: usize = toks.len + 1;
        while (suffix < SUFFIX_CAP and a > 0) : (suffix += 1) {
            a -= 1;
            b -= 1;
            if (toks[a] != (if (b == toks.len) t1 else toks[b])) break;
        }
        return .{ .draft = toks[p..@min(p + max_draft, toks.len)], .suffix = suffix };
    }

    fn key(a: u32, b: u32, c: u32, d: u32) u128 {
        return (@as(u128, a) << 96) | (@as(u128, b) << 64) | (@as(u128, c) << 32) | d;
    }
};

/// Draft length for a match, or 0 to run MTP: sized to what lookups have been
/// landing, or to the cap right after one landed every draft (`streak`), and
/// taken only when it promises more tokens per cost than an MTP chain of
/// `mtp_width` drafts. Never past `remaining` or the draft itself.
pub fn gate(m: ?Match, remaining: u32, lookup_ema: f32, mtp_ema: f32, mtp_width: u32, streak: bool, costs: Costs) u32 {
    const got = m orelse return 0;
    if (got.suffix < MIN_SUFFIX) return 0;
    const cap = if (got.suffix >= STRONG_SUFFIX) MAX_DRAFT_STRONG else MAX_DRAFT;
    const sized: u32 = if (streak) cap else @as(u32, @intFromFloat(@ceil(@max(lookup_ema, 0)))) + 2;
    const k = @min(@min(cap, sized), @min(@as(u32, @intCast(got.draft.len)), remaining));
    if (k == 0) return 0;
    const kf: f32 = @floatFromInt(k);
    const lookup_fit = verifyCost(k + 1);
    const mtp_fit = mtpFitCost(mtp_width);
    // A measured side gives this machine's ms per fit unit; a side not yet measured is its fit in that unit.
    const unit: f32 = if (costs.mtp_ms) |ms| ms / mtp_fit else if (costs.lookup_ms[k]) |ms| ms / lookup_fit else 1;
    const lookup_cost = costs.lookup_ms[k] orelse lookup_fit * unit;
    const mtp_cost = costs.mtp_ms orelse mtp_fit * unit;
    const lookup_rate = ((if (streak) kf else @min(lookup_ema, kf)) + 1) / lookup_cost;
    return if (lookup_rate > (mtp_ema + 1) / mtp_cost) k else 0;
}

const testing = std.testing;

test "lookup: latest earlier continuation, keyed on the tail plus t1, with how far back it agrees" {
    var idx = Index.init(testing.allocator);
    defer idx.deinit();
    // Two earlier "1 2 3 4" sites; the later one agrees back through "0 11 12 13".
    try idx.extend(&.{ 1, 2, 3, 4, 10, 50, 0, 11, 12, 13, 1, 2, 3, 4, 77, 5 });
    try idx.extend(&.{ 60, 0, 11, 12, 13, 1, 2, 3 });
    const m = idx.match(4, 3) orelse return error.NoMatch;
    try testing.expectEqualSlices(u32, &.{ 77, 5, 60 }, m.draft);
    try testing.expectEqual(@as(u32, 8), m.suffix);
    try testing.expect(idx.match(9, 3) == null);
}

test "lookup gate: short matches, weak lookups and exhausted budgets run MTP" {
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const strong = Match{ .draft = &d, .suffix = STRONG_SUFFIX };
    const ordinary = Match{ .draft = &d, .suffix = MIN_SUFFIX };
    try testing.expectEqual(@as(u32, 0), gate(.{ .draft = &d, .suffix = MIN_SUFFIX - 1 }, 100, 14, 0, 6, true, .{}));
    try testing.expectEqual(@as(u32, 0), gate(strong, 100, 3.5, 4.6, 6, false, .{}));
    try testing.expectEqual(@as(u32, 0), gate(ordinary, 100, 4.0, 5.3, 6, false, .{}));
    try testing.expectEqual(@as(u32, 0), gate(strong, 0, 14, 0, 6, true, .{}));
}

test "lookup gate: a landing streak drafts to the cap; otherwise the draft tracks the EMA" {
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const strong = Match{ .draft = &d, .suffix = STRONG_SUFFIX };
    try testing.expectEqual(MAX_DRAFT_STRONG, gate(strong, 100, 8.4, 4.6, 6, true, .{}));
    try testing.expectEqual(MAX_DRAFT, gate(.{ .draft = &d, .suffix = MIN_SUFFIX }, 100, 8.4, 4.6, 6, true, .{}));
    try testing.expectEqual(@as(u32, 11), gate(strong, 100, 9.0, 1.0, 3, false, .{}));
    try testing.expectEqual(@as(u32, 5), gate(strong, 5, 9.0, 1.0, 3, true, .{}));
}

test "lookup gate: one measured side only sets the unit, so the answers match the cold gate" {
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const strong = Match{ .draft = &d, .suffix = STRONG_SUFFIX };
    const ordinary = Match{ .draft = &d, .suffix = MIN_SUFFIX };
    const cases = [_]struct { m: Match, le: f32, me: f32, w: u32, s: bool }{
        .{ .m = strong, .le = 3.5, .me = 4.6, .w = 6, .s = false },
        .{ .m = ordinary, .le = 4.0, .me = 5.3, .w = 6, .s = false },
        .{ .m = strong, .le = 9.0, .me = 1.0, .w = 3, .s = false },
        .{ .m = strong, .le = 8.4, .me = 4.6, .w = 6, .s = true },
    };
    for (cases) |c| {
        const cold = gate(c.m, 100, c.le, c.me, c.w, c.s, .{});
        try testing.expectEqual(cold, gate(c.m, 100, c.le, c.me, c.w, c.s, .{ .mtp_ms = 3.7 * mtpFitCost(c.w) }));
        var only_lookup = Costs{};
        for (&only_lookup.lookup_ms, 0..) |*ms, k| ms.* = 0.8 * verifyCost(@intCast(k + 1));
        try testing.expectEqual(cold, gate(c.m, 100, c.le, c.me, c.w, c.s, only_lookup));
    }
}

test "lookup gate: measured prices on both sides decide" {
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const strong = Match{ .draft = &d, .suffix = STRONG_SUFFIX };
    const ordinary = Match{ .draft = &d, .suffix = MIN_SUFFIX };
    // Cold, MTP wins (5 / 27.0 against 6.3 / 31.5); a lookup measured cheap on this chip wins.
    var cheap = Costs{ .mtp_ms = 31.5 };
    cheap.lookup_ms[6] = 20.0;
    try testing.expectEqual(@as(u32, 6), gate(ordinary, 100, 4.0, 5.3, 6, false, cheap));
    // Cold, lookup wins at 11 drafts; measured dear, MTP runs.
    var dear = Costs{ .mtp_ms = 21.45 };
    dear.lookup_ms[11] = 200.0;
    try testing.expectEqual(@as(u32, 0), gate(strong, 100, 9.0, 1.0, 3, false, dear));
}
