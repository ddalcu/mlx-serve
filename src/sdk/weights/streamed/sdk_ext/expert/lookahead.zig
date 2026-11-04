//! Host half of the expert streamer's lookahead (the lookahead4 lane's
//! Selector and pre-read list). The router barrier also evaluates the next
//! routed layer's own gate on this layer's router input, giving M rows of
//! per-expert scores; per row the top-K experts by score are kept iff their
//! score is at least the row's routed-top-k-th score minus tau, the union is ordered by
//! (best rank over rows, then score), residents are dropped, and the first
//! `budget` become speculative whole-record reads. A score tie breaks by expert
//! id (the Python lane leaves tied experts in numpy's argpartition order).
//! Pure: it chooses records and reads nothing.

const std = @import("std");
const mlx = @import("sdk").mlx;
const expert_policy = @import("policy.zig");

const LayerPolicy = expert_policy.LayerPolicy;

/// The widest per-row candidate list a selector keeps (its `k`).
pub const max_k = 12;
pub const max_budget = 4;
/// Verify rows per layer call.
pub const max_rows = 8;
/// Certain misses handed to one pre-read; the re-validation candidate cap.
pub const max_candidates = 64;

const Entry = struct { rank: u8, score: f32, expert: u16 };

/// Expert `e` is resident under `p`: the residency `select` skips (one layer's policy).
pub fn residentIn(p: *const LayerPolicy, e: u16) bool {
    return p.slotOf(e) != null;
}

/// A selector for an arch routing `routed_top_k` experts per row: tau is measured from each row's routed-top-k-th score,
/// and a selector keeps at least that many candidates per row (`k`, up to `max_k`).
pub fn SelectorOf(comptime routed_top_k: u32) type {
    comptime if (routed_top_k == 0 or routed_top_k > max_k) @compileError("lookahead: routed_top_k outside 1..max_k");
    const min_k = routed_top_k;
    return struct {
        const Selector = @This();

        n_experts: u32,
        k: u32,
        /// inf keeps each row's plain top-K.
        tau: f32,
        budget: u32,
        seen: std.DynamicBitSetUnmanaged,

        pub fn init(a: std.mem.Allocator, n_experts: u32, k: u32, tau: f32, budget: u32) !Selector {
            if (k < min_k or k > max_k or k > n_experts or budget == 0 or budget > max_budget or std.math.isNan(tau) or tau == -std.math.inf(f32))
                return error.InvalidSelector;
            return .{ .n_experts = n_experts, .k = k, .tau = tau, .budget = budget, .seen = try std.DynamicBitSetUnmanaged.initEmpty(a, n_experts) };
        }

        pub fn deinit(self: *Selector, a: std.mem.Allocator) void {
            self.seen.deinit(a);
            self.* = undefined;
        }

        /// The records to read ahead for the next layer: `scores` is rows x
        /// n_experts (row-major f32), `next` that layer's residency. Returns the
        /// first `out.len` non-resident experts of the ordered union.
        pub fn select(self: *Selector, scores: []const f32, next: *const LayerPolicy, out: []u16) []u16 {
            return self.selectWith(scores, next, residentIn, out, null);
        }

        /// `select` with the residency asked of `resident(ctx, expert)` (the next layer's experts over several
        /// policies: one per bank layer).
        pub fn selectWith(self: *Selector, scores: []const f32, ctx: anytype, comptime resident: fn (@TypeOf(ctx), u16) bool, out: []u16, ranks: ?[]u8) []u16 {
            std.debug.assert(scores.len % self.n_experts == 0 and scores.len / self.n_experts <= max_rows);
            var entries: [max_rows * max_k]Entry = undefined;
            const ordered = self.order(scores, &entries);
            var n: usize = 0;
            for (ordered) |en| {
                if (self.seen.isSet(en.expert)) continue;
                self.seen.set(en.expert);
                if (resident(ctx, en.expert)) continue;
                if (ranks) |r| r[n] = en.rank;
                out[n] = en.expert;
                n += 1;
                if (n == out.len) break;
            }
            for (ordered) |en| self.seen.unset(en.expert);
            return out[0..n];
        }

        /// Per row the top-k by (score desc, expert asc), kept iff score >= the
        /// row's routed-top-k-th score - tau (f32); then a stable sort by (rank, score desc),
        /// so equal keys keep row-major order, as the lane's lexsort does.
        fn order(self: *const Selector, scores: []const f32, entries: *[max_rows * max_k]Entry) []Entry {
            const e_n = self.n_experts;
            const rows = scores.len / e_n;
            const finite = std.math.isFinite(self.tau);
            var n: usize = 0;
            for (0..rows) |r| {
                const row = scores[r * e_n ..][0..e_n];
                var top: [max_k]u16 = undefined;
                var cnt: usize = 0;
                for (row, 0..) |v, ei| {
                    const e: u16 = @intCast(ei);
                    if (cnt == self.k and !better(v, e, row[top[cnt - 1]], top[cnt - 1])) continue;
                    var i = if (cnt < self.k) cnt else cnt - 1;
                    if (cnt < self.k) cnt += 1;
                    while (i > 0 and better(v, e, row[top[i - 1]], top[i - 1])) : (i -= 1) top[i] = top[i - 1];
                    top[i] = e;
                }
                const threshold: f32 = if (finite) row[top[routed_top_k - 1]] - self.tau else -std.math.inf(f32);
                for (top[0..cnt], 0..) |e, rank| {
                    const v = row[e];
                    if (finite and !(v >= threshold)) continue;
                    entries[n] = .{ .rank = @intCast(rank), .score = v, .expert = e };
                    n += 1;
                }
            }
            const out = entries[0..n];
            std.sort.insertion(Entry, out, {}, struct {
                fn less(_: void, a: Entry, b: Entry) bool {
                    return if (a.rank != b.rank) a.rank < b.rank else a.score > b.score;
                }
            }.less);
            return out;
        }

        /// The layer call's certain misses: its unique routed experts not resident
        /// in `layer`, in route order, at most `out.len` (they are its plan's misses).
        pub fn certainMisses(self: *Selector, ids: []const u16, layer: *const LayerPolicy, out: []u16) []u16 {
            var n: usize = 0;
            for (ids) |e| {
                if (self.seen.isSet(e)) continue;
                self.seen.set(e);
                if (layer.slotOf(e) == null and n < out.len) {
                    out[n] = e;
                    n += 1;
                }
            }
            for (ids) |e| self.seen.unset(e);
            return out[0..n];
        }
    };
}

fn better(v: f32, e: u16, w: f32, f: u16) bool {
    return v > w or (v == w and e < f);
}

// ── Tests ──
