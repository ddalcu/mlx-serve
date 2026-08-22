//! Measured spec round-cost table: what a round at draft width w costs on
//! THIS model, on THIS machine, at the live context length.
//!
//! Every width decision in spec decode is throughput = accepted tokens over
//! round wall time, and every earlier cost source (the hand-typed chip rows,
//! the fitted EV surfaces, the boot ladder) measured only part of that, in
//! one regime, and shipped it for all of them. This table measures the
//! whole fraction from the rounds the server actually runs. Pure data, no
//! MLX — the generator feeds it and reads it, the fitted surface is the
//! cold-start prior.
//!
//! Width = drafts per round (MTP depth m; DFlash block_size - 1; 0 = serial).
//! Buckets = KV length at the round: <2k, 2-4k, 4-8k, 8-16k, 16-32k, 32k+.
//! Each cell holds an EMA of round ms AND an EMA of emitted tokens — cost is
//! never stored without the tokens it bought.
const std = @import("std");

/// Drafts per round the table covers (MTP depth <= 8, a DFlash block up to 16); index 0 is serial.
pub const MAX_WIDTH: u32 = 16;
pub const N_BUCKETS: usize = 6;
const BUCKET_EDGES = [_]u32{ 2048, 4096, 8192, 16384, 32768 };
pub const BUCKET_NAMES = [N_BUCKETS][]const u8{ "<2k", "2-4k", "4-8k", "8-16k", "16-32k", "32k+" };

/// EMA weight. MIN is wrong here (thermal soak makes the early rounds the
/// fast ones); an EMA tracks the live machine.
pub const BETA: f32 = 0.10;
/// A cell unsampled for this many offered rounds reseeds on its next sample
/// instead of folding: whatever it held was measured in another regime.
pub const RESEED_GAP: u32 = 64;
/// Measured widths a bucket needs before the table replaces the prior.
pub const MIN_WIDTHS: u32 = 2;

pub fn bucketFor(kv_len: u32) usize {
    for (BUCKET_EDGES, 0..) |edge, i| {
        if (kv_len < edge) return i;
    }
    return N_BUCKETS - 1;
}

pub const Cell = struct {
    ms: f32 = 0,
    tok: f32 = 0,
    n: u32 = 0,
    /// `Table.seq` at the last fold — the reseed clock.
    last_seen: u32 = 0,

    pub fn msPerTok(self: Cell) ?f32 {
        if (self.n == 0 or self.tok <= 0) return null;
        return self.ms / self.tok;
    }
};

pub const Verdict = enum { folded, reseeded, contended, transition, bad_sample, out_of_range };

pub const Table = struct {
    cells: [MAX_WIDTH + 1][N_BUCKETS]Cell = @splat(@splat(.{})),
    /// Samples OFFERED (accepted or not): the reseed clock.
    seq: u32 = 0,
    folded: u32 = 0,
    dropped_transition: u32 = 0,
    dropped_contended: u32 = 0,
    dropped_bad: u32 = 0,
    /// One-shot: the first plan that read the table instead of the prior.
    first_use_logged: bool = false,

    /// Feed one realized round. `solo` = this was the only decoding stream
    /// (contention only ever ADDS time, so a busy server stops teaching the
    /// table rather than teaching it a lie); `transition` = the width
    /// differs from the previous round (the width change is a one-off cost
    /// that read the minority shape 5-7% slow in Phase 1).
    pub fn observe(self: *Table, width: u32, kv_len: u32, ms: f32, tokens: f32, solo: bool, transition: bool) Verdict {
        self.seq +%= 1;
        if (width > MAX_WIDTH) return .out_of_range;
        if (!std.math.isFinite(ms) or ms <= 0 or !(tokens > 0)) {
            self.dropped_bad += 1;
            return .bad_sample;
        }
        if (!solo) {
            self.dropped_contended += 1;
            return .contended;
        }
        if (transition) {
            self.dropped_transition += 1;
            return .transition;
        }
        const cell = &self.cells[width][bucketFor(kv_len)];
        self.folded += 1;
        defer cell.last_seen = self.seq;
        if (cell.n == 0 or self.seq -% cell.last_seen > RESEED_GAP) {
            cell.ms = ms;
            cell.tok = tokens;
            cell.n = 1;
            return .reseeded;
        }
        cell.ms += BETA * (ms - cell.ms);
        cell.tok += BETA * (tokens - cell.tok);
        cell.n += 1;
        return .folded;
    }

    /// Measured round ms at exactly this width, or null.
    pub fn measuredMs(self: *const Table, width: u32, bucket: usize) ?f32 {
        if (width > MAX_WIDTH) return null;
        const c = self.cells[width][bucket];
        return if (c.n == 0) null else c.ms;
    }

    pub fn msPerTok(self: *const Table, width: u32, bucket: usize) ?f32 {
        if (width > MAX_WIDTH) return null;
        return self.cells[width][bucket].msPerTok();
    }

    pub fn measuredCount(self: *const Table, bucket: usize) u32 {
        var n: u32 = 0;
        for (0..MAX_WIDTH + 1) |w| {
            if (self.cells[w][bucket].n > 0) n += 1;
        }
        return n;
    }

    pub fn active(self: *const Table, bucket: usize) bool {
        return self.measuredCount(bucket) >= MIN_WIDTHS;
    }

    /// The bucket a plan at `kv_len` reads: its own when active, else the
    /// nearest active one (lower side preferred — cost grows with KV, so a
    /// lower bucket under-bills rather than over-bills). Null = no active
    /// bucket, the prior applies. A bucket boundary crossed mid-generation
    /// must not snap the plan back to the prior.
    pub fn bucketToRead(self: *const Table, kv_len: u32) ?usize {
        const own = bucketFor(kv_len);
        if (self.active(own)) return own;
        var d: usize = 1;
        while (d < N_BUCKETS) : (d += 1) {
            if (own >= d and self.active(own - d)) return own - d;
            if (own + d < N_BUCKETS and self.active(own + d)) return own + d;
        }
        return null;
    }

    /// Narrowest measured width in the bucket (the normalization anchor).
    pub fn narrowestMeasured(self: *const Table, bucket: usize) ?u32 {
        for (0..MAX_WIDTH + 1) |w| {
            if (self.cells[w][bucket].n > 0) return @intCast(w);
        }
        return null;
    }

    /// Round ms at `width`: measured, else linear between the two nearest
    /// measured widths, else (past the widest) extrapolated with the last
    /// slope — cost is near-linear between cliffs, a cliff is found by
    /// measuring it, and the slope past one is the cliff's. BELOW the
    /// narrowest measured width the answer is null and the caller's prior
    /// fills in: the prior's extended rounds land on the cliff first, and
    /// the cliff's slope run downward reads every narrower width as free.
    /// Null while the bucket has fewer than MIN_WIDTHS measured widths.
    pub fn roundMs(self: *const Table, width: u32, bucket: usize) ?f32 {
        if (!self.active(bucket)) return null;
        if (self.measuredMs(width, bucket)) |m| return m;
        var lo: ?u32 = null;
        var lo2: ?u32 = null;
        var hi: ?u32 = null;
        for (0..MAX_WIDTH + 1) |wi| {
            const w: u32 = @intCast(wi);
            if (self.cells[w][bucket].n == 0) continue;
            if (w < width) {
                lo2 = lo;
                lo = w;
            } else if (hi == null) {
                hi = w;
            }
        }
        const at = struct {
            fn f(t: *const Table, w: u32, b: usize) f32 {
                return t.cells[w][b].ms;
            }
        }.f;
        if (lo != null and hi != null) {
            return lerp(lo.?, at(self, lo.?, bucket), hi.?, at(self, hi.?, bucket), width);
        }
        if (lo != null and lo2 != null) {
            return lerp(lo2.?, at(self, lo2.?, bucket), lo.?, at(self, lo.?, bucket), width);
        }
        return null;
    }

    fn lerp(w0: u32, m0: f32, w1: u32, m1: f32, w: u32) f32 {
        const t = (@as(f32, @floatFromInt(w)) - @as(f32, @floatFromInt(w0))) / (@as(f32, @floatFromInt(w1)) - @as(f32, @floatFromInt(w0)));
        return m0 + t * (m1 - m0);
    }

    /// `w3:12.1,w4:11.8` — measured ms per emitted token per width in the
    /// bucket, for `[spec-stats]`. Empty when nothing is measured.
    pub fn formatBucket(self: *const Table, bucket: usize, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (0..MAX_WIDTH + 1) |wi| {
            const c = self.cells[wi][bucket];
            const mpt = c.msPerTok() orelse continue;
            if (!first) w.writeAll(",") catch break;
            first = false;
            w.print("w{d}:{d:.2}", .{ wi, mpt }) catch break;
        }
        return w.buffered();
    }
};

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

test "round_cost: kv buckets" {
    try testing.expectEqual(@as(usize, 0), bucketFor(0));
    try testing.expectEqual(@as(usize, 0), bucketFor(2047));
    try testing.expectEqual(@as(usize, 1), bucketFor(2048));
    try testing.expectEqual(@as(usize, 3), bucketFor(8192));
    try testing.expectEqual(@as(usize, 4), bucketFor(16384));
    try testing.expectEqual(@as(usize, 5), bucketFor(32768));
    try testing.expectEqual(@as(usize, 5), bucketFor(1_000_000));
}

test "round_cost: EMA folds, first sample seeds, a long gap reseeds" {
    var t = Table{};
    try testing.expectEqual(Verdict.reseeded, t.observe(4, 1000, 50.0, 4.0, true, false));
    try testing.expectEqual(Verdict.folded, t.observe(4, 1000, 60.0, 4.0, true, false));
    try testing.expectApproxEqAbs(50.0 + BETA * 10.0, t.measuredMs(4, 0).?, 1e-4);
    try testing.expectApproxEqAbs(t.measuredMs(4, 0).? / 4.0, t.msPerTok(4, 0).?, 1e-4);
    // Other widths tick the clock; the width-4 cell goes stale.
    var i: u32 = 0;
    while (i <= RESEED_GAP) : (i += 1) _ = t.observe(3, 1000, 40.0, 3.0, true, false);
    try testing.expectEqual(Verdict.reseeded, t.observe(4, 1000, 90.0, 4.0, true, false));
    try testing.expectApproxEqAbs(90.0, t.measuredMs(4, 0).?, 1e-4);
}

test "round_cost: a contended, transition or bad sample never moves the estimate" {
    var t = Table{};
    _ = t.observe(4, 1000, 50.0, 4.0, true, false);
    try testing.expectEqual(Verdict.contended, t.observe(4, 1000, 500.0, 4.0, false, false));
    try testing.expectEqual(Verdict.transition, t.observe(4, 1000, 500.0, 4.0, true, true));
    try testing.expectEqual(Verdict.bad_sample, t.observe(4, 1000, 0.0, 4.0, true, false));
    try testing.expectEqual(Verdict.bad_sample, t.observe(4, 1000, 50.0, 0.0, true, false));
    try testing.expectEqual(Verdict.out_of_range, t.observe(MAX_WIDTH + 1, 1000, 50.0, 4.0, true, false));
    try testing.expectApproxEqAbs(50.0, t.measuredMs(4, 0).?, 1e-4);
    try testing.expectEqual(@as(u32, 1), t.dropped_contended);
    try testing.expectEqual(@as(u32, 1), t.dropped_transition);
    try testing.expectEqual(@as(u32, 2), t.dropped_bad);
    try testing.expectEqual(@as(u32, 1), t.folded);
}

test "round_cost: one width is not a table, two interpolate and extrapolate" {
    var t = Table{};
    _ = t.observe(3, 1000, 30.0, 3.0, true, false);
    try testing.expect(!t.active(0));
    try testing.expect(t.roundMs(4, 0) == null);
    try testing.expect(t.bucketToRead(1000) == null);
    _ = t.observe(5, 1000, 50.0, 5.0, true, false);
    try testing.expect(t.active(0));
    try testing.expectApproxEqAbs(40.0, t.roundMs(4, 0).?, 1e-4); // between
    try testing.expectApproxEqAbs(80.0, t.roundMs(8, 0).?, 1e-4); // last slope
    try testing.expect(t.roundMs(2, 0) == null); // below the anchor: the prior's job
    try testing.expectEqual(@as(u32, 3), t.narrowestMeasured(0).?);
    // A measured cliff is read as measured, and the slope past it is the cliff's.
    _ = t.observe(6, 1000, 200.0, 6.0, true, false);
    try testing.expectApproxEqAbs(200.0, t.roundMs(6, 0).?, 1e-4);
    try testing.expectApproxEqAbs(350.0, t.roundMs(7, 0).?, 1e-4);
}

test "round_cost: an unmeasured bucket reads the nearest active one, lower side first" {
    var t = Table{};
    _ = t.observe(3, 3000, 30.0, 3.0, true, false);
    _ = t.observe(4, 3000, 40.0, 4.0, true, false);
    try testing.expectEqual(@as(usize, 1), t.bucketToRead(3000).?);
    try testing.expectEqual(@as(usize, 1), t.bucketToRead(20000).?); // 16-32k falls back down to 2-4k
    try testing.expectEqual(@as(usize, 1), t.bucketToRead(100).?); // <2k falls back up
    _ = t.observe(3, 20000, 60.0, 3.0, true, false);
    _ = t.observe(4, 20000, 80.0, 4.0, true, false);
    try testing.expectEqual(@as(usize, 4), t.bucketToRead(20000).?);
    try testing.expectEqual(@as(usize, 1), t.bucketToRead(6000).?); // 4-8k: lower (2-4k) beats upper (16-32k)
    try testing.expectEqual(@as(usize, 4), t.bucketToRead(10000).?); // 8-16k: 16-32k is nearer than 2-4k
}

test "round_cost: formatBucket lists measured widths as ms/tok" {
    var t = Table{};
    _ = t.observe(3, 1000, 30.0, 3.0, true, false);
    _ = t.observe(5, 1000, 60.0, 4.0, true, false);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("w3:10.00,w5:15.00", t.formatBucket(0, &buf));
    try testing.expectEqualStrings("", t.formatBucket(1, &buf));
}
