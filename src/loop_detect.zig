const std = @import("std");
const testing = std.testing;

pub const degenerate_loop_max_period: usize = 8;

pub const degenerate_loop_reps: usize = 16;

pub const degenerate_loop_min_span: usize = 128;

pub const degenerate_loop_long_max_period: usize = 64;

pub const degenerate_loop_long_reps: usize = 10;

pub const degenerate_loop_long_min_span: usize = 1024;

pub const near_repeat_window: usize = 1024;

pub const near_repeat_ngram: usize = 4;

pub const near_repeat_max_ngram_ratio: f32 = 0.35;

pub const near_repeat_max_token_ratio: f32 = 0.12;

pub const near_repeat_max_novelty: f32 = 0.10;

pub fn DistinctSet(comptime cap: usize) type {
    return struct {
        const Self = @This();
        const empty_key: u64 = std.math.maxInt(u64);
        keys: [cap]u64 = @splat(empty_key),
        n: usize = 0,

        /// True when `key` is already present. Read-only; used to ask whether
        /// the window's second half is introducing anything its first half
        /// did not have.
        fn contains(self: *const Self, key: u64) bool {
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) return false;
                if (self.keys[i] == k) return true;
                i = (i + 1) % cap;
            }
        }

        /// True when `key` was not already present.
        fn insert(self: *Self, key: u64) bool {
            // maxInt is the empty sentinel; fold the one colliding key onto 0.
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) {
                    self.keys[i] = k;
                    self.n += 1;
                    return true;
                }
                if (self.keys[i] == k) return false;
                i = (i + 1) % cap;
            }
        }
    };
}

pub fn nearRepeatWindowIsDegenerate(window: []const u32) bool {
    // Load factor 0.5 keeps the linear probe short even when every entry is
    // distinct (the healthy case, which is also the hot one).
    var toks = DistinctSet(near_repeat_window * 2){};
    for (window) |t| _ = toks.insert(t);
    const token_ratio = @as(f32, @floatFromInt(toks.n)) / @as(f32, @floatFromInt(window.len));
    if (token_ratio > near_repeat_max_token_ratio) return false;

    var grams = DistinctSet(near_repeat_window * 2){};
    var i: usize = 0;
    while (i + near_repeat_ngram <= window.len) : (i += 1) {
        _ = grams.insert(gramHash(window[i .. i + near_repeat_ngram]));
    }
    const n_gram_positions = window.len - near_repeat_ngram + 1;
    const gram_ratio = @as(f32, @floatFromInt(grams.n)) / @as(f32, @floatFromInt(n_gram_positions));
    if (gram_ratio > near_repeat_max_ngram_ratio) return false;

    // Third ratio: is the window still PROGRESSING? Both ratios above are
    // properties of a vocabulary, and procedurally generated code has the same
    // vocabulary profile as a loop — a fixed call template plus a small colour
    // palette (live 2026-08-05: a voxel scene cut at 16241 tokens, the user got
    // no file at all). What a loop does NOT do is keep introducing material:
    // measured, a restatement loop's second half brings 1.9-2.2% n-grams its
    // first half never had, while healthy repetitive output brings 29.8-82.7%.
    // Requiring all THREE keeps the tier's reluctance in the direction that
    // matters — a missed loop still ends at max_tokens, a false cut destroys
    // work that was going fine.
    return halfNovelty(window, near_repeat_ngram) <= near_repeat_max_novelty;
}

pub fn halfNovelty(window: []const u32, n: usize) f32 {
    var first_half = DistinctSet(near_repeat_window * 2){};
    const mid = window.len / 2;
    var fi: usize = 0;
    while (fi + n <= mid) : (fi += 1) {
        _ = first_half.insert(gramHash(window[fi .. fi + n]));
    }
    var second_half = DistinctSet(near_repeat_window * 2){};
    var novel: usize = 0;
    var distinct_second: usize = 0;
    var si: usize = mid;
    while (si + n <= window.len) : (si += 1) {
        const h = gramHash(window[si .. si + n]);
        if (!second_half.insert(h)) continue; // count each distinct gram once
        distinct_second += 1;
        if (!first_half.contains(h)) novel += 1;
    }
    if (distinct_second == 0) return 0;
    return @as(f32, @floatFromInt(novel)) / @as(f32, @floatFromInt(distinct_second));
}

pub fn gramHash(gram: []const u32) u64 {
    var h: u64 = 0;
    for (gram) |t| h = h *% 0x100000001b3 ^ t;
    return h;
}

pub fn isNearRepeatTailLoop(tokens: []const u32) bool {
    if (tokens.len < near_repeat_window) return false;
    return nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..]);
}

pub const near_repeat_step: usize = 128;

pub const near_repeat_min_span: usize = 4096;

pub const near_repeat_max_lookback: usize = 8192;

pub const DegenerateTail = struct {
    tier: Tier,
    /// First index of the degenerate span; `tokens[0..start]` is what a
    /// client should be shown. For the exact tiers ONE copy of the cycle is
    /// deliberately kept — the truncated answer should still show what the
    /// model was doing when it got stuck, and one copy cannot sustain a loop.
    start: usize,

    pub const Tier = enum { exact_cycle, long_cycle, near_repeat };
};

pub fn exactCyclePeriod(tokens: []const u32, min_period: usize, max_period: usize, reps: usize, min_span: usize) ?usize {
    if (max_period == 0 or reps < 2) return null;
    var p: usize = @max(min_period, 1);
    while (p <= max_period) : (p += 1) {
        const span = @max(p * reps, min_span);
        if (tokens.len < span) continue;
        const tail = tokens[tokens.len - span ..];
        var periodic = true;
        var i: usize = p;
        while (i < tail.len) : (i += 1) {
            if (tail[i] != tail[i - p]) {
                periodic = false;
                break;
            }
        }
        if (periodic) return p;
    }
    return null;
}

pub fn trailingCycleStart(tokens: []const u32, p: usize) usize {
    var start = tokens.len - p;
    while (start >= p) {
        if (!std.mem.eql(u32, tokens[start - p .. start], tokens[start .. start + p])) break;
        start -= p;
    }
    return start; // first index of the FIRST copy of the cycle
}

pub fn degenerateTail(tokens: []const u32) ?DegenerateTail {
    if (exactCyclePeriod(tokens, 1, degenerate_loop_max_period, degenerate_loop_reps, degenerate_loop_min_span)) |p| {
        return .{ .tier = .exact_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (exactCyclePeriod(
        tokens,
        degenerate_loop_max_period + 1,
        degenerate_loop_long_max_period,
        degenerate_loop_long_reps,
        degenerate_loop_long_min_span,
    )) |p| {
        return .{ .tier = .long_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (tokens.len < near_repeat_window) return null;
    if (!nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..])) return null;

    // Slide the window back while it keeps convicting. A restatement loop
    // that has run for 3000 tokens is degenerate for all 3000 — trimming only
    // the last window would hand the client the rest of the loop back.
    var start = tokens.len - near_repeat_window;
    const floor = if (tokens.len > near_repeat_window + near_repeat_max_lookback)
        tokens.len - near_repeat_window - near_repeat_max_lookback
    else
        0;
    while (start >= floor + near_repeat_step) {
        const cand = start - near_repeat_step;
        if (!nearRepeatWindowIsDegenerate(tokens[cand .. cand + near_repeat_window])) break;
        start = cand;
    }
    // A file of near-identical rows (a lazy tile map, a zero-heavy table)
    // reads as a loop under every content measure; what it does that a loop
    // never does is END. Convict only once the degenerate span has outrun any
    // such file — a real restatement loop is still cut at ~4k tokens instead
    // of max_tokens, and a false cut destroys work.
    if (tokens.len - start < near_repeat_min_span) return null;
    return .{ .tier = .near_repeat, .start = start };
}

pub const StallClock = struct {
    last_progress_ns: u64 = 0,
    last_progress_count: usize = 0,

    pub fn expired(self: *StallClock, now_ns: u64, generated_count: usize, timeout_ns: u64) bool {
        if (generated_count != self.last_progress_count) {
            self.last_progress_count = generated_count;
            self.last_progress_ns = now_ns;
        }
        if (timeout_ns == 0) return false;
        return now_ns -| self.last_progress_ns >= timeout_ns;
    }
};

pub fn isDegenerateTailLoop(tokens: []const u32, max_period: usize, reps: usize) bool {
    return exactCyclePeriod(tokens, 1, max_period, reps, degenerate_loop_min_span) != null;
}

pub fn isDegenerateTailLoopRange(tokens: []const u32, min_period: usize, max_period: usize, reps: usize) bool {
    return exactCyclePeriod(tokens, min_period, max_period, reps, 0) != null;
}

test "StallClock: progress resets the deadline, silence expires it, 0 disables" {
    var clock = StallClock{};
    const s = std.time.ns_per_s;
    // Producing tokens keeps resetting the deadline — a healthy generation
    // can run arbitrarily long (the live bug: a 33KB tool call at 30 tok/s
    // takes >300s and was guillotined mid-call by the wall-clock timeout).
    try std.testing.expect(!clock.expired(0 * s, 0, 300 * s));
    try std.testing.expect(!clock.expired(299 * s, 1000, 300 * s)); // progress at 299s
    try std.testing.expect(!clock.expired(598 * s, 2000, 300 * s)); // progress again
    // No new tokens for the full window -> stalled.
    try std.testing.expect(!clock.expired(700 * s, 2000, 300 * s));
    try std.testing.expect(clock.expired(898 * s, 2000, 300 * s));
    // 0 = disabled, even after silence.
    var off = StallClock{};
    try std.testing.expect(!off.expired(0, 0, 0));
    try std.testing.expect(!off.expired(10_000 * s, 0, 0));
}

test "isDegenerateTailLoop catches a repeated channel-opener cycle" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Gemma 4 12B failure mode: the model spams the thinking opener
    // `<|channel>thought\n` — model that as a 3-token cycle. After enough
    // identical repetitions the tail is a pure period-3 loop → fire.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 7, 8, 9 }); // some real prefix
        var k: usize = 0;
        while (k < degenerate_loop_min_span / 3 + 1) : (k += 1) {
            try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 }); // <|channel>,thought,\n
        }
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }

    // A single token stuck on repeat (period 1) counts once it fills the span.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        while (k < degenerate_loop_min_span + 1) : (k += 1) try ids.append(testing.allocator, 42);
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }
}

test "isNearRepeatTailLoop catches a VARIED-phrasing restatement loop" {
    // Live 2026-08-04, under pi: the model restated the same
    // intent forever while varying the wording — "I need to break this down."
    // / "I need to break this." / "I need to break this down into pieces." —
    // so no exact cycle exists at ANY period and both exact tiers are blind by
    // construction. What IS invariant is that a long stretch of output recycles
    // a tiny vocabulary and introduces almost no new n-grams.
    const al = testing.allocator;
    const phrasings = [_][]const u32{
        &[_]u32{ 40, 41, 42, 43, 44, 45, 46 }, // I need to break this down .
        &[_]u32{ 40, 41, 42, 43, 44, 46 }, // I need to break this .
        &[_]u32{ 40, 41, 42, 43, 44, 45, 47, 48, 46 }, // ... down into pieces .
        &[_]u32{ 49, 40, 41, 42, 43, 44, 45, 46 }, // So I need to break this down .
        &[_]u32{ 40, 41, 42, 43, 44, 45, 50, 46 }, // ... down first .
        &[_]u32{ 51, 42, 43, 44, 45, 46 }, // Let me break this down .
    };
    var ids = try tVaried(al, &phrasings, near_repeat_window + 64);
    defer ids.deinit(al);
    try testing.expect(isNearRepeatTailLoop(ids.items));

    // Below the window the tier says nothing at all: it is a last-resort net
    // for output that has already run a long way, and a false cut truncates a
    // real answer. Short exact cycles remain tier 1/2's job.
    try testing.expect(!isNearRepeatTailLoop(ids.items[0 .. near_repeat_window - 1]));
}

test "isNearRepeatTailLoop leaves legitimately repetitive output alone" {
    const al = testing.allocator;

    // Healthy prose/code: a repeated scaffold, but every line introduces a new
    // identifier. Recycled STRUCTURE is normal; recycled VOCABULARY is not.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var line: u32 = 0;
        while (ids.items.len < near_repeat_window + 64) : (line += 1) {
            try ids.appendSlice(al, &[_]u32{ 10, 11, 12 }); // `const x =`
            try ids.append(al, 1000 + line); // a fresh identifier
            try ids.appendSlice(al, &[_]u32{ 13, 14 }); // `;\n`
        }
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }

    // A numeric table: FEW distinct tokens (digits + separators, and this
    // family pre-tokenizes digits singly) but the 4-grams keep changing. The
    // two ratios have to be read together — either one alone convicts this.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var seed: u32 = 12345;
        while (ids.items.len < near_repeat_window + 64) {
            try ids.append(al, 200); // '|'
            for (0..4) |_| {
                seed = seed *% 1664525 +% 1013904223;
                try ids.append(al, 100 + (seed >> 16) % 10); // a digit
            }
            try ids.appendSlice(al, &[_]u32{ 200, 201 }); // '|', '\n'
        }
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }

    // Fully novel output.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        for (0..near_repeat_window + 64) |i| try ids.append(al, @intCast(i));
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }
}

test "isDegenerateTailLoop does not fire on healthy or briefly-repeating output" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Strictly increasing ids — no cycle at all.
    {
        var ids: [200]u32 = undefined;
        for (&ids, 0..) |*v, i| v.* = @intCast(i);
        try testing.expect(!isDegenerateTailLoop(&ids, P, R));
    }
    // A short burst of repetition (well under R reps) must be left alone — a
    // model legitimately writing "ha ha ha" or a few identical list bullets.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 1, 2, 3, 4, 5 });
        var k: usize = 0;
        while (k < R - 1) : (k += 1) try ids.appendSlice(testing.allocator, &[_]u32{ 50, 51 });
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Periodic tail but with a longer period than we scan for → ignored.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        var base: u32 = 0;
        while (k < R) : (k += 1) {
            // period = P + 3 (> max_period); never a pure short cycle.
            var j: u32 = 0;
            while (j < P + 3) : (j += 1) try ids.append(testing.allocator, base + j);
            base = 0; // same long block repeats, but its period exceeds the scan window
        }
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Too few tokens to judge.
    try testing.expect(!isDegenerateTailLoop(&[_]u32{ 1, 1 }, P, R));
}

test "isNearRepeatTailLoop leaves PROCEDURAL code alone — it recycles a vocabulary while still progressing" {
    // Live 2026-08-05: a pi session was asked for an elaborate voxel scene and
    // the tier cut it at 16241 generated tokens, so the user got NO file at
    // all. The output was healthy — dense `fillBox(x,y,z, x,y,z, C.name);`
    // lines — but it is exactly the shape the first two ratios were built to
    // tolerate and cannot: a fixed template plus a small colour palette gives
    // a tiny distinct-token ratio, and the templated call shape keeps the
    // 4-gram ratio low too. Measured on the real artifact: 0.068 / 0.351
    // against bars of 0.12 / 0.35 — it cleared conviction by 0.001, and the
    // generation's own tail did not.
    //
    // What separates it from a loop is PROGRESS: every line carries new
    // coordinates, so the window's second half keeps introducing n-grams the
    // first half never had (0.298-0.632 measured, against 0.019-0.022 for the
    // restatement loops this tier exists for).
    // Shape taken from the measured artifact, not invented: a CONTIGUOUS run
    // of template tokens (`\n  fillBox(`, `, C.`, `);`) followed by the
    // varying coordinates. The contiguity is what makes it convict — most
    // 4-gram windows sit entirely inside the fixed run and repeat every line.
    // This fixture scores 0.033 / 0.316 against bars of 0.12 / 0.35.
    const al = testing.allocator;
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    var rng: u32 = 99;
    while (ids.items.len < near_repeat_window * 2) {
        var f: u32 = 0;
        while (f < 14) : (f += 1) try ids.append(al, 500 + f);
        var v: usize = 0;
        while (v < 4) : (v += 1) {
            rng = rng *% 1664525 +% 1013904223;
            try ids.append(al, 600 + (rng >> 16) % 19); // a fresh coordinate
            try ids.append(al, 499); // separator
        }
    }
    try testing.expect(!isNearRepeatTailLoop(ids.items));
    try testing.expect(degenerateTail(ids.items) == null);
}

fn tVaried(al: std.mem.Allocator, phrasings: []const []const u32, n: usize) !std.ArrayList(u32) {
    var ids = std.ArrayList(u32).empty;
    var i: usize = 0;
    while (ids.items.len < n) : (i += 1) {
        try ids.appendSlice(al, phrasings[i % phrasings.len]);
    }
    return ids;
}

test "degenerateTail: a short exact cycle convicts only past the minimum span" {
    const al = testing.allocator;
    // Live 2026-09-15 (pi, Qwen3.8 per-digit tokenizer): a 24-wide map wall
    // row "111111111111111111111111" is 24 identical tokens and was cut as a
    // period-1 loop mid-thought. Short cycles are common in honest code (digit
    // rows, zeroed arrays), so the bar is a SPAN of identical cycling, not a
    // rep count that period 1 reaches in a few dozen bytes.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 7, 8, 9, 1 });
    for (0..24) |_| try ids.append(al, 16);
    try testing.expect(degenerateTail(ids.items) == null);

    // A 32-element zeroed row: `0, 0, 0, ...` is a period-2 cycle of 64 tokens.
    var zeros = std.ArrayList(u32).empty;
    defer zeros.deinit(al);
    try zeros.appendSlice(al, &[_]u32{ 7, 8, 9, 1 });
    for (0..32) |_| try zeros.appendSlice(al, &[_]u32{ 15, 11 });
    try testing.expect(degenerateTail(zeros.items) == null);

    // Stuck for real: one token past the span is a loop and trims to one copy.
    for (24..degenerate_loop_min_span) |_| try ids.append(al, 16);
    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.exact_cycle, d.tier);
    try testing.expectEqual(@as(usize, 5), d.start);
}

test "degenerateTail acquits a low-entropy STRUCTURED file that ends inside the span bar" {
    // Live 2026-09-15 under pi: a tool call rewriting five 24x19 tile maps of
    // '0'/'1' on a per-digit tokenizer was cut as a near-repeat loop at window
    // fill and the agent got an empty turn. Six distinct tokens and sixteen
    // possible 4-grams satisfy every content ratio by construction, and the
    // real 27B output (rows nearly all `100000000000000000000001`) is
    // indistinguishable from a loop by content. What such a file does, and a
    // loop never does, is END: the bar is the degenerate SPAN.
    const al = testing.allocator;
    const Shape = enum { bit_grid, lazy_map, hex_dump };
    for ([_]Shape{ .bit_grid, .lazy_map, .hex_dump }) |shape| {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var seed: u32 = 777;
        var row: usize = 0;
        while (ids.items.len < 2800) : (row += 1) {
            try ids.append(al, 200); // '"'
            for (0..24) |col| {
                seed = seed *% 1664525 +% 1013904223;
                const r = seed >> 16;
                const tok: u32 = switch (shape) {
                    .bit_grid => if (col == 0 or col == 23 or row % 19 == 0) 101 else @as(u32, if (r % 10 < 3) 101 else 100),
                    .lazy_map => if (col == 0 or col == 23 or row % 19 == 0 or (row % 5 == 0 and col != 10)) 101 else 100,
                    .hex_dump => 100 + r % 16,
                };
                try ids.append(al, tok);
            }
            try ids.appendSlice(al, &[_]u32{ 200, 201, 202 }); // '"', ',', '\n'
        }
        try testing.expect(degenerateTail(ids.items) == null);
    }
}

test "degenerateTail: the exact tier reports its tier and keeps ONE cycle" {
    const al = testing.allocator;
    // Identical 3-token cycles past the span bar after a real prefix. The cut
    // is a truncation, so what is emitted should still SHOW what the model got
    // stuck on — one copy of the cycle survives, the rest do not.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 7, 8, 9, 10 });
    var k: usize = 0;
    while (k < degenerate_loop_min_span / 3 + 1) : (k += 1) try ids.appendSlice(al, &[_]u32{ 101, 102, 103 });

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.exact_cycle, d.tier);
    // 4 prefix + 1 kept cycle = 7 tokens survive.
    try testing.expectEqual(@as(usize, 7), d.start);
    // What survives is the honest prefix plus exactly one cycle.
    try testing.expectEqualSlices(u32, &[_]u32{ 7, 8, 9, 10, 101, 102, 103 }, ids.items[0..d.start]);
}

test "degenerateTail: the trim start walks back PAST the near-repeat window" {
    const al = testing.allocator;
    // The near-repeat tier judges the last 1024 tokens, but a restatement
    // loop that has been running for 3000 tokens is degenerate for all 3000.
    // Trimming only the window would hand the client the other ~2000 back,
    // which is the whole failure this exists to stop.
    const phrasings = [_][]const u32{
        &[_]u32{ 1, 2, 3, 4, 5 },
        &[_]u32{ 1, 2, 3, 5, 4 },
        &[_]u32{ 1, 2, 4, 3, 5 },
    };
    var honest = std.ArrayList(u32).empty;
    defer honest.deinit(al);
    var i: u32 = 0;
    while (i < 900) : (i += 1) try honest.append(al, 1000 + i); // all distinct = healthy

    // Pick phrasings pseudo-randomly: a deterministic rotation would be an
    // exact cycle and the long-period tier would convict it first, which is
    // not the tier under test.
    var loop = std.ArrayList(u32).empty;
    defer loop.deinit(al);
    var rng: u32 = 12345;
    while (loop.items.len < near_repeat_min_span + 1000) {
        rng = rng *% 1664525 +% 1013904223;
        try loop.appendSlice(al, phrasings[(rng >> 16) % phrasings.len]);
    }

    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, honest.items);
    try ids.appendSlice(al, loop.items);

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.near_repeat, d.tier);
    // Well past the single window, and never into the healthy prefix.
    try testing.expect(d.start < ids.items.len - near_repeat_window);
    try testing.expect(d.start >= honest.items.len - near_repeat_step);
}

test "degenerateTail: healthy output is never convicted, so nothing is trimmed" {
    var ids: [4000]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    try testing.expect(degenerateTail(&ids) == null);
}

test "degenerateTail: the long-period tier keeps one copy of its sentence cycle" {
    const al = testing.allocator;
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 1, 2 });
    var cycle: [40]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @intCast(500 + i);
    var k: usize = 0;
    while (k < degenerate_loop_long_min_span / cycle.len + 1) : (k += 1) try ids.appendSlice(al, &cycle);

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.long_cycle, d.tier);
    try testing.expectEqual(@as(usize, 2 + cycle.len), d.start);
}
