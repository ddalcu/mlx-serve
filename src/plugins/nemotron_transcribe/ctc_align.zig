//! Word timestamps from the CTC head's log-probabilities: NVIDIA's single-pass banded aligner
//! (`MultiSpeakerSOTWordTimestampAligner`, NeMo Speech `speechlm2/parts/ctc_timestamp_utils.py`) over a t-SOT transcript.
//! Speaker streams are mapped to Sortformer columns by token distribution (a linear assignment), each stream is searched
//! by a max-sum CTC DP in an anchor-guided band over the frames its speaker is active in (the log activity as a prior),
//! and word boundaries are refined inside the diarization's activity segments. MLX-free; `tests/dump_ctc_align_fixtures.py`
//! replays NVIDIA's own functions to make the fixtures these tests demand.
//!
//! The probabilities are `[frames][cols]` f32 with column 0 the blank; a word's tokens are the columns they read.

const std = @import("std");

pub const Error = error{ OutOfMemory, TooShort };

pub const Params = struct {
    /// The speaker prior's weight on the log activity.
    speaker_logprob_weight: f32 = 1.0,
    /// A column counts a token under it while its activity is at least this.
    speaker_activity_threshold: f32 = 0.5,
    /// The search keeps the frames a speaker is active in at least this much.
    support_threshold: f32 = 0.01,
    /// The longest a word may be held, seconds.
    maximum_token_len: f64 = 1.0,
    /// The cap on a word's duration and an onset's snap, in CTC frames.
    max_word_frames: u32 = 24,
    epsilon: f32 = 1e-6,
};

/// One transcript word: its `<spk:N>` turn's tag (-1 before the first tag) and the columns of its tokens (>= 1).
pub const Word = struct { tag: i32, tokens: []const u32 };

/// The speakers over the clip: activity per encoder row (`[n_rows][count]`, resampled onto the CTC grid) and the
/// 10 ms activity labels (`[diar_frames][count]`, active > 0.5) the segments are cut from.
pub const Speakers = struct {
    rows: []const f32,
    n_rows: usize,
    count: usize,
    diar: []const u8,
    diar_frames: usize,
    diar_step: f64 = 0.01,
    /// The clip's duration, seconds: segments end there.
    duration: f64,
};

pub const Timed = struct {
    start: f64,
    end: f64,
    /// The Sortformer column the word's speaker maps to, -1 for none.
    speaker: i32,
    /// No legal path through its stream: the word sits where the previous one ended.
    failed: bool = false,
};

/// Each word's time over `frames` CTC frames of `step` seconds.
pub fn alignWords(a: std.mem.Allocator, lp: []const f32, frames: usize, cols: usize, words: []const Word, spk: ?Speakers, step: f64, p: Params) Error![]Timed {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const w = arena.allocator();
    const out = try a.alloc(Timed, words.len);
    errdefer a.free(out);
    for (out) |*o| o.* = .{ .start = 0, .end = 0, .speaker = -1, .failed = true };
    if (words.len == 0) return out;

    const streams = try buildStreams(w, words, frames);
    const greedy = try w.alloc(u32, frames);
    for (0..frames) |t| greedy[t] = argmax(lp[t * cols ..][0..cols]);

    // The speakers on the CTC grid, and which column each stream is.
    var grid: ?[]f32 = null;
    if (spk) |s| if (s.count > 0 and s.n_rows > 0) {
        grid = try resample(w, s.rows, s.n_rows, s.count, frames);
    };
    const mapping = try w.alloc(i32, streams.len);
    @memset(mapping, -1);
    if (grid) |g| try mapSpeakers(w, cols, greedy, g, spk.?.count, streams, p.speaker_activity_threshold, mapping);

    // The search, stream by stream.
    for (streams, 0..) |st, si| {
        const col = mapping[si];
        const activity = try w.alloc(f32, frames);
        if (grid) |g| {
            const count = spk.?.count;
            for (0..frames) |t| activity[t] = if (col < 0) 0 else g[t * count + @as(usize, @intCast(col))];
        } else @memset(activity, 1);
        var positions: std.ArrayList(u32) = .empty;
        for (activity, 0..) |x, t| if (x >= p.support_threshold) try positions.append(w, @intCast(t));
        const prior = try w.alloc(f32, frames);
        for (activity, prior) |x, *pr| pr.* = p.speaker_logprob_weight * @log(@max(x, p.epsilon));
        const path = try bandedSearch(w, lp, cols, st.labels, positions.items, prior, greedy);
        // Which frames each word of the stream owns on the path.
        const first = try w.alloc(i64, st.n_words);
        const last = try w.alloc(i64, st.n_words);
        @memset(first, -1);
        @memset(last, -1);
        if (path) |states| {
            for (states, positions.items) |state, frame| {
                const wi = st.state_word[@intCast(state)];
                if (wi < 0) continue;
                const k: usize = @intCast(wi);
                if (first[k] < 0) first[k] = frame;
                last[k] = frame;
            }
        }
        for (st.word_idx, 0..) |gi, k| {
            out[gi].speaker = col;
            if (path == null or first[k] < 0) continue;
            const start = @as(f64, @floatFromInt(first[k])) * step;
            const end = @min(@as(f64, @floatFromInt(last[k] + 1)) * step, start + p.maximum_token_len);
            out[gi] = .{ .start = start, .end = end, .speaker = col, .failed = false };
        }
    }
    // A word with no path stands where the one before it ends.
    var prev_end: f64 = 0;
    for (out) |*o| {
        if (o.failed) {
            o.start = prev_end;
            o.end = prev_end;
        } else prev_end = o.end;
    }
    try refine(w, out, words, streams, mapping, spk, step, p);
    return out;
}

// ── streams ──

const Stream = struct {
    tag: i32,
    /// The words of the transcript this stream holds, in order.
    word_idx: []usize,
    n_words: usize,
    /// The blank-expanded target: blank, token, blank, token, ... blank.
    labels: []u32,
    /// The word (its index in the stream) each state belongs to; -1 on blanks.
    state_word: []i32,
};

fn buildStreams(w: std.mem.Allocator, words: []const Word, frames: usize) Error![]Stream {
    var tags: std.ArrayList(i32) = .empty;
    for (words) |x| {
        if (std.mem.indexOfScalar(i32, tags.items, x.tag) == null) try tags.append(w, x.tag);
    }
    const out = try w.alloc(Stream, tags.items.len);
    for (tags.items, out) |tag, *st| {
        var idx: std.ArrayList(usize) = .empty;
        var labels: std.ArrayList(u32) = .empty;
        var state_word: std.ArrayList(i32) = .empty;
        try labels.append(w, 0);
        try state_word.append(w, -1);
        for (words, 0..) |x, gi| {
            if (x.tag != tag) continue;
            const k: i32 = @intCast(idx.items.len);
            try idx.append(w, gi);
            for (x.tokens) |tok| {
                try labels.appendSlice(w, &.{ tok, 0 });
                try state_word.appendSlice(w, &.{ k, -1 });
            }
        }
        st.* = .{ .tag = tag, .word_idx = idx.items, .n_words = idx.items.len, .labels = labels.items, .state_word = state_word.items };
        // The path needs a frame per token and one more between equal neighbours.
        var need: usize = 0;
        var prev: ?u32 = null;
        var i: usize = 1;
        while (i < labels.items.len) : (i += 2) {
            need += 1;
            if (prev != null and prev.? == labels.items[i]) need += 1;
            prev = labels.items[i];
        }
        if (need > frames) return error.TooShort;
    }
    return out;
}

fn argmax(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = @intCast(i);
    };
    return best;
}

/// Linear resampling of `[n][count]` activity (clamped to 0..1) onto `target` frames: torch's linspace positions.
fn resample(w: std.mem.Allocator, probs: []const f32, n: usize, count: usize, target: usize) ![]f32 {
    const out = try w.alloc(f32, target * count);
    const clamped = try w.alloc(f32, n * count);
    for (probs[0 .. n * count], clamped) |x, *c| c.* = std.math.clamp(x, 0, 1);
    if (n == target) {
        @memcpy(out, clamped);
        return out;
    }
    if (n == 1) {
        for (0..target) |t| @memcpy(out[t * count ..][0..count], clamped[0..count]);
        return out;
    }
    const end: f32 = @floatFromInt(n - 1);
    const steps: f32 = @floatFromInt(target - 1);
    const stride = end / steps;
    for (0..target) |t| {
        // torch.linspace: from the start for the first half, back from the end after.
        const pos: f32 = if (t < target / 2) stride * @as(f32, @floatFromInt(t)) else end - stride * @as(f32, @floatFromInt(target - t - 1));
        const lo: usize = @intFromFloat(@floor(pos));
        const hi: usize = @min(n - 1, @as(usize, @intFromFloat(@ceil(pos))));
        const frac = pos - @as(f32, @floatFromInt(lo));
        for (0..count) |s| out[t * count + s] = clamped[lo * count + s] * (1 - frac) + clamped[hi * count + s] * frac;
    }
    return out;
}

// ── speaker mapping ──

fn mapSpeakers(w: std.mem.Allocator, cols: usize, greedy: []const u32, grid: []const f32, count: usize, streams: []const Stream, threshold: f32, mapping: []i32) !void {
    const frames = greedy.len;
    const observed = try w.alloc(f64, count * cols);
    @memset(observed, 0);
    for (0..frames) |t| {
        if (greedy[t] == 0 or (t > 0 and greedy[t] == greedy[t - 1])) continue;
        for (0..count) |c| if (grid[t * count + c] >= threshold) {
            observed[c * cols + greedy[t]] += 1;
        };
    }
    const expected = try w.alloc(f64, streams.len * cols);
    @memset(expected, 0);
    for (streams, 0..) |st, si| {
        var i: usize = 1;
        while (i < st.labels.len) : (i += 2) expected[si * cols + st.labels[i]] += 1;
    }
    const idf = try w.alloc(f64, cols);
    for (0..cols) |k| {
        var seen: f64 = 0;
        for (0..count) |c| seen += @floatFromInt(@intFromBool(observed[c * cols + k] > 0));
        idf[k] = std.math.log1p(@as(f64, @floatFromInt(count)) / (1 + seen));
    }
    const sim = try w.alloc(f64, streams.len * count);
    for (0..streams.len) |si| {
        var na: f64 = 0;
        var exp_sum: f64 = 0;
        for (0..cols) |k| {
            const x = @sqrt(expected[si * cols + k] * idf[k]);
            na += x * x;
            exp_sum += expected[si * cols + k];
        }
        na = @sqrt(na);
        for (0..count) |c| {
            var dot: f64 = 0;
            var nb: f64 = 0;
            var obs_sum: f64 = 0;
            for (0..cols) |k| {
                const x = @sqrt(expected[si * cols + k] * idf[k]);
                const y = @sqrt(observed[c * cols + k] * idf[k]);
                dot += x * y;
                nb += y * y;
                obs_sum += observed[c * cols + k];
            }
            nb = @sqrt(nb);
            var s = dot / @max(1e-8, na * nb);
            s -= 0.1 * @abs(@log((exp_sum + 1) / (obs_sum + 1)));
            sim[si * count + c] = s;
        }
    }
    const assigned = try w.alloc(i32, streams.len);
    try assign(w, sim, streams.len, count, assigned);
    @memcpy(mapping, assigned);
}

/// Maximizes the summed similarity of a one-to-one assignment of rows to columns (the Hungarian method, rows <= or >
/// columns); an unassigned row is -1.
fn assign(w: std.mem.Allocator, sim: []const f64, n_rows: usize, n_cols: usize, out: []i32) !void {
    @memset(out, -1);
    // The classic potentials form wants rows <= columns: transpose when there are more rows.
    const transpose = n_rows > n_cols;
    const n = if (transpose) n_cols else n_rows;
    const m = if (transpose) n_rows else n_cols;
    const cost = try w.alloc(f64, (n + 1) * (m + 1));
    for (1..n + 1) |i| for (1..m + 1) |j| {
        const s = if (transpose) sim[(j - 1) * n_cols + (i - 1)] else sim[(i - 1) * n_cols + (j - 1)];
        cost[i * (m + 1) + j] = -s;
    };
    const u = try w.alloc(f64, n + 1);
    const v = try w.alloc(f64, m + 1);
    const p = try w.alloc(usize, m + 1);
    const way = try w.alloc(usize, m + 1);
    @memset(u, 0);
    @memset(v, 0);
    @memset(p, 0);
    @memset(way, 0);
    for (1..n + 1) |i| {
        p[0] = i;
        var j0: usize = 0;
        const minv = try w.alloc(f64, m + 1);
        const used = try w.alloc(bool, m + 1);
        @memset(minv, std.math.inf(f64));
        @memset(used, false);
        while (true) {
            used[j0] = true;
            const row_i = p[j0];
            var delta = std.math.inf(f64);
            var j1: usize = 0;
            for (1..m + 1) |j| {
                if (used[j]) continue;
                const cur = cost[row_i * (m + 1) + j] - u[row_i] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (0..m + 1) |j| {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else minv[j] -= delta;
            }
            j0 = j1;
            if (p[j0] == 0) break;
        }
        while (j0 != 0) {
            const j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
    }
    for (1..m + 1) |j| {
        if (p[j] == 0) continue;
        if (transpose) out[j - 1] = @intCast(p[j] - 1) else out[p[j] - 1] = @intCast(j - 1);
    }
}

// ── the banded search ──

/// numpy's `rint`: halves round to even.
fn rint(x: f64) i64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5) return @intFromFloat(2 * @round(x / 2));
    return @intFromFloat(r);
}

fn interp(x: f64, xs: []const f64, ys: []const f64) f64 {
    if (x >= xs[xs.len - 1]) return ys[ys.len - 1];
    if (x <= xs[0]) return ys[0];
    var j: usize = 0;
    while (j + 1 < xs.len and xs[j + 1] <= x) j += 1;
    if (j + 1 >= xs.len) return ys[j];
    const slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
    return slope * (x - xs[j]) + ys[j];
}

const Trigram = [3]u32;

fn trigrams(w: std.mem.Allocator, tokens: []const u32) !std.AutoHashMap(Trigram, std.ArrayList(usize)) {
    var table = std.AutoHashMap(Trigram, std.ArrayList(usize)).init(w);
    if (tokens.len < 3) return table;
    for (0..tokens.len - 2) |i| {
        const slot = try table.getOrPut(.{ tokens[i], tokens[i + 1], tokens[i + 2] });
        if (!slot.found_existing) slot.value_ptr.* = .empty;
        try slot.value_ptr.append(w, i);
    }
    return table;
}

const Anchor = struct { time: usize, state: usize };

fn anchorStarts(w: std.mem.Allocator, greedy: []const u32, target: []const u32, states: usize, half: usize, width: usize) ![]usize {
    const frames = greedy.len;
    var observed: std.ArrayList(u32) = .empty;
    var times: std.ArrayList(usize) = .empty;
    for (greedy, 0..) |g, t| {
        if (g != 0 and (t == 0 or g != greedy[t - 1])) {
            try observed.append(w, g);
            try times.append(w, t);
        }
    }
    var left = try trigrams(w, observed.items);
    var right = try trigrams(w, target);
    var pairs: std.ArrayList(Anchor) = .empty;
    var it = left.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.items.len != 1) continue;
        const r = right.get(e.key_ptr.*) orelse continue;
        if (r.items.len != 1) continue;
        try pairs.append(w, .{ .time = times.items[e.value_ptr.items[0] + 1], .state = 2 * (r.items[0] + 1) + 1 });
    }
    std.mem.sort(Anchor, pairs.items, {}, struct {
        fn less(_: void, x: Anchor, y: Anchor) bool {
            return if (x.time != y.time) x.time < y.time else x.state < y.state;
        }
    }.less);
    // The longest chain increasing in time and state (strictly in state).
    var tails: std.ArrayList(usize) = .empty;
    var tail_ids: std.ArrayList(usize) = .empty;
    const parent = try w.alloc(i64, pairs.items.len);
    for (pairs.items, 0..) |pr, i| {
        var lo: usize = 0;
        var hi: usize = tails.items.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (tails.items[mid] < pr.state) lo = mid + 1 else hi = mid;
        }
        parent[i] = if (lo > 0) @intCast(tail_ids.items[lo - 1]) else -1;
        if (lo == tails.items.len) {
            try tails.append(w, pr.state);
            try tail_ids.append(w, i);
        } else {
            tails.items[lo] = pr.state;
            tail_ids.items[lo] = i;
        }
    }
    var chosen: std.ArrayList(Anchor) = .empty;
    var k: i64 = if (tail_ids.items.len > 0) @intCast(tail_ids.items[tail_ids.items.len - 1]) else -1;
    while (k >= 0) : (k = parent[@intCast(k)]) try chosen.append(w, pairs.items[@intCast(k)]);
    std.mem.reverse(Anchor, chosen.items);

    var xs: std.ArrayList(f64) = .empty;
    var ys: std.ArrayList(f64) = .empty;
    try xs.append(w, 0);
    try ys.append(w, 0);
    for (chosen.items) |c| if (c.time > 0 and c.time + 1 < frames) {
        try xs.append(w, @floatFromInt(c.time));
        try ys.append(w, @floatFromInt(c.state));
    };
    try xs.append(w, @floatFromInt(frames - 1));
    try ys.append(w, @floatFromInt(states - 1));

    const starts = try w.alloc(usize, frames);
    const raw = try w.alloc(i64, frames);
    const max_start: i64 = @intCast(states - width);
    for (0..frames) |t| {
        const center = interp(@floatFromInt(t), xs.items, ys.items);
        raw[t] = std.math.clamp(rint(center) - @as(i64, @intCast(half)), 0, max_start);
    }
    // A band moves at most two states a frame, and must still reach the end.
    var run: i64 = std.math.maxInt(i64);
    for (0..frames) |t| {
        const shifted = raw[t] - 2 * @as(i64, @intCast(t));
        run = @min(run, shifted);
        var s = run + 2 * @as(i64, @intCast(t));
        const floor_s = @max(0, max_start - 2 * @as(i64, @intCast(frames - 1 - t)));
        s = @max(s, floor_s);
        starts[t] = @intCast(s);
    }
    return starts;
}

/// The best path through `labels` over `positions`' frames as a state per position; null when none is legal.
fn bandedSearch(w: std.mem.Allocator, lp: []const f32, cols: usize, labels: []const u32, positions: []const u32, prior: []const f32, greedy: []const u32) !?[]u32 {
    const frames = positions.len;
    const states = labels.len;
    if (frames == 0) return null;
    var target: std.ArrayList(u32) = .empty;
    var i: usize = 1;
    while (i < states) : (i += 2) try target.append(w, labels[i]);
    var need: usize = target.items.len;
    for (1..target.items.len) |k| need += @intFromBool(target.items[k] == target.items[k - 1]);
    if (frames < need) return null;

    const half = if (frames < 1000) states else @min(states, @as(usize, @intFromFloat(@ceil(0.10 * @as(f64, @floatFromInt(states)) + 100))));
    const width = @min(states, 2 * half + 1);
    const greedy_pos = try w.alloc(u32, frames);
    for (positions, greedy_pos) |p, *g| g.* = greedy[p];
    const starts = try anchorStarts(w, greedy_pos, target.items, states, half, width);

    const can_skip = try w.alloc(bool, states);
    for (0..states) |s| can_skip[s] = labels[s] != 0 and (s >= 2 and labels[s] != labels[s - 2]);
    const neg = -std.math.inf(f32);
    const pointers = try w.alloc(u8, frames * width);
    @memset(pointers, 0);
    var previous = try w.alloc(f32, width);
    var current = try w.alloc(f32, width);
    @memset(previous, neg);
    {
        const row = lp[@as(usize, positions[0]) * cols ..][0..cols];
        for (0..@min(2, width)) |j| {
            previous[j] = row[labels[j]];
            if (labels[j] != 0) previous[j] = addPrior(previous[j], prior[positions[0]]);
        }
    }
    for (1..frames) |t| {
        const start = starts[t];
        const delta = start - starts[t - 1];
        const row = lp[@as(usize, positions[t]) * cols ..][0..cols];
        for (0..width) |j| {
            const k = j + delta;
            var best: f32 = if (k < width) previous[k] else neg;
            var move: u8 = 0;
            const one: f32 = if (k >= 1 and k - 1 < width) previous[k - 1] else neg;
            if (one > best) {
                best = one;
                move = 1;
            }
            const two: f32 = if (k >= 2 and k - 2 < width) previous[k - 2] else neg;
            if (two > best and can_skip[start + j]) {
                best = two;
                move = 2;
            }
            pointers[t * width + j] = move;
            const label = labels[start + j];
            var v = best + row[label];
            if (label != 0) v = addPrior(v, prior[positions[t]]);
            current[j] = v;
        }
        std.mem.swap([]f32, &previous, &current);
    }

    // The terminal states: the last blank or the last token.
    const last_start = starts[frames - 1];
    var best_state: ?usize = null;
    var best_score: f32 = neg;
    for ([_]usize{ states - 1, states - 2 }) |e| {
        if (e < last_start or e - last_start >= width) continue;
        const sc = previous[e - last_start];
        if (best_state == null or sc > best_score) {
            best_state = e;
            best_score = sc;
        }
    }
    const end = best_state orelse return null;
    if (!std.math.isFinite(best_score)) return null;
    const path = try w.alloc(u32, frames);
    var state = end;
    var t = frames;
    while (t > 0) {
        t -= 1;
        path[t] = @intCast(state);
        if (t > 0) {
            if (state < starts[t]) return null;
            state -= pointers[t * width + (state - starts[t])];
        }
    }
    return path;
}

/// A float32 score plus a log-prior, summed in double and rounded once (numpy's in-place add).
fn addPrior(x: f32, pr: f32) f32 {
    return x + pr;
}

// ── refinement ──

const Interval = struct { start: f64, end: f64 };

/// Same-speaker word boundaries inside the diarization's activity segments (`_refine_word_timestamps`).
fn refine(w: std.mem.Allocator, out: []Timed, words: []const Word, streams: []const Stream, mapping: []const i32, spk: ?Speakers, step: f64, p: Params) !void {
    _ = words;
    const limit = @as(f64, @floatFromInt(p.max_word_frames)) * @min(step, 0.08);
    for (streams, 0..) |st, si| {
        // The stream's words by onset.
        const order = try w.alloc(usize, st.n_words);
        for (order, 0..) |*o, k| o.* = st.word_idx[k];
        std.mem.sort(usize, order, out, struct {
            fn less(o: []Timed, x: usize, y: usize) bool {
                return o[x].start < o[y].start;
            }
        }.less);
        var intervals: []Interval = &.{};
        if (spk != null and mapping[si] >= 0) intervals = try segments(w, spk.?, @intCast(mapping[si]));
        const member = try w.alloc(i64, order.len);
        for (order, member) |gi, *m| {
            // bisect_right(starts, start) - 1
            var lo: usize = 0;
            var hi: usize = intervals.len;
            while (lo < hi) {
                const mid = (lo + hi) / 2;
                if (out[gi].start < intervals[mid].start) hi = mid else lo = mid + 1;
            }
            const idx: i64 = @as(i64, @intCast(lo)) - 1;
            m.* = if (idx >= 0 and out[gi].start < intervals[@intCast(idx)].end) idx else -1;
        }
        var seen: std.ArrayList(i64) = .empty;
        for (order, member) |gi, m| {
            if (m < 0 or std.mem.indexOfScalar(i64, seen.items, m) != null) continue;
            const onset = intervals[@intCast(m)].start;
            if (out[gi].start - onset > 0 and out[gi].start - onset <= limit) out[gi].start = onset;
            try seen.append(w, m);
        }
        for (order, member, 0..) |gi, m, i| {
            const next_start: ?f64 = if (i + 1 < order.len) out[order[i + 1]].start else null;
            const end: f64 = if (m < 0)
                (next_start orelse out[gi].end)
            else if (next_start != null and member[i + 1] == m)
                next_start.?
            else
                intervals[@intCast(m)].end;
            out[gi].end = @max(out[gi].start, @min(end, out[gi].start + limit));
        }
    }
}

/// Speaker `col`'s activity runs as seconds, from the 10 ms labels (`diarization_segments`).
fn segments(w: std.mem.Allocator, s: Speakers, col: usize) ![]Interval {
    var list: std.ArrayList(Interval) = .empty;
    var t: usize = 0;
    while (t < s.diar_frames) {
        if (s.diar[t * s.count + col] == 0) {
            t += 1;
            continue;
        }
        var e = t;
        while (e < s.diar_frames and s.diar[e * s.count + col] != 0) e += 1;
        const start = @as(f64, @floatFromInt(t)) * s.diar_step;
        const end = @min(@as(f64, @floatFromInt(e)) * s.diar_step, s.duration);
        if (end > start) try list.append(w, .{ .start = start, .end = end });
        t = e;
    }
    return list.items;
}

// ── tests ──

const testing = std.testing;

const Fixture = struct {
    name: []const u8,
    frames: usize,
    cols: usize,
    step: f64,
    log_probs: []const f32,
    words: []const struct { tag: i32, tokens: []const u32 },
    speaker_rows: ?[]const f32,
    speakers: usize,
    diar: ?[]const u8,
    diar_frames: usize,
    expected: []const struct { start: f64, end: f64 },
    mapping: std.json.Value,
    failed: bool,
};

test "ctc_align: NVIDIA's aligner on its own fixtures (word times, speaker streams, the band past 1000 frames)" {
    const a = testing.allocator;
    const parsed = try std.json.parseFromSlice(struct {
        params: struct { speaker_activity_threshold: f32, speaker_logprob_weight: f32, maximum_token_len: f64, max_word_frames: u32, epsilon: f32 },
        cases: []const Fixture,
    }, a, @embedFile("fixtures/ctc_align.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const p: Params = .{
        .speaker_activity_threshold = parsed.value.params.speaker_activity_threshold,
        .speaker_logprob_weight = parsed.value.params.speaker_logprob_weight,
        .maximum_token_len = parsed.value.params.maximum_token_len,
        .max_word_frames = parsed.value.params.max_word_frames,
        .epsilon = parsed.value.params.epsilon,
    };
    for (parsed.value.cases) |c| {
        const words = try a.alloc(Word, c.words.len);
        defer a.free(words);
        for (c.words, words) |x, *o| o.* = .{ .tag = x.tag, .tokens = x.tokens };
        var spk: ?Speakers = null;
        if (c.speaker_rows) |rows| {
            spk = .{ .rows = rows, .n_rows = rows.len / c.speakers, .count = c.speakers, .diar = c.diar.?, .diar_frames = c.diar_frames, .duration = @as(f64, @floatFromInt(c.frames)) * c.step };
        }
        const got = try alignWords(a, c.log_probs, c.frames, c.cols, words, spk, c.step, p);
        defer a.free(got);
        for (got, c.expected, 0..) |g, e, i| {
            testing.expectApproxEqAbs(e.start, g.start, 1e-6) catch |err| {
                std.debug.print("{s}: word {d} start {d} want {d}\n", .{ c.name, i, g.start, e.start });
                return err;
            };
            testing.expectApproxEqAbs(e.end, g.end, 1e-6) catch |err| {
                std.debug.print("{s}: word {d} end {d} want {d}\n", .{ c.name, i, g.end, e.end });
                return err;
            };
        }
        // The streams map to the columns NVIDIA's assignment picks.
        var it = c.mapping.object.iterator();
        while (it.next()) |m| {
            const tag = try std.fmt.parseInt(i32, m.key_ptr.*, 10);
            const want: i32 = if (m.value_ptr.* == .integer) @intCast(m.value_ptr.integer) else -1;
            for (words, got) |x, g| if (x.tag == tag) {
                testing.expectEqual(want, g.speaker) catch |err| {
                    std.debug.print("{s}: tag {d} maps to {d}, want {d}\n", .{ c.name, tag, g.speaker, want });
                    return err;
                };
                break;
            };
        }
    }
}

test "ctc_align: an impossible transcript is refused, a word the path cannot reach stands where the last ended" {
    const a = testing.allocator;
    const lp = try a.alloc(f32, 2 * 3);
    defer a.free(lp);
    @memset(lp, -1);
    // Two equal tokens need three frames.
    try testing.expectError(error.TooShort, alignWords(a, lp, 2, 3, &.{.{ .tag = 0, .tokens = &.{ 1, 1 } }}, null, 0.04, .{}));
}

test "ctc_align: the assignment maximizes the similarity (rows or columns in surplus)" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var out: [3]i32 = undefined;
    // 2 rows over 3 columns: best is row0 -> col1 (0.9), row1 -> col2 (0.8).
    try assign(arena.allocator(), &.{ 0.1, 0.9, 0.2, 0.3, 0.1, 0.8 }, 2, 3, out[0..2]);
    try testing.expectEqualSlices(i32, &.{ 1, 2 }, out[0..2]);
    // 3 rows over 2 columns: one row is left out.
    try assign(arena.allocator(), &.{ 0.9, 0.1, 0.8, 0.7, 0.1, 0.2 }, 3, 2, out[0..3]);
    try testing.expectEqualSlices(i32, &.{ 0, 1, -1 }, out[0..3]);
}
