//! A wide prompt layer's indexer spans in groups, MLX-free (the MLX side is a few lines in deepseek_v41.zig's
//! `wideLayer`). Std only, tested by `zig build test`. The bank's bill is as before (: "Group only, keep
//! the bill").
//!
//! An index layer (`ly.idx`) runs its indexer once a 256-row attention span (`attention` -> `indexer`), and a span holds
//! `[span, nb]` cells of scratch until its picks are made: the f32 scores, and in the layers after the candidate source
//! the candidate mask (bool) and the masked scores (f32), 9 bytes a cell at most (`cell_bytes`). The layer's spans are
//! built before the layer is evaluated, and MLX frees a span's scratch only once the GPU has run its picks, so a layer
//! can hold every span's scratch at once: 64 spans x 256 rows x nb cells, about 38 GB at nb 262144 (a 256K prompt).
//! `groupSpans` is how many spans the layer builds before it evaluates their picks: all of them
//! while the whole layer's scratch fits in `bound_bytes`, so short prompts run exactly as before, else the most spans
//! whose scratch fits, at least one. Each span's own graph is untouched (the same kernels over the same inputs), so its
//! bytes are; only where the layer's evaluation is cut moves.

const std = @import("std");

/// The most indexer scratch a wide layer holds at once (2.5 GiB).
pub const bound_bytes: u64 = 5 << 29;
/// A cell's most scratch bytes: the f32 scores, the bool candidate mask and the f32 masked scores.
pub const cell_bytes: u64 = 9;

/// The spans of `n_spans` (each `span` rows, `nb` compressed positions their indexer scores) a wide layer builds before it
/// evaluates their picks.
pub fn groupSpans(n_spans: usize, span: usize, nb: usize) usize {
    const per: u64 = @as(u64, span) * nb * cell_bytes;
    if (per == 0 or @as(u64, n_spans) * per <= bound_bytes) return n_spans;
    return @intCast(@max(1, bound_bytes / per));
}

/// Whether span `k` (0-based) of `n_spans` ends a group of `g` before the layer's own evaluation: its picks and the ones
/// before it in the group are evaluated there (the last group is the layer's own evaluation).
pub fn endsGroup(k: usize, g: usize, n_spans: usize) bool {
    return g < n_spans and (k + 1) % g == 0 and k + 1 < n_spans;
}


// ── tests ──
const testing = std.testing;

test "idx_group: a short prompt's layer runs as one group, a long one's in groups under the bound" {
    // 16K: 64 spans x 256 rows x 16384 cells x 9 B = 2.42 GB, under 2.5 GiB: one group, no cut
    try testing.expectEqual(@as(usize, 64), groupSpans(64, 256, 16384));
    try testing.expectEqual(@as(usize, 64), groupSpans(64, 256, 8192));
    try testing.expectEqual(@as(usize, 1), groupSpans(1, 256, 1)); // one span alone: its own group
    // 64K's last chunk (nb 65536): 17 spans a group; 256K's (262144): 4; 278528: 4
    try testing.expectEqual(@as(usize, 17), groupSpans(64, 256, 65536));
    try testing.expectEqual(@as(usize, 4), groupSpans(64, 256, 262144));
    try testing.expectEqual(@as(usize, 4), groupSpans(64, 256, 278528));
    // a group's scratch is under the bound wherever the layer's is cut
    for ([_]usize{ 16385, 32768, 65536, 131072, 262144, 278528, 1047488 }) |nb| {
        const g = groupSpans(64, 256, nb);
        try testing.expect(g >= 1 and g <= 64);
        if (g < 64) try testing.expect(@as(u64, g) * 256 * nb * cell_bytes <= bound_bytes or g == 1);
    }
}

test "idx_group: the cuts cover every span once, in order" {
    for ([_]usize{ 1, 2, 3, 4, 17, 35, 63, 64 }) |g| {
        var groups: usize = 0;
        var start: usize = 0;
        var seen: usize = 0;
        for (0..64) |k| {
            seen += 1;
            if (endsGroup(k, g, 64)) {
                try testing.expect(k + 1 - start <= g);
                start = k + 1;
                groups += 1;
            }
        }
        try testing.expectEqual(@as(usize, 64), seen);
        try testing.expect(64 - start <= g); // the layer's own evaluation takes the last group
        try testing.expectEqual(if (g >= 64) @as(usize, 0) else (64 + g - 1) / g - 1, groups);
    }
}

// The selection, grouped and ungrouped, on a synthetic case: a CPU model of one index layer's per-span selection as
// `attention` -> `indexerQ` makes it (each span at its own p0, its rows' scores over the groups they see, the candidate
// source's top blocks with a row's newest block pinned, the later layer's scores inside them, top-k in a fixed order
// (larger first, then the lower index), -1 for a pick past what the row sees), over a chunk's spans with the layer's
// evaluation cut every g spans (`endsGroup`) and uncut. A span's picks are a function of its own inputs alone, so every
// cut gives the same picks; on the GPU the same holds for MLX's evaluation (the same kernels over the same inputs), which
// the box's byte gate checks.
const Sel = struct {
    const L = 64;
    const span = 8;
    const ratio = 1;
    const bs = 4; // candidate block size
    const kb = 2; // candidate blocks kept (the newest pinned, one by score)
    const topk = 5;

    fn score(layer: usize, pos: usize, col: usize) f32 {
        // deterministic, with ties: a hash folded to 7 levels
        const h = (layer *% 2654435761 +% pos *% 40503 +% col *% 2246822519) % 7;
        return @floatFromInt(h);
    }
    /// One row's top-k over the first `n` of `s`: larger first, the lower index on ties.
    fn topK(s: []const f32, n: usize, out: []i32) void {
        var used: [256]bool = @splat(false);
        for (out) |*o| {
            var best: ?usize = null;
            for (0..n) |cix| {
                if (used[cix]) continue;
                if (best == null or s[cix] > s[best.?]) best = cix;
            }
            used[best.?] = true;
            o.* = @intCast(best.?);
        }
    }
    /// One span: rows [r0, r0 + span) of a chunk at `p0`, at the span's own position (`attention`'s p0 + lo): the
    /// candidate source's blocks into `cand`, then the later layer's picks into `picks` (`topk` a row, row-major).
    fn spanRows(p0: usize, r0: usize, cand: [][32]bool, picks: []i32) void {
        const sp0 = p0 + r0;
        const nb = (sp0 + span) / ratio;
        const n_blocks = (nb + bs - 1) / bs;
        const k = @min(topk, nb);
        var s: [256]f32 = undefined;
        for (0..span) |l| {
            const row = r0 + l;
            const lens = (sp0 + l + 1) / ratio;
            var blocks: [32]f32 = undefined;
            for (0..n_blocks) |b| {
                var mx: f32 = -std.math.inf(f32);
                for (b * bs..@min(nb, b * bs + bs)) |cix| {
                    if (cix < lens) mx = @max(mx, score(0, sp0 + l, cix));
                }
                blocks[b] = mx;
            }
            blocks[(lens - 1) / bs] = std.math.inf(f32);
            var chosen: [kb]i32 = undefined;
            topK(blocks[0..n_blocks], n_blocks, &chosen);
            cand[row] = @splat(false);
            for (chosen) |b| {
                if (blocks[@intCast(b)] > -std.math.inf(f32)) cand[row][@intCast(b)] = true;
            }
            for (0..nb) |cix| s[cix] = if (cix < lens and cand[row][cix / bs]) score(1, sp0 + l, cix) else -std.math.inf(f32);
            var idx: [topk]i32 = @splat(0);
            topK(s[0..nb], nb, idx[0..k]);
            for (0..topk) |j| picks[row * topk + j] = if (j >= k) -2 else if (idx[j] < @as(i32, @intCast(lens))) idx[j] else -1;
        }
    }
};

test "idx_group: grouped and ungrouped selections are identical on a synthetic case" {
    const n_spans = Sel.L / Sel.span;
    for ([_]usize{ 0, 40 }) |p0| {
        // uncut: the layer's spans in one evaluation
        var cand_all: [Sel.L][32]bool = undefined;
        var picks_all: [Sel.L * Sel.topk]i32 = undefined;
        for (0..n_spans) |k| Sel.spanRows(p0, k * Sel.span, &cand_all, &picks_all);
        for ([_]usize{ 1, 2, 3, 5, n_spans }) |g| {
            var cand: [Sel.L][32]bool = undefined;
            var picks: [Sel.L * Sel.topk]i32 = undefined;
            @memset(&picks, -7);
            // cut every g spans, as wideLayer cuts its evaluation
            var start: usize = 0;
            for (0..n_spans) |k| {
                if (endsGroup(k, g, n_spans) or k + 1 == n_spans) {
                    for (start..k + 1) |j| Sel.spanRows(p0, j * Sel.span, &cand, &picks);
                    start = k + 1;
                }
            }
            try testing.expectEqual(@as(usize, n_spans), start);
            try testing.expectEqualSlices(i32, &picks_all, &picks);
            for (0..Sel.L) |row| try testing.expectEqualSlices(bool, &cand_all[row], &cand[row]);
        }
        // the case has picks a row does not see (-1) at the context's start, and real picks everywhere
        var unseen: usize = 0;
        var real: usize = 0;
        for (picks_all) |v| {
            unseen += @intFromBool(v == -1);
            real += @intFromBool(v >= 0);
        }
        if (p0 == 0) try testing.expect(unseen > 0);
        try testing.expect(real > Sel.L);
    }
}
