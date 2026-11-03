//! Block-sparse attention in float64 on the host, one query row at a time: what the ops in this folder must equal.

const std = @import("std");
const Blocks = @import("params.zig").Blocks;

/// `key_max` block scores of one index head's query `q` (`d` wide) at cache position `pos` over `keys` (`[kv][d]`):
/// per block the max of q . k over its keys at or before `pos`, -inf for a block with none.
pub fn keyMaxScores(out: []f64, b: Blocks, q: []const f32, keys: []const f32, d: usize, pos: usize) void {
    const bs: usize = b.size;
    for (out, 0..) |*o, blk| {
        o.* = -std.math.inf(f64);
        var j = blk * bs;
        while (j < (blk + 1) * bs and j <= pos and j * d < keys.len) : (j += 1) {
            var dot: f64 = 0;
            for (0..d) |i| dot += @as(f64, q[i]) * @as(f64, keys[j * d + i]);
            o.* = @max(o.*, dot);
        }
    }
}

/// The picks of a query at `pos` from its `scores` (`nb` blocks): forced blocks first, then the best by score
/// (ties to the lower index) among `[0, bound)`, `top_k` in all; ascending, without the query's incomplete block
/// (its keys are the tail). Returns how many of `out` it wrote.
pub fn select(b: Blocks, scores: []const f64, pos: usize, out: []i32) usize {
    const ipos: c_int = @intCast(pos);
    const bound: usize = @intCast(@min(@as(c_int, @intCast(scores.len)), b.bound(ipos)));
    const k = @min(bound, b.top_k);
    var n: usize = 0;
    while (n < k) : (n += 1) {
        var best: ?usize = null;
        var best_v: f64 = undefined;
        for (0..bound) |blk| {
            if (std.mem.indexOfScalar(i32, out[0..n], @intCast(blk)) != null) continue;
            const v = if (b.forced(@intCast(blk), ipos)) std.math.inf(f64) else scores[blk];
            if (best == null or v > best_v) {
                best = blk;
                best_v = v;
            }
        }
        out[n] = @intCast(best.?);
    }
    std.mem.sort(i32, out[0..n], {}, std.sort.asc(i32));
    while (n > 0 and out[n - 1] >= b.complete(ipos)) n -= 1;
    return n;
}

/// One head's attention output (`d` wide) for the query at `pos` over the keys of `picks` and its tail
/// `[complete(pos) * block_size, pos]`; `k`, `v` are `[kv][d]`.
pub fn attend(out: []f64, b: Blocks, q: []const f32, k: []const f32, v: []const f32, d: usize, scale: f64, picks: []const i32, pos: usize) void {
    const bs: usize = b.size;
    var m: f64 = -std.math.inf(f64);
    var z: f64 = 0;
    @memset(out, 0);
    const tail: usize = @as(usize, @intCast(b.complete(@intCast(pos)))) * bs;
    var it = KeyIter{ .picks = picks, .bs = bs, .tail = tail, .pos = pos };
    while (it.next()) |j| {
        var dot: f64 = 0;
        for (0..d) |i| dot += @as(f64, q[i]) * @as(f64, k[j * d + i]);
        const s = dot * scale;
        const m_new = @max(m, s);
        const f = @exp(m - m_new);
        const w = @exp(s - m_new);
        z = z * f + w;
        for (out, 0..) |*o, i| o.* = o.* * f + w * @as(f64, v[j * d + i]);
        m = m_new;
    }
    for (out) |*o| o.* /= z;
}

const KeyIter = struct {
    picks: []const i32,
    bs: usize,
    tail: usize,
    pos: usize,
    i: usize = 0,

    fn next(it: *KeyIter) ?usize {
        const in_picks = it.picks.len * it.bs;
        if (it.i < in_picks) {
            defer it.i += 1;
            return @as(usize, @intCast(it.picks[it.i / it.bs])) * it.bs + it.i % it.bs;
        }
        const j = it.tail + (it.i - in_picks);
        if (j > it.pos) return null;
        it.i += 1;
        return j;
    }
};

const testing = std.testing;

test "sparse block reference: forced init and local blocks take top_k slots, the incomplete block rides the tail" {
    const forced: Blocks = .{ .size = 2, .top_k = 3, .init = 1, .local = 1 };
    const scores = [_]f64{ -5, 9, 7, 8, 1, 3 };
    var out: [3]i32 = undefined;
    // pos 9: own block 4 (complete), forced 0 and 4, best other 1.
    try testing.expectEqualSlices(i32, &.{ 0, 1, 4 }, out[0..select(forced, &scores, 9, &out)]);
    // pos 8: own block 4 incomplete (the tail), forced 0, best other 1.
    try testing.expectEqualSlices(i32, &.{ 0, 1 }, out[0..select(forced, &scores, 8, &out)]);
    const plain: Blocks = .{ .size = 2, .top_k = 2 };
    // pos 8: complete blocks 0..3, best two 1 and 3; block 4 is the tail outside top_k.
    try testing.expectEqualSlices(i32, &.{ 1, 3 }, out[0..select(plain, &scores, 8, &out)]);
    const tie = [_]f64{ 1, 1, 1 };
    try testing.expectEqualSlices(i32, &.{ 0, 1 }, out[0..select(plain, &tie, 5, &out)]);

    var keys: [12]f32 = undefined;
    for (&keys, 0..) |*x, i| x.* = @floatFromInt(i);
    var km: [3]f64 = undefined;
    keyMaxScores(&km, plain, &.{1}, &keys, 1, 2);
    try testing.expectEqualSlices(f64, &.{ 1, 2, -std.math.inf(f64) }, &km);

    // Keys 2,3 (block 1) then the tail 4: softmax over scores 2, 3, 4 of v = k.
    var o: [1]f64 = undefined;
    attend(&o, plain, &.{1}, &keys, &keys, 1, 1, &.{1}, 4);
    const w = [_]f64{ @exp(@as(f64, 2)), @exp(@as(f64, 3)), @exp(@as(f64, 4)) };
    try testing.expectApproxEqRel((2 * w[0] + 3 * w[1] + 4 * w[2]) / (w[0] + w[1] + w[2]), o[0], 1e-12);
}
