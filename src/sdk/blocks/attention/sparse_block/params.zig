//! Block-sparse attention's shape: an indexer scores blocks of keys, and each query attends the `top_k` best blocks
//! it can see (its first `init` and last `local` blocks forced among them), plus the keys after its last complete
//! block. Pure host arithmetic; the ops read these answers.

const std = @import("std");

/// Which blocks a query attends.
pub const Blocks = struct {
    size: u32,
    top_k: u32,
    init: u32 = 0,
    /// The blocks ending at the query's own, counted in `top_k`; 0 leaves its incomplete block outside `top_k`.
    local: u32 = 0,

    /// Blocks every key of which a query at cache position `pos` sees; its keys past them are always attended.
    pub fn complete(b: Blocks, pos: c_int) c_int {
        return if (pos < 0) 0 else @divTrunc(pos + 1, @as(c_int, @intCast(b.size)));
    }

    /// The candidate blocks `[0, bound)` of a query at `pos`: its complete blocks, and with local blocks its own.
    pub fn bound(b: Blocks, pos: c_int) c_int {
        if (b.local == 0 or pos < 0) return b.complete(pos);
        return @divTrunc(pos, @as(c_int, @intCast(b.size))) + 1;
    }

    /// Whether block `blk` is a pick of every query at `pos` that can see it, whatever its score.
    pub fn forced(b: Blocks, blk: c_int, pos: c_int) bool {
        if (blk < 0) return false;
        if (blk < b.init) return true;
        const own = @divTrunc(pos, @as(c_int, @intCast(b.size)));
        return blk <= own and blk + @as(c_int, @intCast(b.local)) > own;
    }

    /// Whether every row from `row0` on has all `nb` scored blocks as candidates.
    pub fn allSee(b: Blocks, row0: c_int, nb: c_int) bool {
        return b.bound(row0) >= nb;
    }

    /// Per row `r` (cache position `row0 + r`), the candidate bound clamped to the `nb` scored blocks; `all_vis`
    /// claims every row sees all `nb`, and is honored verbatim.
    pub fn selectBounds(b: Blocks, out: []i32, row0: c_int, nb: c_int, all_vis: bool) void {
        for (out, 0..) |*o, r| o.* = if (all_vis) nb else @min(nb, b.bound(row0 + @as(c_int, @intCast(r))));
    }
};

pub const Score = enum {
    /// sum over the index heads of relu(q_h . pooled block key), one pick list per query (Qwen3.8-Flash-Next QSA:
    /// the pooled key is the mean of the block's keys, normed and roped by the model).
    pooled_relu_sum,
    /// max over the block's visible keys of q_h . k, one pick list per index head (MiniMax-M3 `sparse_score_type`
    /// max; q head h reads index head h / (q heads / index heads)).
    key_max,
};

pub const Params = struct {
    blocks: Blocks,
    index_heads: u32,
    index_dim: u32,
    score: Score,
    /// The attention layers that are sparse; null = every one.
    layers: ?[]const bool = null,

    pub fn validate(p: Params) error{InvalidSparseBlockParams}!void {
        const b = p.blocks;
        if (b.size == 0 or b.top_k == 0 or p.index_heads == 0 or p.index_dim == 0) return error.InvalidSparseBlockParams;
        if (@as(u64, b.init) + b.local > b.top_k) return error.InvalidSparseBlockParams;
    }

    pub fn sparseAt(p: Params, layer: usize) bool {
        const l = p.layers orelse return true;
        return layer < l.len and l[layer];
    }

    /// Pick lists per query row: one, or one per index head.
    pub fn listsPerQuery(p: Params) u32 {
        return if (p.score == .key_max) p.index_heads else 1;
    }
};

const testing = std.testing;

test "sparse block params: bounds are the complete blocks, plus the query's own with local blocks" {
    const qsa: Blocks = .{ .size = 4, .top_k = 512 };
    var out: [6]i32 = undefined;
    qsa.selectBounds(&out, 0, 700, false);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1, 1 }, &out);
    qsa.selectBounds(&out, 2799, 700, false);
    try testing.expectEqualSlices(i32, &.{ 700, 700, 700, 700, 700, 700 }, &out);
    qsa.selectBounds(&out, 2044, 700, false);
    try testing.expectEqualSlices(i32, &.{ 511, 511, 511, 512, 512, 512 }, &out);
    qsa.selectBounds(&out, 4, 24, true);
    try testing.expectEqualSlices(i32, &.{ 24, 24, 24, 24, 24, 24 }, &out);
    try testing.expect(qsa.allSee(2051, 513) and !qsa.allSee(2050, 513) and !qsa.allSee(0, 513) and qsa.allSee(0, 0));

    const mm: Blocks = .{ .size = 4, .top_k = 3, .init = 1, .local = 1 };
    mm.selectBounds(&out, 2, 700, false);
    try testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2, 2, 2 }, &out);
    try testing.expect(mm.forced(0, 30) and mm.forced(7, 30) and !mm.forced(6, 30) and !mm.forced(8, 30));
    try testing.expect(!qsa.forced(0, 30) and !qsa.forced(7, 30));

    const key_max: Params = .{ .blocks = mm, .index_heads = 2, .index_dim = 8, .score = .key_max };
    try key_max.validate();
    try testing.expectEqual(@as(u32, 2), key_max.listsPerQuery());
    try testing.expectEqual(@as(u32, 1), (Params{ .blocks = qsa, .index_heads = 4, .index_dim = 128, .score = .pooled_relu_sum }).listsPerQuery());
    try testing.expectError(error.InvalidSparseBlockParams, (Params{ .blocks = .{ .size = 4, .top_k = 1, .init = 1, .local = 1 }, .index_heads = 1, .index_dim = 1, .score = .key_max }).validate());
    const layers = [_]bool{ false, true };
    const some: Params = .{ .blocks = qsa, .index_heads = 1, .index_dim = 1, .score = .key_max, .layers = &layers };
    try testing.expect(!some.sparseAt(0) and some.sparseAt(1) and !some.sparseAt(2) and key_max.sparseAt(9));
}
