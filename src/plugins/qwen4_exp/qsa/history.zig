//! The per-slot QSA key ring and pooled-block history (`QSA_RING_ROWS`), head marks, and `ssmFreeQsaState`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const round_cost_mod = @import("../../../engines/mlx/spec/round_cost.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const SSMCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").SSMCheckpoint;
const captureSsmCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").captureSsmCheckpoint;
const restoreSsmCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").restoreSsmCheckpoint;
const spanPreservingDropIndex = @import("../../../engines/mlx/ssm/checkpoint.zig").spanPreservingDropIndex;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const qsaRawKeyBytes = @import("../../../engines/mlx/ssm/entry.zig").qsaRawKeyBytes;
const ssmRollbackFromCapture = @import("../../../engines/mlx/ssm/entry.zig").ssmRollbackFromCapture;
const Trunk = @import("../trunk.zig").Trunk;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const applyQsaHistoryAt = @import("checkpoint.zig").applyQsaHistoryAt;
const attachQsaHistoryToLatest = @import("checkpoint.zig").attachQsaHistoryToLatest;
const qsaHistoryRows = @import("checkpoint.zig").qsaHistoryRows;

/// Raw indexer-key rows kept per layer (tokens): verify width plus one block.
pub const QSA_RING_ROWS: c_int = 32;

/// Widest `indexer_compress_ratio` the ring serves; past it a checkpoint is refused at load.
pub const QSA_MAX_COMPRESS_RATIO: c_int = 8;

comptime {
    // A partial-accept rollback re-slices the ring to `keep`, so the ring must still hold a
    // whole verify width plus the partial block that width reopened.
    std.debug.assert(QSA_RING_ROWS >= @as(c_int, @intCast(round_cost_mod.MAX_WIDTH)) + QSA_MAX_COMPRESS_RATIO + 1);
}

/// The ring is a fixed size, so a config asking for a wider block than it can hold is refused
/// by name instead of serving a leftover the rollback cannot re-pool.
pub fn checkQsaCompressRatio(ratio: u32) !void {
    if (ratio == 0 or ratio > QSA_MAX_COMPRESS_RATIO) return error.UnsupportedQsaCompressRatio;
}

/// The `pos % ratio` raw indexer rows ending at MTP-head row `pos`, copied while the ring
/// still held them. The trunk restores at one of its checkpoint positions and the head is
/// clamped to that same row; by commit time the head's ring is a whole generated tail past
/// it, so the rows the next append must re-pool are kept here instead.
pub const QsaHeadMark = struct {
    pos: c_int = 0,
    rows: mlx.mlx_array = .{ .ctx = null },
};

/// Marks a head keeps. A prefill's checkpoints are capped at `ssm_checkpoint_max` (16) and a
/// warm turn merges the donor's; past this the list thins like the checkpoints themselves.
pub const QSA_HEAD_MARKS_MAX: usize = 32;

fn markPosOf(m: *const QsaHeadMark) usize {
    return @intCast(@max(m.pos, 0));
}

/// One owner of a head's marks: the live head holds one, its committed snapshot a
/// refcount-sharing copy. Positions ascend.
pub const QsaHeadMarkSet = struct {
    items: [QSA_HEAD_MARKS_MAX]QsaHeadMark = @splat(.{}),
    len: usize = 0,

    pub fn slice(self: *const QsaHeadMarkSet) []const QsaHeadMark {
        return self.items[0..self.len];
    }

    /// A second owner for every mark; both sets then free independently.
    pub fn share(src: []const QsaHeadMark) QsaHeadMarkSet {
        var out: QsaHeadMarkSet = .{};
        for (src) |m| {
            if (m.rows.ctx == null or out.len == QSA_HEAD_MARKS_MAX) continue;
            var arr = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&arr, m.rows);
            out.items[out.len] = .{ .pos = m.pos, .rows = arr };
            out.len += 1;
        }
        return out;
    }

    pub fn deinit(self: *QsaHeadMarkSet) void {
        for (self.items[0..self.len]) |*m| {
            if (m.rows.ctx != null) _ = mlx.mlx_array_free(m.rows);
            m.* = .{};
        }
        self.len = 0;
    }

    pub fn bytes(self: *const QsaHeadMarkSet) u64 {
        var total: u64 = 0;
        for (self.items[0..self.len]) |m| {
            if (m.rows.ctx == null) continue;
            total += @as(u64, mlx.mlx_array_size(m.rows)) * @as(u64, mlx.mlx_array_itemsize(m.rows));
        }
        return total;
    }

    pub fn find(self: *const QsaHeadMarkSet, pos: c_int) ?mlx.mlx_array {
        for (self.items[0..self.len]) |m| {
            if (m.pos == pos and m.rows.ctx != null) return m.rows;
        }
        return null;
    }

    /// Marks above `pos` describe rows a clamp just discarded — the next append writes
    /// different tokens there.
    pub fn dropAbove(self: *QsaHeadMarkSet, pos: c_int) void {
        var i = self.len;
        while (i > 0 and self.items[i - 1].pos > pos) : (i -= 1) {
            if (self.items[i - 1].rows.ctx != null) _ = mlx.mlx_array_free(self.items[i - 1].rows);
            self.items[i - 1] = .{};
        }
        self.len = i;
    }

    fn dropAt(self: *QsaHeadMarkSet, idx: usize) void {
        if (self.items[idx].rows.ctx != null) _ = mlx.mlx_array_free(self.items[idx].rows);
        var i = idx;
        while (i + 1 < self.len) : (i += 1) self.items[i] = self.items[i + 1];
        self.items[self.len - 1] = .{};
        self.len -= 1;
    }

    /// Takes ownership of `rows`. A repeat at `pos` replaces; a full set thins its interior.
    pub fn put(self: *QsaHeadMarkSet, pos: c_int, rows: mlx.mlx_array) void {
        self.dropAbove(pos);
        if (self.len > 0 and self.items[self.len - 1].pos == pos) self.dropAt(self.len - 1);
        if (self.len == QSA_HEAD_MARKS_MAX) {
            self.dropAt(spanPreservingDropIndex(QsaHeadMark, self.slice(), markPosOf, .min_span_recency, null, 0));
        }
        self.items[self.len] = .{ .pos = pos, .rows = rows };
        self.len += 1;
    }
};

pub fn qsaLeftoverAt(aux: mlx.mlx_array, hist: c_int, pos: c_int, ratio: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return qsaRowsEndingAt(aux, hist, pos, @mod(pos, @max(ratio, 1)), s);
}

/// The `want` raw indexer rows ending at row `pos`, out of a buffer whose rows end at logical
/// row `hist`; null when the buffer no longer physically holds them.
fn qsaRowsEndingAt(aux: mlx.mlx_array, hist: c_int, pos: c_int, want: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (want <= 0 or aux.ctx == null) return .{ .ctx = null };
    const sh = mlx.getShape(aux);
    if (sh.len < 2 or sh[1] < want) return .{ .ctx = null };
    const held = sh[1];
    const ring_start = hist - held;
    const want_start = pos - want;
    if (want_start < ring_start or pos > hist) return .{ .ctx = null };
    const phys_from = want_start - ring_start;
    const phys_to = pos - ring_start;
    var view = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&view, aux, &[_]c_int{ 0, phys_from, 0 }, 3, &[_]c_int{ sh[0], phys_to, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s)) catch |err| {
        _ = mlx.mlx_array_free(view);
        return err;
    };
    defer _ = mlx.mlx_array_free(view);
    return materializedOwnedCopy(s, view);
}

/// Free and null a live entry's whole QSA state: the history, its capacity buffer and the pooled bank move together.
pub fn ssmFreeQsaState(e: *SSMCacheEntry) void {
    if (e.aux_state.ctx != null) _ = mlx.mlx_array_free(e.aux_state);
    if (e.qsa_key_buf.ctx != null) _ = mlx.mlx_array_free(e.qsa_key_buf);
    if (e.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(e.qsa_pooled);
    if (e.qsa_pooled_buf.ctx != null) _ = mlx.mlx_array_free(e.qsa_pooled_buf);
    if (e.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(e.qsa_score_bank);
    if (e.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(e.qsa_score_buf);
    e.aux_state = .{ .ctx = null };
    e.qsa_key_buf = .{ .ctx = null };
    e.qsa_key_rows = 0;
    e.qsa_hist_rows = 0;
    e.qsa_pooled = .{ .ctx = null };
    e.qsa_pooled_buf = .{ .ctx = null };
    e.qsa_pooled_blocks = 0;
    e.qsa_score_bank = .{ .ctx = null };
    e.qsa_score_buf = .{ .ctx = null };
    e.qsa_score_blocks = 0;
}

/// Where the next `add` rows of QSA raw-key history go, and how big the capacity buffer must
/// be. `cap == 0` = no buffer yet. Growth is the KV cache's proportional policy.
const QsaKeyAppendPlan = struct { new_cap: usize, write_at: usize, new_rows: usize };
fn qsaKeyAppendPlan(rows: usize, cap: usize, add: usize, reserve: usize) QsaKeyAppendPlan {
    const need = rows + add;
    // A reservation raises the first grow to the whole request; past it the proportional policy takes over.
    const grown = if (need <= cap) cap else @max(KVCache.nextCapacity(cap, need), reserve);
    return .{
        .new_cap = grown,
        .write_at = rows,
        .new_rows = need,
    };
}

/// How many times a capacity buffer actually allocated.
pub var qsa_cap_buf_allocs: usize = 0;

/// How many times a QSA history accelerator was re-seeded from its authority (`seedCapBuf`).
pub var qsa_history_seeds: usize = 0;

pub var qsa_score_incremental_appends: usize = 0;

/// Ask every live entry's QSA history to hold `tokens` rows up front (same lever as `KVCache.reserve`).
pub fn reserveQsaHistory(entries: ?[]SSMCacheEntry, tokens: usize) void {
    const es = entries orelse return;
    for (es) |*e| {
        if (tokens > e.qsa_reserve_rows) e.qsa_reserve_rows = tokens;
    }
}

pub fn reserveQsaHistoryWithHead(entries: ?[]SSMCacheEntry, head: ?*SSMCacheEntry, tokens: usize) void {
    reserveQsaHistory(entries, tokens);
    if (head) |h| reserveQsaHistory(h[0..1], tokens);
}

/// Append `chunk` into a pre-grown capacity buffer along `axis` and republish `view` as the
/// tight `count`-long prefix. The published view is dropped first: while it lives mlx copies
/// the buffer instead of donating it.
/// Materialize `src` straight into a capacity buffer already `reserve` rows long: the seed
/// half of `capBufAppend`, done once. A tight seed made the next append grow (two full passes
/// over the prefix, ~81 ms at 393k). Byte-identical to the pair it replaces.
pub fn seedCapBuf(s: mlx.mlx_stream, src: mlx.mlx_array, axis: usize, reserve: usize) !mlx.mlx_array {
    const sh = mlx.getShape(src);
    const nd: usize = sh.len;
    std.debug.assert(nd <= 4 and axis < nd);
    const held: usize = @intCast(sh[axis]);
    // No reservation (or one the history exceeds): the tight copy is the right buffer.
    if (reserve <= held) return materializedOwnedCopy(s, src);
    var shape = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    for (sh, 0..) |d, i| {
        shape[i] = d;
        stop[i] = d;
    }
    shape[axis] = @intCast(reserve);
    const start = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var grown = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grown);
    try mlx.check(mlx.mlx_zeros(&grown, &shape, nd, mlx.mlx_array_dtype(src), s));
    var seeded = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(seeded);
    try mlx.check(mlx.mlx_slice_update(&seeded, grown, src, &start, nd, &stop, nd, &strides, nd, s));
    // Not counted in `qsa_cap_buf_allocs`: that counter means "capBufAppend had to grow".
    qsa_history_seeds += 1;
    return seeded;
}

pub fn capBufAppend(s: mlx.mlx_stream, buf: *mlx.mlx_array, view: *mlx.mlx_array, count: *c_int, chunk: mlx.mlx_array, axis: usize, reserve: usize) !void {
    const csh = mlx.getShape(chunk);
    const nd: usize = csh.len;
    std.debug.assert(nd <= 4 and axis < nd);
    const add = csh[axis];
    if (view.ctx != null) _ = mlx.mlx_array_free(view.*);
    view.* = .{ .ctx = null };

    const cap: usize = if (buf.ctx != null) @intCast(mlx.getShape(buf.*)[axis]) else 0;
    const plan = qsaKeyAppendPlan(@intCast(count.*), cap, @intCast(add), reserve);
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    for (csh, 0..) |d, i| stop[i] = d;

    if (plan.new_cap != cap) {
        qsa_cap_buf_allocs += 1;
        var shape = [_]c_int{ 0, 0, 0, 0 };
        for (csh, 0..) |d, i| shape[i] = d;
        shape[axis] = @intCast(plan.new_cap);
        var grown = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(grown);
        try mlx.check(mlx.mlx_zeros(&grown, &shape, nd, mlx.mlx_array_dtype(chunk), s));
        if (buf.ctx != null) {
            if (count.* > 0) {
                stop[axis] = count.*;
                var old = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(old);
                try mlx.check(mlx.mlx_slice(&old, buf.*, &start, nd, &stop, nd, &strides, nd, s));
                var seeded = mlx.mlx_array_new();
                // Safe unscoped: on failure `grown` is still the old handle, so the two errdefers free two different arrays.
                errdefer _ = mlx.mlx_array_free(seeded);
                try mlx.check(mlx.mlx_slice_update(&seeded, grown, old, &start, nd, &stop, nd, &strides, nd, s));
                _ = mlx.mlx_array_free(grown);
                grown = seeded;
            }
            _ = mlx.mlx_array_free(buf.*);
        }
        buf.* = grown;
    }
    start[axis] = count.*;
    stop[axis] = @intCast(plan.new_rows);
    // Block-scoped errdefer on purpose: `updated` becomes `buf.*` on the last line.
    {
        var updated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(updated);
        try mlx.check(mlx.mlx_slice_update(&updated, buf.*, chunk, &start, nd, &stop, nd, &strides, nd, s));
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }
    count.* = @intCast(plan.new_rows);

    start[axis] = 0;
    view.* = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(view.*);
        view.* = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(view, buf.*, &start, nd, &stop, nd, &strides, nd, s));
}

/// Re-publish an entry's QSA views over the first `keep` rows of the capacity buffers it
/// already owns, instead of copying the history out and re-seeding it. False (the caller
/// falls back to the copy) when there is no capacity buffer: a restore, a cold entry, a
/// post-adopt head.
pub fn qsaResliceToKeep(entry: *SSMCacheEntry, keep: c_int, s: mlx.mlx_stream) !bool {
    const hist = qsaHistoryRows(entry);
    if (keep > hist) return false;
    const drop_tail = hist - keep;
    const held = entry.qsa_key_rows;
    if (drop_tail > held) return false;
    const new_held = held - drop_tail;
    // The retained ring must still hold `keep`'s own leftover for the next append to re-pool
    // it; short of that the clamp belongs to the caller's mark path, or to its named decline.
    if (new_held < @mod(keep, @max(entry.qsa_ratio, 1))) return false;
    if (new_held > 0) {
        const buf_ok = entry.qsa_key_buf.ctx != null and
            mlx.getShape(entry.qsa_key_buf)[1] >= new_held;
        if (!buf_ok) return false;
        try capBufReslice(s, entry.qsa_key_buf, &entry.aux_state, &entry.qsa_key_rows, 1, new_held);
    } else {
        if (entry.aux_state.ctx != null) _ = mlx.mlx_array_free(entry.aux_state);
        entry.aux_state = .{ .ctx = null };
        entry.qsa_key_rows = 0;
    }
    entry.qsa_hist_rows = keep;
    const keep_blocks: c_int = @divTrunc(keep, @max(entry.qsa_ratio, 1));
    if (entry.qsa_pooled_buf.ctx != null) {
        if (keep_blocks > 0 and mlx.getShape(entry.qsa_pooled_buf)[1] >= keep_blocks) {
            try capBufReslice(s, entry.qsa_pooled_buf, &entry.qsa_pooled, &entry.qsa_pooled_blocks, 1, keep_blocks);
        } else {
            if (entry.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(entry.qsa_pooled);
            entry.qsa_pooled = .{ .ctx = null };
            entry.qsa_pooled_blocks = 0;
        }
    } else {
        try truncatePooled(&entry.qsa_pooled, keep, entry.qsa_ratio, s, false);
    }
    if (entry.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_bank);
    entry.qsa_score_bank = .{ .ctx = null };
    if (entry.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_buf);
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    return true;
}

fn capBufReslice(s: mlx.mlx_stream, buf: mlx.mlx_array, view: *mlx.mlx_array, count: *c_int, axis: usize, new_count: c_int) !void {
    const sh = mlx.getShape(buf);
    const nd: usize = sh.len;
    if (view.ctx != null) _ = mlx.mlx_array_free(view.*);
    view.* = .{ .ctx = null };
    count.* = 0;
    if (new_count <= 0) return;
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    for (sh, 0..) |d, i| stop[i] = d;
    stop[axis] = new_count;
    view.* = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(view.*);
        view.* = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(view, buf, &start, nd, &stop, nd, &strides, nd, s));
    count.* = new_count;
}

pub fn qsaTestAppendPool(xfm: *Trunk, entry: *SSMCacheEntry, chunk: mlx.mlx_array, offset: c_int, s: mlx.mlx_stream) !void {
    const ratio = @max(entry.qsa_ratio, 1);
    const add = mlx.getShape(chunk)[1];
    const kv = offset + add;
    const nb = @divTrunc(kv, ratio);
    const nb_cached: c_int = if (entry.qsa_pooled.ctx != null) mlx.getShape(entry.qsa_pooled)[1] else 0;
    if (nb > nb_cached) {
        const leftover_n = offset - nb_cached * ratio;
        const n_new = nb - nb_cached;
        const need = n_new * ratio;
        const from_k = need - leftover_n;
        var kb_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb_flat);
        if (leftover_n == 0) {
            try mlx.check(mlx.mlx_slice(&kb_flat, chunk, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, need, mlx.getShape(chunk)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        } else {
            const ash = mlx.getShape(entry.aux_state);
            var leftover = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(leftover);
            try mlx.check(mlx.mlx_slice(&leftover, entry.aux_state, &[_]c_int{ 0, ash[1] - leftover_n, 0 }, 3, &[_]c_int{ ash[0], ash[1], ash[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var k_part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_part);
            try mlx.check(mlx.mlx_slice(&k_part, chunk, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, from_k, mlx.getShape(chunk)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const parts = [_]mlx.mlx_array{ leftover, k_part };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&kb_flat, vec, 1, s));
        }
        const hd = mlx.getShape(chunk)[2];
        var kb4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb4);
        try mlx.check(mlx.mlx_reshape(&kb4, kb_flat, &[_]c_int{ 1, n_new, ratio, hd }, 4, s));
        var f32b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32b);
        try mlx.check(mlx.mlx_astype(&f32b, kb4, .float32, s));
        var mean = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean);
        try mlx.check(mlx.mlx_mean_axis(&mean, f32b, 2, false, s));
        var new3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(new3);
        try mlx.check(mlx.mlx_astype(&new3, mean, .bfloat16, s));
        try xfm.qsaAppendPooled(entry, new3, nb_cached);
    }
    try xfm.qsaAppendKeys(entry, chunk, offset);
}

/// Keep the pooled-key rows that are still complete blocks of a key history
/// truncated to `keep_rows` (the block size is implied by the two shapes).
/// Cut the pooled bank to the blocks `keep_rows` completes. `materialize` = a fresh buffer; false = a slice view.
pub fn truncatePooled(pooled: *mlx.mlx_array, keep_rows: c_int, ratio: c_int, s: mlx.mlx_stream, materialize: bool) !void {
    if (pooled.ctx == null) return;
    const ps = mlx.getShape(pooled.*);
    const keep_blocks: c_int = @divTrunc(keep_rows, ratio);
    if (keep_blocks >= ps[1]) return;
    if (keep_blocks <= 0) {
        _ = mlx.mlx_array_free(pooled.*);
        pooled.* = .{ .ctx = null };
        return;
    }
    const start = [_]c_int{ 0, 0, 0 };
    const stop = [_]c_int{ ps[0], keep_blocks, ps[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var view = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&view, pooled.*, &start, 3, &stop, 3, &strides, 3, s)) catch |err| {
        _ = mlx.mlx_array_free(view);
        return err;
    };
    const owned = if (materialize) blk: {
        defer _ = mlx.mlx_array_free(view);
        break :blk try materializedOwnedCopy(s, view);
    } else view;
    _ = mlx.mlx_array_free(pooled.*);
    pooled.* = owned;
}

// ── tests ──

test "checkQsaCompressRatio: a ratio the ring cannot hold is refused by name" {
    try std.testing.expectError(error.UnsupportedQsaCompressRatio, checkQsaCompressRatio(0));
    try std.testing.expectError(error.UnsupportedQsaCompressRatio, checkQsaCompressRatio(@intCast(QSA_MAX_COMPRESS_RATIO + 1)));
    try checkQsaCompressRatio(@intCast(QSA_MAX_COMPRESS_RATIO));
}

fn qsaRingTail(src: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(src);
    const n: c_int = @min(sh[1], QSA_RING_ROWS);
    if (n == sh[1]) return materializedOwnedCopy(s, src);
    var view = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&view, src, &[_]c_int{ 0, sh[1] - n, 0 }, 3, &[_]c_int{ sh[0], sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s)) catch |err| {
        _ = mlx.mlx_array_free(view);
        return err;
    };
    defer _ = mlx.mlx_array_free(view);
    return materializedOwnedCopy(s, view);
}

test "QSA key history advances by S not 1" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Trunk = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);
    entry.qsa_ratio = 4;
    var prng = std.Random.DefaultPrng.init(0x51A0_0004);
    const chunk = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 4, 16 }, s);
    defer _ = mlx.mlx_array_free(chunk);
    try xfm.qsaAppendKeys(&entry, chunk, 0);
    try t.expectEqual(@as(c_int, 4), qsaHistoryRows(&entry));
}

test "qsa raw-key ring is O(RING_ROWS) at N rows" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Trunk = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);
    entry.qsa_ratio = 4;

    var prng = std.Random.DefaultPrng.init(0x51A0_B17E);
    const rnd = prng.random();
    const n: c_int = 256;
    const hd: c_int = 16;
    try t.expect(n > QSA_RING_ROWS);

    const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, n, hd }, s);
    defer _ = mlx.mlx_array_free(chunk);
    try xfm.qsaAppendKeys(&entry, chunk, 0);
    try mlx.check(mlx.mlx_array_eval(entry.qsa_key_buf));

    const raw = qsaRawKeyBytes(&entry);
    const row: u64 = @as(u64, @intCast(hd)) * @as(u64, mlx.mlx_array_itemsize(entry.qsa_key_buf));
    const ring_cap = @as(u64, @intCast(QSA_RING_ROWS)) * row;
    try t.expectEqual(ring_cap, raw);
}

pub fn qsaTestMeanBlocks(keys: mlx.mlx_array, row0: c_int, n_blocks: c_int, ratio: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(keys);
    const n_rows = n_blocks * ratio;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_slice(&flat, keys, &[_]c_int{ 0, row0, 0 }, 3, &[_]c_int{ sh[0], row0 + n_rows, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var kb4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kb4);
    try mlx.check(mlx.mlx_reshape(&kb4, flat, &[_]c_int{ sh[0], n_blocks, ratio, sh[2] }, 4, s));
    var f32b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32b);
    try mlx.check(mlx.mlx_astype(&f32b, kb4, .float32, s));
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, f32b, 2, false, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, mean, .bfloat16, s));
    return out;
}

test "qsa leftover: restore at L-30 then append/rollback re-pools bit-identically to a full bank" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Trunk = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    var prng = std.Random.DefaultPrng.init(0x1E40_7E11);
    const rnd = prng.random();
    const hd: c_int = 16;
    const ratio: c_int = 4;
    const L: c_int = 37;
    const pos: c_int = L - 30;
    try t.expect(pos > 0 and @mod(pos, ratio) != 0);

    const keys = try attn256RandBf16(rnd, &[_]c_int{ 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(keys);

    var live_arr = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio }};
    defer ssmFreeQsaState(&live_arr[0]);
    {
        var first = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(first);
        try mlx.check(mlx.mlx_slice(&first, keys, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, pos, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &live_arr[0], first, 0, s);
    }
    var cps: [2]SSMCheckpoint = undefined;
    cps[0] = try captureSsmCheckpoint(t.allocator, &live_arr, @intCast(pos), s);
    {
        var rest = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rest);
        try mlx.check(mlx.mlx_slice(&rest, keys, &[_]c_int{ 0, pos, 0 }, 3, &[_]c_int{ 1, L, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &live_arr[0], rest, pos, s);
    }
    cps[1] = try captureSsmCheckpoint(t.allocator, &live_arr, @intCast(L), s);
    defer for (&cps) |*c| c.deinit(t.allocator);
    try attachQsaHistoryToLatest(&cps, &live_arr, s);

    var dest_arr = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true }};
    defer ssmFreeQsaState(&dest_arr[0]);
    try restoreSsmCheckpoint(&dest_arr, &cps[0]);
    try applyQsaHistoryAt(&dest_arr, &cps[1], @intCast(pos), s, false);

    // The restore carries the raw rows of the block `pos` reopened, and nothing else.
    const tail: c_int = @mod(pos, ratio);
    try t.expectEqual(tail, mlx.getShape(dest_arr[0].aux_state)[1]);
    {
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_slice(&want, keys, &[_]c_int{ 0, pos - tail, 0 }, 3, &[_]c_int{ 1, pos, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dest_arr[0].aux_state, want, s));
    }

    const verify_len: c_int = 4;
    {
        var verify = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify);
        try mlx.check(mlx.mlx_slice(&verify, keys, &[_]c_int{ 0, pos, 0 }, 3, &[_]c_int{ 1, pos + verify_len, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &dest_arr[0], verify, pos, s);
    }
    // The block the verify closed was pooled from leftover + chunk: it must already be the
    // full bank's block, before the rollback gets a chance to drop it.
    {
        const closed = @divTrunc(pos + verify_len, ratio);
        try t.expect(closed > @divTrunc(pos, ratio));
        const ref_closed = try qsaTestMeanBlocks(keys, 0, closed, ratio, s);
        defer _ = mlx.mlx_array_free(ref_closed);
        try t.expectEqual(closed, dest_arr[0].qsa_pooled_blocks);
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dest_arr[0].qsa_pooled, ref_closed, s));
    }
    try ssmRollbackFromCapture(&dest_arr[0], 1, @intCast(verify_len), s);
    const keep: c_int = pos + 1 + 1;
    try t.expectEqual(keep, dest_arr[0].qsa_hist_rows);
    {
        var again = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(again);
        try mlx.check(mlx.mlx_slice(&again, keys, &[_]c_int{ 0, keep, 0 }, 3, &[_]c_int{ 1, keep + 3, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &dest_arr[0], again, keep, s);
    }
    const final_len = keep + 3;
    const n_blocks = @divTrunc(final_len, ratio);
    const ref = try qsaTestMeanBlocks(keys, 0, n_blocks, ratio, s);
    defer _ = mlx.mlx_array_free(ref);
    try t.expectEqual(n_blocks, dest_arr[0].qsa_pooled_blocks);
    try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dest_arr[0].qsa_pooled, ref, s));
}

test "qsa rollback: a partial accept keeps the raw-key buffer and re-slices to keep" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Trunk = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;
    xfm.qsa_score_bank_builds = 0;

    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);
    entry.qsa_ratio = 4;
    entry.qsa_reserve_rows = 64;

    var prng = std.Random.DefaultPrng.init(0x51A0_B0FF);
    const rnd = prng.random();
    const hd: c_int = 16;
    const prefix: c_int = 8;
    const verify_len: c_int = 4;
    const accepted: u32 = 1;
    const keep: c_int = prefix + @as(c_int, @intCast(1 + accepted));
    const keep_blocks: c_int = @divTrunc(keep, entry.qsa_ratio);

    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, prefix, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try xfm.qsaAppendKeys(&entry, chunk, 0);
        const blocks = try attn256RandBf16(rnd, &[_]c_int{ 1, @divTrunc(prefix, entry.qsa_ratio), hd }, s);
        defer _ = mlx.mlx_array_free(blocks);
        try xfm.qsaAppendPooled(&entry, blocks, 0);
    }
    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, verify_len, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try xfm.qsaAppendKeys(&entry, chunk, prefix);
        const blocks = try attn256RandBf16(rnd, &[_]c_int{ 1, @divTrunc(verify_len, entry.qsa_ratio), hd }, s);
        defer _ = mlx.mlx_array_free(blocks);
        try xfm.qsaAppendPooled(&entry, blocks, @divTrunc(prefix, entry.qsa_ratio));
    }
    try t.expectEqual(prefix + verify_len, entry.qsa_key_rows);
    try t.expect(entry.qsa_key_buf.ctx != null);
    _ = try xfm.qsaScoreBank(&entry, 1, @divTrunc(prefix + verify_len, entry.qsa_ratio), hd);

    const kept_ref = blk: {
        const ks = mlx.getShape(entry.aux_state);
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(
            &view,
            entry.aux_state,
            &[_]c_int{ 0, 0, 0 },
            3,
            &[_]c_int{ ks[0], keep, ks[2] },
            3,
            &[_]c_int{ 1, 1, 1 },
            3,
            s,
        ));
        break :blk try materializedOwnedCopy(s, view);
    };
    defer _ = mlx.mlx_array_free(kept_ref);

    const kept_pooled = blk: {
        const ps = mlx.getShape(entry.qsa_pooled);
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(
            &view,
            entry.qsa_pooled,
            &[_]c_int{ 0, 0, 0 },
            3,
            &[_]c_int{ ps[0], keep_blocks, ps[2] },
            3,
            &[_]c_int{ 1, 1, 1 },
            3,
            s,
        ));
        break :blk try materializedOwnedCopy(s, view);
    };
    defer _ = mlx.mlx_array_free(kept_pooled);

    const buf_before = entry.qsa_key_buf.ctx;
    const pooled_buf_before = entry.qsa_pooled_buf.ctx;
    try ssmRollbackFromCapture(&entry, accepted, @intCast(verify_len), s);

    try t.expect(entry.qsa_key_buf.ctx == buf_before);
    try t.expectEqual(keep, entry.qsa_key_rows);
    try t.expectEqual(keep, mlx.getShape(entry.aux_state)[1]);
    try t.expect(entry.qsa_pooled_buf.ctx == pooled_buf_before);
    try t.expectEqual(keep_blocks, entry.qsa_pooled_blocks);
    try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.qsa_pooled, kept_pooled, s));
    // The score bank is invalidated rather than truncated: its columns are a function of
    // the pooled bank and the next completed block rebuilds them.
    try t.expect(entry.qsa_score_bank.ctx == null);
    try t.expectEqual(@as(c_int, 0), entry.qsa_score_blocks);

    const tail = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, hd }, s);
    defer _ = mlx.mlx_array_free(tail);
    var reference = mlx.mlx_array_new();
    {
        const parts = [_]mlx.mlx_array{ kept_ref, tail };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&reference, vec, 1, s));
    }
    defer _ = mlx.mlx_array_free(reference);

    try xfm.qsaAppendKeys(&entry, tail, keep);
    try t.expectEqual(keep + 2, mlx.getShape(entry.aux_state)[1]);
    try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.aux_state, reference, s));
}

test "qsa rollback: a rollback that reaches past the ring is a named error, not a negative slice" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    const hd: c_int = 16;
    const ratio: c_int = 4;

    // A restored entry holds only the leftover at its position — no capacity buffer, so the
    // reslice declines and the copy path is what a too-wide verify reaches.
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
    defer ssmFreeQsaState(&entry);
    entry.aux_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&entry.aux_state, &[_]c_int{ 1, 2, hd }, 3, .bfloat16, s));
    entry.qsa_pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&entry.qsa_pooled, &[_]c_int{ 1, 11, hd }, 3, .bfloat16, s));
    entry.qsa_key_rows = 2;
    entry.qsa_pooled_blocks = 11;
    entry.qsa_hist_rows = 46;

    try t.expectError(error.QsaRollbackBeyondRing, ssmRollbackFromCapture(&entry, 0, 16, s));
}

test "qsa key append plan: rows advance by S, capacity grows only when it must" {
    const p0 = qsaKeyAppendPlan(0, 0, 4, 0);
    try testing.expectEqual(@as(usize, 0), p0.write_at);
    try testing.expectEqual(@as(usize, 4), p0.new_rows);
    try testing.expect(p0.new_cap >= 4);

    const p1 = qsaKeyAppendPlan(100, 512, 1, 0);
    try testing.expectEqual(@as(usize, 512), p1.new_cap);
    try testing.expectEqual(@as(usize, 100), p1.write_at);
    try testing.expectEqual(@as(usize, 101), p1.new_rows);

    const p2 = qsaKeyAppendPlan(511, 512, 1, 0);
    try testing.expectEqual(@as(usize, 512), p2.new_cap);
    try testing.expectEqual(@as(usize, 512), p2.new_rows);

    const p3 = qsaKeyAppendPlan(512, 512, 1, 0);
    try testing.expect(p3.new_cap >= 513);
    try testing.expectEqual(@as(usize, 512), p3.write_at);
    try testing.expectEqual(@as(usize, 513), p3.new_rows);

    const p4 = qsaKeyAppendPlan(500, 512, 4096, 0);
    try testing.expect(p4.new_cap >= 4596);
    try testing.expectEqual(@as(usize, 4596), p4.new_rows);
}

test "qsa history reservation: a reserved prefill allocates its buffers ONCE (#353 follow-up)" {
    // These buffers walked the +25% ladder beside a KV cache that had been sized once; the loop
    // below allocated ~30 times before `qsa_reserve_rows`.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5E5E_4E5E);
    const rnd = prng.random();
    const hd: c_int = 128;
    const chunk_rows: c_int = 4096;
    const total: usize = 409_600;

    var t: Trunk = undefined;

    t.s = s;
    t.allocator = ta;
    var entries = [_]SSMCacheEntry{.{ .ssm_state = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new(), .initialized = false }};
    const entry = &entries[0];
    defer {
        ssmFreeQsaState(entry);
        _ = mlx.mlx_array_free(entry.ssm_state);
        _ = mlx.mlx_array_free(entry.conv_state);
    }

    reserveQsaHistory(entries[0..], total);
    try testing.expectEqual(total, entry.qsa_reserve_rows);

    var rows: c_int = 0;
    while (@as(usize, @intCast(rows)) < total) {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, chunk_rows, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try t.qsaAppendKeys(entry, chunk, rows);
        rows += chunk_rows;
    }
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(entry.aux_state)[1]);
    try testing.expect(mlx.getShape(entry.qsa_key_buf)[1] <= QSA_RING_ROWS);
    try testing.expectEqual(@as(c_int, @intCast(total)), entry.qsa_hist_rows);
}

test "qsa key append plan: a reservation raises the FIRST grow and nothing else" {
    try testing.expectEqual(@as(usize, 512), qsaKeyAppendPlan(100, 512, 1, 400_000).new_cap);
    try testing.expectEqual(@as(usize, 400_000), qsaKeyAppendPlan(0, 0, 4096, 400_000).new_cap);
    try testing.expect(qsaKeyAppendPlan(0, 0, 500_000, 400_000).new_cap >= 500_000);
    try testing.expect(qsaKeyAppendPlan(400_000, 400_000, 1, 400_000).new_cap > 400_000);
}

test "qsa key history: capacity-buffer append equals the concatenated reference, across a growth boundary" {
    // The published `aux_state` must stay byte-for-byte the old `concatenate`, including across
    // reallocations and after a restore re-seeds the accelerator.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();

    var t: Trunk = undefined;

    t.s = s;
    t.allocator = testing.allocator;

    const hd: c_int = 16;
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);

    var prng = std.Random.DefaultPrng.init(0xB0FF_2C11);
    const rnd = prng.random();
    var reference = mlx.mlx_array{ .ctx = null };
    defer if (reference.ctx != null) {
        _ = mlx.mlx_array_free(reference);
    };
    var rows: c_int = 0;
    var step_i: usize = 0;
    while (step_i < 601) : (step_i += 1) {
        const add: c_int = if (step_i == 0) 7 else 1;
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, add, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        var next_ref = mlx.mlx_array_new();
        if (reference.ctx != null) {
            const parts = [_]mlx.mlx_array{ reference, chunk };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&next_ref, vec, 1, s));
            _ = mlx.mlx_array_free(reference);
        } else {
            _ = mlx.mlx_array_free(next_ref);
            next_ref = try materializedOwnedCopy(s, chunk);
        }
        reference = next_ref;

        try t.qsaAppendKeys(&entry, chunk, rows);
        rows += add;
        const tail = try qsaRingTail(reference, s);
        defer _ = mlx.mlx_array_free(tail);
        try testing.expectEqual(mlx.getShape(tail)[1], mlx.getShape(entry.aux_state)[1]);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.aux_state, tail, s));
        try testing.expect(mlx.getShape(entry.qsa_key_buf)[1] >= mlx.getShape(entry.aux_state)[1]);
    }
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(entry.aux_state)[1]);

    // A restore replaces the history behind the accelerator's back; the append re-seeds from `aux_state`.
    const restored = try attn256RandBf16(rnd, &[_]c_int{ 1, rows, hd }, s);
    ssmFreeQsaState(&entry);
    entry.aux_state = restored;
    _ = mlx.mlx_array_free(reference);
    reference = try materializedOwnedCopy(s, restored);

    const tail = try attn256RandBf16(rnd, &[_]c_int{ 1, 3, hd }, s);
    defer _ = mlx.mlx_array_free(tail);
    {
        const parts = [_]mlx.mlx_array{ reference, tail };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
        _ = mlx.mlx_array_free(reference);
        reference = cat;
    }
    try t.qsaAppendKeys(&entry, tail, rows);
    {
        const ring = try qsaRingTail(reference, s);
        defer _ = mlx.mlx_array_free(ring);
        try testing.expectEqual(mlx.getShape(ring)[1], mlx.getShape(entry.aux_state)[1]);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.aux_state, ring, s));
    }
    rows += 3;

    // A site that replaced `aux_state` without dropping the buffer still cannot poison the history.
    const shorter = try attn256RandBf16(rnd, &[_]c_int{ 1, 12, hd }, s);
    const stale_view = entry.aux_state;
    entry.aux_state = shorter; // qsa_key_buf deliberately left in place
    _ = mlx.mlx_array_free(stale_view);
    _ = mlx.mlx_array_free(reference);
    reference = try materializedOwnedCopy(s, shorter);
    const tail2 = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, hd }, s);
    defer _ = mlx.mlx_array_free(tail2);
    {
        const parts = [_]mlx.mlx_array{ reference, tail2 };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
        _ = mlx.mlx_array_free(reference);
        reference = cat;
    }
    try t.qsaAppendKeys(&entry, tail2, 12);
    try testing.expectEqual(@as(c_int, 14), mlx.getShape(entry.aux_state)[1]);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.aux_state, reference, s));
}

test "qsa history: a restore-shaped re-seed lands AT the reservation, so the append never grows" {
    // A restore republishes `aux_state` but leaves the append accelerator empty, and the
    // re-seed used to produce a tight buffer that grew one line later (~81 ms at 384k).
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED_5EED);
    const rnd = prng.random();
    const hd: c_int = 128;
    const held: c_int = 4096;
    const reserve: usize = 8192;

    var t: Trunk = undefined;

    t.s = s;
    t.allocator = ta;
    var entries = [_]SSMCacheEntry{.{ .ssm_state = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new(), .initialized = false }};
    const entry = &entries[0];
    defer {
        ssmFreeQsaState(entry);
        _ = mlx.mlx_array_free(entry.ssm_state);
        _ = mlx.mlx_array_free(entry.conv_state);
    }
    reserveQsaHistory(entries[0..], reserve);

    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, held, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try t.qsaAppendKeys(entry, chunk, 0);
    }
    // A restore: `aux_state` survives, the accelerator does not.
    var authority = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&authority, entry.aux_state));
    _ = mlx.mlx_array_free(entry.qsa_key_buf);
    entry.qsa_key_buf = .{ .ctx = null };
    entry.qsa_key_rows = 0;
    _ = mlx.mlx_array_free(entry.aux_state);
    entry.aux_state = authority;

    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, 30, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try t.qsaAppendKeys(entry, chunk, held);
    }
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(entry.aux_state)[1]);
    try testing.expectEqual(@as(c_int, held + 30), entry.qsa_hist_rows);

    entry.qsa_reserve_rows = 0;
    _ = mlx.mlx_array_free(entry.qsa_key_buf);
    entry.qsa_key_buf = .{ .ctx = null };
    entry.qsa_key_rows = 0;
    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, 30, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try t.qsaAppendKeys(entry, chunk, held + 30);
    }
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(entry.aux_state)[1]);
    try testing.expectEqual(@as(c_int, held + 60), entry.qsa_hist_rows);
}
