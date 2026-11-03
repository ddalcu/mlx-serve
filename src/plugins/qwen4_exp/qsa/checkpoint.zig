//! QSA indexer state inside SSM checkpoints and snapshots: whether an entry's aux holds the key history, how many
//! rows it covers, and the capture, trim and restore of the pooled blocks.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const SSMCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").SSMCheckpoint;
const captureSsmCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").captureSsmCheckpoint;
const qsaHistoryBytesHeld = @import("../../../engines/mlx/ssm/checkpoint.zig").qsaHistoryBytesHeld;
const restoreSsmCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").restoreSsmCheckpoint;
const ssmCheckpointBytes = @import("../../../engines/mlx/ssm/checkpoint.zig").ssmCheckpointBytes;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const SSMCacheEntrySnapshot = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntrySnapshot;
const Trunk = @import("../trunk.zig").Trunk;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const qsaHistoryShareEnabled = @import("gather.zig").qsaHistoryShareEnabled;
const QSA_RING_ROWS = @import("history.zig").QSA_RING_ROWS;
const qsaLeftoverAt = @import("history.zig").qsaLeftoverAt;
const ssmFreeQsaState = @import("history.zig").ssmFreeQsaState;
const truncatePooled = @import("history.zig").truncatePooled;

/// QSA indexer key history lives on full-attention layers: `aux_state` is
/// `[B, kv, idx_hd]` and there is no conv recurrence (`conv_state` null or
/// an empty handle). PLE's dilated-conv window is `aux_state` on a GDN
/// layer that HAS a real `conv_state` — that window is O(1) and stays in
/// every checkpoint.
pub fn ssmAuxIsQsaHistory(e: *const SSMCacheEntry) bool {
    if (e.qsa_pooled.ctx != null) return true;
    if (e.aux_state.ctx == null) return false;
    if (e.conv_state.ctx == null) return true;
    return mlx.mlx_array_size(e.conv_state) == 0;
}

pub fn qsaHistoryRows(e: *const SSMCacheEntry) c_int {
    if (e.qsa_hist_rows > 0) return e.qsa_hist_rows;
    if (!ssmAuxIsQsaHistory(e)) return 0;
    if (e.aux_state.ctx == null) return 0;
    const sh = mlx.getShape(e.aux_state);
    return if (sh.len >= 2) sh[1] else 0;
}

/// `ssmAuxIsQsaHistory` over a snapshot layer.
pub fn snapshotHasQsaHistory(l: *const SSMCacheEntrySnapshot) bool {
    return ssmAuxIsQsaHistory(&.{
        .conv_state = l.conv_state,
        .ssm_state = l.ssm_state,
        .aux_state = l.aux_state,
        .qsa_pooled = l.qsa_pooled,
        .initialized = l.initialized,
    });
}

pub fn checkpointHasQsaHistory(cp: *const SSMCheckpoint) bool {
    for (cp.layers) |*l| {
        if (snapshotHasQsaHistory(l)) return true;
    }
    return false;
}

/// Whether the pooled indexer bank — the history proper — lives on THIS checkpoint. Every
/// checkpoint carries its own raw leftover, so `checkpointHasQsaHistory` cannot answer "is this
/// the bank's home": a mid-block snap says yes on the leftover alone.
pub fn checkpointHasQsaPooled(cp: *const SSMCheckpoint) bool {
    for (cp.layers) |*l| {
        if (l.qsa_pooled.ctx != null) return true;
    }
    return false;
}

pub fn checkpointQsaAuxRows(cp: *const SSMCheckpoint) c_int {
    var r: c_int = 0;
    for (cp.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        if (l.qsa_rows > r) r = l.qsa_rows;
    }
    return r;
}

/// Raw leftover rows a checkpoint carries (0 on a non-QSA layer, and on a QSA layer whose
/// position is block-aligned).
fn checkpointQsaLeftoverRows(cp: *const SSMCheckpoint) c_int {
    var rows: c_int = 0;
    for (cp.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        if (l.aux_state.ctx == null) continue;
        const sh = mlx.getShape(l.aux_state);
        if (sh.len >= 2 and sh[1] > rows) rows = sh[1];
    }
    return rows;
}

/// Whether a QSA restore at `pos` has its whole indexer history: `cp` — the checkpoint being
/// restored — carries the raw leftover for `pos` (the `pos % ratio` rows the next append
/// re-pools), and `src` a pooled bank reaching `pos`. A ring sliced at another position is a
/// miss, never a silently wrong leftover.
pub fn checkpointQsaCoversPos(cp: *const SSMCheckpoint, src: *const SSMCheckpoint, pos: usize) bool {
    const logical = checkpointQsaAuxRows(src);
    if (logical == 0) return true;
    const p: c_int = @intCast(pos);
    if (logical < p) return false;
    var has_pooled = false;
    var pblocks: c_int = 0;
    var ratio: c_int = 4;
    for (src.layers) |*l| {
        if (l.qsa_ratio > 0) ratio = l.qsa_ratio;
        if (l.qsa_pooled.ctx == null) continue;
        has_pooled = true;
        const ps = mlx.getShape(l.qsa_pooled);
        if (ps.len >= 2 and ps[1] > pblocks) pblocks = ps[1];
    }
    const r = @max(ratio, 1);
    if (has_pooled and pblocks < @divTrunc(p, r)) return false;
    if (!has_pooled) return qsaSourceHoldsRowsAt(src, p, p);
    const want_lv = @mod(p, r);
    const cp_rows = checkpointQsaLeftoverRows(cp);
    if (cp_rows == 0) return qsaSourceHoldsRowsAt(src, p, want_lv);
    if (cp_rows == want_lv) return true;
    return qsaSourceHoldsRowsAt(cp, p, want_lv) or qsaSourceHoldsRowsAt(src, p, want_lv);
}

pub fn qsaEntrySatisfiesForward(e: *const SSMCacheEntry, pos: c_int) bool {
    if (!ssmAuxIsQsaHistory(e)) return true;
    if (qsaHistoryRows(e) != pos) return false;
    const ratio = @max(e.qsa_ratio, 1);
    const blocks: c_int = if (e.qsa_pooled.ctx != null) mlx.getShape(e.qsa_pooled)[1] else 0;
    const held: c_int = if (e.aux_state.ctx != null) mlx.getShape(e.aux_state)[1] else 0;
    return blocks * ratio + held >= pos;
}

pub fn qsaRestoreSatisfiesForward(entries: []const SSMCacheEntry, pos: usize) bool {
    const p: c_int = @intCast(pos);
    for (entries) |*e| {
        if (!qsaEntrySatisfiesForward(e, p)) return false;
    }
    return true;
}

pub fn checkpointListHasQsaPooled(cps: []const SSMCheckpoint) bool {
    for (cps) |*cp| {
        if (checkpointHasQsaPooled(cp)) return true;
    }
    return false;
}

/// Whether every QSA layer of `src` physically holds the `want` raw rows ending at `pos`.
fn qsaSourceHoldsRowsAt(src: *const SSMCheckpoint, pos: c_int, want: c_int) bool {
    if (want <= 0) return true;
    var any = false;
    for (src.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        if (l.aux_state.ctx == null) return false;
        const sh = mlx.getShape(l.aux_state);
        if (sh.len < 2) return false;
        const hist: c_int = if (l.qsa_rows > 0) l.qsa_rows else sh[1];
        if (pos > hist or pos - want < hist - sh[1]) return false;
        any = true;
    }
    return any;
}

/// Copy QSA aux/pooled from the live entries onto the LATEST checkpoint —
/// ONE materialized copy, sliced to that checkpoint's position (a cancel
/// handoff attaches while the live history is ahead of the last stride
/// snap). Older snaps in `cps` drop any history they carried. No-op when
/// `cps` is empty or no layer holds QSA history.
///
/// The copy arm: the copy coexists with the live buffer for the whole decode.
pub fn attachQsaHistoryToLatest(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream) !void {
    return attachQsaHistoryToLatestMode(cps, live, s, .copy);
}

/// Rows a handed-off history may pin past the checkpoint's position before the handoff
/// materializes instead: an entry must never pin much more than `ssmCheckpointBytes` bills.
pub const QSA_HANDOFF_MAX_SLACK_ROWS: c_int = 16384;

/// The commit boundary: hand the slot's live QSA history to the latest checkpoint as a slice
/// view of the live capacity buffer, sliced to that checkpoint's position. The slot dies right
/// after, so the entry ends up the buffer's only owner. Too much slack materializes instead.
pub fn handoffQsaHistoryToLatest(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream) !void {
    return attachQsaHistoryToLatestMode(cps, live, s, .share);
}

/// Every "the slot is about to die" seam: share when the switch is on, copy otherwise.
pub fn attachQsaHistoryOnHandoff(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream) !void {
    return if (qsaHistoryShareEnabled())
        handoffQsaHistoryToLatest(cps, live, s)
    else
        attachQsaHistoryToLatest(cps, live, s);
}

pub const QsaAttachMode = enum { copy, share };

/// Does sharing `buf_rows` rows of capacity for a `keep`-row snap pin more slack than `QSA_HANDOFF_MAX_SLACK_ROWS`?
pub fn qsaHandoffExceedsSlack(buf_rows: c_int, keep: c_int) bool {
    return buf_rows - keep > QSA_HANDOFF_MAX_SLACK_ROWS;
}

/// The buffer a share of `src` would pin: the capacity buffer when the accelerator holds one, else the published history.
fn qsaHandoffMustMaterialize(src: *const SSMCacheEntry, keep: c_int) bool {
    const buf_rows: c_int = if (src.qsa_key_buf.ctx != null)
        mlx.getShape(src.qsa_key_buf)[1]
    else if (src.aux_state.ctx != null)
        mlx.getShape(src.aux_state)[1]
    else
        0;
    return qsaHandoffExceedsSlack(buf_rows, keep);
}

fn attachQsaHistoryToLatestMode(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream, mode: QsaAttachMode) !void {
    if (cps.len == 0) return;
    const cp = &cps[cps.len - 1];
    if (cp.layers.len != live.len) return error.SsmCheckpointLayerMismatch;
    const keep: c_int = @intCast(cp.pos);
    var copied: usize = 0;
    for (cp.layers, live) |*dst, src| {
        if (!ssmAuxIsQsaHistory(&src)) continue;
        const materialize = switch (mode) {
            .copy => true,
            .share => qsaHandoffMustMaterialize(&src, keep) or src.qsa_pooled_buf.ctx == null,
        };
        if (src.qsa_pooled.ctx != null) {
            if (dst.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(dst.qsa_pooled);
            if (materialize) {
                dst.qsa_pooled = try materializedOwnedCopy(s, src.qsa_pooled);
            } else {
                dst.qsa_pooled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&dst.qsa_pooled, src.qsa_pooled));
            }
            try truncatePooled(&dst.qsa_pooled, keep, src.qsa_ratio, s, materialize);
        }
        dst.qsa_ratio = src.qsa_ratio;
        copied += 1;
    }
    if (copied == 0) return;
    recordQsaRowsOnCheckpoints(cps, live);
    keepOnlyLatestQsaHistory(cps);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var n: usize = 0;
    for (cp.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        inline for (.{ l.aux_state, l.qsa_pooled }) |arr| {
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
                n += 1;
            }
        }
    }
    if (n > 0) _ = mlx.mlx_eval(vec);
}

fn recordQsaRowsOnCheckpoints(cps: []SSMCheckpoint, live: []const SSMCacheEntry) void {
    for (cps) |*cp| {
        if (cp.layers.len != live.len) continue;
        const pos_rows: c_int = @intCast(cp.pos);
        for (cp.layers, live) |*dst, src| {
            if (!ssmAuxIsQsaHistory(&src)) continue;
            if (dst.qsa_rows == 0) dst.qsa_rows = pos_rows;
            dst.qsa_ratio = src.qsa_ratio;
        }
    }
}

/// Keep the QSA history on the NEWEST snap that carries it and drop it from
/// every older one. A prefix-extend merge inherits the previous turn's latest
/// snap, so without this an agent session holds one full copy per turn.
pub fn keepOnlyLatestQsaHistory(cps: []SSMCheckpoint) void {
    var i = cps.len;
    var kept = false;
    while (i > 0) {
        i -= 1;
        if (!checkpointHasQsaPooled(&cps[i])) continue;
        if (!kept) {
            kept = true;
            continue;
        }
        for (cps[i].layers) |*l| {
            if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
            l.qsa_pooled = .{ .ctx = null };
        }
    }
}

/// Any live QSA layer holds indexer history. On a QSA arch a restore that
/// leaves this false cannot prefill (QsaHistoryGap) — the cache treats it
/// as a miss.
pub fn entriesHaveQsaHistory(entries: []const SSMCacheEntry) bool {
    for (entries) |*e| {
        if (ssmAuxIsQsaHistory(e)) return true;
    }
    return false;
}

/// Overlay QSA history from `src_cp` (the entry's latest snap, full length)
/// onto `entries` already restored to `pos`. Slices aux to `pos` rows and
/// pooled to `pos/ratio` blocks. No-op on PLE/GDN layers (they already
/// hold the per-position window from `restoreSsmCheckpoint`).
/// `materialize` false = the destination holds a slice of `src_cp`'s bank (safe when the
/// checkpoint outlives the slot's first append); true = an owned copy, required when the
/// source is a transient such as the SSD tier's overlay.
pub fn applyQsaHistoryAt(entries: []SSMCacheEntry, src_cp: *const SSMCheckpoint, pos: usize, s: mlx.mlx_stream, materialize: bool) !void {
    if (entries.len != src_cp.layers.len) return error.SsmCheckpointLayerMismatch;
    const keep: c_int = @intCast(pos);
    for (entries, src_cp.layers) |*dst, src| {
        if (!snapshotHasQsaHistory(&src)) continue;
        // After restoreSsmCheckpoint a QSA layer has no conv window.
        if (dst.conv_state.ctx != null and mlx.mlx_array_size(dst.conv_state) > 0) continue;
        const src_hist: c_int = if (src.qsa_rows > 0) src.qsa_rows else blk: {
            if (src.qsa_pooled.ctx == null) break :blk 0;
            break :blk keep;
        };
        if (src_hist < keep) return error.QsaHistoryGap;
        const own = dst.aux_state;
        dst.aux_state = .{ .ctx = null };
        if (dst.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(dst.qsa_pooled);
        dst.qsa_pooled = .{ .ctx = null };
        const ratio = @max(src.qsa_ratio, 1);
        if (src.qsa_pooled.ctx != null) {
            const ps = mlx.getShape(src.qsa_pooled);
            const need_blocks = @divTrunc(keep, ratio);
            if (ps.len < 2 or ps[1] < need_blocks) return error.QsaHistoryGap;
            if (materialize) {
                dst.qsa_pooled = try materializedOwnedCopy(s, src.qsa_pooled);
            } else {
                dst.qsa_pooled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&dst.qsa_pooled, src.qsa_pooled));
            }
            try truncatePooled(&dst.qsa_pooled, keep, ratio, s, materialize);
        }
        const from_src = try qsaLeftoverAt(src.aux_state, src_hist, keep, ratio, s);
        if (from_src.ctx != null) {
            if (own.ctx != null) _ = mlx.mlx_array_free(own);
            dst.aux_state = from_src;
        } else {
            dst.aux_state = if (own.ctx != null) own else .{ .ctx = null };
        }
        dst.qsa_ratio = src.qsa_ratio;
        dst.qsa_hist_rows = keep;
        dst.qsa_key_rows = if (dst.aux_state.ctx != null) mlx.getShape(dst.aux_state)[1] else 0;
        if (dst.qsa_pooled.ctx != null) {
            const ps = mlx.getShape(dst.qsa_pooled);
            dst.qsa_pooled_blocks = if (ps.len >= 2) ps[1] else 0;
        }
    }
}

/// Copy QSA aux/pooled from `src` onto `dst`, sliced to `pos` rows. Used when
/// a byte-budget trim would drop the latest snap — the only one that carries
/// the indexer history. A 122880-token trim that freed that snap left aux
/// empty; the next prefill then reshaped a 4096-row chunk into
/// `(1, nb, 4, 128)` and aborted (MLX reshape, 77% RAM — not an OOM).
/// `pos` is `dst.pos`: a checkpoint's aux is normalized to exactly `pos % ratio` rows at
/// capture, which is what the keep-own arm relies on when the source cannot supply them.
pub fn sliceQsaHistoryOntoCheckpoint(dst: *SSMCheckpoint, src: *const SSMCheckpoint, pos: usize, s: mlx.mlx_stream) !void {
    if (dst.layers.len != src.layers.len) return error.SsmCheckpointLayerMismatch;
    const keep: c_int = @intCast(pos);
    for (dst.layers, src.layers) |*d, s_l| {
        if (!snapshotHasQsaHistory(&s_l)) continue;
        if (d.conv_state.ctx != null and mlx.mlx_array_size(d.conv_state) > 0) continue;
        const ratio = @max(s_l.qsa_ratio, 1);
        if (s_l.qsa_pooled.ctx != null) {
            if (d.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(d.qsa_pooled);
            d.qsa_pooled = try materializedOwnedCopy(s, s_l.qsa_pooled);
            try truncatePooled(&d.qsa_pooled, keep, ratio, s, true);
        }
        const src_hist: c_int = if (s_l.qsa_rows > 0) s_l.qsa_rows else keep;
        const leftover_n = @mod(keep, ratio);
        const leftover = try qsaLeftoverAt(s_l.aux_state, src_hist, keep, ratio, s);
        if (leftover.ctx != null) {
            if (d.aux_state.ctx != null) _ = mlx.mlx_array_free(d.aux_state);
            d.aux_state = leftover;
        } else if (leftover_n == 0) {
            if (d.aux_state.ctx != null) _ = mlx.mlx_array_free(d.aux_state);
            d.aux_state = .{ .ctx = null };
        }
        d.qsa_ratio = s_l.qsa_ratio;
        d.qsa_rows = keep;
    }
    // Materialize now: an unevaluated copy node keeps `src`'s buffer alive.
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var n: usize = 0;
    for (dst.layers) |*d| {
        if (!snapshotHasQsaHistory(d)) continue;
        inline for (.{ d.aux_state, d.qsa_pooled }) |arr| {
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
                n += 1;
            }
        }
    }
    if (n > 0) _ = mlx.mlx_eval(vec);
}

// ── tests ──

fn qsaCoverTestCp(ring: c_int, blocks: c_int, rows: c_int, ratio: c_int, hd: c_int, s: mlx.mlx_stream) !SSMCheckpoint {
    const layers = try std.testing.allocator.alloc(SSMCacheEntrySnapshot, 1);
    errdefer std.testing.allocator.free(layers);
    layers[0] = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio, .qsa_rows = rows };
    var cp: SSMCheckpoint = .{ .pos = @intCast(rows), .layers = layers };
    errdefer cp.deinit(std.testing.allocator);
    if (ring > 0) {
        layers[0].aux_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&layers[0].aux_state, &[_]c_int{ 1, ring, hd }, 3, .float32, s));
    }
    if (blocks > 0) {
        layers[0].qsa_pooled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&layers[0].qsa_pooled, &[_]c_int{ 1, blocks, hd }, 3, .float32, s));
    }
    return cp;
}

test "checkpointQsaCoversPos: a bank-less leftover does not cover pos" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var leftover = try qsaCoverTestCp(1, 0, 17, 4, 8, s);
    defer leftover.deinit(t.allocator);
    try t.expect(!checkpointQsaCoversPos(&leftover, &leftover, 17));
    try t.expect(!qsaRestoreSatisfiesForward(&.{
        .{
            .conv_state = .{ .ctx = null },
            .ssm_state = .{ .ctx = null },
            .aux_state = leftover.layers[0].aux_state,
            .qsa_pooled = .{ .ctx = null },
            .initialized = true,
            .qsa_ratio = 4,
            .qsa_hist_rows = 17,
        },
    }, 17));
}

test "checkpointHasQsaPooled: a mid-block leftover is not the pooled bank's home" {
    const t = std.testing;
    const s = mlx.gpuStream();
    // Every site that asks "does the indexer bank live on this checkpoint" must say no to a
    // leftover-only snap, or a warm turn at L-30 never gets the overlay and cold-prefills.
    var leftover_only = try qsaCoverTestCp(2, 0, 46, 4, 8, s);
    defer leftover_only.deinit(t.allocator);
    var with_bank = try qsaCoverTestCp(0, 16, 64, 4, 8, s);
    defer with_bank.deinit(t.allocator);
    try t.expect(checkpointHasQsaHistory(&leftover_only));
    try t.expect(!checkpointHasQsaPooled(&leftover_only));
    try t.expect(checkpointHasQsaPooled(&with_bank));
}

test "checkpointQsaCoversPos: a checkpoint without the leftover for pos is a miss" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const ratio: c_int = 4;
    // The history source: 64 rows as 16 pooled blocks beside a raw ring that is the newest
    // QSA_RING_ROWS rows — a v7 file's aux, the leftover at no position in particular.
    var src = try qsaCoverTestCp(QSA_RING_ROWS, 16, 64, ratio, hd, s);
    defer src.deinit(t.allocator);
    var at48 = try qsaCoverTestCp(0, 0, 48, ratio, hd, s);
    defer at48.deinit(t.allocator);
    var at46 = try qsaCoverTestCp(2, 0, 46, ratio, hd, s);
    defer at46.deinit(t.allocator);
    var at46_bare = try qsaCoverTestCp(0, 0, 46, ratio, hd, s);
    defer at46_bare.deinit(t.allocator);

    try t.expect(checkpointQsaCoversPos(&at48, &src, 48));
    try t.expect(checkpointQsaCoversPos(&at46, &src, 46));
    try t.expect(checkpointQsaCoversPos(&src, &src, 46));
    // A checkpoint with no leftover of its own (a v7 snap) covers pos when the source still
    // physically holds those rows — the whole raw history in a v7 entry file, and here the
    // rows 44..46 the newest ring happens to span.
    try t.expect(checkpointQsaCoversPos(&at46_bare, &src, 46));
    var full = try qsaCoverTestCp(64, 16, 64, ratio, hd, s);
    defer full.deinit(t.allocator);
    try t.expect(checkpointQsaCoversPos(&at46_bare, &full, 46));
    // Rows the source dropped are a miss, never a silently wrong leftover.
    var ring_at_256 = try qsaCoverTestCp(QSA_RING_ROWS, 64, 256, ratio, hd, s);
    defer ring_at_256.deinit(t.allocator);
    try t.expect(!checkpointQsaCoversPos(&at46_bare, &ring_at_256, 46));
    // A source whose pooled bank stops short of pos is still a miss.
    var short = try qsaCoverTestCp(QSA_RING_ROWS, 8, 64, ratio, hd, s);
    defer short.deinit(t.allocator);
    try t.expect(!checkpointQsaCoversPos(&at46, &short, 46));
}

test "checkpointQsaCoversPos: a 32-row ring on the bank covers every residue at or below its pos" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const ratio: c_int = 4;
    var src = try qsaCoverTestCp(QSA_RING_ROWS, 5, 20, ratio, hd, s);
    defer src.deinit(t.allocator);
    var r: usize = 0;
    while (r < 4) : (r += 1) {
        const pos: usize = 16 + r;
        var cp = try qsaCoverTestCp(0, 0, @intCast(pos), ratio, hd, s);
        defer cp.deinit(t.allocator);
        try t.expect(checkpointQsaCoversPos(&cp, &src, pos));
    }
}

test "QSA checkpoint aux+pooled bytes are O(rows + checkpoints)" {
    const s = mlx.gpuStream();
    const n: c_int = 64;
    const hd: c_int = 8;
    const ratio: c_int = 4;
    const aux_shape = [_]c_int{ 1, n, hd };
    const pooled_shape = [_]c_int{ 1, @divExact(n, ratio), hd };
    var aux = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, @floatFromInt(n * hd), 1.0, .float32, s));
        try mlx.check(mlx.mlx_reshape(&aux, flat, &aux_shape, 3, s));
        try mlx.check(mlx.mlx_array_eval(aux));
    }
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 1000.0, 1000.0 + @as(f64, @floatFromInt((n / ratio) * hd)), 1.0, .float32, s));
        try mlx.check(mlx.mlx_reshape(&pooled, flat, &pooled_shape, 3, s));
        try mlx.check(mlx.mlx_array_eval(pooled));
    }
    const live: SSMCacheEntry = .{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux,
        .qsa_pooled = pooled,
        .qsa_ratio = ratio,
    };
    aux = mlx.mlx_array_new();
    pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux, live.aux_state));
    try mlx.check(mlx.mlx_array_set(&pooled, live.qsa_pooled));
    var live_arr = [_]SSMCacheEntry{live};
    defer {
        if (live_arr[0].aux_state.ctx != null) _ = mlx.mlx_array_free(live_arr[0].aux_state);
        if (live_arr[0].qsa_pooled.ctx != null) _ = mlx.mlx_array_free(live_arr[0].qsa_pooled);
    }

    // 46 is mid-block on purpose: a restore there needs the two raw rows of the block it
    // reopens, so the leftover assertions below are not vacuous.
    const positions = [_]usize{ 16, 32, 46, 64 };
    var cps: [positions.len]SSMCheckpoint = undefined;
    for (positions, 0..) |p, i| {
        cps[i] = try captureSsmCheckpoint(testing.allocator, &live_arr, p, s);
    }
    defer for (&cps) |*c| c.deinit(testing.allocator);
    try attachQsaHistoryToLatest(&cps, &live_arr, s);

    const held = qsaHistoryBytesHeld(&cps);
    const one: u64 = @as(u64, @intCast(n)) * @as(u64, @intCast(hd)) * 4 +
        @as(u64, @intCast(@divExact(n, ratio))) * @as(u64, @intCast(hd)) * 4;
    try testing.expect(held <= one + @as(u64, positions.len) * 64);
    for (positions, 0..) |p, i| {
        try testing.expectEqual(@as(c_int, @intCast(p)), cps[i].layers[0].qsa_rows);
        try testing.expectEqual(@as(c_int, @intCast(p % @as(usize, @intCast(ratio)))), checkpointQsaLeftoverRows(&cps[i]));
    }
    try testing.expect(checkpointHasQsaHistory(&cps[positions.len - 1]));

    var dest = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
    }};
    defer ssmFreeQsaState(&dest[0]);
    try restoreSsmCheckpoint(&dest, &cps[2]);
    try applyQsaHistoryAt(&dest, &cps[positions.len - 1], 46, s, false);
    // The leftover is the checkpoint's own raw rows 44 and 45, not the newest ring's overlap.
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(dest[0].aux_state)[1]);
    {
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_slice(&want, aux, &[_]c_int{ 0, 44, 0 }, 3, &[_]c_int{ 1, 46, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dest[0].aux_state, want, s));
    }
    try testing.expectEqual(@as(c_int, 11), dest[0].qsa_pooled_blocks);
    try testing.expectEqual(@as(c_int, 46), dest[0].qsa_hist_rows);

    var t: Trunk = undefined;

    t.s = s;
    t.allocator = testing.allocator;
    const extra_shape = [_]c_int{ 1, 1, hd };
    var extra = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(extra);
    try mlx.check(mlx.mlx_ones(&extra, &extra_shape, 3, .float32, s));
    try t.qsaAppendKeys(&dest[0], extra, 46);
    try mlx.check(mlx.mlx_array_eval(dest[0].aux_state));
    try testing.expectEqual(@as(c_int, 3), mlx.getShape(dest[0].aux_state)[1]);
    try testing.expect(cps[positions.len - 1].layers[0].aux_state.ctx == null);
}

test "qsa_rows is the checkpoint position, never current live length" {
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const ratio: c_int = 4;
    const aux_shape = [_]c_int{ 1, 16, hd };
    var aux = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux);
    try mlx.check(mlx.mlx_ones(&aux, &aux_shape, 3, .float32, s));
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    const pooled_shape = [_]c_int{ 1, 4, hd };
    try mlx.check(mlx.mlx_ones(&pooled, &pooled_shape, 3, .float32, s));
    var live16 = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux,
        .qsa_pooled = pooled,
        .qsa_ratio = ratio,
    }};
    aux = mlx.mlx_array_new();
    pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux, live16[0].aux_state));
    try mlx.check(mlx.mlx_array_set(&pooled, live16[0].qsa_pooled));
    defer ssmFreeQsaState(&live16[0]);

    var cp32 = try captureSsmCheckpoint(testing.allocator, &live16, 32, s);
    defer cp32.deinit(testing.allocator);
    try testing.expectEqual(@as(c_int, 32), cp32.layers[0].qsa_rows);

    const aux64_shape = [_]c_int{ 1, 64, hd };
    var aux64 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux64);
    try mlx.check(mlx.mlx_ones(&aux64, &aux64_shape, 3, .float32, s));
    var pooled64 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled64);
    const pooled64_shape = [_]c_int{ 1, 16, hd };
    try mlx.check(mlx.mlx_ones(&pooled64, &pooled64_shape, 3, .float32, s));
    var live64 = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux64,
        .qsa_pooled = pooled64,
        .qsa_ratio = ratio,
    }};
    aux64 = mlx.mlx_array_new();
    pooled64 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux64, live64[0].aux_state));
    try mlx.check(mlx.mlx_array_set(&pooled64, live64[0].qsa_pooled));
    defer ssmFreeQsaState(&live64[0]);
    var cps = [_]SSMCheckpoint{
        try captureSsmCheckpoint(testing.allocator, &live64, 16, s),
        try captureSsmCheckpoint(testing.allocator, &live64, 64, s),
    };
    defer for (&cps) |*c| c.deinit(testing.allocator);
    try testing.expectEqual(@as(c_int, 16), cps[0].layers[0].qsa_rows);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[0].qsa_rows);

    const aux8_shape = [_]c_int{ 1, 8, hd };
    var aux8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux8);
    try mlx.check(mlx.mlx_ones(&aux8, &aux8_shape, 3, .float32, s));
    var live8 = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux8,
        .qsa_ratio = ratio,
    }};
    aux8 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux8, live8[0].aux_state));
    defer ssmFreeQsaState(&live8[0]);
    try attachQsaHistoryToLatest(&cps, &live8, s);
    try testing.expectEqual(@as(c_int, 16), cps[0].layers[0].qsa_rows);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[0].qsa_rows);
}

test "captureSsmCheckpoint does not copy QSA aux history" {
    // QSA indexer keys grow with seq. Copying them into every stride
    // snapshot is what OOM'd a 400k prefill. Capture must leave them on
    // the live entry; attachQsaHistoryToLatest puts one copy on the latest.
    const s = mlx.gpuStream();
    const aux_shape = [_]c_int{ 1, 2048, 128 };
    var aux = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux);
    try mlx.check(mlx.mlx_ones(&aux, &aux_shape, 3, .bfloat16, s));
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    const pooled_shape = [_]c_int{ 1, 512, 128 };
    try mlx.check(mlx.mlx_ones(&pooled, &pooled_shape, 3, .bfloat16, s));

    const entry: SSMCacheEntry = .{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux,
        .qsa_pooled = pooled,
        .qsa_ratio = 4,
    };
    aux = mlx.mlx_array_new();
    pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux, entry.aux_state));
    try mlx.check(mlx.mlx_array_set(&pooled, entry.qsa_pooled));

    var entries = [_]SSMCacheEntry{entry};
    defer {
        if (entries[0].aux_state.ctx != null) _ = mlx.mlx_array_free(entries[0].aux_state);
        if (entries[0].qsa_pooled.ctx != null) _ = mlx.mlx_array_free(entries[0].qsa_pooled);
    }
    var cp = try captureSsmCheckpoint(testing.allocator, &entries, 2048, s);
    defer cp.deinit(testing.allocator);
    try testing.expect(!checkpointHasQsaHistory(&cp));
    try testing.expect(cp.layers[0].aux_state.ctx == null);
    try testing.expect(cp.layers[0].qsa_pooled.ctx == null);
    try testing.expect(entries[0].aux_state.ctx != null);
    try testing.expectEqual(@as(c_int, 2048), mlx.getShape(entries[0].aux_state)[1]);
}

test "attachQsaHistoryToLatest is one copy; applyQsaHistoryAt slices to pos" {
    const s = mlx.gpuStream();
    const n: c_int = 32;
    const aux_shape = [_]c_int{ 1, n, 8 };
    var aux = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, @floatFromInt(n * 8), 1.0, .float32, s));
        try mlx.check(mlx.mlx_reshape(&aux, flat, &aux_shape, 3, s));
        try mlx.check(mlx.mlx_array_eval(aux));
    }
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_ones(&pooled, &[_]c_int{ 1, n / 4, 8 }, 3, .float32, s));
    const live: SSMCacheEntry = .{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux,
        .qsa_pooled = pooled,
        .qsa_ratio = 4,
    };
    pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&pooled, live.qsa_pooled));
    aux = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux, live.aux_state));

    var live_arr = [_]SSMCacheEntry{live};
    defer {
        if (live_arr[0].aux_state.ctx != null) _ = mlx.mlx_array_free(live_arr[0].aux_state);
        if (live_arr[0].qsa_pooled.ctx != null) _ = mlx.mlx_array_free(live_arr[0].qsa_pooled);
    }
    const cp_a = try captureSsmCheckpoint(testing.allocator, &live_arr, 16, s);
    const cp_b = try captureSsmCheckpoint(testing.allocator, &live_arr, 32, s);
    var cps = [_]SSMCheckpoint{ cp_a, cp_b };
    defer for (&cps) |*c| c.deinit(testing.allocator);

    try testing.expect(!checkpointHasQsaHistory(&cps[0]));
    try testing.expect(!checkpointHasQsaHistory(&cps[1]));
    const bytes_before = ssmCheckpointBytes(&cps[0]) + ssmCheckpointBytes(&cps[1]);

    try attachQsaHistoryToLatest(&cps, &live_arr, s);
    try testing.expect(!checkpointHasQsaHistory(&cps[0]));
    try testing.expect(checkpointHasQsaHistory(&cps[1]));
    const pooled_bytes: u64 = @as(u64, @intCast(n / 4)) * 8 * 4;
    try testing.expectEqual(bytes_before + pooled_bytes, ssmCheckpointBytes(&cps[0]) + ssmCheckpointBytes(&cps[1]));

    var dest_arr = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
    }};
    defer ssmFreeQsaState(&dest_arr[0]);
    try restoreSsmCheckpoint(&dest_arr, &cps[0]);
    try applyQsaHistoryAt(&dest_arr, &cps[1], 16, s, false);
    try testing.expect(dest_arr[0].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 4), dest_arr[0].qsa_pooled_blocks);

    try sliceQsaHistoryOntoCheckpoint(&cps[0], &cps[1], 16, s);
    try testing.expect(checkpointHasQsaHistory(&cps[0]));
    try testing.expect(cps[0].layers[0].aux_state.ctx == null);

    keepOnlyLatestQsaHistory(&cps);
    try testing.expect(!checkpointHasQsaHistory(&cps[0]));
    try testing.expect(checkpointHasQsaHistory(&cps[1]));
    try testing.expect(cps[1].layers[0].aux_state.ctx == null);

    var short = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live_arr, 16, s)};
    defer short[0].deinit(testing.allocator);
    try attachQsaHistoryToLatest(&short, &live_arr, s);
    try testing.expect(short[0].layers[0].aux_state.ctx == null);
    try testing.expect(short[0].layers[0].qsa_pooled.ctx != null);
}

test "handoffQsaHistoryToLatest: the newest snap takes a VIEW of the live buffer — zero allocation, sliced to pos, bytes equal to the copy arm" {
    // The handoff hands the dying slot's buffer to the entry: same rows, same bytes, no allocation.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var t: Trunk = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    const hd: c_int = 16;
    var prng = std.Random.DefaultPrng.init(0x5A5A_1D0F);
    const rnd = prng.random();

    var live = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true }};
    defer ssmFreeQsaState(&live[0]);
    // A reserved append: the buffer carries slack past the published rows.
    const keys = try attn256RandBf16(rnd, &[_]c_int{ 1, QSA_RING_ROWS, hd }, s);
    defer _ = mlx.mlx_array_free(keys);
    try t.qsaAppendKeys(&live[0], keys, 0);
    const blocks = try attn256RandBf16(rnd, &[_]c_int{ 1, QSA_RING_ROWS / 4, hd }, s);
    defer _ = mlx.mlx_array_free(blocks);
    try t.qsaAppendPooled(&live[0], blocks, 0);
    try mlx.check(mlx.mlx_array_eval(live[0].aux_state));
    try mlx.check(mlx.mlx_array_eval(live[0].qsa_pooled));
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(live[0].qsa_key_buf)[1]);

    var ref = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, @intCast(QSA_RING_ROWS), s)};
    defer ref[0].deinit(testing.allocator);
    try attachQsaHistoryToLatest(&ref, &live, s);

    var cps = [_]SSMCheckpoint{
        try captureSsmCheckpoint(testing.allocator, &live, 16, s),
        try captureSsmCheckpoint(testing.allocator, &live, @intCast(QSA_RING_ROWS), s),
    };
    defer for (&cps) |*c| c.deinit(testing.allocator);
    var before: usize = 0;
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_get_active_memory(&before);
    try handoffQsaHistoryToLatest(&cps, &live, s);
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    try testing.expect(after <= before);
    try testing.expect(!checkpointHasQsaHistory(&cps[0]));
    try testing.expect(checkpointHasQsaHistory(&cps[1]));
    const got = &cps[1].layers[0];
    try testing.expect(got.aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, QSA_RING_ROWS / 4), mlx.getShape(got.qsa_pooled)[1]);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got.qsa_pooled, ref[0].layers[0].qsa_pooled, s));
    try testing.expectEqual(ssmCheckpointBytes(&ref[0]), ssmCheckpointBytes(&cps[1]));
    try testing.expectEqual(@as(u64, 8 * @as(u64, @intCast(hd)) * 2), ssmCheckpointBytes(&cps[1]));
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(live[0].aux_state)[1]);
    ssmFreeQsaState(&live[0]);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got.qsa_pooled, ref[0].layers[0].qsa_pooled, s));
}

test "handoffQsaHistoryToLatest materializes when the buffer's slack past the snap exceeds QSA_HANDOFF_MAX_SLACK_ROWS" {
    // Past the slack bar the handoff pays the copy.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var t: Trunk = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    const hd: c_int = 16;
    var prng = std.Random.DefaultPrng.init(0x0DD5_1ACC);
    const rnd = prng.random();
    try testing.expect(!qsaHandoffExceedsSlack(512 + QSA_HANDOFF_MAX_SLACK_ROWS, 512));
    try testing.expect(qsaHandoffExceedsSlack(512 + QSA_HANDOFF_MAX_SLACK_ROWS + 1, 512));

    var live = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true }};
    defer ssmFreeQsaState(&live[0]);
    const keys = try attn256RandBf16(rnd, &[_]c_int{ 1, QSA_RING_ROWS, hd }, s);
    defer _ = mlx.mlx_array_free(keys);
    try t.qsaAppendKeys(&live[0], keys, 0);
    try mlx.check(mlx.mlx_array_eval(live[0].aux_state));
    var ref = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, @intCast(QSA_RING_ROWS), s)};
    defer ref[0].deinit(testing.allocator);
    try attachQsaHistoryToLatest(&ref, &live, s);

    var cps = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, @intCast(QSA_RING_ROWS), s)};
    defer cps[0].deinit(testing.allocator);
    try handoffQsaHistoryToLatest(&cps, &live, s);
    try testing.expect(cps[0].layers[0].aux_state.ctx == null);
    ssmFreeQsaState(&live[0]);
}

test "applyQsaHistoryAt: a sliced restore is a VIEW of the entry's history; the trim's slice is a real copy" {
    // Restore: the slot appends into qsa_pooled_buf via seedCapBuf, so a view of the entry is not written through.
    // Trim: the inherit slice must own its rows; a view would pin the donor's full bank.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const n: c_int = 32;
    const aux_shape = [_]c_int{ 1, n, 8 };
    var aux = mlx.mlx_array_new();
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, @floatFromInt(n * 8), 1.0, .float32, s));
        try mlx.check(mlx.mlx_reshape(&aux, flat, &aux_shape, 3, s));
        try mlx.check(mlx.mlx_array_eval(aux));
    }
    var pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&pooled, &[_]c_int{ 1, n / 4, 8 }, 3, .float32, s));
    var live = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = aux, .qsa_pooled = pooled, .qsa_ratio = 4 }};
    defer ssmFreeQsaState(&live[0]);
    var cps = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, 32, s)};
    defer cps[0].deinit(testing.allocator);
    try attachQsaHistoryToLatest(&cps, &live, s);

    var dest = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true }};
    defer ssmFreeQsaState(&dest[0]);
    try restoreSsmCheckpoint(&dest, &cps[0]);
    try applyQsaHistoryAt(&dest, &cps[0], 16, s, false);
    try testing.expectEqual(@as(c_int, 4), dest[0].qsa_pooled_blocks);
    try mlx.check(mlx.mlx_array_eval(dest[0].qsa_pooled));
    try mlx.check(mlx.mlx_array_eval(cps[0].layers[0].qsa_pooled));
    const dest_ptr = mlx.mlx_array_data_float32(dest[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    const src_ptr = mlx.mlx_array_data_float32(cps[0].layers[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(src_ptr, dest_ptr);

    var t: Trunk = undefined;

    t.s = s;
    t.allocator = testing.allocator;
    var extra = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(extra);
    try mlx.check(mlx.mlx_ones(&extra, &[_]c_int{ 1, 1, 8 }, 3, .float32, s));
    try t.qsaAppendPooled(&dest[0], extra, dest[0].qsa_pooled_blocks);
    try mlx.check(mlx.mlx_array_eval(dest[0].qsa_pooled));
    try mlx.check(mlx.mlx_array_eval(cps[0].layers[0].qsa_pooled));
    const after_ptr = mlx.mlx_array_data_float32(dest[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    const src_after = mlx.mlx_array_data_float32(cps[0].layers[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(src_ptr, src_after);
    try testing.expect(after_ptr != src_ptr);

    var kept = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, 16, s)};
    defer kept[0].deinit(testing.allocator);
    try sliceQsaHistoryOntoCheckpoint(&kept[0], &cps[0], 16, s);
    try testing.expect(kept[0].layers[0].aux_state.ctx == null);
    try testing.expect(kept[0].layers[0].qsa_pooled.ctx != null);
    try mlx.check(mlx.mlx_array_eval(kept[0].layers[0].qsa_pooled));
    const trim_ptr = mlx.mlx_array_data_float32(kept[0].layers[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    try testing.expect(trim_ptr != src_ptr);
}
