//! The QSA indexer: key append into the ring and pooled blocks, the block score sheet, and the mask or block picks
//! each attention arm reads (`qsaMask`, `qsaMaskFromQk`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const qsa_row_mask = @import("../../../sdk/blocks/attention/sparse_block/row_mask.zig");
const history = @import("history.zig");
const mtp = @import("../mtp.zig");
const prof_mod = @import("../prof.zig");
const ProfClock = @import("../../../engines/mlx/diag/profile.zig").ProfClock;
const env = @import("../../../env.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Trunk = @import("../trunk.zig").Trunk;
const concatAxis0 = @import("../../../mlx/array.zig").concatAxis0;
const MropeCosSin = @import("../../../sdk/blocks/rope/mrope.zig").MropeCosSin;
const mropeContext = @import("../../../engines/mlx/methods/mrope.zig").mropeContext;
const qsaEntrySatisfiesForward = @import("checkpoint.zig").qsaEntrySatisfiesForward;
const qsaHistoryRows = @import("checkpoint.zig").qsaHistoryRows;
const qsaBatchedGatherEnabled = @import("gather.zig").qsaBatchedGatherEnabled;
const QSA_RING_ROWS = @import("history.zig").QSA_RING_ROWS;
const capBufAppend = @import("history.zig").capBufAppend;
const seedCapBuf = @import("history.zig").seedCapBuf;
const headPlaceScores = @import("head_place.zig").headPlaceScores;
const qsaMaskFromBlockSel = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlockSel;
const qsaMaskFromBlocks = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlocks;
const qsaAllBlocksVisible = @import("pooled.zig").qsaAllBlocksVisible;
const qsaIdxRopeFusedEnabled = @import("pooled.zig").qsaIdxRopeFusedEnabled;
const qsaPoolNormRopeFused = @import("pooled.zig").qsaPoolNormRopeFused;
const qsaPooledTableEq = @import("pooled.zig").qsaPooledTableEq;
const qsaDumpQsaBlocks = @import("score.zig").qsaDumpQsaBlocks;
const qsaPooledRowContiguous = @import("../../../sdk/blocks/attention/sparse_block/score.zig").qsaPooledRowContiguous;
const qsaScoreFusedEligibleFrom = @import("../../../sdk/blocks/attention/sparse_block/score.zig").qsaScoreFusedEligibleFrom;
const qsaScoreRowsPerChunkFused = @import("../../../sdk/blocks/attention/sparse_block/score.zig").qsaScoreRowsPerChunkFused;
const qsaScoreSheet = @import("../../../sdk/blocks/attention/sparse_block/score.zig").qsaScoreSheet;
const qsaSelectComposedOps = @import("../../../sdk/blocks/attention/sparse_block/select.zig").qsaSelectComposedOps;
const qsaSortWithInvisible = @import("../../../sdk/blocks/attention/sparse_block/select.zig").qsaSortWithInvisible;
const qsaTopBlocksOps = @import("../../../sdk/blocks/attention/sparse_block/select.zig").qsaTopBlocksOps;
const qsaSelectKernelEnabled = @import("../../../sdk/blocks/attention/sparse_block/select.zig").qsaSelectKernelEnabled;
const qsaSelectTopBlocks = @import("../../../sdk/blocks/attention/sparse_block/select.zig").qsaSelectTopBlocks;
const Blocks = @import("../../../sdk/blocks/attention/sparse_block/params.zig").Blocks;
const sb_select = @import("../../../sdk/blocks/attention/sparse_block/select.zig");
const qsaSelectCfgBuilds = sb_select.qsaSelectCfgBuilds;
const tieBias = sb_select.tieBias;
const qsaVisSheetForTest = sb_select.qsaVisSheetForTest;
const qsaArraysAllEqual = @import("../../../sdk/testing/parity.zig").qsaArraysAllEqual;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const testing = std.testing;
const qsaWantsBlocks = @import("sparse_attn.zig").qsaWantsBlocks;

pub fn qsaInstallRing(_: *Trunk, entry: *SSMCacheEntry, src: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const sh = mlx.getShape(src);
    const rows = sh[1];
    const n: c_int = @min(rows, QSA_RING_ROWS);
    var tmp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tmp);
    var view = src;
    if (n < rows) {
        try mlx.check(mlx.mlx_slice(
            &tmp,
            src,
            &[_]c_int{ 0, rows - n, 0 },
            3,
            &[_]c_int{ sh[0], rows, sh[2] },
            3,
            &[_]c_int{ 1, 1, 1 },
            3,
            s,
        ));
        view = tmp;
    }
    if (entry.aux_state.ctx != null) _ = mlx.mlx_array_free(entry.aux_state);
    if (entry.qsa_key_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_key_buf);
    entry.aux_state = .{ .ctx = null };
    entry.qsa_key_buf = .{ .ctx = null };
    entry.qsa_key_rows = 0;
    entry.qsa_key_buf = try seedCapBuf(s, view, 1, @intCast(QSA_RING_ROWS));
    entry.qsa_key_rows = n;
    var start = [_]c_int{ 0, 0, 0 };
    var stop = [_]c_int{ sh[0], n, sh[2] };
    entry.aux_state = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(entry.aux_state);
        entry.aux_state = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(&entry.aux_state, entry.qsa_key_buf, &start, 3, &stop, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
}

/// Append this forward's raw indexer keys `[B, S, hd]` to the layer's QSA ring and
/// republish `entry.aux_state` as the tight view. `offset` is the authority the accelerator is checked against.
pub fn qsaAppendKeys(self: *Trunk, entry: *SSMCacheEntry, k_raw: mlx.mlx_array, offset: c_int) !void {
    const s = self.s;
    const add = mlx.getShape(k_raw)[1];
    const held: c_int = if (entry.aux_state.ctx != null) mlx.getShape(entry.aux_state)[1] else 0;
    const hist = qsaHistoryRows(entry);

    if (!(entry.qsa_key_buf.ctx != null and hist == offset and held == entry.qsa_key_rows)) {
        if (entry.qsa_key_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_key_buf);
        entry.qsa_key_buf = .{ .ctx = null };
        entry.qsa_key_rows = 0;
        if (held > 0) {
            entry.qsa_key_buf = try seedCapBuf(s, entry.aux_state, 1, @intCast(QSA_RING_ROWS));
            entry.qsa_key_rows = held;
        }
    }

    const ring_cap: usize = @intCast(QSA_RING_ROWS);
    if (add >= QSA_RING_ROWS) {
        try self.qsaInstallRing(entry, k_raw, s);
    } else if (held + add <= QSA_RING_ROWS) {
        try capBufAppend(s, &entry.qsa_key_buf, &entry.aux_state, &entry.qsa_key_rows, k_raw, 1, ring_cap);
    } else {
        const keep_old = QSA_RING_ROWS - add;
        const sh = mlx.getShape(entry.aux_state);
        var old_tail = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(old_tail);
        try mlx.check(mlx.mlx_slice(
            &old_tail,
            entry.aux_state,
            &[_]c_int{ 0, held - keep_old, 0 },
            3,
            &[_]c_int{ sh[0], held, sh[2] },
            3,
            &[_]c_int{ 1, 1, 1 },
            3,
            s,
        ));
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        const parts = [_]mlx.mlx_array{ old_tail, k_raw };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
        try self.qsaInstallRing(entry, cat, s);
    }
    entry.qsa_hist_rows = offset + add;
}

/// Extend the pooled block-key bank by `new3` and republish `entry.qsa_pooled` as the tight
/// view, byte-identical to the old `concatenate`. A `nb_cached` mismatch re-seeds.
pub fn qsaAppendPooled(self: *Trunk, entry: *SSMCacheEntry, new3: mlx.mlx_array, nb_cached: c_int) !void {
    const s = self.s;
    if (!(entry.qsa_pooled_buf.ctx != null and entry.qsa_pooled_blocks == nb_cached)) {
        if (entry.qsa_pooled_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_pooled_buf);
        entry.qsa_pooled_buf = .{ .ctx = null };
        entry.qsa_pooled_blocks = 0;
        if (nb_cached > 0 and entry.qsa_pooled.ctx != null) {
            const seed_ratio: usize = @max(@as(usize, @intCast(entry.qsa_ratio)), 1);
            entry.qsa_pooled_buf = try seedCapBuf(s, entry.qsa_pooled, 1, entry.qsa_reserve_rows / seed_ratio);
            entry.qsa_pooled_blocks = nb_cached;
        }
    }
    // One row per completed block: the row reservation divided by the ratio.
    const ratio: usize = @max(@as(usize, @intCast(entry.qsa_ratio)), 1);
    try capBufAppend(s, &entry.qsa_pooled_buf, &entry.qsa_pooled, &entry.qsa_pooled_blocks, new3, 1, entry.qsa_reserve_rows / ratio);
}

/// `n` columns of the block-score matmul's key operand: `[B, 1, hd, n]` f32, the pooled
/// bank transposed and up-cast.
pub fn qsaScoreCols(self: *Trunk, pooled: mlx.mlx_array, batch: c_int, n: c_int, idx_hd: c_int) !mlx.mlx_array {
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    try mlx.check(mlx.mlx_reshape(&k_rope, pooled, &[_]c_int{ batch, 1, n, idx_hd }, 4, self.s));
    const kperm = [_]c_int{ 0, 1, 3, 2 };
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_transpose_axes(&kt, k_rope, &kperm, 4, self.s));
    var k32 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(k32);
    try mlx.check(mlx.mlx_astype(&k32, kt, .float32, self.s));
    return k32;
}

/// The score operand for `nb` blocks, borrowed. Built only on the composed arm.
/// A completed block appends columns; from-scratch rebuilds after restore, a cold
/// entry, or rollback.
pub fn qsaScoreBank(self: *Trunk, entry: *SSMCacheEntry, batch: c_int, nb: c_int, idx_hd: c_int) !mlx.mlx_array {
    if (entry.qsa_score_bank.ctx != null and entry.qsa_score_blocks == nb) return entry.qsa_score_bank;
    const old_nb = entry.qsa_score_blocks;
    const ratio: usize = @max(@as(usize, @intCast(entry.qsa_ratio)), 1);
    const reserve = entry.qsa_reserve_rows / ratio;
    if (entry.qsa_score_buf.ctx != null and old_nb > 0 and nb > old_nb and entry.qsa_pooled.ctx != null) {
        const psh = mlx.getShape(entry.qsa_pooled);
        if (psh[1] >= nb) {
            const n_new = nb - old_nb;
            var new_pooled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(new_pooled);
            try mlx.check(mlx.mlx_slice(
                &new_pooled,
                entry.qsa_pooled,
                &[_]c_int{ 0, old_nb, 0 },
                3,
                &[_]c_int{ psh[0], nb, psh[2] },
                3,
                &[_]c_int{ 1, 1, 1 },
                3,
                self.s,
            ));
            const cols = try self.qsaScoreCols(new_pooled, batch, n_new, idx_hd);
            defer _ = mlx.mlx_array_free(cols);
            try capBufAppend(self.s, &entry.qsa_score_buf, &entry.qsa_score_bank, &entry.qsa_score_blocks, cols, 3, reserve);
            self.qsa_score_bank_builds += 1;
            history.qsa_score_incremental_appends += 1;
            return entry.qsa_score_bank;
        }
    }
    if (entry.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_bank);
    if (entry.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_buf);
    entry.qsa_score_bank = .{ .ctx = null };
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    const cols = try self.qsaScoreCols(entry.qsa_pooled, batch, nb, idx_hd);
    defer _ = mlx.mlx_array_free(cols);
    try capBufAppend(self.s, &entry.qsa_score_buf, &entry.qsa_score_bank, &entry.qsa_score_blocks, cols, 3, reserve);
    self.qsa_score_bank_builds += 1;
    return entry.qsa_score_bank;
}

pub fn qsaScoreOperand(self: *Trunk, entry: *SSMCacheEntry, batch: c_int, nb: c_int, idx_hd: c_int, fused: bool) !mlx.mlx_array {
    if (fused) return .{ .ctx = null };
    return self.qsaScoreBank(entry, batch, nb, idx_hd);
}

/// The pooled-block cos/sin for this forward, built on first ask. Borrowed handles.
pub fn qsaPooledCosSin(self: *Trunk, ctx: *ForwardCtx, rope_dims: c_int, base: c_int, step: c_int, n: c_int, dt: mlx.mlx_dtype) !MropeCosSin {
    const is_mrope = ctx.mrope_pos != null;
    const c = &self.qsa_pooled_rope;
    if (c.stale.load(.acquire)) c.deinit();
    if (c.cos.ctx != null and
        c.base == base and c.step == step and c.n == n and
        c.dtype == dt and c.mrope == is_mrope and
        qsaPooledTableEq(c.mrope_pos, ctx.mrope_pos) and
        c.mrope_total == ctx.mrope_total and c.mrope_delta == ctx.mrope_delta)
        return .{ .cos = c.cos, .sin = c.sin };
    c.deinit();
    const cs = if (is_mrope)
        try self.mropeCosSinAt(mropeContext(ctx), @intCast(base), @intCast(step), @intCast(n), dt)
    else
        try self.ropeCosSinFromFreqs(rope_dims, try self.ropeInvFreq(rope_dims, self.config.rope_theta), @floatFromInt(base), @floatFromInt(step), n, dt);
    c.cos = cs.cos;
    c.sin = cs.sin;
    c.base = base;
    c.step = step;
    c.n = n;
    c.dtype = dt;
    c.mrope = is_mrope;
    c.mrope_pos = ctx.mrope_pos;
    c.mrope_total = ctx.mrope_total;
    c.mrope_delta = ctx.mrope_delta;
    c.builds += 1;
    return cs;
}

/// Pooled block keys `[B, nb, D]` from raw index keys `kb4 [B, nb, ratio, D]`:
/// f32 block mean → bf16 → idx_k_norm → partial RoPE at the block-start
/// positions `base + i*ratio`. One fused kernel on text turns, bit-identical
/// to the composed chain; M-RoPE and odd shapes keep the chain.
pub fn qsaPooledKeys(self: *Trunk, ctx: *ForwardCtx, kb4: mlx.mlx_array, norm_w: mlx.mlx_array, rope_dims: c_int, base: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(kb4);
    if (ctx.mrope_pos == null and qsaIdxRopeFusedEnabled() and mlx.mlx_array_dtype(norm_w) == .bfloat16) {
        const cs = try self.qsaPooledCosSin(ctx, rope_dims, base, sh[2], sh[1], .bfloat16);
        if (try qsaPoolNormRopeFused(self.s, kb4, norm_w, self.config.rms_norm_eps, cs.cos, cs.sin, rope_dims)) |out| return out;
    }
    return self.qsaPooledKeysComposed(ctx, kb4, norm_w, rope_dims, base);
}

/// The composed reference for `qsaPooledKeys` (and its M-RoPE arm).
pub fn qsaPooledKeysComposed(self: *Trunk, ctx: *ForwardCtx, kb4: mlx.mlx_array, norm_w: mlx.mlx_array, rope_dims: c_int, base: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(kb4);
    const pn = try self.qsaPoolNorm(kb4, norm_w);
    defer _ = mlx.mlx_array_free(pn);
    const cs = try self.qsaPooledCosSin(ctx, rope_dims, base, sh[2], sh[1], mlx.mlx_array_dtype(pn));
    if (ctx.mrope_pos == null) return self.qsaPooledRopeComposed(pn, rope_dims, cs.cos, cs.sin);
    var pn4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pn4);
    try mlx.check(mlx.mlx_expand_dims(&pn4, pn, 1, self.s));
    const new_rope = try self.applyMrope(pn4, cs.cos, cs.sin, rope_dims);
    defer _ = mlx.mlx_array_free(new_rope);
    var new3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&new3, new_rope, &[_]c_int{ sh[0], sh[1], sh[3] }, 3, self.s));
    return new3;
}

/// `rms_norm(bf16(mean_f32(kb4, axis 2)), w)` → `[B, nb, D]`.
pub fn qsaPoolNorm(self: *Trunk, kb4: mlx.mlx_array, norm_w: mlx.mlx_array) !mlx.mlx_array {
    var kb_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kb_f32);
    try mlx.check(mlx.mlx_astype(&kb_f32, kb4, .float32, self.s));
    var pooled_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled_f32);
    try mlx.check(mlx.mlx_mean_axis(&pooled_f32, kb_f32, 2, false, self.s));
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_astype(&pooled, pooled_f32, .bfloat16, self.s));
    return self.rmsNorm(pooled, norm_w);
}

/// Scalar partial RoPE of normed pooled keys `pn [B, nb, D]` with row-i cos/sin `[nb, rope_dims]`.
pub fn qsaPooledRopeComposed(self: *Trunk, pn: mlx.mlx_array, rope_dims: c_int, cosv: mlx.mlx_array, sinv: mlx.mlx_array) !mlx.mlx_array {
    const sh = mlx.getShape(pn);
    var pn4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pn4);
    try mlx.check(mlx.mlx_expand_dims(&pn4, pn, 1, self.s));
    const roped = try self.ropeApplyCosSin(pn4, rope_dims, cosv, sinv);
    defer _ = mlx.mlx_array_free(roped);
    var new3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&new3, roped, &[_]c_int{ sh[0], sh[1], sh[2] }, 3, self.s));
    return new3;
}

/// QSA: append this chunk's raw index keys to the layer's history and,
/// past the token budget, build the block-selection mask. Returns the
/// bool `[B,1,S,kv]` mask or null-ctx for dense attention.
/// `cache_len` = key rows already in the history, `pos_base` = absolute
/// position of key row 0 (0 for the trunk; the MTP head's cache starts at
/// the first draft position). Block/tail/causal arithmetic is in cache
/// coordinates; RoPE angles use absolute positions — the SAME M-RoPE table
/// as attention on an image request (queries from the chunk's cos/sin,
/// pooled block keys at 3-D block-start positions), scalar rope otherwise.
pub fn qsaSlotEntry(sc: *ForwardCtx, layer: u32) *SSMCacheEntry {
    if (sc.qsa_entry) |e| return e;
    return &sc.ssm_entries.?[layer];
}

pub fn qsaMask(self: *Trunk, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, layer: u32, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const qk = try self.qmatmul(x, fa.idx_qk_w, fa.idx_qk_s, fa.idx_qk_b); // [B,S,(n+1)*hd]
    defer _ = mlx.mlx_array_free(qk);
    if (ctx.batch_slots) |slots| return self.qsaMaskBatched(slots, qk, fa, layer);
    return self.qsaMaskFromQk(ctx, qk, fa, entry, cache_len, pos_base, batch, seq_len, layer, true);
}

/// Batched decode: the indexer projection ran once on `[N,1,·]`; each row
/// then walks the serial per-slot body against ITS slot's key history and
/// position, and the per-slot masks are false-padded to the group's
/// longest kv and stacked to `[N,1,1,kv_max]`. Null-ctx when no slot is
/// past the budget (dense group — the plain batched mask suffices).
pub fn qsaMaskBatched(self: *Trunk, slots: []const *ForwardCtx, qk: mlx.mlx_array, fa: *const FullAttnWeights, layer: u32) !mlx.mlx_array {
    const N = slots.len;
    // The per-slot ctxs have no other net: without this one throw orphaned a slot's selection.
    errdefer for (slots) |sc| {
        if (sc.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(sc.qsa_blocks);
        sc.qsa_blocks = .{ .ctx = null };
    };
    const qk_sh = mlx.getShape(qk);
    const seq_len: c_int = qk_sh[1];
    const w: c_int = qk_sh[2];
    const masks = try self.allocator.alloc(mlx.mlx_array, N);
    defer self.allocator.free(masks);
    for (masks) |*m| m.* = .{ .ctx = null };
    defer for (masks) |m| {
        if (m.ctx != null) _ = mlx.mlx_array_free(m);
    };
    const kv_lens = try self.allocator.alloc(c_int, N);
    defer self.allocator.free(kv_lens);
    var kv_max: c_int = 0;
    var any = false;
    var any_blocks = false;
    const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
    for (slots, 0..) |sc, i| {
        const e = qsaSlotEntry(sc, layer);
        const cache_len: c_int = @intCast(sc.moe_seq_offset.*);
        const i_c: c_int = @intCast(i);
        var qk_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qk_i);
        try mlx.check(mlx.mlx_slice(&qk_i, qk, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, seq_len, w }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        masks[i] = try self.qsaMaskFromQk(sc, qk_i, fa, e, cache_len, sc.qsa_pos_base, 1, seq_len, layer, false);
        kv_lens[i] = cache_len + seq_len;
        if (kv_lens[i] > kv_max) kv_max = kv_lens[i];
        if (masks[i].ctx != null) any = true;
        if (sc.qsa_blocks.ctx != null) any_blocks = true;
    }
    const keep_blocks = qsaBatchedGatherEnabled() and self.qwen4 != null and any_blocks;
    if (keep_blocks) {
        for (slots, 0..) |sc, i| {
            if (sc.qsa_blocks.ctx == null and masks[i].ctx != null) {
                sc.qsa_mask = masks[i];
                masks[i] = .{ .ctx = null };
            }
        }
        return mlx.mlx_array_new();
    }
    for (slots, 0..) |sc, i| {
        if (sc.qsa_blocks.ctx != null) {
            // Decode-width selection is per-slot unusable in the stacked
            // mask path — expand it to the equivalent dense mask (the same
            // fallback the serial arm uses when its gatherer declines).
            const m = try qsaMaskFromBlocks(self.s, sc.qsa_blocks, kv_lens[i], ratio);
            _ = mlx.mlx_array_free(sc.qsa_blocks);
            sc.qsa_blocks = .{ .ctx = null };
            _ = mlx.mlx_array_free(masks[i]);
            masks[i] = m;
            any = true;
        }
    }
    if (!any) return mlx.mlx_array_new();
    const padded = try self.allocator.alloc(mlx.mlx_array, N);
    defer self.allocator.free(padded);
    for (padded) |*m| m.* = .{ .ctx = null };
    defer for (padded) |m| {
        if (m.ctx != null) _ = mlx.mlx_array_free(m);
    };
    const false_v = mlx.mlx_array_new_bool(false);
    defer _ = mlx.mlx_array_free(false_v);
    for (0..N) |i| {
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        if (masks[i].ctx != null) {
            try mlx.check(mlx.mlx_array_set(&row, masks[i]));
        } else {
            try mlx.check(mlx.mlx_ones(&row, &[_]c_int{ 1, 1, seq_len, kv_lens[i] }, 4, .bool_, self.s));
        }
        const pad_axes = [_]c_int{3};
        const low = [_]c_int{0};
        const high = [_]c_int{kv_max - kv_lens[i]};
        try mlx.check(mlx.mlx_pad(&padded[i], row, &pad_axes, 1, &low, 1, &high, 1, false_v, "constant", self.s));
    }
    return concatAxis0(self.s, padded);
}

/// Serial per-slot body of `qsaMask` over the projected `qk` rows.
pub fn qsaMaskFromQk(self: *Trunk, ctx: *ForwardCtx, qk: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, layer: u32, solo: bool) !mlx.mlx_array {
    const offset = cache_len;
    const cfg = &self.config;
    const n_idx: c_int = @intCast(cfg.indexer_n_heads);
    const idx_hd: c_int = @intCast(cfg.indexer_head_dim);
    const ratio: c_int = @intCast(cfg.indexer_compress_ratio);
    const budget: c_int = @intCast(cfg.indexer_budget);
    const block_topk: c_int = @divTrunc(budget, ratio);
    const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
    const strides3 = [_]c_int{ 1, 1, 1 };
    var k_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_raw);
    try mlx.check(mlx.mlx_slice(&k_raw, qk, &[_]c_int{ 0, 0, n_idx * idx_hd }, 3, &[_]c_int{ batch, seq_len, (n_idx + 1) * idx_hd }, 3, &strides3, 3, self.s));

    const kv: c_int = offset + seq_len;
    entry.qsa_ratio = ratio;
    if (!qsaEntrySatisfiesForward(entry, offset)) return error.QsaHistoryGap;
    if (env.onCached(&prof_mod.qwen4_no_pooled_env, .QWEN4_NO_POOLED)) return error.QsaPooledRequired;
    const nb: c_int = @divTrunc(kv, ratio);
    const nb_cached: c_int = if (entry.qsa_pooled.ctx != null) mlx.getShape(entry.qsa_pooled)[1] else 0;
    if (nb > nb_cached) {
        const leftover_n = offset - nb_cached * ratio;
        if (leftover_n < 0) return error.QsaHistoryGap;
        const held: c_int = if (entry.aux_state.ctx != null) mlx.getShape(entry.aux_state)[1] else 0;
        if (leftover_n > held) return error.QsaHistoryGap;
        const n_new = nb - nb_cached;
        const need = n_new * ratio;
        const from_k = need - leftover_n;
        if (from_k < 0 or from_k > seq_len) return error.QsaHistoryGap;
        var kb_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb_flat);
        if (leftover_n == 0) {
            try mlx.check(mlx.mlx_slice(&kb_flat, k_raw, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, need, idx_hd }, 3, &strides3, 3, self.s));
        } else {
            const ash = mlx.getShape(entry.aux_state);
            var leftover = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(leftover);
            try mlx.check(mlx.mlx_slice(&leftover, entry.aux_state, &[_]c_int{ 0, held - leftover_n, 0 }, 3, &[_]c_int{ ash[0], held, ash[2] }, 3, &strides3, 3, self.s));
            var k_part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_part);
            try mlx.check(mlx.mlx_slice(&k_part, k_raw, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, from_k, idx_hd }, 3, &strides3, 3, self.s));
            const parts = [_]mlx.mlx_array{ leftover, k_part };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&kb_flat, vec, 1, self.s));
        }
        const kb_shape = [_]c_int{ batch, n_new, ratio, idx_hd };
        const want: usize = @as(usize, @intCast(batch)) * @as(usize, @intCast(n_new)) * @as(usize, @intCast(ratio)) * @as(usize, @intCast(idx_hd));
        if (mlx.mlx_array_size(kb_flat) != want) return error.QsaHistoryGap;
        var kb4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb4);
        try mlx.check(mlx.mlx_reshape(&kb4, kb_flat, &kb_shape, 4, self.s));
        const new3 = try self.qsaPooledKeys(ctx, kb4, fa.idx_k_norm, rope_dims, pos_base + nb_cached * ratio);
        defer _ = mlx.mlx_array_free(new3);
        try self.qsaAppendPooled(entry, new3, nb_cached);
    }

    try self.qsaAppendKeys(entry, k_raw, offset);
    var keys = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keys);
    try mlx.check(mlx.mlx_array_set(&keys, entry.aux_state));

    if (kv <= budget + ratio - 1) return mlx.mlx_array_new();
    std.debug.assert(batch == 1);
    if (!Trunk.qsa_engaged_logged) {
        Trunk.qsa_engaged_logged = true;
        log.info("[qsa] sparse attention engaged: kv={d} blocks={d} top-{d} (budget {d}, ratio {d})\n", .{ kv, @divTrunc(kv, ratio), block_topk, budget, ratio });
    }

    // Index queries: per-head norm → [B, n, S, hd] → partial RoPE at the
    // chunk's positions.
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_slice(&q, qk, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, n_idx * idx_hd }, 3, &strides3, 3, self.s));
    const q_shape = [_]c_int{ batch, seq_len, n_idx, idx_hd };
    var q4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q4);
    try mlx.check(mlx.mlx_reshape(&q4, q, &q_shape, 4, self.s));
    const qn = try self.rmsNorm(q4, fa.idx_q_norm);
    defer _ = mlx.mlx_array_free(qn);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var qt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qt);
    try mlx.check(mlx.mlx_transpose_axes(&qt, qn, &perm, 4, self.s));
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    if (ctx.mrope_cos_cur) |cos| {
        _ = mlx.mlx_array_free(q_rope);
        q_rope = try self.applyMrope(qt, cos, ctx.mrope_sin_cur.?, rope_dims);
    } else {
        const eff_off: c_int = pos_base + offset + (if (ctx.mrope_pos != null) ctx.mrope_delta else 0);
        // The SAME spectrum attention rotates with (scaled when the config
        // says YaRN): the index queries and keys are the only place the
        // model asks "how far apart are these two blocks", and a block
        // pick made on unscaled angles at 1M positions would select
        // against a position notion the attention layers no longer have.
        // Indexer is 128-wide / 64 rotated, so score = ms²·A+B and mscale
        // CAN change a top-k. M-RoPE tables already carry mscale via
        // fillCosSin; this scalar arm must too. Never yarnScaleQK here
        // (that table is [head_dim] = 256). Keep mlx_fast_rope scale=1.0.
        const use_yarn = self.yarnActive();
        const freqs: mlx.mlx_array = if (use_yarn) self.rope_freqs_yarn.? else .{ .ctx = null };
        if (ctx.head_rope_pos.ctx != null) {
            try mlx.check(mlx.mlx_fast_rope_dynamic(&q_rope, qt, rope_dims, false, .{ .value = cfg.rope_theta, .has_value = !use_yarn }, 1.0, ctx.head_rope_pos, freqs, self.s));
        } else {
            try mlx.check(mlx.mlx_fast_rope(&q_rope, qt, rope_dims, false, .{ .value = cfg.rope_theta, .has_value = !use_yarn }, 1.0, eff_off, freqs, self.s));
        }
        if (use_yarn) try self.yarnScaleRotated(&q_rope, rope_dims);
    }

    // Pooled block keys: mean of leftover + this chunk, already appended above.
    if (env.onCached(&prof_mod.qwen4_debug_scores_env, .QWEN4_DEBUG_SCORES)) {
        const qs = mlx.getShape(entry.qsa_pooled);
        std.debug.print("[qsa] kv={d} nb={d} cached={d} pooled shape {any} keys {any}\n", .{ kv, nb, nb_cached, qs, mlx.getShape(keys) });
    }
    // scores[b, s, blk] = sum_h relu(q_h . k_blk) in f32 like the reference; the 1/sqrt(hd)
    // scale is dropped because it is monotone for the top-k. Borrowed from the entry.
    const fused = qsaScoreFusedEligibleFrom(q_rope, entry.qsa_pooled) and
        qsaPooledRowContiguous(entry.qsa_pooled) and mlx.streamIsGpu(self.s);
    const k32 = try self.qsaScoreOperand(entry, batch, nb, idx_hd, fused);

    // Block selection (no dense mask) for prefill widths AND decode width:
    // at S==1 the same sorted single-row top-k feeds the decode gatherer
    // in `gatedFullAttnWith`, so a decode step reads O(budget) rows
    // instead of O(kv). Batched slots each take this branch against their
    // own history; `qsaMaskBatched` expands per-slot selections back to
    // masks because the stacked path has no per-slot gather.
    // Verify widths (2 .. FUSED256_MIN_Q_LEN-1, an MTP/DFlash block) take
    // the selection too: `qsaVerifyGatherAttn` reads the UNION of the
    // rows' selections instead of the whole cache. Its own kv floor is
    // higher than the prefill/decode one — the union is fixed-size.
    // A placed draft row takes the mask arm: its dead rows are masked, which no gatherer reads.
    const want_blocks = qsaWantsBlocks(batch, kv, seq_len, ctx.cache.config.scheme == .affine) and ctx.head_place == null;
    // A solo decode row on dense KV hands its picks to `qsa_decode` at every kv; the mask arm needs none.
    const dec_row = solo and batch == 1 and seq_len == 1 and ctx.head_place == null and !ctx.cache.config.isQuant();
    if (want_blocks or dec_row) {
        // Prefill: sorted per-row block indices for the gather kernel;
        // the dense [S, kv] mask is never built. Decode (S==1): the same
        // single-row selection for the decode gatherer.
        const prof = env.onCached(&prof_mod.qwen4_profile_qsa_env, .QWEN4_PROFILE_QSA);
        var clk: ProfClock = undefined;
        if (prof) {
            if (k32.ctx != null) try mlx.check(mlx.mlx_array_eval(k32));
            clk = ProfClock.init();
        }
        // Free before assign: a `defer` registered below a fallible call leaked a handle per tick.
        if (ctx.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_blocks);
        ctx.qsa_blocks = .{ .ctx = null };
        ctx.qsa_blocks = try self.qsaSelectBlocks(q_rope, k32, entry.qsa_pooled, offset, seq_len, nb, block_topk, fused);
        if (ctx.qsa_blocks.ctx != null) {
            const kb_dim: c_int = if (mlx.mlx_array_ndim(ctx.qsa_blocks) == 3) mlx.getShape(ctx.qsa_blocks)[2] else block_topk;
            qsaDumpQsaBlocks(layer, offset, seq_len, kb_dim, ctx.qsa_blocks);
        }
        if (prof) {
            try mlx.check(mlx.mlx_array_eval(ctx.qsa_blocks));
            log.info("[qsa-prof] select S={d} nb={d}: {d:.2} ms\n", .{ seq_len, nb, @as(f64, @floatFromInt(clk.lap())) / 1e6 });
        }
        return mlx.mlx_array_new();
    }

    const sheet = try qsaScoreSheet(self.s, q_rope, entry.qsa_pooled, k32, fused);
    defer _ = mlx.mlx_array_free(sheet);
    const scores = if (ctx.head_place) |p| try headPlaceScores(self.s, sheet, p, ratio, nb) else sheet;
    defer if (ctx.head_place != null) {
        _ = mlx.mlx_array_free(scores);
    };

    if (env.onCached(&prof_mod.qwen4_debug_scores_env, .QWEN4_DEBUG_SCORES)) {
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, scores, .float32, self.s));
        try mlx.check(mlx.mlx_array_eval(f));
        const d = mlx.mlx_array_data_float32(f).?;
        const nbu: usize = @intCast(nb);
        for (0..@intCast(seq_len)) |r| {
            std.debug.print("[qsa scores] row {d}:", .{r});
            for (0..nbu) |c| std.debug.print(" {d:.4}", .{d[r * nbu + c]});
            std.debug.print("\n", .{});
        }
    }
    // Past the budget an all-visible call skips the sheet; the `nb <= block_topk` arm always builds it.
    const all_vis = qsaAllBlocksVisible(offset, nb, ratio);
    var vis3 = mlx.mlx_array{ .ctx = null };
    defer if (vis3.ctx != null) {
        _ = mlx.mlx_array_free(vis3);
    };
    if (!all_vis or nb <= block_topk) vis3 = try self.qsaBlockVisibility(offset, seq_len, nb, ratio);

    try self.qsa_consts.ensure(self.s, nb, ratio);
    return qsaSelectMaskOps(self.allocator, self.s, scores, vis3, self.qsa_consts.tie_bias, offset, seq_len, kv, nb, ratio, block_topk, batch);
}

/// Block selection -> dense `[B,1,S,kv]` QSA mask, as pure ops. `vis3` may be null-ctx
/// (every block complete); every consumer must then skip its own op, since mlx-c aborts
/// on a null handle. Split out so the all-visible claim is testable without a Trunk.
pub fn qsaSelectMaskOps(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    vis3: mlx.mlx_array,
    tie_bias: mlx.mlx_array,
    offset: c_int,
    seq_len: c_int,
    kv: c_int,
    nb: c_int,
    ratio: c_int,
    block_topk: c_int,
    batch: c_int,
) !mlx.mlx_array {
    const strides3 = [_]c_int{ 1, 1, 1 };
    var blk_sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_sel);
    if (nb <= block_topk) {
        // This arm is the sheet; a null sheet means every block is both visible and selected.
        if (vis3.ctx == null) {
            try mlx.check(mlx.mlx_ones(&blk_sel, &[_]c_int{ batch, seq_len, nb }, 3, .bool_, s));
        } else {
            try mlx.check(mlx.mlx_array_set(&blk_sel, vis3));
        }
    } else {
        // One tie rule: the gather path selects with the exact radix-select kernel, so this
        // arm prefers the same kernel and keeps the argpartition chain as its decline path.
        var top_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(top_idx);
        var have_exact = false;
        if (batch == 1) {
            const bounds_host = try alloc.alloc(i32, @intCast(seq_len));
            defer alloc.free(bounds_host);
            (Blocks{ .size = @intCast(ratio), .top_k = @intCast(block_topk) }).selectBounds(bounds_host, offset, nb, vis3.ctx == null);
            const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{seq_len}, 1, .int32);
            defer _ = mlx.mlx_array_free(bounds);
            if (qsaSelectTopBlocks(s, scores, bounds, block_topk) catch null) |picks| {
                defer _ = mlx.mlx_array_free(picks);
                // One row: the sorted picks (INT_MAX where short) go straight to the visibility mask.
                if (seq_len == 1) if (try qsa_row_mask.rowMask(s, picks, kv, ratio)) |m| return m;
                // The kernel pads a short row with INT_MAX; clamping those to 0 is safe because
                // the `logical_and` with the sheet below clears block 0 when it is not visible.
                const nb_v = mlx.mlx_array_new_int(nb);
                defer _ = mlx.mlx_array_free(nb_v);
                var valid = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(valid);
                try mlx.check(mlx.mlx_less(&valid, picks, nb_v, s));
                const zero_i = mlx.mlx_array_new_int(0);
                defer _ = mlx.mlx_array_free(zero_i);
                try mlx.check(mlx.mlx_where(&top_idx, valid, picks, zero_i, s));
                have_exact = true;
            }
        }
        var part = mlx.mlx_array{ .ctx = null };
        defer if (part.ctx != null) {
            _ = mlx.mlx_array_free(part);
        };
        if (!have_exact) {
            part = try qsaTopBlocksOps(s, scores, vis3, tie_bias, nb, block_topk);
            try mlx.check(mlx.mlx_slice(&top_idx, part, &[_]c_int{ 0, 0, nb - block_topk }, 3, &[_]c_int{ batch, seq_len, nb }, 3, &strides3, 3, s));
        }
        var falses = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(falses);
        try mlx.check(mlx.mlx_zeros(&falses, &[_]c_int{ batch, seq_len, nb }, 3, .bool_, s));
        const true_v = mlx.mlx_array_new_bool(true);
        defer _ = mlx.mlx_array_free(true_v);
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_put_along_axis(&picked, falses, top_idx, true_v, -1, s));
        if (vis3.ctx == null) {
            // `picked AND all-true` is `picked`; passing the null handle to mlx aborted the process.
            try mlx.check(mlx.mlx_array_set(&blk_sel, picked));
        } else {
            try mlx.check(mlx.mlx_logical_and(&blk_sel, picked, vis3, s));
        }
    }
    return qsaMaskFromBlockSel(s, blk_sel, offset, seq_len, kv, nb, ratio, batch);
}

/// Block visibility [1,S,nb]: block blk is complete for the query at
/// cache position p iff blk*ratio + ratio - 1 <= p. The block-end column comes from `qsa_consts`.
pub fn qsaBlockVisibility(self: *Trunk, offset: c_int, seq_len: c_int, nb: c_int, ratio: c_int) !mlx.mlx_array {
    const s = self.s;
    try self.qsa_consts.ensure(s, nb, ratio);
    var pos_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_col);
    try mlx.check(mlx.mlx_arange(&pos_col, @floatFromInt(offset), @floatFromInt(offset + seq_len), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos_col, &[_]c_int{ seq_len, 1 }, 2, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_less_equal(&vis, self.qsa_consts.blk_end2, pos2, s)); // [S, nb]
    var vis3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_expand_dims(&vis3, vis, 0, s)); // [1,S,nb]
    return vis3;
}

/// One row-chunk of the block selection: the top-`kb` visible blocks per row, ascending,
/// INT_MAX past the row's count. Under `all_vis` the visibility sheet and both `where`s are
/// identities and are skipped; the fused arm reads the same claim through `Blocks.selectBounds`.
pub fn qsaChunkSelect(self: *Trunk, scores: mlx.mlx_array, canonical: mlx.mlx_array, row0: c_int, rows: c_int, nb: c_int, ratio: c_int, kb: c_int, block_topk: c_int, all_vis: bool) !mlx.mlx_array {
    if (canonical.ctx == null and qsaSelectKernelEnabled() and mlx.streamIsGpu(self.s)) {
        // The whole composed tail in one dispatch; bounds are host arithmetic.
        const bounds_host = try self.allocator.alloc(i32, @intCast(rows));
        defer self.allocator.free(bounds_host);
        (Blocks{ .size = @intCast(ratio), .top_k = @intCast(block_topk) }).selectBounds(bounds_host, row0, nb, all_vis);
        const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
        defer _ = mlx.mlx_array_free(bounds);
        if (try qsaSelectTopBlocks(self.s, scores, bounds, kb)) |fused| return fused;
    }
    var vis3 = mlx.mlx_array{ .ctx = null };
    defer if (vis3.ctx != null) {
        _ = mlx.mlx_array_free(vis3);
    };
    if (!all_vis) vis3 = try self.qsaBlockVisibility(row0, rows, nb, ratio);
    if (canonical.ctx == null) {
        try self.qsa_consts.ensure(self.s, nb, ratio);
        return qsaSelectComposedOps(self.s, scores, vis3, self.qsa_consts.tie_bias, rows, nb, kb, block_topk, all_vis);
    }
    // `nb <= block_topk`: every block is a pick, only visibility can remove one.
    var top_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_idx);
    try mlx.check(mlx.mlx_broadcast_to(&top_idx, canonical, &[_]c_int{ 1, rows, nb }, 3, self.s));
    return qsaSortWithInvisible(self.s, top_idx, vis3, all_vis);
}

/// Prefill block selection for the gather kernel: per row the top-k
/// visible blocks ascending, INT_MAX past the row's count ([1,S,kb]
/// int32). Row-chunked so the [n_idx, rows, nb] f32 score sheet stays
/// under `qsaScoreSheetBudget()`. `k32t` is the pooled key bank
/// [1,1,hd,nb] f32.
pub fn qsaSelectBlocks(self: *Trunk, q_rope: mlx.mlx_array, k32t: mlx.mlx_array, pooled: mlx.mlx_array, offset: c_int, seq_len: c_int, nb: c_int, block_topk: c_int, fused: bool) !mlx.mlx_array {
    const n_idx: c_int = @intCast(self.config.indexer_n_heads);
    const idx_hd: c_int = @intCast(self.config.indexer_head_dim);
    const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
    const kb: c_int = @min(nb, block_topk);
    const rows_per: c_int = @intCast(qsaScoreRowsPerChunkFused(@intCast(n_idx), @intCast(nb), @intCast(seq_len), fused));
    const strides4 = [_]c_int{ 1, 1, 1, 1 };
    var canonical = mlx.mlx_array{ .ctx = null };
    defer if (canonical.ctx != null) {
        _ = mlx.mlx_array_free(canonical);
    };
    if (nb <= block_topk) {
        var ar = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ar);
        try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(nb), 1.0, .int32, self.s));
        canonical = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&canonical, ar, &[_]c_int{ 1, 1, nb }, 3, self.s));
    }

    var parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(self.allocator);
    }
    var r0: c_int = 0;
    while (r0 < seq_len) : (r0 += rows_per) {
        const r1: c_int = @min(r0 + rows_per, seq_len);
        const rows = r1 - r0;
        var q_chunk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_chunk);
        try mlx.check(mlx.mlx_slice(&q_chunk, q_rope, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, n_idx, r1, idx_hd }, 4, &strides4, 4, self.s));
        const scores = try qsaScoreSheet(self.s, q_chunk, pooled, k32t, fused);
        defer _ = mlx.mlx_array_free(scores);
        const sorted = try self.qsaChunkSelect(scores, canonical, offset + r0, rows, nb, ratio, kb, block_topk, qsaAllBlocksVisible(offset + r0, nb, ratio));
        errdefer _ = mlx.mlx_array_free(sorted);
        try parts.append(self.allocator, sorted);
    }
    if (parts.items.len == 1) return parts.pop().?;
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var merged = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&merged, vec, 1, self.s));
    return merged;
}

// ── The QSA all-visible identity skip at the engagement boundary ──

/// Random f32 score sheet (the reference's own dtype).
fn qsaRandScoresF32(rnd: std.Random, shape: []const c_int) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    // relu output: many exact zeros.
    for (data) |*x| {
        const v = rnd.float(f32) - 0.35;
        x.* = if (v > 0) v else 0.0;
    }
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

test "qsa mask: the all-visible skip must not hand a NULL sheet to an mlx op (kv just past the budget)" {
    // `mlx_logical_and(&blk_sel, picked, vis3)` in the nb > block_topk arm passed the null
    // all-visible sheet straight to mlx-c and killed the process. At decode width all-visible
    // is always true, so every generated token past kv = (block_topk + 1) * ratio crashed.
    // Bar: where the predicate claims all-visible, the real sheet is all-true and the null-sheet
    // arm produces the same mask.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ratio: c_int = 4;
    const batch: c_int = 1;
    var prng = std.Random.DefaultPrng.init(0x2068_B0DE);
    const rnd = prng.random();

    var checked_skip: usize = 0;
    // 2049..2051 sit just below the selection boundary; 2052 is the first kv that can crash.
    for ([_]c_int{ 2049, 2052, 2068, 2100, 4100 }) |kv| {
        // `nb` is the complete-block count, floor(kv/ratio).
        const nb = @divTrunc(kv, ratio);
        for ([_]c_int{ 1, 31, 1024 }) |seq_len| {
            if (seq_len > kv) continue;
            const offset = kv - seq_len;
            var topk_buf: [11]c_int = undefined;
            topk_buf[0] = nb;
            topk_buf[1] = nb + 4;
            for (1..9) |d| topk_buf[1 + d] = nb - @as(c_int, @intCast(d));
            topk_buf[10] = 512;

            for (topk_buf) |block_topk| {
                if (block_topk <= 0) continue;
                const scores = try qsaRandScoresF32(rnd, &[_]c_int{ batch, seq_len, nb });
                defer _ = mlx.mlx_array_free(scores);
                const tie = try tieBias(s, nb);
                defer _ = mlx.mlx_array_free(tie);
                const sheet = try qsaVisSheetForTest(s, offset, seq_len, nb, ratio);
                defer _ = mlx.mlx_array_free(sheet);

                const ref = try qsaSelectMaskOps(std.testing.allocator, s, scores, sheet, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
                defer _ = mlx.mlx_array_free(ref);
                try mlx.check(mlx.mlx_array_eval(ref));

                if (!qsaAllBlocksVisible(offset, nb, ratio)) continue;
                var ones = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ones);
                try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{ 1, seq_len, nb }, 3, .bool_, s));
                try std.testing.expect(try qsaArraysAllEqual(sheet, ones, s));
                const null_vis = mlx.mlx_array{ .ctx = null };
                const got = try qsaSelectMaskOps(std.testing.allocator, s, scores, null_vis, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
                defer _ = mlx.mlx_array_free(got);
                try std.testing.expect(try qsaArraysAllEqual(got, ref, s));
                checked_skip += 1;
            }
        }
    }
    std.debug.print("[qsa-boundary] all-visible skip exercised in {d} cases\n", .{checked_skip});
    try std.testing.expect(checked_skip >= 8);
}

test "qsa selection: the mask arm and the gather path use ONE tie rule" {
    // Two selection paths with two different tie rules would let two slots with identical
    // scores attend different blocks. The signal: the mask arm now dispatches the select kernel.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const batch: c_int = 1;
    // All-visible at width S needs S <= (kv % ratio) + 1, so kv must not be a multiple of ratio.
    const seq_len: c_int = 4;
    const kv: c_int = 4099;
    const nb = @divTrunc(kv, ratio);
    // 511 so this test owns a config key no other test builds.
    const block_topk: c_int = 511;
    var prng = std.Random.DefaultPrng.init(0x71E5_10FF);
    const rnd = prng.random();

    const scores = try qsaRandScoresF32(rnd, &[_]c_int{ batch, seq_len, nb });
    defer _ = mlx.mlx_array_free(scores);
    const tie = try tieBias(s, nb);
    defer _ = mlx.mlx_array_free(tie);
    const offset = kv - seq_len;
    try std.testing.expect(qsaAllBlocksVisible(offset, nb, ratio));
    const null_vis = mlx.mlx_array{ .ctx = null };

    const before = qsaSelectCfgBuilds();
    const got = try qsaSelectMaskOps(ta, s, scores, null_vis, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    // A build, not a hit: this key is unique to this test, so the counter moving IS the dispatch.
    try std.testing.expect(qsaSelectCfgBuilds() > before);

    try std.testing.expectEqual(@as(usize, 4), @as(usize, @intCast(mlx.mlx_array_ndim(got))));
    const gsh = mlx.getShape(got);
    try std.testing.expectEqual(batch, gsh[0]);
    try std.testing.expectEqual(seq_len, gsh[2]);
    try std.testing.expectEqual(kv, gsh[3]);

    // With the kernel off both arms fall to the chain: still one rule.
    sb_select.qsa_select_kernel_override = false;
    defer sb_select.qsa_select_kernel_override = null;
    const chain = try qsaSelectMaskOps(ta, s, scores, null_vis, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
    defer _ = mlx.mlx_array_free(chain);
    try mlx.check(mlx.mlx_array_eval(chain));
    try std.testing.expectEqual(kv, mlx.getShape(chain)[3]);
}

test "qsa boundary: past (block_topk+1)*ratio EVERY decode step reaches the all-visible skip" {
    // At decode width offset = kv-1 and nb = floor(kv/ratio), so the all-visible claim holds at
    // every kv and the skip is gated by `nb > block_topk` alone.
    const ratio: c_int = 4;
    const block_topk: c_int = 512; // budget 2048 / ratio
    const first: c_int = (block_topk + 1) * ratio; // 2052
    var kv: c_int = 1;
    while (kv <= 6000) : (kv += 1) {
        const nb = @divTrunc(kv, ratio);
        try std.testing.expect(qsaAllBlocksVisible(kv - 1, nb, ratio));
        try std.testing.expectEqual(kv >= first, nb > block_topk);
    }
    var kv2: c_int = 1;
    while (kv2 < first) : (kv2 += 1) {
        try std.testing.expect(!(@divTrunc(kv2, ratio) > block_topk));
    }
    // A prefill chunk's first row can sit far below the last complete block.
    try std.testing.expect(!qsaAllBlocksVisible(1028, @divTrunc(@as(c_int, 2052), ratio), ratio));
}

test "qsa visibility: decode width is ALWAYS all-visible; a prefill chunk's FIRST row decides" {
    const ratio: c_int = 4;
    // Decode (S == 1): all-visible for every ragged tail.
    var kv: c_int = 2049;
    while (kv <= 2400) : (kv += 1) {
        const nb = @divTrunc(kv, ratio);
        try testing.expect(qsaAllBlocksVisible(kv - 1, nb, ratio));
    }
    try testing.expect(qsaAllBlocksVisible(2051, 513, ratio));
    try testing.expect(!qsaAllBlocksVisible(2050, 513, ratio));
    try testing.expect(!qsaAllBlocksVisible(0, 513, ratio));
    try testing.expect(qsaAllBlocksVisible(0, 0, ratio));
}

test "qsa visibility skip: bit-identical selection where it is an identity, and a REAL difference where it is not" {
    // Where every block is visible the two arms must agree bit for bit; where they are not,
    // the arms must differ (else a blanket "always skip" would pass).
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();

    var t: Trunk = undefined;

    t.s = s;
    t.allocator = testing.allocator;
    t.config = .{};
    t.qsa_consts = .{};
    defer t.qsa_consts.deinit();

    const ratio: c_int = 4;
    const nb: c_int = 24;
    const block_topk: c_int = 8;
    const rows: c_int = 6;
    var prng = std.Random.DefaultPrng.init(0x5EED_9E11);
    const rnd = prng.random();
    var sc_buf: [@as(usize, @intCast(rows)) * @as(usize, @intCast(nb))]f32 = undefined;
    for (&sc_buf) |*v| v.* = if (rnd.boolean()) 0.0 else rnd.float(f32) * 4.0;
    const sc_shape = [_]c_int{ 1, rows, nb };
    const scores = mlx.mlx_array_new_data(&sc_buf, &sc_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const none = mlx.mlx_array{ .ctx = null };

    const row0_all: c_int = nb * ratio - 1;
    try testing.expect(qsaAllBlocksVisible(row0_all, nb, ratio));
    const a = try t.qsaChunkSelect(scores, none, row0_all, rows, nb, ratio, block_topk, block_topk, true);
    defer _ = mlx.mlx_array_free(a);
    const b = try t.qsaChunkSelect(scores, none, row0_all, rows, nb, ratio, block_topk, block_topk, false);
    defer _ = mlx.mlx_array_free(b);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(a, b, s));

    // One row earlier is not all-visible: the two arms must disagree.
    const row0_partial: c_int = 4;
    try testing.expect(!qsaAllBlocksVisible(row0_partial, nb, ratio));
    const c = try t.qsaChunkSelect(scores, none, row0_partial, rows, nb, ratio, block_topk, block_topk, true);
    defer _ = mlx.mlx_array_free(c);
    const d = try t.qsaChunkSelect(scores, none, row0_partial, rows, nb, ratio, block_topk, block_topk, false);
    defer _ = mlx.mlx_array_free(d);
    try testing.expect(try attn256MaxDiff(c, d, s) > 0.0);

    // The `nb <= block_topk` arm takes the same skip.
    var ar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ar);
    try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(nb), 1.0, .int32, s));
    var canonical = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(canonical);
    try mlx.check(mlx.mlx_reshape(&canonical, ar, &[_]c_int{ 1, 1, nb }, 3, s));
    const e = try t.qsaChunkSelect(scores, canonical, row0_all, rows, nb, ratio, nb, nb, true);
    defer _ = mlx.mlx_array_free(e);
    const f = try t.qsaChunkSelect(scores, canonical, row0_all, rows, nb, ratio, nb, nb, false);
    defer _ = mlx.mlx_array_free(f);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(e, f, s));
}
