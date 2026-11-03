//! engines/mlx/ssm/entry.
//! Owns SSMCacheEntry, arrayBytes, qsaRawKeyBytes, ssmEntryBytes, SSMCacheEntrySnapshot, ssmSnapshot, and 7 more.

const qwen4_mod = @import("../../../plugins/qwen4_exp/qwen4_exp.zig");
const mlx = @import("../../../mlx/ffi.zig");
const gdn_decode = @import("../../../sdk/blocks/gdn/decode.zig");
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const plePrevAfterAccept = @import("../../../plugins/qwen4_exp/ple.zig").plePrevAfterAccept;
const qsaHistoryRows = @import("../../../plugins/qwen4_exp/qsa/checkpoint.zig").qsaHistoryRows;
const qsaResliceToKeep = @import("../../../plugins/qwen4_exp/qsa/history.zig").qsaResliceToKeep;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const truncatePooled = @import("../../../plugins/qwen4_exp/qsa/history.zig").truncatePooled;
const PersistentSsmGroup = @import("group.zig").PersistentSsmGroup;
const ssmEvalHandles = @import("group.zig").ssmEvalHandles;
const ssmMaterializeLive = @import("group.zig").ssmMaterializeLive;
const ssmPatchGroupRow = @import("group.zig").ssmPatchGroupRow;

// ── SSM Cache (for GatedDeltaNet linear attention layers) ──

pub const SSMCacheEntry = struct {
    conv_state: mlx.mlx_array, // [B, kernel-1, conv_dim]
    ssm_state: mlx.mlx_array, // [B, Hv, Dv, Dk]
    initialized: bool,
    /// PLD spec-decode capture: per-position SSM states recorded by the GDN
    /// verify forward (`Transformer.spec_capture_ssm`). Shape [T, B, Hv, Dv, Dk]
    /// where T = verify length. Lets partial-accept rollback pick the
    /// accepted-position state WITHOUT a re-forward — the verify already ran the
    /// (sequential, expensive on a 48-layer GatedDeltaNet trunk) recurrence, so
    /// re-running it for the accepted prefix is pure waste. Null outside a
    /// capturing verify; freed at the end of every `nextPld` round.
    spec_state_seq: mlx.mlx_array = .{ .ctx = null },
    /// PLD spec-decode capture: the conv1d input `[B, (kernel-1)+T, conv_dim]`
    /// of the verify forward, so the accepted-position `conv_state` is just a
    /// slice (no re-forward). Null outside capture.
    spec_conv_input: mlx.mlx_array = .{ .ctx = null },
    /// A draft-tree verify's input state and per-row recurrence inputs: the
    /// commit replays the kept path from them (`gdn_decode.replay`).
    spec_state_in: mlx.mlx_array = .{ .ctx = null },
    spec_prework: gdn_decode.Prework = .{},
    /// qwen4_exp per-request state beside the GDN pair: the PLE dilated-conv
    /// window `[B, (K-1)*dilation, hc*hidden]` on the PLE layer, the raw QSA
    /// indexer key history `[B, kv, idx_hd]` on full-attention layers.
    /// QSA history is a prefix of ONE buffer — stride checkpoints must NOT
    /// copy it (attachQsaHistoryToLatest puts one copy on the latest snap).
    aux_state: mlx.mlx_array = .{ .ctx = null },
    /// PLE: the (ngram_size-1) tokens preceding the next forward.
    /// The (ngram_size - 1) tokens preceding this entry.
    ple_prev: [qwen4_mod.MAX_NGRAM_SIZE]u32 = @splat(0),
    ple_prev_valid: bool = false,
    /// Spec-verify capture for the PLE layer: the dilated-conv input
    /// `[B, state_len + T, hc*hidden]` and the token history (prev ++ ids)
    /// of the verify forward, so a partial accept slices instead of
    /// re-forwarding. Null / 0 outside a capturing verify.
    spec_ple_input: mlx.mlx_array = .{ .ctx = null },
    spec_ple_tokens: [32]u32 = @splat(0),
    spec_ple_len: u32 = 0,
    /// QSA: pooled + normed + roped block keys `[B, nb, idx_hd]` for the
    /// complete blocks of `aux_state`, extended incrementally per forward.
    qsa_pooled: mlx.mlx_array = .{ .ctx = null },
    qsa_ratio: c_int = 4,
    /// QSA raw-key history accelerator: a pre-grown `[B, cap, hd]` capacity buffer whose first
    /// `qsa_key_rows` rows are the history; `aux_state` stays the authority as a tight view.
    /// `qsaAppendKeys` trusts it only when `qsa_key_rows` equals the position being appended
    /// at; live sites that replace `aux_state` go through `ssmFreeQsaState`.
    qsa_key_buf: mlx.mlx_array = .{ .ctx = null },
    qsa_key_rows: c_int = 0,
    /// Pooled-bank accelerators, same discipline: `qsa_pooled_buf` is the capacity buffer
    /// `qsa_pooled` views; `qsa_score_bank` is the f32 `[B, 1, hd, nb]` score operand,
    /// a view of `qsa_score_buf`, which grows by appended columns as blocks complete.
    qsa_pooled_buf: mlx.mlx_array = .{ .ctx = null },
    qsa_pooled_blocks: c_int = 0,
    qsa_score_bank: mlx.mlx_array = .{ .ctx = null },
    qsa_score_buf: mlx.mlx_array = .{ .ctx = null },
    qsa_score_blocks: c_int = 0,
    /// Rows this entry's QSA history is asked to hold up front (the counterpart of
    /// `KVCache.reserve_tokens`); the pooled bank derives its own by `ratio`. 0 = none.
    qsa_reserve_rows: usize = 0,
    /// Logical QSA history length in tokens (`pooled_blocks * ratio + ring_tail`).
    qsa_hist_rows: c_int = 0,
    group: ?*PersistentSsmGroup = null,
    group_row: u16 = 0,
    group_layer: u16 = 0,
    /// Bumped on every write to this entry outside its group; the group refuses a
    /// member whose gen moved. Read only from the entries, never from a ctx copy.
    member_gen: u64 = 0,
};

pub const arrayBytes = @import("../../../sdk/kv/cache.zig").arrayBytes;

pub fn qsaRawKeyBytes(e: *const SSMCacheEntry) u64 {
    return arrayBytes(if (e.qsa_key_buf.ctx != null) e.qsa_key_buf else e.aux_state);
}

/// A live request's recurrent state: the hybrid's counterpart of its KV.
pub fn ssmEntryBytes(e: *const SSMCacheEntry) u64 {
    return arrayBytes(e.conv_state) + arrayBytes(e.ssm_state) + qsaRawKeyBytes(e);
}

/// SSM snapshot value. Holds clones of conv_state and ssm_state via refcount —
/// the underlying buffer is shared with the source entry, but the snapshot
/// owns its own array handles and frees them on deinit. Used for
/// speculative-decoding rollback (PLD) where we must be able to revert one
/// decode step on a hybrid model.
pub const SSMCacheEntrySnapshot = struct {
    conv_state: mlx.mlx_array,
    ssm_state: mlx.mlx_array,
    initialized: bool,
    /// qwen4_exp per-request state beside the GDN pair (see SSMCacheEntry):
    /// the PLE conv window / QSA key history, the pooled block keys, and the
    /// PLE token history. Null / invalid on every other arch.
    aux_state: mlx.mlx_array = .{ .ctx = null },
    qsa_pooled: mlx.mlx_array = .{ .ctx = null },
    qsa_ratio: c_int = 4,
    qsa_rows: c_int = 0,
    /// The (ngram_size - 1) tokens preceding this entry.
    ple_prev: [qwen4_mod.MAX_NGRAM_SIZE]u32 = @splat(0),
    ple_prev_valid: bool = false,
};

pub fn ssmSnapshot(src: *const SSMCacheEntry) SSMCacheEntrySnapshot {
    var out: SSMCacheEntrySnapshot = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = src.initialized,
    };
    // mlx_array_set increments refcount on the underlying buffer; both handles
    // point at the same data. Subsequent writes to src.conv_state/ssm_state
    // create NEW handles so the snapshot's view is unaffected.
    //
    // We must guard each field independently — the two states are populated by
    // DIFFERENT code paths and either may legitimately be null even when
    // `initialized == true`:
    //   - LFM2 `gatedConv` writes only `conv_state` (sets `initialized=true`),
    //     so `ssm_state.ctx == null` for the lifetime of that layer.
    //   - Mamba2/GatedDeltaNet flip the order: `conv1dWithCache` sets
    //     `initialized=true` BEFORE the recurrence body initializes
    //     `ssm_state`, so a snapshot taken in the middle would see a null
    //     ssm_state. (Currently snapshots are only taken between full forward
    //     passes, but defensive null-handling prevents future regressions.)
    //
    // Calling `mlx_array_set` with a null source aborts the process via mlx-c's
    // default error handler ("expected a non-empty mlx_array"), so we cannot
    // rely on `try mlx.check(...)`.
    var grouped = false;
    if (src.group) |g| {
        if (g.live and src.group_layer < g.layers.len) {
            const stream = mlx.gpuStream();
            const m = g.layers[src.group_layer];
            const conv_ok = if (m.conv_state.ctx != null) blk: {
                const c = ssmMaterializeLive(stream, src, m.conv_state) catch break :blk false;
                _ = mlx.mlx_array_free(out.conv_state);
                out.conv_state = c;
                break :blk true;
            } else true;
            const ssm_ok = if (m.ssm_state.ctx != null) blk: {
                const c = ssmMaterializeLive(stream, src, m.ssm_state) catch break :blk false;
                _ = mlx.mlx_array_free(out.ssm_state);
                out.ssm_state = c;
                break :blk true;
            } else true;
            const aux_ok = if (m.aux_state.ctx != null) blk: {
                const c = ssmMaterializeLive(stream, src, m.aux_state) catch break :blk false;
                out.aux_state = c;
                break :blk true;
            } else blk: {
                if (src.aux_state.ctx != null) {
                    out.aux_state = mlx.mlx_array_new();
                    _ = mlx.mlx_array_set(&out.aux_state, src.aux_state);
                }
                break :blk true;
            };
            if (conv_ok and ssm_ok and aux_ok) {
                ssmEvalHandles(&.{ out.conv_state, out.ssm_state, out.aux_state });
                grouped = true;
            } else {
                ssmSnapshotDeinit(&out);
                out = .{
                    .conv_state = mlx.mlx_array_new(),
                    .ssm_state = mlx.mlx_array_new(),
                    .initialized = src.initialized,
                };
            }
        }
    }
    if (!grouped) {
        if (src.conv_state.ctx != null) {
            _ = mlx.mlx_array_set(&out.conv_state, src.conv_state);
        }
        if (src.ssm_state.ctx != null) {
            _ = mlx.mlx_array_set(&out.ssm_state, src.ssm_state);
        }
        if (src.aux_state.ctx != null) {
            out.aux_state = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out.aux_state, src.aux_state);
        }
    }
    if (src.qsa_pooled.ctx != null) {
        out.qsa_pooled = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&out.qsa_pooled, src.qsa_pooled);
    }
    out.qsa_ratio = src.qsa_ratio;
    out.qsa_rows = qsaHistoryRows(src);
    out.ple_prev = src.ple_prev;
    out.ple_prev_valid = src.ple_prev_valid;
    return out;
}

pub fn ssmSnapshotDeinit(snap: *SSMCacheEntrySnapshot) void {
    _ = mlx.mlx_array_free(snap.conv_state);
    _ = mlx.mlx_array_free(snap.ssm_state);
    if (snap.aux_state.ctx != null) _ = mlx.mlx_array_free(snap.aux_state);
    if (snap.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(snap.qsa_pooled);
    snap.aux_state = .{ .ctx = null };
    snap.qsa_pooled = .{ .ctx = null };
}

pub fn ssmRestore(dst: *SSMCacheEntry, snap: *const SSMCacheEntrySnapshot) !void {
    _ = mlx.mlx_array_free(dst.conv_state);
    _ = mlx.mlx_array_free(dst.ssm_state);
    dst.conv_state = mlx.mlx_array_new();
    dst.ssm_state = mlx.mlx_array_new();
    dst.initialized = snap.initialized;
    // Mirror snapshot's per-field null guard — the snapshot may legitimately
    // have a null ssm_state (LFM2 gated_conv layers) or null conv_state.
    if (snap.conv_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&dst.conv_state, snap.conv_state));
    }
    if (snap.ssm_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&dst.ssm_state, snap.ssm_state));
    }
    // qwen4_exp aux state: REPLACED, never merged — the entry may hold the
    // previous request's key history.
    ssmFreeQsaState(dst);
    if (snap.aux_state.ctx != null) {
        dst.aux_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&dst.aux_state, snap.aux_state));
    }
    if (snap.qsa_pooled.ctx != null) {
        dst.qsa_pooled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&dst.qsa_pooled, snap.qsa_pooled));
    }
    dst.qsa_ratio = snap.qsa_ratio;
    dst.ple_prev = snap.ple_prev;
    dst.ple_prev_valid = snap.ple_prev_valid;
    dst.qsa_hist_rows = snap.qsa_rows;
    if (snap.aux_state.ctx != null) {
        const sh = mlx.getShape(dst.aux_state);
        dst.qsa_key_rows = if (sh.len >= 2) sh[1] else 0;
    }
    if (snap.qsa_pooled.ctx != null) {
        const ps = mlx.getShape(dst.qsa_pooled);
        dst.qsa_pooled_blocks = if (ps.len >= 2) ps[1] else 0;
    }
    if (dst.group) |g| {
        try ssmPatchGroupRow(
            mlx.gpuStream(),
            g,
            dst.group_row,
            dst.group_layer,
            dst.conv_state,
            dst.ssm_state,
            dst.aux_state,
            .{ .conv = dst.conv_state.ctx != null, .ssm = dst.ssm_state.ctx != null, .aux = dst.aux_state.ctx != null },
        );
    }
}

/// Free the transient PLD spec-decode capture buffers on an SSM entry
/// (`spec_state_seq` / `spec_conv_input`). Idempotent — safe to call on an
/// entry that never captured. Called at the end of every `nextPld` round and
/// in the cache teardown paths so the capture never outlives a round.
pub fn ssmFreeSpecCapture(entry: *SSMCacheEntry) void {
    if (entry.spec_state_seq.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_state_seq);
        entry.spec_state_seq = .{ .ctx = null };
    }
    if (entry.spec_conv_input.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_conv_input);
        entry.spec_conv_input = .{ .ctx = null };
    }
    inline for (.{ &entry.spec_state_in, &entry.spec_prework.k, &entry.spec_prework.v, &entry.spec_prework.gb }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a.*);
        a.* = .{ .ctx = null };
    };
    if (entry.spec_ple_input.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_ple_input);
        entry.spec_ple_input = .{ .ctx = null };
    }
    entry.spec_ple_len = 0;
}

fn ssmAxisView(s: mlx.mlx_stream, arr: mlx.mlx_array, axis: usize, i: usize) !mlx.mlx_array {
    const shape = mlx.getShape(arr);
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    for (0..shape.len) |d| stop[d] = shape[d];
    start[axis] = @intCast(i);
    stop[axis] = @intCast(i + 1);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &start, @intCast(shape.len), &stop, @intCast(shape.len), &strides, @intCast(shape.len), s));
    return out;
}

pub fn splitSpecCaptures(s: mlx.mlx_stream, m: *const SSMCacheEntry, e: *SSMCacheEntry, i: usize) !void {
    if (m.spec_state_seq.ctx != null) {
        const own = try ssmAxisView(s, m.spec_state_seq, 1, i);
        if (e.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(e.spec_state_seq);
        e.spec_state_seq = own;
    }
    if (m.spec_conv_input.ctx != null) {
        const own = try ssmAxisView(s, m.spec_conv_input, 0, i);
        if (e.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(e.spec_conv_input);
        e.spec_conv_input = own;
    }
    if (m.spec_ple_input.ctx != null) {
        const own = try ssmAxisView(s, m.spec_ple_input, 0, i);
        if (e.spec_ple_input.ctx != null) _ = mlx.mlx_array_free(e.spec_ple_input);
        e.spec_ple_input = own;
    }
}

/// PLD partial-accept rollback for a GatedDeltaNet layer, using the
/// per-position capture from the verify forward instead of a re-forward.
/// Sets `conv_state` + `ssm_state` to the state AFTER processing the first
/// `1 + accepted` verify tokens (t1 + accepted accepted drafts).
///
/// Numerically identical to a fresh forward over `[t1, draft[0..accepted-1]]`:
/// the GDN recurrence is sequential, so the state captured at position
/// `accepted` of a length-(1+m) verify run is the exact same float32→bf16
/// value a length-(1+accepted) run would store as its final state. Likewise
/// `conv_state` is the same windowed slice of the same conv input. So this
/// preserves PLD's byte-equivalence guarantee (pinned by
/// `tests/test_pld_equivalence.sh`).
///
/// No-op when the entry holds no capture (non-GDN hybrid layer); the caller
/// only takes the fast path when capture succeeded.
/// `verify_len` is the number of positions the verify forward ran (T). A
/// GatedDeltaNet layer records it implicitly in `spec_state_seq`, but a
/// CONV-ONLY layer (LFM2's gated conv — no recurrent ssm_state at all) has
/// only `spec_conv_input`, whose leading `kernel-1` rows cannot be told from
/// verify positions without it. Passing it explicitly is what lets one helper
/// roll back both shapes.
/// A draft tree's accepted path: the recurrent state after window row
/// `state_row` (every row's state is captured in tree mode) and the conv
/// state from the capture's conv-input rows `conv_rows` (0..2 the pre-verify
/// conv state, 3 + r window row r).
pub fn ssmCommitTreePath(entry: *SSMCacheEntry, path: []const i32, conv_rows: [3]i32, s: mlx.mlx_stream) !void {
    if (entry.spec_prework.k.ctx != null) {
        const sh = mlx.getShape(entry.spec_state_in); // [B, Hv, Dv, Dk]
        const g = gdn_decode.Geometry{ .hk = mlx.getShape(entry.spec_prework.k)[1], .hv = sh[1], .dk = sh[3], .dv = sh[2] };
        const c = (try gdn_decode.replay(g, entry.spec_state_in, entry.spec_prework, path, entry.spec_conv_input, conv_rows, s)) orelse return error.SpecTreeUnsupported;
        _ = mlx.mlx_array_free(entry.ssm_state);
        entry.ssm_state = c.ssm_state;
        _ = mlx.mlx_array_free(entry.conv_state);
        entry.conv_state = c.conv_state;
        return;
    } else if (entry.spec_state_seq.ctx != null) {
        const state_row: u32 = @intCast(path[path.len - 1]);
        const sh = mlx.getShape(entry.spec_state_seq); // [T, B, Hv, Dv, Dk]
        const r: c_int = @intCast(state_row);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, entry.spec_state_seq, &[_]c_int{ r, 0, 0, 0, 0 }, 5, &[_]c_int{ r + 1, sh[1], sh[2], sh[3], sh[4] }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
        var st = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&st, row, &[_]c_int{ sh[1], sh[2], sh[3], sh[4] }, 4, s));
        _ = mlx.mlx_array_free(entry.ssm_state);
        entry.ssm_state = st;
    }
    if (entry.spec_conv_input.ctx != null) {
        const idx = mlx.mlx_array_new_data(&conv_rows, &[_]c_int{3}, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);
        var conv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(&conv, entry.spec_conv_input, idx, 1, s));
        _ = mlx.mlx_array_free(entry.conv_state);
        entry.conv_state = conv;
    }
}

pub fn ssmRollbackFromCapture(entry: *SSMCacheEntry, accepted: u32, verify_len: u32, s: mlx.mlx_stream) !void {
    // qwen4_exp per-request state beside the GDN pair. PLE layer: the conv
    // window + token history slice at 1+accepted (same shape as conv_state
    // below). Attention layer: the QSA raw-key history is positional — keep
    // the rows up to the accepted verify prefix.
    //
    // The two arms are exclusive: `aux_state` holds the PLE window on the PLE layer and the
    // QSA key history on a full-attention layer. `ssmFreeQsaState` frees AND nulls.
    var patch_conv = false;
    var patch_ssm = false;
    var patch_aux = false;
    if (entry.spec_ple_input.ctx != null) {
        const ci_shape = mlx.getShape(entry.spec_ple_input); // [B, state_len+T, hc*H]
        const state_len = ci_shape[1] - @as(c_int, @intCast(verify_len));
        const cstart: c_int = @intCast(1 + accepted);
        const start = [_]c_int{ 0, cstart, 0 };
        const stop = [_]c_int{ ci_shape[0], cstart + state_len, ci_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, entry.spec_ple_input, &start, 3, &stop, 3, &strides, 3, s));
        const owned = try materializedOwnedCopy(s, view);
        ssmFreeQsaState(entry);
        entry.aux_state = owned;
        plePrevAfterAccept(entry, accepted, verify_len);
        patch_aux = true;
    } else if ((entry.aux_state.ctx != null or entry.qsa_pooled.ctx != null) and entry.conv_state.ctx == null) {
        const hist = qsaHistoryRows(entry);
        const keep: c_int = hist - @as(c_int, @intCast(verify_len)) + @as(c_int, @intCast(1 + accepted));
        if (!try qsaResliceToKeep(entry, keep, s)) {
            if (entry.aux_state.ctx == null) {
                entry.qsa_hist_rows = keep;
                try truncatePooled(&entry.qsa_pooled, keep, entry.qsa_ratio, s, true);
            } else {
                const ks = mlx.getShape(entry.aux_state);
                const drop_tail = hist - keep;
                // A slice past the ring's own start clamps to empty instead of failing, so
                // the unservable rollback is named here.
                if (drop_tail > ks[1]) return error.QsaRollbackBeyondRing;
                const new_held = ks[1] - drop_tail;
                const start = [_]c_int{ 0, 0, 0 };
                const stop = [_]c_int{ ks[0], new_held, ks[2] };
                const strides = [_]c_int{ 1, 1, 1 };
                var view = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(view);
                try mlx.check(mlx.mlx_slice(&view, entry.aux_state, &start, 3, &stop, 3, &strides, 3, s));
                const owned = try materializedOwnedCopy(s, view);
                const pooled = entry.qsa_pooled;
                entry.qsa_pooled = .{ .ctx = null };
                ssmFreeQsaState(entry);
                entry.qsa_pooled = pooled;
                entry.aux_state = owned;
                entry.qsa_key_rows = new_held;
                entry.qsa_hist_rows = keep;
                try truncatePooled(&entry.qsa_pooled, keep, entry.qsa_ratio, s, true);
            }
        }
    }
    const acc: c_int = @intCast(accepted);
    if (entry.spec_state_seq.ctx != null) {
        const seq_shape = mlx.getShape(entry.spec_state_seq); // [T, B, Hv, Dv, Dk]

        // ssm_state = spec_state_seq[accepted]  →  [B, Hv, Dv, Dk]
        {
            const start = [_]c_int{ acc, 0, 0, 0, 0 };
            const stop = [_]c_int{ acc + 1, seq_shape[1], seq_shape[2], seq_shape[3], seq_shape[4] };
            const strides = [_]c_int{ 1, 1, 1, 1, 1 };
            var sliced = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&sliced, entry.spec_state_seq, &start, 5, &stop, 5, &strides, 5, s));
            defer _ = mlx.mlx_array_free(sliced);
            const new_shape = [_]c_int{ seq_shape[1], seq_shape[2], seq_shape[3], seq_shape[4] };
            var reshaped = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&reshaped, sliced, &new_shape, 4, s));
            _ = mlx.mlx_array_free(entry.ssm_state);
            entry.ssm_state = reshaped;
            patch_ssm = true;
        }
    }

    // conv_state = spec_conv_input[:, (1+accepted) : (1+accepted)+(kernel-1), :]
    if (entry.spec_conv_input.ctx != null) {
        const ci_shape = mlx.getShape(entry.spec_conv_input); // [B, (k-1)+T, conv_dim]
        const t_len: c_int = @intCast(verify_len);
        const km1 = ci_shape[1] - t_len; // kernel - 1
        const cstart: c_int = @intCast(1 + accepted);
        const start = [_]c_int{ 0, cstart, 0 };
        const stop = [_]c_int{ ci_shape[0], cstart + km1, ci_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var new_conv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&new_conv, entry.spec_conv_input, &start, 3, &stop, 3, &strides, 3, s));
        _ = mlx.mlx_array_free(entry.conv_state);
        entry.conv_state = new_conv;
        patch_conv = true;
    }
    if (entry.group) |g| {
        try ssmPatchGroupRow(s, g, entry.group_row, entry.group_layer, entry.conv_state, entry.ssm_state, entry.aux_state, .{
            .conv = patch_conv,
            .ssm = patch_ssm,
            .aux = patch_aux,
        });
    }
}
