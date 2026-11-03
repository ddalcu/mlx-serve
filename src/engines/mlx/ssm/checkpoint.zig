//! engines/mlx/ssm/checkpoint.
//! Owns SSMCheckpoint, captureSsmCheckpoint, compact_conv_state_override, compact_conv_state_copy_count, cadence_conv_eval_count, compactConvStateEnabled, and 15 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const applyQsaHistoryAt = @import("../../../plugins/qwen4_exp/qsa/checkpoint.zig").applyQsaHistoryAt;
const attachQsaHistoryToLatest = @import("../../../plugins/qwen4_exp/qsa/checkpoint.zig").attachQsaHistoryToLatest;
const qsaHistoryRows = @import("../../../plugins/qwen4_exp/qsa/checkpoint.zig").qsaHistoryRows;
const ssmAuxIsQsaHistory = @import("../../../plugins/qwen4_exp/qsa/checkpoint.zig").ssmAuxIsQsaHistory;
const qsaLeftoverAt = @import("../../../plugins/qwen4_exp/qsa/history.zig").qsaLeftoverAt;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const KVCache = @import("../kv/cache.zig").KVCache;
const testing = @import("../kv/cache.zig").testing;
const SSMCacheEntry = @import("entry.zig").SSMCacheEntry;
const SSMCacheEntrySnapshot = @import("entry.zig").SSMCacheEntrySnapshot;
const ssmFreeSpecCapture = @import("entry.zig").ssmFreeSpecCapture;
const ssmRestore = @import("entry.zig").ssmRestore;
const ssmRollbackFromCapture = @import("entry.zig").ssmRollbackFromCapture;
const ssmSnapshot = @import("entry.zig").ssmSnapshot;
const ssmSnapshotDeinit = @import("entry.zig").ssmSnapshotDeinit;
const ssmLiveAux = @import("group.zig").ssmLiveAux;
const ssmMaterializeLive = @import("group.zig").ssmMaterializeLive;

/// Phase 1 (performance-plan): per-position SSM checkpoint covering ALL hybrid
/// layers in the model. Captures the SSM/conv state after the model has been
/// forwarded over the first `pos` tokens of some prompt. Used by the hot prefix
/// cache to restore mid-sequence SSM state on a multi-turn warm request, so we
/// don't have to cold-prefill the shared prefix every turn.
///
/// A single checkpoint = one `SSMCacheEntrySnapshot` per layer (matching the
/// shape of `ctx.ssm_entries`). Layers whose ssm/conv state is uninitialized
/// at snapshot time hold a null-handle snapshot — `ssmRestore` is null-safe.
///
/// Memory: per GatedDeltaNet layer, the Hv x Dv x Dk state in its own dtype
/// (`ModelConfig.ssmStateDtype`) plus the conv window. Bounded via
/// `HotPrefixCache.max_kv_bytes` (counts toward the same budget as KV).
pub const SSMCheckpoint = struct {
    /// 1-based KV position immediately AFTER forwarding `pos` tokens. The
    /// caller restores the slot's KVCache to exactly this position alongside
    /// the SSM state, so the model behaves as if it had only seen the first
    /// `pos` tokens of the prompt.
    pos: usize,
    /// Per-layer SSM snapshots — same length as `ctx.ssm_entries`. Non-SSM
    /// layers (plain attention) get a null-handle snapshot which is a no-op
    /// in `ssmRestore`.
    layers: []SSMCacheEntrySnapshot,

    pub fn deinit(self: *SSMCheckpoint, allocator: std.mem.Allocator) void {
        for (self.layers) |*l| ssmSnapshotDeinit(l);
        allocator.free(self.layers);
        self.layers = &[_]SSMCacheEntrySnapshot{};
        self.pos = 0;
    }
};

/// Snapshot of every entry in `ssm_entries` at the current point. Caller owns
/// the resulting buffer (free via `SSMCheckpoint.deinit`).
///
/// Unlike the per-decode-step `ssmSnapshot` (PLD rollback — transient, so a
/// cheap refcount-share is right), checkpoints OUTLIVE the request inside the
/// hot prefix cache. The live conv/ssm states are routinely shared-buffer
/// SLICES of much larger parents (the prefill chunk's conv input
/// [B,(k-1)+T,C], the GDN capture sequence), so sharing the handle silently
/// retained the whole parent — the ~3.4x "[hot-cache] resident" under-count
/// on hybrid archs. `materializedOwnedCopy` forces a fresh buffer; the copies
/// are evaluated here, in one batch, so the parents' lifetimes end with the
/// chunk's activation graph instead of with the cache entry.
pub fn captureSsmCheckpoint(
    allocator: std.mem.Allocator,
    ssm_entries: []const SSMCacheEntry,
    pos: usize,
    s: mlx.mlx_stream,
) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, ssm_entries.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (ssm_entries, 0..) |*src, i| {
        var out: SSMCacheEntrySnapshot = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = src.initialized,
        };
        errdefer ssmSnapshotDeinit(&out);
        // Per-field null guards mirror `ssmSnapshot` — either state may
        // legitimately be null (LFM2 gated_conv never sets ssm_state).
        const live_conv = if (src.group) |g| (if (g.live and src.group_layer < g.layers.len) g.layers[src.group_layer].conv_state else src.conv_state) else src.conv_state;
        const live_ssm = if (src.group) |g| (if (g.live and src.group_layer < g.layers.len) g.layers[src.group_layer].ssm_state else src.ssm_state) else src.ssm_state;
        const live_ple = ssmLiveAux(src);
        if (live_conv.ctx != null) {
            const c = try ssmMaterializeLive(s, src, live_conv);
            _ = mlx.mlx_array_free(out.conv_state);
            out.conv_state = c;
        }
        if (live_ssm.ctx != null) {
            const c = try ssmMaterializeLive(s, src, live_ssm);
            _ = mlx.mlx_array_free(out.ssm_state);
            out.ssm_state = c;
        }
        // PLE conv window (O(1), rides a GDN layer) is per-position state and
        // belongs in every checkpoint. QSA indexer history (aux && !conv) is
        // a prefix of one buffer — copying it at every stride is what OOM'd
        // a 400k prefill while auto-context advertised 700k–1M. One copy is
        // attached to the latest checkpoint after prefill.
        const grouped_ple = src.group != null and src.group.?.live and src.group_layer < src.group.?.layers.len and src.group.?.layers[src.group_layer].aux_state.ctx != null;
        if (grouped_ple or (src.aux_state.ctx != null and !ssmAuxIsQsaHistory(src))) {
            out.aux_state = try ssmMaterializeLive(s, src, live_ple);
        }
        out.qsa_ratio = src.qsa_ratio;
        if (!grouped_ple and ssmAuxIsQsaHistory(src)) {
            out.qsa_rows = @intCast(pos);
            if (src.aux_state.ctx != null) {
                out.aux_state = try qsaLeftoverAt(src.aux_state, qsaHistoryRows(src), @intCast(pos), src.qsa_ratio, s);
            }
        }
        out.ple_prev = src.ple_prev;
        out.ple_prev_valid = src.ple_prev_valid;
        layers[i] = out;
        built = i + 1;
    }
    // Materialize all copies in one batch so the parent buffers can be
    // released with the prefill chunk's graph. Without this eval the lazy
    // copy node itself keeps the parent alive.
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var count: usize = 0;
        for (layers) |*l| {
            inline for (.{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled }) |arr| {
                if (arr.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(vec, arr);
                    count += 1;
                }
            }
        }
        if (count > 0) _ = mlx.mlx_eval(vec);
    }
    return .{ .pos = pos, .layers = layers };
}

pub var cadence_conv_eval_count: u64 = 0;

pub const compactConvStateEnabled = @import("../../../sdk/blocks/conv/depthwise.zig").compactConvStateEnabled;

/// A second owner for `cp`'s state: fresh handles and a fresh layer slice,
/// refcount-SHARING every underlying buffer (`mlx_array_set`, the same idiom
/// `ssmSnapshot` and `KVCache.snapshot` use). Both checkpoints then free
/// independently and the DATA outlives whichever dies first.
///
/// This is what makes checkpoint inheritance cheap enough to be unconditional:
/// a commit built on a restored prefix carries the donor entry's checkpoints
/// without re-copying multi-GB of GDN state, and evicting the donor later
/// frees nothing the inheritor still needs. `captureSsmCheckpoint` must
/// MATERIALIZE (its sources are slices of a live prefill graph); a checkpoint
/// is already materialized, so sharing it aliases nothing transient.
pub fn shareSsmCheckpoint(allocator: std.mem.Allocator, cp: *const SSMCheckpoint) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, cp.layers.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (cp.layers, 0..) |*src, i| {
        var out: SSMCacheEntrySnapshot = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = src.initialized,
        };
        // Per-field null guards: mlx_array_set on a null source aborts the
        // process via mlx-c's default error handler (see `ssmSnapshot`).
        if (src.conv_state.ctx != null) _ = mlx.mlx_array_set(&out.conv_state, src.conv_state);
        if (src.ssm_state.ctx != null) _ = mlx.mlx_array_set(&out.ssm_state, src.ssm_state);
        if (src.aux_state.ctx != null) {
            out.aux_state = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out.aux_state, src.aux_state);
        }
        if (src.qsa_pooled.ctx != null) {
            out.qsa_pooled = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out.qsa_pooled, src.qsa_pooled);
        }
        out.qsa_ratio = src.qsa_ratio;
        out.qsa_rows = src.qsa_rows;
        out.ple_prev = src.ple_prev;
        out.ple_prev_valid = src.ple_prev_valid;
        layers[i] = out;
        built = i + 1;
    }
    return .{ .pos = cp.pos, .layers = layers };
}

/// Restore every layer of `ssm_entries` from `cp`. Mirrors the per-layer
/// `ssmRestore` pattern — null-safe on either side.
pub fn restoreSsmCheckpoint(
    ssm_entries: []SSMCacheEntry,
    cp: *const SSMCheckpoint,
) !void {
    if (ssm_entries.len != cp.layers.len) return error.SsmCheckpointLayerMismatch;
    for (ssm_entries, cp.layers) |*dst, *src| {
        try ssmRestore(dst, src);
    }
}

/// Total bytes held by an SSM checkpoint (sum of conv_state + ssm_state across
/// all layers). Used for hot-cache memory budgeting alongside `KVCacheSnapshot`
/// bytes.
pub fn ssmCheckpointBytes(cp: *const SSMCheckpoint) u64 {
    var total: u64 = 0;
    for (cp.layers) |l| {
        inline for (.{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled }) |arr| {
            if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
        }
    }
    return total;
}

/// Which retention policy a checkpoint list is thinned with. The span-preserving thin moves
/// where a warm turn restores from and was measured on qwen4_exp only, so the policy is chosen
/// by `longCtxGated()` at each site and the ungated value names that site's previous
/// behaviour: `.oldest` (prefill capture, disk tier), `.min_span` (hot-cache merge/shed),
/// `.min_span_recency` (`.min_span` plus a dense newest quarter; qwen4_exp).
pub const ThinPolicy = enum { oldest, min_span, min_span_recency };

/// The checkpoint index to drop when a retention list is over its cap: the interior position
/// whose removal widens the coverage gap least; index 0 and the last are always kept. One
/// selection for every retention site (#330: end-anchored survivors left no affordable trim point).
pub fn spanPreservingDropIndex(
    comptime T: type,
    items: []const T,
    comptime posOf: fn (*const T) usize,
    policy: ThinPolicy,
    /// An index the caller knows a later turn must restore from (a media boundary): never
    /// selected. When it is the only candidate the oldest goes instead — that one sits below
    /// it, so the protected position survives either way.
    protect: ?usize,
    /// How many LOW positions are anchored as well (#794: thinning must not
    /// migrate the grid floor upward). 0 = the plain policy; a caller that
    /// wants the legacy drop-oldest behaviour also passes 0.
    low_anchors: usize,
) usize {
    std.debug.assert(items.len > 0);
    const last = items.len - 1;
    // The cap outranks the anchors: the newest position is never a drop candidate,
    // so if every remaining index is anchored the lowest anchor gives way.
    if (items.len <= low_anchors + 1) return @min(low_anchors, last);
    if (policy == .oldest) return @min(low_anchors, last);
    if (items.len < 3) return @min(low_anchors, last);
    // Recency bias: the newest quarter stays at capture density (a warm turn that edits near
    // the end restores from there); everything below it is thinned span-preservingly.
    const scan_end = if (policy == .min_span_recency and items.len >= RECENCY_DENSE_MIN)
        items.len - items.len / 4
    else
        items.len;
    // Index 0 is never a candidate (the span is measured across k-1), so the scan
    // starts one past the last anchor — and at 1 when nothing is anchored, which
    // is the position the plain policies have always started from.
    const scan_start = @max(low_anchors, 1);
    var best_at: usize = scan_start;
    var best_span: usize = std.math.maxInt(usize);
    var k: usize = scan_start;
    while (k + 1 < items.len and k < scan_end) : (k += 1) {
        if (protect) |p| if (k == p) continue;
        const span = posOf(&items[k + 1]) -| posOf(&items[k - 1]);
        if (span < best_span) {
            best_span = span;
            best_at = k;
        }
    }
    // Nothing scannable (recency-protected or anchored): the oldest goes, which
    // sits below every anchor, so the anchors still survive.
    if (best_span == std.math.maxInt(usize)) return 0;
    return best_at;
}

/// `spanPreservingDropIndex` over a bare ascending position list, with the first
/// `low_anchors` positions protected from thinning.
pub fn positionDropIndexLowAnchored(positions: []const u32, policy: ThinPolicy, low_anchors: usize) usize {
    return spanPreservingDropIndex(u32, positions, u32PosOf, policy, null, low_anchors);
}

/// Shortest list that reserves a dense newest quarter.
const RECENCY_DENSE_MIN: usize = 8;

fn checkpointPosOf(cp: *const SSMCheckpoint) usize {
    return cp.pos;
}

fn u32PosOf(p: *const u32) usize {
    return p.*;
}

fn usizePosOf(p: *const usize) usize {
    return p.*;
}

/// `spanPreservingDropIndex` over a checkpoint list.
pub fn ssmCheckpointDropIndex(cps: []const SSMCheckpoint, policy: ThinPolicy, protect: ?usize) usize {
    return spanPreservingDropIndex(SSMCheckpoint, cps, checkpointPosOf, policy, protect, 0);
}

/// `ssmCheckpointDropIndex` with the first `low_anchors` positions protected
/// from thinning: generation-side retention must let the dense low grid anchors
/// (#794) survive the per-prefill cap, or the persist-side `ssm_low_anchors`
/// has nothing left to protect. 0 = the plain policy, byte-identical to before.
pub fn ssmCheckpointDropIndexLowAnchored(
    cps: []const SSMCheckpoint,
    policy: ThinPolicy,
    protect: ?usize,
    low_anchors: usize,
) usize {
    return spanPreservingDropIndex(SSMCheckpoint, cps, checkpointPosOf, policy, protect, low_anchors);
}

/// `spanPreservingDropIndex` over a bare ascending position list.
pub fn positionDropIndex(positions: []const u32, policy: ThinPolicy) usize {
    return spanPreservingDropIndex(u32, positions, u32PosOf, policy, null, 0);
}

pub fn positionDropIndexUsize(positions: []const usize, policy: ThinPolicy) usize {
    return spanPreservingDropIndex(usize, positions, usizePosOf, policy, null, 0);
}

// ── tests ──

test "ssmRollbackFromCapture: padded verify_len keeps PLE tokens inside the real prefix" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var e: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        ssmFreeSpecCapture(&e);
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(&e);
    }
    const state_len: c_int = 2;
    const m: u32 = 2;
    const verify_len: c_int = 4;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const n: f64 = @floatFromInt((state_len + verify_len) * 3);
    try mlx.check(mlx.mlx_arange(&flat, 0.0, n, 1.0, .float32, s));
    const shape = [_]c_int{ 1, state_len + verify_len, 3 };
    try mlx.check(mlx.mlx_reshape(&e.spec_ple_input, flat, &shape, 3, s));
    e.spec_ple_tokens = .{ 9, 8, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    e.spec_ple_len = @intCast(state_len + verify_len);
    try ssmRollbackFromCapture(&e, m, @intCast(verify_len), s);
    try t.expect(e.aux_state.ctx != null);
    try t.expectEqual(state_len, mlx.getShape(e.aux_state)[1]);
    try t.expectEqual(@as(u32, 2), e.ple_prev[0]);
    try t.expectEqual(@as(u32, 3), e.ple_prev[1]);
    try t.expect(e.ple_prev_valid);
}

pub fn qsaHistoryBytesHeld(cps: []const SSMCheckpoint) u64 {
    var total: u64 = 0;
    for (cps) |cp| {
        for (cp.layers) |l| {
            inline for (.{ l.aux_state, l.qsa_pooled }) |arr| {
                if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
            }
        }
    }
    return total;
}

test "SSMCacheEntry snapshot/restore round-trip preserves arrays" {
    const s = mlx.gpuStream();
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }

    // Populate with arbitrary state.
    const conv_shape = [_]c_int{ 1, 3, 4 };
    _ = mlx.mlx_array_free(entry.conv_state);
    entry.conv_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.conv_state, &conv_shape, 3, .float32, s));

    const ssm_shape = [_]c_int{ 1, 2, 8, 4 };
    _ = mlx.mlx_array_free(entry.ssm_state);
    entry.ssm_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.ssm_state, &ssm_shape, 4, .float32, s));
    entry.initialized = true;

    var snap = ssmSnapshot(&entry);
    defer ssmSnapshotDeinit(&snap);

    // Mutate: replace ssm_state with a different shape.
    _ = mlx.mlx_array_free(entry.ssm_state);
    entry.ssm_state = mlx.mlx_array_new();
    const new_shape = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_zeros(&entry.ssm_state, &new_shape, 4, .float32, s));

    try ssmRestore(&entry, &snap);
    try testing.expect(entry.initialized);
    const restored_shape = mlx.getShape(entry.ssm_state);
    try testing.expectEqual(@as(c_int, 1), restored_shape[0]);
    try testing.expectEqual(@as(c_int, 2), restored_shape[1]);
    try testing.expectEqual(@as(c_int, 8), restored_shape[2]);
    try testing.expectEqual(@as(c_int, 4), restored_shape[3]);
}

test "SSMCacheEntry snapshot/restore handles null ssm_state (LFM2 gated_conv)" {
    // LFM2 `gatedConv` populates `conv_state` but never `ssm_state` — the
    // gated-convolution layer doesn't have a recurrence state, only a
    // convolution-window cache. The snapshot/restore code must NOT crash on
    // this shape (`initialized=true`, `conv_state` non-null, `ssm_state.ctx`
    // null). This was the root cause of the Workstream D PLD-on-hybrid bug.
    const s = mlx.gpuStream();
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(), // stays null — no LFM2 layer ever touches it
        .initialized = false,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }

    // Simulate `conv1dWithCache` having run once: conv_state populated,
    // initialized=true, ssm_state still null (its ctx is null).
    const conv_shape = [_]c_int{ 1, 3, 4 };
    _ = mlx.mlx_array_free(entry.conv_state);
    entry.conv_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.conv_state, &conv_shape, 3, .float32, s));
    entry.initialized = true;
    try testing.expect(entry.ssm_state.ctx == null);

    // Snapshot must succeed without dereferencing the null ssm_state.
    var snap = ssmSnapshot(&entry);
    defer ssmSnapshotDeinit(&snap);
    try testing.expect(snap.initialized);
    try testing.expect(snap.conv_state.ctx != null);
    try testing.expect(snap.ssm_state.ctx == null);

    // Mutate conv_state in `entry`, then restore — restore must rebind
    // conv_state without crashing on the still-null ssm_state.
    _ = mlx.mlx_array_free(entry.conv_state);
    entry.conv_state = mlx.mlx_array_new();
    const mutated_shape = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_zeros(&entry.conv_state, &mutated_shape, 3, .float32, s));

    try ssmRestore(&entry, &snap);
    try testing.expect(entry.initialized);
    try testing.expect(entry.ssm_state.ctx == null); // still null after restore
    const restored = mlx.getShape(entry.conv_state);
    try testing.expectEqual(@as(c_int, 1), restored[0]);
    try testing.expectEqual(@as(c_int, 3), restored[1]);
    try testing.expectEqual(@as(c_int, 4), restored[2]);
}

test "captureSsmCheckpoint materializes state copies (parent-buffer retention class)" {
    // Hot-cache SSM checkpoints used to refcount-share the live conv/ssm
    // state handles. Those are routinely SLICES of much larger parent buffers
    // (e.g. the prefill chunk's conv input [B,(k-1)+T,C]), so a committed
    // entry silently retained the whole parent — the measured ~3.4x
    // "[hot-cache] resident" under-count on hybrid archs. Capture must
    // materialize an OWNED copy: the checkpoint's data pointer may not alias
    // the parent's buffer, and ssmCheckpointBytes must equal true retention.
    const s = mlx.gpuStream();

    // Parent buffer: [1, 1024, 8] f32, 32 KB.
    var parent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(parent);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, 8192.0, 1.0, .float32, s));
        const shape = [_]c_int{ 1, 1024, 8 };
        try mlx.check(mlx.mlx_reshape(&parent, flat, &shape, 3, s));
        _ = mlx.mlx_array_eval(parent);
    }

    // conv_state := parent[:, 1021:1024, :] — after eval this is a
    // shared-buffer view into `parent` (MLX slice keeps the parent alive).
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = true,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }
    {
        const st = [_]c_int{ 0, 1021, 0 };
        const sp = [_]c_int{ 1, 1024, 8 };
        const sd = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&entry.conv_state, parent, &st, 3, &sp, 3, &sd, 3, s));
        _ = mlx.mlx_array_eval(entry.conv_state);
    }

    var entries = [_]SSMCacheEntry{entry};
    var cp = try captureSsmCheckpoint(testing.allocator, &entries, 3, s);
    defer cp.deinit(testing.allocator);

    // The checkpoint's conv_state must live in its own buffer, outside the
    // parent's data range.
    _ = mlx.mlx_array_eval(cp.layers[0].conv_state);
    const parent_ptr = mlx.mlx_array_data_float32(parent).?;
    const parent_bytes = @as(u64, mlx.mlx_array_size(parent)) * @as(u64, mlx.mlx_array_itemsize(parent));
    const cp_ptr = mlx.mlx_array_data_float32(cp.layers[0].conv_state).?;
    const p0 = @intFromPtr(parent_ptr);
    const p1 = p0 + parent_bytes;
    const c0 = @intFromPtr(cp_ptr);
    try testing.expect(c0 < p0 or c0 >= p1);

    // Values must survive the copy: parent row 1021 starts at 1021*8.
    try testing.expectEqual(@as(f32, 1021.0 * 8.0), cp_ptr[0]);
    try testing.expectEqual(@as(f32, 1021.0 * 8.0 + 23.0), cp_ptr[23]);

    // Accounting matches true retention: 3*8 f32 elements, nothing more.
    try testing.expectEqual(@as(u64, 3 * 8 * 4), ssmCheckpointBytes(&cp));

    // Null ssm_state stays null (LFM2 gated_conv shape).
    try testing.expect(cp.layers[0].ssm_state.ctx == null);
}

test "ssm checkpoint carries the qwen4_exp aux state (key history, pooled keys, PLE tokens) through capture and restore" {
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA0C5);
    const rnd = prng.random();
    var src_e = [_]SSMCacheEntry{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 3, 64 }, s), .ssm_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 8, 8 }, s), .initialized = true },
    };
    // Layer 0 = a full-attention layer holding the QSA key history + pooled keys
    // (no conv/ssm state); layer 1 = a GDN layer that also carries the PLE
    // window in aux_state and the token history.
    src_e[0].aux_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 37, 16 }, s);
    src_e[0].qsa_pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, 9, 16 }, s);
    src_e[0].qsa_ratio = 4;
    src_e[1].aux_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 6, 32 }, s);
    src_e[1].ple_prev = .{ 11, 22, 0, 0, 0, 0, 0, 0 };
    src_e[1].ple_prev_valid = true;
    defer for (&src_e) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(e);
    };

    var cps = [_]SSMCheckpoint{try captureSsmCheckpoint(a, &src_e, 37, s)};
    defer cps[0].deinit(a);
    try std.testing.expectEqual(@as(c_int, 1), mlx.getShape(cps[0].layers[0].aux_state)[1]);
    try std.testing.expect(cps[0].layers[0].qsa_pooled.ctx == null);
    try std.testing.expect(cps[0].layers[1].ple_prev_valid and cps[0].layers[1].ple_prev[1] == 22);
    try std.testing.expect(cps[0].layers[1].aux_state.ctx != null);
    try attachQsaHistoryToLatest(&cps, &src_e, s);
    const cp = &cps[0];
    try std.testing.expectEqual(@as(c_int, 1), mlx.getShape(cp.layers[0].aux_state)[1]);
    try std.testing.expect(cp.layers[0].qsa_pooled.ctx != null);
    const expect_bytes: u64 = (1 * 16 + 9 * 16 + 6 * 32 + 3 * 64 + 2 * 8 * 8) * 2;
    try std.testing.expectEqual(expect_bytes, ssmCheckpointBytes(cp));

    // A restore target that holds a LATER request's state gets it replaced.
    var dst = [_]SSMCacheEntry{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    dst[0].aux_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 90, 16 }, s);
    dst[1].ple_prev = .{ 7, 7, 0, 0, 0, 0, 0, 0 };
    dst[1].ple_prev_valid = true;
    defer for (&dst) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(e);
    };
    try restoreSsmCheckpoint(&dst, cp);
    try applyQsaHistoryAt(&dst, cp, 37, s, false);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dst[0].aux_state, cp.layers[0].aux_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dst[0].qsa_pooled, src_e[0].qsa_pooled, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dst[1].aux_state, src_e[1].aux_state, s));
    try std.testing.expect(dst[1].qsa_pooled.ctx == null);
    try std.testing.expectEqual(@as(u32, 11), dst[1].ple_prev[0]);
    try std.testing.expect(dst[1].ple_prev_valid and !dst[0].ple_prev_valid);
    // The checkpoint owns its buffers: freeing the source leaves it intact.
    _ = mlx.mlx_array_free(src_e[0].aux_state);
    src_e[0].aux_state = .{ .ctx = null };
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dst[0].aux_state, cp.layers[0].aux_state, s));
}

test "ssm retention: the span-preserving thin keeps both ends and spreads the survivors" {
    // Drop-oldest survivors covered only the last `max * stride` tokens (#330).
    const t = std.testing;
    var positions: [94]usize = undefined;
    for (&positions, 0..) |*p, i| p.* = (i + 1) * 4096;
    var n: usize = positions.len;
    while (n > 16) {
        const drop = positionDropIndexUsize(positions[0..n], .min_span_recency);
        try t.expect(drop != 0);
        try t.expect(drop + 1 != n);
        var k = drop;
        while (k + 1 < n) : (k += 1) positions[k] = positions[k + 1];
        n -= 1;
    }
    try t.expectEqual(@as(usize, 16), n);
    try t.expectEqual(@as(usize, 4096), positions[0]);
    try t.expectEqual(@as(usize, 94 * 4096), positions[n - 1]);
    // Spread, but not evenly: the newest quarter stays at capture density.
    const ideal = (positions[n - 1] - positions[0]) / (n - 1);
    var i: usize = 1;
    while (i < n) : (i += 1) try t.expect(positions[i] - positions[i - 1] <= 2 * ideal);
    try t.expectEqual(@as(usize, 4096), positions[n - 1] - positions[n - 2]);
    try t.expectEqual(@as(usize, 4096), positions[n - 2] - positions[n - 3]);
    try t.expect(positions[1] - positions[0] > 4 * 4096);
    try t.expectEqual(@as(usize, 0), positionDropIndexUsize(positions[0..2], .min_span_recency));
    try t.expectEqual(@as(usize, 0), positionDropIndexUsize(positions[0..1], .min_span_recency));
    const clustered = [_]u32{ 0, 1000, 1010, 2000 };
    try t.expectEqual(@as(usize, 2), positionDropIndex(&clustered, .min_span_recency));

    // Every other arch keeps the previous policy of its own site: `.oldest` (prefill capture,
    // disk tier) and `.min_span` (hot-cache sites, no recency quarter).
    try t.expectEqual(@as(usize, 0), positionDropIndex(&clustered, .oldest));
    try t.expectEqual(@as(usize, 0), positionDropIndexUsize(positions[0..n], .oldest));
    {
        // 12 entries; the tightest pair is inside the newest quarter (starts at 9).
        var pos12: [12]usize = .{ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9010, 10_000 };
        try t.expectEqual(@as(usize, 10), positionDropIndexUsize(&pos12, .min_span));
        try t.expect(positionDropIndexUsize(&pos12, .min_span_recency) < 9);
        try t.expectEqual(
            positionDropIndexUsize(pos12[0..7], .min_span),
            positionDropIndexUsize(pos12[0..7], .min_span_recency),
        );
    }
}

test "the warm-credit capacity gate survives a hybrid trunk's uninitialized layers (W-1)" {
    const t = std.testing;
    // On a GDN trunk only the attention layers reach `update` (12 of 48 on qwen4_exp); a veto
    // on the first uninitialized entry made the warm credit unreachable on the arch it was for.
    const Fold = KVCache.CapacityFold;
    const layers: usize = 48;
    const interval: usize = 4; // full_attention_interval — layer 3, 7, 11, …
    const cap: usize = 787_456;

    var fold = Fold{};
    var attention_layers: usize = 0;
    for (0..layers) |i| {
        const is_attention = (i % interval) == interval - 1;
        if (is_attention) attention_layers += 1;
        fold.add(if (is_attention) cap else null);
    }
    try t.expectEqual(@as(usize, 12), attention_layers);
    try t.expectEqual(cap, fold.result());

    // The minimum still governs among the layers that hold a buffer.
    var ragged = Fold{};
    ragged.add(cap);
    ragged.add(null);
    ragged.add(cap - 4096);
    ragged.add(null);
    try t.expectEqual(cap - 4096, ragged.result());

    var cold = Fold{};
    for (0..layers) |_| cold.add(null);
    try t.expectEqual(@as(usize, 0), cold.result());
    const empty = Fold{};
    try t.expectEqual(@as(usize, 0), empty.result());

    var dense = Fold{};
    for (0..32) |_| dense.add(cap);
    try t.expectEqual(cap, dense.result());
}
