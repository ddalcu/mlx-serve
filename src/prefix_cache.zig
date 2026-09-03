//! Plan 03 — hot prefix cache (Phase 1).
//!
//! Replaces the legacy single-slot `cached_prompt_ids` with a small bounded
//! LRU keyed by `(prompt_ids ++ generated_ids, has_tools)`. Each entry owns a
//! `KVCacheSnapshot` of the live cache at the moment the request finished —
//! refcount-shared handles point at the GPU buffers that filled positions
//! 0..len. On a new request we longest-prefix match across entries, restore
//! the best one back into `xfm.cache`, and let the existing
//! truncate-then-prefill path handle the diverged tail.
//!
//! Hybrid SSM/conv architectures (qwen3_5/qwen3_5_moe/qwen3_next/nemotron_h/
//! lfm2) are excluded in v1: their recurrent state can't be rolled back, so
//! prefix reuse must reset on any divergence anyway. Plan 03's spec calls
//! these "hot tier only" — meaning we keep the single-slot path for them.
//! `HotPrefixCache.shouldUse(config)` returns false for those archs.

const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const model_mod = @import("model.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

const KVCache = transformer_mod.KVCache;
const KVCacheSnapshot = transformer_mod.KVCacheSnapshot;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const restoreSsmCheckpoint = transformer_mod.restoreSsmCheckpoint;
const applyQsaHistoryAt = transformer_mod.applyQsaHistoryAt;
const checkpointHasQsaHistory = transformer_mod.checkpointHasQsaHistory;
const sliceQsaHistoryOntoCheckpoint = transformer_mod.sliceQsaHistoryOntoCheckpoint;
const keepOnlyLatestQsaHistory = transformer_mod.keepOnlyLatestQsaHistory;
const entriesHaveQsaHistory = transformer_mod.entriesHaveQsaHistory;
const ssmCheckpointBytes = transformer_mod.ssmCheckpointBytes;

/// Minimum forwarded-prefix length for committing a CANCELLED prefill
/// (client disconnect mid-prefill). Below this an entry is LRU pollution —
/// chat-template prologues (Gemma=12, Qwen=8, Llama=4 tokens) are identical
/// across every request and "reusable" only in a worthless sense. Same
/// rationale as the llama session pool's `min_prefix_to_claim`, applied at
/// commit time instead of claim time.
pub const MIN_CANCELLED_COMMIT_TOKENS: usize = 256;

/// Result of a cache lookup. Tells the caller how many tokens of `prompt_ids`
/// are already in the live cache after a successful restore — the caller
/// then prefills only the trailing diverged tokens (`prompt_ids[matched..]`).
pub const LookupResult = struct {
    /// Tokens already in the live cache. Caller prefills `prompt_ids[matched..]`.
    matched: usize,
    /// Did the restore land on an entry whose tokens span the FULL new prompt?
    /// Then identical-re-issue logic kicks in (truncate to len-1 and re-forward
    /// the last token), matching the existing reuseKVCache behavior.
    full_match: bool,
    /// Non-null iff a DFlash assistant context was restored into the caller's
    /// target: the absolute trunk position its index 0 represents, with
    /// `base + cache.step == matched` on return. Both tiers serve it (the
    /// SSD tier persists the snapshot in the v4 spec sidecar). Null on EVERY
    /// other path (no target, no payload, miss) and the target is untouched —
    /// the caller then starts the assistant blind at `matched`.
    dflash_base: ?usize = null,
    /// Same contract for the MTP head's committed-history cache: the head's
    /// history is built from trunk hiddens, and a restore forwards NOTHING —
    /// without this every reused prefix drafts against an empty history
    /// (measured on Qwen3.6-27B echo: ~70 → ~38 tok/s on warm repeats).
    mtp_base: ?usize = null,
};

const Entry = struct {
    /// `prompt_ids ++ generated_ids` from the request that produced this snapshot.
    /// Owned by the entry; freed on eviction.
    tokens: []u32,
    /// Whether the request had tools enabled (different chat template, can't
    /// share cache across).
    has_tools: bool,
    /// Hash of the request's media pixels (0 = text only). Image placeholder
    /// tokens are identical across images, so the KV under them is keyed on
    /// the pixels. Non-zero entries stay in RAM (never spilled to the SSD tier).
    vision_key: u64 = 0,
    /// Position of the first dynamic media placeholder in `tokens`. Prefix
    /// state strictly before this boundary is independent of the media pixels
    /// and can be shared across different `vision_key` values.
    media_start: ?usize = null,
    /// Snapshot of the live KVCache at end of generation. Owns refcount-shared
    /// handles to the GPU buffers backing positions 0..tokens.len.
    snapshot: KVCacheSnapshot,
    /// Monotonic counter for LRU. Higher = more recent.
    last_used: u64,
    /// Wave 1.A: full KV-quant config active when this entry was committed.
    /// A new request whose `KVQuantConfig` differs in any field cannot
    /// restore from this entry — the underlying buffer layout (dense bf16 vs
    /// packed uint32 triples; 4-bit vs 8-bit packing) differs, and
    /// dequantization would have happened at commit time anyway. Filter at
    /// lookup so per-request `kv_quant` overrides never produce a hit
    /// against an entry that was committed under another config.
    ///
    /// Storing the full `KVQuantConfig` (not just `Scheme`) is what
    /// distinguishes `affine 4` from `affine 8` and a future TurboQuant
    /// `group_size` change — without that, a 4-bit entry would alias to an
    /// 8-bit slot's findBestMatch lookup and crash SDPA with a packed-shape
    /// mismatch on restore. Repro: `tests/test_kv_quant_per_request.sh`.
    quant_config: kv_quant.KVQuantConfig,
    /// KV-resident bytes for this entry, computed at commit time (Wave 1.B).
    /// Used for `--prefix-cache-mem` memory-budget enforcement; sum across
    /// all entries == `current_kv_bytes`.
    kv_bytes: u64,
    /// Phase 1 (perf-plan): SSM/conv state snapshots taken at stride-aligned
    /// positions during prefill. Sorted by `pos` ascending; the highest `pos`
    /// is at most `tokens.len`. Null for plain-attention archs. The hot
    /// cache restore picks the largest `pos ≤ matched` and rewinds both KV
    /// and SSM to it.
    ssm_checkpoints: ?[]SSMCheckpoint = null,
    /// Bytes resident in `ssm_checkpoints` (sum across all checkpoints and
    /// layers). Folded into `kv_bytes` for the byte-budget accounting so the
    /// memory cap covers both KV and SSM state.
    ssm_bytes: u64 = 0,
    /// DFlash assistant context for this prefix (dflash.zig). The assistant's
    /// K/V is built from trunk hiddens at `target_layer_ids`, and a restore
    /// forwards NOTHING — so without this the assistant starts every reused
    /// turn blind and drafts against an empty context. Optional in both
    /// directions: an entry committed by a non-dflash request has none, and a
    /// request that finds none simply starts blind (the state is DRAFT-side —
    /// a missing or stale context costs acceptance, never a token).
    dflash: ?DflashSnap = null,
    /// Bytes resident in `dflash`, folded into `kv_bytes` like `ssm_bytes`.
    dflash_bytes: u64 = 0,
    /// MTP committed-history cache for this prefix (the head's own dense
    /// KVCache, mtp.zig). Same lifecycle and contract as `dflash`: DRAFT-side
    /// state, best-effort in both directions — a missing or declined snap
    /// starts the history blind, which costs acceptance, never a token. The
    /// committer must snapshot ONLY committed history (no speculative draft
    /// tail — Generator.mtpCommittedHistoryLen is the boundary).
    mtp: ?DflashSnap = null,
    /// Bytes resident in `mtp`, folded into `kv_bytes` like `ssm_bytes`.
    mtp_bytes: u64 = 0,
};

/// A committed speculative-side cache: the snapshot plus the absolute trunk
/// position its index 0 represents (nonzero when the committing request was
/// itself a cache hit). Shared by the DFlash assistant context and the MTP
/// committed-history cache — identical semantics, two Entry fields.
pub const DflashSnap = struct {
    snapshot: KVCacheSnapshot,
    base_pos: usize,

    pub fn deinit(self: *DflashSnap) void {
        self.snapshot.deinit();
    }
};

/// What `commitWithState` reads to build a `DflashSnap`.
pub const DflashCommit = struct { cache: *const KVCache, base_pos: usize };

/// Where `lookupAndRestore` puts a restored assistant context. `base_pos` is
/// written on every path so the caller can build `DflashCtx` from it.
pub const DflashTarget = struct { cache: *KVCache, base_pos: *usize };

pub const HotPrefixCache = struct {
    entries: std.ArrayList(Entry),
    max_entries: u32,
    /// Wave 1.B: total KV bytes the cache is allowed to keep resident across
    /// all entries. 0 disables the byte budget (count cap still applies).
    /// Enforced on `commit`: evict LRU entries (in addition to the count
    /// cap) until `current_kv_bytes + new_entry_bytes <= max_kv_bytes`.
    max_kv_bytes: u64,
    /// Cap on SSM checkpoints kept per entry, mirroring `generate.zig`'s
    /// `ssm_checkpoint_max`. That one bounds a SINGLE prefill; this one bounds
    /// the replace path's merge, which concatenates the previous entry's
    /// checkpoints with this turn's. Without it an entry extended in place —
    /// every turn of an agent conversation — gains one checkpoint per turn for
    /// the life of the session. 0 = unlimited.
    ssm_checkpoint_max: u32 = 0,
    /// Running total of `kv_bytes` across all live entries. Updated on
    /// commit/evict/invalidate.
    current_kv_bytes: u64,
    allocator: std.mem.Allocator,
    counter: u64 = 0,
    /// Set to true once we've called `xfm.resetCache()` at least once after
    /// init. The first commit on a fresh cache must seed an empty entry so
    /// future restores have something to land on.
    initialized: bool = false,
    /// SSD tier (kv_disk_cache.zig). Attached by the scheduler at model load
    /// when `--prefix-cache-disk` is non-zero and the arch is pure-attention.
    /// Lookup falls back to it when it beats the RAM match; commits mark
    /// `disk_dirty` and the scheduler flushes AFTER the response finishes so
    /// the client never waits on the SSD write.
    disk: ?kv_disk_cache.DiskTier = null,
    /// A commit landed since the last `flushPendingDisk`.
    disk_dirty: bool = false,
    /// The arch keeps a QSA indexer history beside its SSM state (qwen4_exp).
    /// A restore that leaves the live entries without it cannot prefill —
    /// `qsaMaskFromQk` errors on every turn on that prefix — so it is a MISS.
    qsa_history_required: bool = false,

    pub fn init(allocator: std.mem.Allocator, max_entries: u32) HotPrefixCache {
        return initWithMem(allocator, max_entries, 0);
    }

    pub fn initWithMem(allocator: std.mem.Allocator, max_entries: u32, max_kv_bytes: u64) HotPrefixCache {
        return .{
            .entries = std.ArrayList(Entry).empty,
            .max_entries = if (max_entries == 0) 1 else max_entries,
            .max_kv_bytes = max_kv_bytes,
            .current_kv_bytes = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HotPrefixCache) void {
        for (self.entries.items) |*e| {
            freeEntryOwnedState(self.allocator, e);
        }
        self.entries.deinit(self.allocator);
        if (self.disk) |*d| d.deinit();
        self.disk = null;
    }

    /// Free everything an Entry owns: token buffer, KV snapshot, SSM
    /// checkpoint array. Used by `deinit`, eviction, and replace paths so
    /// they don't drift apart.
    fn freeEntryOwnedState(allocator: std.mem.Allocator, e: *Entry) void {
        allocator.free(e.tokens);
        e.snapshot.deinit();
        if (e.ssm_checkpoints) |cps| {
            for (cps) |*cp| cp.deinit(allocator);
            allocator.free(cps);
            e.ssm_checkpoints = null;
        }
        if (e.dflash) |*d| {
            d.deinit();
            e.dflash = null;
        }
        if (e.mtp) |*m| {
            m.deinit();
            e.mtp = null;
        }
    }

    /// Restore a committed speculative-side cache (DFlash context or MTP
    /// history) into the caller's cache, clamped to the trunk's restored
    /// length. Returns the base position on success, null when there is
    /// nothing to restore, the snap starts PAST what the trunk actually
    /// reused, or the snap ends BEFORE it (a history with a gap right below
    /// the generation point is worse than a blind start). Best-effort by
    /// contract: a failure leaves the caller blind, never wrong.
    fn restoreSpecSnap(snap_opt: ?*const DflashSnap, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream, what: []const u8) ?usize {
        const t = target orelse return null;
        const snap = snap_opt orelse return null;
        if (snap.base_pos > matched) return null;
        const want = matched - snap.base_pos;
        if (want > snap.snapshot.step) return null;
        t.cache.restore(&snap.snapshot) catch |err| {
            log.warn("  [hot-cache] {s} restore failed: {s} — starts blind\n", .{ what, @errorName(err) });
            return null;
        };
        t.cache.truncate(want, s) catch |err| {
            log.warn("  [hot-cache] {s} clamp failed: {s} — starts blind\n", .{ what, @errorName(err) });
            return null;
        };
        t.base_pos.* = snap.base_pos;
        log.debug("  [hot-cache] {s} restored: {d} tokens from base {d}\n", .{ what, want, snap.base_pos });
        return snap.base_pos;
    }

    fn restoreDflash(e: *const Entry, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream) ?usize {
        return restoreSpecSnap(if (e.dflash) |*d| d else null, target, matched, s, "dflash context");
    }

    fn restoreMtp(e: *const Entry, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream) ?usize {
        return restoreSpecSnap(if (e.mtp) |*m| m else null, target, matched, s, "mtp history");
    }

    /// Disk-tier variant: load the persisted spec snapshot (v4 sidecar) as a
    /// transient and adopt it under the EXACT same clamp rule as the RAM
    /// tier (`restoreSpecSnap`). The trunk restore already forwarded nothing,
    /// so without this a disk hit drafted blind — the 92.6% → 66.5%
    /// acceptance class the RAM tier already fixed.
    fn diskRestoreSpec(
        d: *kv_disk_cache.DiskTier,
        idx: usize,
        target: ?DflashTarget,
        matched: usize,
        s: mlx.mlx_stream,
        which: kv_disk_cache.SpecKind,
    ) ?usize {
        const t = target orelse return null;
        const loaded = d.loadSpecSnap(idx, which, t.cache.entries.len, t.cache.config) orelse return null;
        // restore() refcount-shares the arrays into the target, so the
        // transient snapshot is freed right after.
        var snap = DflashSnap{ .snapshot = loaded.snap, .base_pos = loaded.base };
        defer snap.deinit();
        return restoreSpecSnap(&snap, target, matched, s, switch (which) {
            .dflash => "dflash context",
            .mtp => "mtp history",
        });
    }

    /// The largest checkpoint whose `pos ≤ limit` (checkpoints are sorted
    /// ascending). Shared by the RAM restore and the RAM-vs-disk fairness
    /// comparison — both need the effective restorable length of a hybrid
    /// entry, which is the highest snapshotted position ≤ the token match, NOT
    /// the raw match length (SSM state only exists at snapshotted positions).
    fn highestCheckpointAtOrBelow(cps: []const SSMCheckpoint, limit: usize) ?*const SSMCheckpoint {
        var picked: ?*const SSMCheckpoint = null;
        for (cps) |*cp| {
            // A deinit'd stub (`shedCheckpointsToFit`'s realloc-failure
            // leftover) has zero layers and restores nothing — skip it.
            if (cp.layers.len == 0) continue;
            if (cp.pos > limit) break;
            picked = cp;
        }
        return picked;
    }

    /// Latest checkpoint that carries QSA aux, unless it IS `restored`
    /// (restoreSsmCheckpoint already installed that aux at full length).
    fn qsaHistorySource(cps: []const SSMCheckpoint, restored: *const SSMCheckpoint) ?*const SSMCheckpoint {
        var i = cps.len;
        while (i > 0) {
            i -= 1;
            if (!checkpointHasQsaHistory(&cps[i])) continue;
            if (&cps[i] == restored) return null;
            return &cps[i];
        }
        return null;
    }

    /// Reset every SSM entry to the uninitialized (cold) state. Used on every
    /// miss / failed-restore path so a subsequent prefill starts from a clean
    /// recurrent state instead of stale conv/ssm buffers.
    fn resetSsmEntries(entries: []SSMCacheEntry) void {
        for (entries) |*ssm| {
            _ = mlx.mlx_array_free(ssm.conv_state);
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.conv_state = mlx.mlx_array_new();
            ssm.ssm_state = mlx.mlx_array_new();
            ssm.initialized = false;
            if (ssm.aux_state.ctx != null) _ = mlx.mlx_array_free(ssm.aux_state);
            if (ssm.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(ssm.qsa_pooled);
            ssm.aux_state = .{ .ctx = null };
            ssm.qsa_pooled = .{ .ctx = null };
            ssm.ple_prev_valid = false;
        }
    }

    /// Pure-attention + DSV4 are eligible by default. Hybrid recurrent-state
    /// archs are gated by `enable_ssm_checkpoints` (set by the scheduler
    /// when `--ssm-checkpoint-stride > 0`): with checkpoints we can rewind
    /// both KV and SSM state to a stride-aligned position; without them
    /// every divergence would force a full reset, so we keep the legacy
    /// single-slot path.
    ///
    /// Both `has_hybrid_layers` and `full_attention_interval > 0` signal
    /// the model has SSM/GatedDeltaNet layers somewhere — `has_hybrid_layers`
    /// is set explicitly by the parsers for lfm2 / nemotron_h; the qwen3_5
    /// family sets `full_attention_interval` to N to mark "every Nth layer
    /// is full attention, the rest are GatedDeltaNet". Either way the same
    /// SSM-checkpoint gate applies.
    pub fn shouldUse(
        config: *const model_mod.ModelConfig,
        enable_ssm_checkpoints: bool,
    ) bool {
        // dsv4 keeps its per-request state (raw-kv rings, compressed caches,
        // compressor pending windows) on the module-owned Dsv4Model, not in
        // the KVCache — a snapshot restore would advance cache.step without
        // rebuilding that state. Off until dsv4 state rides the ssm-entry
        // machinery (needsSsmEntries class).
        if (std.mem.eql(u8, config.model_type, "deepseek_v4")) return false;
        const has_ssm_layers = config.has_hybrid_layers or config.full_attention_interval > 0;
        if (has_ssm_layers and !enable_ssm_checkpoints) return false;
        return true;
    }

    fn bumpCounter(self: *HotPrefixCache) u64 {
        self.counter += 1;
        return self.counter;
    }

    /// Wave 1.B: total KV bytes held by a snapshot — sum of `size * itemsize`
    /// across every initialized entry's storage arrays. mlx-c arrays carry
    /// their shape + dtype so this is exact, not a heuristic. Quant schemes
    /// account for q, scales, biases together; future schemes (TurboQuant)
    /// add `kv_quant.snapshotBytesExtra` for per-layer rotation state.
    fn snapshotBytes(snap: *const KVCacheSnapshot) u64 {
        var total: u64 = 0;
        for (snap.entries) |e| {
            if (!e.initialized) continue;
            total += @as(u64, mlx.mlx_array_size(e.keys)) * @as(u64, mlx.mlx_array_itemsize(e.keys));
            total += @as(u64, mlx.mlx_array_size(e.values)) * @as(u64, mlx.mlx_array_itemsize(e.values));
            if (snap.config.scheme != .off) {
                total += @as(u64, mlx.mlx_array_size(e.keys_scales)) * @as(u64, mlx.mlx_array_itemsize(e.keys_scales));
                total += @as(u64, mlx.mlx_array_size(e.keys_biases)) * @as(u64, mlx.mlx_array_itemsize(e.keys_biases));
                total += @as(u64, mlx.mlx_array_size(e.values_scales)) * @as(u64, mlx.mlx_array_itemsize(e.values_scales));
                total += @as(u64, mlx.mlx_array_size(e.values_biases)) * @as(u64, mlx.mlx_array_itemsize(e.values_biases));
            }
        }
        return total;
    }

    /// Issue #330: per-token bytes of a snapshot — what one retained token
    /// costs after a `trimmedCopy` materializes exactly `len` rows. Derived
    /// from each array's own shape (bytes / capacity rows), so it prices
    /// quantized triples correctly too.
    fn snapshotRowBytes(snap: *const KVCacheSnapshot) u64 {
        var total: u64 = 0;
        for (snap.entries) |e| {
            if (!e.initialized) continue;
            inline for (.{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |arr| {
                if (arr.ctx != null) {
                    const rows: u64 = @intCast(mlx.mlx_array_shape(arr)[2]);
                    if (rows > 0) {
                        total += (@as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr))) / rows;
                    }
                }
            }
        }
        return total;
    }

    /// Issue #330: the longest retainable prefix length under `budget`, or
    /// null when nothing at or above the commit floor fits. With checkpoints
    /// (hybrid entry) the trim point must be a RESTORABLE position — a
    /// checkpoint's own `pos` — and its cost includes every checkpoint it
    /// keeps; a KV-only hybrid prefix restores as a cold miss while occupying
    /// an LRU slot, so `budget / row_bytes` is not an answer there. Plain
    /// attention restores at any length, so the budget simply prices tokens.
    /// `limit` caps the answer (tokens.len, or the media boundary — trimming
    /// INTO placeholder rows is not a shape we want to reason about).
    fn trimLenForBudget(
        budget: u64,
        limit: usize,
        row_bytes: u64,
        cps: ?[]const SSMCheckpoint,
    ) ?usize {
        if (cps != null and cps.?.len > 0) {
            const list = cps.?;
            var k = list.len;
            while (k > 0) {
                k -= 1;
                const p = list[k].pos;
                if (p > limit) continue;
                // Ascending positions: everything below here is under the
                // floor too.
                if (p < MIN_CANCELLED_COMMIT_TOKENS) return null;
                var cps_cost: u64 = 0;
                for (list[0 .. k + 1]) |*cp| cps_cost += ssmCheckpointBytes(cp);
                if (@as(u64, p) * row_bytes + cps_cost <= budget) return p;
            }
            return null;
        }
        if (row_bytes == 0) return null;
        const fit: usize = @intCast(budget / row_bytes);
        const len = @min(fit, limit);
        if (len < MIN_CANCELLED_COMMIT_TOKENS) return null;
        return len;
    }

    /// Find the entry with the longest EFFECTIVELY RESTORABLE prefix shared
    /// with `prompt_ids` and matching `(has_tools, quant_config)`. For a hybrid
    /// target that means the highest SSM checkpoint at or below the raw token
    /// match; a longer raw match with no usable checkpoint must not hide an
    /// older entry that can actually restore. Returns the entry index and raw
    /// shared-prefix length; null if no entry matches the key. Wave 1.A:
    /// the config filter exists because cross-config buffer layouts differ
    /// — a slot running `kv_quant=4` cannot restore from an entry committed
    /// in dense (or 8-bit) mode and vice versa. The full `KVQuantConfig`
    /// (scheme + bits + group_size) is compared because `Scheme.affine`
    /// covers BOTH 4-bit and 8-bit packings: filtering on `Scheme` alone
    /// would let a 4-bit entry alias to an 8-bit slot and crash SDPA on
    /// restore. Different media keys may share only the text state before
    /// the earliest known media-placeholder boundary; without a boundary the
    /// lookup stays conservative and rejects the entry. See
    /// `tests/test_kv_quant_per_request.sh`.
    fn findBestRestorableMatch(
        self: *const HotPrefixCache,
        prompt_ids: []const u32,
        has_tools: bool,
        vision_key: u64,
        media_start: ?usize,
        quant_config: kv_quant.KVQuantConfig,
        require_ssm_checkpoint: bool,
    ) ?struct { idx: usize, shared: usize } {
        var best_idx: ?usize = null;
        var best_shared: usize = 0;
        var best_effective: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant_config, quant_config)) continue;

            var max_shared = @min(e.tokens.len, prompt_ids.len);
            if (e.vision_key != vision_key) {
                // Placeholder token IDs do not encode media pixels. Once an
                // image/audio/video row is forwarded, model state depends on
                // the media hash and cannot cross keys. State strictly before
                // the first such row remains ordinary text.
                const safe_boundary = if (e.media_start) |entry_start|
                    if (media_start) |request_start| @min(entry_start, request_start) else entry_start
                else
                    media_start orelse continue;
                max_shared = @min(max_shared, safe_boundary);
            }
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;

            const effective = if (require_ssm_checkpoint) blk: {
                const cps = e.ssm_checkpoints orelse continue;
                const cp = highestCheckpointAtOrBelow(cps, shared) orelse continue;
                break :blk cp.pos;
            } else shared;
            if (effective > best_effective or
                (effective == best_effective and shared > best_shared))
            {
                best_effective = effective;
                best_shared = shared;
                best_idx = i;
            }
        }
        if (best_idx) |idx| return .{ .idx = idx, .shared = best_shared };
        return null;
    }

    fn findBestMatch(self: *const HotPrefixCache, prompt_ids: []const u32, has_tools: bool, vision_key: u64, quant_config: kv_quant.KVQuantConfig) ?struct { idx: usize, shared: usize } {
        const match = self.findBestRestorableMatch(prompt_ids, has_tools, vision_key, null, quant_config, false) orelse return null;
        return .{ .idx = match.idx, .shared = match.shared };
    }

    /// Try to restore a matching entry into `target_cache`. On success, returns
    /// the matched prefix length. On miss, fully resets `target_cache` and
    /// returns 0. Caller should prefill the trailing tokens after this.
    ///
    /// The `target_*` parameters generalize the legacy single-slot path
    /// (`xfm.cache`, `xfm.moe_seq_offset`, `xfm.ssm_entries`) so Phase 2
    /// per-slot caches can reuse the same restore machinery.
    ///
    pub fn lookupAndRestore(
        self: *HotPrefixCache,
        target_cache: *KVCache,
        target_moe_seq_offset: *usize,
        target_ssm_entries: ?[]SSMCacheEntry,
        s: mlx.mlx_stream,
        prompt_ids: []const u32,
        has_tools: bool,
        vision_key: u64,
        dflash_target: ?DflashTarget,
        mtp_target: ?DflashTarget,
    ) !LookupResult {
        return self.lookupAndRestoreWithMedia(
            target_cache,
            target_moe_seq_offset,
            target_ssm_entries,
            s,
            prompt_ids,
            has_tools,
            vision_key,
            null,
            dflash_target,
            mtp_target,
        );
    }

    pub fn lookupAndRestoreWithMedia(
        self: *HotPrefixCache,
        target_cache: *KVCache,
        target_moe_seq_offset: *usize,
        target_ssm_entries: ?[]SSMCacheEntry,
        s: mlx.mlx_stream,
        prompt_ids: []const u32,
        has_tools: bool,
        vision_key: u64,
        media_start: ?usize,
        dflash_target: ?DflashTarget,
        mtp_target: ?DflashTarget,
    ) !LookupResult {
        const match = self.findBestRestorableMatch(
            prompt_ids,
            has_tools,
            vision_key,
            media_start,
            target_cache.config,
            target_ssm_entries != null,
        );

        // ── SSD tier: consult when it can beat the RAM match meaningfully
        // (fresh boot, post-eviction). Phase 3 handles hybrid targets too —
        // the tier persists per-position SSM checkpoints beside the KV chunks
        // and restores both.
        if (self.disk) |*d| disk: {
            if (vision_key != 0) break :disk;
            const dm = d.bestMatch(prompt_ids, has_tools, target_cache.config) orelse break :disk;

            if (target_ssm_entries) |ssm_entries| {
                // Hybrid: compare EFFECTIVE restorable positions — the largest
                // SSM checkpoint ≤ the match on each tier, not the raw prefix
                // length (KV alone is useless without matching SSM state).
                const ram_eff: usize = if (match) |m| blk: {
                    const e = &self.entries.items[m.idx];
                    const cps = e.ssm_checkpoints orelse break :blk 0;
                    const cp = highestCheckpointAtOrBelow(cps, m.shared) orelse break :blk 0;
                    break :blk cp.pos;
                } else 0;
                const disk_cp = d.highestSsmPosAtOrBelow(dm.idx, dm.usable) orelse break :disk;
                if (@as(usize, disk_cp) < ram_eff + kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) break :disk;
                const sw = io_util.Stopwatch.init(d.io);
                const restored = d.restoreIntoHybrid(target_cache, ssm_entries, dm.idx, disk_cp, s) catch |err| {
                    log.warn("  [disk-cache] hybrid restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
                    // A failed restore can leave the cache AND ssm entries
                    // half-rebuilt; reset both before the fall-through.
                    target_cache.truncate(0, s) catch {};
                    resetSsmEntries(ssm_entries);
                    break :disk;
                };
                if (self.qsa_history_required and !entriesHaveQsaHistory(ssm_entries)) {
                    log.warn("  [disk-cache] hybrid restore carries no QSA history — falling back to RAM/cold path\n", .{});
                    target_cache.truncate(0, s) catch {};
                    resetSsmEntries(ssm_entries);
                    break :disk;
                }
                // A checkpoint is always ≤ prompt_len−1, so a hybrid restore
                // never takes the full-match branch (same as the RAM path).
                target_moe_seq_offset.* = restored;
                const ms = sw.read() / std.time.ns_per_ms;
                log.info("  [disk-cache] restored {d}/{d} tokens from SSD in {d}ms (ssm@{d})\n", .{ restored, prompt_ids.len, ms, disk_cp });
                return .{
                    .matched = restored,
                    .full_match = false,
                    .dflash_base = diskRestoreSpec(d, dm.idx, dflash_target, restored, s, .dflash),
                    .mtp_base = diskRestoreSpec(d, dm.idx, mtp_target, restored, s, .mtp),
                };
            }

            const ram_len: usize = if (match) |m| m.shared else 0;
            if (dm.usable <= ram_len) break :disk;
            if (dm.usable - ram_len < kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) break :disk;
            // `dm.usable` is already clamped to the entry's kv_len (see
            // DiskTier.bestMatch), so it IS the restorable length — restore
            // only the chunks covering it. Loading the whole entry
            // (restoreInto) would read a long stored prefix in full to serve a
            // short shared prefix, making a diverged-prefix "hit" slower than a
            // cold miss.
            const effective: usize = dm.usable;
            const full_match = effective == prompt_ids.len;
            const final_len: usize = if (full_match and effective > 1) effective - 1 else effective;
            const sw = io_util.Stopwatch.init(d.io);
            d.restorePrefixInto(target_cache, dm.idx, @intCast(final_len), s) catch |err| {
                log.warn("  [disk-cache] restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
                // A failed restore can leave a half-rebuilt cache; reset it.
                target_cache.truncate(0, s) catch {};
                break :disk;
            };
            target_moe_seq_offset.* = final_len;
            const ms = sw.read() / std.time.ns_per_ms;
            log.info("  [disk-cache] restored {d}/{d} tokens from SSD ({d} chunks) in {d}ms\n", .{ final_len, prompt_ids.len, d.chunks_loaded_last, ms });
            return .{
                .matched = final_len,
                .full_match = full_match,
                .dflash_base = diskRestoreSpec(d, dm.idx, dflash_target, final_len, s, .dflash),
                .mtp_base = diskRestoreSpec(d, dm.idx, mtp_target, final_len, s, .mtp),
            };
        }

        if (match == null) {
            try target_cache.truncate(0, s);
            if (target_ssm_entries) |entries| resetSsmEntries(entries);
            target_moe_seq_offset.* = 0;
            return .{ .matched = 0, .full_match = false };
        }
        const m = match.?;
        const e = &self.entries.items[m.idx];
        e.last_used = self.bumpCounter();

        try target_cache.restore(&e.snapshot);

        // Hybrid path: if the entry carries SSM checkpoints, restore the SSM
        // state at the largest stride-aligned position ≤ m.shared and clamp
        // the effective matched length to that position. KV is positionally
        // trimmable; SSM is only restorable at the snapshotted positions.
        // The two MUST stay in sync, so we rewind KV further too.
        var effective_matched: usize = m.shared;
        if (target_ssm_entries) |entries| {
            if (e.ssm_checkpoints) |cps| {
                if (highestCheckpointAtOrBelow(cps, m.shared)) |cp| {
                    try restoreSsmCheckpoint(entries, cp);
                    effective_matched = cp.pos;
                    // QSA indexer history is stored once on the latest snap
                    // (full length). Intermediate restores slice it to cp.pos.
                    if (qsaHistorySource(cps, cp)) |src| {
                        try applyQsaHistoryAt(entries, src, cp.pos, s);
                    }
                    if (self.qsa_history_required and !entriesHaveQsaHistory(entries)) {
                        // No indexer history after restore: a miss, never
                        // an entry that fails every turn.
                        resetSsmEntries(entries);
                        effective_matched = 0;
                    }
                } else {
                    // No checkpoint at or before this prefix length — reset
                    // SSM and treat the match as zero-effective (we have to
                    // cold-prefill anyway because SSM state would be wrong).
                    resetSsmEntries(entries);
                    effective_matched = 0;
                }
            } else {
                // Hybrid model without checkpoints (e.g., committed pre-Phase-1).
                // Reset and treat as cold prefill — we can't safely reuse.
                resetSsmEntries(entries);
                effective_matched = 0;
            }
        }
        target_moe_seq_offset.* = effective_matched;

        // Miss path (hybrid without a usable checkpoint): also reset KV.
        if (effective_matched == 0) {
            try target_cache.truncate(0, s);
            log.info("  [hot-cache] hybrid miss (no checkpoint ≤ {d} of {d}); cold prefill\n", .{ m.shared, prompt_ids.len });
            return .{ .matched = 0, .full_match = false };
        }

        const full_match = effective_matched == prompt_ids.len;
        const final_len: usize = if (full_match and effective_matched > 1) effective_matched - 1 else effective_matched;

        // ALWAYS clamp the restored cache to the matched length. The old guard
        // (`final_len < e.tokens.len`) skipped this on a WHOLE-entry match — but a
        // snapshot can be committed with a KV buffer LONGER than its logical token
        // count: PLD/speculative decode leaves stale draft positions in the buffer
        // past the committed step, and `commit` snapshots them. Restoring that
        // (then skipping the truncate) left `cache.offset` AHEAD of the matched
        // length that generation tracks (`moe_seq_offset`) — a silent drift that
        // corrupts RoPE positions and CRASHES the Gemma sliding-window prefill mask
        // (`broadcast_shapes` mask-vs-KV mismatch; live 2026-07-09 on
        // gemma-4-26B-A4B at ~16K ctx). truncate is a no-op when the buffer is
        // already `final_len`, so unconditional clamping is safe and restores the
        // invariant cache.offset == matched. The stale KV tail has no matching
        // token id, so the match can never reach into it — discarding it is correct.
        try target_cache.truncate(final_len, s);

        if (full_match and effective_matched > 1) {
            target_moe_seq_offset.* = effective_matched - 1;
            log.info("  [hot-cache] full reuse {d}/{d}, re-forwarding last token\n", .{ effective_matched - 1, prompt_ids.len });
            return .{
                .matched = effective_matched - 1,
                .full_match = true,
                .dflash_base = restoreDflash(e, dflash_target, effective_matched - 1, s),
                .mtp_base = restoreMtp(e, mtp_target, effective_matched - 1, s),
            };
        }

        log.info("  [hot-cache] reused {d}/{d} tokens (matched {d}; entry {d}/{d})\n", .{ effective_matched, prompt_ids.len, m.shared, m.idx + 1, self.entries.items.len });
        return .{
            .matched = effective_matched,
            .full_match = full_match,
            .dflash_base = restoreDflash(e, dflash_target, effective_matched, s),
            .mtp_base = restoreMtp(e, mtp_target, effective_matched, s),
        };
    }

    /// Commit the current `source_cache` state under the given key. Updates
    /// the matching entry if one exists for this exact prefix, otherwise
    /// inserts a new entry, evicting the oldest if at capacity. Snapshot is
    /// taken here (cheap — refcount-share, no data copy).
    ///
    pub fn commit(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
    ) !void {
        return self.commitWithSsm(source_cache, tokens, has_tools, null, null, null);
    }

    /// Commit with optional SSM checkpoint array (Phase 1). The caller
    /// transfers ownership of the slice — the entry frees it on eviction via
    /// the shared `freeEntryOwnedState`. Pass null on plain-attention archs;
    /// the entry stays SSM-free.
    pub fn commitWithSsm(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        ssm_cps: ?[]SSMCheckpoint,
        dflash: ?DflashCommit,
        mtp: ?DflashCommit,
    ) !void {
        return self.commitWithState(source_cache, tokens, has_tools, 0, ssm_cps, dflash, mtp);
    }

    /// Commit with SSM checkpoints; ownership of the payload transfers to
    /// the entry.
    pub fn commitWithState(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        vision_key: u64,
        ssm_cps: ?[]SSMCheckpoint,
        dflash: ?DflashCommit,
        mtp: ?DflashCommit,
    ) !void {
        return self.commitWithMediaState(source_cache, tokens, has_tools, vision_key, null, ssm_cps, dflash, mtp);
    }

    pub fn commitWithMediaState(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        vision_key: u64,
        media_start: ?usize,
        ssm_cps: ?[]SSMCheckpoint,
        dflash: ?DflashCommit,
        mtp: ?DflashCommit,
    ) !void {
        const quant_config = source_cache.config;

        var replace_idx: ?usize = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (e.vision_key != vision_key) continue;
            if (!std.meta.eql(e.quant_config, quant_config)) continue;
            if (e.tokens.len <= tokens.len) {
                var shared: usize = 0;
                while (shared < e.tokens.len and e.tokens[shared] == tokens[shared]) shared += 1;
                if (shared == e.tokens.len) {
                    replace_idx = i;
                    break;
                }
            }
        }

        var new_snap = try source_cache.snapshot();
        // The speculative-side payloads are best-effort: a snapshot failure
        // must not cost the trunk KV entry they ride on.
        var new_dflash: ?DflashSnap = null;
        var new_dflash_bytes: u64 = 0;
        if (dflash) |d| {
            if (d.cache.snapshot()) |snap| {
                new_dflash = .{ .snapshot = snap, .base_pos = d.base_pos };
                new_dflash_bytes = snapshotBytes(&new_dflash.?.snapshot);
            } else |err| {
                log.warn("  [hot-cache] dflash context snapshot failed: {s}\n", .{@errorName(err)});
            }
        }
        var new_mtp: ?DflashSnap = null;
        var new_mtp_bytes: u64 = 0;
        if (mtp) |m2| {
            if (m2.cache.snapshot()) |snap| {
                new_mtp = .{ .snapshot = snap, .base_pos = m2.base_pos };
                new_mtp_bytes = snapshotBytes(&new_mtp.?.snapshot);
            } else |err| {
                log.warn("  [hot-cache] mtp history snapshot failed: {s}\n", .{@errorName(err)});
            }
        }
        var new_kv_bytes = snapshotBytes(&new_snap);
        var new_ssm_bytes: u64 = 0;
        if (ssm_cps) |cps| {
            for (cps) |*cp| new_ssm_bytes += ssmCheckpointBytes(cp);
        }
        var new_bytes = new_kv_bytes + new_ssm_bytes + new_dflash_bytes + new_mtp_bytes;
        // Effective candidate: a byte-budget trim below shortens these.
        var eff_tokens = tokens;
        var eff_media_start = media_start;
        var eff_cps = ssm_cps;
        // The byte budget is a hard retention cap, including for the first (or
        // only) entry. The old eviction loop could empty the cache and then
        // append an entry larger than the cap, defeating the load-time clamp
        // precisely for long single-conversation prefixes (#326). But a flat
        // decline is a CLIFF (#330): a long agent session crosses the budget
        // once mid-conversation and then cold-prefills every turn while the
        // cap "holds" zero bytes. Retain the longest restorable prefix that
        // fits instead; decline only when nothing above the floor does.
        if (self.max_kv_bytes > 0 and new_bytes > self.max_kv_bytes) {
            var trimmed_ok = false;
            trim_blk: {
                const limit = if (eff_media_start) |ms| @min(tokens.len, ms) else tokens.len;
                const tl = trimLenForBudget(self.max_kv_bytes, limit, snapshotRowBytes(&new_snap), eff_cps) orelse break :trim_blk;
                // One-shot: when the resident covered entry already retains
                // the trim target, keep it and drop the candidate — the
                // target is budget-derived and stable, so replacing would
                // re-copy an identical multi-GB prefix every turn.
                if (replace_idx) |idx| {
                    if (self.entries.items[idx].tokens.len >= tl) {
                        var discarded = new_snap;
                        discarded.deinit();
                        if (new_dflash) |*d| d.deinit();
                        if (new_mtp) |*m3| m3.deinit();
                        if (eff_cps) |cps| {
                            for (cps) |*cp| cp.deinit(self.allocator);
                            self.allocator.free(cps);
                        }
                        log.info("  [hot-cache] kept resident {d}-token prefix; oversized candidate ({d} tokens, {d:.2} MB > {d:.2} MB budget) trims no further\n", .{
                            self.entries.items[idx].tokens.len,
                            tokens.len,
                            @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                            @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
                        });
                        return;
                    }
                }
                const trimmed = new_snap.trimmedCopy(tl, mlx.gpuStream()) catch break :trim_blk;
                new_snap.deinit();
                new_snap = trimmed;
                // Spec payloads describe the FULL-length state; a trimmed
                // prefix rebuilds them on its first reused turn.
                if (new_dflash) |*d| {
                    d.deinit();
                    new_dflash = null;
                    new_dflash_bytes = 0;
                }
                if (new_mtp) |*m3| {
                    m3.deinit();
                    new_mtp = null;
                    new_mtp_bytes = 0;
                }
                if (eff_cps) |cps| {
                    var kept: usize = 0;
                    while (kept < cps.len and cps[kept].pos <= tl) kept += 1;
                    if (kept < cps.len) {
                        // QSA history lives only on the latest snap. Slicing
                        // it onto the last KEPT snap before dropping the tail
                        // is what keeps a trimmed 122k entry restorable.
                        if (kept > 0 and checkpointHasQsaHistory(&cps[cps.len - 1])) {
                            sliceQsaHistoryOntoCheckpoint(&cps[kept - 1], &cps[cps.len - 1], cps[kept - 1].pos, mlx.gpuStream()) catch {};
                        }
                        const shrunk = self.allocator.dupe(SSMCheckpoint, cps[0..kept]) catch break :trim_blk;
                        for (cps[kept..]) |*cp| cp.deinit(self.allocator);
                        self.allocator.free(cps);
                        eff_cps = shrunk;
                    }
                }
                eff_tokens = tokens[0..tl];
                if (eff_media_start) |ms| {
                    if (ms >= tl) eff_media_start = null;
                }
                new_kv_bytes = snapshotBytes(&new_snap);
                new_ssm_bytes = 0;
                if (eff_cps) |cps| {
                    for (cps) |*cp| new_ssm_bytes += ssmCheckpointBytes(cp);
                }
                new_bytes = new_kv_bytes + new_ssm_bytes;
                log.info("  [hot-cache] trimmed oversized entry to {d}/{d} tokens ({d:.2} <= {d:.2} MB budget)\n", .{
                    tl,
                    tokens.len,
                    @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
                });
                trimmed_ok = true;
            }
            if (!trimmed_ok) {
                var discarded_snap = new_snap;
                discarded_snap.deinit();
                if (new_dflash) |*d| d.deinit();
                if (new_mtp) |*m3| m3.deinit();
                if (eff_cps) |cps| {
                    for (cps) |*cp| cp.deinit(self.allocator);
                    self.allocator.free(cps);
                }
                log.info("  [hot-cache] skipped oversized entry ({d} tokens, {d:.2} MB > {d:.2} MB budget)\n", .{
                    tokens.len,
                    @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
                });
                return;
            }
        }
        // Ownership of the checkpoint slice transfers to the cache
        // UNCONDITIONALLY — success, decline, or error (#330 adjacent: the
        // scheduler's `catch` arm also freed them, so every failed commit was
        // a double free, with a different allocator). After a trim `eff_cps`
        // may be a cache-allocated replacement of the caller's slice, so the
        // cache is the only party that can still free correctly.
        const tokens_owned = self.allocator.dupe(u32, eff_tokens) catch |err| {
            var snap = new_snap;
            snap.deinit();
            if (new_dflash) |*d| d.deinit();
            if (new_mtp) |*m3| m3.deinit();
            if (eff_cps) |cps| {
                for (cps) |*cp| cp.deinit(self.allocator);
                self.allocator.free(cps);
            }
            return err;
        };

        if (replace_idx) |idx| {
            const e = &self.entries.items[idx];

            // Phase 1: SSM checkpoint inheritance on prefix-extend. The
            // replace path triggers when the new entry's tokens fully
            // extend the old's (i.e., e.tokens is a prefix of `tokens`).
            // The old SSM checkpoints were captured at positions inside
            // e.tokens, so they're still valid for the new entry — those
            // positions are a strict prefix of `tokens`. Inherit them and
            // append any new checkpoints from this turn that don't overlap.
            //
            // Without this, multi-turn flows lose checkpoints fast: turn 2's
            // prefill of the short tail captures few or no checkpoints, so
            // turn 3 has nothing to restore from even though turn 2's match
            // covered nearly the full prefix. (Reproducible by alternating
            // identical-prompt requests at ssm_checkpoint_stride > prompt_len.)
            const merged_cps: ?[]SSMCheckpoint = blk: {
                const old = e.ssm_checkpoints orelse break :blk eff_cps;
                const new = eff_cps orelse {
                    // Detach old so the free-below doesn't touch it; it
                    // becomes the new entry's checkpoint list as-is.
                    e.ssm_checkpoints = null;
                    break :blk old;
                };
                // Both old and new have data. Concat into a sorted-by-pos,
                // dedup-by-pos list. Allocate fresh; transfer ownership.
                var merged = std.ArrayList(SSMCheckpoint).empty;
                errdefer {
                    for (merged.items) |*c| c.deinit(self.allocator);
                    merged.deinit(self.allocator);
                }
                // Detach old + new from their containers so we can move them.
                e.ssm_checkpoints = null;
                // Walk both, picking lower pos each step; on tie, prefer the
                // new one and discard old (new is the more recently observed
                // state at that position).
                var i: usize = 0;
                var j: usize = 0;
                while (i < old.len or j < new.len) {
                    if (i >= old.len) {
                        try merged.append(self.allocator, new[j]);
                        j += 1;
                    } else if (j >= new.len) {
                        try merged.append(self.allocator, old[i]);
                        i += 1;
                    } else if (old[i].pos < new[j].pos) {
                        try merged.append(self.allocator, old[i]);
                        i += 1;
                    } else if (old[i].pos > new[j].pos) {
                        try merged.append(self.allocator, new[j]);
                        j += 1;
                    } else {
                        // Same pos: keep new, drop old.
                        var dropped = old[i];
                        dropped.deinit(self.allocator);
                        i += 1;
                        try merged.append(self.allocator, new[j]);
                        j += 1;
                    }
                }
                self.allocator.free(old);
                self.allocator.free(new);
                // The merged list spans BOTH turns, so the per-prefill cap in
                // generate.zig no longer bounds it. Re-apply it here — but not
                // oldest-first. Within one prefill that is fine; across turns it
                // collapses the survivors onto the end of the prompt, and then a
                // request that diverges early finds no checkpoint at or below its
                // match and pays a FULL cold prefill:
                //     [hot-cache] hybrid miss (no checkpoint <= 16382 of 178509)
                // That one cost 415 s. Oldest-first is also the expensive choice:
                // a checkpoint costs roughly a constant plus a term linear in its
                // position, so it discards the cheap early ones and keeps the
                // large late ones.
                //
                // Thin the interior instead, always keeping the first and the
                // newest: drop whichever checkpoint sits between the closest pair
                // of neighbours, i.e. the one whose removal widens the coverage
                // gap least. The result is a spread over the whole prompt at the
                // same count and LESS memory. `n` is at most ssm_checkpoint_max,
                // so the quadratic scan is trivial.
                while (self.ssm_checkpoint_max > 0 and
                    merged.items.len > self.ssm_checkpoint_max)
                {
                    // Under three there is no interior to thin; honour the cap by
                    // dropping the oldest, which is also the cheapest to redo.
                    const drop = if (merged.items.len < 3) 0 else blk2: {
                        var best_at: usize = 1;
                        var best_span: usize = std.math.maxInt(usize);
                        var k: usize = 1;
                        while (k + 1 < merged.items.len) : (k += 1) {
                            const span = merged.items[k + 1].pos - merged.items[k - 1].pos;
                            if (span < best_span) {
                                best_span = span;
                                best_at = k;
                            }
                        }
                        break :blk2 best_at;
                    };
                    var dropped = merged.orderedRemove(drop);
                    dropped.deinit(self.allocator);
                }
                const owned = try merged.toOwnedSlice(self.allocator);
                // The inherited latest and this turn's latest both carry the
                // indexer history: keep one.
                keepOnlyLatestQsaHistory(owned);
                break :blk owned;
            };

            // Free everything the old entry owned EXCEPT the (now-detached)
            // ssm_checkpoints, which were moved above.
            self.allocator.free(e.tokens);
            e.snapshot.deinit();
            // The old speculative payloads describe a strict PREFIX of the
            // new tokens, but they are keyed to their own base_pos and
            // length; the new ones supersede them outright. A commit with no
            // payload drops the old rather than keeping a shorter stale one.
            if (e.dflash) |*d| d.deinit();
            e.dflash = null;
            if (e.mtp) |*m4| m4.deinit();
            e.mtp = null;
            self.current_kv_bytes -|= e.kv_bytes;

            // Recompute ssm bytes from the merged list.
            var merged_ssm_bytes: u64 = 0;
            if (merged_cps) |cps| {
                for (cps) |*cp| merged_ssm_bytes += ssmCheckpointBytes(cp);
            }
            e.tokens = tokens_owned;
            e.snapshot = new_snap;
            e.has_tools = has_tools;
            e.vision_key = vision_key;
            e.media_start = eff_media_start;
            e.quant_config = quant_config;
            e.kv_bytes = new_kv_bytes + merged_ssm_bytes + new_dflash_bytes + new_mtp_bytes;
            e.ssm_checkpoints = merged_cps;
            e.ssm_bytes = merged_ssm_bytes;
            e.dflash = new_dflash;
            e.dflash_bytes = new_dflash_bytes;
            e.mtp = new_mtp;
            e.mtp_bytes = new_mtp_bytes;
            e.last_used = self.bumpCounter();
            self.current_kv_bytes += e.kv_bytes;
            // Inherited SSM checkpoints can make a replacement larger than
            // `new_bytes`, so enforce the cap again on the final entry — but
            // never by evicting the entry we just paid to update (#330
            // adjacent: near the budget that thrashed commit → evict all →
            // cold prefill, every turn). Evict OTHER entries first, then shed
            // this entry's checkpoints; eviction of the sole entry is the
            // last resort that keeps the load-time headroom clamp real.
            if (self.max_kv_bytes > 0) {
                while (self.current_kv_bytes > self.max_kv_bytes and
                    self.entries.items.len > 1)
                {
                    self.evictOneLru("byte budget");
                }
                self.shedCheckpointsToFit();
                while (self.current_kv_bytes > self.max_kv_bytes and
                    self.entries.items.len > 0)
                {
                    self.evictOneLru("byte budget");
                }
            }
            if (self.disk != null) self.disk_dirty = true;
            self.logResident();
            return;
        }

        while (self.entries.items.len >= self.max_entries) {
            self.evictOneLru("count cap");
        }
        if (self.max_kv_bytes > 0) {
            while (self.current_kv_bytes + new_bytes > self.max_kv_bytes and self.entries.items.len > 0) {
                self.evictOneLru("byte budget");
            }
        }

        self.entries.append(self.allocator, .{
            .tokens = tokens_owned,
            .has_tools = has_tools,
            .vision_key = vision_key,
            .media_start = eff_media_start,
            .snapshot = new_snap,
            .last_used = self.bumpCounter(),
            .quant_config = quant_config,
            .kv_bytes = new_bytes,
            .ssm_checkpoints = eff_cps,
            .ssm_bytes = new_ssm_bytes,
            .dflash = new_dflash,
            .dflash_bytes = new_dflash_bytes,
            .mtp = new_mtp,
            .mtp_bytes = new_mtp_bytes,
        }) catch |err| {
            self.allocator.free(tokens_owned);
            var snap = new_snap;
            snap.deinit();
            if (new_dflash) |*d| d.deinit();
            if (new_mtp) |*m5| m5.deinit();
            if (eff_cps) |cps| {
                for (cps) |*cp| cp.deinit(self.allocator);
                self.allocator.free(cps);
            }
            return err;
        };
        self.current_kv_bytes += new_bytes;
        if (self.disk != null) self.disk_dirty = true;
        self.logResident();
    }

    /// Flush the most recent commit to the SSD tier. Called by the scheduler
    /// AFTER `markFinished` (the client already has its response) — the
    /// chunk-append is bounded (partial tail + new chunks) but synchronous on
    /// the inference thread. Snapshot arrays are refcount-shared with the RAM
    /// entry, so slicing them here reads the same buffers the commit captured.
    pub fn flushPendingDisk(self: *HotPrefixCache, s: mlx.mlx_stream) void {
        if (!self.disk_dirty) return;
        self.disk_dirty = false;
        const d = if (self.disk) |*dd| dd else return;
        if (self.entries.items.len == 0) return;
        var newest: *Entry = &self.entries.items[0];
        for (self.entries.items[1..]) |*e| {
            if (e.last_used > newest.last_used) newest = e;
        }
        if (newest.vision_key != 0) return;
        // Phase 3: hybrid entries persist their SSM checkpoints alongside the
        // KV chunks (immutable per-position s*.safetensors). The snapshot
        // arrays are refcount-shared with the RAM entry, so `appendCommit`
        // reads the same buffers the commit captured.
        // v4: the spec snapshots (dflash context / MTP history) ride along —
        // eligibility was enforced at commitWithState, so the disk tier
        // persists exactly what the RAM entry holds.
        const dflash_spec: ?kv_disk_cache.SpecCommit = if (newest.dflash) |*df| .{
            .entries = df.snapshot.entries,
            .step = df.snapshot.step,
            .config = df.snapshot.config,
            .base_pos = df.base_pos,
        } else null;
        const mtp_spec: ?kv_disk_cache.SpecCommit = if (newest.mtp) |*mm| .{
            .entries = mm.snapshot.entries,
            .step = mm.snapshot.step,
            .config = mm.snapshot.config,
            .base_pos = mm.base_pos,
        } else null;
        const complete = d.appendCommitWithSpec(
            newest.snapshot.entries,
            newest.snapshot.step,
            newest.snapshot.config,
            newest.tokens,
            newest.has_tools,
            newest.ssm_checkpoints,
            dflash_spec,
            mtp_spec,
            s,
        ) catch |err| {
            log.warn("  [disk-cache] persist failed: {s}\n", .{@errorName(err)});
            return;
        };
        // Byte-capped flush: a large entry persists incrementally — keep the
        // dirty flag set so the next finished request continues the write.
        if (!complete) self.disk_dirty = true;
    }

    /// #330 adjacent: when the byte budget is exceeded with the just-updated
    /// entry as the sole survivor, drop its checkpoints (the inherited bytes
    /// the pre-check could not price) instead of evicting it. Reuses the
    /// replace path's interior-thinning rule (keep the first and the newest).
    /// The pre-check guarantees the entry's own KV + this turn's checkpoints
    /// fit, so shedding converges under budget before the list empties in
    /// practice; if it doesn't, the caller's eviction fallback decides.
    fn shedCheckpointsToFit(self: *HotPrefixCache) void {
        if (self.max_kv_bytes == 0 or self.current_kv_bytes <= self.max_kv_bytes) return;
        if (self.entries.items.len == 0) return;
        var newest: *Entry = &self.entries.items[0];
        for (self.entries.items[1..]) |*e| {
            if (e.last_used > newest.last_used) newest = e;
        }
        const cps = newest.ssm_checkpoints orelse return;
        var n = cps.len;
        var shed: usize = 0;
        while (n > 1 and self.current_kv_bytes > self.max_kv_bytes) {
            const drop = if (n < 3) 0 else blk: {
                var best_at: usize = 1;
                var best_span: usize = std.math.maxInt(usize);
                var k: usize = 1;
                while (k + 1 < n) : (k += 1) {
                    const span = cps[k + 1].pos - cps[k - 1].pos;
                    if (span < best_span) {
                        best_span = span;
                        best_at = k;
                    }
                }
                break :blk best_at;
            };
            const freed = ssmCheckpointBytes(&cps[drop]);
            if (drop + 1 == n and drop > 0 and checkpointHasQsaHistory(&cps[drop])) {
                sliceQsaHistoryOntoCheckpoint(&cps[drop - 1], &cps[drop], cps[drop - 1].pos, mlx.gpuStream()) catch {};
            }
            cps[drop].deinit(self.allocator);
            var k = drop;
            while (k + 1 < n) : (k += 1) cps[k] = cps[k + 1];
            n -= 1;
            shed += 1;
            newest.ssm_bytes -|= freed;
            newest.kv_bytes -|= freed;
            self.current_kv_bytes -|= freed;
        }
        if (shed == 0) return;
        // Shrink-in-place realloc cannot practically fail; if it somehow
        // does, the deinit'd tail stubs (pos 0, zero layers) stay in the
        // slice — `highestCheckpointAtOrBelow` skips empty checkpoints and a
        // re-deinit of a stub is a no-op, so they are inert.
        newest.ssm_checkpoints = self.allocator.realloc(cps, n) catch cps;
        log.info("  [hot-cache] shed {d} checkpoints to fit the byte budget ({d} kept)\n", .{ shed, n });
    }

    fn evictOneLru(self: *HotPrefixCache, reason: []const u8) void {
        var lru_idx: usize = 0;
        var lru_used: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used < lru_used) {
                lru_used = e.last_used;
                lru_idx = i;
            }
        }
        var evicted = self.entries.swapRemove(lru_idx);
        const tokens_len = evicted.tokens.len;
        const kv_mb = @as(f64, @floatFromInt(evicted.kv_bytes)) / (1024.0 * 1024.0);
        const had_ssm = evicted.ssm_checkpoints != null;
        const ssm_mb = @as(f64, @floatFromInt(evicted.ssm_bytes)) / (1024.0 * 1024.0);
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        if (had_ssm) {
            log.info("  [hot-cache] evicted LRU entry ({s}; was {d} tokens, {d:.2} MB; ssm {d:.2} MB)\n", .{
                reason, tokens_len, kv_mb, ssm_mb,
            });
        } else {
            log.info("  [hot-cache] evicted LRU entry ({s}; was {d} tokens, {d:.2} MB)\n", .{
                reason, tokens_len, kv_mb,
            });
        }
    }

    fn logResident(self: *const HotPrefixCache) void {
        const mb = @as(f64, @floatFromInt(self.current_kv_bytes)) / (1024.0 * 1024.0);
        if (self.max_kv_bytes == 0) {
            log.info("  [hot-cache] resident={d:.2} MB ({d}/{d} entries)\n", .{ mb, self.entries.items.len, self.max_entries });
        } else {
            const cap_mb = @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0);
            log.info("  [hot-cache] resident={d:.2} / {d:.2} MB ({d}/{d} entries)\n", .{ mb, cap_mb, self.entries.items.len, self.max_entries });
        }
    }

    /// Drop all entries — forces every future request to cold-prefill. Called
    /// when the cache is suspect (pad-only generation, image-bearing prompt,
    /// tools toggle change).
    pub fn invalidateAll(self: *HotPrefixCache, reason: []const u8) void {
        // Suspect state must die on BOTH tiers — a poisoned prefix that
        // survives on disk would be immortal across restarts.
        if (self.disk) |*d| d.invalidateAll();
        self.disk_dirty = false;
        if (self.entries.items.len == 0) return;
        log.info("  [hot-cache] invalidating all {d} entries: {s}\n", .{ self.entries.items.len, reason });
        for (self.entries.items) |*e| {
            freeEntryOwnedState(self.allocator, e);
        }
        self.entries.clearRetainingCapacity();
        self.current_kv_bytes = 0;
    }

    /// Drop the most recently committed entry — used after a pad-only
    /// generation: the entry we just wrote may have stale K/V from the bad
    /// generation in tail positions. Other entries from prior healthy
    /// requests remain untouched (improvement over the legacy nuke-everything).
    pub fn invalidateLatest(self: *HotPrefixCache, reason: []const u8) void {
        if (self.disk) |*d| d.invalidateNewest();
        self.disk_dirty = false;
        if (self.entries.items.len == 0) return;
        var newest_idx: usize = 0;
        var newest_used: u64 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used >= newest_used) {
                newest_used = e.last_used;
                newest_idx = i;
            }
        }
        var evicted = self.entries.swapRemove(newest_idx);
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        log.info("  [hot-cache] invalidated latest entry: {s}\n", .{reason});
    }

    pub fn entryCount(self: *const HotPrefixCache) usize {
        return self.entries.items.len;
    }
};

// ── Tests ──

const testing = std.testing;

test "HotPrefixCache: shouldUse gates hybrid by enable_ssm_checkpoints" {
    var cfg = model_mod.ModelConfig{};
    // Plain attention: always allowed.
    try testing.expect(HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
    // Hybrid (lfm2/nemotron_h-style): only with checkpoints enabled.
    cfg.has_hybrid_layers = true;
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
    // Qwen3.5-style full_attention_interval-marks-hybrid: same gate.
    cfg.has_hybrid_layers = false;
    cfg.full_attention_interval = 4;
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
}

test "HotPrefixCache: shouldUse rejects deepseek_v4 (module-owned decode state)" {
    // dsv4's per-request state (raw-kv rings, compressed caches, compressor
    // pending windows) lives on the Dsv4Model, NOT in the 0-entry KVCache
    // shell — a snapshot restore would set cache.step without rebuilding that
    // state, silently serving a stale ring (or crashing on a null dec_state).
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "deepseek_v4";
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, true));
}

test "HotPrefixCache: init zero capacity clamps to 1" {
    var cache = HotPrefixCache.init(testing.allocator, 0);
    defer cache.deinit();
    try testing.expectEqual(@as(u32, 1), cache.max_entries);
    try testing.expectEqual(@as(usize, 0), cache.entryCount());
}

test "HotPrefixCache: findBestMatch returns longest shared prefix" {
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    // Two synthetic entries (snapshots are no-ops on freshly-zero KVCache; we
    // never restore in this unit test, so no GPU work).
    const ids_a = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    const ids_b = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 9, 9, 9 });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_a,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = 1,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_b,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = 2,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });

    // Looking up [1,2,3,4,5,6] should match entry A (5 shared tokens).
    const lookup_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const m = cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(usize, 0), m.idx);
    try testing.expectEqual(@as(usize, 5), m.shared);

    // Looking up [1,2,3,9,9,9,7] should match entry B (6 shared).
    const lookup_ids2 = [_]u32{ 1, 2, 3, 9, 9, 9, 7 };
    const m2 = cache.findBestMatch(&lookup_ids2, false, 0, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(usize, 1), m2.idx);
    try testing.expectEqual(@as(usize, 6), m2.shared);

    // has_tools mismatch returns null.
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, true, 0, kv_quant.KVQuantConfig.dense));
    // vision_key mismatch returns null both ways: a text entry never serves an
    // image request and an image entry only serves the same pixels.
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, false, 7, kv_quant.KVQuantConfig.dense));
    cache.entries.items[0].vision_key = 7;
    // Text lookup falls through to entry B (3 shared); the keyed lookup gets A.
    try testing.expectEqual(@as(usize, 1), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.dense).?.idx);
    try testing.expectEqual(@as(usize, 0), cache.findBestMatch(&lookup_ids, false, 7, kv_quant.KVQuantConfig.dense).?.idx);
    cache.entries.items[0].vision_key = 0;
    // Scheme mismatch returns null — entries are dense, a query for affine
    // 4-bit cannot match (Wave 1.A: cross-scheme cache hits never happen).
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(4)));
}

test "HotPrefixCache: restore clamps an inflated snapshot to the matched length (gemma mask crash)" {
    // Root cause of the live gemma-4-26B-A4B crash (2026-07-09, broadcast_shapes
    // mask 16890 vs KV 16892 at ~16K ctx): a snapshot committed with a KV buffer
    // LONGER than its logical token count — PLD/speculative decode leaves stale
    // draft positions in the buffer past the committed step. When the NEXT prompt
    // matches the entry's ENTIRE token sequence but is longer (a partial hit,
    // effective_matched == e.tokens.len < prompt_ids.len), the old truncate guard
    // `final_len < e.tokens.len` was FALSE, so the restored cache offset kept the
    // inflated snapshot length — drifting ahead of the matched length generation
    // tracks. That drift corrupts RoPE and crashes the sliding-window prefill mask.
    const s = mlx.gpuStream();

    var toks: [67]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 11);
    const logical_len: usize = 64;

    // Source cache: 64 logical tokens, then 2 STALE tokens (offset 66) — the
    // shape a PLD round leaves behind before commit.
    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, @intCast(logical_len));
    try testFillCache(&src, s, 2, 2); // stale draft tail → offset 66
    try testing.expectEqual(@as(usize, 66), src.step);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    // Commit with the LOGICAL token count (64) — but the snapshot carries 66.
    try hc.commit(&src, toks[0..logical_len], false);

    // Reuse with a prompt that matches all 64 entry tokens but is LONGER (67):
    // effective_matched == e.tokens.len == 64 < prompt_ids.len — the crash path.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&dst, &moe_off, null, s, &toks, false, 0, null, null);

    try testing.expect(!res.full_match);
    try testing.expectEqual(@as(usize, 64), res.matched);
    // The invariant: restored cache offset == matched length, NOT the inflated 66.
    try testing.expectEqual(@as(usize, 64), moe_off);
    try testing.expectEqual(@as(usize, 64), dst.step);
    for (dst.entries) |*e| {
        try testing.expect(e.initialized);
        try testing.expectEqual(@as(usize, 64), e.offset); // clamped, not 66
    }
}

test "prefix cache: DFlash assistant context round-trips, clamped to the trunk's matched length" {
    const s = mlx.gpuStream();
    var toks: [64]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 5);

    // Trunk KV for 64 tokens; the assistant context covers the same span but
    // starts at 10 (the committing request was itself a partial cache hit).
    var trunk = try KVCache.init(testing.allocator, 2);
    defer trunk.deinit();
    try testFillCache(&trunk, s, 2, 64);
    var assist = try KVCache.init(testing.allocator, 2);
    defer assist.deinit();
    try testFillCache(&assist, s, 2, 54);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try hc.commitWithSsm(&trunk, &toks, false, null, .{ .cache = &assist, .base_pos = 10 }, null);
    // The assistant context is billed like SSM state, so the memory cap sees it.
    try testing.expect(hc.entries.items[0].dflash_bytes > 0);
    try testing.expect(hc.entries.items[0].kv_bytes > hc.entries.items[0].dflash_bytes);

    // A shorter prompt that the entry fully covers: the full-match path
    // re-forwards the last token, so the trunk lands at 31 and the assistant
    // context must be clamped to 31-10 = 21 — absLen == matched, or the first
    // round's `dctx.absLen() == anchor_pos` assert fires.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var dfl = try KVCache.init(testing.allocator, 2);
    defer dfl.deinit();
    var moe_off: usize = 0;
    var base: usize = 0;
    const res = try hc.lookupAndRestore(
        &dst,
        &moe_off,
        null,
        s,
        toks[0..32],
        false,
        0,
        .{ .cache = &dfl, .base_pos = &base },
        null,
    );
    try testing.expect(res.full_match);
    try testing.expectEqual(@as(usize, 31), res.matched);
    try testing.expectEqual(@as(?usize, 10), res.dflash_base);
    try testing.expectEqual(@as(usize, 10), base);
    try testing.expectEqual(@as(usize, 21), dfl.step);
    try testing.expectEqual(base + dfl.step, res.matched);

    // An entry with no assistant payload leaves the target untouched, and the
    // caller is told so — a blind start is a valid outcome, never a wrong one.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    try hc2.commit(&trunk, &toks, false);
    var dst2 = try KVCache.init(testing.allocator, 2);
    defer dst2.deinit();
    var dfl2 = try KVCache.init(testing.allocator, 2);
    defer dfl2.deinit();
    var moe_off2: usize = 0;
    var base2: usize = 7;
    const res2 = try hc2.lookupAndRestore(
        &dst2,
        &moe_off2,
        null,
        s,
        toks[0..32],
        false,
        0,
        .{ .cache = &dfl2, .base_pos = &base2 },
        null,
    );
    try testing.expectEqual(@as(usize, 31), res2.matched);
    try testing.expectEqual(@as(?usize, null), res2.dflash_base);
    try testing.expectEqual(@as(usize, 0), dfl2.step);
    try testing.expectEqual(@as(usize, 7), base2); // untouched
}

test "prefix cache: MTP committed history round-trips, clamped; a history ending short is declined" {
    // Same DflashSnap machinery, second Entry field: the head's history is
    // built from trunk hiddens and a restore forwards nothing, so without
    // this every reused prefix drafts blind (~70 -> ~38 tok/s on warm echo,
    // Qwen3.6-27B). Unlike the trunk KV, a history that ends BEFORE the
    // matched cursor cannot be adopted — the missing tail's hiddens are
    // unrecoverable, and a gap right below the generation point is worse
    // than a blind start.
    const s = mlx.gpuStream();
    var toks: [64]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 5);

    var trunk = try KVCache.init(testing.allocator, 2);
    defer trunk.deinit();
    try testFillCache(&trunk, s, 2, 64);
    // Committed history covers 60 of the 64 tokens (the deferred-stash lag).
    var hist = try KVCache.init(testing.allocator, 1);
    defer hist.deinit();
    try testFillCache(&hist, s, 1, 60);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try hc.commitWithSsm(&trunk, &toks, false, null, null, .{ .cache = &hist, .base_pos = 0 });
    try testing.expect(hc.entries.items[0].mtp_bytes > 0);
    try testing.expect(hc.entries.items[0].kv_bytes > hc.entries.items[0].mtp_bytes);

    // Shorter prompt fully covered by the entry: full-match arm lands the
    // trunk at 31 and the history clamps to 31 (base 0).
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var mtp_dst = try KVCache.init(testing.allocator, 1);
    defer mtp_dst.deinit();
    var moe_off: usize = 0;
    var base: usize = 99;
    const res = try hc.lookupAndRestore(&dst, &moe_off, null, s, toks[0..32], false, 0, null, .{ .cache = &mtp_dst, .base_pos = &base });
    try testing.expect(res.full_match);
    try testing.expectEqual(@as(usize, 31), res.matched);
    try testing.expectEqual(@as(?usize, 0), res.mtp_base);
    try testing.expectEqual(@as(usize, 0), base);
    try testing.expectEqual(@as(usize, 31), mtp_dst.step);
    try testing.expectEqual(base + mtp_dst.step, res.matched);

    // Full 64-token re-issue: matched 63 > the 60 the history covers →
    // declined, target untouched, caller starts blind.
    var dst2 = try KVCache.init(testing.allocator, 2);
    defer dst2.deinit();
    var mtp2 = try KVCache.init(testing.allocator, 1);
    defer mtp2.deinit();
    var moe2: usize = 0;
    var base2: usize = 7;
    const res2 = try hc.lookupAndRestore(&dst2, &moe2, null, s, &toks, false, 0, null, .{ .cache = &mtp2, .base_pos = &base2 });
    try testing.expectEqual(@as(usize, 63), res2.matched);
    try testing.expectEqual(@as(?usize, null), res2.mtp_base);
    try testing.expectEqual(@as(usize, 0), mtp2.step);
    try testing.expectEqual(@as(usize, 7), base2); // untouched
}

fn testFillCache(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, tokens: u32) !void {
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const count: f64 = @floatFromInt(step * 8);
            const base: f64 = @floatFromInt(written * 8 + li * 1_000_000);
            try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            const shape = [_]c_int{ 1, 1, @intCast(step), 8 };
            try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s));
            var view = try cache.update(li, k, k, s, 0);
            view.deinit();
        }
        written += step;
    }
}

test "HotPrefixCache: disk tier restores across a fresh cache instance (restart shape)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: commit through the NORMAL RAM path, then flush to disk
    // (the post-markFinished call the scheduler makes).
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hc", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        try hc.commit(&cache, &tokens, false);
        try testing.expect(hc.disk_dirty);
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 ("server restart"): fresh RAM cache, fresh tier over the same
    // root. The lookup must land on the SSD tier and restore the prefix.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hc", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount()); // RAM empty
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
        // Full match: identical re-issue semantics — truncate to len-1 and
        // re-forward the last token, exactly like a RAM full-match hit.
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
        try testing.expectEqual(@as(usize, 599), cache2.step);
        try testing.expectEqual(@as(usize, 599), moe_off);
        for (cache2.entries) |*e| {
            try testing.expect(e.initialized);
            try testing.expectEqual(@as(usize, 599), e.offset);
        }

        // Diverged-tail shape: shares the first 400 tokens, then differs.
        // Restore must land at 400 and leave the tail to prefill.
        var tokens_div: [700]u32 = undefined;
        for (&tokens_div, 0..) |*t, i| t.* = if (i < 400) tokens[i] else @intCast(i + 500_000);
        var cache3 = try KVCache.init(testing.allocator, 2);
        defer cache3.deinit();
        var moe_off3: usize = 0;
        const res3 = try hc2.lookupAndRestore(&cache3, &moe_off3, null, s, &tokens_div, false, 0, null, null);
        try testing.expect(!res3.full_match);
        try testing.expectEqual(@as(usize, 400), res3.matched);
        try testing.expectEqual(@as(usize, 400), cache3.step);
        // A diverged short prefix must read ONLY the chunks covering the
        // usable 400 positions (ceil(400/128) = 4), NOT the whole 600-token
        // stored entry (5 chunks). Loading the full entry to serve a short
        // shared prefix makes a diverged "hit" slower than a cold prefill.
        try testing.expectEqual(@as(u32, 4), hc2.disk.?.chunks_loaded_last);
    }
}

test "HotPrefixCache: dflash + mtp snapshots survive the SSD tier across a restart" {
    // A disk-tier restore forwards NO trunk layers, so state derived from
    // trunk hiddens (dflash context, MTP history) started EMPTY on every
    // disk hit — multi-turn across a restart drafted blind (the same
    // 92.6% → 66.5% acceptance class the RAM tier fixed). v4 persists both
    // in the entry's spec sidecar and restores them under the RAM tier's
    // exact clamp rule.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: commit with BOTH spec payloads through the normal RAM path,
    // then flush (the post-markFinished call the scheduler makes).
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec-hc", 0, 128);
        defer hc.deinit();

        var trunk = try KVCache.init(testing.allocator, 2);
        defer trunk.deinit();
        try testFillCache(&trunk, s, 2, 600);
        var assist = try KVCache.init(testing.allocator, 2);
        defer assist.deinit();
        try testFillCache(&assist, s, 2, 600);
        var hist = try KVCache.init(testing.allocator, 1);
        defer hist.deinit();
        try testFillCache(&hist, s, 1, 600);

        try hc.commitWithSsm(&trunk, &tokens, false, null, .{ .cache = &assist, .base_pos = 0 }, .{ .cache = &hist, .base_pos = 0 });
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expect(hc.disk.?.entries.items[0].spec_dflash != null);
        try testing.expect(hc.disk.?.entries.items[0].spec_mtp != null);
    }

    // Session 2 ("server restart"): RAM empty, disk serves the prefix AND
    // both spec snapshots, clamped to the trunk's matched length.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec-hc", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());

        var trunk2 = try KVCache.init(testing.allocator, 2);
        defer trunk2.deinit();
        var dfl = try KVCache.init(testing.allocator, 2);
        defer dfl.deinit();
        var mtp_dst = try KVCache.init(testing.allocator, 1);
        defer mtp_dst.deinit();
        var moe_off: usize = 0;
        var dbase: usize = 99;
        var mbase: usize = 99;
        const res = try hc2.lookupAndRestore(
            &trunk2,
            &moe_off,
            null,
            s,
            &tokens,
            false,
            0,
            .{ .cache = &dfl, .base_pos = &dbase },
            .{ .cache = &mtp_dst, .base_pos = &mbase },
        );
        // Full match: identical re-issue → trunk lands at 599; both spec
        // caches clamp to base + step == matched.
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
        try testing.expectEqual(@as(?usize, 0), res.dflash_base);
        try testing.expectEqual(@as(usize, 0), dbase);
        try testing.expectEqual(@as(usize, 599), dfl.step);
        try testing.expectEqual(@as(?usize, 0), res.mtp_base);
        try testing.expectEqual(@as(usize, 599), mtp_dst.step);

        // A geometry the persisted snap doesn't fit starts BLIND, never
        // wrong: a 3-layer dflash target declines.
        var trunk3 = try KVCache.init(testing.allocator, 2);
        defer trunk3.deinit();
        var dfl3 = try KVCache.init(testing.allocator, 3);
        defer dfl3.deinit();
        var moe3: usize = 0;
        var dbase3: usize = 42;
        const res3 = try hc2.lookupAndRestore(
            &trunk3,
            &moe3,
            null,
            s,
            &tokens,
            false,
            0,
            .{ .cache = &dfl3, .base_pos = &dbase3 },
            null,
        );
        try testing.expectEqual(@as(usize, 599), res3.matched);
        try testing.expectEqual(@as(?usize, null), res3.dflash_base);
        try testing.expectEqual(@as(usize, 0), dfl3.step);
        try testing.expectEqual(@as(usize, 42), dbase3); // untouched
    }
}

test "HotPrefixCache: RAM match at least as long as disk skips the SSD read" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-skip", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    const disk_uses_before = hc.disk.?.counter;

    // Same prompt again: the RAM entry covers it fully, so the disk tier's
    // LRU counter must not move (no restore happened).
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expect(res.full_match);
    try testing.expectEqual(disk_uses_before, hc.disk.?.counter);
}

test "HotPrefixCache: invalidation propagates to the disk tier" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-inv", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());

    hc.invalidateAll("test poison");
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
    try testing.expect(!hc.disk_dirty);
}

test "HotPrefixCache: findBestMatch isolates affine 4-bit from affine 8-bit" {
    // Regression for the cross-bit-width hit that crashed SDPA in
    // tests/test_kv_quant_per_request.sh. With Entry.scheme tracking only
    // the `Scheme` enum, `affine(4)` and `affine(8)` both matched as
    // `.affine` and a 4-bit snapshot would be restored into an 8-bit slot
    // → broadcast_shapes (1,H,1,64) vs (1,H,1,32) MLX abort. After moving
    // to a full-`KVQuantConfig` filter, the two are distinct keys and
    // can't alias.
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    const ids = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = kv_quant.KVQuantConfig.affine(4) },
        .last_used = 1,
        .quant_config = kv_quant.KVQuantConfig.affine(4),
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });

    const lookup_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    // Matching config (affine 4) hits the entry.
    const hit = cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(4)).?;
    try testing.expectEqual(@as(usize, 0), hit.idx);
    try testing.expectEqual(@as(usize, 5), hit.shared);
    // Same Scheme (.affine) but different bits MUST NOT hit — that's the
    // cross-scheme buffer-layout crash this filter guards against.
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(8)));
    // Dense query against an affine entry: also null (existing guarantee).
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.dense));
}

// ── Phase 3: two-tier hybrid restore (Qwen 3.5/3.6 GatedDeltaNet) ──

const conv_shape_pc = [_]c_int{ 1, 3, 8 };
const ssm_shape_pc = [_]c_int{ 1, 2, 4, 4 };

fn pcArange(s: mlx.mlx_stream, shape: []const c_int, base: f64) mlx.mlx_array {
    var count: f64 = 1;
    for (shape) |d| count *= @floatFromInt(d);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    _ = mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_reshape(&out, flat, shape.ptr, @intCast(shape.len), s);
    _ = mlx.mlx_array_eval(out);
    return out;
}

fn pcSsmVal(arr: mlx.mlx_array, idx: usize, s: mlx.mlx_stream) f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_astype(&f, arr, .float32, s);
    _ = mlx.mlx_array_eval(f);
    return mlx.mlx_array_data_float32(f).?[idx];
}

fn pcBuildHybrid(s: mlx.mlx_stream, conv_base: f64, ssm_base: f64) [3]SSMCacheEntry {
    return .{
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base), .ssm_state = pcArange(s, &ssm_shape_pc, ssm_base), .initialized = true },
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base + 10_000), .ssm_state = mlx.mlx_array_new(), .initialized = true },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

fn pcFreeHybrid(e: *[3]SSMCacheEntry) void {
    for (e) |*x| {
        _ = mlx.mlx_array_free(x.conv_state);
        _ = mlx.mlx_array_free(x.ssm_state);
    }
}

fn pcEmptySsm() [3]SSMCacheEntry {
    return .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

// A new image changes the KV only when its dynamic placeholder rows are
// forwarded. The text prefix before that boundary remains valid even though
// the media hash changes. Hybrid models must restore the last checkpoint at
// or before the boundary, never a later checkpoint whose SSM state has seen
// the old pixels.
test "HotPrefixCache: hybrid lookup reuses only the prefix before changed media" {
    const s = mlx.gpuStream();
    const media_start: usize = 8;
    const cached_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 900, 900, 20, 21 };
    const lookup_tokens = cached_tokens;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var source_cache = try KVCache.init(testing.allocator, 3);
    defer source_cache.deinit();
    try testFillCache(&source_cache, s, 3, cached_tokens.len);
    var source_ssm = pcBuildHybrid(s, 123.0, 456.0);
    defer pcFreeHybrid(&source_ssm);
    const checkpoints = try testing.allocator.alloc(SSMCheckpoint, 2);
    checkpoints[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &source_ssm, media_start, s);
    checkpoints[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &source_ssm, media_start + 2, s);
    try hc.commitWithMediaState(
        &source_cache,
        &cached_tokens,
        false,
        0x1111,
        media_start,
        checkpoints,
        null,
        null,
    );

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target_ssm = pcEmptySsm();
    defer pcFreeHybrid(&target_ssm);
    var moe_off: usize = 0;
    const result = try hc.lookupAndRestoreWithMedia(
        &target_cache,
        &moe_off,
        &target_ssm,
        s,
        &lookup_tokens,
        false,
        0x2222,
        media_start,
        null,
        null,
    );

    try testing.expectEqual(media_start, result.matched);
    try testing.expectEqual(media_start, target_cache.step);
    try testing.expectEqual(media_start, moe_off);
    try testing.expectEqual(@as(f32, 123.0), pcSsmVal(target_ssm[0].conv_state, 0, s));
}

// A vision turn can move the current image span when the same image becomes
// conversation history. The newest entry then has the longest raw token match
// (ending exactly at the old image boundary), but its first SSM checkpoint can
// sit just AFTER that boundary. An older entry for the same pixels may have a
// slightly shorter token match with a usable checkpoint. Picking by raw token
// match alone turns that recoverable case into a full cold prefill.
test "HotPrefixCache: hybrid lookup falls back to the best restorable RAM entry" {
    const s = mlx.gpuStream();
    const vision_key: u64 = 0xdecaf;
    const older_tokens = [_]u32{ 1, 2, 3, 4, 5, 90, 91, 92, 93, 94 };
    const newer_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 80, 81, 82 };
    const lookup_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 70, 71, 72 };

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var older_cache = try KVCache.init(testing.allocator, 3);
    defer older_cache.deinit();
    try testFillCache(&older_cache, s, 3, older_tokens.len);
    var older_ssm = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&older_ssm);
    const older_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    older_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &older_ssm, 4, s);
    try hc.commitWithState(&older_cache, &older_tokens, false, vision_key, older_cps, null, null);

    var newer_cache = try KVCache.init(testing.allocator, 3);
    defer newer_cache.deinit();
    try testFillCache(&newer_cache, s, 3, newer_tokens.len);
    var newer_ssm = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&newer_ssm);
    const newer_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    // The raw match with this entry is 7, so this checkpoint cannot restore it.
    newer_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &newer_ssm, 8, s);
    try hc.commitWithState(&newer_cache, &newer_tokens, false, vision_key, newer_cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target_ssm = pcEmptySsm();
    defer pcFreeHybrid(&target_ssm);
    var moe_off: usize = 0;
    const result = try hc.lookupAndRestore(
        &target_cache,
        &moe_off,
        &target_ssm,
        s,
        &lookup_tokens,
        false,
        vision_key,
        null,
        null,
    );

    // The newer 7-token raw match is unusable; the older checkpoint at 4 is
    // still vastly better than a cold prefill and must win the hybrid lookup.
    try testing.expectEqual(@as(usize, 4), result.matched);
    try testing.expectEqual(@as(usize, 4), target_cache.step);
    try testing.expectEqual(@as(usize, 4), moe_off);
    try testing.expectEqual(@as(f32, 100.0), pcSsmVal(target_ssm[0].conv_state, 0, s));
}

test "HotPrefixCache: hybrid SSM state restores from the SSD tier across a restart" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: hybrid commit (KV + two SSM checkpoints) through the RAM
    // path, then the post-markFinished flush the scheduler makes.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 600);

        var src256 = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src256);
        var src512 = pcBuildHybrid(s, 300.0, 700.0);
        defer pcFreeHybrid(&src512);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s);
        cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
        // commitWithSsm takes ownership of `cps`.
        try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
        try testing.expect(hc.disk_dirty);
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 ("restart"): fresh RAM cache + fresh tier over the same root.
    // A hybrid lookup must restore BOTH KV and SSM state from disk at the
    // highest checkpoint ≤ the match.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var ssm2 = pcEmptySsm();
        defer pcFreeHybrid(&ssm2);
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
        // Highest checkpoint ≤ 600 is 512 — never a full match on hybrid.
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expectEqual(@as(usize, 512), cache2.step);
        try testing.expectEqual(@as(usize, 512), moe_off);
        // SSM state at pos 512 installed (conv base 300 / ssm base 700).
        try testing.expect(ssm2[0].initialized);
        try testing.expectEqual(@as(f32, 300.0), pcSsmVal(ssm2[0].conv_state, 0, s));
        try testing.expectEqual(@as(f32, 700.0), pcSsmVal(ssm2[0].ssm_state, 0, s));
        try testing.expect(ssm2[1].ssm_state.ctx == null);
        try testing.expect(!ssm2[2].initialized);

        // Diverged tail: shares the first 400 tokens → clamps to the largest
        // checkpoint ≤ 400, which is 256.
        var tokens_div: [700]u32 = undefined;
        for (&tokens_div, 0..) |*t, i| t.* = if (i < 400) tokens[i] else @intCast(i + 500_000);
        var cache3 = try KVCache.init(testing.allocator, 3);
        defer cache3.deinit();
        var ssm3 = pcEmptySsm();
        defer pcFreeHybrid(&ssm3);
        var moe_off3: usize = 0;
        const res3 = try hc2.lookupAndRestore(&cache3, &moe_off3, &ssm3, s, &tokens_div, false, 0, null, null);
        try testing.expect(!res3.full_match);
        try testing.expectEqual(@as(usize, 256), res3.matched);
        try testing.expectEqual(@as(usize, 256), cache3.step);
        try testing.expectEqual(@as(f32, 100.0), pcSsmVal(ssm3[0].conv_state, 0, s));
        try testing.expectEqual(@as(f32, 500.0), pcSsmVal(ssm3[0].ssm_state, 0, s));
    }
}

test "HotPrefixCache: hybrid RAM match at least as good as disk skips the SSD read" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hybskip", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, 600);
    var src512 = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&src512);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
    try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
    hc.flushPendingDisk(s);
    const disk_uses_before = hc.disk.?.counter;

    // Same prompt again: the RAM entry's checkpoint at 512 ties the disk's, so
    // the disk advantage gate fails and the SSD read is skipped (counter
    // unchanged) — the RAM path serves the restore.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var ssm2 = pcEmptySsm();
    defer pcFreeHybrid(&ssm2);
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 512), res.matched);
    try testing.expectEqual(disk_uses_before, hc.disk.?.counter);
    // RAM restore installed the SSM state just the same.
    try testing.expectEqual(@as(f32, 300.0), pcSsmVal(ssm2[0].conv_state, 0, s));
}

// Regression: an entry that is EXTENDED in place must not accumulate SSM
// checkpoints without bound, and bounding them must not collapse the survivors
// onto the end of the prompt. `generate.zig` caps what a single prefill
// captures, but the replace path merges the previous entry's checkpoints with
// this turn's, and nothing re-applied a cap to the merged list — so an agent
// conversation gained one checkpoint per turn forever. Observed on
// Qwen3.8-Flash-Next (36 GDN layers): 31237 MB of SSM state in ONE entry under
// `--ssm-checkpoint-max 8`, which starved the prompt-admission check.
//
// Capping oldest-first fixes the size but leaves every survivor near the end,
// and a request diverging earlier then pays a full cold prefill
// ("hybrid miss (no checkpoint <= 16382 of 178509)", 415 s). So the cap thins
// the interior and keeps a spread.
test "HotPrefixCache: replace path bounds SSM checkpoints and keeps them spread" {
    const s = mlx.gpuStream();

    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 3);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssm_checkpoint_max = 4;
    defer hc.deinit();

    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    // Turn 1: four checkpoints at 100..400 over a 450-token prefix.
    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 450);
    const cps1 = try testing.allocator.alloc(SSMCheckpoint, 4);
    for (cps1, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    }
    try hc.commitWithSsm(&c1, tokens[0..450], false, cps1, null, null);
    try testing.expectEqual(@as(usize, 4), hc.entries.items[0].ssm_checkpoints.?.len);

    // Turn 2 extends that exact prefix and brings four more at 500..800.
    // Merged that is eight against a cap of four.
    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, 900);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 4);
    for (cps2, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i + 4], (i + 5) * 100, s);
    }
    try hc.commitWithSsm(&c2, tokens[0..900], false, cps2, null, null);

    // Extended, not appended: still one entry, and the cap holds.
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const kept = hc.entries.items[0].ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 4), kept.len);

    // The first and the newest always survive, the interior is thinned to keep
    // coverage. Oldest-first would have left 500/600/700/800, and a request
    // matching at 150 would then have nothing at or below it to restore from.
    try testing.expectEqual(@as(usize, 100), kept[0].pos);
    try testing.expectEqual(@as(usize, 300), kept[1].pos);
    try testing.expectEqual(@as(usize, 500), kept[2].pos);
    try testing.expectEqual(@as(usize, 800), kept[3].pos);
}

test "HotPrefixCache: byte budget rejects an oversized sole entry and preserves a smaller prefix" {
    const s = mlx.gpuStream();

    var a = try KVCache.init(testing.allocator, 2);
    defer a.deinit();
    try testFillCache(&a, s, 2, 8);
    // KV buffers grow in 256-token chunks, so the sizes must straddle a chunk
    // boundary for the two entries' snapshot bytes to differ.
    var b = try KVCache.init(testing.allocator, 2);
    defer b.deinit();
    try testFillCache(&b, s, 2, 600);

    var toks_a: [8]u32 = undefined;
    for (&toks_a, 0..) |*t2, i| t2.* = @intCast(i + 1);
    var toks_b: [600]u32 = undefined;
    for (&toks_b, 0..) |*t2, i| t2.* = @intCast(i + 100);

    var a_snap = try a.snapshot();
    defer a_snap.deinit();
    var b_snap = try b.snapshot();
    defer b_snap.deinit();
    const small = HotPrefixCache.snapshotBytes(&a_snap);
    const big = HotPrefixCache.snapshotBytes(&b_snap);
    try testing.expect(big > small);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, small);
    defer hc.deinit();

    try hc.commit(&a, &toks_a, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // B extends A, so this exercises the replacement path. It is too large
    // for the cap: the byte budget still holds — but as a TRIM (#330), not a
    // decline. The retained prefix is longer than A's, shorter than B's.
    @memcpy(toks_b[0..toks_a.len], &toks_a);
    try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const kept_len = hc.entries.items[0].tokens.len;
    try testing.expect(kept_len >= MIN_CANCELLED_COMMIT_TOKENS);
    try testing.expect(kept_len < toks_b.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // An empty cache retains the same trimmed prefix from the oversized
    // candidate rather than staying empty (#330: the pre-fix decline held the
    // cap by holding zero bytes).
    hc.invalidateAll("test");
    try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(kept_len, hc.entries.items[0].tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
}

// ── Issue #330: the oversized-entry decline is a cliff, not a cap ──

/// Test-local: dense per-token KV bytes of a snapshot (k + v across layers).
fn pcRowBytes(snap: *const transformer_mod.KVCacheSnapshot) u64 {
    var total: u64 = 0;
    for (snap.entries) |e| {
        if (!e.initialized) continue;
        const rows: u64 = @intCast(mlx.mlx_array_shape(e.keys)[2]);
        if (rows == 0) continue;
        const kb = @as(u64, mlx.mlx_array_size(e.keys)) * @as(u64, mlx.mlx_array_itemsize(e.keys));
        const vb = @as(u64, mlx.mlx_array_size(e.values)) * @as(u64, mlx.mlx_array_itemsize(e.values));
        total += (kb + vb) / rows;
    }
    return total;
}

/// Test-local: assert rows [0:rows] of `a` and `b` (axis 2) are identical.
fn pcExpectRowsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array, rows: usize) !void {
    var sliced_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced_a);
    var sliced_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced_b);
    const sh_a = mlx.mlx_array_shape(a);
    const sh_b = mlx.mlx_array_shape(b);
    const start = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    const stop_a = [_]c_int{ sh_a[0], sh_a[1], @intCast(rows), sh_a[3] };
    const stop_b = [_]c_int{ sh_b[0], sh_b[1], @intCast(rows), sh_b[3] };
    try mlx.check(mlx.mlx_slice(&sliced_a, a, &start, 4, &stop_a, 4, &strides, 4, s));
    try mlx.check(mlx.mlx_slice(&sliced_b, b, &start, 4, &stop_b, 4, &strides, 4, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_equal(&eq, sliced_a, sliced_b, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, eq, false, s));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, all));
    try testing.expect(ok);
}

test "HotPrefixCache: oversized entry trims to the longest prefix that fits (#330)" {
    const s = mlx.gpuStream();

    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, 600);
    var toks: [600]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var probe = try src.snapshot();
    defer probe.deinit();
    const row = pcRowBytes(&probe);
    // Room for exactly 400 tokens — over the 256-token commit floor, under
    // the 600-token candidate.
    const budget = row * 400;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    try hc.commit(&src, &toks, false);

    // The cliff: pre-fix this declines outright and the cache stays empty.
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // The trimmed prefix restores: same prompt matches 400 tokens.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&dst, &moe_off, null, s, &toks, false, 0, null, null);
    try testing.expect(!res.full_match);
    try testing.expectEqual(@as(usize, 400), res.matched);
    try testing.expectEqual(@as(usize, 400), dst.step);
    for (dst.entries, src.entries) |*d, *e| {
        try testing.expectEqual(@as(usize, 400), d.offset);
        // Trimmed rows must be the SOURCE's rows — a slice-math bug here
        // serves a wrong prefix as a cache hit.
        try pcExpectRowsEqual(s, d.keys, e.keys, 400);
        try pcExpectRowsEqual(s, d.values, e.values, 400);
    }
}

test "HotPrefixCache: trimmed entry is one-shot — a covered re-commit keeps the resident copy (#330)" {
    const s = mlx.gpuStream();

    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, 600);
    var toks: [700]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var probe = try src.snapshot();
    defer probe.deinit();
    const budget = pcRowBytes(&probe) * 400;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    try hc.commit(&src, toks[0..600], false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    const resident_keys_ctx = hc.entries.items[0].snapshot.entries[0].keys.ctx;

    // Next turn: the conversation grew, the trim target did not. Re-copying
    // an identical prefix every turn would be a per-turn multi-GB memcpy.
    var src2 = try KVCache.init(testing.allocator, 2);
    defer src2.deinit();
    try testFillCache(&src2, s, 2, 700);
    try hc.commit(&src2, &toks, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    try testing.expectEqual(resident_keys_ctx, hc.entries.items[0].snapshot.entries[0].keys.ctx);
}

test "HotPrefixCache: oversized hybrid entry trims to the highest checkpoint that fits (#330)" {
    const s = mlx.gpuStream();

    var toks: [900]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 3);

    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 900);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 8);
    for (cps, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    }

    var probe = try c1.snapshot();
    defer probe.deinit();
    const row = pcRowBytes(&probe);
    var cps_at_or_below_500: u64 = 0;
    for (cps[0..5]) |*c| cps_at_or_below_500 += transformer_mod.ssmCheckpointBytes(c);
    // Exactly the cost of a 500-token prefix plus its five checkpoints: the
    // trim point must be a RESTORABLE position, so 500 is the answer even
    // though a few more raw tokens would fit.
    const budget = row * 500 + cps_at_or_below_500;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    hc.ssm_checkpoint_max = 8;
    defer hc.deinit();
    try hc.commitWithSsm(&c1, &toks, false, cps, null, null);

    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 500), e.tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    // Checkpoints past the trim point are gone; the one AT it survives.
    const kept = e.ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 5), kept.len);
    try testing.expectEqual(@as(usize, 500), kept[kept.len - 1].pos);
}

test "HotPrefixCache: oversized hybrid entry with no checkpoint under budget declines (#330)" {
    const s = mlx.gpuStream();

    var toks: [900]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 3);

    var hyb = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&hyb);

    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 900);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &hyb, 800, s);

    var probe = try c1.snapshot();
    defer probe.deinit();
    // Sole checkpoint sits at 800; a 100-token budget cannot retain a
    // restorable hybrid prefix, so the commit declines like before.
    const budget = pcRowBytes(&probe) * 100;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    try hc.commitWithSsm(&c1, &toks, false, cps, null, null);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
}

/// qwen4-shaped hybrid: layer 0 is a QSA full-attention layer (no conv/ssm,
/// `aux_state` = `[1, rows, 8]` indexer key history), layer 1 GDN, layer 2 idle.
fn pcBuildQsaHybrid(s: mlx.mlx_stream, rows: c_int, conv_base: f64) [3]SSMCacheEntry {
    const aux_shape = [_]c_int{ 1, rows, 8 };
    return .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = pcArange(s, &aux_shape, 0.0), .qsa_ratio = 4 },
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base), .ssm_state = mlx.mlx_array_new(), .initialized = true },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

fn pcFreeQsaHybrid(e: *[3]SSMCacheEntry) void {
    for (e) |*x| {
        if (x.conv_state.ctx != null) _ = mlx.mlx_array_free(x.conv_state);
        if (x.ssm_state.ctx != null) _ = mlx.mlx_array_free(x.ssm_state);
        if (x.aux_state.ctx != null) _ = mlx.mlx_array_free(x.aux_state);
        if (x.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(x.qsa_pooled);
    }
}

test "HotPrefixCache: a QSA arch restore with no indexer history is a miss, never a poisoned entry" {
    // A snap without QSA history (a cancel handoff whose attach failed, an
    // old on-disk entry) used to restore aux-less; the next prefill then died
    // in qsaMaskFromQk with QsaHistoryGap on EVERY turn on that prefix. A
    // QSA arch treats "no history after restore" like "no checkpoint".
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const lookup_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 70, 71 };
    for ([_]bool{ false, true }) |with_history| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        var live = pcBuildQsaHybrid(s, 10, 100.0);
        defer pcFreeQsaHybrid(&live);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 4, s);
        if (with_history) try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup_tokens, false, 0, null, null);
        if (with_history) {
            try testing.expectEqual(@as(usize, 4), r.matched);
            // Sliced to the snap's position, not the live length.
            try testing.expectEqual(@as(c_int, 4), mlx.getShape(target[0].aux_state)[1]);
        } else {
            try testing.expectEqual(@as(usize, 0), r.matched);
            try testing.expectEqual(@as(usize, 0), moe_off);
            try testing.expect(target[0].aux_state.ctx == null);
        }
    }
}

test "HotPrefixCache: prefix-extend keeps ONE QSA history across turns" {
    // The replace path inherits the old entry's checkpoints. Its latest snap
    // carried the full history and the new latest gets another one: one copy
    // per committed turn, the leak the stride fix closed by another door.
    const s = mlx.gpuStream();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, t1.len);
    var l1 = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&l1);
    const cps1 = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps1[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &l1, 4, s);
    try transformer_mod.attachQsaHistoryToLatest(cps1, &l1, s);
    try hc.commitWithState(&c1, &t1, false, 0, cps1, null, null);

    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, t2.len);
    var l2 = pcBuildQsaHybrid(s, 14, 200.0);
    defer pcFreeQsaHybrid(&l2);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps2[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &l2, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps2, &l2, s);
    try hc.commitWithState(&c2, &t2, false, 0, cps2, null, null);

    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const merged = hc.entries.items[0].ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 2), merged.len);
    try testing.expect(!transformer_mod.checkpointHasQsaHistory(&merged[0]));
    try testing.expect(transformer_mod.checkpointHasQsaHistory(&merged[1]));
    try testing.expectEqual(@as(c_int, 8), mlx.getShape(merged[1].layers[0].aux_state)[1]);
}

test "HotPrefixCache: replace path sheds inherited checkpoints instead of evicting its own entry (#330)" {
    const s = mlx.gpuStream();

    var toks: [900]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 3);

    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    // Turn 1: 450 tokens, checkpoints at 100..400.
    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 450);
    const cps1 = try testing.allocator.alloc(SSMCheckpoint, 4);
    var cps1_bytes: u64 = 0;
    for (cps1, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
        cps1_bytes += transformer_mod.ssmCheckpointBytes(c);
    }

    // Turn 2 extends to 900 with checkpoints at 500..800.
    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, 900);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 4);
    var c2_bytes: u64 = 0;
    for (cps2, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i + 4], (i + 5) * 100, s);
        c2_bytes += transformer_mod.ssmCheckpointBytes(c);
    }
    var c2_snap = try c2.snapshot();
    c2_bytes += HotPrefixCache.snapshotBytes(&c2_snap);
    c2_snap.deinit();

    // Turn 2 alone fits the budget; turn 2 plus the INHERITED turn-1
    // checkpoints does not. The pre-check cannot price the inheritance, so
    // pre-fix the post-merge loop evicted the sole, just-updated entry —
    // commit → evict everything → cold prefill, every turn.
    const budget = c2_bytes + cps1_bytes / 2;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    hc.ssm_checkpoint_max = 8;
    defer hc.deinit();
    try hc.commitWithSsm(&c1, toks[0..450], false, cps1, null, null);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try hc.commitWithSsm(&c2, &toks, false, cps2, null, null);

    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 900), e.tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    // Shedding trimmed the checkpoint list, it did not empty it.
    try testing.expect(e.ssm_checkpoints.?.len >= 1);
}

test "HotPrefixCache: a failed commit still frees the checkpoints it was handed (#330 adjacent)" {
    const s = mlx.gpuStream();

    var src = try KVCache.init(testing.allocator, 1);
    defer src.deinit();
    try testFillCache(&src, s, 1, 8);

    // fail_index 0: the first cache-side allocation (the tokens dupe) fails.
    // Ownership of the checkpoints transfers to the cache UNCONDITIONALLY —
    // the scheduler's catch arm frees nothing (scan-pinned in scheduler.zig;
    // pre-fix it freed too, a double free with a different allocator). The
    // cache's error paths therefore MUST free the slice; std.testing.allocator
    // flags both the leak and a double free.
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    var hc = HotPrefixCache.initWithMem(failing.allocator(), 1, 0);
    defer hc.deinit();

    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = .{ .pos = 4, .layers = try testing.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, 0) };
    var toks = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try testing.expectError(error.OutOfMemory, hc.commitWithMediaState(&src, &toks, false, 0, null, cps, null, null));
    // No frees here: the cache owns the checkpoints on every outcome.
}
