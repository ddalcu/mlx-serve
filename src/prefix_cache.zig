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

/// Why a lookup that found a real raw token match still restored nothing.
/// `findBestRestorableMatch` `continue`s every candidate whose highest SSM
/// checkpoint sits past the shared prefix, so a hybrid lookup can return null
/// with a 393k-token raw match behind it — and the `match == null` arm used to
/// log NOTHING, leaving a 560 s cold prefill with no `[hot-cache]` line at all.
/// Pure so the policy is unit-testable without a cache.
pub const MissKind = enum {
    /// Nothing worth naming: no key-compatible entry, or a shared prefix under
    /// the commit floor. An ordinary cold start; stays quiet.
    cold,
    /// Entries shared a real prefix and not one of them could restore it.
    /// This is the expensive miss and it owes a line.
    no_checkpoint,
};

/// What `findBestRestorableMatch` saw before its restorability filter ran:
/// how many key-compatible entries it considered and the longest RAW token
/// match among them. The filter's `continue`s destroy both, which is why a
/// hybrid miss could not name itself.
pub const MatchProbe = struct { candidates: usize = 0, best_raw: usize = 0 };

pub fn missKind(candidates: usize, best_raw: usize) MissKind {
    if (candidates == 0) return .cold;
    if (best_raw < MIN_CANCELLED_COMMIT_TOKENS) return .cold;
    return .no_checkpoint;
}

/// Why an oversized commit retained nothing; three outcomes used to print one identical line.
pub const TrimDecline = enum {
    no_restorable_prefix,
    snapshot_copy_failed,
    checkpoint_list_copy_failed,

    pub fn reason(self: TrimDecline) []const u8 {
        return switch (self) {
            .no_restorable_prefix => "no restorable prefix fits the budget",
            .snapshot_copy_failed => "every trimmed KV copy failed",
            .checkpoint_list_copy_failed => "the trimmed checkpoint list copy failed",
        };
    }
};

/// Outcome of a cache commit, so callers can log the truth. A budget
/// decline used to exit `commitWithMediaState` via plain return and the
/// scheduler logged "committed N/M" for it (live 2026-09-07: a 122k agent
/// session cold-prefilled ~95k tokens per retry while the log claimed
/// cancelled-prefill commits every time).
pub const CommitStatus = union(enum) {
    /// Entry committed (inserted or replaced) at this many tokens — the
    /// post-trim EFFECTIVE length, never the candidate's forwarded length.
    ok: usize,
    /// The budget decline kept a resident entry that already covers this
    /// many tokens; the longer candidate was discarded (details logged).
    kept_resident: usize,
    /// Declined: oversized with no restorable trim under the budget, or the
    /// trim copy failed. The decline site logs the reason.
    declined,
};

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
    /// Did this restore check out its entry (`checkoutEligible`)? Only then does the first
    /// append donate in place; every other restore is a refcount share copied by that append.
    checked_out: bool = false,
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
    /// Workload the request belonged to (`server.requestCacheKey`, 0 = anonymous).
    /// Eviction is fair across keys: the key holding the most entries pays first.
    cache_key: u64 = 0,
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
    /// Transient, one `spillIdleEntries` pass only: did this pass leave a durable copy on the SSD tier?
    spill_durable: bool = false,
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
    /// Restore by move: the slot that took ownership of this entry's KV buffers (`KVCache.adopt`).
    /// While set the snapshot holds empty handles and the entry is invisible to every other
    /// reader. Cleared by the commit that replaces it; dropped at slot end otherwise.
    checked_out_by: ?usize = null,
    /// Has the slot's append donated these buffers in place (`donateCheckout`)? Until then the
    /// entry's handles are live and a slot that ends hands the entry back; only a donated checkout is dropped.
    checkout_donated: bool = false,
};

/// What a spec-snap adoption may do, decided before any mlx call.
pub const SpecAdopt = union(enum) {
    /// Nothing to adopt: no payload, or the snap does not cover the reused range.
    skip,
    /// A KV-only spec cache (dflash context, sidecar MTP head).
    kv_only: usize,
    /// The qwen4_exp in-checkpoint head: KV + QSA aux together.
    head: usize,
    /// A head target met a payload with no QSA half (pre-v5 sidecar). Head-only miss.
    decline_head_no_history,
};

pub fn specAdoptPlan(base_pos: usize, snap_step: usize, matched: usize, has_head_target: bool, has_head_aux: bool) SpecAdopt {
    if (base_pos > matched) return .skip;
    const want = matched - base_pos;
    if (want > snap_step) return .skip;
    if (!has_head_target) return .{ .kv_only = want };
    if (!has_head_aux) return .decline_head_no_history;
    return .{ .head = want };
}

var ssd_first_env_cached: ?bool = null;
/// Test/bench override for `ssdFirstEnabled()`. Null = read the environment.
pub var ssd_first_override: ?bool = null;

/// Resolve this module's lazily-cached env reads once, from the main thread.
pub fn warmEnvCaches() void {
    _ = ssdFirstEnabled();
    _ = restoreMoveEnabled();
}

/// SSD-first prefix cache mode (`MLX_SERVE_PREFIX_SSD_FIRST=0` restores RAM-first); armed only
/// where `ModelConfig.ssdFirstCapable()`.
pub fn ssdFirstEnabled() bool {
    if (ssd_first_override) |v| return v;
    if (ssd_first_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_PREFIX_SSD_FIRST") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    ssd_first_env_cached = v;
    return v;
}

var restore_move_env_cached: ?bool = null;
/// Test/bench override for `restoreMoveEnabled()`. Null = read the environment.
pub var restore_move_override: ?bool = null;

/// Restore by move. `MLX_SERVE_RESTORE_MOVE=0` restores the refcount share whose first append
/// copies the whole prefix. Armed only where `HotPrefixCache.ssd_first` is.
pub fn restoreMoveEnabled() bool {
    if (restore_move_override) |v| return v;
    if (restore_move_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_RESTORE_MOVE") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    restore_move_env_cached = v;
    return v;
}

/// The SSD-first predicate: arch, env switch, AND a disk tier. Without the tier the mode used
/// to arm with nowhere to spill and a budget floor sized for a tier that did not exist. The
/// budget resolver asks `--prefix-cache-disk > 0`, the arming asks `disk != null`.
pub fn ssdFirstActive(config: *const model_mod.ModelConfig, has_disk: bool) bool {
    return has_disk and config.ssdFirstCapable() and ssdFirstEnabled();
}

/// What the live cache held at commit time, captured before the RAM byte-budget trim: the
/// disk tier gets the full prefix even when RAM keeps a trimmed one. Refcount-shared.
const PendingDiskFlush = struct {
    snapshot: KVCacheSnapshot,
    tokens: []u32,
    has_tools: bool,
    ssm_cps: ?[]SSMCheckpoint = null,
    dflash: ?DflashSnap = null,
    mtp: ?DflashSnap = null,

    fn deinit(self: *PendingDiskFlush, allocator: std.mem.Allocator) void {
        self.snapshot.deinit();
        allocator.free(self.tokens);
        if (self.ssm_cps) |cps| {
            for (cps) |*cp| cp.deinit(allocator);
            allocator.free(cps);
        }
        if (self.dflash) |*d| d.deinit();
        if (self.mtp) |*m| m.deinit();
    }
};

/// A committed speculative-side cache: the snapshot plus the absolute trunk
/// position its index 0 represents (nonzero when the committing request was
/// itself a cache hit). Shared by the DFlash assistant context and the MTP
/// committed-history cache — identical semantics, two Entry fields.
pub const DflashSnap = struct {
    snapshot: KVCacheSnapshot,
    base_pos: usize,
    /// qwen4_exp MTP head only: the head's QSA aux entry and the absolute position of its key
    /// row 0. A snap with one half and not the other is declined.
    head_aux: ?transformer_mod.SSMCacheEntrySnapshot = null,
    head_pos_base: c_int = 0,

    pub fn deinit(self: *DflashSnap) void {
        self.snapshot.deinit();
        if (self.head_aux) |*a| transformer_mod.ssmSnapshotDeinit(a);
        self.head_aux = null;
    }
};

/// What `commitWithState` reads to build a `DflashSnap`. `head` is set only by the qwen4_exp MTP head.
pub const DflashCommit = struct {
    cache: *const KVCache,
    base_pos: usize,
    head: ?*const SSMCacheEntry = null,
    head_pos_base: c_int = 0,
};

/// Where `lookupAndRestore` puts a restored assistant context. `base_pos` is
/// written on every path so the caller can build `DflashCtx` from it. `head` is the qwen4_exp
/// Transformer owning the in-checkpoint MTP head; the adoption then goes through `qwen4MtpAdopt`.
pub const DflashTarget = struct {
    cache: *KVCache,
    base_pos: *usize,
    head: ?*transformer_mod.Transformer = null,
};

/// What `evictLruToAdmit` gave up, and whether it was enough.
pub const EvictionReport = struct {
    entries: usize = 0,
    /// Bytes the allocator actually got back (live delta).
    bytes: u64 = 0,
    /// Bytes the cache had billed for those entries; larger whenever a snapshot is refcount-shared.
    accounted_bytes: u64 = 0,
    /// The pass stopped because an eviction returned nothing (shared with a live request).
    shared_stop: bool = false,
    /// False = the cache is empty (or down to this request's entry) and the request still does not fit.
    admitted: bool = false,
};

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
    /// `last_used` of the entry the current request restored from; `evictLruToAdmit` refuses to evict it.
    last_restored_used: ?u64 = null,
    /// The arch keeps a QSA indexer history beside its SSM state (qwen4_exp).
    /// A restore that leaves the live entries without it cannot prefill —
    /// `qsaMaskFromQk` errors on every turn on that prefix — so it is a MISS.
    qsa_history_required: bool = false,
    /// Checkpoint-retention policy, mirrored once at wiring from `ModelConfig.longCtxGated()`
    /// (this struct never sees a ModelConfig). The default is the previous behaviour.
    cp_thin: transformer_mod.ThinPolicy = .min_span,
    /// SSD-first mode; set by the scheduler at load.
    ssd_first: bool = false,
    /// SSD-first: the RAM allowance for idle entries (the resolved `--prefix-cache-mem`).
    /// `spillIdleEntries` evicts only past it; 0 = nothing idle stays resident.
    ssd_idle_mem: u64 = 0,
    /// The live-cache state of the most recent commit, flushed instead of the (possibly trimmed) RAM entry.
    pending_disk: ?PendingDiskFlush = null,
    /// Byte floor for `restoreWouldPinEntry`; a field so a test can reproduce the live shape.
    restore_pin_min_bytes: u64 = RESTORE_PIN_MIN_BYTES,

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
        if (self.pending_disk) |*p| p.deinit(self.allocator);
        self.pending_disk = null;
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
        const want = switch (specAdoptPlan(snap.base_pos, snap.snapshot.step, matched, t.head != null, snap.head_aux != null)) {
            .skip => {
                log.info("  [hot-cache] {s} not adopted: want {d} tokens from base {d}, snap holds {d} (matched {d})\n", .{ what, matched -| snap.base_pos, snap.base_pos, snap.snapshot.step, matched });
                return null;
            },
            .decline_head_no_history => {
                log.info("  [qwen4] MTP head restore declined (snapshot carries no QSA history) — head starts blind\n", .{});
                return null;
            },
            .kv_only, .head => |w| w,
        };
        // qwen4_exp in-checkpoint head: KV + QSA aux adopt together or not at all.
        if (t.head) |xfm| {
            const aux = &snap.head_aux.?;
            xfm.qwen4MtpAdopt(&snap.snapshot, aux, snap.head_pos_base, want) catch |err| {
                log.warn("  [qwen4] MTP head restore declined ({s}) — head starts blind\n", .{@errorName(err)});
                return null;
            };
            t.base_pos.* = snap.base_pos;
            log.info("  [qwen4] MTP head restored ({d} tokens from base {d})\n", .{ want, snap.base_pos });
            return snap.base_pos;
        }
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
        // transient snapshot is freed right after. A pre-v5 sidecar carries no head half, so a
        // qwen4 head target declines it.
        var snap = DflashSnap{
            .snapshot = loaded.snap,
            .base_pos = loaded.base,
            .head_aux = loaded.head_aux,
            .head_pos_base = loaded.head_pos_base,
        };
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

    /// The rows a restore will deliver, which is not the rows it matched: a hybrid restore is
    /// clamped to its highest SSM checkpoint at or below the match, so the lien test must weigh this.
    pub fn deliverableShare(cps: ?[]const SSMCheckpoint, hybrid: bool, shared: usize) usize {
        if (!hybrid) return shared;
        const list = cps orelse return 0;
        const cp = highestCheckpointAtOrBelow(list, shared) orelse return 0;
        return cp.pos;
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
            transformer_mod.ssmFreeQsaState(ssm);
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

    /// Resident bytes of a speculative-side snap: the KV plus (qwen4_exp) the head's QSA aux half.
    fn specSnapBytes(snap: *const DflashSnap) u64 {
        var total = snapshotBytes(&snap.snapshot);
        if (snap.head_aux) |a| {
            // Only the two the head owns: its `conv_state`/`ssm_state` are empty handles.
            inline for (.{ a.aux_state, a.qsa_pooled }) |arr| {
                if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
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
                // A dense snapshot leaves the quant handles as empty 0-dim arrays; the ndim check makes axis 2 a fact.
                if (arr.ctx != null and mlx.mlx_array_ndim(arr) > 2) {
                    const rows: u64 = @intCast(mlx.mlx_array_shape(arr)[2]);
                    if (rows > 0) {
                        total += (@as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr))) / rows;
                    }
                }
            }
        }
        return total;
    }

    /// Positions printed by the trim-inputs line before it elides; the count is always exact.
    const TRIM_LOG_MAX_POS: usize = 32;

    fn appendTrimFmt(buf: []u8, n: *usize, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(buf[n.*..], fmt, args) catch return;
        n.* += s.len;
    }

    /// The trim decision's inputs as one line (the price, the positions, the chosen bill).
    fn formatTrimInputs(
        buf: []u8,
        tokens_len: usize,
        row_bytes: u64,
        budget: u64,
        positions: []const usize,
        cp_bytes: []const u64,
        total: usize,
        chosen: ?usize,
        gated: bool,
    ) []const u8 {
        var n: usize = 0;
        appendTrimFmt(buf, &n, "  [hot-cache] trim inputs: tokens={d} row_bytes={d} budget={d:.2} MB list_len={d} arm={s} survivors=[", .{
            tokens_len,
            row_bytes,
            @as(f64, @floatFromInt(budget)) / (1024.0 * 1024.0),
            total,
            trimBillArm(total, gated),
        });
        const shown = @min(positions.len, TRIM_LOG_MAX_POS);
        for (positions[0..shown], 0..) |p, i| {
            if (i > 0) appendTrimFmt(buf, &n, ",", .{});
            appendTrimFmt(buf, &n, "{d}", .{p});
        }
        if (shown < total) appendTrimFmt(buf, &n, ",...", .{});
        appendTrimFmt(buf, &n, "] ({d} of {d})", .{ shown, total });
        if (chosen) |tl| {
            appendTrimFmt(buf, &n, " chosen={d}", .{tl});
            var chosen_cp: u64 = 0;
            for (positions, 0..) |p, i| {
                if (p == tl and i < cp_bytes.len) {
                    chosen_cp = cp_bytes[i];
                    break;
                }
            }
            appendTrimFmt(buf, &n, " chosen_cp_bytes={d}", .{chosen_cp});
        } else {
            appendTrimFmt(buf, &n, " chosen=none", .{});
        }
        appendTrimFmt(buf, &n, "\n", .{});
        return buf[0..n];
    }

    /// Fires once per oversized commit.
    fn logTrimInputs(
        tokens_len: usize,
        row_bytes: u64,
        budget: u64,
        cps: ?[]const SSMCheckpoint,
        chosen: ?usize,
        gated: bool,
    ) void {
        var pos_buf: [SHED_SIM_MAX]usize = undefined;
        var byte_buf: [SHED_SIM_MAX]u64 = undefined;
        var k: usize = 0;
        if (cps) |list| {
            while (k < list.len and k < SHED_SIM_MAX) : (k += 1) {
                pos_buf[k] = list[k].pos;
                byte_buf[k] = ssmCheckpointBytes(&list[k]);
            }
        }
        const total = if (cps) |list| list.len else 0;
        var line: [768]u8 = undefined;
        log.info("{s}", .{formatTrimInputs(&line, tokens_len, row_bytes, budget, pos_buf[0..k], byte_buf[0..k], total, chosen, gated)});
    }

    /// Stack bound for the shed simulation; a longer list falls back to billing every lower checkpoint.
    const SHED_SIM_MAX: usize = 128;

    /// Bytes the checkpoints at or below a candidate trim point cost after the commit path's
    /// span-preserving shed to `allowance`; null when even the last survivor is over.
    fn shedSurvivorBytes(positions: []const usize, bytes: []const u64, allowance: u64, policy: transformer_mod.ThinPolicy) ?u64 {
        var total: u64 = 0;
        for (bytes) |b| total += b;
        if (total <= allowance) return total;
        if (positions.len > SHED_SIM_MAX) return null;
        var pos_buf: [SHED_SIM_MAX]usize = undefined;
        var byte_buf: [SHED_SIM_MAX]u64 = undefined;
        @memcpy(pos_buf[0..positions.len], positions);
        @memcpy(byte_buf[0..bytes.len], bytes);
        var n = positions.len;
        while (total > allowance and n > 1) {
            const drop = transformer_mod.positionDropIndexUsize(pos_buf[0..n], policy);
            total -= byte_buf[drop];
            var k = drop;
            while (k + 1 < n) : (k += 1) {
                pos_buf[k] = pos_buf[k + 1];
                byte_buf[k] = byte_buf[k + 1];
            }
            n -= 1;
        }
        return if (total <= allowance) total else null;
    }

    /// The hybrid arm of `trimLenForBudget` as pure arithmetic over positions and per-checkpoint bytes.
    fn trimLenForBudgetPure(
        budget: u64,
        limit: usize,
        row_bytes: u64,
        positions: []const usize,
        cp_bytes: []const u64,
        policy: transformer_mod.ThinPolicy,
    ) ?usize {
        var k = positions.len;
        while (k > 0) {
            k -= 1;
            const p = positions[k];
            if (p > limit) continue;
            if (p < MIN_CANCELLED_COMMIT_TOKENS) return null;
            const rows = @as(u64, p) * row_bytes;
            if (rows > budget) continue;
            if (shedSurvivorBytes(positions[0 .. k + 1], cp_bytes[0 .. k + 1], budget - rows, policy) != null) return p;
        }
        return null;
    }

    /// Which arm `trimLenForBudget` bills a list of this length with (for the log).
    fn trimBillArm(list_len: usize, gated: bool) []const u8 {
        if (!gated) return "all_lower";
        return if (list_len > SHED_SIM_MAX) "all_lower" else "shed";
    }

    /// The pre-shed bill (every lower checkpoint), for a list past `SHED_SIM_MAX` or the ungated arm.
    fn trimLenBillingAllLower(budget: u64, limit: usize, row_bytes: u64, list: []const SSMCheckpoint) ?usize {
        var k = list.len;
        while (k > 0) {
            k -= 1;
            const p = list[k].pos;
            if (p > limit) continue;
            if (p < MIN_CANCELLED_COMMIT_TOKENS) return null;
            var cps_cost: u64 = 0;
            for (list[0 .. k + 1]) |*cp| cps_cost += ssmCheckpointBytes(cp);
            if (@as(u64, p) * row_bytes + cps_cost <= budget) return p;
        }
        return null;
    }

    /// Issue #330: the longest retainable prefix length under `budget`, or
    /// null when nothing at or above the commit floor fits. With checkpoints
    /// (hybrid entry) the trim point must be a RESTORABLE position — a
    /// checkpoint's own `pos`, and its cost includes the checkpoints that survive the commit's
    /// shed. Plain attention restores at any length. `limit` caps the answer.
    fn trimLenForBudget(
        self: *const HotPrefixCache,
        budget: u64,
        limit: usize,
        row_bytes: u64,
        cps: ?[]const SSMCheckpoint,
    ) ?usize {
        if (cps) |list| {
            if (list.len > 0) {
                // Arch gate: the ungated arm prices every lower checkpoint, as before.
                if (self.cp_thin == .min_span) return trimLenBillingAllLower(budget, limit, row_bytes, list);
                if (list.len > SHED_SIM_MAX) return trimLenBillingAllLower(budget, limit, row_bytes, list);
                var pos_buf: [SHED_SIM_MAX]usize = undefined;
                var byte_buf: [SHED_SIM_MAX]u64 = undefined;
                for (list, 0..) |*cp, i| {
                    pos_buf[i] = cp.pos;
                    byte_buf[i] = ssmCheckpointBytes(cp);
                }
                return trimLenForBudgetPure(budget, limit, row_bytes, pos_buf[0..list.len], byte_buf[0..list.len], self.cp_thin);
            }
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
        probe: ?*MatchProbe,
    ) ?struct { idx: usize, shared: usize } {
        var best_idx: ?usize = null;
        var best_shared: usize = 0;
        var best_effective: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            // A checked-out entry's buffers belong to another slot; its snapshot is empty.
            if (e.checked_out_by != null) continue;
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

            // Record the RAW match before the restorability filter can drop
            // this candidate — a null return with a long raw match is the
            // expensive miss, and the only place that fact still exists.
            if (probe) |p| {
                p.candidates += 1;
                if (shared > p.best_raw) p.best_raw = shared;
            }

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
        const match = self.findBestRestorableMatch(prompt_ids, has_tools, vision_key, null, quant_config, false, null) orelse return null;
        return .{ .idx = match.idx, .shared = match.shared };
    }

    /// Try to restore a matching entry into `target_cache`. On success, returns
    /// the matched prefix length. On miss, fully resets `target_cache` and
    /// returns 0. Caller should prefill the trailing tokens after this.
    ///
    /// The `target_*` parameters generalize the legacy single-slot path
    /// (`xfm.cache`, `xfm.moe_seq_offset`, `xfm.ssm_entries`) so Phase 2
    /// per-slot caches can reuse the same restore machinery.
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
            null,
        );
    }

    /// The checkout-capable entry point: `slot_id` names the slot that will own the restored buffers.
    pub fn lookupAndRestoreForSlot(
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
        slot_id: usize,
    ) !LookupResult {
        return self.lookupAndRestoreWithMedia(
            target_cache,
            target_moe_seq_offset,
            target_ssm_entries,
            s,
            prompt_ids,
            has_tools,
            vision_key,
            media_start,
            dflash_target,
            mtp_target,
            slot_id,
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
        /// Restore by move: non-null opts this request into the checkout (the caller promises
        /// `releaseCheckout` on every path that ends the slot).
        slot_id: ?usize,
    ) !LookupResult {
        self.last_restored_used = null;
        var probe: MatchProbe = .{};
        const match = self.findBestRestorableMatch(
            prompt_ids,
            has_tools,
            vision_key,
            media_start,
            target_cache.config,
            target_ssm_entries != null,
            &probe,
        );

        // ── SSD tier: consult when it can beat the RAM match meaningfully
        // (fresh boot, post-eviction). Phase 3 handles hybrid targets too —
        // the tier persists per-position SSM checkpoints beside the KV chunks
        // and restores both.
        if (self.disk) |*d| disk: {
            // A media-carrying request still restores the pure-text prefix
            // before its earliest media row: disk entries are text-only by
            // construction (the flush refuses vision-keyed entries), and
            // state below the boundary is pixel-independent — the RAM path's
            // exact cross-key semantics. The old blanket `vision_key != 0`
            // skip starved image-bearing agent turns of the whole SSD tier
            // (live 2026-09-07: one screenshot in the last turn of a 122k
            // session disabled 40 GB of persisted prefixes). The cap is
            // load-bearing even when tokens match past it: placeholder ids
            // only ever legitimately appear at/after the media position, and
            // a restored row there must come from the vision splice, never
            // from a text prefix.
            const disk_limit: u32 = if (media_start) |ms| @intCast(@min(ms, prompt_ids.len)) else @intCast(prompt_ids.len);
            const dm = d.bestMatch(prompt_ids, has_tools, target_cache.config) orelse {
                // Silent no-entry misses are why a 40 GB disk tier looked
                // dead in a live post-mortem (2026-09-07): nothing in the
                // log ever said the tier was consulted and found nothing.
                if (d.entryCount() > 0)
                    log.debug("  [disk-cache] lookup miss: {d} entries on disk, none share a token prefix under this (tools, quant) key\n", .{d.entryCount()});
                break :disk;
            };
            const usable: u32 = @min(dm.usable, disk_limit);

            if (target_ssm_entries) |ssm_entries| {
                // Hybrid: compare EFFECTIVE restorable positions — the largest
                // SSM checkpoint ≤ the match on each tier, not the raw prefix
                // length (KV alone is useless without matching SSM state).
                // The disk side ranks entries by that checkpoint too (#312's
                // RAM lesson: a longer raw match whose checkpoints sit past
                // the divergence restores nothing — and it must not shadow a
                // shorter entry with a higher restorable position).
                const ram_eff: usize = if (match) |m| blk: {
                    const e = &self.entries.items[m.idx];
                    const cps = e.ssm_checkpoints orelse break :blk 0;
                    const cp = highestCheckpointAtOrBelow(cps, m.shared) orelse break :blk 0;
                    break :blk cp.pos;
                } else 0;
                const hm = d.bestHybridMatch(prompt_ids, has_tools, target_cache.config, disk_limit) orelse break :disk;
                const disk_cp = hm.cp;
                if (@as(usize, disk_cp) < ram_eff + kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) break :disk;
                const sw = io_util.Stopwatch.init(d.io);
                const restored = d.restoreIntoHybrid(target_cache, ssm_entries, hm.idx, disk_cp, s) catch |err| {
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
            if (usable <= ram_len) break :disk;
            if (usable - ram_len < kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) break :disk;
            // `usable` is `dm.usable` clamped to the entry's kv_len (see
            // DiskTier.bestMatch) and to the request's media boundary, so it
            // IS the restorable length — restore
            // only the chunks covering it. Loading the whole entry
            // (restoreInto) would read a long stored prefix in full to serve a
            // short shared prefix, making a diverged-prefix "hit" slower than a
            // cold miss.
            const effective: usize = usable;
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
            // The filter dropped every candidate. This arm used to be silent,
            // so a 393k-token prompt that the cache almost had cold-prefilled
            // for 560 s with no `[hot-cache]` line at all. Same phrasing as
            // the one-entry miss below — one string to grep for.
            switch (missKind(probe.candidates, probe.best_raw)) {
                .cold => {},
                .no_checkpoint => log.info(
                    "  [hot-cache] hybrid miss (no checkpoint ≤ {d} of {d} in {d} entries); cold prefill\n",
                    .{ probe.best_raw, prompt_ids.len, probe.candidates },
                ),
            }
            return .{ .matched = 0, .full_match = false };
        }
        const m = match.?;
        const e = &self.entries.items[m.idx];
        // Decline a restore that is a lien (`restoreWouldPinEntry`), before the bump and before
        // `last_restored_used`; weighed on the DELIVERABLE share (a hybrid is clamped to its
        // highest checkpoint at or below the match). SSD-first only.
        const deliverable = deliverableShare(e.ssm_checkpoints, target_ssm_entries != null, m.shared);
        if (self.ssd_first and restoreWouldPinEntry(e.kv_bytes, self.restore_pin_min_bytes, e.tokens.len, deliverable)) {
            try target_cache.truncate(0, s);
            if (target_ssm_entries) |entries| resetSsmEntries(entries);
            target_moe_seq_offset.* = 0;
            log.info("  [hot-cache] declined a {d}-token restore ({d} deliverable) from a {d}-token entry ({d} MB): the share is a lien on the whole entry; cold prefill\n", .{
                m.shared,
                deliverable,
                e.tokens.len,
                e.kv_bytes / (1024 * 1024),
            });
            return .{ .matched = 0, .full_match = false };
        }
        // The stamp before the bump: two arms after the restore still end in `matched = 0`.
        if (e.ssm_checkpoints) |cps| {
            if (highestCheckpointAtOrBelow(cps, m.shared)) |cp| {
                const src_cp = qsaHistorySource(cps, cp) orelse cp;
                const have = transformer_mod.checkpointQsaAuxRows(src_cp);
                if (have > 0 and have < @as(c_int, @intCast(cp.pos))) {
                    return error.QsaHistoryGap;
                }
            }
        }
        const used_before_restore = e.last_used;
        e.last_used = self.bumpCounter();
        // Identity of the entry this request runs on: evicting it frees nothing (shared buffers).
        self.last_restored_used = e.last_used;

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

        // Miss path (hybrid without a usable checkpoint, and the QSA-history
        // decline that funnels into it): also reset KV.
        if (effective_matched == 0) {
            try target_cache.truncate(0, s);
            // A 0-token outcome is not a restore: the marker was set above the restore (the hybrid
            // clamp needs the entry live) and would otherwise shield a fully reclaimable entry from
            // the admission pass. The LRU hand-back takes the same `ssd_first` gate as the lien decline.
            self.last_restored_used = null;
            if (self.ssd_first) e.last_used = used_before_restore;
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
        var res: LookupResult = .{
            .matched = effective_matched,
            .full_match = full_match,
            .dflash_base = restoreDflash(e, dflash_target, effective_matched, s),
            .mtp_base = restoreMtp(e, mtp_target, effective_matched, s),
        };
        res.checked_out = self.checkoutIfEligible(m.idx, m.shared, prompt_ids.len, slot_id);
        return res;
    }

    /// Restore by move, the decision: only on a full-prefix hit (the commit's replace path lands
    /// on this same entry) with something to append. A partial hit keeps the refcount share.
    pub fn checkoutEligible(
        ssd_first: bool,
        move_enabled: bool,
        pending_disk: bool,
        entry_tokens: usize,
        shared: usize,
        prompt_len: usize,
        has_slot: bool,
    ) bool {
        if (!ssd_first or !move_enabled or !has_slot) return false;
        // The pending disk record shares the same buffers, so a checkout could not donate anyway.
        if (pending_disk) return false;
        if (entry_tokens == 0 or shared != entry_tokens) return false;
        return prompt_len > shared;
    }

    /// Take the checkout: mark the entry as this slot's; `donateCheckout` releases the handles
    /// at the last moment before the slot's first write. Returns whether it was taken (the
    /// admission bill's input).
    fn checkoutIfEligible(self: *HotPrefixCache, idx: usize, shared: usize, prompt_len: usize, slot_id: ?usize) bool {
        const e = &self.entries.items[idx];
        if (!checkoutEligible(
            self.ssd_first,
            restoreMoveEnabled(),
            self.pending_disk != null,
            e.tokens.len,
            shared,
            prompt_len,
            slot_id != null,
        )) return false;
        e.checked_out_by = slot_id;
        log.info("  [hot-cache] checked out {d}-token entry to the slot (restore by move; the append donates in place)\n", .{e.tokens.len});
        return true;
    }

    /// Restore by move, the transfer: give up the entry's own handles so the slot is the sole
    /// owner and its first `writeAtOffset` donates in place. Called right before
    /// `Generator.initWithOptions`, after the admission pass that can still refuse: releasing at
    /// restore time threw away a 364k session for a request that never ran. Idempotent.
    pub fn donateCheckout(self: *HotPrefixCache, slot_id: usize) void {
        for (self.entries.items) |*e| {
            if (e.checked_out_by != slot_id) continue;
            if (e.checkout_donated) continue;
            e.snapshot.releaseHandles();
            e.checkout_donated = true;
        }
    }

    /// End of a slot's life. Not donated: the entry never gave its handles up, hand it back
    /// unchanged. Donated: the bytes die with the slot, drop the record. Reaching here with a
    /// mark set means the slot ended without committing. Idempotent.
    pub fn releaseCheckout(self: *HotPrefixCache, slot_id: usize, reason: []const u8) void {
        var i: usize = self.entries.items.len;
        while (i > 0) {
            i -= 1;
            const e = &self.entries.items[i];
            if (e.checked_out_by != slot_id) continue;
            const tokens_len = e.tokens.len;
            if (!e.checkout_donated) {
                // Nothing was handed over; only the mark is undone (no LRU bump for a request that never ran).
                e.checked_out_by = null;
                log.info("  [hot-cache] checked-out entry returned intact: {s} ({d} tokens; the append never ran)\n", .{ reason, tokens_len });
                continue;
            }
            // Clear the mark before `evictAt`; the snapshot is already empty.
            e.checked_out_by = null;
            e.checkout_donated = false;
            self.evictAt(i, "checked-out entry dropped");
            log.info("  [hot-cache] checked-out entry dropped: {s} ({d} tokens; its KV died with the slot)\n", .{ reason, tokens_len });
        }
    }

    /// Commit the current `source_cache` state under the given key. Updates
    /// the matching entry if one exists for this exact prefix, otherwise
    /// inserts a new entry, evicting the oldest if at capacity. Snapshot is
    /// taken here (cheap — refcount-share, no data copy).
    pub fn commit(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
    ) !CommitStatus {
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
    ) !CommitStatus {
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
    ) !CommitStatus {
        return self.commitWithMediaState(source_cache, tokens, has_tools, vision_key, 0, null, ssm_cps, dflash, mtp);
    }

    pub fn commitWithMediaState(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        vision_key: u64,
        cache_key: u64,
        media_start: ?usize,
        ssm_cps: ?[]SSMCheckpoint,
        dflash: ?DflashCommit,
        mtp: ?DflashCommit,
    ) !CommitStatus {
        const quant_config = source_cache.config;

        // Record what the live cache holds now, before any byte-budget trim.
        if (self.ssd_first and self.disk != null and vision_key == 0) {
            self.capturePendingDisk(source_cache, tokens, has_tools, ssm_cps, dflash, mtp);
        }
        // The record shares the live KV; on an error return nothing consumes it and the slot's
        // KVCache deinit then frees nothing. Function scope on purpose.
        errdefer if (self.pending_disk) |*p| {
            p.deinit(self.allocator);
            self.pending_disk = null;
        };

        // An entry's pixel key applies only to rows it actually covers. When
        // the committed range ends before the request's first media row — a
        // cancelled prefill that forwarded less than the media position —
        // the entry is pure text and must carry the NULL key, or the
        // boundary-less cross-key entry gets conservatively rejected by the
        // lookup (live 2026-09-07: the with-image and no-image retry shapes
        // of one 122k session locked each other out of a shared 121819-token
        // text prefix). A null `media_start` keeps the caller's key as-is
        // (commitWithState's pixel-keyed, boundary-less contract).
        var eff_media_start = media_start;
        var eff_vision_key = vision_key;
        if (media_start) |ms| {
            if (ms >= tokens.len) {
                eff_media_start = null;
                eff_vision_key = 0;
            }
        }

        var replace_idx: ?usize = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (e.vision_key != eff_vision_key) continue;
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
                new_mtp = .{
                    .snapshot = snap,
                    .base_pos = m2.base_pos,
                    .head_aux = if (m2.head) |h| transformer_mod.ssmSnapshot(h) else null,
                    .head_pos_base = m2.head_pos_base,
                };
                new_mtp_bytes = specSnapBytes(&new_mtp.?);
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
            var decline: TrimDecline = .no_restorable_prefix;
            var decline_err: ?anyerror = null;
            var limit = if (eff_media_start) |ms| @min(tokens.len, ms) else tokens.len;
            var inputs_logged = false;
            trim_blk: while (true) {
                const row_bytes = snapshotRowBytes(&new_snap);
                const tl_opt = self.trimLenForBudget(self.max_kv_bytes, limit, row_bytes, eff_cps);
                if (!inputs_logged) {
                    inputs_logged = true;
                    logTrimInputs(tokens.len, row_bytes, self.max_kv_bytes, eff_cps, tl_opt, self.cp_thin != .min_span);
                }
                const tl = tl_opt orelse break :trim_blk;
                // One-shot: when the resident covered entry already retains
                // the trim target, keep it and drop the candidate — the
                // target is budget-derived and stable, so replacing would
                // re-copy an identical multi-GB prefix every turn.
                if (replace_idx) |idx| {
                    // ...unless the entry is checked out: its snapshot is empty, so fall through to the trim/replace.
                    if (self.entries.items[idx].checked_out_by == null and
                        self.entries.items[idx].tokens.len >= tl)
                    {
                        // The resident entry already covers the trim target;
                        // the candidate's EXTRA tokens still belong on disk.
                        if (eff_vision_key == 0) self.spillDeclinedToDisk(&new_snap, tokens, has_tools, eff_cps);
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
                        return .{ .kept_resident = self.entries.items[idx].tokens.len };
                    }
                }
                const trimmed = new_snap.trimmedCopy(tl, mlx.gpuStream()) catch |err| {
                    // A copy that failed at this width is not a verdict on the entry: retry at the next-lower checkpoint.
                    decline = .snapshot_copy_failed;
                    decline_err = err;
                    // Arch gate: a retry allocates again under memory pressure; ungated declines as before.
                    if (self.cp_thin == .min_span) break :trim_blk;
                    log.warn("  [hot-cache] trimmed copy to {d} tokens failed: {s}; retrying at the next-lower checkpoint\n", .{ tl, @errorName(err) });
                    if (tl == 0) break :trim_blk;
                    limit = tl - 1;
                    continue;
                };
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
                        const shrunk = self.allocator.dupe(SSMCheckpoint, cps[0..kept]) catch |err| {
                            decline = .checkpoint_list_copy_failed;
                            decline_err = err;
                            break :trim_blk;
                        };
                        for (cps[kept..]) |*cp| cp.deinit(self.allocator);
                        self.allocator.free(cps);
                        eff_cps = shrunk;
                    }
                }
                eff_tokens = tokens[0..tl];
                if (eff_media_start) |ms| {
                    if (ms >= tl) {
                        // The trim cut below the media boundary: the entry
                        // is pure text now — re-key it, and drop the replace
                        // target (it was scanned under the pixel key, a
                        // different keyspace; the image-keyed entry stays
                        // behind for its own requests).
                        eff_media_start = null;
                        eff_vision_key = 0;
                        replace_idx = null;
                    }
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
                break;
            }
            if (!trimmed_ok) {
                // RAM decline is not a value verdict: offer the candidate to
                // the SSD tier before discarding it.
                if (eff_vision_key == 0) self.spillDeclinedToDisk(&new_snap, tokens, has_tools, eff_cps);
                var discarded_snap = new_snap;
                discarded_snap.deinit();
                if (new_dflash) |*d| d.deinit();
                if (new_mtp) |*m3| m3.deinit();
                if (eff_cps) |cps| {
                    for (cps) |*cp| cp.deinit(self.allocator);
                    self.allocator.free(cps);
                }
                const err_sep: []const u8 = if (decline_err != null) ": " else "";
                const err_name: []const u8 = if (decline_err) |e| @errorName(e) else "";
                log.info("  [hot-cache] skipped oversized entry ({d} tokens, {d:.2} MB > {d:.2} MB budget): {s}{s}{s}\n", .{
                    tokens.len,
                    @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
                    decline.reason(),
                    err_sep,
                    err_name,
                });
                return .declined;
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

        // A commit built on a RESTORED prefix must be at least as restorable
        // as the entry it restored from. The replace path below inherits from
        // the entry it overwrites; a commit that lands as a NEW entry had no
        // inheritance at all, and that is the common shape whenever one prompt
        // is answered twice (an MTP arm then a serial arm, two clients, a
        // retry): the two entries agree on the whole prompt and diverge in
        // their generated tails, so neither is a prefix of the other. The
        // second entry then holds only its own tail prefill's snapshot — which
        // for a ~31-token tail lands AT the prompt end, past any later match
        // (`ssmSnapshotBackoff` is 0 below 31 tokens). Evict the first and the
        // prompt becomes uncacheable.
        //
        // Inherit by refcount-SHARE, never copy: the buffers are already
        // resident, so this costs GPU memory only in the accounting, and only
        // until the donor is evicted.
        if (replace_idx == null) inherit: {
            const donor = self.bestCheckpointDonor(eff_tokens, has_tools, vision_key, quant_config) orelse
                break :inherit;
            const budget: ?u64 = if (self.max_kv_bytes == 0)
                null
            else if (new_bytes >= self.max_kv_bytes)
                break :inherit
            else
                self.max_kv_bytes - new_bytes;
            const donor_cps = self.entries.items[donor.idx].ssm_checkpoints.?;
            const cloned = (cloneCheckpointsUpTo(self.allocator, donor_cps, donor.shared, budget) catch |err| {
                log.warn("  [hot-cache] checkpoint inheritance failed: {s}\n", .{@errorName(err)});
                break :inherit;
            }) orelse break :inherit;
            if (eff_cps) |own| {
                // Consumes both on every path; on error neither survives.
                eff_cps = self.mergeCheckpointLists(cloned, own) catch |err| {
                    log.warn("  [hot-cache] checkpoint merge failed: {s}\n", .{@errorName(err)});
                    eff_cps = null;
                    break :inherit;
                };
            } else {
                eff_cps = cloned;
            }
            var inherited_bytes: u64 = 0;
            for (eff_cps.?) |*cp| inherited_bytes += ssmCheckpointBytes(cp);
            new_bytes = new_bytes - new_ssm_bytes + inherited_bytes;
            new_ssm_bytes = inherited_bytes;
            log.info("  [hot-cache] inherited {d} checkpoints (<= {d} tokens) from a shared prefix\n", .{
                eff_cps.?.len,
                donor.shared,
            });
        }

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
                // Detach old from its container either way: it is either moved
                // wholesale or consumed by the merge, and the free-below must
                // not touch it.
                e.ssm_checkpoints = null;
                const new = eff_cps orelse break :blk old;
                break :blk try self.mergeCheckpointLists(old, new);
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
            e.vision_key = eff_vision_key;
            e.cache_key = cache_key;
            e.media_start = eff_media_start;
            e.quant_config = quant_config;
            e.kv_bytes = new_kv_bytes + merged_ssm_bytes + new_dflash_bytes + new_mtp_bytes;
            e.ssm_checkpoints = merged_cps;
            e.ssm_bytes = merged_ssm_bytes;
            e.dflash = new_dflash;
            e.dflash_bytes = new_dflash_bytes;
            e.mtp = new_mtp;
            e.mtp_bytes = new_mtp_bytes;
            // Restore by move: the replacement the checkout promised; the entry is whole again.
            e.checked_out_by = null;
            e.checkout_donated = false;
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
                    self.evictOneLru("byte budget", null);
                }
                self.shedCheckpointsToFit();
                while (self.current_kv_bytes > self.max_kv_bytes and
                    self.entries.items.len > 0)
                {
                    self.evictOneLru("byte budget", null);
                }
            }
            if (self.disk != null) self.disk_dirty = true;
            self.logResident();
            return .{ .ok = eff_tokens.len };
        }

        while (self.entries.items.len >= self.max_entries) {
            self.evictOneLru("count cap", cache_key);
        }
        if (self.max_kv_bytes > 0) {
            while (self.current_kv_bytes + new_bytes > self.max_kv_bytes and self.entries.items.len > 0) {
                self.evictOneLru("byte budget", cache_key);
            }
        }

        self.entries.append(self.allocator, .{
            .tokens = tokens_owned,
            .has_tools = has_tools,
            .vision_key = eff_vision_key,
            .cache_key = cache_key,
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
        // The trim prices a prefix against the checkpoints that survive a shed, so the shed runs here too.
        if (self.max_kv_bytes > 0) self.shedCheckpointsToFit();
        if (self.disk != null) self.disk_dirty = true;
        self.logResident();
        return .{ .ok = eff_tokens.len };
    }

    /// Offer a budget-declined candidate to the SSD tier before discarding
    /// it. The RAM budget declines RETENTION, not value: under a small
    /// --prefix-cache-mem a long agent session's cancelled-prefill prefixes
    /// were thrown away on every retry while a much larger disk tier sat
    /// idle (live 2026-09-07). Text-prefix candidates only — the disk never
    /// holds pixel-keyed rows. Best-effort: a failed write costs nothing the
    /// RAM decline hadn't already lost, and a byte-capped write continues
    /// from the next commit of the same conversation (chunks are
    /// content-deduped by token range).
    fn spillDeclinedToDisk(
        self: *HotPrefixCache,
        snap: *const KVCacheSnapshot,
        tokens: []const u32,
        has_tools: bool,
        cps: ?[]SSMCheckpoint,
    ) void {
        const d = if (self.disk) |*dd| dd else return;
        // SSD-first captured the live state as `pending_disk` before the trim; the
        // normal flush lands it under the writer's own readback bound.
        if (self.ssd_first) {
            if (self.pending_disk != null) self.disk_dirty = true;
            return;
        }
        if (tokens.len < kv_disk_cache.MIN_PERSIST_TOKENS) return;
        // A decline-spill runs after the client is gone: the per-flush cap
        // exists to bound the stall a LIVE next request pays, and capping
        // here stranded most of each retry's work (live 2026-09-07: 512 MB
        // ≈ 13k tokens banked per retry while ~35k were recomputed every
        // time). Raise the budget for the spill's duration — the tier's
        // byte budget + LRU eviction is the real bound.
        const saved_cap = d.max_flush_bytes;
        d.max_flush_bytes = @max(saved_cap, kv_disk_cache.DECLINE_SPILL_FLUSH_FLOOR);
        defer d.max_flush_bytes = saved_cap;
        const outcome = d.appendCommit(snap.entries, snap.step, snap.config, tokens, has_tools, cps, mlx.gpuStream()) catch |err| {
            log.warn("  [disk-cache] declined-candidate spill failed: {s}\n", .{@errorName(err)});
            return;
        };
        const note: []const u8 = switch (outcome) {
            .persisted => "complete",
            .partial => "byte-capped, continues on later commits",
            .skipped => "already persisted — nothing new to write",
        };
        log.info("  [disk-cache] declined RAM candidate spilled to SSD ({d} tokens, {s})\n", .{ tokens.len, note });
    }

    /// Snapshot the live cache (refcount-shared) plus the full token record and this turn's
    /// checkpoints. Best effort; the caller still owns `ssm_cps`/`dflash`/`mtp`.
    fn capturePendingDisk(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        ssm_cps: ?[]SSMCheckpoint,
        dflash: ?DflashCommit,
        mtp: ?DflashCommit,
    ) void {
        if (self.pending_disk) |*old| {
            old.deinit(self.allocator);
            self.pending_disk = null;
        }
        var snap = source_cache.snapshot() catch |err| {
            log.warn("  [disk-cache] live snapshot failed: {s} — flushing the RAM entry instead\n", .{@errorName(err)});
            return;
        };
        var rec: PendingDiskFlush = .{
            .snapshot = snap,
            .tokens = self.allocator.dupe(u32, tokens) catch {
                snap.deinit();
                return;
            },
            .has_tools = has_tools,
        };
        if (ssm_cps) |cps| {
            rec.ssm_cps = cloneCheckpointsUpTo(self.allocator, cps, std.math.maxInt(usize), null) catch null;
        }
        if (dflash) |d| {
            if (d.cache.snapshot()) |ds| {
                rec.dflash = .{ .snapshot = ds, .base_pos = d.base_pos };
            } else |_| {}
        }
        if (mtp) |m2| {
            if (m2.cache.snapshot()) |ms| {
                rec.mtp = .{
                    .snapshot = ms,
                    .base_pos = m2.base_pos,
                    .head_aux = if (m2.head) |h| transformer_mod.ssmSnapshot(h) else null,
                    .head_pos_base = m2.head_pos_base,
                };
            } else |_| {}
        }
        self.pending_disk = rec;
    }

    const EntrySpecs = struct {
        dflash: ?kv_disk_cache.SpecCommit = null,
        mtp: ?kv_disk_cache.SpecCommit = null,
    };

    /// The disk-tier spec payloads for one RAM entry; the flush and the idle spill must not drift.
    fn entrySpecCommits(e: *Entry) EntrySpecs {
        return .{
            .dflash = if (e.dflash) |*df| .{
                .entries = df.snapshot.entries,
                .step = df.snapshot.step,
                .config = df.snapshot.config,
                .base_pos = df.base_pos,
            } else null,
            .mtp = if (e.mtp) |*mm| .{
                .entries = mm.snapshot.entries,
                .step = mm.snapshot.step,
                .config = mm.snapshot.config,
                .base_pos = mm.base_pos,
                .head_aux = if (mm.head_aux) |*a| a else null,
                .head_pos_base = mm.head_pos_base,
            } else null,
        };
    }

    /// At the end of a request every idle entry is written to the SSD tier, and RAM is trimmed
    /// back to the active session plus the idle allowance. Writing is unconditional; the
    /// allowance is a hard cap shed in two tiers: entries with a proven durable copy first,
    /// then (naming the reason) the rest, oldest first.
    pub fn spillIdleEntries(self: *HotPrefixCache, s: mlx.mlx_stream) void {
        if (!self.ssd_first) return;
        if (self.entries.items.len <= 1) return;
        const d = if (self.disk) |*dd| dd else return;

        // The active session is the single MRU entry; `last_used` is a strictly increasing counter.
        var newest_used: u64 = 0;
        for (self.entries.items) |*e| newest_used = @max(newest_used, e.last_used);

        // Pass 1: write, unconditionally. Attribute anything the writer lost since the last pass
        // before sampling the error counter: a failure between two passes is invisible to the delta.
        _ = d.harvestWriteFailures();
        const errs_before = d.writeErrors();
        var spilled: usize = 0;
        var idle_bytes: u64 = 0;
        for (self.entries.items) |*e| {
            e.spill_durable = false;
            if (e.last_used == newest_used) continue; // the active session
            // Checked out: the bytes are the slot's, the snapshot is empty. Skipped before the allowance count.
            if (e.checked_out_by != null) continue;
            // Counted against the allowance before the vision skip: it occupies RAM either way.
            idle_bytes +|= e.kv_bytes;
            // Vision entries never spill: a token-only disk key is ambiguous.
            if (e.vision_key != 0) continue;
            const specs = entrySpecCommits(e);
            const outcome = d.appendCommitWithSpec(
                e.snapshot.entries,
                e.snapshot.step,
                e.snapshot.config,
                e.tokens,
                e.has_tools,
                e.ssm_checkpoints,
                specs.dflash,
                specs.mtp,
                s,
            ) catch |err| {
                log.warn("  [hot-cache] idle spill failed: {s} — entry stays resident\n", .{@errorName(err)});
                continue;
            };
            // Only `.persisted`: a silent skip or a `.partial` copy is not a copy.
            if (outcome != .persisted) continue;
            // `.persisted` is the write path's claim; the index has to agree.
            const disk_id = d.fullPrefixEntryId(e.snapshot.entries, e.snapshot.step, e.tokens, e.has_tools, e.snapshot.config) orelse {
                log.warn("  [hot-cache] idle spill: the tier does not hold the full prefix — entry stays resident\n", .{});
                continue;
            };
            // A staged copy is not durable yet, and the check must not be a drain (this runs on
            // the inference thread at every finish); the next pass asks again.
            if (d.entryWritesPending(disk_id)) continue;
            if (d.writeErrors() != errs_before) {
                log.warn("  [hot-cache] idle spill: background write failed — entry stays resident\n", .{});
                continue;
            }
            // Ask the filesystem too: one stat per chunk, against the eviction of a whole session.
            if (!d.entryWholeOnDisk(disk_id)) {
                log.warn("  [hot-cache] idle spill: the persisted chunks do not match the index — entry stays resident\n", .{});
                continue;
            }
            e.spill_durable = true;
            spilled += 1;
        }

        // Pass 2: evict the durable, oldest first, down to the allowance.
        var evicted: usize = 0;
        while (idle_bytes > self.ssd_idle_mem) {
            const idx = self.oldestIdleIndex(newest_used, true) orelse break;
            idle_bytes -|= self.entries.items[idx].kv_bytes;
            self.evictAt(idx, "SSD-first idle spill");
            evicted += 1;
        }

        // Pass 3: still over. Entries with no durable copy go too, oldest first, naming why.
        while (idle_bytes > self.ssd_idle_mem) {
            const idx = self.oldestIdleIndex(newest_used, false) orelse break;
            const e = &self.entries.items[idx];
            log.info("  [hot-cache] idle allowance exceeded: dropped unpersistable entry ({s}) {d} tokens, {d:.1} MB\n", .{
                unpersistableReason(d, e),
                e.tokens.len,
                @as(f64, @floatFromInt(e.kv_bytes)) / (1024.0 * 1024.0),
            });
            idle_bytes -|= e.kv_bytes;
            self.evictAt(idx, "SSD-first idle allowance");
            evicted += 1;
        }

        if (spilled > 0 or evicted > 0) {
            log.info("  [hot-cache] SSD-first: wrote {d} idle entries to disk, evicted {d}; RAM holds the active session + {d} MB idle allowance\n", .{
                spilled, evicted, self.ssd_idle_mem >> 20,
            });
        }
    }

    /// Least-recently-used idle entry (never the active session, never a vision entry).
    fn oldestIdleIndex(self: *HotPrefixCache, newest_used: u64, durable_only: bool) ?usize {
        var best: ?usize = null;
        var best_used: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used == newest_used) continue;
            if (e.vision_key != 0) continue;
            if (e.checked_out_by != null) continue;
            if (durable_only and !e.spill_durable) continue;
            if (e.last_used < best_used) {
                best_used = e.last_used;
                best = i;
            }
        }
        return best;
    }

    /// Why pass 1 could not leave a durable copy of `e` (for the log).
    fn unpersistableReason(d: *kv_disk_cache.DiskTier, e: *const Entry) []const u8 {
        if (d.store_declined) return "store declined: volume is short";
        if (e.tokens.len < @as(usize, kv_disk_cache.MIN_PERSIST_TOKENS)) return "under the persist floor";
        switch (e.snapshot.config.scheme) {
            .off, .affine => {},
            else => return "TurboQuant state does not survive a restore",
        }
        const target = kv_disk_cache.persistTargetLen(e.snapshot.entries, e.snapshot.step, e.tokens.len);
        for (e.snapshot.entries) |*le| {
            if (le.initialized and le.offset < target) return "layer offset short of the range";
        }
        return "partial copy";
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
        // Flush the live state captured at commit, not what the RAM entry retained after its trim.
        if (self.pending_disk) |*pending| {
            defer {
                pending.deinit(self.allocator);
                self.pending_disk = null;
            }
            const p_dflash: ?kv_disk_cache.SpecCommit = if (pending.dflash) |*df| .{
                .entries = df.snapshot.entries,
                .step = df.snapshot.step,
                .config = df.snapshot.config,
                .base_pos = df.base_pos,
            } else null;
            const p_mtp: ?kv_disk_cache.SpecCommit = if (pending.mtp) |*mm| .{
                .entries = mm.snapshot.entries,
                .step = mm.snapshot.step,
                .config = mm.snapshot.config,
                .base_pos = mm.base_pos,
                .head_aux = if (mm.head_aux) |*a| a else null,
                .head_pos_base = mm.head_pos_base,
            } else null;
            const ok = d.appendCommitWithSpec(
                pending.snapshot.entries,
                pending.snapshot.step,
                pending.snapshot.config,
                pending.tokens,
                pending.has_tools,
                pending.ssm_cps,
                p_dflash,
                p_mtp,
                s,
            ) catch |err| {
                log.warn("  [disk-cache] persist failed: {s}\n", .{@errorName(err)});
                return;
            };
            // `.partial` is the only outcome with more to write.
            if (!ok.nothingPending()) self.disk_dirty = true;
            return;
        }
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
        const specs = entrySpecCommits(newest);
        const dflash_spec = specs.dflash;
        const mtp_spec = specs.mtp;
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
        if (!complete.nothingPending()) self.disk_dirty = true;
    }

    /// #330 adjacent: when the byte budget is exceeded with the just-updated
    /// entry as the sole survivor, drop its checkpoints (the inherited bytes
    /// the pre-check could not price) instead of evicting it. Reuses the
    /// replace path's interior-thinning rule (keep the first and the newest).
    /// The pre-check guarantees the entry's own KV + this turn's checkpoints
    /// fit, so shedding converges under budget before the list empties in
    /// practice; if it doesn't, the caller's eviction fallback decides.
    /// Merge two OWNED checkpoint lists into one ascending, pos-deduped list
    /// (on a tie the `new` state wins — it is the more recently observed one
    /// at that position), re-apply `ssm_checkpoint_max`, and collapse to ONE
    /// QSA history. Takes ownership of BOTH slices on every path; the caller
    /// must have detached them from whatever owned them.
    ///
    /// The cap is re-applied here because a merged list spans more than one
    /// prefill, so `generate.zig`'s per-prefill cap no longer bounds it — and
    /// NOT oldest-first. Within one prefill oldest-first is fine; across turns
    /// it collapses the survivors onto the end of the prompt, and then a
    /// request that diverges early finds no checkpoint at or below its match
    /// and pays a FULL cold prefill:
    ///     [hot-cache] hybrid miss (no checkpoint <= 16382 of 178509)
    /// That one cost 415 s. Oldest-first is also the expensive choice: a
    /// checkpoint costs roughly a constant plus a term linear in its position,
    /// so it discards the cheap early ones and keeps the large late ones.
    ///
    /// Thin the interior instead, always keeping the first and the newest:
    /// drop whichever checkpoint sits between the closest pair of neighbours,
    /// i.e. the one whose removal widens the coverage gap least. Same count,
    /// spread over the whole prompt, LESS memory. `n` is at most
    /// `ssm_checkpoint_max`, so the quadratic scan is trivial. The selection is
    /// `transformer.ssmCheckpointDropIndex`, shared with every other thinning site.
    fn mergeCheckpointLists(
        self: *HotPrefixCache,
        old: []SSMCheckpoint,
        new: []SSMCheckpoint,
    ) ![]SSMCheckpoint {
        var merged = std.ArrayList(SSMCheckpoint).empty;
        var i: usize = 0;
        var j: usize = 0;
        var sources_freed = false;
        // Ownership of BOTH slices is ours from the first line, so the error
        // path owes the un-moved tails too — items still sitting in old[i..]
        // / new[j..] plus the two backing slices.
        errdefer {
            for (merged.items) |*c| c.deinit(self.allocator);
            merged.deinit(self.allocator);
            if (!sources_freed) {
                for (old[@min(i, old.len)..]) |*c| c.deinit(self.allocator);
                for (new[@min(j, new.len)..]) |*c| c.deinit(self.allocator);
                self.allocator.free(old);
                self.allocator.free(new);
            }
        }
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
                var dropped = old[i];
                dropped.deinit(self.allocator);
                i += 1;
                try merged.append(self.allocator, new[j]);
                j += 1;
            }
        }
        self.allocator.free(old);
        self.allocator.free(new);
        sources_freed = true;
        while (self.ssm_checkpoint_max > 0 and
            merged.items.len > self.ssm_checkpoint_max)
        {
            // Under three there is no interior to thin; honour the cap by
            // dropping the oldest, which is also the cheapest to redo.
            const drop = transformer_mod.ssmCheckpointDropIndex(merged.items, self.cp_thin);
            var dropped = merged.orderedRemove(drop);
            dropped.deinit(self.allocator);
        }
        const owned = try merged.toOwnedSlice(self.allocator);
        // The inherited latest and this turn's latest both carry the indexer
        // history: keep one.
        keepOnlyLatestQsaHistory(owned);
        return owned;
    }

    /// The resident entry whose checkpoints a commit of `tokens` may inherit:
    /// the key-compatible entry maximizing the highest checkpoint at or below
    /// its shared prefix with `tokens`. Returns that entry's index and the
    /// shared length (the inheritance limit — a checkpoint past it describes
    /// state this prompt never reached).
    ///
    /// Checkpoint inheritance was a property of the REPLACE path — an entry
    /// whose tokens are a strict PREFIX of the new ones. It is really a
    /// property of the TOKENS. A prompt sent twice (an MTP arm then a serial
    /// arm; two clients; any retry) commits two entries that agree on the
    /// whole prompt and diverge in their GENERATED tails, so neither replaces
    /// the other — and the second one, having restored ~everything and
    /// prefilled a ~31-token tail, earns no reachable checkpoint of its own
    /// (`ssmSnapshotBackoff` is 0 below 31 tokens, so its sole snapshot lands
    /// AT the prompt end, past any later match). Evict the first and the
    /// prompt is uncacheable: 393k tokens, 560 s of cold prefill, every rung.
    fn bestCheckpointDonor(
        self: *const HotPrefixCache,
        tokens: []const u32,
        has_tools: bool,
        vision_key: u64,
        quant_config: kv_quant.KVQuantConfig,
    ) ?struct { idx: usize, shared: usize } {
        var best_idx: ?usize = null;
        var best_shared: usize = 0;
        var best_pos: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (e.vision_key != vision_key) continue;
            if (!std.meta.eql(e.quant_config, quant_config)) continue;
            const cps = e.ssm_checkpoints orelse continue;
            const max_shared = @min(e.tokens.len, tokens.len);
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == tokens[shared]) shared += 1;
            const cp = highestCheckpointAtOrBelow(cps, shared) orelse continue;
            if (cp.pos > best_pos) {
                best_pos = cp.pos;
                best_shared = shared;
                best_idx = i;
            }
        }
        if (best_idx) |idx| return .{ .idx = idx, .shared = best_shared };
        return null;
    }

    /// Refcount-share `src`'s checkpoints with `pos <= limit` into a fresh
    /// ASCENDING slice the caller owns. Newest-first while a budget remains
    /// (the highest position is the most valuable), then re-sorted; `budget`
    /// null means unbounded. Null when nothing qualifies.
    ///
    /// The clones share the donor's buffers, so this costs no GPU memory —
    /// but `current_kv_bytes` bills them again, because the accounting is
    /// per-entry and cannot see the sharing. That over-bills only while BOTH
    /// entries are resident and self-corrects the moment the donor is evicted
    /// (its bill goes, the buffers stay alive under the inheritor). Erring
    /// toward eviction is the safe direction for a hard cap.
    fn cloneCheckpointsUpTo(
        allocator: std.mem.Allocator,
        src: []const SSMCheckpoint,
        limit: usize,
        budget: ?u64,
    ) !?[]SSMCheckpoint {
        var out = std.ArrayList(SSMCheckpoint).empty;
        errdefer {
            for (out.items) |*c| c.deinit(allocator);
            out.deinit(allocator);
        }
        var spent: u64 = 0;
        var k = src.len;
        while (k > 0) {
            k -= 1;
            const cp = &src[k];
            if (cp.layers.len == 0) continue;
            if (cp.pos > limit) continue;
            const cost = ssmCheckpointBytes(cp);
            if (budget) |b| {
                if (spent + cost > b) break;
            }
            spent += cost;
            try out.append(allocator, try transformer_mod.shareSsmCheckpoint(allocator, cp));
        }
        if (out.items.len == 0) {
            out.deinit(allocator);
            return null;
        }
        // Collected newest-first; the list contract is ascending by pos.
        std.mem.reverse(SSMCheckpoint, out.items);
        return try out.toOwnedSlice(allocator);
    }

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
            const drop = transformer_mod.ssmCheckpointDropIndex(cps[0..n], self.cp_thin);
            const freed = ssmCheckpointBytes(&cps[drop]);
            // Defensive: the shared selection never picks the last.
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

    /// Re-clamp the byte budget after the machine's residency changed (a model loaded
    /// or unloaded beside this one). Shrinking evicts LRU down to the new cap; 0 = uncapped.
    pub fn setBudget(self: *HotPrefixCache, max_kv_bytes: u64) void {
        if (max_kv_bytes == self.max_kv_bytes) return;
        if (max_kv_bytes >> 20 != self.max_kv_bytes >> 20)
            log.info("  [hot-cache] budget revised {d} -> {d} MB\n", .{ self.max_kv_bytes >> 20, max_kv_bytes >> 20 });
        const shrank = max_kv_bytes != 0 and (self.max_kv_bytes == 0 or max_kv_bytes < self.max_kv_bytes);
        self.max_kv_bytes = max_kv_bytes;
        if (!shrank) return;
        while (self.current_kv_bytes > max_kv_bytes and self.entries.items.len > 0) {
            const before = self.entries.items.len;
            self.evictOneLru("budget revised", null);
            if (self.entries.items.len == before) break; // every survivor is checked out
        }
        self.shedCheckpointsToFit();
        self.logResident();
    }

    fn evictOneLru(self: *HotPrefixCache, reason: []const u8, incoming_key: ?u64) void {
        const idx = self.lruIndexExcluding(null, incoming_key) orelse return;
        self.evictAt(idx, reason);
    }

    fn evictAt(self: *HotPrefixCache, lru_idx: usize, reason: []const u8) void {
        var evicted = self.entries.swapRemove(lru_idx);
        const tokens_len = evicted.tokens.len;
        const kv_mb = @as(f64, @floatFromInt(evicted.kv_bytes)) / (1024.0 * 1024.0);
        const had_ssm = evicted.ssm_checkpoints != null;
        const ssm_mb = @as(f64, @floatFromInt(evicted.ssm_bytes)) / (1024.0 * 1024.0);
        const key = evicted.cache_key;
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        if (had_ssm) {
            log.info("  [hot-cache] evicted LRU entry ({s}; key={x}; was {d} tokens, {d:.2} MB; ssm {d:.2} MB)\n", .{
                reason, key, tokens_len, kv_mb, ssm_mb,
            });
        } else {
            log.info("  [hot-cache] evicted LRU entry ({s}; key={x}; was {d} tokens, {d:.2} MB)\n", .{
                reason, key, tokens_len, kv_mb,
            });
        }
    }

    /// Bytes the cache currently holds resident. A hint for the connection thread's admission
    /// guard; the decision that matters is made on the inference thread by `evictLruToAdmit`.
    /// Host bytes this cache's SSD writer holds for files not yet written (up to the permit,
    /// ~1 GiB), peaking at a long prefill's chunk boundary. Inference thread only.
    pub fn stagedHostBytes(self: *HotPrefixCache) u64 {
        const d = if (self.disk) |*dd| dd else return 0;
        return d.stagedHostBytes();
    }

    pub fn residentBytes(self: *const HotPrefixCache) u64 {
        return self.current_kv_bytes;
    }

    /// Bytes an eviction pass can prove it will get back: the residency minus the largest
    /// single entry (a restore pins at most one, and the guard cannot know which).
    pub fn reclaimableBytes(self: *const HotPrefixCache) u64 {
        var largest: u64 = 0;
        // A checked-out entry is neither restorable nor evictable; it comes off the base.
        var checked_out: u64 = 0;
        for (self.entries.items) |*e| {
            if (e.checked_out_by != null) {
                checked_out += e.kv_bytes;
                continue;
            }
            largest = @max(largest, e.kv_bytes);
        }
        return self.current_kv_bytes -| checked_out -| largest;
    }

    /// One resident entry, reduced to what a connection thread may know about it (it may never
    /// dereference `hot_prefix_cache`). `fingerprint` hashes the first `DIGEST_PREFIX_TOKENS`
    /// ids, the restore floor; a shorter entry can never be pinned and gets no digest.
    pub const EntryDigest = struct {
        fingerprint: u64,
        len: u32,
        kv_bytes: u64,
    };

    pub const DIGEST_PREFIX_TOKENS: usize = MIN_CANCELLED_COMMIT_TOKENS;

    /// FNV-1a over the first `DIGEST_PREFIX_TOKENS` ids; null under the restore floor.
    pub fn prefixFingerprint(tokens: []const u32) ?u64 {
        if (tokens.len < DIGEST_PREFIX_TOKENS) return null;
        var h: u64 = 0xcbf29ce484222325;
        for (tokens[0..DIGEST_PREFIX_TOKENS]) |t| {
            h ^= t;
            h *%= 0x100000001b3;
        }
        return h;
    }

    /// Snapshot the resident entries for publication. Caller owns the slice.
    pub fn digestsAlloc(self: *const HotPrefixCache, allocator: std.mem.Allocator) ![]EntryDigest {
        var out = std.ArrayList(EntryDigest).empty;
        errdefer out.deinit(allocator);
        for (self.entries.items) |*e| {
            if (e.checked_out_by != null) continue;
            const fp = prefixFingerprint(e.tokens) orelse continue;
            try out.append(allocator, .{
                .fingerprint = fp,
                .len = @intCast(@min(e.tokens.len, std.math.maxInt(u32))),
                .kv_bytes = e.kv_bytes,
            });
        }
        return out.toOwnedSlice(allocator);
    }

    /// The connection thread's half of the rule: residency minus the largest entry this prompt
    /// could restore from. The pin condition is the fingerprint match alone (a longer record
    /// still restores, clamped); over-crediting is the unsafe direction.
    pub fn reclaimableFromDigests(
        digests: []const EntryDigest,
        residency: u64,
        prompt_fingerprint: ?u64,
    ) u64 {
        const fp = prompt_fingerprint orelse return residency;
        var pinned: u64 = 0;
        for (digests) |d| {
            if (d.fingerprint == fp) pinned = @max(pinned, d.kv_bytes);
        }
        return residency -| pinned;
    }

    /// `reclaimableBytes` with the prompt in hand: only an entry this prompt could restore from
    /// is unevictable. Conservative on both counts (key filters not applied, the LARGEST match
    /// withheld), so never larger than the truth and never smaller than `reclaimableBytes()`.
    pub fn reclaimableBytesFor(self: *const HotPrefixCache, prompt_tokens: []const u32) u64 {
        var pinned: u64 = 0;
        var checked_out: u64 = 0;
        for (self.entries.items) |*e| {
            if (e.checked_out_by != null) {
                checked_out += e.kv_bytes;
                continue;
            }
            const max_shared = @min(e.tokens.len, prompt_tokens.len);
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_tokens[shared]) shared += 1;
            if (shared < MIN_CANCELLED_COMMIT_TOKENS) continue;
            pinned = @max(pinned, e.kv_bytes);
        }
        return self.current_kv_bytes -| checked_out -| pinned;
    }

    /// Evict least-recently-used entries until `fits()` says the request fits (#353): a cached
    /// prefix is an optimization, the request is the work. Never evicts the entry this request
    /// restored from; `fits` is re-asked after every eviction.
    /// Smallest eviction whose live/billed ratio is meaningful.
    pub const SHARED_RATIO_MIN_BYTES: u64 = 1 << 20;

    /// An eviction returning less than 1/Nth of what the entry was billed gave the allocator
    /// nothing (refcount-shared with a live cache). A ratio, not a floor: a small exclusive
    /// entry also returns little.
    pub const SHARED_RETURN_DIVISOR: u64 = 4;

    /// A restore is a lien on the whole entry: `KVCache.restore` shares the entire allocation
    /// and `truncate` frees nothing, so a slot that matched 11 tokens of a 524k entry held all
    /// 11.5 GB and the admission pass could not touch it. A restore may not pin an entry that
    /// hands back less than 1/`RESTORE_PIN_RATIO` of what it pins; small entries never reach the test.
    pub const RESTORE_PIN_MIN_BYTES: u64 = 1 << 30;
    pub const RESTORE_PIN_RATIO: usize = 64;

    /// Would restoring `shared` rows from this entry be a lien rather than a hit?
    pub fn restoreWouldPinEntry(kv_bytes: u64, min_bytes: u64, entry_tokens: usize, shared: usize) bool {
        if (kv_bytes < min_bytes) return false;
        if (shared == 0) return true;
        return entry_tokens > shared *| RESTORE_PIN_RATIO;
    }

    pub fn evictLruToAdmit(
        self: *HotPrefixCache,
        seq_tokens: u64,
        ctx: ?*anyopaque,
        fits: *const fn (?*anyopaque) bool,
        protect_restored: bool,
    ) EvictionReport {
        var report = EvictionReport{};
        while (!fits(ctx)) {
            const idx = self.lruIndexExcluding(if (protect_restored) self.last_restored_used else null, null) orelse break;
            // Accounting bytes are what the entry was billed; live bytes are what the allocator got back.
            var live_before: usize = 0;
            _ = mlx.mlx_get_active_memory(&live_before);
            const acct_before = self.current_kv_bytes;
            self.evictAt(idx, "admitting a long prefill");
            var live_after: usize = 0;
            _ = mlx.mlx_get_active_memory(&live_after);
            const freed_live: u64 = @as(u64, live_before) -| @as(u64, live_after);
            report.entries += 1;
            report.bytes += freed_live;
            const acct_delta = acct_before -| self.current_kv_bytes;
            report.accounted_bytes += acct_delta;
            // Judge only entries big enough for the ratio to mean something.
            if (acct_delta >= SHARED_RATIO_MIN_BYTES and
                freed_live * SHARED_RETURN_DIVISOR < acct_delta)
            {
                report.shared_stop = true;
                break;
            }
        }
        report.admitted = fits(ctx);
        if (report.entries > 0) {
            log.info("  [hot-cache] evicted {d} entries ({d} MB live, {d} MB billed) to admit a {d}-token prefill{s}\n", .{
                report.entries,
                report.bytes / (1024 * 1024),
                report.accounted_bytes / (1024 * 1024),
                seq_tokens,
                if (report.shared_stop) " — stopped: the next entry is shared with a live request" else "",
            });
        }
        return report;
    }

    /// Least-recently-used entry index, skipping the one whose `last_used` equals `protect`.
    /// Workload-fair: only entries of the `cache_key` holding the most eligible entries
    /// (`incoming_key` counts as one more) are candidates, so a sweep evicts its own
    /// documents before another workload's conversation. One key = plain LRU.
    fn lruIndexExcluding(self: *const HotPrefixCache, protect: ?u64, incoming_key: ?u64) ?usize {
        var best: ?usize = null;
        var best_used: u64 = std.math.maxInt(u64);
        var max_count: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            // Held by a live slot that owns its buffers: evicting it frees nothing.
            if (e.checked_out_by != null) continue;
            if (protect) |p| {
                if (e.last_used == p) continue;
            }
            var count: usize = if (incoming_key != null and incoming_key.? == e.cache_key) 1 else 0;
            for (self.entries.items) |*o| {
                if (o.checked_out_by != null or o.cache_key != e.cache_key) continue;
                if (protect) |p| {
                    if (o.last_used == p) continue;
                }
                count += 1;
            }
            if (count > max_count or (count == max_count and e.last_used < best_used)) {
                max_count = count;
                best_used = e.last_used;
                best = i;
            }
        }
        return best;
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

    /// Drop all entries. Called when the cache is suspect (pad-only generation, image-bearing
    /// prompt, tools toggle change).
    pub fn invalidateAll(self: *HotPrefixCache, reason: []const u8) void {
        // Suspect state must die on both tiers.
        if (self.disk) |*d| d.invalidateAll();
        self.disk_dirty = false;
        if (self.pending_disk) |*p| {
            p.deinit(self.allocator);
            self.pending_disk = null;
        }
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
        if (self.pending_disk) |*p| {
            p.deinit(self.allocator);
            self.pending_disk = null;
        }
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
    _ = try hc.commit(&src, toks[0..logical_len], false);

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
    _ = try hc.commitWithSsm(&trunk, &toks, false, null, .{ .cache = &assist, .base_pos = 10 }, null);
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
    _ = try hc2.commit(&trunk, &toks, false);
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
    _ = try hc.commitWithSsm(&trunk, &toks, false, null, null, .{ .cache = &hist, .base_pos = 0 });
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

fn testWriteCacheLayer(cache: *KVCache, s: mlx.mlx_stream, layer: u32, written: u32, step: u32) !void {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const count: f64 = @floatFromInt(step * 8);
    const base: f64 = @floatFromInt(written * 8 + layer * 1_000_000);
    try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    const shape = [_]c_int{ 1, 1, @intCast(step), 8 };
    try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s));
    var view = try cache.update(layer, k, k, s, 0);
    view.deinit();
}

fn testFillCache(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, tokens: u32) !void {
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) try testWriteCacheLayer(cache, s, li, written, step);
        written += step;
    }
}

/// Fill a head-shaped cache: one layer at `layer` (the qwen4_exp MTP head's layer is never 0),
/// driven through `Transformer.qwen4MtpAdvance` as `qwen4MtpForward` drives it.
fn testFillHeadCache(cache: *KVCache, s: mlx.mlx_stream, layer: u32, tokens: u32, seq_offset: *usize) !void {
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        try testWriteCacheLayer(cache, s, layer, written, step);
        transformer_mod.Transformer.qwen4MtpAdvance(cache, seq_offset, @intCast(step));
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
        _ = try hc.commit(&cache, &tokens, false);
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

test "HotPrefixCache: budget decline reports a status, not a silent success" {
    // Live 2026-09-07: every oversized decline from a cancelled prefill
    // logged "[hot-cache] committed N/M" at the scheduler because the
    // decline exits commitWithMediaState via plain return — a 122k agent
    // session looked cached while it cold-prefilled ~95k tokens per retry.
    // The commit contract must be observable: ok / kept_resident / declined.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // 16 KB budget against ~128 B/token rows: even the 256-token commit
    // floor cannot fit, so the candidate must be DECLINED — and the caller
    // must be able to SEE that instead of mistaking it for a commit.
    const st = try hc.commit(&cache, &tokens, false);
    try testing.expect(st == .declined);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());

    // A candidate that fits commits with its effective token length.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    try testFillCache(&cache2, s, 2, 600);
    const st2 = try hc2.commit(&cache2, &tokens, false);
    try testing.expect(st2 == .ok);
    try testing.expectEqual(@as(usize, 600), st2.ok);
}

test "HotPrefixCache: an entry shorter than its media boundary is pure text" {
    // Live 2026-09-07: a cancelled prefill that forwarded fewer tokens than
    // the request's first media position kept the PIXEL vision_key while
    // the boundary was dropped (scheduler shape: media_start=null, key kept)
    // — a boundary-less cross-key entry the lookup conservatively rejects.
    // The with-image and no-image retry shapes of one conversation locked
    // each other out of their shared 121819-token text prefix. An entry
    // whose [0, len) covers no media rows is text: key it as text.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Cancel shape (the scheduler passes the boundary raw): media declared
    // at 500, entry covers [0, 400) — no media rows inside, so the entry
    // must be text-keyed.
    const st = try hc.commitWithMediaState(&cache, tokens[0..400], false, 0xABCD, 0, 500, null, null, null);
    try testing.expect(st == .ok);
    try testing.expectEqual(@as(u64, 0), hc.entries.items[0].vision_key);
    try testing.expect(hc.entries.items[0].media_start == null);

    // Legacy shape (commitWithState's contract): a null boundary KEEPS the
    // caller's pixel key — boundary-less keyed entries stay possible for
    // callers that model their own media state.
    var cache_b = try KVCache.init(testing.allocator, 2);
    defer cache_b.deinit();
    try testFillCache(&cache_b, s, 2, 600);
    _ = try hc.commitWithState(&cache_b, tokens[0..400], false, 0xBEEF, null, null, null);
    try testing.expectEqual(@as(u64, 0xBEEF), hc.entries.items[1].vision_key);

    // The no-image retry shape (vision_key 0, no media) must restore it.
    {
        var c2 = try KVCache.init(testing.allocator, 2);
        defer c2.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreWithMedia(&c2, &moe, null, s, &tokens, false, 0, null, null, null, null);
        try testing.expectEqual(@as(usize, 400), res.matched);
    }
    // A different-image request caps at ITS media boundary (500 > 400 here,
    // so the whole text entry is reusable through it).
    {
        var c3 = try KVCache.init(testing.allocator, 2);
        defer c3.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreWithMedia(&c3, &moe, null, s, &tokens, false, 0xFFFF, 500, null, null, null);
        try testing.expectEqual(@as(usize, 400), res.matched);
    }

    // Trim shape: media INSIDE the candidate, budget forces a trim below the
    // boundary — the trimmed entry is text and must re-key to 0.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 32 * 1024);
    defer hc2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    try testFillCache(&cache2, s, 2, 600);
    const st2 = try hc2.commitWithMediaState(&cache2, &tokens, false, 0xABCD, 0, 300, null, null, null);
    try testing.expect(st2 == .ok);
    try testing.expectEqual(@as(usize, 256), st2.ok); // 32 KB / 128 B rows
    try testing.expectEqual(@as(u64, 0), hc2.entries.items[0].vision_key);
    try testing.expect(hc2.entries.items[0].media_start == null);
}

test "HotPrefixCache: media request restores the pre-media text prefix from SSD" {
    // Live 2026-09-07: the SSD-tier lookup skipped ANY request with a
    // non-zero vision_key (`vision_key != 0 → break :disk`) — a 122k agent
    // session whose LAST turn carries one screenshot never touched the
    // 40 GB of persisted text prefixes, though 99.8% of the prompt is
    // pixel-independent text. Disk entries are text-only by construction
    // (the flush refuses vision-keyed entries), so the restore is exactly
    // the RAM path's pre-media semantics: cap at the request's media_start.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: a text-only conversation commits + flushes to disk.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-vision-disk", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 (RAM empty): an image-bearing request over the same text —
    // the disk tier must serve the shared text prefix, capped at the
    // request's media boundary (400), reading only the covering chunks.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-vision-disk", 0, 128);
        defer hc2.deinit();

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestoreWithMedia(&cache2, &moe, null, s, &tokens, false, 0xDEAD, 400, null, null, null);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 400), res.matched);
        try testing.expectEqual(@as(usize, 400), cache2.step);
        try testing.expectEqual(@as(u32, 4), hc2.disk.?.chunks_loaded_last); // ceil(400/128)
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

        _ = try hc.commitWithSsm(&trunk, &tokens, false, null, .{ .cache = &assist, .base_pos = 0 }, .{ .cache = &hist, .base_pos = 0 });
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
    _ = try hc.commit(&cache, &tokens, false);
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
    _ = try hc.commit(&cache, &tokens, false);
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
    _ = try hc.commitWithMediaState(
        &source_cache,
        &cached_tokens,
        false,
        0x1111,
        0,
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
    _ = try hc.commitWithState(&older_cache, &older_tokens, false, vision_key, older_cps, null, null);

    var newer_cache = try KVCache.init(testing.allocator, 3);
    defer newer_cache.deinit();
    try testFillCache(&newer_cache, s, 3, newer_tokens.len);
    var newer_ssm = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&newer_ssm);
    const newer_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    // The raw match with this entry is 7, so this checkpoint cannot restore it.
    newer_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &newer_ssm, 8, s);
    _ = try hc.commitWithState(&newer_cache, &newer_tokens, false, vision_key, newer_cps, null, null);

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
        _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
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

test "HotPrefixCache: chunk-heavy hybrid flush still lands its SSM checkpoints" {
    // Live 2026-09-07: checkpoints were written AFTER the chunks from the
    // REMAINING per-flush budget — a turn appending ≥ the cap in chunks
    // (cancel-salvage retries: +16 chunks ≈ 544 MB vs a 512 MB cap) left
    // ZERO checkpoint budget every time, and the same-token catch-up path
    // never fires for a growing conversation. Every entry of the Sep-4
    // disk wave landed KV-only: unrestorable, gigabytes of dead chunks.
    // Checkpoints must come off the TOP of the budget, not the bottom.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: hybrid commit + a flush budget tight enough that the KV
    // chunks alone would consume it (5 chunks ≈ 24 KB each vs a 2-chunk
    // cap) — the exact starvation shape, scaled down.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-cp-starve", 0, 128);
        defer hc.deinit();
        hc.disk.?.max_flush_bytes = 2 * 128 * 3 * 2 * 8 * 4; // two KV chunks

        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 600);
        var src = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s);
        cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
        hc.flushPendingDisk(s);
    }

    // Session 2 (RAM empty): the hybrid lookup must restore from the disk
    // entry's highest checkpoint ≤ its flushed kv_len (256 of 5 chunks) —
    // not cold-miss on a checkpoint-less entry.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-cp-starve", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var ssm2 = pcEmptySsm();
        defer pcFreeHybrid(&ssm2);
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 256), res.matched);
        try testing.expectEqual(@as(usize, 256), cache2.step);
        try testing.expectEqual(@as(f32, 100.0), pcSsmVal(ssm2[0].conv_state, 0, s));
    }
}

test "HotPrefixCache: under SSD-first a decline rides pending_disk, never a second synchronous spill" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-spill-ssdfirst", 0, 128);
    defer hc.deinit();
    const cap_before = hc.disk.?.max_flush_bytes;

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const st = try hc.commit(&cache, &tokens, false);
    try testing.expect(st == .declined);
    // The capture at commit time is the one record; nothing lands until the flush.
    try testing.expect(hc.pending_disk != null);
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
    try testing.expectEqual(cap_before, hc.disk.?.max_flush_bytes);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
}

test "HotPrefixCache: a budget-declined candidate spills to the SSD tier" {
    // Live 2026-09-07: an oversized decline discarded the cancelled
    // prefill's work entirely — under a 1 GB hot budget a 122k session's
    // 76-82k-token forwarded prefixes were thrown away on every retry
    // while a 40 GB SSD tier sat idle. The decline must OFFER the
    // candidate to the disk tier instead of discarding it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // A RAM budget too small for even the commit floor forces the decline.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .declined);
        try testing.expectEqual(@as(usize, 0), hc.entryCount()); // RAM keeps nothing
        // The candidate's work must live on the SSD tier.
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // The next request (fresh RAM, generous budget) restores from SSD
    // instead of cold-prefilling the whole prefix.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe, null, s, &tokens, false, 0, null, null);
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
    }
}

test "HotPrefixCache: a decline-spill is not bounded by the per-flush byte cap" {
    // The per-flush cap bounds the stall a LIVE next request pays after a
    // response. A decline-spill runs on a request whose client is already
    // gone — capping it (live 2026-09-07: 512 MB ≈ 13k tokens banked per
    // retry) stranded most of each retry's work while ~35k tokens were
    // recomputed every time. The spill owes no latency; the tier's byte
    // budget + LRU is the real bound.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill-cap", 0, 128);
        defer hc.deinit();
        // A cap of two KV chunks — the exact starvation shape, scaled down.
        hc.disk.?.max_flush_bytes = 2 * 128 * 2 * 2 * 8 * 4;

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .declined);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // The WHOLE candidate must be restorable, not just the cap's worth.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill-cap", 0, 128);
        defer hc2.deinit();
        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe, null, s, &tokens, false, 0, null, null);
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
    }
}

test "HotPrefixCache: hybrid disk restore ranks entries by restorable checkpoint, not raw length" {
    // The RAM tier learned this the hard way (#312): a longer raw match
    // whose checkpoints sit past the divergence restores nothing. The disk
    // tier's hybrid arm used bestMatch (ranked by usable length) and then
    // took that ONE entry's checkpoint — an entry with a high usable length
    // but low checkpoints shadowed a shorter entry with a higher
    // restorable position (live 2026-09-07: the growing conversation entry
    // would have shadowed the old entry's cp@51200 with cp@49152).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Entry A: shares 550 tokens with the prompt (longer) but its only
    // checkpoint is at 256. Diverges from the prompt at 550.
    var a_tokens: [600]u32 = undefined;
    for (&a_tokens, 0..) |*t, i| t.* = if (i < 550) @intCast(i + 7) else @intCast(i + 700);
    // Entry B: shares only 512 tokens (shorter) but carries a checkpoint
    // at 512. Diverges from the prompt at 512 — so neither entry is a
    // prefix of the other and the flush cannot merge them.
    var b_tokens: [520]u32 = undefined;
    for (&b_tokens, 0..) |*t, i| t.* = if (i < 512) @intCast(i + 7) else @intCast(i + 900);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-rank", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 600);
        var src = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src);
        const cps_a = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_a[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s);
        _ = try hc.commitWithSsm(&cache, &a_tokens, false, cps_a, null, null);
        hc.flushPendingDisk(s);
        var cache_b = try KVCache.init(testing.allocator, 3);
        defer cache_b.deinit();
        try testFillCache(&cache_b, s, 3, 600);
        const cps_b = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_b[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache_b, &b_tokens, false, cps_b, null, null);
        hc.flushPendingDisk(s);
    }

    // RAM empty; the hybrid lookup must restore from B's cp@512, not A's
    // cp@256 — even though A's raw usable length (600) beats B's (512).
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-rank", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 2), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var ssm2 = pcEmptySsm();
        defer pcFreeHybrid(&ssm2);
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expectEqual(@as(usize, 512), cache2.step);
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
    _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
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
    _ = try hc.commitWithSsm(&c1, tokens[0..450], false, cps1, null, null);
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
    _ = try hc.commitWithSsm(&c2, tokens[0..900], false, cps2, null, null);

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

test "HotPrefixCache: a revised budget evicts down to fit and a raised one keeps everything" {
    const s = mlx.gpuStream();
    var a = try KVCache.init(testing.allocator, 2);
    defer a.deinit();
    try testFillCache(&a, s, 2, 600);
    var b = try KVCache.init(testing.allocator, 2);
    defer b.deinit();
    try testFillCache(&b, s, 2, 600);
    var toks_a: [600]u32 = undefined;
    for (&toks_a, 0..) |*t2, i| t2.* = @intCast(i + 1);
    var toks_b: [600]u32 = undefined;
    for (&toks_b, 0..) |*t2, i| t2.* = @intCast(i + 1000);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commit(&a, &toks_a, false);
    _ = try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    const one = hc.entries.items[0].kv_bytes;

    // Shrink to one entry's worth: the LRU (A) goes, B stays whole.
    hc.setBudget(one);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    try testing.expectEqual(@as(u32, 1000), hc.entries.items[0].tokens[0]);
    // Raise (0 = uncapped): nothing moves.
    hc.setBudget(0);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.max_kv_bytes);
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

    _ = try hc.commit(&a, &toks_a, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // B extends A, so this exercises the replacement path. It is too large
    // for the cap: the byte budget still holds — but as a TRIM (#330), not a
    // decline. The retained prefix is longer than A's, shorter than B's.
    @memcpy(toks_b[0..toks_a.len], &toks_a);
    _ = try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const kept_len = hc.entries.items[0].tokens.len;
    try testing.expect(kept_len >= MIN_CANCELLED_COMMIT_TOKENS);
    try testing.expect(kept_len < toks_b.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // An empty cache retains the same trimmed prefix from the oversized
    // candidate rather than staying empty (#330: the pre-fix decline held the
    // cap by holding zero bytes).
    hc.invalidateAll("test");
    _ = try hc.commit(&b, &toks_b, false);
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
    _ = try hc.commit(&src, &toks, false);

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
    _ = try hc.commit(&src, toks[0..600], false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    const resident_keys_ctx = hc.entries.items[0].snapshot.entries[0].keys.ctx;

    // Next turn: the conversation grew, the trim target did not. Re-copying
    // an identical prefix every turn would be a per-turn multi-GB memcpy.
    var src2 = try KVCache.init(testing.allocator, 2);
    defer src2.deinit();
    try testFillCache(&src2, s, 2, 700);
    _ = try hc.commit(&src2, &toks, false);
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
    _ = try hc.commitWithSsm(&c1, &toks, false, cps, null, null);

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
    _ = try hc.commitWithSsm(&c1, &toks, false, cps, null, null);
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
        _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

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

test "HotPrefixCache: a history tensor shorter than the checkpoint is a miss, not a short hit" {
    const s = mlx.gpuStream();
    var tokens: [70]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    var lookup_latest: [70]u32 = tokens;
    lookup_latest[64] = 999;
    var lookup_interior: [70]u32 = tokens;
    lookup_interior[16] = 999;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 8, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 16, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 64, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    try testing.expectEqual(@as(c_int, 8), mlx.getShape(cps[1].layers[0].aux_state)[1]);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[0].qsa_rows);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    for ([_][]const u32{ &lookup_latest, &lookup_interior }) |lookup| {
        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        try testing.expectError(error.QsaHistoryGap, hc.lookupAndRestore(&target_cache, &moe_off, &target, s, lookup, false, 0, null, null));
        try testing.expectEqual(@as(usize, 0), target_cache.step);
        try testing.expect(target[0].aux_state.ctx == null);
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
    _ = try hc.commitWithState(&c1, &t1, false, 0, cps1, null, null);

    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, t2.len);
    var l2 = pcBuildQsaHybrid(s, 14, 200.0);
    defer pcFreeQsaHybrid(&l2);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps2[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &l2, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps2, &l2, s);
    _ = try hc.commitWithState(&c2, &t2, false, 0, cps2, null, null);

    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const merged = hc.entries.items[0].ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 2), merged.len);
    try testing.expect(!transformer_mod.checkpointHasQsaHistory(&merged[0]));
    try testing.expect(transformer_mod.checkpointHasQsaHistory(&merged[1]));
    try testing.expectEqual(@as(c_int, 8), mlx.getShape(merged[1].layers[0].aux_state)[1]);
}

test "HotPrefixCache: a handed-off QSA history commits, restores and bills exactly like the prefill-end copy" {
    // The commit handoff gives the newest snap a view of the slot's live history; the entry must
    // be indistinguishable from the copy arm and outlive the slot.
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const lookup_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 70, 71 };
    var restored_val: [2]f32 = .{ -1.0, -2.0 };
    var billed: [2]u64 = .{ 0, 0 };
    for ([_]bool{ false, true }, 0..) |handoff, arm| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        var live = pcBuildQsaHybrid(s, 10, 100.0);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 4, s);
        if (handoff) {
            try transformer_mod.handoffQsaHistoryToLatest(cps, &live, s);
        } else {
            try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        }
        _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);
        pcFreeQsaHybrid(&live);
        billed[arm] = hc.entries.items[0].ssm_bytes;

        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup_tokens, false, 0, null, null);
        try testing.expectEqual(@as(usize, 4), r.matched);
        try testing.expectEqual(@as(c_int, 4), mlx.getShape(target[0].aux_state)[1]);
        var got = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_astype(&got, target[0].aux_state, .float32, s));
        try mlx.check(mlx.mlx_array_eval(got));
        const d = mlx.mlx_array_data_float32(got) orelse return error.TestUnexpectedNullData;
        restored_val[arm] = d[3 * 8 + 5];
    }
    try testing.expectEqual(@as(f32, 3.0 * 8.0 + 5.0), restored_val[1]);
    try testing.expectEqual(restored_val[0], restored_val[1]);
    try testing.expect(billed[0] > 0);
    try testing.expectEqual(billed[0], billed[1]);
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
    _ = try hc.commitWithSsm(&c1, toks[0..450], false, cps1, null, null);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    _ = try hc.commitWithSsm(&c2, &toks, false, cps2, null, null);

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
    try testing.expectError(error.OutOfMemory, hc.commitWithMediaState(&src, &toks, false, 0, 0, null, cps, null, null));
    // No frees here: the cache owns the checkpoints on every outcome.
}

test "HotPrefixCache: a commit from a restored prefix inherits the donor's checkpoints" {
    // The 64k-ladder miss. Each rung sends the same growing prompt twice: an
    // MTP arm cold-prefills and commits entry A (checkpoints at stride), then
    // a serial arm restores ~the whole prompt from A, prefills the ~31-token
    // tail and commits its OWN entry B. B's tokens are NOT a prefix-extension
    // of A's (the two arms generate different tails), so the replace path —
    // the only checkpoint inheritance there was — never runs, and B's own
    // prefill was too short to earn a reachable checkpoint (a <= 30-token
    // tail takes `ssmSnapshotBackoff` 0, so its sole snapshot lands AT the
    // prompt end, past any later match). Once the byte budget evicted A, the
    // next rung found only B, every candidate `continue`d in
    // findBestRestorableMatch, and a 393k-token prompt cold-prefilled for
    // 560 s with no `[hot-cache]` line at all.
    const s = mlx.gpuStream();

    // Shared prompt P, then two different generated tails.
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 1);
    const a_tokens = prompt ++ [_]u32{ 200, 201 };
    const b_tokens = prompt ++ [_]u32{ 210, 211 };

    var srcs: [4][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    hc.ssm_checkpoint_max = 8;
    defer hc.deinit();

    // Entry A: the MTP arm's cold prefill. Checkpoints at 8 and 16 (inside
    // the shared prompt) and one at 21 (inside its OWN generated tail, which
    // B never saw and must not inherit).
    var a_cache = try KVCache.init(testing.allocator, 3);
    defer a_cache.deinit();
    try testFillCache(&a_cache, s, 3, a_tokens.len);
    const a_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    a_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[0], 8, s);
    a_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[1], 16, s);
    a_cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[2], 21, s);
    _ = try hc.commitWithState(&a_cache, &a_tokens, false, 0, a_cps, null, null);

    // Entry B: the serial arm. It restored from A and prefilled a tail too
    // short for a backoff, so its only checkpoint sits at the prompt end.
    var b_cache = try KVCache.init(testing.allocator, 3);
    defer b_cache.deinit();
    try testFillCache(&b_cache, s, 3, b_tokens.len);
    const b_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    b_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[3], 20, s);
    _ = try hc.commitWithState(&b_cache, &b_tokens, false, 0, b_cps, null, null);

    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    // B carries A's in-prompt checkpoints, and NOT the one at 21 (a position
    // only A's own generated tail ever reached).
    const b_idx: usize = if (hc.entries.items[0].tokens[20] == 210) 0 else 1;
    const b_merged = hc.entries.items[b_idx].ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 3), b_merged.len);
    try testing.expectEqual(@as(usize, 8), b_merged[0].pos);
    try testing.expectEqual(@as(usize, 16), b_merged[1].pos);
    try testing.expectEqual(@as(usize, 20), b_merged[2].pos);

    // The count cap evicts A (the byte budget is the same mechanism).
    var c_cache = try KVCache.init(testing.allocator, 3);
    defer c_cache.deinit();
    try testFillCache(&c_cache, s, 3, 4);
    const c_tokens = [_]u32{ 90, 91, 92, 93 };
    _ = try hc.commitWithState(&c_cache, &c_tokens, true, 0, null, null, null);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    for (hc.entries.items) |*e| {
        try testing.expect(e.tokens.len != a_tokens.len or e.tokens[20] != 200);
    }

    // The next rung: shares the first 17 prompt tokens, then diverges (the
    // template's generation suffix renders differently once the turn enters
    // history). B's own checkpoint at 20 cannot serve it; A's at 16 can, and
    // B now carries it.
    const next = prompt[0..17].* ++ [_]u32{ 50, 51, 52 };
    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target_ssm = pcEmptySsm();
    defer pcFreeHybrid(&target_ssm);
    var moe_off: usize = 0;
    const result = try hc.lookupAndRestore(&target_cache, &moe_off, &target_ssm, s, &next, false, 0, null, null);

    try testing.expectEqual(@as(usize, 16), result.matched);
    try testing.expectEqual(@as(usize, 16), target_cache.step);
    try testing.expectEqual(@as(usize, 16), moe_off);
    // The restored state is the checkpoint A captured at 16 (srcs[1]).
    try testing.expectEqual(@as(f32, 200.0), pcSsmVal(target_ssm[0].conv_state, 0, s));
}

test "prefix cache: a hybrid miss with a raw token match names itself" {
    // The silent null: `findBestRestorableMatch` rejects every candidate that
    // has no SSM checkpoint at or below its shared prefix, so the lookup can
    // return null with a LONG raw match behind it — and the `match == null`
    // arm logged nothing at all. `missKind` is the seam: a genuinely cold
    // cache stays quiet, an expensive miss gets a line.
    try testing.expectEqual(MissKind.cold, missKind(0, 0));
    try testing.expectEqual(MissKind.cold, missKind(0, 100_000));
    try testing.expectEqual(MissKind.cold, missKind(3, 0));
    // Below the commit floor there was never a prefix worth restoring.
    try testing.expectEqual(MissKind.cold, missKind(3, MIN_CANCELLED_COMMIT_TOKENS - 1));
    // At and above it, a cold prefill of that many tokens owes an explanation.
    try testing.expectEqual(MissKind.no_checkpoint, missKind(1, MIN_CANCELLED_COMMIT_TOKENS));
    try testing.expectEqual(MissKind.no_checkpoint, missKind(4, 393_000));
}

test "prefix cache: the no-match lookup arm consults missKind, never returns silently" {
    // Class guard for the 560 s unexplained cold prefill: every early return
    // from the lookup owes a reason. The `match == null` arm is the one that
    // had none, and it is reachable only through this file.
    const source = @embedFile("prefix_cache.zig");
    const start = std.mem.indexOf(u8, source, "if (match == null) {") orelse
        return error.MissingNoMatchArm;
    const arm = source[start .. start + 900];
    const end = std.mem.indexOf(u8, arm, "return .{ .matched = 0, .full_match = false };") orelse
        return error.MissingNoMatchReturn;
    try testing.expect(std.mem.indexOf(u8, arm[0..end], "missKind(") != null);
    try testing.expect(std.mem.indexOf(u8, arm[0..end], "[hot-cache] hybrid miss") != null);
    // The probe must be filled BEFORE the restorability filter can `continue`
    // a candidate away, or `best_raw` is always 0 and the line never fires.
    const fbr = std.mem.indexOf(u8, source, "fn findBestRestorableMatch(") orelse
        return error.MissingFinder;
    const body = source[fbr .. fbr + 2600];
    const probe_at = std.mem.indexOf(u8, body, "if (probe) |p| {") orelse return error.MissingProbe;
    const filter_at = std.mem.indexOf(u8, body, "const effective = if (require_ssm_checkpoint)") orelse
        return error.MissingFilter;
    try testing.expect(probe_at < filter_at);
}

test "prefix cache: an inherited checkpoint SHARES the donor's buffers and is budget-bounded" {
    // The two claims inheritance rests on. (1) Sharing: a clone must outlive
    // the donor — the ladder's whole point is that evicting the entry we
    // inherited from frees nothing the inheritor still needs. (2) Bounding:
    // the per-entry accounting bills shared bytes again, so an unbounded
    // inherit could book an entry past a hard cap; the clone takes the
    // HIGHEST positions that fit and stops.
    const s = mlx.gpuStream();

    var srcs: [3][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    const donor = try testing.allocator.alloc(SSMCheckpoint, 3);
    for (donor, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 8, s);
    }
    const one = transformer_mod.ssmCheckpointBytes(&donor[0]);

    // Unbounded, limit past everything: all three, ascending.
    {
        const all = (try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 100, null)).?;
        defer {
            for (all) |*c| c.deinit(testing.allocator);
            testing.allocator.free(all);
        }
        try testing.expectEqual(@as(usize, 3), all.len);
        try testing.expectEqual(@as(usize, 8), all[0].pos);
        try testing.expectEqual(@as(usize, 24), all[2].pos);
    }

    // A position past the shared prefix describes state this prompt never
    // reached and must not be inherited.
    {
        const two = (try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 16, null)).?;
        defer {
            for (two) |*c| c.deinit(testing.allocator);
            testing.allocator.free(two);
        }
        try testing.expectEqual(@as(usize, 2), two.len);
        try testing.expectEqual(@as(usize, 16), two[1].pos);
    }

    // Budget for one and a half: the HIGHEST reachable position wins.
    {
        const one_only = (try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 100, one + one / 2)).?;
        defer {
            for (one_only) |*c| c.deinit(testing.allocator);
            testing.allocator.free(one_only);
        }
        try testing.expectEqual(@as(usize, 1), one_only.len);
        try testing.expectEqual(@as(usize, 24), one_only[0].pos);
    }
    // Nothing fits: null, never an empty slice the caller must special-case.
    try testing.expectEqual(@as(?[]SSMCheckpoint, null), try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 100, 0));

    // (1) The clone outlives the donor. Free the donor list entirely, then
    // read the shared state back — a copy would be fine here too, but a
    // DANGLING handle would not, and the restore below is what the ladder's
    // next rung actually does.
    const kept = (try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 100, null)).?;
    defer {
        for (kept) |*c| c.deinit(testing.allocator);
        testing.allocator.free(kept);
    }
    for (donor) |*c| c.deinit(testing.allocator);
    testing.allocator.free(donor);

    var dst = pcEmptySsm();
    defer pcFreeHybrid(&dst);
    try transformer_mod.restoreSsmCheckpoint(&dst, &kept[1]);
    try testing.expectEqual(@as(f32, 200.0), pcSsmVal(dst[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 1000.0), pcSsmVal(dst[0].ssm_state, 0, s));
}

test "evictLruToAdmit: oldest first, never the entry THIS request restored, and shared bytes are not counted as freed" {
    // "Most recently used" is not "the entry this request restored", and a restored entry's
    // shared buffers return nothing to the allocator when evicted.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var toks_a: [4096]u32 = undefined;
    for (&toks_a, 0..) |*t, i| t.* = @intCast(i + 1);
    var toks_b: [4096]u32 = undefined;
    for (&toks_b, 0..) |*t, i| t.* = @intCast(i + 1_000_001);
    var toks_c: [4096]u32 = undefined;
    for (&toks_c, 0..) |*t, i| t.* = @intCast(i + 2_000_001);

    inline for (.{ &toks_a, &toks_b, &toks_c }) |toks| {
        var cache = try KVCache.init(testing.allocator, 8);
        defer cache.deinit();
        try testFillCache(&cache, s, 8, 4096);
        // Materialize before committing: an unevaluated cache owns no Metal buffer at all.
        for (cache.entries) |*e| {
            if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
            if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
        }
        _ = try hc.commit(&cache, toks, false);
    }
    try testing.expectEqual(@as(usize, 3), hc.entryCount());
    var live_resident: usize = 0;
    _ = mlx.mlx_get_active_memory(&live_resident);
    try testing.expect(live_resident > 4 * 1024 * 1024);

    // This request restores B, the middle entry. The target cache stays alive: that is what makes B shared.
    var live_b = try KVCache.init(testing.allocator, 8);
    defer live_b.deinit();
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&live_b, &moe_off, null, s, &toks_b, false, 0, null, null);
    try testing.expect(hit.full_match);
    try testing.expect(hc.last_restored_used != null);

    const Never = struct {
        fn call(ctx: ?*anyopaque) bool {
            _ = ctx;
            return false;
        }
    };
    const rep = hc.evictLruToAdmit(458_832, null, Never.call, true);
    try testing.expect(!rep.admitted);
    try testing.expectEqual(@as(usize, 2), rep.entries); // A and C, oldest first
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    var probe_cache = try KVCache.init(testing.allocator, 8);
    defer probe_cache.deinit();
    var off2: usize = 0;
    const still_b = try hc.lookupAndRestore(&probe_cache, &off2, null, s, &toks_b, false, 0, null, null);
    try testing.expect(still_b.full_match);
    // An exclusive eviction may return MORE than it was billed (capacity rounding).
    try testing.expect(rep.bytes > 0);
    try testing.expect(rep.bytes * HotPrefixCache.SHARED_RETURN_DIVISOR >= rep.accounted_bytes);
    try testing.expect(!rep.shared_stop);

    // Unprotected, B goes too, and the allocator gets ~nothing back, which the pass notices.
    const rest = hc.evictLruToAdmit(458_832, null, Never.call, false);
    try testing.expect(!rest.admitted);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expect(rest.accounted_bytes > 0);
    try testing.expect(rest.bytes * HotPrefixCache.SHARED_RETURN_DIVISOR < rest.accounted_bytes);
    try testing.expect(rest.shared_stop);
}

test "a trivial share of a whole session is a LIEN, and the admission pass gets the entry back" {
    // Live shape: a cold 786k prompt shared eleven tokens with the one resident 524k entry, the
    // restore shared the whole 11.5 GB into the slot, and the admission pass could not touch it.
    const t = testing;
    try t.expect(23_692 > 14_713); // refused
    try t.expect(14_713 + 11_476 >= 23_692); // admitted

    const MB: u64 = 1 << 20;
    const floor = HotPrefixCache.RESTORE_PIN_MIN_BYTES;
    try t.expect(HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, 11));
    try t.expect(!HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, 100_000));
    try t.expect(!HotPrefixCache.restoreWouldPinEntry(60 * MB, floor, 218, 11));

    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true; // the arm this was measured on
    hc.restore_pin_min_bytes = 1 << 20; // a 2 MB entry stands in for 11.5 GB

    var session: [4096]u32 = undefined;
    for (&session, 0..) |*x, i| x.* = @intCast(i + 1);
    var cold: [4096]u32 = undefined;
    for (&cold, 0..) |*x, i| x.* = if (i < 11) @as(u32, @intCast(i + 1)) else @intCast(i + 5_000_001);

    // One cache: the slot that served the session is handed the next request.
    var slot_cache = try KVCache.init(testing.allocator, 8);
    defer slot_cache.deinit();
    try testFillCache(&slot_cache, s, 8, 4096);
    for (slot_cache.entries) |*e| {
        if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
        if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
    }
    _ = try hc.commit(&slot_cache, &session, false);
    try t.expectEqual(@as(usize, 1), hc.entryCount());
    try t.expect(hc.entries.items[0].kv_bytes > hc.restore_pin_min_bytes);

    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&slot_cache, &moe_off, null, s, &cold, false, 0, null, null);

    // Declined: eleven rows are not worth a lien on the session.
    try t.expectEqual(@as(usize, 0), hit.matched);
    try t.expect(!hit.full_match);
    try t.expectEqual(@as(usize, 0), moe_off);
    try t.expect(hc.last_restored_used == null);
    try t.expectEqual(@as(usize, 0), slot_cache.step);

    // ...so the admission pass reclaims it, and reclaims real bytes.
    const Fits = struct {
        fn call(ctx: ?*anyopaque) bool {
            const cache: *HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() == 0;
        }
    };
    const rep = hc.evictLruToAdmit(786_369, &hc, Fits.call, true);
    try t.expect(rep.admitted);
    try t.expectEqual(@as(usize, 1), rep.entries);
    try t.expect(rep.bytes > 0);
    try t.expect(!rep.shared_stop);
}

test "a PROPORTIONATE share still restores, and is still protected" {
    // The inverse: an entry a request genuinely continues hands back most of what it pins.
    const t = testing;
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    hc.restore_pin_min_bytes = 1 << 20;

    var session: [4096]u32 = undefined;
    for (&session, 0..) |*x, i| x.* = @intCast(i + 1);
    var warm: [4096]u32 = undefined;
    for (&warm, 0..) |*x, i| x.* = if (i < 2048) @as(u32, @intCast(i + 1)) else @intCast(i + 5_000_001);

    var src_cache = try KVCache.init(testing.allocator, 8);
    defer src_cache.deinit();
    try testFillCache(&src_cache, s, 8, 4096);
    for (src_cache.entries) |*e| {
        if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
        if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
    }
    _ = try hc.commit(&src_cache, &session, false);

    var slot_cache = try KVCache.init(testing.allocator, 8);
    defer slot_cache.deinit();
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&slot_cache, &moe_off, null, s, &warm, false, 0, null, null);
    try t.expectEqual(@as(usize, 2048), hit.matched);
    try t.expect(hc.last_restored_used != null);

    const Never = struct {
        fn call(ctx: ?*anyopaque) bool {
            _ = ctx;
            return false;
        }
    };
    const rep = hc.evictLruToAdmit(786_369, null, Never.call, true);
    try t.expectEqual(@as(usize, 0), rep.entries);
    try t.expectEqual(@as(usize, 1), hc.entryCount());

    // The rule is SSD-first's. Off it, even the lien restores.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    hc2.restore_pin_min_bytes = 1 << 20;
    var src2 = try KVCache.init(testing.allocator, 8);
    defer src2.deinit();
    try testFillCache(&src2, s, 8, 4096);
    _ = try hc2.commit(&src2, &session, false);
    var cold: [4096]u32 = undefined;
    for (&cold, 0..) |*x, i| x.* = if (i < 11) @as(u32, @intCast(i + 1)) else @intCast(i + 6_000_001);
    var slot2 = try KVCache.init(testing.allocator, 8);
    defer slot2.deinit();
    var off2: usize = 0;
    const hit2 = try hc2.lookupAndRestore(&slot2, &off2, null, s, &cold, false, 0, null, null);
    try t.expectEqual(@as(usize, 11), hit2.matched);
}

test "a 0-token outcome is not a restore: no LRU bump, no protection, and the entry stays evictable" {
    // A 0-token restore outcome used to leave `last_restored_used` set, shielding a fully
    // reclaimable entry from the admission pass trying to admit the request its miss condemned.
    const t = testing;
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    // The qwen4_exp arm: the indexer history travels with the KV or the
    // restore is a miss (`MtpHeadQsaHistoryGap`).
    hc.qsa_history_required = true;

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i + 7);

    // A DECOY, committed first so it is the LRU victim the pass should take.
    var decoy_ids: [600]u32 = undefined;
    for (&decoy_ids, 0..) |*x, i| x.* = @intCast(i + 900_007);
    var decoy_cache = try KVCache.init(testing.allocator, 3);
    defer decoy_cache.deinit();
    try testFillCache(&decoy_cache, s, 3, 600);
    _ = try hc.commit(&decoy_cache, &decoy_ids, false);

    // The hybrid entry: KV plus one SSM checkpoint, and NO QSA history on it.
    var src = try KVCache.init(testing.allocator, 3);
    defer src.deinit();
    try testFillCache(&src, s, 3, 600);
    var src512 = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&src512);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
    _ = try hc.commitWithSsm(&src, &tokens, false, cps, null, null);
    try t.expectEqual(@as(usize, 2), hc.entryCount());
    const hybrid_idx: usize = 1;
    const used_before = hc.entries.items[hybrid_idx].last_used;

    var slot_cache = try KVCache.init(testing.allocator, 3);
    defer slot_cache.deinit();
    var ssm = pcEmptySsm();
    defer pcFreeHybrid(&ssm);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestoreWithMedia(&slot_cache, &moe_off, &ssm, s, &tokens, false, 0, null, null, null, 0xF5);

    // The outcome: nothing delivered.
    try t.expectEqual(@as(usize, 0), hit.matched);
    try t.expect(!hit.full_match);
    try t.expect(!hit.checked_out);
    try t.expectEqual(@as(usize, 0), moe_off);
    try t.expectEqual(@as(usize, 0), slot_cache.step);

    // ...therefore nothing to protect, nothing to promote, and nothing
    // checked out to another slot.
    try t.expect(hc.last_restored_used == null);
    try t.expectEqual(used_before, hc.entries.items[hybrid_idx].last_used);
    try t.expectEqual(@as(?usize, null), hc.entries.items[hybrid_idx].checked_out_by);

    // The admission pass takes the decoy first, then the entry that delivered nothing.
    const Fits = struct {
        fn call(ctx: ?*anyopaque) bool {
            const cache: *HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() == 0;
        }
    };
    const rep = hc.evictLruToAdmit(600_000, &hc, Fits.call, true);
    try t.expect(rep.admitted);
    try t.expectEqual(@as(usize, 2), rep.entries);
}

test "the lien weighs the share a restore DELIVERS, not the one it matched" {
    // `findBestRestorableMatch` returns the raw token match but the restore clamps to the highest
    // checkpoint, so `restoreWouldPinEntry` weighing `shared` let a 100k match that delivers 1,024
    // rows pin 11.5 GB.
    const t = testing;

    // The decision, pure. Pure attention delivers what it matched.
    try t.expectEqual(@as(usize, 100_000), HotPrefixCache.deliverableShare(null, false, 100_000));
    // A hybrid delivers its checkpoint, and nothing without one. (One
    // null-handle layer: `highestCheckpointAtOrBelow` skips a zero-layer stub.)
    var pure_layers = [_]transformer_mod.SSMCacheEntrySnapshot{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = false,
    }};
    var cps_pure = [_]SSMCheckpoint{.{ .pos = 1024, .layers = &pure_layers }};
    try t.expectEqual(@as(usize, 1024), HotPrefixCache.deliverableShare(&cps_pure, true, 100_000));
    try t.expectEqual(@as(usize, 0), HotPrefixCache.deliverableShare(&cps_pure, true, 512));
    try t.expectEqual(@as(usize, 0), HotPrefixCache.deliverableShare(null, true, 100_000));

    // ...and the lien test at the live numbers, both ways round.
    const MB: u64 = 1 << 20;
    const floor = HotPrefixCache.RESTORE_PIN_MIN_BYTES;
    // The raw share alone acquits it...
    try t.expect(!HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, 100_000));
    // ...the deliverable share convicts it.
    try t.expect(HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, HotPrefixCache.deliverableShare(&cps_pure, true, 100_000)));

    // One hybrid entry, a prompt sharing 400 of 600 tokens, whose only checkpoint is at 8.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    hc.restore_pin_min_bytes = 1 << 20; // a 2 MB entry stands in for 11.5 GB

    var tokens: [4096]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i + 7);

    var src = try KVCache.init(testing.allocator, 8);
    defer src.deinit();
    try testFillCache(&src, s, 8, 4096);
    for (src.entries) |*e| {
        if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
        if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
    }
    var src8 = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&src8);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src8, 8, s);
    _ = try hc.commitWithSsm(&src, &tokens, false, cps, null, null);
    try t.expectEqual(@as(usize, 1), hc.entryCount());
    try t.expect(hc.entries.items[0].kv_bytes > hc.restore_pin_min_bytes);
    const used_before = hc.entries.items[0].last_used;

    // 2048 raw shared rows of a 4096-row entry — a ratio of 2, far inside
    // RESTORE_PIN_RATIO — but the only checkpoint sits at 8.
    var diverged: [4096]u32 = undefined;
    for (&diverged, 0..) |*x, i| x.* = if (i < 2048) tokens[i] else @intCast(i + 800_000);

    var slot_cache = try KVCache.init(testing.allocator, 8);
    defer slot_cache.deinit();
    var ssm = pcEmptySsm();
    defer pcFreeHybrid(&ssm);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&slot_cache, &moe_off, &ssm, s, &diverged, false, 0, null, null);

    // Declined at the gate, before the bump — eight rows are not worth a lien
    // on the session.
    try t.expectEqual(@as(usize, 0), hit.matched);
    try t.expectEqual(@as(usize, 0), moe_off);
    try t.expectEqual(@as(usize, 0), slot_cache.step);
    try t.expect(hc.last_restored_used == null);
    try t.expectEqual(used_before, hc.entries.items[0].last_used);

    // ...and the admission pass gets real bytes back for it.
    const Fits = struct {
        fn call(ctx: ?*anyopaque) bool {
            const cache: *HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() == 0;
        }
    };
    const rep = hc.evictLruToAdmit(600_000, &hc, Fits.call, true);
    try t.expect(rep.admitted);
    try t.expectEqual(@as(usize, 1), rep.entries);
    try t.expect(rep.bytes > 0);
}
test "spec adopt: a qwen4 head target declines a payload with no QSA half; KV-only targets are unaffected" {
    // The qwen4_exp head's KV is meaningless without its index-key history, so the two halves
    // adopt together or not at all. Everything else is KV-only and keeps the old rule.
    const Tag = std.meta.Tag(SpecAdopt);
    const Plan = struct {
        fn tag(p: SpecAdopt) Tag {
            return std.meta.activeTag(p);
        }
        fn len(p: SpecAdopt) usize {
            return switch (p) {
                .kv_only, .head => |w| w,
                else => std.math.maxInt(usize),
            };
        }
    };
    try testing.expectEqual(Tag.kv_only, Plan.tag(specAdoptPlan(10, 40, 31, false, false)));
    try testing.expectEqual(@as(usize, 21), Plan.len(specAdoptPlan(10, 40, 31, false, false)));
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(40, 40, 31, false, false))); // starts past the reuse
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(0, 20, 31, false, false))); // ends short of it
    try testing.expectEqual(@as(usize, 31), Plan.len(specAdoptPlan(0, 31, 31, false, false))); // exact

    try testing.expectEqual(Tag.head, Plan.tag(specAdoptPlan(10, 40, 31, true, true)));
    try testing.expectEqual(@as(usize, 21), Plan.len(specAdoptPlan(10, 40, 31, true, true)));
    try testing.expectEqual(Tag.decline_head_no_history, Plan.tag(specAdoptPlan(10, 40, 31, true, false)));
    // A payload the trunk cannot use is skipped before the aux question.
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(40, 40, 31, true, false)));
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(0, 20, 31, true, false)));
}

test "qwen4 MTP head persist: the head's row count IS its cache step, so a committed history adopts" {
    // `KVCache.update` advances `step` only at layer 0, so `qwen4MtpAdvance` sets the head's;
    // the commit snapshots it and `qwen4MtpAdopt` demands the key history be exactly that long.
    const s = mlx.gpuStream();
    const head_layer: u32 = 3; // stands in for `num_hidden_layers`
    var kv = try KVCache.init(testing.allocator, head_layer + 1);
    defer kv.deinit();
    var seq_offset: usize = 0;
    try testFillHeadCache(&kv, s, head_layer, 100, &seq_offset);
    try testing.expectEqual(@as(usize, 100), seq_offset);
    try testing.expectEqual(seq_offset, kv.step);

    var snap = DflashSnap{ .snapshot = try kv.snapshot(), .base_pos = 0 };
    defer snap.deinit();
    try testing.expectEqual(seq_offset, snap.snapshot.step);

    const Tag = std.meta.Tag(SpecAdopt);
    const plan = specAdoptPlan(snap.base_pos, snap.snapshot.step, seq_offset, true, true);
    try testing.expectEqual(Tag.head, std.meta.activeTag(plan));
    try testing.expectEqual(seq_offset, plan.head);
    try testing.expectEqual(@as(usize, 100), snap.snapshot.step);

    // The shape the bug had: the step `KVCache.update` left at a non-zero layer is not adoptable.
    try testing.expectEqual(Tag.skip, std.meta.activeTag(specAdoptPlan(0, 0, seq_offset, true, true)));
}

test "spec snap bytes: the qwen4 head's QSA half is billed into the entry" {
    const s = mlx.gpuStream();
    // Head-shaped: one layer at the head's own index, never layer 0.
    const head_layer: u32 = 3;
    var kv = try KVCache.init(testing.allocator, head_layer + 1);
    defer kv.deinit();
    var head_rows: usize = 0;
    try testFillHeadCache(&kv, s, head_layer, 16, &head_rows);
    try testing.expectEqual(head_rows, kv.step);
    var snap = DflashSnap{ .snapshot = try kv.snapshot(), .base_pos = 0 };
    defer snap.deinit();
    const kv_only = HotPrefixCache.specSnapBytes(&snap);
    try testing.expect(kv_only > 0);

    var entry: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer transformer_mod.ssmFreeQsaState(&entry);
    defer _ = mlx.mlx_array_free(entry.conv_state);
    defer _ = mlx.mlx_array_free(entry.ssm_state);
    const shape = [_]c_int{ 1, 16, 128 };
    entry.aux_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.aux_state, &shape, 3, .bfloat16, s));
    entry.qsa_ratio = 4;
    snap.head_aux = transformer_mod.ssmSnapshot(&entry);
    const with_head = HotPrefixCache.specSnapBytes(&snap);
    try testing.expectEqual(kv_only + 16 * 128 * 2, with_head);
}

test "SSD-first: the disk flush carries the full prefix while RAM keeps a trim" {
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens: [1200]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 1200);
    var probe = try cache.snapshot();
    const row_bytes = HotPrefixCache.snapshotRowBytes(&probe);
    probe.deinit();
    try testing.expect(row_bytes > 0);
    const budget: u64 = row_bytes * 768;

    // Arm A (ssd_first on): RAM trims, the disk entry covers the full prompt.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-on", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens, false);
        try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
        try testing.expect(hc.entries.items[0].tokens.len < tokens.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, tokens.len), hc.disk.?.entries.items[0].kv_len);
        try testing.expectEqual(@as(usize, tokens.len), hc.disk.?.entries.items[0].tokens.len);
        try testing.expect(hc.pending_disk == null);
    }

    // Arm B (ssd_first off): the disk copy is exactly what RAM retained.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-off", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
        try testing.expect(hc.pending_disk == null);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expect(hc.disk.?.entries.items[0].kv_len < tokens.len);
    }
}

test "SSD-first companion: a restore adopts the entry's buffer when its capacity suffices" {
    // A grow is not in place, so a restore must land in the donor's buffer (which already
    // carries the previous turn's reservation) rather than allocate the entry's whole size.
    const s = mlx.gpuStream();
    const Grows = &transformer_mod.KVCache.kv_cap_buf_grows;

    // Turn 1: a reserved cache grows once, to the reservation.
    var donor = try KVCache.init(testing.allocator, 1);
    defer donor.deinit();
    donor.reserve(4096);
    const g0 = Grows.*;
    try testFillCache(&donor, s, 1, 600);
    try testing.expectEqual(@as(usize, 1), Grows.* - g0);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try hc.commit(&donor, &tokens, false);

    // Turn 2: restore into a fresh slot cache that reserves the same length: nothing allocates.
    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&slot, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expect(res.full_match);
    slot.reserve(4096);
    const g1 = Grows.*;
    try testFillCache(&slot, s, 1, 8); // the diverged tail
    try testing.expectEqual(@as(usize, 0), Grows.* - g1);

    // A reservation is not retroactive: reserving more than the donor holds still allocates nothing.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    _ = try hc.lookupAndRestore(&slot2, &moe_off2, null, s, &tokens, false, 0, null, null);
    slot2.reserve(65536);
    const g2 = Grows.*;
    try testFillCache(&slot2, s, 1, 8);
    try testing.expectEqual(@as(usize, 0), Grows.* - g2);

    // Negative arm: writing past the donor's capacity does grow, exactly once.
    const g3 = Grows.*;
    try testFillCache(&slot2, s, 1, 4096);
    try testing.expect(Grows.* - g3 >= 1);
}

test "SSD-first: an idle entry spills to disk and leaves RAM; the active session stays" {
    // RAM floors at one entry under SSD-first: everything but the active session goes to the
    // SSD, and only once its copy is complete.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // Arm A: SSD-first spills the idle session.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-spill", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());

        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
        // ...and A is still served, from disk.
        try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
        hc.disk.?.drainWriter();
        var back = try KVCache.init(testing.allocator, 1);
        defer back.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestore(&back, &moe_off, null, s, &tokens_a, false, 0, null, null);
        // A restore always leaves the last token to forward.
        try testing.expectEqual(@as(usize, 599), res.matched);
    }

    // Arm B: every other arch keeps both entries resident.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-nospill", 0, 128);
        defer hc.deinit();
        _ = try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

test "SSD-first: an in-flight write does not stall the tick — the entry is re-checked next pass" {
    // The durability check must not drain the writer on the inference thread: an entry whose
    // files are still staged is not evictable on this pass, and the next pass evicts.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-inflight", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    // Generous allowance: this test is about the write state, not the cap.
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);
    _ = try hc.commit(&cache, &tok_b, false);

    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    // It returned, the index knows the entry, and RAM still holds it.
    try testing.expect(hc.disk.?.writer.?.pendingBytes() > 0);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Let the writer run and drop the allowance: the next pass evicts.
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter(); // test-side only: the engine never waits here
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tok_b, hc.entries.items[0].tokens);
}

test "SSD-first: the idle ALLOWANCE bounds eviction, not the fact of being idle" {
    // The spill used to evict every non-newest entry on every finish, ignoring
    // `--prefix-cache-mem`; writing stays unconditional, evicting is what the allowance bounds.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var tok_c: [600]u32 = undefined;
    for (&tok_c, 0..) |*t, i| t.* = @intCast(i + 300_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // Two sessions, an allowance that covers the idle one: both stay.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow2", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for one idle entry
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        // ...and the write still happened.
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // A third session past the allowance: the oldest idle entry goes, and only that one.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow3", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false); // oldest
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for ONE of the two idle
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        for (hc.entries.items) |*e| try testing.expect(!std.mem.eql(u32, e.tokens, &tok_a));
        var saw_b = false;
        var saw_c = false;
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_b)) saw_b = true;
            if (std.mem.eql(u32, e.tokens, &tok_c)) saw_c = true;
        }
        try testing.expect(saw_b and saw_c);
    }

    // Allowance 0 means what it says: nothing idle stays resident.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow0", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false);
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }
}

test "SSD-first: the allowance is a HARD cap, shed in two tiers (durable first)" {
    // Shed the entries that have a durable copy first, then the rest: an unpersistable entry
    // survives while the cache is under the cap and is dropped only past it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var tok_c: [600]u32 = undefined;
    for (&tok_c, 0..) |*t, i| t.* = @intCast(i + 300_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // A is the oldest and unpersistable (TurboQuant); B is newer and persists; C is active.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false);
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) e.snapshot.config = .{ .scheme = .turboquant_4, .bits = 4, .group_size = 64 };
        }
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        hc.spillIdleEntries(s);

        // Tier 1 shed B, the durable one, even though A is older.
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        var saw_a = false;
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) saw_a = true;
            try testing.expect(!std.mem.eql(u32, e.tokens, &tok_b));
        }
        try testing.expect(saw_a);

        // Allowance zero: A has nowhere to go and the cap is hard, so tier 2 drops it.
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }

    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2b", 0, 128);
        defer hc.deinit();
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(10 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        _ = try hc.commit(&cache, &tok_a, false); // oldest
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        for (hc.entries.items) |*e| try testing.expect(!std.mem.eql(u32, e.tokens, &tok_a));
    }
}

test "SSD-first: a silent SKIP is not a durable copy — the idle entry stays resident" {
    // Every silent skip used to read as "the SSD holds this session" and evicted the RAM copy.
    // Four skip reasons, each asserting both halves: the tier holds nothing, RAM still holds the entry.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    // Arm 1: the volume declined the store.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try testFillCache(&cache, s, 1, 600);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-declined", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(10 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        _ = try hc.commit(&cache, &tokens_a, false);
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 2: under `MIN_PERSIST_TOKENS`.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try testFillCache(&cache, s, 1, 400);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-short", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        _ = try hc.commit(&cache, tokens_a[0..400], false);
        _ = try hc.commit(&cache, tokens_b[0..400], false);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 3: TurboQuant, whose rotation state does not survive a restore.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try testFillCache(&cache, s, 1, 600);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tq", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        _ = try hc.commit(&cache, &tokens_a, false);
        _ = try hc.commit(&cache, &tokens_b, false);
        // The scheme is flipped on the committed snapshot rather than on the live cache.
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tokens_a)) e.snapshot.config = .{ .scheme = .turboquant_4, .bits = 4, .group_size = 64 };
        }
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 4: a layer offset short of the persist target.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-offset", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        _ = try hc.commit(&cache, &tokens_a, false);
        _ = try hc.commit(&cache, &tokens_b, false);
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tokens_a)) e.snapshot.entries[1].offset = 300;
        }
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

test "SSD-first: a PARTIAL copy is not a copy — the idle entry stays resident" {
    // A byte-capped flush lands real bytes and stops on a chunk boundary: the entry on disk is short and RAM keeps it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-partial", 0, 128);
    defer hc.deinit();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
    // One byte: the loop writes chunk 0 and stops.
    hc.disk.?.max_flush_bytes = 1;

    _ = try hc.commit(&cache, &tokens_a, false);
    _ = try hc.commit(&cache, &tokens_b, false);
    hc.spillIdleEntries(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.entries.items[0].kv_len < 600);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Lift the cap and the allowance: the next pass completes the copy and the durable entry goes.
    hc.disk.?.max_flush_bytes = 512 * 1024 * 1024;
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
    try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
}

test "DiskTier.holdsFullPrefix: the INDEX must agree before a RAM copy is discarded" {
    // `.persisted` is the write path's claim; this is the manifest's.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-holds", 0, 128);
    defer tier.deinit();

    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, true, cache.config));
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, .{ .scheme = .affine, .bits = 4, .group_size = 64 }));

    // A truncated tail chunk, the shape `scan` records after a kill -9.
    const cb = tier.entries.items[0].chunk_bytes;
    const keep = cb[cb.len - 1];
    cb[cb.len - 1] = 0;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    cb[cb.len - 1] = keep;
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    tier.entries.items[0].kv_len = 400;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
}

test "SSD-first: one resident session makes reclaimableBytes truthfully ZERO" {
    // Under SSD-first RAM holds exactly the active session at rest, so the largest entry is the
    // only entry and the provable discount is 0 (its buffers are the live KV).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-reclaim", 0, 128);
    defer hc.deinit();

    _ = try hc.commit(&cache, &tokens_a, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.residentBytes() > 0);
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // Mid-switch, two sessions are briefly resident and the non-active one is reclaimable.
    _ = try hc.commit(&cache, &tokens_b, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(hc.reclaimableBytes() > 0);

    // ...and the idle spill returns it to 0 without evicting the session being served.
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());
}

test "reclaimableBytesFor: only an entry the PROMPT could restore from is unevictable" {
    // The guard's credit with the prompt in hand: the prompt-blind rule always subtracts the
    // whole cache under one-session-resident.
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    _ = try hc.commit(&cache, &tokens_a, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const resident = hc.residentBytes();
    try testing.expect(resident > 0);

    // (1) The prompt extends the resident session: nothing to reclaim.
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytesFor(&tokens_a));
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // (2) A different session's prompt: the bytes are reclaimable.
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&tokens_b));
    try testing.expect(hc.reclaimableBytesFor(&tokens_b) > hc.reclaimableBytes());

    // A prefix too short to restore from does not pin the entry either.
    var barely: [600]u32 = undefined;
    for (&barely, 0..) |*t, i| t.* = @intCast(i + 7);
    for (barely[MIN_CANCELLED_COMMIT_TOKENS - 8 ..]) |*t| t.* = 424_242;
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&barely));

    // (3) Two entries, prompt matches one: only that one is withheld.
    _ = try hc.commit(&cache, &tokens_b, false);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    const both = hc.residentBytes();
    const credit_a = hc.reclaimableBytesFor(&tokens_a);
    try testing.expect(credit_a > 0 and credit_a < both);
    try testing.expect(credit_a >= hc.reclaimableBytes());
}

test "EntryDigest: the published snapshot answers the reclaimable question without the cache" {
    // The guard reads a published snapshot of these digests instead of the cache.
    const s = mlx.gpuStream();
    const A = testing.allocator;

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var short: [64]u32 = undefined;
    for (&short, 0..) |*t, i| t.* = @intCast(i + 7);

    var cache = try KVCache.init(A, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(A, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    _ = try hc.commit(&cache, &tokens_a, false);
    const resident = hc.residentBytes();

    // Publish, then replace: the superseded slice is the caller's to free.
    var d1 = try hc.digestsAlloc(A);
    try testing.expectEqual(@as(usize, 1), d1.len);
    try testing.expectEqual(resident, d1[0].kv_bytes);
    _ = try hc.commit(&cache, &tokens_b, false);
    const d2 = try hc.digestsAlloc(A);
    A.free(d1);
    d1 = d2;
    defer A.free(d1);
    try testing.expectEqual(@as(usize, 2), d1.len);

    const fp_a = HotPrefixCache.prefixFingerprint(&tokens_a).?;
    const both = hc.residentBytes();
    const credit_a = HotPrefixCache.reclaimableFromDigests(d1, both, fp_a);
    try testing.expect(credit_a > 0 and credit_a < both);

    // A prompt matching neither session credits the whole residency.
    var tokens_c: [600]u32 = undefined;
    for (&tokens_c, 0..) |*t, i| t.* = @intCast(i + 500_000);
    const fp_c = HotPrefixCache.prefixFingerprint(&tokens_c);
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, fp_c));

    // A prompt under the restore floor pins nothing, and hashes to null.
    try testing.expectEqual(@as(?u64, null), HotPrefixCache.prefixFingerprint(&short));
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, null));

    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_a), credit_a);
    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_c), both);
}

test "prefix cache: the trim bill prices only the checkpoints a shed would keep" {
    // `shedCheckpointsToFit` thins the interior the moment an entry lands over the cap, so
    // billing every lower checkpoint prices memory the entry never holds.
    const positions = [_]usize{ 100, 200, 300, 400, 500 };
    const bytes = [_]u64{ 10, 10, 10, 10, 10 };
    try testing.expectEqual(@as(?u64, 50), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 50, .min_span_recency));
    try testing.expectEqual(@as(?u64, 20), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 25, .min_span_recency));
    try testing.expectEqual(@as(?u64, 10), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 15, .min_span_recency));
    try testing.expectEqual(@as(?u64, null), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 5, .min_span_recency));
}

test "prefix cache: a 383k oversized hybrid entry trims instead of flat-declining" {
    // The live #330 follow-up shape: qwen4_exp, stride 4096, a 383,069-token entry at 13,056
    // bytes per KV row, ~26 MB per checkpoint, a 3,873.54 MB budget; it flat-declined.
    const row_bytes: u64 = 13_056;
    const per_cp: u64 = 26 * 1024 * 1024;
    const budget: u64 = 3873 * 1024 * 1024;
    const tokens: usize = 383_069;

    // 93 stride captures plus the end-of-prompt snap.
    var all_pos: [94]usize = undefined;
    for (all_pos[0..93], 0..) |*p, i| p.* = (i + 1) * 4096;
    all_pos[93] = 383_039;
    var bytes: [94]u64 = undefined;
    for (&bytes) |*b| b.* = per_cp;

    // (a) Drop-oldest retention: the lowest survivor already prices past the budget.
    {
        const end_anchored = all_pos[78..94];
        try testing.expect(@as(u64, end_anchored[0]) * row_bytes > budget);
        try testing.expectEqual(
            @as(?usize, null),
            HotPrefixCache.trimLenForBudgetPure(budget, tokens, row_bytes, end_anchored, bytes[0..16], .min_span_recency),
        );
    }

    // (b) Span-preserving retention: the survivors spread over the whole prompt.
    var pos: [94]usize = all_pos;
    var n: usize = pos.len;
    while (n > 16) {
        const drop = transformer_mod.positionDropIndexUsize(pos[0..n], .min_span_recency);
        var k = drop;
        while (k + 1 < n) : (k += 1) pos[k] = pos[k + 1];
        n -= 1;
    }
    try testing.expectEqual(@as(usize, 4096), pos[0]);
    try testing.expectEqual(@as(usize, 383_039), pos[n - 1]);
    const tl = HotPrefixCache.trimLenForBudgetPure(budget, tokens, row_bytes, pos[0..n], bytes[0..n], .min_span_recency) orelse
        return error.NoTrimPoint;
    try testing.expect(tl >= 126_976);
    try testing.expect(std.mem.indexOfScalar(usize, pos[0..n], tl) != null);
    var kept: usize = 0;
    while (kept < n and pos[kept] <= tl) kept += 1;
    const survivors = HotPrefixCache.shedSurvivorBytes(
        pos[0..kept],
        bytes[0..kept],
        budget - @as(u64, tl) * row_bytes,
        .min_span_recency,
    ) orelse return error.ShedDoesNotFit;
    try testing.expect(@as(u64, tl) * row_bytes + survivors <= budget);

    // (c) Pricing every lower checkpoint at the same point buys a strictly shorter prefix.
    var all_lower: ?usize = null;
    var k = n;
    while (k > 0) {
        k -= 1;
        const p = pos[k];
        if (p < MIN_CANCELLED_COMMIT_TOKENS) break;
        if (@as(u64, p) * row_bytes + @as(u64, k + 1) * per_cp <= budget) {
            all_lower = p;
            break;
        }
    }
    try testing.expect(all_lower != null);
    try testing.expect(tl > all_lower.?);
}

test "prefix cache: a failed trimmed copy retries at the next-lower checkpoint" {
    // A `trimmedCopy` failure at one width is not a verdict on the entry.
    const positions = [_]usize{ 4096, 8192, 12288 };
    const bytes = [_]u64{ 1024, 1024, 1024 };
    const budget: u64 = 60_000;
    const tl = HotPrefixCache.trimLenForBudgetPure(budget, 100_000, 4, &positions, &bytes, .min_span_recency) orelse
        return error.NoTrimPoint;
    try testing.expectEqual(@as(usize, 12288), tl);
    try testing.expectEqual(
        @as(?usize, 8192),
        HotPrefixCache.trimLenForBudgetPure(budget, tl - 1, 4, &positions, &bytes, .min_span_recency),
    );
    try testing.expectEqual(
        @as(?usize, null),
        HotPrefixCache.trimLenForBudgetPure(budget, 255, 4, &positions, &bytes, .min_span_recency),
    );
}

test "prefix cache: an oversized commit names WHICH outcome declined it" {
    const a = TrimDecline.no_restorable_prefix.reason();
    const b = TrimDecline.snapshot_copy_failed.reason();
    const c = TrimDecline.checkpoint_list_copy_failed.reason();
    try testing.expect(a.len > 0 and b.len > 0 and c.len > 0);
    try testing.expect(!std.mem.eql(u8, a, b));
    try testing.expect(!std.mem.eql(u8, a, c));
    try testing.expect(!std.mem.eql(u8, b, c));
}

test "prefix cache: the ungated retention + trim arms reproduce the previous policy exactly" {
    // Characterization of the ungated arms: min-span over the whole interior with no recency,
    // and a trim bill of every lower checkpoint.
    const t = std.testing;

    // The cache's default is the ungated policy.
    var hc = HotPrefixCache.init(t.allocator, 4);
    defer hc.deinit();
    try t.expectEqual(transformer_mod.ThinPolicy.min_span, hc.cp_thin);

    try t.expectEqualStrings("all_lower", HotPrefixCache.trimBillArm(4, false));
    try t.expectEqualStrings("all_lower", HotPrefixCache.trimBillArm(32, false));
    try t.expectEqualStrings("shed", HotPrefixCache.trimBillArm(32, true));

    // The two arms really disagree: four checkpoints of 10 bytes, row_bytes 0.
    const positions = [_]usize{ 256, 512, 768, 1024 };
    const bytes = [_]u64{ 10, 10, 10, 10 };
    // shed arm: at position 1024 the shed can thin down to 20 bytes.
    try t.expectEqual(
        @as(?usize, 1024),
        HotPrefixCache.trimLenForBudgetPure(25, 4096, 0, &positions, &bytes, .min_span),
    );
    // ungated arm bills every lower checkpoint: 1024 costs all four (40), over the 25-byte budget.
    var all_lower_at_1024: u64 = 0;
    for (bytes) |b| all_lower_at_1024 += b;
    try t.expectEqual(@as(u64, 40), all_lower_at_1024);
    try t.expect(all_lower_at_1024 > 25);
    try t.expectEqual(
        @as(?u64, 20),
        HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 25, .min_span),
    );
}

test "prefix cache: the trim-inputs line carries the price, the positions and the chosen bill" {
    // Format pinned on the live 383k fixture.
    const row_bytes: u64 = 13_056;
    const per_cp: u64 = 26 * 1024 * 1024;
    const budget: u64 = 3873 * 1024 * 1024;

    var all_pos: [94]usize = undefined;
    for (all_pos[0..93], 0..) |*p, i| p.* = (i + 1) * 4096;
    all_pos[93] = 383_039;
    var bytes: [94]u64 = undefined;
    for (&bytes) |*b| b.* = per_cp;

    var buf: [768]u8 = undefined;
    // Long list: elided at TRIM_LOG_MAX_POS, but the count stays exact.
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 383_069, row_bytes, budget, &all_pos, &bytes, all_pos.len, 126_976, true);
        try testing.expect(std.mem.startsWith(u8, line, "  [hot-cache] trim inputs: tokens=383069 row_bytes=13056 budget=3873.00 MB list_len=94 arm=shed survivors=["));
        try testing.expect(std.mem.endsWith(u8, line, "\n"));
        try testing.expect(std.mem.indexOf(u8, line, "[4096,8192,12288,") != null);
        try testing.expect(std.mem.indexOf(u8, line, ",...] (32 of 94)") != null);
        try testing.expect(std.mem.indexOf(u8, line, " chosen=126976") != null);
        try testing.expect(std.mem.indexOf(u8, line, " chosen_cp_bytes=27262976") != null);
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, line, "\n"));
        try testing.expect(line.len < buf.len);
    }
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 900, row_bytes, budget, all_pos[0..3], bytes[0..3], 3, 8192, true);
        try testing.expect(std.mem.indexOf(u8, line, "survivors=[4096,8192,12288] (3 of 3)") != null);
        try testing.expect(std.mem.indexOf(u8, line, "...") == null);
    }
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 900, row_bytes, budget, all_pos[0..2], bytes[0..2], 2, null, true);
        try testing.expect(std.mem.indexOf(u8, line, " chosen=none") != null);
        try testing.expect(std.mem.indexOf(u8, line, "chosen_cp_bytes") == null);
    }
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 900, row_bytes, budget, all_pos[0..0], bytes[0..0], 0, 512, true);
        try testing.expect(std.mem.indexOf(u8, line, "survivors=[] (0 of 0)") != null);
        try testing.expect(std.mem.indexOf(u8, line, " chosen=512 chosen_cp_bytes=0") != null);
    }
    try testing.expectEqualStrings("shed", HotPrefixCache.trimBillArm(32, true));
    try testing.expectEqualStrings("shed", HotPrefixCache.trimBillArm(HotPrefixCache.SHED_SIM_MAX, true));
    try testing.expectEqualStrings("all_lower", HotPrefixCache.trimBillArm(HotPrefixCache.SHED_SIM_MAX + 1, true));
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 383_069, row_bytes, budget, &all_pos, &bytes, 200, 126_976, true);
        try testing.expect(std.mem.indexOf(u8, line, "list_len=200 arm=all_lower") != null);
        try testing.expect(std.mem.indexOf(u8, line, ",...] (32 of 200)") != null);
    }
}

test "prefix cache: the trim's row price is the entry's own bytes divided by its rows" {
    // The trim price divides each of the six quantized arrays by its own `shape[2]`, a layout
    // assumption; pin the invariant (price x capacity == snapshot bytes) rather than the constant.
    const s = mlx.gpuStream();

    for ([_]kv_quant.KVQuantConfig{
        kv_quant.KVQuantConfig.dense,
        kv_quant.KVQuantConfig.affine(8),
        kv_quant.KVQuantConfig.affine(4),
    }) |cfg| {
        var cache = try KVCache.initWithConfig(testing.allocator, 2, cfg);
        defer cache.deinit();

        // qwen4_exp's own attention shape: 2 kv heads, head_dim 256.
        const mk = struct {
            fn f(str: mlx.mlx_stream, len: c_int) !mlx.mlx_array {
                const shape = [_]c_int{ 1, 2, len, 256 };
                var a = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_ones(&a, &shape, 4, .bfloat16, str));
                return a;
            }
        }.f;
        // Two writes so the second crosses a growth event.
        for ([_]c_int{ 64, 40 }) |n| {
            const k = try mk(s, n);
            defer _ = mlx.mlx_array_free(k);
            var dv = try cache.update(0, k, k, s, 0);
            dv.deinit();
            const k2 = try mk(s, n);
            defer _ = mlx.mlx_array_free(k2);
            var dv2 = try cache.update(1, k2, k2, s, 0);
            dv2.deinit();
        }

        var snap = try cache.snapshot();
        defer snap.deinit();

        const row_bytes = HotPrefixCache.snapshotRowBytes(&snap);
        const total = HotPrefixCache.snapshotBytes(&snap);
        try testing.expect(row_bytes > 0);

        const cap: u64 = @intCast(mlx.getShape(snap.entries[0].keys)[2]);
        try testing.expect(cap >= 104);
        try testing.expectEqual(total, row_bytes * cap);

        // Per token: 2 layers of (K+V) at 2 heads x 256 dims; affine packs to `bits` plus scale and bias per group of 64.
        const per_layer: u64 = switch (cfg.scheme) {
            .off => 2 * (2 * 256 * 2),
            else => blk: {
                const packed_b: u64 = 2 * 256 * @as(u64, cfg.bits) / 8;
                const groups: u64 = 2 * 256 / cfg.group_size;
                break :blk 2 * (packed_b + groups * 2 * 2);
            },
        };
        try testing.expectEqual(2 * per_layer, row_bytes);
    }
}

test "HotPrefixCache: a bounded disk flush leaves disk_dirty set and later flushes complete the entry — never a whole-entry claim in between" {
    // A long entry reaches the tier in pieces; each piece is a valid shorter entry and the next
    // `flushPendingDisk` extends it.
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
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-partial", 0, 128);
    defer hc.deinit();
    hc.disk.?.max_flush_bytes = 1; // one chunk per flush: the bounded shape

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    _ = try hc.commit(&cache, &tokens, false);
    try testing.expect(hc.disk_dirty);

    // 600 tokens at 128/chunk = 5 chunks; each flush lands one.
    var flushes: usize = 0;
    var last_kv: u32 = 0;
    while (hc.disk_dirty and flushes < 10) : (flushes += 1) {
        hc.flushPendingDisk(s);
        const d = &hc.disk.?;
        try testing.expectEqual(@as(usize, 1), d.entryCount());
        const kv = d.entries.items[0].kv_len;
        try testing.expect(kv >= last_kv);
        if (hc.disk_dirty) try testing.expectEqual(@as(u32, 0), kv % 128);
        const m = d.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense).?;
        try testing.expectEqual(kv, m.usable);
        last_kv = kv;
    }
    try testing.expect(!hc.disk_dirty);
    try testing.expectEqual(@as(usize, 5), flushes);
    try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
}

// ── Restore by move (checkout) ──

/// Live Metal bytes, with the allocator pool pinned first.
fn testLiveBytes(s: mlx.mlx_stream) u64 {
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var live: usize = 0;
    _ = mlx.mlx_get_active_memory(&live);
    return @intCast(live);
}

/// Bytes of one layer's key buffer (capacity, not logical rows).
fn testKeyBufferBytes(cache: *KVCache, layer: usize) u64 {
    const sh = mlx.getShape(cache.entries[layer].keys);
    return @as(u64, @intCast(sh[0])) * @as(u64, @intCast(sh[1])) *
        @as(u64, @intCast(sh[2])) * @as(u64, @intCast(sh[3])) * 4; // f32
}

fn testCheckoutCache(hc: *HotPrefixCache, s: mlx.mlx_stream, tokens: []const u32, reserve: usize) !void {
    var donor = try KVCache.init(testing.allocator, 1);
    defer donor.deinit();
    donor.reserve(reserve);
    try testFillCache(&donor, s, 1, @intCast(tokens.len));
    _ = try hc.commit(&donor, tokens, false);
}

test "restore by move: a full-prefix hit checks the entry out and the append donates in place" {
    // `KVCache.restore` binds through `mlx_array_set`, so the entry keeps a second reference and
    // the first `writeAtOffset` cannot donate: `copy_gpu` privatised the whole prefix (5.13 GB /
    // ~110 ms at 393k, 45% of the warm TTFT). The observable is allocation, not address: under
    // suite-wide pressure the buffer pool recycles addresses.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    // The prompt extends the entry: a full-prefix hit whose commit will replace this same entry.
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    // A reservation big enough that one buffer dwarfs any pool noise.
    const reserve: usize = 1 << 20;

    var moved_bytes: [64]f32 = undefined;
    var copied_bytes: [64]f32 = undefined;
    var moved_delta: u64 = 0;
    var copied_delta: u64 = 0;
    var buf_bytes: u64 = 0;

    // Arm A: the move.
    {
        restore_move_override = true;
        defer restore_move_override = null;
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        defer hc.deinit();
        try testCheckoutCache(&hc, s, &tokens, reserve);

        var slot = try KVCache.init(testing.allocator, 1);
        defer slot.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 0xA11CE);
        try testing.expectEqual(@as(usize, 600), res.matched);
        // The transfer is the scheduler's second step, taken at the last point
        // before the first write (see `donateCheckout`).
        hc.donateCheckout(0xA11CE);

        // The entry gave the buffers up: its handles are empty and it names the slot.
        const e = &hc.entries.items[0];
        try testing.expectEqual(@as(?usize, 0xA11CE), e.checked_out_by);
        try testing.expect(e.snapshot.entries[0].keys.ctx == null);
        try testing.expect(e.snapshot.entries[0].values.ctx == null);

        slot.evalState();
        buf_bytes = testKeyBufferBytes(&slot, 0);
        const before = testLiveBytes(s);
        try testWriteCacheLayer(&slot, s, 0, 600, 8);
        slot.evalState();
        moved_delta = testLiveBytes(s) -| before;
        try testReadKeyRows(&slot, 0, 596, &moved_bytes);
    }

    // Arm B: the kill switch, the refcount share, same bytes by a different buffer.
    {
        restore_move_override = false;
        defer restore_move_override = null;
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        defer hc.deinit();
        try testCheckoutCache(&hc, s, &tokens, reserve);

        var slot = try KVCache.init(testing.allocator, 1);
        defer slot.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 0xA11CE);
        try testing.expectEqual(@as(usize, 600), res.matched);
        const e = &hc.entries.items[0];
        try testing.expectEqual(@as(?usize, null), e.checked_out_by);
        try testing.expect(e.snapshot.entries[0].keys.ctx != null);

        slot.evalState();
        try testing.expectEqual(buf_bytes, testKeyBufferBytes(&slot, 0));
        const before = testLiveBytes(s);
        try testWriteCacheLayer(&slot, s, 0, 600, 8);
        slot.evalState();
        copied_delta = testLiveBytes(s) -| before;
        try testReadKeyRows(&slot, 0, 596, &copied_bytes);
    }

    // The share arm had to copy a second capacity-shaped buffer.
    try testing.expect(copied_delta > buf_bytes);
    // The move arm allocated nothing beyond the tail; a quarter of one buffer is a loose ceiling.
    try testing.expect(moved_delta * 4 < buf_bytes);
    try testing.expectEqualSlices(f32, &copied_bytes, &moved_bytes);
}

fn testReadKeyRows(cache: *KVCache, layer: usize, row: usize, out: []f32) !void {
    cache.evalState();
    const p = mlx.mlx_array_data_float32(cache.entries[layer].keys) orelse return error.NotEvaluated;
    for (out, 0..) |*v, i| v.* = p[row * 8 + i];
}

test "restore by move: a partial-prefix hit keeps the refcount-share" {
    // A prompt that diverges from the entry makes no replace promise: its commit lands as a new entry.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [600]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    prompt[500] = 999_999; // diverge

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 7);
    try testing.expectEqual(@as(usize, 500), res.matched);
    try testing.expectEqual(@as(?usize, null), hc.entries.items[0].checked_out_by);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.ctx != null);
}

test "restore by move: a refusal BEFORE the append hands the entry back INTACT" {
    // Turn B restored a 364k entry by move, the admission pass refused before any forward, and
    // `finishSlot` dropped the checked-out entry. The checkout is a promise, not a transfer:
    // a slot that ends before the donate hands the entry back as it was.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);
    const billed_before = hc.current_kv_bytes;
    try testing.expect(billed_before > 0);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    try testing.expect(res.checked_out);
    const lru_before = hc.entries.items[0].last_used;

    // The refusal: `runPrefill` returns before `Generator.initWithOptions`, so
    // `donateCheckout` never ran and the entry never gave its handles up.
    try testing.expect(!hc.entries.items[0].checkout_donated);
    hc.releaseCheckout(42, "prefill refused");

    // The entry is whole: same tokens, live handles, same bill, same LRU position, restorable.
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 600), e.tokens.len);
    try testing.expectEqual(@as(?usize, null), e.checked_out_by);
    try testing.expect(e.snapshot.entries[0].keys.ctx != null);
    try testing.expect(e.snapshot.entries[0].values.ctx != null);
    try testing.expectEqual(billed_before, hc.current_kv_bytes);
    try testing.expectEqual(lru_before, e.last_used);

    // ...and the proof: a second slot restores the same prefix from it.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const again = try hc.lookupAndRestoreForSlot(&slot2, &moe_off2, null, s, &prompt, false, 0, null, null, null, 43);
    try testing.expectEqual(@as(usize, 600), again.matched);
    hc.releaseCheckout(43, "second slot ended");
}

test "restore by move: a slot that ends without committing DROPS its checked-out entry" {
    // The bytes die with the slot; `finishSlot` releases unconditionally. `testing.allocator` is the free-once bar.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);
    const billed_before = hc.current_kv_bytes;
    try testing.expect(billed_before > 0);

    var slot = try KVCache.init(testing.allocator, 1);
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);

    // Admitted and about to write: the handles go over.
    hc.donateCheckout(42);
    try testing.expect(hc.entries.items[0].checkout_donated);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.ctx == null);

    hc.releaseCheckout(42, "cancelled");
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
    // Idempotent.
    hc.releaseCheckout(42, "cancelled");
    hc.releaseCheckout(43, "cancelled");
    slot.deinit();
}

test "restore by move: a commit RECLAIMS the checked-out entry with the grown buffers" {
    // The happy path: the replace arm installs the grown snapshot and clears the mark.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    try testing.expect(hc.entries.items[0].checked_out_by != null);
    try testWriteCacheLayer(&slot, s, 0, 600, 8);

    _ = try hc.commit(&slot, &prompt, false);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(?usize, null), e.checked_out_by);
    try testing.expectEqual(@as(usize, 608), e.tokens.len);
    try testing.expect(e.snapshot.entries[0].keys.ctx != null);
    hc.releaseCheckout(42, "finished");
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);

    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const res2 = try hc.lookupAndRestore(&slot2, &moe_off2, null, s, &prompt, false, 0, null, null);
    try testing.expectEqual(@as(usize, 607), res2.matched);
}

test "restore by move: a checked-out entry is invisible to a second slot, to eviction and to the published residency" {
    // Its snapshot is empty: invisible to a second slot, to eviction and to the guard.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    const billed = hc.entries.items[0].kv_bytes;
    try testing.expect(billed > 0);

    // (c) a second slot misses.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const res2 = try hc.lookupAndRestoreForSlot(&slot2, &moe_off2, null, s, &prompt, false, 0, null, null, null, 43);
    try testing.expectEqual(@as(usize, 0), res2.matched);
    try testing.expect(!res2.full_match);
    try testing.expectEqual(@as(usize, 0), slot2.step);
    try testing.expectEqual(@as(?usize, 42), hc.entries.items[0].checked_out_by);

    // (d) nothing to reclaim, no digest to publish.
    const digests = try hc.digestsAlloc(testing.allocator);
    defer testing.allocator.free(digests);
    try testing.expectEqual(@as(usize, 0), digests.len);
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytesFor(&prompt));
    // The bill still counts it: the bytes really are resident, in the slot.
    try testing.expectEqual(billed, hc.residentBytes());

    // Eviction cannot take it.
    const Never = struct {
        fn fits(_: ?*anyopaque) bool {
            return false;
        }
    };
    const report = hc.evictLruToAdmit(608, null, Never.fits, false);
    try testing.expectEqual(@as(usize, 0), report.entries);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(?usize, 42), hc.entries.items[0].checked_out_by);

    hc.releaseCheckout(42, "test teardown");
}

test "restore by move: the policy is off outside the SSD-first arm and under the kill switch" {
    // Eligible: SSD-first, enabled, no pending flush, whole entry matched, something to append.
    try testing.expect(HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(false, true, false, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, false, false, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 608, false));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, true, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 500, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 600, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 0, 0, 608, true));
}

/// The resident entry whose token record is exactly `toks`.
fn testEntryFor(hc: *HotPrefixCache, toks: []const u32) !*Entry {
    for (hc.entries.items) |*e| {
        if (std.mem.eql(u32, e.tokens, toks)) return e;
    }
    return error.EntryGone;
}

test "SSD-first: a chunk write that fails AFTER the pass invalidates the entry — RAM is never dropped against it" {
    // Pass N stages A and skips it (pending). The writer then loses `c000003` but, FIFO, still
    // lands meta.json, so the index keeps a non-zero `chunk_bytes[3]`. Pass N+1 must not call
    // A durable. The bar is the verdict (`spill_durable`), not the eviction it licenses.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-poison", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);
    _ = try hc.commit(&cache, &tok_b, false);

    // Pass N: the writer is held, so A's files stay staged.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.writer.?.injectFailure("c000003", .write);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    const dead_id = hc.disk.?.entries.items[0].id;

    // Between the passes: chunk 3 dies, meta.json lands.
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter(); // test-side only: the engine never waits here
    try testing.expect(hc.disk.?.writeErrors() > 0);
    try testing.expectEqual(@as(usize, 5), hc.disk.?.entries.items[0].chunk_bytes.len);

    // The failure names its entry, so a restore from it misses.
    try testing.expectEqual(@as(usize, 1), hc.disk.?.harvestWriteFailures());
    try testing.expect(hc.disk.?.entries.items[0].poisoned);
    try testing.expect(hc.disk.?.bestMatch(&tok_a, false, cache.config) == null);
    try testing.expect(!hc.disk.?.holdsFullPrefix(cache.entries, cache.step, &tok_a, false, cache.config));

    // Pass N+1: A is not durable; the dead directory is reclaimed and the rebuild is staged.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.entries.items[0].id != dead_id);

    // Control: with the writer healthy the same entry is durable and restores whole.
    hc.disk.?.writer.?.injectFailure(null, .write);
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter();
    hc.spillIdleEntries(s);
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);

    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tok_b, hc.entries.items[0].tokens);

    var back = try KVCache.init(testing.allocator, 1);
    defer back.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&back, &moe_off, null, s, &tok_a, false, 0, null, null);
    try testing.expectEqual(@as(usize, 599), res.matched);
}

test "SSD-first: a write failure inside the SAME pass still keeps the entry resident" {
    // The same-pass interleaving (`writeErrors() != errs_before`), via a submit-time injection.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-samepass", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);
    _ = try hc.commit(&cache, &tok_b, false);
    hc.disk.?.writer.?.injectFailure("c000002", .submit);
    hc.spillIdleEntries(s);
    hc.disk.?.drainWriter();
    try testing.expect(hc.disk.?.writeErrors() > 0);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(!hc.disk.?.holdsFullPrefix(cache.entries, cache.step, &tok_a, false, cache.config));
}

test "SSD-first: the durability check STATS the chunks — a truncated file is never durable" {
    // A byte can go missing with no write error at all; one stat per chunk catches it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-stat", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);
    _ = try hc.commit(&cache, &tok_b, false);

    // Pass 1 stages A with the writer held; pass 2, after the files land, is the healthy control.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter();
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);

    // One chunk loses its bytes behind the tier's back; every in-memory bar still passes.
    const id = hc.disk.?.entries.items[0].id;
    var sub: [64]u8 = undefined;
    const rel = try std.fmt.bufPrint(&sub, "fp-stat/e{d}/c000002.safetensors", .{id});
    try tmp.dir.writeFile(io, .{ .sub_path = rel, .data = "short" });

    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
}

test "HotPrefixCache: eviction picks the LRU of the key holding the most entries" {
    var cache = HotPrefixCache.init(testing.allocator, 8);
    defer cache.deinit();
    const key_a: u64 = 0xa;
    const key_b: u64 = 0xb;
    // C (conversation, key A, the global LRU) then D1..D3 (sweep docs, key B).
    const keys = [_]u64{ key_a, key_b, key_b, key_b };
    for (keys, 1..) |k, used| {
        try cache.entries.append(testing.allocator, .{
            .tokens = try testing.allocator.dupe(u32, &[_]u32{ 1, @intCast(used) }),
            .has_tools = false,
            .cache_key = k,
            .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
            .last_used = used,
            .quant_config = kv_quant.KVQuantConfig.dense,
            .kv_bytes = 0,
            .ssm_checkpoints = null,
            .ssm_bytes = 0,
        });
    }
    // A fourth sweep doc arriving: the sweep evicts its own oldest, never C.
    try testing.expectEqual(@as(?usize, 1), cache.lruIndexExcluding(null, key_b));
    // No incoming key: the largest existing group (B) still pays.
    try testing.expectEqual(@as(?usize, 1), cache.lruIndexExcluding(null, null));
    // D1 protected / checked out: next LRU within B.
    try testing.expectEqual(@as(?usize, 2), cache.lruIndexExcluding(2, key_b));
    cache.entries.items[1].checked_out_by = 7;
    try testing.expectEqual(@as(?usize, 2), cache.lruIndexExcluding(null, key_b));
    cache.entries.items[1].checked_out_by = null;
    // One workload = plain LRU: C goes first.
    for (cache.entries.items) |*e| e.cache_key = 0;
    try testing.expectEqual(@as(?usize, 0), cache.lruIndexExcluding(null, 0));
}
