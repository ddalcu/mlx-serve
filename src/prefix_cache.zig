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

/// Why an oversized commit retained NOTHING. Three different outcomes used to
/// print one identical `skipped oversized entry` line — the budget arithmetic
/// declining, the trimmed KV copy failing, and the trimmed checkpoint list's
/// `dupe` failing — so a live decline said nothing about which one happened
/// (and the two failures swallowed their error). Pure so the wording is
/// unit-testable.
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
    /// Did this restore CHECK OUT its entry (`checkoutEligible`)? True means
    /// the entry released its own handles and the slot is the sole owner, so
    /// the first append DONATES in place. False — every other restore — is a
    /// refcount SHARE, and mlx privatises the whole prefix on that first
    /// append (`is_donatable()` fails on the entry's second reference). The
    /// admission bill reads this: only a checked-out prefix is a prefix the
    /// request will not allocate (audit B-A3).
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
    /// TRANSIENT, one `spillIdleEntries` pass only: did THIS pass leave a
    /// durable, index-verified copy of the entry on the SSD tier? Reset for
    /// every entry at the top of every pass, so it can never be read stale —
    /// the eviction tiers are decided from it and a stale `true` would
    /// discard a session that is not on disk.
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
    /// RESTORE BY MOVE: the slot that took OWNERSHIP of this entry's KV
    /// buffers (`KVCache.adopt`), identified by its address. While set, the
    /// entry's `snapshot` holds EMPTY handles — the bytes live in that slot's
    /// cache and nowhere else — so the entry is invisible to every other
    /// reader: restore, eviction, the byte-budget shed, the SSD spill, and
    /// both reclaimable/digest publications. Cleared by the commit that
    /// replaces it with the grown buffers; an entry still holding it at slot
    /// end is DROPPED (`releaseCheckout`).
    checked_out_by: ?usize = null,
};

/// What a spec-snap adoption may do, decided BEFORE any mlx call so the whole
/// policy is unit-testable: `.kv_only`/`.head` carry the length to clamp to.
pub const SpecAdopt = union(enum) {
    /// Nothing to adopt: no payload, the snap starts past what the trunk
    /// reused, or it ends before it (a gap right below the generation point
    /// is worse than a blind start).
    skip,
    /// A KV-only spec cache (dflash context, sidecar MTP head).
    kv_only: usize,
    /// The qwen4_exp in-checkpoint head: KV + QSA aux together.
    head: usize,
    /// A head target met a payload with no QSA half — a pre-v5 disk sidecar
    /// or an entry committed before head persistence. Head-only miss.
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

/// Resolve this module's lazily-cached env reads ONCE, from the main thread,
/// before any other thread exists — the same discipline (and the same reason)
/// as `transformer.warmQsaEnvCaches`: `ssd_first_env_cached` is a `?bool`
/// filled on first touch, and a non-atomic optional written from two threads
/// is UB. Only the inference thread touches it today; this is what keeps that
/// true when a second caller appears. Called from `main()`. (audit N7)
pub fn warmEnvCaches() void {
    _ = ssdFirstEnabled();
    _ = restoreMoveEnabled();
}

/// SSD-first prefix cache mode. `MLX_SERVE_PREFIX_SSD_FIRST=0` restores the
/// RAM-first behaviour every other arch already has; the mode is armed only
/// where `ModelConfig.ssdFirstCapable()` is also true (qwen4_exp today), so
/// no other architecture's prefix-cache path changes.
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

/// RESTORE BY MOVE. `MLX_SERVE_RESTORE_MOVE=0` restores the refcount-SHARE
/// every restore did before — the arm whose first append copies the whole
/// prefix. Armed only where `HotPrefixCache.ssd_first` is (qwen4_exp plus the
/// SSD-first env switch), so no other architecture's restore changes.
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

/// THE SSD-first predicate. Arch, env switch, AND a disk tier.
///
/// `ssdFirstEnabled()` alone is only two thirds of the answer, and the missing
/// third is the default: `--prefix-cache-disk` is OFF out of the box, so on
/// qwen4_exp the mode used to arm with no tier underneath it. That handed the
/// budget arm its "one full-context session + idle" floor — ~20 GB plus the
/// whole ask at 1M — while none of the spill machinery could run, because
/// every mechanism needs somewhere to write. A budget sized for a tier that
/// does not exist is just RAM the server cannot use.
///
/// Called at BOTH sites (the load-time budget and the arming), so with the
/// disk off qwen4_exp takes the RAM arm byte for byte, exactly like every
/// other arch. Scan-pinned in `scheduler.zig` and `server.zig`.
///
/// The two sites answer `has_disk` from what they can see, and they are not
/// identical: the budget resolver runs BEFORE the tier is built, so it asks
/// `--prefix-cache-disk > 0` (the operator's ask), while the arming asks
/// `disk != null` (the tier that exists). A boot that asks for a tier and
/// fails to build one therefore gets the SSD budget with the RAM mode for
/// that boot — one log line explains both, and it is a strictly rarer and
/// smaller inconsistency than the default it replaces (the mode armed with no
/// tier at all). (external review item 4)
pub fn ssdFirstActive(config: *const model_mod.ModelConfig, has_disk: bool) bool {
    return has_disk and config.ssdFirstCapable() and ssdFirstEnabled();
}

/// SSD-first mechanism 1: what the LIVE cache held at commit time, captured
/// BEFORE the RAM byte-budget trim. The disk tier is the capacity tier here,
/// so it must receive the full prefix even when the RAM entry keeps a trimmed
/// one (or keeps nothing at all). Every handle is refcount-SHARED with the
/// live KV, so the record costs bookkeeping, not GPU bytes.
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
    /// qwen4_exp MTP head ONLY: the head's QSA aux entry (raw index-key
    /// history + pooled block bank + ratio) and the absolute position of its
    /// key row 0. The head's KV is meaningless without it — `qsaMaskFromQk`
    /// errors `QsaHistoryGap` the moment the key history and the cache
    /// position disagree — so a snap that has one and not the other is
    /// DECLINED, never half-adopted. Null for the dflash context and the
    /// sidecar MTP head, whose state really is KV-only.
    head_aux: ?transformer_mod.SSMCacheEntrySnapshot = null,
    head_pos_base: c_int = 0,

    pub fn deinit(self: *DflashSnap) void {
        self.snapshot.deinit();
        if (self.head_aux) |*a| transformer_mod.ssmSnapshotDeinit(a);
        self.head_aux = null;
    }
};

/// What `commitWithState` reads to build a `DflashSnap`. `head` is set only
/// by the qwen4_exp MTP head (see `DflashSnap.head_aux`).
pub const DflashCommit = struct {
    cache: *const KVCache,
    base_pos: usize,
    head: ?*const SSMCacheEntry = null,
    head_pos_base: c_int = 0,
};

/// Where `lookupAndRestore` puts a restored assistant context. `base_pos` is
/// written on every path so the caller can build `DflashCtx` from it. `head`
/// is the qwen4_exp Transformer owning the in-checkpoint MTP head: present
/// iff `cache` is that head's own KV, and the adoption then goes through
/// `qwen4MtpAdopt` so both halves land or neither does.
pub const DflashTarget = struct {
    cache: *KVCache,
    base_pos: *usize,
    head: ?*transformer_mod.Transformer = null,
};

/// What `evictLruToAdmit` gave up, and whether it was enough.
pub const EvictionReport = struct {
    entries: usize = 0,
    /// Bytes the ALLOCATOR actually got back (live delta).
    bytes: u64 = 0,
    /// Bytes the cache had BILLED for those entries. Larger than `bytes`
    /// whenever a snapshot refcount-shares its buffers with a live cache.
    accounted_bytes: u64 = 0,
    /// The pass stopped because an eviction returned nothing: what is left is
    /// shared with a live request and dropping it would cost hits for free.
    shared_stop: bool = false,
    /// False means the cache is empty (or down to the entry this request is
    /// using) and the request STILL does not fit — refuse it by name.
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
    /// `last_used` of the entry the CURRENT request restored from, or null.
    /// Set by the restore, cleared at the start of every lookup — never a
    /// stale request's entry. `evictLruToAdmit` refuses to evict it.
    last_restored_used: ?u64 = null,
    /// The arch keeps a QSA indexer history beside its SSM state (qwen4_exp).
    /// A restore that leaves the live entries without it cannot prefill —
    /// `qsaMaskFromQk` errors on every turn on that prefix — so it is a MISS.
    qsa_history_required: bool = false,
    /// The checkpoint-retention policy this model's entries are thinned with
    /// (PR #363 item 3). Mirrored ONCE at wiring from
    /// `ModelConfig.longCtxGated()` (`scheduler.zig`, the
    /// `qsa_history_required` pattern), because `HotPrefixCache` never sees a
    /// ModelConfig. The default is a93e2c0's behaviour at the two sites that
    /// read it (`mergeCheckpointLists`, `shedCheckpointsToFit`): min-span over
    /// the whole interior, with NO recency reservation.
    cp_thin: transformer_mod.ThinPolicy = .min_span,
    /// SSD-first mode (`ModelConfig.ssdFirstCapable()` and the env switch).
    /// Set by the scheduler at load; false keeps every legacy path.
    ssd_first: bool = false,
    /// SSD-first: the RAM allowance for IDLE entries, in bytes — the resolved
    /// `--prefix-cache-mem` on this arm (`ssdFirstPrefixCacheMem`'s `idle`
    /// term, published beside the budget). `spillIdleEntries` evicts only
    /// PAST it; 0 really does mean "nothing idle stays resident".
    ///
    /// Without it the spill had no cap to work against and evicted every
    /// non-active entry on every finished request, so two alternating
    /// sessions bounced off the SSD every single turn (external review
    /// item 3).
    ssd_idle_mem: u64 = 0,
    /// SSD-first mechanism 1: the live-cache state of the most recent commit,
    /// flushed instead of the (possibly trimmed) RAM entry.
    pending_disk: ?PendingDiskFlush = null,

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
                // A blind head is never silent: name the two numbers that
                // failed to line up (a snap that starts past the reuse, or
                // ends before it).
                log.info("  [hot-cache] {s} not adopted: want {d} tokens from base {d}, snap holds {d} (matched {d})\n", .{ what, matched -| snap.base_pos, snap.base_pos, snap.snapshot.step, matched });
                return null;
            },
            .decline_head_no_history => {
                log.info("  [qwen4] MTP head restore declined (snapshot carries no QSA history) — head starts blind\n", .{});
                return null;
            },
            .kv_only, .head => |w| w,
        };
        // qwen4_exp in-checkpoint head: KV + QSA aux adopt together or not at
        // all, and the trim is the head's own (it slices the key history and
        // the pooled bank alongside the KV).
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
        // transient snapshot is freed right after. A pre-v5 sidecar carries no
        // head half, so a qwen4 head target declines it (head-only miss —
        // the trunk restore above is untouched).
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

    /// Resident bytes of a speculative-side snap: the KV plus (qwen4_exp) the
    /// head's QSA aux half. The aux is real resident state — a 62.7k-token
    /// index-key history is tens of MB — so a budget that only counted the KV
    /// would under-bill the entry it evicts against.
    fn specSnapBytes(snap: *const DflashSnap) u64 {
        var total = snapshotBytes(&snap.snapshot);
        if (snap.head_aux) |a| {
            // Only the two the head actually owns: its `conv_state` and
            // `ssm_state` are empty handles (the head's one layer is a
            // full-attention layer, so there is no recurrence), and an empty
            // handle's `size` is not a byte count.
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
                // A dense snapshot leaves the four quant handles as EMPTY
                // 0-dim arrays: `ctx != null` is true for those, so reading
                // `shape[2]` walks off the end of the shape. Harmless in
                // effect (size 0 makes the numerator 0) but it is an
                // out-of-bounds read in the price the trim multiplies by a
                // prompt length — the ndim check makes axis 2 a fact.
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

    /// Positions printed by the trim-inputs line before it elides; the COUNT
    /// is always exact.
    const TRIM_LOG_MAX_POS: usize = 32;

    fn appendTrimFmt(buf: []u8, n: *usize, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(buf[n.*..], fmt, args) catch return;
        n.* += s.len;
    }

    /// The trim decision's INPUTS as one line. A live trim that lands far
    /// below the budget cannot be diagnosed from its outcome: the price the
    /// walk used (`row_bytes`), the positions it had to choose among, and the
    /// per-checkpoint bytes it billed for its answer are what separate "the
    /// list stopped there" from "the price was wrong". Pure so the format is
    /// pinned by a test rather than by reading a log.
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

    /// Build the two parallel arrays from a checkpoint list and emit the line.
    /// Fires ONCE per oversized commit (the caller latches it), so it costs
    /// nothing on the path that never trims.
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

    /// Stack bound for the shed simulation. A committed checkpoint list is
    /// capped by `ssm_checkpoint_max` (32 by default), so this never binds in
    /// practice; a longer list falls back to billing every lower checkpoint,
    /// which only ever prices a SHORTER trim point, never a longer one.
    const SHED_SIM_MAX: usize = 128;

    /// Bytes the checkpoints at or below a candidate trim point cost AFTER the
    /// commit path's span-preserving shed to `allowance`; null when even the
    /// last survivor is over. Billing every lower checkpoint instead (the #330
    /// answer) prices a trim point the entry never has to pay for —
    /// `shedCheckpointsToFit` thins the interior the moment the entry lands
    /// over the cap. `positions` is ascending, `bytes` parallel to it.
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

    /// The hybrid arm of `trimLenForBudget` as pure arithmetic over positions
    /// and per-checkpoint bytes, so the live 383k shape is testable without
    /// materializing 8 GB of state.
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
            // Ascending positions: everything below here is under the floor
            // too.
            if (p < MIN_CANCELLED_COMMIT_TOKENS) return null;
            const rows = @as(u64, p) * row_bytes;
            if (rows > budget) continue;
            if (shedSurvivorBytes(positions[0 .. k + 1], cp_bytes[0 .. k + 1], budget - rows, policy) != null) return p;
        }
        return null;
    }

    /// Which arm `trimLenForBudget` bills a list of this length with. Audit
    /// S15b: the `all_lower` arm never simulates the shed, so it is strictly
    /// more pessimistic — a trim that lands far below its budget looks the
    /// same from outside whichever arm ran, and the log has to say which.
    fn trimBillArm(list_len: usize, gated: bool) []const u8 {
        if (!gated) return "all_lower";
        return if (list_len > SHED_SIM_MAX) "all_lower" else "shed";
    }

    /// Defensive arm for a checkpoint list past `SHED_SIM_MAX`: the pre-shed
    /// bill (every lower checkpoint). Unreachable while the retention cap
    /// holds.
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
    /// checkpoint's own `pos` — and its cost includes the checkpoints that
    /// SURVIVE the commit's span-preserving shed, not every lower one; a
    /// KV-only hybrid prefix restores as a cold miss while occupying an LRU
    /// slot, so `budget / row_bytes` is not an answer there. Plain attention
    /// restores at any length, so the budget simply prices tokens. `limit`
    /// caps the answer (tokens.len, or the media boundary — trimming INTO
    /// placeholder rows is not a shape we want to reason about).
    fn trimLenForBudget(
        self: *const HotPrefixCache,
        budget: u64,
        limit: usize,
        row_bytes: u64,
        cps: ?[]const SSMCheckpoint,
    ) ?usize {
        if (cps) |list| {
            if (list.len > 0) {
                // ARCH GATE (PR #363 item 3, the billing half). a93e2c0 priced
                // a trim point at EVERY lower checkpoint; simulating the shed
                // and billing only the survivors accepts a strictly LONGER
                // prefix at the same budget, on every hybrid. `all_lower` is
                // a93e2c0's loop verbatim, so the ungated arm is the pre-PR
                // trim length byte for byte.
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
            // A checked-out entry's buffers belong to another slot; its
            // snapshot is empty. Restoring from it would hand this request an
            // uninitialized cache, so it is not a candidate at all — and it is
            // not counted in `probe` either: it cannot explain a miss.
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
            null,
        );
    }

    /// The checkout-capable entry point: `slot_id` names the slot that will
    /// own the restored buffers. Every other overload passes null.
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
        /// RESTORE BY MOVE: identity of the slot asking. Non-null opts this
        /// request into the checkout — the caller is promising to run
        /// `releaseCheckout` on EVERY path that ends the slot. Null keeps the
        /// refcount-share for every caller that makes no such promise.
        slot_id: ?usize,
    ) !LookupResult {
        // A previous request's restored-entry marker must never protect an
        // entry from THIS request's eviction pass.
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
        e.last_used = self.bumpCounter();
        // Identity of the entry THIS request is about to run on. `restore`
        // refcount-SHARES its buffers with the slot's cache, so evicting it
        // frees nothing and only throws away the hit — and "most recently
        // used" is not the same claim: a concurrent commit bumps the counter
        // past us. Read by `evictLruToAdmit`, cleared at the top of the next
        // lookup.
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
        var res: LookupResult = .{
            .matched = effective_matched,
            .full_match = full_match,
            .dflash_base = restoreDflash(e, dflash_target, effective_matched, s),
            .mtp_base = restoreMtp(e, mtp_target, effective_matched, s),
        };
        // The bill reads this: a SHARED restore is copied by the first append
        // (`is_donatable()` fails on the entry's second reference), so only a
        // checkout makes the restored rows rows nobody allocates (audit B-A3).
        res.checked_out = self.checkoutIfEligible(m.idx, m.shared, prompt_ids.len, slot_id);
        return res;
    }

    /// RESTORE BY MOVE, the decision. PURE so the policy is testable without
    /// mlx: the checkout is taken only on a FULL-PREFIX hit — the entry's
    /// whole token record is a prefix of this prompt, which is what makes the
    /// commit's replace path land on this same entry — and only when the
    /// request has something to append (otherwise nothing would donate and the
    /// entry would be dropped for no gain).
    ///
    /// A PARTIAL hit keeps the refcount-share: the entry still describes a
    /// prefix this request diverged from, it is worth keeping for the next
    /// one, and the commit would land as a NEW entry beside it.
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
        // The pending disk record refcount-SHARES the same buffers, so a
        // checkout taken over it cannot donate anyway (mlx's own use_count
        // test declines) — and reasoning about a flush that reads buffers a
        // slot is appending into is not worth the microseconds. It is
        // consumed by `flushPendingDisk` at the previous slot's end, so this
        // is a guard, not a common path.
        if (pending_disk) return false;
        if (entry_tokens == 0 or shared != entry_tokens) return false;
        return prompt_len > shared;
    }

    /// Take the checkout: hand the entry's KV buffers to the slot outright by
    /// RELEASING the entry's own handles. The slot already holds refcount-
    /// shared handles from `restore`, so this drops the buffers' use_count to
    /// one and the slot's first `writeAtOffset` donates instead of copying.
    ///
    /// Returns whether the checkout was TAKEN — the one answer the admission
    /// bill needs, because only a checked-out prefix is one the request will
    /// not allocate (audit B-A3). The caller never re-derives it from the
    /// conjuncts above; there is one predicate and this is its only reader.
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
        e.snapshot.releaseHandles();
        e.checked_out_by = slot_id;
        log.info("  [hot-cache] checked out {d}-token entry to the slot (restore by move; the append donates in place)\n", .{e.tokens.len});
        return true;
    }

    /// End of a slot's life: any entry that slot still holds is DROPPED.
    ///
    /// The bytes are the slot's KV buffers and die with them, so the record
    /// describes a prefix nothing can restore — leaving it in the cache would
    /// hand the next matching request an empty snapshot. The commit path
    /// clears the mark first when it replaces the entry with the grown
    /// buffers, so reaching here with a mark set means this slot ended
    /// WITHOUT committing: cancelled, errored, or refused.
    ///
    /// Idempotent, and safe when the entry was already removed by an
    /// invalidate.
    pub fn releaseCheckout(self: *HotPrefixCache, slot_id: usize, reason: []const u8) void {
        var i: usize = self.entries.items.len;
        while (i > 0) {
            i -= 1;
            const e = &self.entries.items[i];
            if (e.checked_out_by != slot_id) continue;
            const tokens_len = e.tokens.len;
            // Clear the mark BEFORE `evictAt` so its LRU bookkeeping (and any
            // log it emits) sees an ordinary entry; the snapshot is already
            // empty, so the free returns nothing and bills nothing.
            e.checked_out_by = null;
            self.evictAt(i, "checked-out entry dropped");
            log.info("  [hot-cache] checked-out entry dropped: {s} ({d} tokens; its KV died with the slot)\n", .{ reason, tokens_len });
        }
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

        // SSD-first mechanism 1: record what the LIVE cache holds NOW, before
        // any byte-budget trim below shortens what RAM retains. `flushPendingDisk`
        // prefers this record, so the disk entry's kv_len is the full prompt even
        // when the RAM entry keeps a trimmed prefix (or declines outright).
        if (self.ssd_first and self.disk != null and vision_key == 0) {
            self.capturePendingDisk(source_cache, tokens, has_tools, ssm_cps, dflash, mtp);
        }
        // The record refcount-SHARES the live KV, so it keeps a whole session's
        // buffers alive. On the success path `flushPendingDisk` consumes it; on
        // an ERROR return there is no such consumer, and the slot's KVCache
        // deinit then frees nothing — 24 GB at 1M ctx pinned until the next
        // successful commit, precisely under the memory pressure that caused
        // the error. At FUNCTION scope on purpose: an errdefer inside the block
        // above is discarded when that block exits normally, which is every
        // path that can still fail. (audit S2)
        errdefer if (self.pending_disk) |*p| {
            p.deinit(self.allocator);
            self.pending_disk = null;
        };

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
                new_mtp = .{
                    .snapshot = snap,
                    .base_pos = m2.base_pos,
                    // qwen4_exp: the QSA half. Refcount-shared like the KV.
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
                    // ...unless the entry is CHECKED OUT: it retains nothing
                    // (its snapshot is empty), so "keep the resident prefix"
                    // would keep a record with no bytes behind it and discard
                    // the only copy. Fall through to the trim/replace.
                    if (self.entries.items[idx].checked_out_by == null and
                        self.entries.items[idx].tokens.len >= tl)
                    {
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
                const trimmed = new_snap.trimmedCopy(tl, mlx.gpuStream()) catch |err| {
                    // A copy that failed at THIS width is not a verdict on
                    // the entry: retry at the next-lower checkpoint before
                    // declining. The old arm swallowed the error entirely.
                    decline = .snapshot_copy_failed;
                    decline_err = err;
                    // ARCH GATE (PR #363). a93e2c0 declined the commit on a
                    // failed trimmed copy; retrying at the next-lower
                    // checkpoint means allocating AGAIN, immediately, on a
                    // path whose failure is usually memory pressure — and a
                    // retry loop under pressure is how a clean decline becomes
                    // an uncatchable Metal abort. It was measured on the gated
                    // arch's oversized 383k commits; off it a declined commit
                    // costs one cache entry, which is what a93e2c0 paid.
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
                            // `new_snap` is already the trimmed copy and the
                            // spec payloads are gone; the decline below frees
                            // exactly that, so nothing leaks — but the work
                            // is thrown away, which is worth its own line.
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
                break;
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
            // RESTORE BY MOVE: this is the replacement the checkout promised.
            // `e.snapshot` was empty (the slot owned the buffers) and has just
            // been overwritten with the GROWN ones — the same allocation, now
            // longer — so the entry is whole again and visible to everyone.
            e.checked_out_by = null;
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
        // The trim above prices a prefix against the checkpoints that SURVIVE
        // a shed, so the shed has to actually run here too — the replace path
        // already ends with one. No-op while the cache is under its cap.
        if (self.max_kv_bytes > 0) self.shedCheckpointsToFit();
        if (self.disk != null) self.disk_dirty = true;
        self.logResident();
    }

    /// SSD-first mechanism 1: snapshot the live cache (refcount-shared, so no
    /// GPU bytes) plus the full token record and this turn's checkpoints. Best
    /// effort throughout — a failure costs the disk copy, never the RAM commit.
    /// The caller still owns `ssm_cps`/`dflash`/`mtp`: everything here is a
    /// SHARE.
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

    /// The disk-tier spec payloads for one RAM entry. Both the flush and the
    /// idle spill build them, and they must not drift: passing null where the
    /// entry HAS a payload deletes the sidecar on disk (the supersede rule).
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
                // qwen4_exp: the head's QSA half rides the SAME sidecar (v5).
                .head_aux = if (mm.head_aux) |*a| a else null,
                .head_pos_base = mm.head_pos_base,
            } else null,
        };
    }

    /// SSD-first mechanism 6: at the end of a request every idle entry is
    /// WRITTEN to the SSD tier, and RAM is trimmed back to the active session
    /// plus the idle allowance (`ssd_idle_mem`).
    ///
    /// Write and evict are two decisions, and conflating them was external
    /// review item 3: the spill used to evict every non-newest entry on every
    /// `finishSlot`, so two alternating sessions bounced off the SSD on every
    /// single turn even when RAM had room for both. Writing is cheap and has
    /// no downside — it happens unconditionally. EVICTING is what the
    /// allowance bounds.
    ///
    /// The allowance is a HARD cap in two tiers (review decision (c)):
    ///   1. shed idle entries that have a proven durable copy, oldest first;
    ///   2. only if still over, shed the rest, oldest first, naming the reason
    ///      — that is a cache losing WORK (a cold prefill next time), never
    ///      data, and the alternative is an allowance that quietly does not
    ///      hold whenever the disk refuses.
    /// `ssd_idle_mem == 0` therefore means exactly what it says: nothing idle
    /// stays resident.
    ///
    /// The active entry is the most recently used one: the commit that just
    /// ran bumped it, and a restore bumps the entry it restored from.
    pub fn spillIdleEntries(self: *HotPrefixCache, s: mlx.mlx_stream) void {
        if (!self.ssd_first) return;
        if (self.entries.items.len <= 1) return;
        const d = if (self.disk) |*dd| dd else return;

        // The active session is the single MOST-recently-used entry, and the
        // `== newest_used` test below relies on `last_used` being a strictly
        // increasing counter (`bumpCounter`), so exactly one entry can hold the
        // maximum. If it ever becomes a timestamp, two entries can tie and BOTH
        // would be kept resident — which is the one thing this must not do.
        // (audit N12)
        var newest_used: u64 = 0;
        for (self.entries.items) |*e| newest_used = @max(newest_used, e.last_used);

        // ── Pass 1: WRITE. Unconditional, and independent of the allowance.
        //
        // The writer's error counter as of the start of this pass. A blob it
        // drops after logging an error never reaches the disk, so an entry
        // whose files were staged during a pass that saw ANY error is treated
        // as not-yet-durable and re-checked next pass. Conservative on
        // purpose: the cost of being wrong here is a lost session.
        const errs_before = d.writeErrors();
        var spilled: usize = 0;
        var idle_bytes: u64 = 0;
        for (self.entries.items) |*e| {
            e.spill_durable = false;
            if (e.last_used == newest_used) continue; // the active session
            // Checked out: the bytes are the slot's, the snapshot is empty,
            // and there is nothing here to write. (The spill runs on the
            // inference thread at another slot's `finishSlot`, so it can meet
            // a checkout that is still open.) Skipped BEFORE the allowance
            // count too — billing a slot's bytes to the idle cap would shed
            // some other session to make room for memory the cap does not own.
            if (e.checked_out_by != null) continue;
            // Counted against the allowance BEFORE the vision skip: a vision
            // entry occupies RAM whether or not it can be written, so it is
            // part of what the cap is about even though it is never the one
            // shed (`oldestIdleIndex` skips it — the allowance is not a licence
            // to lose a session to a token-only disk key).
            idle_bytes +|= e.kv_bytes;
            // Vision entries never spill: an image placeholder token is
            // identical across images, so a token-only disk key is ambiguous.
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
            // ONLY `.persisted`. The old test was `complete == true`, which the
            // append path also returns for every SILENT SKIP — a declined
            // volume, a prefix under `MIN_PERSIST_TOKENS`, TurboQuant, a layer
            // offset short of the range. On qwen4_exp with the disk tier on and
            // a disk under ~65 GiB free, that discarded every idle entry from
            // RAM at the end of every request with nothing written in its place.
            // A `.partial` copy is not a copy either; its writer may finish it,
            // and the next pass re-checks.
            if (outcome != .persisted) continue;
            // ...and `.persisted` is the WRITE path's claim. The INDEX has to
            // agree: an entry at this key whose `kv_len` reaches the persist
            // target and whose chunk array has one non-zero size per chunk that
            // length implies.
            const disk_id = d.fullPrefixEntryId(e.snapshot.entries, e.snapshot.step, e.tokens, e.has_tools, e.snapshot.config) orelse {
                log.warn("  [hot-cache] idle spill: the tier does not hold the full prefix — entry stays resident\n", .{});
                continue;
            };
            // ...and a STAGED copy is not a durable one. `appendCommit` reports
            // what it handed the writer, not what reached the disk: the writer
            // logs a failed blob, counts it, and drops it, so discarding RAM on
            // that promise loses the session outright. (audit S3)
            //
            // But the check must NOT be a drain. `drainWriter` waits on the
            // whole queue, and this runs on the INFERENCE thread at the end of
            // every request — a decode stall on every finished turn with a
            // flush outstanding, which is precisely what the background writer
            // exists to remove. An entry whose files are still in flight is
            // simply not evictable on THIS pass; the next pass asks again, and
            // meanwhile the entry is safe in RAM. (external review item 6)
            if (d.entryWritesPending(disk_id)) continue;
            if (d.writeErrors() != errs_before) {
                log.warn("  [hot-cache] idle spill: background write failed — entry stays resident\n", .{});
                continue;
            }
            // Proven durable for THIS pass. The flag is reset at the top of
            // every pass, so it can never be read stale.
            e.spill_durable = true;
            spilled += 1;
        }

        // ── Pass 2: EVICT the durable, oldest first, down to the allowance.
        var evicted: usize = 0;
        while (idle_bytes > self.ssd_idle_mem) {
            const idx = self.oldestIdleIndex(newest_used, true) orelse break;
            idle_bytes -|= self.entries.items[idx].kv_bytes;
            self.evictAt(idx, "SSD-first idle spill");
            evicted += 1;
        }

        // ── Pass 3: still over. An entry with no durable copy is now costing
        // RAM the next admission needs, so it goes too — oldest first, and the
        // log names why, once per entry. This loses WORK (a cold prefill),
        // never data.
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

    /// Least-recently-used IDLE entry (never the active session, never a
    /// vision entry — those are not spillable and the allowance is not a
    /// licence to lose one to a token-only disk key). `durable_only` picks
    /// between the two eviction tiers.
    fn oldestIdleIndex(self: *HotPrefixCache, newest_used: u64, durable_only: bool) ?usize {
        var best: ?usize = null;
        var best_used: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used == newest_used) continue;
            if (e.vision_key != 0) continue;
            // A checked-out entry is a live slot's buffers under another name
            // (restore-move); evicting it is not a cache decision at all.
            if (e.checked_out_by != null) continue;
            if (durable_only and !e.spill_durable) continue;
            if (e.last_used < best_used) {
                best_used = e.last_used;
                best = i;
            }
        }
        return best;
    }

    /// Why pass 1 could not leave a durable copy of `e`. Diagnostic only — the
    /// decision was already made; this names it for the log so a user reading
    /// "dropped unpersistable entry" is not left guessing.
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
        // SSD-first mechanism 1: flush the LIVE state captured at commit, not
        // whatever the RAM entry retained after its byte-budget trim. Consumed
        // once; an incomplete write leaves `disk_dirty` set and the next
        // commit's record resumes the extend.
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
            // `.partial` is the only outcome with more to write. A `.skipped`
            // state does not become writable by retrying, so it must not keep
            // the dirty flag set — exactly today's behaviour, now said out loud.
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
    /// `ssm_checkpoint_max`, so the quadratic scan is trivial. The selection
    /// is `transformer.ssmCheckpointDropIndex` — ONE policy shared with the
    /// prefill capture, the byte-budget shed and the disk tier, so no site can
    /// drift back to drop-oldest.
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
            // Defensive: the shared selection never picks the last (that is
            // where warm turns match), so this arm is a no-op guard against a
            // future policy that would.
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
        const idx = self.lruIndexExcluding(null) orelse return;
        self.evictAt(idx, reason);
    }

    fn evictAt(self: *HotPrefixCache, lru_idx: usize, reason: []const u8) void {
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

    /// Bytes the cache currently holds resident. A hint for the connection
    /// thread's admission guard (it may race a commit by one entry); the
    /// decision that MATTERS is made on the inference thread by
    /// `evictLruToAdmit`, which re-reads live memory after every eviction.
    /// Host bytes this cache's SSD writer is holding for files it has not
    /// written yet. On unified memory those compete with the Metal working
    /// set, so a headroom that reads only `mlx_get_active_memory` is optimistic
    /// by up to the writer's permit (~1 GiB) — and the moment they peak is a
    /// long prefill's chunk boundary, which is exactly when the adaptive width
    /// probe runs. INFERENCE THREAD ONLY: reaches the disk tier directly.
    /// (audit S11 — the subtraction in `prefillHeadroomNow` belongs to the
    /// adaptive owner; this is the number to subtract.)
    pub fn stagedHostBytes(self: *HotPrefixCache) u64 {
        const d = if (self.disk) |*dd| dd else return 0;
        return d.stagedHostBytes();
    }

    pub fn residentBytes(self: *const HotPrefixCache) u64 {
        return self.current_kv_bytes;
    }

    /// Bytes an eviction pass can PROVE it will get back: the residency minus
    /// the largest single entry.
    ///
    /// A prefix restore refcount-shares the matched entry's buffers with the
    /// slot's cache, so evicting it returns nothing — `evictLruToAdmit`
    /// protects it by construction (`protect_restored`). The connection
    /// thread's guard cannot know WHICH entry a prompt will match, but it
    /// knows a restore pins at most ONE, so the largest is the provable
    /// discount. Without it the guard admitted a 383k-token prompt on 1,564
    /// MB of "evictable" bytes that were exactly the entry the prompt then
    /// restored from, and the inference thread refused what the connection
    /// thread had promised (guards run 2026-09-05, issue #353 follow-up).
    pub fn reclaimableBytes(self: *const HotPrefixCache) u64 {
        var largest: u64 = 0;
        // A checked-out entry is neither restorable NOR evictable: its bytes
        // are a live slot's KV. They still count in `current_kv_bytes` (they
        // really are resident), so they come off the base here — crediting
        // them would be the unsafe direction, the one where the connection
        // thread promises a request the inference thread then refuses.
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

    /// One resident entry, reduced to what a CONNECTION thread is allowed to
    /// know about it. `hot_prefix_cache` is inference-thread state — nulled and
    /// freed on every model switch — so the guard may never dereference it; it
    /// reads a published, immutable snapshot of these instead.
    ///
    /// `fingerprint` hashes the entry's first `DIGEST_PREFIX_TOKENS` token ids.
    /// That width is the restore FLOOR (`MIN_CANCELLED_COMMIT_TOKENS`): below
    /// it the lookup reports a cold miss and restores nothing, so an entry that
    /// short can never be pinned by any prompt and gets no digest at all.
    pub const EntryDigest = struct {
        fingerprint: u64,
        len: u32,
        kv_bytes: u64,
    };

    pub const DIGEST_PREFIX_TOKENS: usize = MIN_CANCELLED_COMMIT_TOKENS;

    /// FNV-1a over the first `DIGEST_PREFIX_TOKENS` ids. Null when the record
    /// is shorter than the restore floor — "cannot be restored from", which is
    /// a different answer from "hashes to something".
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
    /// Entries under the restore floor are omitted: nothing can pin them, so
    /// leaving them out only ever CREDITS their bytes, which is correct.
    pub fn digestsAlloc(self: *const HotPrefixCache, allocator: std.mem.Allocator) ![]EntryDigest {
        var out = std.ArrayList(EntryDigest).empty;
        errdefer out.deinit(allocator);
        for (self.entries.items) |*e| {
            // Checked out: no resident bytes to describe, and no prompt can
            // restore from it — publishing it would make the connection
            // thread withhold bytes that are already the slot's.
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

    /// PURE, and the connection thread's half of the rule: residency minus the
    /// LARGEST entry this prompt could restore from.
    ///
    /// The pin condition is the fingerprint match ALONE. It deliberately does
    /// NOT also require `digest.len <= prompt_len`: an entry whose record is
    /// LONGER than the prompt still shares the floor-width prefix, and
    /// `restore` clamps to the shorter of the two — so such an entry really can
    /// be restored from, and excluding it would credit bytes that are about to
    /// be pinned. Over-crediting is the unsafe direction here (the inference
    /// thread then evicts nothing and the request is refused after being
    /// promised), so the length ordering is not part of the test.
    ///
    /// Same conservatism as `reclaimableBytesFor`: the LARGEST match is
    /// withheld, not the longest-matching one, because the real lookup ranks by
    /// restorable position and may pick a different entry than a prefix hash
    /// would. Never smaller than the prompt-blind scalar.
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

    /// The same question asked with the PROMPT in hand: bytes an eviction pass
    /// can prove it will get back, given that only an entry this prompt could
    /// actually restore from is unevictable.
    ///
    /// `reclaimableBytes` above subtracts the largest entry unconditionally
    /// because its caller has no prompt and a restore pins at most one entry.
    /// That rule is maximally pessimistic exactly where SSD-first puts the
    /// steady state: ONE resident entry, so it always subtracts the whole
    /// cache — and a request for a DIFFERENT session is then judged as if a
    /// fully-flushed entry were immovable, though nothing would have shared it.
    ///
    /// Here an entry is excluded only when it shares a restorable prefix with
    /// `prompt_tokens`. Two deliberate conservatisms:
    ///   * the key filters (`has_tools`, quant, vision) are NOT applied — the
    ///     caller may not know them, and skipping them can only ADD candidates,
    ///     never remove one the real lookup would pick;
    ///   * among qualifying entries the LARGEST is subtracted, not the
    ///     longest-matching one. The real lookup ranks by restorable position
    ///     and may pick a different entry than a raw token peek would; taking
    ///     the largest covers whichever it picks.
    /// So this is never larger than the truth, and never smaller than
    /// `reclaimableBytes()` — it degenerates to it when every entry matches.
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
            // Below the floor the lookup reports a cold miss and restores
            // nothing, so the entry is not pinned by this prompt.
            if (shared < MIN_CANCELLED_COMMIT_TOKENS) continue;
            pinned = @max(pinned, e.kv_bytes);
        }
        return self.current_kv_bytes -| checked_out -| pinned;
    }

    /// LIMIT THE CACHE WHILE MEMORY IS LEFT. Evict least-recently-used
    /// entries until `fits()` says the request fits, and report what it cost.
    ///
    /// A cached prefix is an OPTIMIZATION; the request in front of us is the
    /// work. Refusing a 450k-token prefill while holding a 6.5 GB entry from
    /// the previous one — which is what the guard did before #353 — trades a
    /// request the machine can serve for a cache hit nobody asked for. With
    /// `--prefix-cache-disk` on, the evicted entry is still on the SSD tier,
    /// so nothing is lost but the restore time.
    ///
    /// NEVER evicts the entry this request restored from: a restore bumps
    /// that entry to MOST recently used and eviction takes the LEAST, so it
    /// goes last by construction — and the loop stops before the final entry
    /// when it is the one that was just touched (`protect_mru`).
    ///
    /// `fits` is re-asked after every eviction rather than compared against a
    /// precomputed shortfall: freeing an entry moves live MLX memory, and the
    /// one estimator that knows the bill is the one that must answer (#126).
    /// Smallest eviction whose live/billed RATIO is meaningful. Below this the
    /// allocator's page rounding and the graph's own scratch swamp the signal,
    /// so a small entry is never judged — it is simply evicted and the pass
    /// continues.
    pub const SHARED_RATIO_MIN_BYTES: u64 = 1 << 20;

    /// An eviction that returns less than 1/Nth of what the entry was BILLED
    /// gave the allocator nothing: its buffers are refcount-SHARED with a live
    /// cache, so dropping it only cost us the hit.
    ///
    /// A RATIO, not an absolute floor. The floor was the first shape of this
    /// check and it conflated "shared" with "small": a 200-token prefix
    /// returns well under a megabyte when it is exclusively ours, so a pass
    /// that hit one aborted with entries still evictable and refused a request
    /// the machine could serve — the exact failure the eviction exists to
    /// prevent.
    pub const SHARED_RETURN_DIVISOR: u64 = 4;

    pub fn evictLruToAdmit(
        self: *HotPrefixCache,
        seq_tokens: u64,
        ctx: ?*anyopaque,
        fits: *const fn (?*anyopaque) bool,
        protect_restored: bool,
    ) EvictionReport {
        var report = EvictionReport{};
        while (!fits(ctx)) {
            const idx = self.lruIndexExcluding(if (protect_restored) self.last_restored_used else null) orelse break;
            // Accounting bytes are what the entry was BILLED; live bytes are
            // what the allocator actually got back. They differ whenever a
            // snapshot refcount-shares its buffers with a live cache — the
            // restored entry does by construction, and so does anything a
            // still-decoding slot is sitting on. Reporting the accounting
            // number as "freed" is how an eviction pass wipes the cache for
            // nothing and then refuses the request anyway.
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
            // Judge only entries big enough for the ratio to mean something,
            // and judge them by what they RETURNED against what they were
            // billed — never by an absolute number of bytes.
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

    /// Least-recently-used entry index, skipping the one whose `last_used`
    /// equals `protect`. Null when nothing is evictable.
    fn lruIndexExcluding(self: *const HotPrefixCache, protect: ?u64) ?usize {
        var best: ?usize = null;
        var best_used: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |*e, i| {
            // Held by a live slot that owns its buffers: evicting it frees
            // NOTHING (the snapshot is empty) and would silently discard the
            // record the slot's commit is about to replace. Same class as
            // `protect_restored`, one degree stronger — this one is not a
            // heuristic about sharing, it is ownership.
            if (e.checked_out_by != null) continue;
            if (protect) |p| {
                if (e.last_used == p) continue;
            }
            if (e.last_used < best_used) {
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

    /// Drop all entries — forces every future request to cold-prefill. Called
    /// when the cache is suspect (pad-only generation, image-bearing prompt,
    /// tools toggle change).
    pub fn invalidateAll(self: *HotPrefixCache, reason: []const u8) void {
        // Suspect state must die on BOTH tiers — a poisoned prefix that
        // survives on disk would be immortal across restarts.
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

/// Fill a HEAD-shaped cache: ONE layer at `layer` — the qwen4_exp MTP head's
/// single layer sits at `num_hidden_layers`, never 0 — driven through the
/// head's own row-count bookkeeping (`Transformer.qwen4MtpAdvance`) exactly
/// as `qwen4MtpForward` drives it. A head fixture that fills at layer 0 gets
/// `step` for free from `KVCache.update` (which advances it ONLY there), so
/// it cannot see a head step bug at all.
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

test "HotPrefixCache: a handed-off QSA history commits, restores and bills exactly like the prefill-end copy" {
    // The commit handoff gives the newest snap a VIEW of the slot's live
    // history instead of the materialized copy the prefill used to attach.
    // The entry must be indistinguishable from the copy arm: same restored
    // rows and bytes on a warm turn, same `ssm_bytes` on the budget — and it
    // must outlive the slot whose buffer it borrowed.
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
        try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);
        // The slot dies: its handles go, the entry's view keeps the buffer.
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
    try hc.commitWithState(&a_cache, &a_tokens, false, 0, a_cps, null, null);

    // Entry B: the serial arm. It restored from A and prefilled a tail too
    // short for a backoff, so its only checkpoint sits at the prompt end.
    var b_cache = try KVCache.init(testing.allocator, 3);
    defer b_cache.deinit();
    try testFillCache(&b_cache, s, 3, b_tokens.len);
    const b_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    b_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[3], 20, s);
    try hc.commitWithState(&b_cache, &b_tokens, false, 0, b_cps, null, null);

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
    try hc.commitWithState(&c_cache, &c_tokens, true, 0, null, null, null);
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
    // Issue #353 and its audit. The user's rule is "limit cache while free
    // memory still left", so a long prefill evicts rather than being refused —
    // but two things make an eviction pass a lie if they are not handled:
    //
    //   * "most recently used" is not "the entry this request restored". A
    //     concurrent commit bumps the counter past us, and then the pass
    //     evicts the very entry the slot is decoding on.
    //   * a restored entry's buffers are refcount-SHARED with the live cache,
    //     so dropping it returns NOTHING to the allocator. Counting its
    //     billed bytes as freed is how a pass wipes the cache and then
    //     refuses the request anyway.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    // Entries big enough for a live free to be visible (8 layers x 4096
    // tokens x 64 B ~ 2 MB each, over FUTILE_EVICTION_BYTES).
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
        // MATERIALIZE before committing. `update` builds `mlx_zeros` +
        // `slice_update` graph nodes and MLX is lazy, so an unevaluated cache
        // owns no Metal buffer at all: this test measured 1,920 bytes of
        // active memory in total and every eviction "returned" 640 of them.
        // A live-memory assertion over lazy arrays asserts nothing.
        for (cache.entries) |*e| {
            if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
            if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
        }
        try hc.commit(&cache, toks, false);
    }
    try testing.expectEqual(@as(usize, 3), hc.entryCount());
    // The premise of everything below: the entries are real allocations.
    var live_resident: usize = 0;
    _ = mlx.mlx_get_active_memory(&live_resident);
    try testing.expect(live_resident > 4 * 1024 * 1024);

    // THIS request restores B — the middle entry, so "most recently used"
    // and "restored" are only the same by luck. The target cache stays alive
    // for the rest of the test: that is what makes B's buffers shared.
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
    // A pass that can never be satisfied still stops at the restored entry.
    const rep = hc.evictLruToAdmit(458_832, null, Never.call, true);
    try testing.expect(!rep.admitted);
    try testing.expectEqual(@as(usize, 2), rep.entries); // A and C, oldest first
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    // …and the survivor is B, by identity.
    var probe_cache = try KVCache.init(testing.allocator, 8);
    defer probe_cache.deinit();
    var off2: usize = 0;
    const still_b = try hc.lookupAndRestore(&probe_cache, &off2, null, s, &toks_b, false, 0, null, null);
    try testing.expect(still_b.full_match);
    // Evicting entries nothing else holds returns REAL memory — at least the
    // share the pass demands before it calls an entry shared. NOT `bytes <=
    // accounted_bytes`: the billed figure is the LOGICAL KV payload, while
    // the buffers freed carry `nextCapacity`'s rounding and growth headroom,
    // so an exclusive eviction legitimately returns MORE than it was billed.
    try testing.expect(rep.bytes > 0);
    try testing.expect(rep.bytes * HotPrefixCache.SHARED_RETURN_DIVISOR >= rep.accounted_bytes);
    try testing.expect(!rep.shared_stop);

    // Unprotected, B goes too — and because two live caches still hold its
    // buffers, the allocator gets ~nothing back, which the pass NOTICES
    // instead of reporting its billed bytes as freed.
    const rest = hc.evictLruToAdmit(458_832, null, Never.call, false);
    try testing.expect(!rest.admitted);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expect(rest.accounted_bytes > 0);
    try testing.expect(rest.bytes * HotPrefixCache.SHARED_RETURN_DIVISOR < rest.accounted_bytes);
    try testing.expect(rest.shared_stop);
}

test "spec adopt: a qwen4 head target declines a payload with no QSA half; KV-only targets are unaffected" {
    // The qwen4_exp in-checkpoint head's KV is meaningless without the raw
    // index-key history it was built beside — `qsaMaskFromQk` errors
    // `QsaHistoryGap` the moment the two disagree — so the two halves adopt
    // together or not at all. Everything else (dflash context, the sidecar
    // MTP head) is genuinely KV-only and keeps the old rule verbatim.
    // Clamp arithmetic first, shared by both:
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

    // Head target: same arithmetic, plus the aux requirement.
    try testing.expectEqual(Tag.head, Plan.tag(specAdoptPlan(10, 40, 31, true, true)));
    try testing.expectEqual(@as(usize, 21), Plan.len(specAdoptPlan(10, 40, 31, true, true)));
    try testing.expectEqual(Tag.decline_head_no_history, Plan.tag(specAdoptPlan(10, 40, 31, true, false)));
    // A payload the trunk cannot use is skipped BEFORE the aux question —
    // "declined, no history" must name a real head miss, not a length miss.
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(40, 40, 31, true, false)));
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(0, 20, 31, true, false)));
}

test "qwen4 MTP head persist: the head's row count IS its cache step, so a committed history adopts" {
    // The head forwards its one layer at index `num_hidden_layers`, and
    // `KVCache.update` advances `step` ONLY at layer 0 — so nothing the head's
    // forward does moves its cache's step. `qwen4MtpAdvance` is what sets it,
    // and every downstream decision reads it: the commit snapshots `step`, and
    // `qwen4MtpAdopt` demands the QSA key history be EXACTLY that long
    // (`MtpHeadQsaHistoryGap`) after `specAdoptPlan` has already refused any
    // snap shorter than the reuse (a silent `.skip`).
    const s = mlx.gpuStream();
    const head_layer: u32 = 3; // stands in for `num_hidden_layers`
    var kv = try KVCache.init(testing.allocator, head_layer + 1);
    defer kv.deinit();
    var seq_offset: usize = 0;
    try testFillHeadCache(&kv, s, head_layer, 100, &seq_offset);
    try testing.expectEqual(@as(usize, 100), seq_offset);
    try testing.expectEqual(seq_offset, kv.step);

    // What the commit ships: the KV snapshot's step beside a QSA key history
    // as long as the rows the head actually appended.
    var snap = DflashSnap{ .snapshot = try kv.snapshot(), .base_pos = 0 };
    defer snap.deinit();
    try testing.expectEqual(seq_offset, snap.snapshot.step);

    const Tag = std.meta.Tag(SpecAdopt);
    const plan = specAdoptPlan(snap.base_pos, snap.snapshot.step, seq_offset, true, true);
    try testing.expectEqual(Tag.head, std.meta.activeTag(plan));
    try testing.expectEqual(seq_offset, plan.head);
    // The equality `qwen4MtpAdopt` enforces (aux rows == kv_snap.step).
    try testing.expectEqual(@as(usize, 100), snap.snapshot.step);

    // And the shape the bug had: the SAME 100-row history carrying the step
    // `KVCache.update` left behind at a non-zero layer is not adoptable at any
    // length — the head starts blind on a full hot-cache hit, silently.
    try testing.expectEqual(Tag.skip, std.meta.activeTag(specAdoptPlan(0, 0, seq_offset, true, true)));
}

test "spec snap bytes: the qwen4 head's QSA half is billed into the entry" {
    // A 62.7k-token index-key history is tens of MB of resident state. A
    // budget that counted only the head's KV would evict against a number
    // that is not the entry's size.
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
    // Holds roughly 768 of the 1200 positions — the commit must trim.
    const budget: u64 = row_bytes * 768;

    // Arm A (ssd_first ON): RAM trims, the DISK entry covers the full prompt.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-on", 0, 128);
        defer hc.deinit();

        try hc.commit(&cache, &tokens, false);
        try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
        try testing.expect(hc.entries.items[0].tokens.len < tokens.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, tokens.len), hc.disk.?.entries.items[0].kv_len);
        try testing.expectEqual(@as(usize, tokens.len), hc.disk.?.entries.items[0].tokens.len);
        try testing.expect(hc.pending_disk == null);
    }

    // Arm B (ssd_first OFF — every other arch): unchanged, the disk copy is
    // exactly what RAM retained. Red-on-revert bar for arm A.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-off", 0, 128);
        defer hc.deinit();

        try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
        try testing.expect(hc.pending_disk == null);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expect(hc.disk.?.entries.items[0].kv_len < tokens.len);
    }
}

test "SSD-first companion: a restore adopts the entry's buffer when its capacity suffices" {
    // Mechanism 7. The #353 reservation sizes the KV to prompt + max_tokens
    // up front, and a grow is NOT in place — `growQuantBuf` allocates the new
    // capacity and slice_updates the old buffer into it, so both are live at
    // once. A restore must therefore land in the DONOR's buffer, which already
    // carries the previous turn's reservation, rather than provoking a fresh
    // allocation of the entry's whole size at the moment memory is tightest.
    //
    // `KVCache.kv_cap_buf_grows` counts exactly those moments.
    const s = mlx.gpuStream();
    const Grows = &transformer_mod.KVCache.kv_cap_buf_grows;

    // Turn 1: a reserved cache grows ONCE, to the reservation.
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
    try hc.commit(&donor, &tokens, false);

    // Turn 2: restore into a fresh slot cache that reserves the SAME length.
    // The entry's buffer already holds it, so nothing allocates: the restore
    // adopted, it did not copy.
    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&slot, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expect(res.full_match);
    slot.reserve(4096);
    const g1 = Grows.*;
    try testFillCache(&slot, s, 1, 8); // the diverged tail
    try testing.expectEqual(@as(usize, 0), Grows.* - g1);

    // A reservation is NOT retroactive: it raises the capacity of a grow that
    // happens, it does not provoke one. So a restored slot that merely RESERVES
    // more than the donor holds still allocates nothing — the copy is deferred
    // until the data actually needs the room.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    _ = try hc.lookupAndRestore(&slot2, &moe_off2, null, s, &tokens, false, 0, null, null);
    slot2.reserve(65536);
    const g2 = Grows.*;
    try testFillCache(&slot2, s, 1, 8);
    try testing.expectEqual(@as(usize, 0), Grows.* - g2);

    // Negative arm — the bar can SEE a copy, so the zeros above are not
    // vacuous: writing PAST the donor's capacity does grow, exactly once.
    const g3 = Grows.*;
    try testFillCache(&slot2, s, 1, 4096);
    try testing.expect(Grows.* - g3 >= 1);
}

test "SSD-first: an idle entry spills to disk and leaves RAM; the active session stays" {
    // Mechanism 6, RAM half. `ssdFirstPrefixCacheMem` floors RAM at ONE entry
    // because the resident entry for the session being served shares its
    // buffers with the live KV — a second resident session is a second real
    // copy. So at the end of a request everything but the active session goes
    // to the SSD tier, and only once its copy is COMPLETE.
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

        // Session A finishes, then session B finishes: B is now active.
        try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());

        hc.spillIdleEntries(s);
        // RAM holds the active session only...
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
        // ...and A is still served, from disk.
        try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
        hc.disk.?.drainWriter();
        var back = try KVCache.init(testing.allocator, 1);
        defer back.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestore(&back, &moe_off, null, s, &tokens_a, false, 0, null, null);
        // A restore always leaves the last token to forward, so a full-prefix
        // hit on a 600-token record lands at 599.
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
        try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

test "SSD-first: an in-flight write does not stall the tick — the entry is re-checked next pass" {
    // External review item 6. The durability check was `drainWriter()` + a
    // write-error comparison, and `drainWriter` WAITS on the whole queue. It
    // runs on the INFERENCE thread inside `finishSlot`, so every finished
    // request with a flush outstanding parked decode until the background
    // writer caught up — the exact stall the writer was added to remove.
    //
    // The bar is behavioural, with a writer held deliberately: an entry whose
    // files are still staged is NOT evictable, the pass returns anyway, and
    // the next pass evicts once the files have landed.
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
    // Generous allowance: this test is about the WRITE state, not the cap
    // (tier 3 would otherwise drop the entry for a different, named reason).
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    try hc.commit(&cache, &tok_a, false);
    try hc.commit(&cache, &tok_b, false);

    // The writer is held: everything the spill stages stays in flight.
    hc.disk.?.writer.?.setPaused(true);
    // Never let a failed assertion below hang the SUITE: teardown drains,
    // and a drain against a paused writer waits forever. (Scan-pinned: every
    // `setPaused(true)` in a test owes a deferred unpause.)
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    // It RETURNED (a drain would have deadlocked against the paused writer),
    // the index knows the entry, and RAM still holds it.
    try testing.expect(hc.disk.?.writer.?.pendingBytes() > 0);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Let the writer run, and drop the allowance so eviction is permitted at
    // all. The NEXT pass sees the files on disk and evicts.
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter(); // test-side only: the engine never waits here
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tok_b, hc.entries.items[0].tokens);
}

test "the end-of-request path never WAITS on the background writer" {
    // Scan half of item 6. The inference thread runs `flushPendingDisk` then
    // `spillIdleEntries` inside `finishSlot`; neither may block on a write.
    // Needles split so this test's own source cannot satisfy them.
    const src = @embedFile("prefix_cache.zig");
    const waits = [_][]const u8{ "drain" ++ "Writer()", "drain" ++ "Prefix(", ".drain(" };
    for ([_][]const u8{ "pub fn spillIdleEntries(", "pub fn flushPendingDisk(" }) |decl| {
        const at = std.mem.indexOf(u8, src, decl) orelse return error.CallSiteMoved;
        const body = src[at..];
        const end = std.mem.indexOf(u8, body, "\n    }\n") orelse body.len;
        for (waits) |w| {
            if (std.mem.indexOf(u8, body[0..end], w) != null) return error.EndOfRequestPathWaitsOnTheWriter;
        }
    }
    // ...and the non-blocking query it uses instead really is non-blocking:
    // `Writer.pendingPrefix` takes the mutex, reads, and returns — no wait.
    const wsrc = @embedFile("kv_disk_writer.zig");
    const at = std.mem.indexOf(u8, wsrc, "pub fn pendingPrefix(") orelse return error.QueryMissing;
    const body = wsrc[at..];
    const end = std.mem.indexOf(u8, body, "\n    }\n") orelse body.len;
    try testing.expect(std.mem.indexOf(u8, body[0..end], "waitUncancelable") == null);
}

test "SSD-first: the idle ALLOWANCE bounds eviction, not the fact of being idle" {
    // External review item 3. `spillIdleEntries` evicted every non-newest
    // entry on every `finishSlot`, ignoring `--prefix-cache-mem` entirely — so
    // two alternating sessions bounced off the SSD on every single turn even
    // though RAM had been budgeted to hold both. Writing is free and stays
    // unconditional; EVICTING is what the allowance bounds.
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

    // Two sessions, an allowance that covers the idle one: BOTH stay.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow2", 0, 128);
        defer hc.deinit();

        try hc.commit(&cache, &tok_a, false);
        try hc.commit(&cache, &tok_b, false);
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for one idle entry
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        // ...and the WRITE still happened: the idle session is on the SSD too,
        // so the next admission can evict it for free.
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // A third session past the allowance: the OLDEST idle entry goes, and only
    // that one.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow3", 0, 128);
        defer hc.deinit();

        try hc.commit(&cache, &tok_a, false); // oldest
        try hc.commit(&cache, &tok_b, false);
        try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for ONE of the two idle
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        // A is gone; B (the newer idle) and C (active) remain.
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

        try hc.commit(&cache, &tok_a, false);
        try hc.commit(&cache, &tok_b, false);
        try hc.commit(&cache, &tok_c, false);
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }
}

test "SSD-first: the allowance is a HARD cap, shed in two tiers (durable first)" {
    // Review decision (c). Item 2 says an unpersistable entry must not be
    // evicted as if it were on disk; item 3 says the allowance is a real RAM
    // bound the next admission depends on. Both hold at once by ORDER: shed
    // the entries that have a durable copy first, and only if that is not
    // enough shed the rest, naming the reason. An unpersistable entry
    // therefore survives while the cache is under the cap and is dropped only
    // past it — losing WORK (a cold prefill), never data.
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

    // A is the OLDEST and is UNPERSISTABLE (TurboQuant on its snapshot); B is
    // newer and persists fine; C is the active session. The allowance leaves
    // room for exactly one idle entry.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2", 0, 128);
        defer hc.deinit();

        try hc.commit(&cache, &tok_a, false);
        try hc.commit(&cache, &tok_b, false);
        try hc.commit(&cache, &tok_c, false);
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) e.snapshot.config = .{ .scheme = .turboquant_4, .bits = 4, .group_size = 64 };
        }
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        hc.spillIdleEntries(s);

        // Tier 1 shed B — the DURABLE one — even though A is older, because
        // dropping A would have lost a session with no copy anywhere.
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        var saw_a = false;
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) saw_a = true;
            try testing.expect(!std.mem.eql(u32, e.tokens, &tok_b));
        }
        try testing.expect(saw_a);

        // Now take the allowance to zero: A has nowhere to go, and the cap is
        // hard, so tier 2 drops it.
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }

    // ...and past the cap, the unpersistable entries go OLDEST first.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2b", 0, 128);
        defer hc.deinit();
        // Nothing can persist at all: the volume is short.
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(10 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        try hc.commit(&cache, &tok_a, false); // oldest
        try hc.commit(&cache, &tok_b, false);
        try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        for (hc.entries.items) |*e| try testing.expect(!std.mem.eql(u32, e.tokens, &tok_a));
    }
}

test "SSD-first: a silent SKIP is not a durable copy — the idle entry stays resident" {
    // External review item 2, the real bug. `appendCommitWithSpec` returned
    // `true` for "nothing more to write", and every SILENT SKIP returns that
    // too. `spillIdleEntries` read it as "the SSD holds this session" and
    // called `evictAt`. On qwen4_exp with `--prefix-cache-disk` on and a disk
    // under ~65 GiB free, EVERY idle entry was discarded from RAM at the end of
    // EVERY request with nothing whatsoever written in its place.
    //
    // Four skip reasons, each a separate arm, each asserting BOTH halves: the
    // tier holds nothing, and RAM still holds the idle entry.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    // Arm 1 — the volume declined the store (the reviewer's 14 GiB box).
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
        // A generous idle allowance: this test is about the SKIP rule, not
        // about the allowance (item 3's tier-3 drop would evict for a
        // different, named reason).
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(10 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        try hc.commit(&cache, &tokens_a, false);
        try hc.commit(&cache, &tokens_b, false);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 2 — under `MIN_PERSIST_TOKENS`: too short to be worth a file, and
    // therefore too short to justify losing.
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
        // A generous idle allowance: this test is about the SKIP rule, not
        // about the allowance (item 3's tier-3 drop would evict for a
        // different, named reason).
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        try hc.commit(&cache, tokens_a[0..400], false);
        try hc.commit(&cache, tokens_b[0..400], false);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 3 — TurboQuant: the rotation state does not survive a restore, so
    // the tier refuses. Losing RAM on that refusal loses the session.
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
        // A generous idle allowance: this test is about the SKIP rule, not
        // about the allowance (item 3's tier-3 drop would evict for a
        // different, named reason).
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        try hc.commit(&cache, &tokens_a, false);
        try hc.commit(&cache, &tokens_b, false);
        // The scheme is flipped on the COMMITTED snapshot rather than on the
        // live cache: the arrays stay dense (a real TurboQuant fixture is a
        // whole cache shape), and the append path's scheme switch is what
        // this arm is about.
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tokens_a)) e.snapshot.config = .{ .scheme = .turboquant_4, .bits = 4, .group_size = 64 };
        }
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 4 — a layer offset short of the persist target. Mid-spec-decode and
    // batched states look like this, and the tier declines them by design.
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
        // A generous idle allowance: this test is about the SKIP rule, not
        // about the allowance (item 3's tier-3 drop would evict for a
        // different, named reason).
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        try hc.commit(&cache, &tokens_a, false);
        try hc.commit(&cache, &tokens_b, false);
        // The idle entry (A) is the one that is not newest; shorten one of its
        // layers so the snapshot no longer covers the range it claims.
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tokens_a)) e.snapshot.entries[1].offset = 300;
        }
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

test "SSD-first: a PARTIAL copy is not a copy — the idle entry stays resident" {
    // The second half of item 2. A byte-capped flush lands real bytes and
    // stops on a chunk boundary; the entry on disk is genuinely short. The
    // old bool said `false` here and the spill did keep the entry — this pins
    // that the enum did not lose that, and that a partial entry IS indexed
    // (so a `holdsFullPrefix` bar alone would not have been enough either).
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
    // Generous allowance: this test is about PARTIAL, not about the cap.
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
    // One byte: the loop writes chunk 0 and stops.
    hc.disk.?.max_flush_bytes = 1;

    try hc.commit(&cache, &tokens_a, false);
    try hc.commit(&cache, &tokens_b, false);
    hc.spillIdleEntries(s);
    hc.disk.?.drainWriter();
    // A short entry IS on disk — that is what makes this different from a skip.
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.entries.items[0].kv_len < 600);
    // ...and RAM kept both.
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Lift the cap AND the allowance: the next pass completes the copy, and
    // with no idle allowance the now-durable entry is the one that goes.
    hc.disk.?.max_flush_bytes = 512 * 1024 * 1024;
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
    try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
}

test "DiskTier.holdsFullPrefix: the INDEX must agree before a RAM copy is discarded" {
    // `.persisted` is the write path's claim; this is the manifest's. Both,
    // or the entry stays. The negative arms are the ones that matter: a
    // truncated chunk (what `scan` leaves after a kill -9 mid-flush) and a
    // record that does not cover the prefix asked about.
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

    // Nothing stored yet.
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    // A different key is a different entry.
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, true, cache.config));
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, .{ .scheme = .affine, .bits = 4, .group_size = 64 }));

    // A truncated tail chunk — the shape `scan` records after a kill -9.
    const cb = tier.entries.items[0].chunk_bytes;
    const keep = cb[cb.len - 1];
    cb[cb.len - 1] = 0;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    cb[cb.len - 1] = keep;
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    // A shorter persisted extent than the commit would target.
    tier.entries.items[0].kv_len = 400;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
}

test "SSD-first: one resident session makes reclaimableBytes truthfully ZERO" {
    // D2. `reclaimableBytes` is residency minus the largest entry, because a
    // restore refcount-shares the matched entry with the slot's cache and the
    // connection thread cannot know WHICH entry a prompt will match — only
    // that it pins at most one.
    //
    // Under SSD-first that formula needs no SSD-specific accounting, and this
    // pins why: at rest RAM holds exactly the active session, so the largest
    // entry IS the only entry and the provable discount is 0 — correct,
    // because evicting it returns nothing (its buffers are the live KV, which
    // `active_mem` already counts). Crediting it would double-count the very
    // bytes mechanism 5 stopped double-counting.
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

    try hc.commit(&cache, &tokens_a, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.residentBytes() > 0);
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // Mid-switch, two sessions are briefly resident and the non-active one IS
    // genuinely reclaimable — its disk copy is complete (mechanisms 1-4), so
    // the same formula reports a real number rather than a hopeful one.
    try hc.commit(&cache, &tokens_b, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(hc.reclaimableBytes() > 0);

    // ...and the idle spill returns it to 0 without evicting the session being
    // served: RAM is back to one entry, so nothing is claimed that a restore
    // would immediately pin again.
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());
}

test "reclaimableBytesFor: only an entry the PROMPT could restore from is unevictable" {
    // The conn-thread guard's credit, asked with the prompt in hand. Under
    // SSD-first the steady state is ONE resident entry, where the
    // prompt-blind rule always subtracts the whole cache — so a different
    // session's request is judged as if a fully-flushed entry were immovable.
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
    try hc.commit(&cache, &tokens_a, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const resident = hc.residentBytes();
    try testing.expect(resident > 0);

    // (1) The prompt EXTENDS the resident session: the entry is the request's
    // own KV once restored — shared buffers, nothing to reclaim. D2 stands.
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytesFor(&tokens_a));
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // (2) A DIFFERENT session's prompt: nothing would share this entry, so its
    // bytes ARE reclaimable — where the prompt-blind rule says 0.
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&tokens_b));
    try testing.expect(hc.reclaimableBytesFor(&tokens_b) > hc.reclaimableBytes());

    // A prefix too short to restore from does not pin the entry either.
    var barely: [600]u32 = undefined;
    for (&barely, 0..) |*t, i| t.* = @intCast(i + 7);
    for (barely[MIN_CANCELLED_COMMIT_TOKENS - 8 ..]) |*t| t.* = 424_242;
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&barely));

    // (3) Two entries, prompt matches one: only THAT one is withheld.
    try hc.commit(&cache, &tokens_b, false);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    const both = hc.residentBytes();
    const credit_a = hc.reclaimableBytesFor(&tokens_a);
    try testing.expect(credit_a > 0 and credit_a < both);
    // Never larger than the truth, never smaller than the prompt-blind rule.
    try testing.expect(credit_a >= hc.reclaimableBytes());
}

test "EntryDigest: the published snapshot answers the reclaimable question without the cache" {
    // The connection thread may never dereference `hot_prefix_cache` (it is
    // inference-thread state, freed on model switch), so the guard reads a
    // published snapshot of these digests instead. This pins the reduction the
    // guard performs, and the ownership contract around the snapshot.
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
    try hc.commit(&cache, &tokens_a, false);
    const resident = hc.residentBytes();

    // Publish, then REPLACE — the superseded slice is the caller's to free,
    // which is what the inference thread does after the swap.
    var d1 = try hc.digestsAlloc(A);
    try testing.expectEqual(@as(usize, 1), d1.len);
    try testing.expectEqual(resident, d1[0].kv_bytes);
    try hc.commit(&cache, &tokens_b, false);
    const d2 = try hc.digestsAlloc(A);
    A.free(d1);
    d1 = d2;
    defer A.free(d1);
    try testing.expectEqual(@as(usize, 2), d1.len);

    // A prompt that extends session A withholds A's entry and nothing else.
    const fp_a = HotPrefixCache.prefixFingerprint(&tokens_a).?;
    const both = hc.residentBytes();
    const credit_a = HotPrefixCache.reclaimableFromDigests(d1, both, fp_a);
    try testing.expect(credit_a > 0 and credit_a < both);

    // A prompt matching NEITHER session credits the whole residency — the case
    // the prompt-blind scalar gets wrong under one-session-resident.
    var tokens_c: [600]u32 = undefined;
    for (&tokens_c, 0..) |*t, i| t.* = @intCast(i + 500_000);
    const fp_c = HotPrefixCache.prefixFingerprint(&tokens_c);
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, fp_c));

    // A prompt under the restore floor cannot restore from anything, so it
    // pins nothing — and hashes to null rather than to "something".
    try testing.expectEqual(@as(?u64, null), HotPrefixCache.prefixFingerprint(&short));
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, null));

    // The digest agrees with the direct, cache-side answer.
    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_a), credit_a);
    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_c), both);
}

test "prefix cache: the trim bill prices only the checkpoints a shed would keep" {
    // `shedCheckpointsToFit` thins the interior the moment an entry lands over
    // the cap, so billing EVERY lower checkpoint at a candidate trim point
    // (the #330 answer) prices memory the entry never has to hold.
    const positions = [_]usize{ 100, 200, 300, 400, 500 };
    const bytes = [_]u64{ 10, 10, 10, 10, 10 };
    // Everything fits: no shed, full bill.
    try testing.expectEqual(@as(?u64, 50), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 50, .min_span_recency));
    // Sheds down to the two ends.
    try testing.expectEqual(@as(?u64, 20), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 25, .min_span_recency));
    // ... and to the single trim-point checkpoint.
    try testing.expectEqual(@as(?u64, 10), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 15, .min_span_recency));
    // One checkpoint is the floor: under that there is nothing to retain.
    try testing.expectEqual(@as(?u64, null), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 5, .min_span_recency));
}

test "prefix cache: a 383k oversized hybrid entry trims instead of flat-declining" {
    // The live #330 follow-up shape: qwen4_exp, 4096-token SSM checkpoint
    // stride, a 383,069-token entry at 13,056 bytes per KV row, ~26 MB per
    // checkpoint, a 3,873.54 MB hot-cache budget. It committed as
    //   [hot-cache] skipped oversized entry (383069 tokens, 8757.79 MB > 3873.54 MB budget)
    // although #330 promises a TRIM.
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

    // (a) Drop-oldest retention: the survivors are the highest 16, so the
    // LOWEST of them already prices past the budget on KV rows alone and
    // nothing at all fits — the flat decline.
    {
        const end_anchored = all_pos[78..94];
        try testing.expect(@as(u64, end_anchored[0]) * row_bytes > budget);
        try testing.expectEqual(
            @as(?usize, null),
            HotPrefixCache.trimLenForBudgetPure(budget, tokens, row_bytes, end_anchored, bytes[0..16], .min_span_recency),
        );
    }

    // (b) Span-preserving retention: the same 16 survivors spread over the
    // whole prompt, so a long prefix is affordable.
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
    // 126,976 = 31 x 4096 is the arithmetic in the write-up; the thinned list
    // affords considerably more.
    try testing.expect(tl >= 126_976);
    // The answer is a RESTORABLE position, and it fits once the shed runs.
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

    // (c) The bill itself is load-bearing, not just the retention: pricing
    // EVERY lower checkpoint (the #330 answer) at the same candidate point
    // buys a strictly SHORTER prefix, because the commit sheds those
    // checkpoints the moment the entry lands over the cap.
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
    // A `trimmedCopy` failure at one width is not a verdict on the entry —
    // the old arm swallowed the error and declined the whole commit.
    const positions = [_]usize{ 4096, 8192, 12288 };
    const bytes = [_]u64{ 1024, 1024, 1024 };
    const budget: u64 = 60_000;
    const tl = HotPrefixCache.trimLenForBudgetPure(budget, 100_000, 4, &positions, &bytes, .min_span_recency) orelse
        return error.NoTrimPoint;
    try testing.expectEqual(@as(usize, 12288), tl);
    // The retry's limit is `tl - 1`, which lands on the next-lower checkpoint.
    try testing.expectEqual(
        @as(?usize, 8192),
        HotPrefixCache.trimLenForBudgetPure(budget, tl - 1, 4, &positions, &bytes, .min_span_recency),
    );
    // Below the commit floor there is nothing left to retry.
    try testing.expectEqual(
        @as(?usize, null),
        HotPrefixCache.trimLenForBudgetPure(budget, 255, 4, &positions, &bytes, .min_span_recency),
    );
    const source = @embedFile("prefix_cache.zig");
    const at = std.mem.indexOf(u8, source, "new_snap.trimmedCopy(tl, mlx.gpuStream()) catch") orelse
        return error.MissingTrimCopy;
    const arm = source[at..@min(source.len, at + 1700)];
    try testing.expect(std.mem.indexOf(u8, arm, "limit = tl - 1") != null);
    try testing.expect(std.mem.indexOf(u8, arm, "@errorName(err)") != null);
    // ARCH GATE (PR #363). The retry allocates AGAIN, immediately, on a path
    // whose failure is usually memory pressure; a retry loop under pressure is
    // how a clean decline becomes an uncatchable Metal abort. Every other arch
    // keeps a93e2c0's behaviour — decline the commit, at a cost of one cache
    // entry — and the gate is the SAME field the retention policy reads, so a
    // tier cannot have one without the other.
    try testing.expect(std.mem.indexOf(u8, arm, "if (self.cp_thin == .min_span) break :trim_blk;") != null);
    // ...and it is checked BEFORE the retry's own log line, so an ungated
    // decline does not announce a retry it will not do.
    const gate_at = std.mem.indexOf(u8, arm, "if (self.cp_thin == .min_span) break").?;
    const log_at = std.mem.indexOf(u8, arm, "retrying at the next-lower checkpoint").?;
    try testing.expect(gate_at < log_at);
    // The error is still LATCHED on both arms: an ungated decline names its
    // cause rather than swallowing it, which was the original defect.
    const decl_at = std.mem.indexOf(u8, arm, "decline_err = err;").?;
    try testing.expect(decl_at < gate_at);
}

test "prefix cache: an oversized commit names WHICH outcome declined it" {
    // Three outcomes printed the SAME `skipped oversized entry` line, and the
    // two failures swallowed their error.
    const a = TrimDecline.no_restorable_prefix.reason();
    const b = TrimDecline.snapshot_copy_failed.reason();
    const c = TrimDecline.checkpoint_list_copy_failed.reason();
    try testing.expect(a.len > 0 and b.len > 0 and c.len > 0);
    try testing.expect(!std.mem.eql(u8, a, b));
    try testing.expect(!std.mem.eql(u8, a, c));
    try testing.expect(!std.mem.eql(u8, b, c));

    // Scan the CODE, not this test's own literals. Every needle below appears
    // verbatim a few lines further down in this very file, so an unsplit
    // `indexOf` over the whole embed finds ITSELF once the impl is deleted —
    // the scan then passes on exactly the regression it exists for. Both
    // mitigations the project already uses are applied: the source is trimmed
    // at the first test, and the needles are split. (audit B0c)
    const whole = @embedFile("prefix_cache.zig");
    const source = whole[0 .. std.mem.indexOf(u8, whole, "\ntest \"") orelse whole.len];
    const skip_needle = "skipped oversized" ++ " entry ({d} tokens";
    const at = std.mem.indexOf(u8, source, skip_needle) orelse return error.MissingSkipLine;
    const line = source[at..@min(source.len, at + 400)];
    try testing.expect(std.mem.indexOf(u8, line, "decline." ++ "reason()") != null);
    try testing.expect(std.mem.indexOf(u8, line, "err_" ++ "name") != null);
    // Both failure arms set a distinct decline AND keep the error.
    try testing.expect(std.mem.indexOf(u8, source, "decline = ." ++ "snapshot_copy_failed;") != null);
    try testing.expect(std.mem.indexOf(u8, source, "decline = ." ++ "checkpoint_list_copy_failed;") != null);
}

test "prefix cache: every ssm checkpoint retention site asks the ONE thin policy" {
    // Class guard. Five sites bound a checkpoint list — the prefill capture
    // (generate.zig, twice), the hot cache's merge + byte-budget shed, and the
    // disk tier's persisted positions. Each takes a `transformer.ThinPolicy`
    // rather than hard-coding one: on qwen4_exp the span-preserving thin with
    // the dense newest quarter (drop-oldest end-anchors the survivors and
    // #330's trim then has nothing affordable to land on), and on every other
    // arch the a93e2c0 policy of THAT site (PR #363 item 3).
    //
    // A site that stops passing a policy is the failure this pins: it would
    // silently pick one arch's answer for both.
    const gen = @embedFile("generate.zig");
    try testing.expect(std.mem.indexOf(u8, gen, "ssm_checkpoints.orderedRemove(0)") == null);
    var seen: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, gen, i, "transformer_mod.ssmCheckpointDropIndex(ssm_checkpoints.items, cp_thin)")) |at| {
        seen += 1;
        i = at + 1;
    }
    try testing.expectEqual(@as(usize, 2), seen);
    // ...and `cp_thin` there is the ONE predicate, not a hand-rolled conjunct.
    try testing.expect(std.mem.indexOf(u8, gen, "if (xfm.config.longCtx" ++ "Gated()) .min_span_recency else .oldest") != null);

    // Scan the two hot-cache sites INSIDE their own functions — a bare
    // file-wide search would match this test's own text.
    const self_src = @embedFile("prefix_cache.zig");
    for ([_][]const u8{ "fn mergeCheckpointLists(", "fn shedCheckpointsToFit(" }) |sig| {
        const at = std.mem.indexOf(u8, self_src, sig) orelse return error.MissingRetentionSite;
        const body = self_src[at..@min(self_src.len, at + 2400)];
        try testing.expect(std.mem.indexOf(u8, body, "transformer_mod.ssmCheckpointDropIndex(") != null);
    }

    const disk = @embedFile("kv_disk_cache.zig");
    try testing.expect(std.mem.indexOf(u8, disk, "transformer_mod.positionDropIndex(set.items, self.cp_thin)") != null);
    try testing.expect(std.mem.indexOf(u8, disk, "std.mem.copyForwards(u32, set.items") == null);

    // The policy is MIRRORED once, from the ONE predicate, at the ONE wiring
    // site — `HotPrefixCache` and `DiskTier` never see a ModelConfig (the
    // `qsa_history_required` pattern).
    const sch = @embedFile("scheduler.zig");
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, sch, "params.config.longCtx" ++ "Gated()) .min_span_recency else"));
}

test "prefix cache: the ungated retention + trim arms reproduce a93e2c0 exactly" {
    // Characterization. The ungated arms are transcribed from
    // `git show a93e2c0:src/prefix_cache.zig`:
    //   mergeCheckpointLists (line 1340) / shedCheckpointsToFit (line 1465):
    //     `if (len < 3) 0 else` min-span over the WHOLE interior, no recency.
    //   trimLenForBudget (line 457): bills `sum(ssmCheckpointBytes(list[0..k+1]))`
    //     — every LOWER checkpoint, never a shed simulation.
    const t = std.testing;

    // The cache's default IS the a93e2c0 hot-cache policy, so a fixture that
    // never wires an arch gets the old behaviour rather than the new one.
    var hc = HotPrefixCache.init(t.allocator, 4);
    defer hc.deinit();
    try t.expectEqual(transformer_mod.ThinPolicy.min_span, hc.cp_thin);

    // The trim's billing arm follows it: ungated = `all_lower`, which is
    // a93e2c0's loop, at every list length.
    try t.expectEqualStrings("all_lower", HotPrefixCache.trimBillArm(4, false));
    try t.expectEqualStrings("all_lower", HotPrefixCache.trimBillArm(32, false));
    try t.expectEqualStrings("shed", HotPrefixCache.trimBillArm(32, true));

    // And the two arms really disagree: a budget that fits only the survivors
    // of a shed is refused by the a93e2c0 bill. Four checkpoints of 10 bytes
    // at positions 256..1024, row_bytes 0 so only the checkpoints are priced.
    const positions = [_]usize{ 256, 512, 768, 1024 };
    const bytes = [_]u64{ 10, 10, 10, 10 };
    // shed arm: at position 1024 the shed can thin down to 20 bytes.
    try t.expectEqual(
        @as(?usize, 1024),
        HotPrefixCache.trimLenForBudgetPure(25, 4096, 0, &positions, &bytes, .min_span),
    );
    // a93e2c0 arm bills EVERY lower checkpoint: 1024 costs all four (40),
    // which is over the 25-byte budget, so it could not have answered 1024.
    // (`trimLenBillingAllLower` takes real checkpoints; the arithmetic it
    // would do is this sum, and the shed's answer above is strictly longer.)
    var all_lower_at_1024: u64 = 0;
    for (bytes) |b| all_lower_at_1024 += b;
    try t.expectEqual(@as(u64, 40), all_lower_at_1024);
    try t.expect(all_lower_at_1024 > 25);
    // The shed's own survivor bill at the same point fits, which is exactly
    // why the two arms answer differently.
    try t.expectEqual(
        @as(?u64, 20),
        HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 25, .min_span),
    );
}

test "prefix cache: the trim-inputs line carries the price, the positions and the chosen bill" {
    // The instrument the 36,864-token live trim asked for. The outcome line
    // alone cannot say whether the walk ran out of POSITIONS or was quoted a
    // wrong PRICE, so the inputs are logged once per oversized commit. Format
    // pinned here on the live 383k fixture rather than by reading a log.
    const row_bytes: u64 = 13_056;
    const per_cp: u64 = 26 * 1024 * 1024;
    const budget: u64 = 3873 * 1024 * 1024;

    var all_pos: [94]usize = undefined;
    for (all_pos[0..93], 0..) |*p, i| p.* = (i + 1) * 4096;
    all_pos[93] = 383_039;
    var bytes: [94]u64 = undefined;
    for (&bytes) |*b| b.* = per_cp;

    var buf: [768]u8 = undefined;
    // Long list: elided at TRIM_LOG_MAX_POS, but the COUNT stays exact.
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 383_069, row_bytes, budget, &all_pos, &bytes, all_pos.len, 126_976, true);
        try testing.expect(std.mem.startsWith(u8, line, "  [hot-cache] trim inputs: tokens=383069 row_bytes=13056 budget=3873.00 MB list_len=94 arm=shed survivors=["));
        try testing.expect(std.mem.endsWith(u8, line, "\n"));
        try testing.expect(std.mem.indexOf(u8, line, "[4096,8192,12288,") != null);
        try testing.expect(std.mem.indexOf(u8, line, ",...] (32 of 94)") != null);
        try testing.expect(std.mem.indexOf(u8, line, " chosen=126976") != null);
        // The bill for the CHOSEN position, not for the list.
        try testing.expect(std.mem.indexOf(u8, line, " chosen_cp_bytes=27262976") != null);
        // One line, and it fits the site's buffer.
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, line, "\n"));
        try testing.expect(line.len < buf.len);
    }
    // A short list prints whole, with no elision marker.
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 900, row_bytes, budget, all_pos[0..3], bytes[0..3], 3, 8192, true);
        try testing.expect(std.mem.indexOf(u8, line, "survivors=[4096,8192,12288] (3 of 3)") != null);
        try testing.expect(std.mem.indexOf(u8, line, "...") == null);
    }
    // A decline says so, and a plain-attention entry has no positions at all.
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
    // Audit S15b: the line names WHICH bill ran. A list past SHED_SIM_MAX
    // takes the pre-shed `all_lower` arm, which is strictly more pessimistic —
    // the whole point of printing it is that the two are indistinguishable
    // from the outcome alone.
    try testing.expectEqualStrings("shed", HotPrefixCache.trimBillArm(32, true));
    try testing.expectEqualStrings("shed", HotPrefixCache.trimBillArm(HotPrefixCache.SHED_SIM_MAX, true));
    try testing.expectEqualStrings("all_lower", HotPrefixCache.trimBillArm(HotPrefixCache.SHED_SIM_MAX + 1, true));
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 383_069, row_bytes, budget, &all_pos, &bytes, 200, 126_976, true);
        try testing.expect(std.mem.indexOf(u8, line, "list_len=200 arm=all_lower") != null);
        try testing.expect(std.mem.indexOf(u8, line, ",...] (32 of 200)") != null);
    }

    // The site emits it ONCE per oversized commit, before any decision.
    const source = @embedFile("prefix_cache.zig");
    const at = std.mem.indexOf(u8, source, "var inputs_logged = false;") orelse return error.MissingLatch;
    const site = source[at..@min(source.len, at + 640)];
    try testing.expect(std.mem.indexOf(u8, site, "inputs_logged = true;") != null);
    try testing.expect(std.mem.indexOf(u8, site, "logTrimInputs(tokens.len, row_bytes, self.max_kv_bytes, eff_cps, tl_opt,") != null);
    // Before the decision: the latch sits above the `orelse break`.
    const log_at = std.mem.indexOf(u8, site, "logTrimInputs(").?;
    const decide_at = std.mem.indexOf(u8, site, "orelse break :trim_blk").?;
    try testing.expect(log_at < decide_at);
}

test "prefix cache: the trim's row price is the entry's own bytes divided by its rows" {
    // The trim walk multiplies `snapshotRowBytes` by a candidate length and
    // compares against the budget, so the price has to be exactly the
    // snapshot's bytes per row. It divides each of the SIX quantized arrays by
    // its own `shape[2]`; that is only a row count if every array carries rows
    // on axis 2, which is a layout assumption, not an identity. The live 383k
    // trim behaved as though the price were ~8x the arch's per-token KV, so
    // pin the invariant rather than the constant: it holds for any layout, and
    // catches a miscount on any one of the six.
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
        // Two writes so the second crosses a growth event and the buffer
        // capacity is genuinely larger than the logical length — the case
        // where "bytes / rows" and "bytes / length" diverge.
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

        // Every entry's arrays are sized from the same capacity, so the price
        // times that capacity IS the snapshot's bytes. A miscounted axis on
        // any array breaks this by exactly that array's ratio.
        const cap: u64 = @intCast(mlx.getShape(snap.entries[0].keys)[2]);
        try testing.expect(cap >= 104);
        try testing.expectEqual(total, row_bytes * cap);

        // And the price is per TOKEN, not per anything else: 2 layers of
        // (K+V) at 2 heads x 256 dims. Dense is bf16; affine packs to `bits`
        // plus one bf16 scale and bias per group of 64.
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
    // The write-through hook writes ONE chunk per boundary and the SSD-first
    // end-of-request flush is bounded too, so a long entry reaches the tier
    // in pieces. Each piece is a valid SHORTER entry (meta.json's kv_len is
    // the chunk-complete length, restore is clamped to it), `disk_dirty`
    // stays set until the tier reports complete, and the next
    // `flushPendingDisk` — the scheduler's post-finish call — extends it.
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
    try hc.commit(&cache, &tokens, false);
    try testing.expect(hc.disk_dirty);

    // 600 tokens at 128/chunk = 5 chunks; each flush lands one.
    var flushes: usize = 0;
    var last_kv: u32 = 0;
    while (hc.disk_dirty and flushes < 10) : (flushes += 1) {
        hc.flushPendingDisk(s);
        const d = &hc.disk.?;
        try testing.expectEqual(@as(usize, 1), d.entryCount());
        const kv = d.entries.items[0].kv_len;
        // Monotone, chunk-aligned while partial, and what a lookup would
        // restore — never the whole entry before its chunks are there.
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

// ── RESTORE BY MOVE (checkout) ───────────────────────────────────────────────

/// Live Metal bytes, with the allocator pool PINNED first. `mlx_clear_cache()`
/// returns pooled-but-free buffers so the reading reflects what is actually
/// held, and the synchronize makes sure every lazy node this test built has
/// really allocated before we look.
fn testLiveBytes(s: mlx.mlx_stream) u64 {
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var live: usize = 0;
    _ = mlx.mlx_get_active_memory(&live);
    return @intCast(live);
}

/// Bytes of one layer's key BUFFER (capacity, not logical rows) — the size of
/// the second allocation a copy-on-write would have to make.
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
    try hc.commit(&donor, tokens, false);
}

test "restore by move: a full-prefix hit checks the entry out and the append donates in place" {
    // The finding (WARM_TTFT_384k.md §4): `KVCache.restore` binds through
    // `mlx_array_set`, so the hot-cache entry keeps a SECOND reference to every
    // KV buffer for the whole request. mlx's `is_donatable()` wants
    // `use_count() == 1`, so the first `writeAtOffset` cannot donate and
    // `copy_gpu` privatises the entire prefix — 5.13 GB / ~110 ms at 393k
    // tokens on qwen4_exp, 45% of the warm TTFT. Releasing the entry's handles
    // at restore (the checkout) makes the slot the sole owner and the copy
    // disappears rather than moving.
    //
    // THE OBSERVABLE IS ALLOCATION, NOT ADDRESS. The first shape of this test
    // compared the appended buffer's data POINTER against the prefix's and
    // called equality "donated". That passed in isolation and failed inside the
    // full suite with different addresses every run: under suite-wide memory
    // pressure MLX's buffer pool recycles, so an address is a statement about
    // the allocator's mood, not about donation. What a copy cannot hide is the
    // BYTES: `slice_update`'s output is capacity-shaped, so a copy must
    // allocate a whole second buffer, and a donation allocates nothing beyond
    // the appended tail. Measure that, with the pool pinned on both readings.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    // The prompt EXTENDS the entry — that is what makes this a full-prefix hit
    // whose commit will replace this same entry.
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    // A reservation big enough that ONE buffer dwarfs any pool noise the
    // reading could carry: 1 Mi rows x 8 dims x 4 B = 32 MiB of K, same of V.
    const reserve: usize = 1 << 20;

    var moved_bytes: [64]f32 = undefined;
    var copied_bytes: [64]f32 = undefined;
    var moved_delta: u64 = 0;
    var copied_delta: u64 = 0;
    var buf_bytes: u64 = 0;

    // ── Arm A: the move.
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

        // The entry gave the buffers up: its handles are EMPTY and it names
        // the slot that holds them. This is the OWNERSHIP half of the bar, and
        // it is what makes the slot the only reference left — no snapshot, no
        // pending disk record (`checkoutEligible` declines on one), no spec
        // payload, and the donor cache died inside `testCheckoutCache`.
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

    // ── Arm B: the kill switch. `MLX_SERVE_RESTORE_MOVE=0` is today's
    // refcount-share, and it must produce the SAME BYTES by a different
    // buffer — the copy the move removes.
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

    // The share arm HAD to copy: the entry still holds the donor, so the
    // append allocated a second capacity-shaped buffer (K and V both).
    try testing.expect(copied_delta > buf_bytes);
    // The move arm allocated nothing beyond the 8-row tail. A quarter of ONE
    // buffer is a deliberately loose ceiling — the real figure is ~0, and the
    // gap between the arms is two whole buffers.
    try testing.expect(moved_delta * 4 < buf_bytes);
    // Byte-for-byte: the arms differ in ownership and in allocation, never in
    // output.
    try testing.expectEqualSlices(f32, &copied_bytes, &moved_bytes);
}

/// Read 64 f32 elements starting at row `row` of a layer's key buffer.
fn testReadKeyRows(cache: *KVCache, layer: usize, row: usize, out: []f32) !void {
    cache.evalState();
    const p = mlx.mlx_array_data_float32(cache.entries[layer].keys) orelse return error.NotEvaluated;
    // The fixture writes [1, 1, T, 8].
    for (out, 0..) |*v, i| v.* = p[row * 8 + i];
}


test "restore by move: a partial-prefix hit keeps the refcount-share" {
    // The checkout is the promise "the commit replaces this entry". A prompt
    // that DIVERGES from the entry does not make that promise: its commit
    // lands as a new entry beside this one, which stays worth keeping.
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

test "restore by move: a slot that ends without committing DROPS its checked-out entry" {
    // The bytes are the slot's KV buffers and die with them. Leaving the record
    // resident would hand the next matching request an EMPTY snapshot — which
    // is why `finishSlot` releases unconditionally rather than trusting that a
    // commit ran. `testing.allocator` is the free-exactly-once bar.
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

    // The slot is cancelled: no commit, so the record has no bytes behind it.
    hc.releaseCheckout(42, "cancelled");
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
    // Idempotent: a second release (or one for a slot that never checked out)
    // is a no-op, never a second free.
    hc.releaseCheckout(42, "cancelled");
    hc.releaseCheckout(43, "cancelled");
    slot.deinit();
}

test "restore by move: a commit RECLAIMS the checked-out entry with the grown buffers" {
    // The happy path. The replace arm finds the same entry (its tokens are a
    // prefix of the committed ones), installs the grown snapshot and clears the
    // mark — so `releaseCheckout` afterwards finds nothing to drop.
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

    try hc.commit(&slot, &prompt, false);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(?usize, null), e.checked_out_by);
    try testing.expectEqual(@as(usize, 608), e.tokens.len);
    try testing.expect(e.snapshot.entries[0].keys.ctx != null);
    hc.releaseCheckout(42, "finished");
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);

    // ...and the reclaimed entry restores like any other.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const res2 = try hc.lookupAndRestore(&slot2, &moe_off2, null, s, &prompt, false, 0, null, null);
    try testing.expectEqual(@as(usize, 607), res2.matched);
}

test "restore by move: a checked-out entry is invisible to a second slot, to eviction and to the published residency" {
    // Its snapshot is empty. A second slot restoring from it would get an
    // uninitialized cache; an eviction pass would free nothing and discard the
    // record the first slot is about to replace; and the connection thread
    // would credit bytes that are already a live slot's.
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

    // (c) a second slot MISSES — it never touches the moved buffer.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const res2 = try hc.lookupAndRestoreForSlot(&slot2, &moe_off2, null, s, &prompt, false, 0, null, null, null, 43);
    try testing.expectEqual(@as(usize, 0), res2.matched);
    try testing.expect(!res2.full_match);
    try testing.expectEqual(@as(usize, 0), slot2.step);
    // ...and the first slot's checkout is untouched by that miss.
    try testing.expectEqual(@as(?usize, 42), hc.entries.items[0].checked_out_by);

    // (d) the reclaimable/digest helpers exclude it: nothing to reclaim, and
    // no digest to publish.
    const digests = try hc.digestsAlloc(testing.allocator);
    defer testing.allocator.free(digests);
    try testing.expectEqual(@as(usize, 0), digests.len);
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytesFor(&prompt));
    // The bill still counts it — the bytes really are resident, in the slot.
    try testing.expectEqual(billed, hc.residentBytes());

    // Eviction cannot take it, even when nothing else is left and the caller
    // is not protecting the restored entry.
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
    // PURE. The gate is qwen4_exp's SSD-first arm (`ssd_first`) plus the env
    // switch; no other architecture's restore changes, and `=0` is today's
    // behaviour exactly.
    // Eligible: SSD-first, enabled, no pending flush, whole entry matched,
    // something left to append.
    try testing.expect(HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 608, true));
    // Off outside the SSD-first arm.
    try testing.expect(!HotPrefixCache.checkoutEligible(false, true, false, 600, 600, 608, true));
    // Off under the kill switch.
    try testing.expect(!HotPrefixCache.checkoutEligible(true, false, false, 600, 600, 608, true));
    // Off when the caller names no slot (it cannot promise a release).
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 608, false));
    // Off while a disk record shares the same buffers.
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, true, 600, 600, 608, true));
    // Partial hit: the entry is longer than the match.
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 500, 608, true));
    // Nothing to append: no donation to win, and the entry would be dropped
    // for free.
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 600, true));
    // An empty record is never checked out.
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 0, 0, 608, true));
}
