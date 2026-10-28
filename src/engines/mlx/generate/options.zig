//! Generator.InitOptions and the hooks the scheduler hands a generator.
const std = @import("std");
const transformer_mod = @import("../transformer.zig");
const drafter_mod = @import("../../../plugins/gemma4/assistant_drafter.zig");
const mtp_acceptance = @import("../../../sdk/spec/mtp_acceptance.zig");
const dflash_policy = @import("../spec/dflash/policy.zig");
const ForwardCtx = transformer_mod.ForwardCtx;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const DrafterModel = drafter_mod.DrafterModel;
const dflash_mod = @import("../spec/dflash/dflash.zig");
const DflashModel = dflash_mod.DflashModel;
const testing = std.testing;
const AdaptiveWidthState = @import("prefill_chunk.zig").AdaptiveWidthState;
const DFLASH_GATE_MIN_ACCEPTED_PER_ROUND = @import("../spec/gates.zig").DFLASH_GATE_MIN_ACCEPTED_PER_ROUND;
const Generator = @import("generator.zig").Generator;
const LogprobResult = @import("../sampling/logprobs.zig").LogprobResult;
const MtpHeadRef = @import("../spec/mtp_head_ref.zig").MtpHeadRef;
const MtpRestored = @import("../spec/mtp_head_ref.zig").MtpRestored;
const deinit = @import("generator.zig").Generator.deinit;


/// Receiver for SSM checkpoints salvaged out of a cancelled prefill.
/// `initWithOptions` captures stride checkpoints into a local list that
/// its errdefer frees — on `error.Cancelled` it instead moves them here
/// so the scheduler can commit a partial-prefix cache entry. Ownership
/// transfers either to the hot-cache entry (`commitWithSsm`) or back to
/// the sink's `deinit`.
pub const CancelledCheckpointSink = struct {
    /// Absolute prompt-token count forwarded into the KV when the chunk
    /// loop aborted (`ssm_checkpoint_pos_offset + pos`). This — NOT
    /// `cache.step`, which only advances when init completes — is the
    /// authoritative length for a cancelled-prefill commit.
    forwarded: usize = 0,
    checkpoints: []SSMCheckpoint = &.{},
    /// The allocator the checkpoints (and the slice itself) were
    /// allocated with — the same `sch.allocator` family the hot-cache
    /// entry later frees them with.
    alloc: ?std.mem.Allocator = null,

    pub fn deinit(self: *CancelledCheckpointSink) void {
        const a = self.alloc orelse return;
        for (self.checkpoints) |*cp| cp.deinit(a);
        if (self.checkpoints.len > 0) a.free(self.checkpoints);
        self.checkpoints = &.{};
        self.alloc = null;
    }
};

pub const InitOptions = struct {
    /// Skip the lazy pre-forward of the first sampled token. When set,
    /// init samples t1 synchronously and leaves `pending_logits/pending_token`
    /// empty — the cache lands at exactly `prompt_len` with t1 NOT in cache.
    /// `nextPld` v2 (mirroring `nextDrafter`) drives every step from that
    /// invariant: verify input is `[t1, draft[0..m-1]]` length `1+m`; full
    /// accept commits `1+m` tokens with cache landing at `prompt_len + TE_new`
    /// and NO post-step forward. Saves one decode-step forward per accepted
    /// PLD step at the cost of losing the lazy-pipeline overlap on cold
    /// (no-match) steps. The prompt-time gate disables PLD on novel content
    /// where cold-path dominates.
    pld_enabled: bool = false,
    /// The machine-sized prefill chunk frozen at load
    /// (`ModelConfig.pinned_prefill_chunk`, resolved by
    /// `server.resolvePrefillChunk`). 0 = unpinned, keep the launch width.
    ///
    /// It arrives per REQUEST rather than off `xfm.config` because the
    /// Transformer holds a COPY of the config taken when it was built, and
    /// the pin is written to the registry's config after load — the copy
    /// never sees it. The scheduler reads `slot.model.config`, which IS the
    /// object the admission guard bills against, so bill and forward cannot
    /// disagree. (Pinned 4096, prefilled at 8192.)
    pinned_prefill_chunk: usize = 0,
    /// `--prefill-decode-share` width while someone decodes, 0 = none. Applied after
    /// `effectivePrefillChunk`, so it narrows an explicit or env width too: the
    /// operator asked for the share as well.
    decode_share_width_cap: usize = 0,
    /// Enable Gemma 4 assistant drafter. When set, `drafter` must be
    /// non-null and already `bind()`-ed to `xfm`. Init's prefill final-token
    /// forward captures the post-final-norm hidden state into
    /// `Generator.last_hidden` (reused for the drafter's first-step
    /// h_prev — see comment in `nextDrafter`). Same lazy-pre-forward
    /// skip semantics as PLD.
    drafter_enabled: bool = false,
    /// Non-owning pointer to the loaded drafter (must be non-null when
    /// `drafter_enabled` is true).
    drafter: ?*DrafterModel = null,
    /// Number of tokens per draft round. Default 4 (3 drafter steps +
    /// 1 t1 prepend → length-4 verify forward).
    drafter_block_size: u32 = 4,
    /// Enable the DFlash block-drafter. When set, `dflash` must be
    /// non-null and `bind()`-ed to `xfm`. Prefill captures the trunk's
    /// target_layer_ids outputs chunk-by-chunk into the per-request
    /// assistant context cache (`Generator.dflash_ctx`). Same
    /// lazy-pre-forward skip semantics as PLD/drafter/MTP.
    dflash_enabled: bool = false,
    /// Non-owning pointer to the loaded DFlash assistant.
    dflash: ?*DflashModel = null,
    /// Effective DFlash block size (loader-resolved: assistant config
    /// clamped by --draft-block-size). 0 → the assistant config's value.
    dflash_block_size: u32 = 0,
    /// Accepted drafts/round required after warmup. Scheduler scales the
    /// M5/block-16 calibration to the effective draft width and lowers it
    /// for requests whose resolved mode has thinking enabled.
    dflash_min_accepted_per_round: f32 = DFLASH_GATE_MIN_ACCEPTED_PER_ROUND,
    /// Drive the DFlash rounds with `dflash_policy` (a sparse target, where a verify row costs):
    /// plain rounds replace the sticky serial fallback and the rows verified follow the draft's confidence.
    dflash_policy: bool = false,
    /// Enable the Qwen native MTP head. When set, `mtp` must be non-null
    /// and `bind()`-ed to `xfm`. Prefill builds the head's committed-
    /// history KV cache chunk-by-chunk (full-hidden capture) and the
    /// final-token forward captures `last_hidden`, exactly like the
    /// drafter path. Same lazy-pre-forward skip semantics as PLD/drafter.
    mtp_enabled: bool = false,
    /// Explicit verifier selected by the caller at construction.
    mtp_acceptance: mtp_acceptance.Mode = .exact,
    mtp_greedy_tail: bool = false,
    /// Non-owning pointer to the loaded MTP head.
    mtp: ?MtpHeadRef = null,
    /// The model's head before this request's opt-out (the qwen4 weights load with the trunk regardless).
    model_has_mtp: bool = false,
    /// Max tokens drafted per nextMtp round. 0 = auto (`--mtp-max-depth` not
    /// passed): resolved by `resolveMtpDepthCap` — MTP_ADAPTIVE_NAX_CAP
    /// for the measured M5 target+sidecar profile, otherwise
    /// MTP_ADAPTIVE_DEFAULT_CAP under the EV controller; DEFAULT_DEPTH in
    /// fixed mode. Explicit depths remain unchanged.
    mtp_depth: u32 = 0,
    /// When set, this slice (rather than `prompt_ids`) becomes the
    /// `prompt_ids_owned` source for PLD's n-gram lookup. Used by the
    /// server's KV-cache-reuse path to forward only the trailing tokens
    /// while still giving PLD the full prompt for matching.
    lookup_prompt: ?[]const u32 = null,
    /// Per-slot forward context (Phase 2 concurrent batching). When null,
    /// `initWithOptions` builds one from `xfm.defaultCtx()` so the legacy
    /// single-slot path is unchanged. Phase 2 callers pass a ForwardCtx
    /// whose `cache` / `moe_seq_offset` / `ssm_entries` / `vision_embeddings`
    /// point at the slot's own state. Stored by value on the Generator.
    ctx: ?ForwardCtx = null,
    /// Skip the lazy first-token pre-forward (regular path only). When set,
    /// init samples t1 synchronously and leaves `pending_logits` /
    /// `pending_token` empty — cache.step lands at exactly prompt_len with
    /// t1 NOT in cache. The first `next()` call's transition shim will
    /// sync-forward `[t1]` to seed pending_logits before the lazy chain.
    /// Used by the Phase 2 scheduler so a slot's cache state matches
    /// `forwardBatchedDecode`'s expectation (cache.step == prompt_len at
    /// the start of every decode tick). PLD / drafter paths already skip
    /// the lazy pre-forward unconditionally; this flag generalizes that
    /// behavior to the regular sampling path. Has no effect when
    /// `pld_enabled` or `drafter_enabled` is true.
    skip_lazy_preforward: bool = false,
    /// A spoken reply (`generate/speech.zig`): the prefill keeps its last row's logits and hidden, nothing sampled.
    speech: ?@import("speech.zig").Request = null,
    /// Phase 1 (performance-plan): during prefill, capture an SSM
    /// checkpoint every `ssm_checkpoint_stride` tokens. 0 = disabled.
    /// Snapshots land in `Generator.ssm_checkpoints` for the caller to
    /// drain into the hot prefix cache via `takeSsmCheckpoints()`. Only
    /// effective when the model has hybrid layers (otherwise the
    /// `ssm_entries` slice is empty and snapshots become no-op stubs).
    /// Chunked prefill aligns chunk ends to stride positions so each
    /// snapshot reflects a coherent state.
    ssm_checkpoint_stride: u32 = 0,
    /// Cap on retained checkpoints. Past it the list is thinned span-preservingly
    /// (`transformer.ssmCheckpointDropIndex`): lowest and newest always survive. 0 = unlimited.
    ssm_checkpoint_max: u32 = 16,
    /// Protect the first N positions (the lowest grid points) when thinning
    /// the capture list to its cap: they are the only restore anchor for a
    /// request that forked below a coarse grid's first multiple (#794).
    /// 0 = the plain thinning (upstream behaviour, byte-identical).
    ssm_low_anchors: usize = 0,
    /// Phase 1: absolute position of the FIRST token in `prompt_ids`.
    /// On a cold prefill this is 0. On the warm path (where the
    /// scheduler restored some prefix and now forwards only the tail),
    /// callers pass `hot_matched` so the captured checkpoints stamp
    /// absolute positions usable by future warm-path lookups against
    /// the full prompt.
    ssm_checkpoint_pos_offset: usize = 0,
    /// Placeholder rows inside a restored prefix (prefix-cache hit on an
    /// image prompt): the vision splice starts here, not at row 0.
    vision_rows_before: usize = 0,
    /// A DFlash assistant context restored from the prefix cache, whose
    /// `absLen()` already equals `ssm_checkpoint_pos_offset`. Ownership
    /// transfers to the Generator. Null = start the assistant blind at
    /// the offset (a restore forwards no trunk layers, so a cold context
    /// is what a reused prefix would otherwise get).
    dflash_ctx_restored: ?dflash_mod.DflashCtx = null,
    /// An MTP committed-history cache restored from the prefix cache,
    /// whose `base + cache.step()` already equals
    /// `ssm_checkpoint_pos_offset`. Ownership transfers to the
    /// Generator. Null = start the history blind at the offset (same
    /// contract as `dflash_ctx_restored`).
    mtp_cache_restored: ?MtpRestored = null,
    /// Cooperative abort for abandoned requests: checked between prefill
    /// chunks. The scheduler passes `&slot.cancelled`, set by the conn
    /// thread when the client disconnects mid-prefill. When it flips,
    /// `initWithOptions` returns `error.Cancelled` instead of grinding
    /// out the rest of a multi-minute ghost prefill.
    cancel_flag: ?*const std.atomic.Value(bool) = null,
    /// When the chunk loop aborts on `cancel_flag`, the SSM checkpoints
    /// captured so far move into this sink instead of dying with the
    /// failed construction — the scheduler salvages a partial-prefix
    /// cache commit from them. Null keeps the old drop-everything
    /// behaviour.
    cancelled_checkpoint_sink: ?*CancelledCheckpointSink = null,
    /// Per-token logprobs count for this request (0 = disabled). Callers
    /// that set `Generator.logprobs_n` after init must ALSO pass it here:
    /// init's split-prefill final-token forward reads it for the first token.
    logprobs_n: u32 = 0,
    /// `/v1/completions` echo: every prompt token after the first gets its
    /// `logprobs_n`-wide entry here. The prompt must be forwarded whole (no restore).
    prompt_logprobs: ?*std.ArrayList(LogprobResult) = null,
    /// LIVE prefill progress, in tokens actually forwarded so far by THIS
    /// prefill. Bumped once per chunk (not per token), read off-thread by
    /// the metrics gauge sampler.
    ///
    /// Without it the panel is blind during prefill: `prompt_tokens_total`
    /// and `prefill_time_seconds` only advance at request COMPLETION, and
    /// generated tokens only accrue during decode — so a multi-minute
    /// prefill saturates the GPU while both tiles read 0 / "—". The
    /// scheduler resets it to 0 when the prefill ends.
    prefill_progress: ?*std.atomic.Value(u64) = null,
    /// Total tokens this prefill will forward (post-cache tail, same scale
    /// as `prefill_progress`), published once; null on the bypass engines.
    prefill_expected: ?*std.atomic.Value(u64) = null,

    /// Called once per completed prefill chunk boundary (chunk state
    /// evaluated, allocator cache cleared), except the final one. The
    /// scheduler runs decode ticks for the streams already decoding at
    /// this seam, so a long cold prefill stalls them for at most one
    /// chunk-forward instead of its whole duration. Null = the prefill
    /// runs atomically.
    interleave_hook: ?InterleaveHook = null,
    /// SSD-first write-through (qwen4_exp only; see `WriteThroughHook`).
    write_through_hook: ?WriteThroughHook = null,
    /// The per-chunk prefill width, asked at every chunk boundary after that chunk's
    /// `mlx_clear_cache` and before the interleave tick. Null keeps the admitted width.
    /// Presence is NOT the arch gate (`serve` installs it process-wide); read `adaptive_chunk_width`.
    chunk_width_hook: ?ChunkWidthHook = null,
    /// Is the per-chunk adaptive width live for this request (`server.adaptivePrefillChunkEnabled`)?
    /// The only thing that may widen `tailMergeMaxFor`'s bound.
    adaptive_chunk_width: bool = false,
};

pub const InterleaveHook = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque) void,
};

/// SSD-first write-through: called at every completed prefill chunk with the absolute KV
/// position and the SSM checkpoints so far, so a cancelled prefill leaves a restorable prefix.
pub const WriteThroughHook = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, abs_kv_pos: usize, cps: []const SSMCheckpoint) void,
};

/// Returns the width the next chunk should run at (`cur` unchanged = no change).
pub const ChunkWidthHook = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, pos: usize, cur: u32, cap: u32, st: *AdaptiveWidthState) u32,
    /// Re-price a widen after the interleave tick has allocated for co-tenant slots. A
    /// step-down needs no confirmation: it is the safe direction.
    confirm: *const fn (ctx: *anyopaque, pos: usize, want: u32) bool,
};

// ── tests ──


test "InitOptions.lookup_prompt overrides prompt_ids_owned source" {
    // When the server's cache-reuse path forwards only a trailing-tail
    // prompt slice but supplies the full original prompt via
    // `InitOptions.lookup_prompt`, PLD's n-gram buffer must be cloned from
    // the full slice — not the truncated tail.
    const tail = [_]u32{99};
    const full = [_]u32{ 10, 20, 30, 99 };
    const src = Generator.pickLookupPromptSource(&tail, &full);
    try testing.expectEqual(@as(usize, 4), src.len);
    try testing.expectEqualSlices(u32, &full, src);
}

test "InitOptions.lookup_prompt = null preserves existing behavior" {
    // Back-compat path: when callers don't set `lookup_prompt`, the source
    // is the unmodified `prompt_ids` slice — same buffer the Generator
    // received pre-fix.
    const prompt = [_]u32{ 1, 2, 3, 4, 5 };
    const src = Generator.pickLookupPromptSource(&prompt, null);
    try testing.expectEqual(prompt.len, src.len);
    try testing.expectEqualSlices(u32, &prompt, src);
    try testing.expectEqual(@as([*]const u32, prompt[0..].ptr), src.ptr);
}
