const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const dsv4_mod = @import("deepseek_v4.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const json_grammar = @import("json_grammar.zig");
const json_schema = @import("json_schema.zig");
const token_mask = @import("token_mask.zig");
const io_util = @import("io_util.zig");
const pld_index = @import("pld_index.zig");
const drafter_mod = @import("drafter.zig");
const mtp_mod = @import("mtp.zig");
const round_cost = @import("round_cost.zig");
const ane_mod = @import("ane.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ForwardCtx = transformer_mod.ForwardCtx;
const SSMCacheEntrySnapshot = transformer_mod.SSMCacheEntrySnapshot;
const ssmSnapshot = transformer_mod.ssmSnapshot;
const ssmSnapshotDeinit = transformer_mod.ssmSnapshotDeinit;
const ssmRestore = transformer_mod.ssmRestore;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const captureSsmCheckpoint = transformer_mod.captureSsmCheckpoint;
const DrafterModel = drafter_mod.DrafterModel;
const dflash_mod = @import("dflash.zig");
const DflashModel = dflash_mod.DflashModel;
const KVCache = transformer_mod.KVCache;

/// Module-level overrides for prefill behavior. Defaults match the original
/// hardcoded values; main.zig may overwrite these from CLI flags before
/// `serve()` runs. Per-request reads happen on the same thread that did the
/// CLI parse, so no atomicity needed.
pub var prefill_chunk_override: usize = 8192;

/// Set by `--prefill-chunk` in main.zig. An operator-chosen width outranks the
/// per-model chunk the memory sizer pins (`ModelConfig.pinned_prefill_chunk`);
/// without this flag the default 8192 is indistinguishable from a request.
/// Same set-once-at-CLI-parse contract as `prefill_chunk_override`.
pub var prefill_chunk_explicit: bool = false;
pub var prefill_trace_force: bool = false;

/// The width `MLX_SERVE_PREFILL_CHUNK` asked for, or 0 when nothing did.
/// `effectivePrefillChunk` honours it VERBATIM, so it is also the thing that
/// turns the per-chunk adaptive width off: an operator who pinned a width
/// pinned every forward.
pub fn envPrefillChunk() usize {
    return readEnvUsize("MLX_SERVE_PREFILL_CHUNK", 0);
}

/// The state one prefill carries between chunk boundaries for the adaptive
/// width. Lives on `runPrefill`'s stack, so it dies with the request and two
/// concurrent prefills cannot share a ratchet. The POLICY that reads it is
/// `server.adaptivePrefillWidth`; the type lives here because the loop owns
/// the instance and this module has no server.zig import.
/// The ONE place a chosen width becomes the running width, so the summary
/// counters cannot drift from `cur_chunk` (the widen commits from a different
/// point in the loop than the step-down since S17).
pub fn commitAdaptiveWidth(cur: *usize, st: *AdaptiveWidthState, w: u32) void {
    cur.* = w;
    st.transitions +|= 1;
    st.width_min = @min(st.width_min, w);
    st.width_max = @max(st.width_max, w);
}

pub const AdaptiveWidthState = struct {
    /// Consecutive probes that supported the next rung up.
    supporting: u8 = 0,
    /// One-way ratchet: a prefill that has stepped DOWN never widens again.
    ratcheted: bool = false,
    /// Summary, for the one per-request line: how many times the width moved,
    /// and the headroom at the first and last probe.
    transitions: u32 = 0,
    width_min: u32 = 0,
    width_max: u32 = 0,
};

/// MTP prefill-history window (`--mtp-history-window`; 0 = full history).
/// Same set-once-at-CLI-parse contract as `prefill_chunk_override`.
/// DEFAULT 0 (full): the A/B gate failed for windowing — at 64K ctx on the
/// stock Qwen3.6-27B head, window 8192 cost 14 acceptance points (68.2% ->
/// 54.0%) and 4.2 decode tok/s for ZERO prefill benefit (184.7 vs 185.1
/// tok/s); at 32K it was a wash. Qwen's stock head drafts from deep history.
pub var mtp_history_window_override: usize = 0;

/// `--max-mtp-ctx N`: MTP stays OFF past N context tokens (0 = unlimited).
/// Same set-once-at-CLI-parse contract as `mtp_history_window_override` — a
/// machine-wide dispatch limit, so ONE value serves both readers: the HTTP
/// admission gate (`server.admitMtpForCtx`, applied once per request after
/// the prompt is tokenized) and the per-round check at the top of `nextMtp`
/// (prompt + generated, so a long generation switches to serial mid-flight
/// through the existing `spec_disabled_runtime` fallback). The predicate and
/// its inclusive boundary live in `mtp.mtpCtxWithinLimit`.
pub var max_mtp_ctx: u32 = 0;

/// Why a request left speculative decoding for the rest of its generation.
pub const SpecDisableReason = enum {
    none,
    /// Realized draft acceptance did not pay for the verify overhead.
    acceptance,
    /// Context crossed `--max-mtp-ctx` mid-generation.
    max_ctx,
    /// The measured round at the planned width cost more per token than a
    /// measured serial token in this KV bucket (`MtpAdaptive`).
    adaptive,
};

/// Effective MTP history window for a prefill forwarding `prefix_len`
/// positions: 0 (capture everything) unless windowing is on AND the tail is
/// past the threshold — short/medium prompts keep byte-identical behavior.
pub fn effectiveMtpHistoryWindow(prefix_len: usize, window: usize) usize {
    if (window == 0 or prefix_len <= mtp_mod.HISTORY_WINDOW_THRESHOLD) return 0;
    return window;
}

/// Does prefill chunk [pos, end) contribute MTP history? Zero window = all
/// chunks; otherwise only chunks overlapping the last `window` positions of
/// the prefix (a boundary chunk contributes whole — the window is a floor,
/// never an exact cut, so acceptance never loses mid-chunk context).
pub fn chunkNeedsMtpHistory(pos: usize, end: usize, prefix_len: usize, window: usize) bool {
    _ = pos;
    if (window == 0) return true;
    return end > prefix_len - @min(window, prefix_len);
}

/// One layer's materialized-score budget for unfused-SDPA prefill (see
/// boundedPrefillChunk). 4 GiB keeps the full 8K chunk for every context up
/// to ~16K on 16-head models (further on fewer heads) and degrades gradually
/// to the 512 floor as heads × context grows toward 262K.
pub const PREFILL_SCORES_BUDGET_BYTES: u64 = 4 << 30;
/// Lower bound for the auto-capped prefill chunk; also its rounding grain
/// (repeating chunk sizes let the MLX allocator cache reuse score buffers).
pub const PREFILL_CHUNK_FLOOR: usize = 512;

/// MLX's fused SDPA kernels cover head_dim <= 128. Every Gemma-4 and
/// Qwen3.5/3.6 checkpoint ships head_dim=256, which falls back to the
/// composed path that MATERIALIZES a [heads, chunk, total_kv] bf16 score
/// tensor per layer — at an 8K chunk × 255K ctx × 16 heads that is ~67 GB
/// and an uncatchable Metal command-buffer OOM. Cap the chunk so ONE layer's
/// score tensor stays within PREFILL_SCORES_BUDGET_BYTES at this prompt's
/// FINAL KV length (the last chunk attends to everything). Fused head dims
/// and short contexts return `base_chunk` untouched, so typical traffic
/// keeps full prefill throughput; the cap only bites when heads × total_ctx
/// actually outgrows the budget. Never raises a caller-lowered base.
///
/// DELIBERATELY ignores the msv_attn_p256 fused kernel (unlike
/// prefillEvalCadence / prefillMemoryNeeded, which drop their score term via
/// transformer.prefillHeadDimFused): the fused kernel removes the SCORE
/// transient, but a big chunk still scales the OTHER per-chunk transients
/// (MoE gather buffers, per-chunk KV concat) — measured LIVE on
/// gemma-4-26B-A4B at a 99K prompt: fused @ chunk 8192 = 736 tok/s / 61.2 GB
/// peak vs fused @ chunk 1024 = 712 tok/s / 39.5 GB. +3% speed is not worth
/// +22 GB peak (a 64 GB Mac dies), so the cap stays keyed on raw head_dim.
///
/// `sliding_band_arch` (config.has_sliding_window) picks the policy family:
/// archs WITHOUT sliding-band layers (qwen3_5/3_6: GDN + full attention)
/// additionally cap the auto chunk at 2048 — composed-causal prefill
/// measured strictly faster and ~9 GB lighter there (see the inline
/// comment). Gemma keeps the formula-only policy for its fused band layers.
/// `score_head_dim` is the width the SCORE tensor is contracted at
/// (`ModelConfig.prefillScoreHeadDim`) — not necessarily `head_dim`. An MLA arch
/// scores at 192 while storing 128-wide values, and reading `head_dim` there
/// silently exempted the one arch in the tree whose composed path materializes
/// the biggest score tensors. The two hd-256 policy branches below stay keyed
/// on 256 exactly: both were measured on hd-256 checkpoints and neither
/// generalizes (the fused kernel is hd-256-only; the 2048 composed cap was
/// tuned against a 27B's own prefill ladder).
pub fn boundedPrefillChunk(base_chunk: usize, score_head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool) usize {
    const head_dim = score_head_dim;
    if (head_dim <= 128 or n_heads == 0 or total_ctx == 0) return base_chunk;
    // Non-sliding hd-256 archs under FUSED causal (the default since the
    // budgeted-dispatch flip): no score tensor exists, so the scores-budget
    // formula below is moot — and its old shrink (1024 at 64K on 24 heads)
    // starved the dequant+GEMM qmm route, which needs M >= 2048 to engage
    // (the 64K rung was the ladder's weakest for exactly this reason).
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (the gemma-26B@99K lesson: +3% speed for +22 GB peak is a bad trade).
    // DENSE hybrids have no gather transients and a full-size chunk halves
    // the per-chunk dequant sweeps: chunk 8192 measured +1.4% over 4096 at
    // the 8K rung on Qwen3.6-27B dense (M4 Max, 2026-07-30), flat at 32K.
    // Never raises a caller-lowered base.
    if (head_dim == 256 and !sliding_band_arch and transformer_mod.fused256CausalMode() == .all) {
        return @min(base_chunk, if (is_moe) @as(usize, 4096) else @as(usize, 8192));
    }
    // Composed-causal fallback (MLX_SERVE_FUSED_256_CAUSAL=0): SMALL chunks
    // measured strictly faster AND lighter on the 27B (2026-07-12 ladder,
    // M4 Max): 8K 225 -> 235.8 tok/s and peak 28.9 -> 19.8 GB at chunk
    // 2048; 32K 205.4 -> 209.3. Chunk boundaries ARE block-level causal
    // skipping for composed attention. Sliding-band archs (gemma) keep big
    // chunks — their local layers run the fused band kernel, which
    // block-skips in-kernel and wants the fewest KV re-walks (26B@99K:
    // 712 tok/s at the formula chunk).
    const causal_cap: usize = if (sliding_band_arch or head_dim != 256) base_chunk else @min(base_chunk, 2048);
    const per_row: u64 = @as(u64, n_heads) * @as(u64, total_ctx) * 2;
    const max_chunk: u64 = PREFILL_SCORES_BUDGET_BYTES / per_row;
    if (max_chunk >= causal_cap) return causal_cap;
    const floored = @max(
        @as(u64, PREFILL_CHUNK_FLOOR),
        max_chunk - (max_chunk % PREFILL_CHUNK_FLOOR),
    );
    return @intCast(@min(floored, @as(u64, causal_cap)));
}

/// The prefill chunk `initWithOptions` will actually use for a request:
/// MLX_SERVE_PREFILL_CHUNK env (explicit tuning knob — honored verbatim,
/// never safety-capped) > --prefill-chunk / default, capped by
/// boundedPrefillChunk. Exported so server.zig's admission guard
/// (checkAttentionMemory) models the SAME chunk the prefill will run with —
/// the guard and the real prefill must not drift.
pub fn effectivePrefillChunk(head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool, pinned_chunk: usize) usize {
    const env_chunk = readEnvUsize("MLX_SERVE_PREFILL_CHUNK", 0);
    if (env_chunk > 0) return env_chunk;
    // `pinned_chunk` is the machine-sized cap frozen at load
    // (`ModelConfig.pinned_prefill_chunk`, from `server.resolvePrefillChunk`):
    // the widest chunk whose one-off transient reserve still leaves this box a
    // real KV budget. It NARROWS, never raises, and an explicit
    // `--prefill-chunk` outranks it — the operator asked for a width.
    const base: usize = if (prefill_chunk_explicit or pinned_chunk == 0)
        prefill_chunk_override
    else
        @min(prefill_chunk_override, pinned_chunk);
    return boundedPrefillChunk(base, head_dim, n_heads, total_ctx, sliding_band_arch, is_moe);
}

/// Read an unsigned integer from an environment variable, falling back to
/// `default` when unset, empty, or unparseable. Uses libc getenv to stay
/// allocator-free at call sites.
fn readEnvUsize(name: [:0]const u8, default: usize) usize {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return default;
    const slice = std.mem.sliceTo(raw.?, 0);
    if (slice.len == 0) return default;
    return std.fmt.parseInt(usize, slice, 10) catch default;
}

/// Read a finite, NON-NEGATIVE float from an environment variable, falling
/// back to `default` when unset, empty, unparseable, negative or non-finite.
/// (The doc used to say "strictly positive" while the code accepted 0; 0 is a
/// meaningful setting for every current caller — a zero margin, a zero warm —
/// so the comment was the wrong half.)
fn readEnvFloat(name: [:0]const u8, default: f32) f32 {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return default;
    const slice = std.mem.sliceTo(raw.?, 0);
    if (slice.len == 0) return default;
    const v = std.fmt.parseFloat(f32, slice) catch return default;
    return if (std.math.isFinite(v) and v >= 0) v else default;
}

/// Truthy if the env var is exactly "1". Anything else (unset, "0", "true",
/// "yes") is false — keep matching surface tight to avoid surprises.
fn readEnvBool(name: [:0]const u8) bool {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return false;
    const slice = std.mem.sliceTo(raw.?, 0);
    return std.mem.eql(u8, slice, "1");
}

/// Grammar-constrained sampling state. The caller owns `grammar`, `token_bytes`,
/// and `mask_buf`; the generator only reads them. `mask_buf.len` must equal
/// `token_bytes.bytes.len` (the tokenizer's vocab size).
/// Which MTP head a generator drives.
///
/// The controller around it is head-agnostic — the EV planner, the acceptance
/// gate, `mtpBatchedAcceptGraph`, the pre-draft and the horizon valve all work
/// in tokens and probabilities — so the split is exactly these five operations
/// and nothing else.
pub const MtpHeadRef = union(enum) {
    qwen: *mtp_mod.MtpModel,
    /// qwen4_exp: the head and its history live on the Transformer
    /// (`qwen4_mtp`, module-owned ⇒ single-flight); row r of the history is
    /// (pre-mixer stream at position r, token r+1), query position r+1.
    qwen4: *Transformer,

    /// Is this head's decode state MODULE-owned — one instance per MODEL,
    /// shared by every slot on it — rather than per-request?
    ///
    /// The sidecar arm's cache is a `KVCache` this Generator allocated and
    /// owns; the in-checkpoint arm's is `Qwen4Mtp.cache` on the Transformer,
    /// which is why `scheduler.slotExclusiveDecode` gives the slot driving it
    /// the model to itself. That was written as a hand-rolled `head == .qwen4`
    /// on the scheduler side and re-derived again here for S21 — a list of ONE
    /// in two places, the exact shape `ownsModuleDecodeState` was factored out
    /// of. BOTH sides now ask this. A new head arm answers here, once.
    pub fn moduleOwned(self: MtpHeadRef) bool {
        return switch (self) {
            .qwen => false,
            .qwen4 => true,
        };
    }

    pub fn makeCache(self: MtpHeadRef, allocator: std.mem.Allocator) !MtpCacheRef {
        return switch (self) {
            .qwen => |h| .{ .qwen = try h.makeCache(allocator) },
            .qwen4 => |t| blk: {
                try t.qwen4MtpReset();
                break :blk .{ .qwen4 = t };
            },
        };
    }

    /// The head projects only what the caller consumes: this step wants the
    /// LAST row (`want_logits`) or no logits at all, never the S-row block —
    /// the merged history step after a partial accept is `1 + accepted` rows
    /// wide and a full-block mixer + 248320-wide lm_head over it is thrown
    /// away but for one row.
    fn qwen4Step(t: *Transformer, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: c_int, want: mtp_mod.StepWant, mrope_ctx: ?mtp_mod.MropeContext) !mtp_mod.StepOut {
        // `.mixed` is NOT `.none`: a rerank draft still needs the mixer output,
        // which is the vector the lm_head consumes. `hidden_next` is the
        // PRE-mixer stream on this arm and would be a silent shape error there.
        const project: Transformer.Qwen4MtpProject = switch (want) {
            .logits => .last_row,
            .mixed => .mixed_last_row,
            .none => .none,
        };
        const out = try t.qwen4MtpForward(hidden, id_arr, rope_offset + 1, mrope_ctx, project);
        return .{ .logits = out.logits, .hidden_next = out.stream, .rerank_x = out.mixed };
    }

    /// One head forward over L positions. `.logits` returns the LAST row only,
    /// on both arms. `.mixed` asks for the lm_head's INPUT vector instead: on
    /// the sidecar that is `hidden_next` and nothing extra is produced, on
    /// qwen4_exp it is the mixer output, returned in `rerank_x`.
    pub fn forward(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        id_arr: mlx.mlx_array,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        want: mtp_mod.StepWant,
        mrope_ctx: ?mtp_mod.MropeContext,
    ) !mtp_mod.StepOut {
        return switch (self) {
            .qwen => |h| mtp_mod.forwardWithMrope(h, target, &cache.qwen, id_arr, hidden, rope_offset, want == .logits, mrope_ctx),
            .qwen4 => |t| qwen4Step(t, id_arr, hidden, rope_offset, want, mrope_ctx),
        };
    }

    /// Append committed history without projecting logits.
    pub fn appendHistory(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        token_ids: []const u32,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        mrope_ctx: ?mtp_mod.MropeContext,
        allocator: std.mem.Allocator,
    ) !void {
        switch (self) {
            .qwen => |h| try mtp_mod.appendHistoryWithMrope(h, target, &cache.qwen, token_ids, hidden, rope_offset, mrope_ctx),
            .qwen4 => |t| {
                const ids_i32 = try allocator.alloc(i32, token_ids.len);
                defer allocator.free(ids_i32);
                for (token_ids, 0..) |tok, i| ids_i32[i] = @intCast(tok);
                const shape = [_]c_int{@intCast(token_ids.len)};
                const id_arr = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 1, .int32);
                defer _ = mlx.mlx_array_free(id_arr);
                const out = try qwen4Step(t, id_arr, hidden, rope_offset, .none, mrope_ctx);
                _ = mlx.mlx_array_free(out.hidden_next);
            },
        }
    }

    /// Draft-rerank scheme (see mtp.zig): greedy drafts pick via a coarse
    /// 2-bit readout + trunk-head top-32 re-score instead of a full
    /// draft-head projection + argmax.
    pub fn canRerankDrafts(self: MtpHeadRef) bool {
        return switch (self) {
            .qwen => |h| h.canRerankDrafts(),
            // Built lazily on the first ask: `loadQwen4Mtp` runs while the
            // Transformer is still under construction and its lm_head — the
            // very weight the coarse head is a copy of — is not assigned yet.
            .qwen4 => |t| t.qwen4DraftRerankReady(),
        };
    }

    /// `x` is whatever the TARGET's lm_head consumes for this arm — the
    /// sidecar's post-norm hidden, or qwen4_exp's MIXER output. Never
    /// `hidden_next` on qwen4_exp: that is the pre-mixer `[B,S,hc*H]` stream.
    pub fn draftSelect(
        self: MtpHeadRef,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        return switch (self) {
            .qwen => |h| h.draftSelect(target, x, suppress_mask),
            .qwen4 => |t| t.qwen4DraftSelect(x, suppress_mask),
        };
    }

    /// Every G17/NAX cost surface is calibrated against one exact runtime
    /// geometry — the dense Qwen3.6/3.8-27B sidecars on the `.qwen` arm, the
    /// qwen4_exp in-checkpoint head on its own arm — so every other head
    /// plans under `generic`: an off-profile head served by a calibrated
    /// surface would plan depths the measurement never covered.
    pub fn costProfile(self: MtpHeadRef, target: *const Transformer) mtp_mod.MtpCostProfile {
        return switch (self) {
            .qwen => |h| h.m5NaxCostProfile(target),
            .qwen4 => |t| mtp_mod.qwen4G17CostProfile(t),
        };
    }

    /// The head's last healthy acceptance surface, or null when it has never
    /// published one. BOTH arms store it on the head object itself — the
    /// sidecar `MtpModel` and the in-checkpoint `Qwen4Mtp` — so a seed dies
    /// with the head and can never cross models.
    pub fn evSeed(self: MtpHeadRef) ?struct { accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32 } {
        return switch (self) {
            .qwen => |h| if (h.ev_seed_accept) |a| .{ .accept = a, .m_lo = h.ev_seed_m_lo } else null,
            .qwen4 => |t| blk: {
                if (t.qwen4_mtp) |*m| {
                    if (m.ev_seed_accept) |a| break :blk .{ .accept = a, .m_lo = m.ev_seed_m_lo };
                }
                break :blk null;
            },
        };
    }

    pub fn setEvSeed(self: MtpHeadRef, accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32) void {
        switch (self) {
            .qwen => |h| {
                h.ev_seed_accept = accept;
                h.ev_seed_m_lo = m_lo;
            },
            .qwen4 => |t| {
                // A model served with `--no-mtp` has no head to seed.
                if (t.qwen4_mtp) |*m| {
                    m.ev_seed_accept = accept;
                    m.ev_seed_m_lo = m_lo;
                }
            },
        }
    }
};

/// `MLX_SERVE_MTP_HEAD_PERSIST=0` restores the pre-persistence behaviour for
/// the qwen4_exp in-checkpoint head: its committed history is neither
/// snapshotted into a prefix-cache entry nor restored from one, so every
/// reused prefix drafts from an empty head (`qwen4MtpReset`). Read once.
var mtp_head_persist_env: ?bool = null;
/// PURE arm of the switch: absent (and, deliberately, empty) is ON — only a
/// literal "0" turns persistence off, so a harness that exports the variable
/// with no value cannot silently disable it.
pub fn mtpHeadPersistFromEnv(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}
pub fn mtpHeadPersistEnabled() bool {
    if (mtp_head_persist_env) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_MTP_HEAD_PERSIST");
    const on = mtpHeadPersistFromEnv(if (raw) |r| std.mem.sliceTo(r, 0) else null);
    mtp_head_persist_env = on;
    return on;
}

/// The head's committed-history cache.
pub const MtpCacheRef = union(enum) {
    qwen: KVCache,
    qwen4: *Transformer,

    pub fn step(self: *const MtpCacheRef) usize {
        return switch (self.*) {
            .qwen => |*c| c.step,
            .qwen4 => |t| t.qwen4_mtp.?.seq_offset,
        };
    }

    /// The underlying KVCache — what the prefix cache's spec-snap machinery
    /// snapshots and restores. The qwen4 head's state is NOT KV-only (it also
    /// owns the QSA key history + pooled blocks + its own row count), so its
    /// KV alone is never enough: a caller that takes this pointer for that arm
    /// MUST also carry `head()`, and the snap machinery declines an adoption
    /// that has one without the other. Null when head persistence is off
    /// (`MLX_SERVE_MTP_HEAD_PERSIST=0`), which restores the old behaviour of
    /// neither committing nor restoring the qwen4 head.
    pub fn kv(self: *MtpCacheRef) ?*KVCache {
        return switch (self.*) {
            .qwen => |*c| c,
            .qwen4 => |t| if (mtpHeadPersistEnabled()) &t.qwen4_mtp.?.cache else null,
        };
    }

    /// The Transformer owning the in-checkpoint head, for the arms that need
    /// the non-KV half (QSA aux entry, `seq_offset`, `pos_base`). Null on the
    /// sidecar arm and whenever `kv()` is null, so the two always travel
    /// together.
    pub fn head(self: *MtpCacheRef) ?*Transformer {
        return switch (self.*) {
            .qwen => null,
            .qwen4 => |t| if (mtpHeadPersistEnabled()) t else null,
        };
    }

    pub fn truncate(self: *MtpCacheRef, len: usize, s: mlx.mlx_stream) !void {
        switch (self.*) {
            .qwen => |*c| try c.truncate(len, s),
            .qwen4 => |t| try t.qwen4MtpTruncate(len),
        }
    }

    pub fn deinit(self: *MtpCacheRef) void {
        switch (self.*) {
            .qwen => |*c| c.deinit(),
            .qwen4 => {},
        }
    }

    /// Append this cache's storage to a prefill eval batch, so the chunk's
    /// activation graph can be freed with the chunk instead of accumulating
    /// across the whole prompt.
    pub fn appendEvalArrays(self: *const MtpCacheRef, vec: mlx.mlx_vector_array) void {
        switch (self.*) {
            .qwen => |*c| for (c.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values);
            },
            .qwen4 => |t| for (t.qwen4_mtp.?.cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values);
            },
        }
    }
};

/// An MTP committed-history cache restored from the prefix cache, plus the
/// absolute target position its index 0 represents.
pub const MtpRestored = struct { cache: MtpCacheRef, base: usize };

pub const Constraint = struct {
    grammar: *json_grammar.Grammar,
    token_bytes: *const token_mask.TokenBytes,
    mask_buf: []bool,
};

/// RAII bundle for grammar-constrained sampling. Owns the parsed schema,
/// grammar state machine, and per-step mask buffer. The embedded `Constraint`
/// holds pointers/slices into the surrounding struct, so this struct must NOT
/// be moved after `initFromValue`. Construct via `var sc: SchemaConstraint =
/// undefined; try sc.initFromValue(...);` and pass `&sc.constraint` to
/// `SamplingParams`.
pub const SchemaConstraint = struct {
    schema: json_schema.Schema,
    grammar: json_grammar.Grammar,
    mask_buf: []bool,
    constraint: Constraint,
    allocator: std.mem.Allocator,

    /// Initialize in-place from a JSON schema value. On failure, any partial
    /// allocations made during this call are freed and the struct is left
    /// undefined (do not call `deinit`).
    pub fn initFromValue(
        self: *SchemaConstraint,
        allocator: std.mem.Allocator,
        schema_value: std.json.Value,
        token_bytes: *const token_mask.TokenBytes,
    ) !void {
        self.allocator = allocator;
        self.schema = try json_schema.parse(allocator, schema_value);
        errdefer self.schema.deinit();

        self.grammar = try json_grammar.Grammar.init(allocator, &self.schema);
        errdefer self.grammar.deinit();

        self.mask_buf = try allocator.alloc(bool, token_bytes.bytes.len);
        errdefer allocator.free(self.mask_buf);

        self.constraint = .{
            .grammar = &self.grammar,
            .token_bytes = token_bytes,
            .mask_buf = self.mask_buf,
        };
    }

    pub fn deinit(self: *SchemaConstraint) void {
        self.allocator.free(self.mask_buf);
        self.grammar.deinit();
        self.schema.deinit();
    }
};

/// Per-token logprob info (OpenAI format).
pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

/// Logprob result for a single generated token.
pub const LogprobResult = struct {
    token_logprob: f32, // logprob of the chosen token
    top_logprobs: []TokenLogprob, // top N alternatives (caller must free)
};

/// Sampling parameters for token generation.
pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    seed: ?u64 = null,
    /// Draw index under `seed`: every sample takes a fresh key.
    draw: u64 = 0,
    /// When non-null, generation is constrained to outputs that satisfy the
    /// grammar at byte level. Forces a synchronous sampling path (no lazy
    /// pipeline) since grammar advancement requires the realized token id.
    constraint: ?*Constraint = null,
    /// Reserved-token suppression mask: `[vocab]` bool, true = the sampler
    /// must never draw this id (reserved specials like `<|fim_hole|>`, which
    /// a degenerate distribution can rank top-5 at a collapsed position — a
    /// reserved marker in chat output is always a bug). Model-lifetime,
    /// OWNED by the Transformer (`suppress_mask`), non-owning here; wired by
    /// `Generator.initWithOptions` so every sampling path inherits it.
    /// Applied by both samplers and both stochastic-verify filters, fully
    /// lazy (`mlx_where` + -inf, no host sync); logprobs deliberately keep
    /// reading the RAW logits — the field reports the model, the mask is
    /// sampling policy. Null = no suppression (kill switch, no-template
    /// models, every non-suppressing arch).
    suppress_mask: ?mlx.mlx_array = null,
};

/// Build the `[vocab]` bool suppression mask (true = never sample) on the
/// host, once per model load. Caller owns the returned array.
///
/// Two disjoint sources of "never sample this row":
///   - `ids`: the reserved specials (`tokenizer.reservedOutputIds`).
///   - `[defined_vocab, vocab)`: the checkpoint's PADDING rows. A config's
///     `vocab_size` is a padded matrix dimension, not a vocabulary — the
///     tokenizer's highest defined id + 1 is. The rows in between decode to
///     nothing, so a sample from one emits nothing while consuming a step and
///     poisoning the KV; they were the one class of unsampleable row this
///     mask never covered.
/// `defined_vocab == 0` or `>= vocab` means "no padding known" and adds
/// nothing — a mask that suppressed the whole vocab is `-inf` everywhere,
/// which argmaxes to id 0 (the all-false-mask class, from the other side).
pub fn buildSuppressMask(ids: []const u32, defined_vocab: usize, vocab: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    _ = s;
    const alloc = std.heap.page_allocator;
    const buf = try alloc.alloc(bool, vocab);
    defer alloc.free(buf);
    fillSuppressMask(buf, ids, defined_vocab);
    const shape = [_]c_int{@intCast(vocab)};
    return mlx.mlx_array_new_data(buf.ptr, &shape, 1, .bool_);
}

/// The host half of `buildSuppressMask`, pure so the row set is testable
/// without a GPU. `buf.len` is the mask's vocab.
pub fn fillSuppressMask(buf: []bool, ids: []const u32, defined_vocab: usize) void {
    @memset(buf, false);
    for (ids) |id| {
        if (id < buf.len) buf[id] = true;
    }
    if (defined_vocab > 0 and defined_vocab < buf.len) {
        @memset(buf[defined_vocab..], true);
    }
}

/// Derive + install the reserved-token suppression mask on a freshly-built
/// Transformer (both the serve load path and the CLI run path call this).
/// The id set is `tokenizer.reservedOutputIds`: `special: true` added tokens
/// minus EOS/stop ids minus template-emitted markers. Kill switch
/// `MLX_SERVE_SUPPRESS_RESERVED=0`. Never fails a load — any error logs and
/// leaves the mask null (suppression off), and engagement is one-shot-logged
/// so a silent no-op is visible (the silent-fallback class).
pub fn installSuppressMask(xfm: *Transformer, tok: *const Tokenizer, chat_template: []const u8, eos_ids: []const u32) void {
    if (std.c.getenv("MLX_SERVE_SUPPRESS_RESERVED")) |v| {
        if (v[0] == '0') {
            log.info("[suppress] reserved-token suppression disabled by env\n", .{});
            return;
        }
    }
    const ids = tokenizer_mod.reservedOutputIds(xfm.allocator, tok.flagged_specials, chat_template, eos_ids) catch |err| {
        log.warn("[suppress] derivation failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    defer xfm.allocator.free(ids);
    // The mask's length is the LOGITS dim, not the embedding table's:
    // inkling slices its lm_head to `unpadded_vocab_size`, and a mask sized
    // to the padded vocab would fail the `where` broadcast on every sample.
    const logits_dim: usize = if (xfm.config.unpadded_vocab_size > 0)
        xfm.config.unpadded_vocab_size
    else
        xfm.config.vocab_size;
    // Padding rows past the tokenizer's last defined id. A checkpoint that
    // DECLARES `unpadded_vocab_size` has already had those rows sliced off the
    // lm_head, so `logits_dim` IS the real width there — applying the
    // tokenizer's cut on top would be a second, different trim of a vocab that
    // is already trimmed. One trim, whichever the checkpoint provides.
    const defined_vocab: usize = if (xfm.config.unpadded_vocab_size > 0) 0 else tok.definedVocabSize();
    const pad_rows: usize = if (defined_vocab > 0 and defined_vocab < logits_dim)
        logits_dim - defined_vocab
    else
        0;
    if (ids.len == 0 and pad_rows == 0) return;
    xfm.suppress_mask = buildSuppressMask(ids, defined_vocab, logits_dim, xfm.s) catch |err| {
        log.warn("[suppress] mask build failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    log.info(
        "[suppress] {d} of {d} flagged specials masked from sampling (template + eos exempt); {d} padding rows past id {d} of {d}\n",
        .{ ids.len, tok.flagged_specials.len, pad_rows, defined_vocab, logits_dim },
    );
}

/// `out = where(mask, -inf, logits)` — the masked lanes get EXACTLY -inf
/// (never an additive -inf, whose 0×-inf/NaN edge the parity rules exist
/// for). `[V]` broadcasts over both `[1, V]` and `[1, 1, V]` logits.
fn applySuppressMask(out: *mlx.mlx_array, logits: mlx.mlx_array, mask: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(out, mask, neg_inf, logits, s));
}

/// A speculative decode step (PLD / drafter / MTP) cannot honor a grammar
/// constraint — the drafts bypass the per-token grammar mask — nor per-token
/// logprobs, which the draft path never captures. The scheduler and server
/// already select a spec mode only when both are absent, but `nextPld` /
/// `nextDrafter` / `nextMtp` only *documented* that with `std.debug.assert`,
/// which compiles out in the ReleaseFast build users actually run (issue #97).
/// A future dispatch bug that let a constrained request reach a spec step would
/// then read the constrained branch's placeholder token as a real commit and
/// stream silently off-schema output as a normal 200. So the spec steps gate on
/// this real, release-enforced check and return `error.SpecDecodeUnsupported`
/// instead of asserting.
fn specDecodeUnsupported(sampling: SamplingParams, logprobs_n: u32) bool {
    return sampling.constraint != null or logprobs_n != 0;
}

/// Generation result (for non-streaming use).
pub const GenerationResult = struct {
    text: []u8,
    token_ids: []u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    prefill_tps: f64,
    decode_tps: f64,
    /// Wall-clock nanoseconds spent on prefill (prompt processing).
    prefill_ns: u64 = 0,
    /// Wall-clock nanoseconds spent on decode (token generation).
    decode_ns: u64 = 0,
    /// Prompt tokens served from a KV-cache prefix (hot prefix cache for MLX,
    /// persistent-session prefix reuse for llama). `prompt_tokens - cached_tokens`
    /// is what was actually run through the model this turn, so `prefill_tps`
    /// reflects real compute rather than an inflated full-prompt rate.
    cached_tokens: u32 = 0,
    logprobs: ?[]LogprobResult = null, // per-token logprobs (caller must free)
    /// Non-null only when the degenerate-tail guard cut this generation:
    /// the `finish_details.type` value emitted beside `finish_reason`
    /// ("length", which never moves — see `scheduler.loopStopReason`).
    /// Static string; nothing to free.
    finish_details: ?[]const u8 = null,
};

/// Throughput in tokens/sec. Returns 0 when no time elapsed so unmeasured paths
/// report 0 rather than inf / NaN.
pub fn tokensPerSec(tokens: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    const tok_f: f64 = @floatFromInt(tokens);
    const ns_f: f64 = @floatFromInt(elapsed_ns);
    return tok_f * @as(f64, @floatFromInt(std.time.ns_per_s)) / ns_f;
}

/// True prefill compute throughput: divides by the tokens actually pushed through
/// the model (prompt minus the prefix served from KV cache). A near-full cache
/// hit therefore reports the small suffix's real rate, not an inflated
/// full-prompt number. With `cached_tokens == 0` this is just the full-prompt
/// rate, matching the pre-instrumentation behavior.
pub fn prefillTokensPerSec(prompt_tokens: u32, cached_tokens: u32, prefill_ns: u64) f64 {
    const uncached: u32 = if (prompt_tokens > cached_tokens) prompt_tokens - cached_tokens else 0;
    return tokensPerSec(uncached, prefill_ns);
}

/// Pick the end position of the next prefill chunk starting at `pos`.
///
/// Base behavior: advance by `default_chunk` (the memory-bound `PREFILL_CHUNK`),
/// clamped to `prefix_len`. When SSM checkpointing is active, shrink the chunk so
/// it ends exactly on the next `ssm_cp_stride`-aligned ABSOLUTE position — that
/// lays down a stride-aligned SSM snapshot without changing what the model sees
/// (attention is causal; SSM/conv update chunk-locally, so the forward result is
/// identical to an unchunked run).
///
/// Pulled out of the prefill loop so the chunk-count behavior is unit-testable:
/// the stride directly controls how many chunks a cold prefill costs, and on
/// large MoE/hybrid models each extra chunk re-streams the (huge) expert weights
/// from HBM — the dominant cold-prefill cost. A too-small stride therefore
/// silently tanks MoE prefill throughput (~25% on 35B-class models for an
/// 850-token prompt at stride 256). Keeping typical prompts single-chunk is what
/// `ssm_checkpoint_stride`'s default guards.
/// A trailing remainder smaller than this merges into the preceding chunk
/// instead of becoming its own chunk. Chat-templated prompts routinely land a
/// token or two past a chunk multiple (an "8192-token" prompt tokenizes to
/// 8193); a 1-token final chunk pays a full graph build + eval barrier +
/// cache clear for one token. The merged chunk's attention-score transient
/// grows by at most TAIL_MERGE_MAX/default_chunk (~6% at 8192) — within the
/// score-budget slack `boundedPrefillChunk` already carries.
pub const TAIL_MERGE_MAX: usize = 512;

/// PURE: the tail a chunk of `default_chunk` tokens may absorb, WHERE THE
/// WIDTH IS CHOSEN PER CHUNK.
///
/// `TAIL_MERGE_MAX` is a flat 512 justified by "at most ~6% at 8192". Once the
/// width is chosen PER CHUNK that justification stops holding at the bottom of
/// the ladder: at `PREFILL_CHUNK_FLOOR` the same 512 tokens are +100% of the
/// transient the step-down just bought, which is the one thing a step-down
/// exists to prevent. An eighth of the width keeps the original ~6% bound at
/// every rung and is a no-op at 4096 and 8192, where `TAIL_MERGE_MAX` is
/// already the smaller of the two.
///
/// It is NOT the every-arch bound. Reach it through `tailMergeMaxFor`.
pub fn tailMergeMax(default_chunk: usize) usize {
    return @min(TAIL_MERGE_MAX, @max(default_chunk / 8, 1));
}

/// PURE: the tail bound the chunk loop actually uses.
///
/// The scaling above answers a question only the per-chunk adaptive width
/// asks — "this chunk narrowed to protect a transient; do not hand the saving
/// straight back". Every other arch reaches a sub-4096 chunk by a route that
/// never asked it (`resolvePrefillChunk`'s machine ladder puts a 27B on a
/// 16 GB Mac at 512; `boundedPrefillChunk`'s score budget and its
/// composed-causal 2048 cap hit gemma4, qwen3_5/3_6, muse_glimmer,
/// deepseek_v4; any `--prefill-chunk` / `MLX_SERVE_PREFILL_CHUNK` under 4096
/// hits everything). Chunk boundaries are not byte-stable, so an ungated
/// scaling was an unmeasured behaviour change on all of them. Gated: the flat
/// constant everywhere else, byte for byte.
pub fn tailMergeMaxFor(default_chunk: usize, adaptive_width: bool) usize {
    return if (adaptive_width) tailMergeMax(default_chunk) else TAIL_MERGE_MAX;
}

pub fn nextChunkEnd(
    pos: usize,
    prefix_len: usize,
    default_chunk: usize,
    want_ssm_cp: bool,
    ssm_cp_stride: usize,
    ssm_cp_offset: usize,
    // `adaptive_width`: is the per-chunk adaptive width live for THIS prefill
    // (qwen4_exp under `server.adaptivePrefillChunkEnabled`, surfaced to the
    // loop as an installed `chunk_width_hook`)? Only then does the merge bound
    // scale — see `tailMergeMaxFor`.
    adaptive_width: bool,
) usize {
    var end = @min(pos + default_chunk, prefix_len);
    if (want_ssm_cp and ssm_cp_stride > 0) {
        const abs_pos = pos + ssm_cp_offset;
        const abs_end = end + ssm_cp_offset;
        const next_boundary_abs = ((abs_pos / ssm_cp_stride) + 1) * ssm_cp_stride;
        if (next_boundary_abs > abs_pos and next_boundary_abs < abs_end) {
            // A stride boundary lands inside this chunk — end exactly on it
            // (never tail-merge past it; the boundary IS the snapshot point).
            return next_boundary_abs - ssm_cp_offset;
        }
    }
    if (end < prefix_len and prefix_len - end < tailMergeMaxFor(default_chunk, adaptive_width)) {
        // Absorb a tiny tail instead of paying a full graph build + eval
        // barrier for a few tokens. With checkpointing active this can only
        // extend within a boundary-free span (the boundary case returned
        // above), so at most a snapshot < TAIL_MERGE_MAX tokens before the
        // end is skipped — and the always-on end-of-prompt snapshot lands
        // right there anyway.
        end = prefix_len;
    }
    return end;
}

/// Tokens held back from the chunked-prefill loop and forwarded together with
/// the final (logits) forward when SSM checkpointing is active, so the
/// always-on snapshot lands SSM_SNAPSHOT_BACKOFF tokens BEFORE the prompt end.
///
/// A snapshot exactly at the prompt end is unreachable for the next turn's
/// prefix match: the template's generation-prompt suffix
/// ("<|im_start|>assistant\n" + think opener) renders differently once the
/// turn enters history, so the match always falls a few tokens short of the
/// full prompt — "[hot-cache] hybrid miss (no checkpoint ≤ 870 of 897)" was
/// llmprobe's prompt-cache-prefix cell failing (the 2026-06-10 class; fine
/// strides used to mask it by laying boundaries underneath). 30 covers every
/// template suffix in the fleet with margin (ChatML+think ≈ 7 tokens, laguna
/// pre-opened think ≈ 10) while keeping the tail forward UNDER the
/// prefill-eval-cadence threshold (seq >= 32): a 65-token tail was treated as
/// a prefill and paid ~450ms of mid-loop eval bubbles on the 27B, where the
/// 31-token tail rides the verify-shaped fast path. Cold cost is then ~zero
/// (the final forward pays its full weight sweep for ONE token anyway).
/// Warm restores re-forward ≤ backoff+1 tokens.
pub const SSM_SNAPSHOT_BACKOFF: usize = 30;

/// How many trailing prompt tokens the final (logits) forward covers: the
/// held-back snapshot window plus the last token itself. Pure so the
/// backoff/loop-bound interaction is unit-testable.
pub fn ssmSnapshotBackoff(want_ssm_cp: bool, prefix_len: usize) usize {
    if (!want_ssm_cp) return 0;
    if (prefix_len <= SSM_SNAPSHOT_BACKOFF) return 0;
    return SSM_SNAPSHOT_BACKOFF;
}

/// Effective SSM-checkpoint stride for a model, given the base (configured)
/// stride: checkpointing never sub-divides the prefill chunk, on ANY arch.
///
/// The old policy kept the fine base stride on dense (non-MoE) hybrids on the
/// theory that their prefill is compute-bound so extra chunks are ~free. That
/// was true before `prefillDqGemm`: since the dq route only engages at
/// M >= 2048, a fine stride pushes EVERY projection of EVERY chunk onto the
/// slow small-M qmm path, and the per-chunk fixed costs (graph build, eval
/// barrier, MTP history capture) multiply on top. Measured on Qwen3.6-27B
/// dense GDN (M4 Max, 2026-07-30): stride 256 chunked an 8K prompt into 33
/// pieces at 211 tok/s vs 254 at coarse chunks — a 17-20% cold-prefill tax at
/// every context length, which is how llm_context_benchmarks read us 19%
/// behind oMLX. MoE pays even more (expert re-streaming, ~25%).
///
/// Warm mid-prompt reuse granularity drops to the chunk size, but the
/// always-on end-of-prompt snapshot still covers the dominant append-growth
/// multi-turn case (llmprobe's cache-hit tests restore from it at any
/// stride). `base == 0` (checkpointing disabled) is preserved, and a larger
/// explicit stride is never shrunk.
pub fn effectiveSsmCheckpointStride(base: usize, prefill_chunk: usize) usize {
    if (base == 0) return 0;
    return @max(base, prefill_chunk);
}

/// PURE: tokens of KV capacity this prefill reserves UP FRONT, and the ONE
/// place the reservation is gated by architecture.
///
/// The reservation (#353) removes the grow transient a >32k prefill pays, at
/// the price of allocating the generation headroom before a single token of it
/// exists. That trade was measured on qwen4_exp at 100k-1M; on every other
/// arch it is an unmeasured behaviour change — a request that fits today can
/// be refused tomorrow because the guard now bills headroom the prompt may
/// never use. So the reservation is asked for by `ModelConfig.longCtxGated()`
/// and every other arch keeps the pre-#353 proportional growth
/// (`KVCache.nextCapacityPolicy`) BYTE for byte: `reserve_tokens` stays 0, and
/// `nextCapacityReserved` then IS `nextCapacity`.
///
/// `MLX_SERVE_KV_RESERVE=0` still turns it off inside the gate, for a
/// same-boot A/B on the arch that has it.
///
/// The admission side of the same gate is `server.prefillRequestTerms`, which
/// zeroes its per-request terms on the same predicate — the guard and the
/// allocator must agree about whether a reservation exists at all, not only
/// about its size.
pub fn reservedPrefillTokens(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u64,
    chunk: u64,
) u64 {
    if (!config.longCtxGated()) return 0;
    return transformer_mod.KVCache.reservedTokens(
        seq,
        max_tokens,
        chunk,
        config.max_position_embeddings,
    );
}

/// SSM checkpoints exist to feed prefix-cache reuse, and image-bearing
/// prompts are excluded from prefix reuse (equal placeholder IDs do not imply
/// equal images) — so vision prefills skip checkpointing even now that they
/// chunk (the splice scatter itself is chunk-safe via vision_splice_offset).
/// A vision prompt checkpoints like text once it chunks like text (#197):
/// without a checkpoint a hybrid's image turn is a guaranteed hot-cache miss.
pub fn shouldCheckpointSsmPrefill(stride: u32, has_ssm: bool, has_vision: bool) bool {
    return stride > 0 and has_ssm and (!has_vision or visionChunkedPrefillEnabled());
}

/// Chunked vision prefill (issue #197). Default ON: the splice resumes its
/// row index across chunk boundaries via `ForwardCtx.vision_splice_offset`,
/// so an image-bearing prompt prefills under the same chunk-bounded memory
/// envelope as text. MLX_SERVE_VISION_CHUNKED=0 restores the whole-prompt
/// single forward (and the memory guard's full-width bill with it).
var vision_chunked_cached: ?bool = null;
pub fn visionChunkedPrefillEnabled() bool {
    if (vision_chunked_cached) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_VISION_CHUNKED");
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    vision_chunked_cached = on;
    return on;
}

/// What the memory guard should bill a prefill at: full width only for a
/// vision prompt with chunking killed. The guard and this loop must agree or
/// admission either over-refuses (bills seq for a chunked prefill) or
/// under-bills (uncatchable Metal OOM).
pub fn visionPrefillUnchunked(has_vision: bool) bool {
    return has_vision and !visionChunkedPrefillEnabled();
}

/// Placeholder tokens (vision + audio soft tokens) in `ids` — the number of
/// source rows a chunk's splice consumes. Host-side count, no GPU sync.
pub fn countSpliceRows(ids: []const i32, image_token_id: u32, audio_token_id: u32, video_token_id: u32) usize {
    var n: usize = 0;
    for (ids) |id| {
        if (id == @as(i32, @intCast(image_token_id)) or
            (audio_token_id > 0 and id == @as(i32, @intCast(audio_token_id))) or
            (video_token_id > 0 and id == @as(i32, @intCast(video_token_id)))) n += 1;
    }
    return n;
}

/// Number of chunks a cold prefill of `prefix_len` tokens splits into for the
/// given chunk size / SSM-checkpoint stride. Mirrors the loop in `init` exactly
/// (drives the same `nextChunkEnd`), so a test on this is a faithful proxy for
/// the real prefill chunk count. Each chunk on a memory-bound MoE prefill
/// re-streams the expert weights, so this is effectively the cold-prefill
/// weight-traffic multiplier.
pub fn prefillChunkCount(
    prefix_len: usize,
    default_chunk: usize,
    want_ssm_cp: bool,
    ssm_cp_stride: usize,
    ssm_cp_offset: usize,
    adaptive_width: bool,
) usize {
    var pos: usize = 0;
    var n: usize = 0;
    while (pos < prefix_len) {
        const end = nextChunkEnd(pos, prefix_len, default_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset, adaptive_width);
        pos = end;
        n += 1;
    }
    return n;
}

/// Generated tokens between two returns of the decode loop's transients to
/// MLX's allocator. 256 is what the non-speculative paths have always used.
pub const CACHE_CLEAR_INTERVAL: u32 = 256;

/// PURE: has `interval` tokens passed since the last clear?
///
/// Interval arithmetic, not `step % interval == 0`: a spec-decode round emits
/// `1 + accepted` tokens, so a modulo test can step clean over every multiple
/// and never fire (issue #110 — at stride 5 it fires zero times in the first
/// 1024 steps). For the stride-1 paths this is byte-identical to the modulo
/// form it replaces.
pub fn shouldClearAllocatorCache(step: u32, last_clear: u32, interval: u32) bool {
    if (interval == 0) return false;
    return step -| last_clear >= interval;
}

/// Number of accepted draft tokens that may accompany the always-committed
/// anchor token without crossing the request's output-token ceiling.
pub fn capAcceptedForTokenBudget(accepted: u32, completion: u32, max_tokens: u32) u32 {
    const remaining = max_tokens -| completion;
    if (remaining <= 1) return 0;
    return @min(accepted, remaining - 1);
}

test "capAcceptedForTokenBudget keeps speculative commits inside max_tokens" {
    try std.testing.expectEqual(@as(u32, 15), capAcceptedForTokenBudget(15, 0, 400));
    try std.testing.expectEqual(@as(u32, 2), capAcceptedForTokenBudget(15, 397, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 399, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 400, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 401, 400));
}

test "every speculative decoder caps accepted drafts before commit" {
    // Class guard: every speculative path commits an always-emitted anchor
    // plus zero or more accepted drafts. A new/forgotten arm must cap that
    // accepted prefix before it mutates generated ids, KV/module state, or
    // usage. Output-only clipping in the scheduler is already too late.
    const source = @embedFile("generate.zig");
    const names = [_][]const u8{
        "nextDspark",
        "nextPld",
        "nextDrafter",
        "nextDflash",
        "nextMtp",
    };
    for (names) |name| {
        const signature = try std.fmt.allocPrint(testing.allocator, "    pub fn {s}", .{name});
        defer testing.allocator.free(signature);
        const start = std.mem.indexOf(u8, source, signature) orelse return error.MissingSpecDecoder;
        const end = std.mem.indexOfPos(u8, source, start + signature.len, "\n    pub ") orelse source.len;
        if (std.mem.indexOf(u8, source[start..end], "capAcceptedForTokenBudget(") == null) {
            std.debug.print("{s} does not cap accepted drafts before commit\n", .{name});
            return error.UncappedSpecDecoder;
        }
    }
}

/// Step-based generator. Call `init` to prefill, then `next` per token.
/// Uses a fully-lazy async pipeline matching mlx-lm: sample + next forward are
/// built as a single lazy computation graph, async_eval'd together. The GPU
/// never idles between token generation steps.
pub const Generator = struct {
    xfm: *Transformer,
    /// Forward-pass context. Stores per-request KVCache pointer, moe_seq_offset
    /// pointer, ssm_entries slice, vision_embeddings handle, and capture_hidden
    /// override. The legacy single-slot path uses `xfm.defaultCtx()` (pointing at
    /// the Transformer's own fields). Phase 2 concurrent batching constructs a
    /// per-slot ForwardCtx pointing at the slot's own KVCache, etc., so multiple
    /// generators can share one Transformer's weights without colliding on
    /// per-request state. Stored by value; `&self.ctx` is what we pass to
    /// `xfm.forwardWith` / `lazyForward` / drafter step.
    ctx: ForwardCtx,
    tok: *const Tokenizer,
    next_token_id: u32,
    step: u32,
    /// `step` at the last `mlx_clear_cache()`. Advanced only by `advanceStep`,
    /// which is the ONE place `step` may move.
    last_cache_clear_step: u32 = 0,
    max_tokens: u32,
    sampling: SamplingParams,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    done: bool,
    eos_token_ids: []const u32,
    generated_ids: std.ArrayList(u32),
    consecutive_pad: u32 = 0, // count of consecutive token-0 (pad) generations
    timeout_ns: u64, // 0 = no timeout; measures SILENCE, not total time (see StallClock)
    stall: StallClock = .{},
    timer: io_util.Stopwatch,
    logprobs_n: u32 = 0, // 0 = disabled, >0 = number of top_logprobs to return
    last_logprob: ?LogprobResult = null, // logprob result for the most recently returned token
    /// Logprobs for the token the NEXT `next()` call will return.
    ///
    /// The decode loop returns `next_token_id` and, in the same call, forwards
    /// it to sample its successor — so the result `sampleToken` hands back
    /// describes the token AFTER the one being returned. Publishing it as
    /// `last_logprob` there shifted the whole array by one: the caller zips
    /// token_ids with logprobs by index, so at temp 0 every entry reported the
    /// distribution of the token that FOLLOWED it (a one-token "OK" reply came
    /// back with `<|role_end|>` at rank 1). Seeded at init with t1's own
    /// distribution — t1 is sampled from the prefill's final forward, which
    /// this loop never sees.
    pending_logprob: ?LogprobResult = null,
    // Async pipeline state: pre-computed forward pass logits for next decode step
    pending_logits: mlx.mlx_array = .{},
    has_pending_logits: bool = false,
    // Deferred token: lazy array from async pipeline, eval'd at start of next iteration
    pending_token: mlx.mlx_array = .{},
    has_pending_token: bool = false,

    // ── Spec-decode shared state (PLD + drafter) ──
    // Post-final-norm hidden state at the last produced token's position.
    // Owned by the Generator (freed in `deinit`). Captured by
    // `forwardCaptureHidden` during prefill final-token forward and every
    // verify forward — used by drafter as h_prev seed and by PLD verify
    // partial-accept rollback.
    last_hidden: mlx.mlx_array = .{},
    has_last_hidden: bool = false,
    /// PRNG for PLD / drafter stochastic-verify accept test (probability-
    /// ratio requires a uniform draw per draft step). Seeded from
    /// `sampling.seed` when set, otherwise from system time at init.
    prng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),

    // ── PLD (Prompt Lookup Decoding) state ──
    // Owned copy of the input prompt ids — needed because PLD's n-gram lookup
    // table is `prompt + generated_ids`, and the caller-supplied `prompt_ids`
    // slice is freed after `init` returns. `generated_ids` (above) tracks
    // post-prefill tokens; `prompt_ids_owned` is the immutable prefix.
    prompt_ids_owned: []u32 = &.{},
    /// Allocator that owns `prompt_ids_owned`. Stored so `deinit` can free it
    /// without requiring callers to thread the allocator a second time. (Other
    /// owned slices are freed via the `allocator` argument to `deinit` for
    /// historical reasons; this one is set during `initWithOptions`.)
    prompt_ids_alloc: ?std.mem.Allocator = null,
    /// Did init actually arm PLD for this generator (`InitOptions.pld_enabled`
    /// AFTER the deepseek_v4 chokepoint guard)? `nextPld` declines to the
    /// plain serial step when false, and the scheduler's tick dispatch
    /// (`specTickMode`) requires it alongside `slot.enable_pld` — the caller's
    /// flag alone must never put a verify forward through the trunk. This is
    /// PLD's counterpart of the `gen.mtp != null` / `gen.drafter != null`
    /// conjuncts; PLD has no model handle, so the bit has to be explicit.
    pld_enabled: bool = false,
    /// Stats for PLD benchmark logging. `pld_attempted` counts every step
    /// where lookup found a candidate (so a verify forward ran);
    /// `pld_accepted_tokens` is the cumulative number of *drafted* tokens
    /// (not including the always-accepted t1) that were successfully verified.
    pld_attempted: u64 = 0,
    pld_accepted_tokens: u64 = 0,
    /// DeepSeek-V4 DSpark: init armed the native block-parallel draft mode
    /// (dsv4 checkpoint shipping mtp.* stages + a clean request).
    /// Mutually exclusive with pld/drafter/mtp by the chokepoint's
    /// construction; `nextDspark` declines to the serial step when false.
    dspark_enabled: bool = false,
    /// Sampled-request acceptance (the MTP one-hot Leviathan rule over
    /// filtered target probs) instead of raw argmax equality. Set by the
    /// chokepoint when the request samples (temp ≥ 0.01, top_k ≠ 1) and the
    /// stochastic arm isn't env-killed; meaningless unless `dspark_enabled`.
    dspark_stochastic: bool = false,
    dspark_attempted: u64 = 0,
    dspark_accepted_tokens: u64 = 0,

    // ── Gemma 4 assistant drafter state ──
    // External drafter model (cross-attends into target's KV). When
    // `drafter != null`, callers use `nextDrafter` instead of `next`. The
    // drafter is owned by the server (loaded once at startup); the Generator
    // only holds a non-owning pointer.
    drafter: ?*DrafterModel = null,
    /// Number of tokens proposed per round (= drafter forwards + 1 verify token).
    /// Defaults to 4 (3 drafter steps + 1 t1 prepend → length-4 verify).
    drafter_block_size: u32 = 4,
    /// Stats: count of nextDrafter calls that ran a verify forward.
    drafter_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    drafter_accepted_tokens: u64 = 0,

    // ── DFlash block-drafter state ──
    // External block-parallel assistant (src/dflash.zig). When
    // `dflash != null`, callers use `nextDflash` instead of `next`. Weights
    // are owned by the server (per-model); the per-request context cache
    // (`dflash_ctx`) is OWNED by the Generator — built during prefill from
    // the trunk's capture_layers hiddens, freed in `deinit`.
    dflash: ?*DflashModel = null,
    dflash_ctx: ?dflash_mod.DflashCtx = null,
    /// Effective block size (assistant config, clamped by --draft-block-size).
    /// Drafts per round = dflash_block_size - 1.
    dflash_block_size: u32 = 0,
    /// Request-class break-even yield, normalized by the scheduler to the
    /// effective draft width. Thinking requests use the lower calibration
    /// because their less-predictable reasoning preamble pays back later.
    dflash_min_accepted_per_round: f32 = DFLASH_GATE_MIN_ACCEPTED_PER_ROUND,
    /// Per-round width chooser over the model's round-cost table (null =
    /// MLX_SERVE_DFLASH_CHOOSER=0, the fixed block + sticky yield gate).
    dflash_chooser: ?round_cost.WidthChooser = null,
    /// Drafts this round ran (0 = serial), read by the table feed.
    dflash_round_width: u32 = 0,
    /// Stats: count of nextDflash calls that ran a verify forward.
    dflash_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    dflash_accepted_tokens: u64 = 0,
    /// Per-phase wall-time trace (MLX_SERVE_DFLASH_TRACE=1; else untouched).
    /// Unlike the MTP trace this one INSERTS eval barriers to attribute a
    /// fully-lazy round — a traced round is slower than a real one by
    /// whatever overlap the barriers destroy. Diagnostic only.
    dflash_trace: DflashTrace = .{},
    /// Trace-only stopwatch across the scheduler gap (round return → next
    /// round entry); bills the `.gap` phase. Null when not tracing.
    dflash_gap_watch: ?io_util.Stopwatch = null,

    // ── Qwen native MTP head state ──
    // The model's own one-layer multi-token-prediction head (src/mtp.zig).
    // When `mtp != null`, callers use `nextMtp` instead of `next`. The head
    // is owned by the server (loaded with the model); the Generator only
    // holds a non-owning pointer. `mtp_cache` is the head's committed-history
    // KV cache — OWNED by the Generator (built during prefill, freed in
    // `deinit`).
    mtp: ?MtpHeadRef = null,
    /// Does the MODEL have a usable MTP head — i.e. did the registry give
    /// this slot one? `--no-mtp` clears it; a per-REQUEST `enable_mtp:false`
    /// does not, which is deliberate (see `serialCellWanted`).
    model_has_mtp: bool = false,
    mtp_cache: ?MtpCacheRef = null,
    /// Absolute target position represented by MTP-cache position 0. Usually
    /// zero; nonzero when the head keeps only a suffix of a restored/long
    /// prompt. Used to map the sidecar's relative offsets into Qwen M-RoPE.
    mtp_position_base: usize = 0,
    /// CONFIGURED max tokens drafted per round (verify length = depth + 1).
    mtp_depth: u32 = mtp_mod.DEFAULT_DEPTH,
    /// The cap WITHOUT the per-silicon row: an explicit --mtp-depth, else
    /// the adaptive default. The row is the cold-start cap; once the table
    /// has trusted widths it may plan up to this instead (the M4 base row
    /// of 4 measured -6% against what the table found at 6).
    mtp_depth_free: u32 = mtp_mod.DEFAULT_DEPTH,
    /// CURRENT adaptive depth (see updateMtpDepth). Starts at `mtp_depth`,
    /// demoted/promoted per windowed acceptance, never exceeds `mtp_depth`.
    mtp_depth_current: u32 = mtp_mod.DEFAULT_DEPTH,
    /// Stats: count of nextMtp calls that ran a verify forward.
    mtp_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    mtp_accepted_tokens: u64 = 0,
    /// Adaptive-depth moving window: per-round drafted/accepted counts.
    mtp_window_drafted: [MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_accepted: [MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_idx: u32 = 0,
    mtp_rounds_since_switch: u32 = 0,
    /// Rounds remaining during which promotion is blocked (set after a
    /// demotion so a failed depth excursion isn't immediately retried).
    mtp_promote_cooldown: u32 = 0,
    /// Cumulative drafted tokens across rounds. The EV controller varies m
    /// per round, so `attempts x depth` no longer measures proposals — this
    /// is the honest per_draft_pct denominator.
    mtp_drafted_tokens: u64 = 0,
    /// Rounds where the confidence gate extended into chunk B.
    mtp_ext_rounds: u64 = 0,
    /// Extension dry-spell gate: consecutive extension-CONSIDERED rounds
    /// whose confidence gate did not clear, and the single-chunk cooldown
    /// that a full dry streak triggers (see mtpExtDryAllows).
    mtp_ext_dry_streak: u32 = 0,
    mtp_ext_cooldown: u32 = 0,
    /// EV controller: conditional acceptance EMA per draft index,
    /// a[i] = P(draft i accepted | drafts 0..i-1 accepted). Optimistic prior;
    /// warmup rounds pull the low indices to reality before it can matter.
    mtp_ev_accept: [mtp_mod.MAX_DEPTH]f32 = @splat(MTP_EV_PRIOR),
    /// Rounds seen by the EV controller (drives the legacy-behavior warmup).
    mtp_ev_rounds: u32 = 0,
    /// Last round's planned m_lo (base-depth climb damping: +1/round max).
    mtp_ev_m_lo_prev: u32 = 1,
    /// Round-cost surface selected once for this target+MTP head. The M5/G17
    /// NAX surface requires the measured trunk, native sidecar, and 3-bit
    /// draft-only head; every other combination keeps the M1-M4 surface.
    mtp_ev_costs: MtpEvCosts = MTP_EV_DEFAULT_COSTS,
    /// Live-cost EMAs (ms), 0 = unseeded. `mtp_ev_sync_ms` is the measured
    /// chunk-A confidence-read sync — updated ONLY on extension-considered
    /// rounds, so it holds the true cost of a sync when one happens (not the
    /// suppressed-round average). `mtp_ev_round_ms` is the realized round
    /// wall-clock. Their ratio drives the cost-aware exploration throttle
    /// (mtpExtDryThresholdFor) so a dry sync tax self-limits to a small
    /// fraction of the round. Always-on; the trace is not required.
    mtp_ev_sync_ms: f32 = 0,
    mtp_ev_round_ms: f32 = 0,
    mtp_regime: MtpRegime = .{},
    mtp_regime_verdict: ?bool = null,
    /// Width-trial schedule for the round-cost table (one 2-round block per
    /// period at m_lo+1, so the table learns the next cliff once).
    mtp_width_trial: MtpWidthTrial = .{},
    /// Consecutive plans with the same m_lo (the base has stopped climbing).
    mtp_m_lo_streak: u32 = 0,
    /// Drafts of the previous TWO rounds (0 = serial), every round: a round
    /// is a transition unless both predecessors ran its width (the first
    /// round after a width change is the transition, the second is still
    /// elevated — live, every w4 sample taken beside a trial read 17% high
    /// on the M4 base and fed the churn that produced it).
    spec_round_prev_width: ?u32 = null,
    spec_round_prev_width2: ?u32 = null,
    /// Shape of the previous two MTP rounds: a shape change is a transition too.
    spec_round_prev_two_chunk: bool = false,
    spec_round_prev_two_chunk2: bool = false,
    /// Wall clock between round ends: the regime compares the quantity
    /// tok/s reports, so per-round work OUTSIDE the round stopwatch (token
    /// publish, stop checks, scheduler) must be in the denominator — fewer
    /// rounds is part of a wider shape's win.
    mtp_regime_clock: ?io_util.Stopwatch = null,
    /// True while this generator is the ONLY decoding stream. Contention is
    /// strictly one-sided — it only ADDS time — so a busy server simply
    /// stops feeding the kv-term learner rather than teaching it a lie.
    /// Set per tick by the scheduler.
    spec_cost_solo: bool = true,
    /// Per-phase wall-time trace (MLX_SERVE_MTP_TRACE=1; else untouched).
    mtp_trace: MtpTrace = .{},
    /// Trace-only: stopwatch running across the scheduler gap (round return
    /// → next round entry); bills the `.gap` phase. Null when not tracing.
    mtp_gap_watch: ?io_util.Stopwatch = null,
    /// Deferred committed-history append: the round's (tokens, true verify
    /// hiddens) pair, folded into the NEXT round's first draft step as one
    /// multi-row head forward instead of a separate appendHistory forward.
    /// Rounds with no successor (EOS/length/runtime disable) never pay for
    /// the append; the stash is freed unconsumed in `deinit`.
    mtp_hist_stash: ?MtpHistStash = null,
    /// Cross-round pre-draft (round pipelining): the NEXT round's chunk-A
    /// draft chain, built and async-dispatched at the CURRENT round's tail
    /// so the head chain runs on the GPU while the CPU emits tokens. The
    /// build consumes `mtp_hist_stash`, so the two are mutually exclusive
    /// (asserted at consume). Freed unconsumed in `deinit`.
    mtp_pre_draft: ?MtpPreDraft = null,

    // ── Phase 1: SSM checkpoints captured during prefill ──
    /// Owned SSM-state snapshots taken at stride-aligned positions during
    /// chunked prefill. Drained by the scheduler in `commitSlotIfApplicable`
    /// via `takeSsmCheckpoints()`. Empty on non-hybrid models or when
    /// `ssm_checkpoint_stride == 0`. Allocator: the Generator's `allocator`
    /// (passed to `initWithOptions`); the same allocator must be passed to
    /// `deinit` for any checkpoint that wasn't taken.
    ssm_checkpoints: std.ArrayList(SSMCheckpoint) = std.ArrayList(SSMCheckpoint).empty,
    /// Allocator used for `ssm_checkpoints` storage and each checkpoint's
    /// per-layer slice. Set during `initWithOptions`. We track it separately
    /// from the `allocator` argument to `deinit` because `takeSsmCheckpoints`
    /// transfers ownership: the consumer (HotPrefixCache) must use the SAME
    /// allocator to free, since the layer-slice backing memory was allocated
    /// here.
    ssm_checkpoint_alloc: ?std.mem.Allocator = null,

    // ── Runtime acceptance gate ──
    // Set to true mid-request when the per-request acceptance rate
    // (`*_accepted_tokens / *_attempted`) falls below
    // `RUNTIME_GATE_MIN_RATE` after `RUNTIME_GATE_WARMUP` attempts. When set,
    // `nextPld`, `nextDrafter`, and `nextDflash` short-circuit to `next()` for the
    // remainder of the request — the prompt-time gate could not foresee that
    // the workload's *runtime* draft acceptance rate wasn't paying for the
    // per-step verify overhead. The flag is sticky for the rest of the
    // generation; we never re-enable speculation within a single request.
    spec_disabled_runtime: bool = false,
    /// WHY `spec_disabled_runtime` was set, for `[spec-stats]`. A reader
    /// cannot tell "acceptance did not pay" from "the operator's context
    /// ceiling" out of a bare `runtime_disabled=true`, and those two call for
    /// opposite follow-ups.
    spec_disable_reason: SpecDisableReason = .none,
    /// Adaptive serial controller (`MtpAdaptive`): which arm this request is
    /// on, the confirm streak behind it and the KV bucket it was decided in.
    mtp_adaptive: MtpAdaptive = .{},
    /// S21. This request has RELEASED the module-owned MTP head: it armed the
    /// sticky serial arm (so no re-entry, no probe, no round can follow) and
    /// has since reached a serial block boundary with the head detached. From
    /// that point `scheduler.slotExclusiveDecode` stops claiming the model for
    /// this slot, so another request may take the head and the rest of this
    /// one batches like any plain decode.
    ///
    /// Per-REQUEST, deliberately: on the Generator, never on `LoadedModel` or
    /// the Transformer. One slow request must not turn MTP off process-wide.
    mtp_head_released: bool = false,
    /// This request's realized MTP price (ms per emitted token) over a
    /// trailing window — the second of the two measured prices the switch
    /// requires. Zeroed for every new request, hence empty after a restore.
    mtp_price: MtpPriceWindow = .{},
    /// Ticks left in a BOUNDED serial block (the serial probe). Deliberately
    /// not `spec_disabled_runtime`: the probe is internal to `nextMtp`, so
    /// the scheduler keeps dispatching the slot as an MTP slot and the
    /// request comes back on its own.
    mtp_serial_left: u32 = 0,
    /// Where a serial block sits in its ramp back to `nextMtp`'s entry
    /// invariant. `.none` while the block is decoding.
    mtp_serial_exit: MtpSerialExit = .none,
    /// Ticks of the current serial block already spent warming: their wall
    /// time is the previous speculative round's tail, not a serial token's.
    mtp_serial_warm: u32 = 0,
    /// One line per request, not one per tick: `mtpSerialGiveUp` is a
    /// should-never-happen recovery and a tick loop would flood the log.
    mtp_serial_giveup_logged: bool = false,
    /// Inter-tick wall clock feeding the table's serial cell (the twin of
    /// `mtp_regime_clock`). Null = the next read SEEDS instead of measuring.
    mtp_serial_clock: ?io_util.Stopwatch = null,
    /// Yield-gate counters: enabled-mode `nextPld` steps and drafted tokens
    /// accepted since the last (re-)enable. Reset on mid-request re-enable so
    /// a fresh workload region (e.g. file echo after a novel preamble) gets a
    /// fresh economic evaluation instead of inheriting the bad early yield.
    yield_steps: u64 = 0,
    yield_accepted: u64 = 0,
    /// Steps spent in disabled mode since the gate tripped (drives the
    /// periodic `specShouldReenable` re-check).
    disabled_steps: u64 = 0,
    /// Number of attempts before the runtime gate considers disabling.
    /// Below this we trust the prompt-time gate.
    ///
    /// Override at runtime via `SPEC_GATE_WARMUP` env var (parsed in `runtimeGateWarmup()`
    /// once per request). Lower values make the gate trip sooner,
    /// reducing regression-tail damage at the cost of fewer chances for slow-warmup
    /// workloads to amortize spec overhead.
    pub const RUNTIME_GATE_WARMUP: u64 = 5;

    /// Read the warmup threshold for this call. Env-overridable so we can A/B
    /// without rebuilding. Anything outside `[1, 64]` falls back to the default.
    pub fn runtimeGateWarmup() u64 {
        const n = readEnvUsize("SPEC_GATE_WARMUP", @intCast(RUNTIME_GATE_WARMUP));
        if (n < 1 or n > 64) return RUNTIME_GATE_WARMUP;
        return @intCast(n);
    }
    /// Minimum per-draft acceptance probability. Below this after warmup,
    /// speculation is disabled for the rest of the request.
    ///
    /// History: pre-v5 this gate compared `accepted/attempted` (per-round
    /// average) against 0.30 — but with `block_size=4` the max value of that
    /// ratio is 3.0, so the 0.30 threshold corresponded to ~10% per-draft
    /// probability, well below where verify+draft overhead actually breaks
    /// even. Empirically creative-content workloads regress at 22-47% per-draft
    /// acceptance
    /// while the gate stayed off (per-round avg 0.66-1.58, all above 0.30).
    /// Switching to a per-draft probability with threshold 0.50 cleanly cuts
    /// off the regressing tail while leaving heavy-echo workloads (84-97%
    /// per-draft) running unmolested.
    pub const RUNTIME_GATE_MIN_PER_DRAFT_RATE: f32 = 0.50;

    /// Pure helper: should the runtime gate disable speculation given the
    /// observed per-request stats? `drafts_per_round` is the number of
    /// drafted tokens proposed in each verify (= `block_size - 1` for the
    /// drafter, or `pld_draft_len` for PLD); we divide accepts by attempts ×
    /// drafts_per_round to get the per-draft acceptance probability.
    /// Returns true iff `attempted >= warmup` AND per-draft probability is
    /// below `RUNTIME_GATE_MIN_PER_DRAFT_RATE`.
    ///
    /// `drafts_per_round == 0` is treated as "no speculative work happens
    /// per round" → never trip (defensive — current callers always pass
    /// >= 1).
    pub fn runtimeGateShouldDisable(attempted: u64, accepted: u64, drafts_per_round: u32) bool {
        if (attempted < runtimeGateWarmup()) return false;
        if (drafts_per_round == 0) return false;
        const drafts_proposed = attempted * @as(u64, drafts_per_round);
        const rate = @as(f32, @floatFromInt(accepted)) /
            @as(f32, @floatFromInt(drafts_proposed));
        return rate < RUNTIME_GATE_MIN_PER_DRAFT_RATE;
    }

    // DFlash has different economics from sequential-token drafters: one
    // assistant call proposes the whole block and the M5 NAX verify lane makes
    // wide blocks cheap. Per-draft percentage therefore penalizes block 16
    // even when it is decisively profitable. The stable cross-width signal is
    // accepted drafts per verify round. M5 Muse sweeps put code/tool traffic at
    // 4.4-15.0 accepted/round and the regressing prose/vision class at
    // 1.0-1.5; two is the measured break-even boundary. M4 block-5 model-card
    // workloads accepting 62-86% remain above it as well.
    pub const DFLASH_GATE_WARMUP: u64 = 3;
    pub const DFLASH_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 2.0;
    pub const DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.0;
    /// Absolute floor for a SPARSE target, applied after the width scaling.
    /// See `scheduler.dflashGateMinimum` for the measurement it comes from —
    /// a MoE verify reads every expert its block's positions route to, so its
    /// break-even acceptance is several times a dense trunk's.
    pub const DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.8;

    /// Width chooser — OPT-IN (MLX_SERVE_DFLASH_CHOOSER=1) until measured on
    /// the five peer cells; default is the fixed block + sticky yield gate.
    var dflash_chooser_cache: ?bool = null;
    fn dflashChooserEnabled() bool {
        if (dflash_chooser_cache) |v| return v;
        const raw = std.c.getenv("MLX_SERVE_DFLASH_CHOOSER");
        const on = raw != null and std.mem.eql(u8, std.mem.span(raw.?), "1");
        dflash_chooser_cache = on;
        return on;
    }

    pub fn dflashGateWarmup() u64 {
        const n = readEnvUsize("DFLASH_GATE_WARMUP", @intCast(DFLASH_GATE_WARMUP));
        if (n < 1 or n > 64) return DFLASH_GATE_WARMUP;
        return @intCast(n);
    }

    pub fn dflashGateShouldDisable(attempted: u64, accepted: u64, min_avg: f32) bool {
        if (attempted < dflashGateWarmup()) return false;
        const avg = @as(f32, @floatFromInt(accepted)) /
            @as(f32, @floatFromInt(attempted));
        return avg < min_avg;
    }

    // ── PLD yield gate (cold-path economics) ──
    // The per-draft gate above only counts verify ROUNDS, so a workload where
    // the n-gram lookup rarely matches never accumulates enough "attempts" to
    // trip it — yet every no-match step pays PLD's unpipelined cold forward
    // (measured −14% vs the async-pipelined `next()` on creative content).
    // The yield gate instead counts EVERY enabled-mode nextPld step: if the
    // speculation is yielding fewer than YIELD_GATE_MIN_YIELD extra (drafted,
    // accepted) tokens per step after YIELD_GATE_WARMUP steps, the cold-path
    // tax outweighs the wins → disable. Paired with `specShouldReenable`,
    // which flips PLD back on when the generated tail turns repetitive.
    // Warmup 32 (not higher): the re-enable check bounds the cost of a
    // premature trip to ≤SPEC_REENABLE_INTERVAL pipelined-fallback steps,
    // so we can gate early and recover the pipeline sooner on novel content.
    /// Steps of enabled-mode PLD before the yield gate may trip. See the
    /// `yield-gate warmup is 8` test for the sweep this was picked from and
    /// why it moved from 32.
    pub const YIELD_GATE_WARMUP: u64 = 8;
    pub const YIELD_GATE_MIN_YIELD: f32 = 0.25;

    /// Read the yield-gate warmup for this call. Env-overridable
    /// (`SPEC_YIELD_WARMUP`) so the threshold can be swept without a rebuild,
    /// exactly like `runtimeGateWarmup`. Outside `[1, 256]` falls back.
    ///
    /// This number is pure economics, and the economics moved: every warmup
    /// step pays PLD's UNPIPELINED cold forward (plus a synchronous host read
    /// of the sampled token) against `next()`'s async-pipelined step, so the
    /// tax is a fraction of the AR step cost — and the AR step got ~3x cheaper
    /// when the Laguna mscale promotion was fixed, which makes the same
    /// absolute tax a ~3x larger share. 32 was calibrated before that.
    pub fn yieldGateWarmup() u64 {
        const n = readEnvUsize("SPEC_YIELD_WARMUP", @intCast(YIELD_GATE_WARMUP));
        if (n < 1 or n > 256) return YIELD_GATE_WARMUP;
        return @intCast(n);
    }

    pub fn yieldGateShouldDisable(steps_total: u64, accepted: u64) bool {
        if (steps_total < yieldGateWarmup()) return false;
        const yield_rate = @as(f32, @floatFromInt(accepted)) /
            @as(f32, @floatFromInt(steps_total));
        return yield_rate < YIELD_GATE_MIN_YIELD;
    }

    // ── Mid-request spec re-enable ──
    // While the yield gate has PLD disabled, the COMMITTED sequence (prompt +
    // generated) is re-scored every SPEC_REENABLE_INTERVAL steps: what
    // fraction of the recent generated positions would have had a PLD lookup
    // hit (their key-gram appears earlier in committed)? This catches the
    // echo workload where the model repeats PROMPT content (file edits, tool
    // results) — self-repetition scoring misses it because the echoed tail
    // never repeats itself. Above the threshold, PLD is worth re-engaging at
    // the cost of one pipeline drain.
    pub const SPEC_REENABLE_INTERVAL: u64 = 32;
    pub const SPEC_REENABLE_WINDOW: usize = 32;
    pub const SPEC_REENABLE_MIN_FRACTION: f32 = 0.25;
    pub const SPEC_REENABLE_MIN_TOKENS: usize = 16;

    pub fn specShouldReenable(committed: []const u32, generated_len: usize) bool {
        if (generated_len < SPEC_REENABLE_MIN_TOKENS) return false;
        const window = @min(SPEC_REENABLE_WINDOW, generated_len);
        const frac = pld_index.tailMatchFraction(committed, window, 3);
        return frac >= SPEC_REENABLE_MIN_FRACTION;
    }

    /// Emit a stable, easy-to-grep one-line summary of spec-decode acceptance
    /// for this request. External tooling parses the `[spec-stats]` prefix;
    /// keep the format stable.
    ///
    /// No-op when this Generator never ran a speculative path. Drafter and
    /// PLD are mutually exclusive within a single request (drafter > PLD per
    /// dispatch), so the branching here is unambiguous.
    ///
    /// Field semantics:
    /// - `attempts` = number of speculative rounds (one verify forward each).
    /// - `accepts` = total drafted tokens accepted across all rounds (excludes
    ///   the always-committed t1 token at the start of each round).
    /// - `avg_per_round` = accepts/attempts. Bounded by `(block_size - 1)` for
    ///   drafter and `pld_draft_len` for PLD. Equals the metric the runtime
    ///   gate compares against `RUNTIME_GATE_MIN_RATE`.
    /// - `per_draft_pct` (drafter only) = accepts / (attempts × (block_size-1)),
    ///   the per-draft acceptance probability comparable to vLLM's reported
    ///   "62% acceptance rate" metric.
    /// Which QSA arm served this request's attention (qwen4_exp only; no-op
    /// elsewhere). The engaged/declined meters are one-shots for the whole
    /// PROCESS, so they cannot say what ran on THIS request at THIS context
    /// length — a gather that quietly hands every call back to the dense mask
    /// (a full kv8 dequant per layer per forward) otherwise reads like a
    /// healthy run. Emitted next to `[spec-stats]`; resets the tally.
    pub fn logQsaArms(self: *Generator) void {
        // The tally rides THIS request's ForwardCtx — the one every forward
        // of this slot was handed. A Transformer-level counter printed the
        // union of every slot that decoded or prefilled since the last
        // finisher reset it.
        const c = self.ctx.qsa_arms.take() orelse return;
        const arm = c.majority() orelse return;
        log.info(
            "  [qsa-arms] qsa={s} calls={d} mask={d} decode={d} verify={d} prefill={d}\n",
            .{ @tagName(arm), c.total(), c.mask, c.decode, c.verify, c.prefill },
        );
    }

    pub fn logSpecStats(self: *const Generator) void {
        var table_buf: [256]u8 = undefined;
        var hist_buf: [256]u8 = undefined;
        const table_bucket = self.xfm.round_cost.bucketOf(self.mtpKvLen());
        if (self.dspark_enabled and self.dspark_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.dspark_accepted_tokens)) /
                @as(f64, @floatFromInt(self.dspark_attempted));
            log.info(
                "  [spec-stats] mode=dspark attempts={d} accepts={d} avg_per_round={d:.2}\n",
                .{ self.dspark_attempted, self.dspark_accepted_tokens, avg_per_round },
            );
            return;
        }
        if (self.mtp != null and self.mtp_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
                @as(f64, @floatFromInt(self.mtp_attempted));
            // Depth varies per round under the EV controller — the honest
            // denominator is the DRAFTED count, not attempts x cap.
            const drafts_proposed: u64 = if (self.mtp_drafted_tokens > 0)
                self.mtp_drafted_tokens
            else
                self.mtp_attempted * @as(u64, self.mtp_depth);
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  [spec-stats] mode=mtp attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% depth={d} drafted={d} ext_rounds={d} runtime_disabled={s} reason={s} adaptive={s} serial_cell={d:.2} sync_ms={d:.2} round_ms={d:.2} two_ms_tok={d:.2} one_ms_tok={d:.2} verdict_round={d} trials={d} width_trials={d} table={s}:{s} table_drops=t{d}/c{d}/b{d} serial_drops=t{d}/c{d}/b{d}\n",
                .{
                    self.mtp_attempted,
                    self.mtp_accepted_tokens,
                    avg_per_round,
                    per_draft_pct,
                    self.mtp_depth,
                    self.mtp_drafted_tokens,
                    self.mtp_ext_rounds,
                    if (self.spec_disabled_runtime) "true" else "false",
                    @tagName(self.spec_disable_reason),
                    @tagName(self.mtp_adaptive.arm),
                    self.xfm.round_cost.serialMsPerTok(table_bucket) orelse 0.0,
                    self.mtp_ev_sync_ms,
                    self.mtp_ev_round_ms,
                    if (self.mtp_regime.two_tok > 0) self.mtp_regime.two_ms / self.mtp_regime.two_tok else 0.0,
                    if (self.mtp_regime.one_tok > 0) self.mtp_regime.one_ms / self.mtp_regime.one_tok else 0.0,
                    self.mtp_regime.verdict_round,
                    self.mtp_regime.trials,
                    self.mtp_width_trial.trials,
                    round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                    self.xfm.round_cost.formatBucket(table_bucket, &table_buf),
                    self.xfm.round_cost.dropped_transition,
                    self.xfm.round_cost.dropped_contended,
                    self.xfm.round_cost.dropped_bad,
                    self.xfm.round_cost.serial_dropped_transition,
                    self.xfm.round_cost.serial_dropped_contended,
                    self.xfm.round_cost.serial_dropped_bad,
                },
            );
            return;
        }
        if (self.dflash != null and self.dflash_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
                @as(f64, @floatFromInt(self.dflash_attempted));
            const drafts_per_round: u32 = if (self.dflash_block_size >= 1) self.dflash_block_size - 1 else 0;
            // Under the chooser the width varies per round: drafts proposed
            // is the histogram's sum, not attempts x a fixed block.
            const drafts_proposed: u64 = if (self.dflash_chooser) |ch| ch.draftsProposed() else self.dflash_attempted * @as(u64, drafts_per_round);
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  [spec-stats] mode=dflash attempts={d} accepts={d} avg_per_round={d:.2} gate_min={d:.2} per_draft_pct={d:.1}% block_size={d} runtime_disabled={s} table={s}:{s} table_drops=t{d}/c{d}/b{d} block_avg={d:.2} block_hist={s} chooser_trials={d}\n",
                .{
                    self.dflash_attempted,
                    self.dflash_accepted_tokens,
                    avg_per_round,
                    self.dflash_min_accepted_per_round,
                    per_draft_pct,
                    if (self.dflash_chooser) |ch| ch.current + 1 else self.dflash_block_size,
                    if (self.spec_disabled_runtime) "true" else "false",
                    round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                    self.xfm.round_cost.formatBucket(table_bucket, &table_buf),
                    self.xfm.round_cost.dropped_transition,
                    self.xfm.round_cost.dropped_contended,
                    self.xfm.round_cost.dropped_bad,
                    if (self.dflash_chooser) |ch| ch.avgWidth() else @as(f32, @floatFromInt(drafts_per_round)),
                    if (self.dflash_chooser) |*ch| ch.formatHist(&hist_buf) else "",
                    if (self.dflash_chooser) |ch| ch.trial.trials else 0,
                },
            );
            return;
        }
        if (self.drafter != null and self.drafter_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.drafter_accepted_tokens)) /
                @as(f64, @floatFromInt(self.drafter_attempted));
            const drafts_per_round: u32 = if (self.drafter_block_size >= 1) self.drafter_block_size - 1 else 0;
            const drafts_proposed: u64 = self.drafter_attempted * @as(u64, drafts_per_round);
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.drafter_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  [spec-stats] mode=drafter attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% block_size={d} runtime_disabled={s}\n",
                .{
                    self.drafter_attempted,
                    self.drafter_accepted_tokens,
                    avg_per_round,
                    per_draft_pct,
                    self.drafter_block_size,
                    if (self.spec_disabled_runtime) "true" else "false",
                },
            );
        } else if (self.pld_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.pld_accepted_tokens)) /
                @as(f64, @floatFromInt(self.pld_attempted));
            log.info(
                "  [spec-stats] mode=pld attempts={d} accepts={d} avg_per_round={d:.2} runtime_disabled={s}\n",
                .{
                    self.pld_attempted,
                    self.pld_accepted_tokens,
                    avg_per_round,
                    if (self.spec_disabled_runtime) "true" else "false",
                },
            );
        }
    }

    /// Prefill the prompt and prepare for token-by-token generation.
    /// Backwards-compatible — prefer `initWithOptions` for new callers.
    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling: SamplingParams,
        eos_token_ids: []const u32,
    ) !Generator {
        return initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{});
    }

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
        /// disagree. (Live 2026-08-14: pinned 4096, prefilled at 8192.)
        pinned_prefill_chunk: usize = 0,
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
        /// Enable the Qwen native MTP head. When set, `mtp` must be non-null
        /// and `bind()`-ed to `xfm`. Prefill builds the head's committed-
        /// history KV cache chunk-by-chunk (full-hidden capture) and the
        /// final-token forward captures `last_hidden`, exactly like the
        /// drafter path. Same lazy-pre-forward skip semantics as PLD/drafter.
        mtp_enabled: bool = false,
        /// Non-owning pointer to the loaded MTP head.
        mtp: ?MtpHeadRef = null,
        /// The MODEL's head, before this request's opt-out. `Transformer`
        /// loads `qwen4_mtp` with the trunk whatever `--no-mtp` says, so the
        /// weights are the wrong thing to ask; the scheduler passes the
        /// registry's answer.
        model_has_mtp: bool = false,
        /// Max tokens drafted per nextMtp round. 0 = auto (`--mtp-depth` not
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
        /// Phase 1 (performance-plan): during prefill, capture an SSM
        /// checkpoint every `ssm_checkpoint_stride` tokens. 0 = disabled.
        /// Snapshots land in `Generator.ssm_checkpoints` for the caller to
        /// drain into the hot prefix cache via `takeSsmCheckpoints()`. Only
        /// effective when the model has hybrid layers (otherwise the
        /// `ssm_entries` slice is empty and snapshots become no-op stubs).
        /// Chunked prefill aligns chunk ends to stride positions so each
        /// snapshot reflects a coherent state.
        ssm_checkpoint_stride: u32 = 0,
        /// Cap on the number of checkpoints retained. Past the cap the list is
        /// thinned span-preservingly (`transformer.ssmCheckpointDropIndex`):
        /// the lowest and the newest positions always survive, so the
        /// survivors still span the WHOLE prompt and an oversized commit has
        /// an affordable trim point. 0 = unlimited (rely on the hot-cache byte
        /// budget to bound).
        ssm_checkpoint_max: u32 = 16,
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
        /// init's split-prefill final-token forward is a single-row dispatch,
        /// and the certified lm_head prune may only engage when the request
        /// consumes nothing but the argmax — a logprobs request reads the
        /// full logit row, which the pruned head does not produce.
        logprobs_n: u32 = 0,
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

        /// Called once per completed prefill chunk boundary (chunk state
        /// evaluated, allocator cache cleared), except the final one. The
        /// scheduler runs decode ticks for the streams already decoding at
        /// this seam, so a long cold prefill stalls them for at most one
        /// chunk-forward instead of its whole duration. Null = the prefill
        /// runs atomically (pre-interleave behavior, and the
        /// MLX_SERVE_PREFILL_INTERLEAVE=0 kill switch).
        interleave_hook: ?InterleaveHook = null,
        /// SSD-first mechanism 3 (qwen4_exp only; see `WriteThroughHook`).
        write_through_hook: ?WriteThroughHook = null,
        /// The per-CHUNK prefill width. Asked at every chunk boundary AFTER
        /// that chunk's `mlx_clear_cache` and BEFORE the interleave tick, so
        /// the probe behind it reads the steady state the next chunk starts
        /// from and not the tick's allocations. Null keeps the request's
        /// admitted width for the whole prefill — which is every unit test and
        /// every host without the HTTP server.
        ///
        /// PRESENCE IS NOT THE ARCH GATE. `serve` installs the policy
        /// unconditionally and process-wide, so under a real server this is
        /// non-null on EVERY arch; the width then holds because the policy
        /// itself declines (`server.adaptivePrefillChunkEnabled`), not because
        /// the hook is absent. Anything that needs to know whether the
        /// per-chunk width is LIVE reads `adaptive_chunk_width`. (audit B-A2)
        chunk_width_hook: ?ChunkWidthHook = null,
        /// Is the per-chunk adaptive width live for THIS request — the ARCH
        /// predicate (`server.adaptivePrefillChunkEnabled`: qwen4_exp, its two
        /// kill switches, and no operator pin), delivered per model by
        /// `scheduler.adaptiveChunkWidthFor`. False on every other arch, in
        /// every unit test and on every host without the HTTP server, and the
        /// only thing that may widen `tailMergeMaxFor`'s bound.
        adaptive_chunk_width: bool = false,
    };

    pub const InterleaveHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque) void,
    };

    /// SSD-first mechanism 3: called at every completed prefill chunk with the
    /// ABSOLUTE KV position now forwarded and the SSM checkpoints captured so
    /// far. The bytes of a chunk are final once it is evaluated, so persisting
    /// them here — rather than at end of request — means a cancelled or
    /// killed prefill still leaves a chunk-aligned restorable prefix, and the
    /// end-of-request flush is only the tail. Null on every arch but the one
    /// the scheduler arms.
    pub const WriteThroughHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque, abs_kv_pos: usize, cps: []const SSMCheckpoint) void,
    };

    /// Same shape as `InterleaveHook`: the caller owns the context, this loop
    /// owns the `AdaptiveWidthState` it hands in. Returns the width the next
    /// chunk should run at (`cur` unchanged = no change).
    pub const ChunkWidthHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque, pos: usize, cur: u32, cap: u32, st: *AdaptiveWidthState) u32,
        /// Re-price a WIDEN after the interleave tick has allocated for the
        /// co-tenant slots it hosts. `call` runs BEFORE the tick so a
        /// co-tenant's decode is not attributed to this prefill's pressure —
        /// right for attribution, backwards for safety, because a widen
        /// decided on pre-tick headroom then forwards into memory the probe
        /// never saw and a Metal abort cannot be un-decided. A step-down needs
        /// no confirmation: it is the safe direction and waiting costs bytes.
        confirm: *const fn (ctx: *anyopaque, pos: usize, want: u32) bool,
    };

    /// Selects the source slice that `initWithOptions` will dupe into
    /// `prompt_ids_owned`. When `lookup_prompt` is non-null it wins (server
    /// cache-reuse path: full original prompt for PLD lookup); otherwise the
    /// caller's `prompt_ids` is used (back-compat path).
    pub fn pickLookupPromptSource(prompt_ids: []const u32, lookup_prompt: ?[]const u32) []const u32 {
        return lookup_prompt orelse prompt_ids;
    }

    /// Which DSpark accept rule (if any) the dsv4 chokepoint may arm for a
    /// request. Pure over its inputs so every arm is unit-testable.
    pub const DsparkArm = enum { off, greedy, stochastic };

    /// `clean` = nothing consumes logits beyond plain sampling: penalties,
    /// grammar and logprobs stay serial on BOTH arms (matching the greedy-only
    /// contract this generalizes). A clean greedy request (temp < 0.01 or
    /// top_k == 1) gets the raw argmax-equality accept; a clean SAMPLED
    /// request gets the stochastic arm (MTP one-hot Leviathan acceptance over
    /// filtered target probs) unless `stoch_enabled` is false — the
    /// MLX_SERVE_DSV4_DSPARK_STOCH=0 kill switch, which restores greedy-only
    /// gating.
    pub fn dsparkArmFor(sampling: SamplingParams, logprobs_n: u32, stoch_enabled: bool) DsparkArm {
        const clean = sampling.repeat_penalty == 1.0 and
            sampling.presence_penalty == 0.0 and
            sampling.constraint == null and
            logprobs_n == 0;
        if (!clean) return .off;
        const greedy = sampling.temperature < 0.01 or sampling.top_k == 1;
        if (greedy) return .greedy;
        return if (stoch_enabled) .stochastic else .off;
    }

    /// Stochastic-DSpark kill switch — MLX_SERVE_DSV4_DSPARK_STOCH=0
    /// restores the greedy-only chokepoint gate for A/Bs.
    var dspark_stoch_cache: ?bool = null;
    pub fn dsparkStochEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return value.len == 0 or value[0] != '0';
    }

    fn dsparkStochEnabled() bool {
        if (dspark_stoch_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_DSV4_DSPARK_STOCH")) |p| std.mem.span(p) else null;
        const on = dsparkStochEnabledFromEnv(raw);
        dspark_stoch_cache = on;
        return on;
    }

    pub fn initWithOptions(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling_in: SamplingParams,
        eos_token_ids: []const u32,
        options_in: InitOptions,
    ) !Generator {
        // Reserved-token suppression rides the sampling params from HERE —
        // the one chokepoint every init site funnels through — so every
        // sampling path (serial, PLD/drafter/MTP corrections, draft heads,
        // stochastic-verify filters, batched decode via `gen.sampling`)
        // inherits the model's mask without per-site wiring.
        var sampling = sampling_in;
        sampling.suppress_mask = xfm.suppress_mask;
        // DeepSeek-V4 hard-off, at the ONE chokepoint every init site
        // funnels through: dsv4's per-request state lives on the module
        // (rings + compressed caches) and a spec VERIFY forward appends
        // draft tokens to it with NO rollback — two rejected PLD drafts
        // permanently corrupted a live generation (mangled DSML with dropped
        // token runs, 2026-07-31; the per-site `is_dsv4` wiring guard in
        // scheduler.runPrefill demonstrably did not cover the engaged path,
        // and per-site wiring is the class the spec-dispatch rule warns
        // about).
        var options = options_in;
        var dspark_active = false;
        var dspark_stochastic = false;
        if (xfm.dsv4 != null and (options.pld_enabled or options.drafter_enabled or options.mtp_enabled or options.dflash_enabled)) {
            // DSpark lift: dsv4's OWN draft mode (block-parallel stages +
            // snapshot rollback inside deepseek_v4.zig) may engage when the
            // checkpoint ships stages and the request is CLEAN (no
            // penalties, grammar or logprobs — those consume logits the
            // draft path never shapes and stay serial). Greedy requests get
            // the raw argmax-equality accept; sampled requests get the
            // stochastic arm (MTP one-hot Leviathan acceptance over the
            // request's own filtered probs — the agent-default temp 0.6
            // traffic that otherwise always ran serial), env-killable via
            // MLX_SERVE_DSV4_DSPARK_STOCH=0. PLD / drafter / qwen-MTP
            // remain hard-off regardless: their verify forwards go through
            // machinery this arch cannot roll back.
            const mdl_ds = xfm.dsv4.?;
            const dspark_env_off = if (std.c.getenv("MLX_SERVE_DSV4_DSPARK")) |v| v[0] == '0' else false;
            const arm = dsparkArmFor(sampling, options.logprobs_n, dsparkStochEnabled());
            if (mdl_ds.n_mtp > 0 and !dspark_env_off and arm != .off) {
                dspark_active = true;
                dspark_stochastic = arm == .stochastic;
                if (dspark_stochastic) {
                    log.info("  spec=dspark (stochastic; deepseek_v4 native draft stages, block={d})\n", .{mdl_ds.ds_block});
                } else {
                    log.info("  spec=dspark (deepseek_v4 native draft stages, block={d})\n", .{mdl_ds.ds_block});
                }
            } else {
                log.info("  spec=disabled (deepseek_v4 serves serial-only)\n", .{});
            }
            options.pld_enabled = false;
            options.drafter_enabled = false;
            options.drafter = null;
            options.mtp_enabled = false;
            options.mtp = null;
            options.dflash_enabled = false;
            options.dflash = null;
        }
        const s = xfm.s;
        // Per-slot ForwardCtx (Phase 2). Stored by value on the Generator;
        // callers either supply one (scheduler) or fall through to
        // `xfm.defaultCtx()` for the legacy single-slot path. We pass
        // `&ctx` to every forward call below; the cache/moe/ssm fields
        // mutate in-place through their pointers.
        var ctx: ForwardCtx = options.ctx orelse xfm.defaultCtx();

        // Certified lm_head prune gate: the pruned projection proves the
        // ARGMAX, not the tail distribution, so it may engage only when this
        // request consumes nothing else — greedy (or top-1) sampling with no
        // logit-modifying penalties, no per-token logprobs and no grammar
        // mask. Mirrors the `logprobs>0 + grammar disable spec` precedent:
        // no request gets slower, some get faster.
        ctx.argmax_only = (sampling.temperature < 0.01 or sampling.top_k == 1) and
            sampling.repeat_penalty == 1.0 and
            sampling.presence_penalty == 0.0 and
            sampling.constraint == null and
            options.logprobs_n == 0;

        const ids_i32 = try allocator.alloc(i32, prompt_ids.len);
        defer allocator.free(ids_i32);
        for (prompt_ids, 0..) |id, i| {
            ids_i32[i] = @intCast(id);
        }

        // Clone the lookup prompt for the lifetime of the Generator. PLD's
        // n-gram lookup needs `prompt + generated`, and the caller-owned
        // slice can be freed before `nextPld` runs. When `options.lookup_prompt`
        // is set (server cache-reuse path), it carries the full original prompt
        // so PLD's match coverage isn't gutted when only a trailing tail was
        // forwarded into the KV cache. Defaults to `prompt_ids` otherwise.
        // Allocated up front so init's errdefer paths don't have to track
        // partial state.
        const owned_src = pickLookupPromptSource(prompt_ids, options.lookup_prompt);
        const prompt_owned = try allocator.dupe(u32, owned_src);
        errdefer allocator.free(prompt_owned);

        // Split prefill: process first N-1 tokens (cache-only, skip lm_head eval),
        // then the last token (produces logits for sampling). This mirrors mlx-lm's
        // generate_step which avoids the expensive lm_head projection over the full
        // sequence length. For vocab_size=262144, skipping lm_head on N-1 tokens
        // avoids a [N-1, hidden] @ [hidden, 262144] matmul.
        //
        // Chunked prefill: large prompts are processed in PREFILL_CHUNK-sized pieces
        // to bound peak activation memory. Each chunk fills KV cache entries for its
        // positions, gets eval'd, and intermediates are freed before the next chunk.
        // Without chunking, Gemma-4 MoE's 2 MLPs × 4 stacked layers can spike to
        // ~20 GB of activations alone on a 50k-token prompt, causing Metal OOM.
        // Vision requests skip chunking since image token positions must be visible
        // in a single forward pass for spliceVisionEmbeddings to work correctly.
        // PREFILL_CHUNK overridable via env MLX_SERVE_PREFILL_CHUNK for tuning,
        // or via the module-level `prefill_chunk_override` (set by --prefill-chunk
        // CLI flag in main.zig). Env var wins if both are set (and skips the
        // safety cap below — it's the explicit escape hatch).
        //
        // Safety cap: on unfused head dims (>128 — every Gemma-4/Qwen3.5/3.6)
        // the composed SDPA materializes [heads, chunk, total_kv] scores per
        // layer, so the chunk shrinks with the prompt's FINAL KV length to keep
        // that one tensor bounded (boundedPrefillChunk). Warm-path restores
        // start at ssm_checkpoint_pos_offset, so the final KV length is that
        // offset plus everything we're about to forward.
        const total_ctx_for_chunk = options.ssm_checkpoint_pos_offset + prompt_ids.len;
        const PREFILL_CHUNK: usize = effectivePrefillChunk(
            xfm.config.prefillScoreHeadDim(),
            xfm.config.num_attention_heads,
            total_ctx_for_chunk,
            xfm.config.has_sliding_window,
            xfm.config.isMoe(),
            options.pinned_prefill_chunk,
        );
        // Phase-level prefill instrumentation. Enabled at debug level OR via
        // MLX_SERVE_PREFILL_TRACE=1 (which forces the trace line at info).
        // Phase 0 of plan 04 — gives us a decomposed view of where cold prefill
        // time goes (chunked-forward vs eval vs last-token-forward).
        const trace_force: bool = prefill_trace_force or readEnvBool("MLX_SERVE_PREFILL_TRACE");
        const trace_enabled = log.isDebug() or trace_force;
        var prefill_sw = io_util.Stopwatch.init(io);
        var chunked_ns: u64 = 0;
        var eval_ns: u64 = 0;
        var n_chunks: usize = 0;

        // Phase 1: SSM checkpointing during prefill. When enabled, the chunked
        // prefill loop forces a chunk boundary at every multiple of
        // `ssm_checkpoint_stride`, then snapshots `ctx.ssm_entries` after that
        // chunk evaluates. Snapshots accumulate in `Generator.ssm_checkpoints`
        // for the scheduler to drain in `commitSlotIfApplicable`. Plain-attn
        // models have an empty `ssm_entries` slice, so this becomes a no-op
        // even at stride > 0 — but we still bail early so we never allocate
        // empty checkpoints.
        var ssm_checkpoints: std.ArrayList(SSMCheckpoint) = std.ArrayList(SSMCheckpoint).empty;
        errdefer {
            for (ssm_checkpoints.items) |*cp| cp.deinit(allocator);
            ssm_checkpoints.deinit(allocator);
        }
        const has_vision = ctx.vision_embeddings != null;
        const want_ssm_cp = shouldCheckpointSsmPrefill(
            options.ssm_checkpoint_stride,
            ctx.ssm_entries != null and ctx.ssm_entries.?.len > 0,
            has_vision,
        );
        // The snapshot backoff: a checkpoint AT the prompt end is
        // unreachable next turn.
        const want_state_cp = want_ssm_cp;
        // Coarsen the checkpoint stride so checkpointing never sub-divides the
        // prefill chunk, on ANY arch (see effectiveSsmCheckpointStride: fine
        // strides push every projection under prefillDqGemm's M>=2048 floor and
        // multiply per-chunk fixed costs — 17-25% cold-prefill tax measured on
        // dense AND MoE hybrids). Warm reuse keeps chunk-granularity snapshots
        // plus the always-on end-of-prompt snapshot (the append-growth case
        // llmprobe's cache-hit tests pin — verified green at coarse strides).
        // Coarsen against the UNCAPPED base chunk: the head_dim safety cap
        // above must not densify checkpoint spacing (16× more captures at
        // 255K ctx otherwise). nextChunkEnd already shortens a chunk to land
        // on stride boundaries, so a capped chunk stays compatible.
        const ssm_cp_stride: usize = if (want_ssm_cp)
            effectiveSsmCheckpointStride(@intCast(options.ssm_checkpoint_stride), @max(PREFILL_CHUNK, prefill_chunk_override))
        else
            0;
        // Absolute KV position of `prompt_ids[0]`. Warm-path callers (the
        // scheduler after restoring a checkpoint) pass the matched prefix
        // length so the snapshots stamp positions valid in the full original
        // sequence, not relative offsets inside the tail-only prefill.
        const ssm_cp_offset: usize = options.ssm_checkpoint_pos_offset;

        // Qwen native MTP: build the head's committed-history KV cache during
        // prefill. Entry j pairs (trunk hidden at prompt position j, token at
        // j+1); the (hidden[last], t1) pair is appended by the first nextMtp
        // round. On KV-prefix reuse the history covers only the freshly
        // forwarded tail — RoPE offsets are cache-relative, so a late-starting
        // history is self-consistent (sliding-window history semantics).
        const mtp_active = options.mtp_enabled and options.mtp != null;
        var mtp_cache: ?MtpCacheRef = null;
        var mtp_position_base: usize = ssm_cp_offset;
        var mtp_history_started = false;
        if (mtp_active) {
            if (options.mtp_cache_restored) |restored| {
                // A restored history already covers [base, ssm_cp_offset);
                // the tail prefill APPENDS to it (RoPE offsets are
                // cache-relative, so continuation is self-consistent).
                std.debug.assert(restored.base + restored.cache.step() == ssm_cp_offset);
                mtp_cache = restored.cache;
                mtp_position_base = restored.base;
                mtp_history_started = true;
            } else {
                mtp_cache = try options.mtp.?.makeCache(allocator);
            }
        }
        errdefer if (mtp_cache) |*mc| mc.deinit();

        // DFlash: build the assistant's context cache during prefill — every
        // chunk's target_layer_ids captures project through the encoder and
        // append immediately (the chunk's activation graph then frees with
        // the chunk eval). On KV-prefix reuse the context covers only the
        // freshly forwarded tail (`base_pos = ssm_cp_offset`) — sliding-
        // window semantics, same rule as the MTP history.
        const dflash_active = options.dflash_enabled and options.dflash != null;
        var dflash_ctx: ?dflash_mod.DflashCtx = if (!dflash_active)
            null
        else if (options.dflash_ctx_restored) |restored| blk: {
            std.debug.assert(restored.absLen() == ssm_cp_offset);
            break :blk restored;
        } else try dflash_mod.DflashCtx.init(allocator, options.dflash.?, ssm_cp_offset);
        errdefer if (dflash_ctx) |*dc| dc.deinit();
        var dfl_out_buf: []mlx.mlx_array = &.{};
        defer if (dfl_out_buf.len > 0) allocator.free(dfl_out_buf);
        var dfl_cl: transformer_mod.CaptureLayers = undefined;
        if (dflash_active) {
            dfl_out_buf = try allocator.alloc(mlx.mlx_array, options.dflash.?.config.target_layer_ids.len);
            dfl_cl = .{ .ids = options.dflash.?.config.target_layer_ids, .out = dfl_out_buf };
        }

        // Start of the final (logits) forward's token span. Without SSM
        // checkpointing this is the last prompt token (the classic 1-token
        // logits forward); with checkpointing the chunk loop stops
        // SSM_SNAPSHOT_BACKOFF tokens early so the always-on snapshot lands
        // where the next turn's prefix match can actually reach it (see
        // ssmSnapshotBackoff), and the final forward covers the held-back
        // tail + the last token in the same weight sweep.
        var final_start: usize = prompt_ids.len - 1;
        // Chunked vision prefill (issue #197): placeholder rows consumed by
        // completed chunks, fed to ctx.vision_splice_offset per forward
        // (chunk loop AND final-span forward). Stays 0 on the kill-switch arm
        // so that path is byte-identical to the old whole-prompt behavior.
        const vision_chunked = has_vision and visionChunkedPrefillEnabled();
        var vision_rows_consumed: usize = options.vision_rows_before;
        // Per-request adaptive-width bookkeeping. Declared out here so the
        // one summary line can be emitted beside the prefill trace, which
        // sits outside the chunk loop's scope.
        var adapt_state: AdaptiveWidthState = .{
            .width_min = @intCast(PREFILL_CHUNK),
            .width_max = @intCast(PREFILL_CHUNK),
        };

        if (prompt_ids.len > 1) {
            const prefix_len = prompt_ids.len - 1;
            const snapshot_backoff = ssmSnapshotBackoff(want_state_cp, prefix_len);
            const loop_end = prefix_len - snapshot_backoff;
            final_start = loop_end;
            // Vision prompts chunk like text (issue #197) — the splice offset
            // below keeps the row scatter chunk-exact. Kill switch restores
            // the whole-prompt forward.
            const default_chunk = if (has_vision and !vision_chunked) loop_end else PREFILL_CHUNK;
            // PER-CHUNK adaptive width. The request's pin is what the first
            // chunk runs; every boundary after it re-asks the SAME estimator
            // admission used, against the memory the next chunk will actually
            // run in. The prompt's own KV is the thing that moves: at 1M
            // tokens on qwen4_exp the score sheet and the QSA mask grow with
            // the KV until the width that was affordable at token 0 is not,
            // and a fixed width has to be the LAST one to be right for the
            // whole prompt. `cap_adapt` is the widest this arch will forward
            // for this prompt (the pin left out on purpose — widening past it
            // is the point), and it is never wider than `ssm_cp_stride`, so
            // checkpointing still never sub-divides a chunk.
            const adapt_chunked = !(has_vision and !vision_chunked);
            // S18/BL-5: the scaled tail-merge bound belongs to the per-chunk
            // adaptive width and to nothing else — so it reads the ARCH
            // predicate, delivered per model on `adaptive_chunk_width`.
            //
            // NOT `chunk_width_hook != null` (audit B-A2): `serve` installs
            // `prefill_chunk_adapt` unconditionally and process-wide, so under
            // a real server the hook is non-null on every arch and that
            // spelling put the scaled bound back on all of them — the exact
            // defect the gate exists to prevent, with two green tests over it
            // because they exercised the flag and not the wiring.
            const width_is_adaptive = adapt_chunked and options.adaptive_chunk_width;
            const cap_adapt: u32 = if (!adapt_chunked) 0 else @intCast(effectivePrefillChunk(
                xfm.config.prefillScoreHeadDim(),
                xfm.config.num_attention_heads,
                total_ctx_for_chunk,
                xfm.config.has_sliding_window,
                xfm.config.isMoe(),
                0,
            ));
            adapt_state.width_min = @intCast(default_chunk);
            adapt_state.width_max = @intCast(default_chunk);
            var cur_chunk: usize = default_chunk;
            // Last-window MTP history: chunks entirely before the window skip
            // the full-hidden capture AND the head forward (see
            // mtp.SUGGESTED_HISTORY_WINDOW). 0 = capture every chunk.
            const mtp_hist_window = effectiveMtpHistoryWindow(prefix_len, mtp_history_window_override);

            // Reserve the whole request's cache capacity BEFORE the first
            // chunk writes. A grow is not in place — `growQuantBuf` allocates
            // the new capacity with `mlx_zeros` and slice_updates the old
            // buffer into it, and both stay live in the chunk's lazy graph
            // until its eval — so a long prefill's peak carried a second copy
            // of everything accumulated so far (~7.75 GB at 458,832 tokens on
            // qwen4_exp; the runtime floor prices it at a flat 512 MB, issue
            // #353). One allocation up front removes the transient instead of
            // pricing it, and it is exactly what the admission guard billed:
            // both sides call `KVCache.reservedTokens`.
            // On `ctx.cache`, not `xfm.cache`: the scheduler swaps a slot's
            // own cache onto the forward context, and the reservation belongs
            // to the buffers this prefill will actually write.
            // The context bound is the transformer's own: `max_tokens` reaches
            // us already clamped by the server, but a reservation is a number
            // two subsystems must agree on and neither may trust its caller
            // for it (the omitted-max_tokens sentinel, #353 follow-up).
            // ABSOLUTE capacity, so it must be billed in ABSOLUTE positions.
            // `prompt_ids` is the TAIL after a prefix-cache hit, and the cache
            // already holds `ssm_checkpoint_pos_offset` rows — so on every warm
            // path the reservation landed BELOW the capacity the cache already
            // had, `nextCapacityReserved` fell through to the +25% ladder, and
            // the tail prefill re-grew: exactly the old+new coexistence
            // transient #353 measured at 7.75 GB @ 458k, silently absent on
            // the warm long-context path this reservation exists for. The
            // admission guard already bills the FULL prompt length
            // (`server.reservedCacheTokens`), so this also stops the two
            // drifting apart. `total_ctx_for_chunk` is that absolute length.
            // ARCH GATE for checkpoint retention (PR #363 item 3). a93e2c0
            // dropped the OLDEST at both capture sites; the span-preserving
            // thin moves where every warm turn restores from, on every hybrid
            // (lfm2, nemotron_h, qwen3_5*, qwen3_next, bailing_hybrid,
            // inkling), and was measured on the 383k qwen4_exp shape alone.
            // The enum names the a93e2c0 policy of THIS site.
            const cp_thin: transformer_mod.ThinPolicy =
                if (xfm.config.longCtxGated()) .min_span_recency else .oldest;

            const reserved_tokens = reservedPrefillTokens(
                &xfm.config,
                total_ctx_for_chunk,
                max_tokens,
                default_chunk,
            );
            ctx.cache.reserve(@intCast(reserved_tokens));
            // The arch's OWN per-request buffers reserve at the SAME length.
            // The KV cache was sized in one shot while the QSA raw-key history
            // and the pooled bank still walked the +25% ladder underneath it —
            // the transient the guard already bills (`statePerTokenBilled`),
            // paid per layer per rung (#353 follow-up).
            transformer_mod.reserveQsaHistory(ctx.ssm_entries, @intCast(reserved_tokens));

            var pos: usize = 0;
            while (pos < loop_end) {
                // Abandoned-request abort: the client disconnected and the
                // conn thread flagged the slot. Bail before the next chunk —
                // the KV built so far is freed with the slot.
                if (options.cancel_flag) |cf| {
                    if (cf.load(.acquire)) {
                        // Salvage the chunk loop's progress: `forwarded` is
                        // the authoritative length (cache.step only advances
                        // when init completes), the stride checkpoints ride
                        // for hybrid restore (they die with the failed
                        // construction otherwise). toOwnedSlice empties the
                        // local list so the errdefer frees nothing (on OOM
                        // the items stay and the errdefer cleans up — the
                        // sink keeps the length, the commit declines the
                        // checkpoints only).
                        if (options.cancelled_checkpoint_sink) |sink| {
                            sink.forwarded = ssm_cp_offset + pos;
                            if (ssm_checkpoints.items.len > 0) {
                                if (ctx.ssm_entries) |ents| {
                                    // The slot dies with the cancel: share
                                    // the live history when the switch is
                                    // on, copy when it is off.
                                    transformer_mod.attachQsaHistoryOnHandoff(ssm_checkpoints.items, ents, xfm.s) catch {};
                                }
                                if (ssm_checkpoints.toOwnedSlice(allocator)) |owned| {
                                    sink.checkpoints = owned;
                                    sink.alloc = allocator;
                                } else |_| {}
                            }
                        }
                        return error.Cancelled;
                    }
                }
                // An MLX failure inside a PREVIOUS chunk — the Metal
                // working-set OOM of issue #353 is the one this exists for —
                // latched a message instead of killing the process (see
                // `mlx.installErrorHandler`). Everything that chunk produced
                // is garbage, but MLX has cleared its own error state, so the
                // honest move is to abandon THIS request and keep the server.
                // No checkpoint salvage: unlike a client disconnect, the
                // state we would be saving is the state that failed.
                try mlx.checkError();
                // Pick this chunk's end. Normal path: hit the configured chunk
                // size. Phase 1 path: if a checkpoint stride boundary lands
                // inside the would-be chunk, shrink the chunk so it ends
                // exactly on that boundary. That gives us an snapshot-point
                // every `stride` tokens without changing the model's seen
                // input — the forward result is identical to the unchunked
                // version because attention is causal and SSM/conv update
                // chunk-locally. Boundary alignment is in ABSOLUTE position
                // (pos + offset), so the saved snapshot list is correct for
                // the full prompt, not the truncated tail.
                const end = nextChunkEnd(pos, loop_end, cur_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset, width_is_adaptive);
                if (has_vision) ctx.vision_splice_offset = vision_rows_consumed;
                const chunk_len: c_int = @intCast(end - pos);
                const chunk_shape = [_]c_int{ 1, chunk_len };
                const chunk_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[pos]), &chunk_shape, 2, .int32);
                defer _ = mlx.mlx_array_free(chunk_input);

                const chunk_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                // Phase 2 experiment: when MLX_SERVE_COMPILE_FORWARD=1 wired a
                // compiled closure at load time, route this chunk through it.
                // The compiled closure uses xfm.defaultCtx (xfm.cache + xfm.ssm_entries),
                // which matches the prefill `ctx` when the scheduler has swapped
                // the slot's cache onto the Transformer (the single-slot legacy
                // and Phase-2-swapped path both satisfy this). Hidden-capture
                // and vision splice paths don't pass through this chunk loop
                // (they take the last_input branch), so they're already safe.
                // Optional-slice equality: same-ness here means both null or
                // both point at the same backing memory. We accept ssm_entries
                // null↔null too because plain-attn models legitimately have
                // both ctx and xfm carry null.
                const ssm_match = blk: {
                    if (ctx.ssm_entries == null and xfm.ssm_entries == null) break :blk true;
                    if (ctx.ssm_entries == null or xfm.ssm_entries == null) break :blk false;
                    break :blk ctx.ssm_entries.?.ptr == xfm.ssm_entries.?.ptr and
                        ctx.ssm_entries.?.len == xfm.ssm_entries.?.len;
                };
                // History windowing: a chunk before the window needs no
                // capture, which ALSO re-qualifies it for the compiled
                // trunk forward (capture is what disqualifies MTP chunks).
                const mtp_capture = mtp_active and chunkNeedsMtpHistory(pos, end, prefix_len, mtp_hist_window);
                var chunk_hidden_all = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(chunk_hidden_all);
                if (dflash_active) {
                    for (dfl_out_buf) |*a| a.* = mlx.mlx_array_new();
                    ctx.capture_layers = &dfl_cl;
                }
                const chunk_logits = if (xfm.compiled_forward != null and
                    !mtp_capture and
                    !dflash_active and
                    ctx.cache == &xfm.cache and
                    ssm_match and
                    ctx.capture_hidden == null and
                    ctx.vision_embeddings == null)
                    try xfm.forwardCompiled(chunk_input)
                else if (mtp_capture) blk: {
                    var last_unused = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(last_unused);
                    break :blk try xfm.forwardWithCaptureAll(&ctx, chunk_input, &last_unused, &chunk_hidden_all);
                } else try xfm.forwardWith(&ctx, chunk_input);
                _ = mlx.mlx_array_free(chunk_logits);
                if (dflash_active) {
                    ctx.capture_layers = null;
                    try dflash_mod.appendContext(options.dflash.?, &dflash_ctx.?, dfl_out_buf, ssm_cp_offset + pos);
                    for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
                }
                if (trace_enabled) chunked_ns += prefill_sw.read() - chunk_start_ns;

                // MTP history for this chunk: hiddens [pos, end) pair with
                // tokens [pos+1, end+1) — prompt_ids[end] always exists since
                // the chunk loop spans [0, prefix_len) and prompt_ids has
                // prefix_len + 1 entries.
                if (mtp_capture) {
                    if (!mtp_history_started) {
                        std.debug.assert(mtp_cache.?.step() == 0);
                        mtp_position_base = ssm_cp_offset + pos;
                        mtp_history_started = true;
                    }
                    const mtp_mrope_ctx: ?mtp_mod.MropeContext = if (ctx.mrope_pos) |positions| .{
                        .pos = positions,
                        .total = ctx.mrope_total,
                        .delta = ctx.mrope_delta,
                        .base = mtp_position_base,
                    } else null;
                    try options.mtp.?.appendHistory(
                        xfm,
                        &mtp_cache.?,
                        prompt_ids[pos + 1 .. end + 1],
                        chunk_hidden_all,
                        @intCast(mtp_cache.?.step()),
                        mtp_mrope_ctx,
                        allocator,
                    );
                }

                // Eval KV cache — materializes this chunk's K/V, frees activation graph
                const eval_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                {
                    const eval_vec = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(eval_vec);
                    for (ctx.cache.entries) |*entry| {
                        if (!entry.initialized) continue;
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
                    }
                    // Materialize this chunk's MTP history entries alongside
                    // the trunk KV so the chunk's activation graph (incl. the
                    // full-hidden capture) can be freed before the next chunk.
                    if (mtp_cache) |*mc| mc.appendEvalArrays(eval_vec);
                    // Same discipline for the DFlash context appended above.
                    if (dflash_ctx) |*dc| dc.appendEvalArrays(eval_vec);
                    // Phase 1: also force SSM state to materialize so any
                    // snapshot below holds a concrete tensor, not a lazy node
                    // that would re-execute the prefill graph if anyone reads
                    // from it later. Unconditional on EVERY chunk (not just
                    // stride-aligned ones): a tail-merged final chunk ends
                    // off-boundary, and the always-on end-of-prompt snapshot
                    // then hits exactly that re-execution — capturing an
                    // un-evaluated state re-ran the whole last chunk's GDN
                    // scan (measured −4% on an 8K Qwen3.6-27B prefill). The
                    // states are ~KB-to-MB scale; evaluating them alongside
                    // the KV costs nothing measurable.
                    if (want_ssm_cp) {
                        for (ctx.ssm_entries.?) |*ssm| {
                            if (!ssm.initialized) continue;
                            if (ssm.conv_state.ctx != null) {
                                _ = mlx.mlx_vector_array_append_value(eval_vec, ssm.conv_state);
                            }
                            if (ssm.ssm_state.ctx != null) {
                                _ = mlx.mlx_vector_array_append_value(eval_vec, ssm.ssm_state);
                            }
                        }
                    }
                    _ = mlx.mlx_eval(eval_vec);
                }
                _ = mlx.mlx_clear_cache();
                if (trace_enabled) eval_ns += prefill_sw.read() - eval_start_ns;

                // THIS chunk's latch, read HERE — before anything persists it,
                // snapshots it, or yields the thread (B0b). It used to be read
                // at the top of the NEXT iteration, and three things ran in the
                // gap, all on KV that Metal had already abandoned (the rule:
                // Metal at the working-set edge returns ZEROS before it aborts):
                //
                //   * `captureSsmCheckpoint` below snapshots the garbage state;
                //   * `write_through_hook` commits it to the SSD tier as a
                //     durable, indexed restorable prefix that nothing removes,
                //     so later requests restore FROM the failure;
                //   * `interleave_hook` -> `checkErrorDecode` is the same
                //     `consumeLatch`, so a co-tenant decode ate the latch and
                //     was blamed for it — after which this prefill saw a clean
                //     latch at the top of the next iteration and answered 200
                //     with zeros.
                //
                // One failure, two wrong answers: a poisoned disk cache and an
                // error charged to an innocent request. The checks at the top of
                // the loop and after it are now redundant for the chunk's own
                // forward and kept only for what the hooks themselves latch.
                try mlx.checkError();

                // Phase 1: snapshot SSM state at stride-aligned boundaries.
                // We snapshot AFTER the eval above so the underlying buffers
                // are realized; the snapshot is just a refcount-share of the
                // already-resident state.
                const abs_end_for_cp2 = end + ssm_cp_offset;
                if (want_ssm_cp and ssm_cp_stride > 0 and abs_end_for_cp2 % ssm_cp_stride == 0) {
                    const cp = try captureSsmCheckpoint(allocator, ctx.ssm_entries.?, abs_end_for_cp2, xfm.s);
                    try ssm_checkpoints.append(allocator, cp);
                    // Keep the buffer bounded — thin the INTERIOR, never the
                    // oldest (#330 follow-up). Drop-oldest survivors cover
                    // only the last `max * stride` tokens, so a 383k prefill's
                    // lowest checkpoint lands past the hot-cache budget and
                    // the commit has no affordable trim point at all. Removal
                    // is O(n) but `n` is tiny (≤ ssm_checkpoint_max).
                    if (options.ssm_checkpoint_max > 0 and
                        ssm_checkpoints.items.len > options.ssm_checkpoint_max)
                    {
                        var dropped = ssm_checkpoints.orderedRemove(
                            transformer_mod.ssmCheckpointDropIndex(ssm_checkpoints.items, cp_thin),
                        );
                        dropped.deinit(allocator);
                    }
                }

                if (vision_chunked) {
                    vision_rows_consumed += countSpliceRows(
                        ids_i32[pos..end],
                        xfm.config.image_token_id,
                        xfm.config.audio_token_id,
                        xfm.config.video_token_id,
                    );
                }
                pos = end;
                n_chunks += 1;
                // Publish progress once per chunk — same cadence discipline as
                // `inflight_generated_tokens` (once per decode tick), never per token.
                if (options.prefill_progress) |p| p.store(@intCast(pos), .monotonic);
                // Mechanism 3: persist this chunk now. The KV bytes for
                // [0, pos) are final (the eval above materialized them), so
                // the SSD tier can index a chunk-aligned prefix that survives
                // a cancel or a kill. Serialization is bounded (one chunk);
                // the file write is off-thread.
                if (options.write_through_hook) |wt| {
                    wt.call(wt.ctx, pos + ssm_cp_offset, ssm_checkpoints.items);
                }
                // The width of the NEXT chunk, re-priced against live memory
                // at this boundary and nowhere else. The placement is AFTER
                // the chunk's `mlx_clear_cache` above, so the probe reads the
                // steady state the next chunk starts from rather than this
                // one's peak, and BEFORE the interleave tick below, so a
                // co-tenant's decode is not attributed to this prefill's
                // pressure. The SSD write-through above stages HOST bytes the
                // probe cannot see in `mlx_get_active_memory`, so it publishes
                // them and `prefillHeadroomNow` subtracts them (S11).
                // Attribution and safety want opposite orderings here, so a
                // step-down commits from this point and a WIDEN is re-priced
                // after the tick (S17). `pos < loop_end` because the last
                // boundary has no next chunk to size.
                //
                // The first chunk always runs the width admission billed —
                // this only ever moves it afterwards, and the admission bill
                // itself never moves.
                var pending_widen: u32 = 0;
                var widen_confirm: ?*const fn (*anyopaque, usize, u32) bool = null;
                var widen_ctx: ?*anyopaque = null;
                if (adapt_chunked and pos < loop_end) {
                    if (options.chunk_width_hook) |hk| {
                        const next_w = hk.call(hk.ctx, ssm_cp_offset + pos, @intCast(cur_chunk), cap_adapt, &adapt_state);
                        if (next_w != 0 and next_w != cur_chunk) {
                            if (next_w < cur_chunk) {
                                // A step-DOWN is the safe direction and takes
                                // effect at once: waiting for the tick is
                                // waiting inside the abort it prevents.
                                commitAdaptiveWidth(&cur_chunk, &adapt_state, next_w);
                            } else {
                                // A WIDEN is held until after the tick (S17).
                                // The confirm is captured HERE, by value, so
                                // this stays the loop body's ONE unwrapping
                                // of the width hook — the g1 loop-order
                                // contract pins that site BEFORE the tick, and
                                // a second unwrapping after it would make that
                                // scan depend on which occurrence it found.
                                pending_widen = next_w;
                                widen_confirm = hk.confirm;
                                widen_ctx = hk.ctx;
                            }
                        }
                    }
                }
                // Yield to the scheduler between chunks — never after the
                // last (the post-prefill decode tick covers that boundary).
                if (pos < loop_end) {
                    if (options.interleave_hook) |hk| hk.call(hk.ctx);
                }
                // The tick just allocated KV and activations for whatever
                // co-tenant slots it hosted — memory the pre-tick probe could
                // not see. Re-price the widen against what is left; a refusal
                // simply keeps the current width.
                if (pending_widen != 0) {
                    if (widen_confirm) |cf| {
                        if (cf(widen_ctx.?, ssm_cp_offset + pos, pending_widen)) {
                            commitAdaptiveWidth(&cur_chunk, &adapt_state, pending_widen);
                        } else {
                            adapt_state.supporting = 0;
                        }
                    }
                }
            }
            // The last chunk's failure has no next iteration to catch it.
            try mlx.checkError();
            // ONE line per request when the width MOVED — the per-transition
            // lines say where and why, this one says what the prefill ran as
            // a whole. A prefill that held its admitted width says nothing.
            if (adapt_state.transitions > 0) {
                log.info("[prefill] adaptive: {d} chunks, width {d}..{d}, {d} change(s)\n", .{
                    n_chunks,
                    adapt_state.width_min,
                    adapt_state.width_max,
                    adapt_state.transitions,
                });
            }

            // Phase 1: always-on snapshot at the post-prefill position
            // (= prefix_len, i.e., prompt_ids.len - 1). The stride loop
            // captures snapshots at [stride, 2*stride, ...]; this final
            // capture covers the most common warm-path case where the next
            // turn's prompt fully matches turn-1's prompt and matched lands
            // at prompt_ids.len. Without this, a stride=256 setup with a
            // 750-token prompt could only restore at position 512 (losing
            // ~234 tokens of potential reuse to the next stride boundary).
            // With it, the cache restores to position 749 (~99% of the
            // prompt) and only the last token + new tail re-forwards.
            // Skipped on `prompt_ids.len == 1` (no prefill chunks ran).
            if (want_ssm_cp and loop_end > 0) {
                // The snapshot sits at the chunk loop's end — snapshot_backoff
                // tokens BEFORE the prompt end, where the next turn's prefix
                // match can actually reach it (the template's generation
                // suffix renders differently in history, so a match always
                // falls a few tokens short of the full prompt).
                const final_abs = loop_end + ssm_cp_offset;
                // Skip if we already captured at this exact position (the
                // chunked loop would have done so when loop_end happens
                // to be a stride multiple).
                const already_have = ssm_checkpoints.items.len > 0 and
                    ssm_checkpoints.items[ssm_checkpoints.items.len - 1].pos == final_abs;
                if (!already_have) {
                    // SSM state is already materialized — the chunked loop
                    // evaluated it at every chunk boundary. The final chunk
                    // may have been a stride-aligned one (already evaluated)
                    // or a partial tail (also evaluated). The snapshot is a
                    // cheap refcount-share.
                    const cp = try captureSsmCheckpoint(allocator, ctx.ssm_entries.?, final_abs, xfm.s);
                    try ssm_checkpoints.append(allocator, cp);
                    if (options.ssm_checkpoint_max > 0 and
                        ssm_checkpoints.items.len > options.ssm_checkpoint_max)
                    {
                        // Same span-preserving thin as the stride capture.
                        var dropped = ssm_checkpoints.orderedRemove(
                            transformer_mod.ssmCheckpointDropIndex(ssm_checkpoints.items, cp_thin),
                        );
                        dropped.deinit(allocator);
                    }
                }
                // One copy of the QSA key history on the latest snap. Stride
                // captures skipped it so a 400k prefill is not 32× the
                // indexer buffer. With the share switch on (default) NOTHING
                // is attached here: the newest snap takes a VIEW of the live
                // buffer at COMMIT (`scheduler.commitSlotIfApplicable` ->
                // `handoffQsaHistoryToLatest`), so the decode holds ONE copy
                // where the materialized attach kept two (3,840 B/tok on
                // qwen4_exp). `MLX_SERVE_QSA_HISTORY_SHARE=0` restores this
                // copy — and `server.statePerTokenBilled` bills it again.
                if (ssm_checkpoints.items.len > 0 and !transformer_mod.qsaHistoryShareEnabled()) {
                    try transformer_mod.attachQsaHistoryToLatest(ssm_checkpoints.items, ctx.ssm_entries.?, xfm.s);
                }
            }
        }

        // Chunked vision: the final-span forward continues the row scatter
        // where the chunk loop left it (an image ending in the final span
        // otherwise re-splices from row 0). One-shot engagement line — the
        // silent-no-op class; tests grep for it per arm.
        if (has_vision) {
            ctx.vision_splice_offset = vision_rows_consumed;
            if (vision_chunked and n_chunks > 1) log.debug("[vision] chunked prefill: {d} chunks, {d} placeholder rows consumed\n", .{ n_chunks, vision_rows_consumed });
        }

        // Process the final span — the last token, plus (under SSM
        // checkpointing) the snapshot-backoff tail held back from the chunk
        // loop. One forward, one weight sweep, logits sliced to the last
        // position so every consumer sees the classic [1, 1, V] shape.
        const tail_len: usize = prompt_ids.len - final_start;
        const last_shape = [_]c_int{ 1, @as(c_int, @intCast(tail_len)) };
        const last_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[final_start]), &last_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(last_input);

        // Drafter (Gemma 4 assistant) needs the post-final-norm hidden as
        // its first-step h_prev — captured here so we don't need a second
        // forward at the start of `nextDrafter`. `forwardWithCapture`
        // captures the LAST position regardless of span length. When the
        // span holds backed-off tokens AND MTP is active, the head's history
        // must also cover them (a hole right before the generation point is
        // acceptance-critical), so capture ALL positions and append.
        const drafter_active = options.drafter_enabled and options.drafter != null;
        const pld_active = options.pld_enabled;
        const need_capture = drafter_active or mtp_active;
        var captured_hidden: mlx.mlx_array = mlx.mlx_array_new();
        var has_captured_hidden = false;
        const last_start_ns = if (trace_enabled) prefill_sw.read() else 0;
        const tail_mtp_capture = mtp_active and tail_len > 1;
        var tail_hidden_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail_hidden_all);
        // DFlash context for the tail span (rides ctx into all three forward
        // arms below; the forwardWithCapture* wrappers only override
        // capture_hidden/_all).
        if (dflash_active) {
            for (dfl_out_buf) |*a| a.* = mlx.mlx_array_new();
            ctx.capture_layers = &dfl_cl;
        }
        const raw_logits = if (tail_mtp_capture) blk: {
            has_captured_hidden = true;
            break :blk try xfm.forwardWithCaptureAll(&ctx, last_input, &captured_hidden, &tail_hidden_all);
        } else if (need_capture) blk: {
            has_captured_hidden = true;
            break :blk try xfm.forwardWithCapture(&ctx, last_input, &captured_hidden);
        } else try xfm.forwardWith(&ctx, last_input);
        // History entries for the held-back span: (hidden[j], token[j+1]) for
        // j in [final_start, prefix_len) — same pairing as the chunk loop.
        // The last row's pair (hidden[last], t1) is appended by the first
        // nextMtp round, exactly as before.
        if (tail_mtp_capture) {
            if (!mtp_history_started) {
                std.debug.assert(mtp_cache.?.step() == 0);
                mtp_position_base = ssm_cp_offset + final_start;
                mtp_history_started = true;
            }
            var tail_hist_hidden = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tail_hist_hidden);
            const all_shape = mlx.getShape(tail_hidden_all);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ all_shape[0], @intCast(tail_len - 1), all_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&tail_hist_hidden, tail_hidden_all, &start, 3, &stop, 3, &strides, 3, xfm.s));
            const tail_mrope_ctx: ?mtp_mod.MropeContext = if (ctx.mrope_pos) |positions| .{
                .pos = positions,
                .total = ctx.mrope_total,
                .delta = ctx.mrope_delta,
                .base = mtp_position_base,
            } else null;
            try options.mtp.?.appendHistory(
                xfm,
                &mtp_cache.?,
                prompt_ids[final_start + 1 .. prompt_ids.len],
                tail_hist_hidden,
                @intCast(mtp_cache.?.step()),
                tail_mrope_ctx,
                allocator,
            );
        }
        if (dflash_active) {
            ctx.capture_layers = null;
            try dflash_mod.appendContext(options.dflash.?, &dflash_ctx.?, dfl_out_buf, ssm_cp_offset + final_start);
            for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
        }
        // Slice to the last position when the span is longer than one token,
        // so downstream sampling/grammar paths see the classic shape.
        const logits = if (tail_len == 1) raw_logits else blk: {
            defer _ = mlx.mlx_array_free(raw_logits);
            const lshape = mlx.getShape(raw_logits);
            var sliced = mlx.mlx_array_new();
            const start = [_]c_int{ 0, lshape[1] - 1, 0 };
            const stop = [_]c_int{ lshape[0], lshape[1], lshape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&sliced, raw_logits, &start, 3, &stop, 3, &strides, 3, xfm.s));
            break :blk sliced;
        };
        if (trace_enabled) {
            const last_ns = prefill_sw.read() - last_start_ns;
            const total_ns = prefill_sw.read();
            const ms = std.time.ns_per_ms;
            std.debug.print(
                "  [prefill-trace] tokens={d} chunks={d} chunk_size={d} chunk_widths={d}..{d} chunked={d}ms eval={d}ms last_token={d}ms total={d}ms{s}{s}\n",
                .{
                    prompt_ids.len,
                    n_chunks,
                    PREFILL_CHUNK,
                    adapt_state.width_min,
                    adapt_state.width_max,
                    chunked_ns / ms,
                    eval_ns / ms,
                    last_ns / ms,
                    total_ns / ms,
                    if (need_capture) " [capture-hidden]" else "",
                    if (pld_active) " [pld]" else "",
                },
            );
        }
        errdefer if (has_captured_hidden) {
            _ = mlx.mlx_array_free(captured_hidden);
        };

        // Attach the SSM-checkpoint buffer to whichever Generator variant
        // we're about to return. Clears the local list so the errdefer above
        // doesn't double-free. All four init paths below call this once
        // before returning their Generator.
        const attachCp = struct {
            fn f(g: *Generator, list: *std.ArrayList(SSMCheckpoint), a: std.mem.Allocator) void {
                g.ssm_checkpoints = list.*;
                g.ssm_checkpoint_alloc = a;
                list.* = std.ArrayList(SSMCheckpoint).empty;
            }
        }.f;

        // Constrained generation skips the lazy first-sample fast path: we cannot
        // sample the first token until we have applied the grammar mask, and we
        // cannot pipeline because grammar advancement depends on the realized id.
        if (sampling.constraint != null) {
            // Grammar-constrained requests never speculate; release the MTP
            // history cache / DFlash context if dispatch enabled them anyway.
            if (mtp_cache) |*mc| {
                mc.deinit();
                mtp_cache = null;
            }
            if (dflash_ctx) |*dc| {
                dc.deinit();
                dflash_ctx = null;
            }
            var gen = Generator{
                .xfm = xfm,
                .model_has_mtp = options.model_has_mtp,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = 0,
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = sampling,
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
            };
            gen.pending_logits = logits;
            gen.has_pending_logits = true;
            attachCp(&gen, &ssm_checkpoints, allocator);
            return gen;
        }

        // Drafter / PLD-v2 / MTP path: sample synchronously and DO NOT
        // pre-forward the sampled token. The first nextDrafter / nextPld /
        // nextMtp call needs the cache at exactly prompt_len (last prompt
        // token forwarded; first sampled token deferred). The lazy
        // pre-forward path below would over-advance the cache and corrupt
        // every verify forward.
        if (drafter_active or pld_active or mtp_active or dspark_active or dflash_active) {
            const sample_lazy = sampleTokenLazy(logits, sampling, s);
            _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);

            const mtp_cost_profile: mtp_mod.MtpCostProfile = if (mtp_active)
                options.mtp.?.costProfile(xfm)
            else
                .generic;
            const dflash_bs: u32 = if (dflash_active)
                (if (options.dflash_block_size > 0) options.dflash_block_size else options.dflash.?.config.block_size)
            else if (options.dflash_block_size > 0)
                options.dflash_block_size
            else
                0;
            var gen = Generator{
                .xfm = xfm,
                .model_has_mtp = options.model_has_mtp,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = blk: {
                    var sp = sampling;
                    sp.draw = 1; // t1 above was draw 0
                    break :blk sp;
                },
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (need_capture) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = need_capture,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
                .pld_enabled = pld_active,
                .dspark_enabled = dspark_active,
                .dspark_stochastic = dspark_stochastic,
                .drafter = if (drafter_active) options.drafter else null,
                .drafter_block_size = options.drafter_block_size,
                .dflash = if (dflash_active) options.dflash else null,
                .dflash_ctx = dflash_ctx,
                .dflash_block_size = dflash_bs,
                .dflash_chooser = if (dflash_active and dflashChooserEnabled())
                    round_cost.WidthChooser.init(@max(dflash_bs, 2) - 1, options.dflash.?.config.block_size -| 1)
                else
                    null,
                .dflash_min_accepted_per_round = options.dflash_min_accepted_per_round,
                .mtp = if (mtp_active) options.mtp else null,
                .mtp_cache = mtp_cache,
                .mtp_position_base = mtp_position_base,
                .mtp_depth = resolveMtpDepthCapForProfile(options.mtp_depth, mtp_cost_profile),
                .mtp_depth_free = if (xfm.mtp_depth_free != 0) xfm.mtp_depth_free else mtpDepthCapFree(options.mtp_depth),
                .mtp_ev_costs = mtpEvCosts(mtp_cost_profile),
                // Start at depth 1 and climb with evidence: the cheap depth
                // is the safe default (1.11x on cold/creative content), and
                // hot workloads promote within ~8 rounds.
                .mtp_depth_current = 1,
            };
            mtp_cache = null; // ownership transferred to the Generator
            dflash_ctx = null; // ownership transferred to the Generator
            // pending_logits/pending_token left empty — the lazy pipeline is
            // skipped under PLD / drafter / MTP. The speculative `next*` paths
            // drive every subsequent step with predictable cache offset.
            attachCp(&gen, &ssm_checkpoints, allocator);
            return gen;
        }

        // Phase 2: scheduler-managed slots ask init to sample t1 synchronously
        // and skip the lazy pre-forward. Cache lands at prompt_len with t1 NOT
        // in cache — matches `forwardBatchedDecode`'s expectation and the
        // PLD / drafter init path's invariant. Generator.next's transition
        // shim handles the bootstrap on the first decode tick.
        if (options.skip_lazy_preforward) {
            const sample_lazy = sampleTokenLazy(logits, sampling, s);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);
            // t1's own distribution comes from the prefill's final forward and
            // is invisible to the decode loop — read it here or the first
            // returned token has no logprobs and the array shifts by one.
            const first_lp = try firstTokenLogprobs(allocator, logits, @intCast(first_val), options.logprobs_n, s);
            _ = mlx.mlx_array_free(logits);

            var gen = Generator{
                .pending_logprob = first_lp,
                .xfm = xfm,
                .model_has_mtp = options.model_has_mtp,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = blk: {
                    var sp = sampling;
                    sp.draw = 1; // t1 above was draw 0
                    break :blk sp;
                },
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
            };
            attachCp(&gen, &ssm_checkpoints, allocator);
            return gen;
        }

        // Regular path: sample first token lazily, then build the next forward pass
        const lazy_token = sampleTokenLazy(logits, sampling, s);

        const next_logits = try lazyForward(xfm, &ctx, lazy_token);

        // Async-eval the decode pipeline (single-token graphs, much smaller)
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lazy_token);
            _ = mlx.mlx_vector_array_append_value(eval_vec, next_logits);
            _ = mlx.mlx_async_eval(eval_vec);
        }

        // Sync to get the first token value
        try mlx.check(mlx.mlx_array_eval(lazy_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
        _ = mlx.mlx_array_free(lazy_token);
        // See the skip_lazy_preforward branch: t1's distribution lives only in
        // the prefill's own output, so it is read here, against the id that was
        // actually drawn.
        const first_lp = try firstTokenLogprobs(allocator, logits, @intCast(val), options.logprobs_n, s);
        _ = mlx.mlx_array_free(logits);

        var gen = Generator{
            .pending_logprob = first_lp,
            .xfm = xfm,
            .model_has_mtp = options.model_has_mtp,
            .ctx = ctx,
            .tok = tok,
            .next_token_id = @intCast(val),
            .step = 0,
            .max_tokens = max_tokens,
            .sampling = blk: {
                var sp = sampling;
                sp.draw = 1; // t1 above was draw 0
                break :blk sp;
            },
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = 0,
            .finish_reason = "length",
            .done = false,
            .eos_token_ids = eos_token_ids,
            .generated_ids = std.ArrayList(u32).empty,
            .timeout_ns = 0,
            .timer = io_util.Stopwatch.init(io),
            .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
            .has_last_hidden = has_captured_hidden,
            .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
            .prompt_ids_owned = prompt_owned,
            .prompt_ids_alloc = allocator,
        };

        gen.pending_logits = next_logits;
        gen.has_pending_logits = true;

        attachCp(&gen, &ssm_checkpoints, allocator);
        return gen;
    }

    /// The ONE place `step` and `completion_tokens` advance — every decode path
    /// (plain, constrained, PLD, drafter, MTP, batched) routes through here.
    ///
    /// Advancing them by hand is how three paths — `nextDrafter`, `nextMtp` and
    /// `scheduler.runBatchedDecodeTick` — ended up never calling
    /// `mlx_clear_cache()` at all (issue #110). MLX parks freed buffers in a
    /// size-keyed pool instead of returning them to the OS, so a decode path
    /// with no clear ratchets the process footprint while `active_bytes` stays
    /// flat: the reporter's process reached 81 GB with the panel reading 19.6.
    /// A `-mtp` checkpoint on a dense trunk defaults straight onto one of them.
    /// A source-scan test pins that no new path can reintroduce the hole.
    pub fn advanceStep(self: *Generator, n: u32) void {
        self.completion_tokens += n;
        self.step += n;
        if (shouldClearAllocatorCache(self.step, self.last_cache_clear_step, CACHE_CLEAR_INTERVAL)) {
            _ = mlx.mlx_clear_cache();
            self.last_cache_clear_step = self.step;
        }
    }

    pub fn deinit(self: *Generator, allocator: std.mem.Allocator) void {
        if (self.last_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.pending_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        if (self.has_pending_token) {
            _ = mlx.mlx_array_free(self.pending_token);
            self.has_pending_token = false;
        }
        if (self.has_last_hidden) {
            _ = mlx.mlx_array_free(self.last_hidden);
            self.has_last_hidden = false;
        }
        if (self.prompt_ids_alloc) |a| {
            a.free(self.prompt_ids_owned);
            self.prompt_ids_owned = &.{};
            self.prompt_ids_alloc = null;
        }
        if (self.mtp_cache) |*mc| {
            mc.deinit();
            self.mtp_cache = null;
        }
        if (self.dflash_ctx) |*dc| {
            dc.deinit();
            self.dflash_ctx = null;
        }
        if (self.mtp_hist_stash) |*st| {
            st.deinit();
            self.mtp_hist_stash = null;
        }
        if (self.mtp_pre_draft) |*pd| {
            pd.deinit(allocator);
            self.mtp_pre_draft = null;
        }
        // Publish the EV surface for the next request when the experimental
        // cross-request seed is explicitly enabled.
        // Only healthy runs qualify — a runtime-disabled or barely-sampled
        // run would poison the next request's plans (inference thread only,
        // same discipline as every other head-state write).
        if (self.mtp) |head| {
            // A forced-depth run never consulted the EV plan, so its surface
            // is not one the controller chose — publishing it would hand a
            // later ordinary request a diagnostic's numbers.
            if (mtpAdaptiveEnabled() and mtpEvSeedEnabled() and mtpForcedDepth() == null and
                !self.spec_disabled_runtime and self.mtp_attempted >= 8)
            {
                head.setEvSeed(self.mtp_ev_accept, self.mtp_ev_m_lo_prev);
            }
        }
        // Free any SSM checkpoints the caller didn't claim. Each layer-slice
        // was allocated by `ssm_checkpoint_alloc` (= the allocator passed to
        // `initWithOptions`), so we use that one. The ArrayList itself was
        // also created with that allocator.
        if (self.ssm_checkpoint_alloc) |a| {
            for (self.ssm_checkpoints.items) |*cp| cp.deinit(a);
            self.ssm_checkpoints.deinit(a);
            self.ssm_checkpoints = std.ArrayList(SSMCheckpoint).empty;
            self.ssm_checkpoint_alloc = null;
        } else {
            // Defensive: if init never set it, the list is empty — but the
            // backing ArrayList state may still need a deinit call. Use the
            // passed allocator as a fallback.
            self.ssm_checkpoints.deinit(allocator);
            self.ssm_checkpoints = std.ArrayList(SSMCheckpoint).empty;
        }
        self.generated_ids.deinit(allocator);
    }

    /// Transfer ownership of accumulated SSM checkpoints to the caller.
    /// Returns an owned slice; caller must free each `SSMCheckpoint` via
    /// `cp.deinit(allocator)` and the slice itself via `allocator.free`,
    /// where `allocator` is the same one passed to `initWithOptions`.
    /// After return, `ssm_checkpoints` is empty and the Generator owns
    /// nothing related to checkpoints.
    pub fn takeSsmCheckpoints(self: *Generator) []SSMCheckpoint {
        const a = self.ssm_checkpoint_alloc orelse return &[_]SSMCheckpoint{};
        const out = self.ssm_checkpoints.toOwnedSlice(a) catch return &[_]SSMCheckpoint{};
        return out;
    }

    /// Legacy→batched transition (scheduler.runBatchedDecodeTick): consume
    /// the lazy pipeline state so the slot can join a batched tick. The
    /// legacy pipelined decode keeps a lookahead token ALREADY FORWARDED
    /// into the KV cache (`pending_token` / `next_token_id`) plus
    /// `pending_logits` for the position after it. Dropping that state and
    /// re-forwarding `next_token_id` would append a duplicate position to
    /// the cache and re-emit an already-emitted token — corrupting every
    /// stream whose slot enters a batch mid-generation
    /// (tests/test_batched_transition.sh).
    ///
    /// Returns the token to emit this step (the pipelined lookahead), or
    /// null when generation stopped (`checkStop`: EOS / pad-run /
    /// max_tokens / timeout — `finish_reason` is set). On return:
    /// `next_token_id` is sampled but NOT in the cache and pending state is
    /// empty — exactly the batched-tick entry invariant.
    pub fn drainPipelineForBatch(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        try self.resolvePendingToken();
        if (try self.checkStop()) {
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
            return null;
        }
        // Both pipeline shapes (fresh-from-prefill and post-`next()` fast
        // path) carry pending_logits alongside the in-cache lookahead; a
        // lookahead without logits would force a re-forward of an in-cache
        // token, which is the corruption this method exists to prevent.
        if (!self.has_pending_logits) return error.MissingPendingLogits;

        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        const lazy = self.sampleLazy(step_logits);
        _ = mlx.mlx_array_free(step_logits);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);
        self.next_token_id = @intCast(val);
        return token;
    }

    /// Resolve the deferred pending token: eval the lazy array and extract the u32 value.
    /// This is called at the START of each iteration, giving the GPU maximum time
    /// to compute since the async_eval at the END of the previous iteration.
    /// The ONE lazy sampler for a slot's own draws: advances the seed draw index.
    pub fn sampleLazy(self: *Generator, logits: mlx.mlx_array) mlx.mlx_array {
        defer self.sampling.draw +%= 1;
        return sampleTokenLazy(logits, self.sampling, self.xfm.s);
    }

    fn resolvePendingToken(self: *Generator) !void {
        if (!self.has_pending_token) return;
        try mlx.check(mlx.mlx_array_eval(self.pending_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, self.pending_token));
        _ = mlx.mlx_array_free(self.pending_token);
        self.has_pending_token = false;
        self.next_token_id = @intCast(val);
    }

    const DrainResult = union(enum) {
        /// No pending pipeline state — the spec entry invariant already holds.
        already_clean,
        /// One token was emitted while draining; caller returns it this step.
        drained: u32,
        /// The drained token hit a stop condition — generation is over.
        stopped,
        /// Unexpected half-state; do not re-enable speculation.
        stay_disabled,
    };

    /// Transition from the pipelined `next()` state back to the spec-decode
    /// entry invariant (next_token_id known but NOT in cache, no pending
    /// state). The pipeline holds `pending_token` (lazy, its forward already
    /// in the cache) and `pending_logits` (logits for the position after it):
    /// resolving the token, emitting it, and sampling its successor from
    /// `pending_logits` WITHOUT forwarding lands exactly on the invariant.
    /// One sync. Also handles the shim-seeded state (`pending_logits` only).
    fn drainPipelineForSpec(self: *Generator, allocator: std.mem.Allocator) !DrainResult {
        if (!self.has_pending_logits) {
            if (!self.has_pending_token) return .already_clean;
            // pending_token without pending_logits never occurs in the
            // pipelined state machine; bail rather than risk the invariant.
            return .stay_disabled;
        }
        try self.resolvePendingToken();
        if (try self.checkStop()) return .stopped;
        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        // Same hole as `mtpSerialCaptureTick` (N17) and the site that one was
        // copied from: with the latch these checks RETURN instead of ending
        // the process, so the handle needs an owner on the error path.
        const val: i32 = blk: {
            const lazy = self.sampleLazy(step_logits);
            _ = mlx.mlx_array_free(step_logits);
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
            break :blk v;
        };
        self.next_token_id = @intCast(val);
        return .{ .drained = token };
    }

    /// Result of one `nextPld` step. Yields 1..=(1+max_draft_len) tokens.
    /// Caller owns `tokens` (must `allocator.free` it).
    pub const PldStepResult = struct {
        /// Tokens to emit this step (always at least the already-decided t1).
        /// On a full-accept, contains [t1, ...all_drafts]. On partial accept j,
        /// contains [t1, draft[0..j]] (the corrected fallback is stored as the
        /// generator's pending `next_token_id`, NOT included here — same
        /// "pending becomes next-step's first" convention as `nextDrafter`).
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (not counting t1). 0..=draft_len.
        accepted_tokens: u32,
        /// Whether n-gram lookup found a candidate this step. False means PLD
        /// degraded to a single regular forward (no speculative work done).
        used_lookup: bool,
    };

    /// PLD draft+verify decode step. The draft comes from an n-gram lookup
    /// over `prompt_ids_owned ++ generated_ids`, NOT a model call — that's
    /// what makes PLD model-agnostic and cheap.
    ///
    /// `key_len` is the n-gram size used for matching (default 3). `draft_len`
    /// is the maximum number of speculative tokens to verify per step (default
    /// 5). Both are clamped to safe upper bounds internally.
    ///
    /// Returns `null` only when generation is already done. When no n-gram
    /// match exists (cold start, novel output), falls back to the regular
    /// `next()` path and returns a single-token result with `used_lookup=false`.
    /// DeepSeek-V4 DSpark step (the arch's OWN block-parallel spec decode).
    /// Entry/exit share the v2 spec invariant: module state = prompt +
    /// emitted positions, t1 = `next_token_id` NOT in state, pending empty
    /// (init's spec branch establishes it; every exit restores it). The
    /// heavy lifting — draft, batched verify, snapshot rollback — lives in
    /// `deepseek_v4.dsparkRound`; this wrapper only keeps the Generator's
    /// bookkeeping (generated_ids, step accounting, the shell cache.step
    /// that forwardDsv4WithImpl keys fresh-vs-decode on) in sync.
    pub fn nextDspark(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.done) return null;
        if (!self.dspark_enabled) {
            // Same defensive fallback as nextPld's disarmed arm: the
            // dispatching caller's flag alone must never run a draft.
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
        }
        if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;
        const mdl = self.xfm.dsv4.?;
        const t1 = self.next_token_id;
        const accepted_cap = capAcceptedForTokenBudget(
            std.math.maxInt(u32),
            self.completion_tokens,
            self.max_tokens,
        );
        var round = if (self.dspark_stochastic)
            try self.dsparkStochasticRound(allocator, mdl, t1, accepted_cap)
        else
            try dsv4_mod.dsparkRound(mdl, allocator, &mdl.dec_state.?, t1, accepted_cap);
        errdefer round.deinit(allocator);
        // dsparkRound advanced the module state — mirror it on the shell
        // cache verbatim so a later serial fallback (or the fresh-request
        // check keying on step==0) sees a consistent position. Generator.step
        // itself moves through advanceStep below (the clear-cadence clock).
        self.ctx.cache.step = mdl.dec_state.?.n;
        self.dspark_attempted += 1;
        self.dspark_accepted_tokens += round.accepted;
        try self.generated_ids.appendSlice(allocator, round.tokens);
        self.advanceStep(@intCast(round.tokens.len));
        self.next_token_id = round.next_token;
        // tokens ownership transfers to the caller (scheduler frees).
        return DrafterStepResult{ .tokens = round.tokens, .accepted_tokens = round.accepted };
    }

    /// One stochastic DSpark round: dsv4's own greedy stage draft (a one-hot
    /// proposal) verified with the MTP acceptance machinery — filtered target
    /// probs at EVERY verify position (`probsAllPositions`, the request's own
    /// temperature/top-k/top-p), accept draft k with prob `min(1, p_k)`,
    /// first reject at `a` corrected from `normalize(max(p_a − onehot, 0))`,
    /// full accept sampled from the bonus row — corrections pre-sampled in
    /// ONE batched graph (`mtpBatchedAcceptGraph`, one-hot arm) so the round
    /// pays ONE bounded sync. The output distribution equals serial sampling
    /// (the toy-vocab exactness test's invariant), and the correction always
    /// derives from the ORIGINAL verify logits at the acceptance point — the
    /// house partial-accept invariant in sampled form.
    fn dsparkStochasticRound(self: *Generator, allocator: std.mem.Allocator, mdl: *dsv4_mod.Dsv4Model, t1: u32, accepted_cap: u32) !dsv4_mod.DsparkRound {
        const s = self.xfm.s;
        var pending = try dsv4_mod.dsparkBegin(mdl, allocator, &mdl.dec_state.?, t1);
        defer pending.deinit();
        const b: u32 = @intCast(pending.b);

        // Filtered target probs over every verify row: [b+1, V] → [1, b+1, V].
        const vshape = [_]c_int{ 1, @intCast(pending.b + 1), @intCast(mdl.vocab) };
        var vl3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vl3);
        try mlx.check(mlx.mlx_reshape(&vl3, pending.vl_g, &vshape, 3, s));
        const probs_all = try probsAllPositions(vl3, self.sampling, s);
        defer _ = mlx.mlx_array_free(probs_all);

        var accepted: u32 = 0;
        var next_token: u32 = undefined;
        if (b == 0) {
            // The confidence gate submitted nothing: this round verifies t1
            // alone and row 0 IS the bonus row — sample the next trunk token
            // from it directly.
            var log_p = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(log_p);
            try mlx.check(mlx.mlx_log(&log_p, probs_all, s));
            const null_key = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(null_key);
            var sampled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sampled);
            try mlx.check(mlx.mlx_random_categorical(&sampled, log_p, -1, null_key, s));
            var samp_i = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(samp_i);
            try mlx.check(mlx.mlx_astype(&samp_i, sampled, .int32, s));
            try mlx.check(mlx.mlx_array_eval(samp_i));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, samp_i));
            next_token = @intCast(v);
        } else {
            // [1] int32 arrays of the draft ids (already realized host values
            // — dsv4 drafts synchronously, unlike the MTP head's lazy chain).
            var draft_arrs: [16]mlx.mlx_array = undefined;
            var n_arrs: usize = 0;
            defer for (draft_arrs[0..n_arrs]) |arr| {
                _ = mlx.mlx_array_free(arr);
            };
            const idshape = [_]c_int{1};
            for (0..pending.b) |k| {
                const idv = [_]i32{@intCast(pending.verify[k + 1])};
                draft_arrs[k] = mlx.mlx_array_new_data(&idv, &idshape, 1, .int32);
                n_arrs += 1;
            }
            var bg = try mtpBatchedAcceptGraph(probs_all, draft_arrs[0..pending.b], null, b, s);
            defer bg.deinit();

            // ONE bounded sync: the accept vector + pre-sampled corrections
            // (and the whole verify graph beneath them) in one batched eval.
            {
                const ev = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(ev);
                _ = mlx.mlx_vector_array_append_value(ev, bg.accept_p);
                _ = mlx.mlx_vector_array_append_value(ev, bg.corr_samples);
                try mlx.check(mlx.mlx_async_eval(ev));
            }
            try mlx.check(mlx.mlx_array_eval(bg.accept_p));
            const p_data = mlx.mlx_array_data_float32(bg.accept_p) orelse return error.MlxArrayDataNull;
            while (accepted < b) {
                const accept_prob: f32 = @min(1.0, p_data[accepted]);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
            // Pick the correction at the request-budget boundary and let
            // dsparkFinish roll module-owned state back to that same point.
            accepted = @min(accepted, accepted_cap);
            try mlx.check(mlx.mlx_array_eval(bg.corr_samples));
            const corr = mlx.mlx_array_data_int32(bg.corr_samples) orelse return error.MlxArrayDataNull;
            next_token = @intCast(corr[accepted]);
        }
        pending.lapVerify(mdl);

        const round = try dsv4_mod.dsparkFinish(mdl, allocator, &mdl.dec_state.?, &pending, accepted, next_token);
        dsv4_mod.dsparkObserve(mdl, round.phases);
        return round;
    }

    pub fn nextPld(
        self: *Generator,
        allocator: std.mem.Allocator,
        draft_len: u32,
        key_len: u32,
    ) !?PldStepResult {
        if (self.done) return null;
        // Init never armed PLD for this generator (the deepseek_v4 chokepoint
        // guard, or a caller that simply didn't ask). The dispatching caller's
        // flag alone must never put a verify forward through the trunk — on
        // dsv4 the verify appends draft tokens into module-owned state that
        // the KV snapshot rollback cannot restore (the 2026-07-31 mangled-DSML
        // corruption). Unlike `spec_disabled_runtime` below this is permanent:
        // no re-enable check can ever resurrect it.
        if (!self.pld_enabled) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return PldStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
                .used_lookup = false,
            };
        }
        // Release-enforced guard (issue #97): PLD cannot honor a grammar
        // constraint or logprobs. These were std.debug.asserts, compiled out in
        // ReleaseFast; fail loud instead of streaming off-schema output if a
        // dispatch bug ever routes such a request here.
        if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

        // Runtime acceptance gate: if a prior step set the flag, fall back
        // to the regular `next()` path. Under v2, PLD's exit invariant has
        // `t1 NOT in cache` (matches `nextDrafter`) — `next()`'s transition
        // shim seeds `pending_logits` synchronously via `forward([t1])` when
        // it sees `!has_pending_logits and !has_pending_token`. So the
        // hand-off works even though pending state is empty.
        if (self.spec_disabled_runtime) {
            self.disabled_steps += 1;
            // Periodic re-enable check: when the generated tail turns
            // repetitive (file/tool echo after a novel preamble), PLD pays
            // again. Drain the `next()` pipeline back to the spec entry
            // invariant; the drained token (if any) is this step's emit and
            // speculation resumes on the following call.
            if (self.disabled_steps % SPEC_REENABLE_INTERVAL == 0) reenable: {
                const gen = self.generated_ids.items;
                const prompt_toks = self.prompt_ids_owned;
                const committed_check = try allocator.alloc(u32, prompt_toks.len + gen.len);
                defer allocator.free(committed_check);
                @memcpy(committed_check[0..prompt_toks.len], prompt_toks);
                @memcpy(committed_check[prompt_toks.len..], gen);
                //if (log.isDebug()) {
                //    const dbg_frac = pld_index.tailMatchFraction(committed_check, @min(SPEC_REENABLE_WINDOW, gen.len), 3);
                //    log.debug("  pld re-enable check: disabled_steps={d} gen={d} tail_match={d:.2}\n", .{ self.disabled_steps, gen.len, dbg_frac });
                //}
                if (!specShouldReenable(committed_check, gen.len)) break :reenable;
                switch (try self.drainPipelineForSpec(allocator)) {
                    .stay_disabled => break :reenable,
                    .stopped => return null,
                    .already_clean => {
                        //log.info("  pld=re-enabled (generated tail turned repetitive after {d} disabled steps)\n", .{self.disabled_steps});
                        self.spec_disabled_runtime = false;
                        self.disabled_steps = 0;
                        self.yield_steps = 0;
                        self.yield_accepted = 0;
                        // Invariant already holds — fall through to the
                        // enabled flow below in this same call.
                    },
                    .drained => |drained_tok| {
                        //log.info("  pld=re-enabled (generated tail turned repetitive after {d} disabled steps)\n", .{self.disabled_steps});
                        self.spec_disabled_runtime = false;
                        self.disabled_steps = 0;
                        self.yield_steps = 0;
                        self.yield_accepted = 0;
                        const tokens = try allocator.alloc(u32, 1);
                        tokens[0] = drained_tok;
                        return PldStepResult{
                            .tokens = tokens,
                            .accepted_tokens = 0,
                            .used_lookup = false,
                        };
                    },
                }
            }
            if (self.spec_disabled_runtime) {
                const tok_opt = try self.next(allocator);
                if (tok_opt == null) return null;
                const tokens = try allocator.alloc(u32, 1);
                tokens[0] = tok_opt.?;
                return PldStepResult{
                    .tokens = tokens,
                    .accepted_tokens = 0,
                    .used_lookup = false,
                };
            }
        }

        const xfm = self.xfm;
        const s = xfm.s;

        // ── INVARIANT going INTO this call (mirrors `nextDrafter`) ──
        //   cache.step = prompt_len + tokens_emitted   (NOT + 1)
        //   t1 = next_token_id (= "this step's first emit"); NOT in cache yet.
        //   pending_logits / pending_token are empty (init's PLD branch and
        //   every nextPld exit leave them empty under v2).
        //
        // Cold path (no n-gram match): forward([t1]) length 1 advances cache
        // by 1, produces logits at position +1 → sample lookahead → emit t1,
        // set next_token_id = lookahead. Loses A's lazy pipeline overlap on
        // cold steps; the prompt-time n-gram gate disables PLD on novel
        // content where cold-path dominates.
        //
        // Verify path: input = `[t1, draft[0..m-1]]` length 1+m. Walk
        // verify_logits[i] vs draft[i] for i=0..m-1; full accept commits 1+m
        // tokens and exits with cache at prompt_len + TE_new (no post-step
        // forward — that is the per-step saving over v1).
        const t1: u32 = self.next_token_id;

        // Cap draft_len so the verify forward stays a small fixed cost.
        const max_draft: u32 = @min(draft_len, 15);
        const klen: u32 = @max(@as(u32, 1), key_len);

        // ── Phase 1: Lookup ──
        // committed = prompt + generated_ids + [t1]. Key = trailing klen tokens
        // (ends at t1). The lookup returns candidates for "what comes after t1".
        const prompt = self.prompt_ids_owned;
        const generated = self.generated_ids.items;
        const total_len = prompt.len + generated.len + 1;

        var committed = try allocator.alloc(u32, total_len);
        defer allocator.free(committed);
        @memcpy(committed[0..prompt.len], prompt);
        @memcpy(committed[prompt.len .. prompt.len + generated.len], generated);
        committed[total_len - 1] = t1;

        var draft_slice: ?[]const u32 = null;
        if (klen <= total_len - 1) {
            const key_start = total_len - klen;
            const key = committed[key_start..total_len];
            const lookup = pld_index.PldLookup{ .committed = committed, .key_len = klen };
            draft_slice = lookup.findMatch(key, max_draft);
        }
        if (draft_slice) |d| {
            if (d.len == 0) draft_slice = null;
        }

        const stochastic = self.sampling.temperature > 0.01;

        // ── Phase 2: Cold path (no n-gram match) ──
        // Forward([t1]) length 1: cache.step += 1, produces logits at that
        // position. Sample the lookahead, emit t1, set next_token_id =
        // lookahead. Cache exits at prompt_len + TE_new where TE_new = TE + 1.
        if (draft_slice == null) {
            const t1_i32: i32 = @intCast(t1);
            const t1_shape = [_]c_int{ 1, 1 };
            const t1_input = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(t1_input);

            const cold_logits = try xfm.forwardWith(&self.ctx, t1_input); // cache.step += 1
            defer _ = mlx.mlx_array_free(cold_logits);

            const lazy = self.sampleLazy(cold_logits);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var lv: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&lv, lazy));
            _ = mlx.mlx_array_free(lazy);
            const new_t1: u32 = @intCast(lv);

            try self.generated_ids.append(allocator, t1);
            self.advanceStep(1);
            self.next_token_id = new_t1;

            // Yield gate: cold steps pay the unpipelined forward; if the
            // workload isn't yielding accepted drafts to pay for it, fall
            // back to the pipelined `next()` (re-enable check above can
            // bring PLD back when the tail turns repetitive).
            self.yield_steps += 1;
            if (yieldGateShouldDisable(self.yield_steps, self.yield_accepted)) {
                log.info(
                    "  pld=disabled (yield gate: {d} drafted tokens over {d} steps < {d:.2}/step)\n",
                    .{ self.yield_accepted, self.yield_steps, YIELD_GATE_MIN_YIELD },
                );
                self.spec_disabled_runtime = true;
                self.disabled_steps = 0;
            }

            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = t1;
            return PldStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
                .used_lookup = false,
            };
        }

        const draft = draft_slice.?;
        const m: u32 = @intCast(draft.len);

        // ── Phase 3: Snapshot KV + per-layer SSM + moe_seq_offset + DSV4 ──
        // Cache enters at cache.step = prompt_len + TE.
        //
        // The snapshots below are the FALLBACK rollback path (pure-attention,
        // Mamba2, LFM2). On a GatedDeltaNet trunk the verify forward instead
        // CAPTURES per-position SSM/conv state (capture_ssm_seq), and partial
        // accept rolls back by slicing that capture + truncating the KV cache —
        // no re-forward of the accepted prefix, which on this arch re-runs the
        // expensive 48-layer sequential recurrence.
        //
        // The KV-cache truncate length is anchored on `moe_seq_offset`, NOT
        // `cache.step`: on a GDN trunk layer 0 is a linear-attention layer that
        // never calls `cache.update`, and `cache.step` only advances under
        // `if (layer == 0)` — so it stays stale (~0) for this family. The
        // full-attention KV entries instead track `moe_seq_offset` (both advance
        // by seq_len per forward), so that is the real KV length to roll back to.
        var kv_snap = try self.ctx.cache.snapshot();
        defer kv_snap.deinit();
        var ssm_snaps: ?[]SSMCacheEntrySnapshot = null;
        defer if (ssm_snaps) |snaps| {
            for (snaps) |*sn| ssmSnapshotDeinit(sn);
            xfm.allocator.free(snaps);
        };
        if (self.ctx.ssm_entries) |entries| {
            const out = try xfm.allocator.alloc(SSMCacheEntrySnapshot, entries.len);
            for (entries, 0..) |*entry, i| out[i] = ssmSnapshot(entry);
            ssm_snaps = out;
        }
        const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

        // ── Phase 4: Verify forward `[t1, draft[0..m-1]]` length 1+m ──
        // cache.step at start = prompt_len + TE; after = prompt_len + TE + 1 + m.
        //   verify_logits[0]   predicts the slot AFTER t1     → candidate for draft[0]
        //   verify_logits[i]   predicts the slot AFTER draft[i-1] (i = 1..m-1)
        //                                                     → candidate for draft[i]
        //   verify_logits[m]   predicts the slot AFTER draft[m-1]
        //                                                     → "bonus" position (full-accept new_t1)
        const seq_len: c_int = @intCast(1 + m);
        const verify_input_buf = try allocator.alloc(i32, 1 + m);
        defer allocator.free(verify_input_buf);
        verify_input_buf[0] = @intCast(t1);
        for (draft, 0..) |d, i| verify_input_buf[1 + i] = @intCast(d);
        const verify_shape = [_]c_int{ 1, seq_len };
        const verify_input = mlx.mlx_array_new_data(verify_input_buf.ptr, &verify_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(verify_input);

        // Enable per-position SSM capture for the verify pass on a GDN trunk so
        // partial accept can roll back without a re-forward. Self-detecting:
        // only GatedDeltaNet layers actually populate `spec_state_seq`, so
        // pure-attention / Mamba2 / LFM2 fall through to the snapshot fallback.
        self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
        const verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
        self.ctx.capture_ssm_seq = false;
        // Always free the transient capture buffers before returning, however
        // we exit this round (full accept, partial accept, or error).
        defer if (self.ctx.ssm_entries) |entries| {
            for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
        };
        // verify_logits shape [1, 1+m, V]. Sliced and freed below.
        self.pld_attempted += 1;

        const vl_shape = mlx.getShape(verify_logits);
        const slice_strides = [_]c_int{ 1, 1, 1 };

        // Slice all 1+m per-position logits up front so we can sample the
        // correction from the original verify forward (cache state aligned)
        // without re-running forward, and re-use them for both stochastic
        // accept tests and the correction sample.
        const per_pos_logits = try allocator.alloc(mlx.mlx_array, 1 + m);
        defer {
            for (per_pos_logits) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(per_pos_logits);
        }
        for (per_pos_logits, 0..) |*slot, idx| {
            slot.* = mlx.mlx_array_new();
            const start = [_]c_int{ 0, @intCast(idx), 0 };
            const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
            try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
        }
        _ = mlx.mlx_array_free(verify_logits);

        // ── Phase 5: Walk drafts. accepted ∈ [0, m]. Full accept = m. ──
        // verify_logits[i] is the prediction for draft[i] (i = 0..m-1).
        // No separate "first-position" test under v2 — the verify forward
        // covers it.
        var accepted: u32 = 0;
        if (stochastic) {
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                const target_p = try probsAtLastPos(per_pos_logits[i], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try probAt(target_p, draft[i], s);
                const accept_prob: f32 = @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                var argmax_arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(argmax_arr);
                try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, per_pos_logits[i], 2, false, s));
                try mlx.check(mlx.mlx_array_eval(argmax_arr));
                var argmax_val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&argmax_val, argmax_arr));
                if (@as(u32, @intCast(argmax_val)) != draft[i]) break;
                accepted += 1;
            }
        }
        accepted = capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );
        const full_accept = accepted == m;

        // ── Phase 6: Sample new_t1 from per_pos_logits[accepted] ──
        //   - full accept (accepted == m): per_pos_logits[m] predicts the slot
        //     after the last accepted draft (= "bonus" token).
        //   - partial (accepted < m):  per_pos_logits[accepted] is the model's
        //     prediction at the rejected slot. Stochastic samples from the
        //     residual `max(target_p − one_hot(draft[accepted]), 0)` to preserve
        //     the marginal distribution conditional on "not draft[accepted]"
        //     (Leviathan et al). Greedy: argmax of the rejected slot's logits.
        //
        // This indexing differs from v1: v1 sampled from `verify_logits[accepted-1]`
        // because t1 occupied no input slot; v2 has t1 at index 0 of the verify
        // input, so the "prediction one past the last accepted" lives at
        // index `accepted`. Off-by-one here would silently corrupt output.
        const correction_logits = per_pos_logits[accepted];
        const new_t1: u32 = blk: {
            if (stochastic) {
                const probs = try probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (!full_accept) {
                    const onehot = try pldOneHotRow(draft[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(onehot);
                    break :blk try sampleResidual(probs, onehot, s);
                } else {
                    break :blk try sampleFromProbs(probs, s);
                }
            } else {
                const lazy = self.sampleLazy(correction_logits);
                try mlx.check(mlx.mlx_array_eval(lazy));
                var v: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
                _ = mlx.mlx_array_free(lazy);
                break :blk @intCast(v);
            }
        };

        // ── Phase 7: Cache rollback on partial accept ──
        // After verify (length 1+m), cache.step = prompt_len + TE + 1 + m.
        // Full accept: TE_new = TE + 1 + m → no rollback.
        // Partial: must land at prompt_len + TE + 1 + accepted = prompt_len + TE_new
        // (TE_new = TE + 1 + accepted). Rollback then re-forward
        // `[t1, draft[0..accepted-1]]` length 1+accepted (with hidden capture
        // not needed here — just the cache advance).
        //
        // The accepted=0 case (= first draft rejected) MUST still re-forward
        // [t1] length 1: in v1 the t1 forward had been done eagerly before
        // verify; v2 includes t1 IN the verify forward, so rollback rolls
        // both t1 AND the drafts. Skipping the re-forward here would leave
        // the cache at prompt_len + TE — one short of the post-emit invariant.
        if (!full_accept) {
            // Fast GatedDeltaNet path: the verify forward captured per-position
            // SSM/conv state, so roll back by truncating the KV cache to the
            // accepted length (keeping verify's already-correct K/V for those
            // positions) and slicing the captured state — NO re-forward of the
            // accepted prefix. Detect via a populated capture on the first SSM
            // entry; absent it (pure-attention / Mamba2 / LFM2) we take the
            // proven restore + re-forward fallback below. Byte-identical either
            // way (pinned by tests/test_pld_equivalence.sh).
            const gdn_captured = if (self.ctx.ssm_entries) |entries|
                entries.len > 0 and entries[0].spec_state_seq.ctx != null
            else
                false;

            if (gdn_captured) {
                const accepted_len: usize = 1 + @as(usize, accepted);
                // `truncate` overwrites cache.step with its length arg; on this
                // family cache.step is a stale counter the model never reads
                // (positioning is moe_seq_offset), so preserve the snapshot's
                // value to keep the prefix cache's kv_step bookkeeping identical
                // to the restore-based fallback.
                const step_keep = kv_snap.step;
                try self.ctx.cache.truncate(moe_seq_offset_snap + accepted_len, s);
                self.ctx.cache.step = step_keep;
                for (self.ctx.ssm_entries.?) |*entry| {
                    try transformer_mod.ssmRollbackFromCapture(entry, accepted, 1 + m, s);
                }
                self.ctx.moe_seq_offset.* = moe_seq_offset_snap + accepted_len;
            } else {
                try self.ctx.cache.restore(&kv_snap);
                if (ssm_snaps) |snaps| {
                    for (self.ctx.ssm_entries.?, snaps) |*entry, *sn| try ssmRestore(entry, sn);
                }
                self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

                const re_seq_len: c_int = @intCast(1 + accepted);
                const re_input_buf = try allocator.alloc(i32, 1 + accepted);
                defer allocator.free(re_input_buf);
                re_input_buf[0] = @intCast(t1);
                for (draft[0..accepted], 0..) |d, i| re_input_buf[1 + i] = @intCast(d);
                const re_shape = [_]c_int{ 1, re_seq_len };
                const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
                defer _ = mlx.mlx_array_free(re_input);
                const re_logits = try xfm.forwardWith(&self.ctx, re_input);
                _ = mlx.mlx_array_free(re_logits);
            }
        }

        // ── Phase 8: Commit emitted tokens ──
        // Tokens emitted: [t1, draft[0..accepted]] = 1 + accepted.
        const num_emit: u32 = 1 + accepted;
        const tokens = try allocator.alloc(u32, num_emit);
        tokens[0] = t1;
        for (draft[0..accepted], 0..) |d, i| tokens[1 + i] = d;

        try self.generated_ids.append(allocator, t1);
        for (draft[0..accepted]) |d| try self.generated_ids.append(allocator, d);

        self.pld_accepted_tokens += accepted;
        self.advanceStep(num_emit);

        // No post-step forward — `next_token_id = new_t1` and exit. The next
        // nextPld call sees t1 NOT in cache (new invariant).
        self.next_token_id = new_t1;

        // Yield-gate accounting for verify steps (cold steps update in their
        // own branch above).
        self.yield_steps += 1;
        self.yield_accepted += accepted;

        // Runtime acceptance gate: after warmup, if the per-draft acceptance
        // probability is below the threshold, disable speculation for the rest
        // of this request (the re-enable check can bring it back when the
        // generated tail turns repetitive). PLD's `drafts_per_round` is the
        // upper-bound draft length (`max_draft`); matches with shorter accepts
        // still divide by this max so a workload with consistently-short
        // n-gram matches DOES get throttled.
        if (runtimeGateShouldDisable(self.pld_attempted, self.pld_accepted_tokens, max_draft)) {
            const drafts_proposed: u64 = self.pld_attempted * @as(u64, max_draft);
            const rate: f32 = if (drafts_proposed > 0)
                @as(f32, @floatFromInt(self.pld_accepted_tokens)) /
                    @as(f32, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  pld=disabled (runtime per-draft rate {d:.2} < {d:.2} after {d} attempts)\n",
                .{ rate, RUNTIME_GATE_MIN_PER_DRAFT_RATE, self.pld_attempted },
            );
            self.spec_disabled_runtime = true;
            self.disabled_steps = 0;
        }

        return PldStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
            .used_lookup = true,
        };
    }

    /// Result of one `nextDrafter` step. Same shape as PLD's result so the
    /// outer wrapper can share token-emit / EOS-check logic.
    pub const DrafterStepResult = struct {
        /// Tokens to emit this step. On a full accept this is
        /// `[t1, ...all_drafts]` (length `block_size`); on partial accept j
        /// it is `[t1, draft[0..j]]` (length `1+j`). The corrected fallback
        /// becomes `next_token_id` for the next call.
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (excludes always-accepted t1).
        accepted_tokens: u32,
    };

    /// Drafter-assisted decode step. Mirrors `nextPld` but the draft comes
    /// from `block_size - 1` autoregressive forwards through the Gemma 4
    /// assistant drafter (cross-attending into target's KV) instead of an
    /// n-gram lookup. Verify is identical: target forward over
    /// `[t1, draft0..draft_{m-1}]` with greedy / stochastic accept.
    ///
    /// Algorithm:
    ///   1. Run `block_size - 1` drafter steps. Each step's input is
    ///      `concat(target.embed(prev_tok)*scale, h_prev)`. `prev_tok` starts
    ///      at `next_token_id` (= t1); after step i it's the just-sampled
    ///      `draft[i]`. `h_prev` starts at `last_hidden` (captured at
    ///      prefill or the previous accept's verify-forward); after step i
    ///      it's the drafter's own `post_proj` output.
    ///      All drafter forwards in one round share `rope_offset =
    ///      target.cache.step` (per upstream `set_shared_kv`).
    ///   2. Snapshot KV + SSM, run target verify forward over
    ///      `[t1, draft0..draft_{m-1}]` length `block_size` with
    ///      `forwardCaptureHidden` so we get the new `h_prev` at position m.
    ///   3. Walk argmax(verify_logits[i]) vs draft[i] for i in 0..m-1.
    ///      Greedy: equal → accept. Stochastic: standard speculative-decoding
    ///      ratio test using `probAt(target_p, draft[i])` (the drafter's
    ///      masked-LM-head produces probabilistic logits, so we treat its
    ///      sampled draft as a one-hot proposal — same simplification PLD
    ///      uses).
    ///   4. Full accept (j == m): emit drafts, sample new pending from
    ///      verify_logits[m-1] (the target's prediction one position past the
    ///      last accepted draft — already computed during verify), update
    ///      `last_hidden` to the captured post-final-norm hidden.
    ///   5. Partial accept (j < m): roll back KV+SSM, re-forward
    ///      `[t1, draft[0..j-1]]` length `j+1` (with hidden capture) so
    ///      cache lands at exactly `+j+1`. Sample correction from the
    ///      *original* verify_logits[j] (the model's prediction at the
    ///      rejected position).
    pub fn nextDrafter(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.done) return null;
        std.debug.assert(self.drafter != null);
        std.debug.assert(self.has_last_hidden); // captured at init or last accept
        // Release-enforced guard (issue #97): the drafter path cannot honor a
        // grammar constraint or logprobs (compiled-out asserts before).
        if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

        // Runtime acceptance gate: if a prior step set the flag, fall back
        // to the regular `next()` path. Drafter's exit invariant is "t1 NOT
        // in cache" (different from `next()`'s expected entry), so `next()`
        // contains a transition shim that synchronously seeds pending_logits
        // when has_pending_logits is false. The shim makes this hand-off safe.
        if (self.spec_disabled_runtime) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
            };
        }

        const xfm = self.xfm;
        const s = xfm.s;
        const drafter = self.drafter.?;
        const m: u32 = @max(@as(u32, 1), self.drafter_block_size - 1);
        const t1: u32 = self.next_token_id; // already-decided token at position cache.step

        // RoPE offset: position the drafter's queries rotate by. Per upstream
        // `set_shared_kv`, this is `target.cache.step` and stays constant
        // across all `m` drafter steps in this round.
        const rope_offset: c_int = @intCast(self.ctx.cache.step);

        // ── Phase 1: draft `m` tokens lazily, no per-step CPU sync ──
        //
        // The drafter loop builds a chained lazy graph: each step's sampled
        // token is a [1]-shaped mlx_array fed directly to the next step's
        // `embedTargetTokenArr` as the indexer, and forward as the next step's
        // `prev_token`. No `mlx_array_eval` / `mlx_array_item_int32` calls
        // here — the entire m-step chain plus the verify forward (built
        // below) materialize as a single async graph and evaluate together.
        // For block_size=8 (31B), this collapses 7 GPU→CPU syncs into 0,
        // saving ~70-100ms of Metal command-buffer sync latency per round.
        var drafts = try allocator.alloc(u32, m);
        errdefer allocator.free(drafts);

        // `draft_arrs[i]` is the lazy [1] argmax output of drafter step i.
        // Owned here; freed at end of nextDrafter (after verify uses them).
        const draft_arrs = try allocator.alloc(mlx.mlx_array, m);
        defer {
            for (draft_arrs) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(draft_arrs);
        }

        // Wrap t1 as a [1] mlx_array so the FIRST drafter step can use the
        // same lazy-chain helper as subsequent steps. This array is also
        // reshaped + reused as the leading element of the verify input below.
        const t1_i32: i32 = @intCast(t1);
        const t1_shape = [_]c_int{1};
        const t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);

        // `h_prev_owner` rolls forward through the drafter. Starts at the
        // captured target hidden; subsequent steps use the drafter's
        // post_proj output. The output is itself a lazy mlx_array, so the
        // chain stays lazy across all m steps.
        var h_prev_owner: ?mlx.mlx_array = null;
        defer if (h_prev_owner) |h| {
            _ = mlx.mlx_array_free(h);
        };

        {
            var prev_tok_arr: mlx.mlx_array = t1_arr;
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                const h_prev_arg: mlx.mlx_array = if (h_prev_owner) |h| h else self.last_hidden;
                const step_out = try drafter_mod.stepArr(drafter, xfm, self.ctx.cache, prev_tok_arr, h_prev_arg, rope_offset);
                // Sample lazily — `sampleTokenLazy` for greedy returns the
                // argmax as a [1]-shaped lazy array. NO eval here.
                draft_arrs[i] = self.sampleLazy(step_out.logits);
                _ = mlx.mlx_array_free(step_out.logits);

                // Roll h_prev forward.
                if (h_prev_owner) |h_old| {
                    _ = mlx.mlx_array_free(h_old);
                }
                h_prev_owner = step_out.h_prev_next;
                // The next step's prev_token is THIS step's lazy sample.
                prev_tok_arr = draft_arrs[i];
            }
        }

        // ── Phase 2: snapshot KV + SSM + DSV4 ──
        var kv_snap = try self.ctx.cache.snapshot();
        defer kv_snap.deinit();
        var ssm_snaps: ?[]SSMCacheEntrySnapshot = null;
        defer if (ssm_snaps) |snaps| {
            for (snaps) |*sn| ssmSnapshotDeinit(sn);
            xfm.allocator.free(snaps);
        };
        if (self.ctx.ssm_entries) |entries| {
            const out = try xfm.allocator.alloc(SSMCacheEntrySnapshot, entries.len);
            for (entries, 0..) |*entry, idx| out[idx] = ssmSnapshot(entry);
            ssm_snaps = out;
        }
        const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

        // ── Phase 3: build verify input by concatenating [t1, drafts...] ──
        //
        // Build verify_input as a [1, 1+m] tensor without any CPU sync. The
        // m draft tokens are still lazy mlx_arrays at this point; we reshape
        // each [1] → [1,1] and stack along axis=1 with t1 reshaped the same
        // way. The forward pass that consumes verify_input is then chained
        // onto the drafter's lazy graph.
        const reshape_2d = [_]c_int{ 1, 1 };
        var t1_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t1_2d);
        try mlx.check(mlx.mlx_reshape(&t1_2d, t1_arr, &reshape_2d, 2, s));

        // Stack: each draft_arr[i] is shape [1]; reshape each to [1,1] and
        // collect into a vector_array along with t1_2d, then concat axis=1.
        var verify_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_input);
        {
            const drafts_2d = try allocator.alloc(mlx.mlx_array, m);
            defer {
                for (drafts_2d) |arr| _ = mlx.mlx_array_free(arr);
                allocator.free(drafts_2d);
            }
            for (draft_arrs, drafts_2d) |dlazy, *out| {
                out.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(out, dlazy, &reshape_2d, 2, s));
            }
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, t1_2d);
            for (drafts_2d) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
        }

        var new_hidden = mlx.mlx_array_new();
        // Captures the post-final-norm hidden at the LAST input position
        // (= position m, predicting the bonus token if all drafts accept).
        const verify_logits = try xfm.forwardWithCapture(&self.ctx, verify_input, &new_hidden);
        // verify_logits shape: [1, 1+m, V]
        self.drafter_attempted += 1;

        // ── Phase 4: decide longest accepted prefix ──
        //
        // Greedy mode: argmax over the entire [1, 1+m, V] verify_logits in
        // one op (yields [1, 1+m] indices). Stochastic mode: sample-residual
        // / accept-prob path needs per-position logits, so it slices below.
        // Either way, we collapse all per-step syncs into ONE eval at the
        // end of this round.
        const stochastic = self.sampling.temperature > 0.01;
        const vl_shape = mlx.getShape(verify_logits);

        // Stochastic path needs per-position logits to compute target probs
        // and (on partial accept) build the residual. Greedy path skips
        // slicing entirely. `per_pos_logits` is null in greedy mode.
        var per_pos_logits: ?[]mlx.mlx_array = null;
        defer if (per_pos_logits) |slots| {
            for (slots) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };
        if (stochastic) {
            const slots = try allocator.alloc(mlx.mlx_array, 1 + m);
            const slice_strides = [_]c_int{ 1, 1, 1 };
            for (slots, 0..) |*slot, idx| {
                slot.* = mlx.mlx_array_new();
                const start = [_]c_int{ 0, @intCast(idx), 0 };
                const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
                try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
            }
            per_pos_logits = slots;
        }

        // Build the greedy argmax tensor lazily; it'll be eval'd alongside
        // the rest of the round below.
        var verify_argmax = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_argmax);
        if (!stochastic) {
            try mlx.check(mlx.mlx_argmax_axis(&verify_argmax, verify_logits, 2, false, s));
        }
        _ = mlx.mlx_array_free(verify_logits);

        // ── Phase 4b: batched eval — drafts + verify_argmax + new_hidden ──
        //
        // Submit the entire round (drafter chain + verify forward + argmax)
        // to the GPU in a single async dispatch. Then sync ONCE per array we
        // need on the CPU. For block_size=8, this collapses ~14 individual
        // sync points (7 drafter samples + 7 per-position argmaxes in the
        // old code) into approximately 2: one effective sync to wait for
        // GPU completion (the first `mlx_array_eval`), and zero-cost evals
        // afterward since the work is already done.
        //
        // CORRECTNESS: `mlx_array_data_int32` only returns valid data once
        // the array is eval'd. We explicitly eval each array we will read.
        // `verify_input` is NOT eval'd separately because MLX may fuse it
        // into the forward pass without materializing a CPU-readable buffer
        // — instead we read drafts via per-array `mlx_array_item_int32` on
        // each `draft_arrs[i]` (cheap after the first sync).
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (draft_arrs) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
            if (!stochastic) {
                _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax);
            }
            _ = mlx.mlx_vector_array_append_value(eval_vec, new_hidden);
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
        // Extract drafts. First eval sync waits for the GPU; subsequent
        // evals are no-ops since they were queued together.
        for (draft_arrs, 0..) |arr, idx| {
            try mlx.check(mlx.mlx_array_eval(arr));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, arr));
            drafts[idx] = @intCast(v);
        }
        if (!stochastic) {
            // Force verify_argmax to materialize before bulk-reading. It's a
            // separate branch from the drafter chain (drafts → concat →
            // verify → argmax), so eval'ing the drafts above doesn't pull
            // verify_argmax along with them. This was the v26.5.6 bug that
            // produced 0% acceptance on 26B/31B (verify ran longer than the
            // drafter chain, so the data buffer was read while the GPU was
            // still writing it).
            try mlx.check(mlx.mlx_array_eval(verify_argmax));
        }

        var accepted: u32 = 0;
        if (stochastic) {
            // Stochastic verify (Leviathan et al. probability-ratio test).
            // The drafted token came from argmax of the drafter's masked LM
            // head, so we treat it as a one-hot proposal: accept with
            // probability `min(1, target_p[draft[i]])`, otherwise stop and
            // sample from the residual at the rejected position.
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_p = try probsAtLastPos(per_pos_logits.?[k], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try probAt(target_p, drafts[k], s);
                const accept_prob: f32 = @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            // Bulk-read the [1, 1+m] argmax indices and scan for first
            // mismatch in CPU. No more GPU syncs in this branch.
            const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse {
                return error.MlxArrayDataNull;
            };
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_argmax: u32 = @intCast(argmax_data[k]);
                if (target_argmax != drafts[k]) break;
                accepted += 1;
            }
        }

        accepted = capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );

        // Sample the next pending token from the verify output at position
        // `accepted`:
        //   - full accept (accepted == m): position m predicts the bonus
        //     token one past the last draft.
        //   - partial accept: position `accepted` predicts the model's
        //     replacement for the rejected draft.
        // For greedy, position `accepted`'s argmax is already in
        // `argmax_data[accepted]` — no extra GPU work. For stochastic, we
        // need the actual probability distribution at that position, so we
        // sample from `per_pos_logits[accepted]` (with residual correction
        // on partial accept per Leviathan et al).
        const next_pending: u32 = blk: {
            if (stochastic) {
                const correction_logits = per_pos_logits.?[accepted];
                const probs = try probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (accepted < m) {
                    const onehot = try pldOneHotRow(drafts[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(onehot);
                    break :blk try sampleResidual(probs, onehot, s);
                } else {
                    break :blk try sampleFromProbs(probs, s);
                }
            } else {
                // Greedy: reuse the bulk-read argmax row. Already eval'd in
                // the single async eval above; no GPU sync here.
                const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse {
                    return error.MlxArrayDataNull;
                };
                break :blk @intCast(argmax_data[accepted]);
            }
        };

        // ── Phase 5: commit / rollback ──
        if (accepted == m) {
            // Full accept: cache at +1+m. Emit [t1, ...drafts]. Pending = next_pending.
            // The captured `new_hidden` is the post-final-norm hidden at
            // position m — the last accepted draft's position. That's the
            // h_prev for the NEXT round (drafting from t = next_pending; the
            // hidden corresponds to draft[m-1], which is what next_pending
            // follows). This matches the convention `nextDrafter` uses.
            const tokens = try allocator.alloc(u32, 1 + m);
            tokens[0] = t1;
            for (drafts, 0..) |d, idx| tokens[1 + idx] = d;

            try self.generated_ids.append(allocator, t1);
            for (drafts) |d| try self.generated_ids.append(allocator, d);

            if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
            self.last_hidden = new_hidden;
            self.has_last_hidden = true;

            self.drafter_accepted_tokens += m;
            self.next_token_id = next_pending;
            self.advanceStep(1 + m);

            // drafts buffer transferred into tokens copy; free original.
            allocator.free(drafts);
            self.checkDrafterRuntimeGate();
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = m,
            };
        }

        // Partial accept (accepted < m). Cache over-advanced by (m - accepted).
        // The captured new_hidden is for position m (which we're rolling back
        // past) — discard it. Roll back KV+SSM, then re-forward
        // [t1, drafts[0..accepted]] length 1+accepted with hidden capture so
        // last_hidden lands at the position immediately past the last
        // accepted draft (where next_pending will live).
        _ = mlx.mlx_array_free(new_hidden);

        try self.ctx.cache.restore(&kv_snap);
        if (ssm_snaps) |snaps| {
            for (self.ctx.ssm_entries.?, snaps) |*entry, *sn| try ssmRestore(entry, sn);
        }
        self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

        const re_seq_len: c_int = @intCast(1 + accepted);
        const re_input_buf = try allocator.alloc(i32, 1 + accepted);
        defer allocator.free(re_input_buf);
        re_input_buf[0] = @intCast(t1);
        for (drafts[0..accepted], 0..) |d, idx| re_input_buf[1 + idx] = @intCast(d);
        const re_shape = [_]c_int{ 1, re_seq_len };
        const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(re_input);

        var re_new_hidden = mlx.mlx_array_new();
        const re_logits = try xfm.forwardWithCapture(&self.ctx, re_input, &re_new_hidden);
        _ = mlx.mlx_array_free(re_logits);

        const tokens = try allocator.alloc(u32, 1 + accepted);
        tokens[0] = t1;
        for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;

        try self.generated_ids.append(allocator, t1);
        for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);

        if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
        self.last_hidden = re_new_hidden;
        self.has_last_hidden = true;

        self.drafter_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.advanceStep(1 + accepted);

        allocator.free(drafts);
        self.checkDrafterRuntimeGate();
        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    /// DFlash block-drafter step. ONE assistant forward proposes
    /// `block_size - 1` drafts, conditioned on the trunk's cached
    /// target_layer_ids hiddens (the per-request `dflash_ctx`); the trunk
    /// verify over `[t1, drafts...]` follows the standard spec invariant
    /// (`cache.step = prompt_len + emitted`, t1 NOT in cache on entry,
    /// correction from ORIGINAL `verify_logits[accepted]`).
    ///
    /// Rollback is an offset-only `cache.truncate` — legal because
    /// `DflashModel.bind` restricts targets to the pure-KVCache standard
    /// path (no SSM, no module-owned state), where a position's K/V depend
    /// only on earlier positions (causal) and are identical whether computed
    /// in a width-16 verify or any re-forward. No re-forward runs at all:
    /// the next round's context comes from slicing THIS round's verify
    /// captures to the committed prefix — exactly the reference's
    /// `hidden_states[i+1][:, :n_accepted]`.
    ///
    /// Drafts are greedy (argmax over the trunk lm_head on assistant
    /// hiddens, anchor row DROPPED — reference `[:, 1:]`); sampled requests
    /// use the same one-hot Leviathan acceptance the drafter/PLD paths use.
    pub fn nextDflash(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        // The kv term is the same physics for either block decoder (one
        // forward, one KV read, shared across the block's rows), so a DFlash
        // round is an observation for it too — and on a DFlash-only server
        // it is the ONLY source.
        const dflash_kv_watch = io_util.Stopwatch.init(self.timer.io);
        // The round-cost table sees the same round: drafts = block - 1, or
        // 0 when the runtime gate fell back to serial (a serial sample is
        // the "no spec" candidate a width chooser needs).
        const dflash_gen_before = self.generated_ids.items.len;
        self.dflash_round_width = 0;
        const dflash_rounds_before: u64 = if (self.dflash_chooser) |ch| ch.rounds else self.dflash_attempted;
        defer {
            const ms = @as(f32, @floatFromInt(dflash_kv_watch.read())) / @as(f32, std.time.ns_per_ms);
            const emitted = self.generated_ids.items.len - dflash_gen_before;
            const post_warmup = dflash_rounds_before >= dflashGateWarmup();
            const wall = if (post_warmup) self.mtpRegimeWallMs(ms) else ms;
            self.specObserveRound(self.dflash_round_width, wall, @floatFromInt(emitted), post_warmup and emitted > 0, false);
            if (self.dflash_chooser) |*ch| ch.note(self.dflash_round_width);
        }
        if (self.done) return null;
        std.debug.assert(self.dflash != null);
        std.debug.assert(self.dflash_ctx != null);
        if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

        if (self.completion_tokens >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }

        if (self.spec_disabled_runtime) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
        }

        // Per-round width: the chooser's argmax over measured tokens/ms
        // (serial = width 0 is a candidate, so "serial wins" is the gate),
        // the fixed block until it has data.
        var round_width: u32 = @max(self.dflash_block_size, 2) - 1;
        if (self.dflash_chooser) |*ch| {
            if (ch.rounds >= dflashGateWarmup()) {
                const d = ch.choose(&self.xfm.round_cost, self.mtpKvLen(), ch.rounds);
                round_width = d.width;
                // Serial is sticky: a plain decode round does not extend the
                // assistant context, so there is no way back this request.
                if (round_width == 0) self.spec_disabled_runtime = true;
                if (ch.logged != ch.current) {
                    if (self.xfm.round_cost.bucketToRead(self.mtpKvLen())) |b| {
                        ch.logged = ch.current;
                        var buf: [256]u8 = undefined;
                        log.info("[dflash] width chooser: standing w{d} ({s}) from {s} {s} (ms/tok)\n", .{
                            ch.current,
                            if (ch.current == 0) "serial" else "block",
                            round_cost.bucketName(self.xfm.round_cost.layout, b),
                            self.xfm.round_cost.formatBucket(b, &buf),
                        });
                    }
                }
            }
        }
        self.dflash_round_width = round_width;
        if (round_width == 0) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
        }

        const xfm = self.xfm;
        const s = xfm.s;
        const model = self.dflash.?;
        const dctx = &self.dflash_ctx.?;
        const bs: u32 = round_width + 1;
        const m: u32 = bs - 1;
        const t1: u32 = self.next_token_id;
        // On the moe/GDN path positions come from moe_seq_offset (cache.step
        // is a bookkeeping counter the model never reads there — same rule as
        // nextMtp); the standard path keeps cache.step as the anchor.
        const moe_path = self.xfm.moe_layers != null;
        // A hybrid trunk (LFM2 DSpark) positions from moe_seq_offset too, but
        // its cache.step IS genuine (every token passes the attention layers),
        // so it keeps the truncate and only needs the offset + conv-state
        // rollback. Deciding the two independently is what keeps a partial
        // accept from either mis-positioning or double-counting.
        const hybrid_path = self.xfm.hybrid_layers != null;
        const anchor_pos: usize = if (moe_path or hybrid_path) self.ctx.moe_seq_offset.* else self.ctx.cache.step;
        const kv_step_snap = self.ctx.cache.step;
        std.debug.assert(dctx.absLen() == anchor_pos);

        const tracing = dflashTraceEnabled();
        var ph: io_util.Stopwatch = undefined;
        if (tracing) {
            ph = io_util.Stopwatch.init(self.timer.io);
            if (self.dflash_gap_watch) |*gw| self.dflash_trace.add(.gap, gw.read());
            self.dflash_gap_watch = null;
        }

        // ── Phase 1: one assistant forward drafts all m tokens ──
        // Row mapping is the export's convention (`anchor_row_drafts`):
        // DFlash reads mask rows 1..bs-1 (anchor row dropped), DSpark reads
        // ALL rows starting at the anchor — so DSpark needs only m noise rows
        // for the same m drafts and the same verify width.
        const noise_rows: u32 = if (model.config.anchor_row_drafts) m else bs;
        const noise_ids = try allocator.alloc(i32, noise_rows);
        defer allocator.free(noise_ids);
        noise_ids[0] = @intCast(t1);
        for (noise_ids[1..]) |*v| v.* = @intCast(model.config.mask_token_id);
        const noise_shape = [_]c_int{ 1, @intCast(noise_rows) };
        const noise_input = mlx.mlx_array_new_data(noise_ids.ptr, &noise_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(noise_input);
        // RAW table rows — no embed norm, no scale (the DFlash contract).
        const noise_embeds = try xfm.rawEmbedding(noise_input);
        defer _ = mlx.mlx_array_free(noise_embeds);

        const blk_hidden = try dflash_mod.forwardBlock(model, dctx, noise_embeds, anchor_pos);
        defer _ = mlx.mlx_array_free(blk_hidden);
        if (tracing) {
            try mlx.check(mlx.mlx_array_eval(blk_hidden));
            self.dflash_trace.add(.assist, ph.read());
            ph.reset();
        }
        const draft_logits_all = try model.draftLogits(xfm, blk_hidden);
        defer _ = mlx.mlx_array_free(draft_logits_all);

        // Slice the m draft rows: DFlash drops the anchor row (1..bs-1),
        // DSpark reads every row (0..m) — the anchor row IS draft 0.
        const dl_shape = mlx.getShape(draft_logits_all);
        var draft_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_logits);
        {
            const row0: c_int = if (model.config.anchor_row_drafts) 0 else 1;
            const start = [_]c_int{ 0, row0, 0 };
            const stop = [_]c_int{ 1, row0 + @as(c_int, @intCast(m)), dl_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&draft_logits, draft_logits_all, &start, 3, &stop, 3, &strides, 3, s));
        }
        // Sampled requests accept through the full Leviathan ratio
        // min(1, p/q), so a GREEDY draft is a one-hot q — and at temperature
        // the target row is flat, which is exactly where min(1, p(argmax))
        // collapses. Drafting FROM the request's own distribution makes q
        // track p and keeps acceptance flat across temperature. Greedy
        // requests keep the argmax path untouched, so the byte-equality
        // guard is unaffected.
        const stochastic = self.sampling.temperature > 0.01;
        // DFlash2 path selector: when the sidecar ships one, drafts come from
        // the pairwise-scored path trace instead of per-position argmax /
        // block sampling. Greedy requests keep the byte-equality bar (a
        // selector draft only survives verify if it IS the trunk argmax);
        // stochastic requests sample the selector's own candidate softmax and
        // accept through min(1, p/q) with q read off the traced path —
        // exact by construction. `MLX_SERVE_DFLASH_SELECTOR=0` forces the v1
        // arms for A/Bs.
        const use_selector = model.selector != null and dflashSelectorEnabled();
        // DSpark: the block's base logits are position-parallel, but each
        // position's draft is picked from logits CORRECTED by the token
        // drafted at the previous one (the Markov bigram bias). Chaining it
        // is what the head is for — dropping it drafts every position from
        // an uncorrected distribution the sidecar was never trained to emit.
        const use_markov = model.markov != null and dflashMarkovEnabled();
        const sample_drafts = stochastic and !use_selector and !use_markov and dflashSampledDraftsEnabled();
        var sel_path: ?dflash_mod.SelectedPath = null;
        defer if (sel_path) |*sp| sp.deinit(allocator);
        var draft_q: mlx.mlx_array = .{ .ctx = null }; // [m, V] proposal density
        defer if (draft_q.ctx != null) {
            _ = mlx.mlx_array_free(draft_q);
        };
        var draft_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_ids);
        if (use_markov) {
            const mh = &model.markov.?;
            const ids_i32 = try allocator.alloc(i32, m);
            defer allocator.free(ids_i32);
            var q_rows: ?[]mlx.mlx_array = null;
            defer if (q_rows) |rows| {
                for (rows) |r| _ = mlx.mlx_array_free(r);
                allocator.free(rows);
            };
            if (stochastic) {
                const rows = try allocator.alloc(mlx.mlx_array, m);
                for (rows) |*r| r.* = .{ .ctx = null };
                q_rows = rows;
            }
            var prev: u32 = t1;
            var step: u32 = 0;
            while (step < m) : (step += 1) {
                var base_row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(base_row);
                {
                    const start = [_]c_int{ 0, @intCast(step), 0 };
                    const stop = [_]c_int{ 1, @as(c_int, @intCast(step)) + 1, dl_shape[2] };
                    const strides = [_]c_int{ 1, 1, 1 };
                    try mlx.check(mlx.mlx_slice(&base_row, draft_logits, &start, 3, &stop, 3, &strides, 3, s));
                }
                const corrected = try mh.stepLogits(base_row, prev, s);
                defer _ = mlx.mlx_array_free(corrected);
                if (stochastic) {
                    // Sample the step from the request's own filtered
                    // distribution and keep the row as q — the accept ratio
                    // and the reject residual both need the density the
                    // draft was actually drawn from.
                    const qrow = try filteredProbsBlock(corrected, self.sampling, s);
                    var logq = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(logq);
                    try mlx.check(mlx.mlx_log(&logq, qrow, s));
                    const null_key = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(null_key);
                    var sampled = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(sampled);
                    try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
                    try mlx.check(mlx.mlx_array_eval(sampled));
                    const sd = mlx.mlx_array_data_int32(sampled) orelse return error.MlxArrayDataNull;
                    prev = @intCast(sd[0]);
                    q_rows.?[step] = qrow;
                } else {
                    var amax = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(amax);
                    try mlx.check(mlx.mlx_argmax_axis(&amax, corrected, 2, false, s));
                    var as_i32 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(as_i32);
                    try mlx.check(mlx.mlx_astype(&as_i32, amax, .int32, s));
                    try mlx.check(mlx.mlx_array_eval(as_i32));
                    const ad = mlx.mlx_array_data_int32(as_i32) orelse return error.MlxArrayDataNull;
                    prev = @intCast(ad[0]);
                }
                ids_i32[step] = @intCast(prev);
            }
            if (q_rows) |rows| {
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (rows) |r| _ = mlx.mlx_vector_array_append_value(vec, r);
                draft_q = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_concatenate_axis(&draft_q, vec, 0, s));
            }
            const row_shape = [_]c_int{ 1, @as(c_int, @intCast(m)) };
            const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(host_arr);
            try mlx.check(mlx.mlx_array_set(&draft_ids, host_arr));
        } else if (use_selector) {
            const sel_temp: f32 = if (stochastic) self.sampling.temperature else 0.0;
            sel_path = try dflash_mod.selectPath(
                allocator,
                &model.selector.?,
                model.config.selector_top_k,
                blk_hidden,
                draft_logits,
                t1,
                sel_temp,
                self.prng.random(),
                s,
            );
            const ids_i32 = try allocator.alloc(i32, m);
            defer allocator.free(ids_i32);
            for (sel_path.?.ids, ids_i32) |v, *d| d.* = @intCast(v);
            const row_shape = [_]c_int{ 1, @intCast(m) };
            const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(host_arr);
            try mlx.check(mlx.mlx_array_set(&draft_ids, host_arr));
        } else if (sample_drafts) {
            draft_q = try filteredProbsBlock(draft_logits, self.sampling, s);
            var logq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(logq);
            try mlx.check(mlx.mlx_log(&logq, draft_q, s));
            const null_key = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(null_key);
            var sampled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sampled);
            try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
            var as_i32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(as_i32);
            try mlx.check(mlx.mlx_astype(&as_i32, sampled, .int32, s));
            const row_shape = [_]c_int{ 1, @as(c_int, @intCast(m)) };
            try mlx.check(mlx.mlx_reshape(&draft_ids, as_i32, &row_shape, 2, s));
        } else {
            var draft_amax = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(draft_amax);
            try mlx.check(mlx.mlx_argmax_axis(&draft_amax, draft_logits, 2, false, s));
            try mlx.check(mlx.mlx_astype(&draft_ids, draft_amax, .int32, s));
        }
        if (tracing) {
            try mlx.check(mlx.mlx_array_eval(draft_ids));
            self.dflash_trace.add(.head, ph.read());
            ph.reset();
        }

        // ── Phase 2: verify input [t1, drafts...] — [1, bs] int32, lazy ──
        const t1_i32: i32 = @intCast(t1);
        const t1_shape = [_]c_int{ 1, 1 };
        const t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);
        var verify_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_input);
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, t1_arr);
            _ = mlx.mlx_vector_array_append_value(vec, draft_ids);
            try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
        }

        // Verify forward with layer captures — this round's verify IS the
        // next round's context producer.
        const cap_out = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
        defer {
            for (cap_out) |a| _ = mlx.mlx_array_free(a);
            allocator.free(cap_out);
        }
        for (cap_out) |*a| a.* = mlx.mlx_array_new();
        var cl = transformer_mod.CaptureLayers{ .ids = model.config.target_layer_ids, .out = cap_out };
        self.ctx.capture_layers = &cl;
        defer self.ctx.capture_layers = null;
        // Per-position SSM capture on a GDN trunk so partial accept can roll
        // back the recurrent state without re-forwarding (mirrors nextMtp).
        self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
        const verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
        self.ctx.capture_ssm_seq = false;
        defer if (self.ctx.ssm_entries) |entries| {
            for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
        };
        self.dflash_attempted += 1;
        if (tracing) {
            // Captures ride the same forward — eval them here or their cost
            // is billed to `append`.
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, verify_logits);
            for (cap_out) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
            try mlx.check(mlx.mlx_eval(vec));
            self.dflash_trace.add(.verify, ph.read());
            ph.reset();
        }

        // ── Phase 3: decide the longest accepted prefix ──
        const vl_shape = mlx.getShape(verify_logits);
        var per_pos_logits: ?[]mlx.mlx_array = null;
        defer if (per_pos_logits) |slots| {
            for (slots) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };
        if (stochastic) {
            const slots = try allocator.alloc(mlx.mlx_array, bs);
            const slice_strides = [_]c_int{ 1, 1, 1 };
            for (slots, 0..) |*slot, idx| {
                slot.* = mlx.mlx_array_new();
                const start = [_]c_int{ 0, @intCast(idx), 0 };
                const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
                try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
            }
            per_pos_logits = slots;
        }
        var verify_argmax = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_argmax);
        if (!stochastic) {
            try mlx.check(mlx.mlx_argmax_axis(&verify_argmax, verify_logits, 2, false, s));
        }
        _ = mlx.mlx_array_free(verify_logits);

        // One batched dispatch, then read (drafts branch + verify branch are
        // separate graphs — eval BOTH before reading either; the v26.5.6
        // 0%-acceptance class).
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, draft_ids);
            if (!stochastic) _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax);
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
        var drafts = try allocator.alloc(u32, m);
        errdefer allocator.free(drafts);
        if (sel_path) |*sp| {
            @memcpy(drafts, sp.ids);
        } else {
            try mlx.check(mlx.mlx_array_eval(draft_ids));
            const draft_data = mlx.mlx_array_data_int32(draft_ids) orelse return error.MlxArrayDataNull;
            for (drafts, 0..) |*d, idx| d.* = @intCast(draft_data[idx]);
        }
        if (!stochastic) try mlx.check(mlx.mlx_array_eval(verify_argmax));

        var accepted: u32 = 0;
        if (stochastic) {
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_p = try probsAtLastPos(per_pos_logits.?[k], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try probAt(target_p, drafts[k], s);
                const accept_prob: f32 = if (draft_q.ctx != null) blk: {
                    const q_row = try sliceProbRow(draft_q, k, s);
                    defer _ = mlx.mlx_array_free(q_row);
                    break :blk specAcceptProb(p_draft, try probAt(q_row, drafts[k], s));
                } else if (sel_path) |*sp| blk: {
                    // Selector-sampled draft: q is the traced step's own
                    // candidate softmax — exact, no GPU read.
                    const kk = sp.cand_ids.len / @as(usize, m);
                    break :blk specAcceptProb(p_draft, sp.q.?[@as(usize, k) * kk + sp.chosen_idx[k]]);
                } else @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse return error.MlxArrayDataNull;
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                if (@as(u32, @intCast(argmax_data[k])) != drafts[k]) break;
                accepted += 1;
            }
        }

        // A speculative verify may accept the whole block even when only a
        // few output positions remain. Cap the accepted prefix BEFORE cache
        // and assistant-context commit so every representation of the turn —
        // returned tokens, generated_ids, usage, KV and DFlash context — ends
        // at exactly max_tokens.
        accepted = capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );

        const next_pending: u32 = blk: {
            if (stochastic) {
                const correction_logits = per_pos_logits.?[accepted];
                const probs = try probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (accepted < m) {
                    // norm(max(0, p − q)) against the q the draft was drawn
                    // from — a one-hot here is the WRONG residual for a
                    // sampled draft, and wrong silently.
                    const q_row = if (draft_q.ctx != null)
                        try sliceProbRow(draft_q, accepted, s)
                    else if (sel_path) |*sp|
                        try selectorQRow(sp, accepted, m, vl_shape[2], s)
                    else
                        try pldOneHotRow(drafts[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(q_row);
                    break :blk try sampleResidual(probs, q_row, s);
                } else {
                    break :blk try sampleFromProbs(probs, s);
                }
            } else {
                const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse return error.MlxArrayDataNull;
                break :blk @intCast(argmax_data[accepted]);
            }
        };

        if (tracing) {
            self.dflash_trace.add(.accept, ph.read());
            ph.reset();
        }

        // ── Phase 4: commit — truncate on partial accept, then grow the
        // assistant context by exactly the committed positions ──
        const n_commit: usize = 1 + @as(usize, accepted);
        if (accepted < m) {
            try self.ctx.cache.truncate(anchor_pos + n_commit, s);
            if (moe_path) {
                // Same bookkeeping as nextMtp's GDN arm: preserve the
                // pre-verify cache.step (prefix-cache kv_step contract),
                // roll every linear layer's recurrent state back to the
                // accepted position from the verify pass's capture, and
                // re-point moe_seq_offset at the committed length.
                self.ctx.cache.step = kv_step_snap;
                if (self.ctx.ssm_entries) |entries| {
                    const gdn_captured = entries.len > 0 and entries[0].spec_state_seq.ctx != null;
                    if (!gdn_captured) return error.SpecRollbackUnavailable;
                    for (entries) |*entry| {
                        try transformer_mod.ssmRollbackFromCapture(entry, accepted, 1 + m, s);
                    }
                }
                self.ctx.moe_seq_offset.* = anchor_pos + n_commit;
            } else if (hybrid_path) {
                if (self.ctx.ssm_entries) |entries| {
                    // On a hybrid trunk only the CONV layers hold state, so
                    // entry 0 may legitimately be an attention layer with no
                    // capture at all — ask whether ANY layer recorded one.
                    var captured = false;
                    for (entries) |*e| {
                        if (e.spec_conv_input.ctx != null or e.spec_state_seq.ctx != null) {
                            captured = true;
                            break;
                        }
                    }
                    if (!captured) return error.SpecRollbackUnavailable;
                    for (entries) |*entry| {
                        try transformer_mod.ssmRollbackFromCapture(entry, accepted, bs, s);
                    }
                }
                self.ctx.moe_seq_offset.* = anchor_pos + n_commit;
            }
        }
        if (accepted == m) {
            try dflash_mod.appendContext(model, dctx, cap_out, anchor_pos);
        } else {
            const sliced = try allocator.alloc(mlx.mlx_array, cap_out.len);
            defer {
                for (sliced) |a| _ = mlx.mlx_array_free(a);
                allocator.free(sliced);
            }
            for (cap_out, sliced) |full, *out| {
                out.* = mlx.mlx_array_new();
                const fsh = mlx.getShape(full);
                const start = [_]c_int{ 0, 0, 0 };
                const stop = [_]c_int{ 1, @intCast(n_commit), fsh[2] };
                const strides = [_]c_int{ 1, 1, 1 };
                try mlx.check(mlx.mlx_slice(out, full, &start, 3, &stop, 3, &strides, 3, s));
            }
            try dflash_mod.appendContext(model, dctx, sliced, anchor_pos);
        }
        // Materialize the appended context off the critical path so next
        // round's assistant forward doesn't drag this round's verify graph.
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            dctx.appendEvalArrays(eval_vec);
            if (tracing) {
                try mlx.check(mlx.mlx_eval(eval_vec));
                self.dflash_trace.add(.append, ph.read());
                ph.reset();
            } else {
                try mlx.check(mlx.mlx_async_eval(eval_vec));
            }
        }

        const tokens = try allocator.alloc(u32, n_commit);
        tokens[0] = t1;
        for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;
        try self.generated_ids.append(allocator, t1);
        for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);
        allocator.free(drafts);

        self.dflash_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.advanceStep(@intCast(n_commit));
        if (self.dflash_chooser) |*ch| ch.observe(m, accepted, MTP_EV_EMA_BETA);
        // The calibrated sticky gate stays as the bootstrap that gets serial
        // MEASURED (the chooser picks serial only from a measured w0 cell).
        self.checkDflashRuntimeGate();
        if (self.completion_tokens >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
        }

        if (tracing) {
            self.dflashTraceRoundEnd(accepted);
            self.dflash_gap_watch = io_util.Stopwatch.init(self.timer.io);
        }

        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    /// DFlash runtime economics gate. Sticky within the request: once a
    /// three-round sample proves the block-parallel path yields less than its
    /// width-normalized request-class threshold, subsequent ticks use the
    /// regular pipelined decoder through `nextDflash`'s entry fallback.
    fn checkDflashRuntimeGate(self: *Generator) void {
        if (self.spec_disabled_runtime) return;
        if (!dflashGateShouldDisable(
            self.dflash_attempted,
            self.dflash_accepted_tokens,
            self.dflash_min_accepted_per_round,
        )) return;
        const avg = @as(f32, @floatFromInt(self.dflash_accepted_tokens)) /
            @as(f32, @floatFromInt(self.dflash_attempted));
        log.info(
            "  dflash=disabled (runtime yield {d:.2} accepted/round < {d:.2} after {d} attempts)\n",
            .{ avg, self.dflash_min_accepted_per_round, self.dflash_attempted },
        );
        self.spec_disabled_runtime = true;
    }

    /// Runtime acceptance gate for the drafter: after warmup, if the per-draft
    /// acceptance probability is below `RUNTIME_GATE_MIN_PER_DRAFT_RATE`,
    /// disable speculation for the rest of this request. Sticky for the rest
    /// of the generation.
    fn checkDrafterRuntimeGate(self: *Generator) void {
        if (self.spec_disabled_runtime) return;
        const drafts_per_round: u32 = if (self.drafter_block_size >= 1) self.drafter_block_size - 1 else 0;
        if (!runtimeGateShouldDisable(self.drafter_attempted, self.drafter_accepted_tokens, drafts_per_round)) return;
        const drafts_proposed: u64 = self.drafter_attempted * @as(u64, drafts_per_round);
        const rate: f32 = if (drafts_proposed > 0)
            @as(f32, @floatFromInt(self.drafter_accepted_tokens)) /
                @as(f32, @floatFromInt(drafts_proposed))
        else
            0.0;
        log.info(
            "  drafter=disabled (runtime per-draft rate {d:.2} < {d:.2} after {d} attempts)\n",
            .{ rate, RUNTIME_GATE_MIN_PER_DRAFT_RATE, self.drafter_attempted },
        );
        self.spec_disabled_runtime = true;
    }

    /// Qwen native-MTP speculative round. Structure mirrors `nextDrafter`
    /// (same verify invariant: cache.step = prompt_len + emitted, t1 NOT in
    /// cache on entry, verify input `[t1, draft[0..m-1]]`, bonus from row m,
    /// partial-accept snapshot/restore + re-forward) with one addition: the
    /// MTP head's committed-history KV cache. Draft steps append m temporary
    /// entries built from MTP-PREDICTED hiddens; after the verify decision we
    /// restore the round-boundary snapshot and re-append the committed pairs
    /// from TRUE trunk hiddens, so the history never accumulates drift.
    pub const MtpHistStash = struct {
        /// `[n]` int32 committed token ids: `[t1, drafts[0..accepted]]`.
        ids: mlx.mlx_array,
        /// `[1, n, H]` trunk hiddens paired 1:1 with `ids` (a lazy concat of
        /// last_hidden + a verify-capture slice — the handle pins the ~90 KB
        /// parent until consumed, deliberately NOT a deep copy).
        hidden: mlx.mlx_array,
        n: usize,
        /// Head-cache position of ids[0]'s entry (the producing round's
        /// mtp_off0). The consume-time truncate drops the producing round's
        /// stale draft tail past it.
        off0: usize,

        pub fn deinit(self: *MtpHistStash) void {
            _ = mlx.mlx_array_free(self.ids);
            _ = mlx.mlx_array_free(self.hidden);
        }
    };

    /// Round origin of the MTP head cache: with a pending stash the cache
    /// still holds the PREVIOUS round's draft tail (its step is stale), so
    /// the committed length is the stash origin plus the entries the stash
    /// itself will append; without one, the cache is fully committed.
    pub fn mtpRoundOff0(stash: ?MtpHistStash, cache_step: usize) usize {
        if (stash) |st| return st.off0 + st.n;
        return cache_step;
    }

    /// Committed-history length actually IN the head cache at rest — what a
    /// prefix-cache commit may snapshot. Everything past it is speculative:
    /// a pending stash's entries are NOT in the cache yet (the cache past
    /// `stash.off0` is the producing round's stale draft tail), and a built
    /// cross-round pre-draft has already appended NEXT-round draft entries
    /// past ITS `off0`. The min over both boundaries is always safe.
    pub fn mtpCommittedLen(cache_step: usize, pre_draft_off0: ?usize, stash_off0: ?usize) usize {
        var committed = cache_step;
        if (pre_draft_off0) |o| committed = @min(committed, o);
        if (stash_off0) |o| committed = @min(committed, o);
        return committed;
    }

    /// `mtpCommittedLen` over this Generator's live state; 0 when MTP is
    /// not active.
    pub fn mtpCommittedHistoryLen(self: *const Generator) usize {
        if (self.mtp_cache == null) return 0;
        return mtpCommittedLen(
            self.mtp_cache.?.step(),
            if (self.mtp_pre_draft) |*pd| pd.off0 else null,
            if (self.mtp_hist_stash) |*st| st.off0 else null,
        );
    }

    fn mtpMropeContext(self: *const Generator) ?mtp_mod.MropeContext {
        const positions = self.ctx.mrope_pos orelse return null;
        return .{
            .pos = positions,
            .total = self.ctx.mrope_total,
            .delta = self.ctx.mrope_delta,
            .base = self.mtp_position_base,
        };
    }

    /// One round's lazily-built MTP draft chain — the Phase 0/1 state that
    /// cross-round pre-drafting (`mtpMaybePreDraft`) moves into the PREVIOUS
    /// round's tail. Owns every handle it holds; `deinit` frees whatever was
    /// built so far, so it is safe on partial builds and on a pre-draft left
    /// unconsumed at request end.
    pub const MtpPreDraft = struct {
        plan: MtpRoundPlan,
        /// Head-cache position of this round's first draft entry.
        off0: usize,
        t1: u32,
        /// `[1]` int32 t1 — the verify input needs it again in Phase 3.
        t1_arr: mlx.mlx_array,
        /// Host draft ids, filled at the Phase 4b sync. len = plan.m_hi.
        drafts: []u32,
        /// Lazy `[1]` int32 draft ids. len = plan.m_hi; [0..n_drafted) valid.
        draft_arrs: []mlx.mlx_array,
        n_drafted: u32,
        /// Chunk-A log-confidence graphs (two-chunk plans only). len m_lo.
        conf_arrs: ?[]mlx.mlx_array,
        n_conf: u32,
        /// Sharp-draft proposal distributions q. len m_hi; [0..n_qp) valid.
        q_probs: ?[]mlx.mlx_array,
        n_qp: u32,
        /// Chain hidden after the last built step (owned) — a chunk-B
        /// extension resumes from it; freed once the chain is complete.
        h_chain: ?mlx.mlx_array,
        /// Tokens actually drafted this round; starts at m_lo, grows to
        /// m_hi iff the confidence gate clears at the chunk boundary.
        m: u32,

        pub fn deinit(self: *MtpPreDraft, allocator: std.mem.Allocator) void {
            _ = mlx.mlx_array_free(self.t1_arr);
            for (self.draft_arrs[0..self.n_drafted]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(self.draft_arrs);
            if (self.conf_arrs) |slots| {
                for (slots[0..self.n_conf]) |arr| _ = mlx.mlx_array_free(arr);
                allocator.free(slots);
            }
            if (self.q_probs) |slots| {
                for (slots[0..self.n_qp]) |arr| _ = mlx.mlx_array_free(arr);
                allocator.free(slots);
            }
            if (self.h_chain) |h| _ = mlx.mlx_array_free(h);
            allocator.free(self.drafts);
        }
    };

    /// Allocate an empty draft chain for `plan` (nothing built yet).
    fn mtpChainInit(self: *Generator, allocator: std.mem.Allocator, plan: MtpRoundPlan, t1: u32) !MtpPreDraft {
        const consider_ext = plan.m_hi > plan.m_lo;
        const sharp_drafts = mtpDraftSamplingFor(self.sampling, mtpDraftGreedy()).temperature > 0.01;
        const drafts = try allocator.alloc(u32, plan.m_hi);
        errdefer allocator.free(drafts);
        const draft_arrs = try allocator.alloc(mlx.mlx_array, plan.m_hi);
        errdefer allocator.free(draft_arrs);
        const conf_arrs: ?[]mlx.mlx_array = if (consider_ext) try allocator.alloc(mlx.mlx_array, plan.m_lo) else null;
        errdefer if (conf_arrs) |slots| allocator.free(slots);
        const q_probs: ?[]mlx.mlx_array = if (sharp_drafts) try allocator.alloc(mlx.mlx_array, plan.m_hi) else null;
        const t1_i32: i32 = @intCast(t1);
        const t1_shape = [_]c_int{1};
        return .{
            .plan = plan,
            .off0 = mtpRoundOff0(self.mtp_hist_stash, self.mtp_cache.?.step()),
            .t1 = t1,
            .t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 1, .int32),
            .drafts = drafts,
            .draft_arrs = draft_arrs,
            .n_drafted = 0,
            .conf_arrs = conf_arrs,
            .n_conf = 0,
            .q_probs = q_probs,
            .n_qp = 0,
            .h_chain = null,
            .m = plan.m_lo,
        };
    }

    /// Build draft steps [from..to) of `chain` — graph construction only, no
    /// sync; the caller dispatches. Each step's sampled token ([1] lazy
    /// array) feeds the next step's embedding lookup; the MTP post-norm
    /// hidden chains as the next h_prev. Step 0 merges the deferred history
    /// append (consumes `self.mtp_hist_stash`); a chunk-B call resumes from
    /// `chain.h_chain`.
    fn mtpChainBuild(self: *Generator, chain: *MtpPreDraft, from: u32, to: u32) !void {
        const xfm = self.xfm;
        const s = xfm.s;
        const head = self.mtp.?;
        const mc = &self.mtp_cache.?;
        const draft_sampling = mtpDraftSamplingFor(self.sampling, mtpDraftGreedy());
        const mtp_mrope_ctx = self.mtpMropeContext();
        const rerank_drafts = head.canRerankDrafts();
        std.debug.assert(chain.n_drafted == from);
        var i: u32 = from;
        while (i < to) : (i += 1) {
            const h_prev_arg: mlx.mlx_array = if (chain.h_chain) |h| h else self.last_hidden;
            const prev_tok_arr: mlx.mlx_array = if (i == 0) chain.t1_arr else chain.draft_arrs[i - 1];
            // Rerank drafts skip the head's own logits projection entirely:
            // the token comes from `draftSelect` on the chained hidden. The
            // sharp-proposal (q_probs) and confidence paths still need the
            // full distribution, so they keep want_logits.
            const need_logits = chain.q_probs != null or
                (chain.conf_arrs != null and i < chain.plan.m_lo);
            const use_rerank = rerank_drafts and !need_logits;
            // `want_logits: bool` could not tell a rerank draft from a history
            // append: both skip the vocab projection, but the draft still needs
            // the vector that projection would have consumed.
            const want: mtp_mod.StepWant = if (use_rerank) .mixed else .logits;
            const step_out = if (i == 0 and self.mtp_hist_stash != null) blk: {
                // Deferred history append (stashed at the END of the
                // previous round, Phase 5a) merged into this chain's first
                // draft: ONE (n+1)-row head forward appends the
                // committed-history entries AND the first draft entry,
                // replacing the old per-round appendHistory forward. RoPE
                // offsets and cache-append order are byte-identical to the
                // appendHistory-then-stepArr sequence (pinned by the merged-
                // forward equivalence test in mtp.zig).
                var st = self.mtp_hist_stash.?;
                self.mtp_hist_stash = null;
                defer st.deinit();
                // Drop the previous round's stale draft tail — the old
                // Phase 5a truncate, moved to consume time (offset-only).
                try mc.truncate(st.off0, s);
                var merged_ids = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(merged_ids);
                var merged_hidden = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(merged_hidden);
                {
                    const idv = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(idv);
                    _ = mlx.mlx_vector_array_append_value(idv, st.ids);
                    _ = mlx.mlx_vector_array_append_value(idv, prev_tok_arr);
                    try mlx.check(mlx.mlx_concatenate_axis(&merged_ids, idv, 0, s));
                    const hv = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(hv);
                    _ = mlx.mlx_vector_array_append_value(hv, st.hidden);
                    _ = mlx.mlx_vector_array_append_value(hv, h_prev_arg);
                    try mlx.check(mlx.mlx_concatenate_axis(&merged_hidden, hv, 1, s));
                }
                break :blk try head.forward(xfm, mc, merged_ids, merged_hidden, @intCast(st.off0), want, mtp_mrope_ctx);
            } else try head.forward(xfm, mc, prev_tok_arr, h_prev_arg, @intCast(chain.off0 + i), want, mtp_mrope_ctx);
            // A `.mixed` step that publishes `rerank_x` means `hidden_next` is
            // NOT what the lm_head consumes — on qwen4_exp it is the pre-mixer
            // `[B,S,hc*H]` stream and the mixer output is the rerank input.
            // Feeding the wrong one is a shape error, not a wrong answer.
            defer if (step_out.rerank_x.ctx != null) {
                _ = mlx.mlx_array_free(step_out.rerank_x);
            };
            if (use_rerank) {
                const rerank_x = if (step_out.rerank_x.ctx != null) step_out.rerank_x else step_out.hidden_next;
                chain.draft_arrs[i] = try head.draftSelect(xfm, rerank_x, draft_sampling.suppress_mask);
            } else if (chain.q_probs) |slots| {
                // Sharp proposal: q = filtered softmax of the draft-head
                // logits at the FIXED sharpened constants; the draft is
                // sampled from exactly this distribution (log+categorical
                // == categorical over the filtered logits), so the q used
                // in the accept ratio is the true proposal density.
                slots[i] = try probsAtLastPos(step_out.logits, draft_sampling, s);
                chain.n_qp = i + 1;
                chain.draft_arrs[i] = try sampleFromProbsLazy(slots[i], s);
            } else {
                chain.draft_arrs[i] = sampleTokenLazy(step_out.logits, draft_sampling, s);
            }
            chain.n_drafted = i + 1;
            if (chain.conf_arrs != null and i < chain.plan.m_lo) {
                // Chunk-A confidence: log p_head(draft) — built from the
                // step's own logits BEFORE they're freed (lazy graphs
                // hold their inputs internally).
                chain.conf_arrs.?[i] = try draftConfidenceGraph(step_out.logits, chain.draft_arrs[i], s);
                chain.n_conf = i + 1;
            }
            if (step_out.logits.ctx != null) _ = mlx.mlx_array_free(step_out.logits);
            if (chain.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
            chain.h_chain = step_out.hidden_next;
        }
    }

    /// Fire the chain's built graphs [from..to) at the GPU (async, no
    /// sync) so the head chain computes while the CPU builds the round's
    /// remaining graphs. Chunk-A dispatches (from == 0) on two-chunk plans
    /// also carry the confidence graphs and the chain hidden, so a
    /// consume-time extension decision reads materialized arrays (near-free
    /// sync) and chunk B resumes from a realized buffer. Dispatch timing
    /// only — lazy sampling ops bind their PRNG key at graph BUILD time, so
    /// values are identical.
    fn mtpChainDispatch(chain: *const MtpPreDraft, from: u32, to: u32) !void {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        for (chain.draft_arrs[from..to]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
        if (chain.q_probs) |slots| {
            for (slots[from..to]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
        }
        if (from == 0) {
            if (chain.conf_arrs) |slots| {
                for (slots[0..chain.n_conf]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
                if (chain.h_chain) |h| _ = mlx.mlx_vector_array_append_value(ev, h);
            }
        }
        try mlx.check(mlx.mlx_async_eval(ev));
    }

    /// Batched accept/correction graph for one stochastic round. Replaces
    /// the per-position construction — m single-element gathers + concat
    /// for accept_p (again for accept_q under sharp drafts) and 1+m
    /// SEPARATE full-vocab log+categorical correction samplers — with
    /// single batched kernels: one [m,1] take_along_axis per accept vector
    /// and ONE [1+m, V] log+categorical covering every possible reject
    /// position plus the full-accept bonus. Identical math and identical
    /// output distributions; only the dispatch count (and the PRNG key
    /// split pattern — one batched categorical draw instead of 1+m
    /// sequential ones) changes.
    pub const MtpBatchedGraph = struct {
        /// [1+m] int32 pre-sampled corrections (index a = residual sample
        /// for a reject at position a; index m = full-accept bonus).
        corr_samples: mlx.mlx_array,
        /// [m] f32 filtered target probability of each draft.
        accept_p: mlx.mlx_array,
        /// [m] f32 proposal density at each draft (sharp drafts only;
        /// null-ctx under greedy proposals).
        accept_q: mlx.mlx_array,

        pub fn deinit(self: *MtpBatchedGraph) void {
            _ = mlx.mlx_array_free(self.corr_samples);
            _ = mlx.mlx_array_free(self.accept_p);
            if (self.accept_q.ctx != null) _ = mlx.mlx_array_free(self.accept_q);
        }
    };

    pub fn mtpBatchedAcceptGraph(
        probs_all: mlx.mlx_array,
        draft_arrs: []const mlx.mlx_array,
        q_probs: ?[]const mlx.mlx_array,
        m: u32,
        s: mlx.mlx_stream,
    ) !MtpBatchedGraph {
        const shape = mlx.getShape(probs_all); // [1, 1+m, V]
        const vocab = shape[2];
        const mi: c_int = @intCast(m);
        const strides2 = [_]c_int{ 1, 1 };

        var p2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p2d);
        const p2_shape = [_]c_int{ mi + 1, vocab };
        try mlx.check(mlx.mlx_reshape(&p2d, probs_all, &p2_shape, 2, s));

        var ids_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_2d);
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (draft_arrs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_concatenate_axis(&flat, vec, 0, s));
            const id2_shape = [_]c_int{ mi, 1 };
            try mlx.check(mlx.mlx_reshape(&ids_2d, flat, &id2_shape, 2, s));
        }

        // Draft-position rows [m, V] and the bonus row [1, V] (slice views).
        var p_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p_rows);
        var bonus = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bonus);
        {
            const r_start = [_]c_int{ 0, 0 };
            const r_stop = [_]c_int{ mi, vocab };
            try mlx.check(mlx.mlx_slice(&p_rows, p2d, &r_start, 2, &r_stop, 2, &strides2, 2, s));
            const b_start = [_]c_int{ mi, 0 };
            const b_stop = [_]c_int{ mi + 1, vocab };
            try mlx.check(mlx.mlx_slice(&bonus, p2d, &b_start, 2, &b_stop, 2, &strides2, 2, s));
        }

        // Proposal stack [m, V]: the full sharpened q rows under sharp
        // drafts (exact Leviathan residual), one-hot of the lazy draft ids
        // under greedy proposals.
        var proposal = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(proposal);
        if (q_probs) |qs| {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (qs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            try mlx.check(mlx.mlx_concatenate_axis(&proposal, vec, 0, s));
        } else {
            var indices = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(indices);
            try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));
            var onehot_b = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(onehot_b);
            try mlx.check(mlx.mlx_equal(&onehot_b, indices, ids_2d, s));
            try mlx.check(mlx.mlx_astype(&proposal, onehot_b, mlx.mlx_array_dtype(p2d), s));
        }

        // residual = max(p − proposal, 0) rows, then [residual; bonus] and
        // ONE categorical over the whole stack.
        var corr_samples = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(corr_samples);
        {
            var diff = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(diff);
            try mlx.check(mlx.mlx_subtract(&diff, p_rows, proposal, s));
            const zero = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zero);
            var residual = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(residual);
            try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));
            var stack = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stack);
            {
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                _ = mlx.mlx_vector_array_append_value(vec, residual);
                _ = mlx.mlx_vector_array_append_value(vec, bonus);
                try mlx.check(mlx.mlx_concatenate_axis(&stack, vec, 0, s));
            }
            var log_stack = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(log_stack);
            try mlx.check(mlx.mlx_log(&log_stack, stack, s));
            const null_key = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(null_key);
            var sampled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sampled);
            try mlx.check(mlx.mlx_random_categorical(&sampled, log_stack, -1, null_key, s));
            try mlx.check(mlx.mlx_astype(&corr_samples, sampled, .int32, s));
        }

        // accept_p[k] = p_rows[k][draft_k]; accept_q[k] = proposal density.
        var accept_p = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(accept_p);
        var accept_q: mlx.mlx_array = .{ .ctx = null };
        errdefer if (accept_q.ctx != null) {
            _ = mlx.mlx_array_free(accept_q);
        };
        {
            var taken = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(taken);
            try mlx.check(mlx.mlx_take_along_axis(&taken, p_rows, ids_2d, -1, s));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const m_shape = [_]c_int{mi};
            try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
            try mlx.check(mlx.mlx_astype(&accept_p, flat, .float32, s));
        }
        if (q_probs != null) {
            var taken = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(taken);
            try mlx.check(mlx.mlx_take_along_axis(&taken, proposal, ids_2d, -1, s));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const m_shape = [_]c_int{mi};
            try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
            accept_q = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&accept_q, flat, .float32, s));
        }

        return .{ .corr_samples = corr_samples, .accept_p = accept_p, .accept_q = accept_q };
    }

    /// Batched-corrections kill switch — MLX_SERVE_MTP_BATCH_CORR=0
    /// restores the per-position accept/correction graphs for A/Bs.
    var mtp_batch_corr_cache: ?bool = null;
    pub fn mtpBatchCorrEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return value.len == 0 or value[0] != '0';
    }

    fn mtpBatchCorrEnabled() bool {
        if (mtp_batch_corr_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_BATCH_CORR")) |p| std.mem.span(p) else null;
        const on = mtpBatchCorrEnabledFromEnv(raw);
        mtp_batch_corr_cache = on;
        return on;
    }

    /// Cross-round pre-draft (round pipelining): at the round's tail — the
    /// accept decision made, trunk committed/rolled back, EV updated,
    /// last_hidden/next_token_id already pointing at the next round — build
    /// the NEXT round's chunk-A draft chain (consuming the history stash
    /// Phase 5a just created, exactly as the next round's Phase 1 would)
    /// and fire it at the GPU. The CPU then returns to the scheduler for
    /// emit/SSE work while the head chain runs; the next nextMtp call finds
    /// its drafts already materialized. The plan is computed AFTER this
    /// round's EV update, so it is byte-identical to the one the next
    /// round's entry would compute.
    fn mtpMaybePreDraft(self: *Generator, allocator: std.mem.Allocator) !void {
        if (!mtpPredraftEnabled()) return;
        if (self.spec_disabled_runtime) return;
        std.debug.assert(self.mtp_pre_draft == null);
        const plan = self.mtpRoundPlan();
        // The plan itself can end speculation (the adaptive switch) or park
        // the request on a serial probe — either way there is no next round
        // to draft for, and the stash this pre-draft would have consumed is
        // the serial block's to apply.
        if (self.spec_disabled_runtime or self.mtp_serial_left > 0) return;
        var chain = try self.mtpChainInit(allocator, plan, self.next_token_id);
        errdefer chain.deinit(allocator);
        try self.mtpChainBuild(&chain, 0, plan.m_lo);
        try mtpChainDispatch(&chain, 0, plan.m_lo);
        self.mtp_pre_draft = chain;
    }

    // ── Serial blocks inside nextMtp ──
    //
    // Leaving MTP is cheap; coming back is the delicate half, and the exit
    // ramp below exists for ONE reason: `next()`'s pipeline always leaves the
    // trunk one token AHEAD of what was emitted with no captured hidden, and
    // an MTP round needs both `t1 NOT in cache` and an `h_prev` for exactly
    // that position. `drainPipelineForSpec` lands the first half; the capture
    // tick lands the second by forwarding ONE token with capture. The cost is
    // that token's own forward plus the pipelining this one step gives up.
    //
    // What is NOT restored is the head's committed history over the block:
    // those tokens' trunk hiddens were never captured, so the head cache
    // simply does not grow across a serial block. RoPE offsets there are
    // cache-relative, so a mid-sequence start is a state a warm hot-cache hit
    // already produces; the head resumes with a CONTENT gap, which costs
    // acceptance for a while. An M-RoPE turn is the exception and is excluded
    // wholesale by `mtpAdaptiveHeadMayResume`.

    /// Will this serial block be left behind? Only then is applying the
    /// deferred history stash worth a head forward — a block that ends the
    /// request's speculation for good drops it exactly as before.
    fn mtpSerialMayResume(self: *const Generator) bool {
        if (self.mtp_serial_left > 0 or self.mtp_serial_exit != .none) return true;
        return self.spec_disable_reason == .adaptive and self.mtpAdaptiveHeadMayResume();
    }

    /// Leave the head cleanly for a serial block. The round tail may have
    /// built a cross-round pre-draft (drafts for a round that will not run —
    /// dropped, always). `apply_stash` APPLIES the deferred history stash
    /// instead of leaving it for `deinit`: truncate the producing round's
    /// stale draft tail, then one committed-history head forward, so the head
    /// history is complete right up to the block. Idempotent.
    fn mtpDetachHead(self: *Generator, allocator: std.mem.Allocator, apply_stash: bool) !void {
        if (self.mtp_pre_draft) |*pd| {
            pd.deinit(allocator);
            self.mtp_pre_draft = null;
        }
        if (!apply_stash) return;
        var st = self.mtp_hist_stash orelse return;
        self.mtp_hist_stash = null;
        defer st.deinit();
        const mc = &self.mtp_cache.?;
        try mc.truncate(st.off0, self.xfm.s);
        // `.none`, not `.mixed`: this is a committed-history append, so the
        // mixer and the vocab projection are both dead work here.
        const out = try self.mtp.?.forward(
            self.xfm,
            mc,
            st.ids,
            st.hidden,
            @intCast(st.off0),
            .none,
            self.mtpMropeContext(),
        );
        if (out.logits.ctx != null) _ = mlx.mlx_array_free(out.logits);
        if (out.hidden_next.ctx != null) _ = mlx.mlx_array_free(out.hidden_next);
        if (out.rerank_x.ctx != null) _ = mlx.mlx_array_free(out.rerank_x);
    }

    fn mtpSerialOneToken(allocator: std.mem.Allocator, tok: u32) !DrafterStepResult {
        const tokens = try allocator.alloc(u32, 1);
        tokens[0] = tok;
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }

    /// Inter-tick wall clock for the table's serial cell — the twin of
    /// `mtpRegimeWallMs`, and for the same reason: tok/s is what the two arms
    /// are compared on, so per-tick work OUTSIDE the forward (publish, stop
    /// checks, scheduler) belongs in the denominator. Null on the first read
    /// of a block: that one SEEDS.
    fn mtpSerialWallMs(self: *Generator) ?f32 {
        if (self.mtp_serial_clock) |*c| {
            const ns = c.read();
            c.reset();
            return @as(f32, @floatFromInt(ns)) / @as(f32, std.time.ns_per_ms);
        }
        self.mtp_serial_clock = io_util.Stopwatch.init(self.timer.io);
        return null;
    }

    /// S22 / L27. Model-level twin of `mtpAdaptiveArchEligible`: is the
    /// adaptive switch's whole calibration valid for THIS checkpoint?
    ///
    /// `serialCellWanted` runs on requests that never armed a head, so it
    /// cannot ask `self.mtp` — it asks the MODEL. `model_has_mtp` alone was
    /// too wide: every sidecar pack (qwen3.5/3.6/3.8) past 8192 KV read a
    /// stopwatch on every decode token, folded a cell into a row nothing on
    /// that arch will ever read, and rewrote the persisted table at the end
    /// of every request for it. The head KIND is the arch, and only the
    /// in-checkpoint qwen4 head was measured. Widening this is a MEASUREMENT.
    pub fn mtpAdaptiveModelEligible(model_has_mtp: bool, module_head_loaded: bool) bool {
        return model_has_mtp and module_head_loaded;
    }

    fn mtpAdaptiveModelOk(self: *const Generator) bool {
        return mtpAdaptiveModelEligible(self.model_has_mtp, self.xfm.qwen4_mtp != null);
    }

    /// Will ANYONE read a serial cell for this model? The cell exists for
    /// exactly one consumer, the adaptive switch, so folding one anywhere
    /// else is pure cost: `observeSerialTick` sits on the scheduler's regular
    /// decode path, which every model takes, so gemma3, llama, a GGUF and any
    /// `--no-mtp` boot were all folding a serial cell on every token and
    /// rewriting `~/.mlx-serve/round-cost/<key>.txt` at the end of every
    /// request for a table nothing would ever read.
    ///
    /// The gate is on the MODEL, not the request: `enable_mtp:false` against a
    /// checkpoint whose serial row IS read is the cleanest source of serial
    /// data there is, and it must keep feeding the cell. Scope (L27): a
    /// SIDECAR pack never folds a serial cell AT ALL — `mtpAdaptiveModelOk`
    /// declines the whole ARCH, not merely the request that opted out —
    /// because nothing on that arch will ever read the row.
    pub fn serialCellWanted(self: *const Generator) bool {
        if (!mtpAdaptiveSerialEnabled() or !mtpCostTableEnabled()) return false;
        // The ARCH gate first: on a sidecar pack nothing reads the serial
        // row, so folding one is pure cost and a persisted write per request.
        if (!self.mtpAdaptiveModelOk()) return false;
        // NOT `xfm.qwen4_mtp != null`: the in-checkpoint head's weights load
        // with the TRUNK, so they are present even under `--no-mtp`, and a
        // `--no-mtp` boot folded a serial cell on every decoded token and
        // rewrote the persisted table at the end of every request. The
        // registry's answer is the model-level one and honours the flag.
        //
        // Per-REQUEST `enable_mtp:false` deliberately still folds: a plain
        // decode on a checkpoint that CAN speculate is the cleanest serial
        // sample there is, and it is what teaches the cell.
        if (!self.model_has_mtp) return false;
        return mtpAdaptiveKvEligible(self.mtpKvLen(), mtpAdaptiveMinKv());
    }

    /// Throw away the serial cell's PENDING interval: the next read seeds
    /// instead of measuring. Called when something ran between two decode
    /// ticks that is not decode — today, a prefill chunk under
    /// `scheduler.interleaveDecodeTick`, which runs decode ticks from inside
    /// another request's prefill. `spec_cost_solo` reads true throughout
    /// (this stream really is the only one DECODING), so without this the
    /// interval carried a whole prefill chunk into the serial cell, which is
    /// then persisted and used to decide against MTP for the rest of the
    /// process.
    pub fn invalidateSerialClock(self: *Generator) void {
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
    }

    /// Fold ONE plain serial decode token into the model's serial cell. Every
    /// serial decode tick the server runs passes through here: a request that
    /// never armed MTP (`enable_mtp:false`, `--no-mtp`), one whose
    /// speculation turned itself off, and the bounded probe. The first
    /// `MTP_ADAPTIVE_PROBE_WARM` ticks of a block are offered as transitions
    /// — the GPU still holds the previous round's tail — and contention only
    /// ever ADDS time, so a busy server restarts the clock rather than
    /// teaching the table a lie.
    pub fn observeSerialTick(self: *Generator) void {
        if (!self.serialCellWanted()) {
            self.mtp_serial_clock = null;
            self.mtp_serial_warm = 0;
            return;
        }
        if (!self.spec_cost_solo) {
            self.mtp_serial_clock = null;
            self.mtp_serial_warm = 0;
            return;
        }
        const ms = self.mtpSerialWallMs() orelse return;
        const warm = self.mtp_serial_warm < MTP_ADAPTIVE_PROBE_WARM;
        if (warm) self.mtp_serial_warm += 1;
        _ = self.xfm.round_cost.observeSerial(self.mtpKvLen(), ms, true, warm);
    }

    /// May `mtpSerialCaptureTick` run? Its whole contract is to forward the
    /// ONE pending token with capture, so a generator still holding pipelined
    /// logits or a pipelined token would have it forward the wrong position
    /// and publish a hidden for a row that was never committed. Pure, so the
    /// rule is testable without MLX — and a RUNTIME check, not an assert:
    /// `std.debug.assert` compiles to nothing in ReleaseFast, which is the
    /// only build that ever serves, so the guard would have been absent in
    /// exactly the builds that need it.
    pub fn mtpSerialCaptureReady(has_pending_logits: bool, has_pending_token: bool) bool {
        return !has_pending_logits and !has_pending_token;
    }

    /// Abandon the ramp back to MTP and finish this request on the serial
    /// arm. Reached from the two states the pipelined machine is not supposed
    /// to produce; either way the safe answer is the same, so it is ONE
    /// helper. Logged once per request — a tick loop would flood the log.
    fn mtpSerialGiveUp(self: *Generator, allocator: std.mem.Allocator, why: []const u8) !?DrafterStepResult {
        if (!self.mtp_serial_giveup_logged) {
            self.mtp_serial_giveup_logged = true;
            log.info("  [mtp] adaptive: {s} at the serial exit -> serial for the rest of the request\n", .{why});
        }
        self.mtp_serial_exit = .none;
        self.mtp_serial_left = 0;
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .adaptive;
        const tok = try self.next(allocator) orelse return null;
        return try mtpSerialOneToken(allocator, tok);
    }

    /// One tick of a serial block. `.none` is the REAL serial arm — a plain
    /// pipelined `next()`, the thing the serial cell measures; `.drain` and
    /// `.capture` are the exit ramp back to `nextMtp`'s entry invariant.
    fn mtpSerialTick(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.mtp_serial_exit == .capture) return try self.mtpSerialCaptureTick(allocator);
        if (self.mtp_serial_exit == .drain) {
            switch (try self.drainPipelineForSpec(allocator)) {
                .stopped => return null,
                // A half-state the pipelined state machine never produces.
                // Rather than risk nextMtp's entry invariant, this request
                // finishes on the serial arm.
                .stay_disabled => return try self.mtpSerialGiveUp(allocator, "unexpected pipeline state"),
                .already_clean => {
                    self.mtp_serial_exit = .capture;
                    return try self.mtpSerialCaptureTick(allocator);
                },
                .drained => |tok| {
                    self.mtp_serial_exit = .capture;
                    return try mtpSerialOneToken(allocator, tok);
                },
            }
        }
        const tok = try self.next(allocator) orelse return null;
        self.observeSerialTick();
        if (self.mtp_serial_left > 0) {
            self.mtp_serial_left -= 1;
            // A bounded block is over: ramp back rather than run on.
            if (self.mtp_serial_left == 0) self.mtp_serial_exit = .drain;
        }
        return try mtpSerialOneToken(allocator, tok);
    }

    /// The one synchronous step that ends a serial block: forward the pending
    /// token WITH capture so `last_hidden` is the trunk hidden at the last
    /// committed position again, and sample its successor WITHOUT forwarding
    /// it — exactly `nextMtp`'s entry invariant.
    fn mtpSerialCaptureTick(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        // The entry invariant is CHECKED, not asserted: an assert is a no-op
        // in the only optimize mode that serves, and a capture forward on a
        // still-pipelined generator corrupts `last_hidden` silently rather
        // than crashing. The recovery is the drain arm's — finish serial.
        if (!mtpSerialCaptureReady(self.has_pending_logits, self.has_pending_token)) {
            return try self.mtpSerialGiveUp(allocator, "pipeline not drained");
        }
        if (try self.checkStop()) return null;
        const token = self.next_token_id;
        const tok_i32: i32 = @intCast(token);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);

        var hidden = mlx.mlx_array_new();
        const logits = self.xfm.forwardWithCapture(&self.ctx, tok_input, &hidden) catch |e| {
            _ = mlx.mlx_array_free(hidden);
            return e;
        };
        defer _ = mlx.mlx_array_free(logits);
        // `hidden` is adopted here; everything after is bookkeeping the
        // generator must not be left half-way through.
        if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
        self.last_hidden = hidden;
        self.has_last_hidden = true;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        // N17. Both checks below now RETURN on a Metal working-set abort
        // instead of ending the process — that is what the error latch is for,
        // and it is exactly the failure this code path exists to survive — so
        // the handle needs an owner on the error path. A scoped `defer` rather
        // than an `errdefer` plus a manual free: this function ends in
        // `return try mtpSerialOneToken(...)`, and an errdefer paired with an
        // explicit free would double-free the moment anything after the free
        // fails.
        const val: i32 = blk: {
            const lazy = self.sampleLazy(logits);
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
            break :blk v;
        };
        self.next_token_id = @intCast(val);

        // Block over. The next MTP round must not bill the serial block it
        // followed, and the next serial block warms again.
        self.mtp_serial_exit = .none;
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
        self.mtp_regime_clock = null;
        return try mtpSerialOneToken(allocator, token);
    }

    pub fn nextMtp(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.done) return null;
        std.debug.assert(self.mtp != null);
        std.debug.assert(self.mtp_cache != null);
        std.debug.assert(self.has_last_hidden);
        // Release-enforced guard (issue #97): the MTP path cannot honor a
        // grammar constraint or logprobs (compiled-out asserts before).
        if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

        // `--max-mtp-ctx`: checked at the TOP of the round, before any
        // drafting, against prompt + generated (`mtpKvLen` — what the next
        // verify forward would read). A request admitted under the ceiling
        // can generate its way past it, and past it every round costs more
        // than the serial steps it replaces. Sticky for the rest of the
        // generation (context only grows) and routed through the SAME
        // `spec_disabled_runtime` fallback the acceptance gate uses, so the
        // pre-draft is freed and the hand-off contract is the tested one.
        if (!self.spec_disabled_runtime and
            !mtp_mod.mtpCtxWithinLimit(max_mtp_ctx, self.mtpKvLen()))
        {
            log.info(
                "  mtp=off (ctx {d} > --max-mtp-ctx {d})\n",
                .{ self.mtpKvLen(), max_mtp_ctx },
            );
            self.spec_disabled_runtime = true;
            self.spec_disable_reason = .max_ctx;
        }

        // Adaptive serial re-entry. The switch is sticky, with ONE exception:
        // the request generated its way into a different KV bucket, so
        // neither number the switch compared describes this round any more.
        // `--max-mtp-ctx` is checked ABOVE and is not overridable from here —
        // its `.max_ctx` reason never matches, so a ceiling crossing stays
        // off for good.
        if (self.spec_disabled_runtime and self.spec_disable_reason == .adaptive and
            mtpAdaptiveSerialEnabled() and self.mtpAdaptiveHeadMayResume())
        {
            // The SAME resolution the decision used — see mtpAdaptiveBucketOf.
            const b = self.mtpAdaptiveBucket(self.mtpKvLen());
            const prev_bucket = self.mtp_adaptive.bucket;
            const prev_arm = self.mtp_adaptive.arm;
            const action = self.mtp_adaptive.serialTick(b, mtpAdaptiveReentryTokens());
            self.mtpAdaptiveSyncWindow(prev_bucket, prev_arm);
            // Re-entry only when the head can PROVE it is in sync: the next
            // round's offset must be the one `qwen4MtpForward` demands, or it
            // returns `error.MtpPositionGap` and kills the request instead of
            // degrading. A serial block moves the trunk and not the head, so
            // this is exactly the transition that could break it. A decline
            // LATCHES — the drift cannot shrink on its own, so re-asking every
            // period would just log the same refusal forever — and the request
            // simply carries on down the serial branch below.
            var may_reenter = action == .to_mtp;
            if (may_reenter) {
                if (self.mtpHeadDriftNow()) |drift| {
                    if (drift != 0) {
                        log.info(
                            "  [mtp] adaptive: re-entry declined (head out of sync by {d}) -> serial for the rest of the request\n",
                            .{drift},
                        );
                        self.mtp_adaptive.declineReentry();
                        may_reenter = false;
                    }
                }
            }
            if (may_reenter) {
                log.info(
                    "  [mtp] adaptive: kv {d} crossed into bucket {s} -> mtp\n",
                    .{ self.mtpKvLen(), round_cost.bucketName(self.xfm.round_cost.layout, b) },
                );
                self.spec_disabled_runtime = false;
                self.spec_disable_reason = .none;
                self.mtp_serial_exit = .drain;
            }
        }

        // Serial block: the runtime-disabled fallback (acceptance floor,
        // `--max-mtp-ctx`, the adaptive switch), the bounded serial probe, or
        // the ramp back to this function's entry invariant. Same hand-off
        // contract as the drafter (`next()`'s transition shim seeds
        // pending_logits).
        if (self.spec_disabled_runtime or self.mtp_serial_left > 0 or self.mtp_serial_exit != .none) {
            try self.mtpDetachHead(allocator, self.mtpSerialMayResume());
            self.mtpMaybeReleaseModuleHead();
            return try self.mtpSerialTick(allocator);
        }

        const xfm = self.xfm;
        const s = xfm.s;
        const head = self.mtp.?;
        // Cross-request EV seed: inherit the head's last healthy acceptance
        // surface so the controller plans from round 1 instead of re-warming
        // (~10 legacy rounds + a +1/round base climb — a third of a short
        // generation). Demotion stays instant (EMA decay + sticky disable are
        // per-request), so a workload change costs a few rounds, not the win.
        // `MLX_SERVE_MTP_FORCE_DEPTH` is a measurement mode: every round drafts
        // exactly n and the controller never plans, so a seed must not be
        // applied (nor, at deinit, published).
        if (self.mtp_ev_rounds == 0 and self.mtp_attempted == 0 and
            mtpAdaptiveEnabled() and mtpEvSeedEnabled() and mtpForcedDepth() == null)
        {
            if (head.evSeed()) |seed| {
                self.mtp_ev_accept = seed.accept;
                self.mtp_ev_m_lo_prev = @min(@max(seed.m_lo, 1), mtp_mod.MAX_DEPTH);
                self.mtp_ev_rounds = MTP_EV_WARMUP_ROUNDS;
            }
        }
        const tracing = mtpTraceEnabled();
        var ph: io_util.Stopwatch = undefined;
        if (tracing) {
            ph = io_util.Stopwatch.init(self.timer.io);
            // Close the scheduler gap opened at the previous round's return.
            if (self.mtp_gap_watch) |*gw| self.mtp_trace.add(.gap, gw.read());
            self.mtp_gap_watch = null;
        }
        // Always-on round wall-clock for the live-cost round EMA (the
        // denominator of the sync fraction). Read at both commit exits.
        const livecost = mtpLiveCostEnabled();
        var round_watch = io_util.Stopwatch.init(self.timer.io);

        // ── Phase 0/1: acquire this round's draft chain ──
        // Round plan: fixed mode (and EV warmup) is a single chunk at the
        // windowed adaptive depth; post-warmup EV mode plans a base chunk
        // m_lo plus a confidence-gated extension to m_hi (see the EV
        // controller section below). `chain.m` is the tokens actually
        // drafted this round — it grows from m_lo to m_hi iff the gate
        // clears at the chunk boundary below.
        //
        // Either the PREVIOUS round built + dispatched chunk A at its tail
        // (cross-round pipelining, `mtpMaybePreDraft` — the drafts are
        // already materializing on the GPU) or we build it here (round 1,
        // MLX_SERVE_MTP_PREDRAFT=0, or a round with no successor state).
        //
        // No head-cache snapshot in either case: a snapshot refcount-shares
        // the head's KV buffer, which forces every draft append's
        // slice_update to copy-on-write the WHOLE history buffer (~268
        // MB/append at 64k). Rollback is truncate — offset-only — since
        // draft entries only ever append past chain.off0 (the committed
        // origin, resolved from the pending history stash when one exists).
        var chain: MtpPreDraft = if (self.mtp_pre_draft) |pd| blk: {
            self.mtp_pre_draft = null;
            // The pre-draft consumed the history stash when it was built.
            std.debug.assert(self.mtp_hist_stash == null);
            std.debug.assert(pd.t1 == self.next_token_id);
            break :blk pd;
        } else blk: {
            const plan_now = self.mtpRoundPlan();
            var c = try self.mtpChainInit(allocator, plan_now, self.next_token_id);
            errdefer c.deinit(allocator);
            try self.mtpChainBuild(&c, 0, plan_now.m_lo);
            if (mtpEarlyDispatchEnabled()) try mtpChainDispatch(&c, 0, plan_now.m_lo);
            break :blk c;
        };
        defer chain.deinit(allocator);
        const plan = chain.plan;
        const m_lo: u32 = plan.m_lo;
        const m_max: u32 = plan.m_hi;
        const t1: u32 = chain.t1;
        const mtp_off0: usize = chain.off0;
        if (tracing) {
            self.mtp_trace.add(.draft, ph.read());
            ph.reset();
        }

        // ── chunk-A boundary: extension decision (two-chunk EV plans) ──
        // The one bounded sync of the round; near-free when the chain was
        // pre-drafted (ids + confidences already materialized).
        if (m_max > m_lo) {
            var sync_watch = io_util.Stopwatch.init(self.timer.io);
            const chain_ln = try readChainConfidence(chain.draft_arrs[0..m_lo], chain.conf_arrs.?[0..m_lo], s);
            const sync_ns = sync_watch.read();
            // Live sync-cost EMA: measured only on considered rounds, so it
            // holds the true per-sync cost (not the suppressed-round average).
            if (livecost) self.mtp_ev_sync_ms = mtpEmaMs(self.mtp_ev_sync_ms, sync_ns);
            if (tracing) {
                self.mtp_trace.add(.sync, sync_ns);
                ph.reset();
            }
            if (chain_ln >= plan.tau_ln) {
                chain.m = m_max;
                self.mtp_ext_rounds += 1;
                self.mtp_ext_dry_streak = 0;
                try self.mtpChainBuild(&chain, m_lo, m_max);
                if (mtpEarlyDispatchEnabled()) try mtpChainDispatch(&chain, m_lo, m_max);
            } else {
                // Considered but the gate didn't clear — feeds the
                // dry-spell policy (mtpExtDryAllows).
                self.mtp_ext_dry_streak +|= 1;
            }
            if (tracing) {
                self.mtp_trace.add(.ext, ph.read());
                ph.reset();
            }
        }
        const m: u32 = chain.m;
        // Chain complete — release the chain hidden (only chunk B needed it).
        if (chain.h_chain) |h| {
            _ = mlx.mlx_array_free(h);
            chain.h_chain = null;
        }
        const drafts = chain.drafts;
        const draft_arrs = chain.draft_arrs;
        const q_probs = chain.q_probs;

        // ── Phase 2: record rollback anchors (NO snapshot on the GDN path) ──
        // A KVCache.snapshot() refcount-shares the KV buffers, which forces
        // verify's slice_update writes to COPY-on-write every full-attention
        // layer's WHOLE buffer — ~4.3 GB per round at 64k context, the
        // dominant round cost at long context. On a GDN trunk (every real MTP
        // target is qwen3_5-family hybrid) rollback needs only the pre-verify
        // LENGTH (KV truncate is offset-only; the stale tail past it is
        // unreachable) plus the verify pass's per-position SSM capture, so no
        // snapshot is taken at all. A hypothetical pure-attention target
        // (ssm_entries == null) keeps the proven snapshot + re-forward path.
        const kv_step_snap = self.ctx.cache.step;
        const gdn_trunk = self.ctx.ssm_entries != null;
        var kv_snap: ?transformer_mod.KVCacheSnapshot = if (gdn_trunk) null else try self.ctx.cache.snapshot();
        defer if (kv_snap) |*snap| snap.deinit();
        const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

        // ── Phase 3: verify input [t1, drafts...] as one [1, 1+m] tensor ──
        const reshape_2d = [_]c_int{ 1, 1 };
        var t1_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t1_2d);
        try mlx.check(mlx.mlx_reshape(&t1_2d, chain.t1_arr, &reshape_2d, 2, s));

        var verify_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_input);
        {
            const drafts_2d = try allocator.alloc(mlx.mlx_array, m);
            defer {
                for (drafts_2d) |arr| _ = mlx.mlx_array_free(arr);
                allocator.free(drafts_2d);
            }
            for (draft_arrs[0..m], drafts_2d) |dlazy, *out| {
                out.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(out, dlazy, &reshape_2d, 2, s));
            }
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, t1_2d);
            for (drafts_2d) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
        }

        var new_hidden = mlx.mlx_array_new();
        var verify_hidden_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_hidden_all);
        // Enable per-position SSM capture for the verify pass on a GDN trunk
        // so partial accept can roll back without re-forwarding the accepted
        // prefix (mirrors nextPld — the re-forward re-runs the 48-layer
        // sequential recurrence AND a full trunk weight read, and at depth > 1
        // MOST rounds are partial, so it dominated the round cost).
        self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
        // DIAGNOSTIC (MLX_SERVE_MTP_TRACE_SYNC=1): drain the GPU before the
        // verify build so the `sync` lap shows the pending lazy work (draft
        // chain, rollback, commit) and `verify` the forward alone.
        if (tracing and std.c.getenv("MLX_SERVE_MTP_TRACE_SYNC") != null) {
            var sync_watch = io_util.Stopwatch.init(self.timer.io);
            try mlx.check(mlx.mlx_array_eval(verify_input));
            self.mtp_trace.add(.sync, sync_watch.read());
            ph.reset();
        }
        // Captures the post-final-norm hidden at the LAST position (next
        // round's h_prev) AND all 1+m positions (history re-append).
        //
        // qwen4_exp: the n-gram PLE at trunk layer 1 is a HOST gather keyed on
        // these ids, and every id past t1 is still a lazy draft — gathering
        // eagerly parks the build of layers 2..N behind the whole draft chain
        // finishing on the GPU. Defer the leaf so the graph builds while the
        // chain runs, then sync ONCE (`flushDeferredPle`, below) before Phase
        // 4 evaluates anything. Other arches never set `ple_pending`.
        self.ctx.ple_defer = true;
        const verify_logits = xfm.forwardWithCaptureAll(&self.ctx, verify_input, &new_hidden, &verify_hidden_all) catch |e| {
            self.ctx.ple_defer = false;
            self.ctx.capture_ssm_seq = false;
            xfm.discardDeferredPle(&self.ctx);
            return e;
        };
        self.ctx.ple_defer = false;
        self.ctx.capture_ssm_seq = false;
        // Always free the transient capture buffers before returning, however
        // we exit this round (full accept, partial accept, or error).
        defer if (self.ctx.ssm_entries) |entries| {
            for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
        };
        self.mtp_attempted += 1;
        self.mtp_drafted_tokens += m;
        if (tracing) {
            self.mtp_trace.add(.verify, ph.read());
            ph.reset();
        }
        // Fill the deferred PLE leaf: the one host read of the verify ids, and
        // the point at which the entry's n-gram history + `spec_ple_tokens`
        // advance. Must precede BOTH the first eval of anything downstream of
        // the leaf (Phase 4) and `ssmRollbackFromCapture` (Phase 5).
        try xfm.flushDeferredPle(&self.ctx);

        // ── Phase 4: decide longest accepted prefix ──
        // Stochastic path is fully BATCHED: accept probabilities for every
        // draft AND a candidate correction token for every possible reject
        // position are built lazily (draft ids stay lazy arrays — never read
        // on the CPU inside a graph-building loop), then ONE async eval
        // realizes the whole round. The old per-draft probAt()/sampleResidual()
        // calls cost one GPU round-trip sync EACH — 3-5 syncs per round that
        // stalled the pipeline for milliseconds while the GPU sat idle.
        const stochastic = self.sampling.temperature > 0.01;
        const vl_shape = mlx.getShape(verify_logits);

        var per_pos_probs: ?[]mlx.mlx_array = null;
        defer if (per_pos_probs) |slots| {
            for (slots) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };
        var accept_p_vec = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(accept_p_vec);
        var accept_q_vec = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(accept_q_vec);
        var corr_samples: ?[]mlx.mlx_array = null;
        defer if (corr_samples) |slots| {
            for (slots) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };
        // Batched-corrections path: ONE [1+m] pre-sampled correction array
        // instead of the per-position corr_samples slots.
        var corr_batch: mlx.mlx_array = .{ .ctx = null };
        defer if (corr_batch.ctx != null) {
            _ = mlx.mlx_array_free(corr_batch);
        };

        if (stochastic and mtpBatchCorrEnabled()) {
            const probs_all = try probsAllPositions(verify_logits, self.sampling, s);
            defer _ = mlx.mlx_array_free(probs_all);
            const bg = try mtpBatchedAcceptGraph(
                probs_all,
                draft_arrs[0..m],
                if (q_probs) |qs| qs[0..m] else null,
                m,
                s,
            );
            corr_batch = bg.corr_samples;
            _ = mlx.mlx_array_free(accept_p_vec);
            accept_p_vec = bg.accept_p;
            if (bg.accept_q.ctx != null) {
                _ = mlx.mlx_array_free(accept_q_vec);
                accept_q_vec = bg.accept_q;
            }
        } else if (stochastic) {
            const slice_strides = [_]c_int{ 1, 1, 1 };
            // Filtered + softmaxed target probs for ALL 1+m positions in one
            // batched kernel set, then per-position slice VIEWS (no copies).
            const probs_all = try probsAllPositions(verify_logits, self.sampling, s);
            defer _ = mlx.mlx_array_free(probs_all);
            const slots = try allocator.alloc(mlx.mlx_array, 1 + m);
            per_pos_probs = slots;
            for (slots, 0..) |*slot, idx| {
                slot.* = mlx.mlx_array_new();
                const start = [_]c_int{ 0, @intCast(idx), 0 };
                const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
                try mlx.check(mlx.mlx_slice(slot, probs_all, &start, 3, &stop, 3, &slice_strides, 3, s));
            }

            // accept_p_vec[k] = target_p[k][draft_k], gathered with the LAZY
            // draft id array → [m] f32 after one eval.
            {
                const taken = try allocator.alloc(mlx.mlx_array, m);
                defer {
                    for (taken) |arr| _ = mlx.mlx_array_free(arr);
                    allocator.free(taken);
                }
                for (0..m) |k| {
                    taken[k] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_take_axis(&taken[k], slots[k], draft_arrs[k], -1, s));
                }
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (taken) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
                var cat = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(cat);
                try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
                try mlx.check(mlx.mlx_astype(&accept_p_vec, cat, .float32, s));
            }

            // accept_q_vec[k] = q_k[draft_k] — the proposal's own density at
            // the sampled draft, for the Leviathan ratio (sharp drafts only).
            if (q_probs) |qslots| {
                const taken = try allocator.alloc(mlx.mlx_array, m);
                defer {
                    for (taken) |arr| _ = mlx.mlx_array_free(arr);
                    allocator.free(taken);
                }
                for (0..m) |k| {
                    taken[k] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_take_axis(&taken[k], qslots[k], draft_arrs[k], -1, s));
                }
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (taken) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
                var cat = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(cat);
                try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
                try mlx.check(mlx.mlx_astype(&accept_q_vec, cat, .float32, s));
            }

            // Candidate correction for every possible reject position a<m
            // (residual sample) plus the full-accept bonus at a=m. Only the
            // one at the realized `accepted` is read; the rest are a few
            // vocab-length ops of throwaway GPU work — far cheaper than a
            // second synchronous softmax+categorical round-trip.
            var indices = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(indices);
            try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vl_shape[2])), 1, .int32, s));

            const corrs = try allocator.alloc(mlx.mlx_array, 1 + m);
            corr_samples = corrs;
            for (corrs, 0..) |*slot, a| {
                slot.* = mlx.mlx_array_new();
                if (a < m) {
                    // residual = max(target_p − proposal, 0): the proposal is
                    // the FULL sharpened q distribution under sharp drafts
                    // (exact Leviathan residual), the one-hot of the lazy
                    // draft id under greedy drafts (arange == id).
                    var diff = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(diff);
                    if (q_probs) |qslots| {
                        try mlx.check(mlx.mlx_subtract(&diff, per_pos_probs.?[a], qslots[a], s));
                    } else {
                        var onehot_b = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(onehot_b);
                        try mlx.check(mlx.mlx_equal(&onehot_b, indices, draft_arrs[a], s));
                        var onehot = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(onehot);
                        try mlx.check(mlx.mlx_astype(&onehot, onehot_b, .float32, s));
                        try mlx.check(mlx.mlx_subtract(&diff, per_pos_probs.?[a], onehot, s));
                    }
                    const zero = mlx.mlx_array_new_float(0.0);
                    defer _ = mlx.mlx_array_free(zero);
                    var residual = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(residual);
                    try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));
                    var log_res = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(log_res);
                    try mlx.check(mlx.mlx_log(&log_res, residual, s));
                    const null_key = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(null_key);
                    try mlx.check(mlx.mlx_random_categorical(slot, log_res, -1, null_key, s));
                } else {
                    var log_p = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(log_p);
                    try mlx.check(mlx.mlx_log(&log_p, per_pos_probs.?[m], s));
                    const null_key = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(null_key);
                    try mlx.check(mlx.mlx_random_categorical(slot, log_p, -1, null_key, s));
                }
            }
        }

        var verify_argmax = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_argmax);
        if (!stochastic) {
            try mlx.check(mlx.mlx_argmax_axis(&verify_argmax, verify_logits, 2, false, s));
        }
        _ = mlx.mlx_array_free(verify_logits);
        if (tracing) {
            self.mtp_trace.add(.corr, ph.read());
            ph.reset();
        }

        // ── Phase 4b: one batched async eval for the whole round ──
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (draft_arrs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
            if (stochastic) {
                _ = mlx.mlx_vector_array_append_value(eval_vec, accept_p_vec);
                if (q_probs != null) _ = mlx.mlx_vector_array_append_value(eval_vec, accept_q_vec);
                if (corr_batch.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(eval_vec, corr_batch);
                } else {
                    for (corr_samples.?) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
                }
            } else {
                _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax);
            }
            _ = mlx.mlx_vector_array_append_value(eval_vec, new_hidden);
            _ = mlx.mlx_vector_array_append_value(eval_vec, verify_hidden_all);
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
        for (draft_arrs[0..m], 0..) |arr, idx| {
            try mlx.check(mlx.mlx_array_eval(arr));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, arr));
            drafts[idx] = @intCast(v);
        }
        if (!stochastic) {
            // Separate graph branch from the draft chain — force it before
            // bulk-reading (see the v26.5.6 0%-acceptance note in nextDrafter).
            try mlx.check(mlx.mlx_array_eval(verify_argmax));
        }

        var accepted: u32 = 0;
        if (stochastic) {
            // Sharp drafts: full Leviathan ratio min(1, p/q) against the
            // proposal's own density. Greedy-forced drafts keep the exact
            // one-hot rule min(1, target_p).
            try mlx.check(mlx.mlx_array_eval(accept_p_vec));
            const p_data = mlx.mlx_array_data_float32(accept_p_vec) orelse {
                return error.MlxArrayDataNull;
            };
            var q_data: ?[*]const f32 = null;
            if (q_probs != null) {
                try mlx.check(mlx.mlx_array_eval(accept_q_vec));
                q_data = mlx.mlx_array_data_float32(accept_q_vec) orelse {
                    return error.MlxArrayDataNull;
                };
            }
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const accept_prob: f32 = if (q_data) |qd| specAcceptProb(p_data[k], qd[k]) else @min(1.0, p_data[k]);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse {
                return error.MlxArrayDataNull;
            };
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_argmax: u32 = @intCast(argmax_data[k]);
                if (target_argmax != drafts[k]) break;
                accepted += 1;
            }
        }

        accepted = capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );

        const next_pending: u32 = blk: {
            if (stochastic) {
                if (corr_batch.ctx != null) {
                    // Pre-sampled [1+m] batch; index at the realized accept.
                    try mlx.check(mlx.mlx_array_eval(corr_batch));
                    const d = mlx.mlx_array_data_int32(corr_batch) orelse {
                        return error.MlxArrayDataNull;
                    };
                    break :blk @intCast(d[accepted]);
                }
                // Pre-sampled in the round batch; realized already.
                const corr = corr_samples.?[accepted];
                try mlx.check(mlx.mlx_array_eval(corr));
                var v: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&v, corr));
                break :blk @intCast(v);
            } else {
                const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse {
                    return error.MlxArrayDataNull;
                };
                break :blk @intCast(argmax_data[accepted]);
            }
        };

        if (tracing) {
            self.mtp_trace.add(.eval, ph.read());
            ph.reset();
        }
        log.debug("  [mtp-round] off0={d} t1={d} m={d}/{d} drafts={any} accepted={d}\n", .{ mtp_off0, t1, m, m_max, drafts[0..m], accepted });

        // ── Phase 5a: stash the committed history for a DEFERRED append ──
        // The old shape paid a second head forward (appendHistory) here every
        // round to rebuild history from true verify hiddens — then the next
        // round's first draft re-entered the head anyway. Instead, stash the
        // (tokens, hiddens) pair — tokens as [t1, drafts[0..accepted]], the
        // hiddens the SAME concat of h_prev + ORIGINAL verify hiddens the old
        // appendHistory received — and fold the append into that first draft
        // step (the i==0 merged branch above). The head cache keeps this
        // round's draft tail past mtp_off0 until the consume-time truncate;
        // nothing reads it in between. Rounds with no successor (EOS/length/
        // runtime disable) never pay for the append; deinit frees the stash.
        {
            std.debug.assert(self.mtp_hist_stash == null);
            const n_commit: usize = accepted;
            const ids_i32 = try allocator.alloc(i32, 1 + n_commit);
            defer allocator.free(ids_i32);
            ids_i32[0] = @intCast(t1);
            for (drafts[0..n_commit], 0..) |d, idx| ids_i32[1 + idx] = @intCast(d);
            const id_shape = [_]c_int{@intCast(1 + n_commit)};
            const stash_ids = mlx.mlx_array_new_data(ids_i32.ptr, &id_shape, 1, .int32);
            errdefer _ = mlx.mlx_array_free(stash_ids);

            var hist_hidden = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(hist_hidden);
            if (n_commit == 0) {
                try mlx.check(mlx.mlx_array_set(&hist_hidden, self.last_hidden));
            } else {
                const vh_shape = mlx.getShape(verify_hidden_all);
                var vh_slice = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(vh_slice);
                const start = [_]c_int{ 0, 0, 0 };
                const stop = [_]c_int{ 1, @intCast(n_commit), vh_shape[2] };
                const strides = [_]c_int{ 1, 1, 1 };
                try mlx.check(mlx.mlx_slice(&vh_slice, verify_hidden_all, &start, 3, &stop, 3, &strides, 3, s));
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                _ = mlx.mlx_vector_array_append_value(vec, self.last_hidden);
                _ = mlx.mlx_vector_array_append_value(vec, vh_slice);
                try mlx.check(mlx.mlx_concatenate_axis(&hist_hidden, vec, 1, s));
            }
            self.mtp_hist_stash = .{
                .ids = stash_ids,
                .hidden = hist_hidden,
                .n = 1 + n_commit,
                .off0 = mtp_off0,
            };
        }
        if (tracing) {
            self.mtp_trace.add(.hist, ph.read());
            ph.reset();
        }

        // ── Phase 5b: commit / rollback the trunk ──
        if (accepted == m) {
            const tokens = try allocator.alloc(u32, 1 + m);
            tokens[0] = t1;
            for (drafts[0..m], 0..) |d, idx| tokens[1 + idx] = d;

            try self.generated_ids.append(allocator, t1);
            for (drafts[0..m]) |d| try self.generated_ids.append(allocator, d);

            if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
            self.last_hidden = new_hidden;
            self.has_last_hidden = true;

            self.mtp_accepted_tokens += m;
            self.next_token_id = next_pending;
            self.advanceStep(1 + m);

            if (mtpAdaptiveEnabled()) self.updateMtpEvRound(m, m) else self.updateMtpDepth(m, m);
            if (tracing) {
                self.mtp_trace.add(.commit, ph.read());
                ph.reset();
            }
            try self.mtpMaybePreDraft(allocator);
            if (tracing) {
                self.mtp_trace.add(.predraft, ph.read());
                self.mtp_gap_watch = io_util.Stopwatch.init(self.timer.io);
            }
            self.mtpTraceRoundEnd(m, m, m_lo);
            self.mtpRoundEndObserve(m, m + 1, m_max > m_lo, m_lo, plan.width_trial, @as(f32, @floatFromInt(round_watch.read())) / @as(f32, std.time.ns_per_ms));
            if (livecost) self.mtp_ev_round_ms = mtpEmaMs(self.mtp_ev_round_ms, round_watch.read());
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = m,
            };
        }

        // Partial accept: roll back the trunk. On a GDN trunk the verify pass
        // captured per-position SSM/conv state, so roll back by truncating the
        // KV cache to the accepted length and slicing the capture — NO
        // re-forward of the accepted prefix (mirrors nextPld's fast path; the
        // re-forward is a full trunk weight read, and at depth > 1 most rounds
        // are partial, so it dominated the round cost). The next round's
        // h_prev is the TRUE verify hidden at the last committed position
        // (input index `accepted`), which forwardWithCaptureAll captured.
        // Non-GDN archs keep the proven restore + re-forward fallback.
        _ = mlx.mlx_array_free(new_hidden);

        const gdn_captured = if (self.ctx.ssm_entries) |entries|
            entries.len > 0 and entries[0].spec_state_seq.ctx != null
        else
            false;

        var re_new_hidden = mlx.mlx_array_new();
        if (gdn_captured) {
            const accepted_len: usize = 1 + @as(usize, accepted);
            // `truncate` overwrites cache.step with its length arg; on this
            // family cache.step is a stale counter the model never reads
            // (positioning is moe_seq_offset), so preserve the pre-verify
            // value — keeps prefix-cache kv_step bookkeeping identical to
            // the restore-based fallback (same rule as nextPld).
            try self.ctx.cache.truncate(moe_seq_offset_snap + accepted_len, s);
            self.ctx.cache.step = kv_step_snap;
            for (self.ctx.ssm_entries.?) |*entry| {
                try transformer_mod.ssmRollbackFromCapture(entry, accepted, 1 + m, s);
            }
            self.ctx.moe_seq_offset.* = moe_seq_offset_snap + accepted_len;

            const vh_shape = mlx.getShape(verify_hidden_all);
            const start = [_]c_int{ 0, @intCast(accepted), 0 };
            const stop = [_]c_int{ 1, @as(c_int, @intCast(accepted)) + 1, vh_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&re_new_hidden, verify_hidden_all, &start, 3, &stop, 3, &strides, 3, s));
        } else if (kv_snap) |*snap| {
            try self.ctx.cache.restore(snap);
            self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

            const re_seq_len: c_int = @intCast(1 + accepted);
            const re_input_buf = try allocator.alloc(i32, 1 + accepted);
            defer allocator.free(re_input_buf);
            re_input_buf[0] = @intCast(t1);
            for (drafts[0..accepted], 0..) |d, idx| re_input_buf[1 + idx] = @intCast(d);
            const re_shape = [_]c_int{ 1, re_seq_len };
            const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(re_input);

            const re_logits = try xfm.forwardWithCapture(&self.ctx, re_input, &re_new_hidden);
            _ = mlx.mlx_array_free(re_logits);
        } else {
            // GDN trunk whose verify pass produced no capture — cannot roll
            // back safely. Unreachable on real targets (every qwen3_5-family
            // GDN layer populates the capture when capture_ssm_seq is set);
            // pinned by tests/test_mtp_equivalence.sh.
            _ = mlx.mlx_array_free(re_new_hidden);
            return error.MtpRollbackUnavailable;
        }

        const tokens = try allocator.alloc(u32, 1 + accepted);
        tokens[0] = t1;
        for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;

        try self.generated_ids.append(allocator, t1);
        for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);

        if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
        self.last_hidden = re_new_hidden;
        self.has_last_hidden = true;

        self.mtp_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.advanceStep(1 + accepted);

        if (mtpAdaptiveEnabled()) self.updateMtpEvRound(m, accepted) else self.updateMtpDepth(m, accepted);
        if (tracing) {
            self.mtp_trace.add(.commit, ph.read());
            ph.reset();
        }
        try self.mtpMaybePreDraft(allocator);
        if (tracing) {
            self.mtp_trace.add(.predraft, ph.read());
            self.mtp_gap_watch = io_util.Stopwatch.init(self.timer.io);
        }
        self.mtpTraceRoundEnd(m, accepted, m_lo);
        self.mtpRoundEndObserve(m, accepted + 1, m_max > m_lo, m_lo, plan.width_trial, @as(f32, @floatFromInt(round_watch.read())) / @as(f32, std.time.ns_per_ms));
        if (livecost) self.mtp_ev_round_ms = mtpEmaMs(self.mtp_ev_round_ms, round_watch.read());
        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    // ── MTP adaptive depth ──
    // Unlike the drafter's binary gate, the MTP head has a useful fallback
    // BETWEEN "full depth" and "off": depth 1. Measured on Qwen3.6-27B
    // (M4 Max, greedy): creative content runs 48% per-draft at depth 2
    // (a regression vs AR) but 73% at depth 1 (1.11× AR); code runs 89% at
    // depth 2 (1.45× AR). A windowed controller demotes/promotes between
    // depths and only disables outright when even depth 1 can't pay for its
    // verify overhead.
    pub const MTP_DEPTH_WINDOW: u32 = 16; // rounds in the moving window
    pub const MTP_DEPTH_SWITCH_WARMUP: u32 = 5; // rounds before re-evaluating after a switch
    // Thresholds assume the capture-based rollback (no re-forward on partial
    // accept): a rejected draft costs ONLY its own MTP-layer + draft-head
    // pass (~2 ms), not a second trunk forward (~30-50 ms). Extra depth pays
    // whenever the marginal accept probability clears draft-cost/trunk-cost
    // ≈ 0.05-0.10, so the demote floor sits far lower than the old
    // re-forward-era 0.60 — hysteresis band keeps switch churn down.
    pub const MTP_DEMOTE_BELOW: f32 = 0.40; // per-draft rate at depth > 1 → step down
    pub const MTP_PROMOTE_ABOVE: f32 = 0.60; // per-draft rate below configured depth → step up
    // Disable floor = the MEASURED depth-1 breakeven plus margin, not a
    // quality judgment: a d1 round costs ~AR+6 ms (44 vs 38.4 ms at 8K on
    // the 27B, mtp-trace 2026-07-22) and yields (1+p) tokens, so speculation
    // pays down to p ≈ 0.15 — at p=0.45 it is +27% over AR. The old 0.50
    // floor sticky-disabled mid-request on the oQ4e head at long context
    // (window rate dips 0.45-0.55) and cratered 16K/32K ladder decode to
    // bare AR (24-26 tok/s vs oMLX's 41-47, which never fully disables).
    pub const MTP_DISABLE_BELOW: f32 = 0.20; // per-draft rate at depth 1 → disable (sticky)
    pub const MTP_PROMOTE_COOLDOWN: u32 = 32; // rounds promotion stays blocked after a demotion

    /// Greedy (argmax) MTP draft proposals — DEFAULT ON. Measured on the
    /// Jundot oQ4e head (2026-07-22, ladder coding prompts, temp 0.6): the
    /// sharpened stochastic proposal + exact Leviathan ratio (oMLX
    /// Lightning's scheme, see mtpDraftSamplingFor) reads 48-50% per-draft
    /// vs greedy's 58-63% — on LOW-entropy agent/code content the temp-0.6
    /// target is sharper than any sampled proposal, so `min(1,
    /// p_target[argmax])` dominates `1 − TV(p, q)`; draft-head precision
    /// (3-bit/8-bit/trunk q) moved nothing. MLX_SERVE_MTP_DRAFT_GREEDY=0
    /// flips to the sharpened sampled proposal (exactness holds either way —
    /// pinned by the toy-vocab test; only the acceptance RATE differs).
    var mtp_draft_greedy_cache: ?bool = null;
    fn mtpDraftGreedy() bool {
        if (mtp_draft_greedy_cache) |v| return v;
        var on = true;
        if (std.c.getenv("MLX_SERVE_MTP_DRAFT_GREEDY")) |p| {
            const val = std.mem.span(p);
            if (val.len > 0 and val[0] == '0') on = false;
        }
        mtp_draft_greedy_cache = on;
        return on;
    }

    // ── Sharpened stochastic draft proposals (Lightning-class acceptance) ──
    // Drafts for a stochastic target are SAMPLED from a fixed sharper
    // distribution (constants mirror oMLX's _DRAFT_SAMPLER_*: their comment —
    // matched-temp drafting "collapses to ~10-20% on high-entropy content"),
    // acceptance is the full Leviathan/Chen ratio min(1, p/q) with q = the
    // draft sampler's own filtered distribution, and rejection re-samples
    // from normalize(max(p-q, 0)). Output distribution provably equals the
    // target's filtered p for ANY proposal q (pinned by the toy-vocab
    // exactness test); q only moves the ACCEPTANCE RATE, which is why the
    // draft head's quantization never affects correctness.
    pub const MTP_DRAFT_TEMP: f32 = 0.6;
    pub const MTP_DRAFT_TOP_P: f32 = 0.95;
    pub const MTP_DRAFT_TOP_K: u32 = 20;

    /// Draft-proposal sampler for a round: greedy targets keep greedy drafts
    /// (temp-0 identity contract); stochastic targets draft from the fixed
    /// sharpened distribution unless greedy is forced. The constants are not
    /// tunable: swept 2026-08-04 on the oQ4e head (prose@0.6, forced depth 3,
    /// sampled drafts) at draft temp 0.6/0.7/0.85/1.0 and per-draft
    /// acceptance was flat at 49.7-56.9% — no arm beat 0.6.
    pub fn mtpDraftSamplingFor(target: SamplingParams, force_greedy: bool) SamplingParams {
        var d = target;
        if (force_greedy or target.temperature <= 0.01) {
            d.temperature = 0.0;
            return d;
        }
        d.temperature = MTP_DRAFT_TEMP;
        d.top_p = MTP_DRAFT_TOP_P;
        d.top_k = MTP_DRAFT_TOP_K;
        return d;
    }

    /// Full Leviathan acceptance ratio; q clamped so a sampled draft (q > 0
    /// by construction) can never divide by an underflowed zero.
    pub fn specAcceptProb(p: f32, q_draft: f32) f32 {
        return @min(1.0, p / @max(q_draft, 1e-12));
    }

    /// Pure depth-policy step. `rate` is the windowed per-draft acceptance
    /// probability. Returns the new depth; 0 means "disable speculation".
    pub fn mtpNextDepth(current: u32, configured: u32, rate: f32) u32 {
        if (current > 1 and rate < MTP_DEMOTE_BELOW) return current - 1;
        if (current <= 1 and rate < MTP_DISABLE_BELOW) return 0;
        if (current < configured and rate > MTP_PROMOTE_ABOVE) return current + 1;
        return current;
    }

    /// Confidence-gated depth decision. Demoting is cheap (still
    /// speculating) so it reacts on a small sample; DISABLING is sticky and
    /// PROMOTING raises verify cost, so both require more evidence. A
    /// 16-round window at a true 73% per-draft rate essentially never dips
    /// below the 0.50 disable floor; a 5-round window does (observed live:
    /// an early-story cold streak disabled a request that would have run
    /// 1.11x at depth 1).
    pub fn mtpDepthDecision(current: u32, configured: u32, rate: f32, window_rounds: u32, promote_blocked: bool) u32 {
        const next_depth = mtpNextDepth(current, configured, rate);
        if (next_depth == 0 and window_rounds < MTP_DEPTH_WINDOW) return current;
        if (next_depth > current and (window_rounds < 8 or promote_blocked)) return current;
        return next_depth;
    }

    /// Windowed adaptive-depth update, called once per nextMtp round with
    /// that round's (drafted, accepted) counts.
    fn updateMtpDepth(self: *Generator, drafted: u32, accepted: u32) void {
        const idx = self.mtp_window_idx % MTP_DEPTH_WINDOW;
        self.mtp_window_drafted[idx] = @intCast(drafted);
        self.mtp_window_accepted[idx] = @intCast(accepted);
        self.mtp_window_idx += 1;
        if (self.mtp_promote_cooldown > 0) self.mtp_promote_cooldown -= 1;
        if (self.mtp_rounds_since_switch < MTP_DEPTH_SWITCH_WARMUP) {
            self.mtp_rounds_since_switch += 1;
            return;
        }
        const n = @min(self.mtp_window_idx, MTP_DEPTH_WINDOW);
        var drafted_sum: u32 = 0;
        var accepted_sum: u32 = 0;
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            drafted_sum += self.mtp_window_drafted[i];
            accepted_sum += self.mtp_window_accepted[i];
        }
        if (drafted_sum == 0) return;
        const rate = @as(f32, @floatFromInt(accepted_sum)) / @as(f32, @floatFromInt(drafted_sum));
        const next_depth = mtpDepthDecision(self.mtp_depth_current, self.mtp_depth, rate, n, self.mtp_promote_cooldown > 0);
        if (next_depth == self.mtp_depth_current) return;
        if (next_depth == 0) {
            log.info(
                "  mtp=disabled (windowed per-draft rate {d:.2} < {d:.2} at depth 1)\n",
                .{ rate, MTP_DISABLE_BELOW },
            );
            self.spec_disabled_runtime = true;
            self.spec_disable_reason = .acceptance;
            return;
        }
        log.debug("  [mtp-depth] {d} -> {d} (windowed per-draft rate {d:.2})\n", .{ self.mtp_depth_current, next_depth, rate });
        if (next_depth < self.mtp_depth_current) self.mtp_promote_cooldown = MTP_PROMOTE_COOLDOWN;
        self.mtp_depth_current = next_depth;
        self.mtp_rounds_since_switch = 0;
        // Reset the window so the new depth is judged on its own rounds. The
        // price window goes with it for the same reason: its rounds priced a
        // depth this request has just abandoned.
        self.mtp_window_idx = 0;
        self.mtp_price.reset();
    }

    // ── MTP EV (expected-value) adaptive controller ──
    // Fixed-depth drafting is the warm-decode ceiling: at ~77% per-draft the
    // marginal chain decays with index, so one global depth wastes verify
    // width on hard stretches and leaves easy stretches (code boilerplate
    // where 8/8 accept) under-drafted. The EV controller tracks CONDITIONAL
    // per-index acceptance EMAs a[i] = P(draft i accepted | i-1 accepted) and
    // plans each round as two chunks: a base chunk `m_lo` (the static EV
    // optimum), then — when the head's own confidence on chunk A clears a
    // cost-derived threshold tau — an extension to `m_hi`. Only rounds that
    // CONSIDER extension pay the one bounded chunk-A sync; when the plan
    // collapses to m_lo == m_hi the round is byte-identical in shape to the
    // fixed-depth path (no confidence graph, no sync).
    // Disable via MLX_SERVE_MTP_ADAPTIVE=0 (reverts to the windowed
    // fixed-depth controller above).

    /// Default depth cap when `--mtp-depth` is not passed (0 = auto) and the
    /// EV controller is active. 6 keeps the verify forward at seq 1+6 = 7,
    /// the split-K verify-qmm kernel's ceiling on M1-M4. Eligible M5/G17
    /// targets use MTP_ADAPTIVE_NAX_CAP instead: their measured NAX round-cost
    /// surface makes depths 7/8 profitable. Explicit depths always win.
    /// This is the DEFAULT ROW only: `mtp.adaptiveDepthCapForMachine` lowers
    /// it on chips whose verify-width cliff was measured (M1 Pro: 4).
    pub const MTP_ADAPTIVE_DEFAULT_CAP: u32 = 6;
    pub const MTP_ADAPTIVE_NAX_CAP: u32 = 8;
    /// Fraction by which a width's cost-per-position may exceed the running
    /// best before the measured ladder calls it past the cliff. 5% is well
    /// inside the gap the tile cliff opens (the M4 ladder turns up 12% at
    /// width 8) and well outside probe rep-to-rep noise, which is bounded
    /// below by keeping the MIN of the reps.
    pub const MTP_CLIFF_TOLERANCE: f32 = 0.05;
    /// One-shot guard for the per-silicon cap log (the row is a property of
    /// the machine, so it is worth saying once per process, not per request).
    var mtp_depth_cap_logged: bool = false;
    /// Rounds of legacy (fixed-depth windowed) behavior while the EMAs fill.
    /// Warmup, but converges in ROUNDS, not 43 s of offline calibration.
    pub const MTP_EV_WARMUP_ROUNDS: u32 = 10;
    /// EMA step for the per-index acceptance estimates. 0.15 demotes fast on
    /// cold streaks (5 consecutive rejects: 0.72 -> 0.32) without letting a
    /// single unlucky round move the plan.
    pub const MTP_EV_EMA_BETA: f32 = 0.15;
    /// Optimistic prior for unobserved indices. Deliberately ABOVE the
    /// measured average per-draft rate (~77%): a deep index is only ever
    /// observed when extension fires, and on this cost surface the
    /// break-even conditional acceptance for a ramp position is ~0.78 — a
    /// realistic prior would sit razor-under it and extension would never
    /// get its first trial (measured live: ext_rounds=0 on a pure-echo
    /// workload). The tau gate (only near-perfect-confidence rounds extend)
    /// plus demote-fast EMAs bound the cost of an optimistic trial to a few
    /// rounds.
    pub const MTP_EV_PRIOR: f32 = 0.85;
    /// Clamp band for the extension confidence threshold.
    pub const MTP_EV_TAU_MIN: f32 = 0.05;
    pub const MTP_EV_TAU_MAX: f32 = 0.95;
    /// Minimum base-round EV (tokens per round-cost) for the exploration
    /// horizon: below this the base itself barely beats AR, so keeping an
    /// extension position reachable would only add sync tax to a dying
    /// speculation (the sticky-disable floor is likely next anyway). 1.10
    /// keeps fully-cold EMAs collapsed on every cost profile (cold best_r:
    /// generic 1.00, G17-NAX 1.07/1.08).
    pub const MTP_EV_EXPLORE_MIN_R: f32 = 1.10;

    /// Round-cost model in units of the fixed round cost (verify-forward
    /// floor + round eval/read + commit ≈ 1.0 ≈ 32 ms on the 27B since the
    /// deferred history append). Ratios are machine-stable where absolute ms
    /// are not. Refit via MLX_SERVE_MTP_TRACE on Qwen3.6-27B GDN (M4 Max,
    /// 2026-07-13, saturated fixed depths, same-session sweep AFTER the
    /// deferred-append round shape landed): T(1)=42.0, T(3)=62.6,
    /// T(6)=111.6, T(7)=114.9 ms — the surface is PIECEWISE: ~10.3 ms
    /// marginal per position in the flat verify region (seq <= 4), ~16.3
    /// ms/pos for positions 4-6, and position 7 nearly free (+3.3 ms —
    /// verify seq 8 rides the same row tile as 5-7), averaged into
    /// per_pos_hi. ATTRIBUTION (the GDN-vs-qmm µbench in transformer.zig,
    /// MLX_SERVE_GDN_UBENCH=1): the ladder is ~90% qmm ROW-COUNT cost — the
    /// GDN recurrence kernel is nearly flat over verify widths (0.13→0.26 ms
    /// per dispatch, T 2→64) and contributes <1 ms/round; the earlier
    /// "GDN sequential width ramp" reading was a mis-attribution.
    /// The old linear ~1.5 ms/pos model came from a depth-6 run whose
    /// windowed controller was silently demoting underneath — never fit
    /// costs from a run whose realized m_avg you didn't check.
    /// Override for live tuning (an explicit override selects the generic
    /// two-region surface even on M5, so all four values remain the
    /// complete backwards-compatible contract):
    /// MLX_SERVE_MTP_EV_COSTS="draft,per_pos_lo,per_pos_hi,sync".
    pub const MtpEvCosts = struct {
        draft: f32, // one sequential MTP-head step (fwd + draft lm_head)
        per_pos_lo: f32, // marginal verify+capture per position, flat region
        per_pos_hi: f32, // ... beyond flat_max (qmm row-tile ramp)
        flat_max: u32, // last draft index in the flat verify region
        sync: f32, // the chunk-A confidence read-back
        /// First draft depth whose verify forward lands on the M5 NAX tile.
        /// Zero disables the third region. Depth k verifies k+1 rows, so the
        /// default NAX M=8 takeover starts at draft position 7.
        nax_from: u32 = 0,
        per_pos_nax: f32 = 0.0,
        /// Serial-floor cost in ms at kv ~= 0 (the probe's width-1 rung).
        /// Zero — every hand-typed table — makes the kv term below a no-op,
        /// so the tables keep behaving exactly as they did.
        floor_ms: f32 = 0,
        /// Context length the marginals above were FITTED AT. The kv term
        /// scales relative to THIS, not to zero: refit #4 was measured at 8K,
        /// so its 0.20 already contains 8K of KV read and re-scaling it from
        /// a kv~=0 floor would discount it twice. Zero = unknown, which
        /// disables the kv term for that surface rather than guessing.
        kv_ref_tokens: u32 = 0,
        /// Learned per-KV-token round cost B (ms/token), the term that makes
        /// wide speculation approach FREE at long context.
        ///
        /// A verify forward of width k reads the weights once and the KV
        /// once, both SHARED across all k query rows; only arithmetic scales
        /// with k. So T(k, L) ~= W + B*L + C(k) and T(k,L)/T(1,L) -> 1 as L
        /// grows: the optimal width RISES with context, and every constant
        /// this controller ever shipped was fitted at ONE context length.
        /// Learned online rather than probed — a boot probe measures at
        /// kv ~= 0, where B is invisible, and probing it would mean
        /// allocating a 32k KV cache purely to time it.
        kv_ms_per_token: f32 = 0,
    };
    /// 2026-08-15 refit #4, AFTER the hd-256 causal sdpa split
    /// (`splitCausalSdpa` — dense verify q 6..9 now rides the vector path,
    /// invalidating the ramp the old hi partially priced): same-session
    /// saturated ECHO sweep, Jundot oQ4e 27B @8K cold reps
    /// (--prefix-cache-entries 0), M4 Max, MLX_SERVE_MTP_ADAPTIVE=0 forced
    /// depths, saturated (m_avg==N) trace windows only — T(1)=44.6,
    /// T(2)=51.0, T(3)=59.2, T(4)=68.2, T(6)=95.4, T(8)=142.3 ms → floor
    /// ≈ 38.2 ms. Composite marginals in floor units: k<=4 ≈ 0.20 (6.4-9.0
    /// ms/pos — flat_max moves 3 → 4, the old hi over-priced k4 at 0.34 and
    /// under-drafted it on moderate content), k5-6 ≈ 0.36 (13.6 ms/pos),
    /// k7-8 ≈ 0.62 (23.5 ms/pos — the plain-SIMD verify-qmm register cliff
    /// at M 8/9, expressed through the generic third region; only reachable
    /// when --mtp-depth forces past the generic cap of 6). The G17 NAX
    /// tables below predate the sdpa split — refit them on an M5 when one
    /// is available. draft/verify split not separately identifiable; only
    /// the sums enter the controller.
    /// (Refit #3, 2026-07-22, post round-pipelining, for the record:
    /// T(1)=45.4, T(2)=53.3 → .10/.10/.24@3/.01 on a ~37.9 ms floor.
    /// Refit #2, 2026-07-13, post-verify-qmm: T(1)=44.6, T(3)=54.4,
    /// T(6)=89.9 → .06/.06/.24/.02 on a ~40 ms floor.)
    pub const MTP_EV_DEFAULT_COSTS: MtpEvCosts = .{ .draft = 0.10, .per_pos_lo = 0.10, .per_pos_hi = 0.26, .flat_max = 4, .sync = 0.01, .nax_from = 7, .per_pos_nax = 0.52 };
    /// M5 Max/G17 refit (2026-07-17), same-session saturated fixed-depth
    /// sweep after the NAX m16 verify lane landed: T(1..4) ~= 41.35 ms,
    /// T(6)=62.15 ms, T(8)=68.39 ms. In floor units this identifies
    /// draft+hi ~= .21 and draft+nax ~= .10; T(8)/T(6) is reproduced by
    /// 2.19/1.99 = 1.1005 (measured 1.1004). The profile is selected only
    /// for the calibrated dense Qwen3.6-27B homogeneous affine-4/gs-64
    /// checkpoint with its native affine-8/gs-32 sidecar and successfully
    /// built affine-3/gs-64 draft head, when the trunk lm_head routes both
    /// M=8 and M=9 through NAX; every other combination retains DEFAULT.
    pub const MTP_EV_G17_NAX_COSTS: MtpEvCosts = .{
        .draft = 0.06,
        .per_pos_lo = 0.06,
        .per_pos_hi = 0.15,
        .flat_max = 3,
        .sync = 0.02,
        .nax_from = 7,
        .per_pos_nax = 0.04,
    };
    /// M5 Max/G17 affine-4/gs-32 sidecar refit (2026-07-18). A saturated
    /// fixed-depth sweep gave T(1)=36.04, T(3)=43.01, T(6)=62.56, and
    /// T(8)=66.06 ms. The fitted composite marginals (`draft + verify`) are
    /// .107/.200/.054; depths 4 and 5 independently validate the rounded
    /// .11/.20/.05 surface after matching-baseline correction. The split
    /// between draft and verify is not separately identifiable.
    pub const MTP_EV_G17_NAX_Q4_GS32_COSTS: MtpEvCosts = .{
        .draft = 0.03,
        .per_pos_lo = 0.08,
        .per_pos_hi = 0.17,
        .flat_max = 3,
        .sync = 0.02,
        .nax_from = 7,
        .per_pos_nax = 0.02,
    };
    /// M5 Max/G17 uniform affine-4/gs-64 Qwen3.8-27B refit (2026-08-16).
    /// Temperature-gated, reversed two-pass saturated echo gave T(1)=
    /// 33.61-33.91, T(3)=38.455-38.465, T(6)=56.555-57.000, and NAX T(8)=
    /// 61.890-61.925 ms/round. NAX-off T(8)=88.535-88.610 while depth-6
    /// on/off stayed within 0.8%, isolating the M>=8 lane. A 31.4 ms fitted
    /// floor gives rounded composite marginals .075/.195/.08. The profile
    /// stays distinct because this trunk is uniform q4, not mixed q4/q5/q6.
    pub const MTP_EV_G17_NAX_Q4_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.02,
        .per_pos_lo = 0.055,
        .per_pos_hi = 0.175,
        .flat_max = 3,
        .sync = 0.01,
        .nax_from = 7,
        .per_pos_nax = 0.06,
    };
    /// M5 Max/G17 uniform affine-6/gs-64 Qwen3.8-27B refit (2026-08-17).
    /// Temperature-gated, six-pass counterbalanced calibration gave median
    /// traced T(1)=44.98, T(3)=49.895, T(6)=64.495, and NAX T(8)=76.79
    /// ms/round. A 42.52 ms fitted floor yields rounded composite marginals
    /// .06/.115/.145 and reproduces T(8)/T(6): 1.815/1.525=1.1902 versus
    /// 1.1907 measured. Depth 8 was 108.86 tok/s versus 102.93 at depth 7.
    pub const MTP_EV_G17_NAX_Q6_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.02,
        .per_pos_lo = 0.04,
        .per_pos_hi = 0.095,
        .flat_max = 3,
        .sync = 0.01,
        .nax_from = 7,
        .per_pos_nax = 0.125,
    };
    /// M5 Max/G17 uniform affine-8/gs-64 Qwen3.8-27B refit (2026-08-17).
    /// Five settled counterbalanced pairs found the q8-specific NAX takeover
    /// at verify M=7: depth 6 rose from 73.42 to 77.89 tok/s (+6.09%), while
    /// M=6 was -1.14%. Median traced NAX T(6)=85.94 ms and T(8)=90.68 ms over
    /// the 53.0575 ms floor give normalized 1.62/1.71. The additive controller
    /// cannot encode the one-time M7 kernel transition separately, so k=5..6
    /// are a positive .19 bridge and the subsequent k=7..8 NAX marginal is
    /// .045. This reproduces both measured endpoints without negative costs.
    /// The profile is revoked when q8 NAX is off.
    pub const MTP_EV_G17_NAX_Q8_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.005,
        .per_pos_lo = 0.055,
        .per_pos_hi = 0.185,
        .flat_max = 4,
        .sync = 0.01,
        .nax_from = 7,
        .per_pos_nax = 0.04,
    };
    /// M5 Max/G17 oQ4e mixed affine q4/q5/q6, gs64 refit (2026-07-23).
    /// Saturated deterministic echo, same-session fixed widths:
    /// T(1)=34.37, T(3)=40.31, T(6)=61.20, T(8)=62.79 ms/round.
    /// Normalizing the inferred 31.40 ms floor gives composite marginals
    /// .095/.220/.025. The split below is arbitrary but positive; only
    /// `draft + per_pos_*` enters the controller. The profile is selected
    /// solely for the exact resident oQ4e layer-role fingerprint.
    pub const MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.02,
        .per_pos_lo = 0.075,
        .per_pos_hi = 0.20,
        .flat_max = 3,
        .sync = 0.01,
        .nax_from = 7,
        .per_pos_nax = 0.005,
    };
    /// qwen4_exp (Qwen3.8-Flash-Next 125B-A6B) on G17, M5 Max 40-core refit
    /// (2026-08-27/28). Fitted at ~8.5k context like every prior surface
    /// (a short-context fit planned too shallow at 8.5k, measured -4% —
    /// every constant this controller ever shipped was fitted at ONE
    /// context length, and 8K is the house one). Forced-depth saturated
    /// echo (the test_mtp_equivalence echo appended to ~8.3k of neutral
    /// filler, temp 0, `--prefix-cache-entries 0`, round-cost persist off),
    /// depths {1,2,3,4,6} x two reversed passes x 3 reps, medians of
    /// per-request round_ms: T(1)=25.41, T(2)=31.18, T(3)=35.70,
    /// T(4)=41.33, T(6)=53.58 ms -> 20.34 ms floor, composite
    /// (draft + per-position) marginals .257/.303 (SSE 0.28), shipped
    /// house-style below as `.draft = 0.02` plus per-position .237/.283 —
    /// composite minus draft, the same split every calibrated surface
    /// carries. Short-context fit for the record: T(1..4,6,8)=
    /// 23.61/28.12/33.91/38.87/50.16/62.70 -> floor 18.24, composite
    /// .283/.310/.344 (the .344 ships as per-position `.325`).
    /// A verify row on this arch is BYTES — a second row reads its own
    /// experts — so the marginals sit 3-5x above the dense-sidecar surfaces
    /// and there is NO NAX flattening: routed expert banks never ride the
    /// vqmm NAX lane (that is the M5 plan's grouped-expert item, and this
    /// table is its before-number). `per_pos_nax` carries the short-context
    /// k>=7 STEEPENING (+11%/pos, a measurement not a discount) and is
    /// inert under this profile's default cap of 6. sync stays nominal per
    /// the dry-spell-gate doctrine (measured chunk-boundary sync ~1.7 ms is
    /// governed by the realized-rate gate, not the prior). draft/verify
    /// split arbitrary but positive; only the sums enter the controller.
    pub const MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.02,
        .per_pos_lo = 0.237,
        .per_pos_hi = 0.283,
        .flat_max = 4,
        .sync = 0.02,
        .nax_from = 7,
        .per_pos_nax = 0.325,
    };

    /// Minimum kv separation before two anchors can identify B. Below it the
    /// difference is dominated by round-to-round noise, not by the KV read.
    /// Round-end bookkeeping shared by the full- and partial-accept paths:
    /// the kv term, the regime gate and the round-cost table all read ONE
    /// inter-round wall clock (tok/s is measured between round ends, so
    /// per-round work outside the round stopwatch belongs to the width).
    fn mtpRoundEndObserve(self: *Generator, m: u32, tokens: u32, two_chunk: bool, m_lo: u32, width_trial: bool, round_ms: f32) void {
        // The serial cell measures PLAIN decode ticks. An interval that spans
        // a speculative round is neither arm's number, so the clock restarts
        // and the next block warms again.
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
        const post_warmup = self.mtp_ev_rounds >= MTP_EV_WARMUP_ROUNDS;
        // The table wants rounds the EV controller PLANNED: the round that
        // ends warmup was still the legacy controller's (a w2 sample there
        // anchored the table at 2 on the M4 base 9B).
        const ev_planned = self.mtp_ev_rounds > MTP_EV_WARMUP_ROUNDS;
        const wall = if (post_warmup) self.mtpRegimeWallMs(round_ms) else round_ms;
        const tok: f32 = @floatFromInt(tokens);
        if (post_warmup and self.spec_cost_solo and !width_trial) mtpRegimeObserve(&self.mtp_regime, two_chunk, m_lo, wall, tok);
        // A trial round was a single-chunk shape at another depth: not a
        // regime sample, but the shape its successor transitions from.
        if (width_trial) self.mtp_regime.last_two = false;
        // The table is the cost of the SINGLE-CHUNK shape at each width — the
        // quantity the m_lo loop compares. A two-chunk round chose its width
        // by its own confidence gate (tokens biased high) and paid a sync
        // (ms biased high); observed as width m_hi it read single-chunk 6
        // as better than the 5 -> 6 two-chunk it was measured FROM (M4 Max
        // 27B @16k, -4.5%). The shape question stays the regime gate's.
        const shape_changed = self.spec_round_prev_two_chunk != two_chunk or self.spec_round_prev_two_chunk2 != two_chunk;
        self.spec_round_prev_two_chunk2 = self.spec_round_prev_two_chunk;
        self.spec_round_prev_two_chunk = two_chunk;
        // The realized price of speculation for THIS request, on the same
        // wall clock the table folds and under the same solo gate (contention
        // only ever ADDS time). Unlike the table's cells this is not per
        // width: the vote it feeds asks whether the whole speculation is
        // worth running, so every non-trial round counts, extension included.
        // Gated with its ONE consumer (`mtpAdaptiveVoteFor`, arch-scoped):
        // on a sidecar pack this window is written and never read.
        if (post_warmup and self.spec_cost_solo and self.mtpAdaptiveModelOk()) self.mtp_price.observe(wall, tok, width_trial);
        self.specObserveRound(m, wall, tok, ev_planned and !two_chunk, shape_changed);
    }

    /// Feed the model's round-cost table (`Transformer.round_cost`, shared
    /// by every request and both block decoders). Width = drafts this round
    /// (0 = serial). The previous width is tracked on EVERY round so the
    /// first observed one is not a transition by accident; a rejected
    /// sample that is not an expected drop (contended, transition) logs its
    /// numbers — a silent reject is the probe's anti-pattern.
    fn specObserveRound(self: *Generator, width: u32, wall_ms: f32, tokens: f32, observe: bool, shape_changed: bool) void {
        const transition = shape_changed or
            (if (self.spec_round_prev_width) |p| p != width else true) or
            (if (self.spec_round_prev_width2) |p| p != width else true);
        self.spec_round_prev_width2 = self.spec_round_prev_width;
        self.spec_round_prev_width = width;
        if (!observe) return;
        const kv = self.mtpKvLen();
        const v = self.xfm.round_cost.observe(width, kv, wall_ms, tokens, self.spec_cost_solo, transition);
        if (v == .bad_sample or v == .out_of_range) {
            log.warn("[spec-cost] table rejected sample ({s}): width={d} kv={d} ms={d:.2} tokens={d:.1}\n", .{ @tagName(v), width, kv, wall_ms, tokens });
        }
    }

    /// Write the model's round-cost table to disk when this request folded
    /// new samples (request end; inference thread, so no lock).
    pub fn persistRoundCost(self: *Generator) void {
        const t = &self.xfm.round_cost;
        // BOTH clocks: a boot that only learned serial cells still persists
        // them, and a boot that learned nothing writes nothing.
        const folded = round_cost.totalFolded(t);
        if (folded == t.stored_at) return;
        round_cost.storeCached(self.timer.io, self.xfm.round_cost_key_buf[0..self.xfm.round_cost_key_len], t);
        t.stored_at = folded;
    }

    /// Round-cost table kill switch — MLX_SERVE_MTP_COST_TABLE=0 keeps the
    /// table OBSERVING (its `[spec-stats]` fields stay comparable across an
    /// A/B) but the plan reads only the fitted prior and no width trial runs.
    var mtp_cost_table_cache: ?bool = null;
    fn mtpCostTableEnabled() bool {
        if (mtp_cost_table_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_COST_TABLE")) |p| std.mem.span(p) else null;
        const on = mtpLiveCostEnabledFromEnv(raw);
        mtp_cost_table_cache = on;
        return on;
    }

    /// KV length the next verify forward reads.
    fn mtpKvLen(self: *const Generator) u32 {
        return @intCast(self.prompt_ids_owned.len + self.generated_ids.items.len);
    }

    /// Marginal round cost of draft position k (1-based), kv-blind. This is
    /// the fitted SHAPE; `mtpEvMarginalCostAt` applies the kv term.
    pub fn mtpEvMarginalCost(costs: MtpEvCosts, k: u32) f32 {
        const verify_cost = if (costs.nax_from != 0 and k >= costs.nax_from)
            costs.per_pos_nax
        else if (k <= costs.flat_max)
            costs.per_pos_lo
        else
            costs.per_pos_hi;
        return costs.draft + verify_cost;
    }

    /// One round's draft plan. `m_hi > m_lo` means "pay the chunk-A sync and
    /// extend to m_hi when the chain log-confidence clears tau_ln".
    pub const MtpRoundPlan = struct {
        m_lo: u32,
        m_hi: u32,
        tau_ln: f32,
        /// A width-trial round (single-chunk at m_lo+1, measuring the round
        /// cost table): skipped by the regime gate, which compares shapes
        /// at ONE base depth.
        width_trial: bool = false,
    };

    /// Resolve the configured depth cap. 0 = auto (`--mtp-depth` not passed):
    /// MTP_ADAPTIVE_NAX_CAP only when the EV controller and a calibrated G17
    /// cost profile are both active, MTP_ADAPTIVE_DEFAULT_CAP otherwise, and
    /// DEFAULT_DEPTH in fixed mode. Explicit values always win.
    pub fn mtpDepthCapForProfile(configured: u32, adaptive: bool, profile: mtp_mod.MtpCostProfile) u32 {
        const chip = ane_mod.chipBrand();
        const cap = mtpDepthCapResolved(configured, adaptive, profile, chip);
        // Name the row ONCE when a per-silicon measurement is what fenced the
        // depth. Without it `[spec-stats] depth=4` on an M1 Pro reads the same
        // as the EV controller having chosen 4, or as `--mtp-depth 4` — and
        // the fence is exactly what someone debugging MTP there needs to see.
        if (configured == 0 and adaptive and !mtp_depth_cap_logged) {
            const row = mtp_mod.adaptiveDepthCapForMachine(chip, MTP_ADAPTIVE_DEFAULT_CAP);
            if (cap == row.cap and row.cap < MTP_ADAPTIVE_DEFAULT_CAP) {
                mtp_depth_cap_logged = true;
                log.info("[mtp] adaptive depth cap {d} ({s} row, default {d})\n", .{
                    row.cap,
                    row.label,
                    MTP_ADAPTIVE_DEFAULT_CAP,
                });
            }
        }
        return cap;
    }

    /// The cap with no per-silicon row applied: explicit wins, else the
    /// adaptive default (fixed mode keeps DEFAULT_DEPTH).
    pub fn mtpDepthCapFree(configured: u32) u32 {
        if (configured != 0) return @min(mtp_mod.MAX_DEPTH, @max(1, configured));
        return if (mtpAdaptiveEnabled()) MTP_ADAPTIVE_DEFAULT_CAP else mtp_mod.DEFAULT_DEPTH;
    }

    /// Same, with the chip string injected (tests, and the one live caller).
    pub fn mtpDepthCapForProfileChip(configured: u32, adaptive: bool, profile: mtp_mod.MtpCostProfile, chip: []const u8) u32 {
        return mtpDepthCapResolved(configured, adaptive, profile, chip);
    }

    /// Same, with the machine's MEASURED verify ladder when one exists.
    ///
    /// The per-silicon rows exist because the EV controller cannot see time:
    /// it scores accepted-tokens-per-round, which on an M1 Pro IMPROVED
    /// (3.65 -> 4.00) while realized tok/s fell 21%. A measured curve sees
    /// exactly the cliff those rows encode, on whatever chip and whatever
    /// quant geometry is actually resident — which the chip key cannot
    /// express (the split-K lane is 4-bit/g64 only, so a 6-bit pack on the
    /// same box is a different answer). The calibrated NAX profiles keep
    /// their own cap until a probe on that silicon is validated against it.
    pub fn mtpDepthCapResolved(
        configured: u32,
        adaptive: bool,
        profile: mtp_mod.MtpCostProfile,
        chip: []const u8,
    ) u32 {
        if (configured != 0) return @min(mtp_mod.MAX_DEPTH, @max(1, configured));
        if (!adaptive) return mtp_mod.DEFAULT_DEPTH;
        return switch (profile) {
            // The per-silicon row is the COLD-START cap: the measured
            // round-cost table may plan above it on trusted widths.
            .generic => mtp_mod.adaptiveDepthCapForMachine(chip, MTP_ADAPTIVE_DEFAULT_CAP).cap,
            .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q4_gs64, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64 => MTP_ADAPTIVE_NAX_CAP,
            // qwen4's measured surface has no NAX region to reach: depths 7-8
            // price at .345/pos against sub-60% tail acceptance even on
            // saturated echo, so the calibrated cap keeps the default — the
            // NAX cap exists for surfaces that flatten past position 6.
            .g17_nax_qwen4_q4_gs64 => MTP_ADAPTIVE_DEFAULT_CAP,
        };
    }

    pub fn resolveMtpDepthCapForProfile(configured: u32, profile: mtp_mod.MtpCostProfile) u32 {
        return mtpDepthCapForProfile(configured, mtpAdaptiveEnabled(), profile);
    }

    /// Legacy q8 boolean selector retained for source compatibility.
    pub fn mtpDepthCapFor(configured: u32, adaptive: bool, nax_profile: bool) u32 {
        const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
        return mtpDepthCapForProfile(configured, adaptive, profile);
    }

    /// Legacy q8 boolean selector retained for source compatibility.
    pub fn resolveMtpDepthCap(configured: u32, nax_profile: bool) u32 {
        const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
        return resolveMtpDepthCapForProfile(configured, profile);
    }

    /// Expected committed tokens for an m-deep round: the always-committed t1
    /// plus the acceptance chain sum (draft k lands iff drafts 0..k all land).
    pub fn mtpEvExpectedTokens(a: []const f32, m: u32) f32 {
        var chain: f32 = 1.0;
        var tok: f32 = 1.0;
        var k: u32 = 0;
        while (k < m and k < a.len) : (k += 1) {
            chain *= a[k];
            tok += chain;
        }
        return tok;
    }

    pub fn mtpEvMarginalCostAt(costs: MtpEvCosts, k: u32, kv_len: u32) f32 {
        _ = kv_len;
        return mtpEvMarginalCost(costs, k);
    }

    /// Round cost in verify-base units (piecewise per-position marginals).
    pub fn mtpEvRoundCost(costs: MtpEvCosts, m: u32, with_sync: bool) f32 {
        return mtpEvRoundCostAt(costs, m, with_sync, 0);
    }

    pub fn mtpEvRoundCostAt(costs: MtpEvCosts, m: u32, with_sync: bool, kv_len: u32) f32 {
        var c: f32 = 1.0 + (if (with_sync) costs.sync else 0.0);
        var k: u32 = 1;
        while (k <= m) : (k += 1) c += mtpEvMarginalCostAt(costs, k, kv_len);
        return c;
    }

    /// Pure EV plan: pick (m_lo, m_hi, tau) maximizing expected tok/round-cost.
    /// `m_lo_max` damps the base-depth climb (hysteresis — the caller passes
    /// last round's m_lo + 1); demotions are never damped.
    ///  1. m_lo = argmax over single-chunk depths of E(m)/T(m).
    ///  2. m_hi = deepest position whose marginal chain still pays under FULL
    ///     confidence in chunk A (the best case the gate can certify).
    ///  3. tau: extend when the confidence-implied chain beats the stop rate
    ///     on the margin — c*S/dt > r  =>  tau = r*dt/S.
    /// There is deliberately NO separate "is the sync worth it" gate: tau
    /// already keeps low-confidence rounds single-chunk, the horizon check
    /// collapses m_hi on cold EMAs (killing the sync entirely), and a
    /// prior-weighted expected-gain gate measurably starves exploration —
    /// deep indices are only observed when extension fires, so a gate fed by
    /// their priors blocks the first trial forever (live: ext_rounds=0 on
    /// pure echo).
    pub fn mtpEvPlanFor(a: []const f32, cap_in: u32, costs: MtpEvCosts, m_lo_max: u32) MtpRoundPlan {
        return mtpEvPlanForAt(a, cap_in, costs, m_lo_max, 0);
    }

    pub fn mtpEvPlanForAt(a: []const f32, cap_in: u32, costs: MtpEvCosts, m_lo_max: u32, kv_len: u32) MtpRoundPlan {
        return mtpEvPlanSrc(a, cap_in, MtpCostSource.init(costs, kv_len, null), m_lo_max);
    }

    /// Smallest marginal the table may report, in floor units: a measured
    /// pair that reads a wider round CHEAPER is noise, and a non-positive
    /// marginal would make every extension free.
    pub const MTP_EV_TABLE_MIN_MARGINAL: f32 = 0.02;

    /// Where the EV plan's round costs come from: the measured round-cost
    /// table when the kv bucket (or its nearest active neighbour) has
    /// MIN_WIDTHS measured widths, else the fitted surface as the cold-start
    /// prior. The table is in ms; the plan compares ratios but has one
    /// absolute threshold (`MTP_EV_EXPLORE_MIN_R`), so the table is scaled
    /// into FLOOR UNITS at its narrowest measured width — the two sources
    /// agree there by construction and the table's slopes take over.
    pub const MtpCostSource = struct {
        costs: MtpEvCosts,
        kv_len: u32 = 0,
        table: ?*const round_cost.Table = null,
        bucket: usize = 0,
        scale: f32 = 0,

        pub fn init(costs: MtpEvCosts, kv_len: u32, table: ?*const round_cost.Table) MtpCostSource {
            var src = MtpCostSource{ .costs = costs, .kv_len = kv_len, .table = table };
            const t = table orelse return src;
            const b = t.bucketToRead(kv_len) orelse return src;
            const ref = t.narrowestMeasured(b) orelse return src;
            const ref_ms = t.measuredMs(ref, b) orelse return src;
            if (!(ref_ms > 0) or ref > mtp_mod.MAX_DEPTH) return src;
            src.bucket = b;
            src.scale = mtpEvRoundCostAt(costs, ref, false, kv_len) / ref_ms;
            return src;
        }

        pub fn fromTable(self: MtpCostSource) bool {
            return self.scale > 0;
        }

        /// Realized tokens per single-chunk round at a MEASURED width, else
        /// null (the acceptance-EMA model fills in). The model's E(6) on the
        /// M4 Max @16k said 6.85 while the measured single-6 round emitted
        /// 6.0 — the 6th draft's rejections cost a rollback the model cannot
        /// see, and the table can.
        pub fn measuredTokens(self: MtpCostSource, m: u32) ?f32 {
            if (self.scale <= 0) return null;
            return self.table.?.measuredTok(m, self.bucket);
        }

        pub fn roundCost(self: MtpCostSource, m: u32, with_sync: bool) f32 {
            if (self.scale > 0) {
                const sync: f32 = if (with_sync) self.costs.sync else 0.0;
                const t = self.table.?;
                if (t.roundMs(m, self.bucket)) |ms| return ms * self.scale + sync;
                // Past the widest measured width every extra position costs
                // max(last measured slope, prior marginal): continuous with
                // the table, never more optimistic than the prior (a shallow
                // measured slope run upward let the plan race to 8 unmeasured).
                if (t.widestMeasured(self.bucket)) |w| {
                    if (m > w) {
                        const slope = (t.lastSlope(self.bucket) orelse 0.0) * self.scale;
                        var c = t.measuredMs(w, self.bucket).? * self.scale;
                        var k: u32 = w + 1;
                        while (k <= m) : (k += 1) {
                            c += @max(slope, mtpEvMarginalCostAt(self.costs, k, self.kv_len));
                            // One sample is evidence for WORSE, never for
                            // cheaper: an untrusted cell floors the cost.
                            if (t.rawMs(k, self.bucket)) |raw| c = @max(c, raw * self.scale);
                        }
                        return c + sync;
                    }
                }
            }
            return mtpEvRoundCostAt(self.costs, m, with_sync, self.kv_len);
        }

        pub fn marginal(self: MtpCostSource, k: u32) f32 {
            if (self.scale > 0 and k >= 1) {
                return @max(self.roundCost(k, false) - self.roundCost(k - 1, false), MTP_EV_TABLE_MIN_MARGINAL);
            }
            return mtpEvMarginalCostAt(self.costs, k, self.kv_len);
        }
    };

    pub fn mtpEvPlanSrc(a: []const f32, cap_in: u32, src: MtpCostSource, m_lo_max: u32) MtpRoundPlan {
        const cap: u32 = @intCast(@min(@as(usize, @max(1, cap_in)), a.len));
        const lo_cap: u32 = @min(cap, @max(1, m_lo_max));
        var m_lo: u32 = 1;
        var best_r: f32 = 0.0;
        // With measured costs the standing base (m_lo_max - 1, what the
        // caller planned last round) keeps its place unless a challenger
        // beats it by the switch margin: cells are EMAs of a few samples
        // and a 5% reversal between neighbours is noise, while every
        // change of base is a transition round the table will not count
        // and the model pays for.
        const standing: u32 = if (src.fromTable()) @min(lo_cap, m_lo_max -| 1) else 0;
        var standing_r: f32 = 0.0;
        var m: u32 = 1;
        while (m <= lo_cap) : (m += 1) {
            const tok = src.measuredTokens(m) orelse mtpEvExpectedTokens(a, m);
            const r = tok / src.roundCost(m, false);
            if (m == standing) standing_r = r;
            if (r > best_r) {
                best_r = r;
                m_lo = m;
            }
        }
        if (standing >= 1 and m_lo != standing and best_r <= standing_r * (1.0 + round_cost.SWITCH_MARGIN)) {
            m_lo = standing;
            best_r = standing_r;
        }
        if (m_lo >= cap) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        var m_hi: u32 = m_lo;
        var cond: f32 = 1.0;
        var s_sum: f32 = 0.0; // expected extension tokens, conditional on chunk A
        var t_sum: f32 = 0.0; // extension marginal cost (piecewise)
        while (m_hi < cap) {
            cond *= a[m_hi];
            const mc = src.marginal(m_hi + 1);
            if (cond <= best_r * mc) break;
            s_sum += cond;
            t_sum += mc;
            m_hi += 1;
        }
        if (m_hi == m_lo) {
            // Exploration valve for the horizon itself (2026-07-22): the
            // index one past m_lo is observable ONLY when extension fires —
            // at m_lo-deep rounds a[m_lo] never updates — so an EMA dragged
            // cold under an earlier workload (or an easier tau regime)
            // would otherwise close the horizon FOREVER: m stays at m_lo,
            // a[m_lo] never re-proves itself, and pure echo runs at
            // ext_rounds=0 (the exact class the TAU_MAX valve exists for,
            // resurfaced by refit #3's honest marginals). Whenever the base
            // itself pays, keep ONE extension position reachable at the
            // clamped tau; the dry-spell gate bounds the consideration tax
            // on workloads where confidence never clears it.
            if (best_r <= MTP_EV_EXPLORE_MIN_R) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
            const mc = src.marginal(m_lo + 1);
            // A MEASURED marginal the position cannot repay even at full
            // confidence (1 <= best_r * mc) closes the valve: the valve
            // exists to observe a[m_lo], and with the table that is the
            // width trial's job. The fitted prior keeps the valve exactly
            // as Phase 1 measured it.
            if (src.fromTable() and best_r * mc >= 1.0) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
            const s = @max(a[m_lo], 1e-6);
            const tau_x = std.math.clamp(best_r * mc / s, MTP_EV_TAU_MIN, MTP_EV_TAU_MAX);
            return .{ .m_lo = m_lo, .m_hi = m_lo + 1, .tau_ln = @log(tau_x) };
        }
        // The TAU_MAX clamp doubles as the exploration valve: on razor-thin
        // horizons the honest tau approaches 1 ("never extend"), and 0.95
        // lets near-perfect-confidence rounds through so the deep EMAs can
        // observe reality at all.
        const tau = std.math.clamp(best_r * t_sum / s_sum, MTP_EV_TAU_MIN, MTP_EV_TAU_MAX);
        return .{ .m_lo = m_lo, .m_hi = m_hi, .tau_ln = @log(tau) };
    }

    /// Update the conditional acceptance EMAs from one realized round.
    /// Acceptance is prefix-structured: indices < accepted saw a success, the
    /// index AT `accepted` saw the reject (when one happened), and deeper
    /// indices were never conditionally reached — no observation.
    pub fn mtpEvObserve(a: []f32, drafted: u32, accepted: u32, beta: f32) void {
        var i: usize = 0;
        while (i < accepted and i < a.len) : (i += 1) a[i] += beta * (1.0 - a[i]);
        if (accepted < drafted and accepted < a.len) a[accepted] += beta * (0.0 - a[accepted]);
    }

    /// Chain log-confidence of a drafted chunk: sum of per-draft log p_head,
    /// clamped to <= 0 per term (a log-prob is never positive; bf16 noise
    /// can be). NaN poisons to -inf so a broken confidence can never extend.
    pub fn mtpChainLogConf(confs: []const f32) f32 {
        var sum: f32 = 0.0;
        for (confs) |c| {
            if (std.math.isNan(c)) return -std.math.inf(f32);
            sum += @min(0.0, c);
        }
        return sum;
    }

    /// Per-phase wall-time accumulator behind MLX_SERVE_MTP_TRACE=1. Pure
    /// bookkeeping; `nextMtp` stamps phases with a Stopwatch and emits one
    /// summary line every LOG_EVERY rounds. Zero cost when the env is absent
    /// (every stamp is guarded on the cached env check).
    pub const MtpTrace = struct {
        pub const LOG_EVERY: u32 = 32;
        pub const Phase = enum(u4) { draft, sync, ext, verify, corr, eval, hist, commit, predraft, gap };
        pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;

        rounds: u32 = 0,
        ns: [N_PHASES]u64 = @splat(0),
        drafted: u64 = 0,
        accepted: u64 = 0,
        extended: u32 = 0,
        /// Per draft index: rounds that drafted index i / accepted it.
        drafted_idx: [mtp_mod.MAX_DEPTH]u32 = @splat(0),
        accepted_idx: [mtp_mod.MAX_DEPTH]u32 = @splat(0),

        pub fn add(self: *MtpTrace, phase: Phase, dur_ns: u64) void {
            self.ns[@backingInt(phase)] += dur_ns;
        }

        /// Close one round; true when a summary line is due (caller logs,
        /// then calls reset()).
        pub fn endRound(self: *MtpTrace, drafted_n: u32, accepted_n: u32, was_extended: bool) bool {
            self.rounds += 1;
            self.drafted += drafted_n;
            self.accepted += accepted_n;
            if (was_extended) self.extended += 1;
            var i: usize = 0;
            while (i < drafted_n and i < mtp_mod.MAX_DEPTH) : (i += 1) {
                self.drafted_idx[i] += 1;
                if (i < accepted_n) self.accepted_idx[i] += 1;
            }
            return self.rounds >= LOG_EVERY;
        }

        /// `a0/a1/...` acceptance per drafted index (only indices ever drafted).
        pub fn accIdxStr(self: *const MtpTrace, buf: []u8) []const u8 {
            var w: std.Io.Writer = .fixed(buf);
            var i: usize = 0;
            while (i < mtp_mod.MAX_DEPTH and self.drafted_idx[i] > 0) : (i += 1) {
                w.print("{s}{d:.2}", .{ if (i == 0) "" else "/", @as(f64, @floatFromInt(self.accepted_idx[i])) / @as(f64, @floatFromInt(self.drafted_idx[i])) }) catch break;
            }
            return w.buffered();
        }

        pub fn avgMs(self: *const MtpTrace, phase: Phase) f64 {
            if (self.rounds == 0) return 0.0;
            return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
                (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn totalAvgMs(self: *const MtpTrace) f64 {
            if (self.rounds == 0) return 0.0;
            var total: u64 = 0;
            for (self.ns) |v| total += v;
            return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn reset(self: *MtpTrace) void {
            self.* = .{};
        }
    };

    /// Per-phase wall-time accumulator behind MLX_SERVE_DFLASH_TRACE=1.
    /// `nextDflash` stamps phases with a Stopwatch and emits one summary line
    /// every LOG_EVERY rounds. Zero cost when the env is absent.
    pub const DflashTrace = struct {
        pub const LOG_EVERY: u32 = 16;
        pub const Phase = enum(u4) { assist, head, verify, accept, append, gap };
        pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;

        rounds: u32 = 0,
        ns: [N_PHASES]u64 = @splat(0),
        accepted: u64 = 0,

        pub fn add(self: *DflashTrace, phase: Phase, dur_ns: u64) void {
            self.ns[@backingInt(phase)] += dur_ns;
        }

        /// Close one round; true when a summary line is due (caller logs,
        /// then calls reset()).
        pub fn endRound(self: *DflashTrace, accepted_n: u32) bool {
            self.rounds += 1;
            self.accepted += accepted_n;
            return self.rounds >= LOG_EVERY;
        }

        pub fn avgMs(self: *const DflashTrace, phase: Phase) f64 {
            if (self.rounds == 0) return 0.0;
            return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
                (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn totalAvgMs(self: *const DflashTrace) f64 {
            if (self.rounds == 0) return 0.0;
            var total: u64 = 0;
            for (self.ns) |v| total += v;
            return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn reset(self: *DflashTrace) void {
            self.* = .{};
        }
    };

    /// Sampled DFlash drafts — DEFAULT OFF, measured negative.
    /// `MLX_SERVE_DFLASH_SAMPLED_DRAFTS=1` draws each draft from the
    /// request's own filtered distribution and accepts through the full
    /// Leviathan ratio; off keeps the argmax draft and its one-hot q.
    ///
    /// The theory said this should help: a greedy draft's acceptance is
    /// p(argmax q), which a flattened target row deflates, while a matched
    /// proposal accepts at 1 − TV(p, q). Measured on Muse 4-bit at temp 0.7
    /// (3 reps x 4 prompts, same-boot serial reference), it is a LOSS —
    /// 52.4 tok/s / 54.5% per-draft against 55.5 / 57.7% greedy. This
    /// assistant's ARGMAX tracks the trunk well while its distribution SHAPE
    /// does not, so a one-hot proposal is the better proposal and sampling
    /// spends the round's budget landing on tokens the trunk would not pick.
    /// Greedy requests never reach this either way.
    var dflash_sampled_drafts_cache: ?bool = null;
    fn dflashSampledDraftsEnabled() bool {
        if (dflash_sampled_drafts_cache) |v| return v;
        var on = false;
        if (std.c.getenv("MLX_SERVE_DFLASH_SAMPLED_DRAFTS")) |p| {
            const val = std.mem.span(p);
            if (val.len > 0 and val[0] == '1') on = true;
        }
        dflash_sampled_drafts_cache = on;
        return on;
    }

    /// `MLX_SERVE_DFLASH_SELECTOR`: default ON when the sidecar ships a
    /// selector (DFlash2). "0" forces the v1 draft arms (argmax / block
    /// sampling) for A/Bs — the conv layers stay, they are the checkpoint.
    var dflash_selector_cache: ?bool = null;
    /// `MLX_SERVE_DFLASH_MARKOV=0` drafts a DSpark sidecar's block from its
    /// UNCORRECTED base logits (the v1 arm) — an A/B lever for measuring what
    /// the Markov chain is worth, never a default.
    fn dflashMarkovEnabled() bool {
        const p = std.c.getenv("MLX_SERVE_DFLASH_MARKOV") orelse return true;
        return !std.mem.eql(u8, std.mem.span(p), "0");
    }

    fn dflashSelectorEnabled() bool {
        if (dflash_selector_cache) |v| return v;
        var on = true;
        if (std.c.getenv("MLX_SERVE_DFLASH_SELECTOR")) |p| {
            const val = std.mem.span(p);
            if (val.len > 0 and val[0] == '0') on = false;
        }
        dflash_selector_cache = on;
        return on;
    }

    var dflash_trace_cache: ?bool = null;
    fn dflashTraceEnabled() bool {
        if (dflash_trace_cache) |v| return v;
        const on = readEnvBool("MLX_SERVE_DFLASH_TRACE");
        dflash_trace_cache = on;
        return on;
    }

    /// Close one traced dflash round; emits + resets at the cadence.
    fn dflashTraceRoundEnd(self: *Generator, accepted: u32) void {
        if (!self.dflash_trace.endRound(accepted)) return;
        const t = &self.dflash_trace;
        log.info(
            "  [dflash-trace] rounds={d} avg_ms assist={d:.2} head={d:.2} verify={d:.2} accept={d:.2} append={d:.2} gap={d:.2} total={d:.2} | acc_avg={d:.2}\n",
            .{
                t.rounds,
                t.avgMs(.assist),
                t.avgMs(.head),
                t.avgMs(.verify),
                t.avgMs(.accept),
                t.avgMs(.append),
                t.avgMs(.gap),
                t.totalAvgMs(),
                @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
            },
        );
        t.reset();
    }

    /// Adaptive (EV) controller gate — DEFAULT ON. MLX_SERVE_MTP_ADAPTIVE=0
    /// reverts to the fixed-depth windowed controller for same-boot A/Bs.
    var mtp_adaptive_cache: ?bool = null;
    var mtp_force_depth_cache: ??u32 = null;

    pub fn mtpAdaptiveEnabled() bool {
        if (mtp_adaptive_cache) |v| return v;
        var on = true;
        if (std.c.getenv("MLX_SERVE_MTP_ADAPTIVE")) |p| {
            const val = std.mem.span(p);
            if (val.len > 0 and val[0] == '0') on = false;
        }
        mtp_adaptive_cache = on;
        return on;
    }

    /// Cross-request EV seeding gate — default ON; set
    /// MLX_SERVE_MTP_EV_SEED=0 to keep request planning independent.
    var mtp_ev_seed_cache: ?bool = null;
    fn mtpEvSeedEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return value.len == 0 or value[0] != '0';
    }

    fn mtpEvSeedEnabled() bool {
        if (mtp_ev_seed_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_EV_SEED")) |p| std.mem.span(p) else null;
        const on = mtpEvSeedEnabledFromEnv(raw);
        mtp_ev_seed_cache = on;
        return on;
    }

    /// Early draft-chain dispatch (round pipelining) — DEFAULT ON. Fires the
    /// draft-chain graph at the GPU as soon as Phase 1 finishes building it,
    /// so the head chain runs while the CPU builds the verify/accept graphs
    /// (~2 ms the GPU used to spend idle each round). Dispatch timing only —
    /// lazy sampling ops bind their PRNG key at graph BUILD time, so values
    /// are identical. MLX_SERVE_MTP_EARLY_DISPATCH=0 restores the serial
    /// round shape for same-boot A/Bs.
    var mtp_early_dispatch_cache: ?bool = null;
    pub fn mtpEarlyDispatchEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return value.len == 0 or value[0] != '0';
    }

    fn mtpEarlyDispatchEnabled() bool {
        if (mtp_early_dispatch_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_EARLY_DISPATCH")) |p| std.mem.span(p) else null;
        const on = mtpEarlyDispatchEnabledFromEnv(raw);
        mtp_early_dispatch_cache = on;
        return on;
    }

    /// Cross-round pre-draft (round pipelining) — DEFAULT ON. Builds and
    /// dispatches the next round's chunk-A draft chain at the current
    /// round's tail (see mtpMaybePreDraft) so the GPU drafts while the CPU
    /// does emit/SSE bookkeeping. MLX_SERVE_MTP_PREDRAFT=0 reverts to
    /// head-of-round drafting for same-boot A/Bs (combine with
    /// MLX_SERVE_MTP_EARLY_DISPATCH=0 for the fully serial round shape).
    var mtp_predraft_cache: ?bool = null;
    pub fn mtpPredraftEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return value.len == 0 or value[0] != '0';
    }

    fn mtpPredraftEnabled() bool {
        if (mtp_predraft_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_PREDRAFT")) |p| std.mem.span(p) else null;
        const on = mtpPredraftEnabledFromEnv(raw);
        mtp_predraft_cache = on;
        return on;
    }

    var mtp_trace_cache: ?bool = null;
    fn mtpTraceEnabled() bool {
        if (mtp_trace_cache) |v| return v;
        const on = readEnvBool("MLX_SERVE_MTP_TRACE");
        mtp_trace_cache = on;
        return on;
    }

    fn parseMtpEvCostsOverride(raw: []const u8) ?MtpEvCosts {
        var values: [4]f32 = undefined;
        var it = std.mem.splitScalar(u8, raw, ',');
        var i: usize = 0;
        while (i < values.len) : (i += 1) {
            const part = it.next() orelse return null;
            const value = std.fmt.parseFloat(f32, std.mem.trim(u8, part, " ")) catch return null;
            if (!std.math.isFinite(value)) return null;
            values[i] = value;
        }
        if (it.next() != null) return null;
        if (values[0] <= 0.0 or values[1] <= 0.0 or values[2] <= 0.0 or values[3] < 0.0) return null;

        var c = MTP_EV_DEFAULT_COSTS;
        c.draft = values[0];
        c.per_pos_lo = values[1];
        c.per_pos_hi = values[2];
        c.sync = values[3];
        // The override contract is the two-region surface: never inherit a
        // third region (M5 NAX or the generic k>=7 cliff) the four values
        // can't express.
        c.nax_from = 0;
        c.per_pos_nax = 0.0;
        return c;
    }

    /// Pure profile/override selector. A valid explicit four-value override
    /// starts from DEFAULT (rather than silently inheriting the hardware
    /// profile), so a value copied from an M1-M4 tuning run means the same
    /// thing on M5. Empty/partial/malformed values are ignored atomically;
    /// they must not silently leave an auto-cap-8 target on generic costs.
    pub fn mtpEvCostsForProfile(profile: mtp_mod.MtpCostProfile, override: ?[]const u8) MtpEvCosts {
        const selected = switch (profile) {
            .generic => MTP_EV_DEFAULT_COSTS,
            .g17_nax_q8_gs32 => MTP_EV_G17_NAX_COSTS,
            .g17_nax_q4_gs32 => MTP_EV_G17_NAX_Q4_GS32_COSTS,
            .g17_nax_q4_gs64 => MTP_EV_G17_NAX_Q4_GS64_COSTS,
            .g17_nax_q6_gs64 => MTP_EV_G17_NAX_Q6_GS64_COSTS,
            .g17_nax_q8_gs64 => MTP_EV_G17_NAX_Q8_GS64_COSTS,
            .g17_nax_oq4e_q4_gs64 => MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS,
            .g17_nax_qwen4_q4_gs64 => MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS,
        };
        if (override) |raw| {
            return parseMtpEvCostsOverride(raw) orelse selected;
        }
        return selected;
    }

    /// Legacy q8 boolean selector retained for source compatibility.
    pub fn mtpEvCostsFor(nax_profile: bool, override: ?[]const u8) MtpEvCosts {
        const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
        return mtpEvCostsForProfile(profile, override);
    }

    fn mtpEvCosts(profile: mtp_mod.MtpCostProfile) MtpEvCosts {
        return mtpEvCostsForProfile(profile, if (std.c.getenv("MLX_SERVE_MTP_EV_COSTS")) |p| std.mem.span(p) else null);
    }

    /// Extension dry-spell gate constants: after MTP_EXT_DRY_ROUNDS
    /// consecutive extension-CONSIDERED rounds whose confidence gate never
    /// cleared, consideration collapses to single-chunk for
    /// MTP_EXT_DRY_COOLDOWN rounds, then re-opens for a fresh trial.
    /// Rationale (2026-07-22, post-round-pipelining sweep): a two-chunk
    /// round pays a REAL mid-pipeline cost the EV surface can't see — the
    /// chunk-A boundary sync fires ~0.3 ms after the pre-draft dispatch, so
    /// it blocks on the still-running head chain + confidence graphs
    /// (measured sync 0.5-3.7 ms/round at ext_rate 0.00; EV default lost
    /// 3-6 tok/s at 0.5K to fixed depths from this tax alone). The gate is
    /// fed by the REALIZED extension rate, never by priors — the war-story
    /// failure mode (docs/gotchas/engine-mlx.md, ext_rounds=0 on echo) was
    /// a prior-fed expected-gain gate blocking the FIRST trial; this one
    /// guarantees a fresh 16-round trial every 48 rounds, and a single
    /// extension firing resets the streak entirely. Worst-case waste on a
    /// permanently-dry workload: a third of rounds pay the sync. The
    /// cooldown is deliberately SHORT relative to a typical request (a
    /// 160-token echo ≈ 70 rounds): a 64-round blackout swallowed the whole
    /// echoing stretch after a dry preamble and re-created the very
    /// ext_rounds=0 regression this design avoids (caught by
    /// tests/test_mtp_equivalence.sh).
    pub const MTP_EXT_DRY_ROUNDS: u32 = 16;
    pub const MTP_EXT_DRY_COOLDOWN: u32 = 32;
    /// Floor for the cost-aware dry threshold. Even the most expensive sync
    /// tolerates this many dry considered rounds before a cooldown, so the
    /// worst-case fresh-trial cadence (MTP_EXT_DRY_MIN + MTP_EXT_DRY_COOLDOWN)
    /// still fits inside a typical echo stretch (~70 rounds) — the horizon is
    /// never closed regardless of measured cost.
    pub const MTP_EXT_DRY_MIN: u32 = 3;
    /// Tolerated cumulative chunk-A sync per dry exploration burst, expressed
    /// as a fraction of one round. The cost-aware threshold is
    /// SYNC_BUDGET / (sync_ms/round_ms): a costlier sync backs off sooner, a
    /// near-free one keeps the full MTP_EXT_DRY_ROUNDS budget. Calibrated so
    /// the measured 8K operating point (sync ~2.4 ms of a ~45 ms round,
    /// frac ~0.053) lands near a ~15% exploration duty (threshold ~6, cooldown
    /// 32) — matching oMLX's duty-bounded probing.
    pub const MTP_EXT_SYNC_BUDGET: f32 = 0.30;

    /// Cost-aware dry-explore threshold (consecutive dry considered rounds
    /// tolerated before a cooldown), derived from the LIVE-measured sync
    /// fraction `sync_ms / round_ms`. Bounded to
    /// [MTP_EXT_DRY_MIN, MTP_EXT_DRY_ROUNDS] so a fresh trial always fits
    /// inside an echo stretch. Unmeasured (either EMA still 0) keeps the fixed
    /// budget — no behavior change until the live cost is observed.
    pub fn mtpExtDryThresholdFor(sync_ms: f32, round_ms: f32) u32 {
        if (sync_ms <= 0.0 or round_ms <= 0.0) return MTP_EXT_DRY_ROUNDS;
        const frac = sync_ms / round_ms; // measured exploration cost fraction
        if (frac <= 0.0) return MTP_EXT_DRY_ROUNDS;
        const max_f: f32 = @floatFromInt(MTP_EXT_DRY_ROUNDS);
        // Tolerate ~SYNC_BUDGET round-times of accumulated sync per dry burst:
        // a costlier sync (larger frac) yields fewer tolerated dry rounds.
        const raw = @min(MTP_EXT_SYNC_BUDGET / frac, max_f);
        const rounded: u32 = @intFromFloat(@round(@max(0.0, raw)));
        return @min(MTP_EXT_DRY_ROUNDS, @max(MTP_EXT_DRY_MIN, rounded));
    }

    /// Pure dry-spell policy step, called once per post-warmup EV round
    /// whose plan considers extension. Returns whether the two-chunk shape
    /// is allowed this round; mutates the streak/cooldown state. The caller
    /// bumps `streak` on a considered-but-not-extended round and zeroes it
    /// when extension fires. `threshold` is the cost-aware dry limit
    /// (mtpExtDryThresholdFor) — the live-cost lever that shortens dry
    /// exploration bursts when the measured sync is expensive.
    pub fn mtpExtDryAllows(dry_streak: *u32, cooldown: *u32, threshold: u32) bool {
        if (cooldown.* > 0) {
            cooldown.* -= 1;
            return false;
        }
        if (dry_streak.* >= threshold) {
            cooldown.* = MTP_EXT_DRY_COOLDOWN - 1;
            dry_streak.* = 0;
            return false;
        }
        return true;
    }

    /// Dry-spell gate kill switch — MLX_SERVE_MTP_EXT_DRY=0 restores
    /// unconditional extension consideration for same-boot A/Bs.
    var mtp_ext_dry_cache: ?bool = null;
    pub fn mtpExtDryEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return value.len == 0 or value[0] != '0';
    }

    fn mtpExtDryEnabled() bool {
        if (mtp_ext_dry_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_EXT_DRY")) |p| std.mem.span(p) else null;
        const on = mtpExtDryEnabledFromEnv(raw);
        mtp_ext_dry_cache = on;
        return on;
    }

    /// EMA weight for the live round/sync cost EMAs. Slower than the
    /// acceptance EMA (cost drifts with context/thermal, not content) and
    /// paired with the seed-on-first-sample rule so warmup fills them cleanly.
    pub const MTP_EV_COST_BETA: f32 = 0.10;

    /// Fold a nanosecond wall-time sample into a millisecond EMA. Seeds on the
    /// first sample (prev <= 0). Pure — the live-cost measurement plumbing is
    /// just Stopwatch reads feeding this.
    pub fn mtpEmaMs(prev_ms: f32, sample_ns: u64) f32 {
        const sample_ms = @as(f32, @floatFromInt(sample_ns)) / 1.0e6;
        if (prev_ms <= 0.0) return sample_ms;
        return prev_ms + MTP_EV_COST_BETA * (sample_ms - prev_ms);
    }

    /// Measured ms-per-emitted-token of the two round SHAPES the EV plan can
    /// take: a two-chunk plan (extension considered, sync paid whether or not
    /// it fires) against a single-chunk plan at m_lo. Throughput is tokens
    /// over round cost, and both halves are observable per round — no cost
    /// surface, no units. The hand-fitted marginals can only say whether the
    /// extension's VERIFY pays; they cannot see that on an M4 base the whole
    /// two-chunk round runs 17.4 ms/token against 14.4 single (cap 5+ lost
    /// 17% to cap 4 on echo) while on an M4 Max the same shape is a wash.
    pub const MtpRegime = struct {
        two_ms: f32 = 0,
        two_tok: f32 = 0,
        two_m: u32 = 0,
        one_ms: f32 = 0,
        one_tok: f32 = 0,
        one_m: u32 = 0,
        /// Shape of the previous round: a round whose shape differs from its
        /// predecessor is a TRANSITION and is not observed. The minority shape
        /// was only ever measured on transition rounds and read 5-7% slow
        /// (M4 Max 27B @16k interleaved 13.65 vs 12.9 ms/tok; homogeneous arms
        /// 13.0 vs 13.2) — the verify width change is a one-off cost.
        last_two: ?bool = null,
        /// Trial schedule: the minority shape runs for rounds [trial_start,
        /// trial_end) and the next block starts at next_trial. Explicit, not
        /// `idx % period` — the period moves with the EMAs, and a block's own
        /// observation moved it by one per round so `idx % period` stayed
        /// inside the block: M4 base v4 ran 14 of 72 rounds as trials
        /// against 7 on v3 and lost 2-4%.
        sched_verdict: ?bool = null,
        trial_end: u32 = 0,
        next_trial: u32 = 0,
        /// Diagnostics for `[spec-stats]`: the round the first verdict formed
        /// at, and trial blocks started — splits ext_rounds into pre-verdict
        /// (seeding) and scheduled exploration.
        verdict_round: u32 = 0,
        trials: u32 = 0,
        /// Idempotency: `mtpRegimeForce` mutates the schedule, and
        /// `mtpRoundPlan` has two call sites (pre-draft at the previous
        /// round's tail, or the round's own entry). Asking twice for the
        /// same round answers the same and advances nothing.
        last_idx: ?u32 = null,
        last_force: ?bool = null,
    };

    /// The worse regime still runs one round in this many (at least) so its
    /// EMAs keep tracking the live workload (context grows, thermals move).
    pub const MTP_REGIME_EXPLORE_PERIOD: u32 = 8;
    /// Exploration drag the period is sized to: a shape G worse than the
    /// other, run once in G/DRAG rounds, costs ~DRAG of throughput. M4 base
    /// measured two-chunk 26% worse; at 1-in-8 that was a 2% structural
    /// drag that kept cap 5 from cap-4 parity (52.3 vs 54.6).
    pub const MTP_REGIME_EXPLORE_DRAG: f32 = 0.01;
    pub const MTP_REGIME_EXPLORE_PERIOD_MAX: u32 = 128;
    /// A trial is a BLOCK of consecutive rounds: the first is the transition
    /// (unobserved), the second is the steady-state measurement.
    pub const MTP_REGIME_EXPLORE_BLOCK: u32 = 2;

    fn regimeEma(prev: f32, sample: f32) f32 {
        if (prev <= 0.0) return sample;
        return prev + MTP_EV_COST_BETA * (sample - prev);
    }

    /// Wall time since the previous round ended (the first round reads its
    /// own stopwatch), so inter-round work is charged to the shape.
    fn mtpRegimeWallMs(self: *Generator, round_ms: f32) f32 {
        if (self.mtp_regime_clock) |*c| {
            const ns = c.read();
            c.reset();
            return @as(f32, @floatFromInt(ns)) / @as(f32, std.time.ns_per_ms);
        }
        self.mtp_regime_clock = io_util.Stopwatch.init(self.timer.io);
        return round_ms;
    }

    /// Both shapes are compared at the SAME base depth: a shape observed at
    /// a new m_lo reseeds (the warmup climb's depth-1..3 rounds read 19
    /// ms/tok against 11 for a two-chunk round at m_lo 4 — not a comparison).
    pub fn mtpRegimeObserve(r: *MtpRegime, two_chunk: bool, m_lo: u32, round_ms: f32, tokens: f32) void {
        if (round_ms <= 0.0 or tokens <= 0.0) return;
        // The first observed round has no predecessor to transition from
        // (dropping it cost one extra pre-verdict extension round, the whole
        // of the M1 Pro 27B's -2.6% on v5.2).
        const transition = r.last_two != null and r.last_two.? != two_chunk;
        r.last_two = two_chunk;
        if (transition) return;
        if (two_chunk) {
            if (r.two_m != m_lo) {
                r.two_ms = 0;
                r.two_tok = 0;
                r.two_m = m_lo;
            }
            r.two_ms = regimeEma(r.two_ms, round_ms);
            r.two_tok = regimeEma(r.two_tok, tokens);
        } else {
            if (r.one_m != m_lo) {
                r.one_ms = 0;
                r.one_tok = 0;
                r.one_m = m_lo;
            }
            r.one_ms = regimeEma(r.one_ms, round_ms);
            r.one_tok = regimeEma(r.one_tok, tokens);
        }
    }

    /// The minority shape is measured from interleaved rounds and reads a
    /// few percent slow (M4 Max 27B @16k: two-chunk 13.4 ms/tok interleaved
    /// vs 13.0 homogeneous), so a throttle needs a margin the noise cannot
    /// cross; the M4 base loss it exists for is 21%.
    pub const MTP_REGIME_MARGIN: f32 = 0.05;

    /// Null until BOTH shapes have been measured at the same base depth.
    pub fn mtpRegimeTwoChunkWorse(r: MtpRegime) ?bool {
        return mtpRegimeVerdict(r, null);
    }

    /// Verdict with HYSTERESIS: a standing "worse" only flips to "better"
    /// once two-chunk is at or below single, not merely inside the margin.
    /// The majority shape's first observed round after a trial block is
    /// still elevated (M1 Pro 9B: single 23.8 steady, 24.6-24.9 after a
    /// block), which pulled the ratio inside the margin and flipped the
    /// verdict 5-7 times per boot on v5.2, each flip a run of two-chunk
    /// rounds on a box where that shape loses 10%.
    pub fn mtpRegimeVerdict(r: MtpRegime, prev: ?bool) ?bool {
        if (r.two_tok <= 0.0 or r.one_tok <= 0.0 or r.two_m != r.one_m) return null;
        const ratio = (r.two_ms / r.two_tok) / (r.one_ms / r.one_tok);
        if (prev == true) return ratio > 1.0;
        return ratio > 1.0 + MTP_REGIME_MARGIN;
    }

    /// Rounds between trials of the worse shape, from the measured gap.
    pub fn mtpRegimeExplorePeriod(r: MtpRegime) u32 {
        if (r.two_tok <= 0.0 or r.one_tok <= 0.0) return MTP_REGIME_EXPLORE_PERIOD;
        const two = r.two_ms / r.two_tok;
        const one = r.one_ms / r.one_tok;
        const gap = @abs(two - one) / @min(two, one);
        const block: f32 = @floatFromInt(MTP_REGIME_EXPLORE_BLOCK);
        const p: u32 = @intFromFloat(@ceil(block * gap / MTP_REGIME_EXPLORE_DRAG));
        return @min(MTP_REGIME_EXPLORE_PERIOD_MAX, @max(MTP_REGIME_EXPLORE_PERIOD, p));
    }

    /// Which shape this round runs: null = as the plan wrote it (two-chunk),
    /// false = single-chunk. The plan's own default is two-chunk, so that
    /// side measures itself; an unmeasured SINGLE is tried at once (v2
    /// compared one_m to a two_m only a two-chunk round can set, and pinned
    /// single forever). Once both are measured the worse shape runs as a
    /// trial BLOCK every period so its EMA keeps tracking the workload; a
    /// single-chunk trial is what gets it measured at all when the horizon
    /// opens on every round (pure echo never plans one by itself).
    pub fn mtpRegimeForce(r: *MtpRegime, round_idx: u32) ?bool {
        if (r.last_idx == round_idx) return r.last_force;
        const force = mtpRegimeForceAt(r, round_idx);
        r.last_idx = round_idx;
        r.last_force = force;
        return force;
    }

    fn mtpRegimeForceAt(r: *MtpRegime, round_idx: u32) ?bool {
        if (r.two_tok <= 0.0) return null;
        if (r.one_tok <= 0.0 or r.one_m != r.two_m) return false;
        const worse = mtpRegimeVerdict(r.*, r.sched_verdict) orelse return null;
        if (r.sched_verdict != worse) {
            if (r.sched_verdict == null) r.verdict_round = round_idx;
            r.sched_verdict = worse;
            r.trial_end = 0;
            r.next_trial = round_idx + mtpRegimeExplorePeriod(r.*);
        }
        const minority: ?bool = if (worse) null else false;
        const majority: ?bool = if (worse) false else null;
        if (round_idx < r.trial_end) return minority;
        if (round_idx >= r.next_trial) {
            r.trials += 1;
            r.trial_end = round_idx + MTP_REGIME_EXPLORE_BLOCK;
            r.next_trial = r.trial_end + mtpRegimeExplorePeriod(r.*);
            return minority;
        }
        return majority;
    }

    /// Regime gate kill switch — MLX_SERVE_MTP_REGIME=0 leaves every
    /// two-chunk plan as the EV horizon wrote it (same-boot A/B control arm).
    var mtp_regime_cache: ?bool = null;
    fn mtpRegimeGateEnabled() bool {
        if (mtp_regime_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_REGIME")) |p| std.mem.span(p) else null;
        const on = mtpLiveCostEnabledFromEnv(raw);
        mtp_regime_cache = on;
        return on;
    }

    /// Live-cost throttle kill switch — MLX_SERVE_MTP_LIVECOST=0 reverts the
    /// dry-exploration threshold to the fixed MTP_EXT_DRY_ROUNDS (the
    /// pre-live-cost cadence) for same-boot A/Bs.
    var mtp_livecost_cache: ?bool = null;
    pub fn mtpLiveCostEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return value.len == 0 or value[0] != '0';
    }

    fn mtpLiveCostEnabled() bool {
        if (mtp_livecost_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_LIVECOST")) |p| std.mem.span(p) else null;
        const on = mtpLiveCostEnabledFromEnv(raw);
        mtp_livecost_cache = on;
        return on;
    }

    /// Per-round draft plan. Fixed mode (and EV warmup): today's adaptive
    /// depth, no extension — byte-identical round shape to the legacy path.
    /// Post-warmup EV mode: the pure plan over the acceptance EMAs, with the
    /// base-depth climb damped to one step per round and two-chunk plans
    /// gated by the extension dry-spell policy.
    /// DIAGNOSTIC (MLX_SERVE_MTP_FORCE_DEPTH=n): every round drafts exactly
    /// n, the EV/windowed controllers never demote or disable — the
    /// per-index acceptance meter (`acc_idx=` on the trace line).
    fn mtpForcedDepth() ?u32 {
        if (mtp_force_depth_cache) |v| return v;
        const n = readEnvUsize("MLX_SERVE_MTP_FORCE_DEPTH", 0);
        const v: ?u32 = if (n == 0) null else @intCast(@min(n, mtp_mod.MAX_DEPTH));
        mtp_force_depth_cache = v;
        return v;
    }

    // ── Adaptive serial: a measured round against a measured serial token ──
    //
    // The EV controller picks the best DEPTH; nothing in it ever compares a
    // round with the serial step it replaces. The only stop is the acceptance
    // floor (`MTP_DISABLE_BELOW`), a MODEL of break-even fitted where a verify
    // row costs ~0.1 of a trunk forward. On a long-context sparse-attention
    // trunk a verify row is BYTES: measured on qwen4_exp prose, 62.7k ran
    // 47-58 tok/s against serial's 55 and 374k ran 30 against 47 — with
    // acceptance far above the floor the whole way. The fitted prior cannot
    // see it either: it was calibrated at short context, and the serial cost
    // enters the controller only through it.
    //
    // So the two numbers are compared directly, per KV bucket, with no cost
    // model at all:
    //
    //     planned ms/token = roundMs(m_lo) / E[tokens per round at m_lo]
    //     serial  ms/token = the bucket's measured plain-decode token
    //
    // `roundMs` is measured and workload-independent. The tokens are the
    // REQUEST's own acceptance EMAs — never the table's `tok` column, which
    // is a mixture of every workload that fed the bucket (planning the base
    // from that column's twin measured a loss; this is the same trap one
    // level up). Any missing input is `.undecided`: a switch is never made
    // from a guess.
    //
    // This is NOT the parked arm-probe controller. That one re-measured both
    // arms inside every request and lost 3.8% on code to probe tax and an
    // interrupted width climb. Here the measurement is the persisted table's,
    // deciding costs the request nothing, and the ONE thing a request may pay
    // for is a bucket that has never seen a serial token: a bounded probe,
    // once per (model, bucket) per process.

    /// A round worse than serial by less than this is a tie: the cells are
    /// EMAs of a few samples and the two quantities are measured on
    /// different blocks. Same number as the table's own `SWITCH_MARGIN`.
    pub const MTP_ADAPTIVE_MARGIN: f32 = 0.05;
    /// Consecutive rounds that must vote serial. One round can lose to a GPU
    /// hiccup or a single deep-index miss; three in a row cannot.
    pub const MTP_ADAPTIVE_CONFIRM: u32 = 3;
    /// Serial ticks one probe runs. `MTP_ADAPTIVE_PROBE_WARM` of them are
    /// discarded as the previous round's tail, and what is left must still
    /// reach `round_cost.MIN_SAMPLES` or the probe teaches nothing.
    pub const MTP_ADAPTIVE_PROBE_TOKENS: u32 = 8;
    pub const MTP_ADAPTIVE_PROBE_WARM: u32 = 2;
    /// KV below which the switch does not exist: no vote, no probe, no cost.
    /// The symmetric knob to `--max-mtp-ctx` — that one is a ceiling past
    /// which speculation stays off, this is a FLOOR below which it stays on.
    ///
    /// Measured, not guessed: in the 2026-09-04 A/B, 11 of the 14 switches
    /// the controller made were in the `<2k` bucket, all of them llmprobe's
    /// short requests, and short context is where a wrong switch is most
    /// expensive — llmprobe's predictable cell runs 151 tok/s on speculation
    /// against roughly 95 serial, so a bad switch there costs ~2x, against
    /// ~10% at 62.7k. Long context is the only place the feature was ever
    /// argued for, and the whole ladder it was built from starts past 32k.
    pub const MTP_ADAPTIVE_MIN_KV: u32 = 8192;

    /// May the adaptive switch run at this context at all? Pure, and read
    /// BEFORE the vote and the probe alike: a probe below the floor would
    /// spend 8 serial tokens teaching a bucket that can never decide.
    pub fn mtpAdaptiveKvEligible(kv_len: u32, min_kv: u32) bool {
        return kv_len >= min_kv;
    }

    /// THE bucket the adaptive switch speaks in. Pure half, so the rule is
    /// testable: `read` is `Table.bucketToRead(kv)`, which may hand back a
    /// NEIGHBOUR bucket when the request's own one is unmeasured.
    ///
    /// Every site must resolve it the same way. The decision read
    /// `bucketToRead orelse bucketFor` while the re-entry read plain
    /// `bucketFor`: whenever the fallback fired those disagree, so the switch
    /// stored one bucket and the very next serial tick saw a different one,
    /// called it a crossing and re-entered — then the next round switched
    /// again. An oscillation, and not a cheap one: every lap pays the drain
    /// plus the capture forward of the exit ramp.
    /// The fallback is the TABLE's grid (`Table.bucketOf`), never the free
    /// `bucketFor`: the two disagree past 32k the moment a table is on the
    /// legacy layout, and this is the same class of bug the paragraph above
    /// describes — two sites resolving one bucket differently.
    pub fn mtpAdaptiveBucketOf(read: ?usize, kv_len: u32, layout: round_cost.Layout) usize {
        return read orelse round_cost.bucketForLayout(kv_len, layout);
    }

    fn mtpAdaptiveBucket(self: *const Generator, kv_len: u32) usize {
        const t = &self.xfm.round_cost;
        return mtpAdaptiveBucketOf(t.bucketToRead(kv_len), kv_len, t.layout);
    }

    /// Must the realized price window be dropped? It prices ONE arm in ONE
    /// bucket: the moment either moves, its rounds describe a regime the
    /// request has left. `MtpPriceWindow.reset` used to be reachable only
    /// from the legacy depth controller, so after a switch, a re-entry or a
    /// crossing the window still held up to 15 rounds of the old regime and
    /// the 3-round confirm re-decided mostly from them.
    pub fn mtpAdaptiveRegimeMoved(
        prev_bucket: ?usize,
        prev_arm: MtpAdaptiveArm,
        bucket: ?usize,
        arm: MtpAdaptiveArm,
    ) bool {
        if (prev_arm != arm) return true;
        if (prev_bucket == null and bucket == null) return false;
        if (prev_bucket == null or bucket == null) return true;
        return prev_bucket.? != bucket.?;
    }

    /// Apply the rule above around a `MtpAdaptive` transition. Both call
    /// sites (the vote and the re-entry) go through it, so neither can
    /// forget: a switch in EITHER direction, and a crossing, start pricing
    /// from scratch.
    fn mtpAdaptiveSyncWindow(self: *Generator, prev_bucket: ?usize, prev_arm: MtpAdaptiveArm) void {
        if (mtpAdaptiveRegimeMoved(prev_bucket, prev_arm, self.mtp_adaptive.bucket, self.mtp_adaptive.arm)) {
            self.mtp_price.reset();
        }
    }

    /// Rounds in the realized price window. Full-window-only is the point:
    /// the b1-long2 misfire in the A/B voted at round ~3 on rounds that were
    /// all prefix-cache-restore warmup (the head's committed history is a
    /// stub after a restore, so acceptance is unrepresentative for a while).
    /// An EMA cannot express "full", which is why this is a ring and not the
    /// `mtp_ev_round_ms` EMA beside it.
    pub const MTP_PRICE_WINDOW: u32 = 16;

    /// This request's realized ms per EMITTED token over a trailing window of
    /// speculative rounds. Pure; no clock of its own — `mtpRoundEndObserve`
    /// hands it the same inter-round wall the round-cost table folds.
    ///
    /// Width TRIALS are skipped: a trial deliberately runs a width the plan
    /// rejected, and up to 3 of 16 rounds at a rejected width would price a
    /// round the request never intends to run. Two-chunk (extension) rounds
    /// are KEPT — the vote asks whether the whole speculation is worth
    /// running, and an extension is part of it.
    ///
    /// A new request gets a zeroed window, so a prefix-cache restore starts
    /// empty by construction: there is no restore hook to forget.
    pub const MtpPriceWindow = struct {
        ms: [MTP_PRICE_WINDOW]f32 = @splat(0),
        tok: [MTP_PRICE_WINDOW]f32 = @splat(0),
        idx: u32 = 0,
        filled: u32 = 0,

        pub fn reset(self: *MtpPriceWindow) void {
            self.idx = 0;
            self.filled = 0;
        }

        pub fn observe(self: *MtpPriceWindow, round_ms: f32, emitted: f32, width_trial: bool) void {
            if (width_trial) return;
            if (!std.math.isFinite(round_ms) or !(round_ms > 0)) return;
            if (!std.math.isFinite(emitted) or !(emitted > 0)) return;
            const i = self.idx % MTP_PRICE_WINDOW;
            self.ms[i] = round_ms;
            self.tok[i] = emitted;
            self.idx +%= 1;
            if (self.filled < MTP_PRICE_WINDOW) self.filled += 1;
        }

        /// Realized ms per emitted token, or null until the window is FULL.
        /// Summed, not averaged per round: a round is not a unit of work, a
        /// token is.
        pub fn msPerTok(self: *const MtpPriceWindow) ?f32 {
            if (self.filled < MTP_PRICE_WINDOW) return null;
            var ms: f32 = 0;
            var tok: f32 = 0;
            for (self.ms, self.tok) |m, t| {
                ms += m;
                tok += t;
            }
            if (!(ms > 0) or !(tok > 0)) return null;
            const v = ms / tok;
            return if (std.math.isFinite(v)) v else null;
        }
    };

    pub const MtpAdaptiveVote = enum { undecided, mtp, serial };
    pub const MtpAdaptiveArm = enum { undecided, mtp, serial };
    pub const MtpAdaptiveAction = enum { none, to_serial, to_mtp };
    /// Where a serial block sits in its ramp back to `nextMtp`'s entry
    /// invariant: `.none` = decoding, then drain the `next()` pipeline, then
    /// one capture step.
    pub const MtpSerialExit = enum { none, drain, capture };

    /// One round's vote. Pure, and BOTH prices must be measured.
    ///
    /// v1 priced MTP as `measuredMs(m_lo) / mtpEvExpectedTokens(...)`: a
    /// MEASURED numerator over a MODELED denominator. The model is a product
    /// of per-index marginal acceptance EMAs, which under-predicts committed
    /// tokens for two reasons — acceptance is positively correlated within a
    /// round (a predictable stretch takes every draft, a hard one takes none),
    /// so `E[prod] > prod E[]`; and the chain models a round with NO
    /// extension while the cell's ms includes whatever the extension did.
    /// Measured on the qwen4 A/B, the modeled token count ran 12-31% under
    /// the same cell's own `tok` column, and the switch fired on prompts
    /// where MTP was in fact ~10% faster than serial.
    ///
    /// So the vote now reads two MEASURED ms/token and requires both:
    ///   `table_ms_tok`  — `Table.msPerTok(m_lo, bucket)`: ms and tokens
    ///     folded from the SAME rounds of the same cell that used to supply
    ///     only the numerator. Cross-request, but its `tok` column is a
    ///     workload MIXTURE, so a code-fed cell reads optimistic on prose.
    ///   `window_ms_tok` — this REQUEST's realized trailing window
    ///     (`MtpPriceWindow`): immune to the mixture, but per-request and
    ///     empty until it is full.
    /// Their failure modes are disjoint, and a wrong switch costs ~10% while
    /// a missed one costs nothing today, so `serial` needs BOTH to say serial.
    /// Any input missing or degenerate = `.undecided`; a switch is never made
    /// from a guess.
    pub fn mtpAdaptiveVoteFor(
        table_ms_tok: ?f32,
        window_ms_tok: ?f32,
        serial_ms_tok: ?f32,
        margin: f32,
    ) MtpAdaptiveVote {
        const tbl = table_ms_tok orelse return .undecided;
        const win = window_ms_tok orelse return .undecided;
        const ser = serial_ms_tok orelse return .undecided;
        if (!std.math.isFinite(tbl) or !(tbl > 0)) return .undecided;
        if (!std.math.isFinite(win) or !(win > 0)) return .undecided;
        if (!std.math.isFinite(ser) or !(ser > 0)) return .undecided;
        if (!std.math.isFinite(margin) or margin < 0) return .undecided;
        const bar = ser * (1.0 + margin);
        return if (tbl > bar and win > bar) .serial else .mtp;
    }

    /// Arm a bounded serial probe, returning the bucket it will measure.
    /// ONE bucket parameter (H6): the flag used to key on `bucketFor` while
    /// the "already measured?" check keyed on the decision bucket, so a probe
    /// could be spent on a bucket the decision never reads.
    /// A bucket with no measured serial token cannot be decided at all, and
    /// a workload that only ever speculates there would never teach it one.
    /// The flag is CONSUMED here, at arming, so the probe fires at most once
    /// per (model, bucket) per process — a second request, and every later
    /// round of this one, gets null. `decision_bucket` is the bucket the plan
    /// reads (`bucketToRead` may fall back to a neighbour); the sample lands
    /// in the bucket the request is actually IN, which is what the flag keys
    /// on and what the next decision will read once it is active.
    /// A probe buys the LAST missing input, never the first. The vote needs
    /// all three of `table_ms_tok`, `window_ms_tok` and `serial_ms_tok`;
    /// the window alone takes `MTP_PRICE_WINDOW` non-trial rounds to fill,
    /// so a request short enough to end before that can never decide no
    /// matter what the probe teaches it — it just pays 8 serial tokens.
    /// Measured cold on Flash-Next: the short cell ran 82.7 against 88.3 on
    /// a warm boot with the probes as the difference.
    ///
    /// The table cell is NOT part of this: it is cross-request, so a probe
    /// that teaches the serial row leaves something behind for the next
    /// request even when this one never votes. The window does not — it dies
    /// with the request — so it is the honest "can this request ever use
    /// what the probe buys" test.
    pub fn mtpSerialProbeUseful(window_ms_tok: ?f32) bool {
        return window_ms_tok != null;
    }

    pub fn mtpSerialProbeArm(
        t: *round_cost.Table,
        bucket: usize,
        solo: bool,
        idle: bool,
        may_resume: bool,
        useful: bool,
    ) ?usize {
        if (!solo or !idle or !may_resume or !useful) return null;
        if (bucket >= round_cost.N_BUCKETS) return null;
        // Already taught: nothing to probe for.
        if (t.serialMsPerTok(bucket) != null) return null;
        // Bounded RETRIES, not a one-shot flag. The flag was consumed at
        // arming, so a probe interrupted before it folded MIN_SAMPLES — the
        // request ended, the ticks were dropped as contended, a stop sequence
        // landed — burned the bucket's only chance and left it permanently
        // undecidable. The cell being trusted is what ends the retries, above.
        if (t.serial_probes[bucket] >= round_cost.MAX_SERIAL_PROBES) return null;
        t.serial_probes[bucket] += 1;
        return bucket;
    }

    /// The standing choice and the streak behind it. Pure — no MLX, no
    /// clock: `mtpRoundPlan` feeds `round`, the serial block feeds
    /// `serialTick`.
    pub const MtpAdaptive = struct {
        arm: MtpAdaptiveArm = .undecided,
        confirm: u32 = 0,
        /// The bucket the standing choice was decided in.
        bucket: ?usize = null,
        /// Serial ticks since the arm went serial (or since the last
        /// re-decision). Drives the periodic re-open.
        serial_ticks: u32 = 0,
        /// A re-entry was refused because the head could not prove its
        /// position bookkeeping was in sync. The drift cannot shrink by
        /// itself, so this latches for the request.
        reentry_declined: bool = false,
        /// S21. The switch left MTP on a MODULE-OWNED head, so the arm is
        /// STICKY: no re-entry, no probe and no round for the rest of this
        /// request. It is not a stronger opinion about the measurements —
        /// it is what BUYS the release. While a re-arm was possible the
        /// slot had to keep the model's one head reserved forever (the
        /// documented S21 limit); once no round can follow, the head can be
        /// handed back and the other slots on the model can batch.
        /// A KV-only sidecar head never sets this: its cache is per-request,
        /// nothing is reserved, and re-entry stays exactly as it was.
        sticky_serial: bool = false,
        /// Round index of the last `round()`. `mtpRoundPlan` has TWO call
        /// sites per round (the cross-round pre-draft at the previous
        /// round's tail, and the round entry when there is no pre-draft), so
        /// a streak counting both would trip in half the rounds it is sized
        /// for — the same idempotency `TrialSchedule.force` needs.
        last_round: ?u32 = null,
        switches: u32 = 0,

        pub fn round(self: *MtpAdaptive, round_idx: u32, bucket: usize, vote: MtpAdaptiveVote, need: u32) MtpAdaptiveAction {
            // S21. A sticky request has no rounds left to plan — every caller
            // above is already behind `spec_disabled_runtime` — but the refusal
            // is stated HERE, in the controller, so the invariant does not
            // depend on which of the two `mtpRoundPlan` call sites ran.
            if (self.sticky_serial) return .none;
            if (self.last_round) |r| {
                if (r == round_idx) return .none;
            }
            self.last_round = round_idx;
            // A crossing invalidates the streak: the votes behind it priced
            // a different context.
            if (self.bucket == null or self.bucket.? != bucket) {
                self.bucket = bucket;
                self.confirm = 0;
            }
            switch (vote) {
                .undecided => {
                    self.confirm = 0;
                    return .none;
                },
                .mtp => {
                    self.confirm = 0;
                    self.arm = .mtp;
                    return .none;
                },
                .serial => {
                    self.confirm += 1;
                    if (self.confirm < @max(@as(u32, 1), need)) return .none;
                    self.confirm = 0;
                    self.arm = .serial;
                    self.serial_ticks = 0;
                    self.switches += 1;
                    return .to_serial;
                },
            }
        }

        /// Serial tokens after which the decision is re-opened INSIDE the
        /// same bucket. Without it the only re-entry trigger is a bucket
        /// crossing, and the top bucket is unbounded: a request that switched
        /// at 260k stayed serial to 1M no matter how the measurements moved.
        /// It also covers the ordinary long generation that never leaves the
        /// bucket it switched in.
        ///
        /// The re-decision is not free — re-entry pays the exit ramp, and the
        /// price window needs `MTP_PRICE_WINDOW` rounds before it may vote
        /// again — so at ~1 token per round this costs roughly 3-7% of the
        /// interval when MTP is genuinely the worse arm. 512 keeps that small
        /// while still re-testing several times inside one long reply.
        /// Default OFF. A re-decision means a RE-ENTRY, and re-entry after a
        /// serial block resumes a head whose committed history did not grow
        /// across the block — it drafts from a content gap. Until a
        /// head-follow exists (running the head over the serial tokens so its
        /// history keeps up), paying an exit ramp to resume a degraded head on
        /// a timer is not obviously a win. `MLX_SERVE_MTP_ADAPTIVE_REENTRY_TOKENS`
        /// turns it on; 0 (the default) leaves a crossing as the only trigger.
        pub const REDECIDE_SERIAL_TOKENS_DEFAULT: u32 = 0;
        /// The value the lever selects when it is enabled without a number.
        pub const REDECIDE_SERIAL_TOKENS_ON: u32 = 512;

        /// One serial tick while the adaptive arm holds. Returns `.to_mtp`
        /// when the decision should be re-opened: on a bucket CROSSING (the
        /// numbers it compared priced another context), or every
        /// `REDECIDE_SERIAL_TOKENS` ticks inside the same bucket. Either way
        /// the arm goes back to `.undecided` with a clean streak, so the next
        /// vote decides on fresh measurements rather than inheriting the old
        /// verdict.
        /// Latch a refused re-entry: back to the serial arm, and never ask
        /// again for this request.
        pub fn declineReentry(self: *MtpAdaptive) void {
            self.arm = .serial;
            self.serial_ticks = 0;
            self.reentry_declined = true;
        }

        /// S21. Make the serial arm permanent for this request. Armed at the
        /// `to_serial` that moved a MODULE-OWNED head off speculation, so the
        /// head can be released at the next serial block boundary.
        pub fn stickSerial(self: *MtpAdaptive) void {
            self.arm = .serial;
            self.serial_ticks = 0;
            self.sticky_serial = true;
        }

        pub fn serialTick(self: *MtpAdaptive, bucket: usize, redecide_tokens: u32) MtpAdaptiveAction {
            // S21 outranks both re-entry triggers, the bucket CROSSING
            // included: a released head belongs to whoever took it next.
            if (self.sticky_serial) return .none;
            if (self.reentry_declined) return .none;
            if (self.arm != .serial) return .none;
            if (self.bucket) |b| {
                if (b == bucket) {
                    if (redecide_tokens == 0) return .none; // crossing-only
                    self.serial_ticks +|= 1;
                    if (self.serial_ticks < redecide_tokens) return .none;
                    return self.reopen(bucket);
                }
            }
            return self.reopen(bucket);
        }

        fn reopen(self: *MtpAdaptive, bucket: usize) MtpAdaptiveAction {
            self.bucket = bucket;
            self.confirm = 0;
            self.serial_ticks = 0;
            self.arm = .undecided;
            self.switches += 1;
            return .to_mtp;
        }
    };

    /// Whole-mechanism kill switch (`MLX_SERVE_MTP_ADAPTIVE_SERIAL=0`): no
    /// vote, no probe, and — since `serialCellWanted` reads this same lever —
    /// no serial fold and no round-cost write either. Genuinely no cost.
    ///
    /// Independent of `MLX_SERVE_MTP_ADAPTIVE`, which is the DEPTH
    /// controller's lever: that one used to disable this switch as a side
    /// effect, because the decision sat below the depth planner's early
    /// return.
    ///
    /// This comment used to claim the serial cell "keeps being MEASURED so
    /// `[spec-stats]` stays comparable across an A/B". It does not, and the
    /// 2026-09-04 A/B proved it: `serial_cell` read 0.00 for both off-arm
    /// boots. `observeSerialTick` still runs, but with the switch off nothing
    /// ever votes, so no probe arms and no request decodes serially in the
    /// bucket it is speculating in — there is nothing for the cell to fold.
    /// That is the right trade: an off switch that costs literally nothing
    /// beats one that keeps a meter warm for the convenience of an A/B.
    var mtp_adaptive_serial_cache: ?bool = null;
    /// Default ON; only an exact "0" turns it off. First-BYTE matching read
    /// "01" and "0.5" as off, and the sibling `mtpHeadPersistFromEnv` two
    /// screens up already uses exact equality — two kill switches in one file
    /// should not disagree about what "0" means.
    pub fn mtpAdaptiveSerialEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return !std.mem.eql(u8, value, "0");
    }

    fn mtpAdaptiveSerialEnabled() bool {
        if (mtp_adaptive_serial_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_ADAPTIVE_SERIAL")) |p| std.mem.span(p) else null;
        const on = mtpAdaptiveSerialEnabledFromEnv(raw);
        mtp_adaptive_serial_cache = on;
        return on;
    }

    var mtp_adaptive_margin_cache: ?f32 = null;
    fn mtpAdaptiveMargin() f32 {
        if (mtp_adaptive_margin_cache) |v| return v;
        const v = readEnvFloat("MLX_SERVE_MTP_ADAPTIVE_MARGIN", MTP_ADAPTIVE_MARGIN);
        mtp_adaptive_margin_cache = v;
        return v;
    }

    var mtp_reentry_tokens_cache: ?u32 = null;
    /// `MLX_SERVE_MTP_ADAPTIVE_REENTRY_TOKENS`: 0 / unset = crossing-only
    /// re-entry (the default); a bare `1` selects the tuned 512; any other
    /// number is taken literally.
    fn mtpAdaptiveReentryTokens() u32 {
        if (mtp_reentry_tokens_cache) |v| return v;
        const n = readEnvUsize("MLX_SERVE_MTP_ADAPTIVE_REENTRY_TOKENS", MtpAdaptive.REDECIDE_SERIAL_TOKENS_DEFAULT);
        const v: u32 = if (n == 1) MtpAdaptive.REDECIDE_SERIAL_TOKENS_ON else @intCast(@min(n, @as(usize, std.math.maxInt(u32))));
        mtp_reentry_tokens_cache = v;
        return v;
    }

    var mtp_adaptive_min_kv_cache: ?u32 = null;
    fn mtpAdaptiveMinKv() u32 {
        if (mtp_adaptive_min_kv_cache) |v| return v;
        const n = readEnvUsize("MLX_SERVE_MTP_ADAPTIVE_MIN_KV", MTP_ADAPTIVE_MIN_KV);
        const v: u32 = @intCast(@min(n, @as(usize, std.math.maxInt(u32))));
        mtp_adaptive_min_kv_cache = v;
        return v;
    }

    var mtp_adaptive_confirm_cache: ?u32 = null;
    fn mtpAdaptiveConfirm() u32 {
        if (mtp_adaptive_confirm_cache) |v| return v;
        const n = readEnvUsize("MLX_SERVE_MTP_ADAPTIVE_CONFIRM", MTP_ADAPTIVE_CONFIRM);
        const v: u32 = @intCast(@max(@as(usize, 1), @min(n, 64)));
        mtp_adaptive_confirm_cache = v;
        return v;
    }

    /// Is the head's position bookkeeping in sync with what its next forward
    /// will demand? `Transformer.qwen4MtpForward` refuses with
    /// `error.MtpPositionGap` unless
    ///
    ///     pos_offset == pos_base + seq_offset
    ///
    /// with one escape: at `seq_offset == 0` it re-seeds `pos_base` from the
    /// incoming offset, so a fresh head always agrees. Pure, so the rule is
    /// testable on synthetic offsets. Returns the DRIFT (0 = in sync) rather
    /// than a bool, so the decline can say by how much.
    pub fn mtpHeadPositionDrift(pos_base: i64, seq_offset: usize, next_pos_offset: i64) i64 {
        if (seq_offset == 0) return 0;
        return next_pos_offset - (pos_base + @as(i64, @intCast(seq_offset)));
    }

    /// The drift the NEXT round would hit, computed the way the round itself
    /// computes its offset: `mtpRoundOff0` over the head's own cache, plus
    /// one (the head's query sits at r+1). Null = no head state to check.
    ///
    /// A serial block advances the TRUNK by its length while the head's
    /// `seq_offset` deliberately stays put — no trunk hidden is captured for
    /// those tokens — which is why re-entry has to prove this rather than
    /// assume it. The sidecar arm has no such guard and is always in sync.
    fn mtpHeadDriftNow(self: *const Generator) ?i64 {
        if (self.mtp_cache == null) return null;
        const mc = &self.mtp_cache.?;
        return switch (mc.*) {
            .qwen => 0,
            .qwen4 => |t| blk: {
                const m = &(t.qwen4_mtp orelse break :blk 0);
                const off0 = mtpRoundOff0(self.mtp_hist_stash, mc.step());
                break :blk mtpHeadPositionDrift(@intCast(m.pos_base), m.seq_offset, @intCast(off0 + 1));
            },
        };
    }

    /// May this request resume MTP after a serial block? The head's
    /// committed history does NOT grow across one (those tokens' trunk
    /// hiddens were never captured), and cache-relative RoPE offsets take a
    /// content gap in stride — but an M-RoPE turn ropes the head at an
    /// ABSOLUTE `pos_base + seq_offset`, where a gap is a wrong answer
    /// rather than a slow one. Vision turns therefore never come back, and
    /// never probe.
    fn mtpAdaptiveHeadMayResume(self: *const Generator) bool {
        // S21 first. This ONE predicate is what the three round-start doors
        // on the serial side ask — the re-entry gate in `nextMtp`, the
        // `apply_stash` decision in `mtpSerialMayResume`, and the bounded
        // serial PROBE's arming (a probe is a serial block that RETURNS to
        // speculation, so it is a round start like any other). A sticky
        // request has handed the module head back; none of the three may
        // reach for it again.
        if (self.mtp_adaptive.sticky_serial) return false;
        return self.ctx.mrope_pos == null;
    }

    /// S21. Does a `to_serial` on THIS head make the arm sticky? Only a
    /// MODULE-OWNED head: the sidecar arm's cache is this request's own, so
    /// nothing is reserved and nothing is released — its re-entry, its probe
    /// and its exclusivity are all byte-for-byte what they were.
    pub fn stickyOnSerialSwitch(head: ?MtpHeadRef) bool {
        const h = head orelse return false;
        return h.moduleOwned();
    }

    /// S21. Does THIS tick newly release the module head? True exactly once
    /// per request, which is what makes the log line a one-shot and the
    /// exclusivity drop an edge rather than a level.
    pub fn stickyReleaseNow(sticky: bool, already_released: bool) bool {
        return sticky and !already_released;
    }

    /// S21. Has this request released the module-owned MTP head?
    /// `scheduler.slotExclusiveDecode` and the prefix-cache commit both ask.
    pub fn mtpModuleHeadReleased(self: *const Generator) bool {
        return self.mtp_head_released;
    }

    /// S21. Armed but not LANDED: the latch is set and the release has not
    /// run yet. The release lives in `nextMtp`'s serial branch — after the
    /// head is detached, which is what makes it safe — and the BATCHED tick
    /// never calls `nextMtp`. A sticky slot becomes batchable the instant it
    /// sets `spec_disabled_runtime`, so one plain neighbour on the same model
    /// arriving in the very next tick would carry it into a group and it
    /// would hold the model's one MTP head to the end of the request: exactly
    /// the limit this is fixing, now silent instead of documented.
    /// `scheduler.batchable` therefore keeps the slot out of a group for the
    /// ONE tick the release needs. Bounded, and only for a request that
    /// actually took the switch.
    pub fn mtpReleasePending(self: *const Generator) bool {
        return self.mtp_adaptive.sticky_serial and !self.mtp_head_released;
    }

    /// S21. Release the module-owned head, at a serial BLOCK BOUNDARY.
    ///
    /// Called from `nextMtp`'s serial branch and nowhere else, AFTER
    /// `mtpDetachHead` — never from `mtpAdaptiveSerialStep`, where the switch
    /// is decided. The vote is read while a round is being PLANNED and that
    /// round still runs (the log line says so); releasing there would hand a
    /// second slot a head this one is mid-round on. By the time this runs the
    /// pre-draft is dropped, the deferred history stash is dropped too
    /// (`mtpSerialMayResume` is false once sticky, so no head forward is
    /// paid), and this request will not touch the head again.
    fn mtpMaybeReleaseModuleHead(self: *Generator) void {
        if (!stickyReleaseNow(self.mtp_adaptive.sticky_serial, self.mtp_head_released)) return;
        self.mtp_head_released = true;
        log.info(
            "  [mtp] sticky-serial: module head released for the rest of the request (kv {d}) — other slots may claim it\n",
            .{self.mtpKvLen()},
        );
    }

    /// The adaptive serial step: the EV plan answers "which width", this
    /// answers "is a round worth running at all". Returns true when the
    /// request has just left speculation.
    ///
    /// A METHOD, and called from every plan exit, because it used to sit at
    /// the bottom of `mtpRoundPlan` — below the early return taken when
    /// `MLX_SERVE_MTP_ADAPTIVE=0` or during EV warmup. That made the DEPTH
    /// controller's kill switch silently disable the serial switch too, and
    /// left the whole warmup window unable to decide or probe. The two
    /// controllers are independent and are now gated independently;
    /// `MLX_SERVE_MTP_FORCE_DEPTH` still bypasses both, because that mode
    /// never plans.
    /// S22. Every A/B behind the adaptive serial switch — the round-cost
    /// numbers, the margin, the confirm count, the probe budget — was measured
    /// on qwen4_exp, whose verify row is BYTES and whose depth-2 round costs
    /// 2.05 serial forwards. A sidecar pack (qwen3.5/3.6/3.8) has a different
    /// verify surface entirely, so `model_has_mtp` let every such model past
    /// 8192 KV spend up to MAX_SERIAL_PROBES x MTP_ADAPTIVE_PROBE_TOKENS
    /// serial tokens per bucket probing, and switch speculation off on a
    /// calibration that was never taken on it. The head KIND is the arch:
    /// `.qwen4` is the in-checkpoint head, `.qwen` is every sidecar.
    /// Widening this is a MEASUREMENT, not a flag flip.
    ///
    /// NOT `MtpHeadRef.moduleOwned()`, which the S21 sticky arm keys on: the
    /// two agree today because there is one in-checkpoint head, but this asks
    /// "was the switch calibrated here" and that one asks "is the head shared".
    /// A module-owned head on an uncalibrated family must answer no here.
    fn mtpAdaptiveArchEligible(self: *const Generator) bool {
        const head = self.mtp orelse return false;
        return head == .qwen4;
    }

    fn mtpAdaptiveSerialStep(self: *Generator, m_lo: u32, kv_len: u32) bool {
        // Adaptive serial: the EV plan answers "which width"; this answers
        // "is a round worth running at all". Read AFTER the plan (m_lo is
        // the width it prices) and BEFORE the width trial — a trial measures
        // a width for a request that is about to leave speculation.
        if (mtpAdaptiveSerialEnabled() and mtpCostTableEnabled() and
            self.mtpAdaptiveArchEligible() and
            mtpAdaptiveKvEligible(kv_len, mtpAdaptiveMinKv()))
        {
            const t = &self.xfm.round_cost;
            const b = self.mtpAdaptiveBucket(kv_len);
            // Both prices are MEASURED, and both come from rounds that
            // actually ran. `msPerTok` takes ms AND tokens from one cell —
            // the modeled `mtpEvExpectedTokens` denominator that used to sit
            // under this numerator stays behind in `mtpEvPlanSrc`, where it
            // only has to RANK widths and a common bias cancels.
            const table_ms_tok = t.msPerTok(m_lo, b);
            const window_ms_tok = self.mtp_price.msPerTok();
            const serial_ms = t.serialMsPerTok(b);
            const vote = mtpAdaptiveVoteFor(table_ms_tok, window_ms_tok, serial_ms, mtpAdaptiveMargin());
            const prev_bucket = self.mtp_adaptive.bucket;
            const prev_arm = self.mtp_adaptive.arm;
            const action = self.mtp_adaptive.round(self.mtp_ev_rounds, b, vote, mtpAdaptiveConfirm());
            self.mtpAdaptiveSyncWindow(prev_bucket, prev_arm);
            if (action == .to_serial) {
                // "from the next round", not "now": this vote is read while
                // the round is being PLANNED, and `nextMtp` tested
                // `spec_disabled_runtime` before it got here, so the round now
                // under construction still runs speculatively and the switch
                // takes effect on the following tick. Aborting a half-built
                // round instead would have to unwind the pre-draft and the
                // head state mid-flight to land back on `nextMtp`'s entry
                // invariant — the exit ramp exists precisely because that is
                // delicate — and it would buy exactly one round. So the round
                // stands and the LOG is honest about when the switch bites.
                log.info(
                    "  [mtp] adaptive: bucket {s} mtp table {d:.2} / window {d:.2} ms/tok (w{d}) vs serial {d:.2} ms/tok -> serial (from the next round)\n",
                    .{ round_cost.bucketName(self.xfm.round_cost.layout, b), table_ms_tok.?, window_ms_tok.?, m_lo, serial_ms.? },
                );
                self.spec_disabled_runtime = true;
                self.spec_disable_reason = .adaptive;
                // S21. On a MODULE-OWNED head this switch is one-way. The
                // latch is what lets `nextMtp`'s serial branch hand the head
                // back; it is set HERE (with the decision) and consumed there
                // (at the block boundary), never the other way round.
                if (stickyOnSerialSwitch(self.mtp)) self.mtp_adaptive.stickSerial();
                return true;
            }
            // Nothing to decide with: teach the bucket a serial token, once.
            const idle = self.mtp_serial_left == 0 and self.mtp_serial_exit == .none;
            if (mtpSerialProbeArm(t, b, self.spec_cost_solo, idle, self.mtpAdaptiveHeadMayResume(), mtpSerialProbeUseful(window_ms_tok))) |own| {
                self.mtp_serial_left = MTP_ADAPTIVE_PROBE_TOKENS;
                log.info(
                    "  [mtp] adaptive: bucket {s} has no serial cell -> probing {d} serial tokens\n",
                    .{ round_cost.bucketName(self.xfm.round_cost.layout, own), MTP_ADAPTIVE_PROBE_TOKENS },
                );
                return true;
            }
        }
        return false;
    }

    fn mtpRoundPlan(self: *Generator) MtpRoundPlan {
        if (mtpForcedDepth()) |d| {
            self.mtp_ev_m_lo_prev = d;
            return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
        }
        const cap_row: u32 = @min(@max(@as(u32, 1), self.mtp_depth), mtp_mod.MAX_DEPTH);
        const cap_free: u32 = @min(@max(cap_row, self.mtp_depth_free), mtp_mod.MAX_DEPTH);
        var cap: u32 = cap_row;
        const kv_len = self.mtpKvLen();
        if (!mtpAdaptiveEnabled() or self.mtp_ev_rounds < MTP_EV_WARMUP_ROUNDS) {
            const d = @min(@max(@as(u32, 1), self.mtp_depth_current), cap);
            self.mtp_ev_m_lo_prev = d;
            // Independent of the DEPTH controller (L22): the serial switch
            // decides whether to speculate at all, which is a question the EV
            // planner's kill switch and its warmup have no business answering.
            _ = self.mtpAdaptiveSerialStep(d, kv_len);
            return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
        }
        const src = MtpCostSource.init(self.mtp_ev_costs, kv_len, if (mtpCostTableEnabled()) &self.xfm.round_cost else null);
        if (src.fromTable() and !self.xfm.round_cost.first_use_logged) {
            self.xfm.round_cost.first_use_logged = true;
            var buf: [256]u8 = undefined;
            log.info("[mtp] cost table: bucket {s} measured {s} (ms/tok) replaces the fitted surface (scale {d:.4} at w{d})\n", .{
                round_cost.bucketName(self.xfm.round_cost.layout, src.bucket),
                self.xfm.round_cost.formatBucket(src.bucket, &buf),
                src.scale,
                self.xfm.round_cost.narrowestMeasured(src.bucket) orelse 0,
            });
        }
        // The per-silicon row is the COLD-START cap: the plan may exceed it
        // only up to the widest TRUSTED width (never onto the prior's guess
        // above the row — that reopens the regime gate on ties the row had
        // closed), and the width trial may reach one past that to measure.
        if (src.fromTable()) cap = @min(cap_free, @max(cap_row, self.xfm.round_cost.widestMeasured(src.bucket) orelse cap_row));
        var plan = mtpEvPlanSrc(self.mtp_ev_accept[0..cap], cap, src, self.mtp_ev_m_lo_prev + 1);
        if (plan.m_lo == self.mtp_ev_m_lo_prev) self.mtp_m_lo_streak +|= 1 else self.mtp_m_lo_streak = 0;
        self.mtp_ev_m_lo_prev = plan.m_lo;
        // Live-cost lever: shorten dry exploration bursts when the MEASURED
        // chunk-A sync is an expensive fraction of the round (mtpExtDryThresholdFor).
        const dry_threshold = if (mtpLiveCostEnabled())
            mtpExtDryThresholdFor(self.mtp_ev_sync_ms, self.mtp_ev_round_ms)
        else
            MTP_EXT_DRY_ROUNDS;
        if (plan.m_hi > plan.m_lo and mtpExtDryEnabled() and
            !mtpExtDryAllows(&self.mtp_ext_dry_streak, &self.mtp_ext_cooldown, dry_threshold))
        {
            plan.m_hi = plan.m_lo;
            plan.tau_ln = 0.0;
        }
        if (plan.m_hi > plan.m_lo and mtpRegimeGateEnabled()) {
            const force = mtpRegimeForce(&self.mtp_regime, self.mtp_ev_rounds);
            const worse = self.mtp_regime.sched_verdict;
            if (worse != null and worse != self.mtp_regime_verdict) {
                self.mtp_regime_verdict = worse;
                const r = self.mtp_regime;
                log.info("[mtp] regime gate: two-chunk {d:.2} ms/tok vs single {d:.2} ms/tok -> two-chunk {s} (period {d})\n", .{ r.two_ms / r.two_tok, r.one_ms / r.one_tok, if (worse.?) "throttled" else "every round", mtpRegimeExplorePeriod(r) });
            }
            if (force == false) {
                plan.m_hi = plan.m_lo;
                plan.tau_ln = 0.0;
            }
        }
        if (self.mtpAdaptiveSerialStep(plan.m_lo, kv_len)) return plan;
        // Width trial: a single-chunk round at the width the table needs
        // next (`mtpWidthTrialTarget`), one 2-round block per period. Never
        // inside a regime trial block (that block is the regime's own
        // measurement), and only while solo.
        if (mtpCostTableEnabled() and self.spec_cost_solo and self.mtp_ev_rounds >= self.mtp_regime.trial_end) {
            const base_settled = self.mtp_m_lo_streak >= 2;
            if (mtpWidthTrialTarget(&self.xfm.round_cost, kv_len, plan, cap_free, base_settled)) |target| {
                const period = mtpWidthTrialPeriod(&self.xfm.round_cost, kv_len, plan.m_lo);
                if (mtpWidthTrialForce(&self.mtp_width_trial, self.mtp_ev_rounds, period)) {
                    plan = mtpWidthTrialPlan(target);
                }
            }
        }
        return plan;
    }

    /// The single-chunk plan a width trial runs. Built from a scalar on
    /// purpose: `plan = .{ .m_lo = plan.m_lo + 1, .m_hi = plan.m_lo + 1 }`
    /// reads the already-written m_lo for m_hi (result-location aliasing)
    /// and planned a two-chunk round — the simulated loop caught it.
    pub fn mtpWidthTrialPlan(width: u32) MtpRoundPlan {
        return .{ .m_lo = width, .m_hi = width, .tau_ln = 0.0, .width_trial = true };
    }

    /// Which width a trial measures: the plan's own base when the bucket
    /// the plan reads has not measured it, else an unmeasured m_lo+1 under
    /// any shape, else a periodic m_lo+1 on single-chunk plans. Null =
    /// nothing to try.
    pub fn mtpWidthTrialTarget(t: *const round_cost.Table, kv_len: u32, plan: MtpRoundPlan, cap: u32, base_settled: bool) ?u32 {
        const b = t.bucketToRead(kv_len) orelse t.bucketOf(kv_len);
        // A single-chunk plan feeds its own base every round; only a
        // two-chunk plan (extensions and syncs never feed) owes a trial of
        // it — and only once the base has stopped climbing, or the first EV
        // rounds trial widths the plan is merely passing through (M1 Pro
        // 27B warm boot: one block at w3 in rep 1).
        if (t.measuredMs(plan.m_lo, b) == null) return if (plan.m_hi > plan.m_lo and base_settled) plan.m_lo else null;
        // m_lo-1 is deliberately NOT trialled: every trial block is a 3-4%
        // hit on the request that carries it (peer cells, 2026-08-22), the
        // prior below the anchor already prices narrower widths, and the
        // one cliff this was for (M1 Pro 27B, 4 -> 5) is found from 4 by
        // the m_lo+1 trial below.
        if (plan.m_lo >= cap) return null;
        // An unmeasured m_lo+1 is measured under ANY shape: a two-chunk
        // plan extends there every round on echo, blind to its cost (the
        // M1 Pro 27B's 4 -> 5 at +150 ms); the regime gate would catch it,
        // the table should know it — unless its first sample already
        // settled it as clearly worse. A settled one is re-tried only where
        // a single-chunk plan cannot otherwise reach it, at the long period.
        const up = plan.m_lo + 1;
        if (t.measuredMs(up, b) == null and !t.clearlyWorse(up, plan.m_lo, b)) return up;
        if (plan.m_hi == plan.m_lo) return up;
        return null;
    }

    /// Width-trial schedule: same shape as the regime's (explicit
    /// trial_end/next_trial, a 2-round block — transition then measurement),
    /// idempotent per round because `mtpRoundPlan` has two call sites.
    pub const MtpWidthTrial = round_cost.TrialSchedule;

    pub fn mtpWidthTrialForce(t: *MtpWidthTrial, round_idx: u32, period: u32) bool {
        return t.force(round_idx, period);
    }

    /// Rounds between width trials: the regime's drag rule on the measured
    /// ms/tok gap between m_lo+1 and m_lo (a width G worse, tried once in
    /// G/DRAG rounds, costs ~DRAG of throughput); the default period while
    /// either is unmeasured, so an unknown width is learned soon.
    pub fn mtpWidthTrialPeriod(t: *const round_cost.Table, kv_len: u32, m_lo: u32) u32 {
        const b = t.bucketToRead(kv_len) orelse return round_cost.EXPLORE_PERIOD_COLD;
        // An untrusted target keeps the cold period until it is either
        // trusted or settled as clearly worse (then its raw sample sizes
        // the gap): a raw 10%-better w5 read as a 60-round period and no
        // 22-round request ever trialled it again (M4 Max 27B, w5 stuck
        // at one sample).
        if (t.msPerTok(m_lo + 1, b) == null and !t.clearlyWorse(m_lo + 1, m_lo, b)) return round_cost.EXPLORE_PERIOD_COLD;
        return round_cost.trialPeriod(t.msPerTok(m_lo, b), t.rawMsPerTok(m_lo + 1, b));
    }

    /// Track the only evidence that can justify sticky-disable: whether the
    /// first draft landed while the EV base depth was exactly one. Wider base
    /// rounds reset the probation window, and later extension misses do not
    /// count against the depth-one floor. Returns a rate only after a full
    /// fresh window has been observed.
    fn mtpFloorDisableObserve(
        drafted_window: *[MTP_DEPTH_WINDOW]u8,
        accepted_window: *[MTP_DEPTH_WINDOW]u8,
        window_idx: *u32,
        m_lo: u32,
        drafted: u32,
        accepted: u32,
    ) ?f32 {
        std.debug.assert(drafted >= 1);
        if (m_lo != 1) {
            window_idx.* = 0;
            return null;
        }

        const idx = window_idx.* % MTP_DEPTH_WINDOW;
        drafted_window[idx] = 1;
        accepted_window[idx] = @intFromBool(accepted > 0);
        window_idx.* += 1;
        const n = @min(window_idx.*, MTP_DEPTH_WINDOW);
        if (n < MTP_DEPTH_WINDOW) return null;

        var accepted_sum: u32 = 0;
        var i: u32 = 0;
        while (i < n) : (i += 1) accepted_sum += accepted_window[i];
        return @as(f32, @floatFromInt(accepted_sum)) / @as(f32, @floatFromInt(n));
    }

    /// EV-mode per-round update: EMAs always; during warmup the legacy
    /// windowed controller keeps running (today's behavior while EMAs fill);
    /// post-warmup only the sticky disable floor is checked — EV owns depth.
    fn updateMtpEvRound(self: *Generator, drafted: u32, accepted: u32) void {
        mtpEvObserve(&self.mtp_ev_accept, drafted, accepted, MTP_EV_EMA_BETA);
        self.mtp_ev_rounds += 1;
        if (mtpForcedDepth() != null) return;
        if (self.mtp_ev_rounds <= MTP_EV_WARMUP_ROUNDS) {
            self.updateMtpDepth(drafted, accepted);
            // Warmup may evaluate several depths. None of that mixed evidence
            // belongs in the post-warmup depth-one sticky-disable window.
            if (self.mtp_ev_rounds == MTP_EV_WARMUP_ROUNDS) self.mtp_window_idx = 0;
            return;
        }
        // EV owns promotion/demotion. Sticky-disable is judged only from a
        // full, homogeneous window of first-draft outcomes at base depth one.
        const rate = mtpFloorDisableObserve(
            &self.mtp_window_drafted,
            &self.mtp_window_accepted,
            &self.mtp_window_idx,
            self.mtp_ev_m_lo_prev,
            drafted,
            accepted,
        ) orelse return;
        if (rate < MTP_DISABLE_BELOW) {
            log.info(
                "  mtp=disabled (EV: depth-1 first-draft rate {d:.2} < {d:.2})\n",
                .{ rate, MTP_DISABLE_BELOW },
            );
            self.spec_disabled_runtime = true;
            self.spec_disable_reason = .acceptance;
        }
    }

    /// Close one traced round; emits + resets the summary at the cadence.
    fn mtpTraceRoundEnd(self: *Generator, m: u32, accepted: u32, m_lo: u32) void {
        if (!mtpTraceEnabled()) return;
        if (!self.mtp_trace.endRound(m, accepted, m > m_lo)) return;
        const t = &self.mtp_trace;
        var acc_buf: [64]u8 = undefined;
        log.info(
            "  [mtp-trace] rounds={d} avg_ms draft={d:.2} sync={d:.2} ext={d:.2} verify={d:.2} corr={d:.2} eval={d:.2} hist={d:.2} commit={d:.2} predraft={d:.2} gap={d:.2} total={d:.2} | m_avg={d:.2} acc_avg={d:.2} ext_rate={d:.2} acc_idx={s}\n",
            .{
                t.rounds,
                t.avgMs(.draft),
                t.avgMs(.sync),
                t.avgMs(.ext),
                t.avgMs(.verify),
                t.avgMs(.corr),
                t.avgMs(.eval),
                t.avgMs(.hist),
                t.avgMs(.commit),
                t.avgMs(.predraft),
                t.avgMs(.gap),
                t.totalAvgMs(),
                @as(f64, @floatFromInt(t.drafted)) / @as(f64, @floatFromInt(t.rounds)),
                @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
                @as(f64, @floatFromInt(t.extended)) / @as(f64, @floatFromInt(t.rounds)),
                t.accIdxStr(&acc_buf),
            },
        );
        t.reset();
    }

    /// Returns the next token ID, or null when generation is finished.
    ///
    /// Pipeline architecture (matches mlx-lm's generator pattern):
    ///
    ///   The KEY to effective pipelining is the ORDER of operations:
    ///   1. Build next step's lazy graph (depends on pending lazy token)
    ///   2. async_eval the next graph — GPU computes pending token as a DEPENDENCY,
    ///      then continues with the forward pass
    ///   3. eval(pending_token) — returns INSTANTLY since GPU already computed it
    ///   4. Return the token while GPU continues computing the next forward pass
    ///
    ///   This mirrors mlx-lm's: _step(y) → async_eval(next_y) → yield y.item()
    ///   where y.item() is instant because async_eval forced y's computation.
    pub fn next(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (self.done) return null;
        if (self.sampling.constraint != null) return self.nextConstrained(allocator);

        // Transition shim: speculative-decode paths may exit with
        // `next_token_id` set but `pending_logits` unset (drafter's exit
        // invariant is "t1 NOT in cache" — its hand-off to `next()` would
        // otherwise crash on the slow path which assumes pending_logits is
        // always lazily seeded). When we observe that state, synchronously
        // forward `[next_token_id]` to seed `pending_logits` so the fast
        // path picks up cleanly. PLD's exit state already matches `next()`'s
        // invariant, so this only fires for drafter→next runtime-gate
        // fallbacks (and any future spec methods that share drafter's shape).
        if (!self.has_pending_logits and !self.has_pending_token and
            self.step < self.max_tokens and self.logprobs_n == 0)
        {
            const tok_i32: i32 = @intCast(self.next_token_id);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            self.pending_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            self.has_pending_logits = true;
        }

        // ── Phase 1: Build and submit the NEXT step FIRST ──
        // This forces the GPU to compute the pending token as a dependency,
        // so when we eval it in Phase 2, it's already ready.
        if (self.has_pending_logits and self.logprobs_n == 0 and self.step + 1 < self.max_tokens) {
            const step_logits = self.pending_logits;
            self.has_pending_logits = false;

            const lazy_token = self.sampleLazy(step_logits);
            _ = mlx.mlx_array_free(step_logits);

            if (lazyForward(self.xfm, &self.ctx, lazy_token)) |next_logits| {
                const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                _ = mlx.mlx_async_eval(vec);
                _ = mlx.mlx_vector_array_free(vec);

                // NOW resolve the pending token — GPU already computed it as a
                // dependency of the graph we just submitted. Should be instant.
                try self.resolvePendingToken();

                // Check stop conditions on the resolved token
                if (try self.checkStop()) return null;

                const token = self.next_token_id;
                self.advanceStep(1);
                try self.generated_ids.append(allocator, token);

                // Store new pending state
                self.pending_token = lazy_token;
                self.has_pending_token = true;
                self.pending_logits = next_logits;
                self.has_pending_logits = true;

                return token;
            } else |_| {
                // lazyForward failed — fall through to slow path
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                self.has_pending_token = false;
            }
        }

        // ── Phase 2: Slow path (first token, last token, logprobs, or pipeline miss) ──
        try self.resolvePendingToken();

        if (try self.checkStop()) return null;

        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = if (self.has_pending_logits) blk: {
            const logits = self.pending_logits;
            self.has_pending_logits = false;
            break :blk logits;
        } else blk: {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            break :blk try self.xfm.forwardWith(&self.ctx, tok_input);
        };

        // Logprobs: fully synchronous
        if (self.logprobs_n > 0) {
            defer _ = mlx.mlx_array_free(step_logits);
            const result = try sampleToken(allocator, step_logits, self.sampling, self.generated_ids.items, self.logprobs_n, self.xfm.s);
            self.sampling.draw +%= 1;
            self.next_token_id = result.token_id;
            // `result` belongs to the token we just SAMPLED, which the next
            // call returns; `token` is carrying the previous round's result.
            if (self.last_logprob) |*lp| allocator.free(lp.top_logprobs);
            self.last_logprob = self.pending_logprob;
            self.pending_logprob = result.logprob_result;
            if (self.step < self.max_tokens) self.startAsyncForward(result.token_id);
            return token;
        }

        // Last token or pipeline bootstrap
        const lazy_token = self.sampleLazy(step_logits);
        _ = mlx.mlx_array_free(step_logits);

        if (self.step < self.max_tokens) {
            const next_logits = lazyForward(self.xfm, &self.ctx, lazy_token) catch {
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                return token;
            };

            const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);

            self.pending_token = lazy_token;
            self.has_pending_token = true;
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            try mlx.check(mlx.mlx_array_eval(lazy_token));
            var val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
            _ = mlx.mlx_array_free(lazy_token);
            self.next_token_id = @intCast(val);
        }

        return token;
    }

    /// Synchronous, grammar-constrained sampling step. Used whenever
    /// `sampling.constraint` is non-null. Builds a token mask from the grammar's
    /// current state, applies it to the pending logits, samples, advances the
    /// grammar by the sampled token's bytes, and pre-launches the next forward
    /// pass to overlap with the next mask build.
    fn nextConstrained(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (!self.has_pending_logits) {
            self.done = true;
            return null;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }

        const constraint = self.sampling.constraint.?;
        const s = self.xfm.s;

        const allowed = try token_mask.buildMask(constraint.grammar, constraint.token_bytes, constraint.mask_buf);
        if (allowed == 0) {
            // No legal token: every logit would be -inf and argmax over that
            // row returns id 0, whose bytes then fail `acceptByte` and switch
            // enforcement off anyway — one garbage token later, and with the
            // grammar bug reported as model output. Say so and degrade here.
            log.warn("[grammar] no token satisfies the schema at this position — disabling further mask enforcement\n", .{});
            constraint.grammar.dead = true;
            @memset(constraint.mask_buf, true);
        }

        // Also allow every stop-id the generator recognises once the grammar is
        // complete. `token_mask.buildMask` only knows about `tokenizer.eos_id`,
        // but models often have additional stop tokens (e.g. `<|im_end|>` for
        // Qwen, `<end_of_turn>` for Gemma 4) registered via the config — without
        // this, the model can never stop.
        if (constraint.grammar.isComplete()) {
            for (self.eos_token_ids) |eos_id| {
                if (eos_id < constraint.mask_buf.len) constraint.mask_buf[eos_id] = true;
            }
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        defer _ = mlx.mlx_array_free(step_logits);

        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

        // Synchronous sample: we need the realized token id to advance the grammar.
        const lazy = self.sampleLazy(masked_logits);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        // Stop on EOS — do not advance grammar or include in output.
        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        } else {
            self.consecutive_pad = 0;
        }

        // Advance the grammar by the sampled token's byte sequence. The mask
        // guarantees every byte is accepted (or the token has no byte form, e.g. a
        // special tag) — so a rejection here means a bug we want to surface.
        if (token < constraint.token_bytes.bytes.len) {
            if (constraint.token_bytes.bytes[token]) |bytes| {
                for (bytes) |b| {
                    const ok = try constraint.grammar.acceptByte(b);
                    if (!ok) {
                        log.warn("[grammar] sampled token {d} produced byte 0x{x} that was rejected — disabling further mask enforcement\n", .{ token, b });
                        constraint.grammar.dead = true;
                        break;
                    }
                }
            }
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (self.step < self.max_tokens) {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{next_logits};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }

        return token;
    }

    /// Check all stop conditions. Returns true if generation should stop.
    fn checkStop(self: *Generator) !bool {
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        for (self.eos_token_ids) |eos_id| {
            if (self.next_token_id == eos_id) {
                self.done = true;
                self.finish_reason = "stop";
                return true;
            }
        }
        if (self.next_token_id == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                return true;
            }
        } else {
            self.consecutive_pad = 0;
        }
        return false;
    }

    /// Legacy sync forward for logprobs path.
    fn startAsyncForward(self: *Generator, token_id: u32) void {
        const tok_i32: i32 = @intCast(token_id);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);

        const logits = self.xfm.forwardWith(&self.ctx, tok_input) catch return;
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);

        self.pending_logits = logits;
        self.has_pending_logits = true;
    }
};

/// Build forward pass from a lazy sampled token array.
/// Reshapes [1] -> [1, 1] and calls transformer forward. All lazy (no eval).
fn lazyForward(xfm: *Transformer, ctx: *ForwardCtx, lazy_token: mlx.mlx_array) !mlx.mlx_array {
    const tok_shape = [_]c_int{ 1, 1 };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, lazy_token, &tok_shape, 2, xfm.s));
    // The graph is built while the previous step still runs on the GPU; an
    // arch with a host-side token lookup (qwen4 n-gram PLE) fills its leaf
    // afterwards, so the one sync on the token comes AFTER the build.
    ctx.ple_defer = true;
    defer ctx.ple_defer = false;
    const logits = xfm.forwardWith(ctx, reshaped) catch |e| {
        xfm.discardDeferredPle(ctx);
        return e;
    };
    xfm.flushDeferredPle(ctx) catch |e| {
        _ = mlx.mlx_array_free(logits);
        return e;
    };
    return logits;
}

/// Sample a token lazily from logits — returns a lazy MLX array (no eval).
/// Handles temperature scaling, top-k, and top-p, but defers materialization.
/// The returned array has shape [1] with the sampled token ID.
/// Caller must free the returned array.
/// Compute the probability distribution over the vocabulary at the LAST
/// position of `logits_3d` (shape `[B, S, V]`), with the SAME temperature +
/// top-k + top-p masking the sampler would apply. Both `target_p` and `draft_q`
/// in the stochastic-verify accept test must be computed via this function so
/// the ratio `p[draft] / q[draft]` is well-defined over the kept support.
/// Caller owns the returned array; shape `[B, V]`.
/// Batched sibling of `probsAtLastPos`: temperature → top-k → top-p →
/// softmax over EVERY position of `[1, L, V]` logits in one set of
/// row-parallel kernels. A per-position loop pays L separate ~vocab-sized
/// sort/topk kernel launches per spec-decode round; batched it's one each.
/// All filter helpers operate on the last axis, so leading dims pass through.
fn probsAllPositions(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&current, logits_3d));

    // Reserved-token suppression (batched MTP stochastic verify) — same
    // rationale as `probsAtLastPos`.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0) {
        var masked = mlx.mlx_array_new();
        applyTopK(&masked, current, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }
    if (sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        applyTopP(&masked, current, sampling.top_p, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    var probs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    return probs;
}

/// Lazy log-confidence of one MTP draft: `logits[draft] − logsumexp(logits)`
/// = log p_head(draft). Two vocab reductions on the head's own (draft-head)
/// logits — must be built BEFORE the caller frees the logits handle (lazy
/// graphs hold their inputs internally). Returns a `[1]`-shaped lazy array.
fn draftConfidenceGraph(logits: mlx.mlx_array, draft_id: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var lse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lse);
    try mlx.check(mlx.mlx_logsumexp_axis(&lse, logits, -1, false, s));
    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, logits, draft_id, -1, s));
    const flat = [_]c_int{1};
    var t_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t_flat);
    try mlx.check(mlx.mlx_reshape(&t_flat, taken, &flat, 1, s));
    var l_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_flat);
    try mlx.check(mlx.mlx_reshape(&l_flat, lse, &flat, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_subtract(&out, t_flat, l_flat, s));
    return out;
}

/// The chunk-A boundary sync: ONE bounded GPU round-trip that realizes the
/// chunk's draft ids (needed on the CPU later anyway) plus their
/// confidences, and returns the chain log-confidence
/// `Σ min(0, ln p_head(draft_i))` for the extension gate.
fn readChainConfidence(draft_arrs: []const mlx.mlx_array, conf_arrs: []const mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var conf_vec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conf_vec);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (conf_arrs) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
        try mlx.check(mlx.mlx_astype(&conf_vec, cat, .float32, s));
    }
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        for (draft_arrs) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
        _ = mlx.mlx_vector_array_append_value(eval_vec, conf_vec);
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }
    try mlx.check(mlx.mlx_array_eval(conf_vec));
    const data = mlx.mlx_array_data_float32(conf_vec) orelse return error.MlxArrayDataNull;
    return Generator.mtpChainLogConf(data[0..conf_arrs.len]);
}

fn probsAtLastPos(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    const seq_len = shape[1];
    var current = mlx.mlx_array_new();
    if (seq_len == 1) {
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, logits_3d, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ shape[0], seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits_3d, &start, 3, &stop, 3, &strides, 3, s));
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s));
    }

    // Reserved-token suppression: the filtered probs feed spec-verify
    // acceptance AND the residual corrections, so a suppressed draft's
    // acceptance probability is exactly 0 and the residual can't re-draw it.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    return filteredProbsRows(current, sampling, s);
}

/// Temperature → top-k → top-p → softmax over the LAST axis of an owned
/// `[.., V]` array (consumed). The ONE place a proposal or target density is
/// built: a draft sampled from a distribution that is not byte-for-byte the
/// `q` handed to `specAcceptProb` breaks exactness silently, so both sides
/// come through here. Row-independent, so it serves a single `[1, V]` row and
/// a whole `[m, V]` block identically.
fn filteredProbsRows(owned_rows: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = owned_rows;
    // Apply temperature → top-k → top-p (same order as `sampleTokenLazy`).
    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0) {
        var masked = mlx.mlx_array_new();
        applyTopK(&masked, current, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }
    if (sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        applyTopP(&masked, current, sampling.top_p, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    // Softmax: tokens at -inf become 0, kept tokens renormalize to sum=1.
    var probs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    return probs;
}

/// Proposal densities for EVERY row of a `[1, m, V]` draft-logit block, as
/// `[m, V]`. One filter pass and (at the call site) one categorical over m
/// rows, rather than m of each — the draft block is on the critical path
/// ahead of the verify forward.
fn filteredProbsBlock(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    var rows = mlx.mlx_array_new();
    const rows_shape = [_]c_int{ shape[0] * shape[1], shape[2] };
    try mlx.check(mlx.mlx_reshape(&rows, logits_3d, &rows_shape, 2, s));
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        try applySuppressMask(&masked, rows, m, s);
        _ = mlx.mlx_array_free(rows);
        rows = masked;
    }
    return filteredProbsRows(rows, sampling, s);
}

/// Read `probs[0, token_id]` as f32. Forces realization with a single eval.
fn probAt(probs: mlx.mlx_array, token_id: u32, s: mlx.mlx_stream) !f32 {
    const idx_val: i32 = @intCast(token_id);
    const idx_shape = [_]c_int{1};
    const idx_arr = mlx.mlx_array_new_data(&idx_val, &idx_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(idx_arr);

    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, probs, idx_arr, -1, s));

    // Cast to f32 so item_float32 is exact regardless of source dtype (bf16 etc.).
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, taken, .float32, s));
    try mlx.check(mlx.mlx_array_eval(as_f32));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, as_f32));
    return v;
}

/// Sample one token from probability distribution `probs` (shape `[B, V]`).
/// Returns a u32 token id (caller can append directly).
fn sampleFromProbs(probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    // mlx_random_categorical takes logits and applies softmax. Feed log(probs)
    // so the categorical's softmax recovers the original distribution.
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_log(&log_probs, probs, s));

    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, log_probs, -1, null_key, s));
    try mlx.check(mlx.mlx_array_eval(sampled));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, sampled));
    return @intCast(v);
}

/// Build a one-hot float32 row vector of shape `[1, vocab]` with 1.0 at
/// `index` and 0.0 elsewhere. Used by PLD's stochastic-verify reject path,
/// which models the draft (an n-gram lookup, not a probabilistic model) as a
/// degenerate one-hot distribution. Caller owns the returned array.
fn pldOneHotRow(index: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));

    const target_val: i32 = @intCast(index);
    const tgt_shape = [_]c_int{1};
    const target_idx = mlx.mlx_array_new_data(&target_val, &tgt_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(target_idx);

    var mask_bool = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_bool);
    try mlx.check(mlx.mlx_equal(&mask_bool, indices, target_idx, s));

    var mask_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_f32);
    try mlx.check(mlx.mlx_astype(&mask_f32, mask_bool, .float32, s));

    const out_shape = [_]c_int{ 1, vocab };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, mask_f32, &out_shape, 2, s));
    return out;
}

/// One row of a `[m, V]` density block as its own `[1, V]` array, so it can
/// feed `probAt` and `sampleResidual` (both row-shaped). Caller owns it.
fn sliceProbRow(rows_2d: mlx.mlx_array, row: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(rows_2d);
    const start = [_]c_int{ @intCast(row), 0 };
    const stop = [_]c_int{ @as(c_int, @intCast(row)) + 1, shape[1] };
    const strides = [_]c_int{ 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, rows_2d, &start, 2, &stop, 2, &strides, 2, s));
    return out;
}

/// The DFlash2 selector's proposal density for one step as a `[1, V]` row:
/// its per-candidate softmax scattered into zeros — q is zero off the
/// candidate set by construction, so this is the exact density the draft was
/// drawn from (the residual-correction contract).
fn selectorQRow(sp: *const dflash_mod.SelectedPath, step: usize, m: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const kk = sp.cand_ids.len / @as(usize, m);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    const z_shape = [_]c_int{ 1, vocab };
    try mlx.check(mlx.mlx_zeros(&zeros, &z_shape, 2, .float32, s));
    const row_shape = [_]c_int{ 1, @intCast(kk) };
    const ids_arr = mlx.mlx_array_new_data(sp.cand_ids.ptr + step * kk, &row_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);
    const vals_arr = mlx.mlx_array_new_data(sp.q.?.ptr + step * kk, &row_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(vals_arr);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_put_along_axis(&out, zeros, ids_arr, vals_arr, 1, s));
    return out;
}

/// Sample from the residual distribution `residual = max(target - draft, 0)`,
/// renormalized. Used on stochastic-verify reject so the corrected token
/// preserves the target distribution (per Leviathan et al. speculative
/// decoding paper).
fn sampleResidual(target_probs: mlx.mlx_array, draft_probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, target_probs, draft_probs, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var residual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(residual);
    try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));

    return sampleFromProbs(residual, s);
}

/// Lazy categorical sample from an already-filtered probability row
/// ([1, vocab]): log(probs) puts masked tokens at -inf, categorical draws
/// within the kept set — the same distribution as sampling the filtered
/// logits directly, but the caller keeps `probs` as the proposal density q.
fn sampleFromProbsLazy(probs: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var logp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logp);
    try mlx.check(mlx.mlx_log(&logp, probs, s));
    var sampled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sampled);
    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, logp, -1, null_key, s));
    return sampled;
}

pub fn sampleTokenLazy(logits_in: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    // Reserved-token suppression first, so every path below (greedy fast
    // path included — the collapse drew `<|fim_hole|>` at temp 0) sees the
    // masked row. Lazy like everything else here. The handle starts null-ctx
    // and is only materialized when a mask exists — the fast path below
    // exists to save ONE FFI call per decode step, so the no-mask case (every
    // non-suppressing model) must not spend two on an empty array.
    var suppressed: mlx.mlx_array = .{ .ctx = null };
    defer if (suppressed.ctx != null) {
        _ = mlx.mlx_array_free(suppressed);
    };
    const logits = if (sampling.suppress_mask) |m| blk: {
        suppressed = mlx.mlx_array_new();
        applySuppressMask(&suppressed, logits_in, m, s) catch break :blk logits_in;
        break :blk suppressed;
    } else logits_in;

    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Greedy + seq_len==1 (the decode hot path): one mlx op total. argmax_axis
    // over the vocab dim of a `[1, 1, V]` tensor yields a `[1, 1]` int array,
    // which downstream (resolvePendingToken / lazyForward / async_eval vector)
    // treats identically to `[1]`. Skipping the otherwise-needed reshape +
    // argmax-on-2D combo cuts ~one FFI call per decode step.
    if (seq_len == 1 and sampling.temperature < 0.01) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, logits, -1, false, s);
        return result;
    }

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    // `current` is the single owned intermediate — freed before each reassignment.
    var current = mlx.mlx_array_new();

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, logits, &sq_shape, 2, s);
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        _ = mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s);

        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s);
    }

    // Greedy: argmax (no temperature)
    if (sampling.temperature < 0.01) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, current, -1, false, s);
        _ = mlx.mlx_array_free(current);
        return result;
    }

    // Scale by 1/temperature
    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        _ = mlx.mlx_divide(&next, current, temp_arr, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-k filtering (lazy)
    if (sampling.top_k > 0) {
        var next = mlx.mlx_array_new();
        applyTopK(&next, current, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-p filtering (lazy)
    if (sampling.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        applyTopP(&next, current, sampling.top_p, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Sample from categorical distribution (lazy — no eval!)
    var sampled = mlx.mlx_array_new();
    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    _ = mlx.mlx_random_categorical(&sampled, current, -1, key, s);
    _ = mlx.mlx_array_free(current);

    return sampled; // Shape [1], lazy
}

/// PRNG key for one draw: `seed` mixed with the draw index, so a seeded
/// request replays byte-for-byte and consecutive draws never share a key.
/// Null-ctx (MLX global RNG) when the request set no seed.
fn seedKey(sampling: SamplingParams) mlx.mlx_array {
    const seed = sampling.seed orelse return mlx.mlx_array_new();
    var key = mlx.mlx_array_new();
    _ = mlx.mlx_random_key(&key, seed +% sampling.draw *% 0x9E3779B97F4A7C15);
    return key;
}

/// Convenience: generate all tokens at once (non-streaming).
pub fn generate(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    logprobs_n: u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    // logprobs_n rides InitOptions so init's argmax-only gate sees it (a
    // post-init field write would let the split-prefill final-token forward
    // engage the pruned lm_head on a logprobs request).
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .logprobs_n = logprobs_n });
    gen.timeout_ns = timeout_ns;
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var logprob_results = std.ArrayList(LogprobResult).empty;
    defer {
        if (logprobs_n == 0) {
            for (logprob_results.items) |*lp| allocator.free(lp.top_logprobs);
            logprob_results.deinit(allocator);
        }
    }

    timer.reset();
    while (try gen.next(allocator)) |token_id| {
        try output_ids.append(allocator, token_id);
        if (logprobs_n > 0) {
            if (gen.last_logprob) |lp| {
                try logprob_results.append(allocator, lp);
                gen.last_logprob = null; // Transfer ownership
            }
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    log.debug("Decode: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        decode_ns / std.time.ns_per_ms,
        num_decoded,
        decode_tps,
    });

    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = if (logprobs_n > 0) try logprob_results.toOwnedSlice(allocator) else null,
    };
}

/// PLD-enabled non-streaming variant of `generate`. Model-agnostic — works on
/// every supported architecture, no extra weights required. Logprobs and
/// constrained sampling are unsupported (asserted out by `nextPld`).
///
/// `draft_len` and `key_len` come from server config (`--pld-draft-len` /
/// `--pld-key-len`); typical values are 5 and 3 respectively.
pub fn generatePld(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    draft_len: u32,
    key_len: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .pld_enabled = true, .lookup_prompt = lookup_prompt });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (PLD): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    // Decode loop. Each `nextPld` returns 1..=(1+draft_len) tokens. Stop on
    // EOS / max_tokens / timeout. We check stop conditions on every emitted
    // token (drafts can include EOS just like regular sampling) so the early
    // exit is correct.
    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextPld(allocator, draft_len, key_len)) orelse break;
        defer allocator.free(result.tokens);
        // Match `generate`'s convention: stop tokens are NOT included in
        // output_ids. Check before appending — the speculative path has to do
        // this explicitly because `nextPld` emits multiple tokens at once and
        // can't return-null mid-batch like the single-token `next` does.
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishPldResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

/// Drafter-enabled non-streaming variant of `generate`. Mirrors
/// `generatePld` (multi-token-per-step emit pattern) but the draft comes from
/// a Gemma 4 assistant drafter cross-attending into the target's KV cache
/// instead of an n-gram lookup.
///
/// `drafter` must already be `bind()`-ed to `xfm`. `block_size` is the
/// per-round token budget (drafter forwards = block_size - 1; verify forward
/// length = block_size).
pub fn generateDrafter(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    drafter: *DrafterModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    block_size: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
        .drafter_enabled = true,
        .drafter = drafter,
        .drafter_block_size = block_size,
        .lookup_prompt = lookup_prompt,
    });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (drafter): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextDrafter(allocator)) orelse break;
        defer allocator.free(result.tokens);
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishDrafterResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

fn finishDrafterResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.drafter_attempted > 0) {
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.drafter_accepted_tokens)) / @as(f64, @floatFromInt(gen.drafter_attempted));
        log.info("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; drafter accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.drafter_accepted_tokens,
            gen.drafter_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; no draft attempts)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    gen.logQsaArms();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

/// MTP-enabled non-streaming variant of `generate`. Mirrors `generateDrafter`
/// but drives `nextMtp` (the model's own multi-token-prediction head).
/// `head` must already be `bind()`-ed to `xfm`.
pub fn generateMtp(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    head: *mtp_mod.MtpModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    depth: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
        .mtp_enabled = true,
        .mtp = MtpHeadRef{ .qwen = head },
        .mtp_depth = depth,
        .lookup_prompt = lookup_prompt,
    });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (mtp): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextMtp(allocator)) orelse break;
        defer allocator.free(result.tokens);
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.mtp_attempted > 0) {
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.mtp_accepted_tokens)) / @as(f64, @floatFromInt(gen.mtp_attempted));
        log.info("Decode (mtp): {d}ms ({d} tokens, {d:.3} tok/s; mtp accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.mtp_accepted_tokens,
            gen.mtp_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (mtp): {d}ms ({d} tokens, {d:.3} tok/s; no draft attempts)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    gen.logQsaArms();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

fn finishPldResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.pld_attempted > 0) {
        // "Tokens saved" = accepted_tokens (drafts that landed) + 0 from
        // verify forwards that ran. Acceptance ratio is per-position, so we
        // compute average tokens accepted per attempt for visibility.
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.pld_accepted_tokens)) / @as(f64, @floatFromInt(gen.pld_attempted));
        log.info("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; pld accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.pld_accepted_tokens,
            gen.pld_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; no n-gram matches found)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    gen.logQsaArms();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

pub fn isEosId(id: u32, eos: []const u32) bool {
    for (eos) |e| if (id == e) return true;
    return false;
}

/// Max cycle length (in tokens) scanned by `isDegenerateTailLoop`, and how many
/// identical repetitions of that cycle count as "stuck". A real answer — prose,
/// code, a markdown table — essentially never repeats an identical ≤8-token
/// cycle 16 times in a row, so these won't fire on legitimate output, while a
/// model that has collapsed into spamming one short phrase is caught within a
/// few dozen tokens instead of running all the way to `max_tokens`.
pub const degenerate_loop_max_period: usize = 8;
pub const degenerate_loop_reps: usize = 16;
// Tier 2 (2026-08-02 shooter wrap-up class): a two-sentence cycle of ~58
// tokens repeated 26 times evaded the 8-token tier. Long periods demand
// fewer reps — 10 verbatim repetitions of a 9..64-token cycle is
// degeneration with overwhelming probability (identical long lines in real
// code repeat a handful of times, not ten).
pub const degenerate_loop_long_max_period: usize = 64;
pub const degenerate_loop_long_reps: usize = 10;

// Tier 3 (2026-08-04 agent-traffic class): a restatement loop that VARIES its
// phrasing has no exact cycle at any period, so both tiers above are blind by
// construction. What it does have is a long stretch of output that recycles a
// tiny vocabulary and introduces almost no new n-grams.
//
// Everything about this tier is deliberately reluctant, because unlike the
// exact tiers it is a fuzzy judgement and a false cut truncates a real answer:
// the window is LONG (a legitimate repetitive passage — a table, a block of
// near-identical code, a list scaffold — is finite and ends well inside it),
// the two ratios must BOTH be low (either one alone convicts honest output —
// a numeric table has few distinct tokens, a repeated code scaffold has few
// distinct n-grams), and the window is longer than the exact tiers' reach so
// this tier only ever speaks about spans they have already declined.
pub const near_repeat_window: usize = 1024;
pub const near_repeat_ngram: usize = 4;
pub const near_repeat_max_ngram_ratio: f32 = 0.35;
pub const near_repeat_max_token_ratio: f32 = 0.12;
/// Third ratio (2026-08-05): how much of the window's SECOND half is n-grams
/// its first half never had. Measured across both shapes at the shipped
/// window: restatement loops 0.019-0.022, healthy repetitive output
/// 0.298-0.827 (dense procedural scene code, a markdown table, the voxel
/// artifact that was wrongly cut). 0.10 sits ~4.5x above the loops and ~3x
/// below the closest healthy case.
pub const near_repeat_max_novelty: f32 = 0.10;

/// Open-addressed distinct-counter sized for one window. Stack-resident, so
/// the whole tier is allocation-free and runs in O(window) per decode tick.
fn DistinctSet(comptime cap: usize) type {
    return struct {
        const Self = @This();
        const empty_key: u64 = std.math.maxInt(u64);
        keys: [cap]u64 = @splat(empty_key),
        n: usize = 0,

        /// True when `key` is already present. Read-only; used to ask whether
        /// the window's second half is introducing anything its first half
        /// did not have.
        fn contains(self: *const Self, key: u64) bool {
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) return false;
                if (self.keys[i] == k) return true;
                i = (i + 1) % cap;
            }
        }

        /// True when `key` was not already present.
        fn insert(self: *Self, key: u64) bool {
            // maxInt is the empty sentinel; fold the one colliding key onto 0.
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) {
                    self.keys[i] = k;
                    self.n += 1;
                    return true;
                }
                if (self.keys[i] == k) return false;
                i = (i + 1) % cap;
            }
        }
    };
}

/// The two-ratio judgement over ONE window-sized span. Split out so the trim
/// search can slide the same window backwards without re-deriving the rule.
fn nearRepeatWindowIsDegenerate(window: []const u32) bool {
    // Load factor 0.5 keeps the linear probe short even when every entry is
    // distinct (the healthy case, which is also the hot one).
    var toks = DistinctSet(near_repeat_window * 2){};
    for (window) |t| _ = toks.insert(t);
    const token_ratio = @as(f32, @floatFromInt(toks.n)) / @as(f32, @floatFromInt(window.len));
    if (token_ratio > near_repeat_max_token_ratio) return false;

    var grams = DistinctSet(near_repeat_window * 2){};
    var i: usize = 0;
    while (i + near_repeat_ngram <= window.len) : (i += 1) {
        _ = grams.insert(gramHash(window[i .. i + near_repeat_ngram]));
    }
    const n_gram_positions = window.len - near_repeat_ngram + 1;
    const gram_ratio = @as(f32, @floatFromInt(grams.n)) / @as(f32, @floatFromInt(n_gram_positions));
    if (gram_ratio > near_repeat_max_ngram_ratio) return false;

    // Third ratio: is the window still PROGRESSING? Both ratios above are
    // properties of a vocabulary, and procedurally generated code has the same
    // vocabulary profile as a loop — a fixed call template plus a small colour
    // palette (live 2026-08-05: a voxel scene cut at 16241 tokens, the user got
    // no file at all). What a loop does NOT do is keep introducing material:
    // measured, a restatement loop's second half brings 1.9-2.2% n-grams its
    // first half never had, while healthy repetitive output brings 29.8-82.7%.
    // Requiring all THREE keeps the tier's reluctance in the direction that
    // matters — a missed loop still ends at max_tokens, a false cut destroys
    // work that was going fine.
    var first_half = DistinctSet(near_repeat_window * 2){};
    const mid = window.len / 2;
    var fi: usize = 0;
    while (fi + near_repeat_ngram <= mid) : (fi += 1) {
        _ = first_half.insert(gramHash(window[fi .. fi + near_repeat_ngram]));
    }
    var second_half = DistinctSet(near_repeat_window * 2){};
    var novel: usize = 0;
    var distinct_second: usize = 0;
    var si: usize = mid;
    while (si + near_repeat_ngram <= window.len) : (si += 1) {
        const h = gramHash(window[si .. si + near_repeat_ngram]);
        if (!second_half.insert(h)) continue; // count each distinct gram once
        distinct_second += 1;
        if (!first_half.contains(h)) novel += 1;
    }
    if (distinct_second == 0) return true; // nothing new because there is nothing
    const novelty = @as(f32, @floatFromInt(novel)) / @as(f32, @floatFromInt(distinct_second));
    return novelty <= near_repeat_max_novelty;
}

/// One hash for both the ratio pass and the novelty pass — two spellings of
/// this would silently compare different things.
fn gramHash(gram: []const u32) u64 {
    var h: u64 = 0;
    for (gram) |t| h = h *% 0x100000001b3 ^ t;
    return h;
}

/// Detect a NEAR-repeat tail loop: the last `near_repeat_window` tokens keep
/// restating the same thing in slightly different words. Pure; reads only the
/// tail, so cost is independent of total generated length.
pub fn isNearRepeatTailLoop(tokens: []const u32) bool {
    if (tokens.len < near_repeat_window) return false;
    return nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..]);
}

/// How far back the near-repeat trim search steps, and how far it may reach.
/// The window is judged whole at each stop, so the set stays window-sized
/// however long the loop ran — sizing a set to the FULL span instead would
/// put tens of KB on the inference thread's stack.
pub const near_repeat_step: usize = 128;
pub const near_repeat_max_lookback: usize = 8192;

/// A convicted degenerate tail: which tier saw it, and where the degenerate
/// span BEGINS in the generated ids.
///
/// The start is what makes a loop cut recoverable. An agent client re-sends
/// the cut turn's content as history, the model reads its own loop back and
/// resumes it — five loop-stops in a row, each firing sooner than the last
/// (live 2026-08-05, under pi). Emitting only the prefix means
/// the loop cannot round-trip into the next prompt.
pub const DegenerateTail = struct {
    tier: Tier,
    /// First index of the degenerate span; `tokens[0..start]` is what a
    /// client should be shown. For the exact tiers ONE copy of the cycle is
    /// deliberately kept — the truncated answer should still show what the
    /// model was doing when it got stuck, and one copy cannot sustain a loop.
    start: usize,

    pub const Tier = enum { exact_cycle, long_cycle, near_repeat };
};

/// The smallest period in `[min_period, max_period]` whose cycle repeats
/// `reps` times at the tail, or null. `isDegenerateTailLoopRange` is this
/// predicate — one implementation, so the detector and the trim can never
/// disagree about what was convicted.
fn exactCyclePeriod(tokens: []const u32, min_period: usize, max_period: usize, reps: usize) ?usize {
    if (max_period == 0 or reps < 2) return null;
    var p: usize = @max(min_period, 1);
    while (p <= max_period) : (p += 1) {
        const span = p * reps;
        if (tokens.len < span) continue;
        const tail = tokens[tokens.len - span ..];
        var periodic = true;
        var i: usize = p;
        while (i < tail.len) : (i += 1) {
            if (tail[i] != tail[i - p]) {
                periodic = false;
                break;
            }
        }
        if (periodic) return p;
    }
    return null;
}

/// Walk the period-`p` cycle backwards past the `reps` that convicted it: a
/// loop that ran 200 times must be trimmed at 200, not at the threshold.
fn trailingCycleStart(tokens: []const u32, p: usize) usize {
    var start = tokens.len - p;
    while (start >= p) {
        if (!std.mem.eql(u32, tokens[start - p .. start], tokens[start .. start + p])) break;
        start -= p;
    }
    return start; // first index of the FIRST copy of the cycle
}

/// Convict a degenerate tail and say where it starts. Tier order matches
/// `scheduler.loopStopReason`: the exact tiers speak first, and the fuzzy
/// near-repeat tier only ever judges spans they have already declined.
pub fn degenerateTail(tokens: []const u32) ?DegenerateTail {
    if (exactCyclePeriod(tokens, 1, degenerate_loop_max_period, degenerate_loop_reps)) |p| {
        return .{ .tier = .exact_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (exactCyclePeriod(
        tokens,
        degenerate_loop_max_period + 1,
        degenerate_loop_long_max_period,
        degenerate_loop_long_reps,
    )) |p| {
        return .{ .tier = .long_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (tokens.len < near_repeat_window) return null;
    if (!nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..])) return null;

    // Slide the window back while it keeps convicting. A restatement loop
    // that has run for 3000 tokens is degenerate for all 3000 — trimming only
    // the last window would hand the client the rest of the loop back.
    var start = tokens.len - near_repeat_window;
    const floor = if (tokens.len > near_repeat_window + near_repeat_max_lookback)
        tokens.len - near_repeat_window - near_repeat_max_lookback
    else
        0;
    while (start >= floor + near_repeat_step) {
        const cand = start - near_repeat_step;
        if (!nearRepeatWindowIsDegenerate(tokens[cand .. cand + near_repeat_window])) break;
        start = cand;
    }
    return .{ .tier = .near_repeat, .start = start };
}

/// Stall clock for the request timeout: the deadline measures time since the
/// last PRODUCED token, not since the request started. A wall-clock request
/// timeout kills legitimate long generations — live capture 2026-07-03:
/// Qwen3.6-27B writing a 33KB file in one tool call decodes for >300s at
/// ~30 tok/s and was guillotined mid-call by the 300s default, which then
/// surfaced as a "butchered" path-only tool call. Progress is detected from
/// the generated-token COUNT at each check, so every decode path (regular,
/// PLD, drafter, MTP — which don't all share an emit site) resets the clock
/// without instrumentation; a request that stops producing (hung forward,
/// deadlock) still times out after `timeout_ns` of silence.
pub const StallClock = struct {
    last_progress_ns: u64 = 0,
    last_progress_count: usize = 0,

    pub fn expired(self: *StallClock, now_ns: u64, generated_count: usize, timeout_ns: u64) bool {
        if (generated_count != self.last_progress_count) {
            self.last_progress_count = generated_count;
            self.last_progress_ns = now_ns;
        }
        if (timeout_ns == 0) return false;
        return now_ns -| self.last_progress_ns >= timeout_ns;
    }
};

/// Detect a degenerate tail loop: the model is stuck emitting the same short
/// token cycle over and over. Returns true when the last `reps` repetitions of
/// some period-`p` cycle (1 ≤ p ≤ `max_period`) are byte-identical.
///
/// Motivation: Gemma 4 12B sometimes collapses after a large/confusing tool
/// result and spams the thinking opener `<|channel>thought` forever; with no
/// repeat penalty (the default) and a now-generous `max_tokens`, nothing else
/// stops it. The decode loop calls this each tick and cuts the slot short.
///
/// Pure and cheap: only the trailing `max_period * reps` ids are inspected, so
/// cost is independent of total generated length.
pub fn isDegenerateTailLoop(tokens: []const u32, max_period: usize, reps: usize) bool {
    return isDegenerateTailLoopRange(tokens, 1, max_period, reps);
}

/// Range variant so a long-period tier can scan 9..64 without also lowering
/// the rep threshold for short cycles (a few "ha ha ha" reps stay legal).
pub fn isDegenerateTailLoopRange(tokens: []const u32, min_period: usize, max_period: usize, reps: usize) bool {
    return exactCyclePeriod(tokens, min_period, max_period, reps) != null;
}

/// Compute a pooled (per the model's `pooling_mode` — mean by default),
/// L2-normalized embedding from token IDs.
/// Returns a float32 array of shape [hidden_size]. Caller must free the returned slice.
pub fn computeEmbedding(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    token_ids: []const u32,
) ![]f32 {
    const seqs = [_][]const u32{token_ids};
    const rows = try computeEmbeddingsBatch(allocator, xfm, &seqs);
    defer allocator.free(rows);
    return rows[0];
}

/// GPU batch-size cap for encoder embedding forwards: bounds padded-batch
/// memory while keeping the GPU saturated.
pub const EMBED_MAX_BATCH: usize = 64;

/// Padded-token budget per embedding sub-batch (issue #117): the item cap
/// alone lets ONE long input inflate the whole padded allocation (64 rows
/// padded to a 32K outlier = 2M positions of hidden state). The budget bounds
/// rows × padded length at the historical worst case (64 rows × 512 — every
/// legacy BERT batch is unchanged); long-context inputs simply ride smaller
/// sub-batches. A single over-budget input still runs, alone.
pub const EMBED_TOKEN_BUDGET: usize = EMBED_MAX_BATCH * 512;

/// End index (exclusive) of the embedding sub-batch starting at `start`:
/// grows while under BOTH the item cap and the padded-footprint budget
/// (rows × running max length). Always takes at least one item, so an input
/// longer than the whole budget is processed rather than looping forever.
/// Input order is preserved — sub-batches are contiguous slices.
pub fn embedSubBatchEnd(seqs: []const []const u32, start: usize, max_items: usize, budget: usize) usize {
    var end = start;
    var max_len: usize = 0;
    while (end < seqs.len and end - start < max_items) {
        const grown_max = @max(max_len, seqs[end].len);
        if (end > start and (end - start + 1) * grown_max > budget) break;
        max_len = grown_max;
        end += 1;
    }
    return end;
}

/// One padded batch of token sequences ready for an encoder forward.
pub const PaddedBatch = struct {
    ids: []i32, // [B * max_len] row-major, pad id 0
    lengths: []usize, // [B]
    max_len: usize,

    pub fn deinit(self: *PaddedBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.lengths);
    }
};

/// Pad `seqs` into one [B, max_len] i32 buffer (pad id 0). Padded positions
/// are excluded from attention (`buildKeyPadMask`) and pooling
/// (`maskedMeanPoolNormalize`), so the pad id value never leaks into results.
pub fn buildPaddedBatch(allocator: std.mem.Allocator, seqs: []const []const u32) !PaddedBatch {
    var max_len: usize = 0;
    for (seqs) |seq| max_len = @max(max_len, seq.len);
    if (max_len == 0) return error.EmptyInput;

    const ids = try allocator.alloc(i32, seqs.len * max_len);
    errdefer allocator.free(ids);
    const lengths = try allocator.alloc(usize, seqs.len);
    errdefer allocator.free(lengths);
    @memset(ids, 0);
    for (seqs, 0..) |seq, b| {
        lengths[b] = seq.len;
        for (seq, 0..) |id, t| ids[b * max_len + t] = @intCast(id);
    }
    return .{ .ids = ids, .lengths = lengths, .max_len = max_len };
}

/// Additive key-padding mask [B, 1, 1, max_len] (bf16): 0 over real keys,
/// -inf over padding. Broadcasts across heads and query positions; padded
/// QUERIES still produce garbage rows, but pooling drops them.
pub fn buildKeyPadMask(allocator: std.mem.Allocator, lengths: []const usize, max_len: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const buf = try allocator.alloc(f32, lengths.len * max_len);
    defer allocator.free(buf);
    for (lengths, 0..) |len, b| {
        for (0..max_len) |t| {
            buf[b * max_len + t] = if (t < len) 0 else -std.math.inf(f32);
        }
    }
    const shape = [_]c_int{ @intCast(lengths.len), 1, 1, @intCast(max_len) };
    const f32_mask = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_mask);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&mask, f32_mask, .bfloat16, s));
    return mask;
}

/// Mean-pool `hidden` [B, T, H] over each row's first `lengths[b]` positions.
/// Returns the pooled [B, H] mlx array (f32-promoted); caller frees.
pub fn maskedMeanPool(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(hidden);
    const batch: usize = @intCast(shape[0]);
    const seq_len: usize = @intCast(shape[1]);

    // Pool weights [B, T, 1]: 1/len over real positions, 0 over padding — a
    // weighted sum along T is then exactly the masked mean. f32 weights also
    // promote a bf16 hidden so the final data extraction is float32-safe.
    const wbuf = try allocator.alloc(f32, batch * seq_len);
    defer allocator.free(wbuf);
    for (lengths, 0..) |len, b| {
        const denom: f32 = @floatFromInt(@max(len, 1));
        for (0..seq_len) |t| {
            wbuf[b * seq_len + t] = if (t < len) 1.0 / denom else 0.0;
        }
    }
    const wshape = [_]c_int{ shape[0], shape[1], 1 };
    const weights = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 3, .float32);
    defer _ = mlx.mlx_array_free(weights);

    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, hidden, weights, s));

    var pooled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_sum_axis(&pooled, weighted, 1, false, s)); // [B, H]
    return pooled;
}

/// Mean-pool `hidden` [B, T, H] over each row's first `lengths[b]` positions
/// and L2-normalize. Returns B owned rows of H f32 each (plus the outer
/// slice); caller frees all.
pub fn maskedMeanPoolNormalize(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, s: mlx.mlx_stream) ![][]f32 {
    const pooled = try maskedMeanPool(allocator, hidden, lengths, s);
    defer _ = mlx.mlx_array_free(pooled);
    return l2NormalizeRows(allocator, pooled, s);
}

/// Single-position pooling (issue #116): select ONE hidden row per batch
/// element — the CLS token (position 0; bge/mxbai) or the last real,
/// non-padding token (`lengths[b]-1`; Qwen3-Embedding). Same one-hot
/// weighted-sum shape as `maskedMeanPool`, so padded garbage can never leak
/// and a bf16 hidden is promoted to f32 by the weights. Returns the pooled
/// [B, H] mlx array; caller frees.
pub fn gatherTokenPool(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, mode: model_mod.PoolingMode, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(hidden);
    const batch: usize = @intCast(shape[0]);
    const seq_len: usize = @intCast(shape[1]);

    const wbuf = try allocator.alloc(f32, batch * seq_len);
    defer allocator.free(wbuf);
    @memset(wbuf, 0);
    for (lengths, 0..) |len, b| {
        const pos: usize = switch (mode) {
            .cls => 0,
            .last_token => @max(len, 1) - 1,
            .mean => return error.InvalidPoolingMode,
        };
        wbuf[b * seq_len + @min(pos, seq_len - 1)] = 1.0;
    }
    const wshape = [_]c_int{ shape[0], shape[1], 1 };
    const weights = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 3, .float32);
    defer _ = mlx.mlx_array_free(weights);

    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, hidden, weights, s));

    var pooled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_sum_axis(&pooled, weighted, 1, false, s)); // [B, H]
    return pooled;
}

/// L2-normalize each row of `pooled` [B, H] and read out as owned f32 rows.
pub fn l2NormalizeRows(allocator: std.mem.Allocator, pooled: mlx.mlx_array, s: mlx.mlx_stream) ![][]f32 {
    const pshape = mlx.getShape(pooled);
    const batch: usize = @intCast(pshape[0]);
    const dim: usize = @intCast(pshape[1]);

    // L2 normalize rows: pooled / max(sqrt(sum(pooled^2)), eps).
    var squared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squared);
    try mlx.check(mlx.mlx_multiply(&squared, pooled, pooled, s));

    var sum_sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum_sq);
    try mlx.check(mlx.mlx_sum_axis(&sum_sq, squared, -1, true, s));

    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, sum_sq, s));

    const eps = mlx.mlx_array_new_float(1e-12);
    defer _ = mlx.mlx_array_free(eps);
    var norm_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm_safe);
    try mlx.check(mlx.mlx_maximum(&norm_safe, norm, eps, s));

    var normalized = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_divide(&normalized, pooled, norm_safe, s));

    try mlx.check(mlx.mlx_array_eval(normalized));
    const data_ptr = mlx.mlx_array_data_float32(normalized) orelse return error.MlxError;

    const rows = try allocator.alloc([]f32, batch);
    var done: usize = 0;
    errdefer {
        for (rows[0..done]) |r| allocator.free(r);
        allocator.free(rows);
    }
    for (rows, 0..) |*row, b| {
        row.* = try allocator.alloc(f32, dim);
        @memcpy(row.*, data_ptr[b * dim .. (b + 1) * dim]);
        done += 1;
    }
    return rows;
}

/// Compute embeddings for many token sequences in GPU batches: each chunk of
/// up to `EMBED_MAX_BATCH` sequences is padded to its own max length,
/// forwarded ONCE through the encoder with a key-padding mask,
/// masked-mean-pooled, and L2-normalized. Input order preserved. Caller
/// frees every returned row and the outer slice.
pub fn computeEmbeddingsBatch(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    seqs: []const []const u32,
) ![][]f32 {
    const results = try allocator.alloc([]f32, seqs.len);
    var filled: usize = 0;
    errdefer {
        for (results[0..filled]) |r| allocator.free(r);
        allocator.free(results);
    }
    var start: usize = 0;
    while (start < seqs.len) {
        const sub = seqs[start..embedSubBatchEnd(seqs, start, EMBED_MAX_BATCH, EMBED_TOKEN_BUDGET)];
        var pb = try buildPaddedBatch(allocator, sub);
        defer pb.deinit(allocator);

        const shape = [_]c_int{ @intCast(sub.len), @intCast(pb.max_len) };
        const input = mlx.mlx_array_new_data(pb.ids.ptr, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);

        // A single sequence has no padding, so it needs no mask.
        var mask: ?mlx.mlx_array = null;
        defer if (mask) |m| {
            _ = mlx.mlx_array_free(m);
        };
        if (sub.len > 1) mask = try buildKeyPadMask(allocator, pb.lengths, pb.max_len, xfm.s);

        const hidden = try xfm.forwardEmbeddingMasked(input, mask);
        defer _ = mlx.mlx_array_free(hidden);

        // Sentence-transformers pipeline order: pool (per the checkpoint's
        // declared mode — mean by default, CLS for bge/mxbai, last-token for
        // Qwen3-Embedding) → dense head (when the checkpoint ships one —
        // EmbeddingGemma) → normalize.
        const pooled = switch (xfm.config.effectivePooling()) {
            .mean => try maskedMeanPool(allocator, hidden, pb.lengths, xfm.s),
            .cls, .last_token => |m| try gatherTokenPool(allocator, hidden, pb.lengths, m, xfm.s),
        };
        defer _ = mlx.mlx_array_free(pooled);
        const rows = if (xfm.hasEmbedProjection()) blk: {
            const projected = try xfm.embedProjection(pooled);
            defer _ = mlx.mlx_array_free(projected);
            break :blk try l2NormalizeRows(allocator, projected, xfm.s);
        } else try l2NormalizeRows(allocator, pooled, xfm.s);
        defer allocator.free(rows);
        for (rows, 0..) |r, i| {
            results[start + i] = r;
            filled += 1;
        }
        start += sub.len;
    }
    return results;
}

const SampleResult = struct {
    token_id: u32,
    logprob_result: ?LogprobResult = null,
};

/// Sample a token from the last position's logits.
/// temperature <= 0.01: greedy argmax. Otherwise: scale logits, apply top_p, and sample.
/// If logprobs_n > 0, also computes logprobs for the sampled token and top N alternatives.
fn sampleToken(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: ?[]const u32, logprobs_n: u32, s: mlx.mlx_stream) !SampleResult {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    var last_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last_logits);

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));

        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, sliced, &sq_shape, 2, s));
    }

    // Track current working logits (avoid copies when no transform needed)
    var current = last_logits;

    // Reserved-token suppression — before any other transform, so the greedy
    // arm below can't draw a suppressed id. `logprobs_logits` stays
    // `last_logits`: the reported distribution is the MODEL's, so under
    // suppression rank 1 may legitimately differ from the chosen token.
    var suppressed = mlx.mlx_array_new();
    var suppressed_owned = false;
    defer if (suppressed_owned) {
        _ = mlx.mlx_array_free(suppressed);
    };
    if (sampling.suppress_mask) |m| {
        try applySuppressMask(&suppressed, current, m, s);
        current = suppressed;
        suppressed_owned = true;
    }

    // Apply repeat penalty to already-generated tokens
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    const needs_penalty = (sampling.repeat_penalty != 1.0 or sampling.presence_penalty != 0.0);
    if (needs_penalty) {
        if (generated_ids) |ids| {
            if (ids.len > 0) {
                try applyRepeatPenalty(&penalized, current, ids, sampling.repeat_penalty, sampling.presence_penalty, s);
                current = penalized;
                penalized_owned = true;
            }
        }
    }

    // Greedy if temperature is ~0
    if (sampling.temperature < 0.01) {
        const token_id = try argmax(current, s);
        var logprob_result: ?LogprobResult = null;
        if (logprobs_n > 0) {
            logprob_result = try computeLogprobs(allocator, last_logits, token_id, logprobs_n, s);
        }
        return .{ .token_id = token_id, .logprob_result = logprob_result };
    }

    // Scale logits by 1/temperature
    var scaled = mlx.mlx_array_new();
    var scaled_owned = false;
    defer if (scaled_owned) {
        _ = mlx.mlx_array_free(scaled);
    };

    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        try mlx.check(mlx.mlx_divide(&scaled, current, temp_arr, s));
        current = scaled;
        scaled_owned = true;
    }

    // Logprobs report the MODEL's distribution, so they read the position's raw
    // logits — not `current`, which by here carries the client's temperature
    // (and any penalty) and would make the same token's logprob move with a
    // sampling knob.
    const logprobs_logits = last_logits;

    // Apply top-k filtering
    var after_topk = mlx.mlx_array_new();
    var topk_owned = false;
    defer if (topk_owned) {
        _ = mlx.mlx_array_free(after_topk);
    };

    if (sampling.top_k > 0) {
        try applyTopK(&after_topk, current, sampling.top_k, s);
        current = after_topk;
        topk_owned = true;
    }

    // Apply top-p (nucleus) sampling
    var after_topp = mlx.mlx_array_new();
    var topp_owned = false;
    defer if (topp_owned) {
        _ = mlx.mlx_array_free(after_topp);
    };

    if (sampling.top_p < 1.0) {
        try applyTopP(&after_topp, current, sampling.top_p, s);
        current = after_topp;
        topp_owned = true;
    }

    // Sample from categorical distribution
    var sampled = mlx.mlx_array_new();

    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));

    // Eval and extract
    try mlx.check(mlx.mlx_array_eval(sampled));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, sampled));

    const token_id: u32 = @intCast(val);

    // Compute logprobs after sampling (we now know the token_id)
    var logprob_result: ?LogprobResult = null;
    if (logprobs_n > 0) {
        logprob_result = try computeLogprobs(allocator, logprobs_logits, token_id, logprobs_n, s);
    }

    _ = mlx.mlx_array_free(sampled);
    return .{ .token_id = token_id, .logprob_result = logprob_result };
}

/// Logprobs for the FIRST generated token, taken from the prefill's
/// final-position logits — the one distribution the decode loop never sees.
/// `chosen` is the id the lazy sampler actually drew, so the reported
/// `token_logprob` belongs to the token that was really emitted (re-sampling
/// here would disagree with it at any temperature above 0).
fn firstTokenLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen: u32, logprobs_n: u32, s: mlx.mlx_stream) !?LogprobResult {
    if (logprobs_n == 0) return null;
    const shape = mlx.getShape(logits);
    if (shape.len != 3) return null;
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    const sq_shape = [_]c_int{ 1, shape[2] };
    if (shape[1] == 1) {
        try mlx.check(mlx.mlx_reshape(&last, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, shape[1] - 1, 0 };
        const stop = [_]c_int{ 1, shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));
        try mlx.check(mlx.mlx_reshape(&last, sliced, &sq_shape, 2, s));
    }
    return try computeLogprobs(allocator, last, chosen, logprobs_n, s);
}

/// Compute log-probabilities from logits. Returns the logprob of the chosen token
/// and the top N alternatives with their token IDs and logprobs.
///
/// `logits` are the MODEL's logits for the position — before temperature,
/// penalties and top-k/top-p — so the reported distribution is the model's, as
/// OpenAI's is. Reading them after temperature made every value move with a
/// knob the client set (and at temp 0 the distribution SATURATES, so most
/// entries report exactly 0.0).
///
/// Ids travel WITH their values through `mlx_argpartition_axis` + a gather.
/// Recovering them afterwards by scanning the vocab for float equality — what
/// this did — is ambiguous the moment two logits tie, which under the
/// saturation above is everywhere: rank 1 was measured to be the chosen token
/// in 0 of 5 positions on a trivial greedy prompt.
fn computeLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen_token: u32, top_n: u32, s: mlx.mlx_stream) !LogprobResult {
    // Compute log_softmax = log(softmax(logits)) on GPU
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, logits, -1, true, s));

    var log_probs_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs_raw);
    try mlx.check(mlx.mlx_log(&log_probs_raw, probs, s));

    // Cast to float32 for CPU readback (model may produce float16 logits)
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_astype(&log_probs, log_probs_raw, .float32, s));

    const lp_shape = mlx.getShape(log_probs);
    const rank = lp_shape.len;
    const vocab_size: usize = @intCast(lp_shape[rank - 1]);
    const k: usize = @min(@as(usize, @min(top_n, 20)), vocab_size);

    // Top-k INDICES, carried alongside their values. Negating turns "k largest"
    // into the "k smallest" that argpartition puts in the leading slots.
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, log_probs, s));

    var part_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part_idx);
    try mlx.check(mlx.mlx_argpartition_axis(&part_idx, neg, @intCast(if (k == 0) 0 else k - 1), -1, s));

    var start_buf: [8]c_int = @splat(0);
    var stop_buf: [8]c_int = @splat(1);
    var stride_buf: [8]c_int = @splat(1);
    for (0..rank) |i| stop_buf[i] = lp_shape[i];
    stop_buf[rank - 1] = @intCast(k);

    var idx_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_k);
    try mlx.check(mlx.mlx_slice(&idx_k, part_idx, &start_buf, rank, &stop_buf, rank, &stride_buf, rank, s));

    var vals_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_raw);
    try mlx.check(mlx.mlx_take_axis(&vals_raw, log_probs, idx_k, @intCast(rank - 1), s));

    var vals_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_k);
    try mlx.check(mlx.mlx_astype(&vals_k, vals_raw, .float32, s));

    var ids_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_k);
    try mlx.check(mlx.mlx_astype(&ids_k, idx_k, .int32, s));

    try mlx.check(mlx.mlx_array_eval(log_probs));
    try mlx.check(mlx.mlx_array_eval(vals_k));
    try mlx.check(mlx.mlx_array_eval(ids_k));

    // Read the logprob of the chosen token from the full array
    const lp_data = mlx.mlx_array_data_float32(log_probs);
    const chosen_logprob: f32 = if (lp_data) |ptr|
        (if (chosen_token < vocab_size) ptr[chosen_token] else -100.0)
    else
        -100.0;

    const val_ptr = mlx.mlx_array_data_float32(vals_k);
    const id_ptr = mlx.mlx_array_data_int32(ids_k);

    var top_logprobs = try allocator.alloc(TokenLogprob, k);
    errdefer allocator.free(top_logprobs);
    var filled: usize = 0;
    if (val_ptr) |vp| {
        if (id_ptr) |ip| {
            for (0..k) |i| {
                const tid = ip[i];
                if (tid < 0 or @as(usize, @intCast(tid)) >= vocab_size) continue;
                top_logprobs[filled] = .{ .token_id = @intCast(tid), .logprob = vp[i] };
                filled += 1;
            }
        }
    }
    // argpartition leaves the winners UNORDERED; ties break on the lower id so
    // the ranking is deterministic run to run.
    std.mem.sort(TokenLogprob, top_logprobs[0..filled], {}, struct {
        fn lt(_: void, a: TokenLogprob, b: TokenLogprob) bool {
            if (a.logprob != b.logprob) return a.logprob > b.logprob;
            return a.token_id < b.token_id;
        }
    }.lt);

    if (filled < top_logprobs.len) {
        top_logprobs = allocator.realloc(top_logprobs, filled) catch top_logprobs;
    }

    return .{
        .token_logprob = chosen_logprob,
        .top_logprobs = top_logprobs,
    };
}

/// Apply a grammar token mask to logits. `mask[i]==true` keeps `logits[i]`,
/// `false` replaces it with `-inf`. The mask is broadcast over leading dims so
/// `logits` can be either `[1, vocab]` or `[1, 1, vocab]`.
fn applyGrammarMask(allocator: std.mem.Allocator, res: *mlx.mlx_array, logits: mlx.mlx_array, mask: []const bool, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);
    const logit_mask = try maskForLogitVocab(allocator, mask, vocab_size);
    defer logit_mask.deinit(allocator);

    // Zig's `bool` is one byte and matches MLX's `.bool_` storage exactly.
    const arr_shape = [_]c_int{@intCast(vocab_size)};
    const mask_arr = mlx.mlx_array_new_data(@ptrCast(logit_mask.slice.ptr), &arr_shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    try mlx.check(mlx.mlx_where(res, mask_arr, logits, neg_inf, s));
}

const LogitMaskView = struct {
    slice: []const bool,
    owned: ?[]bool = null,

    fn deinit(self: LogitMaskView, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
    }
};

fn maskForLogitVocab(allocator: std.mem.Allocator, mask: []const bool, vocab_size: usize) !LogitMaskView {
    if (mask.len == vocab_size) return .{ .slice = mask };

    var adjusted = try allocator.alloc(bool, vocab_size);
    @memset(adjusted, false);
    const copy_len = @min(mask.len, vocab_size);
    @memcpy(adjusted[0..copy_len], mask[0..copy_len]);
    return .{ .slice = adjusted, .owned = adjusted };
}

/// Apply top-k filtering: keep only the top k logits, set the rest to -inf.
fn applyTopK(res: *mlx.mlx_array, logits: mlx.mlx_array, k: u32, s: mlx.mlx_stream) !void {
    // Per-ROW top-k. `mlx_topk` (no axis) flattens, which is indistinguishable
    // from the right answer for the [1, V] rows every caller passed until the
    // draft block arrived: on [m, V] it returns the k largest of the WHOLE
    // block, so one row's cutoff masks every other row to -inf and softmax
    // hands back NaN. A reduction helper that has only ever seen one row
    // cannot reveal an axis bug.
    var topk_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals);
    try mlx.check(mlx.mlx_topk_axis(&topk_vals, logits, @intCast(k), -1, s));

    // Get the minimum of the top-k values (the k-th largest) as cutoff
    var cutoff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cutoff);
    try mlx.check(mlx.mlx_min_axis(&cutoff, topk_vals, -1, true, s));

    // Mask: logits >= cutoff
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_greater_equal(&mask, logits, cutoff, s));

    // Replace masked-out logits with -inf
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Apply top-p (nucleus) sampling: mask logits outside the top-p probability mass.
/// Works on the original (unsorted) logits by computing which tokens to keep.
fn applyTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, s: mlx.mlx_stream) !void {
    // Sort logits ascending to get sorted probabilities
    var sorted_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_logits);
    try mlx.check(mlx.mlx_sort_axis(&sorted_logits, logits, -1, s));

    // Softmax of sorted logits (ascending order: smallest probs first)
    var sorted_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_probs);
    try mlx.check(mlx.mlx_softmax_axis(&sorted_probs, sorted_logits, -1, true, s));

    // Cumulative sum from smallest to largest
    var cumsum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cumsum);
    try mlx.check(mlx.mlx_cumsum(&cumsum, sorted_probs, -1, false, true, s));

    // Find the cutoff: tokens where cumsum <= (1 - top_p) are outside the nucleus
    const threshold = mlx.mlx_array_new_float(1.0 - top_p);
    defer _ = mlx.mlx_array_free(threshold);

    var outside_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(outside_mask);
    try mlx.check(mlx.mlx_less_equal(&outside_mask, cumsum, threshold, s));

    // Set outside-nucleus logits to -inf in sorted space
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    // where(outside_mask, -inf, sorted_logits) — mask out the low-prob tokens
    try mlx.check(mlx.mlx_where(res, outside_mask, neg_inf, sorted_logits, s));

    // Note: categorical sampling doesn't care about token ordering,
    // but the sampled index will be in sorted space. We need to unsort.
    // Since categorical returns an index into the logits array, and we want
    // the original vocab index, we need to work in original space instead.

    // Better approach: find the minimum logit value that's in the nucleus,
    // then mask original logits below that threshold.
    _ = mlx.mlx_array_free(res.*);
    res.* = mlx.mlx_array_new();

    // The cutoff logit is the smallest logit still in the nucleus.
    // In sorted (ascending) order, tokens with cumsum > (1-top_p) are in nucleus.
    // The first such token's logit value is our threshold.
    // We can achieve this by: where(cumsum > 1-top_p, sorted_logits, +inf) then take min
    var in_nucleus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(in_nucleus);
    try mlx.check(mlx.mlx_greater(&in_nucleus, cumsum, threshold, s));

    const pos_inf = mlx.mlx_array_new_float(std.math.inf(f32));
    defer _ = mlx.mlx_array_free(pos_inf);

    var nucleus_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nucleus_logits);
    try mlx.check(mlx.mlx_where(&nucleus_logits, in_nucleus, sorted_logits, pos_inf, s));

    // Min value = the cutoff
    var min_val = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(min_val);
    try mlx.check(mlx.mlx_min_axis(&min_val, nucleus_logits, -1, true, s));

    // Mask original logits: keep if >= cutoff, else -inf
    var keep_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_mask);
    try mlx.check(mlx.mlx_greater_equal(&keep_mask, logits, min_val, s));

    try mlx.check(mlx.mlx_where(res, keep_mask, logits, neg_inf, s));
}

/// Apply repeat penalty to already-generated tokens.
/// Uses pure MLX GPU ops — no CPU readback, preserves lazy evaluation graph.
fn applyRepeatPenalty(res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, repeat_penalty: f32, presence_penalty: f32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);

    // Collect unique token ids
    var seen_set = std.AutoHashMap(u32, void).init(std.heap.page_allocator);
    defer seen_set.deinit();
    for (generated_ids) |id| {
        if (id < vocab_size) {
            seen_set.put(id, {}) catch continue;
        }
    }

    if (seen_set.count() == 0) return;

    // Build boolean mask: true at positions of seen tokens
    const mask_data = try std.heap.page_allocator.alloc(u8, vocab_size);
    defer std.heap.page_allocator.free(mask_data);
    @memset(mask_data, 0);

    var it = seen_set.keyIterator();
    while (it.next()) |id_ptr| {
        mask_data[id_ptr.*] = 1;
    }

    const arr_shape = [_]c_int{ 1, @intCast(vocab_size) };
    const mask_arr = mlx.mlx_array_new_data(mask_data.ptr, &arr_shape, 2, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    var current = logits;

    // Repeat penalty: multiply seen tokens by 1/penalty (positive) or penalty (negative)
    // This is equivalent to: where(mask & logits > 0, logits / penalty, where(mask, logits * penalty, logits))
    // Simplified: where(mask, where(logits > 0, logits / penalty, logits * penalty), logits)
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (repeat_penalty != 1.0) {
        const rp = mlx.mlx_array_new_float(repeat_penalty);
        defer _ = mlx.mlx_array_free(rp);
        const inv_rp = mlx.mlx_array_new_float(1.0 / repeat_penalty);
        defer _ = mlx.mlx_array_free(inv_rp);
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);

        // positive_mask = logits > 0
        var positive_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positive_mask);
        try mlx.check(mlx.mlx_greater(&positive_mask, current, zero, s));

        // penalized_positive = logits * (1/penalty)
        var pen_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_pos);
        try mlx.check(mlx.mlx_multiply(&pen_pos, current, inv_rp, s));

        // penalized_negative = logits * penalty
        var pen_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_neg);
        try mlx.check(mlx.mlx_multiply(&pen_neg, current, rp, s));

        // sign_selected = where(positive, logits/penalty, logits*penalty)
        var sign_selected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_selected);
        try mlx.check(mlx.mlx_where(&sign_selected, positive_mask, pen_pos, pen_neg, s));

        // result = where(mask, sign_selected, logits)
        try mlx.check(mlx.mlx_where(&penalized, mask_arr, sign_selected, current, s));
        current = penalized;
        penalized_owned = true;
    }

    // Presence penalty: subtract from seen tokens
    if (presence_penalty != 0.0) {
        const pp = mlx.mlx_array_new_float(presence_penalty);
        defer _ = mlx.mlx_array_free(pp);

        // Cast mask to float for arithmetic
        var mask_float = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_float);
        try mlx.check(mlx.mlx_astype(&mask_float, mask_arr, .float16, s));

        // subtract = mask * presence_penalty
        var subtract = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(subtract);
        try mlx.check(mlx.mlx_multiply(&subtract, mask_float, pp, s));

        // result = current - subtract
        try mlx.check(mlx.mlx_subtract(res, current, subtract, s));
    } else {
        try mlx.check(mlx.mlx_copy(res, current, s));
    }
}

/// Greedy argmax over the last axis.
fn argmax(last_logits: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var argmax_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(argmax_arr);
    try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, last_logits, -1, false, s));

    try mlx.check(mlx.mlx_array_eval(argmax_arr));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, argmax_arr));

    return @intCast(val);
}

// ── Tests ──

const testing = std.testing;

test "SamplingParams defaults" {
    const params = SamplingParams{};
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 0), params.top_k);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.repeat_penalty, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), params.presence_penalty, 0.001);
    try testing.expect(params.seed == null);
}

test "SamplingParams custom values" {
    const params = SamplingParams{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .repeat_penalty = 1.1,
        .presence_penalty = 0.5,
        .seed = 42,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.7), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.9), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(u64, 42), params.seed.?);
}

test "specDecodeUnsupported: release-enforced guard for spec + constraint/logprobs (issue #97)" {
    // Speculative decode (PLD/drafter/MTP) cannot honor a grammar constraint or
    // per-token logprobs. nextPld/nextDrafter/nextMtp only asserted this, which
    // compiles out in ReleaseFast (issue #97) — this is the real check they now
    // use to fail loud instead of streaming silently off-schema output.
    try testing.expect(!specDecodeUnsupported(.{}, 0)); // plain sampling → spec OK
    try testing.expect(specDecodeUnsupported(.{}, 1)); // any logprobs → not OK
    try testing.expect(specDecodeUnsupported(.{}, 5));
    // A grammar constraint present → not OK. The guard only null-checks the
    // pointer (never dereferences), so a dummy address is sufficient here.
    var dummy: Constraint = undefined;
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 0));
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 3));
}

test "GenerationResult fields" {
    // Just verifying the struct shape compiles correctly with all fields
    const result = GenerationResult{
        .text = @constCast("hello"),
        .token_ids = @constCast(&[_]u32{ 1, 2, 3 }),
        .prompt_tokens = 10,
        .completion_tokens = 3,
        .finish_reason = "stop",
        .prefill_tps = 100.0,
        .decode_tps = 35.0,
        .logprobs = null,
    };
    try testing.expectEqual(@as(u32, 10), result.prompt_tokens);
    try testing.expectEqual(@as(u32, 3), result.completion_tokens);
    try testing.expectEqualStrings("stop", result.finish_reason);
    try testing.expect(result.logprobs == null);
}

test "tokensPerSec basic and zero-time" {
    // 100 tokens in 1 second = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(100, std.time.ns_per_s), 1e-6);
    // 50 tokens in 0.5s = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(50, std.time.ns_per_s / 2), 1e-6);
    // Zero elapsed → 0, never inf/NaN.
    try testing.expectEqual(@as(f64, 0.0), tokensPerSec(100, 0));
}

test "prefillTokensPerSec divides by uncached tokens" {
    // Cold prefill: 754 tokens, none cached, 2s → 377 tok/s.
    try testing.expectApproxEqAbs(
        @as(f64, 377.0),
        prefillTokensPerSec(754, 0, 2 * std.time.ns_per_s),
        1e-6,
    );
    // Warm prefill: 754-token prompt, 700 cached, only 54 ran. A fast 54-token
    // suffix in 0.1s is 540 tok/s — NOT 7540 (the inflated full-prompt rate).
    try testing.expectApproxEqAbs(
        @as(f64, 540.0),
        prefillTokensPerSec(754, 700, std.time.ns_per_s / 10),
        1e-6,
    );
    // Full cache hit: 0 uncached → 0 tok/s (no compute happened).
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(754, 754, std.time.ns_per_s));
    // Defensive: cached > prompt (shouldn't happen) clamps to 0, no underflow.
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(10, 20, std.time.ns_per_s));
}

test "maskForLogitVocab pads and truncates to logits size" {
    const short_mask = [_]bool{ true, false, true };
    const padded = try maskForLogitVocab(testing.allocator, &short_mask, 5);
    defer padded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 5), padded.slice.len);
    try testing.expect(padded.slice[0]);
    try testing.expect(!padded.slice[1]);
    try testing.expect(padded.slice[2]);
    try testing.expect(!padded.slice[3]);
    try testing.expect(!padded.slice[4]);

    const long_mask = [_]bool{ false, true, true, true };
    const truncated = try maskForLogitVocab(testing.allocator, &long_mask, 2);
    defer truncated.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), truncated.slice.len);
    try testing.expect(!truncated.slice[0]);
    try testing.expect(truncated.slice[1]);
}

test "argmax selects highest value" {
    // Create a simple logits array [1, 5] with values [0.1, 0.5, 0.9, 0.2, 0.3]
    const data = [_]f32{ 0.1, 0.5, 0.9, 0.2, 0.3 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 2), result); // index 2 has value 0.9
}

test "argmax with negative values" {
    const data = [_]f32{ -5.0, -1.0, -3.0, -0.5, -2.0 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 3), result); // index 3 has value -0.5 (highest)
}

test "applyRepeatPenalty reduces seen token logits" {
    const s = mlx.gpuStream();
    // logits: [1.0, 2.0, 3.0, 4.0, 5.0]
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const shape = [_]c_int{ 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize tokens at indices 1 and 3
    const generated = [_]u32{ 1, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: untouched → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[1], 0.01);
    // Index 2: untouched → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
    // Index 4: untouched → 5.0
    try testing.expectApproxEqAbs(@as(f32, 5.0), res_data[4], 0.01);
}

test "applyRepeatPenalty with negative logits" {
    const s = mlx.gpuStream();
    // Mix of positive and negative logits
    const data = [_]f32{ -2.0, 3.0, -1.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize all tokens
    const generated = [_]u32{ 0, 1, 2, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: negative, multiplied by 2.0 → -4.0
    try testing.expectApproxEqAbs(@as(f32, -4.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.5
    try testing.expectApproxEqAbs(@as(f32, 1.5), res_data[1], 0.01);
    // Index 2: negative, multiplied by 2.0 → -2.0
    try testing.expectApproxEqAbs(@as(f32, -2.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
}

test "applyRepeatPenalty presence penalty" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 2 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 1.0, 0.5, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: seen, presence penalty subtracted → 1.0 - 0.5 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.5), res_data[0], 0.01);
    // Index 1: unseen → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[1], 0.01);
    // Index 2: seen → 3.0 - 0.5 = 2.5
    try testing.expectApproxEqAbs(@as(f32, 2.5), res_data[2], 0.01);
    // Index 3: unseen → 4.0
    try testing.expectApproxEqAbs(@as(f32, 4.0), res_data[3], 0.01);
}

test "applyRepeatPenalty combined penalties" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, -1.0, 3.0 };
    const shape = [_]c_int{ 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 1 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 1.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: positive, divide by 2.0 = 1.0, then - 1.0 = 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), res_data[0], 0.01);
    // Index 1: negative, multiply by 2.0 = -2.0, then - 1.0 = -3.0
    try testing.expectApproxEqAbs(@as(f32, -3.0), res_data[1], 0.01);
    // Index 2: unseen → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
}

test "sampleToken greedy selects argmax" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // Create logits [1, 1, 5] — 5 vocab entries, token at index 3 has highest
    const data = [_]f32{ 1.0, 0.5, 0.1, 5.0, 0.2 };
    const logits_shape = [_]c_int{ 1, 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 3), result.token_id);
}

test "sampleToken with temperature produces valid token" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const logits_shape = [_]c_int{ 1, 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.5 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    // Token should be in valid range
    try testing.expect(result.token_id < 3);
}

test "seeded lazy sampling replays the same draws and advances per draw" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 0.0, 0.0 };
    const shape = [_]c_int{ 1, 1, 2 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    var runs: [2][24]i32 = undefined;
    for (&runs) |*run| {
        var params = SamplingParams{ .temperature = 1.0, .seed = 42 };
        for (run) |*out| {
            const lazy = sampleTokenLazy(logits, params, s);
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            try mlx.check(mlx.mlx_array_item_int32(out, lazy));
            params.draw +%= 1;
        }
    }
    try testing.expectEqualSlices(i32, &runs[0], &runs[1]);
    // The same key every draw replays ONE coin flip 24 times (p = 2^-23).
    var all_same = true;
    for (runs[0][1..]) |v| all_same = all_same and v == runs[0][0];
    try testing.expect(!all_same);
}

test "sampleToken from prefill logits (seq_len > 1)" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // [1, 3, 4] — 3 positions, 4 vocab, should take last position
    const data = [_]f32{
        0.1, 0.2, 0.3, 0.4, // pos 0
        0.5, 0.6, 0.7, 0.8, // pos 1
        9.0, 0.1, 0.1, 0.1, // pos 2 — token 0 is clearly highest
    };
    const logits_shape = [_]c_int{ 1, 3, 4 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 0), result.token_id); // pos 2, index 0 = 9.0
}

test "Generator.runtimeGateShouldDisable below warmup never trips" {
    // Even with zero accepts, before the warmup count we trust the prompt-time
    // gate and never disable speculation mid-decode. drafts_per_round is the
    // typical drafter setting (block_size=4 → 3 drafts per round).
    try testing.expect(!Generator.runtimeGateShouldDisable(0, 0, 3));
    try testing.expect(!Generator.runtimeGateShouldDisable(1, 0, 3));
    try testing.expect(!Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP - 1, 0, 3));
}

test "Generator.runtimeGateShouldDisable trips at warmup with low per-draft rate" {
    // Synthetic low-accept scenario: 5 verify attempts, drafts_per_round=3
    // (drafter at block_size=4). 5 attempts × 3 = 15 drafts proposed.
    // 0 accepted → 0.00 < 0.50 → trip. Same with 1 accepted (0.067).
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 0, 3));
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 1, 3));
    // 7 accepted out of 15 = 0.467 — still below 0.50 → trip.
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 7, 3));
    // 8 accepted out of 15 = 0.533 → keeps running.
    try testing.expect(!Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 8, 3));
}

test "Generator.runtimeGateShouldDisable does not trip with high per-draft rate" {
    // Echo workloads on Gemma drafter: ~93% per-draft acceptance (E4B from
    // bench: 67/(24*3) = 93.1%). Well above threshold → keeps running.
    try testing.expect(!Generator.runtimeGateShouldDisable(24, 67, 3));
    // PLD heavy-echo: ~4 of 5 drafts accepted per attempt = 0.80 per-draft.
    try testing.expect(!Generator.runtimeGateShouldDisable(20, 80, 5));
    // Edge case at exactly the threshold (rate == 0.50) — strict less-than,
    // so does NOT trip.
    try testing.expect(!Generator.runtimeGateShouldDisable(10, 15, 3)); // 15/30 = 0.50
}

test "Generator.runtimeGateShouldDisable creative-content regression scenario" {
    // The Phase 1 bench's exact regression cases on creative prompts:
    //   E4B drafter (bs=4 → drafts_per_round=3): 39/59 attempts → 22.0% per-draft → trip
    //   E2B drafter (bs=2 → drafts_per_round=1): 31/66 attempts → 47.0% per-draft → trip
    //   31B drafter (bs=8 → drafts_per_round=7): 60/(38*7) → 22.6% per-draft → trip
    try testing.expect(Generator.runtimeGateShouldDisable(59, 39, 3)); // E4B creative
    try testing.expect(Generator.runtimeGateShouldDisable(66, 31, 1)); // E2B creative
    try testing.expect(Generator.runtimeGateShouldDisable(38, 60, 7)); // 31B creative
    // The 26B-A4B@bs=2 creative case: 37/(60*1) = 61.7% → above threshold,
    // so the runtime gate alone does NOT save it. MoE regressions need the
    // separate `default_enable_drafter` opt-out at startup.
    try testing.expect(!Generator.runtimeGateShouldDisable(60, 37, 1));
}

test "Generator.runtimeGateShouldDisable handles drafts_per_round=0" {
    // Defensive: if a caller somehow passes a degenerate config (block_size=1
    // → drafts_per_round=0), don't divide by zero. We return false (no trip).
    try testing.expect(!Generator.runtimeGateShouldDisable(100, 0, 0));
}

test "Generator.dflashGateShouldDisable uses accepted yield across block widths" {
    // Never decide from fewer than the DFlash warmup number of paid verifies.
    try testing.expect(!Generator.dflashGateShouldDisable(
        Generator.DFLASH_GATE_WARMUP - 1,
        0,
        Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND,
    ));

    // Muse prose/vision measured 1.0-1.5 accepted drafts per round at both
    // block 8 and block 16: that class loses to serial and must fall back.
    try testing.expect(Generator.dflashGateShouldDisable(3, 3, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(Generator.dflashGateShouldDisable(3, 5, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));

    // Exactly two is the strict break-even boundary; code/tool traffic at
    // 4.4+ and echo traffic near a full block remain on DFlash.
    try testing.expect(!Generator.dflashGateShouldDisable(3, 6, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(!Generator.dflashGateShouldDisable(3, 15, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(!Generator.dflashGateShouldDisable(3, 45, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));

    // Thinking preambles recovered from ~1.4 early to 4.4 whole-request; the
    // resolved-mode threshold keeps that path alive while still cutting off
    // a truly non-yielding reasoning request.
    try testing.expect(!Generator.dflashGateShouldDisable(3, 4, Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(Generator.dflashGateShouldDisable(3, 2, Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND));
}

test "Generator.yieldGateShouldDisable trips on cold-path-dominated workloads" {
    // The 2026-06-10 baseline regression: PLD forced on for a creative essay
    // prompt where the n-gram lookup almost never matches. The per-draft gate
    // never trips (it only counts verify ROUNDS, and there are few), but every
    // step pays the unpipelined cold forward → −14% measured on E2B. The
    // yield gate counts ALL enabled-mode steps: accepted-drafted-tokens per
    // step below the threshold after warmup → disable.
    // Creative: 128 steps, ~6 drafted tokens accepted → yield 0.047 → trip.
    try testing.expect(Generator.yieldGateShouldDisable(128, 6));
    // Heavy echo: 40 steps, 80 accepted (2.0/step) → stay on.
    try testing.expect(!Generator.yieldGateShouldDisable(40, 80));
    // Inside warmup: never trip, even at zero yield.
    try testing.expect(!Generator.yieldGateShouldDisable(Generator.YIELD_GATE_WARMUP - 1, 0));
    // Exactly at warmup with healthy yield: stay on.
    try testing.expect(!Generator.yieldGateShouldDisable(Generator.YIELD_GATE_WARMUP, Generator.YIELD_GATE_WARMUP));
}

test "yield-gate warmup is 8 — the cold-path tax tripled when the AR step got 3x cheaper" {
    // This constant is pure economics and the economics MOVED. Every warmup
    // step pays PLD's unpipelined cold forward plus a synchronous host read of
    // the sampled token, against `next()`'s async-pipelined step — an
    // ~absolute tax measured as a share of the AR step. Fixing the Laguna
    // mscale promotion made that AR step ~3x cheaper, so the same tax became a
    // ~3x larger share and 32 steps of it stopped being affordable.
    //
    // Swept on Laguna XS (one boot, serial vs unconstrained alternating per
    // request, server-reported timings, 5 runs median), against a matrix that
    // deliberately contains PLD's WIN case as well as its loss cases —
    // tuning on loss cases alone drives the warmup to zero and throws the win
    // away:
    //
    //   warmup   echo-edit   code-edit  free-form   explain      qa
    //       32     +76.9%       -4.3%      -1.2%     -1.4%     -7.2%
    //       16     +70.1%       -3.4%      -0.4%     -0.3%     -0.8%
    //        8     +77.4%       +0.3%      -0.2%     -0.3%     -1.5%
    //        4     +76.6%       +3.9%      -0.2%     -0.1%     -1.7%
    //
    // 8 recovers essentially the whole loss while leaving the +77% untouched.
    // 4 measured no worse, but a gate that decides on four observations is
    // fitting noise, and a short preamble before a file echo is the NORMAL
    // agent shape — exactly what a too-eager trip would punish. A premature
    // trip is bounded anyway: `specShouldReenable` re-checks every
    // SPEC_REENABLE_INTERVAL steps.
    try testing.expectEqual(@as(u64, 8), Generator.YIELD_GATE_WARMUP);

    // Sweepable without a rebuild, same contract as `runtimeGateWarmup`:
    // out-of-range values fall back rather than silently disabling the gate.
    try testing.expectEqual(Generator.YIELD_GATE_WARMUP, Generator.yieldGateWarmup());
}

test "Generator.specShouldReenable gates mid-request PLD re-activation" {
    // Disabled-mode periodic check on the COMMITTED sequence (prompt +
    // generated). The decisive case: the model echoes PROMPT content (file
    // edit / tool result) after a novel preamble tripped the yield gate. The
    // echoed tail never repeats ITSELF, so self-repetition scoring misses it;
    // tailMatchFraction sees the prompt occurrence.
    var committed: [96]u32 = undefined;
    // prompt = 48-token "file", generated = 16 novel preamble + 32 echo of the file
    for (committed[0..48], 0..) |*t, i| t.* = @intCast(i + 100);
    for (committed[48..64], 0..) |*t, i| t.* = @intCast(i + 9000);
    for (committed[64..96], 0..) |*t, i| t.* = @intCast(i + 100);
    try testing.expect(Generator.specShouldReenable(&committed, 48));

    // Fully novel committed sequence → stay disabled.
    var novel: [96]u32 = undefined;
    for (&novel, 0..) |*t, i| t.* = @intCast(i * 7 + 1);
    try testing.expect(!Generator.specShouldReenable(&novel, 48));

    // Too little generated yet → not enough signal, stay disabled.
    try testing.expect(!Generator.specShouldReenable(&committed, 8));
}

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

test "StallClock: progress resets the deadline, silence expires it, 0 disables" {
    var clock = StallClock{};
    const s = std.time.ns_per_s;
    // Producing tokens keeps resetting the deadline — a healthy generation
    // can run arbitrarily long (the live bug: a 33KB tool call at 30 tok/s
    // takes >300s and was guillotined mid-call by the wall-clock timeout).
    try std.testing.expect(!clock.expired(0 * s, 0, 300 * s));
    try std.testing.expect(!clock.expired(299 * s, 1000, 300 * s)); // progress at 299s
    try std.testing.expect(!clock.expired(598 * s, 2000, 300 * s)); // progress again
    // No new tokens for the full window -> stalled.
    try std.testing.expect(!clock.expired(700 * s, 2000, 300 * s));
    try std.testing.expect(clock.expired(898 * s, 2000, 300 * s));
    // 0 = disabled, even after silence.
    var off = StallClock{};
    try std.testing.expect(!off.expired(0, 0, 0));
    try std.testing.expect(!off.expired(10_000 * s, 0, 0));
}

test "isDegenerateTailLoop catches a repeated channel-opener cycle" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Gemma 4 12B failure mode: the model spams the thinking opener
    // `<|channel>thought\n` — model that as a 3-token cycle. After enough
    // identical repetitions the tail is a pure period-3 loop → fire.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 7, 8, 9 }); // some real prefix
        var k: usize = 0;
        while (k < R + 4) : (k += 1) {
            try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 }); // <|channel>,thought,\n
        }
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }

    // A single token stuck on repeat (period 1) also counts once it passes R.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        while (k < R + 2) : (k += 1) try ids.append(testing.allocator, 42);
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }
}

/// Build `n` tokens by cycling through `phrasings`, which share a vocabulary.
fn tVaried(al: std.mem.Allocator, phrasings: []const []const u32, n: usize) !std.ArrayList(u32) {
    var ids = std.ArrayList(u32).empty;
    var i: usize = 0;
    while (ids.items.len < n) : (i += 1) {
        try ids.appendSlice(al, phrasings[i % phrasings.len]);
    }
    return ids;
}

test "isNearRepeatTailLoop catches a VARIED-phrasing restatement loop" {
    // Live 2026-08-04, under pi: the model restated the same
    // intent forever while varying the wording — "I need to break this down."
    // / "I need to break this." / "I need to break this down into pieces." —
    // so no exact cycle exists at ANY period and both exact tiers are blind by
    // construction. What IS invariant is that a long stretch of output recycles
    // a tiny vocabulary and introduces almost no new n-grams.
    const al = testing.allocator;
    const phrasings = [_][]const u32{
        &[_]u32{ 40, 41, 42, 43, 44, 45, 46 }, // I need to break this down .
        &[_]u32{ 40, 41, 42, 43, 44, 46 }, // I need to break this .
        &[_]u32{ 40, 41, 42, 43, 44, 45, 47, 48, 46 }, // ... down into pieces .
        &[_]u32{ 49, 40, 41, 42, 43, 44, 45, 46 }, // So I need to break this down .
        &[_]u32{ 40, 41, 42, 43, 44, 45, 50, 46 }, // ... down first .
        &[_]u32{ 51, 42, 43, 44, 45, 46 }, // Let me break this down .
    };
    var ids = try tVaried(al, &phrasings, near_repeat_window + 64);
    defer ids.deinit(al);
    try testing.expect(isNearRepeatTailLoop(ids.items));

    // Below the window the tier says nothing at all: it is a last-resort net
    // for output that has already run a long way, and a false cut truncates a
    // real answer. Short exact cycles remain tier 1/2's job.
    try testing.expect(!isNearRepeatTailLoop(ids.items[0 .. near_repeat_window - 1]));
}

test "isNearRepeatTailLoop leaves legitimately repetitive output alone" {
    const al = testing.allocator;

    // Healthy prose/code: a repeated scaffold, but every line introduces a new
    // identifier. Recycled STRUCTURE is normal; recycled VOCABULARY is not.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var line: u32 = 0;
        while (ids.items.len < near_repeat_window + 64) : (line += 1) {
            try ids.appendSlice(al, &[_]u32{ 10, 11, 12 }); // `const x =`
            try ids.append(al, 1000 + line); // a fresh identifier
            try ids.appendSlice(al, &[_]u32{ 13, 14 }); // `;\n`
        }
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }

    // A numeric table: FEW distinct tokens (digits + separators, and this
    // family pre-tokenizes digits singly) but the 4-grams keep changing. The
    // two ratios have to be read together — either one alone convicts this.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var seed: u32 = 12345;
        while (ids.items.len < near_repeat_window + 64) {
            try ids.append(al, 200); // '|'
            for (0..4) |_| {
                seed = seed *% 1664525 +% 1013904223;
                try ids.append(al, 100 + (seed >> 16) % 10); // a digit
            }
            try ids.appendSlice(al, &[_]u32{ 200, 201 }); // '|', '\n'
        }
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }

    // Fully novel output.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        for (0..near_repeat_window + 64) |i| try ids.append(al, @intCast(i));
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }
}

test "isDegenerateTailLoop does not fire on healthy or briefly-repeating output" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Strictly increasing ids — no cycle at all.
    {
        var ids: [200]u32 = undefined;
        for (&ids, 0..) |*v, i| v.* = @intCast(i);
        try testing.expect(!isDegenerateTailLoop(&ids, P, R));
    }
    // A short burst of repetition (well under R reps) must be left alone — a
    // model legitimately writing "ha ha ha" or a few identical list bullets.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 1, 2, 3, 4, 5 });
        var k: usize = 0;
        while (k < R - 1) : (k += 1) try ids.appendSlice(testing.allocator, &[_]u32{ 50, 51 });
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Periodic tail but with a longer period than we scan for → ignored.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        var base: u32 = 0;
        while (k < R) : (k += 1) {
            // period = P + 3 (> max_period); never a pure short cycle.
            var j: u32 = 0;
            while (j < P + 3) : (j += 1) try ids.append(testing.allocator, base + j);
            base = 0; // same long block repeats, but its period exceeds the scan window
        }
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Too few tokens to judge.
    try testing.expect(!isDegenerateTailLoop(&[_]u32{ 1, 1 }, P, R));
}

test "isNearRepeatTailLoop leaves PROCEDURAL code alone — it recycles a vocabulary while still progressing" {
    // Live 2026-08-05: a pi session was asked for an elaborate voxel scene and
    // the tier cut it at 16241 generated tokens, so the user got NO file at
    // all. The output was healthy — dense `fillBox(x,y,z, x,y,z, C.name);`
    // lines — but it is exactly the shape the first two ratios were built to
    // tolerate and cannot: a fixed template plus a small colour palette gives
    // a tiny distinct-token ratio, and the templated call shape keeps the
    // 4-gram ratio low too. Measured on the real artifact: 0.068 / 0.351
    // against bars of 0.12 / 0.35 — it cleared conviction by 0.001, and the
    // generation's own tail did not.
    //
    // What separates it from a loop is PROGRESS: every line carries new
    // coordinates, so the window's second half keeps introducing n-grams the
    // first half never had (0.298-0.632 measured, against 0.019-0.022 for the
    // restatement loops this tier exists for).
    // Shape taken from the measured artifact, not invented: a CONTIGUOUS run
    // of template tokens (`\n  fillBox(`, `, C.`, `);`) followed by the
    // varying coordinates. The contiguity is what makes it convict — most
    // 4-gram windows sit entirely inside the fixed run and repeat every line.
    // This fixture scores 0.033 / 0.316 against bars of 0.12 / 0.35.
    const al = testing.allocator;
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    var rng: u32 = 99;
    while (ids.items.len < near_repeat_window * 2) {
        var f: u32 = 0;
        while (f < 14) : (f += 1) try ids.append(al, 500 + f);
        var v: usize = 0;
        while (v < 4) : (v += 1) {
            rng = rng *% 1664525 +% 1013904223;
            try ids.append(al, 600 + (rng >> 16) % 19); // a fresh coordinate
            try ids.append(al, 499); // separator
        }
    }
    try testing.expect(!isNearRepeatTailLoop(ids.items));
    try testing.expect(degenerateTail(ids.items) == null);
}

test "degenerateTail: the exact tier reports its tier and keeps ONE cycle" {
    const al = testing.allocator;
    // 20 identical 3-token cycles after a real prefix. The cut is a
    // truncation, so what is emitted should still SHOW what the model got
    // stuck on — one copy of the cycle survives, the other 19 do not.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 7, 8, 9, 10 });
    var k: usize = 0;
    while (k < 20) : (k += 1) try ids.appendSlice(al, &[_]u32{ 101, 102, 103 });

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.exact_cycle, d.tier);
    // 4 prefix + 1 kept cycle = 7 tokens survive.
    try testing.expectEqual(@as(usize, 7), d.start);
    // What survives is the honest prefix plus exactly one cycle.
    try testing.expectEqualSlices(u32, &[_]u32{ 7, 8, 9, 10, 101, 102, 103 }, ids.items[0..d.start]);
}

test "degenerateTail: the trim start walks back PAST the near-repeat window" {
    const al = testing.allocator;
    // The near-repeat tier judges the last 1024 tokens, but a restatement
    // loop that has been running for 3000 tokens is degenerate for all 3000.
    // Trimming only the window would hand the client the other ~2000 back,
    // which is the whole failure this exists to stop.
    const phrasings = [_][]const u32{
        &[_]u32{ 1, 2, 3, 4, 5 },
        &[_]u32{ 1, 2, 3, 5, 4 },
        &[_]u32{ 1, 2, 4, 3, 5 },
    };
    var honest = std.ArrayList(u32).empty;
    defer honest.deinit(al);
    var i: u32 = 0;
    while (i < 900) : (i += 1) try honest.append(al, 1000 + i); // all distinct = healthy

    // Pick phrasings pseudo-randomly: a deterministic rotation would be an
    // exact cycle and the long-period tier would convict it first, which is
    // not the tier under test.
    var loop = std.ArrayList(u32).empty;
    defer loop.deinit(al);
    var rng: u32 = 12345;
    while (loop.items.len < 3000) {
        rng = rng *% 1664525 +% 1013904223;
        try loop.appendSlice(al, phrasings[(rng >> 16) % phrasings.len]);
    }

    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, honest.items);
    try ids.appendSlice(al, loop.items);

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.near_repeat, d.tier);
    // Well past the single window, and never into the healthy prefix.
    try testing.expect(d.start < ids.items.len - near_repeat_window);
    try testing.expect(d.start >= honest.items.len - near_repeat_step);
}

test "degenerateTail: healthy output is never convicted, so nothing is trimmed" {
    var ids: [4000]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    try testing.expect(degenerateTail(&ids) == null);
}

test "degenerateTail: the long-period tier keeps one copy of its sentence cycle" {
    const al = testing.allocator;
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 1, 2 });
    var cycle: [40]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @intCast(500 + i);
    var k: usize = 0;
    while (k < degenerate_loop_long_reps + 2) : (k += 1) try ids.appendSlice(al, &cycle);

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.long_cycle, d.tier);
    try testing.expectEqual(@as(usize, 2 + cycle.len), d.start);
}

/// PURE: walk a chunked prefill exactly as `runPrefill` does, taking the
/// width from `widths` at each boundary (the last entry repeats). The loop's
/// only chunk-end decision IS `nextChunkEnd`, so this is a faithful proxy for
/// a prefill whose width moves — the same relationship `prefillChunkCount`
/// has to the fixed-width loop.
fn walkChunkEnds(
    out: []usize,
    prefix_len: usize,
    widths: []const usize,
    want_ssm_cp: bool,
    stride: usize,
    offset: usize,
) []usize {
    var pos: usize = 0;
    var n: usize = 0;
    while (pos < prefix_len and n < out.len) {
        const w = widths[@min(n, widths.len - 1)];
        const end = nextChunkEnd(pos, prefix_len, w, want_ssm_cp, stride, offset, true);
        out[n] = end;
        n += 1;
        pos = end;
    }
    return out[0..n];
}

test "a width that changes at every boundary never moves a checkpoint" {
    // The invariant the whole per-chunk feature rests on. `ssm_cp_stride`
    // derives from the LAUNCH width, not the per-request or per-chunk one, and
    // every ladder rung divides it — so a chunk that starts narrow still ends
    // exactly on the stride boundaries a wide one would have hit, and the
    // prefix cache sees the same restore points either way. If this breaks,
    // a mixed-width prefill silently changes which prefixes a later turn can
    // restore from.
    const t = testing;
    // The widths a real prefill can walk through, in a deliberately nasty
    // order: down several rungs, back up, down to the floor.
    const mixed = [_]usize{ 4096, 4096, 2048, 1024, 512, 512, 1024, 2048, 4096, 512, 2048, 1024, 4096, 512 };
    const fixed = [_]usize{4096};

    for ([_]usize{ 8192, 4096 }) |stride| {
        for ([_]usize{ 0, 1000 }) |offset| {
            for ([_]usize{ 40960, 40000 }) |prefix_len| {
                var fb: [512]usize = undefined;
                var mb: [512]usize = undefined;
                const f = walkChunkEnds(&fb, prefix_len, &fixed, true, stride, offset);
                const m = walkChunkEnds(&mb, prefix_len, &mixed, true, stride, offset);

                // Both cover [0, prefix_len) exactly once, in order, with no
                // empty chunk (an empty one is an infinite loop in `runPrefill`).
                for ([_][]usize{ f, m }) |ends| {
                    var pos: usize = 0;
                    for (ends) |end| {
                        try t.expect(end > pos);
                        try t.expect(end <= prefix_len);
                        // No chunk STRADDLES a stride boundary: the boundary is
                        // the snapshot point, and a chunk that crosses it skips
                        // a checkpoint.
                        const abs_pos = pos + offset;
                        const abs_end = end + offset;
                        const next_b = ((abs_pos / stride) + 1) * stride;
                        if (end != prefix_len) try t.expect(!(next_b > abs_pos and next_b < abs_end));
                        pos = end;
                    }
                    try t.expectEqual(prefix_len, pos);
                }

                // And the checkpoint positions are IDENTICAL — same set, same
                // order — however the width moved in between.
                var fi: usize = 0;
                var mi: usize = 0;
                while (true) {
                    while (fi < f.len and (f[fi] + offset) % stride != 0) fi += 1;
                    while (mi < m.len and (m[mi] + offset) % stride != 0) mi += 1;
                    if (fi >= f.len or mi >= m.len) break;
                    try t.expectEqual(f[fi], m[mi]);
                    fi += 1;
                    mi += 1;
                }
                while (fi < f.len and (f[fi] + offset) % stride != 0) fi += 1;
                while (mi < m.len and (m[mi] + offset) % stride != 0) mi += 1;
                try t.expectEqual(fi >= f.len, mi >= m.len);
            }
        }
    }
}

test "every ladder rung the adaptive width can take divides the checkpoint stride" {
    // The other half of the same invariant, at the source: "SSM-checkpoint
    // stride never sub-divides the chunk" holds for a NARROWED chunk only
    // because the stride is coarsened against the LAUNCH width. Narrowing to
    // 512 must not densify checkpoints, and widening cannot outrun the stride
    // because the arch cap is the top of the same ladder.
    const t = testing;
    const stride = effectiveSsmCheckpointStride(256, prefill_chunk_override);
    try t.expect(stride > 0);
    for ([_]usize{ 8192, 4096, 2048, 1024, PREFILL_CHUNK_FLOOR }) |rung| {
        try t.expect(rung <= stride);
        try t.expectEqual(@as(usize, 0), stride % rung);
    }
}

/// The source of ONE named declaration: from `decl` to the `}` that closes it
/// at the declaration's OWN indentation. zig fmt puts that closer alone on its
/// line, so it is the one brace that cannot also appear inside the body, a
/// string literal or a comment. `decl` must START a line — a mid-line
/// occurrence is a MENTION (a doc comment, a call, a scan's own needle
/// literal) and is skipped, so a window never anchors on one.
///
/// The ONE source-window extractor for scans across the codebase;
/// `server.declBody` delegates here rather than keeping a second copy.
///
/// Every ORDERING scan resolves its needles inside a window like this.
/// `indexOfPos` over a whole embedded file searches FORWARD from a production
/// offset, so deleting the very line a scan pins lets the needle fall through
/// to the TEST'S OWN literal further down the file — the assertion then holds
/// on a full revert (the B0c class). A window holding no test cannot do that,
/// whether or not the individual needle happens to be `++`-split. Same for a
/// bare `indexOf(whole_file, literal) != null` presence check: the test's own
/// copy of the literal keeps it green after the production line is gone.
pub fn productionDeclSource(src: []const u8, decl: []const u8) ?[]const u8 {
    var indent: usize = 0;
    while (indent < decl.len and decl[indent] == ' ') indent += 1;
    var closer_buf: [40]u8 = undefined;
    if (indent + 3 > closer_buf.len) return null;
    closer_buf[0] = '\n';
    @memset(closer_buf[1 .. 1 + indent], ' ');
    closer_buf[1 + indent] = '}';
    closer_buf[2 + indent] = '\n';
    const closer = closer_buf[0 .. indent + 3];
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, src, i, decl)) |at| {
        i = at + 1;
        if (at != 0 and src[at - 1] != '\n') continue; // a mention, not a declaration
        const end = std.mem.indexOfPos(u8, src, at, closer) orelse continue;
        return src[at .. end + closer.len];
    }
    return null;
}

/// True when `window` holds no test block at all — the property that makes a
/// scan resolved inside it red on a revert. `generate.zig` has test blocks
/// ABOVE some implementations, so "ends before the first test" is not the
/// question; "contains no test" is.
pub fn windowHasNoTestBlock(window: []const u8) bool {
    return std.mem.indexOf(u8, window, "\n" ++ "test \"") == null;
}

test "a widen is committed only after the interleave tick, a step-down before it" {
    // S17. Attribution and safety want opposite orderings: the probe must run
    // BEFORE the tick so a co-tenant's decode is not read as this prefill's
    // pressure, but a WIDEN decided on pre-tick headroom then forwards into
    // memory the tick allocated and the probe never saw — and a Metal abort
    // cannot be un-decided. So the growth direction is re-priced after the
    // tick and the safe direction is not. Scan-pinned by index, because the
    // ordering is the whole fix.
    //
    // BL-6: needles resolve inside `initWithOptions`'s OWN body, never the
    // whole file. Three of these five (`hk.call(hk.ctx, …`, `cf(widen_ctx.?,`,
    // `if (next_w < cur_chunk)`) appear verbatim in THIS test's bytes, and
    // with a whole-file `indexOfPos` a full revert of S17 — delete the
    // post-tick confirm block, commit the widen pre-tick — left every needle
    // resolving into this test's own literals below, so all five assertions
    // survived the revert. Splitting the literals would fix these three; the
    // window fixes the CLASS, including the next needle somebody adds.
    const t = testing;
    const src = @embedFile("generate.zig");
    const impl = productionDeclSource(src, "    pub fn initWithOptions(") orelse return error.CallSiteMoved;
    // The window is production bytes only — the property the whole fix rests on.
    try t.expect(windowHasNoTestBlock(impl));
    try t.expect(impl.len < src.len);

    const probe = std.mem.indexOf(u8, impl, "hk.call(hk.ctx, ssm_cp_offset + pos, @intCast(cur_chunk)") orelse return error.CallSiteMoved;
    const tick = std.mem.indexOfPos(u8, impl, probe, "options.interleave" ++ "_hook") orelse return error.CallSiteMoved;
    const confirm = std.mem.indexOfPos(u8, impl, probe, "cf(widen_ctx.?,") orelse return error.CallSiteMoved;
    try t.expect(probe < tick);
    try t.expect(tick < confirm);

    // The step-down commits from the pre-tick branch, the widen only from the
    // post-tick one — both through the ONE commit helper.
    const down = std.mem.indexOfPos(u8, impl, probe, "if (next_w < cur_chunk)") orelse return error.CallSiteMoved;
    try t.expect(down < tick);
    try t.expect(std.mem.indexOfPos(u8, impl, down, "commitAdaptive" ++ "Width(").? < tick);
    try t.expect(std.mem.indexOfPos(u8, impl, confirm, "commitAdaptive" ++ "Width(") != null);

    // The confirm itself is a SERVER-side install, and nothing pinned it. Drop
    // `scheduler_mod.prefill_chunk_widen_ok` and
    // `adaptivePrefillWidenStillFits` becomes unreachable: the scheduler side
    // takes its `orelse return false`, every widen is silently refused, and
    // the whole post-tick half above is a no-op with no red test. Read through
    // `serve`'s OWN body — cross-file already, so it cannot fall through to a
    // literal in this test, and the window stops it falling through to a
    // server-side test's either.
    const srv = @embedFile("server.zig");
    const serve_body = productionDeclSource(srv, "pub fn serve(") orelse return error.CallSiteMoved;
    try t.expect(windowHasNoTestBlock(serve_body));
    try t.expect(std.mem.indexOf(u8, serve_body, "scheduler_mod.prefill_chunk_widen_ok = &adaptivePrefillWiden" ++ "StillFits;") != null);
    // ...and the scheduler consults it rather than assuming a widen fits.
    const sched = @embedFile("scheduler.zig");
    try t.expect(std.mem.indexOf(u8, sched, "const ok = prefill_chunk_widen" ++ "_ok orelse return false;") != null);
}

test "commitAdaptiveWidth: the summary follows the running width from both commit sites" {
    // The widen and the step-down commit from different points in the loop
    // since S17, so the counters live in ONE function or they drift from
    // `cur_chunk` the first time somebody edits one branch.
    const t = testing;
    var cur: usize = 2048;
    var st: AdaptiveWidthState = .{ .width_min = 2048, .width_max = 2048 };
    commitAdaptiveWidth(&cur, &st, 1024);
    try t.expectEqual(@as(usize, 1024), cur);
    try t.expectEqual(@as(u32, 1), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 2048), st.width_max);
    commitAdaptiveWidth(&cur, &st, 4096);
    try t.expectEqual(@as(usize, 4096), cur);
    try t.expectEqual(@as(u32, 2), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 4096), st.width_max);
}

test "the per-chunk width is probed after the cache clear and before the interleave tick" {
    // Both placements are load-bearing and neither is visible from a unit
    // test, so they are scan-pinned. Probing BEFORE the clear reads the
    // chunk's own peak and narrows on memory that is already gone; probing
    // AFTER the interleave tick reads a co-tenant's decode allocations as this
    // prefill's pressure.
    //
    // BL-6 sibling: resolved inside `initWithOptions`'s own body. `clear`'s
    // needle is unsplit and appears verbatim in this test's bytes, so with a
    // whole-file scan a deleted trace line would re-anchor the ordering on the
    // test's own literal. The split needles made it fail closed rather than
    // green, but the window is what makes that structural.
    const t = testing;
    const src = @embedFile("generate.zig");
    const impl = productionDeclSource(src, "    pub fn initWithOptions(") orelse return error.CallSiteMoved;
    try t.expect(windowHasNoTestBlock(impl));
    // Anchored on the eval-trace line, not on the clear plus its next line:
    // the g1 loop-order fix (B0b) inserts `try mlx.checkError()` immediately
    // after the clear, and a needle spanning both lines would go red for that
    // rather than for anything this test is about.
    const clear = std.mem.indexOf(u8, impl, "if (trace_enabled) eval_ns +=") orelse return error.CallSiteMoved;
    const probe = std.mem.indexOfPos(u8, impl, clear, "options.chunk_width" ++ "_hook") orelse return error.CallSiteMoved;
    const tick = std.mem.indexOfPos(u8, impl, clear, "options.interleave" ++ "_hook") orelse return error.CallSiteMoved;
    try t.expect(probe < tick);

    // The unchunked vision arm forwards the whole prompt: there is no next
    // chunk to size, and the guard bills the real width already.
    try t.expect(std.mem.indexOf(u8, impl, "const adapt_chunked = " ++ "!(has_vision and !vision_chunked);") != null);
    // The stride is still coarsened against the LAUNCH width, never the
    // per-request pin and never the per-chunk width.
    try t.expect(std.mem.indexOf(u8, impl, "effectiveSsmCheckpoint" ++ "Stride(@intCast(options.ssm_checkpoint_stride), @max(PREFILL_CHUNK, prefill_chunk_override))") != null);
    // And the first chunk always runs the width admission billed.
    try t.expect(std.mem.indexOf(u8, impl, "var cur_chunk: usize = " ++ "default_chunk;") != null);
}

test "tailMergeMax: the tail a chunk may absorb scales with the width" {
    // S18. The flat 512 was justified as "~6% at 8192". At the ladder floor —
    // which the per-chunk width now reaches by design — it is +100% of the
    // transient the step-down just bought.
    const t = testing;
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMax(8192)); // no-op where it was justified
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMax(4096));
    try t.expectEqual(@as(usize, 256), tailMergeMax(2048));
    try t.expectEqual(@as(usize, 128), tailMergeMax(1024));
    try t.expectEqual(@as(usize, 64), tailMergeMax(PREFILL_CHUNK_FLOOR));
    // Never 0: a 0 merge bound is fine, but a 0 divisor is a trap for the
    // next caller, and `nextChunkEnd` must stay total.
    try t.expectEqual(@as(usize, 1), tailMergeMax(1));
    // The bound it exists to keep: never more than an eighth of the width.
    for ([_]usize{ 8192, 4096, 2048, 1024, PREFILL_CHUNK_FLOOR }) |w| {
        try t.expect(tailMergeMax(w) * 8 <= w);
    }

    // And the loop honours it: a 300-token tail merges at 8192 and does NOT
    // at the floor, where it would double the chunk.
    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false, 0, 0, true));
    try t.expectEqual(@as(usize, 512), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false, 0, 0, true));
}

test "the scaled tail-merge bound is gated on the per-chunk adaptive width" {
    // BL-5. S18 replaced the flat `TAIL_MERGE_MAX` inside `nextChunkEnd` with
    // the width-scaled bound, and `nextChunkEnd` has no arch parameter —
    // so EVERY arch whose resolved chunk is under 4096 started merging a
    // smaller tail, i.e. running an extra chunk on prompts that used to run
    // one. That reaches `resolvePrefillChunk`'s machine ladder (a 27B on a
    // 16 GB Mac lands at 512), `boundedPrefillChunk`'s score-budget floor and
    // its composed-causal 2048 cap (gemma4, qwen3_5/3_6, muse_glimmer,
    // deepseek_v4), and any `--prefill-chunk`/`MLX_SERVE_PREFILL_CHUNK` under
    // 4096 anywhere. Chunk boundaries are not byte-stable, so that was an
    // unmeasured behaviour change on archs the feature never touched.
    const t = testing;

    // The bound itself, both arms.
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(PREFILL_CHUNK_FLOOR, false));
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(1024, false));
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(2048, false));
    try t.expectEqual(@as(usize, 64), tailMergeMaxFor(PREFILL_CHUNK_FLOOR, true));
    try t.expectEqual(@as(usize, 256), tailMergeMaxFor(2048, true));
    // Where the flat 512 was justified the two arms agree by construction.
    try t.expectEqual(tailMergeMaxFor(4096, false), tailMergeMaxFor(4096, true));
    try t.expectEqual(tailMergeMaxFor(8192, false), tailMergeMaxFor(8192, true));

    // ...and the loop honours the gate. A 300-token tail at the ladder's 512
    // rung: a non-adaptive arch merges it, exactly as it did before S18.
    try t.expectEqual(@as(usize, 812), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false, 0, 0, false));
    // The adaptive arm does not — 300 is +59% of the transient the step-down
    // just bought.
    try t.expectEqual(@as(usize, 512), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false, 0, 0, true));
    // Same split one rung up, and none at 4096/8192.
    try t.expectEqual(@as(usize, 1324), nextChunkEnd(0, 1324, 1024, false, 0, 0, false));
    try t.expectEqual(@as(usize, 1024), nextChunkEnd(0, 1324, 1024, false, 0, 0, true));
    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false, 0, 0, false));
    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false, 0, 0, true));

    // Chunk COUNT is what the archs above actually pay: one chunk, not two.
    try t.expectEqual(@as(usize, 1), prefillChunkCount(812, PREFILL_CHUNK_FLOOR, false, 0, 0, false));
    try t.expectEqual(@as(usize, 2), prefillChunkCount(812, PREFILL_CHUNK_FLOOR, false, 0, 0, true));

    // The gate the loop reads is the ARCH predicate, delivered per model on
    // `InitOptions.adaptive_chunk_width` — NEVER the presence of
    // `chunk_width_hook`, which `serve` installs unconditionally and
    // process-wide, so it is non-null on every arch (audit B-A2). The wiring
    // itself is pinned in server.zig, "the tail-merge gate reads the ARCH, not
    // the installed hook"; this half pins what the loop asks. Scanned inside
    // the implementation's own body so the needles cannot fall through to this
    // test's bytes.
    const src = @embedFile("generate.zig");
    const impl = productionDeclSource(src, "    pub fn initWithOptions(") orelse return error.CallSiteMoved;
    try t.expect(windowHasNoTestBlock(impl));
    try t.expect(std.mem.indexOf(u8, impl, "const width_is_adaptive = " ++ "adapt_chunked and options.adaptive_chunk_width;") != null);
    try t.expect(std.mem.indexOf(u8, impl, "adapt_chunked and options.chunk_width" ++ "_hook != null") == null);
    try t.expect(std.mem.indexOf(u8, impl, "ssm_cp_offset, width" ++ "_is_adaptive);") != null);
    // And the scaling is reachable ONLY through the gated helper.
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, src, "tailMergeMax(default" ++ "_chunk)"));
}

test "nextChunkEnd: a tiny trailing remainder merges into the last chunk" {
    // A chat-templated prompt often lands a token or two past the chunk size
    // (8192-target prompts tokenize to 8193). A 1-token trailing chunk pays a
    // FULL graph + eval-barrier + cache-clear for one token — pure overhead.
    // Without checkpoint alignment, remainders under the merge floor extend
    // the current chunk instead.
    try testing.expectEqual(@as(usize, 8193), nextChunkEnd(0, 8193, 8192, false, 0, 0, false));
    // A substantial remainder stays its own chunk.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 8192 + 600, 8192, false, 0, 0, false));
    // Mid-prompt chunks are untouched.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 16385, 8192, false, 0, 0, false));
    try testing.expectEqual(@as(usize, 16385), nextChunkEnd(8192, 16385, 8192, false, 0, 0, false));
    // With SSM-checkpoint alignment active, a tiny tail STILL merges: the old
    // 1-token trailing chunk existed only to lay a snapshot one token before
    // the always-on end-of-prompt snapshot — pure overhead. A boundary strictly
    // INSIDE the chunk still wins over merging (next case).
    try testing.expectEqual(@as(usize, 8193), nextChunkEnd(0, 8193, 8192, true, 8192, 0, false));
    try testing.expectEqual(@as(usize, 4096), nextChunkEnd(0, 8193, 8192, true, 4096, 0, false));
}

test "prefillChunkCount: SSM-checkpoint stride controls cold-prefill chunking" {
    const PREFILL_CHUNK: usize = 8192;
    // Non-hybrid (or checkpointing off): a sub-PREFILL_CHUNK prompt is ONE chunk.
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(851, PREFILL_CHUNK, false, 0, 0, false));
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(8000, PREFILL_CHUNK, false, 0, 0, false));
    // Tail merge: one token past a chunk boundary is still ONE chunk.
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(8193, PREFILL_CHUNK, false, 0, 0, false));
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(16385, PREFILL_CHUNK, false, 0, 0, false));
    // Mechanically, a raw fine stride still splits an 851-token prefill into 4
    // chunks (851 spans boundaries 256/512/768) — which is why
    // effectiveSsmCheckpointStride coarsens every stride to the prefill chunk:
    // per-chunk costs (expert re-streaming on MoE, sub-dq-gemm-floor GEMMs +
    // fixed overhead everywhere) taxed cold prefill 17-25%.
    try testing.expectEqual(@as(usize, 4), prefillChunkCount(851, PREFILL_CHUNK, true, 256, 0, false));
    // Boundary alignment is ABSOLUTE (warm path passes an offset): a tail-only
    // prefill starting mid-sequence still snaps to global strides. offset=2000,
    // prefix tail of 200 (abs 2000..2200), stride 256 -> boundary 2048/2304? only
    // 2048 falls inside (2000..2200) -> 2 chunks.
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(200, PREFILL_CHUNK, true, 256, 2000, false));
}

test "boundedPrefillChunk: fused head dims and short contexts keep the base chunk" {
    // head_dim <= 128 rides MLX's fused SDPA — no materialized scores, no cap,
    // at ANY context length.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 16, 1_000_000, true, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 64, 32, 1_000_000, true, false));
    // hd 256 but short context: 16 heads x 8192 ctx x 8192 chunk x 2B
    // = 2 GiB scores, inside the 4 GiB budget -> full chunk kept. This is the
    // fleet-protection property: every Gemma-4 / Qwen3.5/3.6 checkpoint ships
    // head_dim 256, so typical prompts must keep full prefill throughput.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true, false));
    // Degenerate inputs never cap.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 0, 100_000, true, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 0, true, false));
}

test "boundedPrefillChunk: caps hd-256 long context even with the fused kernel active" {
    // The msv_attn_p256 kernel removes the SCORE transient, but a big chunk
    // still scales the MoE-gather / KV-concat transients — measured +22 GB
    // peak for +3% speed at a 99K prompt. The cap deliberately ignores
    // prefillHeadDimFused (see the fn doc); pin that with the override ON.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true, false));
}

test "boundedPrefillChunk: long context shrinks to the scores budget, floored and rounded" {
    // gemma-4-26B geometry (16 heads): budget/(16*ctx*2) …
    // ctx 32768 -> exactly 4096.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 16, 32768, true, false));
    // ctx 100000 -> raw 1342, rounded down to the 512 grain -> 1024.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true, false));
    // ctx 262144 (the PR-#69 255K case) -> 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 16, 262_144, true, false));
    // Qwen3.6-27B geometry (24 heads) at 262144: raw 341 -> floor 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 24, 262_144, true, false));
    // e4b geometry (8 heads) at 131072: exactly 2048.
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 8, 131_072, true, false));
}

test "boundedPrefillChunk: a 192-wide MLA score is budgeted, and the hd-256 policies stay hd-256" {
    // A hybrid MLA arch can declare head_dim 128 (its value width) so it fell
    // through the "fused SDPA covers it" early-out — while its MLA scores at
    // qk width 192 on the composed path. 32 heads.
    //
    // Short prompts keep the full chunk: 32 * 8192 * 8192 * 2B = 4 GiB exactly.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 192, 32, 8192, false, true));
    // Then it halves with the prompt, holding one score tensor at the budget.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 192, 32, 16384, false, true));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 32768, false, true));
    // The 38201-token prompt that measured a +28.8 GB peak at chunk 8192:
    // 4 GiB / (32 * 38201 * 2) = 1757 -> floored to the 512 grain.
    try testing.expectEqual(@as(usize, 1536), boundedPrefillChunk(8192, 192, 32, 38_201, false, true));
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 192, 32, 262_144, false, true));

    // The two hd-256-measured policies must NOT adopt this arch:
    // - the fused-kernel branch (no score tensor) is hd-256-only,
    // - the 2048 composed cap was tuned on a 27B's own prefill ladder.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    // With the fused override on, a real hd-256 MoE takes the 4096 branch...
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 32, 8192, false, true));
    // ...and the 192-wide arch still gets its honest full chunk at the same shape.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 192, 32, 8192, false, true));
}

test "MTP history window: threshold gate and chunk membership" {
    // Below/at the 16384 threshold the window never engages — behavior (and
    // temp-0 output) stays byte-identical to full-history capture.
    try testing.expectEqual(@as(usize, 0), effectiveMtpHistoryWindow(1000, 8192));
    try testing.expectEqual(@as(usize, 0), effectiveMtpHistoryWindow(16384, 8192));
    try testing.expectEqual(@as(usize, 8192), effectiveMtpHistoryWindow(16385, 8192));
    try testing.expectEqual(@as(usize, 8192), effectiveMtpHistoryWindow(65536, 8192));
    // 0 = full history at any length (the --mtp-history-window 0 escape).
    try testing.expectEqual(@as(usize, 0), effectiveMtpHistoryWindow(65536, 0));

    // Chunk membership at prefix 32768, window 8192: the window starts at
    // 24576. Chunks entirely before it skip capture; the boundary chunk
    // (ending past 24576) captures WHOLE.
    try testing.expect(!chunkNeedsMtpHistory(0, 8192, 32768, 8192));
    try testing.expect(!chunkNeedsMtpHistory(16384, 24576, 32768, 8192));
    try testing.expect(chunkNeedsMtpHistory(24576, 32768, 32768, 8192));
    try testing.expect(chunkNeedsMtpHistory(20000, 24577, 32768, 8192));
    // Zero window: every chunk captures.
    try testing.expect(chunkNeedsMtpHistory(0, 8192, 32768, 0));
    // Window >= prefix degenerates to full capture (no underflow).
    try testing.expect(chunkNeedsMtpHistory(0, 512, 4096, 8192));
}

test "boundedPrefillChunk: never raises a caller-lowered base chunk" {
    // --prefill-chunk 1024 with headroom for 4096: the explicit lower value wins.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 16, 32768, true, false));
    // Even the floor never raises a tiny explicit base.
    try testing.expectEqual(@as(usize, 256), boundedPrefillChunk(256, 256, 16, 262_144, true, false));
}

test "boundedPrefillChunk: fused-causal (default) non-sliding hd-256 — MoE caps at 4096, dense keeps the full chunk" {
    // With the causal arm FUSED (default since the budgeted-dispatch flip) no
    // score tensor exists, so the scores-budget formula is moot for the
    // qwen3_5/3_6 class — and its old shrink to 1024 at 64K starved the
    // dequant+GEMM qmm route (engages at M >= 2048): the 64K rung was the
    // ladder's weakest for exactly this reason.
    std.debug.assert(transformer_mod.fused256_override == null);
    // DENSE hybrids (Qwen3.6-27B class): no expert-gather transients, and a
    // full-size chunk halves per-chunk dequant sweeps — chunk 8192 measured
    // +1.4% over 4096 at the 8K rung (M4 Max, 2026-07-30), flat at 32K.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 8192, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 65536, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, false));
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (gemma-26B@99K: +3% speed for +22 GB peak is a bad trade).
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 8192, false, true));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 140_000, false, true));
    // Never raises a caller-lowered base.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 24, 8192, false, false));
    // Sliding-band archs (gemma: fused band kernel wants big chunks) keep
    // the formula-only policy.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true, false));
    // Fused head dims never cap regardless of arch.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 24, 1_000_000, false, false));
}

test "boundedPrefillChunk: composed-causal (kill switch) keeps the 2048 cap + score formula" {
    // MLX_SERVE_FUSED_256_CAUSAL=0 restores composed causal, where SMALLER
    // chunks measured faster on the 27B ladder (2026-07-12, M4 Max): 8K
    // prompt 225 -> 235.8 tok/s at chunk 2048 (peak 28.9 -> 19.8 GB).
    // Chunking IS block-level causal skipping for composed attention, and
    // the score transient shrinks with it.
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 8192, false, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 32768, false, false));
    // The scores-budget formula still wins BELOW the cap: 64K on 24 heads
    // yields 1024 (measured better than 2048 there: 186 vs 182.3 tok/s).
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 24, 65536, false, false));
}

test "ssmSnapshotBackoff: engages only under checkpointing and past the backoff length" {
    // No checkpointing (pure-attention archs, stride 0, vision): zero — the
    // final forward stays the classic 1-token logits pass.
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(false, 8192));
    // Short prompts: nothing to back off (loop must keep >= 1 token).
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(true, SSM_SNAPSHOT_BACKOFF));
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(true, 1));
    // Checkpointing + long prompt: the always-on snapshot lands backoff
    // tokens before the prompt end, where the next turn's prefix match can
    // reach it (template generation-suffix divergence class).
    try testing.expectEqual(SSM_SNAPSHOT_BACKOFF, ssmSnapshotBackoff(true, 8192));
    try testing.expectEqual(SSM_SNAPSHOT_BACKOFF, ssmSnapshotBackoff(true, SSM_SNAPSHOT_BACKOFF + 1));
    // The tail forward must stay UNDER the prefill-eval-cadence threshold
    // (seq >= 32 turns it into a "prefill" costing ~450ms of eval bubbles):
    // tail = backoff + 1 <= 31.
    try testing.expect(SSM_SNAPSHOT_BACKOFF + 1 < 32);
}

test "effectiveSsmCheckpointStride: checkpointing never sub-divides the prefill chunk (dense AND MoE)" {
    const PREFILL_CHUNK: usize = 8192;
    // Disabled stays disabled.
    try testing.expectEqual(@as(usize, 0), effectiveSsmCheckpointStride(0, PREFILL_CHUNK));
    // Sub-chunk strides coarsen on ALL archs: they push every projection under
    // prefillDqGemm's M>=2048 floor (slow small-M qmm) and multiply per-chunk
    // fixed costs. Measured on Qwen3.6-27B dense (M4 Max, 2026-07-30): stride
    // 256 = 33 chunks at 8K = 211 tok/s vs 254 coarse.
    try testing.expectEqual(@as(usize, 8192), effectiveSsmCheckpointStride(256, PREFILL_CHUNK));
    // A larger explicit stride is respected (never shrunk).
    try testing.expectEqual(@as(usize, 16384), effectiveSsmCheckpointStride(16384, PREFILL_CHUNK));
    // End-to-end: an 851-tok prefill is 1 chunk (was 4 at the raw 256 stride
    // on dense hybrids — the llm_context_benchmarks small-prompt regression).
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(851, PREFILL_CHUNK, true, effectiveSsmCheckpointStride(256, PREFILL_CHUNK), 0, false));
    // An 8K prefill splits only at the (memory-bound) chunk size, never
    // finer: 2 chunks at chunk 4096, not 33.
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(8238, 4096, true, effectiveSsmCheckpointStride(256, PREFILL_CHUNK), 0, false));
}

test "vision prefill checkpoints SSM state only when it chunks like text" {
    const prefix_len: usize = 1587;
    const checkpoint_stride: u32 = 256;

    // Chunked vision (the default) checkpoints like text — a hybrid's image
    // turn is otherwise a guaranteed hot-cache miss (no checkpoint <= match).
    vision_chunked_cached = true;
    defer vision_chunked_cached = null;
    try testing.expect(shouldCheckpointSsmPrefill(checkpoint_stride, true, true));
    // The whole-prompt kill-switch arm has no chunk boundaries to snapshot at.
    vision_chunked_cached = false;
    const vision_checkpoints = shouldCheckpointSsmPrefill(checkpoint_stride, true, true);
    try testing.expect(!vision_checkpoints);
    try testing.expectEqual(
        prefix_len,
        nextChunkEnd(0, prefix_len, prefix_len, vision_checkpoints, @intCast(checkpoint_stride), 0, false),
    );

    const text_checkpoints = shouldCheckpointSsmPrefill(checkpoint_stride, true, false);
    try testing.expect(text_checkpoints);
    try testing.expectEqual(
        @as(usize, checkpoint_stride),
        nextChunkEnd(0, prefix_len, prefix_len, text_checkpoints, @intCast(checkpoint_stride), 0, false),
    );
}

fn mtpEvTestGenerator() Generator {
    var g: Generator = undefined;
    g.mtp_depth = Generator.MTP_ADAPTIVE_DEFAULT_CAP;
    g.mtp_depth_current = 1;
    g.mtp_window_drafted = @splat(0);
    g.mtp_window_accepted = @splat(0);
    g.mtp_window_idx = 0;
    g.mtp_rounds_since_switch = 0;
    g.mtp_promote_cooldown = 0;
    g.mtp_ev_accept = @splat(Generator.MTP_EV_PRIOR);
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS;
    g.mtp_ev_m_lo_prev = 1;
    g.mtp_ev_costs = Generator.MTP_EV_DEFAULT_COSTS;
    g.spec_disabled_runtime = false;
    return g;
}

test "mtpDraftSamplingFor: sharpened fixed proposal for stochastic targets, greedy stays greedy" {
    // Stochastic target: drafts sample from the FIXED sharpened distribution
    // (temp 0.6 / top_p 0.95 / top_k 20 — oMLX Lightning's _DRAFT_SAMPLER_*
    // constants; matched-temp drafting collapses on high-entropy content).
    const target = SamplingParams{ .temperature = 1.0, .top_p = 1.0, .top_k = 0, .repeat_penalty = 1.1 };
    const d = Generator.mtpDraftSamplingFor(target, false);
    try testing.expectEqual(@as(f32, 0.6), d.temperature);
    try testing.expectEqual(@as(f32, 0.95), d.top_p);
    try testing.expectEqual(@as(u32, 20), d.top_k);
    // Non-sampler fields ride through untouched.
    try testing.expectEqual(@as(f32, 1.1), d.repeat_penalty);

    // Greedy target keeps greedy drafts (the temp-0 identity contract).
    const greedy = SamplingParams{ .temperature = 0.0 };
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(greedy, false).temperature);
    const near_greedy = SamplingParams{ .temperature = 0.005 };
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(near_greedy, false).temperature);

    // Explicit greedy override (MLX_SERVE_MTP_DRAFT_GREEDY=1) wins.
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(target, true).temperature);
}

test "filteredProbsBlock: every draft row is the SAME density the sample is drawn from" {
    // Exactness rests on q being the true proposal: the draft must be drawn
    // from byte-for-byte the row handed to `specAcceptProb`. The way that
    // breaks silently is filtering the two differently — so pin that a row
    // is a normalized distribution supported ONLY on the kept set, and that
    // `mlx_random_categorical` over log(q) can never land outside it.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 8;
    const M: c_int = 3;
    // Three rows with deliberately different argmaxes, so a row-blind
    // implementation (broadcasting one row's mask) fails.
    const raw = [_]f32{
        5.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 0.0,
        0.0, 9.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0,
    };
    const shape = [_]c_int{ 1, M, V };
    const logits = mlx.mlx_array_new_data(&raw, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    var sampling = SamplingParams{};
    sampling.temperature = 0.8;
    sampling.top_k = 2;
    const q = try filteredProbsBlock(logits, sampling, s);
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_array_eval(q));
    const qsh = mlx.getShape(q);
    try testing.expectEqual(M, qsh[0]);
    try testing.expectEqual(V, qsh[1]);

    const data = mlx.mlx_array_data_float32(q) orelse return error.MlxArrayDataNull;
    // Per row: sums to 1, and exactly top_k entries carry mass.
    const want_support = [_][2]usize{ .{ 0, 1 }, .{ 5, 6 }, .{ 1, 3 } };
    for (0..@intCast(M)) |r| {
        var sum: f32 = 0;
        var nonzero: usize = 0;
        for (0..@intCast(V)) |c| {
            const v = data[r * @as(usize, @intCast(V)) + c];
            sum += v;
            if (v > 0) nonzero += 1;
        }
        try testing.expect(@abs(sum - 1.0) < 1e-5);
        try testing.expectEqual(@as(usize, 2), nonzero);
        for (want_support[r]) |c| {
            try testing.expect(data[r * @as(usize, @intCast(V)) + c] > 0);
        }
    }

    // Draw the way nextDflash does and assert every sample is in q's support.
    var logq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logq);
    try mlx.check(mlx.mlx_log(&logq, q, s));
    var draws: usize = 0;
    while (draws < 24) : (draws += 1) {
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        var sampled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_astype(&ids, sampled, .int32, s));
        try mlx.check(mlx.mlx_array_eval(ids));
        const idd = mlx.mlx_array_data_int32(ids) orelse return error.MlxArrayDataNull;
        for (0..@intCast(M)) |r| {
            const tok: usize = @intCast(idd[r]);
            // q(draft) > 0 is what keeps the accept ratio finite.
            try testing.expect(data[r * @as(usize, @intCast(V)) + tok] > 0);
            // And `sliceProbRow` must hand back THAT row, not another.
            const row = try sliceProbRow(q, @intCast(r), s);
            defer _ = mlx.mlx_array_free(row);
            const got = try probAt(row, @intCast(tok), s);
            try testing.expectApproxEqAbs(data[r * @as(usize, @intCast(V)) + tok], got, 1e-6);
        }
    }
    _ = allocator;
}

test "specAcceptProb: full Leviathan ratio, q-clamped" {
    // p <= q: accept with p/q.
    try testing.expect(@abs(Generator.specAcceptProb(0.2, 0.4) - 0.5) < 1e-6);
    // p > q: always accept.
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.4, 0.2));
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.4, 0.4));
    // Degenerate q underflow never divides by zero.
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.5, 0.0));
}

test "spec sampling exactness: draft-from-q + ratio-accept + residual reproduces target p (toy vocab)" {
    // Host-level simulation of the exact per-position algorithm the MTP
    // stochastic round runs: draft ~ q, accept with min(1, p/q), on reject
    // sample from normalize(max(p - q, 0)). The output distribution must
    // equal p (Leviathan/Chen) — this is the invariant the one-hot rule
    // broke for sampled proposals.
    const p = [_]f64{ 0.1, 0.2, 0.3, 0.4 };
    const q = [_]f64{ 0.4, 0.3, 0.2, 0.1 };
    var residual: [4]f64 = undefined;
    var res_sum: f64 = 0;
    for (0..4) |i| {
        residual[i] = @max(p[i] - q[i], 0);
        res_sum += residual[i];
    }
    for (&residual) |*r| r.* /= res_sum;

    var prng = std.Random.DefaultPrng.init(0x5A3E);
    const rnd = prng.random();
    var counts = [_]u64{ 0, 0, 0, 0 };
    const N: usize = 400_000;
    for (0..N) |_| {
        // draft ~ q
        var u = rnd.float(f64);
        var draft: usize = 0;
        var acc: f64 = 0;
        for (q, 0..) |qi, i| {
            acc += qi;
            if (u < acc) {
                draft = i;
                break;
            }
        }
        const a = Generator.specAcceptProb(@floatCast(p[draft]), @floatCast(q[draft]));
        if (rnd.float(f32) < a) {
            counts[draft] += 1;
        } else {
            u = rnd.float(f64);
            acc = 0;
            var res_tok: usize = 3;
            for (residual, 0..) |ri, i| {
                acc += ri;
                if (u < acc) {
                    res_tok = i;
                    break;
                }
            }
            counts[res_tok] += 1;
        }
    }
    for (0..4) |i| {
        const freq = @as(f64, @floatFromInt(counts[i])) / @as(f64, @floatFromInt(N));
        try testing.expect(@abs(freq - p[i]) < 0.01);
    }
}

test "mtpNextDepth: adaptive depth policy transitions" {
    const configured: u32 = 3;
    // Hot at configured depth: stay.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.9));
    // Sagging at depth > 1: step down (one level at a time). The demote
    // floor is 0.40 under capture-based rollback (a rejected draft costs
    // only its own head pass, not a trunk re-forward).
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(3, configured, 0.35));
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(2, configured, 0.30));
    // Mid-band (0.40..0.60): hold.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.48));
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(2, configured, 0.55));
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(1, configured, 0.55));
    // Hot below configured depth: promote (band top is 0.60).
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(1, configured, 0.73));
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(2, configured, 0.70));
    // Never exceeds configured.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.99));
    // Depth 1 at 0.40: speculation still pays (~+27% over AR at current
    // round costs — the disable floor sits at the measured breakeven, not
    // at "acceptance feels low"). The old 0.50 floor DISABLED here and
    // cratered the oQ4e 16K/32K ladder cells to bare AR (24-26 tok/s).
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(1, configured, 0.40));
    // Depth 1 below the true breakeven (+margin): disable (0).
    try testing.expectEqual(@as(u32, 0), Generator.mtpNextDepth(1, configured, 0.15));
    // Demote-before-disable: a terrible rate at depth 2 still goes through 1.
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(2, configured, 0.10));
}

test "mtpDepthDecision: confidence gates on disable, promote, cooldown" {
    const W = Generator.MTP_DEPTH_WINDOW;
    // Disable needs a FULL window of evidence; small samples hold at depth 1.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.15, 5, false));
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.15, W - 1, false));
    try testing.expectEqual(@as(u32, 0), Generator.mtpDepthDecision(1, 3, 0.15, W, false));
    // Rates the old 0.50 floor killed keep speculating at any window size.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.40, W, false));
    // Promote needs >= 8 rounds AND no active cooldown.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.95, 7, false));
    try testing.expectEqual(@as(u32, 2), Generator.mtpDepthDecision(1, 3, 0.95, 8, false));
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.95, 8, true));
    // Demote reacts on a small sample, even during cooldown.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(2, 3, 0.30, 5, true));
}

test "MTP EV seed round-trips on the qwen4 head; a fresh or absent head reads null" {
    // The in-checkpoint head stores the seed on `Qwen4Mtp` — per loaded model,
    // exactly like the sidecar's `MtpModel.ev_seed_*`. Before this it was a
    // null/no-op stub, so every qwen4 request re-warmed the controller from
    // scratch. Only the two seed fields are touched here; the rest of the head
    // (and of the Transformer) is never read on this path.
    var t: Transformer = undefined;
    t.qwen4_mtp = null;
    const ref = MtpHeadRef{ .qwen4 = &t };

    // `--no-mtp`: no head to seed, and setting one must not fault.
    try testing.expect(ref.evSeed() == null);
    var zeros: [mtp_mod.MAX_DEPTH]f32 = @splat(0.5);
    ref.setEvSeed(zeros, 4);
    try testing.expect(ref.evSeed() == null);

    var head: transformer_mod.Qwen4Mtp = undefined;
    head.ev_seed_accept = null;
    head.ev_seed_m_lo = 1;
    t.qwen4_mtp = head;
    // A head that has never published reads null — not a zeroed surface.
    try testing.expect(ref.evSeed() == null);

    var accept: [mtp_mod.MAX_DEPTH]f32 = @splat(0.0);
    accept[0] = 0.91;
    accept[1] = 0.80;
    accept[2] = 0.68;
    ref.setEvSeed(accept, 5);
    const got = ref.evSeed() orelse return error.SeedMissing;
    try testing.expectEqual(@as(u32, 5), got.m_lo);
    try testing.expectEqualSlices(f32, &accept, &got.accept);

    // Overwrite, not accumulate: the LAST healthy request wins.
    zeros[0] = 0.10;
    ref.setEvSeed(zeros, 2);
    const again = ref.evSeed() orelse return error.SeedMissing;
    try testing.expectEqual(@as(u32, 2), again.m_lo);
    try testing.expectEqualSlices(f32, &zeros, &again.accept);
}

test "MTP EV seed defaults on and explicit zero disables" {
    try testing.expect(Generator.mtpEvSeedEnabledFromEnv(null));
    try testing.expect(Generator.mtpEvSeedEnabledFromEnv(""));
    try testing.expect(Generator.mtpEvSeedEnabledFromEnv("1"));
    try testing.expect(!Generator.mtpEvSeedEnabledFromEnv("0"));
    try testing.expect(!Generator.mtpEvSeedEnabledFromEnv("0-disabled"));
}

test "MTP early dispatch defaults on and explicit zero disables" {
    try testing.expect(Generator.mtpEarlyDispatchEnabledFromEnv(null));
    try testing.expect(Generator.mtpEarlyDispatchEnabledFromEnv(""));
    try testing.expect(Generator.mtpEarlyDispatchEnabledFromEnv("1"));
    try testing.expect(!Generator.mtpEarlyDispatchEnabledFromEnv("0"));
    try testing.expect(!Generator.mtpEarlyDispatchEnabledFromEnv("0-disabled"));
}

test "MTP cross-round pre-draft defaults on and explicit zero disables" {
    try testing.expect(Generator.mtpPredraftEnabledFromEnv(null));
    try testing.expect(Generator.mtpPredraftEnabledFromEnv(""));
    try testing.expect(Generator.mtpPredraftEnabledFromEnv("1"));
    try testing.expect(!Generator.mtpPredraftEnabledFromEnv("0"));
    try testing.expect(!Generator.mtpPredraftEnabledFromEnv("0-disabled"));
    // The tail-side pre-draft bills its own trace phase; the round total
    // stays the honest sum of all phases.
    var t = Generator.MtpTrace{};
    t.add(.predraft, 2_000_000);
    t.add(.eval, 8_000_000);
    _ = t.endRound(1, 1, false);
    try testing.expectApproxEqAbs(@as(f64, 2.0), t.avgMs(.predraft), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 10.0), t.totalAvgMs(), 1e-9);
}

test "mtpDepthCapFor: auto cap follows the selected cost profile; explicit always wins" {
    // 0 = auto (--mtp-depth not passed).
    // .generic's auto cap is per-silicon, so the chip must be injected here
    // or the assertion is a property of whichever Mac runs the suite.
    try testing.expectEqual(Generator.MTP_ADAPTIVE_DEFAULT_CAP, Generator.mtpDepthCapForProfileChip(0, true, .generic, "Apple M4 Max"));
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapForProfileChip(0, true, .generic, ""));
    try testing.expectEqual(@as(u32, 4), Generator.mtpDepthCapForProfileChip(0, true, .generic, "Apple M1 Pro"));
    // An explicit depth still outranks the table on a measured chip.
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfileChip(8, true, .generic, "Apple M1 Pro"));
    try testing.expectEqual(mtp_mod.DEFAULT_DEPTH, Generator.mtpDepthCapForProfileChip(0, false, .generic, "Apple M1 Pro"));
    for ([_]mtp_mod.MtpCostProfile{ .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q4_gs64, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64 }) |profile| {
        try testing.expectEqual(Generator.MTP_ADAPTIVE_NAX_CAP, Generator.mtpDepthCapForProfile(0, true, profile));
        try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfile(0, true, profile));
    }
    // qwen4's calibrated surface keeps the DEFAULT cap: no NAX region exists
    // to justify opening depths 7-8 on that arch.
    try testing.expectEqual(Generator.MTP_ADAPTIVE_DEFAULT_CAP, Generator.mtpDepthCapForProfile(0, true, .g17_nax_qwen4_q4_gs64));
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapForProfile(0, true, .g17_nax_qwen4_q4_gs64));
    for ([_]mtp_mod.MtpCostProfile{ .generic, .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q4_gs64, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64, .g17_nax_qwen4_q4_gs64 }) |profile| {
        try testing.expectEqual(mtp_mod.DEFAULT_DEPTH, Generator.mtpDepthCapForProfile(0, false, profile));
    }
    // Explicit values ignore both controller mode and profile, and remain
    // clamped to [1, MAX_DEPTH].
    try testing.expectEqual(@as(u32, 5), Generator.mtpDepthCapForProfile(5, true, .generic));
    try testing.expectEqual(@as(u32, 5), Generator.mtpDepthCapForProfile(5, false, .g17_nax_q8_gs32));
    try testing.expectEqual(@as(u32, 7), Generator.mtpDepthCapForProfile(7, true, .generic));
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfile(8, true, .generic));
    try testing.expectEqual(@as(u32, 2), Generator.mtpDepthCapForProfile(2, true, .g17_nax_q4_gs32));
    try testing.expectEqual(mtp_mod.MAX_DEPTH, Generator.mtpDepthCapForProfile(12, true, .generic));

    // The original boolean helpers remain source-compatible and map true to
    // the pre-existing q8 profile.
    // The live resolver must NAME the row it applied, once. A silicon fence
    // nobody can see in the log is indistinguishable from the EV controller's
    // own choice at the same depth (dflash's ready line does the same).
    {
        const src = @embedFile("generate.zig");
        const start = std.mem.indexOf(u8, src, "pub fn mtpDepthCapForProfile(configured") orelse return error.MissingResolver;
        const end = std.mem.indexOfPos(u8, src, start + 1, "\n    }\n") orelse return error.MissingResolverEnd;
        const body = src[start..end];
        try testing.expect(std.mem.indexOf(u8, body, "row.label") != null);
        try testing.expect(std.mem.indexOf(u8, body, "mtp_depth_cap_logged") != null);
    }

    const live_generic = mtp_mod.adaptiveDepthCapForMachine(ane_mod.chipBrand(), Generator.MTP_ADAPTIVE_DEFAULT_CAP).cap;
    try testing.expectEqual(live_generic, Generator.mtpDepthCapFor(0, true, false));
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapFor(0, true, true));
}

test "mtpFloorDisableObserve: extension misses do not poison depth one" {
    const W = Generator.MTP_DEPTH_WINDOW;
    var drafted: [W]u8 = @splat(0);
    var accepted: [W]u8 = @splat(0);
    var idx: u32 = 0;

    var rate: ?f32 = null;
    for (0..W) |_| {
        // An eight-wide extension accepted its first draft but no later one.
        rate = Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 8, 1);
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), rate.?, 1e-5);
    for (drafted) |sample| try testing.expectEqual(@as(u8, 1), sample);

    drafted = @splat(0);
    accepted = @splat(0);
    idx = 0;
    for (0..W) |i| {
        rate = Generator.mtpFloorDisableObserve(
            &drafted,
            &accepted,
            &idx,
            1,
            8,
            @intFromBool(i % 2 == 0),
        );
    }
    try testing.expectApproxEqAbs(@as(f32, 0.5), rate.?, 1e-5);
    // A 50% depth-one window is comfortably above the breakeven floor —
    // this rate must never disable (the old 0.50 floor sat exactly here).
    try testing.expect(!(rate.? < Generator.MTP_DISABLE_BELOW));
}

test "mtpFloorDisableObserve: disable needs 16 fresh failures at base depth one" {
    const W = Generator.MTP_DEPTH_WINDOW;
    var drafted: [W]u8 = @splat(0);
    var accepted: [W]u8 = @splat(0);
    var idx: u32 = 0;

    // Fifteen depth-one failures are insufficient.
    for (0..W - 1) |_| {
        try testing.expectEqual(
            @as(?f32, null),
            Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 1, 0),
        );
    }
    // A wider base round invalidates that probation window.
    try testing.expectEqual(
        @as(?f32, null),
        Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 2, 4, 0),
    );
    try testing.expectEqual(@as(u32, 0), idx);

    // It takes another complete run of depth-one failures to produce a rate.
    for (0..W - 1) |_| {
        try testing.expectEqual(
            @as(?f32, null),
            Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 1, 0),
        );
    }
    const rate = Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 1, 0);
    try testing.expectApproxEqAbs(@as(f32, 0.0), rate.?, 1e-5);
}

test "updateMtpEvRound: sticky disable uses the first draft at base depth one" {
    var good = mtpEvTestGenerator();
    for (0..Generator.MTP_DEPTH_WINDOW * 2) |_| good.updateMtpEvRound(8, 1);
    try testing.expect(!good.spec_disabled_runtime);
    for (good.mtp_window_drafted) |sample| try testing.expectEqual(@as(u8, 1), sample);
    for (good.mtp_window_accepted) |sample| try testing.expectEqual(@as(u8, 1), sample);

    var bad = mtpEvTestGenerator();
    for (0..Generator.MTP_DEPTH_WINDOW - 1) |_| {
        bad.updateMtpEvRound(8, 0);
        try testing.expect(!bad.spec_disabled_runtime);
    }
    bad.updateMtpEvRound(8, 0);
    try testing.expect(bad.spec_disabled_runtime);
}

test "updateMtpEvRound: warmup and wider base rounds reset floor evidence" {
    var g = mtpEvTestGenerator();
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS - 1;
    g.mtp_window_idx = 7;
    g.updateMtpEvRound(4, 0);
    try testing.expectEqual(Generator.MTP_EV_WARMUP_ROUNDS, g.mtp_ev_rounds);
    try testing.expectEqual(@as(u32, 0), g.mtp_window_idx);

    for (0..Generator.MTP_DEPTH_WINDOW - 1) |_| g.updateMtpEvRound(1, 0);
    try testing.expect(!g.spec_disabled_runtime);
    try testing.expectEqual(Generator.MTP_DEPTH_WINDOW - 1, g.mtp_window_idx);

    g.mtp_ev_m_lo_prev = 2;
    g.updateMtpEvRound(4, 0);
    try testing.expectEqual(@as(u32, 0), g.mtp_window_idx);
    try testing.expect(!g.spec_disabled_runtime);
}

test "the qwen4 rerank draft feeds the MIXER output, never the pre-mixer stream" {
    // The one way this change can be wrong without failing to compile. On the
    // sidecar arm `hidden_next` IS the lm_head's input, so the obvious wiring
    // (`draftSelect(xfm, step_out.hidden_next, ...)`) is correct there and
    // silently wrong on qwen4_exp, whose `hidden_next` is the pre-mixer
    // `[B,S,hc*H]` stream — 4x the width the head consumes. The chain must
    // therefore prefer `rerank_x` when a step publishes one.
    const source = @embedFile("generate.zig");
    const build_at = std.mem.indexOf(u8, source, "fn mtpChainBuild(") orelse return error.MissingChainBuild;
    const end = std.mem.indexOfPos(u8, source, build_at, "\n    /// Fire the chain's built graphs") orelse source.len;
    const body = source[build_at..end];

    const sel = std.mem.indexOf(u8, body, "head.draftSelect(") orelse return error.MissingDraftSelect;
    const line_start = std.mem.lastIndexOfScalar(u8, body[0..sel], '\n') orelse 0;
    const line_end = std.mem.indexOfScalarPos(u8, body, sel, '\n') orelse body.len;
    const call = body[line_start..line_end];
    // The rerank input is the resolved vector, never `hidden_next` directly.
    try testing.expect(std.mem.indexOf(u8, call, "rerank_x") != null);
    try testing.expect(std.mem.indexOf(u8, call, "hidden_next") == null);
    try testing.expect(std.mem.indexOf(u8, body, "step_out.rerank_x.ctx != null") != null);

    // ...and a `.mixed` step is what asks for it: a rerank draft must not
    // collapse onto `.none` (the history-append mode), which produces no
    // mixer output at all.
    try testing.expect(std.mem.indexOf(u8, body, "if (use_rerank) .mixed else .logits") != null);

    // The qwen4 arm maps `.mixed` onto the projection that SKIPS the lm_head
    // but still runs the mixer. If it ever mapped to `.none`, `rerank_x` would
    // be null and the chain would silently hand over the pre-mixer stream.
    const step_at = std.mem.indexOf(u8, source, "fn qwen4Step(") orelse return error.MissingQwen4Step;
    const step_end = std.mem.indexOfPos(u8, source, step_at, "\n    /// One head forward") orelse source.len;
    const step_body = source[step_at..step_end];
    try testing.expect(std.mem.indexOf(u8, step_body, ".mixed => .mixed_last_row") != null);
    try testing.expect(std.mem.indexOf(u8, step_body, ".rerank_x = out.mixed") != null);

    // And the owned mixer vector is freed by the chain, not leaked per step.
    try testing.expect(std.mem.indexOf(u8, body, "mlx_array_free(step_out.rerank_x)") != null);
}

test "mtpEvExpectedTokens: 1 + sum of acceptance chain products" {
    const a = [_]f32{ 0.5, 0.5, 0.5 };
    try testing.expectApproxEqAbs(@as(f32, 1.5), Generator.mtpEvExpectedTokens(&a, 1), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.75), Generator.mtpEvExpectedTokens(&a, 2), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.875), Generator.mtpEvExpectedTokens(&a, 3), 1e-5);
    // Zero acceptance: every round still commits exactly the t1 bonus token.
    const z = [_]f32{ 0.0, 0.0 };
    try testing.expectApproxEqAbs(@as(f32, 1.0), Generator.mtpEvExpectedTokens(&z, 2), 1e-5);
}

test "mtpEvRoundCost: piecewise marginals (flat verify region, then the GDN width ramp)" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    try testing.expectApproxEqAbs(@as(f32, 1.19), Generator.mtpEvRoundCost(costs, 1, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.57), Generator.mtpEvRoundCost(costs, 3, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.59), Generator.mtpEvRoundCost(costs, 3, true), 1e-5);
    // Positions 4+ pay the ramp: +0.32 each instead of +0.19.
    try testing.expectApproxEqAbs(@as(f32, 2.21), Generator.mtpEvRoundCost(costs, 5, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.19), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.32), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
}

test "mtpEvCostsFor: G17 profiles are explicit and env tuning stays generic" {
    const generic = Generator.mtpEvCostsForProfile(.generic, null);
    try testing.expectEqual(Generator.MTP_EV_DEFAULT_COSTS, generic);

    const q8 = Generator.mtpEvCostsForProfile(.g17_nax_q8_gs32, null);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_COSTS, q8);
    try testing.expectEqual(@as(u32, 7), q8.nax_from);

    const q4 = Generator.mtpEvCostsForProfile(.g17_nax_q4_gs32, null);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, q4);
    try testing.expectEqual(@as(u32, 7), q4.nax_from);

    const q4_gs64 = Generator.mtpEvCostsForProfile(.g17_nax_q4_gs64, null);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, q4_gs64);
    try testing.expectEqual(@as(u32, 7), q4_gs64.nax_from);

    const oq4e = Generator.mtpEvCostsForProfile(.g17_nax_oq4e_q4_gs64, null);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS, oq4e);
    try testing.expectEqual(@as(u32, 7), oq4e.nax_from);

    const qwen4 = Generator.mtpEvCostsForProfile(.g17_nax_qwen4_q4_gs64, null);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, qwen4);

    // An explicit four-value override retains its historical meaning instead
    // of inheriting an implicit hardware-only third region, for every profile.
    for ([_]mtp_mod.MtpCostProfile{ .generic, .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q4_gs64, .g17_nax_oq4e_q4_gs64, .g17_nax_qwen4_q4_gs64 }) |profile| {
        const tuned = Generator.mtpEvCostsForProfile(profile, "0.10, 0.11, 0.22, 0.03");
        try testing.expectApproxEqAbs(@as(f32, 0.10), tuned.draft, 1e-6);
        try testing.expectApproxEqAbs(@as(f32, 0.11), tuned.per_pos_lo, 1e-6);
        try testing.expectApproxEqAbs(@as(f32, 0.22), tuned.per_pos_hi, 1e-6);
        try testing.expectApproxEqAbs(@as(f32, 0.03), tuned.sync, 1e-6);
        try testing.expectEqual(@as(u32, 0), tuned.nax_from);
        try testing.expectApproxEqAbs(@as(f32, 0.0), tuned.per_pos_nax, 1e-6);
    }

    // Invalid overrides are atomic no-ops. In particular, an empty variable
    // must not combine cap 8 with the generic costs that previously starved it.
    const invalid = [_][]const u8{
        "",
        "0.10,0.11",
        "garbage,0.11,0.22,0.03",
        "nan,0.11,0.22,0.03",
        "0.10,0.11,0.22,0.03,0.04",
        "0.10,-0.11,0.22,0.03",
    };
    for (invalid) |raw| {
        try testing.expectEqual(Generator.MTP_EV_DEFAULT_COSTS, Generator.mtpEvCostsForProfile(.generic, raw));
        try testing.expectEqual(Generator.MTP_EV_G17_NAX_COSTS, Generator.mtpEvCostsForProfile(.g17_nax_q8_gs32, raw));
        try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, Generator.mtpEvCostsForProfile(.g17_nax_q4_gs32, raw));
        try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, Generator.mtpEvCostsForProfile(.g17_nax_q4_gs64, raw));
        try testing.expectEqual(Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS, Generator.mtpEvCostsForProfile(.g17_nax_oq4e_q4_gs64, raw));
        try testing.expectEqual(Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, Generator.mtpEvCostsForProfile(.g17_nax_qwen4_q4_gs64, raw));
    }

    try testing.expectEqual(Generator.MTP_EV_DEFAULT_COSTS, Generator.mtpEvCostsFor(false, null));
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_COSTS, Generator.mtpEvCostsFor(true, null));
}

test "MTP_EV_G17_NAX_COSTS reproduces the measured M5 depth-6/depth-8 ratio" {
    const costs = Generator.MTP_EV_G17_NAX_COSTS;
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.99), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.19), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 68.39 / 62.15), t8 / t6, 2e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.21), Generator.mtpEvMarginalCost(costs, 6), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.10), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
}

test "MTP_EV_G17_NAX_Q4_GS32_COSTS encodes calibrated composite marginals" {
    const costs = Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS;
    try testing.expect(costs.draft > 0.0);
    try testing.expect(costs.per_pos_lo > 0.0);
    try testing.expect(costs.per_pos_hi > 0.0);
    try testing.expect(costs.per_pos_nax > 0.0);
    try testing.expectApproxEqAbs(@as(f32, 0.11), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.20), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.05), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.93), Generator.mtpEvRoundCost(costs, 6, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.03), Generator.mtpEvRoundCost(costs, 8, false), 1e-5);
}

test "MTP_EV_G17_NAX_Q4_GS64_COSTS reproduces the calibrated M5 surface" {
    const costs = Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.075), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.195), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.08), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t1 = Generator.mtpEvRoundCost(costs, 1, false);
    const t3 = Generator.mtpEvRoundCost(costs, 3, false);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.075), t1, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.225), t3, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.81), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.97), t8, 1e-5);
}

test "MTP_EV_G17_NAX_Q6_GS64_COSTS reproduces the calibrated M5 surface" {
    const costs = Generator.MTP_EV_G17_NAX_Q6_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.06), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.115), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.145), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.525), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.815), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 76.79 / 64.495), t8 / t6, 1e-3);
}

test "MTP_EV_G17_NAX_Q8_GS64_COSTS bridges the discontinuous NAX takeover" {
    const costs = Generator.MTP_EV_G17_NAX_Q8_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.06), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.19), Generator.mtpEvMarginalCost(costs, 5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.045), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.62), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.71), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 85.94 / 53.0575), t6, 2e-3);
    try testing.expectApproxEqAbs(@as(f32, 90.68 / 53.0575), t8, 2e-3);
}

test "MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS reproduces the measured M5 surface" {
    const costs = Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.095), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.22), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.025), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.945), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.995), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 62.79 / 61.20), t8 / t6, 5e-4);
}

test "mtpEvPlanFor: M5 NAX surfaces open depth 8 from realistic warmup EMAs" {
    const p = Generator.MTP_EV_PRIOR;
    const a = [_]f32{ 0.97, 0.89, p, p, p, p, p, p };

    // The M1-M4 surface stops at the first expensive ramp position.
    const generic = Generator.mtpEvPlanFor(&a, 8, Generator.MTP_EV_DEFAULT_COSTS, 3);
    try testing.expectEqual(@as(u32, 3), generic.m_lo);
    try testing.expect(generic.m_hi < 8);

    // The M5 fit captures both its cheaper intermediate widths and the NAX
    // takeover at draft position 7, so the same evidence reaches depth 8.
    for ([_]Generator.MtpEvCosts{ Generator.MTP_EV_G17_NAX_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, Generator.MTP_EV_G17_NAX_Q6_GS64_COSTS, Generator.MTP_EV_G17_NAX_Q8_GS64_COSTS, Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS }) |costs| {
        const nax = Generator.mtpEvPlanFor(&a, 8, costs, 3);
        try testing.expectEqual(@as(u32, 3), nax.m_lo);
        try testing.expectEqual(@as(u32, 8), nax.m_hi);
        try testing.expect(nax.tau_ln < 0.0);
    }

    // qwen4_exp's measured surface has no NAX takeover — a verify row is
    // bytes at every width — so the same base depth stands but the same
    // evidence must NOT extend to depth 8.
    const qwen4_plan = Generator.mtpEvPlanFor(&a, 8, Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, 3);
    try testing.expectEqual(@as(u32, 3), qwen4_plan.m_lo);
    try testing.expect(qwen4_plan.m_hi < 8);

    const cold = [_]f32{ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    for ([_]Generator.MtpEvCosts{ Generator.MTP_EV_G17_NAX_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS, Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS }) |costs| {
        const cold_plan = Generator.mtpEvPlanFor(&cold, 8, costs, 8);
        try testing.expectEqual(@as(u32, 1), cold_plan.m_lo);
        try testing.expectEqual(@as(u32, 1), cold_plan.m_hi);
    }
    // Qwen3.8's measured depth-1 marginal is cheap enough that even this
    // synthetic 20% first-draft case clears the exploration floor narrowly.
    // The base stays at one and only one confidence-gated position is exposed.
    const q38_cold = Generator.mtpEvPlanFor(&cold, 8, Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, 8);
    try testing.expectEqual(@as(u32, 1), q38_cold.m_lo);
    try testing.expectEqual(@as(u32, 2), q38_cold.m_hi);
    try testing.expect(q38_cold.tau_ln < 0.0);
}

test "mtpEvPlanFor: mid-decay acceptance picks a shallow base and a confidence-gated extension" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    // Conditional acceptance decays: unconditional EV peaks at m=2, but the
    // marginal chain CONDITIONAL on chunk A landing stays profitable through
    // the flat verify region — the "draft deeper on easy stretches" shape.
    const a = [_]f32{ 0.7, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    try testing.expectEqual(@as(u32, 2), plan.m_lo);
    try testing.expectEqual(@as(u32, 3), plan.m_hi);
    // tau = r(m_lo)*t_ext/S = 1.5362*0.19/0.55 = 0.5307 -> ln = -0.6335.
    try testing.expectApproxEqAbs(@as(f32, -0.6335), plan.tau_ln, 5e-3);
}

test "mtpEvPlanFor: hot flat acceptance rides the flat region and extends into the ramp on confidence" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    const a = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    // Static optimum m=3 (r=2.1904); ramp positions 4..6 pay only under full
    // confidence (0.9^k chain vs r*0.32 = 0.70 threshold).
    try testing.expectEqual(@as(u32, 3), plan.m_lo);
    try testing.expectEqual(@as(u32, 6), plan.m_hi);
    // tau = 2.1904*0.96/2.439 = 0.8621 -> ln = -0.1484.
    try testing.expectApproxEqAbs(@as(f32, -0.1484), plan.tau_ln, 5e-3);
}

test "mtpEvPlanFor: a stale-cold EMA one past m_lo cannot close the horizon permanently" {
    // a[1] is observable ONLY when extension fires (at m=1 rounds it never
    // updates), so a value dragged down under an old workload must not be
    // able to lock the horizon shut — the live regression was ext_rounds=0
    // on the equivalence echo after refit #3's honest marginals. When the
    // base pays, one extension position stays reachable at the clamped
    // exploration tau; realized extensions then re-observe a[1] honestly.
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const a = [_]f32{ 0.75, 0.28, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 2), plan.m_hi);
    // tau clamps to TAU_MAX: only near-perfect-confidence rounds extend.
    try testing.expectApproxEqAbs(@log(Generator.MTP_EV_TAU_MAX), plan.tau_ln, 1e-6);
}

test "mtpEvPlanFor: cold acceptance collapses to depth 1, single chunk" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    const a = [_]f32{ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 1), plan.m_hi);
}

test "mtpEvPlanFor: m_lo_max damps the climb without killing the extension horizon" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    const a = [_]f32{ 0.7, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38 };
    // Same EMAs as the mid-decay case, but the controller may only raise the
    // base one step (hysteresis): m_lo caps at 1 while m_hi stays deeper.
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 1);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 3), plan.m_hi);
    try testing.expect(plan.tau_ln < 0.0);
}

test "mtpEvPlanFor: unobserved deep indices at the prior still open the extension horizon (exploration)" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    // The echo shape after warmup: shallow indices observed hot, deep indices
    // never reached (still at MTP_EV_PRIOR). The horizon must open past m_lo
    // so extension can get its first trial — this is the live ext_rounds=0
    // regression (a prior at the measured average sat razor-under the ramp
    // break-even of ~0.78 and extension never fired on pure echo).
    const p = Generator.MTP_EV_PRIOR;
    const a = [_]f32{ 0.97, 0.97, p, p, p, p, p };
    const plan = Generator.mtpEvPlanFor(&a, 7, costs, 8);
    try testing.expectEqual(@as(u32, 3), plan.m_lo);
    try testing.expect(plan.m_hi > plan.m_lo);
    // tau = r(3)*0.32/0.85 = 0.8898 -> ln = -0.1168 (under the 0.95 clamp).
    try testing.expectApproxEqAbs(@as(f32, -0.1168), plan.tau_ln, 5e-3);
}

test "mtpRegime: the worse shape runs as a scheduled trial block, unmeasured runs as planned" {
    var r = Generator.MtpRegime{};
    // Unseeded: the plan stands (two-chunk measures itself); once two-chunk
    // is measured, an unmeasured single is forced AT ONCE. The first round
    // of a shape is its transition and is not observed.
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, 1) == null);
    Generator.mtpRegimeObserve(&r, true, 4, 129.0, 5.65); // first round counts
    try testing.expect(Generator.mtpRegimeForce(&r, 2) == false);
    // M4 base echo (measured): two-chunk 22.83 vs single 18.11 ms/tok = 26%
    // worse, so a 2-round trial block recurs every 53 rounds (1% drag). The
    // single's first round is a transition (dropped), the second seeds.
    Generator.mtpRegimeObserve(&r, false, 4, 90.5, 5.0);
    try testing.expect(Generator.mtpRegimeForce(&r, 3) == false);
    Generator.mtpRegimeObserve(&r, false, 4, 90.5, 5.0);
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r).?);
    try testing.expectEqual(@as(u32, 53), Generator.mtpRegimeExplorePeriod(r));
    try testing.expect(Generator.mtpRegimeForce(&r, 14) == false); // verdict forms: next trial at 67
    try testing.expect(Generator.mtpRegimeForce(&r, 66) == false);
    try testing.expect(Generator.mtpRegimeForce(&r, 67) == null); // block
    try testing.expect(Generator.mtpRegimeForce(&r, 68) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, 69) == false); // and not a third
    try testing.expectEqual(@as(u32, 69 + 53), r.next_trial);
    try testing.expectEqual(@as(u32, 14), r.verdict_round);
    try testing.expectEqual(@as(u32, 1), r.trials);
    // Asked twice for the same round (pre-draft + entry): same answer, no
    // schedule drift.
    try testing.expect(Generator.mtpRegimeForce(&r, 122) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, 122) == null);
    try testing.expectEqual(@as(u32, 2), r.trials);
    try testing.expectEqual(@as(u32, 124 + 53), r.next_trial);
    // Two-chunk 64 ms for 6 (10.7) against single 56 for 4.96 (11.3):
    // two-chunk runs every round and a single-chunk block is scheduled once
    // per period to keep the other regime's EMA alive.
    var m = Generator.MtpRegime{};
    Generator.mtpRegimeObserve(&m, true, 4, 64.0, 6.0);
    Generator.mtpRegimeObserve(&m, false, 4, 56.0, 4.96);
    Generator.mtpRegimeObserve(&m, false, 4, 56.0, 4.96);
    try testing.expect(!Generator.mtpRegimeTwoChunkWorse(m).?);
    try testing.expectEqual(@as(u32, 12), Generator.mtpRegimeExplorePeriod(m)); // 5.8% gap
    try testing.expect(Generator.mtpRegimeForce(&m, 5) == null); // verdict forms: next trial at 17
    try testing.expect(Generator.mtpRegimeForce(&m, 16) == null);
    try testing.expect(Generator.mtpRegimeForce(&m, 17) == false);
    try testing.expect(Generator.mtpRegimeForce(&m, 18) == false);
    try testing.expect(Generator.mtpRegimeForce(&m, 19) == null);
    // M4 Max 27B @16k: 13.4 vs 12.95 ms/tok is inside the margin — the plan
    // stands (two-chunk measured +3.7% at arm level).
    var n = Generator.MtpRegime{};
    Generator.mtpRegimeObserve(&n, true, 4, 80.4, 6.0);
    Generator.mtpRegimeObserve(&n, false, 4, 64.75, 5.0);
    Generator.mtpRegimeObserve(&n, false, 4, 64.75, 5.0);
    try testing.expect(!Generator.mtpRegimeTwoChunkWorse(n).?);
    // Hysteresis (M1 Pro 9B v5.2): a standing "worse" at 26.4 vs 23.8 does
    // not flip when the post-block single reads 25.3 (ratio 1.04, inside the
    // margin); it flips only once two-chunk is at or below single.
    var h = Generator.MtpRegime{ .two_ms = 26.4, .two_tok = 1.0, .two_m = 4, .one_ms = 25.3, .one_tok = 1.0, .one_m = 4 };
    try testing.expect(Generator.mtpRegimeVerdict(h, true).?);
    try testing.expect(!Generator.mtpRegimeVerdict(h, null).?);
    h.one_ms = 26.5;
    try testing.expect(!Generator.mtpRegimeVerdict(h, true).?);
}

test "mtpRegime: a simulated round loop reaches BOTH shapes, and the worse one keeps being re-tried" {
    // Live 2026-08-22 (M4 base): v2's "try the unmeasured single shape at
    // once" rule compared one_m against a two_m that only a two-chunk round
    // can set, so it forced single-chunk forever — caps 5/6 reported cap-4
    // numbers with ext_rounds=0 and no verdict line, which reads as a PASS
    // on an echo workload. v4's `idx % period` gate then ran trial chains
    // because the block's own observation moved the period (14 of 72 rounds
    // as trials against 7). The gate is driven here exactly as mtpRoundPlan
    // drives it: ask, run that shape, feed the observation.
    const Sim = struct {
        fn run(two_ms: f32, one_ms: f32, rounds: u32) struct { two: u32, one: u32 } {
            var r = Generator.MtpRegime{};
            var two: u32 = 0;
            var one: u32 = 0;
            var i: u32 = 0;
            while (i < rounds) : (i += 1) {
                // The plan always offers two-chunk (pure echo); the gate decides.
                const two_chunk = Generator.mtpRegimeForce(&r, i) orelse true;
                if (two_chunk) two += 1 else one += 1;
                Generator.mtpRegimeObserve(&r, two_chunk, 4, if (two_chunk) two_ms else one_ms, if (two_chunk) 6.0 else 5.0);
            }
            return .{ .two = two, .one = one };
        }
    };
    // M4 base: two-chunk 26% worse. Pre-verdict 4 rounds, then 2-round
    // blocks every 53: 4 + 2*3 = 10 two-chunk rounds in 200, not 20.
    const worse = Sim.run(129.0, 90.5, 200);
    try testing.expect(worse.one > 185);
    try testing.expect(worse.two >= 6 and worse.two <= 12);
    // M4 Max: two-chunk better (5.8% gap): 2-round single blocks every 12.
    const better = Sim.run(64.0, 56.0, 200);
    try testing.expect(better.two > 160);
    try testing.expect(better.one >= 25 and better.one <= 40);
}

test "MtpCostSource: a measured cliff stops the plan where the fitted surface would extend" {
    // M1 Pro 27B: depth 4 -> 5 costs +150 ms/round. The fitted surface
    // prices position 5 at 0.26 floor units and extends; the table has
    // measured it. Same acceptance, same cap, two answers.
    const a = [_]f32{ 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95 };
    const prior = Generator.mtpEvPlanForAt(&a, 8, Generator.MTP_EV_DEFAULT_COSTS, 8, 10000);
    try testing.expect(prior.m_hi >= 5);
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(3, 10000, 60.0, 4.0, true, false);
        _ = t.observe(4, 10000, 70.0, 5.0, true, false);
        _ = t.observe(5, 10000, 220.0, 6.0, true, false);
    }
    const src = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 10000, &t);
    try testing.expect(src.fromTable());
    // Anchored at the narrowest measured width: the two sources agree
    // there, and below it the prior's shape fills in.
    try testing.expectApproxEqAbs(Generator.mtpEvRoundCostAt(Generator.MTP_EV_DEFAULT_COSTS, 3, false, 10000), src.roundCost(3, false), 1e-4);
    try testing.expectApproxEqAbs(Generator.mtpEvRoundCostAt(Generator.MTP_EV_DEFAULT_COSTS, 2, false, 10000), src.roundCost(2, false), 1e-4);
    try testing.expect(src.marginal(5) > 5.0 * src.marginal(4));
    // Past the widest measured width the cliff's slope continues (it is
    // steeper than the prior's marginal here).
    try testing.expectApproxEqAbs(src.roundCost(5, false) + (src.roundCost(5, false) - src.roundCost(4, false)), src.roundCost(6, false), 1e-3);
    const measured = Generator.mtpEvPlanSrc(&a, 8, src, 8);
    try testing.expectEqual(@as(u32, 4), measured.m_hi);
    try testing.expect(measured.m_lo <= 4);
    // ONE sample at the cliff (untrusted) already floors the horizon.
    var t1 = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t1.observe(3, 10000, 60.0, 4.0, true, false);
        _ = t1.observe(4, 10000, 70.0, 5.0, true, false);
    }
    _ = t1.observe(5, 10000, 220.0, 6.0, true, false);
    const src1 = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 10000, &t1);
    try testing.expectEqual(@as(u32, 4), Generator.mtpEvPlanSrc(&a, 8, src1, 8).m_hi);
    // ...and the trial stops asking for it (two-chunk plan, nothing owed).
    try testing.expect(Generator.mtpWidthTrialTarget(&t1, 10000, .{ .m_lo = 4, .m_hi = 5, .tau_ln = 0 }, 8, true) == null);
    // An unmeasured base is trialled only under a two-chunk plan with a settled base.
    const empty_t = round_cost.Table{};
    try testing.expect(Generator.mtpWidthTrialTarget(&empty_t, 1000, .{ .m_lo = 3, .m_hi = 3, .tau_ln = 0 }, 8, true) == null);
    try testing.expect(Generator.mtpWidthTrialTarget(&empty_t, 1000, .{ .m_lo = 3, .m_hi = 5, .tau_ln = 0 }, 8, false) == null);
    try testing.expectEqual(@as(u32, 3), Generator.mtpWidthTrialTarget(&empty_t, 1000, .{ .m_lo = 3, .m_hi = 5, .tau_ln = 0 }, 8, true).?);
    try testing.expect(Generator.mtpWidthTrialPeriod(&t1, 10000, 4) >= 100);
    // An inactive bucket with no active neighbour is the prior, verbatim.
    const empty = round_cost.Table{};
    const cold = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 10000, &empty);
    try testing.expect(!cold.fromTable());
    const cold_plan = Generator.mtpEvPlanSrc(&a, 8, cold, 8);
    try testing.expectEqual(prior.m_hi, cold_plan.m_hi);
    try testing.expectEqual(prior.m_lo, cold_plan.m_lo);
}

test "mtpWidthTrial: blocks per period, idempotent per round, period grows with the measured gap" {
    var wt = Generator.MtpWidthTrial{};
    var forced: u32 = 0;
    var i: u32 = 10;
    while (i < 210) : (i += 1) {
        const f = Generator.mtpWidthTrialForce(&wt, i, 8);
        try testing.expectEqual(f, Generator.mtpWidthTrialForce(&wt, i, 8)); // asked twice, same answer
        if (f) forced += 1;
    }
    // Identity checkable from the log: forced rounds == block * trials.
    try testing.expectEqual(wt.trials * round_cost.EXPLORE_BLOCK, forced);
    try testing.expect(wt.trials >= 17 and wt.trials <= 19);
    // Unmeasured next width: default period. 10% worse: 30. 36% worse: 110.
    var t = round_cost.Table{};
    try testing.expectEqual(round_cost.EXPLORE_PERIOD_COLD, Generator.mtpWidthTrialPeriod(&t, 1000, 4));
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(4, 1000, 40.0, 4.0, true, false);
        _ = t.observe(5, 1000, 55.0, 5.0, true, false);
    }
    try testing.expectEqual(@as(u32, 60), Generator.mtpWidthTrialPeriod(&t, 1000, 4));
    // One better-looking sample is not a period: cold until trusted.
    var u = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| _ = u.observe(4, 1000, 40.0, 4.0, true, false);
    _ = u.observe(5, 1000, 45.0, 5.0, true, false);
    try testing.expectEqual(round_cost.EXPLORE_PERIOD_COLD, Generator.mtpWidthTrialPeriod(&u, 1000, 4));
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observe(6, 1000, 90.0, 6.0, true, false);
    try testing.expectEqual(@as(u32, 219), Generator.mtpWidthTrialPeriod(&t, 1000, 5));
}

test "round_cost: a simulated round loop measures every width the chooser picks and settles under the cliff" {
    // Driven as mtpRoundPlan drives it: ask the plan (prior until the
    // bucket has two widths), run that width, feed the table with a
    // synthetic machine whose round cost is flat-ish to depth 4 and cliffs
    // at 5 (M1 Pro 27B shape). Acceptance is high, so the prior alone
    // would sit at depth 6+.
    const Machine = struct {
        fn roundMs(m: u32) f32 {
            return switch (m) {
                0 => 50.0,
                1 => 56.0,
                2 => 62.0,
                3 => 68.0,
                4 => 74.0,
                else => 74.0 + 150.0 * @as(f32, @floatFromInt(m - 4)),
            };
        }
    };
    const a = [_]f32{ 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95 };
    var t = round_cost.Table{};
    var wt = Generator.MtpWidthTrial{};
    var prev: ?u32 = null;
    var prev2: ?u32 = null;
    var prev_two = false;
    var prev_two2 = false;
    var late_wide: u32 = 0;
    var streak: u32 = 0;
    var picked: [round_cost.MAX_WIDTH + 1]u32 = @splat(0);
    var m_lo_prev: u32 = 1;
    var last_m: u32 = 0;
    var i: u32 = 10;
    while (i < 600) : (i += 1) {
        const src = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 1000, &t);
        var plan = Generator.mtpEvPlanSrc(&a, 8, src, m_lo_prev + 1);
        if (plan.m_lo == m_lo_prev) streak += 1 else streak = 0;
        if (Generator.mtpWidthTrialTarget(&t, 1000, plan, 8, streak >= 2)) |target| {
            if (Generator.mtpWidthTrialForce(&wt, i, Generator.mtpWidthTrialPeriod(&t, 1000, plan.m_lo))) plan = Generator.mtpWidthTrialPlan(target);
        }
        // Run it: a two-chunk plan extends (confidence is high), so the
        // realized width is m_hi — and, as in mtpRoundEndObserve, only a
        // single-chunk round feeds the table.
        const m = plan.m_hi;
        const two_chunk = plan.m_hi > plan.m_lo;
        picked[m] += 1;
        if (i >= 200 and m >= 5) late_wide += 1;
        last_m = m;
        const tokens: f32 = Generator.mtpEvExpectedTokens(&a, m);
        const transition = prev_two or prev_two2 or (if (prev) |p| p != m else true) or (if (prev2) |p| p != m else true);
        if (!two_chunk) _ = t.observe(m, 1000, Machine.roundMs(m), tokens, true, transition);
        prev2 = prev;
        prev = m;
        prev_two2 = prev_two;
        prev_two = two_chunk;
        m_lo_prev = plan.m_lo;
    }
    // The cliff was found from ONE sample (clearly worse: never trusted,
    // never re-trialled at the cold period) and the loop settled under it.
    try testing.expect(t.rawMs(5, 0) != null);
    try testing.expect(t.clearlyWorse(5, 4, 0));
    try testing.expect(t.measuredMs(4, 0) != null);
    try testing.expect(last_m == 4);
    try testing.expect(picked[4] > 450);
    // Before the table activates the prior extends 4 -> 5 blind every round
    // (live, the regime gate throttles that shape; the sim has no gate), so
    // the bar is the settled tail: from round 200 on, width 5 appears only
    // as scheduled re-trial blocks (period capped at 128).
    try testing.expect(late_wide <= 2 * round_cost.EXPLORE_BLOCK);
    try testing.expect(picked[6] + picked[7] + picked[8] <= 4); // the prior's first rounds only
    // Identity: forced rounds == 2 * trials (each block is 2 rounds).
    try testing.expect(wt.trials >= 2);
}

test "mtpRegimeObserve: seeds on the first sample, moves by the cost beta, reseeds on a new base depth" {
    var r = Generator.MtpRegime{};
    Generator.mtpRegimeObserve(&r, true, 3, 45.0, 4.0); // first round counts
    try testing.expect(r.two_tok > 0.0);
    Generator.mtpRegimeObserve(&r, false, 3, 45.0, 4.0); // transition: dropped
    try testing.expect(r.one_tok == 0.0);
    Generator.mtpRegimeObserve(&r, false, 3, 50.0, 4.0);
    try testing.expectApproxEqAbs(@as(f32, 50.0), r.one_ms, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), r.one_tok, 1e-6);
    Generator.mtpRegimeObserve(&r, false, 3, 60.0, 5.0);
    try testing.expectApproxEqAbs(50.0 + Generator.MTP_EV_COST_BETA * 10.0, r.one_ms, 1e-4);
    try testing.expectApproxEqAbs(4.0 + Generator.MTP_EV_COST_BETA * 1.0, r.one_tok, 1e-4);
    // The climb moved m_lo: the depth-3 rounds are not the depth-4 regime.
    Generator.mtpRegimeObserve(&r, false, 4, 70.0, 5.0);
    try testing.expectApproxEqAbs(@as(f32, 70.0), r.one_ms, 1e-6);
    // A two-chunk reading at a different base depth yields no verdict.
    Generator.mtpRegimeObserve(&r, true, 3, 80.0, 6.0);
    Generator.mtpRegimeObserve(&r, true, 3, 80.0, 6.0);
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r) == null);
}

test "mtpEvPlanFor: DEFAULT costs carry the post-sdpa-split surface (2026-08-15 refit #4)" {
    // Pins MTP_EV_DEFAULT_COSTS to the surface measured AFTER the hd-256
    // causal sdpa split (same-session saturated forced-depth echo sweep,
    // Jundot oQ4e 27B @8K cold reps, M4 Max: T(1)=44.6 .. T(8)=142.3 →
    // floor ≈ 38.2 ms, marginals k<=4 ≈ 0.20, k5-6 ≈ 0.36, k7-8 ≈ 0.62).
    // flat_max 4: the old hi over-priced k4 at 0.34 (measured 0.24), so
    // moderate content under-drafted it. The k>=7 third region carries the
    // plain-SIMD verify-qmm register cliff — only reachable when
    // --mtp-depth forces past the generic cap of 6.
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    // Hot uniform 90%: base now rides the WIDER flat region (m_lo 4, was 3
    // under refit #3), extension one step into the ramp.
    const hot = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const hot_plan = Generator.mtpEvPlanFor(&hot, 8, costs, 8);
    try testing.expectEqual(@as(u32, 4), hot_plan.m_lo);
    try testing.expectEqual(@as(u32, 5), hot_plan.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.0943), hot_plan.tau_ln, 5e-3);
    // Marginal 75%: base stays 3, one cheap flat extension position.
    const mid = [_]f32{ 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75 };
    const mid_plan = Generator.mtpEvPlanFor(&mid, 8, costs, 8);
    try testing.expectEqual(@as(u32, 3), mid_plan.m_lo);
    try testing.expectEqual(@as(u32, 4), mid_plan.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.786), mid_plan.tau_ln, 5e-3);
    // The oQ4e head's measured decayed chain still plans a 2-deep base.
    const oq4e = [_]f32{ 0.78, 0.50, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45 };
    const oq4e_plan = Generator.mtpEvPlanFor(&oq4e, 8, costs, 8);
    try testing.expectEqual(@as(u32, 2), oq4e_plan.m_lo);
}

test "mtpExtDryAllows: a dry streak collapses to single-chunk, cooldown expires into a fresh trial" {
    var streak: u32 = 0;
    var cooldown: u32 = 0;
    const thr = Generator.MTP_EXT_DRY_ROUNDS;
    // Considering rounds below the streak threshold stay allowed.
    var i: u32 = 0;
    while (i < thr - 1) : (i += 1) {
        try testing.expect(Generator.mtpExtDryAllows(&streak, &cooldown, thr));
        streak += 1; // caller records "considered, did not extend"
    }
    // The full dry streak trips the cooldown: consideration collapses.
    streak = thr;
    try testing.expect(!Generator.mtpExtDryAllows(&streak, &cooldown, thr));
    try testing.expectEqual(@as(u32, 0), streak);
    // Cooldown rounds stay collapsed and count down.
    var blocked: u32 = 0;
    while (!Generator.mtpExtDryAllows(&streak, &cooldown, thr)) blocked += 1;
    try testing.expectEqual(Generator.MTP_EXT_DRY_COOLDOWN - 1, blocked);
    // After the cooldown, consideration re-opens (fresh trial, streak 0).
    try testing.expectEqual(@as(u32, 0), streak);
    try testing.expect(Generator.mtpExtDryAllows(&streak, &cooldown, thr));
    // An extension firing resets the streak (caller does streak = 0) — the
    // gate never trips on workloads where extension pays.
}

test "mtpExtDryAllows: a lower cost-aware threshold trips the cooldown sooner" {
    var streak: u32 = 0;
    var cooldown: u32 = 0;
    const thr: u32 = 4; // e.g. an expensive sync throttled hard
    var i: u32 = 0;
    while (i < thr - 1) : (i += 1) {
        try testing.expect(Generator.mtpExtDryAllows(&streak, &cooldown, thr));
        streak += 1;
    }
    streak = thr;
    try testing.expect(!Generator.mtpExtDryAllows(&streak, &cooldown, thr));
    try testing.expectEqual(@as(u32, 0), streak);
    // The fresh trial still fits inside an echo stretch even at the floor.
    try testing.expect(Generator.MTP_EXT_DRY_MIN + Generator.MTP_EXT_DRY_COOLDOWN < 70);
}

test "mtpExtDryThresholdFor: cost-aware dry threshold scales inversely with the measured sync fraction" {
    const D = Generator;
    // Unmeasured (either EMA still 0) keeps the fixed dry budget — no behavior
    // change until the live cost is known.
    try testing.expectEqual(D.MTP_EXT_DRY_ROUNDS, D.mtpExtDryThresholdFor(0.0, 45.0));
    try testing.expectEqual(D.MTP_EXT_DRY_ROUNDS, D.mtpExtDryThresholdFor(2.4, 0.0));
    // A near-free sync (tiny fraction) tolerates the full dry budget.
    try testing.expectEqual(D.MTP_EXT_DRY_ROUNDS, D.mtpExtDryThresholdFor(0.2, 45.0));
    // An expensive sync (large fraction) collapses toward the floor.
    try testing.expectEqual(D.MTP_EXT_DRY_MIN, D.mtpExtDryThresholdFor(9.0, 45.0));
    // Monotonic: a costlier sync never tolerates MORE dry rounds.
    try testing.expect(D.mtpExtDryThresholdFor(4.0, 45.0) <= D.mtpExtDryThresholdFor(1.0, 45.0));
    // Always bounded — never below the floor (a zero threshold would trip
    // every round and close the horizon; the fresh-trial guarantee survives
    // any cost).
    try testing.expect(D.mtpExtDryThresholdFor(1000.0, 45.0) >= D.MTP_EXT_DRY_MIN);
    // The measured 8K operating point (sync ~2.4 ms of a ~45 ms round) lands
    // in the throttled band, well below the fixed 16.
    const at_8k = D.mtpExtDryThresholdFor(2.4, 45.0);
    try testing.expect(at_8k >= D.MTP_EXT_DRY_MIN and at_8k < D.MTP_EXT_DRY_ROUNDS);
}

test "mtpEmaMs: seeds on the first sample then folds nanoseconds into a ms EMA" {
    const D = Generator;
    // First sample seeds (prev <= 0): 45 ms == 45_000_000 ns, exactly.
    try testing.expectApproxEqAbs(@as(f32, 45.0), D.mtpEmaMs(0.0, 45_000_000), 1e-3);
    // A subsequent sample folds at MTP_EV_COST_BETA toward the new value.
    const folded = D.mtpEmaMs(45.0, 40_000_000); // 40 ms sample
    try testing.expectApproxEqAbs(45.0 + D.MTP_EV_COST_BETA * (40.0 - 45.0), folded, 1e-3);
    // Env parse: default on, "0" off, "" on.
    try testing.expect(D.mtpLiveCostEnabledFromEnv(null));
    try testing.expect(!D.mtpLiveCostEnabledFromEnv("0"));
    try testing.expect(D.mtpLiveCostEnabledFromEnv(""));
}

test "batched corrections: point-mass residuals sample deterministically, accept vec gathers draft probs" {
    try testing.expect(Generator.mtpBatchCorrEnabledFromEnv(null));
    try testing.expect(!Generator.mtpBatchCorrEnabledFromEnv("0"));

    const s = mlx.gpuStream();
    // V=4, m=2. Row residuals are POINT MASSES so the batched categorical
    // is deterministic end-to-end (no seed dependence):
    //   row0: p=[.5,.5,0,0], draft=0 → residual [0,.5,0,0] → corr 1
    //   row1: p=[0,.25,.75,0], draft=2 → residual [0,.25,0,0] → corr 1
    //   bonus: p=[0,0,0,1] → corr 3
    const p_data = [_]f32{ 0.5, 0.5, 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0, 1.0 };
    const p_shape = [_]c_int{ 1, 3, 4 };
    const probs_all = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs_all);
    const id_shape = [_]c_int{1};
    const d0_data: i32 = 0;
    const d1_data: i32 = 2;
    const d0 = mlx.mlx_array_new_data(&d0_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d0);
    const d1 = mlx.mlx_array_new_data(&d1_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d1);
    const drafts = [_]mlx.mlx_array{ d0, d1 };

    // Greedy proposals (q == null): one-hot residuals.
    var g = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, null, 2, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr[0]);
    try testing.expectEqual(@as(i32, 1), corr[1]);
    try testing.expectEqual(@as(i32, 3), corr[2]);
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    const ap = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.5), ap[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), ap[1], 1e-6);
    try testing.expect(g.accept_q.ctx == null);

    // Sharp proposals: q rows equal to the one-hots reproduce the same
    // residuals, and accept_q gathers the proposal's own density.
    const q_data0 = [_]f32{ 1, 0, 0, 0 };
    const q_data1 = [_]f32{ 0, 0, 1, 0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q0 = mlx.mlx_array_new_data(&q_data0, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q0);
    const q1 = mlx.mlx_array_new_data(&q_data1, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q1);
    const qs = [_]mlx.mlx_array{ q0, q1 };
    var gs = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, &qs, 2, s);
    defer gs.deinit();
    try mlx.check(mlx.mlx_array_eval(gs.corr_samples));
    const corr2 = mlx.mlx_array_data_int32(gs.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr2[0]);
    try testing.expectEqual(@as(i32, 1), corr2[1]);
    try testing.expectEqual(@as(i32, 3), corr2[2]);
    try mlx.check(mlx.mlx_array_eval(gs.accept_q));
    const aq = mlx.mlx_array_data_float32(gs.accept_q) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 1.0), aq[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), aq[1], 1e-6);
}

test "mtpEvPlanFor: cap 1 is a plain depth-1 round" {
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const a = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const plan = Generator.mtpEvPlanFor(&a, 1, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 1), plan.m_hi);
}

test "mtpCommittedLen: speculative tails (pre-draft, stash) bound the committable history" {
    try testing.expectEqual(@as(usize, 42), Generator.mtpCommittedLen(42, null, null));
    // Pending stash: the cache past off0 is the producing round's stale draft tail.
    try testing.expectEqual(@as(usize, 40), Generator.mtpCommittedLen(46, null, 40));
    // Built pre-draft: next-round draft entries sit past ITS off0.
    try testing.expectEqual(@as(usize, 43), Generator.mtpCommittedLen(47, 43, null));
    // Both live: the min wins.
    try testing.expectEqual(@as(usize, 40), Generator.mtpCommittedLen(47, 43, 40));
}

test "mtpRoundOff0: a pending history stash overrides the stale cache length" {
    // No stash: the head cache is fully committed — its step IS the origin.
    try testing.expectEqual(@as(usize, 42), Generator.mtpRoundOff0(null, 42));
    // Pending stash: the cache still carries the previous round's draft tail
    // (step is stale/uncommitted), so the origin is where the stash's entries
    // will END once the consume-time truncate + merged forward run. A round
    // that read mc.step here would draft at the WRONG RoPE offsets.
    const st = Generator.MtpHistStash{
        .ids = .{ .ctx = null },
        .hidden = .{ .ctx = null },
        .n = 3, // t1 + 2 accepted drafts
        .off0 = 40,
    };
    // cache_step (46 = 40 committed + 6 stale draft entries) must be ignored.
    try testing.expectEqual(@as(usize, 43), Generator.mtpRoundOff0(st, 46));
}

test "mtpEvObserve: conditional EMA updates hit accepted indices, the reject index, and nothing past it" {
    var a = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    // 3 drafted, 1 accepted: index 0 saw a success, index 1 saw the reject,
    // index 2 was never conditionally reached (no observation).
    Generator.mtpEvObserve(&a, 3, 1, 0.15);
    try testing.expectApproxEqAbs(@as(f32, 0.575), a[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.425), a[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a[3], 1e-5);
    // Full accept: every drafted index saw a success, none saw a reject.
    var b = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    Generator.mtpEvObserve(&b, 2, 2, 0.15);
    try testing.expectApproxEqAbs(@as(f32, 0.575), b[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.575), b[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), b[2], 1e-5);
}

test "mtpChainLogConf: sums clamped log-confidences; NaN can never pass a gate" {
    try testing.expectApproxEqAbs(@as(f32, -0.3), Generator.mtpChainLogConf(&[_]f32{ -0.1, -0.2 }), 1e-5);
    // Positive numeric noise clamps to 0 (a log-prob is never > 0).
    try testing.expectApproxEqAbs(@as(f32, -0.1), Generator.mtpChainLogConf(&[_]f32{ 0.05, -0.1 }), 1e-5);
    // NaN -> -inf: `chain >= tau_ln` is false for every finite tau.
    const nan_chain = Generator.mtpChainLogConf(&[_]f32{ -0.1, std.math.nan(f32) });
    try testing.expect(nan_chain == -std.math.inf(f32));
    try testing.expect(!(nan_chain >= @log(@as(f32, 0.05))));
}

test "MtpTrace: per-phase accumulation, round averaging, log cadence, reset" {
    var t = Generator.MtpTrace{};
    // 2 rounds: draft 2ms+4ms, eval 10ms+30ms.
    t.add(.draft, 2_000_000);
    t.add(.eval, 10_000_000);
    try testing.expect(!t.endRound(3, 2, false));
    t.add(.draft, 4_000_000);
    t.add(.eval, 30_000_000);
    try testing.expect(!t.endRound(7, 6, true));
    try testing.expectApproxEqAbs(@as(f64, 3.0), t.avgMs(.draft), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.eval), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 23.0), t.totalAvgMs(), 1e-9);
    try testing.expectEqual(@as(u64, 10), t.drafted);
    try testing.expectEqual(@as(u64, 8), t.accepted);
    try testing.expectEqual(@as(u32, 1), t.extended);
    // Log line falls due exactly at LOG_EVERY rounds.
    var i: u32 = 2;
    while (i < Generator.MtpTrace.LOG_EVERY - 1) : (i += 1) {
        try testing.expect(!t.endRound(1, 1, false));
    }
    try testing.expect(t.endRound(1, 1, false));
    t.reset();
    try testing.expectEqual(@as(u32, 0), t.rounds);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.draft), 1e-9);
}

test "buildPaddedBatch pads to max length with zeros and records lengths" {
    const seqs = [_][]const u32{
        &[_]u32{ 101, 7592, 102 },
        &[_]u32{ 101, 102 },
    };
    var pb = try buildPaddedBatch(testing.allocator, &seqs);
    defer pb.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), pb.max_len);
    try testing.expectEqual(@as(usize, 2), pb.lengths.len);
    try testing.expectEqual(@as(usize, 3), pb.lengths[0]);
    try testing.expectEqual(@as(usize, 2), pb.lengths[1]);
    const expected = [_]i32{ 101, 7592, 102, 101, 102, 0 };
    try testing.expectEqualSlices(i32, &expected, pb.ids);
}

test "buildKeyPadMask is additive zero on real keys, -inf on padding" {
    const s = mlx.gpuStream();
    const lengths = [_]usize{ 3, 1 };
    const mask = try buildKeyPadMask(testing.allocator, &lengths, 3, s);
    defer _ = mlx.mlx_array_free(mask);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 1, 1, 3 }, mlx.getShape(mask));
    var f32_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_mask);
    try mlx.check(mlx.mlx_astype(&f32_mask, mask, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32_mask));
    const data = mlx.mlx_array_data_float32(f32_mask).?;
    // Batch row 0: all three keys real.
    try testing.expectEqual(@as(f32, 0), data[0]);
    try testing.expectEqual(@as(f32, 0), data[1]);
    try testing.expectEqual(@as(f32, 0), data[2]);
    // Batch row 1: one real key, two padded.
    try testing.expectEqual(@as(f32, 0), data[3]);
    try testing.expect(std.math.isInf(data[4]) and data[4] < 0);
    try testing.expect(std.math.isInf(data[5]) and data[5] < 0);
}

test "maskedMeanPoolNormalize excludes padded positions and unit-normalizes" {
    const s = mlx.gpuStream();
    // hidden [2, 3, 2]; row 0 has 2 real positions (pad slot holds garbage
    // that must not leak into the pool), row 1 has 3.
    const data = [_]f32{
        1, 0, 3, 4, 100, 100,
        0, 2, 0, 4, 0,   6,
    };
    const shape = [_]c_int{ 2, 3, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{ 2, 3 };
    const rows = try maskedMeanPoolNormalize(testing.allocator, hidden, &lengths, s);
    defer {
        for (rows) |r| testing.allocator.free(r);
        testing.allocator.free(rows);
    }
    // Row 0: mean of (1,0),(3,4) = (2,2) → L2-normalized (1/√2, 1/√2).
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][1], 1e-4);
    // Row 1: mean of (0,2),(0,4),(0,6) = (0,4) → normalized (0, 1).
    try testing.expectApproxEqAbs(@as(f32, 0.0), rows[1][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), rows[1][1], 1e-4);
}

test "embedSubBatchEnd: item cap, padded-footprint budget, oversize singleton, order" {
    const short = [_]u32{ 1, 2 };
    const long: [600]u32 = @splat(0);
    // Item cap alone (all short): full batch.
    const all_short = [_][]const u32{ &short, &short, &short };
    try testing.expectEqual(@as(usize, 3), embedSubBatchEnd(&all_short, 0, 64, 64 * 512));
    try testing.expectEqual(@as(usize, 2), embedSubBatchEnd(&all_short, 0, 2, 64 * 512));

    // One long input caps how many rows pad to its length: 600-token rows fit
    // budget/600 = 54 per sub-batch at the default budget, not 64.
    const many_long: [60][]const u32 = @splat(&long);
    const end = embedSubBatchEnd(&many_long, 0, 64, 64 * 512);
    try testing.expectEqual(@as(usize, (64 * 512) / 600), end);

    // A mixed batch stops BEFORE a long input would inflate the padded
    // footprint of everything before it.
    const mixed = [_][]const u32{ &short, &short, &long };
    const mixed_end = embedSubBatchEnd(&mixed, 0, 64, 4);
    try testing.expectEqual(@as(usize, 2), mixed_end);

    // An input larger than the whole budget still runs — alone.
    const huge: [40000]u32 = @splat(0);
    const with_huge = [_][]const u32{ &huge, &short };
    try testing.expectEqual(@as(usize, 1), embedSubBatchEnd(&with_huge, 0, 64, 64 * 512));
    // ...and the next sub-batch starts right after it (order preserved).
    try testing.expectEqual(@as(usize, 2), embedSubBatchEnd(&with_huge, 1, 64, 64 * 512));
}

test "gatherTokenPool: cls takes position 0, last_token takes the last real position" {
    const s = mlx.gpuStream();
    // hidden [2, 3, 2]; row 0 has 2 real positions (position 2 is pad garbage
    // that must never be selected), row 1 has 3.
    const data = [_]f32{
        1, 0, 3, 4, 100, 100,
        0, 2, 0, 4, 0,   6,
    };
    const shape = [_]c_int{ 2, 3, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{ 2, 3 };

    const cls = try gatherTokenPool(testing.allocator, hidden, &lengths, .cls, s);
    defer _ = mlx.mlx_array_free(cls);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 2 }, mlx.getShape(cls));
    try mlx.check(mlx.mlx_array_eval(cls));
    const cls_data = mlx.mlx_array_data_float32(cls).?;
    try testing.expectEqual(@as(f32, 1), cls_data[0]); // hidden[0,0,:]
    try testing.expectEqual(@as(f32, 0), cls_data[1]);
    try testing.expectEqual(@as(f32, 0), cls_data[2]); // hidden[1,0,:]
    try testing.expectEqual(@as(f32, 2), cls_data[3]);

    const last = try gatherTokenPool(testing.allocator, hidden, &lengths, .last_token, s);
    defer _ = mlx.mlx_array_free(last);
    try mlx.check(mlx.mlx_array_eval(last));
    const last_data = mlx.mlx_array_data_float32(last).?;
    try testing.expectEqual(@as(f32, 3), last_data[0]); // hidden[0,1,:] — NOT the pad slot
    try testing.expectEqual(@as(f32, 4), last_data[1]);
    try testing.expectEqual(@as(f32, 0), last_data[2]); // hidden[1,2,:]
    try testing.expectEqual(@as(f32, 6), last_data[3]);
}

test "gatherTokenPool: mean mode is not a gather — callers must dispatch it to maskedMeanPool" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1, 2 };
    const shape = [_]c_int{ 1, 1, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{1};
    try testing.expectError(error.InvalidPoolingMode, gatherTokenPool(testing.allocator, hidden, &lengths, .mean, s));
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

// ── Allocator-cache clear cadence (issue #110) ───────────────────────────────

test "clear cadence survives variable spec strides" {
    // A spec-decode round emits `1 + accepted` tokens, so `step` advances by a
    // VARIABLE amount and `step % 256 == 0` can walk clean over every multiple
    // — silently, because a decode path that never clears is output-identical
    // to one that does. Interval arithmetic against the last clear cannot be
    // stepped over.
    for ([_]u32{ 1, 3, 5, 9 }) |stride| {
        var step: u32 = 0;
        var last_clear: u32 = 0;
        var clears: usize = 0;
        var max_gap: u32 = 0;
        while (step < 4096) {
            step += stride;
            if (shouldClearAllocatorCache(step, last_clear, CACHE_CLEAR_INTERVAL)) {
                const gap = step - last_clear;
                if (gap > max_gap) max_gap = gap;
                last_clear = step;
                clears += 1;
            }
        }
        try testing.expect(clears >= 15);
        // A stride can overshoot the interval by at most stride-1 tokens.
        try testing.expect(max_gap <= CACHE_CLEAR_INTERVAL + stride);
    }
}

test "no decode path advances `step` outside advanceStep" {
    // Class guard for #110. `Generator.step` is the clear cadence's clock, so a
    // path that bumps it by hand is a path that strands its round's transients
    // in MLX's buffer pool forever — which is exactly what `nextDrafter`,
    // `nextMtp` and `scheduler.runBatchedDecodeTick` did, and the `-mtp`
    // checkpoints default onto one of those. A new decode path added later must
    // route through `advanceStep` or fail here.
    //
    // The needles are concatenated so this test's OWN source doesn't match them.
    const needle = ".step" ++ " +=";
    const allowed = "self.step" ++ " += n";
    const sources = [_]struct { name: []const u8, src: []const u8 }{
        .{ .name = "generate.zig", .src = @embedFile("generate.zig") },
        .{ .name = "scheduler.zig", .src = @embedFile("scheduler.zig") },
    };
    var total: usize = 0;
    for (sources) |file| {
        var it = std.mem.splitScalar(u8, file.src, '\n');
        var lineno: usize = 0;
        while (it.next()) |raw| {
            lineno += 1;
            // Strip trailing line comments: several of them spell out
            // `cache.step += 1` while describing the KV cache's own counter.
            const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
            if (std.mem.indexOf(u8, line, needle) == null) continue;
            total += 1;
            if (std.mem.indexOf(u8, line, allowed) == null) {
                std.debug.print("{s}:{d}: raw step advance outside advanceStep: {s}\n", .{
                    file.name, lineno, std.mem.trim(u8, line, " "),
                });
                return error.RawStepAdvance;
            }
        }
    }
    // Exactly one: the assignment inside `advanceStep`. Zero means the field was
    // renamed and this guard went vacuous — update it with the rename.
    try testing.expectEqual(@as(usize, 1), total);
}

test "dsv4: nextPld on a chokepoint-disabled generator stays serial (DSV4_MINI)" {
    // DSpark is opt-in at load; the nextDspark arm below needs it armed.
    _ = setenv("MLX_SERVE_DSV4_DSPARK", "1", 1);
    // The live corruption path (2026-07-31, log 166348-166361): the scheduler
    // decode tick dispatched on `slot.enable_pld` alone, so it called
    // `nextPld` on a generator whose init the dsv4 guard had already flipped
    // to pld_enabled=false — and nextPld trusted its caller, ran lookup +
    // verify forwards, and the rejected drafts left dsv4's module-owned
    // state (rings + kv/comp caches) permanently ahead of the rolled-back
    // KVCache shell. This test reproduces the bypassing caller directly:
    // nextPld on such a generator must (a) report the chokepoint flip via
    // `gen.pld_enabled == false`, (b) never run a verify forward
    // (`pld_attempted == 0`), and (c) emit the exact serial-decode tokens.
    //
    // Fabricate the mini with:
    //   python3 tests/dsv4_mlx_ref.py --fabricate /tmp/dsv4-mini
    //   DSV4_MINI=/tmp/dsv4-mini zig build test -Dtest-filter=DSV4_MINI
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    if (mlx.noGpuBackend()) return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model_mod.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model_mod.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    // Never read by the Generator (stored only) — see the field comment.
    var tok_dummy: Tokenizer = undefined;

    // Prompt = every vocab id once. With key_len=1 any sampled t1 < V has an
    // earlier occurrence, so PLD's lookup ALWAYS proposes a draft — on the
    // pre-fix code that guarantees a verify forward (and the corruption);
    // random-content prompts can idle in the cold path and mask the bug
    // (exactly how the first two live requests read as "guards held").
    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i);
    const greedy = SamplingParams{ .temperature = 0.0 };
    const want: usize = 10;

    // Serial baseline: the regular scheduler shape (skip_lazy_preforward).
    var serial: [want]u32 = undefined;
    {
        var xfm = try Transformer.init(io, allocator, cfg, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, greedy, &.{}, .{
            .skip_lazy_preforward = true,
        });
        defer gen.deinit(allocator);
        var n: usize = 0;
        while (n < want) {
            const t = (try gen.next(allocator)) orelse break;
            serial[n] = t;
            n += 1;
        }
        try testing.expectEqual(want, n);
    }

    // Bypass arm: init asks for PLD (the app's always-on flags), the dsv4
    // chokepoint flips it off, and the caller drives nextPld anyway — the
    // scheduler tick's exact live behavior before specTickMode grew the
    // generator-state conjunct.
    {
        var xfm = try Transformer.init(io, allocator, cfg, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, greedy, &.{}, .{
            .pld_enabled = true,
            .skip_lazy_preforward = true,
            .lookup_prompt = &prompt,
        });
        defer gen.deinit(allocator);
        try testing.expect(!gen.pld_enabled);

        var pld_toks: [want]u32 = undefined;
        var n: usize = 0;
        while (n < want) {
            const r = (try gen.nextPld(allocator, 4, 1)) orelse break;
            defer allocator.free(r.tokens);
            for (r.tokens) |t| {
                if (n < want) {
                    pld_toks[n] = t;
                    n += 1;
                }
            }
        }
        try testing.expectEqual(want, n);
        try testing.expectEqual(@as(u64, 0), gen.pld_attempted);
        try testing.expectEqualSlices(u32, serial[0..], pld_toks[0..]);
    }

    // DSpark arm: the SAME app-shaped init (spec flags on, greedy) now arms
    // dsv4's own draft mode on a stage-bearing checkpoint. The tick driver
    // is nextDspark; the sequence must match serial (the batch-verify vs
    // single-token kernel-choice class allows a late near-tie flip on the
    // random mini — the module-level dsparkRound gate pins the loop itself,
    // this pins the GENERATOR wiring: engagement, step accounting, and the
    // shell-cache mirror).
    {
        var xfm = try Transformer.init(io, allocator, cfg, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, greedy, &.{}, .{
            .pld_enabled = true,
            .mtp_enabled = true,
            .skip_lazy_preforward = true,
            .lookup_prompt = &prompt,
        });
        defer gen.deinit(allocator);
        try testing.expect(!gen.pld_enabled);
        try testing.expect(gen.mtp == null);
        try testing.expect(gen.dspark_enabled);

        var ds_toks: [want]u32 = undefined;
        var n: usize = 0;
        while (n < want) {
            const r = (try gen.nextDspark(allocator)) orelse break;
            defer allocator.free(r.tokens);
            for (r.tokens) |t| {
                if (n < want) {
                    ds_toks[n] = t;
                    n += 1;
                }
            }
        }
        try testing.expectEqual(want, n);
        // ENGAGEMENT: silent serial fallback is output-identical — count rounds.
        try testing.expect(gen.dspark_attempted >= 1);
        // Shell cache mirrors the module state exactly.
        try testing.expectEqual(xfm.dsv4.?.dec_state.?.n, gen.ctx.cache.step);
        // Sequence agreement: exact up to the sanctioned near-tie window.
        var first_div: usize = want;
        for (0..want) |k| {
            if (serial[k] != ds_toks[k]) {
                first_div = k;
                break;
            }
        }
        if (first_div < 4) {
            std.debug.print("nextDspark serial={any} dspark={any}\n", .{ serial, ds_toks });
            try testing.expect(false);
        }
        std.debug.print("dsv4 nextDspark (generator): first_div={d}/{d}, rounds={d}, accepts={d}\n", .{ first_div, want, gen.dspark_attempted, gen.dspark_accepted_tokens });
    }
}

test "dsparkArmFor: greedy and stochastic arms gate on clean sampling, kill switch restores greedy-only" {
    // Greedy clean → the argmax-equality arm (temp 0 or top_k 1), with or
    // without the stochastic arm enabled.
    try testing.expectEqual(Generator.DsparkArm.greedy, Generator.dsparkArmFor(.{ .temperature = 0.0 }, 0, true));
    try testing.expectEqual(Generator.DsparkArm.greedy, Generator.dsparkArmFor(.{ .temperature = 0.6, .top_k = 1 }, 0, true));
    try testing.expectEqual(Generator.DsparkArm.greedy, Generator.dsparkArmFor(.{ .temperature = 0.0 }, 0, false));
    // Sampled clean → stochastic (the checkpoint-default temp 0.6 agent
    // shape: pi and friends omit temperature, generation_config fills 0.6).
    try testing.expectEqual(Generator.DsparkArm.stochastic, Generator.dsparkArmFor(.{ .temperature = 0.6, .top_p = 0.95 }, 0, true));
    try testing.expectEqual(Generator.DsparkArm.stochastic, Generator.dsparkArmFor(.{ .temperature = 1.0, .top_k = 40 }, 0, true));
    // Kill switch: sampled requests fall back to serial-only.
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6, .top_p = 0.95 }, 0, false));
    // Penalties / grammar / logprobs stay serial on BOTH arms — the
    // pre-stochastic contract, unchanged.
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.0, .repeat_penalty = 1.1 }, 0, true));
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6, .presence_penalty = 0.5 }, 0, true));
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6 }, 5, true));
    var c: Constraint = undefined;
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6, .constraint = &c }, 0, true));
    // Env parser: unset/empty/anything-but-0 → on, "0" → off.
    try testing.expect(Generator.dsparkStochEnabledFromEnv(null));
    try testing.expect(Generator.dsparkStochEnabledFromEnv(""));
    try testing.expect(Generator.dsparkStochEnabledFromEnv("1"));
    try testing.expect(!Generator.dsparkStochEnabledFromEnv("0"));
}

test "dsv4: stochastic dspark engages at sampled temperature and keeps the exit invariant (DSV4_MINI)" {
    // The motivating traffic shape: the checkpoint ships generation_config
    // temp 0.6 and agent CLIs omit temperature, so every real agent request
    // ran serial while only pinned `--temp 0` earned DSpark. The stochastic
    // arm ports the MTP probsAllPositions acceptance (one-hot Leviathan over
    // filtered target probs) onto dsv4's own draft stages. Engagement is
    // COUNTED (dspark_attempted) — a silent serial fallback emits perfectly
    // plausible tokens.
    _ = setenv("MLX_SERVE_DSV4_DSPARK", "1", 1);
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    if (mlx.noGpuBackend()) return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model_mod.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model_mod.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    // Never read by the Generator (stored only) — see the field comment.
    var tok_dummy: Tokenizer = undefined;

    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i);
    // The agent-default request shape, seeded so the run is reproducible.
    const sampled = SamplingParams{ .temperature = 0.6, .top_p = 0.95, .seed = 0xD54A };

    // The kill-switch cache is process-global (set at first chokepoint use),
    // so honor whatever env this test binary was LAUNCHED with and assert
    // the matching behavior — that makes the `MLX_SERVE_DSV4_DSPARK_STOCH=0`
    // run a real test of the fallback, not a skip.
    const stoch_raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_DSV4_DSPARK_STOCH")) |p| std.mem.span(p) else null;
    const stoch_on = Generator.dsparkStochEnabledFromEnv(stoch_raw);

    var xfm = try Transformer.init(io, allocator, cfg, &weights);
    defer xfm.deinit();
    var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 64, sampled, &.{}, .{
        .pld_enabled = true,
        .mtp_enabled = true,
        .skip_lazy_preforward = true,
        .lookup_prompt = &prompt,
    });
    defer gen.deinit(allocator);

    if (!stoch_on) {
        // Kill-switch arm: the chokepoint declines the sampled request and
        // nextDspark serves the defensive serial step.
        try testing.expect(!gen.dspark_enabled);
        const r = (try gen.nextDspark(allocator)) orelse return error.TestExpectedResult;
        defer allocator.free(r.tokens);
        try testing.expectEqual(@as(usize, 1), r.tokens.len);
        try testing.expectEqual(@as(u64, 0), gen.dspark_attempted);
        return;
    }

    try testing.expect(gen.dspark_enabled);
    try testing.expect(gen.dspark_stochastic);

    const mdl = xfm.dsv4.?;
    const want: usize = 10;
    var n: usize = 0;
    while (n < want) {
        const t1 = gen.next_token_id;
        const r = (try gen.nextDspark(allocator)) orelse break;
        defer allocator.free(r.tokens);
        try testing.expect(r.tokens.len >= 1);
        try testing.expectEqual(t1, r.tokens[0]);
        try testing.expect(r.accepted_tokens <= mdl.ds_block);
        for (r.tokens) |tok| {
            try testing.expect(tok < mdl.vocab);
            n += 1;
        }
        // Module state and the shell-cache mirror stay in lockstep with
        // exactly what was committed (the v2 spec exit invariant).
        try testing.expectEqual(prompt.len + n, mdl.dec_state.?.n);
        try testing.expectEqual(mdl.dec_state.?.n, gen.ctx.cache.step);
    }
    try testing.expect(n >= want);
    // ENGAGEMENT: count rounds, never output shape.
    try testing.expect(gen.dspark_attempted >= 1);
    // Exit invariant survives a hand-off: a following serial step decodes a
    // valid token (finite logits) from the state the rounds left behind.
    const t = (try gen.next(allocator)) orelse return error.TestExpectedResult;
    try testing.expect(t < mdl.vocab);
    std.debug.print("dsv4 stochastic dspark (generator): {d} tokens over {d} rounds, {d} drafts accepted\n", .{ n, gen.dspark_attempted, gen.dspark_accepted_tokens });

    // b==0 arm: an over-threshold confidence gate submits NOTHING — the round
    // verifies t1 alone, commits it, and samples the next trunk token from
    // row 0's filtered probs (the stochastic sibling of the greedy
    // confidence-gate test). The env is read at initModel, so this needs a
    // fresh Transformer; unset after — test order must not inherit the gate.
    _ = setenv("MLX_SERVE_DSV4_DSPARK_CONF", "999999", 1);
    defer _ = unsetenv("MLX_SERVE_DSV4_DSPARK_CONF");
    var xfm2 = try Transformer.init(io, allocator, cfg, &weights);
    defer xfm2.deinit();
    var gen2 = try Generator.initWithOptions(io, allocator, &xfm2, &tok_dummy, &prompt, 64, sampled, &.{}, .{
        .pld_enabled = true,
        .mtp_enabled = true,
        .skip_lazy_preforward = true,
        .lookup_prompt = &prompt,
    });
    defer gen2.deinit(allocator);
    try testing.expect(gen2.dspark_enabled);
    var n2: usize = 0;
    while (n2 < 4) {
        const r = (try gen2.nextDspark(allocator)) orelse break;
        defer allocator.free(r.tokens);
        try testing.expectEqual(@as(u32, 0), r.accepted_tokens);
        try testing.expectEqual(@as(usize, 1), r.tokens.len);
        try testing.expect(r.tokens[0] < mdl.vocab);
        n2 += 1;
    }
    try testing.expectEqual(@as(usize, 4), n2);
    try testing.expect(gen2.dspark_attempted >= 4);
    try testing.expectEqual(xfm2.dsv4.?.dec_state.?.n, gen2.ctx.cache.step);
    std.debug.print("dsv4 stochastic dspark (b==0 gate): {d} single-token rounds\n", .{n2});
}

fn evalLazyToken(lazy: mlx.mlx_array) !u32 {
    defer _ = mlx.mlx_array_free(lazy);
    try mlx.check(mlx.mlx_array_eval(lazy));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
    return @intCast(val);
}

test "fillSuppressMask: padding rows past the tokenizer's last defined id are suppressed" {
    // A config's `vocab_size` is a padded matrix dimension, not a vocabulary.
    // qwen4_exp declares 248320 while the tokenizer defines 248044 + 33 =
    // 248077 ids: 243 rows that decode to NOTHING, carry whatever the
    // initializer left, and were sampleable — a drawn one emits no text while
    // consuming a step and entering the KV. Same mask, second row class.
    var buf: [16]bool = undefined;
    fillSuppressMask(&buf, &[_]u32{3}, 12);
    for (buf, 0..) |v, i| {
        const want = (i == 3) or (i >= 12);
        try testing.expectEqual(want, v);
    }

    // No padding known (0) and a fully-defined vocab (== len) both add
    // nothing: a mask over the WHOLE vocab is -inf everywhere, and argmax
    // over that row returns id 0 — the all-false-mask class from the other
    // side. The reserved id must still be the only true.
    for ([_]usize{ 0, 16, 99 }) |defined| {
        fillSuppressMask(&buf, &[_]u32{3}, defined);
        for (buf, 0..) |v, i| try testing.expectEqual(i == 3, v);
    }
}

test "Tokenizer.definedVocabSize: highest defined id + 1, 0 when nothing is defined" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .byte_level_bpe);
    defer tok.deinit();
    try testing.expectEqual(@as(usize, 0), tok.definedVocabSize());
    // Added specials sit ABOVE the base vocab and are not contiguous with it,
    // so the answer is the maximum key, never the entry count.
    try tok.id_to_token.put(0, "a");
    try tok.id_to_token.put(1, "b");
    try tok.id_to_token.put(11, "<|special|>");
    try testing.expectEqual(@as(usize, 12), tok.definedVocabSize());
}

test "suppress_mask: a suppressed id is unreachable from both samplers, everything else stays" {
    // A collapsed distribution can rank `<|fim_hole|>` (a reserved FIM marker) in the
    // top-5 at every degenerate position, and greedy DREW it live. A reserved
    // marker in chat output is always a bug, so the sampler carries a
    // model-lifetime suppression mask. Contract pinned here: with the mask,
    // neither sampler can return a suppressed id on any of its paths (greedy
    // fast path, greedy general, categorical, sync sampleToken); with a null
    // mask the same logits DO pick it (red-on-revert); and logprobs keep
    // reporting the MODEL's raw distribution — the suppressed id still ranks
    // where the model put it, because the field reports the model while the
    // mask is sampling policy.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const V: usize = 8;
    const mask = try buildSuppressMask(&[_]u32{3}, 0, V, s);
    defer _ = mlx.mlx_array_free(mask);

    // id 3 (suppressed) dominates, id 5 is the runner-up, the rest are so far
    // down that after masking the categorical underflows to exactly id 5.
    var data = [_]f32{ -1000, -1000, -1000, 100, -1000, 50, -1000, -1000 };

    // Greedy fast path: [1, 1, V].
    const shape3 = [_]c_int{ 1, 1, V };
    const logits3 = mlx.mlx_array_new_data(&data, &shape3, 3, .float32);
    defer _ = mlx.mlx_array_free(logits3);
    const masked_sp = SamplingParams{ .temperature = 0.0, .suppress_mask = mask };
    const open_sp = SamplingParams{ .temperature = 0.0 };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_sp, s)));
    try testing.expectEqual(@as(u32, 3), try evalLazyToken(sampleTokenLazy(logits3, open_sp, s)));

    // Categorical path at temp 1.0 (exercises the seq_len==1 reshape arm and
    // the sampler proper; the masked lane's probability is exactly 0 and the
    // -1000 lanes underflow, so the draw is deterministic).
    const masked_t1 = SamplingParams{ .temperature = 1.0, .suppress_mask = mask };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_t1, s)));

    // Sync sampler (logprobs/penalty path): same policy, and the reported
    // distribution stays the model's own — rank 1 is the SUPPRESSED id.
    const r = try sampleToken(allocator, logits3, masked_sp, null, 2, s);
    const lp = r.logprob_result orelse return error.NoLogprobs;
    defer allocator.free(lp.top_logprobs);
    try testing.expectEqual(@as(u32, 5), r.token_id);
    try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);

    // Stochastic-verify filters: a suppressed draft's acceptance probability
    // must be exactly 0 (always rejected), and the residual it corrects from
    // can never re-draw it.
    const probs = try probsAtLastPos(logits3, masked_t1, s);
    defer _ = mlx.mlx_array_free(probs);
    try testing.expectEqual(@as(f32, 0.0), try probAt(probs, 3, s));
    try testing.expect(try probAt(probs, 5, s) > 0.99);
}

test "computeLogprobs: rank 1 is the argmax and ranks descend, under a tie-saturated distribution" {
    // The producer used to recover token ids by scanning the whole vocab for
    // FLOAT EQUALITY with each `mlx_topk` value. Ties make that ambiguous, and
    // at temp 0 the post-temperature distribution saturates so ties are
    // everywhere — measured 0/5 positions where rank 1 was the chosen token on
    // a trivial prompt. The observable contract: with greedy sampling the
    // chosen token IS the argmax, so rank 1 must equal it.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    // Six exactly-tied entries, then a strict runner-up and a strict max.
    const data = [_]f32{ 1, 1, 1, 1, 1, 1, 2, 3 };
    const shape = [_]c_int{ 1, 8 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const chosen: u32 = 7; // argmax
    const r = try computeLogprobs(allocator, logits, chosen, 3, s);
    defer allocator.free(r.top_logprobs);

    try testing.expectEqual(@as(usize, 3), r.top_logprobs.len);
    try testing.expectEqual(chosen, r.top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 6), r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[2].token_id < 6);
    // Strictly descending, and ids distinct.
    try testing.expect(r.top_logprobs[0].logprob > r.top_logprobs[1].logprob);
    try testing.expect(r.top_logprobs[1].logprob > r.top_logprobs[2].logprob);
    try testing.expect(r.top_logprobs[0].token_id != r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[1].token_id != r.top_logprobs[2].token_id);
    // The chosen token's own logprob must agree with its rank-1 entry.
    try testing.expectApproxEqAbs(r.top_logprobs[0].logprob, r.token_logprob, 1e-6);
}

test "sampleToken: reported logprobs are the model's, not the client's temperature" {
    // Same prompt, same chosen token, three temperatures used to report three
    // different logprobs (-0.2129 / -0.0607 / -2.1566 at 0 / 0.6 / 2.0). The
    // value belongs to the model, so it must not move with a sampling knob.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const data = [_]f32{ 0, 1, 2, 3 };
    const shape = [_]c_int{ 1, 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // log_softmax([0,1,2,3])[3], computed independently.
    var denom: f64 = 0;
    for (data) |v| denom += @exp(@as(f64, v));
    const expect_top: f32 = @floatCast(3.0 - @log(denom));

    const temps = [_]f32{ 0.0, 0.6, 2.0 };
    for (temps) |t| {
        const sp = SamplingParams{ .temperature = t, .seed = 7 };
        const r = try sampleToken(allocator, logits, sp, null, 2, s);
        const lp = r.logprob_result orelse return error.NoLogprobs;
        defer allocator.free(lp.top_logprobs);
        try testing.expectEqual(@as(usize, 2), lp.top_logprobs.len);
        try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);
        try testing.expectEqual(@as(u32, 2), lp.top_logprobs[1].token_id);
        try testing.expectApproxEqAbs(expect_top, lp.top_logprobs[0].logprob, 1e-5);
    }
}

test "logprobs publish through the one-token delay, never straight from sampleToken" {
    // The decode loop returns `next_token_id` and, in the SAME call, forwards
    // it to sample its successor — so a path that publishes `sampleToken`'s
    // result as `last_logprob` shifts the whole array by one. Live symptom: a
    // one-token "OK" reply came back with `<|role_end|>` at rank 1, which reads
    // as a broken ranking rather than a broken pairing, and sent a real
    // quantization hunt down the wrong path for a day.
    //
    // The needles are concatenated so this test's OWN source doesn't match.
    const needle = "last_logprob" ++ " = ";
    const allowed = "last_logprob" ++ " = self.pending_logprob";
    const src = @embedFile("generate.zig");
    var it = std.mem.splitScalar(u8, src, '\n');
    var lineno: usize = 0;
    var seen: usize = 0;
    while (it.next()) |raw| {
        lineno += 1;
        const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
        if (std.mem.indexOf(u8, line, needle) == null) continue;
        // `= null` is the ownership hand-off to the scheduler, not a publish.
        if (std.mem.indexOf(u8, line, "= null") != null) continue;
        seen += 1;
        if (std.mem.indexOf(u8, line, allowed) == null) {
            std.debug.print("generate.zig:{d}: logprobs published without the one-token delay: {s}\n", .{
                lineno, std.mem.trim(u8, line, " "),
            });
            return error.LogprobPairingBypass;
        }
    }
    try testing.expect(seen >= 1);
}

test "dflash: nextDflash greedy equals serial decode, invariants exact each round" {
    // Hermetic end-to-end round loop on the tiny llama trunk + tiny DFlash
    // assistant (dflash.TinyFix). The bar is the spec-decode contract:
    //   - greedy dflash-on emits EXACTLY the serial greedy tokens (any bug in
    //     verify alignment, rollback truncate, correction indexing, or the
    //     anchor-row drop shifts the stream);
    //   - after EVERY round: trunk cache.step == prompt_len + emitted and the
    //     assistant context tracks it exactly (base + step == cache.step) at
    //     whatever accepted counts the round produced.
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_trunk = std.testing.tmpDir(.{});
    defer tmp_trunk.cleanup();
    var trunk_buf: [512]u8 = undefined;
    const trunk_len = try tmp_trunk.dir.realPath(io, &trunk_buf);
    const trunk_path = trunk_buf[0..trunk_len];
    try dflash_mod.TinyFix.writeTrunk(io, tmp_trunk.dir, trunk_path, s);

    var tmp_asst = std.testing.tmpDir(.{});
    defer tmp_asst.cleanup();
    var asst_buf: [512]u8 = undefined;
    const asst_len = try tmp_asst.dir.realPath(io, &asst_buf);
    const asst_path = asst_buf[0..asst_len];
    try dflash_mod.TinyFix.writeAssistant(io, tmp_asst.dir, asst_path, s);

    var config = try model_mod.parseConfig(io, allocator, trunk_path);
    var weights = try model_mod.loadWeights(io, allocator, trunk_path);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);

    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    const greedy = SamplingParams{ .temperature = 0.0 };
    const want: usize = 16;

    // Serial baseline.
    var serial: [want]u32 = undefined;
    {
        var xfm = try Transformer.init(io, allocator, config, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), greedy, &.{}, .{
            .skip_lazy_preforward = true,
        });
        defer gen.deinit(allocator);
        var n: usize = 0;
        while (n < want) {
            const t = (try gen.next(allocator)) orelse break;
            serial[n] = t;
            n += 1;
        }
        try testing.expectEqual(want, n);
    }

    // DFlash arm.
    {
        var xfm = try Transformer.init(io, allocator, config, &weights);
        defer xfm.deinit();
        var dm = try dflash_mod.loadDflash(io, allocator, s, asst_path);
        defer dm.deinit();
        try dm.bind(&xfm);

        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), greedy, &.{}, .{
            .dflash_enabled = true,
            .dflash = &dm,
            // This is the numeric DFlash guard, not the economics-gate test:
            // every emitted token must pass through nextDflash itself.
            .dflash_min_accepted_per_round = 0,
        });
        defer gen.deinit(allocator);
        try testing.expectEqual(@as(u32, 4), gen.dflash_block_size);
        // Prefill built context for the whole prompt.
        try testing.expectEqual(prompt.len, gen.dflash_ctx.?.absLen());
        try testing.expectEqual(prompt.len, gen.ctx.cache.step);

        var got = std.ArrayList(u32).empty;
        defer got.deinit(allocator);
        while (true) {
            const attempts_before = gen.dflash_attempted;
            const res = (try gen.nextDflash(allocator)) orelse break;
            defer allocator.free(res.tokens);
            try got.appendSlice(allocator, res.tokens);
            // A real DFlash round commits trunk and assistant context to the
            // same exact boundary at every accepted count. Falling back here
            // would make the remaining equality checks compare serial decode
            // with itself and gut the default-on draft-quantization guard.
            try testing.expect(gen.dflash_attempted != attempts_before);
            try testing.expectEqual(prompt.len + gen.generated_ids.items.len, gen.ctx.cache.step);
            try testing.expectEqual(gen.ctx.cache.step, gen.dflash_ctx.?.absLen());
        }
        try testing.expect(gen.dflash_attempted > 0);
        try testing.expect(!gen.spec_disabled_runtime);
        try testing.expectEqual(want, got.items.len);
        try testing.expectEqual(@as(u32, @intCast(want)), gen.completion_tokens);
        try testing.expectEqual(want, gen.generated_ids.items.len);
        for (serial, got.items) |a, b| try testing.expectEqual(a, b);
    }

    // DFlash2 arm: selector-traced greedy drafts + dyn-conv forward. The bar
    // is unchanged — a selector draft only survives verify when it IS the
    // trunk argmax, so the emitted stream must still be byte-identical to
    // serial at every accepted count.
    {
        var tmp_a2 = std.testing.tmpDir(.{});
        defer tmp_a2.cleanup();
        var a2_buf: [512]u8 = undefined;
        const a2_path = a2_buf[0..try tmp_a2.dir.realPath(io, &a2_buf)];
        try dflash_mod.TinyFix.writeAssistant2(io, tmp_a2.dir, a2_path, s);

        var xfm = try Transformer.init(io, allocator, config, &weights);
        defer xfm.deinit();
        var dm = try dflash_mod.loadDflash(io, allocator, s, a2_path);
        defer dm.deinit();
        try testing.expect(dm.selector != null);
        try dm.bind(&xfm);

        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), greedy, &.{}, .{
            .dflash_enabled = true,
            .dflash = &dm,
            .dflash_min_accepted_per_round = 0,
        });
        defer gen.deinit(allocator);

        var got = std.ArrayList(u32).empty;
        defer got.deinit(allocator);
        while (true) {
            const attempts_before = gen.dflash_attempted;
            const res = (try gen.nextDflash(allocator)) orelse break;
            defer allocator.free(res.tokens);
            try got.appendSlice(allocator, res.tokens);
            try testing.expect(gen.dflash_attempted != attempts_before);
            try testing.expectEqual(prompt.len + gen.generated_ids.items.len, gen.ctx.cache.step);
            try testing.expectEqual(gen.ctx.cache.step, gen.dflash_ctx.?.absLen());
        }
        try testing.expect(gen.dflash_attempted > 0);
        try testing.expectEqual(want, got.items.len);
        for (serial, got.items) |a, b| try testing.expectEqual(a, b);
    }
}

test "prefill chunk loop yields to the interleave hook between chunks, never after the last" {
    // The scheduler runs decode ticks for active streams at this seam so a
    // long prefill cannot stall them for its whole duration. The final
    // boundary is excluded: the post-prefill decode tick covers it.
    const src = @embedFile("generate.zig");
    const progress = "if (options.prefill_progress) |p| p.store(@intCast(pos), .monotonic);";
    const at = std.mem.indexOf(u8, src, progress) orelse return error.ProgressPublishMissing;
    const guard = "if (pos < loop_end) {";
    const guard_at = std.mem.indexOfPos(u8, src, at, guard) orelse return error.InterleaveGuardMissing;
    const call = "if (options.interleave" ++ "_hook) |hk| hk.call(hk.ctx);";
    const call_at = std.mem.indexOfPos(u8, src, guard_at, call) orelse return error.InterleaveCallMissing;
    // The call sits INSIDE the guard (same statement block, before the loop
    // re-tests `pos`), not somewhere later in the file.
    try std.testing.expect(call_at - guard_at < 200);
}

test "countSpliceRows counts image, audio and video placeholders, extras only when declared" {
    const ids = [_]i32{ 7, 99, 99, 8, 88, 9, 77 };
    try testing.expectEqual(@as(usize, 2), countSpliceRows(&ids, 99, 0, 0));
    try testing.expectEqual(@as(usize, 3), countSpliceRows(&ids, 99, 88, 0));
    try testing.expectEqual(@as(usize, 4), countSpliceRows(&ids, 99, 88, 77));
    try testing.expectEqual(@as(usize, 0), countSpliceRows(ids[0..1], 99, 88, 77));
}

test "vision prefill chunks by default and the splice offset feeds every prefill forward" {
    // Source scan: the chunk-size pick keys on the kill switch (not bare
    // has_vision), the chunk loop and the final-span forward both set
    // ctx.vision_splice_offset, and the loop advances the consumed-row count.
    const src = @embedFile("generate.zig");
    const marker = "const default_chunk = if (has_vision and !vision" ++ "_chunked) loop_end else PREFILL_CHUNK;";
    try testing.expect(std.mem.indexOf(u8, src, marker) != null);
    const set_off = "ctx.vision_splice" ++ "_offset = vision_rows_consumed;";
    const first = std.mem.indexOf(u8, src, set_off) orelse return error.ChunkOffsetMissing;
    try testing.expect(std.mem.indexOfPos(u8, src, first + 1, set_off) != null); // final-span site too
    try testing.expect(std.mem.indexOf(u8, src, "vision_rows_consumed += countSplice" ++ "Rows(") != null);
}

test "MTP verify forward defers the PLE gather" {
    // Source scan: on qwen4_exp the n-gram PLE at trunk layer 1 is a HOST
    // gather keyed on the verify ids, and every id past t1 is still a lazy
    // draft — an eager gather parks the build of layers 2..N behind the whole
    // draft chain finishing on the GPU. The verify build must therefore run
    // with `ple_defer` armed, hand the leaf back on the error path, and flush
    // it (the round's one host sync) before Phase 4 evaluates anything or
    // Phase 5 rolls the n-gram history back.
    const src = @embedFile("generate.zig");
    const build = "xfm.forwardWithCaptureAll(&self.ctx, verify" ++ "_input, &new_hidden, &verify_hidden_all)";
    const at = std.mem.indexOf(u8, src, build) orelse return error.VerifyBuildMissing;

    // Armed immediately before the build, disarmed right after it.
    const arm = "self.ctx.ple" ++ "_defer = true;";
    const arm_at = std.mem.lastIndexOf(u8, src[0..at], arm) orelse return error.PleDeferNotArmed;
    try testing.expect(at - arm_at < 600); // the comment block, nothing else
    const disarm = "self.ctx.ple" ++ "_defer = false;";
    try testing.expect(std.mem.indexOfPos(u8, src, at, disarm) != null);
    // The build's error path hands the unfilled leaf back.
    const discard = "xfm.discardDeferred" ++ "Ple(&self.ctx);";
    const discard_at = std.mem.indexOfPos(u8, src, at, discard) orelse return error.PleDiscardMissing;
    try testing.expect(discard_at - at < 400);
    // The flush follows the build and precedes Phase 4 (the first eval).
    const flush = "try xfm.flushDeferred" ++ "Ple(&self.ctx);";
    const flush_at = std.mem.indexOfPos(u8, src, at, flush) orelse return error.PleFlushMissing;
    const phase4 = std.mem.indexOfPos(u8, src, at, "// ── Phase 4:") orelse return error.Phase4Missing;
    try testing.expect(flush_at < phase4);
}

test "MtpTrace per-index acceptance: index i counts only rounds that drafted it" {
    var t: Generator.MtpTrace = .{};
    _ = t.endRound(3, 3, false); // all three accepted
    _ = t.endRound(3, 1, false); // a0 accepted, a1 rejected, a2 never judged
    _ = t.endRound(1, 0, false); // depth-1 round: only index 0 drafted
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("0.67/0.50/0.50", t.accIdxStr(&buf));
    t.reset();
    try std.testing.expectEqualStrings("", t.accIdxStr(&buf));
}

test "qwen4: the coarse rerank head is READY before any draft has run" {
    // The head is a full `requantizeRows` of the trunk lm_head plus a
    // synchronous eval of its ~240 MB. Built lazily it landed inside the
    // FIRST request's draft chain — on the inference thread, mid-round,
    // draining the stream, and paid as first-token latency by whoever
    // happened to be first after a load. `qwen4BuildDraftRerank` is that
    // build as a standalone LOAD-time step: no Generator, no chain, no
    // request. Afterwards `canRerankDrafts()` is a pure read.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();

    var fx = try mtp_mod.RerankFixture.init(s, mtp_mod.TOP32_MIN_ROWS + 96, 256, 8, 64, 0xC0DE);
    defer fx.deinit();
    // Only the rerank fields are read by the build; the rest of the head is
    // the trunk's business.
    var head: transformer_mod.Qwen4Mtp = undefined;
    head.rerank = null;
    head.rerank_logged = false;
    head.rerank_tried = false;
    fx.xfm.qwen4_mtp = head;
    defer if (fx.xfm.qwen4_mtp) |*m| {
        if (m.rerank) |*rc| rc.deinit();
    };

    // Nothing is built by construction...
    try testing.expect(fx.xfm.qwen4_mtp.?.rerank == null);
    // ...the eager build stands on its own...
    try testing.expect(fx.xfm.qwen4BuildDraftRerank());
    const built = fx.xfm.qwen4_mtp.?.rerank orelse return error.NoCoarseHead;
    try testing.expectEqual(fx.vocab, built.rows);
    try testing.expect(built.bits > 0 and built.bits < 8);

    // ...and the draft path now only READS it. Same mlx handle => the ask
    // did not rebuild anything (which is what the lazy path did).
    const head_ctx = built.q.w.ctx;
    const ref = MtpHeadRef{ .qwen4 = &fx.xfm };
    try testing.expect(ref.canRerankDrafts());
    try testing.expectEqual(head_ctx, fx.xfm.qwen4_mtp.?.rerank.?.q.w.ctx);

    // One-shot: a second eager build (both load paths firing, a re-probe)
    // keeps the head it already has.
    try testing.expect(fx.xfm.qwen4BuildDraftRerank());
    try testing.expectEqual(head_ctx, fx.xfm.qwen4_mtp.?.rerank.?.q.w.ctx);
}

test "every request-finalize seam that logs [spec-stats] also logs [qsa-arms]" {
    // Class guard for a wiring bug this test was written after: `logQsaArms`
    // was added beside the three `gen.logSpecStats()` calls in THIS file, which
    // are the legacy/CLI path only. The SERVER finalizes requests in
    // `scheduler.finishSlot` ("scheduler-driven slots finalize here instead"),
    // so `[qsa-arms]` was dead on every served request while looking wired.
    // A per-request meter is worth nothing if it misses the path that actually
    // serves requests, so the SEAM — not the symbol — is what gets pinned:
    // every call site of the request summary must carry both. Needles are
    // assembled at comptime so this test's own source cannot satisfy the scan.
    const spec_call = "logSpec" ++ "Stats();";
    const qsa_call = "logQsa" ++ "Arms();";
    inline for (.{ @embedFile("generate.zig"), @embedFile("scheduler.zig") }) |src| {
        var i: usize = 0;
        var seams: usize = 0;
        while (std.mem.indexOfPos(u8, src, i, spec_call)) |at| {
            seams += 1;
            // The paired call must sit in the same short block, not merely
            // somewhere else in the file.
            const win_end = @min(src.len, at + 400);
            if (std.mem.indexOfPos(u8, src, at, qsa_call)) |q| {
                if (q >= win_end) return error.SpecStatsSeamMissesQsaArms;
            } else return error.SpecStatsSeamMissesQsaArms;
            i = at + spec_call.len;
        }
        try std.testing.expect(seams > 0);
    }
}

test "nextMtp consults the --max-mtp-ctx ceiling at round entry, before drafting" {
    const src = @embedFile("generate.zig");

    // The check lives INSIDE nextMtp and ahead of its `spec_disabled_runtime`
    // short-circuit, so a crossing lands on the tested AR-fallback hand-off
    // (which frees `mtp_pre_draft`) instead of a second, parallel bail-out.
    const fn_at = std.mem.indexOf(u8, src, "pub fn next" ++ "Mtp(self: *Generator") orelse
        return error.NextMtpMissing;
    const needle = "mtpCtx" ++ "WithinLimit(max_mtp_ctx, self.mtpKvLen())";
    const check_at = std.mem.indexOfPos(u8, src, fn_at, needle) orelse
        return error.CeilingCheckMissing;
    const bail_at = std.mem.indexOfPos(u8, src, fn_at, "if (self.spec_disabled" ++ "_runtime or self.mtp_serial_left") orelse
        return error.RuntimeBailMissing;
    try testing.expect(check_at < bail_at);

    // It disables through the EXISTING runtime path and names its reason, so
    // `[spec-stats]` can tell an operator ceiling from a bad acceptance rate.
    const window = src[check_at..bail_at];
    try testing.expect(std.mem.indexOf(u8, window, "spec_disabled" ++ "_runtime = true") != null);
    try testing.expect(std.mem.indexOf(u8, window, "spec_disable" ++ "_reason = .max_ctx") != null);

    // The reason rides the [spec-stats] mtp line.
    try testing.expect(std.mem.indexOf(u8, src, "runtime_disabled={s} reason={s}") != null);

    // MTP only: the ceiling must not appear in the PLD / drafter / DFlash
    // round entries (they have their own economics and their own gates).
    for ([_][]const u8{ "fn next" ++ "Pld(", "fn next" ++ "Drafter(", "fn next" ++ "Dflash(" }) |name| {
        const at = std.mem.indexOf(u8, src, name) orelse continue;
        const end = @min(src.len, at + 4096);
        try testing.expect(std.mem.indexOf(u8, src[at..end], "max_mtp_ctx") == null);
    }
}

test "mtpAdaptiveVoteFor: a switch needs BOTH measured prices past the margin" {
    const G = Generator;
    const margin = G.MTP_ADAPTIVE_MARGIN;
    // Both prices past the bar (serial 10 -> bar 10.5): serial wins.
    try testing.expectEqual(G.MtpAdaptiveVote.serial, G.mtpAdaptiveVoteFor(12.0, 12.0, 10.0, margin));
    // Either price inside the bar keeps speculation. This is the whole point
    // of requiring two: the table's `tok` column is a workload MIXTURE and
    // the window is per-request, so each can be wrong alone.
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(12.0, 9.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(9.0, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(9.0, 9.0, 10.0, margin));
    // The band: serial 11.5 puts the bar at 12.075, so 12.0 is a TIE on both
    // and stays on MTP; serial 11.4 (bar 11.97) is a loss on both.
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(12.0, 12.0, 11.5, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.serial, G.mtpAdaptiveVoteFor(12.0, 12.0, 11.4, margin));
    // A zero margin still declines an exact tie (strictly worse only).
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(12.0, 12.0, 12.0, 0.0));

    // Any missing or degenerate input is UNDECIDED — never a switch. An
    // unfilled window is a null, so a request cannot switch before it has
    // priced itself.
    const nan = std.math.nan(f32);
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(null, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, null, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, null, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(0.0, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 0.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, 0.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(nan, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, nan, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, nan, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, 10.0, nan));
}

test "mtpAdaptiveVoteFor: the four real switch events of the 2026-09-04 qwen4 A/B" {
    // Every row is a switch this controller ACTUALLY made on the 62.7k prose
    // prompt (root_adapt, 4 boots), read out of the server logs. On all four
    // MTP was the faster arm (56.8 tok/s against serial's 51.6), so the
    // correct vote is `mtp` every time.
    //
    // v1 priced MTP as measuredMs(m_lo) / mtpEvExpectedTokens(...) and voted
    // `serial` on all four. The modeled token count ran 12-31% under the
    // SAME cell's measured `tok`:
    //
    //   event     w   modeled exp   cell tok   error
    //   b1 long1  2      1.81         2.28      -21%
    //   b1 long2  2      1.81         2.32      -22%
    //   b4 long1  4      1.84         2.67      -31%
    //   b4 long2  1      1.46         1.66      -12%
    const G = Generator;
    const margin = G.MTP_ADAPTIVE_MARGIN;
    const Event = struct {
        name: []const u8,
        table_ms_tok: f32, // Table.msPerTok(m_lo, bucket), from `table=` in [spec-stats]
        serial_ms_tok: f32, // serial_cell= at the moment of the switch
        window_ms_tok: f32, // round_ms EMA / (avg_per_round + 1)
        window_full: bool, // rounds run before the vote >= MTP_PRICE_WINDOW
    };
    const events = [_]Event{
        // round_ms 38.97 / (1.53 + 1) = 15.40
        .{ .name = "b1 long1", .table_ms_tok = 16.12, .serial_ms_tok = 19.20, .window_ms_tok = 15.40, .window_full = true },
        // The one v1 rule B would ALSO have got wrong: it voted at round ~3,
        // and all 9 of its rounds were prefix-cache-restore warmup
        // (avg_per_round 0.89, round_ms 45.69 -> 24.17 ms/tok). The
        // full-window gate is what saves it — the window is not readable yet.
        .{ .name = "b1 long2", .table_ms_tok = 15.85, .serial_ms_tok = 18.16, .window_ms_tok = 24.17, .window_full = false },
        // round_ms 41.74 / (1.51 + 1) = 16.63
        .{ .name = "b4 long1", .table_ms_tok = 17.28, .serial_ms_tok = 19.53, .window_ms_tok = 16.63, .window_full = true },
        // round_ms 34.49 / (1.16 + 1) = 15.97
        .{ .name = "b4 long2", .table_ms_tok = 17.91, .serial_ms_tok = 17.08, .window_ms_tok = 15.97, .window_full = true },
    };
    for (events) |e| {
        // Rule A alone is right on all four: the measured cell says MTP.
        try testing.expectEqual(
            G.MtpAdaptiveVote.mtp,
            G.mtpAdaptiveVoteFor(e.table_ms_tok, e.table_ms_tok, e.serial_ms_tok, margin),
        );
        // The shipped rule, with the window gated on being full. What must
        // hold on every row is that it does NOT vote serial; an unfilled
        // window is `.undecided`, which is stronger than `.mtp` — the
        // controller cannot even evaluate the round, and `MtpAdaptive.round`
        // clears the confirm streak on it. Both outcomes leave speculation on.
        const win: ?f32 = if (e.window_full) e.window_ms_tok else null;
        const got = G.mtpAdaptiveVoteFor(e.table_ms_tok, win, e.serial_ms_tok, margin);
        try testing.expect(got != .serial);
        try testing.expectEqual(
            if (e.window_full) G.MtpAdaptiveVote.mtp else G.MtpAdaptiveVote.undecided,
            got,
        );
    }
    // And the guard against over-fitting: rule B UNGATED reproduces v1's
    // b1-long2 mistake, which is exactly why the full-window gate exists.
    const b1l2 = events[1];
    try testing.expectEqual(
        G.MtpAdaptiveVote.serial,
        G.mtpAdaptiveVoteFor(30.0, b1l2.window_ms_tok, b1l2.serial_ms_tok, margin),
    );
}

test "mtpAdaptiveBucketOf: the decision and the re-entry resolve ONE bucket (H6 oscillation)" {
    const G = Generator;
    const kv: u32 = 40_000;
    const own = round_cost.bucketFor(kv);

    // No measured neighbour: both sites land on the request's own bucket.
    try testing.expectEqual(own, G.mtpAdaptiveBucketOf(null, kv, .long));
    // A measured NEIGHBOUR is what the prices came from, so it is the bucket
    // the switch speaks in — at BOTH sites.
    try testing.expectEqual(@as(usize, own + 1), G.mtpAdaptiveBucketOf(own + 1, kv, .long));
    try testing.expect(G.mtpAdaptiveBucketOf(own + 1, kv, .long) != round_cost.bucketFor(kv));

    // The oscillation the mismatch produced: a switch decided in the read
    // bucket, followed by a serial tick that resolved the bucket the OTHER
    // way, reads as a crossing and re-enters immediately — then the next
    // round switches again, each lap paying the drain + capture ramp.
    var osc = G.MtpAdaptive{};
    _ = osc.round(0, G.mtpAdaptiveBucketOf(own + 1, kv, .long), .serial, 1);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, osc.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, osc.serialTick(round_cost.bucketFor(kv), 0));

    // With one resolver the switch is sticky, which is the contract.
    var ok = G.MtpAdaptive{};
    const b = G.mtpAdaptiveBucketOf(own + 1, kv, .long);
    _ = ok.round(0, b, .serial, 1);
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, ok.serialTick(G.mtpAdaptiveBucketOf(own + 1, kv, .long), 0));
    }
    try testing.expectEqual(G.MtpAdaptiveArm.serial, ok.arm);
}

test "mtpAdaptiveRegimeMoved: a switch either way, or a crossing, drops the price window (H7)" {
    const G = Generator;
    const A = G.MtpAdaptiveArm;
    // Nothing moved.
    try testing.expect(!G.mtpAdaptiveRegimeMoved(null, .undecided, null, .undecided));
    try testing.expect(!G.mtpAdaptiveRegimeMoved(3, .mtp, 3, .mtp));
    // Arm moved — BOTH directions. The window priced the arm the request has
    // just left, so 3 fresh rounds must not re-decide from 13 stale ones.
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.mtp, 3, A.serial));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.serial, 3, A.undecided));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.undecided, 3, A.mtp));
    // Bucket moved, including in and out of "not yet decided anywhere".
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.mtp, 4, A.mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(null, A.mtp, 4, A.mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.mtp, null, A.mtp));

    // The transitions the two call sites actually produce.
    var a = G.MtpAdaptive{};
    var pb = a.bucket;
    var pa = a.arm;
    _ = a.round(0, 5, .serial, 1); // first vote: bucket null -> 5, arm -> serial
    try testing.expect(G.mtpAdaptiveRegimeMoved(pb, pa, a.bucket, a.arm));
    pb = a.bucket;
    pa = a.arm;
    _ = a.serialTick(6, 0); // crossing: re-opens the decision
    try testing.expect(G.mtpAdaptiveRegimeMoved(pb, pa, a.bucket, a.arm));
    // A tick that changes nothing must NOT drop the window, or it can never fill.
    pb = a.bucket;
    pa = a.arm;
    _ = a.serialTick(6, 0);
    try testing.expect(!G.mtpAdaptiveRegimeMoved(pb, pa, a.bucket, a.arm));
}

test "serialCellWanted: --no-mtp must not fold a serial cell (the head weights are still loaded)" {
    const src = @embedFile("generate.zig");
    const at = std.mem.indexOf(u8, src, "fn serialCell" ++ "Wanted(self: *const Generator)") orelse
        return error.MissingSerialCellWanted;
    const end = std.mem.indexOfPos(u8, src, at, "\n    }\n") orelse src.len;
    const body = src[at..end];

    // `Transformer.qwen4_mtp` is loaded with the TRUNK — `--no-mtp` gates
    // `entry.mtp`, not the weights — so "the head exists" is not the same
    // question as "this model may speculate", and using it as the proxy made
    // a `--no-mtp` boot fold a serial cell per token and persist the table.
    // The model-level answer is the one the registry already computed.
    // The CODE form, not the word: the comment above the predicate names the
    // rejected expression on purpose, and must stay allowed to.
    try testing.expect(std.mem.indexOf(u8, body, "self.xfm.qwen4_" ++ "mtp") == null);
    try testing.expect(std.mem.indexOf(u8, body, "self.model_has_" ++ "mtp") != null);
}

test "mtpAdaptiveKvEligible: short context is below the floor, and the floor is the default" {
    const G = Generator;
    const floor = G.MTP_ADAPTIVE_MIN_KV;
    try testing.expectEqual(@as(u32, 8192), floor);

    // llmprobe-shaped traffic: in the 2026-09-04 A/B, 11 of 14 switches were
    // in the `<2k` bucket and every one of them was a short llmprobe request.
    // Thirty of those in a boot must not reach the vote even once.
    var reached: u32 = 0;
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        // 60-token prompt, 200 tokens of reply: the whole request lives far
        // below the floor even at its longest.
        const kv: u32 = 60 + i * 7 + 200;
        if (G.mtpAdaptiveKvEligible(kv, floor)) reached += 1;
    }
    try testing.expectEqual(@as(u32, 0), reached);

    // The boundary is inclusive on the eligible side, and the contexts the
    // feature was actually argued for are all above it.
    try testing.expect(!G.mtpAdaptiveKvEligible(0, floor));
    try testing.expect(!G.mtpAdaptiveKvEligible(floor - 1, floor));
    try testing.expect(G.mtpAdaptiveKvEligible(floor, floor));
    try testing.expect(G.mtpAdaptiveKvEligible(62_755, floor)); // the A/B prose prompt
    try testing.expect(G.mtpAdaptiveKvEligible(374_000, floor)); // the ladder's top rung

    // The knob can be moved either way, including all the way open.
    try testing.expect(G.mtpAdaptiveKvEligible(1, 0));
    try testing.expect(!G.mtpAdaptiveKvEligible(62_755, 131_072));
}

test "MtpPriceWindow: reads nothing until FULL, skips width trials, prices per TOKEN" {
    const G = Generator;
    const N = G.MTP_PRICE_WINDOW;
    var w = G.MtpPriceWindow{};
    // A fresh window — every new request, and therefore every prefix-cache
    // restore — has no price at all.
    try testing.expect(w.msPerTok() == null);

    var i: u32 = 0;
    while (i < N - 1) : (i += 1) {
        w.observe(20.0, 2.0, false);
        try testing.expect(w.msPerTok() == null); // still not full
    }
    w.observe(20.0, 2.0, false);
    try testing.expectApproxEqAbs(@as(f32, 10.0), w.msPerTok().?, 1e-4);

    // Width trials never enter: a trial prices a width the plan rejected.
    var t = G.MtpPriceWindow{};
    i = 0;
    while (i < N * 2) : (i += 1) t.observe(999.0, 1.0, true);
    try testing.expect(t.msPerTok() == null);
    try testing.expectEqual(@as(u32, 0), t.filled);

    // Degenerate samples are dropped, not folded as zeros.
    var d = G.MtpPriceWindow{};
    d.observe(std.math.nan(f32), 2.0, false);
    d.observe(20.0, 0.0, false);
    d.observe(-1.0, 2.0, false);
    try testing.expectEqual(@as(u32, 0), d.filled);

    // Summed over the window, not averaged per round: a round is not a unit
    // of work. Eight rounds emitting 1 token in 30 ms and eight emitting 4 in
    // 60 ms cost (8*30 + 8*60) / (8*1 + 8*4) = 720/40 = 18.0 ms/tok — the
    // mean of the per-round ratios (30 and 15) would have said 22.5.
    var m = G.MtpPriceWindow{};
    i = 0;
    while (i < N / 2) : (i += 1) m.observe(30.0, 1.0, false);
    while (i < N) : (i += 1) m.observe(60.0, 4.0, false);
    try testing.expectApproxEqAbs(@as(f32, 18.0), m.msPerTok().?, 1e-4);

    // The ring keeps only the last N: a reset, or N newer rounds, retires the
    // old regime entirely.
    i = 0;
    while (i < N) : (i += 1) m.observe(10.0, 2.0, false);
    try testing.expectApproxEqAbs(@as(f32, 5.0), m.msPerTok().?, 1e-4);
    m.reset();
    try testing.expect(m.msPerTok() == null);
}

test "MtpAdaptive: the switch needs CONFIRM consecutive rounds, counted once per round" {
    const G = Generator;
    const need = G.MTP_ADAPTIVE_CONFIRM;
    var a = G.MtpAdaptive{};
    var r: u32 = 0;
    while (r < need - 1) : (r += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, a.round(r, 3, .serial, need));
    }
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, a.round(need - 1, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);
    try testing.expectEqual(@as(u32, 1), a.switches);

    // `mtpRoundPlan` has two call sites per round: the second call at the
    // SAME round index must not advance the streak (it would trip in half
    // the rounds the confirm count is sized for).
    var b = G.MtpAdaptive{};
    var i: u32 = 0;
    while (i < need - 1) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, b.round(i, 3, .serial, need));
        try testing.expectEqual(G.MtpAdaptiveAction.none, b.round(i, 3, .serial, need));
    }
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, b.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, b.round(need - 1, 3, .serial, need));

    // One winning round resets the streak: a switch is a run, not a tally.
    var c = G.MtpAdaptive{};
    _ = c.round(0, 3, .serial, need);
    _ = c.round(1, 3, .serial, need);
    try testing.expectEqual(G.MtpAdaptiveAction.none, c.round(2, 3, .mtp, need));
    try testing.expectEqual(G.MtpAdaptiveArm.mtp, c.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.none, c.round(3, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveAction.none, c.round(4, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, c.round(5, 3, .serial, need));

    // Undecided is not a vote either way — it clears the streak.
    var d = G.MtpAdaptive{};
    _ = d.round(0, 3, .serial, need);
    _ = d.round(1, 3, .serial, need);
    try testing.expectEqual(G.MtpAdaptiveAction.none, d.round(2, 3, .undecided, need));
    try testing.expectEqual(G.MtpAdaptiveAction.none, d.round(3, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, d.arm);

    // A crossing invalidates the streak: those votes priced another context.
    var e = G.MtpAdaptive{};
    _ = e.round(0, 3, .serial, need);
    _ = e.round(1, 3, .serial, need);
    try testing.expectEqual(G.MtpAdaptiveAction.none, e.round(2, 4, .serial, need));
    try testing.expectEqual(@as(u32, 1), e.confirm);

    // A confirm count of 0 is still one round (never a switch on no evidence).
    var f = G.MtpAdaptive{};
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, f.round(0, 3, .serial, 0));
}

test "MtpAdaptive: crossing always re-opens; the periodic re-open is OFF by default (M16)" {
    const G = Generator;
    const N = G.MtpAdaptive.REDECIDE_SERIAL_TOKENS_ON;
    // Default is crossing-only: a re-entry resumes a head whose committed
    // history did not grow across the block, so paying an exit ramp on a timer
    // is not a proven win until a head-follow exists.
    try testing.expectEqual(@as(u32, 0), G.MtpAdaptive.REDECIDE_SERIAL_TOKENS_DEFAULT);

    var off = G.MtpAdaptive{};
    _ = off.round(0, 3, .serial, 1);
    var i: u32 = 0;
    while (i < N * 2) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, off.serialTick(3, 0));
    }
    try testing.expectEqual(G.MtpAdaptiveArm.serial, off.arm);
    // ...but a CROSSING re-opens even with the timer off.
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, off.serialTick(4, 0));

    // With the lever on, the timer fires — including in the UNBOUNDED top
    // bucket, where a crossing may never come.
    var a = G.MtpAdaptive{};
    _ = a.round(0, 3, .serial, 1);
    i = 0;
    while (i < N - 1) : (i += 1) try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(3, N));
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, a.serialTick(3, N));
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, a.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(3, N)); // not twice

    var top = G.MtpAdaptive{};
    const b_top = round_cost.bucketFor(300_000);
    try testing.expectEqual(b_top, round_cost.bucketFor(900_000));
    _ = top.round(0, b_top, .serial, 1);
    var fired: u32 = 0;
    i = 0;
    while (i < N * 3) : (i += 1) {
        if (top.serialTick(b_top, N) == .to_mtp) {
            fired += 1;
            _ = top.round(i, b_top, .serial, 1);
        }
    }
    try testing.expectEqual(@as(u32, 3), fired);

    // Entering serial restarts the clock.
    var d = G.MtpAdaptive{};
    _ = d.round(0, 3, .serial, 1);
    i = 0;
    while (i < N / 2) : (i += 1) _ = d.serialTick(3, N);
    _ = d.round(1, 3, .mtp, 1);
    _ = d.round(2, 3, .serial, 1);
    try testing.expectEqual(@as(u32, 0), d.serial_ticks);
}

test "mtpHeadPositionDrift: re-entry may only resume a head that is provably in sync" {
    const G = Generator;
    // `qwen4MtpForward` demands pos_offset == pos_base + seq_offset.
    try testing.expectEqual(@as(i64, 0), G.mtpHeadPositionDrift(100, 20, 120));
    // A serial block of L moves the TRUNK and not the head: if the next
    // offset were absolute it would run ahead by exactly L.
    try testing.expectEqual(@as(i64, 7), G.mtpHeadPositionDrift(100, 20, 127));
    try testing.expectEqual(@as(i64, -3), G.mtpHeadPositionDrift(100, 20, 117));
    // seq_offset == 0 re-seeds pos_base, so a fresh head always agrees —
    // whatever offset it is handed.
    try testing.expectEqual(@as(i64, 0), G.mtpHeadPositionDrift(100, 0, 999));
    try testing.expectEqual(@as(i64, 0), G.mtpHeadPositionDrift(0, 0, 0));

    // A refused re-entry LATCHES: the drift cannot shrink on its own, so the
    // request must not keep asking (and keep logging) every period.
    var a = G.MtpAdaptive{};
    _ = a.round(0, 3, .serial, 1);
    a.declineReentry();
    try testing.expect(a.reentry_declined);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(4, 512)); // crossing
    var i: u32 = 0;
    while (i < 2000) : (i += 1) try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(3, 512));
}

test "mtpSerialProbeArm: bounded RETRIES per bucket, and none once the cell is trusted (M15)" {
    const G = Generator;
    var t = round_cost.Table{};
    const b = round_cost.bucketFor(20_000);

    // A probe may be retried: an attempt that never folded MIN_SAMPLES (the
    // request ended, its ticks were dropped as contended) must not burn the
    // bucket's only chance the way the arming flag used to.
    var fired: u32 = 0;
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        if (G.mtpSerialProbeArm(&t, b, true, true, true, true) != null) fired += 1;
    }
    try testing.expectEqual(@as(u32, round_cost.MAX_SERIAL_PROBES), fired);
    try testing.expect(round_cost.MAX_SERIAL_PROBES > 1); // else it is the old flag

    // What ENDS the retries is the cell becoming trusted, not the attempts.
    var u = round_cost.Table{};
    const kv: u32 = 20_000;
    try testing.expect(G.mtpSerialProbeArm(&u, b, true, true, true, true) != null);
    var k: u32 = 0;
    while (k < round_cost.MIN_SAMPLES) : (k += 1) _ = u.observeSerial(kv, 16.0, true, false);
    try testing.expect(u.serialMsPerTok(b) != null);
    try testing.expect(G.mtpSerialProbeArm(&u, b, true, true, true, true) == null);

    // Another bucket is its own decision.
    var v = round_cost.Table{};
    try testing.expect(G.mtpSerialProbeArm(&v, round_cost.bucketFor(1000), true, true, true, true) != null);
    try testing.expect(G.mtpSerialProbeArm(&v, b, true, true, true, true) != null);

    // Refusals: contended, mid-block, an M-RoPE turn, and a bucket that is
    // not a bucket.
    var w = round_cost.Table{};
    try testing.expect(G.mtpSerialProbeArm(&w, b, false, true, true, true) == null);
    try testing.expect(G.mtpSerialProbeArm(&w, b, true, false, true, true) == null);
    try testing.expect(G.mtpSerialProbeArm(&w, b, true, true, false, true) == null);
    try testing.expect(G.mtpSerialProbeArm(&w, round_cost.N_BUCKETS, true, true, true, true) == null);
    // None of those consumed an attempt.
    try testing.expectEqual(@as(u8, 0), w.serial_probes[b]);
}

test "S21: a to_serial on a MODULE-OWNED head is sticky, and the release is a one-shot" {
    const G = Generator;
    // The in-checkpoint qwen4_exp head is the module-owned one; the pointer is
    // never dereferenced, only the tag is read.
    try testing.expect(G.stickyOnSerialSwitch(MtpHeadRef{ .qwen4 = undefined }));

    var a = G.MtpAdaptive{};
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, a.round(0, 3, .serial, 1));
    a.stickSerial();
    try testing.expect(a.sticky_serial);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);

    // The two re-entry triggers are both refused. A bucket CROSSING is the
    // one that always fired before (the periodic re-open is off by default),
    // so it is the one that proves the latch.
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(4, 0));
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(5, 0));
    const on = G.MtpAdaptive.REDECIDE_SERIAL_TOKENS_ON;
    var i: u32 = 0;
    while (i < on * 2) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(3, on));
    }
    // ...and so is a fresh round, whichever way it votes: the controller
    // refuses before it looks at the vote, so neither call site can re-arm.
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.round(1, 3, .mtp, 1));
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.round(2, 4, .serial, 1));
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);

    // The release edge fires exactly once — the log line and the exclusivity
    // drop both key on it. `mtpReleasePending` is its complement: armed and
    // not yet landed, which is the one tick `scheduler.batchable` holds the
    // slot out of a group so the release cannot be skipped.
    try testing.expect(G.stickyReleaseNow(true, false));
    try testing.expect(!G.stickyReleaseNow(true, true));
    try testing.expect(!G.stickyReleaseNow(false, false));
    try testing.expect(!G.stickyReleaseNow(false, true));
}

test "S21: a KV-only sidecar head keeps its re-arm, and nothing about it changes" {
    const G = Generator;
    // A sidecar cache is this request's own KVCache: nothing is reserved on
    // the model, so there is nothing to release and no reason to give up the
    // re-entry. No head at all (a request that never armed MTP) likewise.
    try testing.expect(!G.stickyOnSerialSwitch(MtpHeadRef{ .qwen = undefined }));
    try testing.expect(!G.stickyOnSerialSwitch(null));

    var b = G.MtpAdaptive{};
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, b.round(0, 3, .serial, 1));
    try testing.expect(!b.sticky_serial);
    // Byte-for-byte the pre-S21 behaviour: a crossing re-opens.
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, b.serialTick(4, 0));
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, b.arm);

    // And the whole mechanism off (`MLX_SERVE_MTP_ADAPTIVE_SERIAL=0`) means no
    // vote, so no `to_serial`, so no latch and no release: unchanged by
    // construction.
    try testing.expect(!Generator.mtpAdaptiveSerialEnabledFromEnv("0"));
}

test "S21: every site that can start an MTP round refuses a sticky-serial request" {
    const src = @embedFile("generate.zig");

    // (i) THE gate. One predicate answers for the re-entry, for the deferred
    // history stash, and for the bounded serial PROBE (a probe is a serial
    // block that RETURNS to speculation, i.e. a round start).
    const may_at = std.mem.indexOf(u8, src, "fn mtpAdaptiveHeadMay" ++ "Resume(self: *const Generator)") orelse
        return error.MissingMayResume;
    const sticky_at = std.mem.indexOfPos(u8, src, may_at, "if (self.mtp_adaptive.sticky" ++ "_serial) return false;") orelse
        return error.MissingStickyGate;
    const mrope_at = std.mem.indexOfPos(u8, src, may_at, "self.ctx.mrope" ++ "_pos == null") orelse
        return error.MissingMropeGate;
    try testing.expect(sticky_at < mrope_at);

    // (ii) re-entry in `nextMtp` — the ONE place that clears the adaptive
    // disable — reads it.
    const fn_at = std.mem.indexOf(u8, src, "pub fn next" ++ "Mtp(self: *Generator") orelse
        return error.MissingNextMtp;
    const reentry_at = std.mem.indexOfPos(u8, src, fn_at, "self.mtp_adaptive." ++ "serialTick(") orelse
        return error.MissingReentry;
    try testing.expect(std.mem.indexOf(u8, src[fn_at..reentry_at], "self.mtpAdaptiveHeadMay" ++ "Resume()") != null);

    // (iii) the probe arm takes the same predicate as its `may_resume`.
    try testing.expect(std.mem.indexOf(u8, src, "mtpSerialProbe" ++ "Arm(t, b, self.spec_cost_solo, idle, self.mtpAdaptiveHeadMayResume(), ") != null);

    // (iv) the deferred stash: a sticky request drops it instead of paying a
    // head forward to keep a head it is handing back.
    const resume_at = std.mem.indexOf(u8, src, "fn mtpSerialMay" ++ "Resume(self: *const Generator)") orelse
        return error.MissingSerialMayResume;
    try testing.expect(std.mem.indexOf(u8, src[resume_at .. resume_at + 400], "self.mtpAdaptiveHeadMay" ++ "Resume()") != null);

    // (v) the round PLANNER and the width trial inside it are reached from
    // exactly two places: the cross-round pre-draft, which returns on the
    // runtime disable both BEFORE and AFTER planning, and `nextMtp` below its
    // serial branch. (Their ordering inside the planner is pinned by "the
    // adaptive decision is read after the EV plan..." above.)
    const pre_at = std.mem.indexOf(u8, src, "fn mtpMaybePre" ++ "Draft(self: *Generator") orelse
        return error.MissingPreDraft;
    const pre_gate = std.mem.indexOfPos(u8, src, pre_at, "if (self.spec_disabled" ++ "_runtime) return;") orelse
        return error.MissingPreDraftGate;
    const pre_plan = std.mem.indexOfPos(u8, src, pre_at, "self.mtpRound" ++ "Plan();") orelse
        return error.MissingPreDraftPlan;
    const pre_after = std.mem.indexOfPos(u8, src, pre_plan, "if (self.spec_disabled" ++ "_runtime or self.mtp_serial_left > 0) return;") orelse
        return error.MissingPreDraftRecheck;
    try testing.expect(pre_gate < pre_plan);
    try testing.expect(pre_plan < pre_after);

    // (vi) the serial branch is the FIRST thing past the re-entry gate, so
    // the round body — the EV seed consume included — is unreachable while
    // the disable stands. A sticky request never inherits a seed.
    const serial_at = std.mem.indexOfPos(u8, src, reentry_at, "return try self.mtpSerial" ++ "Tick(allocator);") orelse
        return error.MissingSerialBranch;
    const seed_at = std.mem.indexOfPos(u8, src, reentry_at, "head.ev" ++ "Seed()") orelse
        return error.MissingSeedConsume;
    try testing.expect(serial_at < seed_at);

    // (vii) the EV seed PUBLISH at deinit is gated on the same disable, so a
    // request that left speculation never teaches the next one its surface.
    try testing.expect(std.mem.indexOf(u8, src, "!self.spec_disabled" ++ "_runtime and self.mtp_attempted >= 8") != null);

    // (viii) and the controller itself refuses both entry points, so the
    // invariant does not rest on any of the call-site orderings above.
    const ctrl_round = std.mem.indexOf(u8, src, "pub fn round(self: *MtpAdaptive") orelse
        return error.MissingCtrlRound;
    try testing.expect(std.mem.indexOf(u8, src[ctrl_round .. ctrl_round + 600], "if (self.sticky" ++ "_serial) return .none;") != null);
    const ctrl_tick = std.mem.indexOf(u8, src, "pub fn serialTick(self: *MtpAdaptive") orelse
        return error.MissingCtrlTick;
    try testing.expect(std.mem.indexOf(u8, src[ctrl_tick .. ctrl_tick + 400], "if (self.sticky" ++ "_serial) return .none;") != null);
}

test "S21: the head is released at the block boundary, after the detach, and logged once" {
    const src = @embedFile("generate.zig");

    // ONE call site, and it is the serial branch of `nextMtp` — the pre-commit
    // block boundary. Releasing where the switch is DECIDED
    // (`mtpAdaptiveSerialStep`) would hand a second slot a head this request
    // is mid-round on: the vote is read while a round is being planned and
    // that round still runs.
    try testing.expectEqual(
        @as(usize, 1),
        std.mem.count(u8, src, "self.mtpMaybeReleaseModule" ++ "Head();"),
    );
    const detach_at = std.mem.indexOf(u8, src, "try self.mtpDetach" ++ "Head(allocator, self.mtpSerialMayResume());") orelse
        return error.MissingDetach;
    const rel_at = std.mem.indexOfPos(u8, src, detach_at, "self.mtpMaybeReleaseModule" ++ "Head();") orelse
        return error.MissingReleaseCall;
    const tick_at = std.mem.indexOfPos(u8, src, detach_at, "return try self.mtpSerial" ++ "Tick(allocator);") orelse
        return error.MissingSerialTick;
    try testing.expect(detach_at < rel_at);
    try testing.expect(rel_at < tick_at);
    const step_at = std.mem.indexOf(u8, src, "fn mtpAdaptiveSerial" ++ "Step(self: *Generator") orelse
        return error.MissingStepFn;
    const plan_at = std.mem.indexOfPos(u8, src, step_at, "fn mtpRound" ++ "Plan(self: *Generator)") orelse
        return error.MissingRoundPlan;
    try testing.expect(std.mem.indexOf(u8, src[step_at..plan_at], "mtpMaybeReleaseModule" ++ "Head") == null);
    // The switch's own site arms the LATCH and nothing else.
    try testing.expect(std.mem.indexOf(u8, src[step_at..plan_at], "self.mtp_adaptive.stick" ++ "Serial()") != null);

    // Inside the release: guard, then the flag, then the log — so the line is
    // emitted exactly once per request (`stickyReleaseNow` is the one-shot).
    const fn_at = std.mem.indexOf(u8, src, "fn mtpMaybeReleaseModule" ++ "Head(self: *Generator)") orelse
        return error.MissingReleaseFn;
    const guard_at = std.mem.indexOfPos(u8, src, fn_at, "stickyRelease" ++ "Now(self.mtp_adaptive.sticky_serial, self.mtp_head_released)") orelse
        return error.MissingReleaseGuard;
    const flag_at = std.mem.indexOfPos(u8, src, fn_at, "self.mtp_head" ++ "_released = true;") orelse
        return error.MissingReleaseFlag;
    const log_at = std.mem.indexOfPos(u8, src, fn_at, "[mtp] sticky-serial: module head released") orelse
        return error.MissingReleaseLog;
    try testing.expect(guard_at < flag_at);
    try testing.expect(flag_at < log_at);
    // The flag is written in exactly one place, and read through one accessor.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "self.mtp_head" ++ "_released = true;"));
    try testing.expect(std.mem.indexOf(u8, src, "pub fn mtpModuleHead" ++ "Released(self: *const Generator) bool") != null);
}

test "mtpAdaptiveSerialEnabledFromEnv: the mechanism is on unless the lever says 0" {
    try testing.expect(Generator.mtpAdaptiveSerialEnabledFromEnv(null));
    try testing.expect(Generator.mtpAdaptiveSerialEnabledFromEnv(""));
    try testing.expect(Generator.mtpAdaptiveSerialEnabledFromEnv("1"));
    try testing.expect(!Generator.mtpAdaptiveSerialEnabledFromEnv("0"));
}

test "the adaptive decision is read after the EV plan, before the width trial, and after the ctx ceiling" {
    const src = @embedFile("generate.zig");

    // ── In `mtpRoundPlan`: plan first (m_lo is the width being priced), then
    // the adaptive step, then the width trial — a trial measures a width for
    // a request that is about to leave speculation.
    const plan_at = std.mem.indexOf(u8, src, "fn mtpRound" ++ "Plan(self: *Generator)") orelse
        return error.MissingRoundPlan;
    const ev_at = std.mem.indexOfPos(u8, src, plan_at, "mtpEvPlan" ++ "Src(self.mtp_ev_accept") orelse
        return error.MissingEvPlan;
    const step_at = std.mem.indexOfPos(u8, src, ev_at, "self.mtpAdaptiveSerial" ++ "Step(plan.m_lo, kv_len)") orelse
        return error.MissingAdaptiveStep;
    const trial_at = std.mem.indexOfPos(u8, src, ev_at, "mtpWidthTrial" ++ "Target(") orelse
        return error.MissingWidthTrial;
    try testing.expect(ev_at < step_at);
    try testing.expect(step_at < trial_at);

    // L22: the serial switch is INDEPENDENT of the depth controller. It is
    // also called on the exit taken when MLX_SERVE_MTP_ADAPTIVE=0 or during EV
    // warmup, which used to return before the decision was ever read.
    const warm_at = std.mem.indexOfPos(u8, src, plan_at, "self.mtpAdaptiveSerial" ++ "Step(d, kv_len)") orelse
        return error.MissingWarmupAdaptiveStep;
    try testing.expect(warm_at < step_at);

    // ── Inside the step itself.
    const fn_step = std.mem.indexOf(u8, src, "fn mtpAdaptiveSerial" ++ "Step(self: *Generator") orelse
        return error.MissingStepFn;
    const gate_at = std.mem.indexOfPos(u8, src, fn_step, "mtpAdaptiveKv" ++ "Eligible(kv_len") orelse
        return error.MissingKvFloor;
    const decide_at = std.mem.indexOfPos(u8, src, fn_step, "self.mtp_adaptive." ++ "round(") orelse
        return error.MissingDecision;
    const probe_at = std.mem.indexOfPos(u8, src, fn_step, "mtpSerialProbe" ++ "Arm(t, b,") orelse
        return error.MissingProbeArm;
    // The floor gates the whole block, so it covers the probe as well as the
    // vote: a probe below it would spend 8 serial tokens teaching a bucket
    // that is never allowed to decide.
    try testing.expect(gate_at < decide_at);
    try testing.expect(gate_at < probe_at);

    const step_win = src[fn_step..probe_at];
    // BOTH of the vote's MTP prices are MEASURED, and the modeled chain is not
    // one of them: a measured numerator over a modeled denominator is how v1
    // came to switch on prompts where MTP was 10% faster.
    try testing.expect(std.mem.indexOf(u8, step_win, "t.msPer" ++ "Tok(m_lo, b)") != null);
    try testing.expect(std.mem.indexOf(u8, step_win, "self.mtp_price.msPer" ++ "Tok()") != null);
    try testing.expect(std.mem.indexOf(u8, step_win, "mtpEvExpected" ++ "Tokens(&self.mtp_ev_accept") == null);
    // ONE bucket resolver (H6), and the window is dropped when the regime
    // moves (H7).
    try testing.expect(std.mem.indexOf(u8, step_win, "self.mtpAdaptive" ++ "Bucket(kv_len)") != null);
    try testing.expectEqual(
        @as(usize, 2),
        std.mem.count(u8, src, "self.mtpAdaptiveSync" ++ "Window(prev_bucket, prev_arm)"),
    );
    // The decision disables through the EXISTING runtime fallback and names
    // its own reason, so `[spec-stats]` can tell it from the acceptance floor
    // and from the operator's ceiling.
    const disable_win = src[decide_at..probe_at];
    try testing.expect(std.mem.indexOf(u8, disable_win, "spec_disabled" ++ "_runtime = true") != null);
    try testing.expect(std.mem.indexOf(u8, disable_win, "spec_disable" ++ "_reason = .adaptive") != null);
    try testing.expect(std.mem.indexOf(u8, src, "reason={s} adaptive={s} serial_cell={d:.2}") != null);
    // M14: the vote is read while the round is being planned and `nextMtp`
    // already tested `spec_disabled_runtime`, so the round under construction
    // still runs speculatively. The line must say so — and must keep the
    // "-> serial" substring every guard and analyzer greps for.
    try testing.expect(std.mem.indexOf(u8, src, "-> serial (from the next round)") != null);

    // ── In `nextMtp`: `--max-mtp-ctx` is checked FIRST and is not
    // overridable — the re-entry keys on `.adaptive`, so a ceiling crossing
    // stays off for good.
    const fn_at = std.mem.indexOf(u8, src, "pub fn next" ++ "Mtp(self: *Generator") orelse
        return error.MissingNextMtp;
    const ceiling_at = std.mem.indexOfPos(u8, src, fn_at, "mtpCtx" ++ "WithinLimit(max_mtp_ctx") orelse
        return error.MissingCeiling;
    const reentry_at = std.mem.indexOfPos(u8, src, fn_at, "self.mtp_adaptive." ++ "serialTick(") orelse
        return error.MissingReentry;
    try testing.expect(ceiling_at < reentry_at);
    try testing.expect(std.mem.indexOf(u8, src[fn_at..reentry_at], "spec_disable" ++ "_reason == .adaptive") != null);
    // ONE bucket resolver at BOTH sites (H6).
    const reentry_win = src[fn_at .. reentry_at + 64];
    try testing.expect(std.mem.indexOf(u8, reentry_win, "self.mtpAdaptive" ++ "Bucket(self.mtpKvLen())") != null);
    try testing.expect(std.mem.indexOf(u8, reentry_win, "round_cost.bucket" ++ "For(self.mtpKvLen())") == null);
    // Re-entry is gated on the head PROVING its position bookkeeping is in
    // sync: `qwen4MtpForward` returns `error.MtpPositionGap` rather than
    // degrading, and a serial block is exactly the transition that moves the
    // trunk without moving the head.
    const gate_win = src[reentry_at .. reentry_at + 1400];
    try testing.expect(std.mem.indexOf(u8, gate_win, "self.mtpHeadDrift" ++ "Now()") != null);
    try testing.expect(std.mem.indexOf(u8, gate_win, "declineRe" ++ "entry()") != null);
    try testing.expect(std.mem.indexOf(u8, src, "re-entry declined (head out of sync by {d})") != null);

    // Every serial tick folds through ONE helper, and that helper is GATED
    // (H8): a model with no MTP head must not fold a cell nothing reads, nor
    // rewrite the persisted table on every request.
    try testing.expect(std.mem.indexOf(u8, src, "self.observe" ++ "SerialTick();") != null);
    try testing.expect(std.mem.indexOf(u8, src, "if (!self.serialCell" ++ "Wanted())") != null);
    const sched = @embedFile("scheduler.zig");
    try testing.expect(std.mem.indexOf(u8, sched, "gen.observe" ++ "SerialTick();") != null);

    // M13: an INTERLEAVED decode tick runs from inside another request's
    // prefill, so the interval since the previous tick contains a prefill
    // chunk while `spec_cost_solo` still reads true. The pending interval must
    // be dropped BEFORE the tick runs, or the chunk is folded as a token.
    const il_at = std.mem.indexOf(u8, sched, "fn interleaveDecode" ++ "Tick(sch: *Scheduler)") orelse
        return error.MissingInterleave;
    const inval_at = std.mem.indexOfPos(u8, sched, il_at, "g.invalidateSerial" ++ "Clock()") orelse
        return error.MissingSerialClockInvalidation;
    const tick_at = std.mem.indexOfPos(u8, sched, il_at, "runDecode" ++ "Tick(sch, buf[0..n])") orelse
        return error.MissingRunDecodeTick;
    try testing.expect(inval_at < tick_at);
}

test "mtpSerialCaptureReady: the capture step's entry invariant is a RUNTIME check, not an assert" {
    const G = Generator;
    // The only state the capture forward may run in.
    try testing.expect(G.mtpSerialCaptureReady(false, false));
    try testing.expect(!G.mtpSerialCaptureReady(true, false));
    try testing.expect(!G.mtpSerialCaptureReady(false, true));
    try testing.expect(!G.mtpSerialCaptureReady(true, true));

    const src = @embedFile("generate.zig");
    const at = std.mem.indexOf(u8, src, "fn mtpSerialCapture" ++ "Tick(self: *Generator") orelse
        return error.MissingCaptureTick;
    const end = std.mem.indexOfPos(u8, src, at, "\n    }\n") orelse src.len;
    const body = src[at..end];
    // A `std.debug.assert` here compiles to nothing in ReleaseFast — the only
    // mode that ever serves — so the invariant must be a real branch that
    // takes the give-up arm, exactly like the drain path's `.stay_disabled`.
    try testing.expect(std.mem.indexOf(u8, body, "std.debug." ++ "assert") == null);
    try testing.expect(std.mem.indexOf(u8, body, "mtpSerialCapture" ++ "Ready(") != null);
    try testing.expect(std.mem.indexOf(u8, body, "mtpSerialGive" ++ "Up(") != null);
    // ONE recovery, shared with the drain arm: two spellings of "finish this
    // request serial" would drift apart.
    try testing.expect(std.mem.indexOf(u8, src, ".stay_disabled => return try self.mtpSerialGive" ++ "Up(") != null);
    // Logged once per request, never once per tick.
    const give_at = std.mem.indexOf(u8, src, "fn mtpSerialGive" ++ "Up(self: *Generator") orelse
        return error.MissingGiveUp;
    const give_end = std.mem.indexOfPos(u8, src, give_at, "\n    }\n") orelse src.len;
    try testing.expect(std.mem.indexOf(u8, src[give_at..give_end], "mtp_serial_giveup_" ++ "logged") != null);
}

test "MTP head persistence kill switch: only a literal 0 turns it off" {
    // `MLX_SERVE_MTP_HEAD_PERSIST=0` restores the pre-persistence behaviour
    // for the qwen4_exp in-checkpoint head — `kv()`/`head()` return null, so
    // its committed history is neither snapshotted nor restored and every
    // reused prefix starts from `qwen4MtpReset`. Absent OR empty is ON: a
    // harness that exports the variable bare must not silently disable a
    // default-on path (the `QWEN4_PROFILE_FWD=0` class).
    try testing.expect(mtpHeadPersistFromEnv(null));
    try testing.expect(mtpHeadPersistFromEnv(""));
    try testing.expect(mtpHeadPersistFromEnv("1"));
    try testing.expect(!mtpHeadPersistFromEnv("0"));
}

test "scan: the prefill chunk loop's post-eval order is a contract (B0b + S17)" {
    // SHARED CONTRACT with the adaptive-width owner (a8bb30c23939f18f4). The
    // order after a chunk's forward is, and must stay:
    //
    //   eval -> mlx_clear_cache -> checkError -> write_through_hook
    //        -> chunk_width_hook -> interleaveDecodeTick
    //
    // Each edge is load-bearing and they were established for different
    // reasons, so a change that looks local to one owner can silently undo
    // the other's:
    //
    //   clear   -> check : the chunk's own latch, read before ANYTHING acts
    //                      on the KV it produced (B0b).
    //   check   -> wt    : the write-through indexes a DURABLE restorable
    //                      prefix on the SSD tier. Metal returns ZEROS before
    //                      it aborts, so persisting first publishes garbage
    //                      that later requests restore FROM, permanently.
    //   check   -> tick  : `interleaveDecodeTick` -> `checkErrorDecode` is the
    //                      same `consumeLatch`. Running it first let a
    //                      co-tenant decode EAT this prefill's latch and be
    //                      blamed, after which the failing prefill saw a clean
    //                      latch and answered 200 with zeros.
    //   wt      -> width : the width probe sizes the NEXT chunk; the hook
    //                      above it persists the one just completed.
    //   width   -> tick  : the probe reads steady-state headroom, so a
    //                      co-tenant's decode allocations are not charged to
    //                      this prefill (the adaptive owner's attribution
    //                      argument, S17).
    //
    // Every needle is SPLIT: this scan sits BELOW the impl in this file, and
    // an unsplit needle matches the test's own literal first and can never go
    // red — the failure mode the audit found in two other guards.
    const src = @embedFile("generate.zig");
    const wt = std.mem.indexOf(u8, src, "write_through" ++ "_hook) |wt|").?;
    const before = src[0..wt];
    // `lastIndexOf` anchors on THIS chunk's eval/clear pair, so the check at
    // the TOP of the loop — which precedes the clear — cannot satisfy it.
    const clear = std.mem.lastIndexOf(u8, before, "mlx_clear" ++ "_cache();").?;
    const check = std.mem.lastIndexOf(u8, before, "try mlx.check" ++ "Error();").?;
    const width = std.mem.indexOfPos(u8, src, wt, "chunk_width" ++ "_hook) |hk|").?;
    const tick = std.mem.indexOfPos(u8, src, wt, "interleave" ++ "_hook) |hk|").?;

    try std.testing.expect(clear < check);
    try std.testing.expect(check < wt);
    try std.testing.expect(wt < width);
    try std.testing.expect(width < tick);

    // The SSM checkpoint capture is on the same poisoned path — the snapshot
    // is what the write-through then persists — so the check precedes it too.
    const cp = std.mem.indexOfPos(u8, src, clear, "captureSsm" ++ "Checkpoint(").?;
    try std.testing.expect(check < cp);
}

test "scan: the prefill-end QSA history attach is gated by the share switch; the cancel sink hands off" {
    // With `MLX_SERVE_QSA_HISTORY_SHARE` on, the prefill must NOT materialize
    // a second copy of the indexer history — the commit hands the live buffer
    // over as a view. An ungated attach here re-doubles the decode's
    // residency behind a bill that says one copy.
    const src = @embedFile("generate.zig");
    const gate = "ssm_checkpoints.items.len > 0 and !transformer_mod.qsaHistoryShare" ++ "Enabled()";
    const gate_at = std.mem.indexOf(u8, src, gate) orelse return error.MissingShareGate;
    const attach = "transformer_mod.attachQsaHistoryTo" ++ "Latest(ssm_checkpoints.items, ctx.ssm_entries.?, xfm.s)";
    const attach_at = std.mem.indexOf(u8, src, attach) orelse return error.MissingPrefillEndAttach;
    // The gate is the condition of the statement the attach sits in.
    try std.testing.expect(gate_at < attach_at and attach_at - gate_at < 200);
    // No other prefill-end attach exists: every remaining call is the
    // dispatcher on the cancel handoff.
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "transformer_mod.attachQsaHistoryTo" ++ "Latest("));
    try std.testing.expect(std.mem.indexOf(u8, src, "transformer_mod.attachQsaHistory" ++ "OnHandoff(ssm_checkpoints.items, ents, xfm.s)") != null);
}

test "L27 characterization: a sidecar (legacy-layout) boot plans EXACTLY as a93e2c0 did" {
    // The regression this pins: on a qwen3_5 trunk with a MTP SIDECAR (the
    // 27B pack), a PR that bumped the round-cost store version made every
    // boot COLD. `MtpCostSource.fromTable()` is the ONE term deciding
    // whether the EV plan prices extension from measurements or from the
    // fitted prior, and the prior's extension valve is deliberately
    // always-open — it exists to observe `a[m_lo]`, which no other round
    // can. Cold therefore means EVERY round is a two-chunk round: measured
    // in the field at 37 of 40 extension rounds at 77 ms against a warm
    // table's 1 of 52 at 51 ms, same acceptance, -25% decode.
    //
    // Every expectation below was RECORDED by running the a93e2c0 planner on
    // these exact inputs (scratch worktree at a93e2c0, probe over
    // `mtpEvPlanSrc` / `MtpCostSource.init`), never derived from this tree.
    const G = Generator;
    const a = [_]f32{ 0.85, 0.78, 0.70, 0.60, 0.50, 0.42, 0.35, 0.30 };
    const kv: u32 = 8192; // one 27B rung; ~3.0-3.5 tok/round at m_lo 2-3
    const costs = G.MTP_EV_DEFAULT_COSTS;
    const cap: u32 = 6;

    // The bucket an 8k request reads is 3 on BOTH grids — the split is a
    // long-context change — so the scenario isolates the FORMAT, not the
    // edges.
    const legacy_empty = round_cost.Table{ .layout = .legacy };
    try testing.expectEqual(@as(usize, 3), legacy_empty.bucketOf(kv));
    try testing.expectEqual(@as(usize, 3), round_cost.bucketFor(kv));

    // ── the warm table, restored from the file 26.9.1 wrote ──
    const rc1_body = "rc1\n2 3 44.0000 2.7000 3\n3 3 88.0000 3.0500 3\n4 3 140.0000 3.2000 3\n";
    const warm = round_cost.parse(rc1_body, .legacy) orelse return error.TestUnexpectedResult;
    const warm_src = G.MtpCostSource.init(costs, kv, &warm);
    try testing.expect(warm_src.fromTable());

    // a93e2c0, warm: a MEASURED marginal the position cannot repay even at
    // full confidence closes the valve. Every round is SINGLE-chunk.
    var prev: u32 = 1;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, warm_src, prev + 1);
        try testing.expectEqual(@as(u32, 2), p.m_lo);
        try testing.expectEqual(@as(u32, 2), p.m_hi); // no extension
        try testing.expectEqual(@as(f32, 0.0), p.tau_ln);
        prev = p.m_lo;
    }

    // ── the same request on a COLD table: the flip, and why the bump cost
    // 25%. Recorded from a93e2c0 too, so this arm is not a claim about the
    // fix — it is the shape both trees produce when the file is missing.
    const cold_src = G.MtpCostSource.init(costs, kv, null);
    try testing.expect(!cold_src.fromTable());
    const cold0 = G.mtpEvPlanSrc(&a, cap, cold_src, 2);
    try testing.expectEqual(@as(u32, 2), cold0.m_lo);
    try testing.expectEqual(@as(u32, 4), cold0.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.444614), cold0.tau_ln, 1e-5);
    prev = cold0.m_lo;
    for (0..7) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, cold_src, prev + 1);
        try testing.expectEqual(@as(u32, 3), p.m_lo);
        try testing.expectEqual(@as(u32, 4), p.m_hi); // two-chunk, every round
        try testing.expectApproxEqAbs(@as(f32, -0.477666), p.tau_ln, 1e-5);
        prev = p.m_lo;
    }

    // ── a warm table whose cells make ONE extension position pay: the plan
    // extends by exactly one, and stops. (a93e2c0 literals.)
    var shallow = round_cost.Table{ .layout = .legacy };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = shallow.observe(2, kv, 44.0, 2.70, true, false);
        _ = shallow.observe(3, kv, 51.0, 3.20, true, false);
        _ = shallow.observe(4, kv, 62.0, 3.50, true, false);
    }
    const shallow_src = G.MtpCostSource.init(costs, kv, &shallow);
    prev = 1;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, shallow_src, prev + 1);
        try testing.expectEqual(@as(u32, 2), p.m_lo);
        try testing.expectEqual(@as(u32, 3), p.m_hi);
        try testing.expectApproxEqAbs(@as(f32, -0.488353), p.tau_ln, 1e-5);
        prev = p.m_lo;
    }

    // ── the WIDTH is a table-state property too, not only the extension.
    // Same acceptance stream, a warm table whose cells make a wider base
    // look good: a93e2c0 holds the base at 2 (the standing-base hysteresis
    // measured cells earn and the prior does not have), while the SAME
    // request on a cold table climbs to 3. Recorded from a93e2c0.
    var wide = round_cost.Table{ .layout = .legacy };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = wide.observe(2, kv, 44.0, 2.70, true, false);
        _ = wide.observe(3, kv, 51.0, 3.20, true, false);
        _ = wide.observe(4, kv, 55.0, 4.00, true, false);
        _ = wide.observe(5, kv, 75.0, 4.20, true, false);
    }
    const wide_src = G.MtpCostSource.init(costs, kv, &wide);
    prev = 1;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, wide_src, prev + 1);
        try testing.expectEqual(@as(u32, 2), p.m_lo);
        try testing.expectEqual(@as(u32, 4), p.m_hi);
        try testing.expectApproxEqAbs(@as(f32, -0.506371), p.tau_ln, 1e-5);
        prev = p.m_lo;
    }
    // The cold arm over the SAME acceptance stream settles one width wider —
    // a cheaper round that earns less, which is the shape the field saw
    // (tok/step 3.31 -> 2.67 with the round 53.9 -> 44.6 ms). The width the
    // plan picks is a property of the TABLE, so the table's identity, grid
    // and store version have to be the arch's, not the build's.
    prev = 1;
    var cold_final: u32 = 0;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, cold_src, prev + 1);
        cold_final = p.m_lo;
        prev = p.m_lo;
    }
    try testing.expectEqual(@as(u32, 3), cold_final);
    try testing.expect(cold_final != G.mtpEvPlanSrc(&a, cap, wide_src, 8).m_lo);

    // The width trial reads the SAME bucket through the table's own grid,
    // and owes the same width a93e2c0 owed (recorded: 3).
    try testing.expectEqual(
        @as(?u32, 3),
        G.mtpWidthTrialTarget(&warm, kv, .{ .m_lo = 2, .m_hi = 2, .tau_ln = 0 }, cap, true),
    );
}

test "mtpAdaptiveModelEligible: the serial row and its price window are the module head's, not every MTP model's (L27)" {
    const G = Generator;
    // A sidecar pack HAS an MTP head, so `model_has_mtp` alone let every
    // qwen3.5/3.6/3.8 boot past 8192 KV read a stopwatch per decode token,
    // fold a cell nothing on that arch reads, and rewrite the persisted
    // table at the end of every request for it.
    try testing.expect(!G.mtpAdaptiveModelEligible(true, false));
    try testing.expect(G.mtpAdaptiveModelEligible(true, true));
    // `--no-mtp` on the calibrated arch still declines: the in-checkpoint
    // head's weights load with the trunk, so presence alone is not consent.
    try testing.expect(!G.mtpAdaptiveModelEligible(false, true));
    try testing.expect(!G.mtpAdaptiveModelEligible(false, false));
}

test "mtpSerialProbeUseful: a probe buys the LAST missing input, never the first (L27)" {
    const G = Generator;
    // The vote needs table, window and serial. The window takes
    // MTP_PRICE_WINDOW non-trial rounds to fill and dies with the request,
    // so a request that cannot fill it can never use what a probe buys — it
    // just pays 8 serial tokens. Measured cold on Flash-Next: the short cell
    // ran 82.7 against 88.3 warm, the probes being the difference.
    try testing.expect(!G.mtpSerialProbeUseful(null));
    try testing.expect(G.mtpSerialProbeUseful(12.5));

    var t = round_cost.Table{ .layout = .long };
    const b = round_cost.bucketFor(20_000);
    // Every other precondition satisfied, window empty: no probe, no cost,
    // and — the point of a COUNT rather than a flag — no budget spent either.
    try testing.expect(G.mtpSerialProbeArm(&t, b, true, true, true, G.mtpSerialProbeUseful(null)) == null);
    try testing.expectEqual(@as(u8, 0), t.serial_probes[b]);
    // Once the window is full the probe arms exactly as before.
    try testing.expectEqual(@as(?usize, b), G.mtpSerialProbeArm(&t, b, true, true, true, G.mtpSerialProbeUseful(12.5)));
    try testing.expectEqual(@as(u8, 1), t.serial_probes[b]);
}

test "reservedPrefillTokens: the KV capacity reservation is qwen4_exp-only; every other arch keeps a93e2c0's growth" {
    // PR #363 item 1. The reservation pre-buys the prompt + a generation
    // headroom + a chunk of slack past 32k tokens, and the admission guard
    // bills it. On qwen4_exp that removes a measured 7.75 GB coexistence
    // transient at 458k. On a 27B qwen3_5 at 131k it is a pure cost: memory
    // reserved for a generation that may never happen, on a path nobody
    // measured. Gated.
    const t = std.testing;
    const chunk: u64 = 4096;
    const seq: u64 = 200_000;
    const max_tokens: u64 = 2048;

    var qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .max_position_embeddings = 1_048_576 };
    // The gated arch reserves exactly what `KVCache.reservedTokens` says — the
    // guard's own function, so the two cannot drift. Reached through an alias
    // so the scan pin below still counts exactly ONE direct call site.
    const KVC = transformer_mod.KVCache;
    try t.expectEqual(
        KVC.reservedTokens(seq, max_tokens, chunk, 1_048_576),
        reservedPrefillTokens(&qwen4, seq, max_tokens, chunk),
    );
    try t.expect(reservedPrefillTokens(&qwen4, seq, max_tokens, chunk) > seq);

    // Every other arch: ZERO, which is `KVCache.reserve`'s no-op and leaves
    // `reserve_tokens` at its default — `nextCapacityReserved` is then
    // `nextCapacity`, the a93e2c0 policy, byte for byte.
    for ([_][]const u8{ "qwen3_5", "qwen3_5_moe", "lfm2", "nemotron_h", "bailing_hybrid", "llama" }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt, .max_position_embeddings = 262_144 };
        try t.expectEqual(@as(u64, 0), reservedPrefillTokens(&cfg, seq, max_tokens, chunk));
    }

    // ... and a reservation of 0 really is the a93e2c0 growth ladder. Pinned
    // on the pure policy rather than on a live cache: `reserve(0)` never
    // raises `reserve_tokens`, so `nextCapacityReserved(cap, needed)` reduces
    // to `nextCapacityPolicy(cap, needed, linear)` at every capacity.
    var cache = try KVC.init(t.allocator, 4);
    defer cache.deinit();
    cache.reserve(0);
    try t.expectEqual(@as(usize, 0), cache.reserve_tokens);
}

test "reservedPrefillTokens: the prefill's ONE reservation site asks the gate, not the cache" {
    // Scan pin. `KVCache.reservedTokens` is ungated by construction (the guard
    // calls it too), so a call site that reaches it directly re-arms the
    // reservation on every arch. There is exactly one such call in this file
    // and it is inside the gated helper.
    const t = std.testing;
    const src = @embedFile("generate.zig");
    const call = "transformer_mod.KVCache.reserved" ++ "Tokens(";
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, src, call));
    const gate = "pub fn reservedPrefill" ++ "Tokens(";
    const at = std.mem.indexOf(u8, src, gate) orelse return error.HelperMoved;
    const body = src[at..@min(src.len, at + 500)];
    try t.expect(std.mem.indexOf(u8, body, "config.longCtxGated()") != null);
    try t.expect(std.mem.indexOf(u8, body, call) != null);
}
