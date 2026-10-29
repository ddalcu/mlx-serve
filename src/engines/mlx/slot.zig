//! A request slot: per-request state the inference thread drives and the connection thread drains.

const std = @import("std");
const abi = @import("../../abi/engine.zig");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const rp_mod = @import("../../sdk/grammar/protocol.zig");
const drafter_mod = @import("../../plugins/gemma4/assistant_drafter.zig");
const model_mod = @import("../../sdk/config/model.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const metrics_mod = @import("../../host/metrics/metrics.zig");
const model_registry_mod = @import("../../host/registry/registry.zig");
const KVCache = transformer_mod.KVCache;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const ForwardCtx = transformer_mod.ForwardCtx;
const ModelConfig = model_mod.ModelConfig;
const Generator = generate_mod.Generator;
const SamplingParams = generate_mod.SamplingParams;
const DrafterModel = drafter_mod.DrafterModel;
const dflash_mod = @import("spec/dflash/dflash.zig");
const DflashModel = dflash_mod.DflashModel;

const SubmitParams = @import("scheduler/params.zig").SubmitParams;
const assistantSidecarEnabledForRequest = @import("spec/wiring.zig").assistantSidecarEnabledForRequest;

pub const SlotState = enum { pending_prefill, decoding, finished, errored };

/// Result of `Slot.waitNext`. Driven by the inference thread; consumed by
/// the connection thread.
pub const NextResult = union(enum) {
    /// Next decoded token id.
    token: u32,
    /// Generation completed (EOS, max_tokens, timeout, etc.). `finish_reason`
    /// is set on the slot at this point.
    done: void,
    /// Generation errored. `error_code` is set on the slot; the caller
    /// should surface it to the client and call `complete(slot)`.
    err: void,
};

/// Per-request state. Owned by the Scheduler from `submit` until `complete`.
pub const Slot = struct {
    allocator: std.mem.Allocator,
    /// io reference for Stopwatch / async-eval (captured from Scheduler).
    io: std.Io,

    /// Plan 05 Phase D: target model for this request. Captured from
    /// `SubmitParams.model` and borrowed for the slot's lifetime; the conn
    /// thread holds a refcount (via `scheduler.ensureLoaded`) so the
    /// pointer stays valid until `complete()`. Prefill/decode route forward
    /// passes through `model.transformer.?` (and `model.vision_encoder`,
    /// `model.drafter`, `model.prefix_cache`). Batched decode groups slots
    /// by this field so kernels never cross model boundaries.
    model: *model_registry_mod.LoadedModel,

    // ── Per-slot model state. Owned by the slot. ──
    cache: KVCache,
    moe_seq_offset: usize,
    ssm_entries: ?[]SSMCacheEntry,
    /// SSM stride checkpoints salvaged from a prefill the client cancelled:
    /// `Generator.initWithOptions` moves its captured checkpoints into this
    /// sink before returning `error.Cancelled` (they die with the failed
    /// construction otherwise). Consumed by `commitCancelledPrefillSlot`
    /// (ownership transfers into the hot-cache entry); freed by `deinit`
    /// when never consumed.
    cancelled_prefill: Generator.CancelledCheckpointSink = .{},
    vision_embeddings: ?mlx.mlx_array,
    /// Media items in `full_prompt` (owned): the prefix-cache key of their rows.
    media: []prefix_cache_mod.MediaSpan,
    cache_key: u64 = 0,
    client: metrics_mod.Client = .other,
    request_id: u64 = 0,
    /// Hot-cache entry this request restored from (`LookupResult.entry_id`).
    restored_entry: u64 = 0,
    skip_prefix_cache: bool = false,
    /// Set while the slot is prefilled at another prefill's chunk boundary: the most uncached tokens it
    /// may forward there. A restore that covers less than the pre-filter peeked hands the request back.
    boundary_max_tail: ?usize = null,
    /// Qwen3-VL M-RoPE position-id table (flat [3 × mrope_total]) + decode delta.
    /// Owned by the slot; `mrope_pos` freed on deinit.
    mrope_pos: ?[]const i32,
    mrope_total: usize,
    mrope_delta: i32,

    /// Forward context backed by the fields above. Initialized in `init`
    /// and aliased by the Generator's own `ctx` field at prefill time.
    ctx: ForwardCtx,

    /// Generator (constructed on inference thread post-prefill).
    legacy_gen: ?Generator,

    /// True when this slot holds one of `model.session_claims` from `submit` (an engine session beside MLX);
    /// released in `complete()`.
    holds_session: bool = false,

    // ── Submission data. Owned by the slot, freed in deinit. ──
    prompt_ids: []u32,
    full_prompt: []u32,
    sampling: SamplingParams,
    eos_token_ids: []u32,
    max_tokens: u32,
    timeout_ns: u64,
    has_tools: bool,
    enable_thinking: bool,
    enable_pld: bool,
    enable_drafter: bool,
    drafter: ?*DrafterModel,
    dflash: ?*DflashModel,
    drafter_block_size: u32,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    mtp: ?generate_mod.MtpHeadRef,
    mtp_depth: u32,
    pld_draft_len: u32,
    pld_key_len: u32,
    /// Phase 2 (Plan ricky): see SubmitParams.kv_attn_fused.
    kv_attn_fused: bool,
    cached_tokens: u32,
    logprobs_n: u32,
    /// A spoken reply (`SubmitParams.speech`).
    speech: ?@import("../../sdk/request/speech.zig").Request = null,
    serial_reason_logged: bool = false,
    memory_hold_logged: bool = false,
    memory_promised: u64 = 0,
    /// This tick decodes plain (batched) although the slot's MTP head is armed: the
    /// batched forward captures its hidden so the next solo tick can resume speculating.
    mtp_plain_tick: bool = false,
    planner_plain_transition: bool = false,
    planner_price_transition: bool = false,
    planner_last_width: u8 = 255,
    planner_force_plain: bool = false,
    company_ticks: u8 = 0,
    mtp_publish_ns: u64 = 0,
    mtp_publish_gap_ms: f32 = 0,

    // ── State + output channel. ──
    state: SlotState,
    /// The host's stream (`SubmitParams.sink`): every token and the terminator also go out as `abi.Event`s.
    sink: ?abi.Sink = null,
    /// A constraint span published before the token that carries it rides that token's event.
    sink_constraint: ?[2]u32 = null,

    out_mu: std.Io.Mutex,
    out_cond: std.Io.Condition,
    /// Wake signal for `waitNextTimeout` — set alongside every `out_cond`
    /// broadcast. Events support timed waits (Io.Condition does not), which
    /// is what lets the conn thread poll the peer socket during long
    /// prefills instead of blocking until the first token.
    out_event: std.Io.Event,
    out_buf: std.ArrayList(u32),
    out_idx: usize,
    finished: bool,
    error_code: ?[]const u8,
    /// {needed, available} bytes the inference thread compared when it refused this prefill, so the
    /// client's `PrefillDoesNotFit` can quote them as the connection thread's refusal does.
    refused_bill: ?[2]u64 = null,
    finish_reason: []const u8,
    /// Set ONLY by the degenerate-tail guard. The wire reason is "stop" so a
    /// client does not mistake the guard for output/context exhaustion; this
    /// sibling signal preserves the specific cause. Static string, never freed.
    finish_details: ?[]const u8,
    /// Index into the emitted tokens where the degenerate span begins;
    /// everything from here on is the loop. Non-streaming responses are cut
    /// here so the client cannot round-trip the loop into the next prompt.
    loop_trim_start: ?usize,
    cancelled: std.atomic.Value(bool),
    /// Inference thread only: the request's outcome has been counted in `Metrics`.
    metrics_recorded: bool = false,
    /// Inference-thread passes (a prefill, a decode tick) holding this slot, taken
    /// under `queue_mu`. `complete` waits it out: the handler owns sampling state
    /// the pass reads (`think_bound`, `constraint`) and frees it once `complete` returns.
    in_pass: std.atomic.Value(u32) = .init(0),
    /// Reasoning-protocol payload boundary, published by the inference thread
    /// BEFORE the token that carries it is pushed (single-writer atomics, so
    /// the conn thread reading token i already sees its span). `token_index`
    /// is the index into the generated stream; `maxInt(u32)` = not started.
    constraint_payload_at: std.atomic.Value(u32),
    constraint_payload_offset: std.atomic.Value(u32),

    // ── Stats (filled by inference thread, safe to read after finish). ──
    prompt_tokens: u32,
    completion_tokens: u32,
    prefill_tps: f64,
    decode_tps: f64,
    /// Monotonic timestamp captured in `Slot.init`, BEFORE the queue wait.
    /// Anchors the exact time-to-first-token measurement.
    request_start_ts: std.Io.Timestamp,
    /// Wall-clock nanoseconds from `request_start_ts` (request arrival) to
    /// prefill completion = queue_wait + prefill = real time-to-first-token.
    /// Captured directly when prefill finishes (never derived by subtracting
    /// decode time), so it stays exact even if a slot finishes mid-tick. Used
    /// as `real_ttft_ns` for the metrics histogram; e2e = first_token_ns + decode_ns.
    first_token_ns: u64,
    /// Wall-clock nanoseconds spent in `runPrefill` for this slot. Includes
    /// hot-prefix-cache lookup/restore and the model forward over the
    /// uncached tail. Populated by the scheduler main loop.
    prefill_ns: u64,
    /// The prefill's sampled token, already published; its next push is swallowed.
    early_first: ?u32 = null,
    /// Wall-clock nanoseconds of interleaved decode ticks hosted INSIDE this
    /// slot's prefill (chunk-boundary yields). Charged to the decoding slots
    /// that received the tokens; subtracted from this slot's `prefill_ns` so
    /// prefill_tps stays a statement about the prefill forward.
    prefill_interleaved_ns: u64,
    /// Wall-clock nanoseconds the slot spent in decode ticks. For batched
    /// decode the full tick wall-clock is added to every participating slot,
    /// so this matches the per-slot throughput a user actually observes
    /// (`completion_tokens / decode_ns`).
    decode_ns: u64,
    /// Actual generated ids (from legacy_gen.generated_ids). Shallow copy at
    /// completion so the connection thread can read them without locking.
    generated_ids: ?[]u32,
    /// pad-only flag: set by inference thread when the entire generation was
    /// token id 0. Server-level cache invalidation reads this after complete.
    was_pad_only: bool,
    /// Phase A5: per-token logprobs accumulated by the inference thread when
    /// `logprobs_n > 0`. The conn thread takes ownership via
    /// `nonStreamingViaScheduler` (or equivalent) at completion; if the
    /// caller doesn't consume, `Slot.deinit` frees the contents.
    logprobs_buf: std.ArrayList(generate_mod.LogprobResult),
    /// Prompt-token logprobs for an echo request (`SubmitParams.prompt_logprobs`),
    /// filled by prefill; taken by the conn thread like `logprobs_buf`.
    prompt_logprobs: bool,
    prompt_logprobs_buf: std.ArrayList(generate_mod.LogprobResult),

    /// Initialize but do NOT take ownership of caches — those are allocated
    /// inside `init` from the slot's allocator.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        config: *const ModelConfig,
        params: SubmitParams,
        kv_quant_config: transformer_mod.KVQuantConfig,
    ) !*Slot {
        const slot = try allocator.create(Slot);
        errdefer allocator.destroy(slot);

        // A model on an engine beside MLX (ds4, llama.cpp, the Neural Engine): skip all MLX per-slot
        // allocations — the engine owns its own KV cache and forward. Build
        // sentinel-empty fields so `Slot.deinit` is well-defined on both paths.
        const is_embedded = params.model.engineBacked();

        // Per-slot KVCache, honoring the process-level kv-quant setting. For
        // embedded slots the engine owns its own cache — we initialize a
        // zero-layer shell so `Slot.deinit` is symmetric with the MLX path.
        const slot_kv_layers: u32 = if (is_embedded) 0 else config.num_hidden_layers;
        var cache = try KVCache.initWithConfig(allocator, slot_kv_layers, kv_quant_config);
        errdefer cache.deinit();

        // Per-slot SSM cache. Mirror the same predicate `Transformer.init`
        // uses to allocate `xfm.ssm_entries` (transformer.zig: `has_hybrid_layers`
        // OR `full_attention_interval > 0`). Without this branch the
        // slot's `ctx.ssm_entries` is null and `forwardMoeWith`'s
        // linear-attention layers crash on `ctx.ssm_entries.?` for Qwen 3.5/3.6
        // MoE (which carries GatedDeltaNet inside its MoE structure but does
        // NOT set `has_hybrid_layers`). Pure-attention MoE (qwen3_moe, Gemma 4
        // MoE — isMoe() but interval == 0) must NOT get entries: a non-null
        // slice makes the hot prefix cache treat the model as hybrid and
        // cold-prefill every request.
        var ssm_entries: ?[]SSMCacheEntry = null;
        if (!is_embedded and config.needsSsmEntries()) {
            const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
            for (entries) |*e| {
                e.* = .{
                    .conv_state = mlx.mlx_array_new(),
                    .ssm_state = mlx.mlx_array_new(),
                    .initialized = false,
                };
            }
            ssm_entries = entries;
        }
        errdefer if (ssm_entries) |entries| {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                transformer_mod.ssmFreeQsaState(e);
            }
            allocator.free(entries);
        };

        // Dup owned slices.
        const prompt_owned = try allocator.dupe(u32, params.prompt_ids);
        errdefer allocator.free(prompt_owned);
        const full_prompt_src = params.full_prompt orelse params.prompt_ids;
        const full_prompt_owned = try allocator.dupe(u32, full_prompt_src);
        errdefer allocator.free(full_prompt_owned);
        const media_owned = try allocator.dupe(prefix_cache_mod.MediaSpan, params.media);
        errdefer allocator.free(media_owned);
        const eos_owned = try allocator.dupe(u32, params.eos_token_ids);
        errdefer allocator.free(eos_owned);

        slot.* = .{
            .allocator = allocator,
            .io = io,
            .model = params.model,
            .cache = cache,
            .moe_seq_offset = 0,
            .ssm_entries = ssm_entries,
            .vision_embeddings = params.vision_embeddings,
            .media = media_owned,
            .cache_key = params.cache_key,
            .client = params.client,
            .request_id = metrics_mod.nextRequestId(),
            .mrope_pos = params.mrope_pos,
            .mrope_total = params.mrope_total,
            .mrope_delta = params.mrope_delta,
            .ctx = undefined, // set after slot is in stable storage so pointers are valid
            .legacy_gen = null,
            .prompt_ids = prompt_owned,
            .full_prompt = full_prompt_owned,
            .sampling = params.sampling,
            .eos_token_ids = eos_owned,
            .max_tokens = params.max_tokens,
            .timeout_ns = params.timeout_ns,
            .has_tools = params.has_tools,
            .enable_thinking = params.enable_thinking,
            // A spoken reply forwards one row a slot: no speculative round.
            .enable_pld = params.enable_pld and params.speech == null,
            // Muse and Nemotron RADIO DFlash consume vision-conditioned
            // trunk captures; Qwen's external drafter lacks M-RoPE positions.
            .enable_drafter = assistantSidecarEnabledForRequest(
                params.enable_drafter and params.speech == null,
                params.vision_embeddings != null,
                params.mrope_pos != null,
                params.dflash != null,
                config,
            ),
            .drafter = params.drafter,
            .dflash = params.dflash,
            .drafter_block_size = params.drafter_block_size,
            .enable_mtp = params.enable_mtp and params.speech == null,
            .speech = params.speech,
            .allow_batch_mtp = params.allow_batch_mtp,
            .mtp = if (params.mtp) |m| m.* else null,
            .mtp_depth = params.mtp_depth,
            .pld_draft_len = params.pld_draft_len,
            .pld_key_len = params.pld_key_len,
            .kv_attn_fused = params.kv_attn_fused,
            .cached_tokens = params.cached_tokens,
            .logprobs_n = params.logprobs_n,
            .state = .pending_prefill,
            .sink = params.sink,
            .out_mu = .init,
            .out_cond = .init,
            .out_event = .unset,
            .out_buf = std.ArrayList(u32).empty,
            .out_idx = 0,
            .finished = false,
            .error_code = null,
            .finish_reason = "length",
            .finish_details = null,
            .loop_trim_start = null,
            .cancelled = std.atomic.Value(bool).init(false),
            .constraint_payload_at = std.atomic.Value(u32).init(std.math.maxInt(u32)),
            .constraint_payload_offset = std.atomic.Value(u32).init(0),
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .prefill_tps = 0.0,
            .decode_tps = 0.0,
            .request_start_ts = std.Io.Timestamp.now(io, .boot),
            .first_token_ns = 0,
            .prefill_ns = 0,
            .prefill_interleaved_ns = 0,
            .decode_ns = 0,
            .generated_ids = null,
            .was_pad_only = true,
            .logprobs_buf = .empty,
            .prompt_logprobs = params.prompt_logprobs,
            .prompt_logprobs_buf = .empty,
            .skip_prefix_cache = params.prompt_logprobs,
        };

        // ForwardCtx points at fields owned by `slot` — must outlive the
        // Generator. The slot is heap-allocated so addresses are stable
        // until `complete` frees it.
        slot.ctx = .{
            .cache = &slot.cache,
            .moe_seq_offset = &slot.moe_seq_offset,
            .ssm_entries = slot.ssm_entries,
            .ssm_member_gen = transformer_mod.nextSsmMemberGen(),
            .vision_embeddings = slot.vision_embeddings,
            .mrope_pos = slot.mrope_pos,
            .mrope_total = slot.mrope_total,
            .mrope_delta = slot.mrope_delta,
            .capture_hidden = null,
            .kv_attn_fused = params.kv_attn_fused,
        };

        return slot;
    }

    /// Free everything the slot owns. Only safe to call when no thread can
    /// observe the slot anymore (i.e. after the inference thread has
    /// finished/errored it AND the connection thread has consumed the final
    /// `done`/`err` from `waitNext`).
    pub fn deinit(self: *Slot) void {
        // A slot freed before its terminator (a cancel pulled into cleanup) still tells its stream it is done.
        if (self.sink != null) self.emitFinished(.cancelled, null);
        if (self.model.transformer) |xfm| if (xfm.qwen4Trunk()) |t| t.markQsaPooledRopeStale();
        if (self.legacy_gen) |*gen| {
            gen.deinit(self.allocator);
        }
        // Salvaged-but-never-consumed cancelled-prefill checkpoints.
        self.cancelled_prefill.deinit();
        self.cache.deinit();
        if (self.ssm_entries) |entries| {
            if (self.model.transformer) |xfm| xfm.ssmGroupDrop(entries);
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                transformer_mod.ssmFreeQsaState(e);
            }
            self.allocator.free(entries);
        }
        if (self.vision_embeddings) |ve| _ = mlx.mlx_array_free(ve);
        if (self.mrope_pos) |mp| self.allocator.free(mp);
        self.allocator.free(self.prompt_ids);
        self.allocator.free(self.full_prompt);
        self.allocator.free(self.media);
        self.allocator.free(self.eos_token_ids);
        if (self.error_code) |code| self.allocator.free(code);
        if (self.generated_ids) |g| self.allocator.free(g);
        // Free any logprobs the conn thread didn't claim. After
        // `nonStreamingViaScheduler` calls `toOwnedSlice`, items.len becomes
        // 0 so this is a no-op on the success path.
        for (self.logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.logprobs_buf.deinit(self.allocator);
        for (self.prompt_logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.prompt_logprobs_buf.deinit(self.allocator);
        self.out_buf.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Inference thread: the echo sink, emptied first, since a retried prefill starts over.
    pub fn freshPromptLogprobs(self: *Slot) *std.ArrayList(generate_mod.LogprobResult) {
        for (self.prompt_logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.prompt_logprobs_buf.clearRetainingCapacity();
        return &self.prompt_logprobs_buf;
    }

    /// Inference thread: record the reasoning→payload boundary so it is
    /// visible to the conn thread no later than the token that carries it.
    pub fn publishConstraintSpan(self: *Slot, token_index: u32, byte_offset: u32) void {
        self.constraint_payload_offset.store(byte_offset, .release);
        self.constraint_payload_at.store(token_index, .release);
        if (self.sink != null) self.sink_constraint = .{ token_index, byte_offset };
    }

    /// Connection thread: where the constrained payload begins relative to
    /// the generated stream, or null while it has not begun. `token_index`
    /// matches the reader's own token count (token i carries the payload
    /// start when the returned index equals i).
    pub fn constraintPayloadStart(self: *const Slot) ?rp_mod.ConstraintSpan {
        const at = self.constraint_payload_at.load(.acquire);
        if (at == std.math.maxInt(u32)) return null;
        return .{ .token_index = at, .byte_offset = self.constraint_payload_offset.load(.acquire) };
    }

    /// Inference thread: enqueue a generated token for the consumer.
    pub fn pushToken(self: *Slot, t: u32) void {
        self.pushTokenWithLogprob(t, null);
    }

    /// Publish one token and, when the request asked for logprobs, the entry
    /// describing it — in ONE critical section.
    ///
    /// Both must move under `out_mu` together. The streaming path reads entry
    /// i as soon as it is handed token i, so an append outside the lock races
    /// the reader two ways: the entry may not be visible yet (a silently
    /// missing trailing logprob), and a concurrent grow reallocates the
    /// backing array under a reader mid-copy. Non-streaming consumes the whole
    /// buffer at completion and is blind to both, which is why the streaming
    /// gap survived: it is invisible to output-equality tests AND to llmprobe,
    /// which probes logprobs non-streaming only.
    pub fn pushTokenWithLogprob(self: *Slot, t: u32, lp: ?generate_mod.LogprobResult) void {
        // The prefill's token went out early (`publishFirstToken`): the decoder's own push of it is swallowed.
        if (self.early_first) |e| {
            self.early_first = null;
            if (e == t and lp == null) return;
        }
        // A request with a stream reads it there alone: the event carries the token and a copy of its logprob.
        if (self.sink != null) {
            self.emitToken(t, lp);
            if (lp) |entry| self.allocator.free(entry.top_logprobs);
            return;
        }
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (lp) |entry| {
            // Degrade to error like the token append below — and do NOT return:
            // the broadcast at the bottom is what wakes a reader blocked on the
            // condvar, so an early exit here trades an OOM for a hang.
            self.logprobs_buf.append(self.allocator, entry) catch |err| {
                self.error_code = self.allocator.dupe(u8, @errorName(err)) catch null;
                self.state = .errored;
            };
        }
        self.out_buf.append(self.allocator, t) catch |err| {
            // Allocation failure: degrade to error.
            self.error_code = self.allocator.dupe(u8, @errorName(err)) catch null;
            self.state = .errored;
        };
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Connection thread: copy logprob entries produced since `cursor` into
    /// `out`, returning the new cursor. Under `out_mu`, so it cannot observe a
    /// half-written entry or a reallocating buffer.
    ///
    /// The COPY is shallow and that is deliberate: each entry's `top_logprobs`
    /// is its own allocation, stable for the life of the slot, and owned by
    /// the slot (`Slot.deinit` frees it). The caller borrows and must not free
    /// — only the ArrayList's backing array is at risk from a concurrent grow,
    /// and that is exactly what the lock covers.
    pub fn copyLogprobsFrom(
        self: *Slot,
        allocator: std.mem.Allocator,
        cursor: usize,
        out: *std.ArrayList(generate_mod.LogprobResult),
    ) !usize {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const items = self.logprobs_buf.items;
        if (cursor >= items.len) return cursor;
        try out.appendSlice(allocator, items[cursor..]);
        return items.len;
    }

    /// Inference thread: signal normal completion. Safe to call multiple
    /// times (idempotent on `finished`).
    pub fn markFinished(self: *Slot, reason: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.finished) return;
        self.finished = true;
        self.state = .finished;
        self.finish_reason = reason;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
        self.emitFinished(if (std.mem.eql(u8, reason, "length")) .length else if (std.mem.eql(u8, reason, "cancelled")) .cancelled else .stop, null);
    }

    /// Inference thread: signal error. `name` is borrowed; we dupe so the
    /// connection thread can read it after the inference loop drops the slot.
    /// Whether a slot error name names a memory failure (`OutOfMemory` from the MLX latch or Zig's allocator).
    pub fn errorNameIsMemory(name: []const u8) bool {
        return std.mem.eql(u8, name, "OutOfMemory") or
            std.mem.eql(u8, name, "InsufficientMemory");
    }

    /// Whether the slot's latched error name is exactly `name`.
    pub fn errorNameIs(self: *Slot, name: []const u8) bool {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const have = self.error_code orelse return false;
        return std.mem.eql(u8, have, name);
    }

    pub fn errorIsMemory(self: *Slot) bool {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const name = self.error_code orelse return false;
        return errorNameIsMemory(name);
    }

    pub fn markError(self: *Slot, name: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.error_code != null or self.finished) return;
        self.error_code = self.allocator.dupe(u8, name) catch null;
        self.state = .errored;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
        self.emitFinished(.failed, name);
    }

    /// The prefill is done: the prompt's counts and its logprobs go out before its first token.
    pub fn emitPrefilled(self: *Slot) void {
        const sk = self.sink orelse return;
        var lps: std.ArrayList(abi.Logprob) = .empty;
        defer lps.deinit(self.allocator);
        var tops: std.ArrayList(abi.Bias) = .empty;
        defer tops.deinit(self.allocator);
        for (self.prompt_logprobs_buf.items) |lp| tops.appendSlice(self.allocator, @ptrCast(lp.top_logprobs)) catch {};
        var at: usize = 0;
        for (self.prompt_logprobs_buf.items) |lp| {
            lps.append(self.allocator, .{ .token = 0, .logprob = lp.token_logprob, .top = abi.span(abi.Bias, tops.items[at..][0..lp.top_logprobs.len]) }) catch {};
            at += lp.top_logprobs.len;
        }
        const ev: abi.Event = .{ .kind = .prefilled, .cached = self.cached_tokens, .prompt_tokens = self.prompt_tokens, .prompt_logprobs = abi.span(abi.Logprob, lps.items) };
        if (sk.event(sk.ctx, @intFromPtr(self), &ev) == .stop) self.cancelled.store(true, .release);
    }

    fn emitToken(self: *Slot, t: u32, lp: ?generate_mod.LogprobResult) void {
        const sk = self.sink orelse return;
        const one = [1]u32{t};
        var entry: [1]abi.Logprob = undefined;
        if (lp) |e| entry[0] = .{ .token = t, .logprob = e.token_logprob, .top = abi.span(abi.Bias, @ptrCast(e.top_logprobs)) };
        const span = self.sink_constraint;
        self.sink_constraint = null;
        const ev: abi.Event = .{
            .kind = .tokens,
            .tokens = abi.span(u32, &one),
            .logprobs = if (lp != null) abi.span(abi.Logprob, &entry) else .{},
            .constraint_at = if (span) |c| c[0] else std.math.maxInt(u32),
            .constraint_offset = if (span) |c| c[1] else 0,
        };
        if (sk.event(sk.ctx, @intFromPtr(self), &ev) == .stop) self.cancelled.store(true, .release);
    }

    /// Spoken frames and the samples they made final (`.audio`); a request without a stream has no reader for them.
    pub fn emitAudio(self: *Slot, codes: []const u32, samples: []const f32, sample_rate: u32) void {
        const sk = self.sink orelse return;
        const ev: abi.Event = .{ .kind = .audio, .codes = abi.span(u32, codes), .samples = abi.span(f32, samples), .sample_rate = sample_rate };
        if (sk.event(sk.ctx, @intFromPtr(self), &ev) == .stop) self.cancelled.store(true, .release);
    }

    fn emitFinished(self: *Slot, reason: @FieldType(abi.Event, "reason"), failure: ?[]const u8) void {
        const sk = self.sink orelse return;
        self.sink = null;
        const st: abi.Stats = .{ .prefill_ns = self.prefill_ns, .decode_ns = self.decode_ns };
        const ev: abi.Event = .{
            .kind = .finished,
            .reason = reason,
            .stats = &st,
            .verdict = if (failure) |name| .{ .code = .failed, .why = abi.Str.of(name) } else .{},
            .finish_details = if (self.finish_details) |d| abi.Str.of(d) else .{},
            .loop_trim_start = if (self.loop_trim_start) |i| i else std.math.maxInt(u64),
            .refused_needed = if (self.refused_bill) |b| b[0] else 0,
            .refused_available = if (self.refused_bill) |b| b[1] else 0,
        };
        _ = sk.event(sk.ctx, @intFromPtr(self), &ev);
    }

    /// Connection thread: block until the next token, completion, or error.
    /// Returns `.token` for each generated id, then exactly one terminator
    /// (`.done` or `.err`).
    pub fn waitNext(self: *Slot) NextResult {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        while (true) {
            if (self.out_idx < self.out_buf.items.len) {
                const t = self.out_buf.items[self.out_idx];
                self.out_idx += 1;
                return .{ .token = t };
            }
            if (self.error_code != null) return .{ .err = {} };
            if (self.finished) return .{ .done = {} };
            // Cancellation (client disconnect or server shutdown) must unblock a
            // blocked reader — `cancel()` only broadcasts; without this the
            // reader would sleep until the inference thread happened to finish
            // the slot, so shutdown could never drain in-flight requests and
            // raced `Scheduler.deinit` into a use-after-free (SIGSEGV in
            // `complete`). Buffered tokens above still drain first.
            if (self.cancelled.load(.acquire)) return .{ .done = {} };
            self.out_cond.waitUncancelable(self.io, &self.out_mu);
        }
    }

    /// Connection thread: like `waitNext`, but wakes with `null` (idle)
    /// after `timeout_ms` with no token or terminator. Lets the caller poll
    /// the peer socket and emit SSE keepalives during long prefills —
    /// Claude Code disconnects after ~60s of stream silence, and pre-2026-06
    /// the handler sat blocked in `waitNext` for the whole multi-minute
    /// prefill, never noticed the disconnect, and abandoned giant prefills
    /// piled up serially behind every client retry (the server looked dead
    /// while the GPU ground ghosts; observed live with Claude Code + a
    /// 40K-token MCP prompt on gemma-4-12b).
    pub fn waitNextTimeout(self: *Slot, timeout_ms: i64) ?NextResult {
        while (true) {
            self.out_mu.lockUncancelable(self.io);
            if (self.out_idx < self.out_buf.items.len) {
                const t = self.out_buf.items[self.out_idx];
                self.out_idx += 1;
                self.out_mu.unlock(self.io);
                return .{ .token = t };
            }
            if (self.error_code != null) {
                self.out_mu.unlock(self.io);
                return .{ .err = {} };
            }
            if (self.finished) {
                self.out_mu.unlock(self.io);
                return .{ .done = {} };
            }
            // Cancellation unblocks the reader promptly (see waitNext) so a
            // disconnected/shutdown request stops instead of waiting out the
            // generation — the precondition for draining conn threads before
            // teardown.
            if (self.cancelled.load(.acquire)) {
                self.out_mu.unlock(self.io);
                return .{ .done = {} };
            }
            // Arm the event under the lock: producers mutate under this lock
            // and set() before releasing it, so a set racing our reset leaves
            // the event set and the wait below returns immediately — no lost
            // wakeups. A spurious wake just reads as an early idle (benign).
            self.out_event.reset();
            self.out_mu.unlock(self.io);
            self.out_event.waitTimeout(self.io, .{ .duration = .{
                .raw = .fromMilliseconds(timeout_ms),
                .clock = .awake,
            } }) catch return null;
        }
    }

    /// Connection thread: signal cancellation. The inference thread will
    /// drop this slot at the next tick boundary.
    pub fn cancel(self: *Slot) void {
        self.cancelled.store(true, .release);
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }
};
