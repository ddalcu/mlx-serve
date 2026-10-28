//! Generator.initWithOptions: the prompt prefill and every spec-decode arm's setup.
const speech_mod = @import("speech.zig");
const depth_bounds = @import("../spec/mtp_depth_bounds.zig");
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const mlx_stream = @import("../modules/stream.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const mtp_acceptance = @import("../../../sdk/spec/mtp_acceptance.zig");
const round_cost = @import("../spec/round_cost.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ForwardCtx = transformer_mod.ForwardCtx;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const captureSsmCheckpoint = transformer_mod.captureSsmCheckpoint;
const dflash_mod = @import("../spec/dflash/dflash.zig");
const generate_mod = @import("../generate.zig");
const AdaptiveWidthState = @import("prefill_chunk.zig").AdaptiveWidthState;
const DsparkArm = @import("../spec/dspark.zig").DsparkArm;
const Generator = @import("generator.zig").Generator;
const InitOptions = @import("options.zig").InitOptions;
const MtpCacheRef = @import("../spec/mtp_head_ref.zig").MtpCacheRef;
const MtpGraphFn = @import("../spec/tree.zig").MtpGraphFn;
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const chunkNeedsMtpHistory = @import("prefill_chunk.zig").chunkNeedsMtpHistory;
const commitAdaptiveWidth = @import("prefill_chunk.zig").commitAdaptiveWidth;
const countSpliceRows = @import("prefill_chunk.zig").countSpliceRows;
const decodeShareCapped = @import("prefill_chunk.zig").decodeShareCapped;
const dsparkArmFor = @import("../spec/dspark.zig").dsparkArmFor;
const dsparkEnabled = @import("../spec/dspark.zig").dsparkEnabled;
const effectiveMtpHistoryWindow = @import("prefill_chunk.zig").effectiveMtpHistoryWindow;
const effectivePrefillChunk = @import("prefill_chunk.zig").effectivePrefillChunk;
const effectiveSsmCheckpointStride = @import("prefill_chunk.zig").effectiveSsmCheckpointStride;
const firstTokenLogprobs = @import("../sampling/logprobs.zig").firstTokenLogprobs;
const lazyForward = @import("step.zig").lazyForward;
const mtpBatchedExactGraph = @import("../spec/mtp/batched_graph.zig").mtpBatchedExactGraph;
const mtpBatchedTokenV3Graph = @import("../spec/mtp/batched_graph.zig").mtpBatchedTokenV3Graph;
const mtpBatchedTypicalGraph = @import("../spec/mtp/batched_graph.zig").mtpBatchedTypicalGraph;
const mtpDepthCapFree = @import("../spec/mtp/ev_plan.zig").mtpDepthCapFree;
const mtpDraftGreedyFor = @import("../spec/mtp/depth.zig").mtpDraftGreedyFor;
const mtpEvCosts = @import("../spec/mtp/ev_costs.zig").mtpEvCosts;
const nextChunkEnd = @import("prefill_chunk.zig").nextChunkEnd;
const pickLookupPromptSource = @import("../spec/pld.zig").pickLookupPromptSource;
const promptLogprobs = @import("../sampling/logprobs.zig").promptLogprobs;
const reservedPrefillTokens = @import("prefill_chunk.zig").reservedPrefillTokens;
const resolveMtpDepthCapForProfile = @import("../spec/mtp/ev_plan.zig").resolveMtpDepthCapForProfile;
const sampleTokenLazy = @import("../sampling/lazy.zig").sampleTokenLazy;
const shouldCheckpointSsmPrefill = @import("prefill_chunk.zig").shouldCheckpointSsmPrefill;
const ssmCapturePos = @import("prefill_chunk.zig").ssmCapturePos;
const ssmSnapshotBackoff = @import("prefill_chunk.zig").ssmSnapshotBackoff;
const intree = @import("../plugin/intree.zig");
const plugin_driver = @import("../plugin/driver.zig");


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
    sampling.keyed = xfm.config.rowExactDecode() and mlx.metalKernelsAvailable(); // the keyed sampler is a Metal kernel
    sampling.position_base = pickLookupPromptSource(prompt_ids, options_in.lookup_prompt).len;
    // Spec drafts and verify rows share the serial sampler's keys (and
    // mlx-stream's lane draws from the seed), so an unseeded sampled
    // request gets its own random seed.
    if ((sampling.keyed or xfm.mlx_stream_ext != null) and sampling.seed == null and sampling.temperature > 0.01) {
        var prng = std.Random.DefaultPrng.init(@truncate(@as(u96, @bitCast(std.Io.Timestamp.now(io, .real).toNanoseconds())) ^ @intFromPtr(xfm)));
        sampling.seed = prng.random().int(u64);
    }
    // DeepSeek-V4 hard-off, at the ONE chokepoint every init site
    // funnels through: dsv4's per-request state lives on the module
    // (rings + compressed caches) and a spec VERIFY forward appends
    // draft tokens to it with NO rollback — two rejected PLD drafts
    // permanently corrupted a live generation (mangled DSML with dropped
    // token runs; the per-site `is_dsv4` wiring guard in
    // scheduler.runPrefill demonstrably did not cover the engaged path,
    // and per-site wiring is the class the spec-dispatch rule warns
    // about).
    var options = options_in;
    var dspark_active = false;
    var dspark_stochastic = false;
    if ((intree.dsv4(xfm) != null or intree.dsv41(xfm) != null or xfm.mlx_stream_ext != null or plugin_driver.nativeDrafter(xfm) != null) and (options.pld_enabled or options.drafter_enabled or options.mtp_enabled or options.dflash_enabled)) {
        // DSpark lift: dsv4's OWN draft mode (block-parallel stages +
        // snapshot rollback inside deepseek_v4.zig) may engage when the
        // checkpoint ships stages and the request is CLEAN (no
        // penalties, grammar or logprobs — those consume logits the
        // draft path never shapes and stay serial). Greedy requests get
        // the raw argmax-equality accept; sampled requests get the
        // stochastic arm (MTP one-hot Leviathan acceptance over the
        // request's own filtered probs — the agent-default temp 0.6
        // traffic that otherwise always ran serial). PLD / drafter / qwen-MTP
        // remain hard-off regardless: their verify forwards go through
        // machinery this arch cannot roll back.
        const native = plugin_driver.nativeDrafter(xfm) != null;
        const ds_block = if (intree.dsv4(xfm)) |d| d.ds_block else if (intree.dsv41(xfm)) |d| d.ds_block else if (native) xfm.config.plugin.?.parsed.spec.native_block else mlx_stream.blockSize(xfm.mlx_stream_ext.?);
        const dspark_env_off = !dsparkEnabled();
        // A plugin whose drafter cannot hand the verify back accepts greedily: a sampled request runs serial.
        const arm = if (!options.mtp_enabled or xfm.config.mtp_override == false)
            DsparkArm.off
        else if (native and !plugin_driver.nativeSplit(xfm) and dsparkArmFor(sampling, options.logprobs_n) == .stochastic)
            DsparkArm.off
        else
            dsparkArmFor(sampling, options.logprobs_n);
        // mlx-stream's lane samples a sampled request itself, from the request's own settings.
        const lane_sampling: ?mlx_stream.SamplingParams = if (dspark_env_off or arm == .off) null else .{
            .temperature = sampling.temperature,
            .top_p = sampling.top_p,
            .top_k = sampling.top_k,
            .min_p = sampling.min_p,
            .seed = sampling.seed orelse 0,
        };
        const lane_arms = if (xfm.mlx_stream_ext) |m| mlx_stream.arm(m, lane_sampling) else true;
        if (xfm.dsparkStages() > 0 and !dspark_env_off and arm != .off and lane_arms) {
            dspark_active = true;
            dspark_stochastic = arm == .stochastic;
            if (dspark_stochastic) {
                log.info("  spec=dspark (stochastic; {s} native draft stages, block={d})\n", .{ xfm.config.model_type, ds_block });
            } else {
                log.info("  spec=dspark ({s} native draft stages, block={d})\n", .{ xfm.config.model_type, ds_block });
            }
        } else {
            log.info("  spec=disabled ({s} serves serial-only)\n", .{xfm.config.model_type});
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
    // Vision requests chunk too: the splice resumes its row index across
    // chunks (#197). An explicit `--prefill-chunk` outranks the machine pin;
    // the safety cap below still applies.
    //
    // Safety cap: on unfused head dims (>128 — every Gemma-4/Qwen3.5/3.6)
    // the composed SDPA materializes [heads, chunk, total_kv] scores per
    // layer, so the chunk shrinks with the prompt's FINAL KV length to keep
    // that one tensor bounded (boundedPrefillChunk). Warm-path restores
    // start at ssm_checkpoint_pos_offset, so the final KV length is that
    // offset plus everything we're about to forward.
    const total_ctx_for_chunk = options.ssm_checkpoint_pos_offset + prompt_ids.len;
    const PREFILL_CHUNK: usize = decodeShareCapped(effectivePrefillChunk(
        xfm.config.prefillScoreHeadDim(),
        xfm.config.num_attention_heads,
        total_ctx_for_chunk,
        xfm.config.has_sliding_window,
        xfm.config.isMoe(),
        xfm.config.longCtxGated(),
        options.pinned_prefill_chunk,
    ), options.decode_share_width_cap);
    // Phase-level prefill instrumentation. Enabled at debug level OR via
    // diag prefill-trace (which forces the trace line at info).
    // Phase 0 of plan 04 — gives us a decomposed view of where cold prefill
    // time goes (chunked-forward vs eval vs last-token-forward).
    const trace_force: bool = generate_mod.prefill_trace_force or env.one(.PREFILL_TRACE);
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
        effectiveSsmCheckpointStride(@intCast(options.ssm_checkpoint_stride), @max(PREFILL_CHUNK, generate_mod.prefillCli().override))
    else
        0;
    // Absolute KV position of `prompt_ids[0]`. Warm-path callers (the
    // scheduler after restoring a checkpoint) pass the matched prefix
    // length so the snapshots stamp positions valid in the full original
    // sequence, not relative offsets inside the tail-only prefill.
    const ssm_cp_offset: usize = options.ssm_checkpoint_pos_offset;
    // Gate for the dense low grid (#794): the same predicate as the
    // checkpoint-retention `ssm_low_anchors` wiring. A non-gated arch keeps the
    // coarsened stride's own boundary chunking — the grid adds it no chunks.
    const dense_low_grid: bool = xfm.config.longCtxGated();

    // Qwen native MTP: build the head's committed-history KV cache during
    // prefill. Entry j pairs (trunk hidden at prompt position j, token at
    // j+1); the (hidden[last], t1) pair is appended by the first nextMtp
    // round. On KV-prefix reuse the history covers only the freshly
    // forwarded tail — RoPE offsets are cache-relative, so a late-starting
    // history is self-consistent (sliding-window history semantics).
    const mtp_active = options.mtp_enabled and options.mtp != null;
    const AcceptRoute = struct {
        graph: MtpGraphFn,
        prefix: mtp_acceptance.PrefixFn,
        param: f32,
    };
    const accept_route: AcceptRoute = switch (options.mtp_acceptance) {
        .exact => .{ .graph = mtpBatchedExactGraph, .prefix = mtp_acceptance.exactPrefix, .param = 0 },
        .typical => |cfg| .{ .graph = mtpBatchedTypicalGraph, .prefix = mtp_acceptance.typicalPrefix, .param = cfg.delta },
        .tokenv3 => |alpha| .{ .graph = mtpBatchedTokenV3Graph, .prefix = mtp_acceptance.tokenV3Prefix, .param = alpha },
    };
    if (mtp_active and std.meta.activeTag(options.mtp_acceptance) != .exact) {
        switch (options.mtp_acceptance) {
            .typical => |cfg| log.info("  [mtp-accept] typical delta={d:.3} eps={d:.3} NOT distribution-exact\n", .{ cfg.delta, cfg.eps }),
            .tokenv3 => |alpha| log.info("  [mtp-accept] TokenV3 alpha={d:.3} NOT distribution-exact\n", .{alpha}),
            .exact => unreachable,
        }
    }
    if (mtp_active and options.mtp_greedy_tail and sampling.temperature > 0.01 and !mtpDraftGreedyFor(sampling))
        log.info("  [mtp-draft] greedy tail engaged: depth 0 samples, later depths take the head's argmax\n", .{});
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
        dfl_cl = .{ .ids = options.dflash.?.config.captureIds(), .out = dfl_out_buf };
    }

    // Start of the final (logits) forward's token span. Without SSM
    // checkpointing this is the last prompt token (the classic 1-token
    // logits forward); with checkpointing the chunk loop stops
    // SSM_SNAPSHOT_BACKOFF tokens early so the always-on snapshot lands
    // where the next turn's prefix match can actually reach it (see
    // ssmSnapshotBackoff), and the final forward covers the held-back
    // tail + the last token in the same weight sweep.
    var final_start: usize = if (xfm.prefillsWholePrompt()) 0 else prompt_ids.len - 1;
    // Chunked vision prefill (issue #197): placeholder rows consumed by
    // completed chunks, fed to ctx.vision_splice_offset per forward
    // (chunk loop AND final-span forward).
    var vision_rows_consumed: usize = options.vision_rows_before;
    var adapt_state: AdaptiveWidthState = .{
        .width_min = @intCast(PREFILL_CHUNK),
        .width_max = @intCast(PREFILL_CHUNK),
    };

    if (final_start > 0) {
        const prefix_len = prompt_ids.len - 1;
        const snapshot_backoff = ssmSnapshotBackoff(want_state_cp, prefix_len, ssm_cp_offset > 0);
        const loop_end = prefix_len - snapshot_backoff;
        final_start = loop_end;
        // Per-chunk adaptive width: the first chunk runs the admitted width; every boundary
        // after it re-asks the same estimator. `cap_adapt` is the widest this arch forwards
        // for this prompt, never wider than `ssm_cp_stride`.
        // The scaled tail-merge bound reads the arch predicate, never `chunk_width_hook != null`
        // (installed process-wide).
        const width_is_adaptive = options.adaptive_chunk_width;
        const cap_adapt: u32 = @intCast(effectivePrefillChunk(
            xfm.config.prefillScoreHeadDim(),
            xfm.config.num_attention_heads,
            total_ctx_for_chunk,
            xfm.config.has_sliding_window,
            xfm.config.isMoe(),
            xfm.config.longCtxGated(),
            0,
        ));
        adapt_state.width_min = @intCast(PREFILL_CHUNK);
        adapt_state.width_max = @intCast(PREFILL_CHUNK);
        var cur_chunk: usize = PREFILL_CHUNK;
        // Last-window MTP history: chunks entirely before the window skip
        // the full-hidden capture AND the head forward (see
        // mtp.SUGGESTED_HISTORY_WINDOW). 0 = capture every chunk.
        const mtp_hist_window = effectiveMtpHistoryWindow(prefix_len, generate_mod.mtp_history_window_override);

        // Reserve the whole request's cache capacity before the first chunk writes: a grow
        // is not in place, so a long prefill's peak carried a second copy of everything so
        // far (~7.75 GB at 458k on qwen4_exp, #353). Same number the admission guard billed
        // (`KVCache.reservedTokens`), in ABSOLUTE positions: `prompt_ids` is the tail after
        // a prefix-cache hit. `cp_thin` names this site's previous retention policy.
        const cp_thin: transformer_mod.ThinPolicy =
            if (xfm.config.longCtxGated()) .min_span_recency else .oldest;

        const reserved_tokens = reservedPrefillTokens(
            &xfm.config,
            total_ctx_for_chunk,
            max_tokens,
            PREFILL_CHUNK,
        );
        ctx.cache.reserve(@intCast(reserved_tokens));
        // The arch's own per-request buffers reserve at the same length.
        if (mtp_cache) |*mc| mc.activate();
        transformer_mod.reserveQsaHistoryWithHead(
            ctx.ssm_entries,
            if (xfm.qwen4Trunk()) |t| if (t.qwen4_mtp) |*m| &m.entry else null else null,
            @intCast(reserved_tokens),
        );

        var pos: usize = 0;
        if (options.prefill_expected) |e| e.store(@intCast(loop_end), .monotonic);
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
            // An MLX failure in a previous chunk latched instead of killing the process (#353);
            // abandon this request, keep the server. No checkpoint salvage: the state failed.
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
            const end = nextChunkEnd(pos, loop_end, cur_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset, width_is_adaptive, dense_low_grid);
            if (has_vision) ctx.vision_splice_offset = vision_rows_consumed;
            const chunk_len: c_int = @intCast(end - pos);
            const chunk_shape = [_]c_int{ 1, chunk_len };
            const chunk_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[pos]), &chunk_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(chunk_input);

            const chunk_start_ns = if (trace_enabled) prefill_sw.read() else 0;
            // History windowing: a chunk before the window needs no capture.
            const mtp_capture = mtp_active and chunkNeedsMtpHistory(pos, end, prefix_len, mtp_hist_window);
            var chunk_hidden_all = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(chunk_hidden_all);
            if (dflash_active) {
                for (dfl_out_buf) |*a| a.* = mlx.mlx_array_new();
                ctx.capture_layers = &dfl_cl;
            }
            errdefer if (dflash_active) {
                for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
            };
            const chunk_logits = if (mtp_capture) blk: {
                var last_unused = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(last_unused);
                break :blk try xfm.forwardWithCaptureAll(&ctx, chunk_input, &last_unused, &chunk_hidden_all);
            } else try xfm.forwardWith(&ctx, chunk_input);
            if (options.prompt_logprobs) |out| promptLogprobs(allocator, chunk_logits, prompt_ids[pos + 1 .. end + 1], options.logprobs_n, xfm.s, out) catch |err| {
                _ = mlx.mlx_array_free(chunk_logits);
                return err;
            };
            _ = mlx.mlx_array_free(chunk_logits);
            if (dflash_active) {
                ctx.capture_layers = null;
                try dflash_mod.appendContext(options.dflash.?, &dflash_ctx.?, dfl_out_buf, ssm_cp_offset + pos);
                for (dfl_out_buf) |*a| {
                    _ = mlx.mlx_array_free(a.*);
                    a.* = .{ .ctx = null };
                }
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

            // This chunk's latch, read before anything persists it, snapshots it or yields
            // the thread: Metal returns zeros before it aborts, so a later read let the
            // write-through index garbage and a co-tenant decode eat the latch.
            try mlx.checkError();

            // Phase 1: snapshot SSM state at stride-aligned boundaries.
            // We snapshot AFTER the eval above so the underlying buffers
            // are realized; the snapshot is just a refcount-share of the
            // already-resident state.
            const abs_end_for_cp2 = end + ssm_cp_offset;
            if (want_ssm_cp and ssmCapturePos(ssm_cp_stride, abs_end_for_cp2, dense_low_grid)) {
                const cp = try captureSsmCheckpoint(allocator, ctx.ssm_entries.?, abs_end_for_cp2, xfm.s);
                ssm_checkpoints.append(allocator, cp) catch |e| {
                    var doomed = cp;
                    doomed.deinit(allocator);
                    return e;
                };
                // The head's own QSA leftover at this position: its ring is 32 rows and
                // the clamp back to this checkpoint comes a whole generated tail later.
                if (mtp_active) {
                    mtp_cache.?.activate();
                }
                if (mtp_active) if (xfm.qwen4Trunk()) |t| t.qwen4MtpMarkQsaLeftover(abs_end_for_cp2 -| mtp_position_base) catch |e| {
                    log.debug("[qwen4] MTP head leftover mark failed: {s}\n", .{@errorName(e)});
                };
                // Thin the interior, never the oldest (#330): drop-oldest survivors covered only
                // the last `max * stride` tokens and left no affordable trim point.
                if (options.ssm_checkpoint_max > 0 and
                    ssm_checkpoints.items.len > options.ssm_checkpoint_max)
                {
                    var dropped = ssm_checkpoints.orderedRemove(
                        transformer_mod.ssmCheckpointDropIndexLowAnchored(
                            ssm_checkpoints.items,
                            cp_thin,
                            null,
                            options.ssm_low_anchors,
                        ),
                    );
                    dropped.deinit(allocator);
                }
            }

            if (has_vision) {
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
            // Persist this chunk now: the KV bytes for [0, pos) are final.
            if (options.write_through_hook) |wt| {
                wt.call(wt.ctx, pos + ssm_cp_offset, ssm_checkpoints.items);
            }
            // The next chunk's width, re-priced at this boundary: after the `mlx_clear_cache`
            // (steady state) and before the interleave tick (a co-tenant's decode is not this
            // prefill's pressure). A step-down commits here; a widen is re-priced after the tick.
            var pending_widen: u32 = 0;
            var widen_confirm: ?*const fn (*anyopaque, usize, u32) bool = null;
            var widen_ctx: ?*anyopaque = null;
            if (pos < loop_end) {
                if (options.chunk_width_hook) |hk| {
                    const next_w = hk.call(hk.ctx, ssm_cp_offset + pos, @intCast(cur_chunk), cap_adapt, &adapt_state);
                    if (next_w != 0 and next_w != cur_chunk) {
                        if (next_w < cur_chunk) {
                            commitAdaptiveWidth(&cur_chunk, &adapt_state, next_w);
                        } else {
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
            // The tick allocated for co-tenant slots; re-price the widen against what is left.
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
        try mlx.checkError();
        // One line per request when the width moved.
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
                ssm_checkpoints.append(allocator, cp) catch |e| {
                    var doomed = cp;
                    doomed.deinit(allocator);
                    return e;
                };
                // The head's own QSA leftover at this position: its ring is 32 rows and
                // the clamp back to this checkpoint comes a whole generated tail later.
                if (mtp_active) {
                    mtp_cache.?.activate();
                }
                if (mtp_active) if (xfm.qwen4Trunk()) |t| t.qwen4MtpMarkQsaLeftover(final_abs -| mtp_position_base) catch |e| {
                    log.debug("[qwen4] MTP head leftover mark failed: {s}\n", .{@errorName(e)});
                };
                if (options.ssm_checkpoint_max > 0 and
                    ssm_checkpoints.items.len > options.ssm_checkpoint_max)
                {
                    var dropped = ssm_checkpoints.orderedRemove(
                        transformer_mod.ssmCheckpointDropIndexLowAnchored(
                            ssm_checkpoints.items,
                            cp_thin,
                            null,
                            options.ssm_low_anchors,
                        ),
                    );
                    dropped.deinit(allocator);
                }
            }
            // One copy of the QSA key history on the latest snap. Stride
            // captures skipped it so a 400k prefill is not 32× the
            // indexer buffer. With the share switch on nothing is attached here: the newest
            // snap takes a view of the live buffer at commit (`handoffQsaHistoryToLatest`).
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
        if (has_vision and n_chunks > 1) log.debug("[vision] chunked prefill: {d} chunks, {d} placeholder rows consumed\n", .{ n_chunks, vision_rows_consumed });
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
    const need_capture = drafter_active or mtp_active or options.speech != null;
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
    errdefer if (dflash_active) {
        for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
    };
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
        for (dfl_out_buf) |*a| {
            _ = mlx.mlx_array_free(a.*);
            a.* = .{ .ctx = null };
        }
    }
    if (options.prompt_logprobs) |out| {
        errdefer _ = mlx.mlx_array_free(raw_logits);
        try promptLogprobs(allocator, raw_logits, prompt_ids[final_start + 1 ..], options.logprobs_n, xfm.s, out);
        if (out.items.len != prompt_ids.len - 1) return error.PromptLogprobsUnavailable;
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

    if (dspark_active) if (intree.dsv41(xfm)) |m41| {
        m41.ds_want_conf = true;
    };
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
            options.mtp.?.costProfile(xfm, ctx.cache.config)
        else
            .generic;
        const mtp_cap = @min(resolveMtpDepthCapForProfile(xfm.config.mtpDepth(options.mtp_depth), mtp_cost_profile), if (options.mtp) |h| h.maxDepth() else mtp_mod.MAX_DEPTH);
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
            .dspark_pol = if (dspark_active and intree.dsv41(xfm) != null) .{} else null,
            .drafter = if (drafter_active) options.drafter else null,
            .drafter_block_size = options.drafter_block_size,
            .dflash = if (dflash_active) options.dflash else null,
            .dflash_ctx = dflash_ctx,
            .dflash_block_size = dflash_bs,
            .dflash_pol = if (dflash_active and options.dflash_policy) .{} else null,
            .dflash_min_accepted_per_round = options.dflash_min_accepted_per_round,
            .mtp = if (mtp_active) options.mtp else null,
            .mtp_accept_graph = accept_route.graph,
            .mtp_accept_prefix = accept_route.prefix,
            .mtp_accept_param = accept_route.param,
            .mtp_greedy_tail = options.mtp_greedy_tail,
            .mtp_serial_accept = mtp_active and xfm.config.rowExactDecode() and std.meta.activeTag(options.mtp_acceptance) == .exact,
            .mtp_cache = mtp_cache,
            .mtp_position_base = mtp_position_base,
            .mtp_depth = mtp_cap,
            .mtp_depth_free = if (xfm.mtp_depth_free != 0) xfm.mtp_depth_free else mtpDepthCapFree(xfm.config.mtpDepth(options.mtp_depth)),
            .mtp_ev_costs = mtpEvCosts(mtp_cost_profile),
            // Start at depth 1 and climb with evidence: the cheap depth
            // is the safe default (1.11x on cold/creative content), and
            // hot workloads promote within ~8 rounds.
            .mtp_depth_current = depth_bounds.floorFor(depth_bounds.active, mtp_cap),
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
    if (options.speech) |req| {
        const b = xfm.plugin orelse return error.SpeechUnsupported;
        const hooks = b.table.speech orelse return error.SpeechUnsupported;
        const parsed = &xfm.config.plugin.?.parsed;
        const st = try allocator.create(speech_mod.State);
        errdefer allocator.destroy(st);
        st.* = try speech_mod.State.init(allocator, hooks, b.model, parsed.speech, parsed.placeholder_ids.audio, @intCast(mlx.getShape(logits)[mlx.getShape(logits).len - 1]), xfm.suppress_mask, prompt_ids[prompt_ids.len - 1], req, sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds()), s);
        var gen = Generator{
            .xfm = xfm,
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
            .last_hidden = captured_hidden,
            .has_last_hidden = true,
            .prng = std.Random.DefaultPrng.init(0),
            .prompt_ids_owned = prompt_owned,
            .prompt_ids_alloc = allocator,
            .speech = st,
        };
        gen.pending_logits = logits;
        gen.has_pending_logits = true;
        attachCp(&gen, &ssm_checkpoints, allocator);
        return gen;
    }
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
