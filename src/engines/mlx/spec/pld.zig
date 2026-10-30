//! Prompt-lookup decoding (PLD) rounds.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const log = @import("../../../sdk/util/log.zig");
const pld_index = @import("../../../sdk/spec/pld_index.zig");
const SSMCacheEntrySnapshot = transformer_mod.SSMCacheEntrySnapshot;
const ssmSnapshot = transformer_mod.ssmSnapshot;
const ssmSnapshotDeinit = transformer_mod.ssmSnapshotDeinit;
const ssmRestore = transformer_mod.ssmRestore;
const Generator = @import("../generate/generator.zig").Generator;
const RUNTIME_GATE_MIN_PER_DRAFT_RATE = @import("gates.zig").RUNTIME_GATE_MIN_PER_DRAFT_RATE;
const SPEC_REENABLE_INTERVAL = @import("gates.zig").SPEC_REENABLE_INTERVAL;
const YIELD_GATE_MIN_YIELD = @import("gates.zig").YIELD_GATE_MIN_YIELD;
const capAcceptedForTokenBudget = @import("verify.zig").capAcceptedForTokenBudget;
const pldOneHotRow = @import("../sampling/spec_sampling.zig").pldOneHotRow;
const probAt = @import("../sampling/spec_sampling.zig").probAt;
const probsAtLastPos = @import("../sampling/spec_sampling.zig").probsAtLastPos;
const runtimeGateShouldDisable = @import("gates.zig").runtimeGateShouldDisable;
const sampleFromProbs = @import("../sampling/spec_sampling.zig").sampleFromProbs;
const sampleResidual = @import("../sampling/spec_sampling.zig").sampleResidual;
const specDecodeUnsupported = @import("verify.zig").specDecodeUnsupported;
const specShouldReenable = @import("gates.zig").specShouldReenable;
const yieldGateShouldDisable = @import("gates.zig").yieldGateShouldDisable;


/// Selects the source slice that `initWithOptions` will dupe into
/// `prompt_ids_owned`. When `lookup_prompt` is non-null it wins (server
/// cache-reuse path: full original prompt for PLD lookup); otherwise the
/// caller's `prompt_ids` is used (back-compat path).
pub fn pickLookupPromptSource(prompt_ids: []const u32, lookup_prompt: ?[]const u32) []const u32 {
    return lookup_prompt orelse prompt_ids;
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
    // the KV snapshot rollback cannot restore (the mangled-DSML
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

    if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting

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
    var verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
    errdefer _ = mlx.mlx_array_free(verify_logits);
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
    var per_pos_logits_n: usize = 0;
    defer {
        for (per_pos_logits[0..per_pos_logits_n]) |arr| _ = mlx.mlx_array_free(arr);
        allocator.free(per_pos_logits);
    }
    for (per_pos_logits, 0..) |*slot, idx| {
        slot.* = mlx.mlx_array_new();
        per_pos_logits_n = idx + 1;
        const start = [_]c_int{ 0, @intCast(idx), 0 };
        const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
        try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
    }
    _ = mlx.mlx_array_free(verify_logits);
    verify_logits = .{ .ctx = null };

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
                break :blk try sampleResidual(probs, onehot, self.correctionDraw(accepted), s);
            } else {
                break :blk try sampleFromProbs(probs, self.correctionDraw(accepted), s);
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
            try self.rollbackSsmFromCapture(self.ctx.ssm_entries.?, accepted, 1 + m, s);
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
