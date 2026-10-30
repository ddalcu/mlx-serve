//! Gemma assistant drafter rounds and the correction draw.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const drafter_mod = @import("../../../plugins/gemma4/assistant_drafter.zig");
const SSMCacheEntrySnapshot = transformer_mod.SSMCacheEntrySnapshot;
const ssmSnapshot = transformer_mod.ssmSnapshot;
const ssmSnapshotDeinit = transformer_mod.ssmSnapshotDeinit;
const ssmRestore = transformer_mod.ssmRestore;
const CommittedArgmax = @import("verify.zig").CommittedArgmax;
const Generator = @import("../generate/generator.zig").Generator;
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const capAcceptedForTokenBudget = @import("verify.zig").capAcceptedForTokenBudget;
const pldOneHotRow = @import("../sampling/spec_sampling.zig").pldOneHotRow;
const probAt = @import("../sampling/spec_sampling.zig").probAt;
const probsAtLastPos = @import("../sampling/spec_sampling.zig").probsAtLastPos;
const sampleFromProbs = @import("../sampling/spec_sampling.zig").sampleFromProbs;
const sampleResidual = @import("../sampling/spec_sampling.zig").sampleResidual;
const specDecodeUnsupported = @import("verify.zig").specDecodeUnsupported;
const verifyArgmax = @import("verify.zig").verifyArgmax;


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
/// The sampling for a verify round's correction token (the bonus or the residual
/// draw after `accepted` drafts): keyed at that token's output position, so a
/// seeded request replays under every drafter and no two positions share a key.
pub fn correctionDraw(self: *const Generator, accepted: u32) SamplingParams {
    var sp = self.sampling;
    sp.draw = self.generated_ids.items.len + accepted + 2;
    return sp;
}

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

    if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting

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
    var draft_arrs_n: usize = 0;
    defer {
        for (draft_arrs[0..draft_arrs_n]) |arr| _ = mlx.mlx_array_free(arr);
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
            draft_arrs_n = i + 1;
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
        var drafts_2d_n: usize = 0;
        defer {
            for (drafts_2d[0..drafts_2d_n]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(drafts_2d);
        }
        for (draft_arrs, drafts_2d) |dlazy, *out| {
            out.* = mlx.mlx_array_new();
            drafts_2d_n += 1;
            try mlx.check(mlx.mlx_reshape(out, dlazy, &reshape_2d, 2, s));
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, t1_2d);
        for (drafts_2d) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
    }

    var new_hidden = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(new_hidden);
    // Captures the post-final-norm hidden at the LAST input position
    // (= position m, predicting the bonus token if all drafts accept).
    var verify_logits = try xfm.forwardWithCapture(&self.ctx, verify_input, &new_hidden);
    errdefer _ = mlx.mlx_array_free(verify_logits);
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
    var per_pos_logits_n: usize = 0;
    defer if (per_pos_logits) |slots| {
        for (slots[0..per_pos_logits_n]) |arr| _ = mlx.mlx_array_free(arr);
        allocator.free(slots);
    };
    if (stochastic) {
        const slots = try allocator.alloc(mlx.mlx_array, 1 + m);
        per_pos_logits = slots;
        const slice_strides = [_]c_int{ 1, 1, 1 };
        for (slots, 0..) |*slot, idx| {
            slot.* = mlx.mlx_array_new();
            per_pos_logits_n = idx + 1;
            const start = [_]c_int{ 0, @intCast(idx), 0 };
            const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
            try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
        }
    }

    // Build the greedy argmax tensor lazily; it'll be eval'd alongside
    // the rest of the round below. Its entries are COMMITTED, so they mask
    // reserved ids like every other block decoder's.
    var verify_argmax: CommittedArgmax = .{};
    defer verify_argmax.deinit();
    if (!stochastic) {
        verify_argmax = try verifyArgmax(verify_logits, self.sampling.suppress_mask, s);
    }
    _ = mlx.mlx_array_free(verify_logits);
    verify_logits = .{ .ctx = null };

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
            _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax.lazy());
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
        try mlx.check(mlx.mlx_array_eval(verify_argmax.lazy()));
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
        const argmax_data = try verify_argmax.ids(1 + m);
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
                break :blk try sampleResidual(probs, onehot, self.correctionDraw(accepted), s);
            } else {
                break :blk try sampleFromProbs(probs, self.correctionDraw(accepted), s);
            }
        } else {
            // Greedy: reuse the bulk-read argmax row. Already eval'd in
            // the single async eval above; no GPU sync here.
            const argmax_data = try verify_argmax.ids(1 + m);
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
        new_hidden = .{ .ctx = null };

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
    new_hidden = .{ .ctx = null };

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
    errdefer _ = mlx.mlx_array_free(re_new_hidden);
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
    re_new_hidden = .{ .ctx = null };

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
