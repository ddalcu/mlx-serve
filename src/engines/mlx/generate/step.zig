//! The serial decode step: next / reasoning / choice / constrained steps, stop checks, lazy forward.
const std = @import("std");
const env = @import("../../../env.zig");
const plugin_settle = @import("../plugin/settle.zig");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const log = @import("../../../sdk/util/log.zig");
const token_mask = @import("../../../sdk/grammar/token_mask.zig");
const rp_mod = @import("../../../sdk/grammar/protocol.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const Transformer = transformer_mod.Transformer;
const ForwardCtx = transformer_mod.ForwardCtx;
const Constraint = @import("../grammar/constraint.zig").Constraint;
const Generator = @import("generator.zig").Generator;
const applyGrammarMask = @import("../sampling/sampler.zig").applyGrammarMask;
const computeLogprobs = @import("../sampling/logprobs.zig").computeLogprobs;
const sampleToken = @import("../sampling/sampler.zig").sampleToken;
const tokenStops = @import("../../../sdk/request/stops.zig").tokenStops;


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
    // A penalty stays on the slow path: here the pending token is not in `generated_ids` yet.
    if (self.has_pending_logits and self.logprobs_n == 0 and !self.sampling.penalized() and self.step + 1 < self.max_tokens) {
        const step_logits = self.pending_logits;
        self.has_pending_logits = false;

        const lazy_token = self.sampleLazy(step_logits);
        _ = mlx.mlx_array_free(step_logits);

        var adopted = false;
        defer if (!adopted) {
            _ = mlx.mlx_array_free(lazy_token);
        };
        self.pushSsmLag();
        if (lazyForward(self.xfm, &self.ctx, lazy_token)) |next_logits| {
            defer if (!adopted) {
                _ = mlx.mlx_array_free(next_logits);
            };
            var sw = StepTrace.watch();
            const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            StepTrace.lap(&sw, .submit);

            // NOW resolve the pending token — GPU already computed it as a
            // dependency of the graph we just submitted. Should be instant.
            try self.resolvePendingToken();
            StepTrace.lap(&sw, .resolve);
            if (sw != null) StepTrace.steps += 1;

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
            adopted = true;

            return token;
        } else |_| {
            // lazyForward failed — fall through to slow path
            try mlx.check(mlx.mlx_array_eval(lazy_token));
            var val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
            _ = mlx.mlx_array_free(lazy_token);
            adopted = true;
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

/// Sample one token while the reasoning protocol has not reached the JSON
/// body: drain a forced recovery sequence, work the unresolved opener
/// choice under the union mask, or sample unconstrained reasoning while
/// the close-delimiter matcher watches. The active JSON phase reuses the
/// existing grammar mask builder.
pub fn nextReasoningStep(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint, proto: *const rp_mod.Protocol) !?u32 {
    // Recovery drain: at most one canonical transition token per tick,
    // through the same forward/accounting path as a sampled token.
    if (constraint.pstate.recovering) {
        return try self.commitForcedConstraintToken(allocator, constraint);
    }
    switch (constraint.pstate.phase) {
        .choice, .header => return self.nextChoiceStep(allocator, constraint, proto),
        .reasoning => {},
        .json_body => unreachable,
    }

    const step_logits = self.pending_logits;
    self.has_pending_logits = false;
    var owns_step_logits = true;
    defer {
        if (owns_step_logits) _ = mlx.mlx_array_free(step_logits);
    }

    // Boundary-crossing candidates (tokens carrying the close delimiter
    // plus a payload suffix) are validated BEFORE they can be sampled: a
    // suffix the schema rejects is masked out here, never sampled and
    // degraded afterwards.
    const invalid_crossers = try rp_mod.applyReasoningMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf);
    const lazy = blk: {
        if (invalid_crossers == 0) break :blk self.sampleLazy(step_logits);
        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, self.xfm.s);
        break :blk self.sampleLazy(masked_logits);
    };
    defer _ = mlx.mlx_array_free(lazy);
    try mlx.check(mlx.mlx_array_eval(lazy));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
    const token: u32 = @intCast(val);
    self.next_token_id = token;

    for (self.eos_token_ids) |eos_id| {
        if (token == eos_id and (proto.closer_atomic == null or token != proto.closer_atomic.?)) {
            _ = mlx.mlx_array_free(step_logits);
            owns_step_logits = false;
            switch (try self.forceConstraintTransition(allocator)) {
                .committed => |forced| {
                    log.info("[grammar] reasoning boundary forced after premature EOS\n", .{});
                    return forced;
                },
                .activated => unreachable, // the reasoning phase always needs its close delimiter
                .refused => {
                    self.done = true;
                    self.finish_reason = "stop";
                    return null;
                },
            }
        }
    }
    if (token == 0) {
        self.consecutive_pad += 1;
        if (self.consecutive_pad >= 3) {
            _ = mlx.mlx_array_free(step_logits);
            owns_step_logits = false;
            switch (try self.forceConstraintTransition(allocator)) {
                .committed => |forced| {
                    log.info("[grammar] reasoning boundary forced after terminal padding\n", .{});
                    return forced;
                },
                .activated => unreachable,
                .refused => {
                    self.done = true;
                    self.finish_reason = "stop";
                    return null;
                },
            }
        }
    } else {
        self.consecutive_pad = 0;
    }

    const token_bytes = if (token < constraint.token_bytes.bytes.len) constraint.token_bytes.bytes[token] else null;
    const payload_suffix = rp_mod.observeProtocolToken(proto, &constraint.pstate, token, token_bytes);
    if (constraint.pstate.invalid) return error.InvalidProtocolTransition;
    if (payload_suffix) |suffix| {
        // The close delimiter completed inside this token; any bytes after
        // it are the first payload bytes. The pre-sample mask validated
        // every crossing candidate's suffix. A mismatch is an internal
        // error and must never disable the constraint.
        for (suffix) |b| {
            const ok = try constraint.grammar.acceptByte(b);
            if (!ok) {
                return error.InvalidProtocolPayload;
            }
        }
        constraint.pstate.phase = .json_body;
        // A byte-spelled close shares this token: the payload begins at
        // byte_offset inside it. An atomic close has no bytes at all —
        // the payload begins with the NEXT token (never unwrap the
        // null token bytes).
        constraint.pending_span = if (token_bytes == null)
            .{ .token_index = @intCast(self.generated_ids.items.len + 1), .byte_offset = 0 }
        else
            .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = rp_mod.payloadByteOffset(token_bytes, suffix) };
        self.loop_guard_start = self.generated_ids.items.len;
        log.info("[grammar] reasoning boundary reached; enforcing final-answer schema\n", .{});
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

/// One constrained sample at the unresolved opener choice. The mask is
/// the union of schema-legal tokens (the direct answer is constrained
/// from its first byte) and opener-prefix candidates; once opener bytes
/// are in flight only opener continuations remain. The JSON grammar never
/// consumes protocol bytes.
pub fn nextChoiceStep(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint, proto: *const rp_mod.Protocol) !?u32 {
    const s = self.xfm.s;

    const allowed = if (constraint.pstate.phase == .header)
        try rp_mod.applyHeaderMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf)
    else
        try rp_mod.applyChoiceMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf);
    if (allowed == 0) {
        return error.NoValidProtocolToken;
    }

    const step_logits = self.pending_logits;
    self.has_pending_logits = false;
    defer _ = mlx.mlx_array_free(step_logits);

    var masked_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked_logits);
    try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

    const lazy = self.sampleLazy(masked_logits);
    defer _ = mlx.mlx_array_free(lazy);
    try mlx.check(mlx.mlx_array_eval(lazy));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
    const token: u32 = @intCast(val);
    self.next_token_id = token;

    const token_bytes = if (token < constraint.token_bytes.bytes.len) constraint.token_bytes.bytes[token] else null;

    // The choice mask never admits EOS (the grammar is incomplete) and
    // pad tokens carry no admissible bytes; stop the way the constrained
    // path does if one is sampled anyway.
    for (self.eos_token_ids) |eos_id| {
        if (token == eos_id and constraint.pstate.phase != .header) {
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

    const outcome: rp_mod.ChoiceOutcome = if (constraint.pstate.phase == .header) blk: {
        const suffix = rp_mod.observeProtocolToken(proto, &constraint.pstate, token, token_bytes);
        if (constraint.pstate.invalid) return error.InvalidProtocolTransition;
        if (suffix) |bytes| break :blk .{ .payload = bytes };
        break :blk if (constraint.pstate.phase == .reasoning) .opened else .progress;
    } else rp_mod.observeChoice(proto, &constraint.pstate, token, token_bytes);
    switch (outcome) {
        .progress => {},
        .opened => {
            self.loop_guard_start = self.generated_ids.items.len;
            log.info("[grammar] reasoning opener generated; reasoning unconstrained\n", .{});
        },
        .json, .payload => {
            const payload = if (outcome == .payload) outcome.payload else token_bytes;
            if (payload) |bytes| {
                for (bytes) |b| {
                    const ok = try constraint.grammar.acceptByte(b);
                    if (!ok) {
                        return error.InvalidProtocolPayload;
                    }
                }
            }
            constraint.pstate.phase = .json_body;
            constraint.pending_span = .{
                .token_index = @intCast(self.generated_ids.items.len + @as(usize, if (token_bytes == null) 1 else 0)),
                .byte_offset = if (outcome == .payload) rp_mod.payloadByteOffset(token_bytes, outcome.payload) else 0,
            };
            self.loop_guard_start = self.generated_ids.items.len;
        },
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

/// Synchronous, grammar-constrained sampling step. Used whenever
/// `sampling.constraint` is non-null. Builds a token mask from the grammar's
/// current state, applies it to the pending logits, samples, advances the
/// grammar by the sampled token's bytes, and pre-launches the next forward
/// pass to overlap with the next mask build.
pub fn nextConstrained(self: *Generator, allocator: std.mem.Allocator) !?u32 {
    if (self.logprobs_n == 0 or !self.has_pending_logits) return self.nextConstrainedToken(allocator);
    // Every constrained arm returns a token drawn (or forced) at THIS position, with no
    // one-token delay, so its entry comes from these logits: raw, before the grammar mask.
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_set(&logits, self.pending_logits));
    const token = (try self.nextConstrainedToken(allocator)) orelse return null;
    if (self.last_logprob) |*lp| allocator.free(lp.top_logprobs);
    self.pending_logprob = try computeLogprobs(allocator, logits, token, self.logprobs_n, self.xfm.s);
    self.last_logprob = self.pending_logprob;
    self.pending_logprob = null;
    return token;
}

pub fn nextConstrainedToken(self: *Generator, allocator: std.mem.Allocator) !?u32 {
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
    if (constraint.proto) |proto| {
        if (constraint.pstate.phase != .json_body) return self.nextReasoningStep(allocator, constraint, proto);
    }
    const s = self.xfm.s;

    var allowed = (try token_mask.buildMask(constraint.grammar, constraint.token_bytes, constraint.mask_buf)).allowed;
    // The tokenizer's EOS is not necessarily the model's only stop id.
    if (constraint.grammar.isComplete()) {
        for (self.eos_token_ids) |eos_id| {
            if (eos_id < constraint.mask_buf.len and !constraint.mask_buf[eos_id]) {
                constraint.mask_buf[eos_id] = true;
                allowed += 1;
            }
        }
    }
    if (allowed == 0) {
        log.warn("[grammar] no token satisfies the schema at this position — disabling further mask enforcement\n", .{});
        constraint.grammar.dead = true;
        @memset(constraint.mask_buf, true);
    }

    const step_logits = self.pending_logits;
    self.has_pending_logits = false;
    defer _ = mlx.mlx_array_free(step_logits);

    var masked_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked_logits);
    try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

    // The next forward is built on the still-lazy sample so the CPU graph
    // build overlaps this step's GPU work; the token is realized after
    // dispatch and only then advances the grammar (the mask for the next
    // logits is built on the next call, off the realized state).
    const lazy = self.sampleLazy(masked_logits);
    defer _ = mlx.mlx_array_free(lazy);
    var next_logits: ?mlx.mlx_array = null;
    if (self.step + 1 < self.max_tokens) {
        if (lazyForward(self.xfm, &self.ctx, lazy)) |nl| {
            const arr = [_]mlx.mlx_array{ lazy, nl };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            next_logits = nl;
        } else |_| {}
    }
    errdefer if (next_logits) |nl| {
        _ = mlx.mlx_array_free(nl);
    };
    try mlx.check(mlx.mlx_array_eval(lazy));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
    const token: u32 = @intCast(val);
    self.next_token_id = token;

    // Stop on EOS — do not advance grammar or include in output. A
    // dispatched forward on the stop token is dropped, as on the lazy path.
    for (self.eos_token_ids) |eos_id| {
        if (token == eos_id) {
            if (next_logits) |nl| _ = mlx.mlx_array_free(nl);
            self.done = true;
            self.finish_reason = "stop";
            return null;
        }
    }
    if (token == 0) {
        self.consecutive_pad += 1;
        if (self.consecutive_pad >= 3) {
            if (next_logits) |nl| _ = mlx.mlx_array_free(nl);
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

    if (next_logits) |nl| {
        self.pending_logits = nl;
        self.has_pending_logits = true;
    } else if (self.step < self.max_tokens) {
        // Pipeline miss: forward the realized token.
        const tok_i32: i32 = @intCast(token);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);
        const nl = try self.xfm.forwardWith(&self.ctx, tok_input);
        const arr = [_]mlx.mlx_array{nl};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);
        self.pending_logits = nl;
        self.has_pending_logits = true;
    } else {
        self.done = true;
        self.finish_reason = "length";
    }

    return token;
}

/// Check all stop conditions. Returns true if generation should stop.
pub fn checkStop(self: *Generator) !bool {
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
    if (tokenStops(self.next_token_id, self.eos_token_ids, &self.consecutive_pad)) {
        self.done = true;
        self.finish_reason = "stop";
        return true;
    }
    return false;
}

/// Legacy sync forward for logprobs path.
pub fn startAsyncForward(self: *Generator, token_id: u32) void {
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

/// Diagnostic (diag `step-trace`): host time of a pipelined serial step, per phase,
/// summed across the request and reported with the spec stats.
pub const StepTrace = struct {
    const Phase = enum { build, flush, submit, resolve };
    var on: ?bool = null;
    var steps: u64 = 0;
    var ns: [4]u64 = .{ 0, 0, 0, 0 };

    fn enabled() bool {
        if (on) |v| return v;
        const raw = env.get(.STEP_TRACE);
        const v = raw != null and raw.?[0] != '0';
        on = v;
        return v;
    }
    fn watch() ?io_util.Stopwatch {
        return if (enabled()) io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io()) else null;
    }
    fn lap(sw: *?io_util.Stopwatch, phase: Phase) void {
        if (sw.*) |*w| {
            ns[@backingInt(phase)] += w.read();
            w.reset();
        }
    }
    pub fn report() void {
        if (!enabled() or steps == 0) return;
        const n: f64 = @floatFromInt(steps);
        log.info("  [step-trace] steps={d} build={d:.3} flush={d:.3} submit={d:.3} resolve={d:.3} ms/step\n", .{
            steps,
            @as(f64, @floatFromInt(ns[0])) / n / 1e6,
            @as(f64, @floatFromInt(ns[1])) / n / 1e6,
            @as(f64, @floatFromInt(ns[2])) / n / 1e6,
            @as(f64, @floatFromInt(ns[3])) / n / 1e6,
        });
        steps = 0;
        ns = .{ 0, 0, 0, 0 };
    }
};

/// Build forward pass from a lazy sampled token array.
/// Reshapes [1] -> [1, 1] and calls transformer forward. All lazy (no eval).
pub fn lazyForward(xfm: *Transformer, ctx: *ForwardCtx, lazy_token: mlx.mlx_array) !mlx.mlx_array {
    const tok_shape = [_]c_int{ 1, 1 };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, lazy_token, &tok_shape, 2, xfm.s));
    // Every array of an async batch carries that batch's END event, so the GPU PLE arm sends
    // the sampled token out as its own batch: its deferred history settle then waits on the
    // sampler, not on the forward built on it.
    if (plugin_settle.defersHost(xfm)) {
        const tv = mlx.mlx_vector_array_new_value(reshaped);
        defer _ = mlx.mlx_vector_array_free(tv);
        try mlx.check(mlx.mlx_async_eval(tv));
    }
    // The graph is built while the previous step still runs on the GPU; an
    // arch with a host-side token lookup (qwen4 n-gram PLE) fills its leaf
    // afterwards, so the one sync on the token comes AFTER the build.
    ctx.ple_defer = true;
    defer ctx.ple_defer = false;
    var sw = StepTrace.watch();
    const logits = xfm.forwardWith(ctx, reshaped) catch |e| {
        xfm.discardDeferredPle(ctx);
        return e;
    };
    StepTrace.lap(&sw, .build);
    // The GPU arm owes only the host history, and its ids are the token this very forward
    // waits on; settling now would idle the GPU for the round trip. The next PLE embedding
    // (or the slot's drain / finish) settles it, when those ids are long evaluated.
    if (!Transformer.deferredPleOnGpu(ctx)) xfm.flushDeferredPle(ctx) catch |e| {
        xfm.discardDeferredPle(ctx);
        _ = mlx.mlx_array_free(logits);
        return e;
    };
    StepTrace.lap(&sw, .flush);
    return logits;
}
