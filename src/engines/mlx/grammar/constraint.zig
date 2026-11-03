//! Grammar constraints on generation: schema state, forced transitions and their commit.
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const plugin_driver = @import("../plugin/driver.zig");
const log = @import("../../../sdk/util/log.zig");
const json_grammar = @import("../../../sdk/grammar/json_grammar.zig");
const json_schema = @import("../../../sdk/grammar/json_schema.zig");
const token_mask = @import("../../../sdk/grammar/token_mask.zig");
const rp_mod = @import("../../../sdk/grammar/protocol.zig");
const testing = std.testing;
const Generator = @import("../generate/generator.zig").Generator;
const generate = @import("../generate/oneshot.zig").generate;

pub const Constraint = @import("../../../sdk/grammar/constraint.zig").Constraint;
pub const SchemaConstraint = @import("../../../sdk/grammar/constraint.zig").SchemaConstraint;
pub const forcedBoundaryCanContinue = @import("../../../sdk/grammar/constraint.zig").forcedBoundaryCanContinue;

pub fn hasPrePayloadConstraint(self: *const Generator) bool {
    const constraint = self.sampling.constraint orelse return false;
    return constraint.proto != null and constraint.pstate.phase != .json_body;
}

/// The pending reasoning→payload boundary, consumed by the scheduler when
/// it publishes the token that carries it (same delivery path as the
/// token itself, so a streaming reader cannot see one without the other).
pub fn takeConstraintSpan(self: *Generator) ?rp_mod.ConstraintSpan {
    const span = self.sampling.constraint orelse return null;
    const s = span.pending_span orelse return null;
    span.pending_span = null;
    return s;
}

pub const ConstraintRecovery = union(enum) {
    /// One forced token was committed through the live forward path; the
    /// caller publishes it like a sampled token.
    committed: u32,
    /// The choice state recovers by answering directly: nothing to
    /// commit, the next sample is JSON-masked.
    activated,
    /// The remaining transition does not fit the completion allowance.
    refused,
};

/// Force the reasoning→constrained transition when the protocol has not
/// reached the JSON body: a reasoning loop, premature EOS, or terminal
/// padding recovers into a valid final answer when the remaining
/// completion allowance fits the WHOLE remaining transition plus one
/// answer token. Commits at most one forced token per call (the next
/// tick drains the rest), so cancellation, stop sequences, inference
/// errors, and stalled forwards keep their usual precedence.
pub fn forceConstraintTransition(self: *Generator, allocator: std.mem.Allocator) !ConstraintRecovery {
    const constraint = self.sampling.constraint orelse return .refused;
    const proto = constraint.proto orelse return .refused;
    std.debug.assert(!self.has_pending_token);

    switch (constraint.pstate.phase) {
        .json_body => return .refused,
        .choice => {
            if (constraint.pstate.open_cursor == 0) {
                // The model never committed a channel and emitted no
                // opener bytes: a direct constrained answer needs no
                // transition token at all. The pending logits stay —
                // they are the distribution the next JSON-masked sample
                // reads.
                constraint.pstate.phase = .json_body;
                constraint.pstate.recovering = false;
                constraint.pending_span = .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = 0 };
                self.loop_guard_start = self.generated_ids.items.len;
                log.info("[grammar] reasoning loop at the channel choice; enforcing JSON schema immediately\n", .{});
                return .activated;
            }
            // Opener bytes are already in the output: starting the JSON
            // here would strand them (the response split reads them as an
            // unclosed opener and files the answer as reasoning).
            // Complete the opener, close it, then constrain. Fall through
            // to the shared planner below.
            log.info("[grammar] completing the generated reasoning opener before the boundary\n", .{});
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
            if (!constraint.pstate.recovering) {
                var planned: [rp_mod.MAX_TRANSITION_TOKENS]u32 = undefined;
                const planned_len = proto.planRecovery(&constraint.pstate, &planned) orelse return .refused;
                if (!forcedBoundaryCanContinue(self.completion_tokens, self.max_tokens, planned_len)) return .refused;
                constraint.pstate.recovering = true;
                constraint.pstate.forced_cursor = 0;
                constraint.pstate.pending_len = @intCast(planned_len);
                @memcpy(constraint.pstate.pending[0..planned_len], planned[0..planned_len]);
            }
            return .{ .committed = try self.commitForcedConstraintToken(allocator, constraint) };
        },
        .reasoning, .header => {
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
            if (!constraint.pstate.recovering) {
                var planned: [rp_mod.MAX_TRANSITION_TOKENS]u32 = undefined;
                const planned_len = proto.planRecovery(&constraint.pstate, &planned) orelse return .refused;
                if (!forcedBoundaryCanContinue(self.completion_tokens, self.max_tokens, planned_len)) return .refused;
                constraint.pstate.recovering = true;
                constraint.pstate.forced_cursor = 0;
                constraint.pstate.pending_len = @intCast(planned_len);
                @memcpy(constraint.pstate.pending[0..planned_len], planned[0..planned_len]);
            }
            return .{ .committed = try self.commitForcedConstraintToken(allocator, constraint) };
        },
    }
}

/// Commit ONE canonical transition token through the live model state,
/// then leave its logits pending for the next sample. Called once per
/// scheduler tick while `pstate.recovering` drains.
pub fn commitForcedConstraintToken(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint) !u32 {
    if (self.has_pending_logits) {
        _ = mlx.mlx_array_free(self.pending_logits);
        self.has_pending_logits = false;
    }
    const seq = constraint.pstate.pending[0..constraint.pstate.pending_len];
    const tok = seq[constraint.pstate.forced_cursor];
    constraint.pstate.forced_cursor += 1;

    try self.generated_ids.append(allocator, tok);
    self.loop_guard_start = self.generated_ids.items.len;
    self.next_token_id = tok;
    self.advanceStep(1);
    self.consecutive_pad = 0;

    const tok_i32: i32 = @intCast(tok);
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

    if (constraint.pstate.forced_cursor >= seq.len) {
        // The transition drained: the next sample is the first constrained
        // payload token. Reset the repetition window at the final-body
        // transition so the drained sequence cannot re-trigger recovery.
        constraint.pstate.recovering = false;
        constraint.pstate.phase = .json_body;
        constraint.pending_span = .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = 0 };
        self.loop_guard_start = self.generated_ids.items.len;
    }
    return tok;
}

// ── tests ──


test "forced recovery leaves room for the whole transition and one answer token" {
    try std.testing.expect(forcedBoundaryCanContinue(0, 2, 1));
    try std.testing.expect(forcedBoundaryCanContinue(8, 10, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 1, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(9, 10, 1));
    // A multi-token transition plans against the WHOLE sequence.
    try std.testing.expect(forcedBoundaryCanContinue(0, 4, 3));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 3, 3));
    // An in-place activation needs no transition token.
    try std.testing.expect(forcedBoundaryCanContinue(0, 1, 0));
}

test "constrained generation returns one logprob entry per token, paired with that token" {
    // Bar: a json_schema request with logprobs gets an entry for every returned token, describing it.
    const path = env.get(.LOGPROBS_TEST_MODEL) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(path);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    var xfm = try plugin_driver.load(io, a, &config, dir, mlx.gpuStream());
    defer xfm.deinit();
    var tok = try tokenizer_mod.loadTokenizer(io, a, dir);
    defer tok.deinit();
    var token_bytes = try token_mask.build(a, &tok);
    defer token_bytes.deinit();

    const schema_src =
        \\{"type":"object","properties":{"a":{"type":"string","enum":["yes","no"]}},"required":["a"],"additionalProperties":false}
    ;
    var schema = try std.json.parseFromSlice(std.json.Value, a, schema_src, .{});
    defer schema.deinit();
    var sc: SchemaConstraint = undefined;
    try sc.initFromValue(a, schema.value, &token_bytes);
    defer sc.deinit();

    const prompt = try tok.encode(a, "Answer yes or no in JSON: is water wet?\n");
    defer a.free(prompt);
    const result = try generate(io, a, &xfm, &tok, prompt, 16, .{ .temperature = 0, .constraint = &sc.constraint }, config.eosTokenSlice(), 0, 5);
    defer {
        a.free(result.text);
        a.free(result.token_ids);
        if (result.logprobs) |lps| {
            for (lps) |lp| a.free(lp.top_logprobs);
            a.free(lps);
        }
    }

    const lps = result.logprobs orelse return error.NoLogprobs;
    try testing.expect(result.token_ids.len > 0);
    try testing.expectEqual(result.token_ids.len, lps.len);
    var paired: usize = 0;
    for (result.token_ids, lps) |id, lp| {
        try testing.expect(lp.token_logprob <= 0);
        for (lp.top_logprobs) |alt| {
            if (alt.token_id != id) continue;
            try testing.expectApproxEqAbs(alt.logprob, lp.token_logprob, 1e-5);
            paired += 1;
        }
    }
    try testing.expect(paired > 0);
}
