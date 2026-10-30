//! The pipelined serial step: SSM lag, pending-token resolution, drains, forced-token commits.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const log = @import("../../../sdk/util/log.zig");
const SSMCacheEntrySnapshot = transformer_mod.SSMCacheEntrySnapshot;
const ssmSnapshot = transformer_mod.ssmSnapshot;
const ssmSnapshotDeinit = transformer_mod.ssmSnapshotDeinit;
const ssmRestore = transformer_mod.ssmRestore;
const Generator = @import("generator.zig").Generator;
const applyPenalties = @import("../sampling/filters.zig").applyPenalties;
const sampleTokenLazy = @import("../sampling/lazy.zig").sampleTokenLazy;


/// One pipelined step's recurrent state: the position it describes and a refcount snapshot per layer.
pub const SsmLag = struct {
    pos: usize = 0,
    snaps: []SSMCacheEntrySnapshot = &.{},

    pub fn deinit(self: *SsmLag, allocator: std.mem.Allocator) void {
        for (self.snaps) |*s| ssmSnapshotDeinit(s);
        allocator.free(self.snaps);
        self.* = .{};
    }
};

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
    // The batched tick hashes against this slot's history: settle the pipelined step's owed advance first.
    try self.xfm.flushDeferredPle(&self.ctx);
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
    if (self.sampling.penalized() and self.generated_ids.items.len > 0) {
        var pen = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen);
        if (applyPenalties(&pen, logits, self.generated_ids.items, self.sampling, self.xfm.s)) |_| {
            return sampleTokenLazy(pen, self.sampling, self.xfm.s);
        } else |err| log.warn("[sampling] repeat penalty failed ({s}); this draw is unpenalized\n", .{@errorName(err)});
    }
    return sampleTokenLazy(logits, self.sampling, self.xfm.s);
}

/// The token the next publish emits; a pipelined one is resolved now (one sync).
pub fn upcomingToken(self: *Generator) !u32 {
    try self.resolvePendingToken();
    return self.next_token_id;
}

pub fn resolvePendingToken(self: *Generator) !void {
    if (!self.has_pending_token) return;
    try mlx.check(mlx.mlx_array_eval(self.pending_token));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, self.pending_token));
    _ = mlx.mlx_array_free(self.pending_token);
    self.has_pending_token = false;
    self.next_token_id = @intCast(val);
}

pub const DrainResult = union(enum) {
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
pub fn drainPipelineForSpec(self: *Generator, allocator: std.mem.Allocator) !DrainResult {
    // The spec rounds roll the n-gram history back and forth: settle the pipelined step's owed advance first.
    try self.xfm.flushDeferredPle(&self.ctx);
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
    // With the latch these checks return instead of ending the process, so the handle needs an owner.
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

pub const ForcedCommit = struct {
    /// Tokens emitted by this call: any pending token plus every forced
    /// token but the last, which stays as `next_token_id` for the next
    /// regular tick to publish.
    emitted: []const u32,
    stopped: bool,
    /// The forward's DFlash taps joined the assistant context: DFlash may keep drafting.
    dflash_synced: bool = false,
};

/// Commit `forced` through the model as ONE multi-token forward, from any
/// of the three inter-tick states (pipelined, spec exit, shim-seeded), and
/// leave the generator in the shim-seeded state a regular `next()` reads.
/// A bound DFlash drafter's taps for the span are parked (`ForcedCommit.dflash_synced`).
pub fn commitForcedTokens(self: *Generator, allocator: std.mem.Allocator, forced: []const u32) !ForcedCommit {
    std.debug.assert(forced.len > 0);
    var emitted = std.ArrayList(u32).empty;
    errdefer emitted.deinit(allocator);
    var input = std.ArrayList(u32).empty;
    defer input.deinit(allocator);

    if (self.has_pending_token or self.has_pending_logits) {
        // Pipelined or shim-seeded: `next_token_id` is in the cache, unpublished.
        try self.resolvePendingToken();
        if (try self.checkStop()) return .{ .emitted = try emitted.toOwnedSlice(allocator), .stopped = true };
        try self.generated_ids.append(allocator, self.next_token_id);
        try emitted.append(allocator, self.next_token_id);
        self.advanceStep(1);
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
    } else {
        // Spec exit: `next_token_id` is decided but not in the cache.
        try input.append(allocator, self.next_token_id);
        try self.generated_ids.append(allocator, self.next_token_id);
        try emitted.append(allocator, self.next_token_id);
        self.advanceStep(1);
    }
    try input.appendSlice(allocator, forced);
    for (forced[0 .. forced.len - 1]) |f| {
        try self.generated_ids.append(allocator, f);
        try emitted.append(allocator, f);
    }
    self.advanceStep(@intCast(forced.len - 1));
    self.consecutive_pad = 0;
    self.loop_guard_start = self.generated_ids.items.len;

    const ids_i32 = try allocator.alloc(i32, input.items.len);
    defer allocator.free(ids_i32);
    for (input.items, 0..) |t, i| ids_i32[i] = @intCast(t);
    const shape = [_]c_int{ 1, @intCast(input.items.len) };
    const tok_input = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tok_input);
    const dflash_live = self.dflash != null and self.dflash_ctx != null and !self.spec_disabled_runtime;
    const caps = try allocator.alloc(mlx.mlx_array, if (dflash_live) self.dflash.?.config.target_layer_ids.len else 0);
    defer {
        for (caps) |a| _ = mlx.mlx_array_free(a);
        allocator.free(caps);
    }
    for (caps) |*a| a.* = mlx.mlx_array_new();
    var cl = transformer_mod.CaptureLayers{ .ids = if (dflash_live) self.dflash.?.config.captureIds() else &.{}, .out = caps };
    if (dflash_live) {
        self.ctx.capture_layers = &cl;
        self.ctx.capture_ssm_seq = false; // a forced span never rolls back
    }
    const forward = self.xfm.forwardWith(&self.ctx, tok_input);
    self.ctx.capture_layers = null;
    const all_logits = try forward;
    defer _ = mlx.mlx_array_free(all_logits);
    const shp = mlx.getShape(all_logits);
    var logits = mlx.mlx_array_new();
    const start = [_]c_int{ 0, shp[1] - 1, 0 };
    const stop = [_]c_int{ shp[0], shp[1], shp[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&logits, all_logits, &start, 3, &stop, 3, &strides, 3, self.xfm.s));
    const arr = [_]mlx.mlx_array{logits};
    const vec = mlx.mlx_vector_array_new_data(&arr, 1);
    for (caps) |c| _ = mlx.mlx_vector_array_append_value(vec, c);
    _ = mlx.mlx_async_eval(vec);
    _ = mlx.mlx_vector_array_free(vec);
    self.pending_logits = logits;
    self.has_pending_logits = true;
    self.next_token_id = forced[forced.len - 1];
    self.mtp_hidden_stale = true;
    if (dflash_live) try self.dflashPark(allocator, caps, @intCast(input.items.len));
    return .{ .emitted = try emitted.toOwnedSlice(allocator), .stopped = false, .dflash_synced = dflash_live };
}

/// The ONE place a partial accept rolls recurrent state back from the verify capture:
/// the loop and the `partial_rounds` count travel together, so a caller cannot take the
/// rollback without being counted.
pub fn rollbackSsmFromCapture(self: *Generator, entries: []transformer_mod.SSMCacheEntry, accepted: u32, verify_len: u32, s: mlx.mlx_stream) !void {
    self.partial_rounds += 1;
    for (entries) |*entry| {
        try transformer_mod.ssmRollbackFromCapture(entry, accepted, verify_len, s);
    }
}

/// Keep the recurrent state this pipelined step starts from, dropping the oldest. Solo slots
/// only: a live batched group's states are copies, and a model that writes its state in place keeps none.
pub fn pushSsmLag(self: *Generator) void {
    const entries = self.ctx.ssm_entries orelse return;
    if (self.xfm.config.plugin) |b| if (b.parsed.cache.ssm_state.snapshot_policy == .in_place) return;
    for (entries) |*e| if (e.group) |g| if (g.live) return;
    var oldest = self.ssm_lag[0];
    self.ssm_lag[0] = self.ssm_lag[1];
    self.ssm_lag[1] = .{};
    for (oldest.snaps) |*s| ssmSnapshotDeinit(s);
    if (oldest.snaps.len != entries.len) {
        self.xfm.allocator.free(oldest.snaps);
        oldest.snaps = self.xfm.allocator.alloc(SSMCacheEntrySnapshot, entries.len) catch return;
    }
    for (entries, oldest.snaps) |*e, *s| s.* = ssmSnapshot(e);
    oldest.pos = self.ctx.moe_seq_offset.*;
    self.ssm_lag[1] = oldest;
}

/// Restore the recurrent state a pipelined step started from at `pos` (the committed length), so
/// the finishing slot's live state is exactly what it commits. False when no kept step is there.
pub fn rollbackSsmTo(self: *Generator, pos: usize) bool {
    const entries = self.ctx.ssm_entries orelse return false;
    for (&self.ssm_lag) |*lag| {
        if (lag.snaps.len != entries.len or lag.pos != pos) continue;
        for (entries, lag.snaps) |*e, *s| ssmRestore(e, s) catch return false;
        self.ctx.moe_seq_offset.* = pos;
        return true;
    }
    return false;
}
