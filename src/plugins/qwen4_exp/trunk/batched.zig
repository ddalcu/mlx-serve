//! The trunk's batched ticks: one decode row (or one verify block) per slot through ONE forward, every slot's GDN
//! state and PLE window merged into a batch-N entry (kept merged across ticks by the persistent group) and each
//! slot's QSA keys read from its own cache.
//! copied_from: engines/mlx/batched/{layered,caps}.zig, engines/mlx/ssm/group.zig (the qwen4_exp arm).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const Trunk = @import("../trunk.zig").Trunk;
const ple = @import("../ple.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const splitSpecCaptures = @import("../../../engines/mlx/ssm/entry.zig").splitSpecCaptures;
const ssmFreeSpecCapture = @import("../../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const group = @import("../../../engines/mlx/ssm/group.zig");
const sliceBatchRows = @import("../../../engines/mlx/batched/layered.zig").sliceBatchRows;

var decode_logged = false;
var verify_rows_logged: c_int = 0;

/// Every slot's next token `[N, 1]` in one forward; N logits `[1, 1, V]` (and, with `hidden_rows`, each slot's
/// pre-mixer stream row) the caller owns.
pub fn forwardMoeBatchedDecode(self: *Trunk, next_tokens: []const u32, ctxs: []const *ForwardCtx, rope_offsets: []const u32, hidden_rows: ?*?[]mlx.mlx_array) ![]mlx.mlx_array {
    std.debug.assert(next_tokens.len == ctxs.len);
    if (!decode_logged) {
        decode_logged = true;
        log.info("[batched] gdn batched decode engaged (slots={d})\n", .{next_tokens.len});
    }
    const token_buf = try self.allocator.alloc(i32, next_tokens.len);
    defer self.allocator.free(token_buf);
    for (next_tokens, 0..) |t, i| token_buf[i] = @intCast(t);
    const token_arr = mlx.mlx_array_new_data(token_buf.ptr, &[_]c_int{ @intCast(next_tokens.len), 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(token_arr);
    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    const logits = try self.forwardMoeBatchedRows(token_arr, ctxs, rope_offsets, false, if (hidden_rows != null) &hidden else null, null);
    defer _ = mlx.mlx_array_free(logits);
    if (hidden_rows) |out| out.* = try sliceBatchRows(self.allocator, self.s, hidden, next_tokens.len);
    return sliceBatchRows(self.allocator, self.s, logits, next_tokens.len);
}

/// A batched spec verify: `[N, S]` rows (every slot's `[t1, drafts…]`) with each recurrent row's state captured for
/// the slot's own rollback; logits `[N, S, V]`, the stream at the last row and at every row.
pub fn forwardMoeBatchedVerify(self: *Trunk, token_arr: mlx.mlx_array, ctxs: []const *ForwardCtx, rope_offsets: []const u32, hidden_last: *mlx.mlx_array, hidden_all: *mlx.mlx_array) !mlx.mlx_array {
    const width = mlx.getShape(token_arr)[1];
    const rows = @as(c_int, @intCast(ctxs.len)) * width;
    if (rows > verify_rows_logged) {
        verify_rows_logged = rows;
        log.info("[batched] gdn batched verify engaged (slots={d}, width={d})\n", .{ ctxs.len, width });
    }
    return self.forwardMoeBatchedRows(token_arr, ctxs, rope_offsets, true, hidden_last, hidden_all);
}

pub fn forwardMoeBatchedRows(self: *Trunk, token_arr: mlx.mlx_array, ctxs: []const *ForwardCtx, rope_offsets: []const u32, capture_ssm: bool, hidden_last: ?*mlx.mlx_array, hidden_all: ?*mlx.mlx_array) !mlx.mlx_array {
    self.fwd_gen +%= 1;
    const N: c_int = @intCast(ctxs.len);
    std.debug.assert(ctxs.len == rope_offsets.len);
    std.debug.assert(N >= 1);
    const ml = self.moe_layers;

    // Per-slot RoPE offsets, each already carrying its own M-RoPE delta.
    const off_buf = try self.allocator.alloc(i32, rope_offsets.len);
    defer self.allocator.free(off_buf);
    for (rope_offsets, 0..) |o, i| {
        const d: i32 = if (ctxs[i].mrope_pos != null) ctxs[i].mrope_delta else 0;
        off_buf[i] = @as(i32, @intCast(o)) + d;
    }
    const rope_offset_arr = mlx.mlx_array_new_data(off_buf.ptr, &[_]c_int{N}, 1, .int32);
    defer _ = mlx.mlx_array_free(rope_offset_arr);

    // Each linear (and PLE) layer's per-slot state merged into one batch-N entry; `batchedGdnReady` gates on every
    // slot carrying it.
    const persist = self.persistent_group_state_override orelse true;
    self.ssm_group.allocator = self.allocator;
    const stable = persist and self.ssm_group.live and self.ssm_group.matches(ctxs);
    var kinds_buf: []group.SsmMergeKind = &.{};
    defer if (kinds_buf.len != 0) self.allocator.free(kinds_buf);
    const kinds: []const group.SsmMergeKind = if (stable) self.ssm_group.kinds else blk: {
        kinds_buf = try self.allocator.alloc(group.SsmMergeKind, ml.len);
        for (ml, 0..) |*lw, l| kinds_buf[l] = .{ .linear = lw.is_linear, .ple = lw.ple != null };
        break :blk kinds_buf;
    };
    const bind = try group.ssmBindTickState(self.allocator, self.s, &self.ssm_group, ctxs, kinds, persist);
    const merged = bind.merged;
    defer if (bind.owned) {
        for (merged) |*m| group.ssmFreeMergedLayer(m);
        self.allocator.free(merged);
    };

    // `cache` and `moe_seq_offset` are never read on this path (attention reaches each slot's cache through
    // `batch_slots`, rope through the offsets array); slot 0's stand in for the non-optional fields.
    var scratch_offset: usize = rope_offsets[0];
    var bctx: ForwardCtx = .{
        .cache = ctxs[0].cache,
        .moe_seq_offset = &scratch_offset,
        .ssm_entries = merged,
        .capture_hidden = hidden_last,
        .capture_hidden_all = hidden_all,
        .capture_ssm_seq = capture_ssm,
        .vision_embeddings = null,
        .batch_slots = ctxs,
        .batch_rope_offsets = rope_offset_arr,
        .ple_defer = true,
    };
    // A verify's captures live on the merged entries until split below.
    defer if (capture_ssm) {
        for (merged) |*m| ssmFreeSpecCapture(m);
    };
    const logits = self.forwardQwen4With(&bctx, token_arr) catch |e| {
        ple.discardDeferredPle(self, &bctx);
        return e;
    };
    errdefer {
        ple.discardDeferredPle(self, &bctx);
        _ = mlx.mlx_array_free(logits);
    }
    try ple.flushDeferredPle(self, &bctx);

    if (persist) {
        if (capture_ssm) {
            for (merged, 0..) |*m, li| {
                if (!kinds[li].linear and !kinds[li].ple) continue;
                for (ctxs, 0..) |c, i| try splitSpecCaptures(self.s, m, &c.ssm_entries.?[li], i);
            }
        }
        if (!stable) group.ssmBindSlotRefs(&self.ssm_group, ctxs);
    } else {
        try group.ssmSplitAll(self.s, merged, ctxs, kinds);
    }
    return logits;
}

/// Can these slots take one batched tick? Every slot must carry live state on every linear layer and its PLE
/// window and token history (true after any prefill).
pub fn batchedGdnReady(self: *const Trunk, ctxs: []const *ForwardCtx) bool {
    const ml = self.moe_layers;
    for (ctxs) |c| {
        const entries = c.ssm_entries orelse return false;
        if (entries.len != ml.len) return false;
        for (ml, 0..) |*lw, i| {
            if (lw.ple != null and (entries[i].aux_state.ctx == null or !entries[i].ple_prev_valid)) return false;
            if (!lw.is_linear) continue;
            if (!entries[i].initialized) return false;
            if (entries[i].conv_state.ctx == null or entries[i].ssm_state.ctx == null) return false;
        }
    }
    return true;
}

/// At the entry of every non-group forward: a member's live row goes back to its own entries first.
pub fn ssmGroupRelease(self: *Trunk, ctx: *ForwardCtx) !void {
    try group.ssmGroupReleaseCtx(&self.ssm_group, self.s, ctx);
}

pub fn ssmGroupDrop(self: *Trunk, entries: []SSMCacheEntry) void {
    group.ssmGroupDropMember(&self.ssm_group, self.s, entries) catch {};
}

/// One layer's per-slot recurrent state merged into a batch-N entry.
/// copied_from: engines/mlx/ssm/group.zig (`mergeSsmAcrossSlots`).
pub fn mergeSsmAcrossSlots(self: *Trunk, ctxs: []const *ForwardCtx, layer: usize) !SSMCacheEntry {
    const lw = &self.moe_layers[layer];
    return group.ssmMergeLayer(self.s, ctxs, layer, .{ .linear = lw.is_linear, .ple = lw.ple != null });
}

/// Writes the batch-N entry `m` back to each slot's own entry at `layer`.
/// copied_from: engines/mlx/ssm/group.zig (`splitSsmToSlots`).
pub fn splitSsmToSlots(self: *Trunk, m: *const SSMCacheEntry, ctxs: []const *ForwardCtx, layer: usize) !void {
    const lw = &self.moe_layers[layer];
    try group.ssmSplitLayer(self.s, m, ctxs, layer, .{ .linear = lw.is_linear, .ple = lw.ple != null });
}
