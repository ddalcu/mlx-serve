//! engines/mlx/batched/layered: the engine's batched decode and verify entries, routed to the served plugin.

const std = @import("std");
const plugin_driver = @import("../plugin/driver.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;

pub var per_slot_attn_logged: bool = false;
var gdn_batched_verify_rows_logged: c_int = 0; // widest row total logged so far
var gdn_batched_logged: bool = false;
pub var row_axis_verify_logged: bool = false;
pub var verify_fail_after_flush: ?usize = null;

pub const MAX_BATCH_ROWS = @import("../../../sdk/bases/run.zig").MAX_BATCH_ROWS;
pub var std_batched_logged: bool = false; // one-shot log guard

/// One batched decode tick for a served model: one row per slot at its own position. Returns N logits `[1,1,V]`, caller
/// owns each and the slice. `hidden_rows`: when given, receives every slot's post-norm hidden `[1,1,H]` (owned slice),
/// for a slot that keeps an MTP head resumable across plain ticks.
pub fn forwardMoeBatchedDecode(
    self: *Transformer,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    std.debug.assert(next_tokens.len == ctxs.len);
    // The guards grep this marker (`tests/test_batched_equivalence.sh`): equal output cannot tell batched from serial.
    if (!gdn_batched_logged) {
        gdn_batched_logged = true;
        log.info("[batched] gdn batched decode engaged (slots={d})\n", .{next_tokens.len});
    }
    const b = self.plugin orelse return error.BatchedDecodeUnsupported;
    return plugin_driver.forwardBatched(self, b, next_tokens, ctxs, rope_offsets, hidden_rows);
}

/// `forwardMoeBatchedDecode` for an attention-only trunk (its own engagement line).
pub fn forwardBatchedDecode(
    self: *Transformer,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
) ![]mlx.mlx_array {
    std.debug.assert(next_tokens.len == ctxs.len and next_tokens.len == rope_offsets.len);
    // Output equality alone cannot tell a batched run from N serial ones: every guard greps for this marker.
    if (!std_batched_logged) {
        std_batched_logged = true;
        log.info("[batched] batched decode engaged (slots={d})\n", .{next_tokens.len});
    }
    const b = self.plugin orelse return error.BatchedDecodeUnsupported;
    return plugin_driver.forwardBatched(self, b, next_tokens, ctxs, rope_offsets, null);
}

/// Batched spec VERIFY: `[N, S]` rows (every slot's `[t1, drafts…]`, padded to the
/// group's widest), per-position SSM capture ON so each slot rolls back on its own,
/// post-norm hidden captured at the last row (`hidden_last` `[N,1,H]`) and at every
/// row (`hidden_all` `[N,S,H]`). Returns logits `[N, S, V]`; caller owns everything.
pub fn forwardMoeBatchedVerify(
    self: *Transformer,
    token_arr: mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    hidden_last: *mlx.mlx_array,
    hidden_all: *mlx.mlx_array,
) !mlx.mlx_array {
    const width = mlx.getShape(token_arr)[1];
    const rows = @as(c_int, @intCast(ctxs.len)) * width;
    if (rows > gdn_batched_verify_rows_logged) {
        gdn_batched_verify_rows_logged = rows;
        log.info("[batched] gdn batched verify engaged (slots={d}, width={d})\n", .{ ctxs.len, width });
    }
    const b = self.plugin orelse return error.BatchedDecodeUnsupported;
    return plugin_driver.forwardBatchedVerify(self, b, token_arr, ctxs, rope_offsets, hidden_last, hidden_all);
}

/// Row `i` of `[N, S, V]` logits as an owned `[1, S, V]` array, for every row.
pub fn sliceBatchRows(allocator: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array, n: usize) ![]mlx.mlx_array {
    const shape = mlx.getShape(arr);
    const out = try allocator.alloc(mlx.mlx_array, n);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(out);
    }
    for (out, 0..) |*slot, i| {
        const i_c: c_int = @intCast(i);
        const start = [_]c_int{ i_c, 0, 0 };
        const stop = [_]c_int{ i_c + 1, shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        slot.* = mlx.mlx_array_new();
        built = i + 1;
        try mlx.check(mlx.mlx_slice(slot, arr, &start, 3, &stop, 3, &strides, 3, s));
    }
    return out;
}
