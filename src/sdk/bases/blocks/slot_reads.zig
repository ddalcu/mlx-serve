//! A batched forward's attention reads over its slots' KV views: each slot's rows against its own view, or every view
//! padded to the longest and stacked under one array mask. Shared by the decoder and gated attention blocks.
//! copied_from: sdk/blocks/attention/batched.zig (`perSlotBatchedAttn`, `stackedBatchedAttn`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const da = @import("decoder_attention.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");
const sdpa = @import("../../blocks/attention/sdpa.zig");
const kvq = @import("../../blocks/attention/kv_quant_attn.zig");
const DenseKVView = @import("../../blocks/attention/kv_view.zig").DenseKVView;

const Array = ops.Array;
const Graph = ops.Graph;
const own = da.own;
const sdpaRaw = da.sdpaRaw;

/// Slots a batched forward takes (the engine's `MAX_BATCH_ROWS`).
pub const MAX_SLOTS = 32;

/// Each slot's query rows attend its own view: no pad, no stack, no mask (`perSlotBatchedAttn`).
pub fn perSlot(parent: *Graph, q: Array, views: []const DenseKVView, packed_reads: []const bool, seq: c_int, scale: f32) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    var outs: [MAX_SLOTS]Array = undefined;
    const none = Array{};
    for (views, 0..) |*dv, i| {
        const q_i = try g.adopt(try @import("../../../mlx/array.zig").axisView(g.s, q, 0, i));
        const mode: [:0]const u8 = if (seq > 1) "causal" else "";
        var out: ?Array = null;
        if (packed_reads[i] and kvq.kvAttnFusedEligible(dv, seq)) {
            out = try own(g, try kvq.qkvAttnDecodeKernel(g.s, q_i, dv, scale, mode, none));
            if (out != null) kvq.logKvAttnFusedEngaged(dv, q_i, seq);
        } else if (packed_reads[i] and kvq.kvAttnVerifyEligible(dv, seq)) {
            out = try own(g, try kvq.qkvAttnVerifyKernel(g.s, q_i, dv, scale, mode));
        }
        if (out == null and seq > 1) {
            out = try own(g, try attn256.fusedSdpa256Prefill(g.s, q_i, dv.k, dv.v, scale, 0));
            if (out == null) out = try own(g, try sdpa.splitCausalSdpa(g.s, q_i, dv.k, dv.v, scale));
        }
        if (out == null) out = try sdpaRaw(g, q_i, dv.k, dv.v, scale, mode.ptr, none, seq > 1 and sdpa.sdpaForceFused(q_i, dv.k));
        outs[i] = out.?;
    }
    return parent.keep(try g.concat(outs[0..views.len], 0));
}

/// Every slot's view padded to the longest and stacked, one array-masked read (`stackedBatchedAttn`).
pub fn stacked(parent: *Graph, q: Array, views: []const DenseKVView, seq: c_int, scale: f32) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    var lens: [MAX_SLOTS]i32 = undefined;
    var kv_max: c_int = 0;
    for (views, 0..) |dv, i| {
        lens[i] = ops.shape(dv.k)[2];
        kv_max = @max(kv_max, lens[i]);
    }
    const k = try padStack(g, views, true, kv_max);
    const v = try padStack(g, views, false, kv_max);
    const mask = try decodeMask(g, lens[0..views.len], kv_max, seq, ops.dtype(q));
    return parent.keep(try sdpaRaw(g, q, k, v, scale, "array", mask, false));
}

fn padStack(parent: *Graph, views: []const DenseKVView, keys: bool, kv_max: c_int) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const zero = try g.scalar(0, .bfloat16);
    var padded: [MAX_SLOTS]Array = undefined;
    for (views, 0..) |dv, i| {
        const x = if (keys) dv.k else dv.v;
        const hi = [_]c_int{kv_max - ops.shape(x)[2]};
        padded[i] = try g.call(mlx.mlx_pad, .{ x, &[_]c_int{2}, @as(usize, 1), &[_]c_int{0}, @as(usize, 1), &hi, @as(usize, 1), zero, "constant" });
    }
    return parent.keep(try g.concat(padded[0..views.len], 0));
}

/// Additive `[N, 1, S, kv_max]` mask: row j of slot n sees positions below `len_n - S + 1 + j`.
fn decodeMask(parent: *Graph, lens: []const i32, kv_max: c_int, seq: c_int, dt: mlx.mlx_dtype) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const n: c_int = @intCast(lens.len);
    const pos = try g.reshape(try g.arange(0, @floatFromInt(kv_max), 1, .int32), &.{ 1, 1, 1, kv_max });
    var lim: [MAX_SLOTS * 16]i32 = undefined;
    const rows: usize = @intCast(seq);
    if (lens.len * rows > lim.len) return error.BatchedMaskTooWide;
    for (lens, 0..) |l, i| for (0..rows) |j| {
        lim[i * rows + j] = l - seq + 1 + @as(i32, @intCast(j));
    };
    const limits = try g.fromSlice(i32, lim[0 .. lens.len * rows], &.{ n, 1, seq, 1 });
    const valid = try g.less(pos, limits);
    return parent.keep(try g.where(valid, try g.scalar(0, dt), try g.scalar(-std.math.inf(f32), dt)));
}
