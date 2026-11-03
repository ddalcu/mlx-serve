//! The legacy `Transformer`'s methods over `sdk/blocks/attention/batched.zig` (what reads the engine's model state).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const layered = @import("../batched/layered.zig");
const MAX_BATCH_ROWS = @import("../../../sdk/bases/run.zig").MAX_BATCH_ROWS;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const Transformer = @import("../transformer.zig").Transformer;
const axisView = @import("../../../mlx/array.zig").axisView;
const fusedSdpa256Prefill = @import("../../../sdk/blocks/attention/attn256.zig").fusedSdpa256Prefill;
const kvAttnFusedEligible = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").kvAttnFusedEligible;
const kvAttnVerifyEligible = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").kvAttnVerifyEligible;
const logKvAttnFusedEngaged = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").logKvAttnFusedEngaged;
const qkvAttnDecodeKernel = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnDecodeKernel;
const qkvAttnVerifyKernel = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnVerifyKernel;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const sdpaForceFused = @import("../../../sdk/blocks/attention/sdpa.zig").sdpaForceFused;
const splitCausalSdpa = @import("../../../sdk/blocks/attention/sdpa.zig").splitCausalSdpa;
const BATCHED_PER_SLOT_ATTN_MIN_KV = @import("../../../sdk/blocks/attention/batched.zig").BATCHED_PER_SLOT_ATTN_MIN_KV;

/// The stacked batched read: every slot's view padded to the longest and concatenated,
/// one array-mask SDPA for the group. `narrow` (bool, null-ctx for none) removes more
/// positions from the pad mask.
pub fn stackedBatchedAttn(
    self: *const Transformer,
    q: mlx.mlx_array,
    views: []const DenseKVView,
    seq_len: c_int,
    attn_scale: f32,
    narrow: mlx.mlx_array,
) !mlx.mlx_array {
    var kv_len_buf: [MAX_BATCH_ROWS]i32 = undefined;
    var kv_max: c_int = 0;
    for (views, 0..) |dv, i| {
        kv_len_buf[i] = mlx.getShape(dv.k)[2];
        kv_max = @max(kv_max, kv_len_buf[i]);
    }
    const stacked_k = try self.padAndStackBatchedKV(views, true, kv_max);
    defer _ = mlx.mlx_array_free(stacked_k);
    const stacked_v = try self.padAndStackBatchedKV(views, false, kv_max);
    defer _ = mlx.mlx_array_free(stacked_v);
    var mask = try self.buildBatchedDecodeMask(kv_len_buf[0..views.len], kv_max, seq_len, mlx.mlx_array_dtype(q));
    defer _ = mlx.mlx_array_free(mask);
    if (narrow.ctx != null) {
        const neg_inf = try scalarOf(-std.math.inf(f32), mlx.mlx_array_dtype(mask), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var narrowed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&narrowed, narrow, mask, neg_inf, self.s));
        _ = mlx.mlx_array_free(mask);
        mask = narrowed;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, stacked_k, stacked_v, attn_scale, "array", mask, .{ .ctx = null }, false, self.s));
    return out;
}

/// The per-slot batched read: each slot's query rows attend its OWN view, so nothing is
/// padded, copied or masked, and the causal and quantized-KV kernels serve it as they
/// serve a solo slot. `fused[i]` is slot i's `ForwardCtx.kv_attn_fused`.
pub fn perSlotBatchedAttn(
    self: *const Transformer,
    q: mlx.mlx_array,
    views: []const DenseKVView,
    fused: []const bool,
    seq_len: c_int,
    attn_scale: f32,
) !mlx.mlx_array {
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var outs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (outs[0..n]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    if (!layered.per_slot_attn_logged) {
        layered.per_slot_attn_logged = true;
        log.info("[batched] per-slot attention engaged (slots={d}, rows={d}): no KV pad or stack past {d} tokens\n", .{ views.len, seq_len, BATCHED_PER_SLOT_ATTN_MIN_KV });
    }
    for (views, 0..) |*dv, i| {
        const q_i = try axisView(self.s, q, 0, i);
        defer _ = mlx.mlx_array_free(q_i);
        const mode: [:0]const u8 = if (seq_len > 1) "causal" else "";
        var out: ?mlx.mlx_array = null;
        if (fused[i] and kvAttnFusedEligible(dv, seq_len)) {
            out = try qkvAttnDecodeKernel(self.s, q_i, dv, attn_scale, mode, none_mask);
            if (out != null) logKvAttnFusedEngaged(dv, q_i, seq_len);
        } else if (fused[i] and kvAttnVerifyEligible(dv, seq_len)) {
            out = try qkvAttnVerifyKernel(self.s, q_i, dv, attn_scale, mode);
        }
        if (out == null and seq_len > 1) {
            out = try fusedSdpa256Prefill(self.s, q_i, dv.k, dv.v, attn_scale, 0);
            if (out == null) out = try splitCausalSdpa(self.s, q_i, dv.k, dv.v, attn_scale);
        }
        if (out == null) {
            var o = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q_i, dv.k, dv.v, attn_scale, mode.ptr, none_mask, .{ .ctx = null }, seq_len > 1 and sdpaForceFused(q_i, dv.k), self.s));
            out = o;
        }
        outs[n] = out.?;
        n += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&outs, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var joined = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, self.s));
    return joined;
}

pub fn padAndStackBatchedKV(
    self: *const Transformer,
    views: []const DenseKVView,
    key_not_value: bool,
    kv_max: c_int,
) !mlx.mlx_array {
    const pad_axes = [_]c_int{2};
    const pad_value = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(pad_value);

    const padded_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(padded_vec);
    // Track padded arrays so we can free them after the concat.
    var padded_arrs = try self.allocator.alloc(mlx.mlx_array, views.len);
    defer {
        for (padded_arrs) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(padded_arrs);
    }

    for (views, 0..) |dv, i| {
        const view = if (key_not_value) dv.k else dv.v;
        const vshape = mlx.getShape(view);
        const klen: c_int = vshape[2];
        const high_pad: c_int = kv_max - klen;
        const low_pad_arr = [_]c_int{0};
        const high_pad_arr = [_]c_int{high_pad};
        padded_arrs[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_pad(
            &padded_arrs[i],
            view,
            &pad_axes,
            1,
            &low_pad_arr,
            1,
            &high_pad_arr,
            1,
            pad_value,
            "constant",
            self.s,
        ));
        _ = mlx.mlx_vector_array_append_value(padded_vec, padded_arrs[i]);
    }

    var stacked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, padded_vec, 0, self.s));
    return stacked;
}

/// `[N, 1, S, kv_max]` additive mask: row j of slot n sees positions below
/// `kv_len_n - S + 1 + j` (its own causal prefix; the pad past `kv_len_n` never).
/// Additive [N, 1, S, kv_max] mask in `dt`, the queries' dtype: sdpa
/// rejects a mask that does not promote to its output type.
pub fn buildBatchedDecodeMask(
    self: *const Transformer,
    kv_lens: []const i32,
    kv_max: c_int,
    seq_len: c_int,
    dt: mlx.mlx_dtype,
) !mlx.mlx_array {
    const N: c_int = @intCast(kv_lens.len);
    var positions = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(positions);
    try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_max), 1, .int32, self.s));
    const pos_shape = [_]c_int{ 1, 1, 1, kv_max };
    var pos_4d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_4d);
    try mlx.check(mlx.mlx_reshape(&pos_4d, positions, &pos_shape, 4, self.s));

    // limit[n, 0, j, 0] = kv_len_n - S + 1 + j
    var lim_buf: [MAX_BATCH_ROWS * 16]i32 = undefined;
    if (kv_lens.len * @as(usize, @intCast(seq_len)) > lim_buf.len) return error.BatchedMaskTooWide;
    for (kv_lens, 0..) |kl, n| {
        for (0..@intCast(seq_len)) |j| lim_buf[n * @as(usize, @intCast(seq_len)) + j] = kl - seq_len + 1 + @as(i32, @intCast(j));
    }
    const lim_shape = [_]c_int{ N, 1, seq_len, 1 };
    const lens_4d = mlx.mlx_array_new_data(lim_buf[0 .. kv_lens.len * @as(usize, @intCast(seq_len))].ptr, &lim_shape, 4, .int32);
    defer _ = mlx.mlx_array_free(lens_4d);

    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less(&valid, pos_4d, lens_4d, self.s));

    const zero = try scalarOf(0.0, dt, self.s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = try scalarOf(-std.math.inf(f32), dt, self.s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&mask, valid, zero, neg_inf, self.s));
    return mask;
}
