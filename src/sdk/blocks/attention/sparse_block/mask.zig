//! The dense key mask of block picks, and the masked attention arm that reads it.

const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const qsa_row_mask = @import("row_mask.zig");
const ops = @import("../../../../mlx/ops.zig");
const Graph = ops.Graph;
const Blocks = @import("params.zig").Blocks;
const scalarOf = @import("../../../../mlx/array.zig").scalarOf;
const fusedSdpa256Masked = @import("../attn256.zig").fusedSdpa256Masked;
const splitMaskedSdpa256 = @import("../sdpa.zig").splitMaskedSdpa256;

/// Dense [B,1,S,kv] bool mask equivalent to a block selection (fallback arm
/// + fixture trace): the selected blocks' tokens ∪ each row's own incomplete
/// tail, ∧ causal. Sentinel entries land in a spare column and are sliced off.
pub fn qsaMaskFromBlocks(s: mlx.mlx_stream, blocks: mlx.mlx_array, kv: c_int, ratio: c_int) !mlx.mlx_array {
    const bs = mlx.getShape(blocks);
    const batch = bs[0];
    const seq_len = bs[1];
    if (batch == 1 and seq_len == 1) if (try qsa_row_mask.rowMask(s, blocks, kv, ratio)) |m| return m;
    const nb: c_int = @divTrunc(kv, ratio);
    const nb_i = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_i);
    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less(&valid, blocks, nb_i, s));
    var safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(safe);
    try mlx.check(mlx.mlx_where(&safe, valid, blocks, nb_i, s));
    var falses = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(falses);
    try mlx.check(mlx.mlx_zeros(&falses, &[_]c_int{ batch, seq_len, nb + 1 }, 3, .bool_, s));
    const true_v = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(true_v);
    var put = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(put);
    try mlx.check(mlx.mlx_put_along_axis(&put, falses, safe, true_v, -1, s));
    var blk_sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_sel);
    const strides3 = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&blk_sel, put, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, nb }, 3, &strides3, 3, s));
    return qsaMaskFromBlockSel(s, blk_sel, kv - seq_len, seq_len, kv, nb, ratio, batch);
}

/// Expand a [B,S,nb] bool block selection to the [B,1,S,kv] token mask:
/// selected blocks' tokens (zero-padded to kv) ∪ each query's own incomplete
/// tail (tokens at/after ratio*floor((p+1)/ratio) — always visible), ∧ the
/// causal bound. `offset` is the cache position of row 0.
pub fn qsaMaskFromBlockSel(s: mlx.mlx_stream, blk_sel: mlx.mlx_array, offset: c_int, seq_len: c_int, kv: c_int, nb: c_int, ratio: c_int, batch: c_int) !mlx.mlx_array {
    var pos_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_col);
    try mlx.check(mlx.mlx_arange(&pos_col, @floatFromInt(offset), @floatFromInt(offset + seq_len), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos_col, &[_]c_int{ seq_len, 1 }, 2, s));
    var sel4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel4);
    try mlx.check(mlx.mlx_expand_dims(&sel4, blk_sel, -1, s)); // [B,S,nb,1]
    var sel_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_b);
    try mlx.check(mlx.mlx_broadcast_to(&sel_b, sel4, &[_]c_int{ batch, seq_len, nb, ratio }, 4, s));
    var sel_tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_tok);
    try mlx.check(mlx.mlx_reshape(&sel_tok, sel_b, &[_]c_int{ batch, seq_len, nb * ratio }, 3, s));
    var sel_full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_full);
    const pad: c_int = kv - nb * ratio;
    if (pad > 0) {
        var zeros_pad = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zeros_pad);
        try mlx.check(mlx.mlx_zeros(&zeros_pad, &[_]c_int{ batch, seq_len, pad }, 3, .bool_, s));
        const parts = [_]mlx.mlx_array{ sel_tok, zeros_pad };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&sel_full, vec, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&sel_full, sel_tok));
    }
    var kv_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_idx);
    try mlx.check(mlx.mlx_arange(&kv_idx, 0, @floatFromInt(kv), 1.0, .int32, s));
    var kv_idx2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_idx2);
    try mlx.check(mlx.mlx_reshape(&kv_idx2, kv_idx, &[_]c_int{ 1, kv }, 2, s));
    // tail_start[p] = ratio * floor((p+1)/ratio), via integer arithmetic.
    const one_i = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(one_i);
    const ratio_i = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_i);
    var p1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p1);
    try mlx.check(mlx.mlx_add(&p1, pos2, one_i, s));
    var nb_p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb_p);
    try mlx.check(mlx.mlx_floor_divide(&nb_p, p1, ratio_i, s));
    var tail_start = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_start);
    try mlx.check(mlx.mlx_multiply(&tail_start, nb_p, ratio_i, s)); // [S,1]
    var tail_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_mask);
    try mlx.check(mlx.mlx_greater_equal(&tail_mask, kv_idx2, tail_start, s)); // [S,kv]
    var tok_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_mask);
    try mlx.check(mlx.mlx_logical_or(&tok_mask, sel_full, tail_mask, s));
    var causal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(causal);
    try mlx.check(mlx.mlx_less_equal(&causal, kv_idx2, pos2, s)); // [S, kv]
    var mask3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask3);
    try mlx.check(mlx.mlx_logical_and(&mask3, tok_mask, causal, s));
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask, mask3, &[_]c_int{ batch, 1, seq_len, kv }, 4, s));
    return mask;
}

/// The key mask of picks `[1, lists * rows, K]` for the last `rows` queries of a `kv`-key cache: each list's blocks
/// and the rows' tails, `[1, 1, rows, kv]` with one list, else `[1, heads, rows, kv]` with q head h reading list
/// h / (heads / lists).
pub fn picksMask(a: std.mem.Allocator, s: mlx.mlx_stream, b: Blocks, picks: mlx.mlx_array, rows: c_int, kv: c_int, heads: c_int) !mlx.mlx_array {
    var g = Graph.init(a, s);
    defer g.deinit();
    const ps = ops.shape(picks);
    const lists = @divExact(ps[1], rows);
    const m = try g.adopt(try qsaMaskFromBlocks(s, try g.reshape(picks, &.{ lists, rows, ps[2] }), kv, @intCast(b.size)));
    if (lists == 1) return g.escape(m);
    const per = @divExact(heads, lists);
    const wide = try g.broadcast(try g.reshape(m, &.{ 1, lists, 1, rows, kv }), &.{ 1, lists, per, rows, kv });
    return g.escape(try g.reshape(wide, &.{ 1, heads, rows, kv }));
}

/// The dense-mask arm: the fused masked kernel, the split vector sdpa, then stock sdpa.
pub fn qsaMaskArm(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask: mlx.mlx_array) !mlx.mlx_array {
    if (try fusedSdpa256Masked(s, q, k, v, scale, mask)) |o| return o;
    if (try splitMaskedSdpa256(s, q, k, v, scale, mask)) |o| return o;
    // MLX's CUDA SDPA turns a bool mask into 0 / -inf after broadcasting it over the heads, three
    // [B, H, S, kv] arrays; the same values over one head reach cuDNN as a broadcast view.
    var additive = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(additive);
    if (mlx.cudaAvailable() and mlx.mlx_array_dtype(mask) == .bool_) {
        const zero = try scalarOf(0, mlx.mlx_array_dtype(q), s);
        defer _ = mlx.mlx_array_free(zero);
        const neg_inf = try scalarOf(-std.math.inf(f32), mlx.mlx_array_dtype(q), s);
        defer _ = mlx.mlx_array_free(neg_inf);
        try mlx.check(mlx.mlx_where(&additive, mask, zero, neg_inf, s));
    } else try mlx.check(mlx.mlx_array_set(&additive, mask));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k, v, scale, "array", additive, .{ .ctx = null }, false, s));
    return out;
}
