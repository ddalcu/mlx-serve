//! Hy4-flash's compressed indexer (`HYV4FlashCompressedIndexer`): every `compress_ratio` consecutive token
//! keys pool into one group key by a learned gated convex combination (fp32, `intra_group_ape` breaks slot ties),
//! group keys carry the RoPE phase of their group's first token, and a query keeps the `group_topk - 1` best
//! CLOSED groups plus the group it sits in — attention then reads the tokens those groups expand to.
//!
//! `compressed_indexer_skip_fine_select: true`: there is no second, per-token stage; the recall set is exactly
//! `group_topk * compress_ratio` token slots (2048 at the pack's 512 x 4).
//!
//! Reference: `scores = relu(q @ group_keys^T)` per head, `logits = sum_h head_weight[h] * scores[h]`,
//! `head_weight = head_weight_projection(x) * (heads*dim)^-0.5`, groups at or past the query's own group masked.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const proj = @import("proj.zig");
const sliding = @import("../../../sdk/bases/blocks/sliding.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Proj = proj.Proj;

pub const Indexer = struct {
    /// `q_lora_rank -> heads*head_dim`.
    query: Proj,
    /// `hidden -> heads`, f32 weights for the head mix.
    head_weight: Proj,
    /// `hidden -> head_dim`, the per-token key.
    token_key: Proj,
    token_key_norm: Array,
    /// Dense f32 `[head_dim, head_dim]`, pre-transposed to `[in, out]`.
    value_proj: Proj,
    /// Dense f32 `[head_dim, head_dim]`, pre-transposed.
    gate_proj: Proj,
    key_norm: Array,
    /// `[compress_ratio, head_dim]` f32 inside-group position embedding.
    ape: Array,
    heads: u32,
    dim: u32,
    ratio: u32,
    /// Complete closed groups a query keeps (`group_topk - 1`, its own group added).
    group_topk: u32,
    /// The recall budget in tokens (`index_topk`): below it the layer attends densely.
    budget: u32,
    /// The rotary width the indexer rotates on its keys and queries (the layer's rope tail).
    rope_dim: u32,
    theta: f32,
    rms_eps: f32,

    /// The per-token keys `[slots, rows, dim]` (bf16).
    pub fn tokenKeys(ix: *const Indexer, g: *Graph, x: Array) !Array {
        const raw = try ix.token_key.forward(g, x);
        return g.rmsNorm(raw, ix.token_key_norm, ix.rms_eps);
    }

    /// The recall mask `[1, 1, rows, kv_len]` (additive: 0 where a token slot is recalled, -inf elsewhere),
    /// already causal. `token_keys_all` is the cached history `[1, 1, kv_len, dim]`; `base_pos` is the slot's
    /// position of history row 0, and `positions` the query rows' absolute positions.
    pub fn recallMask(ix: *const Indexer, g: *Graph, x: Array, q_resid: Array, token_keys_all: Array,
        kv_len: c_int, base_pos: c_int, positions: []const u32) !Array {
        const rows: c_int = @intCast(positions.len);
        const ng: c_int = @divTrunc(kv_len, @as(c_int, @intCast(ix.ratio)));
        if (ng <= 0) return causalOnly(g, rows, kv_len);
        var t = g.scope();
        defer t.deinit();

        // Group keys: the gated pooling over complete groups, in fp32.
        const first = try t.slice(token_keys_all, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ng * @as(c_int, @intCast(ix.ratio)), ops.shape(token_keys_all)[3] });
        const grouped = try t.reshape(first, &.{ ng, @intCast(ix.ratio), @intCast(ix.dim) });
        const grouped_f = try t.astype(grouped, .float32);
        const values = try ix.value_proj.forward(&t, grouped_f); // [ng, ratio, dim]
        const gates = try t.add(try ix.gate_proj.forward(&t, grouped_f), ix.ape); // ape [ratio, dim] broadcasts
        const soft = try t.softmax(gates, 1, true);
        const pooled = try t.sum(try t.mul(values, soft), 1, false); // [ng, dim]
        const pooled_n = try t.astype(try t.rmsNorm(pooled, ix.key_norm, ix.rms_eps), ops.dtype(token_keys_all));
        const gpos = try t.arange(@floatFromInt(base_pos), @floatFromInt(base_pos + ng * @as(c_int, @intCast(ix.ratio))),
            @floatFromInt(ix.ratio), .float32); // [ng]
        const group_keys = try ropeTail(&t, pooled_n, @intCast(ix.rope_dim), ix.theta, gpos); // [ng, dim]

        // Queries: the q_lora projection, roped at their own positions.
        const q_heads = try t.reshape(try ix.query.forward(&t, q_resid), &.{ rows * @as(c_int, @intCast(ix.heads)), @intCast(ix.dim) });
        // Each head rotates at its row's position: repeat every position `heads` times.
        const qpos = blk: {
            const buf = try t.a.alloc(f32, positions.len * ix.heads);
            defer t.a.free(buf);
            for (positions, 0..) |p_, i| for (0..ix.heads) |h| {
                buf[i * ix.heads + h] = @floatFromInt(p_);
            };
            break :blk try t.fromSlice(f32, buf, &.{ rows * @as(c_int, @intCast(ix.heads)) });
        };
        const query = try t.reshape(try ropeTail(&t, q_heads, @intCast(ix.rope_dim), ix.theta, qpos),
            &.{ rows, @intCast(ix.heads), @intCast(ix.dim) });

        // Scores: relu(q @ keys^T) per head, mixed by the head weights, scaled by (heads*dim)^-0.5.
        const raw_scores = try t.matmul(try t.astype(query, .float32),
            try t.transpose(try t.astype(group_keys, .float32), &.{ 1, 0 })); // [rows, heads, ng]
        const scores = try t.maximum(raw_scores, try t.scalar(0, .float32));
        const scale: f32 = std.math.pow(f32, @floatFromInt(ix.heads * ix.dim), -0.5);
        const weights = try t.mulScalar(
            try t.reshape(try t.astype(try ix.head_weight.forward(&t, x), .float32), &.{ rows, @intCast(ix.heads) }),
            scale,
        ); // [rows, heads]
        const logits = try t.sum(try t.mul(scores, try t.expandDims(weights, -1)), 1, false); // [rows, ng]

        // A query sees only groups before its own; its own group is recalled unconditionally.
        // The group a query sits in follows its SLOT (the reference's `query_slots // ratio`), not its
        // absolute position: a prefix-cache restore keeps slots contiguous while positions carry the prefix.
        var own: [4096]i32 = undefined;
        if (positions.len > own.len) return error.TooManyRows;
        for (0..positions.len) |i| own[i] = @intCast((kv_len - rows + @as(c_int, @intCast(i))));
        for (own[0..positions.len]) |*o| o.* = @divTrunc(o.*, @as(i32, @intCast(ix.ratio)));
        const own_arr = try t.fromSlice(i32, own[0..positions.len], &.{rows});
        const group_ids = try t.arange(0, @floatFromInt(ng), 1, .int32);
        const past = try t.call(mlx.mlx_greater_equal, .{ try t.broadcast(group_ids, &.{ rows, ng }), try t.expandDims(own_arr, -1) });
        const neg_inf = try t.scalar(-std.math.inf(f32), .float32);
        const masked = try t.where(past, neg_inf, logits);

        const k: c_int = @intCast(@min(ix.group_topk - 1, @as(u32, @intCast(ng))));
        const ids = try t.argsort(try t.neg(masked), -1); // [rows, ng], descending logits
        const top = try t.slice(ids, &.{ 0, 0 }, &.{ rows, k });
        const recalled = try t.concat(&.{ top, try t.expandDims(own_arr, -1) }, -1); // [rows, group_topk]

        // Recall slots: `group * ratio + i` for every i in the group; all complete, so none overflows.
        const slots = try t.reshape(
            try t.add(
                try t.mul(try t.expandDims(recalled, -1), try t.scalar(@floatFromInt(ix.ratio), .int32)),
                try t.reshape(try t.arange(0, @floatFromInt(ix.ratio), 1, .int32), &.{ 1, 1, @intCast(ix.ratio) }),
            ),
            &.{ rows, @intCast(ix.group_topk * ix.ratio) },
        );
        const neg_inf16 = try t.scalar(-std.math.inf(f32), .bfloat16);
        const outside = try t.full(&.{ rows, kv_len }, neg_inf16, .bfloat16);
        var put = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_put_along_axis(&put, outside, slots, try t.zeros(&.{ rows, @intCast(ix.group_topk * ix.ratio) }, .bfloat16), -1, t.s));
        // Both arms are additive: causal + recall is 0 only where the query may read the slot.
        const causal = try causalMask(&t, rows, kv_len);
        const put_owned = try t.adopt(put);
        const mask4 = try t.add(try t.reshape(put_owned, &.{ 1, 1, rows, kv_len }), causal);
        return t.escape(mask4);
    }
};

/// A plain causal additive mask `[1, 1, rows, kv_len]` bf16 (the SDK's proven builder: query rows are the
/// cache's last `rows` slots).
fn causalMask(t: *Graph, rows: c_int, kv_len: c_int) !Array {
    return sliding.causalMask(t, rows, kv_len);
}

fn causalOnly(g: *Graph, rows: c_int, kv_len: c_int) !Array {
    var t = g.scope();
    defer t.deinit();
    return t.escape(try causalMask(&t, rows, kv_len));
}

/// Adjacent-pair rotation of the TRAILING `dim` channels of `x` `[n, width]` at `positions` `[n]` f32.
/// The tables round to the activation dtype before the rotation, as the reference does.
fn ropeTail(t: *Graph, x: Array, dim: c_int, theta: f32, positions: Array) !Array {
    const sh = ops.shape(x);
    const width = sh[sh.len - 1];
    const half = @divExact(dim, 2);
    const dt = ops.dtype(x);
    const n: c_int = sh[0];

    // inv_freq[i] = theta^(-i/half); angle = pos * inv_freq.
    const idx = try t.arange(0, @floatFromInt(half), 1, .float32);
    const inv = try t.exp(try t.mulScalar(idx, -@log(theta) / @as(f32, @floatFromInt(half))));
    const ang = try t.mul(try t.reshape(positions, &.{ n, 1 }), try t.reshape(inv, &.{ 1, half }));
    const cos_t = try t.astype(try t.cos(ang), dt);
    const sin_t = try t.astype(try t.sin(ang), dt);

    // Split the head: passthrough `[n, width-dim]`, rotary `[n, dim]` as `[n, half, 2]` pairs.
    const pass = try t.slice(x, &.{ 0, 0 }, &.{ n, width - dim });
    const rot = try t.slice(x, &.{ 0, width - dim }, &.{ n, width });
    const pairs = try t.reshape(rot, &.{ n, half, 2 });
    const x0 = try t.slice(pairs, &.{ 0, 0, 0 }, &.{ n, half, 1 });
    const x1 = try t.slice(pairs, &.{ 0, 0, 1 }, &.{ n, half, 2 });
    const c = try t.expandDims(cos_t, -1); // [n, half, 1]
    const s = try t.expandDims(sin_t, -1);
    const o0 = try t.sub(try t.mul(x0, c), try t.mul(x1, s));
    const o1 = try t.add(try t.mul(x0, s), try t.mul(x1, c));
    const joined = try t.reshape(try t.concat(&.{ o0, o1 }, -1), &.{ n, dim });
    return t.concat(&.{ pass, joined }, -1);
}

// ── tests ──

const testing = std.testing;

fn readI32(a: Array) ![]const i32 {
    try ops.eval(a);
    const n = mlx.mlx_array_size(a);
    const data = mlx.mlx_array_data_int32(a) orelse return error.NoData;
    return data[0..n];
}

/// An array's values as f32: bf16/f16 casts first (`mlx_array_data_float32` reads f32 buffers only).
fn readF32(a: Array) ![]const f32 {
    if (mlx.mlx_array_dtype(a) == .float32) {
        try ops.eval(a);
        const n = mlx.mlx_array_size(a);
        const data = mlx.mlx_array_data_float32(a) orelse return error.NoData;
        return data[0..n];
    }
    var cast = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, a, .float32, mlx.gpuStream()));
    try ops.eval(cast);
    const n = mlx.mlx_array_size(cast);
    const data = mlx.mlx_array_data_float32(cast) orelse return error.NoData;
    return data[0..n];
}
