//! The cached decoder's attention is `DecoderAttention(.{})` (llama, qwen3; batched rows too); here are the other two
//! in the legacy forwards' exact op order: `HybridAttention` the hybrid stacks' (LFM2, Nemotron-H), causal over the
//! slot's KV cache, and `BidirAttention` the encoder's (BERT). copied_from: sdk/blocks/attention/hybrid_attn.zig.

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const kv = @import("../kv.zig");
const linear = @import("linear.zig");
const row_exact = @import("row_exact.zig");
const da = @import("decoder_attention.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");
const sdpa = @import("../../blocks/attention/sdpa.zig");
const FUSED_ROWS_MAX_M = @import("../../blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;

const Array = ops.Array;
const Ctx = block.Ctx;
const Linear = linear.Linear;
const Joined = linear.Joined;
const rope = da.ropeAt;
const sdpaRaw = da.sdpaRaw;
const own = da.own;

const perm = [_]c_int{ 0, 2, 1, 3 };

pub const Geo = da.Geo;
/// The dense decoder's attention.
pub const Attention = da.DecoderAttention(.{});

/// The hybrid stacks' attention (LFM2 with q/k norms, Nemotron-H NoPE): unit-free causal reads, the cache keeping
/// `keep` rows of view.
pub const HybridAttention = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        q_norm: Array = .{},
        k_norm: Array = .{},
        qkv: Joined = .{},
        geo: Geo,
        /// The view's tail span on every append (the legacy passes `max_position_embeddings`).
        keep: u32 = 0,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        if (c.mode == .batched_rows) return error.BatchedRowsUnsupported;
        const g = c.g;
        const geo = &w.geo;
        const b: c_int = 1;
        const s = c.rows;
        const offset: c_int = @intCast(c.offset());
        const fused_rows = w.qkv.built() and b * s <= FUSED_ROWS_MAX_M;
        const lane = c.row_exact;
        const all: Array = if (fused_rows) try row_exact.linear(lane, g, &w.qkv.lin, x) else .{};
        const q_raw = if (fused_rows) try w.qkv.part(g, all, 0) else try row_exact.linear(lane, g, &w.q, x);
        const k_raw = if (fused_rows) try w.qkv.part(g, all, 1) else try row_exact.linear(lane, g, &w.k, x);
        const v_raw = if (fused_rows) try w.qkv.part(g, all, 2) else try row_exact.linear(lane, g, &w.v, x);
        var q = try g.reshape(q_raw, &.{ b, s, geo.heads, geo.head_dim });
        var k = try g.reshape(k_raw, &.{ b, s, geo.kv_heads, geo.head_dim });
        const v = try g.reshape(v_raw, &.{ b, s, geo.kv_heads, geo.head_dim });
        if (w.q_norm.ctx != null) {
            q = try g.rmsNorm(q, w.q_norm, c.shared.eps);
            k = try g.rmsNorm(k, w.k_norm, c.shared.eps);
        }
        const q_t = try rope(g, try g.transpose(q, &perm), geo, offset);
        const k_t = try rope(g, try g.transpose(k, &perm), geo, offset);
        const v_t = try g.transpose(v, &perm);

        var view = try kv.append(c, k_t, v_t, w.keep);
        defer kv.release(c, &view);
        const none = Array{};
        const out = if (try row_exact.attend(lane, g, q_t, mxa(view.k), mxa(view.v), geo.scale, s)) |rows|
            rows
        else if (c.wide())
            (try own(g, try attn256.fusedSdpa256Prefill(g.s, q_t, mxa(view.k), mxa(view.v), geo.scale, 0))) orelse
                try sdpaRaw(g, q_t, mxa(view.k), mxa(view.v), geo.scale, "causal", none, sdpa.sdpaForceFused(q_t, mxa(view.k)))
        else
            try sdpaRaw(g, q_t, mxa(view.k), mxa(view.v), geo.scale, "", none, false);
        return row_exact.linear(lane, g, &w.o, try g.mergeHeads(out));
    }
};

/// Bidirectional attention with additive projection biases and no cache (BERT); `Ctx.mask` masks padded keys.
/// copied_from: plugins/bert/forward.zig (`forwardBertWith`'s attention).
pub const BidirAttention = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        heads: c_int,
        head_dim: c_int,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const q = try g.splitHeads(try w.q.forward(g, x), w.heads, w.head_dim);
        const k = try g.splitHeads(try w.k.forward(g, x), w.heads, w.head_dim);
        const v = try g.splitHeads(try w.v.forward(g, x), w.heads, w.head_dim);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(w.head_dim)));
        const mode: [*:0]const u8 = if (c.mask.ctx != null) "array" else "";
        return w.o.forward(g, try g.mergeHeads(try sdpaRaw(g, q, k, v, scale, mode, c.mask, false)));
    }
};

fn mxa(a: @import("../../../abi/plugin.zig").Array) Array {
    return .{ .ctx = a.ctx };
}
