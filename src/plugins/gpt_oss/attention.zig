//! gpt-oss attention: plain GQA with additive q/k/v/o biases, no q/k norm, learned per-head sinks as an extra softmax
//! column, YaRN on every layer (its mscale on the rotated dims), sliding layers over a block-sized tail with a band
//! mask at prefill and a window mask at decode. Serial only.
//! copied_from: sdk/blocks/attention/sinks.zig (`sinkAttnWith`, `sinkAttnCore`, `sinkAttnOut` without MiMo's
//! per-layer geometry, value scale, ring and decode copies, and without the batched slots).

const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const kv = @import("../../sdk/bases/kv.zig");
const linear = @import("../../sdk/bases/blocks/linear.zig");
const sliding = @import("../../sdk/bases/blocks/sliding.zig");
const attention = @import("../../sdk/bases/blocks/attention.zig");
const da = @import("../../sdk/bases/blocks/decoder_attention.zig");
const attn256 = @import("../../sdk/blocks/attention/attn256.zig");
const dec_attn = @import("../mimo_v2/dec_attn.zig");

const Array = ops.Array;
const Ctx = block.Ctx;
const Linear = linear.Linear;

const perm = [_]c_int{ 0, 2, 1, 3 };

pub const SinkAttention = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        /// `[heads]`; null = none.
        sinks: Array = .{},
        /// `geo.freqs` set = YaRN; `mscale` its f32 `[head_dim]` attention-factor table.
        geo: attention.Geo,
        mscale: Array = .{},
        /// Keys a sliding layer attends; 0 = a full layer.
        window: u32 = 0,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const geo = &w.geo;
        const s = c.rows;
        const offset: c_int = @intCast(c.offset());
        const q_t = try g.transpose(try g.reshape(try w.q.forward(g, x), &.{ 1, s, geo.heads, geo.head_dim }), &perm);
        const kv_shape = [_]c_int{ 1, s, geo.kv_heads, geo.head_dim };
        const k_t = try g.transpose(try g.reshape(try w.k.forward(g, x), &kv_shape), &perm);
        var q = try da.ropeAt(g, q_t, geo, offset);
        var k = try da.ropeAt(g, k_t, geo, offset);
        if (geo.freqs.ctx != null and w.mscale.ctx != null) {
            const ms = try g.astype(w.mscale, ops.dtype(q));
            q = try g.mul(q, ms);
            k = try g.mul(k, ms);
        }
        const v = try g.transpose(try g.reshape(try w.v.forward(g, x), &kv_shape), &perm);

        const total = offset + s;
        const sv = sliding.view(w.window, geo.head_dim, total, s);
        var view = try kv.append(c, k, v, if (w.window > 0) sv.span else 0);
        defer kv.release(c, &view);
        const dv = kv.dense(view);
        const out = try core(w, c, q, dv.k, dv.v, total, sv.kv_len);
        return w.o.forward(g, try g.mergeHeads(out));
    }

    /// Fused kernels where they serve, else MLX's SDPA with the sinks: causal at more than one row, banded to the
    /// window on a sliding layer's prefill, the window mask on its decode past the window.
    fn core(w: *const Weights, c: *Ctx, q: Array, k: Array, v: Array, total: c_int, kv_len: c_int) !Array {
        const g = c.g;
        const geo = &w.geo;
        const s = c.rows;
        const full = w.window == 0;
        const wide = s > 1;
        const sinks: ?Array = if (w.sinks.ctx != null) w.sinks else null;
        const fused: ?Array = if (wide and s >= attn256.FUSED256_MIN_Q_LEN)
            try da.own(g, try attn256.fusedSinkAttnPrefill(g.s, q, k, v, geo.scale, if (full) 0 else @intCast(w.window), sinks))
        else if (full and sinks == null)
            try da.own(g, try dec_attn.attention(g.s, q, k, v, geo.scale))
        else
            null;
        if (fused) |f| return f;
        if (full) return da.sdpaSinks(g, q, k, v, geo.scale, if (wide) "causal" else "", .{}, w.sinks, false);
        const sw: c_int = @intCast(w.window);
        if (wide and total <= sw) return da.sdpaSinks(g, q, k, v, geo.scale, "causal", .{}, w.sinks, false);
        if (wide) return da.sdpaSinks(g, q, k, v, geo.scale, "array", try sliding.bandMask(g, s, kv_len, sw), w.sinks, false);
        if (total <= sw) return da.sdpaSinks(g, q, k, v, geo.scale, "", .{}, w.sinks, false);
        return da.sdpaSinks(g, q, k, v, geo.scale, "array", try sliding.decodeMask(g, @min(total, sw), sw), w.sinks, false);
    }
};
