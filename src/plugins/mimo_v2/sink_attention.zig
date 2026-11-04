//! MiMo's attention: plain GQA with additive q/k/v/o biases and no q/k norm, per-layer geometry (global layers 4 KV
//! heads at theta 1e7, sliding layers 8 at theta 1e4 under a 128-row band with learned sinks), K 192 / V 128 with
//! RoPE on the leading 64 channels, V scaled before it is cached, sliding layers caching a ring of their newest rows.
//! Decode rows and short verify windows read the requantized side copies of the projections; global layers decode
//! through the split-K kernel. Batched rows attend per slot.
//! copied_from: sdk/blocks/attention/sinks.zig (`sinkAttnWith`, `sinkAttnSlots`, `sinkAttnCore`, `sinkAttnOut`,
//! `mimoDecodeCopies`), engines/mlx/forward/layered.zig (the forward-level sliding masks).

const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const kv = @import("../../sdk/bases/kv.zig");
const sliding = @import("../../sdk/bases/blocks/sliding.zig");
const da = @import("../../sdk/bases/blocks/decoder_attention.zig");
const Proj = @import("../../sdk/bases/blocks/gated_attention.zig").Proj;
const attn256 = @import("../../sdk/blocks/attention/attn256.zig");
const dec_attn = @import("../../sdk/blocks/attention/dec_attn.zig");
const DenseKVView = @import("../../sdk/blocks/attention/kv_view.zig").DenseKVView;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = bases.Ctx;
const Geo = da.Geo;

const perm = [_]c_int{ 0, 2, 1, 3 };
const MAX_SLOTS = @import("../../sdk/bases/blocks/slot_reads.zig").MAX_SLOTS;

pub const SinkAttention = struct {
    pub const kind: bases.block.Kind = .mixer;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        q: Proj,
        k: Proj,
        v: Proj,
        o: Proj,
        /// `[heads]`; null = none (MiMo's global layers).
        sinks: Array = .{},
        geo: Geo,
        v_dim: c_int,
        /// V is multiplied by this (in V's dtype) before it is cached; 1 = none.
        value_scale: f32 = 1.0,
        /// Keys a sliding layer attends; 0 = a global layer.
        window: u32 = 0,
        /// Rows a sliding layer's cache keeps; 0 = every row.
        ring: u32 = 0,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const geo = &w.geo;
        const n = c.slots;
        const s = c.rows;
        // The side copies serve decode width and a speculative verify window; a prompt chunk prefills dense.
        const copies = n == 1 and (s == 1 or (c.mode == .verify and s <= dec_attn.MAX_ROWS));
        var q = try g.transpose(try g.reshape(try w.q.forward(g, x, copies), &.{ n, s, geo.heads, geo.head_dim }), &perm);
        var k = try g.transpose(try g.reshape(try w.k.forward(g, x, copies), &.{ n, s, geo.kv_heads, geo.head_dim }), &perm);
        if (c.mode == .batched_rows) {
            var off_buf: [MAX_SLOTS]i32 = undefined;
            if (c.positions.len > MAX_SLOTS) return error.TooManySlots;
            for (c.positions, 0..) |p, i| off_buf[i] = @intCast(p);
            const offs = try g.fromSlice(i32, off_buf[0..c.positions.len], &.{n});
            q = try da.ropeDyn(g, q, geo, offs);
            k = try da.ropeDyn(g, k, geo, offs);
        } else {
            q = try da.ropeAt(g, q, geo, @intCast(c.offset()));
            k = try da.ropeAt(g, k, geo, @intCast(c.offset()));
        }
        var v = try g.transpose(try g.reshape(try w.v.forward(g, x, copies), &.{ n, s, geo.kv_heads, w.v_dim }), &perm);
        if (w.value_scale != 1.0) v = try g.mul(v, try g.scalar(w.value_scale, ops.dtype(v)));
        const out = if (c.mode == .batched_rows) try perSlot(w, c, q, k, v) else try solo(w, c, q, k, v);
        const flat = try g.reshape(try g.transpose(out, &perm), &.{ n, s, geo.heads * w.v_dim });
        return w.o.forward(g, flat, copies);
    }

    /// The solo slot: its (sliding) view, the band over a sliding prefill and the decode mask past the window.
    fn solo(w: *const Weights, c: *Ctx, q: Array, k: Array, v: Array) !Array {
        const s = c.rows;
        const offset: c_int = @intCast(c.offset());
        const total = offset + s;
        const sv = sliding.view(w.window, w.geo.head_dim, total, s);
        var view = try append(w, c, c.cache(), k, v, if (w.window > 0) sv.span else 0);
        defer kv.release(c, &view);
        const wc: c_int = @intCast(w.window);
        const decode_mask: Array = if (w.window > 0 and s == 1 and total > wc) try sliding.decodeOnce(c, @min(total, wc), wc) else .{};
        return core(w, c, q, kv.dense(view), offset, sv.kv_len, decode_mask);
    }

    /// Each batched slot's row of q/k/v appended at the slot's own length and read through its own view.
    fn perSlot(w: *const Weights, c: *Ctx, q: Array, k: Array, v: Array) !Array {
        const g = c.g;
        const s = c.rows;
        if (c.caches.len > MAX_SLOTS) return error.TooManySlots;
        const o = try c.cacheOps();
        var outs: [MAX_SLOTS]Array = undefined;
        for (c.caches, 0..) |h, i| {
            const at: c_int = @intCast(i);
            const offset: c_int = @intCast(o.len(h, c.layer));
            const sv = sliding.view(w.window, w.geo.head_dim, offset + s, s);
            var view = try append(w, c, h, try g.sliceAxis(k, 0, at, at + 1), try g.sliceAxis(v, 0, at, at + 1), if (w.window > 0) sv.span else 0);
            defer kv.release(c, &view);
            // A slot's decode view is exactly its window: no decode mask.
            outs[i] = try core(w, c, try g.sliceAxis(q, 0, at, at + 1), kv.dense(view), offset, sv.kv_len, .{});
        }
        return g.concat(outs[0..c.caches.len], 0);
    }

    /// One slot's attention over its view (keys before this block numbering `offset`): the fused prefill kernels
    /// where they serve, the split-K decode kernel on a global layer, else MLX's SDPA with the sinks.
    fn core(w: *const Weights, c: *Ctx, q: Array, dv: DenseKVView, offset: c_int, kv_len: c_int, decode_mask: Array) !Array {
        const g = c.g;
        const s = c.rows;
        const scale = w.geo.scale;
        const sinks = w.sinks;
        const full = w.window == 0;
        const prefill = s > 1;
        const fused: ?Array = if (prefill and s >= attn256.FUSED256_MIN_Q_LEN)
            try da.own(g, try attn256.fusedSinkAttnPrefill(g.s, q, dv.k, dv.v, scale, @intCast(w.window), if (sinks.ctx != null) sinks else null))
        else if (full and sinks.ctx == null)
            try da.own(g, try dec_attn.attention(g.s, q, dv.k, dv.v, scale))
        else
            null;
        if (fused) |f| return f;
        if (full) return da.sdpaSinks(g, q, dv.k, dv.v, scale, if (prefill) "causal" else "", .{}, sinks, false);
        const wc: c_int = @intCast(w.window);
        const total = offset + s;
        if (prefill and total <= wc) return da.sdpaSinks(g, q, dv.k, dv.v, scale, "causal", .{}, sinks, false);
        if (prefill) return da.sdpaSinks(g, q, dv.k, dv.v, scale, "array", try sliding.bandOnce(c, s, kv_len, wc), sinks, false);
        if (total <= wc or decode_mask.ctx == null) return da.sdpaSinks(g, q, dv.k, dv.v, scale, "", .{}, sinks, false);
        return da.sdpaSinks(g, q, dv.k, dv.v, scale, "array", decode_mask, sinks, false);
    }

    /// Appends to `h` at the current layer: a sliding layer keeps only its ring.
    fn append(w: *const Weights, c: *Ctx, h: abi.Handle, k: Array, v: Array, keep: u32) !abi.KvView {
        const o = try c.cacheOps();
        const st: abi.Stream = .{ .ctx = c.stream().ctx };
        var out: abi.KvView = .{};
        const verdict = if (keep > 0 and w.ring > 0)
            o.append_ring(h, c.layer, .{ .ctx = k.ctx }, .{ .ctx = v.ctx }, keep, w.window, w.ring, st, &out)
        else
            o.append(h, c.layer, .{ .ctx = k.ctx }, .{ .ctx = v.ctx }, keep, st, &out);
        if (!verdict.isOk()) return error.CacheOpFailed;
        return out;
    }
};
