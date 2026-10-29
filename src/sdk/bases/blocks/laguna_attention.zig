//! Laguna's attention arm and its relatives (Flint, Kolibri): per-head q/k RMS norms before RoPE (one fused
//! dispatch at an hd-128 decode step), per-layer head counts and RoPE (YaRN with its mscale on full layers), sliding
//! layers reading the block-sized tail of the cache, and comptime extras: a softplus per-head output gate (Laguna),
//! learned per-head sinks (Flint), and a ring cache for sliding layers (`append_ring`, rows past the ring dropped).
//! Batched rows attend per slot. copied_from: plugins/laguna/attention.zig (`lagunaAttnWith`, no fused-QKV env arm,
//! no decode-attn-quant side copies), sdk/blocks/attention/sinks.zig (`sinkAttnSlots`, `sinkAttnCore`,
//! `sinkCacheUpdate`).

const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const kv = @import("../kv.zig");
const linear = @import("linear.zig");
const sliding = @import("sliding.zig");
const da = @import("decoder_attention.zig");
const qk_rope = @import("qk_rope.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");
const sdpa = @import("../../blocks/attention/sdpa.zig");
const kvq = @import("../../blocks/attention/kv_quant_attn.zig");
const gates = @import("../../blocks/attention/gates.zig");
const nax_attention = @import("../../blocks/attention/nax_attention.zig");
const qk_norm_rope = @import("../../blocks/rope/qk_norm_rope.zig");
const dec_attn = @import("../../blocks/attention/dec_attn.zig");
const DenseKVView = @import("../../blocks/attention/kv_view.zig").DenseKVView;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Linear = linear.Linear;
const Geo = da.Geo;

const perm = [_]c_int{ 0, 2, 1, 3 };
const MAX_SLOTS = @import("slot_reads.zig").MAX_SLOTS;

pub const Options = struct {
    /// `softplus(gate(x))` per head multiplies the attention output before o_proj.
    softplus_gate: bool = false,
    /// Learned per-head sinks ride every SDPA as an extra softmax column.
    sinks: bool = false,
};

pub fn LagunaAttention(comptime opts: Options) type {
    return struct {
        pub const kind: block.Kind = .mixer;
        pub const supports_batched_rows = true;
        pub const Weights = struct {
            q: Linear,
            k: Linear,
            v: Linear,
            o: Linear,
            q_norm: Array,
            k_norm: Array,
            gate: Linear = .{},
            /// `[heads]` in the activation dtype.
            sinks: Array = .{},
            /// `geo.freqs` set = a YaRN layer; `mscale` is its f32 `[head_dim]` attention-factor table.
            geo: Geo,
            mscale: Array = .{},
            /// Keys a sliding layer attends; 0 = a full layer.
            window: u32 = 0,
            /// Rows a sliding layer's cache keeps (>= the widest view); 0 = every row.
            ring: u32 = 0,
        };

        fn sinksOf(w: *const Weights) Array {
            return if (opts.sinks) w.sinks else .{};
        }

        pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
            const g = c.g;
            const geo = &w.geo;
            const n = c.slots;
            const s = c.rows;
            const qk = try roped(w, c, x);
            const v = try g.transpose(try g.reshape(try w.v.forward(g, x), &.{ n, s, geo.kv_heads, geo.head_dim }), &perm);
            const out = if (c.mode == .batched_rows) try perSlot(w, c, qk[0], qk[1], v) else try solo(w, c, qk[0], qk[1], v);
            const flat = try g.mergeHeads(out);
            if (!opts.softplus_gate or w.gate.w.ctx == null) return w.o.forward(g, flat);
            return w.o.forward(g, try softplusGate(w, g, x, flat));
        }

        /// q and k `[N, H, S, D]`, normed and roped: one fused dispatch at a solo hd-128 decode step, else the ops.
        fn roped(w: *const Weights, c: *Ctx, x: Array) ![2]Array {
            const g = c.g;
            const geo = &w.geo;
            const s = c.rows;
            const yarn = geo.freqs.ctx != null;
            const q_proj = try w.q.forward(g, x);
            if (c.mode != .batched_rows and s == 1 and geo.head_dim == 128 and qk_norm_rope.qkNormRopeFusedEnabled()) fused: {
                const k_proj = try w.k.forward(g, x);
                const ms: ?Array = if (yarn) try c.castOnce(if (w.mscale.ctx != null) w.mscale else break :fused, ops.dtype(q_proj)) else null;
                // A NoPE layer runs the norm alone at rd 0; q_norm stands in for the unread angles.
                const angles = if (geo.nope) w.q_norm else qk_rope.angles(c, null, 0, geo.rope_dims, ropeBase(geo), geo.freqs, @intCast(c.offset()), 1, 1.0, 1.0) catch break :fused;
                const pair = (qk_norm_rope.fusedQkNormRope(g.s, q_proj, k_proj, w.q_norm, w.k_norm, angles, if (geo.nope) null else ms, c.shared.eps_arr, geo.heads, geo.kv_heads, if (geo.nope) 0 else geo.rope_dims) catch null) orelse break :fused;
                const q = g.adopt(pair[0]) catch |e| {
                    _ = mlx.mlx_array_free(pair[1]);
                    return e;
                };
                return .{ q, try g.adopt(pair[1]) };
            }
            const k_proj = try w.k.forward(g, x);
            var q = try headsRoped(c, q_proj, w.q_norm, geo.heads, geo);
            var k = try headsRoped(c, k_proj, w.k_norm, geo.kv_heads, geo);
            // YaRN mscale on the rotated slice, cast to q/k's dtype so the product stays narrow.
            if (yarn and w.mscale.ctx != null) {
                const ms = try c.castOnce(w.mscale, ops.dtype(q));
                q = try g.mul(q, ms);
                k = try g.mul(k, ms);
            }
            return .{ q, k };
        }

        fn headsRoped(c: *Ctx, proj: Array, norm: Array, heads: c_int, geo: *const Geo) !Array {
            const g = c.g;
            const r = try g.reshape(proj, &.{ c.slots, c.rows, heads, geo.head_dim });
            const t = try g.transpose(try g.rmsNorm(r, norm, c.shared.eps), &perm);
            if (c.mode != .batched_rows) return da.ropeAt(g, t, geo, @intCast(c.offset()));
            var off_buf: [MAX_SLOTS]i32 = undefined;
            if (c.positions.len > MAX_SLOTS) return error.TooManySlots;
            for (c.positions, 0..) |p, i| off_buf[i] = @intCast(p);
            return da.ropeDyn(g, t, geo, try g.fromSlice(i32, off_buf[0..c.positions.len], &.{c.slots}));
        }

        /// The solo slot: the cache's (sliding) view, packed reads where they serve, the band kernel on a sinks
        /// layer's sliding prefill, else MLX's SDPA with the sinks.
        fn solo(w: *const Weights, c: *Ctx, q: Array, k: Array, v: Array) !Array {
            const g = c.g;
            const geo = &w.geo;
            const s = c.rows;
            const window = w.window;
            const total: c_int = @as(c_int, @intCast(c.offset())) + s;
            const sv = sliding.view(window, geo.head_dim, total, s);
            var view = try append(w, c, c.cache(), k, v, if (window > 0) sv.span else 0);
            defer kv.release(c, &view);
            const dv = kv.dense(view);
            const sinks = sinksOf(w);
            const w_c: c_int = @intCast(window);
            const short = window == 0 or total <= w_c;

            if (sinks.ctx == null and view.packed_reads and kvq.kvAttnFusedEligible(&dv, s)) {
                var mode: [:0]const u8 = "";
                var mask: Array = .{};
                if (c.wide() and short) {
                    mode = "causal";
                } else if (c.wide()) {
                    mode = "array";
                    mask = try sliding.bandOnce(c, s, sv.kv_len, w_c);
                }
                if (try da.own(g, try kvq.qkvAttnDecodeKernel(g.s, q, &dv, geo.scale, mode, mask))) |o| {
                    kvq.logKvAttnFusedEngaged(&dv, q, s);
                    return o;
                }
            }
            if (c.wide() and window > 0 and sinks.ctx != null) {
                if (try da.own(g, try nax_attention.attention(g.s, q, dv.k, dv.v, geo.scale, w_c, sinks))) |o| return o;
            }
            if (!c.wide()) return da.sdpaSinks(g, q, dv.k, dv.v, geo.scale, "", .{}, sinks, false);
            if (short) return da.sdpaSinks(g, q, dv.k, dv.v, geo.scale, "causal", .{}, sinks, sinks.ctx == null and sdpa.sdpaForceFused(q, dv.k));
            return da.sdpaSinks(g, q, dv.k, dv.v, geo.scale, "array", try sliding.bandOnce(c, s, sv.kv_len, w_c), sinks, false);
        }

        /// Each batched slot's row of q/k/v appended at the slot's own length and read through its own view.
        fn perSlot(w: *const Weights, c: *Ctx, q: Array, k: Array, v: Array) !Array {
            const g = c.g;
            const geo = &w.geo;
            const s = c.rows;
            if (c.caches.len > MAX_SLOTS) return error.TooManySlots;
            const o = try c.cacheOps();
            var outs: [MAX_SLOTS]Array = undefined;
            for (c.caches, 0..) |h, i| {
                const at: c_int = @intCast(i);
                const offset: c_int = @intCast(o.len(h, c.layer));
                const sv = sliding.view(w.window, geo.head_dim, offset + s, s);
                var view = try append(w, c, h, try g.sliceAxis(k, 0, at, at + 1), try g.sliceAxis(v, 0, at, at + 1), if (w.window > 0) sv.span else 0);
                defer kv.release(c, &view);
                outs[i] = try core(w, c, try g.sliceAxis(q, 0, at, at + 1), kv.dense(view), offset, sv.kv_len);
            }
            return g.concat(outs[0..c.caches.len], 0);
        }

        /// One slot's attention over its view, keys before this block numbering `offset`.
        fn core(w: *const Weights, c: *Ctx, q: Array, dv: DenseKVView, offset: c_int, kv_len: c_int) !Array {
            const g = c.g;
            const geo = &w.geo;
            const s = c.rows;
            const sinks = sinksOf(w);
            const full = w.window == 0;
            const prefill = s > 1;
            const fused: ?Array = if (prefill and s >= attn256.FUSED256_MIN_Q_LEN)
                try da.own(g, try attn256.fusedSinkAttnPrefill(g.s, q, dv.k, dv.v, geo.scale, @intCast(w.window), if (sinks.ctx != null) sinks else null))
            else if (full and sinks.ctx == null)
                try da.own(g, try dec_attn.attention(g.s, q, dv.k, dv.v, geo.scale))
            else
                null;
            if (fused) |f| return f;
            const w_c: c_int = @intCast(w.window);
            if (full or !prefill or offset + s <= w_c)
                return da.sdpaSinks(g, q, dv.k, dv.v, geo.scale, if (prefill) "causal" else "", .{}, sinks, false);
            return da.sdpaSinks(g, q, dv.k, dv.v, geo.scale, "array", try sliding.bandOnce(c, s, kv_len, w_c), sinks, false);
        }

        /// Appends to `h` at the current layer: a sliding layer with a ring keeps only its ring.
        fn append(w: *const Weights, c: *Ctx, h: abi.Handle, k: Array, v: Array, keep: u32) !abi.KvView {
            const o = try c.cacheOps();
            const s: abi.Stream = .{ .ctx = c.stream().ctx };
            var out: abi.KvView = .{};
            const verdict = if (keep > 0 and w.ring > 0)
                o.append_ring(h, c.layer, .{ .ctx = k.ctx }, .{ .ctx = v.ctx }, keep, w.window, w.ring, s, &out)
            else
                o.append(h, c.layer, .{ .ctx = k.ctx }, .{ .ctx = v.ctx }, keep, s, &out);
            if (!verdict.isOk()) return error.CacheOpFailed;
            return out;
        }

        /// `flat * softplus(gate(x))` per head: the f32 softplus rounded to bf16 before the multiply.
        fn softplusGate(w: *const Weights, parent: *Graph, x: Array, flat: Array) !Array {
            var sc = parent.scope();
            defer sc.deinit();
            const g = &sc;
            const geo = &w.geo;
            const logits = try w.gate.forward(g, x);
            const sp = try g.call(mlx.mlx_logaddexp, .{ try g.scalar(0, .float32), try g.astype(logits, .float32) });
            const sh = ops.shape(flat);
            const gate4 = try g.reshape(try g.astype(sp, .bfloat16), &.{ sh[0], sh[1], geo.heads, 1 });
            const attn4 = try g.reshape(flat, &.{ sh[0], sh[1], geo.heads, geo.head_dim });
            return parent.keep(try g.reshape(try g.mul(attn4, gate4), sh));
        }
    };
}

fn ropeBase(geo: *const Geo) mlx.mlx_optional_float {
    return .{ .value = geo.rope_base, .has_value = geo.freqs.ctx == null };
}
