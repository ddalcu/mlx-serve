//! The dense decoder's attention with its comptime extras (Gemma 3/4, Muse, Spark, and with none of them llama/qwen3):
//! sliding layers reading a block-sized tail of the cache under band masks, layers reading an earlier layer's KV,
//! a parameter-free value norm, K=V layers, and a sigmoid output gate per channel or per head. Batched rows too.
//! copied_from: engines/mlx/forward/dense.zig (`forwardStandardWith`'s attention),
//! engines/mlx/batched/dense.zig (`forwardBatchedDecode`'s attention), sdk/blocks/attention/gates.zig
//! (`attnOutGate`).

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const kv = @import("../kv.zig");
const linear = @import("linear.zig");
const slot_reads = @import("slot_reads.zig");
const qk_rope = @import("qk_rope.zig");
const sliding = @import("sliding.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");
const sdpa = @import("../../blocks/attention/sdpa.zig");
const kvq = @import("../../blocks/attention/kv_quant_attn.zig");
const batched = @import("../../blocks/attention/batched.zig");
const gates = @import("../../blocks/attention/gates.zig");
const qk_norm_rope = @import("../../blocks/rope/qk_norm_rope.zig");
const DenseKVView = @import("../../blocks/attention/kv_view.zig").DenseKVView;
const FUSED_ROWS_MAX_M = @import("../../blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const rope_scaling = @import("../../config/rope_scaling.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Linear = linear.Linear;
const Joined = linear.Joined;

const perm = [_]c_int{ 0, 2, 1, 3 };
const MAX_SLOTS = slot_reads.MAX_SLOTS;

/// One head geometry and its rotary embedding.
pub const Geo = struct {
    heads: c_int,
    kv_heads: c_int,
    head_dim: c_int,
    scale: f32,
    rope_dims: c_int,
    rope_base: f32,
    rope_scale: f32,
    /// No rotary position on this layer (NoPE).
    nope: bool = false,
    /// A frequency table (llama3 / proportional RoPE) in place of `rope_base`; null = none.
    freqs: Array = .{},
    /// YaRN's factor on q and k after the rotation (`rope_dims == head_dim`); 1 = none.
    mscale: f32 = 1.0,
    /// Dynamic NTK: past the window the base grows with the sequence; null = the static base.
    ntk: ?rope_scaling.Ntk = null,
};

pub const Gate = enum { none, channel, head };

pub const Options = struct {
    /// Layers with a `window` attend only their last `window` keys.
    sliding: bool = false,
    /// Layers with a `kv_source` read that layer's KV and project none of their own.
    kv_sharing: bool = false,
    /// A parameter-free RMS norm on values (`v_norm`, ones) and K=V layers (`k_eq_v`).
    value_norm: bool = false,
    /// `sigmoid(gate(x))` multiplies the attention output before o_proj.
    gate: Gate = .none,
};

/// Whether a step reads its keys through the sliding band: a sliding layer, unless `fit_causal` and a prefill whose
/// keys all fit the window (the band is then plain causal).
pub fn banded(window: u32, fit_causal: bool, wide: bool, total_kv: c_int) bool {
    return window > 0 and !(fit_causal and wide and total_kv <= window);
}

test "decoder attention: a fitting prefill reads causal only where the layer asks for it" {
    try std.testing.expect(!banded(0, false, true, 32));
    try std.testing.expect(banded(1024, false, true, 32));
    try std.testing.expect(!banded(1024, true, true, 32));
    try std.testing.expect(!banded(1024, true, true, 1024));
    try std.testing.expect(banded(1024, true, true, 1025));
    try std.testing.expect(banded(1024, true, false, 32));
}

pub fn DecoderAttention(comptime opts: Options) type {
    return struct {
        pub const kind: block.Kind = .mixer;
        pub const supports_batched_rows = true;
        pub const Weights = struct {
            q: Linear,
            k: Linear = .{},
            v: Linear = .{},
            o: Linear,
            q_norm: Array = .{},
            k_norm: Array = .{},
            /// q|k|v (| gate) joined for one-row steps.
            qkv: Joined = .{},
            geo: Geo,
            /// Keys a sliding layer attends; 0 = every key (a global layer).
            window: u32 = 0,
            kv_source: ?u32 = null,
            /// A sliding layer whose keys all fit its window prefills as plain causal: the kernel and reduction order
            /// mlx-lm takes for short prompts, which Gemma 4's MoE forward keeps so near-tie routes do not flip.
            fit_causal: bool = false,
            v_norm: Array = .{},
            k_eq_v: bool = false,
            out_gate: Linear = .{},
            /// The model's angle rows for the fused hd-256 decode step; null = built per call.
            angles: ?*qk_rope.AngleCache = null,
        };

        fn shared(w: *const Weights) ?u32 {
            return if (opts.kv_sharing) w.kv_source else null;
        }
        fn windowOf(w: *const Weights) u32 {
            return if (opts.sliding) w.window else 0;
        }

        /// k and v `[N, kv_heads, S, D]` before the cache: normed, roped (k) by `ropeFn`.
        fn ownKv(w: *const Weights, c: *Ctx, n: c_int, own_k: Array, own_v: Array, roped: anytype) ![2]Array {
            const g = c.g;
            const geo = &w.geo;
            const kv_shape = [_]c_int{ n, c.rows, geo.kv_heads, geo.head_dim };
            const k_r = try g.reshape(own_k, &kv_shape);
            const v_r = try g.reshape(own_v, &kv_shape);
            const k_n = if (w.k_norm.ctx != null) try g.rmsNorm(k_r, w.k_norm, c.shared.eps) else k_r;
            const v_n = if (opts.value_norm and w.v_norm.ctx != null) try g.rmsNorm(v_r, w.v_norm, c.shared.eps) else v_r;
            return .{ try roped.apply(g, try g.transpose(k_n, &perm)), try g.transpose(v_n, &perm) };
        }

        pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
            if (c.mode == .batched_rows) return batchedForward(w, c, x);
            if (c.caches.len == 0) return embedBatchForward(w, c, x);
            const g = c.g;
            const geo = &w.geo;
            const s = c.rows;
            const offset: c_int = @intCast(c.offset());
            const src = shared(w);
            const kev = opts.value_norm and w.k_eq_v;

            const fused_rows = w.qkv.built() and s <= FUSED_ROWS_MAX_M;
            const all: Array = if (fused_rows) try w.qkv.lin.forward(g, x) else .{};
            const q = if (fused_rows) try w.qkv.part(g, all, 0) else try w.q.forward(g, x);
            const q_r = try g.reshape(q, &.{ 1, s, geo.heads, geo.head_dim });
            var own_k: Array = .{};
            var own_v: Array = .{};
            if (src == null) {
                own_k = if (fused_rows) try w.qkv.part(g, all, 1) else try w.k.forward(g, x);
                own_v = if (kev) own_k else if (fused_rows) try w.qkv.part(g, all, 2) else try w.v.forward(g, x);
            }
            const window = windowOf(w);
            // A decode step at hd 256: one kernel norms, transposes and ropes q and k (and norms v).
            var fused: ?[3]Array = null;
            const has_vn = opts.value_norm and w.v_norm.ctx != null;
            const kv_shape = [_]c_int{ 1, s, geo.kv_heads, geo.head_dim };
            if (src == null and !kev and !geo.nope and geo.mscale == 1.0 and geo.ntk == null and s == 1 and geo.head_dim == 256 and (w.q_norm.ctx != null) == (w.k_norm.ctx != null) and
                (w.q_norm.ctx != null or !has_vn) and qk_norm_rope.qkNormRopeFusedEnabled())
            blk: {
                const family: usize = if (window > 0) 0 else 1;
                const ang = qk_rope.angles(c, w.angles, family, geo.rope_dims, base(geo), geo.freqs, offset, 1, 1.0, geo.rope_scale) catch break :blk;
                const v_r: Array = if (has_vn) try g.reshape(own_v, &kv_shape) else .{};
                const v_norm: Array = if (has_vn) w.v_norm else .{};
                fused = try qk_rope.fused256(g, q_r, try g.reshape(own_k, &kv_shape), v_r, w.q_norm, w.k_norm, v_norm, ang, c.shared.eps_arr, geo.heads, geo.kv_heads, 1, geo.rope_dims);
            }

            const q_rope = if (fused) |f| f[0] else blk: {
                const q_n = if (w.q_norm.ctx != null) try g.rmsNorm(q_r, w.q_norm, c.shared.eps) else q_r;
                break :blk try ropeAt(g, try g.transpose(q_n, &perm), geo, offset);
            };

            const total_kv = offset + s;
            const sv = sliding.view(window, geo.head_dim, total_kv, s);
            var view: abi.KvView = if (src) |l|
                try kv.view(c, c.cache(), l)
            else blk: {
                const kvs = if (fused) |f|
                    [2]Array{ f[1], if (f[2].ctx != null) f[2] else try g.transpose(try g.reshape(own_v, &kv_shape), &perm) }
                else
                    try ownKv(w, c, 1, own_k, own_v, Roper{ .geo = geo, .offset = offset });
                break :blk try kv.append(c, kvs[0], kvs[1], if (window > 0) sv.span else 0);
            };
            defer kv.release(c, &view);
            const dv = kv.dense(view);

            // The mode and mask the dense arms and the packed kernels share.
            var mode: [:0]const u8 = if (c.wide()) "causal" else "";
            var mask: Array = .{};
            if (banded(window, w.fit_causal, c.wide(), total_kv)) {
                const w_c: c_int = @intCast(window);
                if (c.wide()) {
                    mode = "array";
                    if (!sv.band_in_kernel) mask = try sliding.bandOnce(c, s, sv.kv_len, w_c);
                } else {
                    const held: c_int = @intCast((try c.cacheOps()).len(c.cache(), src orelse c.layer));
                    if (held > w_c) {
                        mode = "array";
                        mask = try sliding.decodeOnce(c, @min(total_kv, w_c), w_c);
                    }
                }
            }

            // A caller's additive term (a left-padded prompt) reaches every layer's scores as an explicit array, the band
            // summed in on a sliding layer; the fused kernels mask from geometry alone and never see it.
            const extra = c.wide() and c.mask_add.ctx != null;
            if (extra) {
                mode = "array";
                mask = if (window > 0) try g.add(try sliding.bandOnce(c, s, sv.kv_len, @intCast(window)), c.mask_add) else c.mask_add;
            }

            var out: ?Array = null;
            if (extra) {} else if (view.packed_reads and kvq.kvAttnFusedEligible(&dv, s)) {
                if (std.mem.eql(u8, mode, "array") and mask.ctx == null) mask = try sliding.bandOnce(c, s, sv.kv_len, @intCast(window));
                out = try own(g, try kvq.qkvAttnDecodeKernel(g.s, q_rope, &dv, geo.scale, mode, mask));
                if (out != null) kvq.logKvAttnFusedEngaged(&dv, q_rope, s);
            } else if (view.packed_reads and kvq.kvAttnVerifyEligible(&dv, s)) {
                out = try own(g, try kvq.qkvAttnVerifyKernel(g.s, q_rope, &dv, geo.scale, mode));
            }
            if (out == null) {
                if (std.mem.eql(u8, mode, "causal")) {
                    out = try own(g, try attn256.fusedSdpa256Prefill(g.s, q_rope, dv.k, dv.v, geo.scale, 0));
                    if (out == null) out = try own(g, try sdpa.splitCausalSdpa(g.s, q_rope, dv.k, dv.v, geo.scale));
                    if (out == null) out = try sdpaRaw(g, q_rope, dv.k, dv.v, geo.scale, "causal", .{}, sdpa.sdpaForceFused(q_rope, dv.k));
                } else if (std.mem.eql(u8, mode, "array")) {
                    if (c.wide() and !extra) out = try own(g, try attn256.fusedSdpa256Prefill(g.s, q_rope, dv.k, dv.v, geo.scale, @intCast(window)));
                    if (out == null) {
                        if (mask.ctx == null) mask = try sliding.bandOnce(c, s, sv.kv_len, @intCast(window));
                        out = try sdpaRaw(g, q_rope, dv.k, dv.v, geo.scale, "array", mask, false);
                    }
                } else {
                    out = try sdpaRaw(g, q_rope, dv.k, dv.v, geo.scale, "", .{}, false);
                }
            }
            const flat = try g.mergeHeads(out.?);
            const g_pre: Array = if (opts.gate != .none and fused_rows and w.qkv.widths[3] != 0) try w.qkv.part(g, all, 3) else .{};
            return w.o.forward(g, try outGate(w, g, x, flat, g_pre));
        }

        /// A decoder embedder's padded batch `[N, S]` (no cache, every sequence from position 0): causal over its own
        /// rows, as the dense forward ran it through a fresh cache. Padding is on the right, behind every real row's
        /// causal reach, so the key mask is not read.
        fn embedBatchForward(w: *const Weights, c: *Ctx, x: Array) !Array {
            if (shared(w) != null or windowOf(w) > 0 or (opts.value_norm and w.k_eq_v)) return error.EmbedBatchUnsupported;
            const g = c.g;
            const geo = &w.geo;
            const n = c.slots;
            const s = c.rows;
            const fused_rows = w.qkv.built() and n * s <= FUSED_ROWS_MAX_M;
            const all: Array = if (fused_rows) try w.qkv.lin.forward(g, x) else .{};
            const q = if (fused_rows) try w.qkv.part(g, all, 0) else try w.q.forward(g, x);
            const own_k = if (fused_rows) try w.qkv.part(g, all, 1) else try w.k.forward(g, x);
            const own_v = if (fused_rows) try w.qkv.part(g, all, 2) else try w.v.forward(g, x);
            const q_r = try g.reshape(q, &.{ n, s, geo.heads, geo.head_dim });
            const q_n = if (w.q_norm.ctx != null) try g.rmsNorm(q_r, w.q_norm, c.shared.eps) else q_r;
            const q_rope = try ropeAt(g, try g.transpose(q_n, &perm), geo, 0);
            const kvs = try ownKv(w, c, n, own_k, own_v, Roper{ .geo = geo, .offset = 0 });
            var out: ?Array = if (s == 1) try sdpaRaw(g, q_rope, kvs[0], kvs[1], geo.scale, "", .{}, false) else null;
            if (out == null) out = try own(g, try attn256.fusedSdpa256Prefill(g.s, q_rope, kvs[0], kvs[1], geo.scale, 0));
            if (out == null) out = try own(g, try sdpa.splitCausalSdpa(g.s, q_rope, kvs[0], kvs[1], geo.scale));
            if (out == null) out = try sdpaRaw(g, q_rope, kvs[0], kvs[1], geo.scale, "causal", .{}, sdpa.sdpaForceFused(q_rope, kvs[0]));
            const g_pre: Array = if (opts.gate != .none and fused_rows and w.qkv.widths[3] != 0) try w.qkv.part(g, all, 3) else .{};
            return w.o.forward(g, try outGate(w, g, x, try g.mergeHeads(out.?), g_pre));
        }

        /// N one-row slots: projections at width N, each slot roped at its own position, keys appended per slot
        /// (a sliding layer's cache keeping its window), then a per-slot or a padded stacked read.
        fn batchedForward(w: *const Weights, c: *Ctx, x: Array) !Array {
            const g = c.g;
            const geo = &w.geo;
            const n = c.slots;
            const s = c.rows;
            if (n > MAX_SLOTS) return error.TooManySlots;
            var off_buf: [MAX_SLOTS]i32 = undefined;
            for (c.positions, 0..) |p, i| off_buf[i] = @intCast(p);
            const offsets = try g.fromSlice(i32, off_buf[0..c.positions.len], &.{n});
            const src = shared(w);

            const q_r = try g.reshape(try w.q.forward(g, x), &.{ n, s, geo.heads, geo.head_dim });
            const q_n = if (w.q_norm.ctx != null) try g.rmsNorm(q_r, w.q_norm, c.shared.eps) else q_r;
            const q_rope = try ropeDyn(g, try g.transpose(q_n, &perm), geo, offsets);

            if (src == null) {
                const own_k = try w.k.forward(g, x);
                const own_v = if (opts.value_norm and w.k_eq_v) own_k else try w.v.forward(g, x);
                const kvs = try ownKv(w, c, n, own_k, own_v, DynRoper{ .geo = geo, .offsets = offsets });
                try kv.appendRows(c, kvs[0], kvs[1], windowOf(w));
            }

            var views: [MAX_SLOTS]abi.KvView = undefined;
            var dense: [MAX_SLOTS]DenseKVView = undefined;
            var packed_reads: [MAX_SLOTS]bool = undefined;
            var got: usize = 0;
            defer for (views[0..got]) |*v| kv.release(c, v);
            for (c.caches, 0..) |h, i| {
                views[i] = try kv.view(c, h, src orelse c.layer);
                got = i + 1;
                dense[i] = kv.dense(views[i]);
                packed_reads[i] = views[i].packed_reads;
            }
            const attn = if (batched.batchedAttnPerSlot(dense[0..got]))
                try slot_reads.perSlot(g, q_rope, dense[0..got], packed_reads[0..got], s, geo.scale)
            else
                try slot_reads.stacked(g, q_rope, dense[0..got], s, geo.scale);
            return w.o.forward(g, try outGate(w, g, x, try g.mergeHeads(attn), .{}));
        }

        /// The output gate over `flat` `[N, S, heads * head_dim]`: one exact dispatch where it serves, else
        /// sigmoid then a per-channel or per-head multiply. `g_pre` is the gate projection when the joined rows made it.
        fn outGate(w: *const Weights, parent: *Graph, x: Array, flat: Array, g_pre: Array) !Array {
            if (opts.gate == .none) return flat;
            var sc = parent.scope();
            defer sc.deinit();
            const g = &sc;
            const raw = if (g_pre.ctx != null) g_pre else try w.out_gate.forward(g, x);
            const heads: c_int = if (opts.gate == .head) w.geo.heads else 0;
            if (try gates.fusedSigmoidGateMul(g.s, flat, raw, heads)) |y| return parent.keep(try g.adopt(y));
            const sig = try g.sigmoid(raw);
            if (opts.gate == .channel) return parent.keep(try g.mul(flat, sig));
            const sh = ops.shape(flat);
            const a4 = try g.reshape(flat, &.{ sh[0], sh[1], heads, @divExact(sh[2], heads) });
            const g4 = try g.reshape(sig, &.{ sh[0], sh[1], heads, 1 });
            return parent.keep(try g.reshape(try g.mul(a4, g4), sh));
        }
    };
}

const Roper = struct {
    geo: *const Geo,
    offset: c_int,
    fn apply(r: Roper, g: *Graph, x: Array) !Array {
        return ropeAt(g, x, r.geo, r.offset);
    }
};

const DynRoper = struct {
    geo: *const Geo,
    offsets: Array,
    fn apply(r: DynRoper, g: *Graph, x: Array) !Array {
        return ropeDyn(g, x, r.geo, r.offsets);
    }
};

fn base(geo: *const Geo) mlx.mlx_optional_float {
    return .{ .value = geo.rope_base, .has_value = geo.freqs.ctx == null };
}

/// `x` `[B, H, S, D]` roped from `offset` (dynamic NTK's base for the positions it reaches).
pub fn ropeAt(g: *Graph, x: Array, geo: *const Geo, offset: c_int) !Array {
    if (geo.nope) return x;
    var b = base(geo);
    if (geo.ntk) |n| b.value = @floatCast(n.base(geo.rope_base, @intCast(geo.rope_dims), @intCast(offset + ops.shape(x)[2])));
    const r = try g.call(mlx.mlx_fast_rope, .{ x, geo.rope_dims, false, b, geo.rope_scale, offset, geo.freqs });
    return if (geo.mscale == 1.0) r else g.mulScalar(r, geo.mscale);
}

/// Each slot's rows roped from its own offset (a dynamic NTK model never batches).
pub fn ropeDyn(g: *Graph, x: Array, geo: *const Geo, offsets: Array) !Array {
    if (geo.nope) return x;
    const r = try g.call(mlx.mlx_fast_rope_dynamic, .{ x, geo.rope_dims, false, base(geo), geo.rope_scale, offsets, geo.freqs });
    return if (geo.mscale == 1.0) r else g.mulScalar(r, geo.mscale);
}

pub fn sdpaRaw(g: *Graph, q: Array, k: Array, v: Array, scale: f32, mode: [*:0]const u8, mask: Array, force: bool) !Array {
    return sdpaSinks(g, q, k, v, scale, mode, mask, .{}, force);
}

pub fn sdpaSinks(g: *Graph, q: Array, k: Array, v: Array, scale: f32, mode: [*:0]const u8, mask: Array, sinks: Array, force: bool) !Array {
    return g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, k, v, scale, mode, mask, sinks, force });
}

pub fn own(g: *Graph, x: ?Array) !?Array {
    return if (x) |a| try g.adopt(a) else null;
}

// Bar: each scaling mlx-lm's `initialize_rope` builds ropes q as mlx-lm does (angle `p * scale / freqs[i]` over the
// scaled spectrum or dynamic NTK's base, NeoX halves, YaRN's factor after), against the host in f64, on both
// attention blocks.
test "decoder and gated attention: each rope scaling ropes as mlx-lm's initialize_rope does" {
    const scaled = @import("../../blocks/rope/scaled.zig");
    const gated = @import("gated_attention.zig");
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const D = 64;
    const S = 5;
    const base_theta: f32 = 10000;
    const max_pos: u32 = 4096;
    const cases = [_]rope_scaling.Scaling{
        .none,
        .{ .linear = 4 },
        .{ .dynamic = 2 },
        .{ .yarn = .{ .factor = 4, .original_max_position_embeddings = 1024 } },
        .{ .llama3 = .{ .factor = 8, .original_max_position_embeddings = 2048 } },
    };
    var prng = std.Random.DefaultPrng.init(0x5ca1ed);
    var xs: [S * D]f32 = undefined;
    for (&xs) |*v| v.* = prng.random().floatNorm(f32);
    for (cases) |sc| for ([_]c_int{ 3, 5000 }) |off| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const x = try g.adopt(mlx.mlx_array_new_data(&xs, &[_]c_int{ 1, 1, S, D }, 4, .float32));
        const freqs: Array = if (try scaled.freqs(sc, D, base_theta)) |f| try g.adopt(f) else .{};
        const ms: f32 = @floatCast(rope_scaling.mscale(sc));
        const ntk = rope_scaling.ntk(sc, max_pos);
        const geo: Geo = .{ .heads = 1, .kv_heads = 1, .head_dim = D, .scale = 1, .rope_dims = D, .rope_base = base_theta, .rope_scale = rope_scaling.positionScale(sc), .freqs = freqs, .mscale = ms, .ntk = ntk };
        const table: Array = if (ms == 1.0) .{} else try g.adopt(try scaled.mscaleTable(ms, D, D, .float32, s));
        const ggeo: gated.Geo = .{ .heads = 1, .kv_heads = 1, .head_dim = D, .scale = 1, .rope_dims = D, .rope_base = base_theta, .yarn_freqs = freqs, .yarn_mscale = table, .yarn_factor = ms, .rope_scale = rope_scaling.positionScale(sc), .ntk = ntk };
        // The host reference.
        var inv: [D / 2]f64 = undefined;
        var f64s: [D / 2]f64 = undefined;
        if (rope_scaling.freqs(sc, &f64s, D, base_theta)) {
            for (&inv, f64s) |*o, f| o.* = 1.0 / @as(f64, @as(f32, @floatCast(f)));
        } else {
            const b: f64 = if (ntk) |n| n.base(base_theta, D, @intCast(off + S)) else base_theta;
            for (&inv, 0..) |*o, i| o.* = 1.0 / std.math.pow(f64, b, @as(f64, @floatFromInt(2 * i)) / D);
        }
        var want: [S * D]f64 = undefined;
        for (0..S) |t| for (0..D / 2) |i| {
            const p: f64 = @floatFromInt(@as(i64, off) + @as(i64, @intCast(t)));
            const ang = p * rope_scaling.positionScale(sc) * inv[i];
            const x1: f64 = xs[t * D + i];
            const x2: f64 = xs[t * D + i + D / 2];
            want[t * D + i] = (x1 * @cos(ang) - x2 * @sin(ang)) * ms;
            want[t * D + i + D / 2] = (x2 * @cos(ang) + x1 * @sin(ang)) * ms;
        };
        for ([_]Array{ try ropeAt(&g, x, &geo, off), try gated.ropeAt(&g, x, &ggeo, off) }) |got| {
            const c = try g.astype(got, .float32);
            try mlx.check(mlx.mlx_array_eval(c));
            const d = mlx.mlx_array_data_float32(c).?;
            for (want, d[0 .. S * D]) |w, v| try std.testing.expectApproxEqAbs(w, @as(f64, v), 2e-3);
        }
    };
}
