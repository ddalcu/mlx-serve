//! Spark's attention: sliding layers over a block-sized tail under band masks, a per-head sigmoid output gate (joined
//! into q|k|v|g at one-row steps), and at an hd-256 decode step the one fused RoPE kernel (Spark carries no q/k norms).
//! Batched rows too.
//! copied_from: sdk/bases/blocks/decoder_attention.zig (`DecoderAttention(.{ .sliding, .gate = .head })`), plus
//! engines/mlx/forward/dense.zig's hd-256 `fusedQkNormRope256` decode arm, which that block does not carry.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const kv = @import("../../sdk/bases/kv.zig");
const da = @import("../../sdk/bases/blocks/decoder_attention.zig");
const qk_rope = @import("../../sdk/bases/blocks/qk_rope.zig");
const attn256 = @import("../../sdk/blocks/attention/attn256.zig");
const sdpa = @import("../../sdk/blocks/attention/sdpa.zig");
const kvq = @import("../../sdk/blocks/attention/kv_quant_attn.zig");
const batched = @import("../../sdk/blocks/attention/batched.zig");
const gates = @import("../../sdk/blocks/attention/gates.zig");
const qk_norm_rope = @import("../../sdk/blocks/rope/qk_norm_rope.zig");
const DenseKVView = @import("../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const FUSED_ROWS_MAX_M = @import("../../sdk/blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;

const b = bases.blocks;
const sliding = b.sliding;
const attention = b.attention;
const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = bases.Ctx;
const Linear = b.linear.Linear;
const Joined = b.linear.Joined;
pub const Geo = attention.Geo;

const perm = [_]c_int{ 0, 2, 1, 3 };
const slot_reads = @import("../../sdk/bases/blocks/slot_reads.zig");
const MAX_SLOTS = slot_reads.MAX_SLOTS;

pub const Attention = struct {
    pub const kind: bases.block.Kind = .mixer;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        /// q|k|v|g joined for one-row steps.
        qkv: Joined = .{},
        geo: Geo,
        /// Keys a sliding layer attends; 0 = every key (a full layer).
        window: u32 = 0,
        /// One logit per head (`g_proj`).
        out_gate: Linear,
        /// The model's rotary angle rows for the fused decode step, and this layer's family in them (1 = full).
        angles: ?*qk_rope.AngleCache = null,
        family: usize = 0,
    };

    fn ropeBase(geo: *const Geo) @import("../../mlx/ffi.zig").mlx_optional_float {
        return .{ .value = geo.rope_base, .has_value = geo.freqs.ctx == null };
    }

    /// q and k `[1, H, 1, 256]` roped by the one hd-256 kernel; null where it does not serve the step.
    fn fusedDecode(w: *const Weights, c: *Ctx, q_r: Array, own_k: Array) !?[2]Array {
        const g = c.g;
        const geo = &w.geo;
        if (geo.nope or c.rows != 1 or geo.head_dim != 256 or !qk_norm_rope.qkNormRopeFusedEnabled()) return null;
        const k_r4 = try g.reshape(own_k, &.{ 1, 1, geo.kv_heads, geo.head_dim });
        const offset: c_int = @intCast(c.offset());
        const ang = qk_rope.angles(c, w.angles, w.family, geo.rope_dims, ropeBase(geo), geo.freqs, offset, 1, 1.0, geo.rope_scale) catch return null;
        const f = (try qk_rope.fused256(g, q_r, k_r4, .{}, .{}, .{}, .{}, ang, c.shared.eps_arr, geo.heads, geo.kv_heads, 1, geo.rope_dims)) orelse return null;
        return .{ f[0], f[1] };
    }

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        if (c.mode == .batched_rows) return batchedForward(w, c, x);
        const g = c.g;
        const geo = &w.geo;
        const s = c.rows;
        const offset: c_int = @intCast(c.offset());

        const fused_rows = w.qkv.built() and s <= FUSED_ROWS_MAX_M;
        const all: Array = if (fused_rows) try w.qkv.lin.forward(g, x) else .{};
        const q = if (fused_rows) try w.qkv.part(g, all, 0) else try w.q.forward(g, x);
        const q_r = try g.reshape(q, &.{ 1, s, geo.heads, geo.head_dim });
        const own_k = if (fused_rows) try w.qkv.part(g, all, 1) else try w.k.forward(g, x);
        const own_v = if (fused_rows) try w.qkv.part(g, all, 2) else try w.v.forward(g, x);
        const fused = try fusedDecode(w, c, q_r, own_k);

        const q_rope = if (fused) |f| f[0] else try da.ropeAt(g, try g.transpose(q_r, &perm), geo, offset);
        const kv_shape = [_]c_int{ 1, s, geo.kv_heads, geo.head_dim };
        const k_rope = if (fused) |f| f[1] else try da.ropeAt(g, try g.transpose(try g.reshape(own_k, &kv_shape), &perm), geo, offset);
        const v_t = try g.transpose(try g.reshape(own_v, &kv_shape), &perm);

        const window = w.window;
        const total_kv = offset + s;
        const sv = sliding.view(window, geo.head_dim, total_kv, s);
        var view = try kv.append(c, k_rope, v_t, if (window > 0) sv.span else 0);
        defer kv.release(c, &view);
        const dv = kv.dense(view);

        // The mode and mask the dense arms and the packed kernels share.
        var mode: [:0]const u8 = if (c.wide()) "causal" else "";
        var mask: Array = .{};
        if (window > 0) {
            const w_c: c_int = @intCast(window);
            if (c.wide()) {
                mode = "array";
                if (!sv.band_in_kernel) mask = try sliding.bandOnce(c, s, sv.kv_len, w_c);
            } else {
                const held: c_int = @intCast((try c.cacheOps()).len(c.cache(), c.layer));
                if (held > w_c) {
                    mode = "array";
                    mask = try sliding.decodeOnce(c, @min(total_kv, w_c), w_c);
                }
            }
        }

        var out: ?Array = null;
        if (view.packed_reads and kvq.kvAttnFusedEligible(&dv, s)) {
            if (std.mem.eql(u8, mode, "array") and mask.ctx == null) mask = try sliding.bandOnce(c, s, sv.kv_len, @intCast(window));
            out = try da.own(g, try kvq.qkvAttnDecodeKernel(g.s, q_rope, &dv, geo.scale, mode, mask));
            if (out != null) kvq.logKvAttnFusedEngaged(&dv, q_rope, s);
        } else if (view.packed_reads and kvq.kvAttnVerifyEligible(&dv, s)) {
            out = try da.own(g, try kvq.qkvAttnVerifyKernel(g.s, q_rope, &dv, geo.scale, mode));
        }
        if (out == null) {
            if (std.mem.eql(u8, mode, "causal")) {
                out = try da.own(g, try attn256.fusedSdpa256Prefill(g.s, q_rope, dv.k, dv.v, geo.scale, 0));
                if (out == null) out = try da.own(g, try sdpa.splitCausalSdpa(g.s, q_rope, dv.k, dv.v, geo.scale));
                if (out == null) out = try da.sdpaRaw(g, q_rope, dv.k, dv.v, geo.scale, "causal", .{}, sdpa.sdpaForceFused(q_rope, dv.k));
            } else if (std.mem.eql(u8, mode, "array")) {
                if (c.wide()) out = try da.own(g, try attn256.fusedSdpa256Prefill(g.s, q_rope, dv.k, dv.v, geo.scale, @intCast(window)));
                if (out == null) {
                    if (mask.ctx == null) mask = try sliding.bandOnce(c, s, sv.kv_len, @intCast(window));
                    out = try da.sdpaRaw(g, q_rope, dv.k, dv.v, geo.scale, "array", mask, false);
                }
            } else {
                out = try da.sdpaRaw(g, q_rope, dv.k, dv.v, geo.scale, "", .{}, false);
            }
        }
        const flat = try g.mergeHeads(out.?);
        const g_pre: Array = if (fused_rows and w.qkv.widths[3] != 0) try w.qkv.part(g, all, 3) else .{};
        return w.o.forward(g, try outGate(w, g, x, flat, g_pre));
    }

    /// N one-row slots: projections at width N, each slot roped at its own position, keys appended per slot (a
    /// sliding layer's cache keeping its window), then a per-slot or a padded stacked read.
    fn batchedForward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const geo = &w.geo;
        const n = c.slots;
        const s = c.rows;
        if (n > MAX_SLOTS) return error.TooManySlots;
        var off_buf: [MAX_SLOTS]i32 = undefined;
        for (c.positions, 0..) |p, i| off_buf[i] = @intCast(p);
        const offsets = try g.fromSlice(i32, off_buf[0..c.positions.len], &.{n});

        const q_r = try g.reshape(try w.q.forward(g, x), &.{ n, s, geo.heads, geo.head_dim });
        const q_rope = try da.ropeDyn(g, try g.transpose(q_r, &perm), geo, offsets);
        const kv_shape = [_]c_int{ n, s, geo.kv_heads, geo.head_dim };
        const k_rope = try da.ropeDyn(g, try g.transpose(try g.reshape(try w.k.forward(g, x), &kv_shape), &perm), geo, offsets);
        const v_t = try g.transpose(try g.reshape(try w.v.forward(g, x), &kv_shape), &perm);
        try kv.appendRows(c, k_rope, v_t, w.window);

        var views: [MAX_SLOTS]abi.KvView = undefined;
        var dense: [MAX_SLOTS]DenseKVView = undefined;
        var packed_reads: [MAX_SLOTS]bool = undefined;
        var got: usize = 0;
        defer for (views[0..got]) |*v| kv.release(c, v);
        for (c.caches, 0..) |h, i| {
            views[i] = try kv.view(c, h, c.layer);
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

    /// `flat * sigmoid(gate(x))` per head: one exact dispatch where it serves, else the ops. `g_pre` is the gate
    /// projection when the joined rows made it.
    fn outGate(w: *const Weights, parent: *Graph, x: Array, flat: Array, g_pre: Array) !Array {
        var sc = parent.scope();
        defer sc.deinit();
        const g = &sc;
        const raw = if (g_pre.ctx != null) g_pre else try w.out_gate.forward(g, x);
        const heads = w.geo.heads;
        if (try gates.fusedSigmoidGateMul(g.s, flat, raw, heads)) |y| return parent.keep(try g.adopt(y));
        const sig = try g.sigmoid(raw);
        const sh = ops.shape(flat);
        const a4 = try g.reshape(flat, &.{ sh[0], sh[1], heads, @divExact(sh[2], heads) });
        const g4 = try g.reshape(sig, &.{ sh[0], sh[1], heads, 1 });
        return parent.keep(try g.reshape(try g.mul(a4, g4), sh));
    }
};
