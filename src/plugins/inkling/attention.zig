//! Inkling's attention block and its helpers (the short conv, the relative bias, the mask, the log-scaling), with the
//! dense MLP and the vocab-trimming head. Reference: inkling_mlx/attention.py.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const testing = std.testing;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const jsonF64 = @import("../../sdk/testing/parity.zig").jsonF64;
const jsonToMlxArray = @import("../../sdk/testing/parity.zig").jsonToMlxArray;
const maxAbsDiffF32 = @import("../../sdk/testing/parity.zig").maxAbsDiffF32;
const inklingRouterChain = @import("moe.zig").inklingRouterChain;

// ── Inkling (inkling_mm_model) forward helpers ──
// Reference: the checkpoint's bundled inkling_mlx/ package (Apache-2.0,
// parity-validated vs transformers). Dtype discipline mirrors the
// reference EXACTLY: the residual stream runs in the checkpoint's dtype (bf16
// on the shipped builds — the dense mlp.global_scale tensors are STORED bf16,
// so no promotion), while the short convs, the router chain and the rel-bias
// build compute in f32 and the shared-expert sum is f32-then-cast.

const InklingSconvOut = struct { out: mlx.mlx_array, new_state: mlx.mlx_array };

/// Depthwise causal short-convolution with residual add, computed in fp32
/// (`InklingShortConvolution`). `prev_state` = the last kernel-1 f32 inputs
/// (null ⇒ zeros — sequence start). Returns the output in x's dtype and the
/// new f32 state; caller owns both.
pub fn inklingSconvCore(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, prev_state: ?mlx.mlx_array) !InklingSconvOut {
    const in_dtype = mlx.mlx_array_dtype(x);
    const x_shape = mlx.getShape(x); // [B, S, C]
    const w_shape = mlx.getShape(w); // [C, K, 1]
    const kernel = w_shape[1];
    const channels = x_shape[2];

    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    try mlx.check(mlx.mlx_astype(&w32, w, .float32, s));

    var left = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(left);
    if (prev_state) |st| {
        try mlx.check(mlx.mlx_astype(&left, st, .float32, s));
    } else {
        const zshape = [_]c_int{ x_shape[0], kernel - 1, channels };
        try mlx.check(mlx.mlx_zeros(&left, &zshape, 3, .float32, s));
    }

    var conv_in = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_in);
    {
        const arr = [_]mlx.mlx_array{ left, x32 };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&conv_in, vec, 1, s));
    }

    // "valid" conv over [left, x] yields exactly S causal outputs.
    var conv_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_out);
    try mlx.check(mlx.mlx_conv1d(&conv_out, conv_in, w32, 1, 0, 1, channels, s));

    var out32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out32);
    try mlx.check(mlx.mlx_add(&out32, conv_out, x32, s));

    var new_state = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(new_state);
    {
        const start = [_]c_int{ 0, x_shape[1], 0 }; // last kernel-1 rows of [K-1+S]
        const stop = [_]c_int{ x_shape[0], x_shape[1] + kernel - 1, channels };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&new_state, conv_in, &start, 3, &stop, 3, &strides, 3, s));
    }

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, out32, in_dtype, s));
    return .{ .out = out, .new_state = new_state };
}

/// Hidden-state-conditioned relative position bias (`InklingRelativeLogits`):
/// rel_states [B, Lq, H, d_rel] × proj [d_rel, extent] → per-distance bias,
/// gathered per (q, kv) pair as bias[b,h,i,j] = profile[clip(qpos_i - kvpos_j)]
/// and zeroed outside 0 <= distance < extent. `kv_start` = the absolute
/// position of the view's first key (non-zero when a sliding-layer decode
/// reads the last-window KV view). Returns [B, H, Lq, kv_len] f32.
fn inklingRelBias(s: mlx.mlx_stream, rel_states: mlx.mlx_array, proj: mlx.mlx_array, q_start: i64, kv_len: c_int, kv_start: c_int) !mlx.mlx_array {
    const rs_shape = mlx.getShape(rel_states); // [B, Lq, H, d_rel]
    const batch = rs_shape[0];
    const lq = rs_shape[1];
    const heads = rs_shape[2];
    const extent = mlx.getShape(proj)[1];

    var rs32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs32);
    try mlx.check(mlx.mlx_astype(&rs32, rel_states, .float32, s));
    var proj32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(proj32);
    try mlx.check(mlx.mlx_astype(&proj32, proj, .float32, s));

    // [B, Lq, H, extent] → [B, H, Lq, extent]
    var rel_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rel_logits);
    try mlx.check(mlx.mlx_matmul(&rel_logits, rs32, proj32, s));
    var rel_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rel_t);
    try mlx.check(mlx.mlx_transpose_axes(&rel_t, rel_logits, &[_]c_int{ 0, 2, 1, 3 }, 4, s));

    // distance[i, j] = (q_start + i) - j, [Lq, kv_len] int32
    var q_pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_pos);
    try mlx.check(mlx.mlx_arange(&q_pos, @floatFromInt(q_start), @floatFromInt(q_start + lq), 1, .int32, s));
    var q_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_col);
    try mlx.check(mlx.mlx_reshape(&q_col, q_pos, &[_]c_int{ lq, 1 }, 2, s));
    var kv_pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_pos);
    try mlx.check(mlx.mlx_arange(&kv_pos, @floatFromInt(kv_start), @floatFromInt(kv_start + kv_len), 1, .int32, s));
    var distance = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(distance);
    try mlx.check(mlx.mlx_subtract(&distance, q_col, kv_pos, s));

    // gather = clip(distance, 0, extent-1), broadcast [B, H, Lq, kv_len]
    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    const max_i = mlx.mlx_array_new_int(extent - 1);
    defer _ = mlx.mlx_array_free(max_i);
    var lo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo);
    try mlx.check(mlx.mlx_maximum(&lo, distance, zero_i, s));
    var gather = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather);
    try mlx.check(mlx.mlx_minimum(&gather, lo, max_i, s));
    var gather4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather4);
    try mlx.check(mlx.mlx_reshape(&gather4, gather, &[_]c_int{ 1, 1, lq, kv_len }, 4, s));
    var gather_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather_b);
    try mlx.check(mlx.mlx_broadcast_to(&gather_b, gather4, &[_]c_int{ batch, heads, lq, kv_len }, 4, s));

    var bias = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias);
    try mlx.check(mlx.mlx_take_along_axis(&bias, rel_t, gather_b, 3, s));

    // valid = 0 <= distance < extent (per the reference, additive-zero outside)
    var ge0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ge0);
    try mlx.check(mlx.mlx_greater_equal(&ge0, distance, zero_i, s));
    const ext_i = mlx.mlx_array_new_int(extent);
    defer _ = mlx.mlx_array_free(ext_i);
    var lt_ext = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lt_ext);
    try mlx.check(mlx.mlx_less(&lt_ext, distance, ext_i, s));
    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_logical_and(&valid, ge0, lt_ext, s));
    var valid4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid4);
    try mlx.check(mlx.mlx_reshape(&valid4, valid, &[_]c_int{ 1, 1, lq, kv_len }, 4, s));

    const zero_f = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero_f);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_where(&out, valid4, bias, zero_f, s));
    return out;
}

/// Fold the causal (or sliding-window) constraint into the rel bias as ONE
/// additive mask: where allowed, the bias; elsewhere -1e30 (the reference's
/// NEG_INF — built with where, never indicator × -inf). Returns f32
/// [.., Lq, kv_len] broadcastable against the bias shape.
fn inklingCombineMask(s: mlx.mlx_stream, bias: mlx.mlx_array, q_start: c_int, kv_len: c_int, kv_start: c_int, window: ?c_int) !mlx.mlx_array {
    const b_shape = mlx.getShape(bias); // [B, H, Lq, kv_len]
    const lq = b_shape[2];
    var q_pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_pos);
    try mlx.check(mlx.mlx_arange(&q_pos, @floatFromInt(q_start), @floatFromInt(q_start + lq), 1, .int32, s));
    var q_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_col);
    try mlx.check(mlx.mlx_reshape(&q_col, q_pos, &[_]c_int{ lq, 1 }, 2, s));
    var kv_pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_pos);
    try mlx.check(mlx.mlx_arange(&kv_pos, @floatFromInt(kv_start), @floatFromInt(kv_start + kv_len), 1, .int32, s));
    var distance = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(distance);
    try mlx.check(mlx.mlx_subtract(&distance, q_col, kv_pos, s));

    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    var allowed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(allowed);
    try mlx.check(mlx.mlx_greater_equal(&allowed, distance, zero_i, s));
    if (window) |w| {
        const w_i = mlx.mlx_array_new_int(w);
        defer _ = mlx.mlx_array_free(w_i);
        var in_win = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(in_win);
        try mlx.check(mlx.mlx_less(&in_win, distance, w_i, s));
        var both = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_logical_and(&both, allowed, in_win, s));
        _ = mlx.mlx_array_free(allowed);
        allowed = both;
    }
    var allowed4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(allowed4);
    try mlx.check(mlx.mlx_reshape(&allowed4, allowed, &[_]c_int{ 1, 1, lq, kv_len }, 4, s));

    const neg_inf = mlx.mlx_array_new_float(-1e30);
    defer _ = mlx.mlx_array_free(neg_inf);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_where(&out, allowed4, bias, neg_inf, s));
    return out;
}

/// Per-query log-scaling factor for global layers past the floor:
/// tau_i = 1 + alpha·ln(max((q_start + i + 1)/n_floor, 1)), f32 [1,1,Lq,1].
fn inklingLogScaleTau(s: mlx.mlx_stream, q_start: c_int, lq: c_int, n_floor: u32, alpha: f32) !mlx.mlx_array {
    var eff_n = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eff_n);
    try mlx.check(mlx.mlx_arange(&eff_n, @floatFromInt(q_start + 1), @floatFromInt(q_start + lq + 1), 1, .float32, s));
    const floor_f = mlx.mlx_array_new_float(@floatFromInt(n_floor));
    defer _ = mlx.mlx_array_free(floor_f);
    var ratio = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ratio);
    try mlx.check(mlx.mlx_divide(&ratio, eff_n, floor_f, s));
    const one_f = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_f);
    var clamped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(clamped);
    try mlx.check(mlx.mlx_maximum(&clamped, ratio, one_f, s));
    var logv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logv);
    try mlx.check(mlx.mlx_log(&logv, clamped, s));
    const alpha_f = mlx.mlx_array_new_float(alpha);
    defer _ = mlx.mlx_array_free(alpha_f);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, logv, alpha_f, s));
    var tau_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tau_flat);
    try mlx.check(mlx.mlx_add(&tau_flat, scaled, one_f, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, tau_flat, &[_]c_int{ 1, 1, lq, 1 }, 4, s));
    return out;
}

// ── the plugin's block ──

const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const kv = @import("../../sdk/bases/kv.zig");
const band = @import("../../sdk/bases/blocks/sliding.zig");
const sdpaSinks = @import("../../sdk/bases/blocks/decoder_attention.zig").sdpaSinks;
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const dq = @import("../../sdk/blocks/attention/decode_attn_quant.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Ctx = block.Ctx;

/// `[1, S, C]` through a short conv behind `prev` (null = zeros): the output and the new window, owned by `g`.
fn sconv(g: *Graph, x: Array, w: Array, prev: ?Array) ![2]Array {
    const r = try inklingSconvCore(g.s, x, w, prev);
    const out = g.adopt(r.out) catch |e| {
        _ = mlx.mlx_array_free(r.new_state);
        return e;
    };
    return .{ out, try g.adopt(r.new_state) };
}

/// Inkling's attention with its output short conv (`attn_sconv`): q/k/v/relative-state projections, causal short
/// convs on k and v, per-head q/k RMSNorm, no RoPE; the hidden-conditioned relative bias and the causal (or band)
/// constraint as one additive mask, log-scaled with q on a global layer past `log_floor`. The layer's conv state
/// `[k | v | attn | mlp]` (f32) is the slot's `.conv`: this block writes the first three windows and carries the
/// MLP's, which `Mlp` replaces.
pub const Attention = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        rel: Linear,
        q_norm: Array,
        k_norm: Array,
        /// `[d_rel, extent]`.
        rel_proj: Array,
        k_sconv: Array,
        v_sconv: Array,
        /// The output's short conv `[hidden, kernel, 1]`.
        out_sconv: Array,
        heads: c_int,
        kv_heads: c_int,
        head_dim: c_int,
        d_rel: c_int,
        /// 0 = a global layer.
        window: u32,
        /// Log-scaling past this many positions on a global layer; 0 = off.
        log_floor: u32,
        log_alpha: f32,
        /// q, k, v, rel, o decode-only side copies (`--decode-attn-quant`); a null `w` keeps the weight.
        dq: [5]Linear = @splat(.{}),
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const s = c.rows;
        const hd = w.head_dim;
        const kv_dim = w.kv_heads * hd;
        const offset: c_int = @intCast(c.offset());
        const total: c_int = offset + s;
        // A solo one-row step reads the side copies, as the legacy `attnProj`.
        const step = c.slots == 1 and c.rows == 1 and dq.decodeAttnQuantEnabled();
        var lins = [_]Linear{ w.q, w.k, w.v, w.rel, w.o };
        if (step) for (&lins, w.dq) |*l, d| {
            if (d.w.ctx != null) l.* = d;
        };
        const q_proj = try lins[0].forward(g, x);
        const k_proj = try lins[1].forward(g, x);
        const v_proj = try lins[2].forward(g, x);
        const rel_out = try lins[3].forward(g, x);

        const st = try kv.state(c, .conv);
        const width: c_int = if (st) |a| ops.shape(a)[2] else 0;
        const window = struct {
            fn at(gr: *Graph, a: ?Array, from: c_int, n: c_int) !?Array {
                return if (a) |x_| try gr.sliceAxis(x_, 2, from, from + n) else null;
            }
        }.at;
        const ks = try sconv(g, k_proj, w.k_sconv, try window(g, st, 0, kv_dim));
        const vs = try sconv(g, v_proj, w.v_sconv, try window(g, st, kv_dim, kv_dim));

        const perm = [_]c_int{ 0, 2, 1, 3 };
        const q = try g.transpose(try g.rmsNorm(try g.reshape(q_proj, &.{ 1, s, w.heads, hd }), w.q_norm, c.shared.eps), &perm);
        const k = try g.transpose(try g.rmsNorm(try g.reshape(ks[0], &.{ 1, s, w.kv_heads, hd }), w.k_norm, c.shared.eps), &perm);
        const v = try g.transpose(try g.reshape(vs[0], &.{ 1, s, w.kv_heads, hd }), &perm);

        const sv = band.view(w.window, hd, total, s);
        var view = try kv.append(c, k, v, if (w.window > 0) sv.span else 0);
        defer kv.release(c, &view);
        const dv = kv.dense(view);
        const kv_len = ops.shape(dv.k)[2];
        const kv_start = total - kv_len;

        const rel = try g.reshape(rel_out, &.{ 1, s, w.heads, w.d_rel });
        const bias = try g.adopt(try inklingRelBias(g.s, rel, w.rel_proj, offset, kv_len, kv_start));
        var mask = try g.adopt(try inklingCombineMask(g.s, bias, offset, kv_len, kv_start, if (w.window > 0) @intCast(w.window) else null));
        var qf = q;
        if (w.window == 0 and w.log_floor > 0 and total > @as(c_int, @intCast(w.log_floor))) {
            const tau = try g.adopt(try inklingLogScaleTau(g.s, offset, s, w.log_floor, w.log_alpha));
            qf = try g.astype(try g.mul(try g.astype(q, .float32), tau), ops.dtype(q));
            mask = try g.mul(mask, tau);
        }
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd * hd)));
        const attn = try sdpaSinks(g, qf, dv.k, dv.v, scale, "", try g.astype(mask, ops.dtype(qf)), .{}, false);
        const flat = try g.reshape(try g.transpose(attn, &perm), &.{ 1, s, w.heads * hd });
        const out = try lins[4].forward(g, flat);

        const hidden = ops.shape(out)[2];
        const as = try sconv(g, out, w.out_sconv, try window(g, st, 2 * kv_dim, hidden));
        var parts = [_]Array{ ks[1], vs[1], as[1], undefined };
        const n: usize = if (width > 2 * kv_dim + hidden) 4 else 3;
        if (n == 4) parts[3] = try g.sliceAxis(st.?, 2, 2 * kv_dim + hidden, width);
        try kv.setState(c, .conv, try g.concat(parts[0..n], 2));
        return as[0];
    }
};

/// The MLP slot: the dense SwiGLU times its learned gain (`mlp.global_scale`, in the stored dtype) below
/// `dense_mlp_idx`, the routed experts above, then the output's short conv (`mlp_sconv`), whose window replaces the
/// last columns of the layer's conv state.
pub const Mlp = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        body: union(enum) {
            dense: struct { mlp: GatedMlp.Weights, gain: Array = .{} },
            moe: moe_block.Moe.Weights,
        },
        out_sconv: Array,
        /// Columns of the conv state before this window (`Config.mlpStateOffset`).
        state_offset: c_int,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const y = switch (w.body) {
            .dense => |*d| blk: {
                const m = try GatedMlp.forward(&d.mlp, c, x);
                break :blk if (d.gain.ctx != null) try g.mul(m, d.gain) else m;
            },
            .moe => |*m| try moe_block.Moe.forward(m, c, x),
        };
        // The attention block wrote this layer's state; its MLP window is there once a forward has run.
        const st = (try kv.state(c, .conv)) orelse return error.InklingStateMissing;
        const width = ops.shape(st)[2];
        const prev: ?Array = if (width > w.state_offset) try g.sliceAxis(st, 2, w.state_offset, width) else null;
        const ms = try sconv(g, y, w.out_sconv, prev);
        try kv.setState(c, .conv, try g.concat(&.{ try g.sliceAxis(st, 2, 0, w.state_offset), ms[1] }, 2));
        return ms[0];
    }
};

const GatedMlp = @import("../../sdk/bases/blocks/mlp.zig").GatedMlp;
const moe_block = @import("moe.zig");

/// Head: the LM head, then the padded vocab's real rows (`unpadded_vocab_size`).
pub const Head = struct {
    pub const kind: block.Kind = .head;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        lin: Linear,
        /// Logit columns kept; 0 = all.
        vocab: c_int = 0,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const logits = try head.project(c.g, &w.lin, x);
        return if (w.vocab > 0 and w.vocab < ops.shape(logits)[2]) c.g.sliceAxis(logits, 2, 0, w.vocab) else logits;
    }
};

const head = @import("../../sdk/bases/blocks/head.zig");

// ── tests ──

fn loadInklingFixtures(allocator: std.mem.Allocator) !?std.json.Parsed(std.json.Value) {
    const path_z = std.c.getenv("INKLING_FIXTURES") orelse return null;
    const path = std.mem.span(path_z);
    if (path.len == 0) return null;
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const data = try reader_state.interface.allocRemaining(allocator, .limited(1 << 26));
    defer allocator.free(data);
    return try std.json.parseFromSlice(std.json.Value, allocator, data, .{});
}

test "inkling sconv parity vs reference ShortConvolution (INKLING_FIXTURES)" {
    const allocator = testing.allocator;
    var parsed = (try loadInklingFixtures(allocator)) orelse return error.SkipZigTest;
    defer parsed.deinit();
    const sc = parsed.value.object.get("sconv").?.object;
    const s = mlx.gpuStream();

    const w = try jsonToMlxArray(allocator, sc.get("weight").?);
    defer _ = mlx.mlx_array_free(w);
    const x = try jsonToMlxArray(allocator, sc.get("x").?);
    defer _ = mlx.mlx_array_free(x);
    const y_ref = try jsonToMlxArray(allocator, sc.get("y_full").?);
    defer _ = mlx.mlx_array_free(y_ref);
    const state_ref = try jsonToMlxArray(allocator, sc.get("final_cache_state").?);
    defer _ = mlx.mlx_array_free(state_ref);

    // Full-sequence (no prior state).
    const full = try inklingSconvCore(s, x, w, null);
    defer _ = mlx.mlx_array_free(full.out);
    defer _ = mlx.mlx_array_free(full.new_state);
    try testing.expect(try maxAbsDiffF32(full.out, y_ref, s) < 1e-5);

    // Incremental: one position at a time through the carried state must
    // reproduce the full-sequence output AND the reference's final state.
    const x_shape = mlx.getShape(x);
    const seq: usize = @intCast(x_shape[1]);
    var state: ?mlx.mlx_array = null;
    defer if (state) |st| {
        _ = mlx.mlx_array_free(st);
    };
    var worst: f32 = 0;
    for (0..seq) |t| {
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        const start = [_]c_int{ 0, @intCast(t), 0 };
        const stop = [_]c_int{ x_shape[0], @intCast(t + 1), x_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&xt, x, &start, 3, &stop, 3, &strides, 3, s));
        const step = try inklingSconvCore(s, xt, w, state);
        if (state) |st| _ = mlx.mlx_array_free(st);
        state = step.new_state;
        defer _ = mlx.mlx_array_free(step.out);
        var yt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(yt);
        try mlx.check(mlx.mlx_slice(&yt, y_ref, &start, 3, &stop, 3, &strides, 3, s));
        const e = try maxAbsDiffF32(step.out, yt, s);
        if (e > worst) worst = e;
    }
    try testing.expect(worst < 1e-5);
    try testing.expect(try maxAbsDiffF32(state.?, state_ref, s) < 1e-5);
    std.debug.print("[inkling-sconv] full+incremental parity ok (worst step err {e:.3})\n", .{worst});
}

test "inkling relative-logits bias parity vs reference (INKLING_FIXTURES)" {
    const allocator = testing.allocator;
    var parsed = (try loadInklingFixtures(allocator)) orelse return error.SkipZigTest;
    defer parsed.deinit();
    const rb = parsed.value.object.get("rel_bias").?.object;
    const s = mlx.gpuStream();

    const proj = try jsonToMlxArray(allocator, rb.get("proj").?);
    defer _ = mlx.mlx_array_free(proj);

    // Prefill: Lq == Lkv, start 0.
    {
        const pf = rb.get("prefill").?.object;
        const rel_states = try jsonToMlxArray(allocator, pf.get("rel_states").?);
        defer _ = mlx.mlx_array_free(rel_states);
        const bias_ref = try jsonToMlxArray(allocator, pf.get("bias").?);
        defer _ = mlx.mlx_array_free(bias_ref);
        const shape = mlx.getShape(rel_states);
        const bias = try inklingRelBias(s, rel_states, proj, 0, shape[1], 0);
        defer _ = mlx.mlx_array_free(bias);
        try testing.expect(try maxAbsDiffF32(bias, bias_ref, s) < 1e-5);
    }
    // Decode: Lq = 1 at a position past the extent (zero-beyond-extent region).
    {
        const dc = rb.get("decode").?.object;
        const rel_states = try jsonToMlxArray(allocator, dc.get("rel_states").?);
        defer _ = mlx.mlx_array_free(rel_states);
        const bias_ref = try jsonToMlxArray(allocator, dc.get("bias").?);
        defer _ = mlx.mlx_array_free(bias_ref);
        const kv_len: c_int = @intCast(dc.get("kv_len").?.integer);
        const q_pos: i64 = dc.get("q_pos").?.integer;
        const bias = try inklingRelBias(s, rel_states, proj, q_pos, kv_len, 0);
        defer _ = mlx.mlx_array_free(bias);
        try testing.expect(try maxAbsDiffF32(bias, bias_ref, s) < 1e-5);
    }
    std.debug.print("[inkling-relbias] prefill+decode parity ok\n", .{});
}

test "inkling router chain parity vs reference (INKLING_FIXTURES)" {
    const allocator = testing.allocator;
    var parsed = (try loadInklingFixtures(allocator)) orelse return error.SkipZigTest;
    defer parsed.deinit();
    const rt = parsed.value.object.get("router").?.object;
    const cfgj = parsed.value.object.get("config").?.object;
    const s = mlx.gpuStream();

    const w = try jsonToMlxArray(allocator, rt.get("weight").?); // [E+ns, H]
    defer _ = mlx.mlx_array_free(w);
    const bias = try jsonToMlxArray(allocator, rt.get("bias").?); // [E]
    defer _ = mlx.mlx_array_free(bias);
    const x = try jsonToMlxArray(allocator, rt.get("x").?); // [1, T, H]
    defer _ = mlx.mlx_array_free(x);
    const tw_ref = try jsonToMlxArray(allocator, rt.get("topk_weights").?);
    defer _ = mlx.mlx_array_free(tw_ref);
    const sg_ref = try jsonToMlxArray(allocator, rt.get("shared_gammas").?);
    defer _ = mlx.mlx_array_free(sg_ref);

    const top_k: c_int = @intCast(cfgj.get("num_experts_per_tok").?.integer);
    const n_shared: c_int = @intCast(cfgj.get("n_shared_experts").?.integer);
    const route_scale: f32 = @floatCast(jsonF64(cfgj.get("route_scale").?));
    const global_scale: f32 = @floatCast(jsonF64(rt.get("global_scale").?));

    // logits = flatten(x) @ w.T in f32 (the caller's job in the forward).
    const x_shape = mlx.getShape(x);
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    const flat_shape = [_]c_int{ x_shape[0] * x_shape[1], x_shape[2] };
    try mlx.check(mlx.mlx_reshape(&xf, x, &flat_shape, 2, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose_axes(&wt, w, &[_]c_int{ 1, 0 }, 2, s));
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_matmul(&logits, xf, wt, s));

    const r = try inklingRouterChain(s, logits, bias, top_k, n_shared, route_scale, global_scale);
    defer _ = mlx.mlx_array_free(r.topk_weights);
    defer _ = mlx.mlx_array_free(r.topk_idx);
    defer _ = mlx.mlx_array_free(r.shared_gammas);

    // Indices: order within the top-k is irrelevant downstream, but the
    // reference dumps argpartition order — compare as SETS per token, and
    // weights via a value-sorted comparison per token.
    const ti_ref = rt.get("topk_idx").?.array;
    const idx_c = try materializedOwnedCopy(s, r.topk_idx);
    defer _ = mlx.mlx_array_free(idx_c);
    try mlx.check(mlx.mlx_array_eval(idx_c));
    const ip = mlx.mlx_array_data_uint32(idx_c) orelse return error.NoData;
    const T: usize = @intCast(flat_shape[0]);
    const K: usize = @intCast(top_k);
    for (0..T) |t| {
        const row_ref = ti_ref.items[t].array;
        for (0..K) |k| {
            const want: u32 = @intCast(row_ref.items[k].integer);
            var found = false;
            for (0..K) |k2| {
                if (ip[t * K + k2] == want) found = true;
            }
            try testing.expect(found);
        }
    }
    // Weights: match by expert id (reference order vs ours may differ).
    const tw_c = try materializedOwnedCopy(s, r.topk_weights);
    defer _ = mlx.mlx_array_free(tw_c);
    try mlx.check(mlx.mlx_array_eval(tw_c));
    const wp = mlx.mlx_array_data_float32(tw_c) orelse return error.NoData;
    try mlx.check(mlx.mlx_array_eval(tw_ref));
    const wr = mlx.mlx_array_data_float32(tw_ref) orelse return error.NoData;
    for (0..T) |t| {
        const row_ref = ti_ref.items[t].array;
        for (0..K) |k| {
            const expert: u32 = @intCast(row_ref.items[k].integer);
            var ours: f32 = std.math.nan(f32);
            for (0..K) |k2| {
                if (ip[t * K + k2] == expert) ours = wp[t * K + k2];
            }
            try testing.expect(std.math.isFinite(ours));
            try testing.expect(@abs(ours - wr[t * K + k]) < 1e-4);
        }
    }
    try testing.expect(try maxAbsDiffF32(r.shared_gammas, sg_ref, s) < 1e-4);
    std.debug.print("[inkling-router] top-k select + sink weights parity ok\n", .{});
}
