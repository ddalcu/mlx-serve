//! qwen3_5's MTP head (`SpecHooks.head`): the qwen-layout head layer the engine's sidecar arm runs. `sdk/spec/mtp`'s
//! loader reads the weights (sidecar or in-checkpoint, delta norms folded, quant solved per weight); the forward is
//! the arm's (`mtp.forwardWithMrope`, `appendKvOnly`) over the plugin's own embedding table and the head's one-layer
//! cache behind `cache_ops`. A routed-expert head (a MoE trunk's) runs its experts through the plugin's MoE block, as
//! the arm runs them through the trunk's; a Hy3/Nemotron-shaped head is refused at load.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const env = @import("../../env.zig");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const quantParamsFromGeometry = @import("../../sdk/quant/params.zig").quantParamsFromGeometry;
const verifyQmm = @import("../../sdk/blocks/linear/verify_qmm.zig").verifyQmm;
const fusedSdpa256Prefill = @import("../../sdk/blocks/attention/attn256.zig").fusedSdpa256Prefill;
const sdpaForceFused = @import("../../sdk/blocks/attention/sdpa.zig").sdpaForceFused;
const config = @import("config.zig");
const ld = @import("../qwen3/load.zig");
const ops = @import("../../mlx/ops.zig");
const moe_block = @import("../../sdk/bases/blocks/moe.zig");
const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;
const rht = @import("../../sdk/quant/hadamard/rht.zig");
const axisView = @import("../../mlx/array.zig").axisView;
const gated_attention = @import("../../sdk/bases/blocks/gated_attention.zig");

const Array = mlx.mlx_array;
const Verdict = abi.Verdict;
const MtpModel = mtp_mod.MtpModel;
const rope_scaling = @import("../../sdk/config/rope_scaling.zig");
const QLinear = mtp_mod.QLinear;

/// The loaded head and what its forward reads of the trunk.
pub const Head = struct {
    m: MtpModel,
    c: *const config.Config,
    /// The trunk's embedding table (borrowed from the model).
    embed: *const ld.Linear,
    /// The trunk's scaled rope (borrowed from the model).
    rope: Rope,
    /// A routed-expert head's MLP over the head's own tensors (borrowed from `m`); null = dense.
    moe: ?moe_block.Moe.Weights = null,

    pub fn deinit(h: *Head) void {
        h.m.deinit();
    }

    fn scale(h: *const Head) f32 {
        return 1.0 / @sqrt(@as(f32, @floatFromInt(h.c.query_pre_attn_scalar)));
    }

    /// `HeadHooks.cost_fingerprint`: the engine's measured-layout key for this head, 0 = none.
    pub fn costFingerprint(h: *const Head) u32 {
        const fp = h.m.m5NaxHeadFingerprint(.{
            .hidden = h.c.hidden_size,
            .heads = h.c.num_attention_heads,
            .kv_heads = h.c.num_key_value_heads,
            .head_dim = h.c.head_dim,
            .intermediate = h.c.intermediate_size,
        }) orelse return 0;
        return fp.bits << 16 | fp.group_size;
    }
};

/// The trunk's scaled spectrum (null = the base's) and YaRN's `[head_dim]` factor (null = none).
pub const Rope = struct { freqs: Array = .{ .ctx = null }, mscale: Array = .{ .ctx = null } };

/// Loads the pack's head; `error.UnsupportedMtpHead` for a layout this port does not run.
pub fn load(c: *const config.Config, embed: *const ld.Linear, rope_tables: Rope, model_dir: []const u8, s: mlx.mlx_stream) !Head {
    var m = try mtp_mod.loadMtp(std.Io.Threaded.global_single_threaded.io(), ld.gpa, s, model_dir);
    errdefer m.deinit();
    if (m.layout != .qwen or m.eh_proj != null) return error.UnsupportedMtpHead;
    // The fc reads `[emb | hidden]` (2H wide) into H.
    const fs = mlx.getShape(m.fc.w);
    const h: c_int = @intCast(c.hidden_size);
    if (fs.len != 2 or !(fs[0] == 2 * h and fs[1] == h or fs[0] == h)) return error.MtpTargetMismatch;
    const moe: ?moe_block.Moe.Weights = switch (m.mlp) {
        .moe => |*mw| try moeWeights(c, mw),
        else => null,
    };
    return .{ .m = m, .c = c, .embed = embed, .rope = rope_tables, .moe = moe };
}

/// One projection of the head's MoE layer, its quant solved from the weight's geometry against `in_dim`.
fn projOf(w: Array, sc: ?Array, bi: ?Array, in_dim: u32) !ld.Linear {
    const s = sc orelse Array{ .ctx = null };
    const b = bi orelse Array{ .ctx = null };
    if (s.ctx == null) return .{ .w = w };
    const qp = quantParamsFromGeometry(w, s, b.ctx != null, in_dim) orelse return error.UnsupportedMtpHead;
    return .{ .w = w, .s = s, .b = b, .bits = qp.bits, .group_size = qp.group_size, .mode = qp.mode };
}

fn bankOf(w: Array, sc: Array, bi: Array, in_dim: u32) !moe_block.Bank {
    if (sc.ctx == null) return error.UnsupportedMtpHead;
    const qp: QuantParams = quantParamsFromGeometry(w, sc, bi.ctx != null, in_dim) orelse return error.UnsupportedMtpHead;
    return .{ .w = w, .s = sc, .b = bi, .qp = qp };
}

/// The head's router, stacked experts and gated shared expert as the plugin's MoE block reads them: no one-expert
/// views and no compiled routing, the shared gate as loaded (the arm's `MoeMlpWeights` carry neither).
fn moeWeights(c: *const config.Config, mw: anytype) !moe_block.Moe.Weights {
    const hidden: u32 = @intCast(c.hidden_size);
    const inter: u32 = @intCast(c.moe_intermediate_size);
    const shared_inter: u32 = @intCast(c.shared_expert_intermediate_size);
    const seg_w = mw.shared_expert_gate_w orelse return error.UnsupportedMtpHead;
    const router = try projOf(mw.router_w, mw.router_s, mw.router_b, hidden);
    const seg = try projOf(seg_w, mw.shared_expert_gate_s, mw.shared_expert_gate_b, hidden);
    // The arm reads the gate as a quantized projection.
    if (seg.s.ctx == null) return error.UnsupportedMtpHead;
    const shared: moe_block.SharedExpert = .{
        .gate = try projOf(mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, hidden),
        .up = try projOf(mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, hidden),
        .down = try projOf(mw.shared_down_w, mw.shared_down_s, mw.shared_down_b, shared_inter),
        .gate_logit = seg,
    };
    return .{
        .router = router,
        .route = .softmax,
        .gate = try bankOf(mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, hidden),
        .up = try bankOf(mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, hidden),
        .down = try bankOf(mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inter),
        .shared = shared,
        .top_k = @intCast(c.num_experts_per_tok),
        // The head's own router: a pruned (REAP) trunk's `num_experts` is smaller than the head it ships.
        .experts = mlx.getShape(mw.switch_gate_w)[0],
    };
}

/// One head forward as the engine calls it (`HeadHooks.forward`): `.last_row` lets leading rows append K/V only,
/// `.none` with `want_hidden` runs every row, `.none` without it appends history only. `out.hidden` = the last row's
/// normed output.
pub fn forward(h: *const Head, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const s = h.m.s;
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_reshape(&ids, .{ .ctx = f.ids.ctx }, &[_]c_int{@intCast(f.rows)}, 1, s));
    const hidden: Array = .{ .ctx = f.hidden_in.ctx };
    const off: c_int = @intCast(f.positions.slice()[0]);
    const kv: Kv = .{ .ops = f.cache_ops orelse return error.NoCacheOps, .cache = f.caches.slice()[0], .s = s };
    var mr: ?[2]Array = null;
    if (f.mrope.ctx != null) {
        // The engine splits a merged `.last_row` step itself, each part roped on its own side of the table's end.
        if (f.logits == .last_row and f.rows > 1) return error.UnsupportedMropeSplit;
        mr = .{ try angleHalf(.{ .ctx = f.mrope.ctx }, 0, s), try angleHalf(.{ .ctx = f.mrope.ctx }, 1, s) };
    }
    defer if (mr) |m| for (m) |x| {
        _ = mlx.mlx_array_free(x);
    };
    if (f.logits == .none and !f.want_hidden) return appendKvOnly(h, kv, ids, hidden, off, mr);
    const post = try forwardRows(h, kv, ids, hidden, off, f.logits == .last_row, mr);
    out.hidden = .{ .ctx = post.ctx };
}

/// Half `i` (cos, sin) of `Forward.mrope` `[2, 1, 1, rows, rope_dims]`.
fn angleHalf(x: Array, i: c_int, s: mlx.mlx_stream) !Array {
    const sh = mlx.getShape(x);
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_slice(&one, x, &[_]c_int{ i, 0, 0, 0, 0 }, 5, &[_]c_int{ i + 1, sh[1], sh[2], sh[3], sh[4] }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
    return reshaped(one, sh[1..], s);
}

const Kv = struct {
    ops: *const abi.CacheOps,
    cache: abi.Handle,
    s: mlx.mlx_stream,

    fn append(kv: Kv, k: Array, v: Array) !abi.KvView {
        var view: abi.KvView = .{};
        const vd = kv.ops.append(kv.cache, 0, .{ .ctx = k.ctx }, .{ .ctx = v.ctx }, 0, .{ .ctx = kv.s.ctx }, &view);
        if (!vd.isOk()) return error.KvAppendFailed;
        return view;
    }
};



fn rmsNorm(x: Array, w: Array, eps: f32, s: mlx.mlx_stream) !Array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

/// The arm's projection: dense matmul, else the quant params solved from the weight's geometry against `x`'s width,
/// the verify-width kernels first on affine weights.
fn linear(m: *const MtpModel, x: Array, lin: *const QLinear) !Array {
    var out = mlx.mlx_array_new();
    if (lin.s.ctx == null) {
        try mlx.check(mlx.mlx_matmul(&out, x, lin.w, m.s));
        return out;
    }
    var bits = m.quant_bits;
    var group = m.quant_group_size;
    var mode = m.quant_mode;
    const xs = mlx.getShape(x);
    if (xs.len > 0 and xs[xs.len - 1] > 0) if (quantParamsFromGeometry(lin.w, lin.s, lin.b.ctx != null, @intCast(xs[xs.len - 1]))) |qp| {
        bits = qp.bits;
        group = qp.group_size;
        mode = qp.mode;
    };
    if (mode == .affine) if (try verifyQmm(m.s, x, lin.w, lin.s, lin.b, bits, group)) |y| {
        _ = mlx.mlx_array_free(out);
        return y;
    };
    try mlx.check(mlx.mlx_quantized_matmul(&out, x, lin.w, lin.s, lin.b, true, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), m.s));
    return out;
}

/// `[n]` ids through the trunk's table into `[1, n, H]` bf16 (dequantized rows, as the arm embeds them).
fn embedRows(h: *const Head, ids: Array, n: c_int) !Array {
    const s = h.m.s;
    const e = h.embed;
    const shape = [_]c_int{ 1, n, @intCast(h.c.hidden_size) };
    var tw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tw);
    try mlx.check(mlx.mlx_take_axis(&tw, e.w, ids, 0, s));
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    if (e.s.ctx == null) {
        try mlx.check(mlx.mlx_astype(&rows, tw, .bfloat16, s));
    } else {
        var ts = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ts);
        try mlx.check(mlx.mlx_take_axis(&ts, e.s, ids, 0, s));
        var tb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tb);
        if (e.b.ctx != null) try mlx.check(mlx.mlx_take_axis(&tb, e.b, ids, 0, s));
        try mlx.check(mlx.mlx_dequantize(&rows, tw, ts, tb, mlx.mlx_optional_int.some(@intCast(e.group_size)), mlx.mlx_optional_int.some(@intCast(e.bits)), e.mode.cstr(), .{}, .{ .value = .bfloat16, .has_value = true }, s));
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, rows, &shape, 3, s));
    // A Hadamard pack's rows back to the residual basis (the arm's `unrotateEmbedding`).
    const reg = e.rht orelse return out;
    if (e.signs.ctx == null) return out;
    defer _ = mlx.mlx_array_free(out);
    return rht.transform(out, e.signs, reg.block, true, s);
}

fn cast(x: Array, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !Array {
    var out = mlx.mlx_array_new();
    if (mlx.mlx_array_dtype(x) == dt) {
        try mlx.check(mlx.mlx_array_set(&out, x));
    } else try mlx.check(mlx.mlx_astype(&out, x, dt, s));
    return out;
}

/// `fc([norm(embed(ids)) | norm(hidden)])` in the head's dtype.
fn fcConcat(h: *const Head, ids: Array, hidden: Array, n: c_int) !Array {
    const m = &h.m;
    const s = m.s;
    const eps = h.c.rms_norm_eps;
    const dt = mlx.mlx_array_dtype(m.pre_fc_norm_hidden);
    const raw = try embedRows(h, ids, n);
    defer _ = mlx.mlx_array_free(raw);
    const e = try cast(raw, dt, s);
    defer _ = mlx.mlx_array_free(e);
    const hid = try cast(hidden, dt, s);
    defer _ = mlx.mlx_array_free(hid);
    const en = try rmsNorm(e, m.pre_fc_norm_emb, eps, s);
    defer _ = mlx.mlx_array_free(en);
    const hn = try rmsNorm(hid, m.pre_fc_norm_hidden, eps, s);
    defer _ = mlx.mlx_array_free(hn);
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, en);
    _ = mlx.mlx_vector_array_append_value(vec, hn);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 2, s));
    return linear(m, cat, &m.fc);
}

fn ropeDims(c: *const config.Config) c_int {
    return @intCast(c.ropeDims());
}

/// The trunk's rope base for a forward reaching `seq` positions (null under a scaled spectrum).
fn ropeBase(h: *const Head, seq: c_int) mlx.mlx_optional_float {
    const c = h.c;
    if (h.rope.freqs.ctx != null) return .{ .value = 0, .has_value = false };
    const n = rope_scaling.ntk(c.rope, c.max_position_embeddings) orelse return mlx.mlx_optional_float.some(c.rope_theta);
    return mlx.mlx_optional_float.some(@floatCast(n.base(c.rope_theta, c.ropeDims(), @intCast(seq))));
}

/// YaRN's factor on `x`'s rotated dims (a new reference).
fn mscaled(h: *const Head, x: Array, s: mlx.mlx_stream) !Array {
    var out = mlx.mlx_array_new();
    if (h.rope.mscale.ctx == null) {
        try mlx.check(mlx.mlx_array_set(&out, x));
    } else try mlx.check(mlx.mlx_multiply(&out, x, h.rope.mscale, s));
    return out;
}

/// RoPE at `off` as the trunk ropes, or the per-row M-RoPE angles `mr` (cos, sin) when the rows sit inside a prompt's
/// position table.
fn rope(x: Array, h: *const Head, off: c_int, mr: ?[2]Array, s: mlx.mlx_stream) !Array {
    const c = h.c;
    if (mr) |m| {
        var g = ops.Graph.init(ld.gpa, s);
        defer g.deinit();
        return g.escape(try gated_attention.applyMrope(&g, x, m[0], m[1], ropeDims(c)));
    }
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_rope(&out, x, ropeDims(c), false, ropeBase(h, off + mlx.getShape(x)[2]), rope_scaling.positionScale(c.rope), off, h.rope.freqs, s));
    return mscaled(h, out, s);
}

fn transposed(x: Array, s: mlx.mlx_stream) !Array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, x, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    return out;
}

fn reshaped(x: Array, shape: []const c_int, s: mlx.mlx_stream) !Array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, shape.len, s));
    return out;
}

/// Rows `[start, stop)` of `[n]` ids or of `[1, n, H]` hidden (axis 1).
fn sliceRows(x: Array, start: c_int, stop: c_int, s: mlx.mlx_stream) !Array {
    const sh = mlx.getShape(x);
    var out = mlx.mlx_array_new();
    if (sh.len == 1) {
        try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{start}, 1, &[_]c_int{stop}, 1, &[_]c_int{1}, 1, s));
    } else {
        try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ 0, start, 0 }, 3, &[_]c_int{ sh[0], stop, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    }
    return out;
}

/// Leading rows into the head's cache: fc + input norm + K/V + k norm + rope, no query, attention or MLP.
fn appendKvOnly(h: *const Head, kv: Kv, ids: Array, hidden: Array, off: c_int, mr: ?[2]Array) !void {
    const m = &h.m;
    const s = m.s;
    const c = h.c;
    const n: c_int = mlx.getShape(hidden)[1];
    const x = try fcConcat(h, ids, hidden, n);
    defer _ = mlx.mlx_array_free(x);
    const normed = try rmsNorm(x, m.input_norm, c.rms_norm_eps, s);
    defer _ = mlx.mlx_array_free(normed);
    const kp = try linear(m, normed, &m.k);
    defer _ = mlx.mlx_array_free(kp);
    const vp = try linear(m, normed, &m.v);
    defer _ = mlx.mlx_array_free(vp);
    const kv_shape = [_]c_int{ 1, n, @intCast(c.num_key_value_heads), @intCast(c.head_dim) };
    const kr = try reshaped(kp, &kv_shape, s);
    defer _ = mlx.mlx_array_free(kr);
    const vr = try reshaped(vp, &kv_shape, s);
    defer _ = mlx.mlx_array_free(vr);
    const kn = try rmsNorm(kr, m.k_norm, c.rms_norm_eps, s);
    defer _ = mlx.mlx_array_free(kn);
    const kt = try transposed(kn, s);
    defer _ = mlx.mlx_array_free(kt);
    const vt = try transposed(vr, s);
    defer _ = mlx.mlx_array_free(vt);
    const kro = try rope(kt, h, off, mr, s);
    defer _ = mlx.mlx_array_free(kro);
    var view = try kv.append(kro, vt);
    kv.ops.release(&view);
}

/// The head layer's pre-rope half: the fc residual `x`, q/k/v as `[1, heads, n, D]` (q and k normed) and the raw
/// output gate `[1, n, heads, D]`.
const Front = struct {
    x: Array,
    q: Array,
    k: Array,
    v: Array,
    gate: Array,

    fn deinit(f: *const Front) void {
        for ([_]Array{ f.x, f.q, f.k, f.v, f.gate }) |a| _ = mlx.mlx_array_free(a);
    }
};

fn front(h: *const Head, ids: Array, hidden: Array, n: c_int) !Front {
    const m = &h.m;
    const s = m.s;
    const c = h.c;
    const heads: c_int = @intCast(c.num_attention_heads);
    const kvh: c_int = @intCast(c.num_key_value_heads);
    const hd: c_int = @intCast(c.head_dim);
    const eps = c.rms_norm_eps;
    const x = try fcConcat(h, ids, hidden, n);
    errdefer _ = mlx.mlx_array_free(x);
    const normed = try rmsNorm(x, m.input_norm, eps, s);
    defer _ = mlx.mlx_array_free(normed);
    const qp = try linear(m, normed, &m.q);
    defer _ = mlx.mlx_array_free(qp);
    const qg = try reshaped(qp, &.{ 1, n, heads, hd * 2 }, s);
    defer _ = mlx.mlx_array_free(qg);
    var split = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(split);
    try mlx.check(mlx.mlx_split(&split, qg, 2, -1, s));
    var queries = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(queries);
    var gate = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_vector_array_get(&queries, split, 0));
    try mlx.check(mlx.mlx_vector_array_get(&gate, split, 1));
    const kp = try linear(m, normed, &m.k);
    defer _ = mlx.mlx_array_free(kp);
    const vp = try linear(m, normed, &m.v);
    defer _ = mlx.mlx_array_free(vp);
    const kr = try reshaped(kp, &.{ 1, n, kvh, hd }, s);
    defer _ = mlx.mlx_array_free(kr);
    const vr = try reshaped(vp, &.{ 1, n, kvh, hd }, s);
    defer _ = mlx.mlx_array_free(vr);
    const qn = try rmsNorm(queries, m.q_norm, eps, s);
    defer _ = mlx.mlx_array_free(qn);
    const kn = try rmsNorm(kr, m.k_norm, eps, s);
    defer _ = mlx.mlx_array_free(kn);
    const qt = try transposed(qn, s);
    errdefer _ = mlx.mlx_array_free(qt);
    const kt = try transposed(kn, s);
    errdefer _ = mlx.mlx_array_free(kt);
    const vt = try transposed(vr, s);
    return .{ .x = x, .q = qt, .k = kt, .v = vt, .gate = gate };
}

/// The post-attention half over `attn` `[1, heads, n, D]`: o(attn * sigmoid(gate)), the residual, the MLP, the final
/// norm; `[1, n, H]`.
fn back(h: *const Head, fr: *const Front, attn: Array, n: c_int) !Array {
    const m = &h.m;
    const s = m.s;
    const c = h.c;
    const heads: c_int = @intCast(c.num_attention_heads);
    const hd: c_int = @intCast(c.head_dim);
    const eps = c.rms_norm_eps;
    const at = try transposed(attn, s);
    defer _ = mlx.mlx_array_free(at);
    var gs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gs);
    try mlx.check(mlx.mlx_sigmoid(&gs, fr.gate, s));
    var g4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g4);
    try mlx.check(mlx.mlx_multiply(&g4, at, gs, s));
    const gated = try reshaped(g4, &.{ 1, n, heads * hd }, s);
    defer _ = mlx.mlx_array_free(gated);
    const o = try linear(m, gated, &m.o);
    defer _ = mlx.mlx_array_free(o);
    var h1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h1);
    try mlx.check(mlx.mlx_add(&h1, fr.x, o, s));
    const ff = try rmsNorm(h1, m.post_attn_norm, eps, s);
    defer _ = mlx.mlx_array_free(ff);
    const down = if (h.moe) |*mw| try moeMlp(mw, ff, s) else try denseMlp(m, ff);
    defer _ = mlx.mlx_array_free(down);
    var xo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xo);
    try mlx.check(mlx.mlx_add(&xo, h1, down, s));
    return rmsNorm(xo, m.final_norm, eps, s);
}

/// The head layer over `n` rows at `off`; the last row's normed output.
fn forwardRows(h: *const Head, kv: Kv, ids: Array, hidden: Array, off: c_int, want_logits: bool, mr: ?[2]Array) !Array {
    const m = &h.m;
    const s = m.s;
    const c = h.c;
    const n: c_int = mlx.getShape(hidden)[1];
    const hs: c_int = @intCast(c.hidden_size);
    if (want_logits and n > 1) {
        const ids_head = try sliceRows(ids, 0, n - 1, s);
        defer _ = mlx.mlx_array_free(ids_head);
        const ids_last = try sliceRows(ids, n - 1, n, s);
        defer _ = mlx.mlx_array_free(ids_last);
        const hid_head = try sliceRows(hidden, 0, n - 1, s);
        defer _ = mlx.mlx_array_free(hid_head);
        const hid_last = try sliceRows(hidden, n - 1, n, s);
        defer _ = mlx.mlx_array_free(hid_last);
        try appendKvOnly(h, kv, ids_head, hid_head, off, null);
        return forwardRows(h, kv, ids_last, hid_last, off + n - 1, true, null);
    }
    const fr = try front(h, ids, hidden, n);
    defer fr.deinit();
    const qro = try rope(fr.q, h, off, mr, s);
    defer _ = mlx.mlx_array_free(qro);
    const kro = try rope(fr.k, h, off, mr, s);
    defer _ = mlx.mlx_array_free(kro);
    var view = try kv.append(kro, fr.v);
    defer kv.ops.release(&view);
    const vk: Array = .{ .ctx = view.k.ctx };
    const vv: Array = .{ .ctx = view.v.ctx };

    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const fused = if (n > 1) try fusedSdpa256Prefill(s, qro, vk, vv, h.scale(), 0) else null;
    if (fused) |a| {
        _ = mlx.mlx_array_free(attn);
        attn = a;
    } else {
        const none = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none);
        const mode: [*:0]const u8 = if (n > 1) "causal" else "";
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qro, vk, vv, h.scale(), mode, none, .{ .ctx = null }, n > 1 and sdpaForceFused(qro, vk), s));
    }
    const post = try back(h, &fr, attn, n);
    if (n == 1) return post;
    defer _ = mlx.mlx_array_free(post);
    var last = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&last, post, &[_]c_int{ 0, n - 1, 0 }, 3, &[_]c_int{ 1, n, hs }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return last;
}

/// One draft step for several requests (`HeadHooks.forward_lanes`): the lanes are the rows of one front and back,
/// each roped at its own position and attending its own cache. `out.hidden` = `[1, n, H]`.
pub fn forwardLanes(h: *const Head, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const s = h.m.s;
    const c = h.c;
    const handles = f.caches.slice();
    const positions = f.positions.slice();
    const n = handles.len;
    if (n == 0 or n > mtp_mod.MAX_LANES or positions.len != n) return error.MtpLaneCount;
    const ops_ = f.cache_ops orelse return error.NoCacheOps;
    const nc: c_int = @intCast(n);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_reshape(&ids, .{ .ctx = f.ids.ctx }, &[_]c_int{nc}, 1, s));
    const fr = try front(h, ids, .{ .ctx = f.hidden_in.ctx }, nc);
    defer fr.deinit();

    var offsets: [mtp_mod.MAX_LANES]i32 = undefined;
    for (positions, 0..) |p, k| offsets[k] = @intCast(p);
    const off_arr = mlx.mlx_array_new_data(&offsets, &[_]c_int{nc}, 1, .int32);
    defer _ = mlx.mlx_array_free(off_arr);
    var roped: [2]Array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (roped) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]Array{ fr.q, fr.k }, &roped) |x, *o| {
        var by_lane = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(by_lane);
        try mlx.check(mlx.mlx_transpose_axes(&by_lane, x, &[_]c_int{ 2, 1, 0, 3 }, 4, s));
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_fast_rope_dynamic(&r, by_lane, ropeDims(c), false, ropeBase(h, 0), rope_scaling.positionScale(c.rope), off_arr, h.rope.freqs, s));
        _ = mlx.mlx_array_free(o.*);
        o.* = try mscaled(h, r, s);
    }

    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    var outs: [mtp_mod.MAX_LANES]Array = undefined;
    var built: usize = 0;
    defer for (outs[0..built]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (handles, 0..) |handle, k| {
        const q = try axisView(s, roped[0], 0, k);
        defer _ = mlx.mlx_array_free(q);
        const key = try axisView(s, roped[1], 0, k);
        defer _ = mlx.mlx_array_free(key);
        const val = try axisView(s, fr.v, 2, k);
        defer _ = mlx.mlx_array_free(val);
        const kv: Kv = .{ .ops = ops_, .cache = handle, .s = s };
        var view = try kv.append(key, val);
        defer kv.ops.release(&view);
        outs[built] = mlx.mlx_array_new();
        built += 1;
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&outs[k], q, .{ .ctx = view.k.ctx }, .{ .ctx = view.v.ctx }, h.scale(), "", none, .{ .ctx = null }, false, s));
    }
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const vec = mlx.mlx_vector_array_new_data(&outs, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(&attn, vec, 2, s));
    const post = try back(h, &fr, attn, nc);
    out.hidden = .{ .ctx = post.ctx };
}

/// The plugin's MoE block over `x` (owned by the caller).
fn moeMlp(w: *const moe_block.Moe.Weights, x: Array, s: mlx.mlx_stream) !Array {
    var g = ops.Graph.init(ld.gpa, s);
    defer g.deinit();
    const y = try moe_block.mlp(w, &g, x);
    return g.escape(y);
}

/// The dense SwiGLU (`silu(gate) * up`, then down).
fn denseMlp(m: *const MtpModel, ff: Array) !Array {
    const s = m.s;
    const d = &m.mlp.dense;
    const g = try linear(m, ff, &d.gate);
    defer _ = mlx.mlx_array_free(g);
    const up = try linear(m, ff, &d.up);
    defer _ = mlx.mlx_array_free(up);
    var gsig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gsig);
    try mlx.check(mlx.mlx_sigmoid(&gsig, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, gsig, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_multiply(&act, silu, up, s));
    return linear(m, act, &d.down);
}
