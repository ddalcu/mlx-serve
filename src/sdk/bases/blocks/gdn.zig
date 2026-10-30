//! The GatedDeltaNet mixer (the qwen3_5 family's linear-attention layers) over the slot's recurrent state, read and
//! written through `CacheOps`: prefill (composed chain, blocked/pipelined recurrence, the wide fused prework where a
//! pack asks), the fused one-dispatch decode, verify rows with the per-row state capture a partial accept rolls back
//! to, and batched rows (every slot's state stacked on the batch axis).
//! copied_from: sdk/blocks/gdn/gated_delta_net.zig (`gatedDeltaNetProjected` without the KDA, low-rank gate,
//! grouped-verify and diagnostic arms), sdk/blocks/gdn/gate.zig (`computeGdnGate`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const log = @import("../../util/log.zig");
const block = @import("../block.zig");
const linear = @import("linear.zig");
const conv1d = @import("conv1d.zig");
const state = @import("state.zig");
const compiled = @import("compiled.zig");
const gdn_decode = @import("../../blocks/gdn/decode.zig");
const fused = @import("../../blocks/gdn/decode_fused.zig");
const kernels = @import("../../blocks/gdn/kernels.zig");
const gate_mod = @import("../../blocks/gdn/gate.zig");
const row_exact = @import("row_exact.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Dtype = ops.Dtype;
const Ctx = block.Ctx;
const Linear = linear.Linear;
const Joined = linear.Joined;

/// One model's linear-attention geometry and the constants every layer shares.
pub const Geo = struct {
    hk: c_int,
    hv: c_int,
    dk: c_int,
    dv: c_int,
    /// The conv window plus one.
    kernel: c_int,
    /// The recurrent state's dtype (f32 where the pack's reference keeps it: `ssm_state_f32`, Hadamard packs).
    state: Dtype,
    /// The q/k RMS norms' epsilon (`ModelConfig.gdnQkNormEps`).
    qk_eps: f32,
    /// `y * sigmoid(z)` (qwen4_exp) in place of `silu(z) * y`.
    sigmoid_gate: bool = false,
    /// A prompt chunk takes the wide fused prework (qwen4_exp's long-context arm).
    wide_prework: bool = false,
    /// The model type the pipelined prefill kernel's measured table keys on.
    arch: []const u8,
    /// `1/dk` and `1/sqrt(dk)` as 0-d arrays in the activation dtype; `[dk]` ones (the q/k norms have no weight).
    q_scale: Array,
    k_scale: Array,
    ones: Array,
    /// The compiled gate chain (`compileGate`); null = the chain op by op.
    gate_fn: compiled.Closure = .{ .ctx = null },

    fn keyDim(geo: *const Geo) c_int {
        return geo.hk * geo.dk;
    }
    fn valueDim(geo: *const Geo) c_int {
        return geo.hv * geo.dv;
    }
    fn convDim(geo: *const Geo) c_int {
        return 2 * geo.keyDim() + geo.valueDim();
    }
    fn dec(geo: *const Geo) gdn_decode.Geometry {
        return .{ .hk = geo.hk, .hv = geo.hv, .dk = geo.dk, .dv = geo.dv };
    }
};

/// The per-model constants of `Geo`, in activation dtype `act`, on `s`. Freed with `freeConsts`.
pub fn consts(geo: *Geo, act: Dtype, s: ops.Stream) !void {
    var g = Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    const inv: f32 = 1.0 / @as(f32, @floatFromInt(geo.dk));
    geo.q_scale = g.escape(try g.astype(try g.adopt(mlx.mlx_array_new_float(inv)), act));
    geo.k_scale = g.escape(try g.astype(try g.adopt(mlx.mlx_array_new_float(@sqrt(inv))), act));
    geo.ones = g.escape(try g.ones(&.{geo.dk}, act));
}

pub fn freeConsts(geo: *Geo) void {
    inline for (.{ &geo.q_scale, &geo.k_scale, &geo.ones }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a.*);
        a.* = .{ .ctx = null };
    };
    compiled.free(&geo.gate_fn);
}

const GatePayload = struct { state: Dtype, s: ops.Stream };

fn gateBody(p: *const GatePayload, in: []const Array, out: *[1]Array) anyerror!void {
    out[0] = try gate_mod.gdnGateChain(in[0], in[1], in[2], gate_mod.gdnChainGateDtype(mlx.mlx_array_dtype(in[1]), p.state), p.s);
}

/// The gate chain `exp(-exp(A_log) * softplus(a + dt_bias))` compiled, as the legacy load compiles it.
pub fn compileGate(state_dtype: Dtype, s: ops.Stream) compiled.Closure {
    return compiled.compile(GatePayload, .{ .state = state_dtype, .s = s }, 1, gateBody, true);
}

pub const Gdn = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        /// The separate projections; `qkv` is null when q/k/v ship at different widths (`qkv_parts`).
        qkv: Linear = .{},
        qkv_parts: [3]Linear = @splat(.{}),
        z: Linear = .{},
        a: Linear = .{},
        b: Linear = .{},
        out: Linear,
        /// `[qkv | z | a | b]` joined for decode widths; null `lin.w` = not built.
        in: Joined = .{},
        /// qwen3_next: `qkv` holds q|k|v|z and `b` holds b|a, interleaved per key head.
        combined: bool = false,
        conv_w: Array,
        A_log: Array,
        dt_bias: Array,
        norm_w: Array,
        geo: Geo,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        return mix(w, c, x);
    }
};

/// The four projections in `[B, S, ...]`, each owned by `g`; `all` is the joined matmul's output when one ran.
const Proj = struct { qkv: Array, z: Array, a: Array, b: Array, all: ?Array = null };

fn project(w: *const Gdn.Weights, c: *const Ctx, x: Array, batch: c_int, seq: c_int) !Proj {
    const lane = c.row_exact;
    const g = c.g;
    if (w.combined) return projectCombined(w, g, x, batch, seq);
    if (w.in.built() and row_exact.groupServes(lane, batch * seq)) {
        const all = try row_exact.linear(lane, g, &w.in.lin, x);
        return .{ .qkv = try w.in.part(g, all, 0), .z = try w.in.part(g, all, 1), .a = try w.in.part(g, all, 2), .b = try w.in.part(g, all, 3), .all = all };
    }
    if (try c.offloadGdn(x)) |qz| return .{ .qkv = qz[0], .z = qz[1], .a = try row_exact.linear(lane, g, &w.a, x), .b = try row_exact.linear(lane, g, &w.b, x) };
    const qkv = if (w.qkv.w.ctx != null) try row_exact.linear(lane, g, &w.qkv, x) else blk: {
        var parts: [3]Array = undefined;
        for (&parts, &w.qkv_parts) |*p, *l| p.* = try row_exact.linear(lane, g, l, x);
        break :blk try g.concat(&parts, -1);
    };
    const z = try row_exact.linear(lane, g, &w.z, x);
    return .{ .qkv = qkv, .z = z, .a = try row_exact.linear(lane, g, &w.a, x), .b = try row_exact.linear(lane, g, &w.b, x) };
}

/// qwen3_next's interleaved q|k|v|z and b|a projections, split per key head and flattened.
fn projectCombined(w: *const Gdn.Weights, g: *Graph, x: Array, batch: c_int, seq: c_int) !Proj {
    const geo = &w.geo;
    const nk = geo.hk;
    const dk = geo.dk;
    const vph = @divExact(geo.hv, nk);
    const per_head = dk + dk + vph * geo.dv + vph * geo.dv;
    const qkvz = try g.reshape(try w.qkv.forward(g, x), &.{ batch, seq, nk, per_head });
    const v_off = dk * 2;
    const v_end = v_off + vph * geo.dv;
    const q = try g.slice(qkvz, &.{ 0, 0, 0, 0 }, &.{ batch, seq, nk, dk });
    const k = try g.slice(qkvz, &.{ 0, 0, 0, dk }, &.{ batch, seq, nk, dk * 2 });
    const v = try g.slice(qkvz, &.{ 0, 0, 0, v_off }, &.{ batch, seq, nk, v_end });
    const z = try g.slice(qkvz, &.{ 0, 0, 0, v_end }, &.{ batch, seq, nk, per_head });
    const qk_flat = [_]c_int{ batch, seq, geo.keyDim() };
    const vz_flat = [_]c_int{ batch, seq, geo.valueDim() };
    const parts = [_]Array{ try g.reshape(q, &qk_flat), try g.reshape(k, &qk_flat), try g.reshape(v, &vz_flat) };
    const qkv = try g.concat(&parts, 2);
    const z_flat = try g.reshape(z, &vz_flat);
    const ba = try g.reshape(try w.b.forward(g, x), &.{ batch, seq, nk, vph * 2 });
    const b_g = try g.slice(ba, &.{ 0, 0, 0, 0 }, &.{ batch, seq, nk, vph });
    const a_g = try g.slice(ba, &.{ 0, 0, 0, vph }, &.{ batch, seq, nk, vph * 2 });
    const ba_flat = [_]c_int{ batch, seq, geo.hv };
    return .{ .qkv = qkv, .z = z_flat, .a = try g.reshape(a_g, &ba_flat), .b = try g.reshape(b_g, &ba_flat) };
}

fn gdnInputs(w: *const Gdn.Weights, p: Proj, conv: Array, ssm: Array, eps: Array) gdn_decode.Inputs {
    return .{
        .qkv = p.qkv,
        .z = p.z,
        .a = p.a,
        .b = p.b,
        .conv_state = conv,
        .ssm_state = ssm,
        .conv_w = w.conv_w,
        .A_log = w.A_log,
        .dt_bias = w.dt_bias,
        .q_scale = w.geo.q_scale,
        .k_scale = w.geo.k_scale,
        .norm_w = w.norm_w,
        .eps = eps,
        .signs = .{ .ctx = null },
    };
}

fn adopt3(g: *Graph, xs: anytype) !void {
    inline for (xs) |x| _ = try g.adopt(x);
}

fn mix(w: *const Gdn.Weights, c: *Ctx, x: Array) !Array {
    const g = c.g;
    const geo = &w.geo;
    const batch: c_int = if (c.mode == .batched_rows) c.slots else 1;
    const seq = c.rows;
    const capture = c.capture_spec;
    const value_dim = geo.valueDim();
    const conv_dim = geo.convDim();
    const eps = c.shared.eps_arr;
    const swish = !geo.sigmoid_gate;

    const p = try project(w, c, x, batch, seq);
    var conv = try state.get(c, .conv);
    var ssm: ?Array = if (conv != null) try state.get(c, .ssm) else null;

    // A Hadamard pack, one decode token: prework, recurrence, norm-gate and the out projection's input rotation in two
    // dispatches, the rotation seeded so the projection reads it as rotated.
    if (w.out.rht) |reg| if (batch == 1 and seq == 1 and !capture and conv != null and ssm != null and geo.kernel == 4 and w.out.signs.ctx != null) fast: {
        var ins = gdnInputs(w, p, conv.?, ssm.?, eps);
        ins.signs = w.out.signs;
        const r = (try gdn_decode.step(geo.dec(), reg.block, ins, g.s)) orelse break :fast;
        try adopt3(g, .{ r.rot, r.conv_state, r.ssm_state });
        try state.put(c, .conv, r.conv_state);
        try state.put(c, .ssm, r.ssm_state);
        try reg.seed(r.rot, w.out.signs, r.rot);
        return w.out.forward(g, r.rot);
    };

    // One decode token: prework and recurrence in one dispatch, then the norm-gate.
    if (w.out.rht == null and batch == 1 and seq == 1 and !capture and conv != null and ssm != null and geo.kernel == 4 and fused.gdnDecodeRecurEnabled()) fast: {
        const r = (try gdn_decode.recur(geo.dec(), gdnInputs(w, p, conv.?, ssm.?, eps), g.s)) orelse break :fast;
        try adopt3(g, .{ r.y, r.conv_state, r.ssm_state });
        const flat = (try fused.gdnNormGateFused(g.s, r.y, p.z, 0, value_dim, w.norm_w, eps, swish, geo.hv, geo.dv, 1, 1)) orelse break :fast;
        try state.put(c, .conv, r.conv_state);
        try state.put(c, .ssm, r.ssm_state);
        if (!fused.gdn_decode_recur_engaged) {
            fused.gdn_decode_recur_engaged = true;
            log.info("[gdn] decode recur engaged: Hk={d} Hv={d} gate={s}\n", .{ geo.hk, geo.hv, if (geo.sigmoid_gate) "sigmoid" else "swish" });
        }
        return row_exact.linear(c.row_exact, g, &w.out, try g.adopt(flat));
    }

    // Verify rows: the same kernel over S tokens, keeping every row's state for the rollback.
    if (w.out.rht == null and batch == 1 and seq >= 1 and seq <= gdn_decode.MAX_SEQ and capture and conv != null and ssm != null and geo.kernel == 4 and fused.gdnDecodeRecurEnabled()) fast: {
        const ins = gdnInputs(w, p, conv.?, ssm.?, eps);
        if (c.tree) |t| return treeVerify(w, c, p, ins, ssm.?, t.table);
        if (fused.gdnVerifyFoldEnabled()) fold: {
            const f = (try gdn_decode.recurSeqFold(geo.dec(), seq, ins, swish, g.s)) orelse break :fold;
            try adopt3(g, .{ f.gated, f.conv_input, f.state_seq, f.conv_state, f.ssm_state });
            try state.put(c, .conv_input, f.conv_input);
            try state.put(c, .state_seq, f.state_seq);
            try state.put(c, .conv, f.conv_state);
            try state.put(c, .ssm, f.ssm_state);
            if (!fused.gdn_verify_fold_engaged) {
                fused.gdn_verify_fold_engaged = true;
                log.info("[gdn] verify fold engaged: S={d} Hk={d} Hv={d}\n", .{ seq, geo.hk, geo.hv });
            }
            return row_exact.linear(c.row_exact, g, &w.out, f.gated);
        }
        const r = (try gdn_decode.recurSeq(geo.dec(), seq, ins, g.s)) orelse break :fast;
        try adopt3(g, .{ r.y, r.conv_state, r.ssm_state, r.state_seq });
        const flat = (try fused.gdnNormGateFused(g.s, r.y, p.z, 0, value_dim, w.norm_w, eps, swish, geo.hv, geo.dv, 1, seq)) orelse break :fast;
        const conv_input = try g.concat(&.{ conv.?, p.qkv }, 1);
        try state.put(c, .conv_input, conv_input);
        try state.put(c, .state_seq, r.state_seq);
        try state.put(c, .conv, r.conv_state);
        try state.put(c, .ssm, r.ssm_state);
        if (!fused.gdn_verify_recur_engaged) {
            fused.gdn_verify_recur_engaged = true;
            log.info("[gdn] verify recur engaged: S={d} Hk={d} Hv={d}\n", .{ seq, geo.hk, geo.hv });
        }
        return row_exact.linear(c.row_exact, g, &w.out, try g.adopt(flat));
    }

    if (c.tree != null) return error.SpecTreeUnsupported;
    var q_scaled: Array = undefined;
    var k_scaled: Array = undefined;
    var v_heads: Array = undefined;
    var fused_prework = false;
    var g_fused: ?Array = null;
    var beta_fused: ?Array = null;

    const prefill_fused = geo.wide_prework and seq > 1 and fused.gdnPrefillFusedFor(seq, batch);
    if ((batch * seq <= fused.GDN_FUSED_MAX_ROWS or prefill_fused) and geo.kernel == 4 and (conv != null or prefill_fused)) blk: {
        const incoming = conv orelse try g.zeros(&.{ batch, 3, conv_dim }, ops.dtype(p.qkv));
        const pre = (fused.gdnPreworkFused(g.s, .{
            .qkv = p.qkv,
            .qkv_off = 0,
            .qkv_stride = conv_dim,
            .b = p.b,
            .b_off = 0,
            .b_stride = geo.hv,
            .a = p.a,
            .a_off = 0,
            .a_stride = geo.hv,
            .A_log = w.A_log,
            .dt_bias = w.dt_bias,
            .conv_state = incoming,
            .conv_w = w.conv_w,
            .q_scale = geo.q_scale,
            .k_scale = geo.k_scale,
            .qk_eps = geo.qk_eps,
            .hk = geo.hk,
            .hv = geo.hv,
            .dk = geo.dk,
            .dv = geo.dv,
            .seq = seq,
            .batch = batch,
            .gate = true,
            .g_dtype = gdn_decode.gateDtype(ops.dtype(p.qkv), geo.state),
        }) catch null) orelse break :blk;
        try adopt3(g, .{ pre.q, pre.k, pre.v, pre.conv_state, pre.g, pre.beta });
        if (capture) try state.put(c, .conv_input, try g.concat(&.{ incoming, p.qkv }, 1));
        if (prefill_fused and seq >= 512 and !fused.gdn_prefill_engaged) {
            fused.gdn_prefill_engaged = true;
            log.info("[gdn-prefill] engaged: S={d} B={d} cold={}\n", .{ seq, batch, conv == null });
        }
        conv = pre.conv_state;
        q_scaled = pre.q;
        k_scaled = pre.k;
        v_heads = pre.v;
        fused_prework = true;
        if (fused.gdnDecodeFusedEnabled()) {
            g_fused = pre.g;
            beta_fused = pre.beta;
        }
    }

    if (!fused_prework) {
        const cv = try conv1d.run(g, p.qkv, conv, w.conv_w, null, batch, conv_dim, geo.kernel, true);
        conv = cv.window;
        if (capture) try state.put(c, .conv_input, cv.input);
        const kd = geo.keyDim();
        const q_flat = try g.slice(cv.y, &.{ 0, 0, 0 }, &.{ batch, seq, kd });
        const k_flat = try g.slice(cv.y, &.{ 0, 0, kd }, &.{ batch, seq, kd * 2 });
        const v_flat = try g.slice(cv.y, &.{ 0, 0, kd * 2 }, &.{ batch, seq, kd * 2 + value_dim });
        const q_heads = try g.reshape(q_flat, &.{ batch, seq, geo.hk, geo.dk });
        const k_heads = try g.reshape(k_flat, &.{ batch, seq, geo.hk, geo.dk });
        v_heads = try g.reshape(v_flat, &.{ batch, seq, geo.hv, geo.dv });
        q_scaled = try g.mul(try g.rmsNorm(q_heads, geo.ones, geo.qk_eps), geo.q_scale);
        k_scaled = try g.mul(try g.rmsNorm(k_heads, geo.ones, geo.qk_eps), geo.k_scale);
    }
    try state.put(c, .conv, conv.?);

    const gate = g_fused orelse try gdnGate(g, geo, w.A_log, p.a, w.dt_bias);
    const beta = beta_fused orelse try g.sigmoid(p.b);
    const ssm_in = ssm orelse try g.zeros(&.{ batch, geo.hv, geo.dv, geo.dk }, geo.state);
    ssm = ssm_in;

    const in_dtype = ops.dtype(q_scaled);
    const state_dtype = ops.dtype(ssm_in);
    // y leaves in the activation dtype (an f32 activation keeps a bf16 y).
    const out_dtype: Dtype = if (in_dtype == .float16) .float16 else .bfloat16;
    var y: Array = undefined;

    if (!mlx.streamIsGpu(g.s)) {
        const r = (try kernels.gdnRecurCuda(g.s, q_scaled, k_scaled, v_heads, gate, beta, ssm_in, false, capture, out_dtype)) orelse
            try kernels.gdnRecurOps(g.s, q_scaled, k_scaled, v_heads, gate, beta, ssm_in, false, capture, out_dtype);
        y = try g.adopt(r.y);
        const st = try g.adopt(r.state);
        try state.put(c, .ssm, st);
        if (capture) try state.put(c, .state_seq, try g.adopt(r.state_seq));
    } else if (capture) {
        const outs = try recurrence(g, geo, .stock, 0, &.{ q_scaled, k_scaled, v_heads, gate, beta, ssm_in }, batch, seq, true, in_dtype, state_dtype, out_dtype);
        y = outs[0];
        try state.put(c, .state_seq, outs[1]);
        try state.put(c, .ssm, outs[2]);
    } else {
        const tb: ?u32 = if (kernels.gdnBlockedEnabled() and kernels.gdnBlockedEligible(seq, geo.dk, geo.dv, geo.hk, geo.hv))
            kernels.gdnBlockTFor(kernels.GDN_BLOCK_T, geo.dk, mlx.mlx_array_itemsize(q_scaled))
        else
            null;
        const route: kernels.GdnRoute = if (tb == null) .stock else if (kernels.gdnPipelinedEnabled(geo.arch)) .pipelined else .blocked;
        const outs = try recurrence(g, geo, route, tb orelse 0, &.{ q_scaled, k_scaled, v_heads, gate, beta, ssm_in }, batch, seq, false, in_dtype, state_dtype, out_dtype);
        y = outs[0];
        try state.put(c, .ssm, outs[1]);
    }

    // Fused epilogue: rms_norm(y) * act(z) straight to the out projection's input.
    if (fused_prework) {
        if (try fused.gdnNormGateFused(g.s, y, p.z, 0, value_dim, w.norm_w, eps, swish, geo.hv, geo.dv, batch, seq)) |flat|
            return row_exact.linear(c.row_exact, g, &w.out, try g.adopt(flat));
    }
    const z_heads = try g.reshape(p.z, &.{ batch, seq, geo.hv, geo.dv });
    const y_normed = try g.rmsNorm(y, w.norm_w, c.shared.eps);
    const gated = if (geo.sigmoid_gate)
        try g.mul(y_normed, try g.sigmoid(z_heads))
    else
        try g.mul(try g.mul(z_heads, try g.sigmoid(z_heads)), y_normed);
    return row_exact.linear(c.row_exact, g, &w.out, try g.reshape(gated, &.{ batch, seq, value_dim }));
}

/// A draft tree's verify rows, each row's recurrence along its own ancestors. The conv and scan states stay the round's
/// input: the commit replays the kept path from the input state, the conv input and every row's prework left here.
fn treeVerify(w: *const Gdn.Weights, c: *Ctx, p: Proj, ins: gdn_decode.Inputs, ssm: Array, table: Array) !Array {
    const g = c.g;
    const geo = &w.geo;
    const wd = w.in.widths;
    const joined: ?gdn_decode.Joined = if (p.all) |all| .{ .all = all, .a_off = wd[0] + wd[1], .b_off = wd[0] + wd[1] + wd[2] } else null;
    const r = (try gdn_decode.recurTree(geo.dec(), c.rows, ins, joined, table, g.s)) orelse return error.SpecTreeUnsupported;
    inline for (.{ r.y, r.conv_input, r.prework.k, r.prework.v, r.prework.gb }) |x| _ = try g.adopt(x);
    const z_src = if (joined) |j| j.all else p.z;
    const z_off: c_int = if (joined != null) wd[0] else 0;
    const z_stride = if (joined) |j| ops.shape(j.all)[2] else geo.valueDim();
    const flat = (try fused.gdnNormGateFused(g.s, r.y, z_src, z_off, z_stride, w.norm_w, c.shared.eps_arr, !geo.sigmoid_gate, geo.hv, geo.dv, 1, c.rows)) orelse return error.SpecTreeUnsupported;
    try state.put(c, .state_seq, .{ .ctx = null });
    try state.put(c, .conv_input, r.conv_input);
    try state.put(c, .state_in, ssm);
    try state.put(c, .prework_k, r.prework.k);
    try state.put(c, .prework_v, r.prework.v);
    try state.put(c, .prework_gb, r.prework.gb);
    if (!fused.gdn_verify_tree_engaged) {
        fused.gdn_verify_tree_engaged = true;
        log.info("[gdn] verify tree engaged: S={d} Hk={d} Hv={d}\n", .{ c.rows, geo.hk, geo.hv });
    }
    return row_exact.linear(c.row_exact, g, &w.out, try g.adopt(flat));
}

/// The decay gate `[B, S, Hv]`: the compiled closure where one was built, else the chain.
fn gdnGate(g: *Graph, geo: *const Geo, A_log: Array, a: Array, dt_bias: Array) !Array {
    if (geo.gate_fn.ctx != null) if (try compiled.apply(g, geo.gate_fn, &.{ A_log, a, dt_bias }, 1)) |out| return out[0];
    return g.adopt(try gate_mod.gdnGateChain(A_log, a, dt_bias, gate_mod.gdnChainGateDtype(ops.dtype(a), geo.state), g.s));
}

/// The T-step delta recurrence in one Metal dispatch. Returns y and the final state, with the per-row state
/// sequence between them when `capture` (`[T, B, Hv, Dv, Dk]`, its last row unwritten).
fn recurrence(g: *Graph, geo: *const Geo, route: kernels.GdnRoute, tb: u32, ins: *const [6]Array, batch: c_int, seq: c_int, capture: bool, in_dt: Dtype, st_dt: Dtype, out_dt: Dtype) ![3]Array {
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ batch, seq, geo.hv, geo.dv };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, out_dt));
    const st_shape = [_]c_int{ batch, geo.hv, geo.dv, geo.dk };
    if (capture) {
        const seq_shape = [_]c_int{ seq, batch, geo.hv, geo.dv, geo.dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &seq_shape, 5, st_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &st_shape, 4, st_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, geo.dv, batch * geo.hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &st_shape, 4, st_dt));
        try kernels.gdnSetGrid(config, route, geo.dv, geo.hv, batch);
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", in_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", geo.dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", geo.dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", geo.hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", geo.hv));

    const t_scalar = try g.adopt(mlx.mlx_array_new_int(seq));
    var in_buf: [8]Array = undefined;
    @memcpy(in_buf[0..6], ins);
    in_buf[6] = t_scalar;
    var n_in: usize = 7;
    if (capture) {
        in_buf[7] = try g.adopt(mlx.mlx_array_new_int(batch * geo.hv * geo.dv * geo.dk));
        n_in = 8;
    }
    const in_vec = mlx.mlx_vector_array_new_data(&in_buf, n_in);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    const kernel = if (capture) try kernels.getGdnKernelSeq(false) else try kernels.gdnRouteKernel(route, tb, false);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, config, g.s));
    const n_out: usize = if (capture) 3 else 2;
    if (mlx.mlx_vector_array_size(out_vec) != n_out) return error.MetalKernelBadOutputCount;
    var out: [3]Array = @splat(.{ .ctx = null });
    for (0..n_out) |i| {
        var a = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&a, out_vec, i)) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        out[i] = try g.adopt(a);
    }
    return out;
}
